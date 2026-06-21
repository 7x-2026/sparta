from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from src.metrics import collect_predictions, compute_attribution_metrics, compute_classification_metrics
from src.utils.config import resolve_path
from src.utils.io import ensure_dir


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, config) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "loss_risk": 0.0, "loss_node": 0.0, "loss_link": 0.0, "loss_metric": 0.0, "loss_metric_soft": 0.0}
    batches = 0
    grad_clip = float(config["train"].get("grad_clip", 1.0))
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        loss_dict = loss_fn(outputs, batch)
        loss = loss_dict["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        for key in totals:
            value = loss_dict.get(key)
            if value is not None:
                totals[key] += float(value.detach().cpu())
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, config) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "loss_risk": 0.0, "loss_node": 0.0, "loss_link": 0.0, "loss_metric": 0.0, "loss_metric_soft": 0.0}
    batches = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        loss_dict = loss_fn(outputs, batch)
        for key in totals:
            value = loss_dict.get(key)
            if value is not None:
                totals[key] += float(value.detach().cpu())
        batches += 1
    labels, preds, has_attr = collect_predictions(model, dataloader, device)
    metrics = compute_classification_metrics(labels["risk_label"], preds["risk_label"])
    metrics.update(compute_attribution_metrics(labels, preds, has_attr))
    for key, value in totals.items():
        metrics[key] = value / max(batches, 1)
    return metrics


def fit(model, train_loader, val_loader, loss_fn, optimizer, config: dict, model_name: str, device: torch.device) -> list[dict]:
    checkpoint_dir = resolve_path(config, config["project"]["checkpoint_dir"])
    log_dir = resolve_path(config, config["project"]["log_dir"])
    ensure_dir(checkpoint_dir)
    ensure_dir(log_dir)
    variant = config.get("variant") or config.get("run", {}).get("variant") or ""
    artifact_name = f"{model_name}_{variant}" if variant else model_name
    loss_cfg = config.get("loss", {})
    best_metric_name = config["train"].get("best_metric", "macro_f1")
    best_metric = -float("inf")
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, config)
        val_metrics = validate(model, val_loader, loss_fn, device, config)
        best_checkpoint_path = checkpoint_dir / f"{artifact_name}_best.pth"
        last_checkpoint_path = checkpoint_dir / f"{artifact_name}_last.pth"
        row = {
            "model": model_name,
            "artifact_model": artifact_name,
            "variant": variant,
            "run_id": config.get("run", {}).get("run_id", ""),
            "dataset_path": str(resolve_path(config, config["data"].get("train_path", Path(config["data"]["dataset_dir"]) / "train.pkl"))),
            "checkpoint_save_path": str(checkpoint_dir),
            "epoch": epoch,
            "lambda_metric": loss_cfg.get("lambda_metric", 0.2),
            "lambda_metric_soft": loss_cfg.get("lambda_metric_soft", 0.0),
            "metric_loss_type": loss_cfg.get("metric_loss_type", "weighted_ce"),
            "metric_soft_loss_type": loss_cfg.get("metric_soft_loss_type", "none"),
            "metric_soft_temperature": loss_cfg.get("metric_soft_temperature", 2.0),
            "metric_class_weighting": loss_cfg.get("metric_class_weighting", "none"),
            "metric_focal_gamma": loss_cfg.get("metric_focal_gamma", 2.0),
            "risk_metric_scores_use_future_information": True,
            "used_as_supervision_only": True,
            "used_as_model_input": False,
            "train_loss": train_metrics["loss"],
            "loss_risk": train_metrics.get("loss_risk", 0.0),
            "loss_node": train_metrics.get("loss_node", 0.0),
            "loss_link": train_metrics.get("loss_link", 0.0),
            "loss_metric": train_metrics.get("loss_metric", 0.0),
            "loss_metric_soft": train_metrics.get("loss_metric_soft", 0.0),
            "metric_class_weights": json.dumps(
                config.get("loss", {}).get("metric_class_weights_resolved", []),
                ensure_ascii=False,
            ),
            "val_loss": val_metrics["loss"],
            "val_loss_risk": val_metrics.get("loss_risk", 0.0),
            "val_loss_node": val_metrics.get("loss_node", 0.0),
            "val_loss_link": val_metrics.get("loss_link", 0.0),
            "val_loss_metric": val_metrics.get("loss_metric", 0.0),
            "val_loss_metric_soft": val_metrics.get("loss_metric_soft", 0.0),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_risk_recall": val_metrics["risk_recall"],
            "val_violation_recall": val_metrics["violation_recall"],
            "val_node_attr_acc": val_metrics["node_attr_acc"],
            "val_link_attr_acc": val_metrics["link_attr_acc"],
            "val_metric_attr_acc": val_metrics["metric_attr_acc"],
            "best_epoch": best_epoch,
            "best_checkpoint_path": str(best_checkpoint_path),
        }
        current = val_metrics.get(best_metric_name, val_metrics["macro_f1"])
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            torch.save(
                {
                    "model_name": model_name,
                    "artifact_model": artifact_name,
                    "variant": variant,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "config": config,
                },
                best_checkpoint_path,
            )
        torch.save(
            {
                "model_name": model_name,
                "artifact_model": artifact_name,
                "variant": variant,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_metric": best_metric,
                "config": config,
            },
            last_checkpoint_path,
        )
        row["best_epoch"] = best_epoch
        history.append(row)
        print(
            f"{model_name} epoch {epoch}: "
            f"variant={variant} "
            f"train_loss={row['train_loss']:.4f} "
            f"loss_risk={row['loss_risk']:.4f} "
            f"loss_node={row['loss_node']:.4f} "
            f"loss_link={row['loss_link']:.4f} "
            f"loss_metric={row['loss_metric']:.4f} "
            f"loss_metric_soft={row['loss_metric_soft']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f}"
        )

    fieldnames = list(history[0].keys()) if history else []
    for log_path in [log_dir / f"train_log_{artifact_name}.csv", log_dir / f"{artifact_name}_train.log"]:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
    return history
