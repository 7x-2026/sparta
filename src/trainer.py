from __future__ import annotations

import csv
from pathlib import Path

import torch

from src.metrics import collect_predictions, compute_attribution_metrics, compute_classification_metrics
from src.utils.config import resolve_path
from src.utils.io import ensure_dir


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, config) -> dict[str, float]:
    model.train()
    total_loss = 0.0
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
        total_loss += float(loss.detach().cpu())
        batches += 1
    return {"loss": total_loss / max(batches, 1)}


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, config) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        loss_dict = loss_fn(outputs, batch)
        total_loss += float(loss_dict["loss"].detach().cpu())
        batches += 1
    labels, preds, has_attr = collect_predictions(model, dataloader, device)
    metrics = compute_classification_metrics(labels["risk_label"], preds["risk_label"])
    metrics.update(compute_attribution_metrics(labels, preds, has_attr))
    metrics["loss"] = total_loss / max(batches, 1)
    return metrics


def fit(model, train_loader, val_loader, loss_fn, optimizer, config: dict, model_name: str, device: torch.device) -> list[dict]:
    checkpoint_dir = resolve_path(config, config["project"]["checkpoint_dir"])
    log_dir = resolve_path(config, config["project"]["log_dir"])
    ensure_dir(checkpoint_dir)
    ensure_dir(log_dir)
    best_metric_name = config["train"].get("best_metric", "macro_f1")
    best_metric = -float("inf")
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, config)
        val_metrics = validate(model, val_loader, loss_fn, device, config)
        best_checkpoint_path = checkpoint_dir / f"{model_name}_best.pth"
        last_checkpoint_path = checkpoint_dir / f"{model_name}_last.pth"
        row = {
            "model": model_name,
            "run_id": config.get("run", {}).get("run_id", ""),
            "dataset_path": str(resolve_path(config, config["data"].get("train_path", Path(config["data"]["dataset_dir"]) / "train.pkl"))),
            "checkpoint_save_path": str(checkpoint_dir),
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
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
        print(f"{model_name} epoch {epoch}: train_loss={row['train_loss']:.4f} val_macro_f1={row['val_macro_f1']:.4f}")

    fieldnames = list(history[0].keys()) if history else []
    for log_path in [log_dir / f"train_log_{model_name}.csv", log_dir / f"{model_name}_train.log"]:
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
    return history
