from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sparta_dataset import SPARTADataset
from src.metrics import compute_attribution_majority_baseline
from src.train import apply_resume_dataset_run, build_model, normalize_variant, prepare_metric_evidence_stats, select_device
from src.utils.config import load_config
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


METRIC_NAMES = {0: "delay", 1: "loss", 2: "cpu", 3: "queue", 4: "bandwidth"}
OBJECTIVES = {"balanced_acc", "macro_f1", "acc_plus_balanced"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="acc_plus_balanced")
    parser.add_argument("--grid_min", type=float, default=-2.0)
    parser.add_argument("--grid_max", type=float, default=2.0)
    parser.add_argument("--grid_step", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_passes", type=int, default=5)
    parser.add_argument("--calibration_name", default=None)
    return parser.parse_args()


def artifact_name(variant: str | None) -> str:
    return f"sparta_{variant}" if variant else "sparta"


def normalize_calibration_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if any(ch in text for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]):
        raise ValueError(f"Invalid calibration_name: {value}")
    return text


def output_suffix(calibration_name: str | None) -> str:
    return f"_{calibration_name}" if calibration_name else ""


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_run_config(run_dir: Path, variant: str | None) -> dict:
    config_path = run_dir / "config" / "resolved_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    config = load_config(config_path)
    return apply_resume_dataset_run(config, str(run_dir), variant)


def load_sparta_model(config: dict, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    prepare_metric_evidence_stats(config, None, write=False)
    model = build_model(config, "sparta").to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_metric_logits(
    model,
    dataset_path: Path,
    config: dict,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray | int]:
    dataset = SPARTADataset(dataset_path, config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.get("train", {}).get("num_workers", 0)),
    )
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    total_samples = 0
    valid_attr_samples = 0
    for batch in loader:
        total_samples += int(batch["risk_metric"].shape[0])
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        outputs = model(batch)
        if "metric_logits" not in outputs:
            raise RuntimeError("SPARTA checkpoint did not produce metric_logits")
        attr_mask = batch["attr_mask"].bool()
        valid_attr_samples += int(attr_mask.sum().item())
        if attr_mask.any():
            logits_parts.append(outputs["metric_logits"][attr_mask].detach().cpu())
            label_parts.append(batch["risk_metric"][attr_mask].detach().cpu())
    logits = torch.cat(logits_parts, dim=0).numpy() if logits_parts else np.zeros((0, 5), dtype=np.float32)
    labels = torch.cat(label_parts, dim=0).numpy() if label_parts else np.zeros((0,), dtype=np.int64)
    return {
        "logits": logits,
        "labels": labels.astype(np.int64),
        "total_samples": total_samples,
        "valid_attr_samples": valid_attr_samples,
    }


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def metric_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    total = int(y_true.shape[0])
    recalls: dict[str, float] = {}
    precisions: dict[str, float] = {}
    f1s: list[float] = []
    for metric_id, name in METRIC_NAMES.items():
        tp = int(np.sum((y_true == metric_id) & (y_pred == metric_id)))
        fp = int(np.sum((y_true != metric_id) & (y_pred == metric_id)))
        fn = int(np.sum((y_true == metric_id) & (y_pred != metric_id)))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        precisions[name] = precision
        recalls[name] = recall
        f1s.append(safe_div(2.0 * precision * recall, precision + recall))
    return {
        "metric_attr_acc": safe_div(int(np.sum(y_true == y_pred)), total),
        "metric_attr_macro_f1": sum(f1s) / len(f1s),
        "metric_attr_balanced_acc": sum(recalls.values()) / len(recalls),
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "pred_counts": {METRIC_NAMES[idx]: int(np.sum(y_pred == idx)) for idx in METRIC_NAMES},
        "support": {METRIC_NAMES[idx]: int(np.sum(y_true == idx)) for idx in METRIC_NAMES},
    }


def predictions(logits: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    if bias is None:
        bias = np.zeros((5,), dtype=np.float32)
    return np.argmax(logits + bias.reshape(1, 5), axis=1).astype(np.int64)


def objective_score(metrics: dict, objective: str) -> float:
    if objective == "balanced_acc":
        return float(metrics["metric_attr_balanced_acc"])
    if objective == "macro_f1":
        return float(metrics["metric_attr_macro_f1"])
    return (
        float(metrics["metric_attr_acc"])
        + float(metrics["metric_attr_balanced_acc"])
        + 0.5 * float(metrics["metric_attr_macro_f1"])
    )


def coordinate_grid_search(
    logits: np.ndarray,
    labels: np.ndarray,
    grid_values: np.ndarray,
    objective: str = "acc_plus_balanced",
    max_passes: int = 5,
) -> tuple[np.ndarray, dict]:
    if logits.shape[0] == 0:
        return np.zeros((5,), dtype=np.float32), {"score": math.nan, "passes": 0, "evaluations": 0}
    bias = np.zeros((5,), dtype=np.float32)
    best_metrics = metric_metrics(labels, predictions(logits, bias))
    best_score = objective_score(best_metrics, objective)
    evaluations = 1
    for pass_idx in range(max(1, int(max_passes))):
        changed = False
        for dim in range(5):
            dim_best_value = float(bias[dim])
            dim_best_score = best_score
            dim_best_metrics = best_metrics
            for value in grid_values:
                candidate = bias.copy()
                candidate[dim] = float(value)
                candidate_metrics = metric_metrics(labels, predictions(logits, candidate))
                candidate_score = objective_score(candidate_metrics, objective)
                evaluations += 1
                if candidate_score > dim_best_score + 1e-12:
                    dim_best_value = float(value)
                    dim_best_score = candidate_score
                    dim_best_metrics = candidate_metrics
            if abs(float(bias[dim]) - dim_best_value) > 1e-12:
                bias[dim] = dim_best_value
                best_score = dim_best_score
                best_metrics = dim_best_metrics
                changed = True
        if not changed:
            return bias.astype(np.float32), {"score": best_score, "passes": pass_idx + 1, "evaluations": evaluations}
    return bias.astype(np.float32), {"score": best_score, "passes": int(max_passes), "evaluations": evaluations}


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    rows = []
    for true_id, true_name in METRIC_NAMES.items():
        for pred_id, pred_name in METRIC_NAMES.items():
            rows.append(
                {
                    "true_metric_id": true_id,
                    "true_metric": true_name,
                    "pred_metric_id": pred_id,
                    "pred_metric": pred_name,
                    "count": int(np.sum((y_true == true_id) & (y_pred == pred_id))),
                }
            )
    return rows


def flatten_calibration_row(summary: dict) -> dict:
    row: dict = {
        "method": "sparta_metric_logit_calibration",
        "model": "sparta",
        "variant": summary["variant"],
        "checkpoint_path": summary["checkpoint_path"],
        "dataset_path": summary["test_dataset_path"],
        "calibration_dataset_path": summary["val_dataset_path"],
        "objective": summary["objective"],
        "metric_majority_baseline": summary["metric_majority_baseline"],
        "after_acc_minus_majority": summary["after_acc_minus_majority"],
        "before_metric_attr_acc": summary["before_metric_attr_acc"],
        "after_metric_attr_acc": summary["after_metric_attr_acc"],
        "before_metric_attr_macro_f1": summary["before_metric_attr_macro_f1"],
        "after_metric_attr_macro_f1": summary["after_metric_attr_macro_f1"],
        "before_metric_attr_balanced_acc": summary["before_metric_attr_balanced_acc"],
        "after_metric_attr_balanced_acc": summary["after_metric_attr_balanced_acc"],
    }
    for name in METRIC_NAMES.values():
        row[f"before_recall_{name}"] = summary["before_per_class_recall"][name]
        row[f"after_recall_{name}"] = summary["after_per_class_recall"][name]
        row[f"best_bias_{name}"] = summary[f"best_bias_{name}"]
    return row


def calibrate_run(
    run_dir: Path,
    variant: str | None,
    objective: str,
    grid_min: float,
    grid_max: float,
    grid_step: float,
    batch_size: int | None = None,
    max_passes: int = 5,
    calibration_name: str | None = None,
) -> dict:
    if grid_step <= 0:
        raise ValueError("grid_step must be positive")
    if grid_max < grid_min:
        raise ValueError("grid_max must be >= grid_min")
    if objective not in OBJECTIVES:
        raise ValueError(f"Unsupported objective: {objective}")
    variant = normalize_variant(variant)
    calibration_name = normalize_calibration_name(calibration_name)
    artifact = artifact_name(variant)
    config = load_run_config(run_dir, variant)
    set_seed(int(config.get("seed", 42)))
    device = select_device(config)
    checkpoint_path = run_dir / "checkpoints" / f"{artifact}_best.pth"
    model = load_sparta_model(config, checkpoint_path, device)
    batch_size = int(batch_size or config.get("train", {}).get("batch_size", 64))
    val_path = run_dir / "dataset" / "val.pkl"
    test_path = run_dir / "dataset" / "test.pkl"
    val_data = collect_metric_logits(model, val_path, config, device, batch_size)
    test_data = collect_metric_logits(model, test_path, config, device, batch_size)
    grid_values = np.round(np.arange(grid_min, grid_max + 0.5 * grid_step, grid_step), 10).astype(np.float32)
    best_bias, search_info = coordinate_grid_search(
        val_data["logits"],
        val_data["labels"],
        grid_values,
        objective=objective,
        max_passes=max_passes,
    )
    before_pred = predictions(test_data["logits"])
    after_pred = predictions(test_data["logits"], best_bias)
    before = metric_metrics(test_data["labels"], before_pred)
    after = metric_metrics(test_data["labels"], after_pred)
    test_dataset = SPARTADataset(test_path, config)
    majority = compute_attribution_majority_baseline(test_dataset.samples, split="test", attr_mask_only=True)
    majority_baseline = majority["metric_majority_acc_valid_attr_only"]
    summary = {
        "run_dir": str(run_dir),
        "variant": variant or "",
        "artifact": artifact,
        "checkpoint_path": str(checkpoint_path),
        "val_dataset_path": str(val_path),
        "test_dataset_path": str(test_path),
        "objective": objective,
        "calibration_name": calibration_name or "",
        "grid_min": grid_min,
        "grid_max": grid_max,
        "grid_step": grid_step,
        "grid_search_method": "coordinate_grid_search",
        "grid_search": search_info,
        "val_total_samples": int(val_data["total_samples"]),
        "val_valid_attr_samples": int(val_data["valid_attr_samples"]),
        "test_total_samples": int(test_data["total_samples"]),
        "test_valid_attr_samples": int(test_data["valid_attr_samples"]),
        "normal_samples_excluded_from_calibration": True,
        "fit_split": "val",
        "apply_split": "test",
        "metric_majority_baseline": majority_baseline,
        "metric_majority_acc_valid_attr_only": majority["metric_majority_acc_valid_attr_only"],
        "metric_majority_acc_including_normal": majority["metric_majority_acc_including_normal"],
        "metric_majority_name": majority.get("metric_majority_name", ""),
        "before_metric_attr_acc": before["metric_attr_acc"],
        "after_metric_attr_acc": after["metric_attr_acc"],
        "before_metric_attr_macro_f1": before["metric_attr_macro_f1"],
        "after_metric_attr_macro_f1": after["metric_attr_macro_f1"],
        "before_metric_attr_balanced_acc": before["metric_attr_balanced_acc"],
        "after_metric_attr_balanced_acc": after["metric_attr_balanced_acc"],
        "before_per_class_recall": before["per_class_recall"],
        "after_per_class_recall": after["per_class_recall"],
        "before_pred_counts": before["pred_counts"],
        "after_pred_counts": after["pred_counts"],
        "support": after["support"],
        "after_acc_minus_majority": after["metric_attr_acc"] - majority_baseline,
    }
    for idx, name in METRIC_NAMES.items():
        summary[f"best_bias_{name}"] = float(best_bias[idx])
    analysis_dir = ensure_dir(run_dir / "analysis")
    results_dir = ensure_dir(run_dir / "results")
    analysis_prefix = variant or "sparta"
    suffix = output_suffix(calibration_name)
    json_path = analysis_dir / f"{analysis_prefix}_metric_calibration{suffix}.json"
    confusion_path = analysis_dir / f"{analysis_prefix}_metric_calibrated_confusion{suffix}.csv"
    result_path = results_dir / f"{artifact}_calibrated_test_results{suffix}.csv"
    summary["result_source_file"] = str(result_path)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(confusion_path, confusion_rows(test_data["labels"], after_pred))
    result_row = flatten_calibration_row(summary)
    write_csv(result_path, [result_row])
    print(f"Wrote {json_path}")
    print(f"Wrote {confusion_path}")
    print(f"Wrote {result_path}")
    return summary


def main() -> None:
    args = parse_args()
    calibrate_run(
        run_dir=Path(args.run_dir).resolve(),
        variant=args.variant,
        objective=args.objective,
        grid_min=args.grid_min,
        grid_max=args.grid_max,
        grid_step=args.grid_step,
        batch_size=args.batch_size,
        max_passes=args.max_passes,
        calibration_name=args.calibration_name,
    )


if __name__ == "__main__":
    main()
