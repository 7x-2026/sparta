from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sparta_dataset import SPARTADataset
from src.metrics import collect_predictions, compute_attribution_metrics, compute_classification_metrics
from src.train import SUPPORTED_MODELS, apply_resume_dataset_run, build_model, normalize_variant, prepare_metric_evidence_stats, select_device
from src.utils.config import load_config, resolve_path
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


RESULT_COLUMNS = [
    "method",
    "model",
    "variant",
    "has_attribution_head",
    "checkpoint_path",
    "dataset_path",
    "result_source_file",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_acc",
    "risk_recall",
    "violation_recall",
    "auc",
    "risk_auc_macro_ovr",
    "risk_auc_weighted_ovr",
    "risk_auc_binary",
    "auc_error",
    "node_attr_acc",
    "link_attr_acc",
    "metric_attr_acc",
    "metric_attr_macro_f1",
    "metric_attr_balanced_acc",
    "metric_attr_per_class_recall_delay",
    "metric_attr_per_class_recall_loss",
    "metric_attr_per_class_recall_cpu",
    "metric_attr_per_class_recall_queue",
    "metric_attr_per_class_recall_bandwidth",
    "metric_pred_count_delay",
    "metric_pred_count_loss",
    "metric_pred_count_cpu",
    "metric_pred_count_queue",
    "metric_pred_count_bandwidth",
    "metric_recall_delay",
    "metric_recall_loss",
    "metric_recall_cpu",
    "metric_recall_queue",
    "metric_recall_bandwidth",
    "metric_majority_baseline",
    "metric_majority_acc_valid_attr_only",
    "metric_attr_acc_minus_majority",
    "uses_future_scores_as_supervision_only",
    "risk_metric_scores_use_future_information",
    "used_future_scores_as_model_input",
    "inference_time_ms",
]
ATTR_COLUMNS = {
    "node_attr_acc",
    "link_attr_acc",
    "metric_attr_acc",
    "metric_attr_macro_f1",
    "metric_attr_balanced_acc",
    "metric_attr_per_class_recall_delay",
    "metric_attr_per_class_recall_loss",
    "metric_attr_per_class_recall_cpu",
    "metric_attr_per_class_recall_queue",
    "metric_attr_per_class_recall_bandwidth",
    "metric_pred_count_delay",
    "metric_pred_count_loss",
    "metric_pred_count_cpu",
    "metric_pred_count_queue",
    "metric_pred_count_bandwidth",
    "metric_recall_delay",
    "metric_recall_loss",
    "metric_recall_cpu",
    "metric_recall_queue",
    "metric_recall_bandwidth",
    "metric_majority_baseline",
    "metric_majority_acc_valid_attr_only",
    "metric_attr_acc_minus_majority",
}
LABELS = [0, 1, 2]
LABEL_NAMES = {0: "normal", 1: "risky", 2: "violated"}
METRIC_LABELS = [0, 1, 2, 3, 4]
METRIC_NAMES = {0: "delay", 1: "loss", 2: "cpu", 3: "queue", 4: "bandwidth"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_debug.yaml")
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume_dataset_run", default=None)
    parser.add_argument("--variant", default=None)
    return parser.parse_args()


def _format_value(value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _format_row(row: dict) -> dict:
    formatted = {key: _format_value(row.get(key, math.nan)) for key in RESULT_COLUMNS}
    if not bool(row.get("has_attribution_head", False)):
        for key in ATTR_COLUMNS:
            formatted[key] = "N/A"
    formatted["has_attribution_head"] = "true" if bool(row.get("has_attribution_head", False)) else "false"
    return formatted


def write_result(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow(_format_row(row))


def artifact_name(model_name: str, variant: str | None = None) -> str:
    return f"{model_name}_{variant}" if variant else model_name


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def extra_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    supports = {label: sum(int(true == label) for true in y_true) for label in LABELS}
    f1s = {}
    recalls = {}
    for label in LABELS:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = _safe_div(tp, tp + fp)
        recalls[label] = _safe_div(tp, tp + fn)
        f1s[label] = _safe_div(2 * precision * recalls[label], precision + recalls[label])
    total = max(len(y_true), 1)
    return {
        "weighted_f1": sum(f1s[label] * supports[label] for label in LABELS) / total,
        "balanced_acc": sum(recalls.values()) / len(LABELS),
    }


def metric_prediction_diagnostics(labels: dict, preds: dict, has_attr: bool) -> dict[str, float | int]:
    if not has_attr:
        return {
            **{f"metric_pred_count_{name}": math.nan for name in METRIC_NAMES.values()},
            **{f"metric_recall_{name}": math.nan for name in METRIC_NAMES.values()},
            "metric_majority_baseline": math.nan,
            "metric_attr_acc_minus_majority": math.nan,
        }
    valid = [idx for idx, mask in enumerate(labels["attr_mask"]) if int(mask) == 1]
    if not valid:
        return {
            **{f"metric_pred_count_{name}": math.nan for name in METRIC_NAMES.values()},
            **{f"metric_recall_{name}": math.nan for name in METRIC_NAMES.values()},
            "metric_majority_baseline": math.nan,
            "metric_attr_acc_minus_majority": math.nan,
        }
    true_values = [int(labels["risk_metric"][idx]) for idx in valid]
    pred_values = [int(preds["risk_metric"][idx]) for idx in valid]
    pred_counts = Counter(pred_values)
    true_counts = Counter(true_values)
    majority = true_counts.most_common(1)[0][1] / len(valid) if true_counts else math.nan
    correct = sum(int(true == pred) for true, pred in zip(true_values, pred_values))
    acc = correct / len(valid)
    out: dict[str, float | int] = {}
    for metric_id, name in METRIC_NAMES.items():
        support = true_counts.get(metric_id, 0)
        hit = sum(1 for true, pred in zip(true_values, pred_values) if true == metric_id and pred == metric_id)
        out[f"metric_pred_count_{name}"] = pred_counts.get(metric_id, 0)
        out[f"metric_recall_{name}"] = hit / support if support else 0.0
    out["metric_majority_baseline"] = majority
    out["metric_majority_acc_valid_attr_only"] = majority
    out["metric_attr_acc_minus_majority"] = acc - majority
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_classification_artifacts(output_dir: Path, name: str, y_true: list[int], y_pred: list[int]) -> None:
    cm_rows = []
    for true_label in LABELS:
        for pred_label in LABELS:
            cm_rows.append(
                {
                    "true_label": true_label,
                    "true_name": LABEL_NAMES[true_label],
                    "pred_label": pred_label,
                    "pred_name": LABEL_NAMES[pred_label],
                    "count": sum(1 for true, pred in zip(y_true, y_pred) if true == true_label and pred == pred_label),
                }
            )
    write_csv(output_dir / f"{name}_confusion_matrix.csv", cm_rows)

    pred_counts = Counter(y_pred)
    total = max(len(y_pred), 1)
    distribution_rows = [
        {
            "pred_label": label,
            "pred_name": LABEL_NAMES[label],
            "count": pred_counts.get(label, 0),
            "ratio": pred_counts.get(label, 0) / total,
        }
        for label in LABELS
    ]
    write_csv(output_dir / f"{name}_prediction_distribution.csv", distribution_rows)


def update_all_results(output_dir: Path) -> None:
    rows: list[dict] = []
    for model_name in ["lstm", "transformer", "sparta"]:
        path = output_dir / f"{model_name}_test_results.csv"
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    method = row.get("method", model_name)
                    has_attr = str(row.get("has_attribution_head", "")).lower() == "true" or method == "sparta"
                    row["method"] = method
                    row["has_attribution_head"] = "true" if has_attr else "false"
                    if not has_attr:
                        for key in ATTR_COLUMNS:
                            row[key] = "N/A"
                    rows.append(row)
    if rows:
        with (output_dir / "all_test_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in RESULT_COLUMNS})


@torch.no_grad()
def _evaluate_with_predictions(
    config: dict,
    model_name: str,
    checkpoint_path: str | Path | None = None,
    variant: str | None = None,
) -> tuple[dict, dict, dict, Path, Path]:
    set_seed(int(config.get("seed", 42)))
    prepare_metric_evidence_stats(config, None, write=False)
    device = select_device(config)
    model = build_model(config, model_name).to(device)
    name = artifact_name(model_name, variant)
    ckpt_path = Path(checkpoint_path) if checkpoint_path else resolve_path(config, config["project"]["checkpoint_dir"]) / f"{name}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_path = resolve_path(config, config["data"].get("test_path", Path(config["data"]["dataset_dir"]) / "test.pkl"))
    dataset = SPARTADataset(test_path, config)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 0)),
    )
    start = time.perf_counter()
    labels, preds, has_attr = collect_predictions(model, loader, device)
    elapsed = time.perf_counter() - start
    metrics = compute_classification_metrics(labels["risk_label"], preds["risk_label"], preds.get("risk_probs"))
    metrics.update(extra_classification_metrics(labels["risk_label"], preds["risk_label"]))
    metrics.update(compute_attribution_metrics(labels, preds, has_attr))
    metrics.update(metric_prediction_diagnostics(labels, preds, has_attr))
    metrics["method"] = model_name
    metrics["model"] = model_name
    metrics["variant"] = variant or ""
    metrics["has_attribution_head"] = bool(has_attr)
    metrics["checkpoint_path"] = str(ckpt_path)
    metrics["dataset_path"] = str(test_path)
    metrics["result_source_file"] = ""
    metrics["inference_time_ms"] = (elapsed / max(len(dataset), 1)) * 1000.0
    loss_cfg = config.get("loss", {})
    uses_soft = float(loss_cfg.get("lambda_metric_soft", 0.0)) > 0.0
    metrics["uses_future_scores_as_supervision_only"] = bool(uses_soft)
    metrics["risk_metric_scores_use_future_information"] = bool(uses_soft)
    metrics["used_future_scores_as_model_input"] = False
    return metrics, labels, preds, ckpt_path, test_path


@torch.no_grad()
def evaluate_once(config: dict, model_name: str, checkpoint_path: str | Path | None = None) -> dict:
    metrics, _labels, _preds, _ckpt_path, _test_path = _evaluate_with_predictions(config, model_name, checkpoint_path)
    return metrics


def main() -> None:
    args = parse_args()
    variant = normalize_variant(args.variant)
    config = load_config(args.config)
    config = apply_resume_dataset_run(config, args.resume_dataset_run, variant)
    row, labels, preds, _ckpt_path, _test_path = _evaluate_with_predictions(config, args.model, args.checkpoint, variant)
    output_dir = resolve_path(config, config["project"]["output_dir"])
    name = artifact_name(args.model, variant)
    out_path = output_dir / f"{name}_test_results.csv"
    row["result_source_file"] = str(out_path)
    write_result(out_path, row)
    write_classification_artifacts(output_dir, name, labels["risk_label"], preds["risk_label"])
    if not variant:
        update_all_results(output_dir)
    print(f"Wrote evaluation result to {out_path}: {row}")


if __name__ == "__main__":
    main()
