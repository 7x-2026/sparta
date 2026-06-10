from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        recall_score,
        roc_auc_score,
    )

    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local binary packages
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sparta_dataset import SPARTADataset
from src.train import build_model, select_device
from src.utils.config import load_config, resolve_path
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


MODELS = ["lstm", "transformer", "sparta"]
LABELS = [0, 1, 2]
LABEL_NAMES = {0: "normal", 1: "risky", 2: "violated"}
METRIC_NAMES = {0: "delay", 1: "loss", 2: "cpu", 3: "queue", 4: "bandwidth"}
RESULT_COLUMNS = [
    "run_id",
    "model",
    "has_attribution_head",
    "checkpoint_path",
    "checkpoint_mtime",
    "dataset_path",
    "result_source_file",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_acc",
    "risk_recall",
    "violation_recall",
    "auc",
    "node_attr_acc",
    "link_attr_acc",
    "metric_attr_acc",
    "inference_time_ms",
]
ATTR_COLUMNS = {"node_attr_acc", "link_attr_acc", "metric_attr_acc"}


if not SKLEARN_AVAILABLE:

    def accuracy_score(y_true, y_pred):
        return sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)

    def precision_recall_fscore_support(y_true, y_pred, labels, zero_division=0):
        precisions, recalls, f1s, supports = [], [], [], []
        for label in labels:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            support = sum(1 for t in y_true if t == label)
            precision = tp / (tp + fp) if (tp + fp) else float(zero_division)
            recall = tp / (tp + fn) if (tp + fn) else float(zero_division)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float(zero_division)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            supports.append(support)
        return precisions, recalls, f1s, supports

    def f1_score(y_true, y_pred, labels, average, zero_division=0):
        _, _, f1s, supports = precision_recall_fscore_support(y_true, y_pred, labels, zero_division)
        if average == "macro":
            return sum(f1s) / max(len(f1s), 1)
        if average == "weighted":
            total = sum(supports)
            return sum(f1 * support for f1, support in zip(f1s, supports)) / total if total else 0.0
        raise ValueError(f"Unsupported average={average}")

    def recall_score(y_true, y_pred, labels, average=None, zero_division=0):
        _, recalls, _, supports = precision_recall_fscore_support(y_true, y_pred, labels, zero_division)
        if average is None:
            return recalls
        if average == "macro":
            return sum(recalls) / max(len(recalls), 1)
        if average == "weighted":
            total = sum(supports)
            return sum(recall * support for recall, support in zip(recalls, supports)) / total if total else 0.0
        raise ValueError(f"Unsupported average={average}")

    def balanced_accuracy_score(y_true, y_pred):
        return recall_score(y_true, y_pred, LABELS, average="macro", zero_division=0)

    def confusion_matrix(y_true, y_pred, labels):
        matrix = [[0 for _ in labels] for _ in labels]
        index = {label: idx for idx, label in enumerate(labels)}
        for true, pred in zip(y_true, y_pred):
            if true in index and pred in index:
                matrix[index[true]][index[pred]] += 1
        return matrix

    def _binary_auc(y_true_binary: list[int], y_score: list[float]) -> float:
        pairs = sorted(zip(y_score, y_true_binary), key=lambda item: item[0])
        n_pos = sum(y_true_binary)
        n_neg = len(y_true_binary) - n_pos
        if n_pos == 0 or n_neg == 0:
            return math.nan
        rank_sum = 0.0
        i = 0
        while i < len(pairs):
            j = i + 1
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
            i = j
        return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    def roc_auc_score(y_true, y_score, multi_class, average, labels):
        if multi_class != "ovr" or average != "macro":
            raise ValueError("Fallback roc_auc_score only supports multi_class='ovr', average='macro'")
        aucs = []
        for label in labels:
            binary = [1 if true == label else 0 for true in y_true]
            scores = [row[label] for row in y_score]
            aucs.append(_binary_auc(binary, scores))
        return sum(aucs) / len(aucs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--backup_existing", action="store_true", default=True)
    return parser.parse_args()


def format_value(value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def format_result_row(row: dict) -> dict:
    out = {key: format_value(row.get(key, math.nan)) for key in RESULT_COLUMNS}
    has_attr = bool(row.get("has_attribution_head", False))
    out["has_attribution_head"] = "true" if has_attr else "false"
    if not has_attr:
        for key in ATTR_COLUMNS:
            out[key] = "N/A"
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config" / "resolved_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    return load_config(config_path)


def backup_existing_outputs(run_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ensure_dir(run_dir / "artifacts" / f"eval_supplement_backup_{timestamp}")
    for subdir in ["results", "logs"]:
        src_dir = run_dir / subdir
        if not src_dir.exists():
            continue
        for path in src_dir.glob("*"):
            if path.is_file() and (path.suffix.lower() in {".csv", ".log", ".txt"}):
                dst = backup_dir / subdir / path.name
                ensure_dir(dst.parent)
                shutil.copy2(path, dst)
    return backup_dir


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def collect_eval_outputs(config: dict, model_name: str, checkpoint_path: Path, dataset_path: Path) -> dict:
    set_seed(int(config.get("seed", 42)))
    device = select_device(config)
    model = build_model(copy.deepcopy(config), model_name).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SPARTADataset(dataset_path, config)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 0)),
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[list[float]] = []
    labels = {"risk_node": [], "risk_link": [], "risk_metric": [], "attr_mask": []}
    preds = {"risk_node": [], "risk_link": [], "risk_metric": []}
    has_attr = False
    start = time.perf_counter()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        logits = outputs["risk_logits"]
        probs = torch.softmax(logits, dim=-1)
        y_score.extend(probs.cpu().tolist())
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
        y_true.extend(batch["risk_label"].cpu().tolist())
        labels["risk_node"].extend(batch["risk_node"].cpu().tolist())
        labels["risk_link"].extend(batch["risk_link"].cpu().tolist())
        labels["risk_metric"].extend(batch["risk_metric"].cpu().tolist())
        labels["attr_mask"].extend(batch["attr_mask"].cpu().tolist())
        if all(key in outputs for key in ["node_logits", "link_logits", "metric_logits"]):
            has_attr = True
            preds["risk_node"].extend(outputs["node_logits"].argmax(dim=-1).cpu().tolist())
            preds["risk_link"].extend(outputs["link_logits"].argmax(dim=-1).cpu().tolist())
            preds["risk_metric"].extend(outputs["metric_logits"].argmax(dim=-1).cpu().tolist())
        else:
            n = batch["risk_label"].shape[0]
            preds["risk_node"].extend([-100] * n)
            preds["risk_link"].extend([-100] * n)
            preds["risk_metric"].extend([-100] * n)
    elapsed = time.perf_counter() - start
    return {
        "y_true": [int(x) for x in y_true],
        "y_pred": [int(x) for x in y_pred],
        "y_score": y_score,
        "labels": labels,
        "preds": preds,
        "has_attr": has_attr,
        "inference_time_ms": (elapsed / max(len(dataset), 1)) * 1000.0,
    }


def safe_auc(y_true: list[int], y_score: list[list[float]]) -> float:
    if set(y_true) != set(LABELS):
        return math.nan
    if not y_score or len(y_score[0]) != 3:
        raise RuntimeError(f"Expected y_score shape [N,3], got N={len(y_score)}")
    auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro", labels=LABELS)
    if math.isnan(float(auc)):
        raise RuntimeError("AUC is nan even though y_true contains all three classes")
    return float(auc)


def attribution_accuracy(labels: dict, preds: dict, field: str, has_attr: bool) -> float:
    if not has_attr:
        return math.nan
    valid = [idx for idx, mask in enumerate(labels["attr_mask"]) if int(mask) == 1]
    if not valid:
        return math.nan
    return sum(int(labels[field][idx] == preds[field][idx]) for idx in valid) / len(valid)


def build_result_row(run_dir: Path, model_name: str, checkpoint_path: Path, dataset_path: Path, eval_data: dict, result_path: Path) -> dict:
    y_true = eval_data["y_true"]
    y_pred = eval_data["y_pred"]
    row = {
        "run_id": run_dir.name,
        "model": model_name,
        "has_attribution_head": bool(eval_data["has_attr"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_mtime": checkpoint_path.stat().st_mtime,
        "dataset_path": str(dataset_path),
        "result_source_file": str(result_path),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "risk_recall": recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)[1],
        "violation_recall": recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)[2],
        "auc": safe_auc(y_true, eval_data["y_score"]),
        "node_attr_acc": attribution_accuracy(eval_data["labels"], eval_data["preds"], "risk_node", eval_data["has_attr"]),
        "link_attr_acc": attribution_accuracy(eval_data["labels"], eval_data["preds"], "risk_link", eval_data["has_attr"]),
        "metric_attr_acc": attribution_accuracy(eval_data["labels"], eval_data["preds"], "risk_metric", eval_data["has_attr"]),
        "inference_time_ms": eval_data["inference_time_ms"],
    }
    return row


def write_classification_artifacts(run_dir: Path, model_name: str, y_true: list[int], y_pred: list[int]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_rows = [
        {
            "true_label": true_label,
            "true_name": LABEL_NAMES[true_label],
            "pred_label": pred_label,
            "pred_name": LABEL_NAMES[pred_label],
            "count": int(cm[i][j]),
        }
        for i, true_label in enumerate(LABELS)
        for j, pred_label in enumerate(LABELS)
    ]
    write_csv(run_dir / "results" / f"{model_name}_confusion_matrix.csv", cm_rows)

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
    write_csv(run_dir / "results" / f"{model_name}_prediction_distribution.csv", distribution_rows)

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    precision_rows = [{"class_id": label, "class_name": LABEL_NAMES[label], "precision": float(precision[i]), "support": int(support[i])} for i, label in enumerate(LABELS)]
    recall_rows = [{"class_id": label, "class_name": LABEL_NAMES[label], "recall": float(recall[i]), "support": int(support[i])} for i, label in enumerate(LABELS)]
    f1_rows = [{"class_id": label, "class_name": LABEL_NAMES[label], "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(LABELS)]
    write_csv(run_dir / "results" / f"{model_name}_per_class_precision.csv", precision_rows)
    write_csv(run_dir / "results" / f"{model_name}_per_class_recall.csv", recall_rows)
    write_csv(run_dir / "results" / f"{model_name}_per_class_f1.csv", f1_rows)


def write_sparta_attribution_artifacts(run_dir: Path, eval_data: dict) -> None:
    labels = eval_data["labels"]
    preds = eval_data["preds"]
    valid = [idx for idx, mask in enumerate(labels["attr_mask"]) if int(mask) == 1]
    baseline_rows = []
    per_class_rows = []
    fields = [
        ("risk_node", "node", None),
        ("risk_link", "link", None),
        ("risk_metric", "metric", METRIC_NAMES),
    ]
    for field, field_name, names in fields:
        values = [int(labels[field][idx]) for idx in valid]
        total = len(values)
        counts = Counter(values)
        majority_value, majority_count = counts.most_common(1)[0] if counts else ("", 0)
        baseline_rows.append(
            {
                "field": field_name,
                "majority_value": majority_value,
                "majority_name": names.get(majority_value, str(majority_value)) if names and majority_value != "" else str(majority_value),
                "majority_count": majority_count,
                "valid_attr_samples": total,
                "majority_acc": majority_count / total if total else math.nan,
            }
        )
        for value in sorted(counts):
            class_indices = [idx for idx in valid if int(labels[field][idx]) == value]
            correct = sum(int(labels[field][idx] == preds[field][idx]) for idx in class_indices)
            per_class_rows.append(
                {
                    "field": field_name,
                    "class_value": value,
                    "class_name": names.get(value, str(value)) if names else str(value),
                    "support": len(class_indices),
                    "correct": correct,
                    "accuracy": correct / len(class_indices) if class_indices else math.nan,
                }
            )
    write_csv(run_dir / "results" / "sparta_attribution_majority_baseline.csv", baseline_rows)
    write_csv(run_dir / "results" / "sparta_attribution_per_class.csv", per_class_rows)


def write_eval_log(run_dir: Path, model_name: str, row: dict) -> None:
    log_path = run_dir / "logs" / f"{model_name}_eval.log"
    ensure_dir(log_path.parent)
    formatted = format_result_row(row)
    with log_path.open("w", encoding="utf-8") as f:
        for key in RESULT_COLUMNS:
            f.write(f"{key}={formatted.get(key, '')}\n")


def read_single_model_results(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for model_name in MODELS:
        path = run_dir / "results" / f"{model_name}_test_results.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing single-model result: {path}")
        with path.open("r", newline="", encoding="utf-8") as f:
            model_rows = list(csv.DictReader(f))
        if len(model_rows) != 1:
            raise RuntimeError(f"Expected exactly one row in {path}, got {len(model_rows)}")
        rows.append(model_rows[0])
    return rows


def update_manifest(run_dir: Path, backup_dir: Path) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = list(manifest.get("completed_stages", []))
    if "supplement_eval_outputs" not in completed:
        completed.append("supplement_eval_outputs")
    manifest.update(
        {
            "status": "completed",
            "completed_stages": completed,
            "missing_outputs": [],
            "error": None,
            "last_eval_supplement_backup": str(backup_dir),
            "last_eval_supplemented_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = load_run_config(run_dir)
    dataset_path = resolve_path(config, config["data"].get("test_path", Path(config["data"]["dataset_dir"]) / "test.pkl"))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {dataset_path}")
    backup_dir = backup_existing_outputs(run_dir) if args.backup_existing else run_dir / "artifacts"
    print(f"Backed up existing eval outputs to {backup_dir}")

    for model_name in MODELS:
        checkpoint_path = resolve_path(config, config["project"]["checkpoint_dir"]) / f"{model_name}_best.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        eval_data = collect_eval_outputs(config, model_name, checkpoint_path, dataset_path)
        result_path = run_dir / "results" / f"{model_name}_test_results.csv"
        row = build_result_row(run_dir, model_name, checkpoint_path, dataset_path, eval_data, result_path)
        write_csv(result_path, [format_result_row(row)], RESULT_COLUMNS)
        write_classification_artifacts(run_dir, model_name, eval_data["y_true"], eval_data["y_pred"])
        if model_name == "sparta":
            write_sparta_attribution_artifacts(run_dir, eval_data)
        write_eval_log(run_dir, model_name, row)
        print(f"Wrote {result_path}")

    all_rows = read_single_model_results(run_dir)
    write_csv(run_dir / "results" / "all_test_results.csv", all_rows, RESULT_COLUMNS)
    update_manifest(run_dir, backup_dir)
    print(f"Wrote {run_dir / 'results' / 'all_test_results.csv'}")


if __name__ == "__main__":
    main()
