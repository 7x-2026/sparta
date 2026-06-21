from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sparta_dataset import SPARTADataset
from src.metrics import compute_attribution_majority_baseline
from src.models.sparta import (
    DEFAULT_FEATURE_SCHEMA,
    build_metric_evidence,
    compute_metric_evidence_stats_from_loader,
    load_metric_evidence_stats,
    save_metric_evidence_stats,
)
from src.preprocessing.build_path_graph import load_logs
from src.preprocessing.generate_labels import compute_metric_risk_scores
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_pickle


METRIC_NAMES = {0: "delay", 1: "loss", 2: "cpu", 3: "queue", 4: "bandwidth"}
DEFAULT_EVIDENCE_NORMS = ["raw", "zscore", "pressure01", "pressure02"]
EVIDENCE_NORMS = DEFAULT_EVIDENCE_NORMS + ["labelrule"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--evidence_norm", choices=EVIDENCE_NORMS + ["all"], default="all")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--debug_topk", type=int, default=0)
    return parser.parse_args()


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
    config = load_config(config_path)
    dataset_dir = run_dir / "dataset"
    config.setdefault("data", {})
    config["data"]["dataset_dir"] = str(dataset_dir)
    config["data"]["train_path"] = str(dataset_dir / "train.pkl")
    config["data"]["val_path"] = str(dataset_dir / "val.pkl")
    config["data"]["test_path"] = str(dataset_dir / "test.pkl")
    return config


def schema_from_config(config: dict) -> dict[str, list[str]]:
    data_cfg = config.get("data", {})
    schema = {}
    for group, defaults in DEFAULT_FEATURE_SCHEMA.items():
        names = data_cfg.get(f"{group}_feature_names") or data_cfg.get(f"{group}_feat_names") or defaults
        schema[group] = [str(name) for name in names]
    return schema


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    total = len(y_true)
    f1s = []
    recalls = {}
    for cls, name in METRIC_NAMES.items():
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred == cls)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != cls and pred == cls)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == cls and pred != cls)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        recalls[name] = recall
        f1s.append(safe_div(2 * precision * recall, precision + recall))
    return {
        "metric_evidence_acc": safe_div(sum(int(t == p) for t, p in zip(y_true, y_pred)), total),
        "metric_evidence_macro_f1": sum(f1s) / len(f1s),
        "metric_evidence_balanced_acc": sum(recalls.values()) / len(recalls),
        "per_class_recall": recalls,
    }


def confusion_rows(y_true: list[int], y_pred: list[int]) -> list[dict]:
    rows = []
    for true_id, true_name in METRIC_NAMES.items():
        for pred_id, pred_name in METRIC_NAMES.items():
            rows.append(
                {
                    "true_metric_id": true_id,
                    "true_metric": true_name,
                    "pred_metric_id": pred_id,
                    "pred_metric": pred_name,
                    "count": sum(1 for true, pred in zip(y_true, y_pred) if true == true_id and pred == pred_id),
                }
            )
    return rows


def metric_stats_path(run_dir: Path, evidence_norm: str = "pressure01") -> Path:
    if evidence_norm == "pressure02":
        return run_dir / "artifacts" / "metric_evidence_stats_pressure02.json"
    return run_dir / "artifacts" / "metric_evidence_stats.json"


def expected_stats_version(evidence_norm: str) -> str:
    return "pressure02_v5_growth_rank" if evidence_norm == "pressure02" else "pressure01_v3_excess"


def load_or_compute_pressure_stats(run_dir: Path, config: dict, schema: dict, batch_size: int, evidence_norm: str) -> dict:
    path = metric_stats_path(run_dir, evidence_norm)
    if path.exists():
        model_cfg = config.setdefault("model", {})
        model_cfg["metric_evidence_stats_path"] = str(path)
        loaded = load_metric_evidence_stats(config)
        has_expected_stats = (
            isinstance(loaded, dict)
            and loaded.get("version") == expected_stats_version(evidence_norm)
            and isinstance(loaded.get("evidence_raw"), dict)
            and all(name in loaded["evidence_raw"] for name in METRIC_NAMES.values())
        )
        if has_expected_stats:
            return loaded

    train_path = run_dir / "dataset" / "train.pkl"
    if not train_path.exists():
        raise FileNotFoundError(f"{evidence_norm} requires train.pkl to compute stats, missing: {train_path}")
    train_ds = SPARTADataset(train_path, config)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    stats = compute_metric_evidence_stats_from_loader(train_loader, schema, norm=evidence_norm)
    save_metric_evidence_stats(path, stats)
    config.setdefault("model", {})["metric_evidence_stats_path"] = str(path)
    return stats


def tensor_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": math.nan, "std": math.nan, "p50": math.nan, "p90": math.nan, "p95": math.nan, "max": math.nan}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "max": float(tensor.max().item()),
    }


@torch.no_grad()
def collect_predictions(run_dir: Path, config: dict, evidence_norm: str, batch_size: int) -> tuple[list[int], list[int], dict | None, list[dict], list[list[float]], list[int]]:
    test_path = run_dir / "dataset" / "test.pkl"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {test_path}")
    schema = schema_from_config(config)
    evidence_stats = None
    if evidence_norm in {"pressure01", "pressure02"}:
        evidence_stats = load_or_compute_pressure_stats(run_dir, config, schema, batch_size, evidence_norm)
    dataset = SPARTADataset(test_path, config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    norm = "none" if evidence_norm == "raw" else evidence_norm
    y_true: list[int] = []
    y_pred: list[int] = []
    evidence_rows: list[list[float]] = []
    sample_indices: list[int] = []
    offset = 0
    for batch in loader:
        evidence = build_metric_evidence(batch, schema, norm=norm, evidence_stats=evidence_stats)
        attr_mask = batch["attr_mask"].bool()
        if attr_mask.any():
            y_true.extend(int(x) for x in batch["risk_metric"][attr_mask].cpu().tolist())
            y_pred.extend(int(x) for x in evidence.argmax(dim=-1)[attr_mask].cpu().tolist())
            evidence_rows.extend([[float(v) for v in row] for row in evidence[attr_mask].cpu().tolist()])
            selected = torch.nonzero(attr_mask, as_tuple=False).reshape(-1).cpu().tolist()
            sample_indices.extend([offset + int(idx) for idx in selected])
        offset += int(batch["risk_metric"].shape[0])
    return y_true, y_pred, evidence_stats, dataset.samples, evidence_rows, sample_indices


def collect_labelrule_predictions(run_dir: Path, config: dict) -> tuple[list[int], list[int], None, list[dict], list[list[float]], list[int]]:
    test_path = run_dir / "dataset" / "test.pkl"
    raw_dir = run_dir / "raw_logs"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {test_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"labelrule evidence requires raw logs, missing: {raw_dir}")
    samples = load_pickle(test_path)
    logs = load_logs(raw_dir)
    y_true: list[int] = []
    y_pred: list[int] = []
    evidence_rows: list[list[float]] = []
    sample_indices: list[int] = []
    for sample_index, sample in enumerate(samples):
        if int(sample.get("attr_mask", 0)) != 1:
            continue
        info = compute_metric_risk_scores(sample, logs, config)
        scores = [float(x) for x in info["scores"]]
        y_true.append(int(sample["risk_metric"]))
        y_pred.append(int(info["label_rule_argmax"]))
        evidence_rows.append(scores)
        sample_indices.append(sample_index)
    return y_true, y_pred, None, samples, evidence_rows, sample_indices


def evidence_dimension_summary(evidence_rows: list[list[float]], y_pred: list[int]) -> dict:
    out = {}
    for idx, name in METRIC_NAMES.items():
        values = [row[idx] for row in evidence_rows]
        stats = tensor_stats(values)
        stats["pred_count"] = int(sum(1 for pred in y_pred if pred == idx))
        out[name] = stats
    return out


def top_predicted_class_ratio(y_pred: list[int]) -> float:
    if not y_pred:
        return math.nan
    return max(Counter(y_pred).values()) / len(y_pred)


def debug_cases(samples: list[dict], sample_indices: list[int], y_true: list[int], y_pred: list[int], evidence_rows: list[list[float]], topk: int) -> list[dict]:
    if topk <= 0:
        return []
    ranked = []
    for pos, row in enumerate(evidence_rows):
        sorted_vals = sorted(row, reverse=True)
        margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else sorted_vals[0]
        ranked.append((margin, pos))
    rows = []
    for _margin, pos in sorted(ranked, reverse=True)[:topk]:
        sample = samples[sample_indices[pos]]
        rows.append(
            {
                "sample_index": sample_indices[pos],
                "service_id": sample.get("service_id", ""),
                "time": sample.get("time", ""),
                "scenario": sample.get("scenario", ""),
                "true_metric": METRIC_NAMES.get(y_true[pos], str(y_true[pos])),
                "pred_metric": METRIC_NAMES.get(y_pred[pos], str(y_pred[pos])),
                "metric_evidence": evidence_rows[pos],
            }
        )
    return rows


def evaluate_one(run_dir: Path, config: dict, evidence_norm: str, batch_size: int, debug_topk: int = 0) -> tuple[dict, list[dict], dict | None]:
    if evidence_norm == "labelrule":
        y_true, y_pred, evidence_stats, samples, evidence_rows, sample_indices = collect_labelrule_predictions(run_dir, config)
    else:
        y_true, y_pred, evidence_stats, samples, evidence_rows, sample_indices = collect_predictions(run_dir, config, evidence_norm, batch_size)
    result = compute_metrics(y_true, y_pred)
    pred_counts = Counter(y_pred)
    majority = compute_attribution_majority_baseline(samples, split="test", attr_mask_only=True)
    majority_baseline = majority["metric_majority_acc_valid_attr_only"]
    result.update(
        {
            "run_dir": str(run_dir),
            "evidence_norm": evidence_norm,
            "valid_attr_samples": len(y_true),
            "pred_count_delay": pred_counts.get(0, 0),
            "pred_count_loss": pred_counts.get(1, 0),
            "pred_count_cpu": pred_counts.get(2, 0),
            "pred_count_queue": pred_counts.get(3, 0),
            "pred_count_bandwidth": pred_counts.get(4, 0),
            "majority_baseline": majority_baseline,
            "metric_majority_acc_valid_attr_only": majority["metric_majority_acc_valid_attr_only"],
            "metric_majority_acc_including_normal": majority["metric_majority_acc_including_normal"],
            "metric_majority_count_by_class": majority["metric_majority_count_by_class"],
            "metric_majority_count_by_class_named": majority["metric_majority_count_by_class_named"],
            "metric_majority_name": majority["metric_majority_name"],
            "denominator": majority["denominator"],
            "acc_minus_majority": result["metric_evidence_acc"] - majority_baseline if y_true else math.nan,
            "metric_evidence_stats_path": str(metric_stats_path(run_dir, evidence_norm)) if evidence_norm in {"pressure01", "pressure02"} else "",
            "metric_evidence_stats_source": "train.pkl_or_existing_artifact" if evidence_norm in {"pressure01", "pressure02"} else "",
            "evidence_stats_loaded": evidence_stats is not None,
            "evidence_dimension_summary": evidence_dimension_summary(evidence_rows, y_pred),
            "top_predicted_class_ratio": top_predicted_class_ratio(y_pred),
        }
    )
    if evidence_norm == "labelrule":
        result.update(
            {
                "labelrule_evidence_acc": result["metric_evidence_acc"],
                "labelrule_evidence_macro_f1": result["metric_evidence_macro_f1"],
                "labelrule_evidence_balanced_acc": result["metric_evidence_balanced_acc"],
                "uses_future_information": True,
                "allowed_for_model_input": False,
                "future_information_note": (
                    "labelrule reproduces generate_labels.py horizon-based risk_metric scores and is diagnostic only."
                ),
            }
        )
    calibration = None
    if evidence_norm in {"pressure01", "pressure02"}:
        calibration = {
            "run_dir": str(run_dir),
            "evidence_norm": evidence_norm,
            "valid_attr_samples": len(y_true),
            "metrics": {
                "metric_evidence_acc": result["metric_evidence_acc"],
                "metric_evidence_macro_f1": result["metric_evidence_macro_f1"],
                "metric_evidence_balanced_acc": result["metric_evidence_balanced_acc"],
                "per_class_recall": result["per_class_recall"],
            },
            "evidence_dimension_summary": result["evidence_dimension_summary"],
            "pred_counts": {
                name: int(sum(1 for pred in y_pred if pred == idx))
                for idx, name in METRIC_NAMES.items()
            },
            "majority_baseline": result["majority_baseline"],
            "acc_minus_majority": result["acc_minus_majority"],
            "metric_evidence_stats_path": result["metric_evidence_stats_path"],
            "top_predicted_class_ratio": result["top_predicted_class_ratio"],
            "debug_topk": debug_cases(samples, sample_indices, y_true, y_pred, evidence_rows, debug_topk),
        }
    return result, confusion_rows(y_true, y_pred), calibration


def write_outputs(run_dir: Path, evidence_norm: str, result: dict, confusion: list[dict], calibration: dict | None = None) -> None:
    analysis_dir = ensure_dir(run_dir / "analysis")
    json_path = analysis_dir / f"metric_evidence_baseline_{evidence_norm}.json"
    confusion_path = analysis_dir / f"metric_evidence_baseline_{evidence_norm}_confusion.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(confusion_path, confusion)
    print(f"Wrote {json_path}")
    print(f"Wrote {confusion_path}")
    if calibration is not None:
        calibration_path = analysis_dir / f"metric_evidence_{evidence_norm}_calibration.json"
        calibration_path.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {calibration_path}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = load_run_config(run_dir)
    batch_size = int(args.batch_size or config.get("train", {}).get("batch_size", 64))
    norms = DEFAULT_EVIDENCE_NORMS if args.evidence_norm == "all" else [args.evidence_norm]
    for norm in norms:
        result, confusion, calibration = evaluate_one(run_dir, config, norm, batch_size, debug_topk=args.debug_topk)
        write_outputs(run_dir, norm, result, confusion, calibration)


if __name__ == "__main__":
    main()
