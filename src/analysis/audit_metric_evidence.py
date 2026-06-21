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
from src.models.sparta import DEFAULT_FEATURE_SCHEMA, build_metric_evidence, load_metric_evidence_stats
from src.train import build_model, select_device
from src.utils.config import load_config
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


METRIC_NAMES = {0: "delay", 1: "loss", 2: "cpu", 3: "queue", 4: "bandwidth"}
REQUIRED_FEATURES = {
    "node": {
        "cpu_util": ["cpu_util", "cpu"],
        "available_cpu": ["available_cpu", "free_cpu"],
        "queue_len": ["queue_len", "queue"],
    },
    "link": {
        "delay": ["delay_ms", "delay"],
        "loss": ["loss", "packet_loss"],
        "bandwidth_util": ["bandwidth_util", "bw_util"],
        "queue_delay": ["queue_delay_ms", "queue_delay"],
    },
    "service": {
        "response_time": ["response_time", "path_delay"],
        "request_rate": ["request_rate"],
    },
    "sla": {
        "max_delay": ["max_delay"],
        "max_loss": ["max_loss"],
        "min_bandwidth": ["min_bandwidth"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--variant", default="metricheadv2")
    parser.add_argument("--max_cases", type=int, default=50)
    return parser.parse_args()


def normalize_variant(variant: str | None) -> str | None:
    if variant is None:
        return None
    text = str(variant).strip()
    if not text:
        return None
    if any(ch in text for ch in ["/", "\\", ":"]):
        raise ValueError(f"Invalid variant name: {variant}")
    return text


def sparta_artifact_name(variant: str | None) -> str:
    return f"sparta_{variant}" if variant else "sparta"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def load_run_config(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    candidates = [
        run_dir / "config" / "resolved_config.yaml",
        Path(str(manifest.get("resolved_config_path", ""))),
    ]
    for path in candidates:
        if path and path.exists():
            return load_config(path)
    raise FileNotFoundError(f"Missing resolved config under {run_dir / 'config'}")


def schema_from_config(config: dict) -> dict[str, list[str]]:
    data_cfg = config.get("data", {})
    schema = {}
    for group, defaults in DEFAULT_FEATURE_SCHEMA.items():
        names = data_cfg.get(f"{group}_feature_names") or data_cfg.get(f"{group}_feat_names") or defaults
        schema[group] = [str(name) for name in names]
    return schema


def find_feature_index(names: list[str], aliases: list[str]) -> int | None:
    lowered = [name.lower() for name in names]
    alias_set = {alias.lower() for alias in aliases}
    for idx, name in enumerate(lowered):
        if name in alias_set or any(alias in name for alias in alias_set):
            return idx
    return None


def check_feature_schema(config: dict) -> dict:
    data_cfg = config.get("data", {})
    schema = schema_from_config(config)
    warnings = []
    checks = {}
    for group, required in REQUIRED_FEATURES.items():
        names_from_config = data_cfg.get(f"{group}_feature_names") or data_cfg.get(f"{group}_feat_names")
        if not names_from_config:
            warnings.append(f"{group}_feature_names missing in config; using default SPARTA feature schema.")
        checks[group] = {}
        for canonical, aliases in required.items():
            idx = find_feature_index(schema.get(group, []), aliases)
            checks[group][canonical] = {
                "found": idx is not None,
                "index": idx,
                "aliases": aliases,
                "feature_names": schema.get(group, []),
            }
            if idx is None:
                warnings.append(f"Missing or unmapped feature: {group}.{canonical} aliases={aliases}")
    return {"schema": schema, "checks": checks, "warnings": warnings}


def load_checkpoint_config(run_dir: Path, variant: str | None, fallback_config: dict) -> tuple[dict, Path, dict]:
    checkpoint_path = run_dir / "checkpoints" / f"{sparta_artifact_name(variant)}_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("config", fallback_config) if isinstance(checkpoint, dict) else fallback_config
    return model_config, checkpoint_path, checkpoint


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


@torch.no_grad()
def collect_evidence_and_predictions(run_dir: Path, config: dict, variant: str | None) -> dict:
    test_path = run_dir / "dataset" / "test.pkl"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {test_path}")

    model_config, checkpoint_path, checkpoint = load_checkpoint_config(run_dir, variant, config)
    set_seed(int(model_config.get("seed", 42)))
    device = select_device(model_config)
    model = build_model(model_config, "sparta").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SPARTADataset(test_path, model_config)
    batch_size = int(model_config.get("train", {}).get("batch_size", config.get("train", {}).get("batch_size", 64)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    schema = schema_from_config(model_config)
    norm = str(model_config.get("model", {}).get("metric_evidence_norm", "zscore"))
    evidence_stats = load_metric_evidence_stats(model_config)
    if norm == "pressure01" and evidence_stats is None:
        raise FileNotFoundError(
            "metric_evidence_norm=pressure01 requires train evidence stats. "
            "Run variant training first so artifacts/metric_evidence_stats.json is created."
        )

    true_metric: list[int] = []
    risk_label: list[int] = []
    attr_mask: list[int] = []
    pred_metric: list[int] = []
    evidence_rows: list[list[float]] = []
    raw_evidence_rows: list[list[float]] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        evidence = build_metric_evidence(batch, schema, norm=norm, evidence_stats=evidence_stats)
        raw_evidence = build_metric_evidence(batch, schema, norm="none")
        true_metric.extend(batch["risk_metric"].cpu().tolist())
        risk_label.extend(batch["risk_label"].cpu().tolist())
        attr_mask.extend(batch["attr_mask"].cpu().int().tolist())
        pred_metric.extend(outputs["metric_logits"].argmax(dim=-1).cpu().tolist())
        evidence_rows.extend(evidence.cpu().tolist())
        raw_evidence_rows.extend(raw_evidence.cpu().tolist())

    return {
        "samples": dataset.samples,
        "test_path": test_path,
        "checkpoint_path": checkpoint_path,
        "model_config": model_config,
        "feature_schema": schema,
        "evidence_norm": norm,
        "evidence_stats": evidence_stats,
        "true_metric": np.asarray(true_metric, dtype=np.int64),
        "risk_label": np.asarray(risk_label, dtype=np.int64),
        "attr_mask": np.asarray(attr_mask, dtype=np.int64),
        "pred_metric": np.asarray(pred_metric, dtype=np.int64),
        "evidence": np.asarray(evidence_rows, dtype=np.float64),
        "raw_evidence": np.asarray(raw_evidence_rows, dtype=np.float64),
    }


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {
            "accuracy": math.nan,
            "macro_f1": math.nan,
            "balanced_acc": math.nan,
            "per_class_recall": {name: math.nan for name in METRIC_NAMES.values()},
        }
    accuracy = float((y_true == y_pred).mean())
    f1s = []
    recalls = {}
    for cls, name in METRIC_NAMES.items():
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        recalls[name] = recall
        f1s.append(safe_div(2 * precision * recall, precision + recall))
    return {
        "accuracy": accuracy,
        "macro_f1": float(sum(f1s) / len(f1s)),
        "balanced_acc": float(sum(recalls.values()) / len(recalls)),
        "per_class_recall": recalls,
    }


def by_class_rows(evidence: np.ndarray, y_true: np.ndarray) -> list[dict]:
    rows = []
    for true_id, true_name in METRIC_NAMES.items():
        mask = y_true == true_id
        values = evidence[mask]
        row = {"true_metric_id": true_id, "true_metric": true_name, "support": int(mask.sum())}
        for dim, dim_name in METRIC_NAMES.items():
            dim_values = values[:, dim] if len(values) else np.asarray([], dtype=np.float64)
            row[f"{dim_name}_mean"] = float(dim_values.mean()) if len(dim_values) else math.nan
            row[f"{dim_name}_std"] = float(dim_values.std()) if len(dim_values) else math.nan
            row[f"{dim_name}_median"] = float(np.median(dim_values)) if len(dim_values) else math.nan
        rows.append(row)
    return rows


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def evidence_correlations(evidence: np.ndarray, y_true: np.ndarray) -> dict:
    correlations = {}
    for dim, dim_name in METRIC_NAMES.items():
        dim_values = evidence[:, dim]
        item = {"pearson_with_metric_id": pearson_corr(dim_values, y_true.astype(np.float64))}
        for cls, cls_name in METRIC_NAMES.items():
            item[f"pearson_with_true_{cls_name}"] = pearson_corr(dim_values, (y_true == cls).astype(np.float64))
        correlations[dim_name] = item
    return correlations


def cpu_distribution(evidence: np.ndarray, raw_evidence: np.ndarray, y_true: np.ndarray) -> dict:
    mask = y_true == 2
    values = evidence[mask]
    raw_values = raw_evidence[mask]
    if len(values) == 0:
        return {"support": 0}
    cpu_values = values[:, 2]
    comparisons = {}
    for dim, name in METRIC_NAMES.items():
        if dim == 2:
            continue
        comparisons[f"cpu_gt_{name}_ratio"] = float((values[:, 2] > values[:, dim]).mean())
        comparisons[f"cpu_minus_{name}_mean"] = float((values[:, 2] - values[:, dim]).mean())
    return {
        "support": int(mask.sum()),
        "normalized_cpu_evidence": {
            "mean": float(cpu_values.mean()),
            "std": float(cpu_values.std()),
            "median": float(np.median(cpu_values)),
            "min": float(cpu_values.min()),
            "max": float(cpu_values.max()),
            "p05": float(np.quantile(cpu_values, 0.05)),
            "p95": float(np.quantile(cpu_values, 0.95)),
        },
        "raw_cpu_evidence": {
            "mean": float(raw_values[:, 2].mean()),
            "std": float(raw_values[:, 2].std()),
            "median": float(np.median(raw_values[:, 2])),
            "min": float(raw_values[:, 2].min()),
            "max": float(raw_values[:, 2].max()),
            "p05": float(np.quantile(raw_values[:, 2], 0.05)),
            "p95": float(np.quantile(raw_values[:, 2], 0.95)),
        },
        "normalized_cpu_vs_other_evidence": comparisons,
        "normalized_argmax_counts_for_cpu_samples": {
            METRIC_NAMES[idx]: int(count)
            for idx, count in Counter(np.argmax(values, axis=1).tolist()).items()
        },
        "raw_argmax_counts_for_cpu_samples": {
            METRIC_NAMES[idx]: int(count)
            for idx, count in Counter(np.argmax(raw_values, axis=1).tolist()).items()
        },
    }


def json_array(value) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return json.dumps(value, ensure_ascii=False)


def summarize_sample_features(sample: dict, evidence: np.ndarray, raw_evidence: np.ndarray, true_metric: int, pred_metric: int) -> dict:
    return {
        "sample_id": sample.get("sample_id", ""),
        "service_id": sample.get("service_id", ""),
        "time": sample.get("time", ""),
        "scenario": sample.get("scenario", ""),
        "risk_label": sample.get("risk_label", ""),
        "true_metric_id": int(true_metric),
        "true_metric": METRIC_NAMES.get(int(true_metric), str(true_metric)),
        "pred_metric_id": int(pred_metric),
        "pred_metric": METRIC_NAMES.get(int(pred_metric), str(pred_metric)),
        "evidence_argmax_id": int(np.argmax(evidence)),
        "evidence_argmax": METRIC_NAMES.get(int(np.argmax(evidence)), str(int(np.argmax(evidence)))),
        "metric_evidence": json_array(evidence),
        "raw_metric_evidence": json_array(raw_evidence),
        "node_features": json_array(sample.get("node_x", [])),
        "link_features": json_array(sample.get("link_x", [])),
        "service_features": json_array(sample.get("service_x", [])),
        "sla_features": json_array(sample.get("sla_x", [])),
        "node_ids": json_array(sample.get("node_ids", [])),
        "link_ids": json_array(sample.get("link_ids", [])),
    }


def cpu_error_cases(samples: list[dict], evidence: np.ndarray, raw_evidence: np.ndarray, y_true: np.ndarray, pred_metric: np.ndarray, max_cases: int) -> list[dict]:
    rows = []
    for idx, true_value in enumerate(y_true):
        if int(true_value) != 2:
            continue
        if int(pred_metric[idx]) == 2 and int(np.argmax(evidence[idx])) == 2:
            continue
        rows.append(summarize_sample_features(samples[idx], evidence[idx], raw_evidence[idx], int(true_value), int(pred_metric[idx])))
        if len(rows) >= max_cases:
            break
    return rows


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    variant = normalize_variant(args.variant)
    artifact = sparta_artifact_name(variant)
    analysis_dir = ensure_dir(run_dir / "analysis")

    config = load_run_config(run_dir)
    collected = collect_evidence_and_predictions(run_dir, config, variant)
    model_config = collected["model_config"]
    feature_check = check_feature_schema(model_config)

    valid = collected["attr_mask"] == 1
    y_true = collected["true_metric"][valid]
    pred_metric = collected["pred_metric"][valid]
    evidence = collected["evidence"][valid]
    raw_evidence = collected["raw_evidence"][valid]
    samples = [sample for sample, is_valid in zip(collected["samples"], valid.tolist()) if is_valid]

    evidence_argmax = np.argmax(evidence, axis=1) if len(evidence) else np.asarray([], dtype=np.int64)
    raw_evidence_argmax = np.argmax(raw_evidence, axis=1) if len(raw_evidence) else np.asarray([], dtype=np.int64)
    evidence_metrics = classification_metrics(y_true, evidence_argmax)
    raw_evidence_metrics = classification_metrics(y_true, raw_evidence_argmax)

    by_class = by_class_rows(evidence, y_true)
    error_rows = cpu_error_cases(samples, evidence, raw_evidence, y_true, pred_metric, args.max_cases)
    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(pred_metric.tolist())
    evidence_argmax_counts = Counter(evidence_argmax.tolist())

    report = {
        "run_dir": str(run_dir),
        "variant": variant or "",
        "dataset_path": str(collected["test_path"]),
        "checkpoint_path": str(collected["checkpoint_path"]),
        "result_path": str(run_dir / "results" / f"{artifact}_test_results.csv"),
        "use_metric_evidence_head": bool(model_config.get("model", {}).get("use_metric_evidence_head", False)),
        "metric_evidence_norm": collected["evidence_norm"],
        "valid_attr_samples": int(valid.sum()),
        "true_metric_counts": {METRIC_NAMES[idx]: int(true_counts.get(idx, 0)) for idx in METRIC_NAMES},
        "model_pred_metric_counts": {METRIC_NAMES[idx]: int(pred_counts.get(idx, 0)) for idx in METRIC_NAMES},
        "evidence_argmax_counts": {METRIC_NAMES[idx]: int(evidence_argmax_counts.get(idx, 0)) for idx in METRIC_NAMES},
        "evidence_argmax_metrics": {
            "evidence_argmax_acc": evidence_metrics["accuracy"],
            "evidence_argmax_macro_f1": evidence_metrics["macro_f1"],
            "evidence_argmax_balanced_acc": evidence_metrics["balanced_acc"],
            "per_class_recall": evidence_metrics["per_class_recall"],
        },
        "raw_evidence_argmax_metrics": {
            "raw_evidence_argmax_acc": raw_evidence_metrics["accuracy"],
            "raw_evidence_argmax_macro_f1": raw_evidence_metrics["macro_f1"],
            "raw_evidence_argmax_balanced_acc": raw_evidence_metrics["balanced_acc"],
            "per_class_recall": raw_evidence_metrics["per_class_recall"],
        },
        "evidence_correlations": evidence_correlations(evidence, y_true),
        "raw_evidence_correlations": evidence_correlations(raw_evidence, y_true),
        "cpu_evidence_distribution": cpu_distribution(evidence, raw_evidence, y_true),
        "feature_schema_check": feature_check,
        "interpretation_hints": [
            "If raw_evidence CPU separates but normalized evidence does not, suspect normalization squeezing CPU information.",
            "If evidence_argmax CPU recall is near zero, suspect evidence extraction or feature mapping for CPU.",
            "If evidence_argmax CPU recall is reasonable but model pred CPU recall is near zero, suspect metric head learning/use of evidence.",
        ],
    }

    json_path = analysis_dir / f"{variant}_metric_evidence_audit.json"
    by_class_path = analysis_dir / f"{variant}_metric_evidence_by_class.csv"
    error_path = analysis_dir / f"{variant}_metric_evidence_error_cases.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(by_class_path, by_class)
    write_csv(error_path, error_rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {by_class_path}")
    print(f"Wrote {error_path}")
    if feature_check["warnings"]:
        print("Metric evidence audit warnings:")
        for warning in feature_check["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
