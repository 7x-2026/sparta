from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir


MODELS = ["lstm", "transformer", "sparta"]
METRICS = [
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
    "metric_attr_macro_f1",
    "metric_attr_balanced_acc",
    "metric_attr_per_class_recall_delay",
    "metric_attr_per_class_recall_loss",
    "metric_attr_per_class_recall_cpu",
    "metric_attr_per_class_recall_queue",
    "metric_attr_per_class_recall_bandwidth",
    "inference_time",
]
ATTR_METRICS = {
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
}

SUMMARY_COLUMNS = [
    "run_id",
    "run_dir",
    "data_seed",
    "train_seed",
    "model",
    "checkpoint_path",
    "dataset_path",
    "result_source_file",
    *METRICS,
]
MEAN_STD_COLUMNS = ["model"] + [f"{metric}_{suffix}" for metric in METRICS for suffix in ["mean", "std"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="run_groups/edgesimpy_real_strict_mid_3seed")
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def parse_float(value) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text == "" or text.upper() == "N/A" or text.lower() == "nan":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def format_metric(value) -> str | float:
    parsed = parse_float(value)
    if math.isnan(parsed):
        return "N/A"
    return parsed


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def validate_manifest(run_dir: Path, manifest: dict) -> None:
    required = {
        "effective_simulator_backend": "edgesimpy_real",
        "edgesimpy_adapter_fallback": False,
        "real_object_created": True,
        "real_simulation_ran": True,
        "provenance_inconsistent": False,
        "status": "completed",
    }
    failures: list[str] = []
    for key, expected in required.items():
        actual = manifest.get(key)
        if actual != expected:
            failures.append(f"{key}={actual!r}, expected {expected!r}")
    if failures:
        raise RuntimeError(f"Run does not satisfy strict real EdgeSimPy criteria: {run_dir}; " + "; ".join(failures))


def validate_audit_inputs(run_dir: Path) -> tuple[str, list[dict]]:
    audit_report = run_dir / "audit" / "audit_report.txt"
    majority_path = run_dir / "audit" / "attribution_majority_baseline.csv"
    require_file(audit_report)
    require_file(majority_path)
    report_text = audit_report.read_text(encoding="utf-8", errors="replace")
    majority_rows = read_csv_rows(majority_path)
    if not majority_rows:
        raise RuntimeError(f"Empty attribution majority baseline: {majority_path}")
    if not any(row.get("split") == "all" for row in majority_rows):
        raise RuntimeError(f"Missing split=all in attribution majority baseline: {majority_path}")
    return report_text, majority_rows


def result_by_model(rows: list[dict], result_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        model = str(row.get("model") or row.get("method") or "").strip()
        if model:
            out[model] = row
    missing = sorted(set(MODELS) - set(out))
    if missing:
        raise RuntimeError(f"Missing model rows in {result_path}: {missing}")
    return out


def attribution_value(model: str, metric: str, result: dict) -> str | float:
    if metric in ATTR_METRICS and str(result.get("has_attribution_head", "")).lower() == "false":
        return "N/A"
    if metric in ATTR_METRICS and model in {"lstm", "transformer"}:
        return "N/A"
    return format_metric(result.get(metric))


def collect_run_rows(run_dir: Path) -> list[dict]:
    manifest_path = run_dir / "run_manifest.json"
    result_path = run_dir / "results" / "all_test_results.csv"
    require_file(manifest_path)
    require_file(result_path)
    manifest = read_json(manifest_path)
    validate_manifest(run_dir, manifest)
    validate_audit_inputs(run_dir)
    results = result_by_model(read_csv_rows(result_path), result_path)

    rows: list[dict] = []
    for model in MODELS:
        result = results[model]
        row: dict = {
            "run_id": manifest.get("run_id", run_dir.name),
            "run_dir": str(run_dir),
            "data_seed": manifest.get("data_seed", manifest.get("seed", "")),
            "train_seed": manifest.get("train_seed", ""),
            "model": model,
            "checkpoint_path": result.get("checkpoint_path", ""),
            "dataset_path": result.get("dataset_path", ""),
            "result_source_file": result.get("result_source_file", str(result_path)),
        }
        for metric in METRICS:
            if metric == "inference_time":
                row[metric] = format_metric(result.get("inference_time", result.get("inference_time_ms")))
            elif metric in ATTR_METRICS:
                row[metric] = attribution_value(model, metric, result)
            else:
                row[metric] = format_metric(result.get(metric))
        rows.append(row)
    return rows


def mean_std_rows(summary_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["model"])].append(row)

    rows: list[dict] = []
    for model in MODELS:
        model_rows = grouped.get(model, [])
        if not model_rows:
            continue
        out: dict = {"model": model}
        for metric in METRICS:
            values = [parse_float(row.get(metric)) for row in model_rows]
            values = [value for value in values if not math.isnan(value)]
            if values:
                out[f"{metric}_mean"] = statistics.mean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                out[f"{metric}_mean"] = "N/A"
                out[f"{metric}_std"] = "N/A"
        rows.append(out)
    return rows


def main() -> None:
    args = parse_args()
    run_dirs = [resolve_path(value) for value in args.run_dirs]
    output_dir = ensure_dir(resolve_path(args.output_dir))

    summary_rows: list[dict] = []
    for run_dir in run_dirs:
        summary_rows.extend(collect_run_rows(run_dir))

    expected_rows = len(run_dirs) * len(MODELS)
    if len(summary_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} summary rows, got {len(summary_rows)}")

    summary_path = output_dir / "multiseed_summary.csv"
    mean_std_path = output_dir / "multiseed_mean_std.csv"
    write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    write_csv(mean_std_path, mean_std_rows(summary_rows), MEAN_STD_COLUMNS)
    print(f"Wrote {summary_path}")
    print(f"Wrote {mean_std_path}")


if __name__ == "__main__":
    main()
