from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir


OUTPUT_DIR_DEFAULT = "run_groups/cpu_precursor_v1_final_3seed"
REQUIRED_FILES = {
    "sparta_result": Path("results/sparta_cpu_precursor_v1_test_results.csv"),
    "diagnosis": Path("analysis/cpu_precursor_v1_attribution_diagnosis.json"),
    "macro_f1_calibration": Path("analysis/cpu_precursor_v1_metric_calibration_macro_f1.json"),
    "audit_report": Path("audit/audit_report.txt"),
    "cpu_precursor_correlation": Path("analysis/cpu_precursor_correlation.json"),
    "manifest": Path("run_manifest.json"),
}

RISK_METRICS = [
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
]
RAW_ATTR_METRICS = [
    "node_attr_acc",
    "link_attr_acc",
    "metric_attr_acc",
    "metric_attr_macro_f1",
    "metric_attr_balanced_acc",
    "metric_recall_delay",
    "metric_recall_loss",
    "metric_recall_cpu",
    "metric_recall_queue",
    "metric_recall_bandwidth",
    "metric_majority_baseline",
    "metric_attr_acc_minus_majority",
]
CALIBRATION_METRICS = [
    "after_metric_attr_acc",
    "after_metric_attr_macro_f1",
    "after_metric_attr_balanced_acc",
    "after_acc_minus_majority",
    "after_recall_delay",
    "after_recall_loss",
    "after_recall_cpu",
    "after_recall_queue",
    "after_recall_bandwidth",
    "best_bias_delay",
    "best_bias_loss",
    "best_bias_cpu",
    "best_bias_queue",
    "best_bias_bandwidth",
]
PROVENANCE_COLUMNS = [
    "run_id",
    "run_dir",
    "data_seed",
    "train_seed",
    "effective_simulator_backend",
    "edgesimpy_backend_state",
    "real_object_created",
    "real_simulation_ran",
    "risk_injection",
    "cpu_precursor_enabled",
    "cpu_precursor_valid",
    "episode_count",
    "affected_node_count",
    "affected_service_count",
]
CORRELATION_COLUMNS = [
    "node_cpu_util_slope_pearson_future_cpu_score",
    "node_cpu_util_slope_spearman_future_cpu_score",
    "node_available_cpu_slope_pearson_future_cpu_score",
    "node_available_cpu_slope_spearman_future_cpu_score",
    "service_path_cpu_pressure_slope_pearson_future_cpu_score",
    "service_path_cpu_pressure_slope_spearman_future_cpu_score",
]
SUMMARY_COLUMNS = PROVENANCE_COLUMNS + RISK_METRICS + ["auc_error"] + RAW_ATTR_METRICS + CALIBRATION_METRICS + CORRELATION_COLUMNS
NUMERIC_COLUMNS = [
    "data_seed",
    "train_seed",
    "episode_count",
    "affected_node_count",
    "affected_service_count",
    *RISK_METRICS,
    *RAW_ATTR_METRICS,
    *CALIBRATION_METRICS,
    *CORRELATION_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", default=OUTPUT_DIR_DEFAULT)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def nan() -> float:
    return math.nan


def parse_float(value) -> float:
    if value is None:
        return nan()
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.upper() == "N/A" or text.lower() == "nan":
        return nan()
    try:
        return float(text)
    except ValueError:
        return nan()


def scalar(value):
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "NaN"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def read_json(path: Path, missing: list[str]) -> dict:
    if not path.exists():
        missing.append(str(path))
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        missing.append(f"{path} (parse failed: {exc})")
        return {}


def read_csv_first(path: Path, missing: list[str]) -> dict:
    if not path.exists():
        missing.append(str(path))
        return {}
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[0] if rows else {}
    except Exception as exc:
        missing.append(f"{path} (parse failed: {exc})")
        return {}


def read_text(path: Path, missing: list[str]) -> str:
    if not path.exists():
        missing.append(str(path))
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        missing.append(f"{path} (read failed: {exc})")
        return ""


def metric_value(row: dict, key: str) -> float:
    if key.startswith("metric_recall_"):
        metric_name = key.replace("metric_recall_", "")
        return parse_float(row.get(key, row.get(f"metric_attr_per_class_recall_{metric_name}")))
    return parse_float(row.get(key))


def nested_metric(payload: dict, path: list[str], default=math.nan):
    current = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def collect_run(run_dir: Path) -> tuple[dict, list[str], str]:
    missing: list[str] = []
    result = read_csv_first(run_dir / REQUIRED_FILES["sparta_result"], missing)
    diagnosis = read_json(run_dir / REQUIRED_FILES["diagnosis"], missing)
    calibration = read_json(run_dir / REQUIRED_FILES["macro_f1_calibration"], missing)
    audit_text = read_text(run_dir / REQUIRED_FILES["audit_report"], missing)
    cpu_corr = read_json(run_dir / REQUIRED_FILES["cpu_precursor_correlation"], missing)
    manifest = read_json(run_dir / REQUIRED_FILES["manifest"], missing)
    _ = diagnosis  # The file is required as an experiment artifact; current summary fields come from result/correlation JSON.

    row: dict = {
        "run_id": manifest.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "data_seed": manifest.get("data_seed", manifest.get("seed", nan())),
        "train_seed": manifest.get("train_seed", nan()),
        "effective_simulator_backend": manifest.get("effective_simulator_backend", "NaN"),
        "edgesimpy_backend_state": manifest.get("edgesimpy_backend_state", "NaN"),
        "real_object_created": manifest.get("real_object_created", "NaN"),
        "real_simulation_ran": manifest.get("real_simulation_ran", "NaN"),
        "risk_injection": manifest.get("risk_injection", "NaN"),
        "cpu_precursor_enabled": manifest.get("cpu_precursor_enabled", "NaN"),
        "cpu_precursor_valid": cpu_corr.get("cpu_precursor_valid", "NaN"),
        "episode_count": cpu_corr.get("episode_count", nan()),
        "affected_node_count": cpu_corr.get("affected_node_count", nan()),
        "affected_service_count": cpu_corr.get("affected_service_count", nan()),
    }
    for key in RISK_METRICS:
        row[key] = parse_float(result.get(key))
    row["auc_error"] = result.get("auc_error", "")
    for key in ["auc", "risk_auc_macro_ovr", "risk_auc_weighted_ovr", "risk_auc_binary"]:
        if key not in result:
            missing.append(
                f"{run_dir / REQUIRED_FILES['sparta_result']} missing column {key}; rerun evaluate.py to populate risk AUC"
            )
    for key in RAW_ATTR_METRICS:
        row[key] = metric_value(result, key)

    recall = calibration.get("after_per_class_recall", {})
    for key in CALIBRATION_METRICS:
        if key.startswith("after_recall_"):
            row[key] = parse_float(recall.get(key.replace("after_recall_", "")))
        else:
            row[key] = parse_float(calibration.get(key))

    corr = cpu_corr.get("correlations", {})
    corr_map = {
        "node_cpu_util_slope": "node.cpu_util_slope",
        "node_available_cpu_slope": "node.available_cpu_slope",
        "service_path_cpu_pressure_slope": "service.path_cpu_pressure_slope",
    }
    for out_prefix, source_key in corr_map.items():
        stats = corr.get(source_key, {})
        row[f"{out_prefix}_pearson_future_cpu_score"] = parse_float(stats.get("pearson_with_future_cpu_score"))
        row[f"{out_prefix}_spearman_future_cpu_score"] = parse_float(stats.get("spearman_with_future_cpu_score"))

    if audit_text and "通过正式实验数据要求" not in audit_text and "未发现失败项" not in audit_text:
        missing.append(f"{run_dir / REQUIRED_FILES['audit_report']} (audit text does not contain pass phrase)")
    return row, missing, audit_text


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: scalar(row.get(key, "NaN")) for key in fieldnames} for row in rows])


def mean_std_rows(summary_rows: list[dict]) -> list[dict]:
    row: dict = {"group": "cpu_precursor_v1_final_3seed", "num_runs": len(summary_rows)}
    for column in NUMERIC_COLUMNS:
        values = [parse_float(item.get(column)) for item in summary_rows]
        values = [value for value in values if not math.isnan(value)]
        if values:
            row[f"{column}_mean"] = statistics.mean(values)
            row[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        else:
            row[f"{column}_mean"] = "NaN"
            row[f"{column}_std"] = "NaN"
    return [row]


def mean_std_columns() -> list[str]:
    return ["group", "num_runs"] + [f"{column}_{suffix}" for column in NUMERIC_COLUMNS for suffix in ["mean", "std"]]


def pct(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value * 100:.2f}%"


def avg(summary_rows: list[dict], key: str) -> float:
    values = [parse_float(row.get(key)) for row in summary_rows]
    values = [value for value in values if not math.isnan(value)]
    return statistics.mean(values) if values else nan()


def build_report(summary_rows: list[dict], missing_by_run: dict[str, list[str]], output_dir: Path) -> str:
    missing_total = sum(len(items) for items in missing_by_run.values())
    lines = [
        "# CPU Precursor v1 Final 3-Seed Report",
        "",
        "## 中文总结",
        "",
        f"本次汇总包含 {len(summary_rows)} 个 run，实验组合为 `cpu_precursor_v1 数据 + 原始 SPARTA 训练 + macro_f1 metric logit calibration`。",
        f"原始风险检测平均 Macro-F1 为 {pct(avg(summary_rows, 'macro_f1'))}，平均 Balanced Acc 为 {pct(avg(summary_rows, 'balanced_acc'))}。",
        f"风险检测平均 AUC macro OVR 为 {pct(avg(summary_rows, 'risk_auc_macro_ovr'))}，weighted OVR 为 {pct(avg(summary_rows, 'risk_auc_weighted_ovr'))}，binary risk-vs-normal AUC 为 {pct(avg(summary_rows, 'risk_auc_binary'))}。",
        f"原始 metric attribution 平均 Acc 为 {pct(avg(summary_rows, 'metric_attr_acc'))}，平均 Macro-F1 为 {pct(avg(summary_rows, 'metric_attr_macro_f1'))}。",
        f"macro_f1 校准后 metric attribution 平均 Acc 为 {pct(avg(summary_rows, 'after_metric_attr_acc'))}，平均 Macro-F1 为 {pct(avg(summary_rows, 'after_metric_attr_macro_f1'))}，平均 Balanced Acc 为 {pct(avg(summary_rows, 'after_metric_attr_balanced_acc'))}。",
        f"CPU precursor 审计平均 episode_count 为 {avg(summary_rows, 'episode_count'):.2f}，平均 affected_node_count 为 {avg(summary_rows, 'affected_node_count'):.2f}，平均 affected_service_count 为 {avg(summary_rows, 'affected_service_count'):.2f}。",
        "",
        "## 输出文件",
        "",
        f"- `{output_dir / 'summary.csv'}`",
        f"- `{output_dir / 'mean_std.csv'}`",
        f"- `{output_dir / 'final_report.md'}`",
        "",
        "## Missing Files",
        "",
    ]
    if missing_total == 0:
        lines.append("未发现缺失文件。")
    else:
        for run_id, items in missing_by_run.items():
            if not items:
                continue
            lines.append(f"### {run_id}")
            for item in items:
                lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Provenance 检查提示",
            "",
            "请确认 summary.csv 中 `effective_simulator_backend=edgesimpy_real`、`risk_injection=cpu_precursor_v1`、`cpu_precursor_valid=true` 的 run 才用于正式论文表格。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dirs = [resolve_path(value) for value in args.run_dirs]
    output_dir = ensure_dir(resolve_path(args.output_dir))
    summary_rows: list[dict] = []
    missing_by_run: dict[str, list[str]] = {}
    for run_dir in run_dirs:
        row, missing, _audit_text = collect_run(run_dir)
        summary_rows.append(row)
        missing_by_run[str(row.get("run_id", run_dir.name))] = missing

    summary_path = output_dir / "summary.csv"
    mean_std_path = output_dir / "mean_std.csv"
    report_path = output_dir / "final_report.md"
    write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    write_csv(mean_std_path, mean_std_rows(summary_rows), mean_std_columns())
    report_path.write_text(build_report(summary_rows, missing_by_run, output_dir), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {mean_std_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
