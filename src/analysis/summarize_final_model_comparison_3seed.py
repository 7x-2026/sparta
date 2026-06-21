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


OUTPUT_DIR_DEFAULT = "run_groups/cpu_precursor_v1_final_model_comparison_3seed"
MODEL_ROWS = ["LSTM-CP", "Transformer-CP", "SPARTA-CP", "SPARTA-CP + Cal."]
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
    "inference_time_ms",
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
SUMMARY_COLUMNS = [
    "run_id",
    "run_dir",
    "data_seed",
    "train_seed",
    "model",
    "result_source_file",
    *PROVENANCE_COLUMNS,
    *RISK_METRICS,
    *RAW_ATTR_METRICS,
    *CALIBRATION_METRICS,
]
NUMERIC_COLUMNS = [
    "data_seed",
    "train_seed",
    "episode_count",
    "affected_node_count",
    "affected_service_count",
    *RISK_METRICS,
    *RAW_ATTR_METRICS,
    *CALIBRATION_METRICS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", default=OUTPUT_DIR_DEFAULT)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def parse_float(value) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "" or text.upper() == "N/A" or text.lower() == "nan":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


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


def read_csv_first_existing(paths: list[Path], missing: list[str]) -> tuple[dict, str]:
    for path in paths:
        if path.exists():
            try:
                with path.open("r", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                return (rows[0] if rows else {}), str(path)
            except Exception as exc:
                missing.append(f"{path} (parse failed: {exc})")
                return {}, str(path)
    missing.append(" or ".join(str(path) for path in paths))
    return {}, ""


def read_text(path: Path, missing: list[str]) -> str:
    if not path.exists():
        missing.append(str(path))
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def metric_value(row: dict, key: str) -> float:
    if key.startswith("metric_recall_"):
        name = key.replace("metric_recall_", "")
        return parse_float(row.get(key, row.get(f"metric_attr_per_class_recall_{name}")))
    return parse_float(row.get(key))


def base_row(run_dir: Path, manifest: dict, cpu_corr: dict, model_name: str, result_source_file: str) -> dict:
    return {
        "run_id": manifest.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "data_seed": manifest.get("data_seed", manifest.get("seed", math.nan)),
        "train_seed": manifest.get("train_seed", math.nan),
        "model": model_name,
        "result_source_file": result_source_file,
        "effective_simulator_backend": manifest.get("effective_simulator_backend", "NaN"),
        "edgesimpy_backend_state": manifest.get("edgesimpy_backend_state", "NaN"),
        "real_object_created": manifest.get("real_object_created", "NaN"),
        "real_simulation_ran": manifest.get("real_simulation_ran", "NaN"),
        "risk_injection": manifest.get("risk_injection", "NaN"),
        "cpu_precursor_enabled": manifest.get("cpu_precursor_enabled", "NaN"),
        "cpu_precursor_valid": cpu_corr.get("cpu_precursor_valid", "NaN"),
        "episode_count": cpu_corr.get("episode_count", math.nan),
        "affected_node_count": cpu_corr.get("affected_node_count", math.nan),
        "affected_service_count": cpu_corr.get("affected_service_count", math.nan),
    }


def fill_risk_metrics(row: dict, result: dict) -> None:
    for key in RISK_METRICS:
        row[key] = parse_float(result.get(key))


def fill_raw_attr_metrics(row: dict, result: dict) -> None:
    for key in RAW_ATTR_METRICS:
        row[key] = metric_value(result, key)


def fill_nan_metrics(row: dict, metrics: list[str]) -> None:
    for key in metrics:
        row[key] = math.nan


def fill_calibration_metrics(row: dict, calibration: dict) -> None:
    recall = calibration.get("after_per_class_recall", {})
    for key in CALIBRATION_METRICS:
        if key.startswith("after_recall_"):
            row[key] = parse_float(recall.get(key.replace("after_recall_", "")))
        else:
            row[key] = parse_float(calibration.get(key))


def collect_run(run_dir: Path) -> tuple[list[dict], list[str]]:
    missing: list[str] = []
    manifest = read_json(run_dir / "run_manifest.json", missing)
    cpu_corr = read_json(run_dir / "analysis" / "cpu_precursor_correlation.json", missing)
    _diagnosis = read_json(run_dir / "analysis" / "cpu_precursor_v1_attribution_diagnosis.json", missing)
    audit_text = read_text(run_dir / "audit" / "audit_report.txt", missing)
    if audit_text and "通过正式实验数据要求" not in audit_text and "未发现失败项" not in audit_text:
        missing.append(f"{run_dir / 'audit' / 'audit_report.txt'} (audit text does not contain pass phrase)")

    lstm_result, lstm_path = read_csv_first_existing(
        [run_dir / "results" / "lstm_cpu_precursor_v1_test_results.csv", run_dir / "results" / "lstm_test_results.csv"],
        missing,
    )
    transformer_result, transformer_path = read_csv_first_existing(
        [
            run_dir / "results" / "transformer_cpu_precursor_v1_test_results.csv",
            run_dir / "results" / "transformer_test_results.csv",
        ],
        missing,
    )
    sparta_result, sparta_path = read_csv_first_existing(
        [run_dir / "results" / "sparta_cpu_precursor_v1_test_results.csv"],
        missing,
    )
    calibration = read_json(run_dir / "analysis" / "cpu_precursor_v1_metric_calibration_macro_f1.json", missing)

    rows: list[dict] = []
    for model_name, result, result_path in [
        ("LSTM-CP", lstm_result, lstm_path),
        ("Transformer-CP", transformer_result, transformer_path),
    ]:
        row = base_row(run_dir, manifest, cpu_corr, model_name, result_path)
        fill_risk_metrics(row, result)
        fill_nan_metrics(row, RAW_ATTR_METRICS + CALIBRATION_METRICS)
        rows.append(row)

    sparta_row = base_row(run_dir, manifest, cpu_corr, "SPARTA-CP", sparta_path)
    fill_risk_metrics(sparta_row, sparta_result)
    fill_raw_attr_metrics(sparta_row, sparta_result)
    fill_nan_metrics(sparta_row, CALIBRATION_METRICS)
    rows.append(sparta_row)

    cal_row = base_row(run_dir, manifest, cpu_corr, "SPARTA-CP + Cal.", str(run_dir / "analysis" / "cpu_precursor_v1_metric_calibration_macro_f1.json"))
    fill_risk_metrics(cal_row, sparta_result)
    fill_nan_metrics(cal_row, RAW_ATTR_METRICS)
    fill_calibration_metrics(cal_row, calibration)
    rows.append(cal_row)
    return rows, missing


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: scalar(row.get(key, "NaN")) for key in fieldnames} for row in rows])


def mean_std_rows(summary_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in summary_rows:
        grouped.setdefault(str(row["model"]), []).append(row)
    rows: list[dict] = []
    for model in MODEL_ROWS:
        model_rows = grouped.get(model, [])
        out: dict = {"model": model, "num_runs": len(model_rows)}
        for column in NUMERIC_COLUMNS:
            values = [parse_float(row.get(column)) for row in model_rows]
            values = [value for value in values if not math.isnan(value)]
            if values:
                out[f"{column}_mean"] = statistics.mean(values)
                out[f"{column}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                out[f"{column}_mean"] = "NaN"
                out[f"{column}_std"] = "NaN"
        rows.append(out)
    return rows


def mean_std_columns() -> list[str]:
    return ["model", "num_runs"] + [f"{column}_{suffix}" for column in NUMERIC_COLUMNS for suffix in ["mean", "std"]]


def avg(rows: list[dict], model: str, key: str) -> float:
    values = [parse_float(row.get(key)) for row in rows if row.get("model") == model]
    values = [value for value in values if not math.isnan(value)]
    return statistics.mean(values) if values else math.nan


def pct(value: float) -> str:
    return "NaN" if math.isnan(value) else f"{value * 100:.2f}%"


def build_report(summary_rows: list[dict], missing_by_run: dict[str, list[str]], output_dir: Path) -> str:
    seeds = sorted({str(row.get("data_seed")) for row in summary_rows if row.get("data_seed") not in {"NaN", math.nan}})
    missing_total = sum(len(items) for items in missing_by_run.values())
    sparta_macro = avg(summary_rows, "SPARTA-CP", "macro_f1")
    sparta_balanced = avg(summary_rows, "SPARTA-CP", "balanced_acc")
    cal_metric_f1 = avg(summary_rows, "SPARTA-CP + Cal.", "after_metric_attr_macro_f1")
    cal_metric_acc = avg(summary_rows, "SPARTA-CP + Cal.", "after_metric_attr_acc")
    cal_minus_majority = avg(summary_rows, "SPARTA-CP + Cal.", "after_acc_minus_majority")
    lines = [
        "# CPU Precursor v1 Final Model Comparison",
        "",
        "## 中文总结",
        "",
        f"本次模型对比包含 seeds: {', '.join(seeds)}，完整性：{'完整' if len(seeds) == 3 else '不完整'}。",
        f"SPARTA-CP 风险检测均值：Macro-F1={pct(sparta_macro)}，Balanced Acc={pct(sparta_balanced)}。",
        f"SPARTA-CP + Cal. attribution 均值：metric Acc={pct(cal_metric_acc)}，metric Macro-F1={pct(cal_metric_f1)}。",
        f"SPARTA-CP + Cal. 相对 metric majority baseline 的平均差值：{pct(cal_minus_majority)}。",
        "结论：" + ("校准后平均超过 majority baseline。" if not math.isnan(cal_minus_majority) and cal_minus_majority > 0 else "校准后平均未超过或无法确认超过 majority baseline。"),
        "",
        "## 输出文件",
        "",
        f"- `{output_dir / 'model_comparison_summary.csv'}`",
        f"- `{output_dir / 'model_comparison_mean_std.csv'}`",
        f"- `{output_dir / 'model_comparison_report.md'}`",
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
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dirs = [resolve_path(value) for value in args.run_dirs]
    output_dir = ensure_dir(resolve_path(args.output_dir))
    summary_rows: list[dict] = []
    missing_by_run: dict[str, list[str]] = {}
    for run_dir in run_dirs:
        rows, missing = collect_run(run_dir)
        summary_rows.extend(rows)
        run_id = rows[0]["run_id"] if rows else run_dir.name
        missing_by_run[str(run_id)] = missing

    summary_path = output_dir / "model_comparison_summary.csv"
    mean_std_path = output_dir / "model_comparison_mean_std.csv"
    report_path = output_dir / "model_comparison_report.md"
    write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    write_csv(mean_std_path, mean_std_rows(summary_rows), mean_std_columns())
    report_path.write_text(build_report(summary_rows, missing_by_run, output_dir), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {mean_std_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
