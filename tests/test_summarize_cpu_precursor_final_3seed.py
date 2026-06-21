from __future__ import annotations

import csv
import json
from pathlib import Path

from src.analysis.summarize_cpu_precursor_final_3seed import main


def write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_run(root: Path, seed: int, missing_calibration: bool = False) -> Path:
    run_dir = root / f"run_seed{seed}"
    result_row = {
        "accuracy": "0.70",
        "macro_f1": "0.71",
        "weighted_f1": "0.72",
        "balanced_acc": "0.73",
        "risk_recall": "0.74",
        "violation_recall": "0.75",
        "auc": "0.76",
        "risk_auc_macro_ovr": "0.76",
        "risk_auc_weighted_ovr": "0.77",
        "risk_auc_binary": "0.78",
        "auc_error": "",
        "node_attr_acc": "0.40",
        "link_attr_acc": "0.41",
        "metric_attr_acc": "0.42",
        "metric_attr_macro_f1": "0.43",
        "metric_attr_balanced_acc": "0.44",
        "metric_recall_delay": "0.50",
        "metric_recall_loss": "0.51",
        "metric_recall_cpu": "0.52",
        "metric_recall_queue": "0.53",
        "metric_recall_bandwidth": "0.54",
        "metric_majority_baseline": "0.55",
        "metric_attr_acc_minus_majority": "-0.13",
    }
    write_csv(run_dir / "results" / "sparta_cpu_precursor_v1_test_results.csv", result_row)
    write_json(run_dir / "analysis" / "cpu_precursor_v1_attribution_diagnosis.json", {"ok": True})
    if not missing_calibration:
        write_json(
            run_dir / "analysis" / "cpu_precursor_v1_metric_calibration_macro_f1.json",
            {
                "after_metric_attr_acc": 0.61,
                "after_metric_attr_macro_f1": 0.62,
                "after_metric_attr_balanced_acc": 0.63,
                "after_acc_minus_majority": 0.06,
                "after_per_class_recall": {
                    "delay": 0.1,
                    "loss": 0.2,
                    "cpu": 0.3,
                    "queue": 0.4,
                    "bandwidth": 0.5,
                },
                "best_bias_delay": -0.1,
                "best_bias_loss": -0.2,
                "best_bias_cpu": -0.3,
                "best_bias_queue": 0.4,
                "best_bias_bandwidth": -0.5,
            },
        )
    (run_dir / "audit").mkdir(parents=True, exist_ok=True)
    (run_dir / "audit" / "audit_report.txt").write_text("通过正式实验数据要求\n未发现失败项\n", encoding="utf-8")
    write_json(
        run_dir / "analysis" / "cpu_precursor_correlation.json",
        {
            "cpu_precursor_valid": True,
            "episode_count": 6 + seed % 3,
            "affected_node_count": 3,
            "affected_service_count": 20,
            "correlations": {
                "node.cpu_util_slope": {
                    "pearson_with_future_cpu_score": 0.11,
                    "spearman_with_future_cpu_score": 0.12,
                },
                "node.available_cpu_slope": {
                    "pearson_with_future_cpu_score": -0.21,
                    "spearman_with_future_cpu_score": -0.22,
                },
                "service.path_cpu_pressure_slope": {
                    "pearson_with_future_cpu_score": 0.31,
                    "spearman_with_future_cpu_score": 0.32,
                },
            },
        },
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": f"run_seed{seed}",
            "data_seed": seed,
            "train_seed": 0,
            "effective_simulator_backend": "edgesimpy_real",
            "edgesimpy_backend_state": "edgesimpy_real_simulation_ran",
            "real_object_created": True,
            "real_simulation_ran": True,
            "risk_injection": "cpu_precursor_v1",
            "cpu_precursor_enabled": True,
        },
    )
    return run_dir


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_summarize_cpu_precursor_final_3seed_outputs(tmp_path: Path, monkeypatch):
    runs = [make_run(tmp_path, seed) for seed in [42, 2025, 3407]]
    output_dir = tmp_path / "group"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_cpu_precursor_final_3seed.py",
            "--run_dirs",
            *(str(run) for run in runs),
            "--output_dir",
            str(output_dir),
        ],
    )

    main()

    summary_path = output_dir / "summary.csv"
    mean_std_path = output_dir / "mean_std.csv"
    report_path = output_dir / "final_report.md"
    assert summary_path.exists()
    assert mean_std_path.exists()
    assert report_path.exists()
    summary = read_csv_rows(summary_path)
    mean_std = read_csv_rows(mean_std_path)
    assert len(summary) == 3
    assert summary[0]["risk_injection"] == "cpu_precursor_v1"
    assert summary[0]["risk_auc_binary"] == "0.78"
    assert summary[0]["after_metric_attr_macro_f1"] == "0.62"
    assert "macro_f1_mean" in mean_std[0]
    assert "risk_auc_binary_mean" in mean_std[0]
    assert "未发现缺失文件" in report_path.read_text(encoding="utf-8")


def test_summarize_cpu_precursor_final_3seed_missing_file_does_not_crash(tmp_path: Path, monkeypatch):
    runs = [
        make_run(tmp_path, 42),
        make_run(tmp_path, 2025, missing_calibration=True),
        make_run(tmp_path, 3407),
    ]
    output_dir = tmp_path / "group_missing"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_cpu_precursor_final_3seed.py",
            "--run_dirs",
            *(str(run) for run in runs),
            "--output_dir",
            str(output_dir),
        ],
    )

    main()

    report = (output_dir / "final_report.md").read_text(encoding="utf-8")
    summary = read_csv_rows(output_dir / "summary.csv")
    missing_row = next(row for row in summary if row["data_seed"] == "2025")
    assert "cpu_precursor_v1_metric_calibration_macro_f1.json" in report
    assert missing_row["after_metric_attr_acc"] == "NaN"
