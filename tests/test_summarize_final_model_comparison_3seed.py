from __future__ import annotations

import csv
import json
from pathlib import Path

from src.analysis.summarize_final_model_comparison_3seed import main


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def result_row(**overrides) -> dict:
    row = {
        "accuracy": "0.71",
        "macro_f1": "0.72",
        "weighted_f1": "0.73",
        "balanced_acc": "0.74",
        "risk_recall": "0.75",
        "violation_recall": "0.76",
        "auc": "0.88",
        "risk_auc_macro_ovr": "0.88",
        "risk_auc_weighted_ovr": "0.89",
        "risk_auc_binary": "0.92",
        "inference_time_ms": "0.4",
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
    row.update(overrides)
    return row


def make_run(root: Path, seed: int, include_baselines: bool) -> Path:
    run_dir = root / f"run_seed{seed}"
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
    write_json(
        run_dir / "analysis" / "cpu_precursor_correlation.json",
        {
            "cpu_precursor_valid": True,
            "episode_count": 6,
            "affected_node_count": 3,
            "affected_service_count": 50,
        },
    )
    write_json(run_dir / "analysis" / "cpu_precursor_v1_attribution_diagnosis.json", {"ok": True})
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
            "best_bias_queue": 0.0,
            "best_bias_bandwidth": -0.5,
        },
    )
    (run_dir / "audit").mkdir(parents=True, exist_ok=True)
    (run_dir / "audit" / "audit_report.txt").write_text("通过正式实验数据要求\n未发现失败项\n", encoding="utf-8")
    write_csv(run_dir / "results" / "sparta_cpu_precursor_v1_test_results.csv", result_row())
    if include_baselines:
        write_csv(run_dir / "results" / "lstm_cpu_precursor_v1_test_results.csv", result_row(accuracy="0.60"))
        write_csv(run_dir / "results" / "transformer_cpu_precursor_v1_test_results.csv", result_row(accuracy="0.65"))
    return run_dir


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_final_model_comparison_outputs_with_missing_baselines(tmp_path: Path, monkeypatch):
    runs = [
        make_run(tmp_path, 42, include_baselines=True),
        make_run(tmp_path, 2025, include_baselines=False),
        make_run(tmp_path, 3407, include_baselines=True),
    ]
    output_dir = tmp_path / "comparison"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_final_model_comparison_3seed.py",
            "--run_dirs",
            *(str(run) for run in runs),
            "--output_dir",
            str(output_dir),
        ],
    )

    main()

    summary_path = output_dir / "model_comparison_summary.csv"
    mean_std_path = output_dir / "model_comparison_mean_std.csv"
    report_path = output_dir / "model_comparison_report.md"
    assert summary_path.exists()
    assert mean_std_path.exists()
    assert report_path.exists()
    rows = read_csv_rows(summary_path)
    assert len(rows) == 12
    models = {row["model"] for row in rows}
    assert models == {"LSTM-CP", "Transformer-CP", "SPARTA-CP", "SPARTA-CP + Cal."}
    missing_lstm = next(row for row in rows if row["data_seed"] == "2025" and row["model"] == "LSTM-CP")
    assert missing_lstm["accuracy"] == "NaN"
    cal_row = next(row for row in rows if row["data_seed"] == "42" and row["model"] == "SPARTA-CP + Cal.")
    assert cal_row["after_metric_attr_macro_f1"] == "0.62"
    report = report_path.read_text(encoding="utf-8")
    assert "Missing Files" in report
    assert "lstm_cpu_precursor_v1_test_results.csv" in report


def test_final_model_comparison_mean_std_by_model(tmp_path: Path, monkeypatch):
    runs = [make_run(tmp_path, seed, include_baselines=True) for seed in [42, 2025, 3407]]
    output_dir = tmp_path / "comparison"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize_final_model_comparison_3seed.py",
            "--run_dirs",
            *(str(run) for run in runs),
            "--output_dir",
            str(output_dir),
        ],
    )

    main()

    mean_rows = read_csv_rows(output_dir / "model_comparison_mean_std.csv")
    sparta = next(row for row in mean_rows if row["model"] == "SPARTA-CP")
    cal = next(row for row in mean_rows if row["model"] == "SPARTA-CP + Cal.")
    assert sparta["accuracy_mean"] == "0.71"
    assert cal["after_metric_attr_macro_f1_mean"] == "0.62"
