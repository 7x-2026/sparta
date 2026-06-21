from __future__ import annotations

import csv

from .conftest import run_cmd
from .test_variant_training import make_variant_run


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train_variant(run_dir, config_path):
    run_cmd(
        [
            "src/train.py",
            "--config",
            str(config_path),
            "--model",
            "sparta",
            "--resume_dataset_run",
            str(run_dir),
            "--variant",
            "metricfix",
        ]
    )


def test_variant_eval_and_diagnosis_use_variant_paths(tmp_path):
    run_dir, config_path = make_variant_run(tmp_path)
    original_result = run_dir / "results" / "sparta_test_results.csv"
    original_text = original_result.read_text(encoding="utf-8")
    train_variant(run_dir, config_path)

    run_cmd(
        [
            "src/evaluate.py",
            "--config",
            str(config_path),
            "--model",
            "sparta",
            "--resume_dataset_run",
            str(run_dir),
            "--variant",
            "metricfix",
        ]
    )

    assert original_result.read_text(encoding="utf-8") == original_text
    variant_result = run_dir / "results" / "sparta_metricfix_test_results.csv"
    assert variant_result.exists()
    assert (run_dir / "results" / "sparta_metricfix_prediction_distribution.csv").exists()
    assert (run_dir / "results" / "sparta_metricfix_confusion_matrix.csv").exists()
    row = read_rows(variant_result)[0]
    assert row["checkpoint_path"].endswith("sparta_metricfix_best.pth")
    assert row["result_source_file"].endswith("sparta_metricfix_test_results.csv")

    run_cmd(
        [
            "src/analysis/diagnose_attribution.py",
            "--run_dir",
            str(run_dir),
            "--variant",
            "metricfix",
            "--max_error_cases",
            "5",
        ]
    )
    assert (run_dir / "analysis" / "metricfix_attribution_diagnosis.json").exists()
    assert (run_dir / "analysis" / "metricfix_attribution_metric_confusion.csv").exists()
    assert (run_dir / "analysis" / "metricfix_attribution_error_cases.csv").exists()

    run_cmd(
        [
            "src/analysis/check_metricfix_effective.py",
            "--run_dir",
            str(run_dir),
            "--variant",
            "metricfix",
            "--metricfix_config",
            str(config_path),
        ]
    )
    assert (run_dir / "analysis" / "metricfix_effective_check.json").exists()
