from __future__ import annotations

import json
import shutil

import yaml

from src.utils.config import load_config

from .conftest import ROOT, ensure_dataset, run_cmd


def make_mock_run(tmp_path):
    ensure_dataset()
    run_dir = tmp_path / "baseline_run"
    shutil.copytree(ROOT / "data" / "debug" / "dataset", run_dir / "dataset")
    (run_dir / "config").mkdir(parents=True)
    config = load_config(ROOT / "configs" / "sparta_debug.yaml")
    config["data"]["dataset_dir"] = str(run_dir / "dataset")
    config["data"]["train_path"] = str(run_dir / "dataset" / "train.pkl")
    config["data"]["val_path"] = str(run_dir / "dataset" / "val.pkl")
    config["data"]["test_path"] = str(run_dir / "dataset" / "test.pkl")
    config["train"]["batch_size"] = 16
    (run_dir / "config" / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return run_dir


def test_metric_evidence_baseline_outputs_json_and_csv(tmp_path):
    run_dir = make_mock_run(tmp_path)

    run_cmd(["src/analysis/evaluate_metric_evidence_baseline.py", "--run_dir", str(run_dir)])

    for norm in ["raw", "zscore", "pressure01"]:
        json_path = run_dir / "analysis" / f"metric_evidence_baseline_{norm}.json"
        confusion_path = run_dir / "analysis" / f"metric_evidence_baseline_{norm}_confusion.csv"
        assert json_path.exists()
        assert confusion_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "metric_evidence_acc" in data
        assert "metric_evidence_macro_f1" in data
        assert "metric_evidence_balanced_acc" in data
        assert "per_class_recall" in data
        assert "pred_count_cpu" in data
        assert "majority_baseline" in data
        assert "acc_minus_majority" in data

    assert (run_dir / "artifacts" / "metric_evidence_stats.json").exists()
    calibration_path = run_dir / "analysis" / "metric_evidence_pressure01_calibration.json"
    assert calibration_path.exists()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert "evidence_dimension_summary" in calibration


def test_metric_evidence_baseline_single_norm(tmp_path):
    run_dir = make_mock_run(tmp_path)

    run_cmd(
        [
            "src/analysis/evaluate_metric_evidence_baseline.py",
            "--run_dir",
            str(run_dir),
            "--evidence_norm",
            "raw",
        ]
    )

    assert (run_dir / "analysis" / "metric_evidence_baseline_raw.json").exists()
    assert (run_dir / "analysis" / "metric_evidence_baseline_raw_confusion.csv").exists()
