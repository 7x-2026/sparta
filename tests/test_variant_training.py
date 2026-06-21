from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from src.utils.config import load_config

from .conftest import ROOT, ensure_dataset, run_cmd


def make_variant_run(tmp_path: Path) -> tuple[Path, Path]:
    ensure_dataset()
    run_dir = tmp_path / "variant_run"
    shutil.copytree(ROOT / "data" / "debug" / "dataset", run_dir / "dataset")
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "config").mkdir(parents=True)

    original_checkpoint = run_dir / "checkpoints" / "sparta_best.pth"
    original_result = run_dir / "results" / "sparta_test_results.csv"
    original_checkpoint.write_bytes(b"original checkpoint")
    original_result.write_text("method,accuracy\nsparta,0.0\n", encoding="utf-8")

    config = load_config(ROOT / "configs" / "sparta_debug.yaml")
    config["project"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    config["project"]["log_dir"] = str(run_dir / "logs")
    config["project"]["output_dir"] = str(run_dir / "results")
    config["data"]["dataset_dir"] = str(run_dir / "dataset")
    config["data"]["train_path"] = str(run_dir / "dataset" / "train.pkl")
    config["data"]["val_path"] = str(run_dir / "dataset" / "val.pkl")
    config["data"]["test_path"] = str(run_dir / "dataset" / "test.pkl")
    config["train"]["epochs"] = 1
    config["train"]["batch_size"] = 8
    config["loss"].update(
        {
            "lambda_node": 0.2,
            "lambda_link": 0.2,
            "lambda_metric": 0.8,
            "metric_loss_type": "focal",
            "metric_class_weighting": "inverse_sqrt_freq",
            "metric_focal_gamma": 2.0,
        }
    )
    config["run"] = {"run_id": run_dir.name}

    config_path = tmp_path / "variant_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "config" / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return run_dir, config_path


def test_variant_training_does_not_overwrite_original_checkpoint(tmp_path):
    run_dir, config_path = make_variant_run(tmp_path)
    original_checkpoint = run_dir / "checkpoints" / "sparta_best.pth"
    original_bytes = original_checkpoint.read_bytes()

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

    assert original_checkpoint.read_bytes() == original_bytes
    assert (run_dir / "checkpoints" / "sparta_metricfix_best.pth").exists()
    assert (run_dir / "checkpoints" / "sparta_metricfix_last.pth").exists()
    log_path = run_dir / "logs" / "sparta_metricfix_train.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "variant" in log_text
    assert "metric_loss_type" in log_text
    assert "metric_focal_gamma" in log_text
    assert "metric_class_weights" in log_text
    assert "loss_metric" in log_text
