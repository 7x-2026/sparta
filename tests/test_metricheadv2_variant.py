from __future__ import annotations

import yaml

from .conftest import run_cmd
from .test_variant_training import make_variant_run


def configure_metricheadv2(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("experiment", {})["name"] = "debug_metricheadv2"
    config["model"]["use_metric_evidence_head"] = True
    config["model"]["metric_evidence_bias_alpha"] = 0.5
    config["model"]["metric_evidence_norm"] = "zscore"
    config["model"]["metric_evidence_use_bias"] = True
    config["loss"]["lambda_metric"] = 1.0
    config["loss"]["metric_loss_type"] = "focal"
    config["loss"]["metric_class_weighting"] = "inverse_sqrt_freq"
    config["loss"]["metric_focal_gamma"] = 2.0
    config["train"]["epochs"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_metricheadv2_variant_paths_do_not_overwrite_existing_outputs(tmp_path):
    run_dir, config_path = make_variant_run(tmp_path)
    configure_metricheadv2(config_path)
    original_checkpoint = run_dir / "checkpoints" / "sparta_best.pth"
    metricfix_checkpoint = run_dir / "checkpoints" / "sparta_metricfix_best.pth"
    metricfix_checkpoint.write_bytes(b"metricfix checkpoint")
    original_bytes = original_checkpoint.read_bytes()
    metricfix_bytes = metricfix_checkpoint.read_bytes()

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
            "metricheadv2",
        ]
    )

    assert original_checkpoint.read_bytes() == original_bytes
    assert metricfix_checkpoint.read_bytes() == metricfix_bytes
    assert (run_dir / "checkpoints" / "sparta_metricheadv2_best.pth").exists()

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
            "metricheadv2",
        ]
    )
    assert (run_dir / "results" / "sparta_metricheadv2_test_results.csv").exists()
    assert (run_dir / "results" / "sparta_metricheadv2_prediction_distribution.csv").exists()
    assert (run_dir / "results" / "sparta_metricheadv2_confusion_matrix.csv").exists()

    run_cmd(
        [
            "src/analysis/diagnose_attribution.py",
            "--run_dir",
            str(run_dir),
            "--variant",
            "metricheadv2",
            "--max_error_cases",
            "5",
        ]
    )
    assert (run_dir / "analysis" / "metricheadv2_attribution_diagnosis.json").exists()
    assert (run_dir / "analysis" / "metricheadv2_attribution_metric_confusion.csv").exists()
    assert (run_dir / "analysis" / "metricheadv2_attribution_error_cases.csv").exists()
