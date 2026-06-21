from __future__ import annotations

import json
import yaml

from .conftest import run_cmd
from .test_variant_training import make_variant_run


def configure_metricheadv3(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("experiment", {})["name"] = "debug_metricheadv3"
    config["model"]["use_metric_evidence_head"] = True
    config["model"]["metric_evidence_norm"] = "pressure01"
    config["model"]["metric_evidence_use_bias"] = True
    config["model"]["metric_evidence_bias_alpha"] = 3.0
    config["model"]["metric_evidence_bias_type"] = "logit"
    config["model"]["metric_evidence_detach"] = False
    config["loss"]["lambda_metric"] = 1.0
    config["loss"]["metric_loss_type"] = "focal"
    config["loss"]["metric_class_weighting"] = "inverse_sqrt_freq"
    config["loss"]["metric_focal_gamma"] = 2.0
    config["train"]["epochs"] = 1
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_metricheadv3_variant_outputs_and_train_stats(tmp_path):
    run_dir, config_path = make_variant_run(tmp_path)
    configure_metricheadv3(config_path)
    original_checkpoint = run_dir / "checkpoints" / "sparta_best.pth"
    metricfix_checkpoint = run_dir / "checkpoints" / "sparta_metricfix_best.pth"
    metricheadv2_checkpoint = run_dir / "checkpoints" / "sparta_metricheadv2_best.pth"
    metricfix_checkpoint.write_bytes(b"metricfix checkpoint")
    metricheadv2_checkpoint.write_bytes(b"metricheadv2 checkpoint")
    original_bytes = original_checkpoint.read_bytes()
    metricfix_bytes = metricfix_checkpoint.read_bytes()
    metricheadv2_bytes = metricheadv2_checkpoint.read_bytes()

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
            "metricheadv3",
        ]
    )

    stats_path = run_dir / "artifacts" / "metric_evidence_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert "queue_len_p95" in stats
    assert "evidence" in stats
    assert original_checkpoint.read_bytes() == original_bytes
    assert metricfix_checkpoint.read_bytes() == metricfix_bytes
    assert metricheadv2_checkpoint.read_bytes() == metricheadv2_bytes
    assert (run_dir / "checkpoints" / "sparta_metricheadv3_best.pth").exists()

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
            "metricheadv3",
        ]
    )

    assert (run_dir / "results" / "sparta_metricheadv3_test_results.csv").exists()
    assert (run_dir / "results" / "sparta_metricheadv3_prediction_distribution.csv").exists()
    assert (run_dir / "results" / "sparta_metricheadv3_confusion_matrix.csv").exists()

    run_cmd(
        [
            "src/analysis/diagnose_attribution.py",
            "--run_dir",
            str(run_dir),
            "--variant",
            "metricheadv3",
            "--max_error_cases",
            "5",
        ]
    )
    assert (run_dir / "analysis" / "metricheadv3_attribution_diagnosis.json").exists()


def test_use_metric_evidence_head_false_old_model_unaffected():
    from src.models.sparta import SPARTA
    from src.utils.config import load_config
    from .test_metric_evidence_head import make_batch
    from .conftest import ROOT

    config = load_config(ROOT / "configs" / "sparta_debug.yaml")
    config["model"]["use_metric_evidence_head"] = False
    model = SPARTA(config)
    outputs = model(make_batch())
    assert "metric_evidence" not in outputs
