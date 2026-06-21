from __future__ import annotations

import json
import shutil

import torch
import yaml

from src.models.sparta import build_metric_evidence, compute_metric_evidence_stats_from_loader
from src.utils.config import load_config

from .conftest import ROOT, ensure_dataset, run_cmd


def make_pressure02_batch() -> dict[str, torch.Tensor]:
    batch_size, L, N, E = 4, 4, 3, 2
    node_x = torch.zeros(batch_size, L, N, 8)
    link_x = torch.zeros(batch_size, L, E, 8)
    service_x = torch.zeros(batch_size, L, 5)
    sla_x = torch.ones(batch_size, 5)
    node_mask = torch.ones(batch_size, N, dtype=torch.bool)
    link_mask = torch.ones(batch_size, E, dtype=torch.bool)

    node_x[:, :, :, 0] = 0.25
    node_x[:, :, :, 3] = 0.85
    link_x[:, :, :, 1] = 0.90
    sla_x[:, 0] = 0.80
    sla_x[:, 1] = 0.20
    sla_x[:, 2] = 0.70

    node_x[0, :, :, 0] = 0.95
    node_x[0, :, :, 3] = 0.05

    for t in range(L):
        node_x[1, t, :, 2] = 0.10 + 0.20 * t
    link_x[1, :, :, 4] = 0.70

    link_x[2, :, :, 1] = 0.20
    link_x[2, :, :, 5] = 0.96
    sla_x[2, 2] = 0.90

    service_x[3, :, 1] = 0.90
    link_x[3, :, :, 0] = 0.80
    return {
        "node_x": node_x,
        "link_x": link_x,
        "service_x": service_x,
        "sla_x": sla_x,
        "node_mask": node_mask,
        "link_mask": link_mask,
    }


def make_mock_run(tmp_path):
    ensure_dataset()
    run_dir = tmp_path / "pressure02_run"
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


def test_pressure02_evidence_shape_and_range():
    batch = make_pressure02_batch()
    stats = compute_metric_evidence_stats_from_loader([batch], norm="pressure02")
    evidence = build_metric_evidence(batch, norm="pressure02", evidence_stats=stats)

    assert tuple(evidence.shape) == (4, 5)
    assert torch.all(evidence >= 0.0)
    assert torch.all(evidence <= 1.0)


def test_pressure02_stats_are_train_split_artifacts():
    batch = make_pressure02_batch()
    stats = compute_metric_evidence_stats_from_loader([batch], norm="pressure02")

    assert stats["version"] == "pressure02_v5_growth_rank"
    assert stats["norm"] == "pressure02"
    assert set(stats["evidence"]) == {"delay", "loss", "cpu", "queue", "bandwidth"}
    assert "rank_quantiles" in stats
    assert "queue_growth" in stats["aux"]
    assert "min_bandwidth_over_bandwidth" in stats["aux"]


def test_queue_pressure_responds_to_growth_and_queue_delay():
    batch = make_pressure02_batch()
    stats = compute_metric_evidence_stats_from_loader([batch], norm="pressure02")
    evidence = build_metric_evidence(batch, norm="pressure02", evidence_stats=stats)

    assert evidence[1, 3] > evidence[0, 3]


def test_bandwidth_pressure_responds_to_util_and_min_bandwidth_ratio():
    batch = make_pressure02_batch()
    stats = compute_metric_evidence_stats_from_loader([batch], norm="pressure02")
    evidence = build_metric_evidence(batch, norm="pressure02", evidence_stats=stats)

    assert evidence[2, 4] > evidence[0, 4]


def test_cpu_pressure_does_not_dominate_all_samples():
    batch = make_pressure02_batch()
    stats = compute_metric_evidence_stats_from_loader([batch], norm="pressure02")
    evidence = build_metric_evidence(batch, norm="pressure02", evidence_stats=stats)
    preds = evidence.argmax(dim=-1).tolist()

    assert preds.count(2) < len(preds)
    assert 3 in preds or 4 in preds


def test_metric_evidence_baseline_pressure02_outputs(tmp_path):
    run_dir = make_mock_run(tmp_path)

    run_cmd(
        [
            "src/analysis/evaluate_metric_evidence_baseline.py",
            "--run_dir",
            str(run_dir),
            "--evidence_norm",
            "pressure02",
        ]
    )

    json_path = run_dir / "analysis" / "metric_evidence_baseline_pressure02.json"
    confusion_path = run_dir / "analysis" / "metric_evidence_baseline_pressure02_confusion.csv"
    calibration_path = run_dir / "analysis" / "metric_evidence_pressure02_calibration.json"
    stats_path = run_dir / "artifacts" / "metric_evidence_stats_pressure02.json"
    assert json_path.exists()
    assert confusion_path.exists()
    assert calibration_path.exists()
    assert stats_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "metric_evidence_acc" in data
    assert "metric_majority_acc_valid_attr_only" in data
    assert "top_predicted_class_ratio" in data
