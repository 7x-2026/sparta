from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.datasets.sparta_dataset import SPARTADataset
from src.models.sparta import build_metric_evidence, compute_metric_evidence_stats_from_loader

from .conftest import ROOT, ensure_dataset


def make_pressure_batch() -> dict[str, torch.Tensor]:
    batch_size, L, N, E = 2, 4, 3, 2
    node_x = torch.zeros(batch_size, L, N, 8)
    link_x = torch.zeros(batch_size, L, E, 8)
    service_x = torch.zeros(batch_size, L, 5)
    sla_x = torch.ones(batch_size, 5)
    node_mask = torch.ones(batch_size, N, dtype=torch.bool)
    link_mask = torch.ones(batch_size, E, dtype=torch.bool)

    node_x[0, :, :, 0] = 0.20
    node_x[0, :, 0, 3] = 0.10
    node_x[0, :, 1:, 3] = 0.90
    node_x[1, :, :, 0] = 0.80
    node_x[1, :, :, 3] = 0.90
    return {
        "node_x": node_x,
        "link_x": link_x,
        "service_x": service_x,
        "sla_x": sla_x,
        "node_mask": node_mask,
        "link_mask": link_mask,
    }


def test_pressure01_evidence_shape_and_range():
    evidence = build_metric_evidence(make_pressure_batch(), norm="pressure01")
    assert tuple(evidence.shape) == (2, 5)
    assert torch.all(evidence >= 0.0)
    assert torch.all(evidence <= 1.0)


def test_cpu_pressure_uses_cpu_util_and_inverse_available_cpu():
    evidence = build_metric_evidence(make_pressure_batch(), norm="pressure01")
    cpu_pressure = evidence[:, 2]
    assert torch.isclose(cpu_pressure[0], torch.tensor(0.90), atol=1e-5)
    assert torch.isclose(cpu_pressure[1], torch.tensor(0.80), atol=1e-5)


def test_available_cpu_direction_is_not_reversed():
    batch = make_pressure_batch()
    high_available = batch.copy()
    low_available = make_pressure_batch()
    high_available["node_x"] = batch["node_x"].clone()
    high_available["node_x"][:, :, :, 3] = 0.95
    low_available["node_x"][:, :, :, 3] = 0.05

    high_cpu_pressure = build_metric_evidence(high_available, norm="pressure01")[:, 2].mean()
    low_cpu_pressure = build_metric_evidence(low_available, norm="pressure01")[:, 2].mean()

    assert low_cpu_pressure > high_cpu_pressure


def test_metric_evidence_stats_are_computed_from_train_split():
    ensure_dataset()
    config = {
        "data": {
            "dataset_dir": str(ROOT / "data/debug/dataset"),
            "input_window": 4,
            "max_nodes": 6,
            "max_links": 8,
            "node_feat_dim": 8,
            "link_feat_dim": 8,
            "service_feat_dim": 5,
            "sla_feat_dim": 5,
        }
    }
    train_ds = SPARTADataset(ROOT / "data/debug/dataset/train.pkl", config)
    loader = DataLoader(train_ds, batch_size=16, shuffle=False, num_workers=0)
    stats = compute_metric_evidence_stats_from_loader(loader)

    assert "queue_len_p95" in stats
    assert "queue_delay_p95" in stats
    assert "delay_ratio_p95" in stats
    assert "loss_ratio_p95" in stats
    assert "bandwidth_pressure_p95" in stats
    assert stats["version"] == "pressure01_v3_excess"
    assert "evidence_raw" in stats
    assert set(stats["evidence"]) == {"delay", "loss", "cpu", "queue", "bandwidth"}
    for metric_name in ["delay", "loss", "cpu", "queue", "bandwidth"]:
        for key in ["p50", "p90", "p95", "max"]:
            assert key in stats["evidence_raw"][metric_name]
