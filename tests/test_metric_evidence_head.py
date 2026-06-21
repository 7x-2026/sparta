from __future__ import annotations

import copy

import torch

from src.models.sparta import MetricEvidenceHead, SPARTA, build_metric_evidence
from src.utils.config import load_config

from .conftest import ROOT


def make_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "node_x": torch.rand(batch_size, 4, 6, 8),
        "link_x": torch.rand(batch_size, 4, 8, 8),
        "service_x": torch.rand(batch_size, 4, 5),
        "sla_x": torch.rand(batch_size, 5).clamp_min(0.1),
        "adj": torch.eye(6).repeat(batch_size, 1, 1),
        "node_mask": torch.ones(batch_size, 6, dtype=torch.bool),
        "link_mask": torch.ones(batch_size, 8, dtype=torch.bool),
    }


def test_metric_evidence_shape():
    batch = make_batch()
    evidence = build_metric_evidence(batch)
    assert tuple(evidence.shape) == (4, 5)
    assert torch.isfinite(evidence).all()


def test_metric_evidence_head_logits_shape():
    head = MetricEvidenceHead(hidden_dim=32, dropout=0.0, alpha=0.5)
    z_sla = torch.randn(4, 32)
    evidence = torch.randn(4, 5)
    logits = head(z_sla, evidence)
    assert tuple(logits.shape) == (4, 5)


def test_sparta_without_metric_evidence_head_is_unchanged():
    config = load_config(ROOT / "configs" / "sparta_debug.yaml")
    config = copy.deepcopy(config)
    config["model"]["use_metric_evidence_head"] = False
    model = SPARTA(config)
    out = model(make_batch())
    assert tuple(out["metric_logits"].shape) == (4, 5)
    assert "metric_evidence" not in out


def test_sparta_with_metric_evidence_head_uses_evidence():
    config = load_config(ROOT / "configs" / "sparta_debug.yaml")
    config = copy.deepcopy(config)
    config["model"]["use_metric_evidence_head"] = True
    config["model"]["metric_evidence_bias_alpha"] = 0.5
    config["model"]["metric_evidence_norm"] = "zscore"
    config["model"]["metric_evidence_use_bias"] = True
    model = SPARTA(config)
    out = model(make_batch())
    assert tuple(out["metric_logits"].shape) == (4, 5)
    assert tuple(out["metric_evidence"].shape) == (4, 5)


def test_metric_evidence_bias_alpha_changes_logits():
    head = MetricEvidenceHead(hidden_dim=32, dropout=0.0, alpha=0.0, use_bias=True)
    head.eval()
    z_sla = torch.randn(4, 32)
    evidence = torch.randn(4, 5)
    logits_alpha_0 = head(z_sla, evidence)
    head.alpha = 1.0
    logits_alpha_1 = head(z_sla, evidence)
    assert not torch.allclose(logits_alpha_0, logits_alpha_1)
