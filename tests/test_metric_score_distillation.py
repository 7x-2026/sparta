from __future__ import annotations

import copy

import torch

from src.datasets.sparta_dataset import SPARTADataset
from src.losses.multitask_loss import SPARTALoss
from src.models.sparta import SPARTA
from src.utils.io import save_pickle


def make_sample(metric_scores=None, attr_mask: int = 1) -> dict:
    L, N, E = 4, 6, 8
    risk_metric = 2 if attr_mask else -100
    return {
        "sample_id": "s0",
        "service_id": 0,
        "time": 10,
        "scenario": "node_overload",
        "node_x": torch.zeros(L, N, 8).numpy(),
        "link_x": torch.zeros(L, E, 8).numpy(),
        "service_x": torch.zeros(L, 5).numpy(),
        "sla_x": torch.ones(5).numpy(),
        "adj": torch.eye(N).numpy(),
        "node_mask": torch.ones(N, dtype=torch.bool).numpy(),
        "link_mask": torch.ones(E, dtype=torch.bool).numpy(),
        "node_ids": [f"n{i}" for i in range(N)],
        "link_ids": [f"l{i}" for i in range(E)],
        "risk_label": 1 if attr_mask else 0,
        "risk_node": 1 if attr_mask else -100,
        "risk_link": 1 if attr_mask else -100,
        "risk_metric": risk_metric,
        "risk_metric_scores": metric_scores or [0.1, 0.2, 1.2, 0.3, 0.4],
        "attr_mask": attr_mask,
    }


def make_batch(attr_mask=None) -> dict[str, torch.Tensor]:
    attr_mask = attr_mask if attr_mask is not None else torch.tensor([True, False, True])
    batch_size = int(attr_mask.numel())
    return {
        "risk_label": torch.tensor([1, 0, 2][:batch_size], dtype=torch.long),
        "risk_node": torch.tensor([1, -100, 2][:batch_size], dtype=torch.long),
        "risk_link": torch.tensor([1, -100, 2][:batch_size], dtype=torch.long),
        "risk_metric": torch.tensor([2, -100, 4][:batch_size], dtype=torch.long),
        "risk_metric_scores": torch.tensor(
            [
                [0.1, 0.2, 1.2, 0.3, 0.4],
                [9.0, 9.0, 9.0, 9.0, 9.0],
                [0.2, 0.1, 0.3, 0.4, 1.3],
            ][:batch_size],
            dtype=torch.float32,
        ),
        "attr_mask": attr_mask.bool(),
    }


def make_outputs(batch_size: int = 3) -> dict[str, torch.Tensor]:
    return {
        "risk_logits": torch.randn(batch_size, 3, requires_grad=True),
        "node_logits": torch.randn(batch_size, 6, requires_grad=True),
        "link_logits": torch.randn(batch_size, 8, requires_grad=True),
        "metric_logits": torch.randn(batch_size, 5, requires_grad=True),
    }


def test_dataset_returns_risk_metric_scores_shape(tmp_path):
    path = tmp_path / "samples.pkl"
    save_pickle(path, [make_sample()])
    ds = SPARTADataset(path)

    item = ds[0]

    assert tuple(item["risk_metric_scores"].shape) == (5,)


def test_metric_soft_loss_only_uses_attr_mask_samples():
    loss_fn = SPARTALoss(lambda_metric_soft=0.5, metric_soft_loss_type="kl", metric_soft_temperature=2.0)
    outputs = make_outputs()
    batch = make_batch()
    loss_a = loss_fn(outputs, batch)["loss_metric_soft"]

    changed = copy.deepcopy(batch)
    changed["risk_metric_scores"] = changed["risk_metric_scores"].clone()
    changed["risk_metric_scores"][1] = torch.tensor([100.0, -100.0, -100.0, -100.0, -100.0])
    loss_b = loss_fn(outputs, changed)["loss_metric_soft"]

    assert torch.isclose(loss_a, loss_b)


def test_normal_only_batch_has_zero_metric_soft_loss():
    loss_fn = SPARTALoss(lambda_metric_soft=0.5, metric_soft_loss_type="kl")
    outputs = make_outputs(batch_size=1)
    batch = make_batch(attr_mask=torch.tensor([False]))

    loss = loss_fn(outputs, batch)

    assert torch.isclose(loss["loss_metric_soft"], torch.tensor(0.0))


def test_metric_soft_kl_loss_backpropagates():
    loss_fn = SPARTALoss(lambda_metric_soft=0.5, metric_soft_loss_type="kl", metric_soft_temperature=2.0)
    outputs = make_outputs()
    batch = make_batch()

    loss = loss_fn(outputs, batch)["loss"]
    loss.backward()

    assert outputs["metric_logits"].grad is not None
    assert torch.isfinite(outputs["metric_logits"].grad).all()


def test_risk_metric_scores_do_not_change_model_forward():
    config = {
        "data": {
            "input_window": 4,
            "max_nodes": 6,
            "max_links": 8,
            "node_feat_dim": 8,
            "link_feat_dim": 8,
            "service_feat_dim": 5,
            "sla_feat_dim": 5,
        },
        "model": {
            "hidden_dim": 16,
            "temporal_layers": 1,
            "topology_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "use_topology": True,
            "use_sla_gate": True,
            "use_metric_evidence_head": False,
        },
    }
    model = SPARTA(config).eval()
    batch = {
        "node_x": torch.zeros(2, 4, 6, 8),
        "link_x": torch.zeros(2, 4, 8, 8),
        "service_x": torch.zeros(2, 4, 5),
        "sla_x": torch.ones(2, 5),
        "adj": torch.eye(6).unsqueeze(0).repeat(2, 1, 1),
        "node_mask": torch.ones(2, 6, dtype=torch.bool),
        "link_mask": torch.ones(2, 8, dtype=torch.bool),
    }
    with_scores = {**batch, "risk_metric_scores": torch.randn(2, 5)}

    with torch.no_grad():
        out_a = model(batch)
        out_b = model(with_scores)

    assert torch.allclose(out_a["metric_logits"], out_b["metric_logits"])
