from __future__ import annotations

import torch

from src.losses.multitask_loss import SPARTALoss, compute_metric_class_weights


def make_outputs(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "risk_logits": torch.randn(batch_size, 3, requires_grad=True),
        "node_logits": torch.randn(batch_size, 6, requires_grad=True),
        "link_logits": torch.randn(batch_size, 8, requires_grad=True),
        "metric_logits": torch.randn(batch_size, 5, requires_grad=True),
    }


def make_batch(metric_labels: list[int]) -> dict[str, torch.Tensor]:
    return {
        "risk_label": torch.tensor([0, 1, 2, 1], dtype=torch.long),
        "risk_node": torch.tensor([0, -100, 2, -100], dtype=torch.long),
        "risk_link": torch.tensor([1, -100, 3, -100], dtype=torch.long),
        "risk_metric": torch.tensor(metric_labels, dtype=torch.long),
        "attr_mask": torch.tensor([1, 0, 1, 0], dtype=torch.bool),
    }


def test_metric_loss_ignores_attr_mask_zero_samples():
    outputs = make_outputs()
    loss_fn = SPARTALoss(lambda_node=0.0, lambda_link=0.0, lambda_metric=1.0, metric_loss_type="weighted_ce")
    batch_a = make_batch([2, -100, 4, -100])
    batch_b = make_batch([2, 1, 4, 3])

    loss_a = loss_fn(outputs, batch_a)["loss_metric"]
    loss_b = loss_fn(outputs, batch_b)["loss_metric"]

    assert torch.allclose(loss_a, loss_b)


def test_metric_class_weights_are_finite_and_nonzero_for_supported_classes():
    weights = compute_metric_class_weights([10, 2, 5, 1, 3], method="inverse_sqrt_freq")
    assert weights.shape == (5,)
    assert torch.isfinite(weights).all()
    assert float(weights[2]) > 0.0
    assert float(weights[4]) > 0.0


def test_metric_class_weights_smooth_missing_classes():
    weights = compute_metric_class_weights([10, 0, 5, 0, 3], method="effective_number")
    assert weights.shape == (5,)
    assert torch.isfinite(weights).all()
    assert (weights > 0).all()


def test_metric_focal_loss_backward():
    outputs = make_outputs()
    weights = compute_metric_class_weights([1, 2, 8, 3, 1], method="inverse_sqrt_freq")
    loss_fn = SPARTALoss(
        lambda_node=0.0,
        lambda_link=0.0,
        lambda_metric=1.0,
        metric_class_weights=weights,
        metric_loss_type="focal",
        metric_focal_gamma=2.0,
    )
    batch = make_batch([2, -100, 4, -100])
    loss = loss_fn(outputs, batch)["loss"]

    assert torch.isfinite(loss)
    loss.backward()
    assert outputs["metric_logits"].grad is not None
    assert torch.isfinite(outputs["metric_logits"].grad).all()
