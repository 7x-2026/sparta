from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def compute_metric_class_weights(
    counts: list[int] | torch.Tensor,
    method: str = "inverse_sqrt_freq",
    smoothing: float = 1.0,
    effective_beta: float = 0.999,
) -> torch.Tensor:
    count_tensor = torch.as_tensor(counts, dtype=torch.float32)
    if count_tensor.numel() != 5:
        raise ValueError(f"Expected 5 metric class counts, got {count_tensor.numel()}")
    smooth_counts = count_tensor + float(smoothing)
    method = str(method or "inverse_sqrt_freq")
    if method == "inverse_sqrt_freq":
        weights = 1.0 / torch.sqrt(smooth_counts)
    elif method == "effective_number":
        beta = float(effective_beta)
        effective_num = 1.0 - torch.pow(torch.full_like(smooth_counts, beta), smooth_counts)
        weights = (1.0 - beta) / effective_num.clamp_min(1e-12)
    elif method in {"none", "uniform"}:
        weights = torch.ones_like(smooth_counts)
    else:
        raise ValueError(f"Unsupported metric_class_weighting: {method}")
    weights = weights / weights.mean().clamp_min(1e-12)
    return weights


def focal_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none", ignore_index=ignore_index)
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    pt = torch.exp(-ce[valid])
    focal = torch.pow(1.0 - pt, float(gamma)) * ce[valid]
    return focal.mean()


class SPARTALoss(nn.Module):
    def __init__(
        self,
        lambda_node: float = 0.3,
        lambda_link: float = 0.3,
        lambda_metric: float = 0.2,
        ignore_index: int = -100,
        risk_class_weights=None,
        metric_class_weights=None,
        metric_loss_type: str = "weighted_ce",
        metric_focal_gamma: float = 2.0,
        lambda_metric_soft: float = 0.0,
        metric_soft_loss_type: str = "none",
        metric_soft_temperature: float = 2.0,
    ):
        super().__init__()
        self.lambda_node = lambda_node
        self.lambda_link = lambda_link
        self.lambda_metric = lambda_metric
        self.ignore_index = ignore_index
        self.metric_loss_type = str(metric_loss_type or "weighted_ce")
        self.metric_focal_gamma = float(metric_focal_gamma)
        self.lambda_metric_soft = float(lambda_metric_soft)
        self.metric_soft_loss_type = str(metric_soft_loss_type or "none")
        self.metric_soft_temperature = float(metric_soft_temperature)
        if risk_class_weights is not None:
            self.register_buffer("risk_class_weights", torch.as_tensor(risk_class_weights, dtype=torch.float32))
        else:
            self.risk_class_weights = None
        if metric_class_weights is not None:
            self.register_buffer("metric_class_weights", torch.as_tensor(metric_class_weights, dtype=torch.float32))
        else:
            self.metric_class_weights = None

    def _zero(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return outputs["risk_logits"].sum() * 0.0

    def _metric_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.metric_class_weights
        if weights is not None:
            weights = weights.to(logits.device)
        if self.metric_loss_type == "focal":
            return focal_cross_entropy(
                logits,
                target,
                weight=weights,
                gamma=self.metric_focal_gamma,
                ignore_index=self.ignore_index,
            )
        if self.metric_loss_type == "weighted_ce":
            return F.cross_entropy(logits, target, weight=weights, ignore_index=self.ignore_index)
        if self.metric_loss_type == "ce":
            return F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        raise ValueError(f"Unsupported metric_loss_type: {self.metric_loss_type}")

    def _metric_soft_loss(self, logits: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        if self.lambda_metric_soft <= 0.0 or self.metric_soft_loss_type in {"", "none"}:
            return logits.sum() * 0.0
        if self.metric_soft_loss_type != "kl":
            raise ValueError(f"Unsupported metric_soft_loss_type: {self.metric_soft_loss_type}")
        temperature = max(self.metric_soft_temperature, 1e-6)
        target = torch.softmax(target_scores / temperature, dim=-1)
        pred_log = torch.log_softmax(logits / temperature, dim=-1)
        return F.kl_div(pred_log, target, reduction="batchmean") * (temperature * temperature)

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        weights = self.risk_class_weights
        if weights is not None:
            weights = weights.to(outputs["risk_logits"].device)
        risk_loss = F.cross_entropy(outputs["risk_logits"], batch["risk_label"], weight=weights)
        valid = batch["attr_mask"].bool()
        if "node_logits" in outputs and valid.any():
            node_loss = F.cross_entropy(outputs["node_logits"][valid], batch["risk_node"][valid], ignore_index=self.ignore_index)
        else:
            node_loss = self._zero(outputs)
        if "link_logits" in outputs and valid.any():
            link_loss = F.cross_entropy(outputs["link_logits"][valid], batch["risk_link"][valid], ignore_index=self.ignore_index)
        else:
            link_loss = self._zero(outputs)
        if "metric_logits" in outputs and valid.any():
            metric_loss = self._metric_loss(outputs["metric_logits"][valid], batch["risk_metric"][valid])
            if "risk_metric_scores" in batch:
                metric_soft_loss = self._metric_soft_loss(
                    outputs["metric_logits"][valid],
                    batch["risk_metric_scores"][valid].to(outputs["metric_logits"].device),
                )
            else:
                metric_soft_loss = self._zero(outputs)
        else:
            metric_loss = self._zero(outputs)
            metric_soft_loss = self._zero(outputs)
        total = (
            risk_loss
            + self.lambda_node * node_loss
            + self.lambda_link * link_loss
            + self.lambda_metric * metric_loss
            + self.lambda_metric_soft * metric_soft_loss
        )
        return {
            "loss": total,
            "loss_risk": risk_loss.detach(),
            "loss_node": node_loss.detach(),
            "loss_link": link_loss.detach(),
            "loss_metric": metric_loss.detach(),
            "loss_metric_soft": metric_soft_loss.detach(),
            "risk_loss": risk_loss.detach(),
            "node_loss": node_loss.detach(),
            "link_loss": link_loss.detach(),
            "metric_loss": metric_loss.detach(),
            "metric_soft_loss": metric_soft_loss.detach(),
        }
