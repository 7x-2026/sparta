from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SPARTALoss(nn.Module):
    def __init__(
        self,
        lambda_node: float = 0.3,
        lambda_link: float = 0.3,
        lambda_metric: float = 0.2,
        ignore_index: int = -100,
        risk_class_weights=None,
    ):
        super().__init__()
        self.lambda_node = lambda_node
        self.lambda_link = lambda_link
        self.lambda_metric = lambda_metric
        self.ignore_index = ignore_index
        if risk_class_weights is not None:
            self.register_buffer("risk_class_weights", torch.as_tensor(risk_class_weights, dtype=torch.float32))
        else:
            self.risk_class_weights = None

    def _zero(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return outputs["risk_logits"].sum() * 0.0

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
            metric_loss = F.cross_entropy(outputs["metric_logits"][valid], batch["risk_metric"][valid], ignore_index=self.ignore_index)
        else:
            metric_loss = self._zero(outputs)
        total = risk_loss + self.lambda_node * node_loss + self.lambda_link * link_loss + self.lambda_metric * metric_loss
        return {
            "loss": total,
            "risk_loss": risk_loss.detach(),
            "node_loss": node_loss.detach(),
            "link_loss": link_loss.detach(),
            "metric_loss": metric_loss.detach(),
        }
