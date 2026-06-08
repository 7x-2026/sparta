from __future__ import annotations

import torch
from torch import nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.to(device=x.device, dtype=x.dtype)
    if x.dim() == 4 and dim == 2:
        mask = mask[:, None, :, None]
    elif x.dim() == 3 and dim == 1:
        mask = mask[:, :, None]
    else:
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)
    masked = x * mask
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return masked.sum(dim=dim) / denom


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None, dropout: float = 0.1, layers: int = 2):
        super().__init__()
        hidden = hidden_dim or out_dim
        if layers <= 1:
            self.net = nn.Linear(in_dim, out_dim)
            return
        modules: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(layers - 2):
            modules.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        modules.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PathTokenEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        link_feat_dim: int,
        service_feat_dim: int,
        sla_feat_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        raw_dim = node_feat_dim + link_feat_dim + service_feat_dim + sla_feat_dim
        self.token_mlp = MLP(raw_dim, hidden_dim, hidden_dim=hidden_dim, dropout=dropout, layers=2)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        node_x = batch["node_x"]
        link_x = batch["link_x"]
        service_x = batch["service_x"]
        sla_x = batch["sla_x"]
        node_mask = batch.get("node_mask")
        link_mask = batch.get("link_mask")
        B, L = node_x.shape[:2]
        node_pool = masked_mean(node_x, node_mask, dim=2)
        link_pool = masked_mean(link_x, link_mask, dim=2)
        sla_expand = sla_x[:, None, :].expand(B, L, -1)
        raw = torch.cat([node_pool, link_pool, service_x, sla_expand], dim=-1)
        return self.token_mlp(raw)
