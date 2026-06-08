from __future__ import annotations

import torch
from torch import nn

from src.models.common import MLP, masked_mean


class TopologyEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_h: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        mask = node_mask.to(dtype=node_h.dtype, device=node_h.device)
        A = adj.to(device=node_h.device, dtype=node_h.dtype) * mask[:, :, None] * mask[:, None, :]
        eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).unsqueeze(0)
        A = torch.clamp(A + eye * mask[:, :, None], 0.0, 1.0)
        deg = A.sum(dim=-1, keepdim=True).clamp_min(1.0)
        A = A / deg
        h = node_h * mask[:, :, None]
        for linear, norm in zip(self.layers, self.norms):
            msg = torch.bmm(A, h)
            update = self.dropout(torch.relu(linear(msg)))
            h = norm(h + update)
            h = h * mask[:, :, None]
        return h


class SPARTA(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        data_cfg = config["data"]
        model_cfg = config["model"]
        self.input_window = int(data_cfg["input_window"])
        self.max_nodes = int(data_cfg["max_nodes"])
        self.max_links = int(data_cfg["max_links"])
        self.node_feat_dim = int(data_cfg["node_feat_dim"])
        self.link_feat_dim = int(data_cfg["link_feat_dim"])
        self.service_feat_dim = int(data_cfg["service_feat_dim"])
        self.sla_feat_dim = int(data_cfg["sla_feat_dim"])
        self.hidden_dim = int(model_cfg.get("hidden_dim", 32))
        self.num_classes = 3
        self.num_metrics = 5
        self.use_topology = bool(model_cfg.get("use_topology", True))
        self.use_sla_gate = bool(model_cfg.get("use_sla_gate", True))
        dropout = float(model_cfg.get("dropout", 0.1))
        n_heads = int(model_cfg.get("n_heads", 2))
        if self.hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by n_heads={n_heads}")

        self.node_mlp = MLP(self.node_feat_dim, self.hidden_dim, dropout=dropout, layers=2)
        self.link_mlp = MLP(self.link_feat_dim, self.hidden_dim, dropout=dropout, layers=2)
        self.service_mlp = MLP(self.service_feat_dim, self.hidden_dim, dropout=dropout, layers=2)
        self.sla_mlp = MLP(self.sla_feat_dim, self.hidden_dim, dropout=dropout, layers=2)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=n_heads,
            dim_feedforward=int(model_cfg.get("ffn_dim", self.hidden_dim * 4)),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=int(model_cfg.get("temporal_layers", 1)))
        self.topology = TopologyEncoder(self.hidden_dim, int(model_cfg.get("topology_layers", 1)), dropout=dropout)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.sla_gate = MLP(self.hidden_dim, self.hidden_dim, hidden_dim=self.hidden_dim, dropout=dropout, layers=2)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.node_head = nn.Linear(self.hidden_dim, 1)
        self.link_head = nn.Linear(self.hidden_dim, 1)
        self.metric_head = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.num_metrics))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        node_x = batch["node_x"]
        link_x = batch["link_x"]
        service_x = batch["service_x"]
        sla_x = batch["sla_x"]
        adj = batch["adj"]
        node_mask = batch["node_mask"]
        link_mask = batch["link_mask"]
        B, L, N, FN = node_x.shape
        B2, L2, E, FE = link_x.shape
        assert B2 == B and L2 == L
        assert L == self.input_window
        assert N == self.max_nodes
        assert E == self.max_links
        assert FN == self.node_feat_dim
        assert FE == self.link_feat_dim
        assert service_x.shape == (B, L, self.service_feat_dim)
        assert sla_x.shape == (B, self.sla_feat_dim)
        assert adj.shape == (B, N, N)
        assert node_mask.shape == (B, N)
        assert link_mask.shape == (B, E)

        node_h = self.node_mlp(node_x)
        link_h = self.link_mlp(link_x)
        service_h = self.service_mlp(service_x)
        sla_h = self.sla_mlp(sla_x)
        node_ctx = masked_mean(node_h, node_mask, dim=2)
        link_ctx = masked_mean(link_h, link_mask, dim=2)
        path_token = node_ctx + link_ctx + service_h
        temp_h = self.temporal(path_token)
        z_temp = temp_h[:, -1, :]

        node_topo_in = node_h.mean(dim=1)
        node_topo_h = self.topology(node_topo_in, adj, node_mask)
        z_topo = masked_mean(node_topo_h, node_mask, dim=1)
        z = self.fusion_norm(z_temp + z_topo) if self.use_topology else z_temp
        if self.use_sla_gate:
            gate = torch.sigmoid(self.sla_gate(sla_h))
            z_sla = z * (1.0 + gate)
        else:
            z_sla = z

        risk_logits = self.risk_head(z_sla)
        node_logits = self.node_head(node_topo_h).squeeze(-1)
        link_attr_h = link_h.mean(dim=1)
        link_logits = self.link_head(link_attr_h).squeeze(-1)
        metric_logits = self.metric_head(z_sla)
        node_logits = node_logits.masked_fill(~node_mask, -1e9)
        link_logits = link_logits.masked_fill(~link_mask, -1e9)

        assert risk_logits.shape == (B, self.num_classes)
        assert node_logits.shape == (B, N)
        assert link_logits.shape == (B, E)
        assert metric_logits.shape == (B, self.num_metrics)
        return {
            "risk_logits": risk_logits,
            "node_logits": node_logits,
            "link_logits": link_logits,
            "metric_logits": metric_logits,
            "z": z_sla,
        }
