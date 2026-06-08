from __future__ import annotations

from torch import nn

from src.models.common import PathTokenEncoder


class TransformerBaseline(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        data_cfg = config["data"]
        model_cfg = config["model"]
        hidden_dim = int(model_cfg.get("hidden_dim", 32))
        n_heads = int(model_cfg.get("n_heads", 2))
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}")
        self.encoder = PathTokenEncoder(
            data_cfg["node_feat_dim"],
            data_cfg["link_feat_dim"],
            data_cfg["service_feat_dim"],
            data_cfg["sla_feat_dim"],
            hidden_dim,
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=int(model_cfg.get("ffn_dim", hidden_dim * 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=int(model_cfg.get("temporal_layers", 1)))
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))

    def forward(self, batch: dict):
        x = self.encoder(batch)
        h = self.temporal(x)
        return {"risk_logits": self.head(h[:, -1, :])}


VanillaTransformer = TransformerBaseline
