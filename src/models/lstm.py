from __future__ import annotations

from torch import nn

from src.models.common import PathTokenEncoder


class LSTMBaseline(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        data_cfg = config["data"]
        model_cfg = config["model"]
        hidden_dim = int(model_cfg.get("hidden_dim", 32))
        self.encoder = PathTokenEncoder(
            data_cfg["node_feat_dim"],
            data_cfg["link_feat_dim"],
            data_cfg["service_feat_dim"],
            data_cfg["sla_feat_dim"],
            hidden_dim,
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=int(model_cfg.get("temporal_layers", 1)),
            batch_first=True,
            dropout=float(model_cfg.get("dropout", 0.1)) if int(model_cfg.get("temporal_layers", 1)) > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))

    def forward(self, batch: dict):
        x = self.encoder(batch)
        out, _ = self.lstm(x)
        return {"risk_logits": self.head(out[:, -1, :])}


LSTM = LSTMBaseline
