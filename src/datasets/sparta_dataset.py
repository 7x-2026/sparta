from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.utils.io import load_pickle


def _metric_score_fallback(sample: dict) -> list[float]:
    metric = int(sample.get("risk_metric", -100))
    if 0 <= metric < 5:
        scores = [0.0] * 5
        scores[metric] = 1.0
        return scores
    return [0.0] * 5


class SPARTADataset(Dataset):
    def __init__(self, pkl_path: str | Path, config: dict | None = None):
        self.path = Path(pkl_path)
        self.samples = load_pickle(self.path)
        self.config = config or {}
        if not isinstance(self.samples, list):
            raise TypeError(f"{self.path} must contain a list of sample dicts")
        if len(self.samples) == 0:
            raise ValueError(f"{self.path} contains no samples")
        self._check_first_sample()

    def _check_first_sample(self) -> None:
        sample = self.samples[0]
        data_cfg = self.config.get("data", {})
        L = data_cfg.get("input_window", sample["node_x"].shape[0])
        max_nodes = data_cfg.get("max_nodes", sample["node_x"].shape[1])
        max_links = data_cfg.get("max_links", sample["link_x"].shape[1])
        node_feat_dim = data_cfg.get("node_feat_dim", sample["node_x"].shape[2])
        link_feat_dim = data_cfg.get("link_feat_dim", sample["link_x"].shape[2])
        service_feat_dim = data_cfg.get("service_feat_dim", sample["service_x"].shape[1])
        sla_feat_dim = data_cfg.get("sla_feat_dim", sample["sla_x"].shape[0])
        assert sample["node_x"].shape == (L, max_nodes, node_feat_dim)
        assert sample["link_x"].shape == (L, max_links, link_feat_dim)
        assert sample["service_x"].shape == (L, service_feat_dim)
        assert sample["sla_x"].shape == (sla_feat_dim,)
        assert sample["adj"].shape == (max_nodes, max_nodes)
        assert sample["node_mask"].shape == (max_nodes,)
        assert sample["link_mask"].shape == (max_links,)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        item = {
            "node_x": torch.as_tensor(sample["node_x"], dtype=torch.float32),
            "link_x": torch.as_tensor(sample["link_x"], dtype=torch.float32),
            "service_x": torch.as_tensor(sample["service_x"], dtype=torch.float32),
            "sla_x": torch.as_tensor(sample["sla_x"], dtype=torch.float32),
            "adj": torch.as_tensor(sample["adj"], dtype=torch.float32),
            "node_mask": torch.as_tensor(sample["node_mask"], dtype=torch.bool),
            "link_mask": torch.as_tensor(sample["link_mask"], dtype=torch.bool),
            "risk_label": torch.tensor(int(sample["risk_label"]), dtype=torch.long),
            "risk_node": torch.tensor(int(sample["risk_node"]), dtype=torch.long),
            "risk_link": torch.tensor(int(sample["risk_link"]), dtype=torch.long),
            "risk_metric": torch.tensor(int(sample["risk_metric"]), dtype=torch.long),
            "risk_metric_scores": torch.as_tensor(
                sample.get("risk_metric_scores", _metric_score_fallback(sample)),
                dtype=torch.float32,
            ),
            "attr_mask": torch.tensor(int(sample["attr_mask"]), dtype=torch.bool),
            "sample_time": torch.tensor(int(sample["time"]), dtype=torch.long),
            "service_id": torch.tensor(int(sample["service_id"]), dtype=torch.long),
        }
        return item
