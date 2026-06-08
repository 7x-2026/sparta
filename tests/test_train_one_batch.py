from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.datasets.sparta_dataset import SPARTADataset
from src.losses.multitask_loss import SPARTALoss
from src.models.sparta import SPARTA
from src.utils.config import load_config

from .conftest import ROOT, ensure_dataset


def test_train_one_batch():
    ensure_dataset()
    config = load_config(ROOT / "configs/sparta_debug.yaml")
    batch = next(iter(DataLoader(SPARTADataset(ROOT / "data/debug/dataset/train.pkl", config), batch_size=2)))
    model = SPARTA(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = SPARTALoss()(model(batch), batch)["loss"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
