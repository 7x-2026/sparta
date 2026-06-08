from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.datasets.sparta_dataset import SPARTADataset
from src.losses.multitask_loss import SPARTALoss
from src.models.sparta import SPARTA
from src.utils.config import load_config

from .conftest import ROOT, ensure_dataset


def test_loss_backward():
    ensure_dataset()
    config = load_config(ROOT / "configs/sparta_debug.yaml")
    batch = next(iter(DataLoader(SPARTADataset(ROOT / "data/debug/dataset/train.pkl", config), batch_size=2)))
    model = SPARTA(config)
    loss = SPARTALoss()(model(batch), batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)
