from __future__ import annotations

from torch.utils.data import DataLoader

from src.datasets.sparta_dataset import SPARTADataset
from src.models.lstm import LSTMBaseline
from src.models.sparta import SPARTA
from src.models.transformer import TransformerBaseline
from src.utils.config import load_config

from .conftest import ROOT, ensure_dataset


def test_model_forward():
    ensure_dataset()
    config = load_config(ROOT / "configs/sparta_debug.yaml")
    batch = next(iter(DataLoader(SPARTADataset(ROOT / "data/debug/dataset/train.pkl", config), batch_size=2)))
    outputs = SPARTA(config)(batch)
    assert tuple(outputs["risk_logits"].shape) == (2, 3)
    assert tuple(outputs["node_logits"].shape) == (2, 6)
    assert tuple(outputs["link_logits"].shape) == (2, 8)
    assert tuple(outputs["metric_logits"].shape) == (2, 5)
    assert tuple(LSTMBaseline(config)(batch)["risk_logits"].shape) == (2, 3)
    assert tuple(TransformerBaseline(config)(batch)["risk_logits"].shape) == (2, 3)
