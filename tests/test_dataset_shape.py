from __future__ import annotations

from src.datasets.sparta_dataset import SPARTADataset
from src.utils.config import load_config

from .conftest import ROOT, ensure_dataset


def test_dataset_shape():
    ensure_dataset()
    config = load_config(ROOT / "configs/sparta_debug.yaml")
    dataset = SPARTADataset(ROOT / "data/debug/dataset/train.pkl", config)
    sample = dataset[0]
    assert tuple(sample["node_x"].shape) == (4, 6, 8)
    assert tuple(sample["link_x"].shape) == (4, 8, 8)
    assert tuple(sample["service_x"].shape) == (4, 5)
    assert tuple(sample["sla_x"].shape) == (5,)
    assert tuple(sample["adj"].shape) == (6, 6)
    assert tuple(sample["node_mask"].shape) == (6,)
    assert tuple(sample["link_mask"].shape) == (8,)
    assert int(sample["risk_label"]) in {0, 1, 2}
