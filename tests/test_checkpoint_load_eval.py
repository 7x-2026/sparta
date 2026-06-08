from __future__ import annotations

from pathlib import Path

from src.evaluate import evaluate_once
from src.utils.config import load_config

from .conftest import ROOT, ensure_dataset, run_cmd


def test_checkpoint_load_eval():
    ensure_dataset()
    checkpoint = ROOT / "checkpoints/debug/sparta_best.pth"
    if not checkpoint.exists():
        run_cmd(["src/train.py", "--config", "configs/sparta_debug.yaml", "--model", "sparta"])
    assert checkpoint.exists()
    metrics = evaluate_once(load_config(ROOT / "configs/sparta_debug.yaml"), "sparta", checkpoint)
    assert "macro_f1" in metrics
    assert metrics["method"] == "sparta"
