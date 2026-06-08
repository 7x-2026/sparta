from __future__ import annotations

from .conftest import ROOT, run_cmd


def test_run_debug_pipeline():
    run_cmd(["src/run_debug_pipeline.py", "--config", "configs/sparta_debug.yaml"])
    assert (ROOT / "data/debug/dataset/train.pkl").exists()
    assert (ROOT / "data/debug/dataset/val.pkl").exists()
    assert (ROOT / "data/debug/dataset/test.pkl").exists()
    assert (ROOT / "checkpoints/debug/lstm_best.pth").exists()
    assert (ROOT / "checkpoints/debug/transformer_best.pth").exists()
    assert (ROOT / "checkpoints/debug/sparta_best.pth").exists()
    assert (ROOT / "results/debug/lstm_test_results.csv").exists()
    assert (ROOT / "results/debug/transformer_test_results.csv").exists()
    assert (ROOT / "results/debug/sparta_test_results.csv").exists()
