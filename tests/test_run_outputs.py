from __future__ import annotations

from src.utils.output_checker import REQUIRED_OUTPUTS_FULL, REQUIRED_OUTPUTS_GENERATE


def test_required_output_lists_include_run_artifacts():
    assert "raw_logs/node_log.csv" in REQUIRED_OUTPUTS_GENERATE
    assert "dataset/train.pkl" in REQUIRED_OUTPUTS_FULL
    assert "checkpoints/lstm_best.pth" in REQUIRED_OUTPUTS_FULL
    assert "results/all_test_results.csv" in REQUIRED_OUTPUTS_FULL
    assert "logs/sparta_train.log" in REQUIRED_OUTPUTS_FULL
