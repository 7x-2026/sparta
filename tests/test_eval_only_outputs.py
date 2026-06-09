from __future__ import annotations

from src.utils.output_checker import REQUIRED_OUTPUTS_EVAL, check_outputs_or_report


def test_eval_outputs_require_per_model_files(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "all_test_results.csv").write_text("run_id,model\nrun_1,lstm\n", encoding="utf-8")
    checked = check_outputs_or_report(tmp_path, REQUIRED_OUTPUTS_EVAL)
    assert "results/all_test_results.csv" in checked["existing_outputs"]
    assert "results/lstm_test_results.csv" in checked["missing_outputs"]
    assert "results/transformer_test_results.csv" in checked["missing_outputs"]
    assert "results/sparta_test_results.csv" in checked["missing_outputs"]
