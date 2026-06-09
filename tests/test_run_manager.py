from __future__ import annotations

import json

from src.utils.run_manager import (
    check_required_outputs,
    create_run_dir,
    init_run_structure,
    save_run_manifest,
    update_run_manifest,
)


def test_run_manager_creates_incrementing_run_dirs(tmp_path):
    first = create_run_dir(tmp_path, "synthetic_full")
    second = create_run_dir(tmp_path, "synthetic_full")
    assert first != second
    assert first.name.endswith("_001_synthetic_full")
    assert second.name.endswith("_002_synthetic_full")

    paths = init_run_structure(first)
    for name in ["config", "raw_logs", "processed", "dataset", "checkpoints", "logs", "audit", "results", "artifacts"]:
        assert paths[name].is_dir()

    save_run_manifest(first, {"run_id": first.name, "status": "running", "completed_stages": [], "missing_outputs": []})
    update_run_manifest(first, {"status": "completed", "completed_stages": ["generate"]})
    manifest = json.loads((first / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == first.name
    assert manifest["status"] == "completed"
    assert manifest["completed_stages"] == ["generate"]


def test_check_required_outputs_reports_missing(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "all_test_results.csv").write_text("x\n", encoding="utf-8")
    checked = check_required_outputs(tmp_path, ["results/lstm_test_results.csv", "results/all_test_results.csv"])
    assert checked["existing_outputs"] == ["results/all_test_results.csv"]
    assert checked["missing_outputs"] == ["results/lstm_test_results.csv"]
