from __future__ import annotations

import json
import os
import subprocess
import sys
import csv
from pathlib import Path

import yaml

from src.simulation.edgesimpy_adapter import try_import_edgesimpy

from .conftest import ROOT


def run_cmd(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, text=True, capture_output=True, check=True)


def test_edgesimpy_stub_generate_only_pipeline(tmp_path):
    config = yaml.safe_load((ROOT / "configs/sparta_edgesimpy_mvp.yaml").read_text(encoding="utf-8"))
    config["run"]["output_root"] = str(tmp_path / "run")
    config["experiment"]["name"] = "edgesimpy_mvp_test"
    config_path = tmp_path / "edgesimpy_mvp.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--generate_only", "--fast_dev_run"])

    runs = sorted((tmp_path / "run").glob("*_edgesimpy_mvp_test"))
    assert runs
    run_dir = runs[-1]
    raw_dir = run_dir / "raw_logs"
    for filename in ["node_log.csv", "link_log.csv", "service_log.csv", "path_log.csv", "sla_log.csv"]:
        assert (raw_dir / filename).exists()
    assert (raw_dir / "check_raw_log_schema_report.txt").exists()
    for split in ["train", "val", "test"]:
        assert (run_dir / "dataset" / f"{split}.pkl").exists()

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_source"] == "edgesimpy"
    assert manifest["simulator_backend"] == "edgesimpy_stub"
    assert manifest["effective_simulator_backend"] == "edgesimpy_stub"
    assert manifest["raw_log_schema_version"] == "v1"
    assert "check_raw_log_schema" in manifest["completed_stages"]
    audit_report = (run_dir / "audit" / "audit_report.txt").read_text(encoding="utf-8")
    assert "effective_simulator_backend: edgesimpy_stub" in audit_report

    with (run_dir / "audit" / "attribution_distribution.csv").open("r", newline="", encoding="utf-8") as f:
        attr_rows = list(csv.DictReader(f))
    cpu_rows = [
        row
        for row in attr_rows
        if row["field"] == "risk_metric" and row["value_name"] == "cpu" and row["split"] in {"train", "val", "test"}
    ]
    assert {row["split"] for row in cpu_rows} == {"train", "val", "test"}
    assert all(float(row["ratio"]) > 0.0 for row in cpu_rows)


def test_edgesimpy_real_generate_only_when_installed(tmp_path):
    installed, _, _ = try_import_edgesimpy()
    config = yaml.safe_load((ROOT / "configs/sparta_edgesimpy_real_mvp.yaml").read_text(encoding="utf-8"))
    config["run"]["output_root"] = str(tmp_path / "run")
    config["experiment"]["name"] = "edgesimpy_real_mvp_test"
    config_path = tmp_path / "edgesimpy_real_mvp.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--generate_only", "--fast_dev_run"])

    runs = sorted((tmp_path / "run").glob("*_edgesimpy_real_mvp_test"))
    assert runs
    run_dir = runs[-1]
    for filename in ["node_log.csv", "link_log.csv", "service_log.csv", "path_log.csv", "sla_log.csv"]:
        assert (run_dir / "raw_logs" / filename).exists()

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_source"] == "edgesimpy"
    assert manifest["simulator_backend"] == "edgesimpy"
    assert manifest["risk_injection"] == "adapter_cpu_overload_balancer_v1"
    assert bool(manifest["edgesimpy_installed"]) is installed
    assert bool(manifest["edgesimpy_adapter_fallback"]) is True
    assert bool(manifest["real_edgesimpy_objects_created"]) is False
    expected_effective_backend = "edgesimpy_adapter_fallback"
    assert manifest["effective_simulator_backend"] == expected_effective_backend
    assert manifest["effective_simulator_backend_at_generation"] == expected_effective_backend
    assert Path(manifest["raw_log_generation_provenance"]).exists()
    audit_report = (run_dir / "audit" / "audit_report.txt").read_text(encoding="utf-8")
    assert f"effective_simulator_backend: {expected_effective_backend}" in audit_report
    assert f"effective_simulator_backend_at_generation: {expected_effective_backend}" in audit_report

    with (run_dir / "audit" / "attribution_distribution.csv").open("r", newline="", encoding="utf-8") as f:
        attr_rows = list(csv.DictReader(f))
    cpu_rows = [
        row
        for row in attr_rows
        if row["field"] == "risk_metric" and row["value_name"] == "cpu" and row["split"] in {"train", "val", "test"}
    ]
    assert {row["split"] for row in cpu_rows} == {"train", "val", "test"}
    assert all(float(row["ratio"]) > 0.0 for row in cpu_rows)

    with (run_dir / "audit" / "attribution_majority_baseline.csv").open("r", newline="", encoding="utf-8") as f:
        majority_rows = list(csv.DictReader(f))
    all_row = next(row for row in majority_rows if row["split"] == "all")
    assert float(all_row["metric_majority_acc"]) <= 0.50
