from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

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
    assert manifest["raw_log_schema_version"] == "v1"
    assert "check_raw_log_schema" in manifest["completed_stages"]
