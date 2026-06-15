from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from src.simulation.edgesimpy_adapter import run_edgesimpy_real_smoke, try_import_edgesimpy

from .conftest import ROOT


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, text=True, capture_output=True, check=check)


def write_config(tmp_path: Path, source_config: str, experiment_name: str) -> Path:
    config = yaml.safe_load((ROOT / source_config).read_text(encoding="utf-8"))
    config.setdefault("run", {})["output_root"] = str(tmp_path / "run")
    config.setdefault("experiment", {})["name"] = experiment_name
    path = tmp_path / f"{experiment_name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def latest_run(tmp_path: Path, experiment_name: str) -> Path:
    runs = sorted((tmp_path / "run").glob(f"*_{experiment_name}"))
    assert runs
    return runs[-1]


def manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))


def test_synthetic_full_unaffected_by_edgesimpy_real_detection(tmp_path):
    config_path = write_config(tmp_path, "configs/sparta_synthetic_full.yaml", "synthetic_full_real_backend_guard")
    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--generate_only", "--fast_dev_run"])
    run_dir = latest_run(tmp_path, "synthetic_full_real_backend_guard")
    data = manifest(run_dir)
    assert data["effective_simulator_backend"] == "synthetic_full"
    assert data["real_object_created"] is False
    assert data["real_simulation_ran"] is False


def test_edgesimpy_stub_unaffected_by_real_detection(tmp_path):
    config_path = write_config(tmp_path, "configs/sparta_edgesimpy_mvp.yaml", "edgesimpy_stub_real_backend_guard")
    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--generate_only", "--fast_dev_run"])
    run_dir = latest_run(tmp_path, "edgesimpy_stub_real_backend_guard")
    data = manifest(run_dir)
    assert data["simulator_backend"] == "edgesimpy_stub"
    assert data["effective_simulator_backend"] == "edgesimpy_stub"
    assert data["edgesimpy_adapter_fallback"] is False


def test_edgesimpy_real_smoke_metadata_is_explicit():
    installed, _, _ = try_import_edgesimpy()
    smoke = run_edgesimpy_real_smoke({"edgesimpy": {"backend": "edgesimpy"}})
    assert smoke["import_success"] is installed
    for key in ["real_object_created", "real_simulation_ran", "created_object_types", "edgesimpy_object_counts", "error"]:
        assert key in smoke
    if smoke["real_simulation_ran"]:
        assert smoke["real_object_created"] is True
        assert {"Simulator", "EdgeServer", "Service", "User", "Application"}.issubset(set(smoke["created_object_types"]))


def test_edgesimpy_fallback_or_real_manifest_fields(tmp_path):
    config_path = write_config(tmp_path, "configs/sparta_edgesimpy_real_mvp.yaml", "edgesimpy_real_or_fallback_guard")
    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--generate_only", "--fast_dev_run"])
    run_dir = latest_run(tmp_path, "edgesimpy_real_or_fallback_guard")
    data = manifest(run_dir)
    real_ready = data["real_object_created"] and data["real_simulation_ran"]
    if real_ready:
        assert data["effective_simulator_backend"] == "edgesimpy_real"
        assert data["edgesimpy_backend_state"] == "edgesimpy_real_simulation_ran"
        assert data["edgesimpy_adapter_fallback"] is False
        assert data["created_object_types"]
    else:
        assert data["effective_simulator_backend"] == "edgesimpy_adapter_fallback"
        assert data["edgesimpy_backend_state"] == "edgesimpy_adapter_fallback"
        assert data["edgesimpy_adapter_fallback"] is True


def test_edgesimpy_strict_requires_real_object_and_simulation(tmp_path):
    config_path = write_config(tmp_path, "configs/sparta_edgesimpy_real_strict_mvp.yaml", "edgesimpy_strict_guard")
    result = run_cmd(
        ["src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--generate_only", "--fast_dev_run"],
        check=False,
    )
    if result.returncode == 0:
        run_dir = latest_run(tmp_path, "edgesimpy_strict_guard")
        data = manifest(run_dir)
        assert data["effective_simulator_backend"] == "edgesimpy_real"
        assert data["edgesimpy_backend_state"] == "edgesimpy_real_simulation_ran"
        assert data["edgesimpy_adapter_fallback"] is False
        assert data["real_object_created"] is True
        assert data["real_simulation_ran"] is True
        return
    combined = result.stdout + result.stderr
    assert "EdgeSimPy Adapter Failed" in combined
