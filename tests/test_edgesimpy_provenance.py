from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .conftest import ROOT


def run_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


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


def load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))


def load_provenance(run_dir: Path) -> dict:
    return json.loads((run_dir / "artifacts" / "raw_log_generation_provenance.json").read_text(encoding="utf-8"))


def test_edgesimpy_strict_generate_only_fails_without_verified_real_objects(tmp_path):
    config_path = write_config(tmp_path, "configs/sparta_edgesimpy_real_strict_mvp.yaml", "edgesimpy_strict_provenance_test")
    result = run_cmd(
        ["src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--generate_only", "--fast_dev_run"],
        check=False,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "EdgeSimPy Adapter Failed" in combined
    assert "verified real EdgeSimPy objects" in combined or "pip install -r requirements-edgesimpy.txt" in combined


def test_edgesimpy_fallback_provenance_and_eval_only_preserves_generation_fields(tmp_path):
    config_path = write_config(tmp_path, "configs/sparta_edgesimpy_adapter_mid.yaml", "edgesimpy_fallback_provenance_test")
    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--generate_only", "--fast_dev_run"])
    run_dir = latest_run(tmp_path, "edgesimpy_fallback_provenance_test")

    manifest = load_manifest(run_dir)
    provenance_path = Path(manifest["raw_log_generation_provenance"])
    assert provenance_path.exists()
    assert manifest["effective_simulator_backend"] == "edgesimpy_adapter_fallback"
    assert manifest["effective_simulator_backend_at_generation"] == "edgesimpy_adapter_fallback"
    assert manifest["edgesimpy_adapter_fallback_at_generation"] is True

    provenance = load_provenance(run_dir)
    assert provenance["effective_simulator_backend_at_generation"] == "edgesimpy_adapter_fallback"
    assert provenance["edgesimpy_adapter_fallback_at_generation"] is True
    assert provenance["real_edgesimpy_objects_created"] is False
    assert provenance["raw_log_generator_function"] == "src.simulation.synthetic_full_generator.generate_synthetic_full_logs"

    check = run_cmd(["src/analysis/check_run_provenance.py", "--run_dir", str(run_dir)])
    assert "PROVENANCE VERIFIED: effective_simulator_backend=edgesimpy_adapter_fallback" in check.stdout
    assert (run_dir / "artifacts" / "run_provenance_check_report.txt").exists()

    provenance_before = provenance_path.read_text(encoding="utf-8")
    generation_created_at = manifest["generation_created_at"]
    eval_result = run_cmd(
        ["src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--eval_only", "--resume_run", str(run_dir)],
        check=False,
    )
    assert eval_result.returncode != 0
    manifest_after = load_manifest(run_dir)
    assert provenance_path.read_text(encoding="utf-8") == provenance_before
    assert manifest_after["generation_created_at"] == generation_created_at
    assert manifest_after["effective_simulator_backend_at_generation"] == "edgesimpy_adapter_fallback"


def test_provenance_checker_detects_manifest_mismatch(tmp_path):
    config_path = write_config(tmp_path, "configs/sparta_edgesimpy_adapter_mid.yaml", "edgesimpy_mismatch_provenance_test")
    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--generate_only", "--fast_dev_run"])
    run_dir = latest_run(tmp_path, "edgesimpy_mismatch_provenance_test")

    manifest_path = run_dir / "run_manifest.json"
    manifest = load_manifest(run_dir)
    manifest["effective_simulator_backend"] = "edgesimpy_real"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    result = run_cmd(["src/analysis/check_run_provenance.py", "--run_dir", str(run_dir)], check=False)
    assert result.returncode != 0
    report = (run_dir / "artifacts" / "run_provenance_check_report.txt").read_text(encoding="utf-8")
    assert "PROVENANCE MISMATCH" in report
    assert load_manifest(run_dir)["provenance_inconsistent"] is True


def test_synthetic_full_provenance_is_unaffected(tmp_path):
    config_path = write_config(tmp_path, "configs/sparta_synthetic_full.yaml", "synthetic_full_provenance_test")
    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--generate_only", "--fast_dev_run"])
    run_dir = latest_run(tmp_path, "synthetic_full_provenance_test")

    manifest = load_manifest(run_dir)
    provenance = load_provenance(run_dir)
    assert manifest["effective_simulator_backend"] == "synthetic_full"
    assert manifest["effective_simulator_backend_at_generation"] == "synthetic_full"
    assert provenance["effective_simulator_backend_at_generation"] == "synthetic_full"

    result = run_cmd(["src/analysis/check_run_provenance.py", "--run_dir", str(run_dir)])
    assert "PROVENANCE VERIFIED: effective_simulator_backend=synthetic_full" in result.stdout
