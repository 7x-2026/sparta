from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .conftest import ROOT


def run_cmd(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, text=True, capture_output=True, check=True, timeout=timeout)


def csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_multiseed_config(tmp_path: Path) -> Path:
    config = yaml.safe_load((ROOT / "configs/sparta_synthetic_full.yaml").read_text(encoding="utf-8"))
    config["run"]["output_root"] = str(tmp_path / "run")
    config["train"]["epochs"] = 1
    config["validation"]["min_samples_per_class"] = 1
    config_path = tmp_path / "sparta_synthetic_full.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_three_data_seed_runs_and_summary(tmp_path):
    config_path = write_multiseed_config(tmp_path)
    run_root = tmp_path / "run"
    seeds = [42, 2025, 3407]
    run_dirs: list[Path] = []

    for seed in seeds:
        run_cmd(
            [
                "src/run_synthetic_full_pipeline.py",
                "--config",
                str(config_path),
                "--data_seed",
                str(seed),
                "--train_seed",
                "0",
                "--experiment_name",
                f"synthetic_full_seed{seed}",
                "--fast_dev_run",
            ],
            timeout=360,
        )
        matches = sorted(run_root.glob(f"*_synthetic_full_seed{seed}"))
        assert matches
        run_dirs.append(matches[-1])

    assert len({path.name for path in run_dirs}) == 3

    for seed, run_dir in zip(seeds, run_dirs):
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["data_seed"] == seed
        assert manifest["train_seed"] == 0
        for dirname in ["raw_logs", "dataset", "audit", "checkpoints", "results"]:
            assert (run_dir / dirname).is_dir()
        assert (run_dir / "raw_logs" / "node_log.csv").exists()
        assert (run_dir / "dataset" / "train.pkl").exists()
        assert (run_dir / "audit" / "audit_report.txt").exists()
        for model in ["lstm", "transformer", "sparta"]:
            assert (run_dir / "checkpoints" / f"{model}_best.pth").exists()
        all_results = run_dir / "results" / "all_test_results.csv"
        assert all_results.exists()
        assert {row["model"] for row in csv_rows(all_results)} == {"lstm", "transformer", "sparta"}

    summary_path = tmp_path / "run" / "multiseed_summary.csv"
    run_cmd(
        [
            "src/analysis/summarize_multiseed.py",
            "--run_root",
            str(run_root),
            "--pattern",
            "synthetic_full_seed",
            "--output",
            str(summary_path),
        ]
    )
    mean_std_path = summary_path.with_name("multiseed_mean_std.csv")
    assert summary_path.exists()
    assert mean_std_path.exists()
    assert len(csv_rows(summary_path)) >= 9
