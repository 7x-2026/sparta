from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from src.utils.io import load_pickle

from .conftest import ROOT


REQUIRED_RAW_LOGS = [
    "node_log.csv",
    "link_log.csv",
    "service_log.csv",
    "path_log.csv",
    "sla_log.csv",
]


def run_cmd(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, text=True, capture_output=True, check=True)


def latest_run_dir() -> Path:
    runs = sorted((ROOT / "run").glob("*_synthetic_full"), key=lambda path: path.stat().st_mtime)
    assert runs
    return runs[-1]


def mtimes(paths: list[Path]) -> dict[Path, float | None]:
    return {path: path.stat().st_mtime if path.exists() else None for path in paths}


def csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_synthetic_full_run_dirs_and_eval_outputs():
    before = set((ROOT / "run").glob("*_synthetic_full")) if (ROOT / "run").exists() else set()
    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", "configs/sparta_synthetic_full.yaml", "--generate_only", "--fast_dev_run"])
    after = set((ROOT / "run").glob("*_synthetic_full"))
    new_runs = sorted(after - before, key=lambda path: path.name)
    assert new_runs
    run_dir = new_runs[-1]

    for dirname in ["config", "raw_logs", "processed", "dataset", "checkpoints", "logs", "audit", "results", "artifacts"]:
        assert (run_dir / dirname).is_dir()
    for filename in REQUIRED_RAW_LOGS:
        assert (run_dir / "raw_logs" / filename).exists()
    for split in ["train", "val", "test"]:
        assert (run_dir / "dataset" / f"{split}.pkl").exists()
    for filename in ["label_distribution.csv", "scenario_distribution.csv", "label_by_scenario.csv", "attribution_distribution.csv", "attribution_majority_baseline.csv", "split_shift_report.csv", "audit_report.txt"]:
        assert (run_dir / "audit" / filename).exists()
    assert not list((run_dir / "checkpoints").glob("*.pth"))

    samples = load_pickle(run_dir / "dataset" / "train.pkl")
    assert samples
    for key in ["node_x", "link_x", "service_x", "sla_x", "adj", "risk_label", "risk_node", "risk_link", "risk_metric"]:
        assert key in samples[0]

    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", "configs/sparta_synthetic_full.yaml", "--train_only", "--resume_run", str(run_dir)])
    for model in ["lstm", "transformer", "sparta"]:
        assert (run_dir / "checkpoints" / f"{model}_best.pth").exists()
        assert (run_dir / "checkpoints" / f"{model}_last.pth").exists()
        assert (run_dir / "logs" / f"{model}_train.log").exists()

    protected = [
        run_dir / "raw_logs" / "node_log.csv",
        run_dir / "processed" / "path_graphs.pkl",
        run_dir / "dataset" / "train.pkl",
        run_dir / "audit" / "audit_report.txt",
    ]
    protected_before = mtimes(protected)
    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", "configs/sparta_synthetic_full.yaml", "--eval_only", "--resume_run", str(run_dir)])
    assert mtimes(protected) == protected_before

    all_results = run_dir / "results" / "all_test_results.csv"
    assert all_results.exists()
    rows = csv_rows(all_results)
    assert {row["model"] for row in rows} == {"lstm", "transformer", "sparta"}
    assert len({row["result_source_file"] for row in rows}) == 3
    assert len({row["checkpoint_path"] for row in rows}) == 3
    for row in rows:
        assert row["run_id"] == run_dir.name
        assert Path(row["result_source_file"]).exists()
        assert Path(row["checkpoint_path"]).exists()
    for model in ["lstm", "transformer", "sparta"]:
        for suffix in ["test_results", "confusion_matrix", "prediction_distribution", "per_class_recall", "per_class_f1"]:
            assert (run_dir / "results" / f"{model}_{suffix}.csv").exists()
        assert (run_dir / "logs" / f"{model}_eval.log").exists()

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == run_dir.name
    assert manifest["status"] == "completed"
    assert manifest["missing_outputs"] == []
    assert {"generate_raw_logs", "build_path_graph", "generate_labels_attribution", "split_dataset", "audit", "train", "eval"}.issubset(set(manifest["completed_stages"]))


def test_synthetic_full_train_only_requires_dataset(tmp_path):
    config = yaml.safe_load((ROOT / "configs/sparta_synthetic_full.yaml").read_text(encoding="utf-8"))
    config["run"]["output_root"] = str(tmp_path / "run")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "src/run_synthetic_full_pipeline.py", "--config", str(config_path), "--train_only"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Dataset files not found. Please run:" in result.stderr
