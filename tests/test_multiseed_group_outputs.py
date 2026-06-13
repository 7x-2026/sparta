from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from src.utils.group_manager import create_group_dir, save_group_manifest, write_run_list

from .conftest import ROOT


MODELS = ["lstm", "transformer", "sparta"]
SUMMARY_METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_acc",
    "risk_recall",
    "violation_recall",
    "auc",
    "node_attr_acc",
    "link_attr_acc",
    "metric_attr_acc",
]


def csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_fake_run(run_root: Path, seed: int, offset: float = 0.0) -> Path:
    run_dir = run_root / f"20260612_001_synthetic_full_seed{seed}"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "dataset").mkdir()
    manifest = {
        "run_id": run_dir.name,
        "data_seed": seed,
        "train_seed": 0,
        "status": "completed",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    for idx, model in enumerate(MODELS):
        checkpoint = run_dir / "checkpoints" / f"{model}_best.pth"
        dataset = run_dir / "dataset" / "test.pkl"
        result_source = run_dir / "results" / f"{model}_test_results.csv"
        checkpoint.write_text("fake", encoding="utf-8")
        dataset.write_text("fake", encoding="utf-8")
        result_source.write_text("fake", encoding="utf-8")
        row = {
            "run_id": run_dir.name,
            "model": model,
            "checkpoint_path": str(checkpoint),
            "dataset_path": str(dataset),
            "result_source_file": str(result_source),
        }
        for metric in SUMMARY_METRICS:
            row[metric] = "N/A" if model != "sparta" and metric.endswith("_attr_acc") else 0.50 + offset + idx * 0.01
        rows.append(row)
    write_csv(run_dir / "results" / "all_test_results.csv", rows)
    return run_dir


def run_cmd(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, text=True, capture_output=True, check=True)


def test_summarize_multiseed_uses_run_list_only(tmp_path):
    run_root = tmp_path / "run"
    selected_runs = [make_fake_run(run_root, seed, idx * 0.02) for idx, seed in enumerate([42, 2025, 3407])]
    make_fake_run(run_root, 9999, 0.90)

    group_dir = create_group_dir(tmp_path / "run_groups", "multiseed_synthetic_full")
    write_run_list(group_dir, selected_runs)
    save_group_manifest(group_dir, {"group_id": group_dir.name, "status": "running", "run_dirs": [str(path) for path in selected_runs]})

    run_cmd(
        [
            "src/analysis/summarize_multiseed.py",
            "--run_list",
            str(group_dir / "run_list.txt"),
            "--output_dir",
            str(group_dir),
            "--group_name",
            "multiseed_synthetic_full",
        ]
    )

    summary_path = group_dir / "multiseed_summary.csv"
    mean_std_path = group_dir / "multiseed_mean_std.csv"
    winner_path = group_dir / "per_seed_winner_report.csv"
    winner_count_path = group_dir / "metric_winner_counts.csv"
    assert summary_path.exists()
    assert mean_std_path.exists()
    assert winner_path.exists()
    assert winner_count_path.exists()
    assert (group_dir / "group_manifest.json").exists()
    assert (group_dir / "logs" / "summarize_multiseed.log").exists()
    assert not (run_root / "multiseed_summary.csv").exists()

    summary_rows = csv_rows(summary_path)
    mean_std_rows = csv_rows(mean_std_path)
    winner_rows = csv_rows(winner_path)
    winner_count_rows = csv_rows(winner_count_path)
    assert len(summary_rows) == 9
    assert len(mean_std_rows) == 3
    assert len(winner_rows) == 33
    assert len(winner_count_rows) == 11
    assert {row["model"] for row in summary_rows} == set(MODELS)
    assert {row["data_seed"] for row in summary_rows} == {"42", "2025", "3407"}
    assert "9999" not in {row["data_seed"] for row in summary_rows}
    assert all(row["group_id"] == group_dir.name for row in summary_rows + mean_std_rows + winner_rows + winner_count_rows)
    for run_dir in selected_runs:
        assert (run_dir / "results" / "seed_winner_report.csv").exists()
