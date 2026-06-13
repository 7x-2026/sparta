from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from .conftest import ROOT


MODELS = ["lstm", "transformer", "sparta"]
CLASSIFICATION_METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_acc",
    "risk_recall",
    "violation_recall",
    "auc",
]
ATTR_METRICS = ["node_attr_acc", "link_attr_acc", "metric_attr_acc"]
WINNER_METRICS = CLASSIFICATION_METRICS + ATTR_METRICS + ["overall_score"]


def csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_fake_run(run_root: Path, seed: int, idx: int, offset: float = 0.0) -> Path:
    run_dir = run_root / f"20260612_{idx:03d}_synthetic_full_seed{seed}"
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "dataset").mkdir()
    manifest = {
        "run_id": run_dir.name,
        "data_seed": seed,
        "train_seed": 0,
        "created_at": f"2026-06-12 10:{idx:02d}:00",
        "status": "completed",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows = []
    for model_idx, model in enumerate(MODELS):
        checkpoint = run_dir / "checkpoints" / f"{model}_best.pth"
        dataset = run_dir / "dataset" / "test.pkl"
        result_source = run_dir / "results" / f"{model}_test_results.csv"
        checkpoint.write_text("fake", encoding="utf-8")
        dataset.write_text("fake", encoding="utf-8")
        result_source.write_text("fake", encoding="utf-8")
        base = 0.55 + offset + model_idx * 0.02
        row = {
            "run_id": run_dir.name,
            "model": model,
            "checkpoint_path": str(checkpoint),
            "dataset_path": str(dataset),
            "result_source_file": str(result_source),
        }
        for metric in CLASSIFICATION_METRICS:
            row[metric] = base
        for metric in ATTR_METRICS:
            row[metric] = base if model == "sparta" else "N/A"
        rows.append(row)
    write_csv(run_dir / "results" / "all_test_results.csv", rows)
    return run_dir


def run_cmd(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, text=True, capture_output=True, check=True)


def latest_group(group_root: Path) -> Path:
    groups = sorted(group_root.glob("*_multiseed_synthetic_full"))
    assert groups
    return groups[-1]


def test_multiseed_winner_reports_from_run_ids(tmp_path):
    run_root = tmp_path / "run"
    group_root = tmp_path / "run_groups"
    runs = [
        make_fake_run(run_root, 42, 1, 0.00),
        make_fake_run(run_root, 2025, 2, 0.01),
        make_fake_run(run_root, 3407, 3, 0.02),
    ]

    run_cmd(
        [
            "src/analysis/summarize_multiseed.py",
            "--run_ids",
            *(str(path) for path in runs),
            "--group_name",
            "multiseed_synthetic_full",
            "--output_root",
            str(group_root),
            "--strict",
        ]
    )

    group_dir = latest_group(group_root)
    summary_rows = csv_rows(group_dir / "multiseed_summary.csv")
    mean_std_rows = csv_rows(group_dir / "multiseed_mean_std.csv")
    per_seed_rows = csv_rows(group_dir / "per_seed_winner_report.csv")
    count_rows = csv_rows(group_dir / "metric_winner_counts.csv")

    assert len(summary_rows) == 9
    assert len(mean_std_rows) == 3
    assert len(per_seed_rows) == len(runs) * len(WINNER_METRICS)
    assert {row["total_seeds"] for row in count_rows} == {"3"}
    assert (group_dir / "group_manifest.json").exists()
    assert (group_dir / "logs" / "summarize_multiseed.log").exists()
    assert not (run_root / "multiseed_summary.csv").exists()
    for run_dir in runs:
        seed_winner = run_dir / "results" / "seed_winner_report.csv"
        assert seed_winner.exists()
        assert len(csv_rows(seed_winner)) == len(WINNER_METRICS)


def test_multiseed_summary_dedupes_duplicate_seed_model_keys(tmp_path):
    run_root = tmp_path / "run"
    group_root = tmp_path / "run_groups"
    old_seed42 = make_fake_run(run_root, 42, 1, 0.00)
    runs = [
        old_seed42,
        make_fake_run(run_root, 2025, 2, 0.01),
        make_fake_run(run_root, 3407, 3, 0.02),
        make_fake_run(run_root, 42, 4, 0.10),
    ]

    run_cmd(
        [
            "src/analysis/summarize_multiseed.py",
            "--run_ids",
            *(str(path) for path in runs),
            "--group_name",
            "multiseed_synthetic_full",
            "--output_root",
            str(group_root),
        ]
    )

    group_dir = latest_group(group_root)
    summary_rows = csv_rows(group_dir / "multiseed_summary.csv")
    log_text = (group_dir / "logs" / "summarize_multiseed.log").read_text(encoding="utf-8")
    assert len(summary_rows) == 9
    assert "WARNING: duplicated key" in log_text
    assert str(old_seed42) not in {row["run_dir"] for row in summary_rows}
