from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/sparta_debug.yaml"


def run_cmd(args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def ensure_dataset() -> None:
    if (ROOT / "data/debug/dataset/train.pkl").exists():
        return
    run_cmd(["src/simulation/make_debug_trace.py", "--config", CONFIG])
    run_cmd(["src/simulation/run_synthetic_simulation.py", "--config", CONFIG])
    run_cmd(["src/preprocessing/build_path_graph.py", "--config", CONFIG])
    run_cmd(["src/preprocessing/generate_labels.py", "--config", CONFIG])
    run_cmd(["src/preprocessing/split_dataset.py", "--config", CONFIG])
    run_cmd(["src/preprocessing/check_dataset_stats.py", "--config", CONFIG])
