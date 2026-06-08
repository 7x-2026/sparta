from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_debug.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    py = sys.executable
    steps = [
        [py, "src/simulation/make_debug_trace.py", "--config", config_path],
        [py, "src/simulation/run_synthetic_simulation.py", "--config", config_path],
        [py, "src/preprocessing/build_path_graph.py", "--config", config_path],
        [py, "src/preprocessing/generate_labels.py", "--config", config_path],
        [py, "src/preprocessing/split_dataset.py", "--config", config_path],
        [py, "src/preprocessing/check_dataset_stats.py", "--config", config_path],
        [py, "src/train.py", "--config", config_path, "--model", "lstm"],
        [py, "src/train.py", "--config", config_path, "--model", "transformer"],
        [py, "src/train.py", "--config", config_path, "--model", "sparta"],
        [py, "src/evaluate.py", "--config", config_path, "--model", "lstm"],
        [py, "src/evaluate.py", "--config", config_path, "--model", "transformer"],
        [py, "src/evaluate.py", "--config", config_path, "--model", "sparta"],
    ]
    for step in steps:
        print(f"[run_debug_pipeline] {' '.join(step)}")
        subprocess.run(step, check=True)
    print("SPARTA debug pipeline complete")


if __name__ == "__main__":
    main()
