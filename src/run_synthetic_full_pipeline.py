from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_synthetic_full.yaml")
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def make_runtime_config(config_path: str, fast_dev_run: bool) -> str:
    if not fast_dev_run:
        return config_path

    config = load_config(config_path)
    runtime = copy.deepcopy(config)
    runtime["data"]["num_steps"] = 120
    runtime["data"]["num_services"] = 8
    runtime["data"]["num_users"] = 32
    runtime["train"]["epochs"] = 1
    runtime["train"]["batch_size"] = 16
    runtime["validation"]["min_samples_per_class"] = 1
    runtime_path = resolve_path(runtime, Path(runtime["data"]["base_dir"]) / "runtime_fast_dev_config.yaml")
    ensure_dir(runtime_path.parent)
    with runtime_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(runtime, f, sort_keys=False)
    return str(runtime_path)


def main() -> None:
    args = parse_args()
    config_path = make_runtime_config(args.config, args.fast_dev_run)
    py = sys.executable
    steps = [
        [py, "src/simulation/synthetic_full_generator.py", "--config", config_path],
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
        print(f"[run_synthetic_full_pipeline] {' '.join(step)}")
        subprocess.run(step, check=True)
    print("SPARTA synthetic full pipeline complete")


if __name__ == "__main__":
    main()
