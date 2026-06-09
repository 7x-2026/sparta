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
from src.utils.io import ensure_dir, load_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_synthetic_full.yaml")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--generate_only", action="store_true")
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


def run_step(label: str, step: list[str]) -> None:
    print(label, flush=True)
    print(f"[run_synthetic_full_pipeline] {' '.join(step)}", flush=True)
    subprocess.run(step, check=True)


def verify_attribution(config_path: str) -> None:
    config = load_config(config_path)
    processed_dir = resolve_path(config, config["data"]["processed_dir"])
    labeled_path = processed_dir / "samples_labeled.pkl"
    samples = load_pickle(labeled_path)
    required = {"risk_label", "risk_node", "risk_link", "risk_metric", "attr_mask"}
    if not samples:
        raise RuntimeError(f"No labeled samples found in {labeled_path}")
    missing = required.difference(samples[0])
    if missing:
        raise RuntimeError(f"Missing attribution fields in {labeled_path}: {sorted(missing)}")
    print(f"[run_synthetic_full_pipeline] Attribution fields verified in {labeled_path}", flush=True)


def main() -> None:
    args = parse_args()
    config_path = make_runtime_config(args.config, args.fast_dev_run)
    py = sys.executable

    if args.generate_only:
        config = load_config(config_path)
        dataset_dir = resolve_path(config, config["data"]["dataset_dir"])
        raw_dir = resolve_path(config, config["data"]["raw_logs_dir"])
        output_dir = resolve_path(config, config["project"]["output_dir"]) / "audit"
        audit_script = Path("src/analysis/audit_synthetic_full.py")

        run_step("[1/6] Generating raw logs...", [py, "src/simulation/synthetic_full_generator.py", "--config", config_path])
        run_step("[2/6] Building path graph...", [py, "src/preprocessing/build_path_graph.py", "--config", config_path])
        run_step("[3/6] Generating labels...", [py, "src/preprocessing/generate_labels.py", "--config", config_path])
        print("[4/6] Generating attribution...", flush=True)
        verify_attribution(config_path)
        run_step("[5/6] Splitting dataset...", [py, "src/preprocessing/split_dataset.py", "--config", config_path])
        if audit_script.exists():
            run_step(
                "[6/6] Running audit...",
                [
                    py,
                    str(audit_script),
                    "--dataset_dir",
                    str(dataset_dir),
                    "--raw_dir",
                    str(raw_dir),
                    "--output_dir",
                    str(output_dir),
                ],
            )
        else:
            print("[6/6] Running audit...", flush=True)
            print("[run_synthetic_full_pipeline] audit_synthetic_full.py not found; audit skipped.", flush=True)
        print("Generate-only pipeline finished.", flush=True)
        return

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
