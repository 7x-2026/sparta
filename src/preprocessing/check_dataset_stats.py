from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.io import load_pickle


def _describe_array(samples: list[dict], key: str) -> dict:
    if not samples:
        return {"shape": [], "min": None, "max": None, "mean": None, "std": None}
    arr = np.stack([sample[key] for sample in samples])
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def _stats(samples: list[dict]) -> dict:
    counts = Counter(int(sample["risk_label"]) for sample in samples)
    attr_valid = sum(int(sample["attr_mask"]) for sample in samples)
    return {
        "num_samples": len(samples),
        "class_counts": {"normal": counts.get(0, 0), "risky": counts.get(1, 0), "violated": counts.get(2, 0)},
        "attr_valid": attr_valid,
        "node_x": _describe_array(samples, "node_x"),
        "link_x": _describe_array(samples, "link_x"),
        "service_x": _describe_array(samples, "service_x"),
        "sla_x": _describe_array(samples, "sla_x"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_dir = resolve_path(config, config["data"]["dataset_dir"])
    all_samples: list[dict] = []
    for split in ["train", "val", "test"]:
        samples = load_pickle(dataset_dir / f"{split}.pkl")
        all_samples.extend(samples)
        print(f"{split}: {_stats(samples)}")
    all_counts = Counter(int(sample["risk_label"]) for sample in all_samples)
    if any(all_counts.get(label, 0) == 0 for label in [0, 1, 2]):
        raise RuntimeError(f"Dataset must contain normal/risky/violated classes; got {dict(all_counts)}")
    risky_violated = (all_counts.get(1, 0) + all_counts.get(2, 0)) / max(len(all_samples), 1)
    min_ratio = float(config.get("validation", {}).get("min_risky_violated_ratio", 0.0))
    if risky_violated < min_ratio:
        raise RuntimeError(f"risky+violated ratio {risky_violated:.3f} < {min_ratio:.3f}")
    print("dataset stats ok")


if __name__ == "__main__":
    main()
