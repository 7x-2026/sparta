from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.io import load_pickle, save_json, save_pickle


def time_based_split(samples: list[dict], train_ratio: float, val_ratio: float) -> tuple[list[dict], list[dict], list[dict]]:
    samples = sorted(samples, key=lambda sample: (sample["time"], sample["service_id"]))
    times = sorted({int(sample["time"]) for sample in samples})
    n_times = len(times)
    train_end = int(n_times * train_ratio)
    val_end = int(n_times * (train_ratio + val_ratio))
    train_times = set(times[:train_end])
    val_times = set(times[train_end:val_end])
    test_times = set(times[val_end:])
    train = [sample for sample in samples if int(sample["time"]) in train_times]
    val = [sample for sample in samples if int(sample["time"]) in val_times]
    test = [sample for sample in samples if int(sample["time"]) in test_times]
    return train, val, test


def class_counts(samples: list[dict]) -> dict[str, int]:
    counts = Counter(int(sample["risk_label"]) for sample in samples)
    return {"normal": counts.get(0, 0), "risky": counts.get(1, 0), "violated": counts.get(2, 0)}


def validate_classes(samples: list[dict], config: dict) -> None:
    counts = class_counts(samples)
    total = max(sum(counts.values()), 1)
    risky_violated_ratio = (counts["risky"] + counts["violated"]) / total
    validation = config.get("validation", {})
    if validation.get("require_all_classes", True) and any(counts[name] == 0 for name in counts):
        raise RuntimeError(f"Missing risk class in full dataset: {counts}")
    min_ratio = float(validation.get("min_risky_violated_ratio", 0.0))
    if risky_violated_ratio < min_ratio:
        raise RuntimeError(f"risky+violated ratio {risky_violated_ratio:.3f} < {min_ratio:.3f}; counts={counts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = resolve_path(config, config["data"]["processed_dir"])
    dataset_dir = resolve_path(config, config["data"]["dataset_dir"])
    samples = load_pickle(processed_dir / "samples_labeled.pkl")
    validate_classes(samples, config)
    train, val, test = time_based_split(samples, float(config["data"]["train_ratio"]), float(config["data"]["val_ratio"]))

    save_pickle(dataset_dir / "train.pkl", train)
    save_pickle(dataset_dir / "val.pkl", val)
    save_pickle(dataset_dir / "test.pkl", test)
    metadata = {
        "format": "pkl",
        "schema_version": "debug_v0",
        "risk_metric_scores": {
            "shape": [5],
            "use_future_information": True,
            "used_as_supervision_only": True,
            "used_as_model_input": False,
        },
        "input_window": config["data"]["input_window"],
        "pred_horizon": config["data"]["pred_horizon"],
        "max_nodes": config["data"]["max_nodes"],
        "max_links": config["data"]["max_links"],
        "node_feat_dim": config["data"]["node_feat_dim"],
        "link_feat_dim": config["data"]["link_feat_dim"],
        "service_feat_dim": config["data"]["service_feat_dim"],
        "sla_feat_dim": config["data"]["sla_feat_dim"],
        "splits": {
            "train": {"num_samples": len(train), "class_counts": class_counts(train)},
            "val": {"num_samples": len(val), "class_counts": class_counts(val)},
            "test": {"num_samples": len(test), "class_counts": class_counts(test)},
            "all": {"num_samples": len(samples), "class_counts": class_counts(samples)},
        },
    }
    save_json(dataset_dir / "metadata.json", metadata)
    print(f"Wrote train/val/test pkl to {dataset_dir}; metadata={metadata['splits']}")


if __name__ == "__main__":
    main()
