from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sparta_dataset import SPARTADataset
from src.metrics import collect_predictions, compute_attribution_metrics, compute_classification_metrics
from src.train import SUPPORTED_MODELS, build_model, select_device
from src.utils.config import load_config, resolve_path
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


RESULT_COLUMNS = [
    "method",
    "has_attribution_head",
    "accuracy",
    "macro_f1",
    "risk_recall",
    "violation_recall",
    "auc",
    "node_attr_acc",
    "link_attr_acc",
    "metric_attr_acc",
    "inference_time_ms",
]
ATTR_COLUMNS = {"node_attr_acc", "link_attr_acc", "metric_attr_acc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_debug.yaml")
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def _format_value(value):
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def _format_row(row: dict) -> dict:
    formatted = {key: _format_value(row.get(key, math.nan)) for key in RESULT_COLUMNS}
    if not bool(row.get("has_attribution_head", False)):
        for key in ATTR_COLUMNS:
            formatted[key] = "N/A"
    formatted["has_attribution_head"] = "true" if bool(row.get("has_attribution_head", False)) else "false"
    return formatted


def write_result(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow(_format_row(row))


def update_all_results(output_dir: Path) -> None:
    rows: list[dict] = []
    for model_name in ["lstm", "transformer", "sparta"]:
        path = output_dir / f"{model_name}_test_results.csv"
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    method = row.get("method", model_name)
                    has_attr = str(row.get("has_attribution_head", "")).lower() == "true" or method == "sparta"
                    row["method"] = method
                    row["has_attribution_head"] = "true" if has_attr else "false"
                    if not has_attr:
                        for key in ATTR_COLUMNS:
                            row[key] = "N/A"
                    rows.append(row)
    if rows:
        with (output_dir / "all_test_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in RESULT_COLUMNS})


@torch.no_grad()
def evaluate_once(config: dict, model_name: str, checkpoint_path: str | Path | None = None) -> dict:
    set_seed(int(config.get("seed", 42)))
    device = select_device(config)
    model = build_model(config, model_name).to(device)
    ckpt_path = Path(checkpoint_path) if checkpoint_path else resolve_path(config, config["project"]["checkpoint_dir"]) / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_path = resolve_path(config, config["data"].get("test_path", Path(config["data"]["dataset_dir"]) / "test.pkl"))
    dataset = SPARTADataset(test_path, config)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 0)),
    )
    start = time.perf_counter()
    labels, preds, has_attr = collect_predictions(model, loader, device)
    elapsed = time.perf_counter() - start
    metrics = compute_classification_metrics(labels["risk_label"], preds["risk_label"])
    metrics.update(compute_attribution_metrics(labels, preds, has_attr))
    metrics["method"] = model_name
    metrics["has_attribution_head"] = bool(has_attr)
    metrics["inference_time_ms"] = (elapsed / max(len(dataset), 1)) * 1000.0
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    row = evaluate_once(config, args.model, args.checkpoint)
    output_dir = resolve_path(config, config["project"]["output_dir"])
    out_path = output_dir / f"{args.model}_test_results.csv"
    write_result(out_path, row)
    update_all_results(output_dir)
    print(f"Wrote evaluation result to {out_path}: {row}")


if __name__ == "__main__":
    main()
