from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sparta_dataset import SPARTADataset
from src.losses.multitask_loss import SPARTALoss
from src.models.lstm import LSTMBaseline
from src.models.sparta import SPARTA
from src.models.transformer import TransformerBaseline
from src.trainer import fit
from src.utils.config import load_config, resolve_path


SUPPORTED_MODELS = {"lstm", "transformer", "sparta"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_debug.yaml")
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    return parser.parse_args()


def build_model(config: dict, model_name: str):
    config["model"]["name"] = model_name
    if model_name == "lstm":
        return LSTMBaseline(config)
    if model_name == "transformer":
        return TransformerBaseline(config)
    if model_name == "sparta":
        return SPARTA(config)
    raise ValueError(f"Unsupported model: {model_name}")


def get_train_seed(config: dict) -> int:
    experiment_cfg = config.get("experiment", {})
    return int(experiment_cfg.get("train_seed", experiment_cfg.get("seed", config.get("seed", 42))))


def set_train_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = config["data"]
    train_path = resolve_path(config, data_cfg.get("train_path", Path(data_cfg["dataset_dir"]) / "train.pkl"))
    val_path = resolve_path(config, data_cfg.get("val_path", Path(data_cfg["dataset_dir"]) / "val.pkl"))
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"].get("num_workers", 0))
    train_ds = SPARTADataset(train_path, config)
    val_ds = SPARTADataset(val_path, config)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def build_loss(config: dict) -> SPARTALoss:
    loss_cfg = config["loss"]
    return SPARTALoss(
        lambda_node=float(loss_cfg.get("lambda_node", 0.3)),
        lambda_link=float(loss_cfg.get("lambda_link", 0.3)),
        lambda_metric=float(loss_cfg.get("lambda_metric", 0.2)),
        ignore_index=int(loss_cfg.get("ignore_index", -100)),
    )


def select_device(config: dict) -> torch.device:
    requested = config["train"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_train_seed(get_train_seed(config))
    device = select_device(config)
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config, args.model).to(device)
    loss_fn = build_loss(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    fit(model, train_loader, val_loader, loss_fn, optimizer, config, args.model, device)
    print(f"Training complete for {args.model}")


if __name__ == "__main__":
    main()
