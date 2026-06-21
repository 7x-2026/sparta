from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sparta_dataset import SPARTADataset
from src.losses.multitask_loss import SPARTALoss, compute_metric_class_weights
from src.models.lstm import LSTMBaseline
from src.models.sparta import SPARTA, _schema_from_config, compute_metric_evidence_stats_from_loader, save_metric_evidence_stats
from src.models.transformer import TransformerBaseline
from src.trainer import fit
from src.utils.config import load_config, resolve_path
from src.utils.io import load_pickle


SUPPORTED_MODELS = {"lstm", "transformer", "sparta"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_debug.yaml")
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--resume_dataset_run", default=None)
    parser.add_argument("--variant", default=None)
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


def normalize_variant(variant: str | None) -> str | None:
    if variant is None:
        return None
    text = str(variant).strip()
    if not text:
        return None
    if any(ch in text for ch in ['/', '\\', ':']):
        raise ValueError(f"Invalid variant name: {variant}")
    return text


def apply_resume_dataset_run(config: dict, resume_dataset_run: str | None, variant: str | None = None) -> dict:
    if resume_dataset_run is None:
        if variant:
            config["variant"] = variant
            config.setdefault("run", {})["variant"] = variant
        return config

    run_dir = Path(resume_dataset_run).resolve()
    dataset_dir = run_dir / "dataset"
    for name in ["train.pkl", "val.pkl", "test.pkl"]:
        path = dataset_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file for resume_dataset_run: {path}")

    data_cfg = config.setdefault("data", {})
    data_cfg["dataset_dir"] = str(dataset_dir)
    data_cfg["train_path"] = str(dataset_dir / "train.pkl")
    data_cfg["val_path"] = str(dataset_dir / "val.pkl")
    data_cfg["test_path"] = str(dataset_dir / "test.pkl")
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        data_cfg["metadata_path"] = str(metadata_path)

    project_cfg = config.setdefault("project", {})
    project_cfg["checkpoint_dir"] = str(run_dir / "checkpoints")
    project_cfg["log_dir"] = str(run_dir / "logs")
    project_cfg["output_dir"] = str(run_dir / "results")

    run_cfg = config.setdefault("run", {})
    run_cfg["run_id"] = run_dir.name
    run_cfg["resume_dataset_run"] = str(run_dir)
    if variant:
        config["variant"] = variant
        run_cfg["variant"] = variant
    return config


def metric_evidence_stats_path(config: dict) -> Path:
    run_cfg = config.get("run", {})
    resume_run = run_cfg.get("resume_dataset_run") or run_cfg.get("resume_run")
    if resume_run:
        return Path(resume_run).resolve() / "artifacts" / "metric_evidence_stats.json"
    output_dir = resolve_path(config, config["project"]["output_dir"])
    return output_dir.parent / "artifacts" / "metric_evidence_stats.json"


def prepare_metric_evidence_stats(config: dict, train_loader: DataLoader | None, write: bool = True) -> Path | None:
    model_cfg = config.get("model", {})
    if not bool(model_cfg.get("use_metric_evidence_head", False)):
        return None
    if str(model_cfg.get("metric_evidence_norm", "")).lower() != "pressure01":
        return None
    stats_path = metric_evidence_stats_path(config)
    if write:
        if train_loader is None:
            raise ValueError("train_loader is required when write=True")
        stats = compute_metric_evidence_stats_from_loader(train_loader, _schema_from_config(config))
        save_metric_evidence_stats(stats_path, stats)
        print(f"metric_evidence_stats_path={stats_path}", flush=True)
    elif not stats_path.exists():
        raise FileNotFoundError(f"Missing train metric evidence stats: {stats_path}")
    model_cfg["metric_evidence_stats_path"] = str(stats_path)
    return stats_path


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


def metric_class_weights_from_train(config: dict) -> torch.Tensor | None:
    loss_cfg = config.get("loss", {})
    method = str(loss_cfg.get("metric_class_weighting", "none"))
    if method in {"", "none", "uniform"}:
        return None
    data_cfg = config["data"]
    train_path = resolve_path(config, data_cfg.get("train_path", Path(data_cfg["dataset_dir"]) / "train.pkl"))
    samples = load_pickle(train_path)
    counts = Counter(int(sample["risk_metric"]) for sample in samples if int(sample.get("attr_mask", 0)) == 1)
    count_list = [counts.get(metric_id, 0) for metric_id in range(5)]
    return compute_metric_class_weights(
        count_list,
        method=method,
        smoothing=float(loss_cfg.get("metric_class_weight_smoothing", 1.0)),
        effective_beta=float(loss_cfg.get("metric_effective_beta", 0.999)),
    )


def build_loss(config: dict) -> SPARTALoss:
    loss_cfg = config["loss"]
    metric_weights = metric_class_weights_from_train(config)
    if metric_weights is not None:
        loss_cfg["metric_class_weights_resolved"] = [float(value) for value in metric_weights.tolist()]
        print(f"metric_class_weights={json.dumps(loss_cfg['metric_class_weights_resolved'])}", flush=True)
    return SPARTALoss(
        lambda_node=float(loss_cfg.get("lambda_node", 0.3)),
        lambda_link=float(loss_cfg.get("lambda_link", 0.3)),
        lambda_metric=float(loss_cfg.get("lambda_metric", 0.2)),
        ignore_index=int(loss_cfg.get("ignore_index", -100)),
        metric_class_weights=metric_weights,
        metric_loss_type=str(loss_cfg.get("metric_loss_type", "weighted_ce")),
        metric_focal_gamma=float(loss_cfg.get("metric_focal_gamma", 2.0)),
        lambda_metric_soft=float(loss_cfg.get("lambda_metric_soft", 0.0)),
        metric_soft_loss_type=str(loss_cfg.get("metric_soft_loss_type", "none")),
        metric_soft_temperature=float(loss_cfg.get("metric_soft_temperature", 2.0)),
    )


def select_device(config: dict) -> torch.device:
    requested = config["train"].get("device", "auto")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    variant = normalize_variant(args.variant)
    config = load_config(args.config)
    config = apply_resume_dataset_run(config, args.resume_dataset_run, variant)
    set_train_seed(get_train_seed(config))
    device = select_device(config)
    train_loader, val_loader = build_dataloaders(config)
    prepare_metric_evidence_stats(config, train_loader, write=True)
    model = build_model(config, args.model).to(device)
    loss_fn = build_loss(config)
    loss_cfg = config.get("loss", {})
    print(
        "training_config "
        f"model={args.model} "
        f"variant={variant or ''} "
        f"lambda_metric={loss_cfg.get('lambda_metric', 0.2)} "
        f"lambda_metric_soft={loss_cfg.get('lambda_metric_soft', 0.0)} "
        f"metric_loss_type={loss_cfg.get('metric_loss_type', 'weighted_ce')} "
        f"metric_soft_loss_type={loss_cfg.get('metric_soft_loss_type', 'none')} "
        f"metric_soft_temperature={loss_cfg.get('metric_soft_temperature', 2.0)} "
        f"metric_class_weighting={loss_cfg.get('metric_class_weighting', 'none')} "
        f"metric_focal_gamma={loss_cfg.get('metric_focal_gamma', 2.0)} "
        f"metric_class_weights={json.dumps(loss_cfg.get('metric_class_weights_resolved', []))} "
        "risk_metric_scores_use_future_information=true "
        "used_as_supervision_only=true "
        "used_as_model_input=false",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"].get("weight_decay", 0.0)),
    )
    fit(model, train_loader, val_loader, loss_fn, optimizer, config, args.model, device)
    print(f"Training complete for {args.model}")


if __name__ == "__main__":
    main()
