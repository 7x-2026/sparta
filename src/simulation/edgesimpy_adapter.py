from __future__ import annotations

import argparse
import copy
import importlib
import random
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for raw log generation
    torch = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.export_logs import export_logs
from src.simulation.log_schema import LOG_SCHEMA_VERSION
from src.simulation.synthetic_scenarios import SCENARIOS
from src.simulation.synthetic_full_generator import generate_synthetic_full_logs
from src.utils.config import load_config, resolve_path


def try_import_edgesimpy() -> tuple[bool, object | None, str | None]:
    try:
        module = importlib.import_module("edge_sim_py")
        return True, module, None
    except Exception as exc:
        return False, None, str(exc)


def _data_seed(config: dict) -> int:
    experiment_cfg = config.get("experiment", {})
    return int(experiment_cfg.get("data_seed", experiment_cfg.get("seed", config.get("seed", 42))))


def seed_adapter(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _adapter_generation_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    data_cfg = cfg.setdefault("data", {})
    edge_cfg = cfg.setdefault("edgesimpy", {})

    data_cfg["source"] = "edgesimpy"
    data_cfg.setdefault("input_window", data_cfg.get("window", 12))
    data_cfg.setdefault("pred_horizon", data_cfg.get("horizon", 5))
    data_cfg.setdefault("max_nodes", 8)
    data_cfg.setdefault("max_links", 12)
    data_cfg.setdefault("node_feat_dim", 8)
    data_cfg.setdefault("link_feat_dim", 8)
    data_cfg.setdefault("service_feat_dim", 5)
    data_cfg.setdefault("sla_feat_dim", 5)
    data_cfg.setdefault("num_nodes", 8)
    data_cfg.setdefault("num_links", 12)
    data_cfg.setdefault("num_edge_nodes", edge_cfg.get("num_edge_nodes", 5))
    data_cfg.setdefault("num_access_nodes", 2)
    data_cfg.setdefault("num_users", edge_cfg.get("num_users", 50))
    data_cfg.setdefault("num_services", edge_cfg.get("num_services", 30))
    data_cfg.setdefault("num_steps", edge_cfg.get("num_steps", 300))
    data_cfg.setdefault("train_ratio", 0.6)
    data_cfg.setdefault("val_ratio", 0.2)
    data_cfg.setdefault("test_ratio", 0.2)

    sim_cfg = cfg.setdefault("simulation", {})
    sim_cfg.setdefault("topology", edge_cfg.get("topology", "small_world"))
    sim_cfg.setdefault("scenario_schedule", "interleaved")
    sim_cfg.setdefault("max_scenario_ratio_per_split", 0.45)
    sim_cfg.setdefault("max_attribution_majority_ratio", 0.55)
    sim_cfg.setdefault("target_label_ratio", {"normal": 0.45, "risky": 0.35, "violated": 0.20})
    sim_cfg.setdefault(
        "allowed_label_ratio_range",
        {"normal": [0.20, 0.70], "risky": [0.05, 0.60], "violated": [0.02, 0.45]},
    )
    split_check = sim_cfg.setdefault("split_balance_check", {})
    split_check.setdefault("enabled", True)
    split_check.setdefault("max_label_ratio_gap", 0.30)
    split_check.setdefault("max_scenario_ratio_per_split", 0.55)
    split_check.setdefault("max_top_node_attr_ratio", 0.70)
    split_check.setdefault("max_top_link_attr_ratio", 0.70)
    split_check.setdefault("min_metric_ratio", {"queue": 0.0})
    sim_cfg.setdefault("generation_retry", {"enabled": True, "max_retries": 20})
    sim_cfg.setdefault("scenarios", SCENARIOS)
    sim_cfg.setdefault("delay_cap_ms", 300.0)
    sim_cfg.setdefault("bandwidth_cap", 100.0)
    sim_cfg.setdefault("queue_cap", 50.0)
    sim_cfg.setdefault("queue_metric_cap", 28.0)
    sim_cfg.setdefault("queue_growth_cap", 8.0)
    sim_cfg.setdefault("request_cap", 100.0)
    sim_cfg.setdefault("response_cap_ms", 420.0)

    validation_cfg = cfg.setdefault("validation", {})
    validation_cfg.setdefault("require_all_classes", True)
    validation_cfg.setdefault("min_risky_violated_ratio", 0.15)
    validation_cfg.setdefault("min_samples_per_class", 1)
    return cfg


def run_edgesimpy_adapter(config: dict, output_dir: Path) -> dict:
    edge_cfg = config.get("edgesimpy", {})
    backend = str(edge_cfg.get("backend", "edgesimpy_stub"))
    installed, module, import_error = try_import_edgesimpy()
    if backend not in {"edgesimpy_stub", "edgesimpy"}:
        raise ValueError(f"Unsupported EdgeSimPy backend: {backend}")
    if backend == "edgesimpy" and not installed:
        raise RuntimeError(
            "[EdgeSimPy Adapter Failed]\n"
            "backend=edgesimpy was requested, but EdgeSimPy is not installed or not importable.\n"
            "Please run:\n"
            "pip install -r requirements-edgesimpy.txt"
        )

    generation_config = _adapter_generation_config(config)
    seed_adapter(_data_seed(generation_config))
    logs = generate_synthetic_full_logs(generation_config)
    output_dir = Path(output_dir)
    export_logs(output_dir, logs)

    return {
        "node_log": str(output_dir / "node_log.csv"),
        "link_log": str(output_dir / "link_log.csv"),
        "service_log": str(output_dir / "service_log.csv"),
        "path_log": str(output_dir / "path_log.csv"),
        "sla_log": str(output_dir / "sla_log.csv"),
        "data_source": "edgesimpy",
        "simulator_backend": backend,
        "raw_log_schema_version": LOG_SCHEMA_VERSION,
        "edgesimpy_installed": bool(installed),
        "edgesimpy_import_error": None if installed else import_error,
        "edgesimpy_config": edge_cfg,
        "edgesimpy_module": getattr(module, "__file__", None) if module is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_edgesimpy_mvp.yaml")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else resolve_path(config, config["data"]["raw_logs_dir"])
    metadata = run_edgesimpy_adapter(config, output_dir)
    print(f"[edgesimpy_adapter] backend={metadata['simulator_backend']} raw_dir={output_dir}")


if __name__ == "__main__":
    main()
