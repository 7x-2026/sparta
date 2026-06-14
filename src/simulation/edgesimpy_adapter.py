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


def _split_pipe(value: str) -> list[str]:
    return [part for part in str(value or "").split("|") if part]


def _int_value(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float_value(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sample_split_times(config: dict) -> dict[str, list[int]]:
    data_cfg = config["data"]
    L = int(data_cfg["input_window"])
    H = int(data_cfg["pred_horizon"])
    times = list(range(L - 1, int(data_cfg["num_steps"]) - H))
    train_end = int(len(times) * float(data_cfg.get("train_ratio", 0.6)))
    val_end = int(len(times) * (float(data_cfg.get("train_ratio", 0.6)) + float(data_cfg.get("val_ratio", 0.2))))
    return {"train": times[:train_end], "val": times[train_end:val_end], "test": times[val_end:]}


def _row_index(rows: list[dict], *keys: str) -> dict[tuple, dict]:
    return {tuple(row[key] for key in keys): row for row in rows}


def _service_sla(logs: dict[str, list[dict]]) -> dict[int, dict]:
    return {_int_value(row.get("service_id")): row for row in logs.get("sla_log.csv", [])}


def _cpu_target_cells(config: dict, split_times: dict[str, list[int]], rng: np.random.Generator) -> dict[str, list[tuple[int, int]]]:
    data_cfg = config["data"]
    num_services = int(data_cfg["num_services"])
    num_steps = int(data_cfg["num_steps"])
    target_ratio = float(config.get("edgesimpy", {}).get("stub_cpu_metric_target_ratio", 0.01))
    if num_steps <= 150 or num_services <= 10:
        target_ratio = min(target_ratio, 0.02)
    cells: dict[str, list[tuple[int, int]]] = {}
    split_factors = {"train": 0.75, "val": 1.00, "test": 1.40}
    for split, times in split_times.items():
        if not times:
            cells[split] = []
            continue
        effective_ratio = target_ratio * split_factors.get(split, 1.0)
        target_count = max(1, int(round(len(times) * num_services * effective_ratio)))
        time_step = max(1, len(times) // max(8, min(36, len(times))))
        time_slots = times[::time_step]
        selected: list[tuple[int, int]] = []
        cursor = 0
        while len(selected) < target_count:
            t = time_slots[cursor % len(time_slots)]
            service_id = int((cursor * 7 + rng.integers(0, num_services)) % num_services)
            cell = (int(t), service_id)
            if cell not in selected:
                selected.append(cell)
            cursor += 1
        cells[split] = selected
    return cells


def inject_cpu_dominant_overload(logs: dict[str, list[dict]], config: dict) -> None:
    """Add EdgeSimPy-stub-specific CPU-dominant overload windows.

    The synthetic generator often makes node overload and queue growth rise
    together. For the EdgeSimPy stub we keep the same five-scenario surface but
    carve out evenly interleaved windows where CPU is the dominant risk signal.
    """
    data_cfg = config["data"]
    H = int(data_cfg["pred_horizon"])
    num_steps = int(data_cfg["num_steps"])
    queue_cap = float(config.get("simulation", {}).get("queue_cap", 50.0))
    rng = np.random.default_rng(_data_seed(config) + 2027)
    split_times = _sample_split_times(config)
    targets = _cpu_target_cells(config, split_times, rng)

    node_by_time_id = _row_index(logs.get("node_log.csv", []), "time", "node_id")
    service_by_time_id = _row_index(logs.get("service_log.csv", []), "time", "service_id")
    path_by_time_id = _row_index(logs.get("path_log.csv", []), "time", "service_id")
    sla_by_service = _service_sla(logs)
    episode_base = 900000
    node_position_cursor = 0

    for split, cells in targets.items():
        for target_idx, (sample_time, service_id) in enumerate(cells):
            risk_node_for_window = ""
            event_id = episode_base + len(split) * 100000 + target_idx
            for ft in range(sample_time + 1, min(sample_time + H, num_steps - 1) + 1):
                path_row = path_by_time_id.get((ft, service_id))
                service_row = service_by_time_id.get((ft, service_id))
                if path_row is None or service_row is None:
                    continue
                path_nodes = _split_pipe(path_row.get("path_nodes", ""))
                edge_nodes = [node for node in path_nodes if node.startswith("edge")]
                if not edge_nodes:
                    continue
                risk_node = edge_nodes[(node_position_cursor + target_idx + ft) % len(edge_nodes)]
                risk_node_for_window = risk_node
                queue_len = float(rng.uniform(0.36, 0.48) * queue_cap)
                for node_id in path_nodes:
                    node_row = node_by_time_id.get((ft, node_id))
                    if node_row is None:
                        continue
                    if node_id == risk_node:
                        cpu_util = float(rng.uniform(0.78, 0.84))
                        available_cpu = float(rng.uniform(0.02, 0.08))
                    else:
                        cpu_util = min(_float_value(node_row.get("cpu_util"), 0.55), float(rng.uniform(0.48, 0.66)))
                        available_cpu = max(0.10, 1.0 - cpu_util)
                    node_row.update(
                        {
                            "scenario": "node_overload",
                            "event_id": event_id,
                            "severity": 0.72,
                            "affected_node": risk_node,
                            "affected_link": path_row.get("bottleneck_link", ""),
                            "cpu_util": round(cpu_util, 5),
                            "mem_util": round(max(_float_value(node_row.get("mem_util"), 0.35), min(0.82, 0.30 + 0.45 * cpu_util)), 5),
                            "queue_len": round(queue_len, 5),
                            "available_cpu": round(available_cpu, 5),
                            "available_mem": round(max(0.08, 1.0 - _float_value(node_row.get("mem_util"), 0.35)), 5),
                        }
                    )
                sla = sla_by_service.get(service_id, {})
                max_delay = max(_float_value(sla.get("max_delay"), 120.0), 1.0)
                service_row.update(
                    {
                        "scenario": "node_overload",
                        "event_id": event_id,
                        "severity": 0.72,
                        "affected_node": risk_node,
                        "affected_link": path_row.get("bottleneck_link", ""),
                        "request_rate": round(_float_value(service_row.get("request_rate"), 10.0), 5),
                        "response_time": round(_float_value(service_row.get("response_time"), max_delay * 0.82), 5),
                    }
                )
                path_row.update(
                    {
                        "scenario": "node_overload",
                        "event_id": event_id,
                        "severity": 0.72,
                        "affected_node": risk_node,
                        "affected_link": path_row.get("bottleneck_link", ""),
                        "bottleneck_node": risk_node,
                    }
                )
            if risk_node_for_window:
                node_position_cursor += 1


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
    if backend == "edgesimpy_stub":
        inject_cpu_dominant_overload(logs, generation_config)
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
