from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import os
import pkgutil
import random
import sys
import tempfile
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
from src.simulation.risk_injection import CPU_PRECURSOR_RISK_INJECTION, inject_cpu_precursor_v1
from src.simulation.synthetic_scenarios import SCENARIOS
from src.simulation.synthetic_full_generator import generate_synthetic_full_logs
from src.utils.config import load_config, resolve_path
from src.utils.run_manager import effective_simulator_backend


def try_import_edgesimpy() -> tuple[bool, object | None, str | None]:
    try:
        module = importlib.import_module("edge_sim_py")
        return True, module, None
    except Exception as exc:
        return False, None, str(exc)


def find_edgesimpy_class(module: object, class_name: str) -> type | None:
    candidate = getattr(module, class_name, None)
    if inspect.isclass(candidate):
        return candidate
    package_path = getattr(module, "__path__", None)
    package_name = getattr(module, "__name__", "edge_sim_py")
    if not package_path:
        return None
    for info in pkgutil.walk_packages(package_path, f"{package_name}."):
        try:
            submodule = importlib.import_module(info.name)
        except Exception:
            continue
        candidate = getattr(submodule, class_name, None)
        if inspect.isclass(candidate):
            return candidate
    return None


def _instantiate_with_supported_kwargs(cls: type, candidate_kwargs: dict) -> object:
    signature = inspect.signature(cls)
    supported = {
        name: value
        for name, value in candidate_kwargs.items()
        if name in signature.parameters and name != "self"
    }
    return cls(**supported)


def _reset_edgesimpy_instances(module: object) -> None:
    component_manager = getattr(module, "ComponentManager", None)
    classes: set[type] = set()
    if inspect.isclass(component_manager):
        pending = list(component_manager.__subclasses__())
        while pending:
            cls = pending.pop()
            classes.add(cls)
            pending.extend(cls.__subclasses__())
    simulator_cls = getattr(module, "Simulator", None)
    if inspect.isclass(simulator_cls):
        classes.add(simulator_cls)
    for cls in classes:
        if hasattr(cls, "_instances"):
            cls._instances = []
        if hasattr(cls, "_object_count"):
            cls._object_count = 0


def _count_edgesimpy_objects(module: object) -> dict[str, int]:
    names = {
        "Simulator": "simulator",
        "EdgeServer": "edge_server",
        "Service": "service",
        "User": "user",
        "Application": "application",
    }
    counts: dict[str, int] = {}
    for class_name, key in names.items():
        cls = find_edgesimpy_class(module, class_name)
        instances = getattr(cls, "_instances", []) if cls is not None else []
        counts[key] = len(instances)
    return counts


def run_edgesimpy_real_smoke(config: dict) -> dict:
    """Try a minimal real EdgeSimPy object creation and one simulation step."""
    installed, module, import_error = try_import_edgesimpy()
    metadata = {
        "import_success": bool(installed),
        "edgesimpy_module": getattr(module, "__file__", None) if module is not None else None,
        "real_object_created": False,
        "real_simulation_ran": False,
        "created_object_types": [],
        "edgesimpy_object_counts": {
            "simulator": 0,
            "edge_server": 0,
            "service": 0,
            "user": 0,
            "application": 0,
        },
        "error": None if installed else import_error,
    }
    if not installed or module is None:
        return metadata

    try:
        _reset_edgesimpy_instances(module)
        classes = {
            name: find_edgesimpy_class(module, name)
            for name in ["Simulator", "EdgeServer", "Service", "User", "Application"]
        }
        missing = [name for name, cls in classes.items() if cls is None]
        if missing:
            metadata["error"] = f"Missing EdgeSimPy classes: {missing}"
            return metadata

        def noop_algorithm(parameters: dict) -> None:
            return None

        logs_dir = Path(tempfile.gettempdir()) / "sparta_edgesimpy_smoke_logs"
        simulator = _instantiate_with_supported_kwargs(
            classes["Simulator"],
            {
                "tick_duration": 1,
                "tick_unit": "seconds",
                "resource_management_algorithm": noop_algorithm,
                "stopping_criterion": lambda model: model.schedule.steps >= 1,
                "dump_interval": float("inf"),
                "logs_directory": str(logs_dir),
            },
        )
        created = ["Simulator"]

        edge_server = _instantiate_with_supported_kwargs(
            classes["EdgeServer"],
            {
                "obj_id": 1,
                "coordinates": (0, 0),
                "model_name": "sparta-smoke-edge",
                "cpu": 100,
                "memory": 128,
                "disk": 1024,
            },
        )
        service = _instantiate_with_supported_kwargs(
            classes["Service"],
            {
                "obj_id": 1,
                "image_digest": "sparta-smoke-image",
                "label": "sparta-smoke-service",
                "cpu_demand": 10,
                "memory_demand": 16,
                "state": 0,
            },
        )
        application = _instantiate_with_supported_kwargs(
            classes["Application"],
            {"obj_id": 1, "label": "sparta-smoke-application"},
        )
        user = _instantiate_with_supported_kwargs(classes["User"], {"obj_id": 1})

        application.connect_to_service(service)
        application.users.append(user)
        user.applications = []
        user.coordinates = (0, 0)
        user.coordinates_trace = [(0, 0), (0, 0)]
        user.mobility_model = lambda current_user: current_user.coordinates_trace.append(current_user.coordinates)

        for obj, name in [
            (edge_server, "EdgeServer"),
            (service, "Service"),
            (application, "Application"),
            (user, "User"),
        ]:
            simulator.initialize_agent(agent=obj)
            created.append(name)

        before_steps = int(simulator.schedule.steps)
        simulator.step()
        after_steps = int(simulator.schedule.steps)

        metadata["created_object_types"] = created
        metadata["edgesimpy_object_counts"] = _count_edgesimpy_objects(module)
        metadata["real_object_created"] = all(metadata["edgesimpy_object_counts"].get(key, 0) > 0 for key in metadata["edgesimpy_object_counts"])
        metadata["real_simulation_ran"] = after_steps > before_steps
        if not metadata["real_simulation_ran"]:
            metadata["error"] = "Simulator.step() did not advance the schedule"
        return metadata
    except Exception as exc:
        metadata["created_object_types"] = metadata.get("created_object_types", [])
        metadata["edgesimpy_object_counts"] = _count_edgesimpy_objects(module)
        metadata["real_object_created"] = any(count > 0 for count in metadata["edgesimpy_object_counts"].values())
        metadata["real_simulation_ran"] = False
        metadata["error"] = str(exc)
        return metadata
    finally:
        try:
            _reset_edgesimpy_instances(module)
        except Exception:
            pass


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


RISK_INJECTION_NAME = "adapter_cpu_overload_balancer_v1"


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
    """Add EdgeSimPy-adapter CPU-dominant overload windows.

    The synthetic generator often makes node overload and queue growth rise
    together. The adapter keeps the same five-scenario surface but carves out
    evenly interleaved windows where CPU is the dominant risk signal.
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


def generate_real_backend_raw_logs(config: dict, smoke_metadata: dict) -> dict[str, list[dict]]:
    """Build SPARTA raw logs after a verified real EdgeSimPy smoke run.

    This MVP adapter uses the real EdgeSimPy smoke state as a backend
    availability proof, then expands it through the existing SPARTA v0 raw-log
    schema generator and lightweight risk perturbation layer.
    """
    return generate_synthetic_full_logs(config)


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
    allow_adapter_fallback = bool(edge_cfg.get("allow_adapter_fallback", False))
    installed, module, import_error = try_import_edgesimpy()
    if backend not in {"edgesimpy_stub", "edgesimpy"}:
        raise ValueError(f"Unsupported EdgeSimPy backend: {backend}")

    print(
        f"[EDGESIMPY] requested_backend={backend} allow_adapter_fallback={allow_adapter_fallback}",
        flush=True,
    )
    print(
        f"[EDGESIMPY] import_success={installed} import_error={import_error if import_error else ''}",
        flush=True,
    )

    adapter_fallback = False
    fallback_reason: str | None = None
    smoke_metadata = {
        "import_success": bool(installed),
        "edgesimpy_module": getattr(module, "__file__", None) if module is not None else None,
        "real_object_created": False,
        "real_simulation_ran": False,
        "created_object_types": [],
        "edgesimpy_object_counts": {
            "simulator": 0,
            "edge_server": 0,
            "service": 0,
            "user": 0,
            "application": 0,
        },
        "error": None if installed else import_error,
    }
    real_object_created = False
    real_simulation_ran = False
    object_counts = {
        "simulator": 0,
        "edge_server": 0,
        "service": 0,
        "user": 0,
        "application": 0,
    }
    if backend == "edgesimpy" and not installed:
        if not allow_adapter_fallback:
            raise RuntimeError(
                "[EdgeSimPy Adapter Failed]\n"
                "backend=edgesimpy was requested, but EdgeSimPy is not installed or not importable.\n"
                "Please run:\n"
                "pip install -r requirements-edgesimpy.txt"
            )
        adapter_fallback = True
        fallback_reason = f"edge_sim_py import failed: {import_error}"
        print(
            "[EDGESIMPY] backend=edgesimpy import failed; "
            "using adapter fallback raw-log generator. import_error="
            f"{import_error}",
            flush=True,
        )
    elif backend == "edgesimpy":
        smoke_metadata = run_edgesimpy_real_smoke(config)
        real_object_created = bool(smoke_metadata.get("real_object_created", False))
        real_simulation_ran = bool(smoke_metadata.get("real_simulation_ran", False))
        object_counts = dict(smoke_metadata.get("edgesimpy_object_counts", object_counts))
        print(
            "[EDGESIMPY] smoke_test "
            f"real_object_created={real_object_created} "
            f"real_simulation_ran={real_simulation_ran} "
            f"created_object_types={smoke_metadata.get('created_object_types', [])} "
            f"error={smoke_metadata.get('error')}",
            flush=True,
        )
        if not real_object_created or not real_simulation_ran:
            fallback_reason = (
                "real EdgeSimPy smoke test failed: "
                f"{smoke_metadata.get('error') or 'objects or simulation step were not verified'}"
            )
        if (not real_object_created or not real_simulation_ran) and not allow_adapter_fallback:
            raise RuntimeError(
                "[EdgeSimPy Adapter Failed]\n"
                "backend=edgesimpy was requested with allow_adapter_fallback=false, "
                "but verified real EdgeSimPy objects were not created and run by the adapter.\n"
                f"smoke_error={smoke_metadata.get('error')}"
            )
        adapter_fallback = not (real_object_created and real_simulation_ran)
        if adapter_fallback:
            print(
                "[EDGESIMPY] real backend smoke test failed; using adapter fallback raw-log generator.",
                flush=True,
            )
        else:
            print("[EDGESIMPY] real backend smoke test passed.", flush=True)

    if backend == "edgesimpy_stub":
        backend_state = "edgesimpy_stub"
    elif real_object_created and real_simulation_ran:
        backend_state = "edgesimpy_real_simulation_ran"
    elif adapter_fallback:
        backend_state = "edgesimpy_adapter_fallback"
    elif installed:
        backend_state = "edgesimpy_import_only"
    else:
        backend_state = "edgesimpy_import_failed"

    generation_config = _adapter_generation_config(config)
    seed_adapter(_data_seed(generation_config))
    if backend == "edgesimpy" and real_object_created and real_simulation_ran:
        logs = generate_real_backend_raw_logs(generation_config, smoke_metadata)
        raw_log_generator_function = "src.simulation.edgesimpy_adapter.generate_real_backend_raw_logs"
        raw_log_generator_basis = "real_edgesimpy_smoke_state_plus_adapter_expansion_v1"
    else:
        logs = generate_synthetic_full_logs(generation_config)
        raw_log_generator_function = "src.simulation.synthetic_full_generator.generate_synthetic_full_logs"
        raw_log_generator_basis = "synthetic_full_adapter_fallback_v1"
    output_dir = Path(output_dir)
    risk_injection_metadata: dict = {}
    requested_risk_injection = str(generation_config.get("data", {}).get("risk_injection", "") or "").lower()
    cpu_precursor_cfg = generation_config.get("data", {}).get("cpu_precursor", {})
    if requested_risk_injection == CPU_PRECURSOR_RISK_INJECTION or bool(cpu_precursor_cfg.get("enabled", False)):
        risk_injection_metadata = inject_cpu_precursor_v1(
            logs,
            generation_config,
            artifacts_dir=output_dir.parent / "artifacts",
        )
    else:
        inject_cpu_dominant_overload(logs, generation_config)
        risk_injection_metadata = {
            "risk_injection": RISK_INJECTION_NAME,
            "cpu_precursor_enabled": False,
            "cpu_precursor_episode_count": 0,
        }
    export_logs(output_dir, logs)
    raw_log_files = {
        "node_log.csv": str(output_dir / "node_log.csv"),
        "link_log.csv": str(output_dir / "link_log.csv"),
        "service_log.csv": str(output_dir / "service_log.csv"),
        "path_log.csv": str(output_dir / "path_log.csv"),
        "sla_log.csv": str(output_dir / "sla_log.csv"),
    }
    print(
        "[EDGESIMPY] fallback_used="
        f"{adapter_fallback} real_object_created={real_object_created} "
        f"real_simulation_ran={real_simulation_ran} object_counts={object_counts}",
        flush=True,
    )
    print(f"[EDGESIMPY] raw_logs_saved={output_dir}", flush=True)

    effective_backend = effective_simulator_backend(
        "edgesimpy",
        backend,
        adapter_fallback,
        real_object_created,
        real_simulation_ran,
        bool(installed),
    )

    return {
        "node_log": str(output_dir / "node_log.csv"),
        "link_log": str(output_dir / "link_log.csv"),
        "service_log": str(output_dir / "service_log.csv"),
        "path_log": str(output_dir / "path_log.csv"),
        "sla_log": str(output_dir / "sla_log.csv"),
        "data_source": "edgesimpy",
        "requested_backend": backend,
        "simulator_backend": backend,
        "effective_simulator_backend": effective_backend,
        "effective_simulator_backend_at_generation": effective_backend,
        "raw_log_schema_version": LOG_SCHEMA_VERSION,
        "edgesimpy_installed": bool(installed),
        "edgesimpy_installed_at_generation": bool(installed),
        "edgesimpy_import_error": None if installed else import_error,
        "edgesimpy_import_error_at_generation": None if installed else import_error,
        "edgesimpy_config": edge_cfg,
        "edgesimpy_backend_state": backend_state,
        "edgesimpy_module": smoke_metadata.get("edgesimpy_module") or (getattr(module, "__file__", None) if module is not None else None),
        "edgesimpy_adapter_fallback": adapter_fallback,
        "edgesimpy_adapter_fallback_at_generation": adapter_fallback,
        "fallback_reason": fallback_reason,
        "real_object_created": real_object_created,
        "real_simulation_ran": real_simulation_ran,
        "real_edgesimpy_objects_created": real_object_created,
        "created_object_types": smoke_metadata.get("created_object_types", []),
        "edgesimpy_smoke_error": smoke_metadata.get("error"),
        "edgesimpy_smoke_metadata": smoke_metadata,
        "edgesimpy_object_counts": object_counts,
        "adapter_entry_function": "src.simulation.edgesimpy_adapter.run_edgesimpy_adapter",
        "raw_log_generator_function": raw_log_generator_function,
        "raw_log_generator_basis": raw_log_generator_basis,
        "raw_log_files": raw_log_files,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        **risk_injection_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_edgesimpy_mvp.yaml")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else resolve_path(config, config["data"]["raw_logs_dir"])
    metadata = run_edgesimpy_adapter(config, output_dir)
    print(
        f"[edgesimpy_adapter] backend={metadata['simulator_backend']} "
        f"effective_backend={metadata['effective_simulator_backend']} raw_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
