from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for raw log generation
    torch = None


CPU_PRECURSOR_RISK_INJECTION = "cpu_precursor_v1"


def _data_seed(config: dict) -> int:
    experiment_cfg = config.get("experiment", {})
    return int(experiment_cfg.get("data_seed", experiment_cfg.get("seed", config.get("seed", 42))))


def _seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _f(row: dict | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _i(row: dict | None, key: str, default: int = 0) -> int:
    if row is None:
        return default
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def _split_pipe(value: str) -> list[str]:
    return [part for part in str(value or "").split("|") if part]


def _index_logs(logs: dict[str, list[dict]]) -> dict[str, dict]:
    node_by_time_id: dict[tuple[int, str], dict] = {}
    service_by_time_id: dict[tuple[int, int], dict] = {}
    path_by_time_id: dict[tuple[int, int], dict] = {}
    services_by_time: dict[int, list[dict]] = {}
    paths_by_time: dict[int, list[dict]] = {}
    node_ids: set[str] = set()
    service_ids: set[int] = set()

    for row in logs.get("node_log.csv", []):
        time = _i(row, "time")
        node_id = str(row.get("node_id", ""))
        node_by_time_id[(time, node_id)] = row
        node_ids.add(node_id)
    for row in logs.get("service_log.csv", []):
        time = _i(row, "time")
        service_id = _i(row, "service_id")
        service_by_time_id[(time, service_id)] = row
        services_by_time.setdefault(time, []).append(row)
        service_ids.add(service_id)
    for row in logs.get("path_log.csv", []):
        time = _i(row, "time")
        service_id = _i(row, "service_id")
        path_by_time_id[(time, service_id)] = row
        paths_by_time.setdefault(time, []).append(row)
        service_ids.add(service_id)

    return {
        "node": node_by_time_id,
        "service": service_by_time_id,
        "path": path_by_time_id,
        "services_by_time": services_by_time,
        "paths_by_time": paths_by_time,
        "node_ids": sorted(node_ids),
        "service_ids": sorted(service_ids),
    }


def _sample_split_times(config: dict) -> dict[str, list[int]]:
    data_cfg = config["data"]
    L = int(data_cfg.get("input_window", data_cfg.get("window", 12)))
    H = int(data_cfg.get("pred_horizon", data_cfg.get("horizon", 5)))
    num_steps = int(data_cfg.get("num_steps", config.get("edgesimpy", {}).get("num_steps", 1000)))
    times = list(range(L - 1, max(L, num_steps - H)))
    train_ratio = float(data_cfg.get("train_ratio", 0.6))
    val_ratio = float(data_cfg.get("val_ratio", 0.2))
    train_end = int(len(times) * train_ratio)
    val_end = int(len(times) * (train_ratio + val_ratio))
    return {"train": times[:train_end], "val": times[train_end:val_end], "test": times[val_end:]}


def _path_edge_nodes(path_row: dict | None) -> list[str]:
    if not path_row:
        return []
    return [node for node in _split_pipe(path_row.get("path_nodes", "")) if node.startswith("edge")]


def _services_on_node(index: dict[str, dict], time: int, node_id: str) -> list[int]:
    service_ids: list[int] = []
    for path_row in index["paths_by_time"].get(time, []):
        path_nodes = _split_pipe(path_row.get("path_nodes", ""))
        if node_id in path_nodes:
            service_ids.append(_i(path_row, "service_id"))
    return sorted(set(service_ids))


def _event_times(split_times: list[int], count: int, rng: np.random.Generator) -> list[int]:
    if not split_times:
        return []
    if len(split_times) <= count:
        return list(split_times)
    base_positions = np.linspace(0, len(split_times) - 1, num=count + 2, dtype=int)[1:-1]
    jitter = max(1, len(split_times) // max(count * 8, 1))
    chosen: list[int] = []
    for pos in base_positions:
        shifted = int(np.clip(pos + rng.integers(-jitter, jitter + 1), 0, len(split_times) - 1))
        chosen.append(int(split_times[shifted]))
    return sorted(set(chosen))


def _ramp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _update_cpu_node(
    row: dict,
    *,
    event_id: int,
    affected_node: str,
    severity: float,
    cpu_phase: float,
    queue_phase: float,
    config: dict,
    rng: np.random.Generator,
) -> tuple[float, float]:
    precursor_cfg = config.get("data", {}).get("cpu_precursor", {})
    sim_cfg = config.get("simulation", {})
    queue_cap = float(sim_cfg.get("queue_cap", 50.0))
    max_cpu_util = float(precursor_cfg.get("max_cpu_util", 0.98))
    min_available_cpu = float(precursor_cfg.get("min_available_cpu", 0.02))
    noise_std = float(precursor_cfg.get("noise_std", 0.03))
    cpu_growth = rng.uniform(
        float(precursor_cfg.get("cpu_pressure_growth_min", 0.05)),
        float(precursor_cfg.get("cpu_pressure_growth_max", 0.15)),
    )
    noise = float(rng.normal(0.0, noise_std))
    existing_cpu = _f(row, "cpu_util", 0.45)
    existing_available = _f(row, "available_cpu", 1.0 - existing_cpu)
    peak_cpu_target = min(max_cpu_util, max(existing_cpu, 0.88 + cpu_growth * severity + noise))
    target_cpu = existing_cpu + (peak_cpu_target - existing_cpu) * cpu_phase
    peak_available_target = max(min_available_cpu, min(existing_available, 0.08 - 0.04 * severity - noise))
    target_available = existing_available + (peak_available_target - existing_available) * cpu_phase
    existing_queue = _f(row, "queue_len", 0.0)
    target_queue = existing_queue
    if queue_phase > 0.0:
        target_queue = max(existing_queue, (0.24 + 0.16 * queue_phase + 0.04 * severity) * queue_cap)
    row.update(
        {
            "scenario": "node_overload",
            "event_id": event_id,
            "severity": round(severity, 5),
            "affected_node": affected_node,
            "cpu_util": round(float(np.clip(target_cpu, 0.0, 1.0)), 5),
            "available_cpu": round(float(np.clip(target_available, 0.0, 1.0)), 5),
            "mem_util": round(float(np.clip(max(_f(row, "mem_util", 0.35), 0.36 + 0.34 * cpu_phase), 0.0, 1.0)), 5),
            "available_mem": round(float(np.clip(1.0 - max(_f(row, "mem_util", 0.35), 0.36 + 0.34 * cpu_phase), 0.0, 1.0)), 5),
            "queue_len": round(float(np.clip(target_queue, 0.0, queue_cap)), 5),
        }
    )
    return float(row["cpu_util"]), float(row["queue_len"])


def _update_service_and_path(
    service_row: dict,
    path_row: dict,
    *,
    event_id: int,
    affected_node: str,
    affected_services: set[int],
    severity: float,
    request_phase: float,
    delay_phase: float,
    config: dict,
    rng: np.random.Generator,
) -> float:
    precursor_cfg = config.get("data", {}).get("cpu_precursor", {})
    sim_cfg = config.get("simulation", {})
    request_cap = float(sim_cfg.get("request_cap", 100.0))
    response_cap = float(sim_cfg.get("response_cap_ms", 420.0))
    service_id = _i(service_row, "service_id")
    is_direct = service_id in affected_services
    growth_min = float(precursor_cfg.get("request_rate_growth_min", 0.08))
    growth_max = float(precursor_cfg.get("request_rate_growth_max", 0.18))
    colocated_boost = float(precursor_cfg.get("colocated_service_boost", 0.20))
    growth = rng.uniform(growth_min, growth_max)
    boost = 1.0 if is_direct else colocated_boost
    existing_request = _f(service_row, "request_rate", 10.0)
    request_add = request_cap * growth * request_phase * boost * (0.75 + 0.50 * severity)
    new_request = min(request_cap * 1.15, max(existing_request, existing_request + request_add))
    existing_response = _f(service_row, "response_time", _f(service_row, "base_response_time", 80.0))
    response_add = response_cap * (0.06 + 0.16 * severity) * delay_phase * boost
    new_response = min(response_cap * 0.82, max(existing_response, existing_response + response_add))
    service_row.update(
        {
            "scenario": "node_overload",
            "event_id": event_id,
            "severity": round(severity, 5),
            "affected_node": affected_node,
            "request_rate": round(float(new_request), 5),
            "response_time": round(float(new_response), 5),
        }
    )
    path_row.update(
        {
            "scenario": "node_overload",
            "event_id": event_id,
            "severity": round(severity, 5),
            "affected_node": affected_node,
            "bottleneck_node": affected_node,
            "path_delay": round(float(max(_f(path_row, "path_delay", 0.0), new_response * 0.82)), 5),
        }
    )
    return float(new_response)


def inject_cpu_precursor_v1(
    logs: dict[str, list[dict]],
    config: dict,
    artifacts_dir: str | Path | None = None,
) -> dict:
    """Inject observable CPU precursor episodes into raw logs.

    The injection happens before path graph construction and labels are still
    produced by the existing future-horizon rules. No future risk labels or
    label-rule scores are written into model input features.
    """
    precursor_cfg = config.get("data", {}).get("cpu_precursor", {})
    if not bool(precursor_cfg.get("enabled", False)):
        return {"risk_injection": "none", "cpu_precursor_enabled": False, "cpu_precursor_episode_count": 0}

    seed = _data_seed(config) + 4041
    rng = _seed(seed)
    data_cfg = config.get("data", {})
    L = int(data_cfg.get("input_window", data_cfg.get("window", 12)))
    H = int(data_cfg.get("pred_horizon", data_cfg.get("horizon", 5)))
    num_steps = int(data_cfg.get("num_steps", config.get("edgesimpy", {}).get("num_steps", 1000)))
    num_services = int(data_cfg.get("num_services", config.get("edgesimpy", {}).get("num_services", 60)))
    index = _index_logs(logs)
    edge_nodes = [node for node in index["node_ids"] if str(node).startswith("edge")]
    if not edge_nodes:
        return {"risk_injection": CPU_PRECURSOR_RISK_INJECTION, "cpu_precursor_enabled": True, "cpu_precursor_episode_count": 0}

    affected_node_ratio = float(precursor_cfg.get("affected_node_ratio", 0.25))
    affected_service_ratio = float(precursor_cfg.get("affected_service_ratio", 0.35))
    warmup_steps = int(precursor_cfg.get("warmup_steps", 8))
    growth_steps = int(precursor_cfg.get("growth_steps", 12))
    queue_lag_steps = int(precursor_cfg.get("queue_lag_steps", 2))
    delay_lag_steps = int(precursor_cfg.get("delay_lag_steps", 3))
    episodes_per_split = max(2, int(round(max(1, len(edge_nodes)) * affected_node_ratio * 2.0)))
    episode_rows: list[dict] = []
    episode_id = 700000
    node_cursor = 0

    for split, times in _sample_split_times(config).items():
        for sample_time in _event_times(times, episodes_per_split, rng):
            reference_service = int(rng.integers(0, max(num_services, 1)))
            reference_path = index["path"].get((sample_time, reference_service))
            candidates = _path_edge_nodes(reference_path)
            if not candidates:
                for service_id in rng.permutation(num_services).tolist():
                    reference_path = index["path"].get((sample_time, int(service_id)))
                    candidates = _path_edge_nodes(reference_path)
                    if candidates:
                        reference_service = int(service_id)
                        break
            if not candidates:
                candidates = edge_nodes
            affected_node = candidates[node_cursor % len(candidates)]
            node_cursor += 1

            matched_services = _services_on_node(index, sample_time, affected_node)
            target_services = max(1, int(round(num_services * affected_service_ratio)))
            if len(matched_services) > target_services:
                matched_services = sorted(rng.choice(matched_services, size=target_services, replace=False).tolist())
            if not matched_services:
                matched_services = [reference_service]
            affected_services = set(int(sid) for sid in matched_services)

            start = int(max(0, sample_time - max(2, min(L - 2, warmup_steps))))
            peak = int(min(num_steps - 1, max(sample_time + 1, sample_time + min(H, max(1, growth_steps // 3)))))
            end = int(min(num_steps - 1, peak + queue_lag_steps + delay_lag_steps + 2))
            severity = float(rng.uniform(0.68, 0.86))
            current_episode_id = episode_id
            episode_id += 1
            peak_cpu = 0.0
            min_available = 1.0
            peak_queue = 0.0
            peak_response = 0.0

            for t in range(start, end + 1):
                cpu_phase = _ramp((t - start) / max(peak - start, 1))
                queue_phase = _ramp((t - start - queue_lag_steps) / max(peak - start, 1))
                delay_phase = _ramp((t - start - queue_lag_steps - delay_lag_steps) / max(peak - start, 1))
                node_row = index["node"].get((t, affected_node))
                if node_row is not None:
                    cpu_value, queue_value = _update_cpu_node(
                        node_row,
                        event_id=current_episode_id,
                        affected_node=affected_node,
                        severity=severity,
                        cpu_phase=cpu_phase,
                        queue_phase=queue_phase,
                        config=config,
                        rng=rng,
                    )
                    peak_cpu = max(peak_cpu, cpu_value)
                    min_available = min(min_available, _f(node_row, "available_cpu", min_available))
                    peak_queue = max(peak_queue, queue_value)

                colocated_services = set(_services_on_node(index, t, affected_node))
                for service_id in sorted(affected_services | colocated_services):
                    service_row = index["service"].get((t, int(service_id)))
                    path_row = index["path"].get((t, int(service_id)))
                    if service_row is None or path_row is None:
                        continue
                    if affected_node not in _split_pipe(path_row.get("path_nodes", "")) and service_id not in affected_services:
                        continue
                    response_value = _update_service_and_path(
                        service_row,
                        path_row,
                        event_id=current_episode_id,
                        affected_node=affected_node,
                        affected_services=affected_services,
                        severity=severity,
                        request_phase=cpu_phase,
                        delay_phase=delay_phase,
                        config=config,
                        rng=rng,
                    )
                    peak_response = max(peak_response, response_value)

            episode_rows.append(
                {
                    "episode_id": current_episode_id,
                    "split": split,
                    "time_start": start,
                    "time_peak": peak,
                    "node_id": affected_node,
                    "affected_services": "|".join(str(sid) for sid in sorted(affected_services)),
                    "peak_cpu_util": round(peak_cpu, 5),
                    "min_available_cpu": round(min_available, 5),
                    "peak_queue_len": round(peak_queue, 5),
                    "peak_response_time": round(peak_response, 5),
                }
            )

    episodes_path = ""
    if artifacts_dir is not None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / "cpu_precursor_episodes.csv"
        fieldnames = [
            "episode_id",
            "split",
            "time_start",
            "time_peak",
            "node_id",
            "affected_services",
            "peak_cpu_util",
            "min_available_cpu",
            "peak_queue_len",
            "peak_response_time",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_rows)
        episodes_path = str(path)

    affected_nodes = sorted({row["node_id"] for row in episode_rows})
    affected_services_all = sorted(
        {
            service_id
            for row in episode_rows
            for service_id in _split_pipe(row.get("affected_services", ""))
        }
    )
    return {
        "risk_injection": CPU_PRECURSOR_RISK_INJECTION,
        "cpu_precursor_enabled": True,
        "cpu_precursor_episode_count": len(episode_rows),
        "cpu_precursor_episodes_path": episodes_path,
        "cpu_precursor_affected_node_count": len(affected_nodes),
        "cpu_precursor_affected_service_count": len(affected_services_all),
        "cpu_precursor_seed": seed,
    }
