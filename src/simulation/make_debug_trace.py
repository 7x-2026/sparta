from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.io import write_csv_rows
from src.utils.seed import set_seed


SERVICE_TYPES = ["latency", "reliability", "cost"]


def _burst_multiplier(t: int) -> float:
    if 45 <= t < 70:
        return 1.55
    if 115 <= t < 145:
        return 2.45
    if 165 <= t < 178:
        return 1.85
    return 1.0


def generate_service_trace(config: dict) -> list[dict]:
    data_cfg = config["data"]
    rng = np.random.default_rng(config.get("seed", 42))
    rows: list[dict] = []
    for t in range(data_cfg["num_steps"]):
        daily = 1.0 + 0.22 * math.sin(2 * math.pi * t / 48.0)
        burst = _burst_multiplier(t)
        for service_id in range(data_cfg["num_services"]):
            service_type = SERVICE_TYPES[service_id % len(SERVICE_TYPES)]
            base_rate = 14.0 + 4.5 * service_id
            noise = rng.normal(0.0, 0.8)
            request_rate = max(1.0, base_rate * daily * burst + noise)
            type_base = {"latency": 38.0, "reliability": 58.0, "cost": 74.0}[service_type]
            base_response_time = max(5.0, type_base + rng.normal(0.0, 2.0))
            rows.append(
                {
                    "time": t,
                    "service_id": service_id,
                    "service_type": service_type,
                    "request_rate": round(request_rate, 4),
                    "base_response_time": round(base_response_time, 4),
                    "dependency_count": 1 + (service_id % 3),
                }
            )
    return rows


def generate_node_trace(config: dict) -> list[dict]:
    data_cfg = config["data"]
    rng = np.random.default_rng(config.get("seed", 42) + 7)
    rows: list[dict] = []
    for t in range(data_cfg["num_steps"]):
        smooth = 0.46 + 0.14 * math.sin(2 * math.pi * (t + 5) / 55.0)
        overload = 0.0
        arrival = 1.0
        if 45 <= t < 70:
            overload = 0.22
            arrival = 1.55
        if 115 <= t < 145:
            overload = 0.40
            arrival = 2.35
        if 165 <= t < 178:
            overload = 0.26
            arrival = 1.85
        cpu_scale = float(np.clip(smooth + overload + rng.normal(0.0, 0.025), 0.15, 0.98))
        mem_scale = float(np.clip(cpu_scale * 0.82 + 0.08 + rng.normal(0.0, 0.018), 0.10, 0.97))
        rows.append(
            {
                "time": t,
                "node_group": "edge",
                "cpu_scale": round(cpu_scale, 4),
                "mem_scale": round(mem_scale, 4),
                "arrival_scale": round(arrival, 4),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    traces_dir = resolve_path(config, config["data"]["traces_dir"])
    service_rows = generate_service_trace(config)
    node_rows = generate_node_trace(config)
    write_csv_rows(traces_dir / "debug_service_trace.csv", service_rows)
    write_csv_rows(traces_dir / "debug_node_trace.csv", node_rows)
    print(f"Wrote {len(service_rows)} service rows and {len(node_rows)} node rows to {traces_dir}")


if __name__ == "__main__":
    main()
