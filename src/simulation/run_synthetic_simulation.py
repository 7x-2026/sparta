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
from src.utils.io import read_csv_rows, write_csv_rows
from src.utils.seed import set_seed


SERVICE_SLA = {
    "latency": {"max_delay": 100.0, "max_loss": 0.050, "min_bandwidth": 20.0, "reliability_req": 0.990, "cost_weight": 0.20},
    "reliability": {"max_delay": 150.0, "max_loss": 0.020, "min_bandwidth": 15.0, "reliability_req": 0.999, "cost_weight": 0.30},
    "cost": {"max_delay": 200.0, "max_loss": 0.080, "min_bandwidth": 10.0, "reliability_req": 0.950, "cost_weight": 0.70},
}


def _node_type(node_id: str) -> str:
    if node_id.startswith("cloud"):
        return "cloud"
    if node_id.startswith("edge"):
        return "edge"
    return "access"


def _link_id(src: str, dst: str) -> str:
    return f"{src}-{dst}"


def _burst_level(t: int) -> float:
    if 45 <= t < 70:
        return 0.45
    if 115 <= t < 145:
        return 0.90
    if 165 <= t < 178:
        return 0.58
    return 0.0


def _build_topology(num_access: int, num_edge: int) -> tuple[list[str], list[tuple[str, str]]]:
    nodes = [f"access_{i}" for i in range(num_access)] + [f"edge_{i}" for i in range(num_edge)] + ["cloud_0"]
    links: list[tuple[str, str]] = []
    for access in [n for n in nodes if n.startswith("access")]:
        for edge_idx in range(num_edge):
            links.append((access, f"edge_{edge_idx}"))
    for edge_idx in range(num_edge):
        links.append((f"edge_{edge_idx}", "cloud_0"))
    for edge_idx in range(num_edge):
        links.append((f"edge_{edge_idx}", f"edge_{(edge_idx + 1) % num_edge}"))
    return nodes, links


def run_simulation(config: dict) -> dict[str, list[dict]]:
    data_cfg = config["data"]
    sim_cfg = config["simulation"]
    rng = np.random.default_rng(config.get("seed", 42) + 21)
    traces_dir = resolve_path(config, data_cfg["traces_dir"])
    service_trace = read_csv_rows(traces_dir / "debug_service_trace.csv")
    node_trace = {int(row["time"]): row for row in read_csv_rows(traces_dir / "debug_node_trace.csv")}
    service_by_time = {
        (int(row["time"]), int(row["service_id"])): row
        for row in service_trace
    }

    nodes, links = _build_topology(data_cfg["num_access_nodes"], data_cfg["num_edge_nodes"])
    degree = {node: 0 for node in nodes}
    for src, dst in links:
        degree[src] += 1
        degree[dst] += 1

    node_log: list[dict] = []
    link_log: list[dict] = []
    service_log: list[dict] = []
    path_log: list[dict] = []
    last_violation = {service_id: 0 for service_id in range(data_cfg["num_services"])}

    for t in range(data_cfg["num_steps"]):
        trace_row = node_trace[t]
        cpu_scale = float(trace_row["cpu_scale"])
        mem_scale = float(trace_row["mem_scale"])
        arrival_scale = float(trace_row["arrival_scale"])
        burst = _burst_level(t)

        node_state: dict[str, dict] = {}
        for node in nodes:
            ntype = _node_type(node)
            if ntype == "edge":
                edge_idx = int(node.split("_")[1])
                hot = 0.11 if edge_idx in {1, 3} and burst > 0 else 0.0
                cpu = np.clip(cpu_scale + hot + rng.normal(0.0, 0.025), 0.05, 0.99)
                mem = np.clip(mem_scale + 0.05 * hot + rng.normal(0.0, 0.020), 0.05, 0.98)
                req_load = np.clip(arrival_scale * (0.45 + 0.08 * edge_idx), 0.0, 3.0)
                queue = max(0.0, (cpu - 0.50) * sim_cfg["queue_cap"] * 1.35 + burst * 8.0 + rng.normal(0, 1.1))
            elif ntype == "cloud":
                cpu = np.clip(0.34 + 0.18 * cpu_scale + rng.normal(0.0, 0.015), 0.05, 0.80)
                mem = np.clip(0.40 + 0.12 * mem_scale + rng.normal(0.0, 0.015), 0.05, 0.85)
                req_load = np.clip(0.40 + 0.20 * arrival_scale, 0.0, 2.0)
                queue = max(0.0, (cpu - 0.55) * sim_cfg["queue_cap"] * 0.5)
            else:
                cpu = np.clip(0.18 + 0.10 * arrival_scale + rng.normal(0.0, 0.012), 0.02, 0.62)
                mem = np.clip(0.20 + 0.08 * arrival_scale + rng.normal(0.0, 0.010), 0.02, 0.60)
                req_load = np.clip(0.20 + 0.16 * arrival_scale, 0.0, 1.2)
                queue = max(0.0, (cpu - 0.40) * sim_cfg["queue_cap"] * 0.35)

            node_state[node] = {
                "cpu_util": float(cpu),
                "mem_util": float(mem),
                "queue_len": float(queue),
                "available_cpu": float(np.clip(1.0 - cpu, 0.0, 1.0)),
                "available_mem": float(np.clip(1.0 - mem, 0.0, 1.0)),
                "request_load_on_node": float(req_load),
                "degree": degree[node],
            }
            node_log.append(
                {
                    "time": t,
                    "node_id": node,
                    "node_type": ntype,
                    "cpu_util": round(float(cpu), 5),
                    "mem_util": round(float(mem), 5),
                    "queue_len": round(float(queue), 5),
                    "available_cpu": round(float(1.0 - cpu), 5),
                    "available_mem": round(float(1.0 - mem), 5),
                    "request_load_on_node": round(float(req_load), 5),
                    "node_degree": degree[node],
                }
            )

        link_state: dict[str, dict] = {}
        for src, dst in links:
            src_type = _node_type(src)
            dst_type = _node_type(dst)
            base_delay = 8.0 if "cloud" not in (src_type, dst_type) else 30.0
            base_bw = 60.0 if "cloud" not in (src_type, dst_type) else 85.0
            edge_hot = any(x in {src, dst} for x in ["edge_1", "edge_3"])
            congestion = burst * (0.50 if edge_hot else 0.28)
            bandwidth_util = float(np.clip(0.28 + 0.28 * arrival_scale + congestion + rng.normal(0, 0.035), 0.02, 0.99))
            delay_ms = float(max(1.0, base_delay * (1.0 + 1.25 * bandwidth_util) + rng.normal(0, 1.2)))
            loss = float(np.clip(0.004 + 0.045 * max(0.0, bandwidth_util - 0.65) + 0.030 * burst + rng.normal(0, 0.002), 0.0, 0.30))
            jitter_ms = float(max(0.0, delay_ms * (0.04 + 0.08 * bandwidth_util) + rng.normal(0, 0.2)))
            queue_delay_ms = float(max(0.0, delay_ms * bandwidth_util * 0.28))
            bandwidth = float(max(1.0, base_bw * (1.0 - 0.45 * bandwidth_util)))
            lid = _link_id(src, dst)
            link_state[lid] = {
                "src": src,
                "dst": dst,
                "delay_ms": delay_ms,
                "bandwidth": bandwidth,
                "loss": loss,
                "jitter_ms": jitter_ms,
                "queue_delay_ms": queue_delay_ms,
                "bandwidth_util": bandwidth_util,
            }
            link_log.append(
                {
                    "time": t,
                    "src": src,
                    "dst": dst,
                    "delay_ms": round(delay_ms, 5),
                    "bandwidth": round(bandwidth, 5),
                    "loss": round(loss, 6),
                    "jitter_ms": round(jitter_ms, 5),
                    "queue_delay_ms": round(queue_delay_ms, 5),
                    "bandwidth_util": round(bandwidth_util, 5),
                }
            )

        for service_id in range(data_cfg["num_services"]):
            src_trace = service_by_time[(t, service_id)]
            service_type = src_trace["service_type"]
            access = f"access_{service_id % data_cfg['num_access_nodes']}"
            current_edge = f"edge_{(service_id * 2 + (t // 35)) % data_cfg['num_edge_nodes']}"
            path_nodes = [access, current_edge, "cloud_0"]
            path_links = [_link_id(access, current_edge), _link_id(current_edge, "cloud_0")]
            path_link_state = [link_state[lid] for lid in path_links]
            path_delay = sum(row["delay_ms"] for row in path_link_state)
            path_loss = 1.0
            for row in path_link_state:
                path_loss *= 1.0 - row["loss"]
            path_loss = 1.0 - path_loss
            bottleneck_node = max(path_nodes, key=lambda node: node_state[node]["cpu_util"] + node_state[node]["queue_len"] / sim_cfg["queue_cap"])
            bottleneck_link = max(path_links, key=lambda lid: link_state[lid]["bandwidth_util"] + 4.0 * link_state[lid]["loss"])
            request_rate = float(src_trace["request_rate"])
            base_response = float(src_trace["base_response_time"])
            node_pressure = max(node_state[node]["cpu_util"] for node in path_nodes)
            queue_pressure = max(node_state[node]["queue_len"] for node in path_nodes)
            response_time = (
                base_response
                + 0.55 * path_delay
                + 0.70 * request_rate
                + 1.35 * queue_pressure
                + 42.0 * max(0.0, node_pressure - 0.78)
                + rng.normal(0, 2.0)
            )
            response_time = float(np.clip(response_time, 1.0, sim_cfg["response_cap_ms"]))
            sla = SERVICE_SLA[service_type]
            violated = int(response_time >= sla["max_delay"] or path_loss >= sla["max_loss"])
            service_log.append(
                {
                    "time": t,
                    "service_id": service_id,
                    "user_id": service_id % data_cfg["num_users"],
                    "service_type": service_type,
                    "request_rate": round(request_rate, 5),
                    "response_time": round(response_time, 5),
                    "base_response_time": round(base_response, 5),
                    "dependency_count": int(src_trace["dependency_count"]),
                    "current_edge": current_edge,
                    "cloud_node": "cloud_0",
                    "last_violation": last_violation[service_id],
                }
            )
            path_log.append(
                {
                    "time": t,
                    "service_id": service_id,
                    "path_nodes": "|".join(path_nodes),
                    "path_links": "|".join(path_links),
                    "path_delay": round(path_delay, 5),
                    "path_loss": round(path_loss, 6),
                    "bottleneck_node": bottleneck_node,
                    "bottleneck_link": bottleneck_link,
                }
            )
            last_violation[service_id] = violated

    sla_log = []
    for service_id in range(data_cfg["num_services"]):
        service_type = service_by_time[(0, service_id)]["service_type"]
        sla = SERVICE_SLA[service_type]
        sla_log.append(
            {
                "service_id": service_id,
                "service_type": service_type,
                "max_delay": sla["max_delay"],
                "max_loss": sla["max_loss"],
                "min_bandwidth": sla["min_bandwidth"],
                "reliability_req": sla["reliability_req"],
                "cost_weight": sla["cost_weight"],
            }
        )

    return {
        "node_log.csv": node_log,
        "link_log.csv": link_log,
        "service_log.csv": service_log,
        "path_log.csv": path_log,
        "sla_log.csv": sla_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    raw_dir = resolve_path(config, config["data"]["raw_logs_dir"])
    outputs = run_simulation(config)
    for filename, rows in outputs.items():
        write_csv_rows(raw_dir / filename, rows)
        print(f"Wrote {len(rows)} rows to {raw_dir / filename}")


if __name__ == "__main__":
    main()
