from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.export_logs import export_logs
from src.simulation.synthetic_scenarios import SCENARIOS, scenario_state
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


SERVICE_TYPES = ["latency", "reliability", "cost"]
SERVICE_SLA = {
    "latency": {"max_delay": 105.0, "max_loss": 0.050, "min_bandwidth": 20.0, "reliability_req": 0.990, "cost_weight": 0.20},
    "reliability": {"max_delay": 150.0, "max_loss": 0.022, "min_bandwidth": 15.0, "reliability_req": 0.999, "cost_weight": 0.30},
    "cost": {"max_delay": 205.0, "max_loss": 0.080, "min_bandwidth": 10.0, "reliability_req": 0.950, "cost_weight": 0.70},
}


def _link_id(src: str, dst: str) -> str:
    return f"{src}-{dst}"


def _node_type(node_id: str) -> str:
    if node_id.startswith("cloud"):
        return "cloud"
    if node_id.startswith("edge"):
        return "edge"
    return "access"


def build_full_topology() -> tuple[list[str], list[tuple[str, str]]]:
    nodes = ["access_0", "access_1", "edge_0", "edge_1", "edge_2", "edge_3", "edge_4", "cloud_0"]
    links = [
        ("access_0", "edge_0"),
        ("access_0", "edge_1"),
        ("access_1", "edge_2"),
        ("access_1", "edge_3"),
        ("edge_0", "edge_1"),
        ("edge_1", "edge_2"),
        ("edge_2", "edge_3"),
        ("edge_3", "edge_4"),
        ("edge_4", "edge_0"),
        ("edge_0", "cloud_0"),
        ("edge_2", "cloud_0"),
        ("edge_4", "cloud_0"),
    ]
    return nodes, links


def _path_for_service(service_id: int, local_t: int) -> tuple[list[str], list[str], str]:
    if service_id % 2 == 0:
        access = "access_0"
        edge = "edge_0" if (service_id + local_t // 25) % 3 != 1 else "edge_1"
        if edge == "edge_0":
            return [access, "edge_0", "cloud_0"], [_link_id(access, "edge_0"), "edge_0-cloud_0"], edge
        return [access, "edge_1", "edge_0", "cloud_0"], [_link_id(access, "edge_1"), "edge_0-edge_1", "edge_0-cloud_0"], edge
    access = "access_1"
    edge = "edge_2" if (service_id + local_t // 30) % 3 != 1 else "edge_3"
    if edge == "edge_2":
        return [access, "edge_2", "cloud_0"], [_link_id(access, "edge_2"), "edge_2-cloud_0"], edge
    return [access, "edge_3", "edge_2", "cloud_0"], [_link_id(access, "edge_3"), "edge_2-edge_3", "edge_2-cloud_0"], edge


def _scenario_schedule(total_steps: int, scenarios: list[str]) -> list[tuple[str, int, int]]:
    base = total_steps // len(scenarios)
    schedule = []
    start = 0
    for idx, scenario in enumerate(scenarios):
        end = start + base
        if idx == len(scenarios) - 1:
            end = total_steps
        schedule.append((scenario, start, end))
        start = end
    return schedule


def generate_synthetic_full_logs(config: dict) -> dict[str, list[dict]]:
    data_cfg = config["data"]
    sim_cfg = config["simulation"]
    scenarios = sim_cfg.get("scenarios", SCENARIOS)
    if not scenarios:
        raise ValueError("simulation.scenarios must not be empty")
    total_steps = int(data_cfg["num_steps"])
    num_services = int(data_cfg["num_services"])
    rng = np.random.default_rng(int(config.get("seed", 42)))
    nodes, links = build_full_topology()
    if int(data_cfg.get("num_nodes", len(nodes))) != len(nodes):
        raise ValueError("synthetic full v0 uses exactly 8 nodes")
    if int(data_cfg.get("num_links", len(links))) != len(links):
        raise ValueError("synthetic full v0 uses exactly 12 links")

    degree = {node: 0 for node in nodes}
    for src, dst in links:
        degree[src] += 1
        degree[dst] += 1

    node_log: list[dict] = []
    link_log: list[dict] = []
    service_log: list[dict] = []
    path_log: list[dict] = []
    last_violation = {service_id: 0 for service_id in range(num_services)}

    schedule = _scenario_schedule(total_steps, scenarios)
    scenario_by_time = {}
    for scenario, start, end in schedule:
        for t in range(start, end):
            scenario_by_time[t] = (scenario, t - start, end - start)

    for t in range(total_steps):
        scenario, local_t, scenario_steps = scenario_by_time[t]
        state = scenario_state(scenario, local_t, scenario_steps)
        node_state: dict[str, dict] = {}
        link_state: dict[str, dict] = {}

        for node in nodes:
            ntype = _node_type(node)
            hot = 1.0 if node in state.hot_edges else 0.0
            if ntype == "edge":
                base_cpu = 0.36 + 0.04 * (int(node.split("_")[1]) % 3)
                cpu = np.clip(base_cpu + 0.34 * state.node_pressure * hot + 0.12 * state.request_multiplier + rng.normal(0, 0.025 * state.noise_scale), 0.05, 0.99)
                mem = np.clip(0.30 + 0.52 * cpu + rng.normal(0, 0.018 * state.noise_scale), 0.05, 0.98)
                req_load = np.clip(0.30 + 0.55 * state.request_multiplier + 0.35 * hot, 0.0, 3.0)
                queue = max(0.0, (cpu - 0.48) * float(sim_cfg.get("queue_cap", 50.0)) * 1.25 + 11.0 * state.node_pressure * hot + rng.normal(0, 1.0))
            elif ntype == "cloud":
                cpu = np.clip(0.30 + 0.16 * state.request_multiplier + rng.normal(0, 0.012), 0.05, 0.82)
                mem = np.clip(0.38 + 0.15 * state.request_multiplier + rng.normal(0, 0.012), 0.05, 0.86)
                req_load = np.clip(0.40 + 0.24 * state.request_multiplier, 0.0, 2.0)
                queue = max(0.0, (cpu - 0.58) * float(sim_cfg.get("queue_cap", 50.0)) * 0.45)
            else:
                cpu = np.clip(0.16 + 0.10 * state.request_multiplier + rng.normal(0, 0.010), 0.02, 0.62)
                mem = np.clip(0.20 + 0.08 * state.request_multiplier + rng.normal(0, 0.010), 0.02, 0.62)
                req_load = np.clip(0.18 + 0.18 * state.request_multiplier, 0.0, 1.4)
                queue = max(0.0, (cpu - 0.40) * float(sim_cfg.get("queue_cap", 50.0)) * 0.25)

            node_state[node] = {
                "cpu_util": float(cpu),
                "mem_util": float(mem),
                "queue_len": float(queue),
                "available_cpu": float(np.clip(1.0 - cpu, 0.0, 1.0)),
                "available_mem": float(np.clip(1.0 - mem, 0.0, 1.0)),
                "request_load_on_node": float(req_load),
                "node_degree": int(degree[node]),
            }
            node_log.append(
                {
                    "time": t,
                    "scenario": scenario,
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

        for src, dst in links:
            lid = _link_id(src, dst)
            hot = 1.0 if lid in state.hot_links else 0.0
            cloud_link = 1.0 if "cloud_0" in (src, dst) else 0.0
            base_delay = 8.0 + 21.0 * cloud_link
            base_bw = 62.0 + 22.0 * cloud_link
            bandwidth_util = float(np.clip(0.24 + 0.22 * state.request_multiplier + 0.48 * state.link_pressure * hot + rng.normal(0, 0.030 * state.noise_scale), 0.02, 0.99))
            delay_ms = float(max(1.0, base_delay * (1.0 + 1.30 * bandwidth_util) + 13.0 * state.link_pressure * hot + rng.normal(0, 1.1)))
            loss = float(np.clip(0.003 + 0.035 * max(0.0, bandwidth_util - 0.62) + 0.055 * state.link_pressure * hot + rng.normal(0, 0.002), 0.0, 0.35))
            jitter_ms = float(max(0.0, delay_ms * (0.04 + 0.07 * bandwidth_util) + rng.normal(0, 0.2)))
            queue_delay_ms = float(max(0.0, delay_ms * bandwidth_util * (0.24 + 0.18 * hot)))
            bandwidth = float(max(1.0, base_bw * (1.0 - 0.48 * bandwidth_util)))
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
                    "scenario": scenario,
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

        for service_id in range(num_services):
            service_type = SERVICE_TYPES[service_id % len(SERVICE_TYPES)]
            path_nodes, path_links, current_edge = _path_for_service(service_id, local_t)
            path_link_state = [link_state[link] for link in path_links]
            path_delay = sum(row["delay_ms"] for row in path_link_state)
            path_loss_survival = 1.0
            for row in path_link_state:
                path_loss_survival *= 1.0 - row["loss"]
            path_loss = 1.0 - path_loss_survival
            base_rate = 7.0 + (service_id % 11) * 1.8
            service_wave = 1.0 + 0.12 * np.sin(2 * np.pi * (local_t + service_id) / 60.0)
            request_rate = float(max(0.5, base_rate * state.request_multiplier * service_wave + rng.normal(0, 0.5 * state.noise_scale)))
            base_response = {"latency": 34.0, "reliability": 54.0, "cost": 70.0}[service_type] + (service_id % 5) * 1.5
            max_cpu = max(node_state[node]["cpu_util"] for node in path_nodes)
            max_queue = max(node_state[node]["queue_len"] for node in path_nodes)
            response_time = float(
                np.clip(
                    base_response
                    + 0.50 * path_delay
                    + 0.55 * request_rate
                    + 1.25 * max_queue
                    + 45.0 * max(0.0, max_cpu - 0.76)
                    + rng.normal(0, 2.0 * state.noise_scale),
                    1.0,
                    float(sim_cfg.get("response_cap_ms", 420.0)),
                )
            )
            sla = SERVICE_SLA[service_type]
            violated = int(response_time >= sla["max_delay"] or path_loss >= sla["max_loss"])
            bottleneck_node = max(path_nodes, key=lambda node: node_state[node]["cpu_util"] + node_state[node]["queue_len"] / float(sim_cfg.get("queue_cap", 50.0)))
            bottleneck_link = max(path_links, key=lambda link: link_state[link]["bandwidth_util"] + 4.0 * link_state[link]["loss"])
            service_log.append(
                {
                    "time": t,
                    "scenario": scenario,
                    "service_id": service_id,
                    "user_id": service_id % int(data_cfg.get("num_users", 300)),
                    "service_type": service_type,
                    "request_rate": round(request_rate, 5),
                    "response_time": round(response_time, 5),
                    "base_response_time": round(base_response, 5),
                    "dependency_count": 1 + (service_id % 5),
                    "current_edge": current_edge,
                    "cloud_node": "cloud_0",
                    "last_violation": last_violation[service_id],
                }
            )
            path_log.append(
                {
                    "time": t,
                    "scenario": scenario,
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
    for service_id in range(num_services):
        service_type = SERVICE_TYPES[service_id % len(SERVICE_TYPES)]
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
    parser.add_argument("--config", default="configs/sparta_synthetic_full.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    logs = generate_synthetic_full_logs(config)
    raw_dir = resolve_path(config, config["data"]["raw_logs_dir"])
    export_logs(raw_dir, logs)


if __name__ == "__main__":
    main()
