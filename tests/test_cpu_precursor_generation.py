from __future__ import annotations

from pathlib import Path

from src.simulation.risk_injection import inject_cpu_precursor_v1
from src.utils.io import read_csv_rows


def make_config() -> dict:
    return {
        "seed": 7,
        "experiment": {"seed": 7, "data_seed": 7},
        "data": {
            "input_window": 6,
            "pred_horizon": 3,
            "num_steps": 40,
            "num_services": 4,
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "risk_injection": "cpu_precursor_v1",
            "cpu_precursor": {
                "enabled": True,
                "affected_node_ratio": 0.5,
                "affected_service_ratio": 0.5,
                "warmup_steps": 4,
                "growth_steps": 6,
                "queue_lag_steps": 2,
                "delay_lag_steps": 2,
                "enable_cpu_slope_features": True,
            },
        },
        "simulation": {"queue_cap": 50.0, "request_cap": 100.0, "response_cap_ms": 400.0},
    }


def make_logs() -> dict[str, list[dict]]:
    logs = {name: [] for name in ["node_log.csv", "link_log.csv", "service_log.csv", "path_log.csv", "sla_log.csv"]}
    nodes = ["access_0", "edge_0", "edge_1", "cloud_0"]
    paths = {
        0: (["access_0", "edge_0", "cloud_0"], ["access_0-edge_0", "edge_0-cloud_0"], "edge_0"),
        1: (["access_0", "edge_0", "cloud_0"], ["access_0-edge_0", "edge_0-cloud_0"], "edge_0"),
        2: (["access_0", "edge_1", "cloud_0"], ["access_0-edge_1", "edge_1-cloud_0"], "edge_1"),
        3: (["access_0", "edge_1", "cloud_0"], ["access_0-edge_1", "edge_1-cloud_0"], "edge_1"),
    }
    for t in range(40):
        for node in nodes:
            logs["node_log.csv"].append(
                {
                    "time": t,
                    "node_id": node,
                    "node_type": "edge" if node.startswith("edge") else "access",
                    "cpu_util": 0.35,
                    "mem_util": 0.30,
                    "queue_len": 2.0,
                    "available_cpu": 0.65,
                    "available_mem": 0.70,
                    "request_load_on_node": 10.0,
                    "node_degree": 2,
                    "scenario": "normal",
                }
            )
        for service_id, (path_nodes, path_links, current_edge) in paths.items():
            logs["service_log.csv"].append(
                {
                    "time": t,
                    "service_id": service_id,
                    "user_id": service_id,
                    "service_type": "latency",
                    "request_rate": 10.0,
                    "response_time": 60.0,
                    "current_edge": current_edge,
                    "cloud_node": "cloud_0",
                    "base_response_time": 60.0,
                    "dependency_count": 1,
                    "scenario": "normal",
                }
            )
            logs["path_log.csv"].append(
                {
                    "time": t,
                    "service_id": service_id,
                    "path_nodes": "|".join(path_nodes),
                    "path_links": "|".join(path_links),
                    "path_delay": 60.0,
                    "path_loss": 0.005,
                    "bottleneck_node": current_edge,
                    "bottleneck_link": path_links[-1],
                    "scenario": "normal",
                }
            )
        for src, dst in [("access_0", "edge_0"), ("edge_0", "cloud_0"), ("access_0", "edge_1"), ("edge_1", "cloud_0")]:
            logs["link_log.csv"].append(
                {
                    "time": t,
                    "src": src,
                    "dst": dst,
                    "delay_ms": 5.0,
                    "bandwidth": 80.0,
                    "loss": 0.005,
                    "jitter_ms": 1.0,
                    "queue_delay_ms": 0.5,
                    "bandwidth_util": 0.25,
                    "scenario": "normal",
                }
            )
    for service_id in paths:
        logs["sla_log.csv"].append(
            {
                "service_id": service_id,
                "service_type": "latency",
                "max_delay": 120.0,
                "max_loss": 0.05,
                "min_bandwidth": 20.0,
                "reliability_req": 0.99,
                "cost_weight": 0.2,
            }
        )
    return logs


def by_time_node(logs: dict[str, list[dict]], time: int, node: str) -> dict:
    return next(row for row in logs["node_log.csv"] if int(row["time"]) == time and row["node_id"] == node)


def by_time_service(logs: dict[str, list[dict]], time: int, service_id: int) -> dict:
    return next(row for row in logs["service_log.csv"] if int(row["time"]) == time and int(row["service_id"]) == service_id)


def test_cpu_precursor_generates_episode_and_lagged_signals(tmp_path: Path):
    config = make_config()
    logs = make_logs()

    metadata = inject_cpu_precursor_v1(logs, config, artifacts_dir=tmp_path)

    episode_path = tmp_path / "cpu_precursor_episodes.csv"
    rows = read_csv_rows(episode_path)
    assert metadata["risk_injection"] == "cpu_precursor_v1"
    assert episode_path.exists()
    assert rows

    episode = rows[0]
    start = int(episode["time_start"])
    peak = int(episode["time_peak"])
    node_id = episode["node_id"]
    service_id = int(episode["affected_services"].split("|")[0])

    node_start = by_time_node(logs, start, node_id)
    node_peak = by_time_node(logs, peak, node_id)
    node_before_queue = by_time_node(logs, start + 1, node_id)
    node_after_queue = by_time_node(logs, start + 3, node_id)
    service_before_delay = by_time_service(logs, start + 1, service_id)
    service_after_delay = by_time_service(logs, peak, service_id)

    assert float(node_peak["cpu_util"]) > float(node_start["cpu_util"])
    assert float(node_peak["available_cpu"]) < float(node_start["available_cpu"])
    assert float(node_after_queue["queue_len"]) > float(node_before_queue["queue_len"])
    assert float(service_after_delay["response_time"]) > float(service_before_delay["response_time"])


def test_cpu_precursor_does_not_modify_link_loss_or_bandwidth(tmp_path: Path):
    config = make_config()
    logs = make_logs()
    original_links = [(row["loss"], row["bandwidth"], row["bandwidth_util"]) for row in logs["link_log.csv"]]

    inject_cpu_precursor_v1(logs, config, artifacts_dir=tmp_path)

    after_links = [(row["loss"], row["bandwidth"], row["bandwidth_util"]) for row in logs["link_log.csv"]]
    assert after_links == original_links
