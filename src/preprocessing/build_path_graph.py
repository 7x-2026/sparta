from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.io import load_pickle, read_csv_rows, save_pickle


NODE_TYPE_ID = {"access": 0.25, "edge": 0.65, "cloud": 1.0}
SERVICE_TYPE_ID = {"latency": 0.0, "reliability": 0.5, "cost": 1.0}


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _i(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def _link_id(src: str, dst: str) -> str:
    return f"{src}-{dst}"


def _split_pipe(value: str) -> list[str]:
    if not value:
        return []
    return [part for part in value.split("|") if part]


def _node_type(node_id: str) -> str:
    if node_id.startswith("cloud"):
        return "cloud"
    if node_id.startswith("edge"):
        return "edge"
    return "access"


def load_logs(raw_dir: str | Path) -> dict:
    raw_dir = Path(raw_dir)
    rows = {
        "node": read_csv_rows(raw_dir / "node_log.csv"),
        "link": read_csv_rows(raw_dir / "link_log.csv"),
        "service": read_csv_rows(raw_dir / "service_log.csv"),
        "path": read_csv_rows(raw_dir / "path_log.csv"),
        "sla": read_csv_rows(raw_dir / "sla_log.csv"),
    }
    indexes = {
        "node": {},
        "link": {},
        "service": {},
        "path": {},
        "sla": {},
        "all_nodes": set(),
        "all_links": set(),
        "link_pairs": {},
    }
    for row in rows["node"]:
        t = _i(row, "time")
        node_id = row["node_id"]
        indexes["node"][(t, node_id)] = row
        indexes["all_nodes"].add(node_id)
    for row in rows["link"]:
        t = _i(row, "time")
        src, dst = row["src"], row["dst"]
        lid = _link_id(src, dst)
        indexes["link"][(t, lid)] = row
        indexes["link"][(t, _link_id(dst, src))] = row
        indexes["all_links"].add(lid)
        indexes["link_pairs"][lid] = (src, dst)
    for row in rows["service"]:
        indexes["service"][(_i(row, "time"), _i(row, "service_id"))] = row
    for row in rows["path"]:
        indexes["path"][(_i(row, "time"), _i(row, "service_id"))] = row
    for row in rows["sla"]:
        indexes["sla"][_i(row, "service_id")] = row
    return {"rows": rows, "idx": indexes}


def select_nodes(path_row: dict, logs: dict, max_nodes: int) -> list[str]:
    path_nodes = _split_pipe(path_row["path_nodes"])
    chosen: list[str] = []
    for node in path_nodes + [path_row.get("bottleneck_node", "")]:
        if node and node not in chosen:
            chosen.append(node)
    edge_nodes = sorted(node for node in logs["idx"]["all_nodes"] if str(node).startswith("edge"))
    for node in edge_nodes:
        if len(chosen) >= max_nodes:
            break
        if node not in chosen:
            chosen.append(node)
    return chosen[:max_nodes]


def select_links(path_row: dict, logs: dict, node_ids: list[str], max_links: int) -> list[str]:
    path_links = _split_pipe(path_row["path_links"])
    chosen: list[str] = []
    for link in path_links + [path_row.get("bottleneck_link", "")]:
        if link and link not in chosen:
            chosen.append(link)
    node_set = set(node_ids)
    for link_id in sorted(logs["idx"]["all_links"]):
        if len(chosen) >= max_links:
            break
        src, dst = logs["idx"]["link_pairs"].get(link_id, ("", ""))
        if src in node_set and dst in node_set and link_id not in chosen:
            chosen.append(link_id)
    return chosen[:max_links]


def make_node_feature(row: dict | None, node_id: str, current_edge: str, degree_max: float, queue_cap: float) -> list[float]:
    if row is None:
        return [0.0] * 8
    queue_norm = min(_f(row, "queue_len") / max(queue_cap, 1.0), 1.0)
    return [
        float(np.clip(_f(row, "cpu_util"), 0.0, 1.0)),
        float(np.clip(_f(row, "mem_util"), 0.0, 1.0)),
        float(np.clip(queue_norm, 0.0, 1.0)),
        float(np.clip(_f(row, "available_cpu"), 0.0, 1.0)),
        float(np.clip(_f(row, "available_mem"), 0.0, 1.0)),
        NODE_TYPE_ID.get(_node_type(node_id), 0.0),
        float(np.clip(_f(row, "node_degree") / max(degree_max, 1.0), 0.0, 1.0)),
        1.0 if node_id == current_edge else 0.0,
    ]


def make_link_feature(row: dict | None, link_id: str, path_links: list[str], caps: dict) -> list[float]:
    if row is None:
        return [0.0] * 8
    if link_id in path_links:
        hop_position = path_links.index(link_id) / max(len(path_links) - 1, 1)
        is_current = 1.0
    else:
        hop_position = 0.0
        is_current = 0.0
    return [
        float(np.clip(_f(row, "delay_ms") / caps["delay"], 0.0, 1.0)),
        float(np.clip(_f(row, "bandwidth") / caps["bandwidth"], 0.0, 1.0)),
        float(np.clip(_f(row, "loss"), 0.0, 1.0)),
        float(np.clip(_f(row, "jitter_ms") / caps["delay"], 0.0, 1.0)),
        float(np.clip(_f(row, "queue_delay_ms") / caps["delay"], 0.0, 1.0)),
        float(np.clip(_f(row, "bandwidth_util"), 0.0, 1.0)),
        is_current,
        float(np.clip(hop_position, 0.0, 1.0)),
    ]


def make_service_feature(row: dict | None, caps: dict) -> list[float]:
    if row is None:
        return [0.0] * 5
    return [
        float(np.clip(_f(row, "request_rate") / caps["request"], 0.0, 1.0)),
        float(np.clip(_f(row, "response_time") / caps["response"], 0.0, 1.0)),
        SERVICE_TYPE_ID.get(row.get("service_type", "latency"), 0.0),
        float(np.clip(_f(row, "base_response_time") / caps["response"], 0.0, 1.0)),
        float(np.clip(_f(row, "dependency_count") / 5.0, 0.0, 1.0)),
    ]


def make_sla_feature(row: dict, caps: dict) -> list[float]:
    return [
        float(np.clip(_f(row, "max_delay") / caps["response"], 0.0, 1.0)),
        float(np.clip(_f(row, "max_loss") / 0.10, 0.0, 1.0)),
        float(np.clip(_f(row, "min_bandwidth") / caps["bandwidth"], 0.0, 1.0)),
        float(np.clip((_f(row, "reliability_req") - 0.90) / 0.10, 0.0, 1.0)),
        float(np.clip(_f(row, "cost_weight"), 0.0, 1.0)),
    ]


def build_sample(service_id: int, time: int, logs: dict, config: dict) -> dict:
    data_cfg = config["data"]
    sim_cfg = config["simulation"]
    L = data_cfg["input_window"]
    max_nodes = data_cfg["max_nodes"]
    max_links = data_cfg["max_links"]
    path_row = logs["idx"]["path"][(time, service_id)]
    service_row = logs["idx"]["service"][(time, service_id)]
    node_ids = select_nodes(path_row, logs, max_nodes)
    link_ids = select_links(path_row, logs, node_ids, max_links)
    path_links = _split_pipe(path_row["path_links"])
    degree_max = max((_f(row, "node_degree") for row in logs["rows"]["node"]), default=1.0)
    caps = {
        "delay": float(sim_cfg.get("delay_cap_ms", 300.0)),
        "bandwidth": float(sim_cfg.get("bandwidth_cap", 100.0)),
        "request": float(sim_cfg.get("request_cap", 100.0)),
        "response": float(sim_cfg.get("response_cap_ms", 400.0)),
    }

    node_x = np.zeros((L, max_nodes, data_cfg["node_feat_dim"]), dtype=np.float32)
    link_x = np.zeros((L, max_links, data_cfg["link_feat_dim"]), dtype=np.float32)
    service_x = np.zeros((L, data_cfg["service_feat_dim"]), dtype=np.float32)
    node_mask = np.zeros((max_nodes,), dtype=np.bool_)
    link_mask = np.zeros((max_links,), dtype=np.bool_)
    node_mask[: len(node_ids)] = True
    link_mask[: len(link_ids)] = True

    history = list(range(time - L + 1, time + 1))
    for li, ht in enumerate(history):
        hist_service = logs["idx"]["service"].get((ht, service_id), service_row)
        hist_path = logs["idx"]["path"].get((ht, service_id), path_row)
        hist_path_links = _split_pipe(hist_path.get("path_links", ""))
        current_edge = hist_service.get("current_edge", service_row.get("current_edge", ""))
        for ni, node_id in enumerate(node_ids):
            node_x[li, ni] = np.asarray(
                make_node_feature(
                    logs["idx"]["node"].get((ht, node_id)),
                    node_id,
                    current_edge,
                    degree_max,
                    float(sim_cfg.get("queue_cap", 50.0)),
                ),
                dtype=np.float32,
            )
        for ei, link_id in enumerate(link_ids):
            link_x[li, ei] = np.asarray(
                make_link_feature(logs["idx"]["link"].get((ht, link_id)), link_id, hist_path_links, caps),
                dtype=np.float32,
            )
        service_x[li] = np.asarray(make_service_feature(hist_service, caps), dtype=np.float32)

    sla_x = np.asarray(make_sla_feature(logs["idx"]["sla"][service_id], caps), dtype=np.float32)
    adj = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    for link_id in link_ids:
        src, dst = logs["idx"]["link_pairs"].get(link_id, ("", ""))
        if src in node_index and dst in node_index:
            i, j = node_index[src], node_index[dst]
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    for idx in range(len(node_ids)):
        adj[idx, idx] = 1.0

    sample = {
        "sample_id": f"debug_s{service_id}_t{time}",
        "service_id": int(service_id),
        "time": int(time),
        "scenario": path_row.get("scenario", "unknown"),
        "node_x": node_x,
        "link_x": link_x,
        "service_x": service_x,
        "sla_x": sla_x,
        "adj": adj,
        "node_mask": node_mask,
        "link_mask": link_mask,
        "node_ids": node_ids,
        "link_ids": link_ids,
        "path_nodes": _split_pipe(path_row["path_nodes"]),
        "path_links": path_links,
    }
    assert sample["node_x"].shape == (L, max_nodes, data_cfg["node_feat_dim"])
    assert sample["link_x"].shape == (L, max_links, data_cfg["link_feat_dim"])
    assert sample["service_x"].shape == (L, data_cfg["service_feat_dim"])
    assert sample["sla_x"].shape == (data_cfg["sla_feat_dim"],)
    assert sample["adj"].shape == (max_nodes, max_nodes)
    return sample


def build_path_graphs(config: dict) -> list[dict]:
    logs = load_logs(resolve_path(config, config["data"]["raw_logs_dir"]))
    data_cfg = config["data"]
    L = data_cfg["input_window"]
    H = data_cfg["pred_horizon"]
    max_t = data_cfg["num_steps"] - H - 1
    samples: list[dict] = []
    for time in range(L - 1, max_t + 1):
        for service_id in range(data_cfg["num_services"]):
            if (time, service_id) in logs["idx"]["path"]:
                samples.append(build_sample(service_id, time, logs, config))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    samples = build_path_graphs(config)
    out_path = resolve_path(config, config["data"]["processed_dir"]) / "path_graphs.pkl"
    save_pickle(out_path, samples)
    print(f"Wrote {len(samples)} path graph samples to {out_path}")


if __name__ == "__main__":
    main()
