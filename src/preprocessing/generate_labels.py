from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.build_path_graph import _f, _i, _split_pipe, load_logs
from src.utils.config import load_config, resolve_path
from src.utils.io import load_pickle, save_pickle


METRIC_NAMES = ["delay", "loss", "cpu", "queue", "bandwidth"]


def _future_times(time: int, horizon: int, num_steps: int) -> list[int]:
    return [t for t in range(time + 1, min(time + horizon, num_steps - 1) + 1)]


def _link_score(row: dict | None, delay_cap: float) -> float:
    if row is None:
        return -1.0
    return (
        0.4 * min(_f(row, "delay_ms") / delay_cap, 1.0)
        + 0.3 * min(_f(row, "loss"), 1.0)
        + 0.3 * min(_f(row, "bandwidth_util"), 1.0)
    )


def _node_score(row: dict | None, queue_cap: float) -> float:
    if row is None:
        return -1.0
    return (
        0.4 * min(_f(row, "cpu_util"), 1.0)
        + 0.3 * min(_f(row, "queue_len") / queue_cap, 1.0)
        + 0.3 * min(1.0 - _f(row, "available_cpu"), 1.0)
    )


def assign_risk_label(service_id: int, time: int, logs: dict, sla_row: dict, horizon: int, config: dict) -> int:
    sim_cfg = config["simulation"]
    max_delay = _f(sla_row, "max_delay")
    max_loss = _f(sla_row, "max_loss")
    queue_cap = float(sim_cfg.get("queue_cap", 50.0))
    future = _future_times(time, horizon, config["data"]["num_steps"])

    violated = False
    risky = False
    for ft in future:
        service = logs["idx"]["service"].get((ft, service_id))
        path = logs["idx"]["path"].get((ft, service_id))
        if service is None or path is None:
            continue
        response = _f(service, "response_time")
        loss = _f(path, "path_loss")
        if response >= max_delay or loss >= max_loss:
            violated = True
        if response >= 0.75 * max_delay or loss >= 0.75 * max_loss:
            risky = True
        for node_id in _split_pipe(path.get("path_nodes", "")):
            node = logs["idx"]["node"].get((ft, node_id))
            if node and (_f(node, "cpu_util") >= 0.85 or _f(node, "queue_len") / queue_cap >= 0.85):
                risky = True
        for link_id in _split_pipe(path.get("path_links", "")):
            link = logs["idx"]["link"].get((ft, link_id))
            if link and _f(link, "bandwidth_util") >= 0.85:
                risky = True
    if violated:
        return 2
    if risky:
        return 1
    return 0


def assign_attribution(sample: dict, logs: dict, config: dict) -> tuple[int, int, int, int]:
    if sample["risk_label"] == 0:
        return -100, -100, -100, 0

    sim_cfg = config["simulation"]
    service_id = int(sample["service_id"])
    time = int(sample["time"])
    horizon = config["data"]["pred_horizon"]
    queue_cap = float(sim_cfg.get("queue_cap", 50.0))
    delay_cap = float(sim_cfg.get("delay_cap_ms", 300.0))
    future = _future_times(time, horizon, config["data"]["num_steps"])
    node_scores = {node: -1.0 for node in sample["node_ids"]}
    link_scores = {link: -1.0 for link in sample["link_ids"]}
    metric_scores = np.zeros((5,), dtype=np.float32)
    sla = logs["idx"]["sla"][service_id]
    max_delay = max(_f(sla, "max_delay"), 1e-6)
    max_loss = max(_f(sla, "max_loss"), 1e-6)

    for ft in future:
        service = logs["idx"]["service"].get((ft, service_id))
        path = logs["idx"]["path"].get((ft, service_id))
        if service is None or path is None:
            continue
        path_nodes = _split_pipe(path.get("path_nodes", ""))
        path_links = _split_pipe(path.get("path_links", ""))
        metric_scores[0] = max(metric_scores[0], _f(service, "response_time") / max_delay)
        metric_scores[1] = max(metric_scores[1], _f(path, "path_loss") / max_loss)
        for node in path_nodes:
            row = logs["idx"]["node"].get((ft, node))
            score = _node_score(row, queue_cap)
            if node in node_scores:
                node_scores[node] = max(node_scores[node], score)
            if row:
                metric_scores[2] = max(metric_scores[2], _f(row, "cpu_util") / 0.85)
                metric_scores[3] = max(metric_scores[3], _f(row, "queue_len") / queue_cap)
        for link in path_links:
            row = logs["idx"]["link"].get((ft, link))
            score = _link_score(row, delay_cap)
            if link in link_scores:
                link_scores[link] = max(link_scores[link], score)
            if row:
                metric_scores[4] = max(metric_scores[4], _f(row, "bandwidth_util") / 0.85)

    best_node = max(node_scores, key=node_scores.get)
    best_link = max(link_scores, key=link_scores.get)
    risk_node = sample["node_ids"].index(best_node)
    risk_link = sample["link_ids"].index(best_link)
    risk_metric = int(metric_scores.argmax())
    return risk_node, risk_link, risk_metric, 1


def label_samples(samples: list[dict], logs: dict, config: dict) -> list[dict]:
    labeled: list[dict] = []
    for sample in samples:
        service_id = int(sample["service_id"])
        risk_label = assign_risk_label(
            service_id,
            int(sample["time"]),
            logs,
            logs["idx"]["sla"][service_id],
            config["data"]["pred_horizon"],
            config,
        )
        sample = dict(sample)
        sample["risk_label"] = int(risk_label)
        sample["risk_node"], sample["risk_link"], sample["risk_metric"], sample["attr_mask"] = assign_attribution(sample, logs, config)
        if sample["risk_label"] == 0:
            assert sample["risk_node"] == -100
            assert sample["risk_link"] == -100
            assert sample["risk_metric"] == -100
            assert sample["attr_mask"] == 0
        labeled.append(sample)
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_debug.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    processed_dir = resolve_path(config, config["data"]["processed_dir"])
    raw_dir = resolve_path(config, config["data"]["raw_logs_dir"])
    samples = load_pickle(processed_dir / "path_graphs.pkl")
    logs = load_logs(raw_dir)
    labeled = label_samples(samples, logs, config)
    out_path = processed_dir / "samples_labeled.pkl"
    save_pickle(out_path, labeled)
    counts = {label: sum(1 for sample in labeled if sample["risk_label"] == label) for label in [0, 1, 2]}
    print(f"Wrote {len(labeled)} labeled samples to {out_path}; class counts={counts}")


if __name__ == "__main__":
    main()
