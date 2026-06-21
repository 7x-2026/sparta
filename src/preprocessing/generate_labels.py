from __future__ import annotations

import argparse
from collections import Counter
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


def _dominant_scenario(counts: Counter, fallback: str) -> str:
    if counts:
        return str(counts.most_common(1)[0][0])
    return fallback or "unknown"


def _choose_risk_metric(
    metric_scores: np.ndarray,
    scenario: str,
    queue_abs_score: float,
    queue_growth_score: float,
    queue_trend: float,
    request_trend: float,
) -> int:
    if scenario == "node_overload":
        if metric_scores[3] >= 0.72 and metric_scores[3] >= 0.96 * metric_scores[2]:
            return 3
        if metric_scores[2] >= 0.68:
            return 2

    if scenario == "burst":
        if request_trend >= 0.45 and queue_trend >= 0.45 and max(metric_scores[3], queue_growth_score) >= 0.70:
            return 3
        adjusted = metric_scores.copy()
        adjusted[3] *= 0.85
        return int(adjusted.argmax())

    if scenario == "link_congestion":
        link_metric_scores = {
            0: float(metric_scores[0]),
            1: float(metric_scores[1]),
            4: float(metric_scores[4] + 0.05 * max(0.0, metric_scores[4] - 0.90)),
        }
        return max(link_metric_scores, key=link_metric_scores.get)

    return int(metric_scores.argmax())


def _config_horizon(config: dict) -> int:
    data_cfg = config.get("data", {})
    return int(data_cfg.get("pred_horizon", data_cfg.get("horizon", 5)))


def _config_num_steps(config: dict, logs: dict) -> int:
    data_cfg = config.get("data", {})
    if data_cfg.get("num_steps") is not None:
        return int(data_cfg["num_steps"])
    edge_cfg = config.get("edgesimpy", {})
    if edge_cfg.get("num_steps") is not None:
        return int(edge_cfg["num_steps"])
    times = [_i(row, "time") for row in logs.get("rows", {}).get("path", [])]
    return max(times, default=0) + 1


def compute_metric_risk_scores(sample_or_window, logs: dict | None = None, config: dict | None = None) -> dict:
    """Return the exact five metric scores used by risk_metric labeling.

    When called with a dataset sample plus raw logs this uses the future horizon
    and is therefore diagnostic only, not valid model input.
    """
    if logs is None:
        if isinstance(sample_or_window, dict):
            for key in ("metric_rule_scores", "labelrule_metric_scores"):
                if key in sample_or_window:
                    return {
                        "scores": sample_or_window[key],
                        "uses_future_information": False,
                        "allowed_for_model_input": True,
                    }
        raise ValueError(
            "labelrule metric evidence requires raw logs/config or a precomputed metric_rule_scores tensor."
        )
    if config is None:
        raise ValueError("config is required when computing label-rule scores from raw logs.")

    sample = sample_or_window
    sim_cfg = config.get("simulation", {})
    service_id = int(sample["service_id"])
    time = int(sample["time"])
    horizon = _config_horizon(config)
    num_steps = _config_num_steps(config, logs)
    metric_cap = float(sim_cfg.get("metric_score_cap", 1.60))
    queue_metric_cap = float(sim_cfg.get("queue_metric_cap", 28.0))
    queue_growth_cap = float(sim_cfg.get("queue_growth_cap", 8.0))
    queue_step_threshold = float(sim_cfg.get("queue_growth_step_threshold", 0.50))
    request_step_threshold = float(sim_cfg.get("request_growth_step_threshold", 0.20))
    future = _future_times(time, horizon, num_steps)
    metric_scores = np.zeros((5,), dtype=np.float32)
    sla = logs["idx"]["sla"][service_id]
    max_delay = max(_f(sla, "max_delay"), 1e-6)
    max_loss = max(_f(sla, "max_loss"), 1e-6)
    scenario_counts: Counter[str] = Counter()
    queue_values: list[float] = []
    max_queue_growth = 0.0
    queue_increase_steps = 0
    request_increase_steps = 0
    prev_queue: float | None = None
    prev_request: float | None = None

    for ft in future:
        service = logs["idx"]["service"].get((ft, service_id))
        path = logs["idx"]["path"].get((ft, service_id))
        if service is None or path is None:
            continue
        scenario_counts[path.get("scenario", sample.get("scenario", "unknown"))] += 1
        path_nodes = _split_pipe(path.get("path_nodes", ""))
        path_links = _split_pipe(path.get("path_links", ""))
        metric_scores[0] = max(metric_scores[0], min(_f(service, "response_time") / max_delay, metric_cap))
        metric_scores[1] = max(metric_scores[1], min(_f(path, "path_loss") / max_loss, metric_cap))
        path_max_queue = 0.0
        for node in path_nodes:
            row = logs["idx"]["node"].get((ft, node))
            if row:
                cpu_pressure = max(_f(row, "cpu_util"), 1.0 - _f(row, "available_cpu", 1.0))
                metric_scores[2] = max(metric_scores[2], min(cpu_pressure / 0.85, metric_cap))
                path_max_queue = max(path_max_queue, _f(row, "queue_len"))
        queue_values.append(path_max_queue)
        if prev_queue is not None:
            growth = path_max_queue - prev_queue
            max_queue_growth = max(max_queue_growth, growth)
            if growth > queue_step_threshold:
                queue_increase_steps += 1
        prev_queue = path_max_queue
        request = _f(service, "request_rate")
        if prev_request is not None and request > prev_request + request_step_threshold:
            request_increase_steps += 1
        prev_request = request
        for link in path_links:
            row = logs["idx"]["link"].get((ft, link))
            if row:
                metric_scores[4] = max(metric_scores[4], min(_f(row, "bandwidth_util") / 0.85, metric_cap))

    denom = max(len(queue_values) - 1, 1)
    queue_abs_score = max(queue_values, default=0.0) / max(queue_metric_cap, 1e-6)
    queue_growth_score = max_queue_growth / max(queue_growth_cap, 1e-6)
    queue_trend = queue_increase_steps / denom
    request_trend = request_increase_steps / denom
    metric_scores[3] = min(max(queue_abs_score, 0.82 * queue_abs_score + 0.18 * queue_trend), metric_cap)
    scenario = _dominant_scenario(scenario_counts, str(sample.get("scenario", "unknown")))
    label_rule_argmax = _choose_risk_metric(metric_scores, scenario, queue_abs_score, queue_growth_score, queue_trend, request_trend)
    return {
        "scores": metric_scores,
        "scenario_used_by_label": scenario,
        "queue_abs_score": float(queue_abs_score),
        "queue_growth_score": float(queue_growth_score),
        "queue_trend": float(queue_trend),
        "request_trend": float(request_trend),
        "label_rule_argmax": int(label_rule_argmax),
        "uses_future_information": True,
        "allowed_for_model_input": False,
    }


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

    for ft in future:
        service = logs["idx"]["service"].get((ft, service_id))
        path = logs["idx"]["path"].get((ft, service_id))
        if service is None or path is None:
            continue
        path_nodes = _split_pipe(path.get("path_nodes", ""))
        path_links = _split_pipe(path.get("path_links", ""))
        for node in path_nodes:
            row = logs["idx"]["node"].get((ft, node))
            score = _node_score(row, queue_cap)
            if node in node_scores:
                node_scores[node] = max(node_scores[node], score)
        for link in path_links:
            row = logs["idx"]["link"].get((ft, link))
            score = _link_score(row, delay_cap)
            if link in link_scores:
                link_scores[link] = max(link_scores[link], score)

    best_node = max(node_scores, key=node_scores.get)
    best_link = max(link_scores, key=link_scores.get)
    risk_node = sample["node_ids"].index(best_node)
    risk_link = sample["link_ids"].index(best_link)
    metric_info = compute_metric_risk_scores(sample, logs, config)
    risk_metric = int(metric_info["label_rule_argmax"])
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
        metric_info = compute_metric_risk_scores(sample, logs, config)
        sample["risk_metric_scores"] = [float(value) for value in metric_info["scores"]]
        sample["risk_metric_scores_use_future_information"] = bool(metric_info["uses_future_information"])
        sample["risk_metric_scores_used_as_supervision_only"] = True
        sample["risk_metric_scores_used_as_model_input"] = False
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
