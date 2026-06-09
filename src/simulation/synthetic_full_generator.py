from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SyntheticEpisode:
    episode_id: int
    scenario_type: str
    start_time: int
    duration: int
    severity: float
    affected_services: tuple[int, ...]
    affected_node: str
    affected_link: str
    affected_node_position: int
    affected_link_position: int

    @property
    def end_time(self) -> int:
        return self.start_time + self.duration


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


TOPOLOGY_NODES, TOPOLOGY_LINKS = build_full_topology()
TOPOLOGY_LINK_IDS = {_link_id(src, dst): (src, dst) for src, dst in TOPOLOGY_LINKS}
TOPOLOGY_LINK_BY_PAIR = {frozenset((src, dst)): _link_id(src, dst) for src, dst in TOPOLOGY_LINKS}

PATH_NODE_CANDIDATES = [
    ["access_0", "edge_0", "cloud_0"],
    ["access_0", "edge_1", "edge_0", "cloud_0"],
    ["access_0", "edge_1", "edge_2", "cloud_0"],
    ["access_0", "edge_0", "edge_4", "cloud_0"],
    ["access_0", "edge_1", "edge_2", "edge_3", "edge_4", "cloud_0"],
    ["access_1", "edge_2", "cloud_0"],
    ["access_1", "edge_3", "edge_2", "cloud_0"],
    ["access_1", "edge_3", "edge_4", "cloud_0"],
    ["access_1", "edge_2", "edge_1", "edge_0", "cloud_0"],
    ["access_1", "edge_2", "edge_1", "edge_0", "edge_4", "cloud_0"],
]


def _canonical_link_id(src: str, dst: str) -> str:
    try:
        return TOPOLOGY_LINK_BY_PAIR[frozenset((src, dst))]
    except KeyError as exc:
        raise ValueError(f"No synthetic topology link between {src} and {dst}") from exc


def _path_for_service(service_id: int, t: int) -> tuple[list[str], list[str], str]:
    slot = t // 17
    idx = (service_id * 7 + (service_id // 3) * 2 + slot * 3) % len(PATH_NODE_CANDIDATES)
    path_nodes = list(PATH_NODE_CANDIDATES[idx])
    path_links = [_canonical_link_id(src, dst) for src, dst in zip(path_nodes[:-1], path_nodes[1:])]
    edge_nodes = [node for node in path_nodes if node.startswith("edge")]
    current_edge = edge_nodes[(service_id + slot) % len(edge_nodes)] if edge_nodes else path_nodes[0]
    return path_nodes, path_links, current_edge


def _sample_split_times(config: dict) -> dict[str, list[int]]:
    data_cfg = config["data"]
    L = int(data_cfg["input_window"])
    H = int(data_cfg["pred_horizon"])
    times = list(range(L - 1, int(data_cfg["num_steps"]) - H))
    train_end = int(len(times) * float(data_cfg["train_ratio"]))
    val_end = int(len(times) * (float(data_cfg["train_ratio"]) + float(data_cfg["val_ratio"])))
    return {"train": times[:train_end], "val": times[train_end:val_end], "test": times[val_end:]}


def _event_choices(scenario: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if scenario == "burst":
        return ("edge_1", "edge_3"), ("edge_0-edge_1", "edge_2-edge_3")
    if scenario == "node_overload":
        return ("edge_2", "edge_3"), ("edge_2-cloud_0", "edge_2-edge_3")
    if scenario == "link_congestion":
        return ("edge_0", "edge_4"), ("edge_4-edge_0", "edge_4-cloud_0", "edge_0-cloud_0")
    if scenario == "mixed":
        return ("edge_1", "edge_2", "edge_4"), ("edge_0-edge_1", "edge_2-cloud_0", "edge_4-cloud_0")
    return ("edge_0",), ("edge_0-cloud_0",)


def _severity_range(scenario: str) -> tuple[float, float]:
    return {
        "burst": (0.34, 0.62),
        "node_overload": (0.36, 0.64),
        "link_congestion": (0.34, 0.62),
        "mixed": (0.38, 0.68),
    }.get(scenario, (0.10, 0.20))


def _cycle_position(positions: tuple[int, ...], cursor: int, max_length: int) -> int:
    valid = [pos for pos in positions if 0 <= pos < max_length]
    if not valid:
        valid = list(range(max_length))
    if not valid:
        return 0
    return valid[cursor % len(valid)]


def _reference_path(
    num_services: int,
    start: int,
    node_position: int,
    link_position: int,
    rng: np.random.Generator,
) -> tuple[int, list[str], list[str]]:
    fallback: tuple[int, list[str], list[str]] | None = None
    positioned_fallback: tuple[int, list[str], list[str]] | None = None
    for service_id in rng.permutation(num_services).tolist():
        path_nodes, path_links, _ = _path_for_service(int(service_id), start)
        if fallback is None or len(path_nodes) + len(path_links) > len(fallback[1]) + len(fallback[2]):
            fallback = (int(service_id), path_nodes, path_links)
        if node_position < len(path_nodes) and link_position < len(path_links):
            candidate = (int(service_id), path_nodes, path_links)
            if positioned_fallback is None:
                positioned_fallback = candidate
            if path_nodes[node_position].startswith("edge"):
                return candidate
    if positioned_fallback is not None:
        return positioned_fallback
    if fallback is None:
        raise ValueError("No service path candidates available")
    return fallback


def _services_for_event(
    num_services: int,
    start: int,
    scenario: str,
    affected_node: str,
    affected_link: str,
    target_node_position: int,
    target_link_position: int,
    target_size: int,
    reference_service: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    exact_matches: list[int] = []
    strong_matches: list[int] = []
    weak_matches: list[int] = []
    for service_id in rng.permutation(num_services).tolist():
        path_nodes, path_links, _ = _path_for_service(int(service_id), start)
        node_match = affected_node in path_nodes
        link_match = affected_link in path_links
        node_position = path_nodes.index(affected_node) if node_match else -1
        link_position = path_links.index(affected_link) if link_match else -1
        if scenario == "node_overload":
            strong = node_match
            exact = node_position == target_node_position
        elif scenario == "link_congestion":
            strong = link_match
            exact = link_position == target_link_position
        else:
            strong = node_match or link_match
            exact = node_position == target_node_position or link_position == target_link_position
        weak = node_match or link_match
        if exact:
            exact_matches.append(int(service_id))
        elif strong:
            strong_matches.append(int(service_id))
        elif weak:
            weak_matches.append(int(service_id))

    chosen: list[int] = []
    for service_id in [reference_service] + exact_matches + strong_matches + weak_matches:
        if service_id not in chosen:
            chosen.append(service_id)
        if len(chosen) >= target_size:
            break
    if not chosen:
        chosen = [reference_service]
    return tuple(sorted(chosen))


def generate_event_schedule(config: dict, rng: np.random.Generator, severity_scale: float) -> tuple[dict[tuple[int, int], SyntheticEpisode], list[SyntheticEpisode]]:
    data_cfg = config["data"]
    scenarios = [scenario for scenario in config["simulation"].get("scenarios", SCENARIOS) if scenario in SCENARIOS]
    non_normal = [scenario for scenario in scenarios if scenario != "normal"]
    num_services = int(data_cfg["num_services"])
    split_times = _sample_split_times(config)
    event_grid: dict[tuple[int, int], SyntheticEpisode] = {}
    episodes: list[SyntheticEpisode] = []
    episode_id = 0
    target_labels = config["simulation"].get("target_label_ratio", {})
    normal_target = min(0.36, max(0.30, float(target_labels.get("normal", 0.45)) - 0.10))
    per_event_target = (1.0 - normal_target) / max(len(non_normal), 1)
    node_position_cursor: Counter[str] = Counter()
    link_position_cursor: Counter[str] = Counter()

    for _, times in split_times.items():
        if not times:
            continue
        split_time_set = set(times)
        total_cells = len(times) * num_services
        service_group_size = max(1, int(round(num_services * 0.20)))
        min_duration = max(3, min(8, len(times) // 10))
        max_duration = max(min_duration, min(24, max(4, len(times) // 4)))
        for scenario in non_normal:
            target_cells = int(total_cells * per_event_target)
            attempts = 0
            while attempts < 3000:
                current = sum(1 for (t, _), episode in event_grid.items() if t in split_time_set and episode.scenario_type == scenario)
                if current >= target_cells:
                    break
                duration = int(rng.integers(min_duration, max_duration + 1))
                start = int(rng.choice(times))
                start = min(start, times[-1])
                template = scenario_state(scenario, 0, 1)
                node_position = _cycle_position(
                    template.target_node_positions or (1, 2, 3, 4),
                    node_position_cursor[scenario],
                    int(data_cfg.get("max_nodes", 8)),
                )
                link_position = _cycle_position(
                    template.target_link_positions or (0, 1, 2, 3),
                    link_position_cursor[scenario],
                    int(data_cfg.get("max_links", 12)),
                )
                reference_service, path_nodes, path_links = _reference_path(num_services, start, node_position, link_position, rng)
                node_position = min(node_position, len(path_nodes) - 1)
                link_position = min(link_position, len(path_links) - 1)
                affected_node = path_nodes[node_position]
                affected_link = path_links[link_position]
                services = _services_for_event(
                    num_services,
                    start,
                    scenario,
                    affected_node,
                    affected_link,
                    int(node_position),
                    int(link_position),
                    min(service_group_size, num_services),
                    reference_service,
                    rng,
                )
                severity_lo, severity_hi = _severity_range(scenario)
                episode = SyntheticEpisode(
                    episode_id=episode_id,
                    scenario_type=scenario,
                    start_time=start,
                    duration=duration,
                    severity=float(rng.uniform(severity_lo, severity_hi) * severity_scale),
                    affected_services=services,
                    affected_node=affected_node,
                    affected_link=affected_link,
                    affected_node_position=int(node_position),
                    affected_link_position=int(link_position),
                )
                filled = 0
                for t in range(start, min(start + duration, times[-1] + 1)):
                    if t not in split_time_set:
                        continue
                    for service_id in services:
                        key = (t, int(service_id))
                        if key not in event_grid:
                            event_grid[key] = episode
                            filled += 1
                if filled:
                    episodes.append(episode)
                    episode_id += 1
                    node_position_cursor[scenario] += 1
                    link_position_cursor[scenario] += 1
                attempts += 1
    return event_grid, episodes


def _episode_state(episode: SyntheticEpisode | None, t: int):
    if episode is None:
        return scenario_state("normal", t % 24, 24), "normal", 0.0, "", "", -1
    local_t = max(0, t - episode.start_time)
    state = scenario_state(episode.scenario_type, local_t, max(episode.duration, 1))
    return state, episode.scenario_type, episode.severity, episode.affected_node, episode.affected_link, episode.episode_id


def _dominant_node_events(episodes_by_time: dict[int, list[SyntheticEpisode]], t: int) -> dict[str, SyntheticEpisode]:
    out: dict[str, SyntheticEpisode] = {}
    for episode in episodes_by_time.get(t, []):
        prev = out.get(episode.affected_node)
        if prev is None or episode.severity > prev.severity:
            out[episode.affected_node] = episode
    return out


def _dominant_link_events(episodes_by_time: dict[int, list[SyntheticEpisode]], t: int) -> dict[str, SyntheticEpisode]:
    out: dict[str, SyntheticEpisode] = {}
    for episode in episodes_by_time.get(t, []):
        prev = out.get(episode.affected_link)
        if prev is None or episode.severity > prev.severity:
            out[episode.affected_link] = episode
    return out


def _build_indexes(logs: dict[str, list[dict]]) -> dict:
    all_nodes = {row["node_id"] for row in logs["node_log.csv"]}
    all_links = {_link_id(row["src"], row["dst"]) for row in logs["link_log.csv"]}
    return {
        "service": {(int(row["time"]), int(row["service_id"])): row for row in logs["service_log.csv"]},
        "path": {(int(row["time"]), int(row["service_id"])): row for row in logs["path_log.csv"]},
        "node": {(int(row["time"]), row["node_id"]): row for row in logs["node_log.csv"]},
        "link": {(int(row["time"]), f"{row['src']}-{row['dst']}"): row for row in logs["link_log.csv"]},
        "sla": {int(row["service_id"]): row for row in logs["sla_log.csv"]},
        "all_nodes": all_nodes,
        "all_links": all_links,
        "link_pairs": {link_id: pair for link_id, pair in TOPOLOGY_LINK_IDS.items() if link_id in all_links},
    }


def _split_pipe(value: str) -> list[str]:
    return [part for part in value.split("|") if part]


def _risk_label_for(indexes: dict, config: dict, time: int, service_id: int) -> int:
    H = int(config["data"]["pred_horizon"])
    num_steps = int(config["data"]["num_steps"])
    queue_cap = float(config["simulation"].get("queue_cap", 50.0))
    sla = indexes["sla"][service_id]
    max_delay = float(sla["max_delay"])
    max_loss = float(sla["max_loss"])
    violated = False
    risky = False
    for ft in range(time + 1, min(time + H, num_steps - 1) + 1):
        service = indexes["service"].get((ft, service_id))
        path = indexes["path"].get((ft, service_id))
        if service is None or path is None:
            continue
        response = float(service["response_time"])
        loss = float(path["path_loss"])
        if response >= max_delay or loss >= max_loss:
            violated = True
        if response >= 0.75 * max_delay or loss >= 0.75 * max_loss:
            risky = True
        for node in _split_pipe(path["path_nodes"]):
            row = indexes["node"].get((ft, node))
            if row and (float(row["cpu_util"]) >= 0.85 or float(row["queue_len"]) / queue_cap >= 0.85):
                risky = True
        for link in _split_pipe(path["path_links"]):
            row = indexes["link"].get((ft, link))
            if row and float(row["bandwidth_util"]) >= 0.85:
                risky = True
    if violated:
        return 2
    if risky:
        return 1
    return 0


def _validation_node_score(row: dict | None, queue_cap: float) -> float:
    if row is None:
        return -1.0
    cpu = float(row["cpu_util"])
    queue = float(row["queue_len"]) / max(queue_cap, 1.0)
    available = float(row["available_cpu"])
    return 0.4 * min(cpu, 1.0) + 0.3 * min(queue, 1.0) + 0.3 * min(1.0 - available, 1.0)


def _validation_link_score(row: dict | None, delay_cap: float) -> float:
    if row is None:
        return -1.0
    return (
        0.4 * min(float(row["delay_ms"]) / max(delay_cap, 1.0), 1.0)
        + 0.3 * min(float(row["loss"]), 1.0)
        + 0.3 * min(float(row["bandwidth_util"]), 1.0)
    )


def _candidate_node_ids(path_row: dict, indexes: dict, max_nodes: int) -> list[str]:
    chosen: list[str] = []
    for node in _split_pipe(path_row.get("path_nodes", "")) + [path_row.get("bottleneck_node", "")]:
        if node and node not in chosen:
            chosen.append(node)
    for node in sorted(node for node in indexes["all_nodes"] if str(node).startswith("edge")):
        if len(chosen) >= max_nodes:
            break
        if node not in chosen:
            chosen.append(node)
    return chosen[:max_nodes]


def _candidate_link_ids(path_row: dict, indexes: dict, node_ids: list[str], max_links: int) -> list[str]:
    chosen: list[str] = []
    for link in _split_pipe(path_row.get("path_links", "")) + [path_row.get("bottleneck_link", "")]:
        if link and link not in chosen:
            chosen.append(link)
    node_set = set(node_ids)
    for link_id in sorted(indexes["all_links"]):
        if len(chosen) >= max_links:
            break
        src, dst = indexes["link_pairs"].get(link_id, ("", ""))
        if src in node_set and dst in node_set and link_id not in chosen:
            chosen.append(link_id)
    return chosen[:max_links]


def _validation_attribution(indexes: dict, config: dict, time: int, service_id: int) -> tuple[int, int]:
    data_cfg = config["data"]
    sim_cfg = config["simulation"]
    H = int(data_cfg["pred_horizon"])
    num_steps = int(data_cfg["num_steps"])
    current_path = indexes["path"].get((time, service_id))
    if current_path is None:
        return 0, 0
    node_ids = _candidate_node_ids(current_path, indexes, int(data_cfg.get("max_nodes", 8)))
    link_ids = _candidate_link_ids(current_path, indexes, node_ids, int(data_cfg.get("max_links", 12)))
    node_scores = {node: -1.0 for node in node_ids}
    link_scores = {link: -1.0 for link in link_ids}
    queue_cap = float(sim_cfg.get("queue_cap", 50.0))
    delay_cap = float(sim_cfg.get("delay_cap_ms", 300.0))
    for ft in range(time + 1, min(time + H, num_steps - 1) + 1):
        future_path = indexes["path"].get((ft, service_id))
        if future_path is None:
            continue
        for node in _split_pipe(future_path.get("path_nodes", "")):
            if node in node_scores:
                node_scores[node] = max(node_scores[node], _validation_node_score(indexes["node"].get((ft, node)), queue_cap))
        for link in _split_pipe(future_path.get("path_links", "")):
            if link in link_scores:
                link_scores[link] = max(link_scores[link], _validation_link_score(indexes["link"].get((ft, link)), delay_cap))
    best_node = max(node_scores, key=node_scores.get) if node_scores else ""
    best_link = max(link_scores, key=link_scores.get) if link_scores else ""
    risk_node = node_ids.index(best_node) if best_node in node_ids else 0
    risk_link = link_ids.index(best_link) if best_link in link_ids else 0
    return risk_node, risk_link


def _validate_logs(logs: dict[str, list[dict]], config: dict) -> tuple[bool, str]:
    indexes = _build_indexes(logs)
    data_cfg = config["data"]
    sim_cfg = config["simulation"]
    num_services = int(data_cfg["num_services"])
    split_check = sim_cfg.get("split_balance_check", {})
    split_check_enabled = bool(split_check.get("enabled", True))
    max_scenario_ratio = float(split_check.get("max_scenario_ratio_per_split", sim_cfg.get("max_scenario_ratio_per_split", 0.40)))
    max_node_attr_ratio = float(split_check.get("max_top_node_attr_ratio", sim_cfg.get("max_attribution_majority_ratio", 0.50)))
    max_link_attr_ratio = float(split_check.get("max_top_link_attr_ratio", sim_cfg.get("max_attribution_majority_ratio", 0.50)))
    max_label_ratio_gap = float(split_check.get("max_label_ratio_gap", 1.0))
    allowed_label_ranges = sim_cfg.get("allowed_label_ratio_range", {})
    label_names = {0: "normal", 1: "risky", 2: "violated"}
    split_label_ratios: dict[str, dict[str, float]] = {}
    split_times = _sample_split_times(config)
    for split, times in split_times.items():
        label_counts = Counter()
        scenario_counts = Counter()
        node_attr_counts = Counter()
        link_attr_counts = Counter()
        for t in times:
            for service_id in range(num_services):
                label = _risk_label_for(indexes, config, t, service_id)
                label_counts[label] += 1
                if label > 0:
                    risk_node, risk_link = _validation_attribution(indexes, config, t, service_id)
                    node_attr_counts[risk_node] += 1
                    link_attr_counts[risk_link] += 1
                path = indexes["path"].get((t, service_id))
                scenario_counts[path.get("scenario", "unknown") if path else "unknown"] += 1
        total = max(sum(label_counts.values()), 1)
        ratios = {label_names[label_id]: label_counts[label_id] / total for label_id in label_names}
        split_label_ratios[split] = ratios
        top_scenario, top_count = scenario_counts.most_common(1)[0]
        top_ratio = top_count / total
        missing = [scenario for scenario in SCENARIOS if scenario_counts.get(scenario, 0) == 0]
        for label_name, ratio in ratios.items():
            if label_name in allowed_label_ranges:
                low, high = allowed_label_ranges[label_name]
                if ratio < float(low) or ratio > float(high):
                    return False, f"{split} {label_name} ratio {ratio:.3f} outside [{float(low):.3f}, {float(high):.3f}]"
        if split_check_enabled and missing:
            return False, f"{split} missing scenarios {missing}"
        if split_check_enabled and top_ratio > max_scenario_ratio:
            return False, f"{split} top scenario {top_scenario} ratio {top_ratio:.3f} > {max_scenario_ratio:.3f}"
        for field, counts, max_attr_ratio in [
            ("risk_node", node_attr_counts, max_node_attr_ratio),
            ("risk_link", link_attr_counts, max_link_attr_ratio),
        ]:
            attr_total = sum(counts.values())
            if not attr_total:
                continue
            top_value, top_count = counts.most_common(1)[0]
            top_ratio = top_count / attr_total
            if split_check_enabled and top_ratio > max_attr_ratio:
                return False, f"{split} top {field} {top_value} ratio {top_ratio:.3f} > {max_attr_ratio:.3f}"
    if split_check_enabled:
        for label_name in label_names.values():
            values = [ratios.get(label_name, 0.0) for ratios in split_label_ratios.values()]
            if values:
                gap = max(values) - min(values)
                if gap > max_label_ratio_gap:
                    return False, f"{label_name} ratio gap {gap:.3f} > {max_label_ratio_gap:.3f}"
    return True, "ok"


def _generate_once(config: dict, seed: int, severity_scale: float) -> tuple[dict[str, list[dict]], str]:
    data_cfg = config["data"]
    sim_cfg = config["simulation"]
    total_steps = int(data_cfg["num_steps"])
    num_services = int(data_cfg["num_services"])
    rng = np.random.default_rng(seed)
    nodes, links = build_full_topology()
    if int(data_cfg.get("num_nodes", len(nodes))) != len(nodes):
        raise ValueError("synthetic full v0 uses exactly 8 nodes")
    if int(data_cfg.get("num_links", len(links))) != len(links):
        raise ValueError("synthetic full v0 uses exactly 12 links")

    event_grid, episodes = generate_event_schedule(config, rng, severity_scale)
    episodes_by_time: dict[int, list[SyntheticEpisode]] = defaultdict(list)
    for episode in episodes:
        for t in range(episode.start_time, min(episode.end_time, total_steps)):
            episodes_by_time[t].append(episode)

    degree = {node: 0 for node in nodes}
    for src, dst in links:
        degree[src] += 1
        degree[dst] += 1

    node_log: list[dict] = []
    link_log: list[dict] = []
    service_log: list[dict] = []
    path_log: list[dict] = []
    last_violation = {service_id: 0 for service_id in range(num_services)}

    for t in range(total_steps):
        node_events = _dominant_node_events(episodes_by_time, t)
        link_events = _dominant_link_events(episodes_by_time, t)
        node_state: dict[str, dict] = {}
        link_state: dict[str, dict] = {}

        for node in nodes:
            event = node_events.get(node)
            state, scenario, severity, affected_node, affected_link, event_id = _episode_state(event, t)
            ntype = _node_type(node)
            hot = 1.0 if event is not None else 0.0
            if ntype == "edge":
                base_cpu = 0.28 + 0.03 * (int(node.split("_")[1]) % 3)
                cpu = np.clip(
                    base_cpu
                    + 0.42 * state.node_pressure * severity * hot
                    + 0.08 * severity * hot
                    + 0.08 * state.request_multiplier
                    + rng.normal(0, 0.020 * state.noise_scale),
                    0.04,
                    0.98,
                )
                mem = np.clip(0.28 + 0.48 * cpu + rng.normal(0, 0.016 * state.noise_scale), 0.04, 0.96)
                req_load = np.clip(0.28 + 0.45 * state.request_multiplier + 0.18 * severity * hot, 0.0, 3.0)
                queue = max(
                    0.0,
                    (cpu - 0.48) * float(sim_cfg.get("queue_cap", 50.0)) * 1.05
                    + 18.0 * state.node_pressure * severity * hot
                    + 6.0 * severity * hot
                    + (32.0 * severity * hot if scenario == "node_overload" else 10.0 * severity * hot if scenario == "mixed" else 0.0)
                    + rng.normal(0, 0.8),
                )
            elif ntype == "cloud":
                cpu = np.clip(0.28 + 0.12 * state.request_multiplier + 0.20 * state.node_pressure * severity * hot + rng.normal(0, 0.010), 0.04, 0.90)
                mem = np.clip(0.36 + 0.12 * state.request_multiplier + rng.normal(0, 0.010), 0.04, 0.82)
                req_load = np.clip(0.36 + 0.20 * state.request_multiplier, 0.0, 2.0)
                queue = max(0.0, (cpu - 0.58) * float(sim_cfg.get("queue_cap", 50.0)) * 0.45 + 8.0 * state.node_pressure * severity * hot)
            else:
                cpu = np.clip(0.15 + 0.08 * state.request_multiplier + 0.16 * state.node_pressure * severity * hot + rng.normal(0, 0.009), 0.02, 0.72)
                mem = np.clip(0.20 + 0.06 * state.request_multiplier + rng.normal(0, 0.009), 0.02, 0.58)
                req_load = np.clip(0.16 + 0.16 * state.request_multiplier, 0.0, 1.3)
                queue = max(0.0, (cpu - 0.40) * float(sim_cfg.get("queue_cap", 50.0)) * 0.28 + 5.0 * state.node_pressure * severity * hot)

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
                    "event_id": event_id,
                    "severity": round(severity, 5),
                    "affected_node": affected_node,
                    "affected_link": affected_link,
                    "affected_node_position": event.affected_node_position if event is not None else -1,
                    "affected_link_position": event.affected_link_position if event is not None else -1,
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
            event = link_events.get(lid)
            state, scenario, severity, affected_node, affected_link, event_id = _episode_state(event, t)
            hot = 1.0 if event is not None else 0.0
            cloud_link = 1.0 if "cloud_0" in (src, dst) else 0.0
            base_delay = 7.0 + 20.0 * cloud_link
            base_bw = 62.0 + 22.0 * cloud_link
            bandwidth_util = float(np.clip(0.22 + 0.18 * state.request_multiplier + 0.44 * state.link_pressure * severity * hot + rng.normal(0, 0.025 * state.noise_scale), 0.02, 0.98))
            delay_ms = float(max(1.0, base_delay * (1.0 + 1.10 * bandwidth_util) + 12.0 * state.link_pressure * severity * hot + rng.normal(0, 0.9)))
            loss = float(np.clip(0.002 + 0.018 * max(0.0, bandwidth_util - 0.62) + 0.036 * state.link_pressure * severity * hot + rng.normal(0, 0.0015), 0.0, 0.25))
            jitter_ms = float(max(0.0, delay_ms * (0.035 + 0.060 * bandwidth_util) + rng.normal(0, 0.18)))
            queue_delay_ms = float(max(0.0, delay_ms * bandwidth_util * (0.21 + 0.12 * hot)))
            bandwidth = float(max(1.0, base_bw * (1.0 - 0.44 * bandwidth_util)))
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
                    "event_id": event_id,
                    "severity": round(severity, 5),
                    "affected_node": affected_node,
                    "affected_link": affected_link,
                    "affected_node_position": event.affected_node_position if event is not None else -1,
                    "affected_link_position": event.affected_link_position if event is not None else -1,
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
            episode = event_grid.get((t, service_id))
            state, scenario, severity, affected_node, affected_link, event_id = _episode_state(episode, t)
            service_type = SERVICE_TYPES[service_id % len(SERVICE_TYPES)]
            path_nodes, path_links, current_edge = _path_for_service(service_id, t)
            path_link_state = [link_state[link] for link in path_links]
            path_delay = sum(row["delay_ms"] for row in path_link_state)
            path_loss_survival = 1.0
            for row in path_link_state:
                path_loss_survival *= 1.0 - row["loss"]
            path_loss = 1.0 - path_loss_survival
            base_rate = 7.0 + (service_id % 11) * 1.8
            service_wave = 1.0 + 0.12 * np.sin(2 * np.pi * (t + service_id) / 60.0)
            request_rate = float(max(0.5, base_rate * state.request_multiplier * service_wave + rng.normal(0, 0.45 * state.noise_scale)))
            base_response = {"latency": 34.0, "reliability": 54.0, "cost": 70.0}[service_type] + (service_id % 5) * 1.5
            max_cpu = max(node_state[node]["cpu_util"] for node in path_nodes)
            max_queue = max(node_state[node]["queue_len"] for node in path_nodes)
            response_time = float(
                np.clip(
                    base_response
                    + 0.36 * path_delay
                    + 0.45 * request_rate
                    + 0.78 * max_queue
                    + 26.0 * max(0.0, max_cpu - 0.76)
                    + 10.0 * severity * (1.0 if scenario in {"burst", "mixed"} else 0.0)
                    + rng.normal(0, 1.8 * state.noise_scale),
                    1.0,
                    float(sim_cfg.get("response_cap_ms", 420.0)),
                )
            )
            sla = SERVICE_SLA[service_type]
            violated = int(response_time >= sla["max_delay"] or path_loss >= sla["max_loss"])
            bottleneck_node = max(path_nodes, key=lambda node: node_state[node]["cpu_util"] + node_state[node]["queue_len"] / float(sim_cfg.get("queue_cap", 50.0)))
            bottleneck_link = max(path_links, key=lambda link: link_state[link]["bandwidth_util"] + 4.0 * link_state[link]["loss"])
            common = {
                "time": t,
                "scenario": scenario,
                "event_id": event_id,
                "severity": round(severity, 5),
                "affected_node": affected_node,
                "affected_link": affected_link,
                "affected_node_position": episode.affected_node_position if episode is not None else -1,
                "affected_link_position": episode.affected_link_position if episode is not None else -1,
                "service_id": service_id,
            }
            service_log.append(
                {
                    **common,
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
                    **common,
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
        sla_log.append({"service_id": service_id, "service_type": service_type, **sla})

    return {
        "node_log.csv": node_log,
        "link_log.csv": link_log,
        "service_log.csv": service_log,
        "path_log.csv": path_log,
        "sla_log.csv": sla_log,
    }, f"episodes={len(episodes)}, seed={seed}, severity_scale={severity_scale:.3f}"


def generate_synthetic_full_logs(config: dict) -> dict[str, list[dict]]:
    base_seed = int(config.get("seed", 42))
    retry_cfg = config.get("simulation", {}).get("generation_retry", {})
    max_retries = int(retry_cfg.get("max_retries", 20)) if bool(retry_cfg.get("enabled", True)) else 1
    last_reason = ""
    for attempt in range(max_retries):
        severity_scale = max(0.35, 1.0 - 0.035 * attempt)
        logs, summary = _generate_once(config, base_seed + attempt, severity_scale)
        ok, reason = _validate_logs(logs, config)
        print(f"synthetic full attempt {attempt + 1}/{max_retries}: {summary}; validation={reason}")
        if ok:
            return logs
        last_reason = reason
    raise RuntimeError(f"Failed to generate valid synthetic full logs after {max_retries} attempts: {last_reason}")


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
