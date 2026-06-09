from __future__ import annotations

from dataclasses import dataclass
import math


SCENARIOS = ["normal", "burst", "node_overload", "link_congestion", "mixed"]


@dataclass(frozen=True)
class ScenarioState:
    request_multiplier: float
    node_pressure: float
    link_pressure: float
    noise_scale: float
    hot_edges: tuple[str, ...]
    hot_links: tuple[str, ...]
    target_node_positions: tuple[int, ...]
    target_link_positions: tuple[int, ...]


def _pulse(progress: float, center: float, width: float) -> float:
    return math.exp(-((progress - center) ** 2) / max(width, 1e-6))


def scenario_state(name: str, local_t: int, scenario_steps: int) -> ScenarioState:
    if name not in SCENARIOS:
        raise ValueError(f"Unsupported synthetic scenario: {name}")
    progress = local_t / max(scenario_steps - 1, 1)
    wave = 0.5 + 0.5 * math.sin(2 * math.pi * progress)

    if name == "normal":
        return ScenarioState(
            request_multiplier=0.80 + 0.18 * wave,
            node_pressure=0.10 + 0.06 * wave,
            link_pressure=0.08 + 0.05 * wave,
            noise_scale=0.7,
            hot_edges=("edge_0",),
            hot_links=("edge_0-cloud_0",),
            target_node_positions=(),
            target_link_positions=(),
        )

    if name == "burst":
        pulse = _pulse(progress, 0.50, 0.065)
        return ScenarioState(
            request_multiplier=1.00 + 1.25 * pulse + 0.18 * wave,
            node_pressure=0.18 + 0.20 * pulse,
            link_pressure=0.16 + 0.18 * pulse,
            noise_scale=1.0,
            hot_edges=("edge_1", "edge_3"),
            hot_links=("edge_0-edge_1", "edge_2-edge_3"),
            target_node_positions=(1, 2, 3, 4),
            target_link_positions=(0, 1, 2, 3),
        )

    if name == "node_overload":
        pulse = max(_pulse(progress, 0.36, 0.08), _pulse(progress, 0.70, 0.09))
        return ScenarioState(
            request_multiplier=0.95 + 0.28 * wave,
            node_pressure=0.28 + 0.62 * pulse,
            link_pressure=0.12 + 0.14 * pulse,
            noise_scale=1.0,
            hot_edges=("edge_2", "edge_3"),
            hot_links=("edge_2-cloud_0", "edge_2-edge_3"),
            target_node_positions=(1, 2, 3, 4),
            target_link_positions=(0, 1, 2, 3),
        )

    if name == "link_congestion":
        pulse = max(_pulse(progress, 0.45, 0.09), _pulse(progress, 0.78, 0.06))
        return ScenarioState(
            request_multiplier=0.98 + 0.25 * wave,
            node_pressure=0.16 + 0.12 * pulse,
            link_pressure=0.32 + 0.60 * pulse,
            noise_scale=1.1,
            hot_edges=("edge_0", "edge_4"),
            hot_links=("edge_4-edge_0", "edge_4-cloud_0", "edge_0-cloud_0"),
            target_node_positions=(1, 2, 3, 4),
            target_link_positions=(0, 1, 2, 3),
        )

    pulse = max(_pulse(progress, 0.32, 0.07), _pulse(progress, 0.58, 0.08), _pulse(progress, 0.84, 0.05))
    return ScenarioState(
        request_multiplier=1.08 + 1.00 * pulse + 0.25 * wave,
        node_pressure=0.25 + 0.48 * pulse,
        link_pressure=0.28 + 0.50 * pulse,
        noise_scale=1.2,
        hot_edges=("edge_1", "edge_2", "edge_4"),
        hot_links=("edge_0-edge_1", "edge_2-cloud_0", "edge_4-cloud_0"),
        target_node_positions=(1, 2, 3, 4),
        target_link_positions=(0, 1, 2, 3),
    )
