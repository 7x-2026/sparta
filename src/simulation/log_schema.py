from __future__ import annotations

LOG_SCHEMA_VERSION = "v1"

NODE_LOG_COLUMNS = [
    "time",
    "node_id",
    "node_type",
    "cpu_util",
    "mem_util",
    "queue_len",
    "available_cpu",
    "available_mem",
]

LINK_LOG_COLUMNS = [
    "time",
    "src",
    "dst",
    "delay",
    "bandwidth",
    "loss",
    "jitter",
    "queue_delay",
    "bandwidth_util",
]

SERVICE_LOG_COLUMNS = [
    "time",
    "service_id",
    "user_id",
    "service_type",
    "request_rate",
    "response_time",
    "current_edge",
    "cloud_node",
]

PATH_LOG_COLUMNS = [
    "time",
    "service_id",
    "path_nodes",
    "path_links",
    "path_delay",
    "path_loss",
    "bottleneck_node",
    "bottleneck_link",
]

SLA_LOG_COLUMNS = [
    "service_id",
    "service_type",
    "max_delay",
    "max_loss",
    "min_bandwidth",
    "reliability_req",
    "cost_weight",
]

LOG_COLUMNS = {
    "node_log.csv": NODE_LOG_COLUMNS,
    "link_log.csv": LINK_LOG_COLUMNS,
    "service_log.csv": SERVICE_LOG_COLUMNS,
    "path_log.csv": PATH_LOG_COLUMNS,
    "sla_log.csv": SLA_LOG_COLUMNS,
}

COMMON_EVENT_COLUMNS = [
    "scenario",
    "event_id",
    "severity",
    "affected_node",
    "affected_link",
    "affected_node_position",
    "affected_link_position",
]

MVP_EXTRA_COLUMNS = {
    "node_log.csv": COMMON_EVENT_COLUMNS + ["request_load_on_node", "node_degree"],
    "link_log.csv": COMMON_EVENT_COLUMNS + ["delay_ms", "jitter_ms", "queue_delay_ms"],
    "service_log.csv": COMMON_EVENT_COLUMNS + ["base_response_time", "dependency_count", "last_violation"],
    "path_log.csv": COMMON_EVENT_COLUMNS,
    "sla_log.csv": [],
}

EXPORT_LOG_COLUMNS = {
    filename: columns + [column for column in MVP_EXTRA_COLUMNS.get(filename, []) if column not in columns]
    for filename, columns in LOG_COLUMNS.items()
}

LOG_FILENAMES = list(LOG_COLUMNS)

LINK_ALIAS_PAIRS = [
    ("delay", "delay_ms"),
    ("jitter", "jitter_ms"),
    ("queue_delay", "queue_delay_ms"),
]


def normalize_log_row(filename: str, row: dict) -> dict:
    normalized = dict(row)
    if filename == "link_log.csv":
        for canonical, legacy in LINK_ALIAS_PAIRS:
            if canonical not in normalized and legacy in normalized:
                normalized[canonical] = normalized[legacy]
            if legacy not in normalized and canonical in normalized:
                normalized[legacy] = normalized[canonical]
    return normalized


def export_fieldnames(filename: str, rows: list[dict]) -> list[str]:
    base = list(EXPORT_LOG_COLUMNS.get(filename, []))
    extras = sorted({key for row in rows for key in row if key not in base})
    return base + extras
