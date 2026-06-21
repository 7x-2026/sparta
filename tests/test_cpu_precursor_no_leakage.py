from __future__ import annotations

from pathlib import Path

from src.preprocessing.build_path_graph import build_sample, load_logs
from src.simulation.export_logs import export_logs
from src.simulation.risk_injection import inject_cpu_precursor_v1

from tests.test_cpu_precursor_generation import make_config, make_logs


NODE_FEATURE_NAMES = [
    "cpu_util",
    "mem_util",
    "queue_len",
    "available_cpu",
    "available_mem",
    "node_type",
    "node_degree",
    "is_current_edge",
    "cpu_util_slope",
    "available_cpu_slope",
    "cpu_pressure_persistence",
    "colocated_service_count",
    "colocated_request_rate_sum",
    "colocated_request_rate_slope",
]
SERVICE_FEATURE_NAMES = [
    "request_rate",
    "response_time",
    "service_type",
    "base_response_time",
    "dependency_count",
    "request_rate_slope",
    "response_time_slope",
    "path_cpu_pressure_mean",
    "path_cpu_pressure_max",
    "path_cpu_pressure_slope",
    "path_available_cpu_min",
    "path_queue_len_slope",
]


def graph_config(raw_dir: Path) -> dict:
    config = make_config()
    config["data"].update(
        {
            "raw_logs_dir": str(raw_dir),
            "processed_dir": str(raw_dir.parent / "processed"),
            "dataset_dir": str(raw_dir.parent / "dataset"),
            "max_nodes": 4,
            "max_links": 4,
            "node_feat_dim": len(NODE_FEATURE_NAMES),
            "link_feat_dim": 8,
            "service_feat_dim": len(SERVICE_FEATURE_NAMES),
            "sla_feat_dim": 5,
            "node_feature_names": NODE_FEATURE_NAMES,
            "service_feature_names": SERVICE_FEATURE_NAMES,
            "link_feature_names": [
                "delay_ms",
                "bandwidth",
                "loss",
                "jitter_ms",
                "queue_delay_ms",
                "bandwidth_util",
                "is_current_link",
                "hop_position",
            ],
            "sla_feature_names": ["max_delay", "max_loss", "min_bandwidth", "reliability_req", "cost_weight"],
        }
    )
    return config


def test_cpu_precursor_features_do_not_include_future_label_fields(tmp_path: Path):
    raw_dir = tmp_path / "raw_logs"
    logs = make_logs()
    inject_cpu_precursor_v1(logs, make_config(), artifacts_dir=tmp_path / "artifacts")
    export_logs(raw_dir, logs)
    config = graph_config(raw_dir)
    loaded_logs = load_logs(raw_dir)

    sample = build_sample(service_id=0, time=10, logs=loaded_logs, config=config)

    assert sample["node_x"].shape == (6, 4, 14)
    assert sample["service_x"].shape == (6, 12)
    forbidden = ("risk_metric", "risk_metric_scores", "label_score", "future", "horizon")
    all_feature_names = NODE_FEATURE_NAMES + SERVICE_FEATURE_NAMES
    assert not any(token in name for name in all_feature_names for token in forbidden)
    assert "risk_metric_scores" not in sample
    assert "risk_metric" not in sample


def test_cpu_precursor_manifest_metadata_preserves_strict_backend_semantics(tmp_path: Path):
    logs = make_logs()
    metadata = inject_cpu_precursor_v1(logs, make_config(), artifacts_dir=tmp_path)

    assert metadata["risk_injection"] == "cpu_precursor_v1"
    assert metadata["cpu_precursor_enabled"] is True
    assert metadata["cpu_precursor_episode_count"] > 0
