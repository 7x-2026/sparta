from __future__ import annotations

from src.analysis.check_raw_log_schema import check_raw_log_schema, write_report
from src.simulation.export_logs import export_logs
from src.simulation.log_schema import LOG_COLUMNS


def test_raw_log_schema_checker_accepts_standard_logs(tmp_path):
    logs = {
        "node_log.csv": [
            {
                "time": 0,
                "node_id": "access_0",
                "node_type": "access",
                "cpu_util": 0.1,
                "mem_util": 0.2,
                "queue_len": 0.0,
                "available_cpu": 0.9,
                "available_mem": 0.8,
            },
            {
                "time": 0,
                "node_id": "edge_0",
                "node_type": "edge",
                "cpu_util": 0.3,
                "mem_util": 0.2,
                "queue_len": 1.0,
                "available_cpu": 0.7,
                "available_mem": 0.8,
            },
        ],
        "link_log.csv": [
            {
                "time": 0,
                "src": "access_0",
                "dst": "edge_0",
                "delay": 10.0,
                "bandwidth": 50.0,
                "loss": 0.001,
                "jitter": 0.2,
                "queue_delay": 1.0,
                "bandwidth_util": 0.3,
            }
        ],
        "service_log.csv": [
            {
                "time": 0,
                "service_id": 0,
                "user_id": 0,
                "service_type": "latency",
                "request_rate": 5.0,
                "response_time": 30.0,
                "current_edge": "edge_0",
                "cloud_node": "cloud_0",
            }
        ],
        "path_log.csv": [
            {
                "time": 0,
                "service_id": 0,
                "path_nodes": "access_0|edge_0",
                "path_links": "access_0-edge_0",
                "path_delay": 10.0,
                "path_loss": 0.001,
                "bottleneck_node": "edge_0",
                "bottleneck_link": "access_0-edge_0",
            }
        ],
        "sla_log.csv": [
            {
                "service_id": 0,
                "service_type": "latency",
                "max_delay": 100.0,
                "max_loss": 0.05,
                "min_bandwidth": 10.0,
                "reliability_req": 0.99,
                "cost_weight": 0.2,
            }
        ],
    }
    export_logs(tmp_path, logs)
    passed, errors = check_raw_log_schema(tmp_path)
    report_path = write_report(tmp_path, passed, errors)
    assert passed, errors
    assert report_path.exists()
    for filename, columns in LOG_COLUMNS.items():
        header = (tmp_path / filename).read_text(encoding="utf-8").splitlines()[0].split(",")
        for column in columns:
            assert column in header
