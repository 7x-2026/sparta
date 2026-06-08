from __future__ import annotations

import csv

from src.utils.io import load_pickle

from .conftest import ROOT, run_cmd


REQUIRED_COLUMNS = {
    "node_log.csv": {
        "time",
        "node_id",
        "node_type",
        "cpu_util",
        "mem_util",
        "queue_len",
        "available_cpu",
        "available_mem",
        "request_load_on_node",
        "node_degree",
    },
    "link_log.csv": {
        "time",
        "src",
        "dst",
        "delay_ms",
        "bandwidth",
        "loss",
        "jitter_ms",
        "queue_delay_ms",
        "bandwidth_util",
    },
    "service_log.csv": {
        "time",
        "service_id",
        "user_id",
        "service_type",
        "request_rate",
        "response_time",
        "base_response_time",
        "dependency_count",
        "current_edge",
        "cloud_node",
        "last_violation",
    },
    "path_log.csv": {
        "time",
        "service_id",
        "path_nodes",
        "path_links",
        "path_delay",
        "path_loss",
        "bottleneck_node",
        "bottleneck_link",
    },
    "sla_log.csv": {
        "service_id",
        "service_type",
        "max_delay",
        "max_loss",
        "min_bandwidth",
        "reliability_req",
        "cost_weight",
    },
}


def _csv_header(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return set(next(csv.reader(f)))


def _csv_row_count(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def test_synthetic_full_pipeline():
    run_cmd(["src/run_synthetic_full_pipeline.py", "--config", "configs/sparta_synthetic_full.yaml", "--fast_dev_run"])

    raw_dir = ROOT / "data/synthetic_full/raw_logs"
    for filename, required in REQUIRED_COLUMNS.items():
        path = raw_dir / filename
        assert path.exists(), filename
        assert required.issubset(_csv_header(path)), filename
        assert _csv_row_count(path) > 0, filename

    dataset_dir = ROOT / "data/synthetic_full/dataset"
    for split in ["train", "val", "test"]:
        assert (dataset_dir / f"{split}.pkl").exists()

    samples = load_pickle(dataset_dir / "train.pkl")
    assert samples
    sample = samples[0]
    for key in ["node_x", "link_x", "service_x", "sla_x", "adj", "risk_label", "risk_node", "risk_link", "risk_metric"]:
        assert key in sample
    assert sample["node_x"].shape == (12, 8, 8)
    assert sample["link_x"].shape == (12, 12, 8)
    assert sample["service_x"].shape == (12, 5)
    assert sample["sla_x"].shape == (5,)
    assert sample["adj"].shape == (8, 8)

    for model in ["lstm", "transformer", "sparta"]:
        assert (ROOT / f"checkpoints/synthetic_full/{model}_best.pth").exists()
        log_path = ROOT / f"logs/synthetic_full/train_log_{model}.csv"
        assert log_path.exists()
        assert _csv_row_count(log_path) >= 1

    all_results = ROOT / "results/synthetic_full/all_test_results.csv"
    assert all_results.exists()
    with all_results.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {row["method"] for row in rows} == {"lstm", "transformer", "sparta"}
