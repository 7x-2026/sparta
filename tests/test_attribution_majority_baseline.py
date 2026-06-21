from __future__ import annotations

from src.metrics import compute_attribution_majority_baseline


def test_attribution_majority_baseline_excludes_normal_by_default():
    samples = [
        {"risk_label": 0, "attr_mask": 0, "risk_node": -100, "risk_link": -100, "risk_metric": -100},
        {"risk_label": 0, "attr_mask": 0, "risk_node": -100, "risk_link": -100, "risk_metric": -100},
        {"risk_label": 1, "attr_mask": 1, "risk_node": 1, "risk_link": 0, "risk_metric": 2},
        {"risk_label": 2, "attr_mask": 1, "risk_node": 1, "risk_link": 1, "risk_metric": 2},
        {"risk_label": 1, "attr_mask": 1, "risk_node": 2, "risk_link": 1, "risk_metric": 3},
    ]

    baseline = compute_attribution_majority_baseline(samples, split="test")

    assert baseline["denominator"] == 3
    assert baseline["valid_attr_samples"] == 3
    assert baseline["metric_majority_value"] == 2
    assert baseline["metric_majority_name"] == "cpu"
    assert baseline["metric_majority_count"] == 2
    assert baseline["metric_majority_acc_valid_attr_only"] == 2 / 3
    assert baseline["metric_majority_acc"] == 2 / 3
    assert baseline["metric_majority_count_by_class"] == {"2": 2, "3": 1}


def test_attribution_majority_baseline_can_report_including_normal_separately():
    samples = [
        {"risk_label": 0, "attr_mask": 0, "risk_node": -100, "risk_link": -100, "risk_metric": -100},
        {"risk_label": 0, "attr_mask": 0, "risk_node": -100, "risk_link": -100, "risk_metric": -100},
        {"risk_label": 1, "attr_mask": 1, "risk_node": 1, "risk_link": 0, "risk_metric": 2},
    ]

    baseline = compute_attribution_majority_baseline(samples, split="test")

    assert baseline["metric_majority_acc_valid_attr_only"] == 1.0
    assert baseline["metric_majority_acc_including_normal"] == 2 / 3
    assert baseline["metric_majority_value"] == 2
