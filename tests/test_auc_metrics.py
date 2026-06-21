from __future__ import annotations

import math

from src.metrics import compute_classification_metrics, compute_risk_auc_metrics


def test_multiclass_risk_probs_compute_macro_auc():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    risk_probs = [
        [0.90, 0.07, 0.03],
        [0.08, 0.84, 0.08],
        [0.05, 0.10, 0.85],
        [0.80, 0.15, 0.05],
        [0.12, 0.76, 0.12],
        [0.07, 0.18, 0.75],
    ]

    metrics = compute_classification_metrics(y_true, y_pred, risk_probs)

    assert metrics["risk_auc_macro_ovr"] > 0.99
    assert metrics["risk_auc_weighted_ovr"] > 0.99
    assert metrics["auc"] == metrics["risk_auc_macro_ovr"]
    assert metrics["auc_error"] == ""


def test_binary_risk_auc_normal_vs_risky_violated():
    y_true = [0, 0, 1, 2, 1, 2]
    risk_probs = [
        [0.90, 0.05, 0.05],
        [0.85, 0.10, 0.05],
        [0.20, 0.70, 0.10],
        [0.10, 0.20, 0.70],
        [0.15, 0.60, 0.25],
        [0.05, 0.25, 0.70],
    ]

    metrics = compute_risk_auc_metrics(y_true, risk_probs)

    assert metrics["risk_auc_binary"] > 0.99


def test_hard_labels_do_not_compute_auc():
    y_true = [0, 1, 2, 0, 1, 2]
    hard_pred = [0, 1, 2, 0, 1, 2]

    metrics = compute_classification_metrics(y_true, hard_pred)

    assert math.isnan(metrics["auc"])
    assert math.isnan(metrics["risk_auc_macro_ovr"])
    assert "y_score" in metrics["auc_error"]


def test_missing_class_auc_returns_nan_and_error():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    risk_probs = [
        [0.90, 0.08, 0.02],
        [0.85, 0.10, 0.05],
        [0.20, 0.70, 0.10],
        [0.15, 0.75, 0.10],
    ]

    metrics = compute_classification_metrics(y_true, y_pred, risk_probs)

    assert math.isnan(metrics["risk_auc_macro_ovr"])
    assert math.isnan(metrics["auc"])
    assert "present=[0, 1]" in metrics["auc_error"]
    assert not math.isnan(metrics["risk_auc_binary"])
