from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from src.analysis import calibrate_metric_logits as calibration
from src.utils.io import save_pickle


def make_sample(metric: int, logits: list[float], attr_mask: int = 1) -> dict:
    L, N, E = 3, 4, 4
    service_x = np.zeros((L, 5), dtype=np.float32)
    service_x[-1, :] = np.asarray(logits, dtype=np.float32)
    return {
        "sample_id": f"s_{metric}_{attr_mask}_{sum(logits):.2f}",
        "service_id": 0,
        "time": 10,
        "scenario": "node_overload",
        "node_x": np.zeros((L, N, 8), dtype=np.float32),
        "link_x": np.zeros((L, E, 8), dtype=np.float32),
        "service_x": service_x,
        "sla_x": np.ones((5,), dtype=np.float32),
        "adj": np.eye(N, dtype=np.float32),
        "node_mask": np.ones((N,), dtype=bool),
        "link_mask": np.ones((E,), dtype=bool),
        "node_ids": [f"n{i}" for i in range(N)],
        "link_ids": [f"l{i}" for i in range(E)],
        "risk_label": 1 if attr_mask else 0,
        "risk_node": 1 if attr_mask else -100,
        "risk_link": 1 if attr_mask else -100,
        "risk_metric": metric if attr_mask else -100,
        "risk_metric_scores": [0.0] * 5,
        "attr_mask": attr_mask,
    }


def make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run" / "mock_cpu_precursor"
    for name in ["config", "dataset", "checkpoints", "results", "analysis"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    config = {
        "seed": 1,
        "project": {
            "checkpoint_dir": str(run_dir / "checkpoints"),
            "output_dir": str(run_dir / "results"),
            "log_dir": str(run_dir / "logs"),
        },
        "run": {"run_id": run_dir.name},
        "data": {
            "input_window": 3,
            "max_nodes": 4,
            "max_links": 4,
            "node_feat_dim": 8,
            "link_feat_dim": 8,
            "service_feat_dim": 5,
            "sla_feat_dim": 5,
            "dataset_dir": str(run_dir / "dataset"),
            "train_path": str(run_dir / "dataset" / "train.pkl"),
            "val_path": str(run_dir / "dataset" / "val.pkl"),
            "test_path": str(run_dir / "dataset" / "test.pkl"),
        },
        "model": {
            "hidden_dim": 8,
            "temporal_layers": 1,
            "topology_layers": 1,
            "n_heads": 2,
            "dropout": 0.0,
            "use_topology": True,
            "use_sla_gate": True,
            "use_metric_evidence_head": False,
        },
        "train": {"batch_size": 4, "num_workers": 0, "device": "cpu"},
    }
    (run_dir / "config" / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    checkpoint_bytes = b"original checkpoint bytes"
    (run_dir / "checkpoints" / "sparta_cpu_precursor_v1_best.pth").write_bytes(checkpoint_bytes)
    val_samples = [
        make_sample(3, [1.2, 1.0, 0.8, 0.1, 1.1], attr_mask=1),
        make_sample(3, [1.1, 0.9, 0.7, 0.0, 1.0], attr_mask=1),
        make_sample(1, [0.0, 1.4, 0.1, 0.0, 0.2], attr_mask=1),
        make_sample(-100, [0.0, 0.0, 0.0, 9.0, 0.0], attr_mask=0),
    ]
    test_samples = [
        make_sample(3, [1.1, 0.8, 0.7, 0.1, 1.0], attr_mask=1),
        make_sample(3, [1.0, 0.8, 0.6, 0.0, 0.9], attr_mask=1),
        make_sample(1, [0.0, 1.2, 0.0, 0.0, 0.1], attr_mask=1),
        make_sample(-100, [0.0, 0.0, 0.0, 9.0, 0.0], attr_mask=0),
    ]
    save_pickle(run_dir / "dataset" / "train.pkl", val_samples)
    save_pickle(run_dir / "dataset" / "val.pkl", val_samples)
    save_pickle(run_dir / "dataset" / "test.pkl", test_samples)
    return run_dir


class DummySparta(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = batch["service_x"].shape[0]
        return {
            "risk_logits": torch.zeros(batch_size, 3, device=batch["service_x"].device),
            "node_logits": torch.zeros(batch_size, 4, device=batch["service_x"].device),
            "link_logits": torch.zeros(batch_size, 4, device=batch["service_x"].device),
            "metric_logits": batch["service_x"][:, -1, :5],
        }


def test_metric_logit_calibration_outputs_and_preserves_checkpoint(tmp_path: Path, monkeypatch):
    run_dir = make_run(tmp_path)
    checkpoint_path = run_dir / "checkpoints" / "sparta_cpu_precursor_v1_best.pth"
    before_bytes = checkpoint_path.read_bytes()
    searched = {}
    original_search = calibration.coordinate_grid_search

    def wrapped_search(logits, labels, grid_values, objective="acc_plus_balanced", max_passes=5):
        searched["labels"] = labels.tolist()
        return original_search(logits, labels, grid_values, objective=objective, max_passes=max_passes)

    monkeypatch.setattr(calibration, "select_device", lambda config: torch.device("cpu"))
    monkeypatch.setattr(calibration, "load_sparta_model", lambda config, checkpoint_path, device: DummySparta())
    monkeypatch.setattr(calibration, "coordinate_grid_search", wrapped_search)

    summary = calibration.calibrate_run(
        run_dir=run_dir,
        variant="cpu_precursor_v1",
        objective="acc_plus_balanced",
        grid_min=-1.0,
        grid_max=1.0,
        grid_step=0.5,
        max_passes=2,
    )

    assert checkpoint_path.read_bytes() == before_bytes
    assert searched["labels"] == [3, 3, 1]
    assert summary["val_total_samples"] == 4
    assert summary["val_valid_attr_samples"] == 3
    assert summary["test_valid_attr_samples"] == 3
    assert summary["normal_samples_excluded_from_calibration"] is True
    assert (run_dir / "analysis" / "cpu_precursor_v1_metric_calibration.json").exists()
    assert (run_dir / "analysis" / "cpu_precursor_v1_metric_calibrated_confusion.csv").exists()
    assert (run_dir / "results" / "sparta_cpu_precursor_v1_calibrated_test_results.csv").exists()
    assert not (run_dir / "results" / "sparta_cpu_precursor_v1_test_results.csv").exists()


def test_metric_logit_calibration_uses_test_only_for_application(tmp_path: Path, monkeypatch):
    run_dir = make_run(tmp_path)
    collected_paths = []
    original_collect = calibration.collect_metric_logits

    def wrapped_collect(model, dataset_path, config, device, batch_size):
        collected_paths.append(Path(dataset_path).name)
        return original_collect(model, dataset_path, config, device, batch_size)

    monkeypatch.setattr(calibration, "select_device", lambda config: torch.device("cpu"))
    monkeypatch.setattr(calibration, "load_sparta_model", lambda config, checkpoint_path, device: DummySparta())
    monkeypatch.setattr(calibration, "collect_metric_logits", wrapped_collect)

    calibration.calibrate_run(
        run_dir=run_dir,
        variant="cpu_precursor_v1",
        objective="balanced_acc",
        grid_min=-0.5,
        grid_max=0.5,
        grid_step=0.5,
        max_passes=1,
    )

    assert collected_paths == ["val.pkl", "test.pkl"]


def test_metric_logit_calibration_name_keeps_outputs_separate(tmp_path: Path, monkeypatch):
    run_dir = make_run(tmp_path)
    monkeypatch.setattr(calibration, "select_device", lambda config: torch.device("cpu"))
    monkeypatch.setattr(calibration, "load_sparta_model", lambda config, checkpoint_path, device: DummySparta())

    calibration.calibrate_run(
        run_dir=run_dir,
        variant="cpu_precursor_v1",
        objective="acc_plus_balanced",
        grid_min=-0.5,
        grid_max=0.5,
        grid_step=0.5,
        max_passes=1,
        calibration_name="acc_plus_balanced",
    )
    calibration.calibrate_run(
        run_dir=run_dir,
        variant="cpu_precursor_v1",
        objective="balanced_acc",
        grid_min=-0.5,
        grid_max=0.5,
        grid_step=0.5,
        max_passes=1,
        calibration_name="balanced_acc",
    )

    assert (run_dir / "analysis" / "cpu_precursor_v1_metric_calibration_acc_plus_balanced.json").exists()
    assert (run_dir / "analysis" / "cpu_precursor_v1_metric_calibrated_confusion_acc_plus_balanced.csv").exists()
    assert (run_dir / "results" / "sparta_cpu_precursor_v1_calibrated_test_results_acc_plus_balanced.csv").exists()
    assert (run_dir / "analysis" / "cpu_precursor_v1_metric_calibration_balanced_acc.json").exists()
    assert (run_dir / "analysis" / "cpu_precursor_v1_metric_calibrated_confusion_balanced_acc.csv").exists()
    assert (run_dir / "results" / "sparta_cpu_precursor_v1_calibrated_test_results_balanced_acc.csv").exists()
    assert not (run_dir / "results" / "sparta_cpu_precursor_v1_test_results.csv").exists()
