from __future__ import annotations

import inspect
import json
import shutil

import torch
import yaml

from src.models import sparta
from src.models.sparta import build_metric_evidence
from src.preprocessing.build_path_graph import load_logs
from src.preprocessing.generate_labels import compute_metric_risk_scores
from src.utils.config import load_config
from src.utils.io import load_pickle

from .conftest import ROOT, ensure_dataset, run_cmd


def make_labelrule_batch() -> dict[str, torch.Tensor]:
    batch_size, L, N, E = 2, 4, 3, 2
    return {
        "node_x": torch.zeros(batch_size, L, N, 8),
        "link_x": torch.zeros(batch_size, L, E, 8),
        "service_x": torch.zeros(batch_size, L, 5),
        "sla_x": torch.ones(batch_size, 5),
        "node_mask": torch.ones(batch_size, N, dtype=torch.bool),
        "link_mask": torch.ones(batch_size, E, dtype=torch.bool),
        "metric_rule_scores": torch.tensor(
            [
                [0.1, 0.2, 0.8, 0.3, 0.4],
                [0.9, 0.1, 0.2, 0.3, 0.4],
            ],
            dtype=torch.float32,
        ),
    }


def make_mock_run(tmp_path):
    ensure_dataset()
    run_dir = tmp_path / "labelrule_run"
    shutil.copytree(ROOT / "data" / "debug" / "dataset", run_dir / "dataset")
    shutil.copytree(ROOT / "data" / "debug" / "raw_logs", run_dir / "raw_logs")
    (run_dir / "config").mkdir(parents=True)
    config = load_config(ROOT / "configs" / "sparta_debug.yaml")
    config["data"]["dataset_dir"] = str(run_dir / "dataset")
    config["data"]["raw_logs_dir"] = str(run_dir / "raw_logs")
    config["data"]["train_path"] = str(run_dir / "dataset" / "train.pkl")
    config["data"]["val_path"] = str(run_dir / "dataset" / "val.pkl")
    config["data"]["test_path"] = str(run_dir / "dataset" / "test.pkl")
    config["train"]["batch_size"] = 16
    (run_dir / "config" / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return run_dir


def test_labelrule_evidence_shape_from_precomputed_scores():
    batch = make_labelrule_batch()
    evidence = build_metric_evidence(batch, norm="labelrule")

    assert tuple(evidence.shape) == (2, 5)
    assert torch.equal(evidence, batch["metric_rule_scores"])


def test_labelrule_reproduces_metric_or_marks_future_leakage():
    ensure_dataset()
    config = load_config(ROOT / "configs" / "sparta_debug.yaml")
    logs = load_logs(ROOT / "data" / "debug" / "raw_logs")
    samples = load_pickle(ROOT / "data" / "debug" / "dataset" / "test.pkl")
    sample = next(sample for sample in samples if int(sample["attr_mask"]) == 1)

    info = compute_metric_risk_scores(sample, logs, config)

    assert info["uses_future_information"] is True
    assert info["allowed_for_model_input"] is False
    assert int(info["label_rule_argmax"]) == int(sample["risk_metric"])


def test_generate_labels_and_build_metric_evidence_use_same_function():
    source = inspect.getsource(sparta.build_metric_evidence)

    assert "compute_metric_risk_scores" in source


def test_metric_evidence_baseline_labelrule_outputs(tmp_path):
    run_dir = make_mock_run(tmp_path)

    run_cmd(
        [
            "src/analysis/evaluate_metric_evidence_baseline.py",
            "--run_dir",
            str(run_dir),
            "--evidence_norm",
            "labelrule",
        ]
    )

    json_path = run_dir / "analysis" / "metric_evidence_baseline_labelrule.json"
    confusion_path = run_dir / "analysis" / "metric_evidence_baseline_labelrule_confusion.csv"
    assert json_path.exists()
    assert confusion_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["uses_future_information"] is True
    assert data["allowed_for_model_input"] is False
    assert data["labelrule_evidence_acc"] >= 0.99
