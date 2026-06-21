from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sparta_dataset import SPARTADataset
from src.models.sparta import DEFAULT_FEATURE_SCHEMA, build_metric_evidence, compute_metric_evidence_stats_from_loader, save_metric_evidence_stats
from src.preprocessing.build_path_graph import _f, _split_pipe, load_logs
from src.preprocessing.generate_labels import compute_metric_risk_scores
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_pickle


METRIC_NAMES = ["delay", "loss", "cpu", "queue", "bandwidth"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--max_cases", type=int, default=500)
    return parser.parse_args()


def schema_from_config(config: dict) -> dict[str, list[str]]:
    data_cfg = config.get("data", {})
    schema = {}
    for group, defaults in DEFAULT_FEATURE_SCHEMA.items():
        names = data_cfg.get(f"{group}_feature_names") or data_cfg.get(f"{group}_feat_names") or defaults
        schema[group] = [str(name) for name in names]
    return schema


def load_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config" / "resolved_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    config = load_config(config_path)
    dataset_dir = run_dir / "dataset"
    config.setdefault("data", {})
    config["data"]["dataset_dir"] = str(dataset_dir)
    config["data"]["train_path"] = str(dataset_dir / "train.pkl")
    config["data"]["test_path"] = str(dataset_dir / "test.pkl")
    return config


def pressure02_evidence(run_dir: Path, config: dict, samples: list[dict]) -> dict[int, list[float]]:
    test_path = run_dir / "dataset" / "test.pkl"
    train_path = run_dir / "dataset" / "train.pkl"
    stats_path = run_dir / "artifacts" / "metric_evidence_stats_pressure02.json"
    schema = schema_from_config(config)
    expected_version = "pressure02_v5_growth_rank"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats.get("version") != expected_version:
            stats = None
    else:
        stats = None
    if stats is None:
        train_ds = SPARTADataset(train_path, config)
        train_loader = DataLoader(train_ds, batch_size=int(config.get("train", {}).get("batch_size", 64)), shuffle=False, num_workers=0)
        stats = compute_metric_evidence_stats_from_loader(train_loader, schema, norm="pressure02")
        save_metric_evidence_stats(stats_path, stats)
    test_ds = SPARTADataset(test_path, config)
    loader = DataLoader(test_ds, batch_size=int(config.get("train", {}).get("batch_size", 64)), shuffle=False, num_workers=0)
    out: dict[int, list[float]] = {}
    offset = 0
    with torch.no_grad():
        for batch in loader:
            evidence = build_metric_evidence(batch, schema, norm="pressure02", evidence_stats=stats)
            rows = evidence.cpu().tolist()
            for idx, row in enumerate(rows):
                if offset + idx < len(samples):
                    out[offset + idx] = [float(v) for v in row]
            offset += len(rows)
    return out


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _corr(x: list[float], y: list[float], method: str) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if method == "spearman":
        xa = _rankdata(xa)
        ya = _rankdata(ya)
    if float(xa.std()) <= 1e-12 or float(ya.std()) <= 1e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def compare_rule_and_pressure(rows: list[dict]) -> dict:
    correlations = {}
    by_class = {}
    for idx, name in enumerate(METRIC_NAMES):
        rule_values = [float(row[f"{name}_score_used_by_label"]) for row in rows]
        pressure_values = [float(row[f"pressure02_{name}_evidence"]) for row in rows]
        correlations[name] = {
            "pearson": _corr(rule_values, pressure_values, "pearson"),
            "spearman": _corr(rule_values, pressure_values, "spearman"),
        }
    for metric_id, metric_name in enumerate(METRIC_NAMES):
        cls_rows = [row for row in rows if int(row["true_metric"]) == metric_id]
        by_class[metric_name] = {}
        for name in METRIC_NAMES:
            by_class[metric_name][f"labelrule_{name}"] = _mean_std(
                [float(row[f"{name}_score_used_by_label"]) for row in cls_rows]
            )
            by_class[metric_name][f"pressure02_{name}"] = _mean_std(
                [float(row[f"pressure02_{name}_evidence"]) for row in cls_rows]
            )
    mismatch_counter = Counter(
        (row["true_metric_name"], row["pressure02_argmax_name"])
        for row in rows
        if int(row["pressure02_argmax"]) != int(row["true_metric"])
    )
    return {
        "per_dimension_correlation": correlations,
        "by_true_metric_summary": by_class,
        "top_mismatch_pairs": [
            {"true_metric": true_name, "pressure02_pred": pred_name, "count": count}
            for (true_name, pred_name), count in mismatch_counter.most_common(20)
        ],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = load_run_config(run_dir)
    test_path = run_dir / "dataset" / "test.pkl"
    raw_dir = run_dir / "raw_logs"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test dataset: {test_path}")
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw logs dir: {raw_dir}")
    samples = load_pickle(test_path)
    logs = load_logs(raw_dir)
    pressure02_by_index = pressure02_evidence(run_dir, config, samples)

    rows: list[dict] = []
    match_count = 0
    pressure_match_count = 0
    valid_count = 0
    mismatch_examples: list[dict] = []
    for sample_index, sample in enumerate(samples):
        if int(sample.get("attr_mask", 0)) != 1:
            continue
        score_info = compute_metric_risk_scores(sample, logs, config)
        metric_scores = [float(x) for x in score_info["scores"]]
        true_metric = int(sample.get("risk_metric", -100))
        label_rule_argmax = int(score_info["label_rule_argmax"])
        pressure_scores = pressure02_by_index.get(sample_index, [0.0] * len(METRIC_NAMES))
        pressure_argmax = int(np.asarray(pressure_scores).argmax())
        score_fields = {
            "scenario_used_by_label": score_info.get("scenario_used_by_label", ""),
            "delay_score_used_by_label": metric_scores[0],
            "loss_score_used_by_label": metric_scores[1],
            "cpu_score_used_by_label": metric_scores[2],
            "queue_score_used_by_label": metric_scores[3],
            "bandwidth_score_used_by_label": metric_scores[4],
            "queue_abs_score": float(score_info.get("queue_abs_score", 0.0)),
            "queue_growth_score": float(score_info.get("queue_growth_score", 0.0)),
            "queue_trend": float(score_info.get("queue_trend", 0.0)),
            "request_trend": float(score_info.get("request_trend", 0.0)),
            "label_rule_argmax": label_rule_argmax,
        }
        pressure_fields = {
            f"pressure02_{name}_evidence": pressure_scores[idx] for idx, name in enumerate(METRIC_NAMES)
        }
        row = {
            "sample_index": sample_index,
            "sample_id": sample.get("sample_id", ""),
            "service_id": sample.get("service_id", ""),
            "time": sample.get("time", ""),
            "scenario": sample.get("scenario", ""),
            "true_metric": true_metric,
            "true_metric_name": METRIC_NAMES[true_metric] if 0 <= true_metric < len(METRIC_NAMES) else str(true_metric),
            **score_fields,
            "label_rule_argmax_name": METRIC_NAMES[label_rule_argmax],
            "label_rule_matches_true": int(label_rule_argmax == true_metric),
            **pressure_fields,
            "pressure02_argmax": pressure_argmax,
            "pressure02_argmax_name": METRIC_NAMES[pressure_argmax] if 0 <= pressure_argmax < len(METRIC_NAMES) else str(pressure_argmax),
            "pressure02_matches_label_rule": int(pressure_argmax == label_rule_argmax),
        }
        rows.append(row)
        valid_count += 1
        match_count += int(label_rule_argmax == true_metric)
        pressure_match_count += int(pressure_argmax == label_rule_argmax)
        if label_rule_argmax != true_metric or pressure_argmax != label_rule_argmax:
            mismatch_examples.append(row)

    report = {
        "run_dir": str(run_dir),
        "valid_attr_samples": valid_count,
        "label_rule_replay_matches_true": match_count,
        "label_rule_replay_accuracy": match_count / valid_count if valid_count else 0.0,
        "pressure02_matches_label_rule": pressure_match_count,
        "pressure02_matches_label_rule_rate": pressure_match_count / valid_count if valid_count else 0.0,
        "note": "pressure02 evidence and label-rule scores differ on many samples; compare *_score_used_by_label with pressure02 evidence calibration."
        if valid_count and pressure_match_count / valid_count < 0.8
        else "pressure02 evidence is broadly aligned with replayed label-rule argmax.",
    }
    compare_report = {
        **report,
        **compare_rule_and_pressure(rows),
    }
    analysis_dir = ensure_dir(run_dir / "analysis")
    report_path = analysis_dir / "metric_label_rule_replay.json"
    cases_path = analysis_dir / "metric_label_rule_replay_cases.csv"
    compare_path = analysis_dir / "metric_label_rule_vs_pressure02.json"
    compare_cases_path = analysis_dir / "metric_label_rule_vs_pressure02_cases.csv"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(cases_path, rows[: args.max_cases] if args.max_cases > 0 else rows)
    compare_path.write_text(json.dumps(compare_report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(compare_cases_path, rows[: args.max_cases] if args.max_cases > 0 else rows)
    print(f"Wrote {report_path}")
    print(f"Wrote {cases_path}")
    print(f"Wrote {compare_path}")
    print(f"Wrote {compare_cases_path}")


if __name__ == "__main__":
    main()
