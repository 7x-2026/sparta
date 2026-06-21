from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.sparta_dataset import SPARTADataset
from src.metrics import compute_attribution_majority_baseline
from src.train import build_model, select_device
from src.utils.config import load_config
from src.utils.io import ensure_dir, load_pickle
from src.utils.seed import set_seed


METRIC_NAMES = {0: "delay", 1: "loss", 2: "cpu", 3: "queue", 4: "bandwidth"}
ATTR_FIELDS = ["risk_node", "risk_link", "risk_metric"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--max_error_cases", type=int, default=100)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_float(value) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text == "" or text.upper() == "N/A" or text.lower() == "nan":
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def load_run_config(run_dir: Path) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    candidates = [
        run_dir / "config" / "resolved_config.yaml",
        Path(str(manifest.get("resolved_config_path", ""))),
    ]
    for path in candidates:
        if path and path.exists():
            return load_config(path)
    raise FileNotFoundError(f"Missing resolved config under {run_dir / 'config'}")


def normalize_variant(variant: str | None) -> str | None:
    if variant is None:
        return None
    text = str(variant).strip()
    if not text:
        return None
    if any(ch in text for ch in ["/", "\\", ":"]):
        raise ValueError(f"Invalid variant name: {variant}")
    return text


def sparta_artifact_name(variant: str | None) -> str:
    return f"sparta_{variant}" if variant else "sparta"


def result_row(run_dir: Path, variant: str | None = None) -> dict:
    artifact = sparta_artifact_name(variant)
    path = run_dir / "results" / f"{artifact}_test_results.csv"
    if not path.exists():
        if variant:
            raise FileNotFoundError(f"Missing variant SPARTA result file: {path}")
        all_path = run_dir / "results" / "all_test_results.csv"
        if not all_path.exists():
            raise FileNotFoundError(f"Missing SPARTA result file: {path}")
        rows = [row for row in read_csv_rows(all_path) if (row.get("model") or row.get("method")) == "sparta"]
    else:
        rows = read_csv_rows(path)
    if not rows:
        raise RuntimeError("No SPARTA result row found")
    return rows[0]


def load_model_predictions(run_dir: Path, config: dict, variant: str | None = None) -> tuple[list[dict], dict[str, list[int]], dict[str, list[list[float]]], dict]:
    test_path = run_dir / "dataset" / "test.pkl"
    artifact = sparta_artifact_name(variant)
    checkpoint_path = run_dir / "checkpoints" / f"{artifact}_best.pth"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing dataset/test.pkl: {test_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing SPARTA checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("config", config) if isinstance(checkpoint, dict) else config
    set_seed(int(model_config.get("seed", 42)))
    device = select_device(model_config)
    model = build_model(model_config, "sparta").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SPARTADataset(test_path, model_config)
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"].get("num_workers", 0)),
    )
    preds = {"risk_label": [], "risk_node": [], "risk_link": [], "risk_metric": []}
    logits = {"node_logits": [], "link_logits": [], "metric_logits": []}
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            outputs = model(batch)
            preds["risk_label"].extend(outputs["risk_logits"].argmax(dim=-1).cpu().tolist())
            preds["risk_node"].extend(outputs["node_logits"].argmax(dim=-1).cpu().tolist())
            preds["risk_link"].extend(outputs["link_logits"].argmax(dim=-1).cpu().tolist())
            preds["risk_metric"].extend(outputs["metric_logits"].argmax(dim=-1).cpu().tolist())
            logits["node_logits"].extend(outputs["node_logits"].cpu().tolist())
            logits["link_logits"].extend(outputs["link_logits"].cpu().tolist())
            logits["metric_logits"].extend(outputs["metric_logits"].cpu().tolist())
    return dataset.samples, preds, logits, model_config


def valid_attr_indices(samples: list[dict]) -> list[int]:
    return [idx for idx, sample in enumerate(samples) if int(sample.get("attr_mask", 0)) == 1]


def attr_acc(samples: list[dict], preds: dict[str, list[int]], field: str, indices: list[int]) -> float:
    if not indices:
        return math.nan
    return sum(int(int(samples[idx][field]) == int(preds[field][idx])) for idx in indices) / len(indices)


def attr_acc_including_normal(samples: list[dict], preds: dict[str, list[int]], field: str) -> float:
    if not samples:
        return math.nan
    return sum(int(int(sample[field]) == int(preds[field][idx])) for idx, sample in enumerate(samples)) / len(samples)


def close_to(a: float, b: float, tol: float = 1e-6) -> bool:
    return not math.isnan(a) and not math.isnan(b) and abs(a - b) <= tol


def metric_confusion(samples: list[dict], preds: dict[str, list[int]], indices: list[int]) -> dict[tuple[int, int], int]:
    counts: Counter[tuple[int, int]] = Counter()
    for idx in indices:
        counts[(int(samples[idx]["risk_metric"]), int(preds["risk_metric"][idx]))] += 1
    return dict(counts)


def per_metric_rows(confusion: dict[tuple[int, int], int]) -> list[dict]:
    rows: list[dict] = []
    for metric_id, metric_name in METRIC_NAMES.items():
        tp = confusion.get((metric_id, metric_id), 0)
        support = sum(count for (true, _pred), count in confusion.items() if true == metric_id)
        predicted = sum(count for (_true, pred), count in confusion.items() if pred == metric_id)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append(
            {
                "metric_id": metric_id,
                "metric": metric_name,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def metric_prediction_summary(confusion: dict[tuple[int, int], int], total_valid: int, metric_acc: float, baseline: dict) -> dict:
    predicted_counts = {
        f"metric_pred_count_{METRIC_NAMES[metric_id]}": sum(count for (_true, pred), count in confusion.items() if pred == metric_id)
        for metric_id in METRIC_NAMES
    }
    recalls = {}
    for metric_id, name in METRIC_NAMES.items():
        support = sum(count for (true, _pred), count in confusion.items() if true == metric_id)
        hit = confusion.get((metric_id, metric_id), 0)
        recalls[f"metric_recall_{name}"] = hit / support if support else 0.0
    majority = baseline.get("metric_majority_acc", math.nan)
    if math.isnan(majority):
        true_counts = Counter()
        for (true, _pred), count in confusion.items():
            true_counts[true] += count
        majority = true_counts.most_common(1)[0][1] / total_valid if true_counts and total_valid else math.nan
    return {
        **predicted_counts,
        **recalls,
        "metric_majority_baseline": majority,
        "metric_attr_acc_minus_majority": metric_acc - majority if not math.isnan(metric_acc) and not math.isnan(majority) else math.nan,
    }


def confusion_rows(confusion: dict[tuple[int, int], int]) -> list[dict]:
    rows: list[dict] = []
    for true_id, true_name in METRIC_NAMES.items():
        for pred_id, pred_name in METRIC_NAMES.items():
            rows.append(
                {
                    "true_metric": true_name,
                    "pred_metric": pred_name,
                    "true_metric_id": true_id,
                    "pred_metric_id": pred_id,
                    "count": confusion.get((true_id, pred_id), 0),
                }
            )
    return rows


def list_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    return str(value)


def id_at(values, idx: int) -> str:
    if not isinstance(values, list) or idx < 0 or idx >= len(values):
        return ""
    return str(values[idx])


def label_space_check(samples: list[dict], preds: dict[str, list[int]], indices: list[int]) -> dict:
    node_label_valid = 0
    link_label_valid = 0
    node_pred_valid = 0
    link_pred_valid = 0
    node_path_contained = 0
    link_path_contained = 0
    for idx in indices:
        sample = samples[idx]
        node_ids = sample.get("node_ids", [])
        link_ids = sample.get("link_ids", [])
        node_mask = sample.get("node_mask", [])
        link_mask = sample.get("link_mask", [])
        risk_node = int(sample["risk_node"])
        risk_link = int(sample["risk_link"])
        pred_node = int(preds["risk_node"][idx])
        pred_link = int(preds["risk_link"][idx])
        if 0 <= risk_node < len(node_ids) and bool(node_mask[risk_node]):
            node_label_valid += 1
        if 0 <= risk_link < len(link_ids) and bool(link_mask[risk_link]):
            link_label_valid += 1
        if 0 <= pred_node < len(node_ids) and bool(node_mask[pred_node]):
            node_pred_valid += 1
        if 0 <= pred_link < len(link_ids) and bool(link_mask[pred_link]):
            link_pred_valid += 1
        path_nodes = set(sample.get("path_nodes", []) or [])
        path_links = set(sample.get("path_links", []) or [])
        if not path_nodes or id_at(node_ids, risk_node) in path_nodes:
            node_path_contained += 1
        if not path_links or id_at(link_ids, risk_link) in path_links:
            link_path_contained += 1
    total = len(indices)
    return {
        "risk_node_label_space": "path_local_index" if node_label_valid == total else "mixed_or_global_id",
        "risk_link_label_space": "path_local_index" if link_label_valid == total else "mixed_or_global_id",
        "valid_risk_node_index_ratio": node_label_valid / total if total else math.nan,
        "valid_risk_link_index_ratio": link_label_valid / total if total else math.nan,
        "valid_pred_node_index_ratio": node_pred_valid / total if total else math.nan,
        "valid_pred_link_index_ratio": link_pred_valid / total if total else math.nan,
        "risk_node_id_in_path_nodes_ratio": node_path_contained / total if total else math.nan,
        "risk_link_id_in_path_links_ratio": link_path_contained / total if total else math.nan,
        "node_logits_order_check": "node_logits position i maps to sample['node_ids'][i] when node_mask[i]=true",
        "link_logits_order_check": "link_logits position i maps to sample['link_ids'][i] when link_mask[i]=true",
    }


def error_case_rows(samples: list[dict], preds: dict[str, list[int]], indices: list[int], max_cases: int) -> list[dict]:
    rows: list[dict] = []
    for idx in indices:
        sample = samples[idx]
        true_metric = int(sample["risk_metric"])
        pred_metric = int(preds["risk_metric"][idx])
        true_node = int(sample["risk_node"])
        pred_node = int(preds["risk_node"][idx])
        true_link = int(sample["risk_link"])
        pred_link = int(preds["risk_link"][idx])
        if true_metric == pred_metric and true_node == pred_node and true_link == pred_link:
            continue
        node_ids = sample.get("node_ids", [])
        link_ids = sample.get("link_ids", [])
        rows.append(
            {
                "sample_index": idx,
                "sample_id": sample.get("sample_id", ""),
                "service_id": sample.get("service_id", ""),
                "time": sample.get("time", ""),
                "scenario": sample.get("scenario", ""),
                "risk_label": sample.get("risk_label", ""),
                "true_metric": METRIC_NAMES.get(true_metric, str(true_metric)),
                "pred_metric": METRIC_NAMES.get(pred_metric, str(pred_metric)),
                "true_metric_id": true_metric,
                "pred_metric_id": pred_metric,
                "true_node": true_node,
                "pred_node": pred_node,
                "true_node_id": id_at(node_ids, true_node),
                "pred_node_id": id_at(node_ids, pred_node),
                "true_link": true_link,
                "pred_link": pred_link,
                "true_link_id": id_at(link_ids, true_link),
                "pred_link_id": id_at(link_ids, pred_link),
                "node_ids": list_text(node_ids),
                "link_ids": list_text(link_ids),
            }
        )
        if len(rows) >= max_cases:
            break
    return rows


def normal_label_diagnostics(samples: list[dict]) -> dict:
    normal = [sample for sample in samples if int(sample["risk_label"]) == 0]
    valid_normal = [sample for sample in normal if int(sample.get("attr_mask", 0)) == 1]
    ignored = {
        field: sum(int(int(sample[field]) == -100) for sample in normal)
        for field in ATTR_FIELDS
    }
    return {
        "normal_samples": len(normal),
        "normal_attr_mask_1": len(valid_normal),
        "normal_attr_mask_1_ratio": len(valid_normal) / len(normal) if normal else math.nan,
        "normal_ignore_label_counts": ignored,
        "normal_labels_are_ignore_index": all(count == len(normal) for count in ignored.values()) if normal else True,
    }


def main() -> None:
    args = parse_args()
    variant = normalize_variant(args.variant)
    artifact = sparta_artifact_name(variant)
    run_dir = Path(args.run_dir).resolve()
    analysis_dir = ensure_dir(run_dir / "analysis")
    config = load_run_config(run_dir)
    result = result_row(run_dir, variant)
    samples, preds, _logits, model_config = load_model_predictions(run_dir, config, variant)
    baseline = compute_attribution_majority_baseline(samples, split="test", attr_mask_only=True)
    valid_indices = valid_attr_indices(samples)
    normal_diag = normal_label_diagnostics(samples)

    label_counts = Counter(int(sample["risk_label"]) for sample in samples)
    attr_label_counts = Counter(int(samples[idx]["risk_label"]) for idx in valid_indices)
    reported = {
        "node_attr_acc": parse_float(result.get("node_attr_acc")),
        "link_attr_acc": parse_float(result.get("link_attr_acc")),
        "metric_attr_acc": parse_float(result.get("metric_attr_acc")),
    }
    recomputed_valid = {
        "node_attr_acc": attr_acc(samples, preds, "risk_node", valid_indices),
        "link_attr_acc": attr_acc(samples, preds, "risk_link", valid_indices),
        "metric_attr_acc": attr_acc(samples, preds, "risk_metric", valid_indices),
    }
    recomputed_all = {
        "node_attr_acc": attr_acc_including_normal(samples, preds, "risk_node"),
        "link_attr_acc": attr_acc_including_normal(samples, preds, "risk_link"),
        "metric_attr_acc": attr_acc_including_normal(samples, preds, "risk_metric"),
    }
    normal_included_check = {
        metric: {
            "reported": reported[metric],
            "valid_attr_only": recomputed_valid[metric],
            "including_normal": recomputed_all[metric],
            "matches_valid_attr_only": close_to(reported[metric], recomputed_valid[metric]),
            "matches_including_normal": close_to(reported[metric], recomputed_all[metric]),
        }
        for metric in reported
    }

    confusion = metric_confusion(samples, preds, valid_indices)
    per_metric = per_metric_rows(confusion)
    metric_summary = metric_prediction_summary(confusion, len(valid_indices), recomputed_valid["metric_attr_acc"], baseline)
    error_rows = error_case_rows(samples, preds, valid_indices, args.max_error_cases)

    diagnosis = {
        "run_dir": str(run_dir),
        "variant": variant or "",
        "dataset_path": str(run_dir / "dataset" / "test.pkl"),
        "result_path": str(run_dir / "results" / f"{artifact}_test_results.csv"),
        "checkpoint_path": str(run_dir / "checkpoints" / f"{artifact}_best.pth"),
        "model_config_experiment_name": model_config.get("experiment", {}).get("name", ""),
        "use_metric_evidence_head": bool(model_config.get("model", {}).get("use_metric_evidence_head", False)),
        "sample_counts": {
            "total": len(samples),
            "normal": label_counts.get(0, 0),
            "risky": label_counts.get(1, 0),
            "violated": label_counts.get(2, 0),
            "valid_attribution": len(valid_indices),
            "valid_attribution_by_risk_label": {
                "normal": attr_label_counts.get(0, 0),
                "risky": attr_label_counts.get(1, 0),
                "violated": attr_label_counts.get(2, 0),
            },
        },
        "normal_label_diagnostics": normal_diag,
        "attribution_eval_normal_inclusion_check": normal_included_check,
        "label_space_and_logit_order_checks": label_space_check(samples, preds, valid_indices),
        "metric_head_order": {
            "assumed_order": METRIC_NAMES,
            "metric_logits_dim": 5,
            "check": "metric_logits[:, 0..4] interpreted as delay, loss, cpu, queue, bandwidth",
        },
        "metric_per_class": per_metric,
        "reported_sparta_attribution_acc": reported,
        "recomputed_sparta_attribution_acc_valid_attr_only": recomputed_valid,
        "recomputed_sparta_attribution_acc_including_normal": recomputed_all,
        "majority_baseline": baseline,
        "metric_prediction_summary": metric_summary,
        "metric_majority_acc_valid_attr_only": baseline.get("metric_majority_acc_valid_attr_only"),
        "metric_attr_acc_minus_majority": metric_summary.get("metric_attr_acc_minus_majority"),
        "uses_future_scores_as_supervision_only": bool(
            float(model_config.get("loss", {}).get("lambda_metric_soft", 0.0)) > 0.0
        ),
        "risk_metric_scores_use_future_information": bool(
            float(model_config.get("loss", {}).get("lambda_metric_soft", 0.0)) > 0.0
        ),
        "used_future_scores_as_model_input": False,
        "sparta_vs_majority": {
            "node_attr_acc_minus_majority": recomputed_valid["node_attr_acc"] - baseline.get("node_majority_acc", math.nan),
            "link_attr_acc_minus_majority": recomputed_valid["link_attr_acc"] - baseline.get("link_majority_acc", math.nan),
            "metric_attr_acc_minus_majority": recomputed_valid["metric_attr_acc"] - baseline.get("metric_majority_acc", math.nan),
        },
        "notes": [
            "Attribution accuracy should be computed only on attr_mask=1 samples.",
            "Normal samples should have risk_node/risk_link/risk_metric=-100 and attr_mask=0.",
            "Node/link labels are expected to be path-local positions aligned with node_ids/link_ids and logits dimensions.",
        ],
    }

    prefix = f"{variant}_" if variant else ""
    json_path = analysis_dir / f"{prefix}attribution_diagnosis.json"
    confusion_path = analysis_dir / f"{prefix}attribution_metric_confusion.csv"
    error_path = analysis_dir / f"{prefix}attribution_error_cases.csv"
    json_path.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(confusion_path, confusion_rows(confusion), ["true_metric", "pred_metric", "true_metric_id", "pred_metric_id", "count"])
    write_csv(
        error_path,
        error_rows,
        [
            "sample_index",
            "sample_id",
            "service_id",
            "time",
            "scenario",
            "risk_label",
            "true_metric",
            "pred_metric",
            "true_metric_id",
            "pred_metric_id",
            "true_node",
            "pred_node",
            "true_node_id",
            "pred_node_id",
            "true_link",
            "pred_link",
            "true_link_id",
            "pred_link_id",
            "node_ids",
            "link_ids",
        ],
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {confusion_path}")
    print(f"Wrote {error_path}")


if __name__ == "__main__":
    main()
