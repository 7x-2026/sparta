from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.io import ensure_dir, load_json, load_pickle, read_csv_rows


LEAKAGE_TOKENS = ("risk_metric", "risk_metric_scores", "label_score", "future", "horizon")
FEATURES_TO_AUDIT = [
    ("node", "cpu_util_slope"),
    ("node", "available_cpu_slope"),
    ("node", "colocated_request_rate_slope"),
    ("service", "request_rate_slope"),
    ("service", "path_cpu_pressure_slope"),
    ("service", "path_queue_len_slope"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    return parser.parse_args()


def load_samples(run_dir: Path) -> list[dict]:
    samples: list[dict] = []
    for split in ["train", "val", "test"]:
        path = run_dir / "dataset" / f"{split}.pkl"
        if path.exists():
            split_samples = load_pickle(path)
            for sample in split_samples:
                sample = dict(sample)
                sample["_split"] = split
                samples.append(sample)
    return samples


def ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = np.empty_like(order, dtype=float)
    ranked[order] = np.arange(len(values), dtype=float)
    unique_values, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    _ = unique_values
    sums = np.bincount(inverse, ranked)
    return sums[inverse] / np.maximum(counts[inverse], 1)


def corr(x: list[float], y: list[float], method: str) -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2 or np.std(x_arr) <= 1e-12 or np.std(y_arr) <= 1e-12:
        return math.nan
    if method == "spearman":
        x_arr = ranks(x_arr)
        y_arr = ranks(y_arr)
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def feature_names(config: dict, group: str) -> list[str]:
    data_cfg = config.get("data", {})
    defaults = {
        "node": ["cpu_util", "mem_util", "queue_len", "available_cpu", "available_mem", "node_type", "node_degree", "is_current_edge"],
        "service": ["request_rate", "response_time", "service_type", "base_response_time", "dependency_count"],
    }
    return [str(name) for name in data_cfg.get(f"{group}_feature_names", defaults[group])]


def feature_value(sample: dict, group: str, name: str, names: list[str]) -> float:
    if name not in names:
        return math.nan
    idx = names.index(name)
    if group == "node":
        node_x = np.asarray(sample["node_x"], dtype=float)
        mask = np.asarray(sample.get("node_mask", np.ones(node_x.shape[1])), dtype=bool)
        if idx >= node_x.shape[-1] or not mask.any():
            return math.nan
        return float(np.mean(node_x[-1, mask, idx]))
    service_x = np.asarray(sample["service_x"], dtype=float)
    if idx >= service_x.shape[-1]:
        return math.nan
    return float(service_x[-1, idx])


def leakage_report(config: dict) -> dict:
    names = feature_names(config, "node") + feature_names(config, "service")
    leaked = [name for name in names if any(token in name.lower() for token in LEAKAGE_TOKENS)]
    return {"has_leakage_feature_name": bool(leaked), "leakage_feature_names": leaked}


def load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(run_dir: Path) -> tuple[dict, str]:
    config_path = run_dir / "config" / "resolved_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {config_path}")
    config = load_config(config_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    episodes_path = run_dir / "artifacts" / "cpu_precursor_episodes.csv"
    episodes = read_csv_rows(episodes_path) if episodes_path.exists() else []
    samples = load_samples(run_dir)
    node_names = feature_names(config, "node")
    service_names = feature_names(config, "service")

    valid_samples = [sample for sample in samples if int(sample.get("attr_mask", 0)) == 1]
    cpu_binary = [1.0 if int(sample.get("risk_metric", -100)) == 2 else 0.0 for sample in valid_samples]
    cpu_scores = [
        float(sample.get("risk_metric_scores", [0, 0, 0, 0, 0])[2])
        if len(sample.get("risk_metric_scores", [0, 0, 0, 0, 0])) >= 3
        else math.nan
        for sample in valid_samples
    ]

    correlations: dict[str, dict[str, float]] = {}
    for group, name in FEATURES_TO_AUDIT:
        names = node_names if group == "node" else service_names
        values = [feature_value(sample, group, name, names) for sample in valid_samples]
        correlations[f"{group}.{name}"] = {
            "pearson_with_future_cpu_label": corr(values, cpu_binary, "pearson"),
            "spearman_with_future_cpu_label": corr(values, cpu_binary, "spearman"),
            "pearson_with_future_cpu_score": corr(values, cpu_scores, "pearson"),
            "spearman_with_future_cpu_score": corr(values, cpu_scores, "spearman"),
        }

    leakage = leakage_report(config)
    pressure02_vs_labelrule = load_optional_json(run_dir / "analysis" / "metric_label_rule_vs_pressure02.json") or {}
    pressure02_baseline = load_optional_json(run_dir / "analysis" / "metric_evidence_baseline_pressure02.json") or {}
    affected_services = {
        service
        for row in episodes
        for service in str(row.get("affected_services", "")).split("|")
        if service != ""
    }
    affected_nodes = {row.get("node_id", "") for row in episodes if row.get("node_id")}
    provenance_ok = (
        manifest.get("effective_simulator_backend") == "edgesimpy_real"
        and bool(manifest.get("real_object_created", False))
        and bool(manifest.get("real_simulation_ran", False))
        and not bool(manifest.get("edgesimpy_adapter_fallback", False))
    )
    cpu_precursor_valid = bool(episodes) and not leakage["has_leakage_feature_name"] and provenance_ok
    report = {
        "run_dir": str(run_dir),
        "risk_injection": manifest.get("risk_injection", config.get("data", {}).get("risk_injection")),
        "cpu_precursor_enabled": bool(manifest.get("cpu_precursor_enabled", config.get("data", {}).get("cpu_precursor", {}).get("enabled", False))),
        "episode_count": len(episodes),
        "affected_node_count": len(affected_nodes),
        "affected_service_count": len(affected_services),
        "valid_attr_samples": len(valid_samples),
        "future_cpu_label_count": int(sum(cpu_binary)),
        "correlations": correlations,
        "pressure02_matches_label_rule_rate": pressure02_vs_labelrule.get("pressure02_matches_label_rule_rate"),
        "pressure02_cpu_recall": (pressure02_baseline.get("per_class_recall") or {}).get("cpu"),
        "leakage": leakage,
        "provenance": {
            "effective_simulator_backend": manifest.get("effective_simulator_backend"),
            "edgesimpy_backend_state": manifest.get("edgesimpy_backend_state"),
            "edgesimpy_adapter_fallback": manifest.get("edgesimpy_adapter_fallback"),
            "real_object_created": manifest.get("real_object_created"),
            "real_simulation_ran": manifest.get("real_simulation_ran"),
            "provenance_ok": provenance_ok,
        },
        "cpu_precursor_valid": cpu_precursor_valid,
    }

    lines = [
        "CPU precursor audit",
        f"run_dir: {run_dir}",
        f"risk_injection: {report['risk_injection']}",
        f"cpu_precursor_enabled: {report['cpu_precursor_enabled']}",
        f"episode_count: {len(episodes)}",
        f"affected_node_count: {len(affected_nodes)}",
        f"affected_service_count: {len(affected_services)}",
        f"valid_attr_samples: {len(valid_samples)}",
        f"future_cpu_label_count: {int(sum(cpu_binary))}",
        f"leakage_features: {leakage['leakage_feature_names']}",
        f"provenance_ok: {provenance_ok}",
        f"pressure02_matches_label_rule_rate: {report['pressure02_matches_label_rule_rate']}",
        f"pressure02_cpu_recall: {report['pressure02_cpu_recall']}",
        f"cpu_precursor_valid: {cpu_precursor_valid}",
    ]
    if not episodes:
        lines.append("结论：未找到 cpu_precursor_episodes.csv 或 episode 数为 0。")
    elif leakage["has_leakage_feature_name"]:
        lines.append("结论：发现疑似未来标签泄漏特征名，请先修复特征 schema。")
    elif not provenance_ok:
        lines.append("结论：EdgeSimPy strict provenance 未通过，不应作为真实 strict precursor run。")
    else:
        lines.append("结论：CPU precursor 注入和输入窗口特征审计通过基础检查。")
    return report, "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    report, text = build_report(run_dir)
    audit_dir = ensure_dir(run_dir / "audit")
    analysis_dir = ensure_dir(run_dir / "analysis")
    (audit_dir / "cpu_precursor_audit.txt").write_text(text, encoding="utf-8")
    (analysis_dir / "cpu_precursor_correlation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {audit_dir / 'cpu_precursor_audit.txt'}")
    print(f"Wrote {analysis_dir / 'cpu_precursor_correlation.json'}")


if __name__ == "__main__":
    main()
