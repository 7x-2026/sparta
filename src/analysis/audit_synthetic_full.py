from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_pickle, read_csv_rows, write_csv_rows


LABEL_NAMES = {0: "normal", 1: "risky", 2: "violated"}
METRIC_NAMES = {0: "delay", 1: "loss", 2: "cpu", 3: "queue", 4: "bandwidth", -100: "ignored"}
SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/synthetic_full/dataset")
    parser.add_argument("--raw_dir", default="data/synthetic_full/raw_logs")
    parser.add_argument("--output_dir", default="results/synthetic_full/audit")
    return parser.parse_args()


def load_splits(dataset_dir: Path) -> dict[str, list[dict]]:
    splits = {}
    for split in SPLITS:
        path = dataset_dir / f"{split}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset split: {path}")
        samples = load_pickle(path)
        if not isinstance(samples, list):
            raise TypeError(f"{path} must contain a list of sample dicts")
        splits[split] = samples
    return splits


def load_scenario_map(raw_dir: Path) -> dict[tuple[int, int], str]:
    scenario_map: dict[tuple[int, int], str] = {}
    for filename in ["path_log.csv", "service_log.csv"]:
        path = raw_dir / filename
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            if "scenario" not in row:
                continue
            try:
                key = (int(float(row["time"])), int(float(row["service_id"])))
            except (KeyError, TypeError, ValueError):
                continue
            scenario_map.setdefault(key, row["scenario"])
    return scenario_map


def sample_scenario(sample: dict, scenario_map: dict[tuple[int, int], str]) -> str:
    if sample.get("scenario") not in [None, ""]:
        return str(sample["scenario"])
    key = (int(sample["time"]), int(sample["service_id"]))
    return scenario_map.get(key, "unknown")


def label_distribution_rows(splits: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for split, samples in splits.items():
        counts = Counter(int(sample["risk_label"]) for sample in samples)
        total = len(samples)
        for label, name in LABEL_NAMES.items():
            count = counts.get(label, 0)
            rows.append(
                {
                    "split": split,
                    "label": name,
                    "label_id": label,
                    "count": count,
                    "ratio": count / total if total else math.nan,
                }
            )
    return rows


def attribution_distribution_rows(splits: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    fields = ["risk_node", "risk_link", "risk_metric"]
    for split, samples in splits.items():
        valid_samples = [sample for sample in samples if int(sample.get("attr_mask", 0)) == 1]
        for field in fields:
            counts = Counter(int(sample[field]) for sample in valid_samples)
            total = len(valid_samples)
            for value, count in sorted(counts.items()):
                value_name = METRIC_NAMES.get(value, str(value)) if field == "risk_metric" else str(value)
                rows.append(
                    {
                        "split": split,
                        "field": field,
                        "value": value,
                        "value_name": value_name,
                        "count": count,
                        "ratio": count / total if total else math.nan,
                    }
                )
    return rows


def scenario_distribution_rows(splits: dict[str, list[dict]], scenario_map: dict[tuple[int, int], str]) -> list[dict]:
    rows: list[dict] = []
    for split, samples in splits.items():
        counts = Counter(sample_scenario(sample, scenario_map) for sample in samples)
        total = len(samples)
        for scenario, count in sorted(counts.items()):
            rows.append({"split": split, "scenario": scenario, "count": count, "ratio": count / total if total else math.nan})
    return rows


def time_range_rows(splits: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for split, samples in splits.items():
        times = [int(sample["time"]) for sample in samples]
        rows.append(
            {
                "split": split,
                "min_time": min(times) if times else "",
                "max_time": max(times) if times else "",
                "num_samples": len(samples),
                "num_unique_times": len(set(times)),
            }
        )
    return rows


def _array_summary(arrays: Iterable[np.ndarray]) -> dict:
    arr = np.concatenate([np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays])
    finite = np.isfinite(arr)
    finite_arr = arr[finite]
    return {
        "mean": float(finite_arr.mean()) if finite_arr.size else math.nan,
        "std": float(finite_arr.std()) if finite_arr.size else math.nan,
        "min": float(finite_arr.min()) if finite_arr.size else math.nan,
        "max": float(finite_arr.max()) if finite_arr.size else math.nan,
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "num_values": int(arr.size),
    }


def feature_summary_rows(splits: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for split, samples in splits.items():
        for field in ["node_x", "link_x", "service_x", "sla_x"]:
            summary = _array_summary(sample[field] for sample in samples)
            rows.append({"split": split, "feature": field, **summary})
    return rows


def path_size_summary_rows(splits: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for split, samples in splits.items():
        node_counts = Counter(int(np.asarray(sample["node_mask"]).sum()) for sample in samples)
        link_counts = Counter(int(np.asarray(sample["link_mask"]).sum()) for sample in samples)
        for size, count in sorted(node_counts.items()):
            rows.append({"split": split, "kind": "nodes", "size": size, "count": count, "ratio": count / len(samples) if samples else math.nan})
        for size, count in sorted(link_counts.items()):
            rows.append({"split": split, "kind": "links", "size": size, "count": count, "ratio": count / len(samples) if samples else math.nan})
    return rows


def _label_issue(label_rows: list[dict]) -> tuple[list[str], dict[str, float]]:
    all_counts = Counter()
    for row in label_rows:
        all_counts[row["label"]] += int(row["count"])
    total = sum(all_counts.values())
    ratios = {label: count / total if total else 0.0 for label, count in all_counts.items()}
    issues = []
    if ratios and max(ratios.values()) > 0.80:
        issues.append(f"类别极端不平衡：最大类别占比 {max(ratios.values()):.2%}。")
    risky_violated_ratio = ratios.get("risky", 0.0) + ratios.get("violated", 0.0)
    if ratios.get("risky", 0.0) < 0.05 or ratios.get("violated", 0.0) < 0.05:
        issues.append(f"risky 或 violated 过少：risky={ratios.get('risky', 0.0):.2%}, violated={ratios.get('violated', 0.0):.2%}。")
    elif risky_violated_ratio < 0.20:
        issues.append(f"risky/violated 总占比偏低：{risky_violated_ratio:.2%}。")
    return issues, ratios


def _concentration_issue(attr_rows: list[dict]) -> list[str]:
    issues = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in attr_rows:
        grouped[row["field"]].append(row)
    for field in ["risk_node", "risk_link"]:
        rows = grouped.get(field, [])
        if not rows:
            issues.append(f"{field} 没有有效归因样本。")
            continue
        total_by_value = Counter()
        for row in rows:
            total_by_value[int(row["value"])] += int(row["count"])
        total = sum(total_by_value.values())
        value, count = total_by_value.most_common(1)[0]
        ratio = count / total if total else 0.0
        if ratio > 0.70:
            issues.append(f"{field} 过度集中：编号 {value} 占 {ratio:.2%}。")
    return issues


def _time_overlap_issue(time_rows: list[dict]) -> list[str]:
    ranges = {row["split"]: (int(row["min_time"]), int(row["max_time"])) for row in time_rows if row["min_time"] != ""}
    issues = []
    pairs = [("train", "val"), ("val", "test"), ("train", "test")]
    for left, right in pairs:
        if left not in ranges or right not in ranges:
            continue
        l_min, l_max = ranges[left]
        r_min, r_max = ranges[right]
        if max(l_min, r_min) <= min(l_max, r_max):
            issues.append(f"{left}/{right} 时间范围重叠：{ranges[left]} vs {ranges[right]}。")
    ordered = ["train", "val", "test"]
    if all(split in ranges for split in ordered):
        if not (ranges["train"][1] < ranges["val"][0] <= ranges["val"][1] < ranges["test"][0]):
            issues.append(f"时间划分不是严格递增：train={ranges['train']}, val={ranges['val']}, test={ranges['test']}。")
    return issues


def _feature_issue(feature_rows: list[dict]) -> list[str]:
    issues = []
    bad = [row for row in feature_rows if int(row["nan_count"]) > 0 or int(row["inf_count"]) > 0]
    for row in bad:
        issues.append(f"{row['split']} {row['feature']} 存在 NaN/inf：nan={row['nan_count']}, inf={row['inf_count']}。")
    return issues


def _path_size_issue(path_rows: list[dict]) -> list[str]:
    issues = []
    for row in path_rows:
        size = int(row["size"])
        if row["kind"] == "nodes" and (size <= 0 or size > 8):
            issues.append(f"{row['split']} 节点数异常：Np={size}。")
        if row["kind"] == "links" and (size <= 0 or size > 12):
            issues.append(f"{row['split']} 链路数异常：Ep={size}。")
    return issues


def _scenario_shortcut_issue(scenario_rows: list[dict], label_rows: list[dict]) -> list[str]:
    issues = []
    scenarios = {row["scenario"] for row in scenario_rows if row["scenario"] != "unknown"}
    if not scenarios:
        return ["样本没有 scenario 字段，且 raw logs 无法映射 scenario；暂不能判断场景捷径风险。"]
    split_scenario_counts = defaultdict(Counter)
    for row in scenario_rows:
        split_scenario_counts[row["split"]][row["scenario"]] += int(row["count"])
    for split, counts in split_scenario_counts.items():
        total = sum(counts.values())
        if total and counts.most_common(1)[0][1] / total > 0.85:
            scenario, count = counts.most_common(1)[0]
            issues.append(f"{split} 几乎由单一场景 {scenario} 构成，占 {count / total:.2%}，模型可能学到场景标签。")
    if len(scenarios) >= 3:
        issues.append(
            "存在多个连续场景块。若时间划分刚好切开场景，SPARTA 可能部分利用场景/时间块差异，建议后续加入跨场景混合切分或每个 split 覆盖所有场景。"
        )
    return issues


def write_audit_report(
    output_path: Path,
    label_rows: list[dict],
    attr_rows: list[dict],
    scenario_rows: list[dict],
    time_rows: list[dict],
    feature_rows: list[dict],
    path_rows: list[dict],
) -> None:
    issues = []
    label_issues, ratios = _label_issue(label_rows)
    issues.extend(label_issues)
    issues.extend(_concentration_issue(attr_rows))
    issues.extend(_time_overlap_issue(time_rows))
    issues.extend(_feature_issue(feature_rows))
    issues.extend(_path_size_issue(path_rows))
    issues.extend(_scenario_shortcut_issue(scenario_rows, label_rows))

    ok_lines = []
    if not label_issues:
        ok_lines.append("类别分布未触发极端不平衡阈值，risky/violated 数量充足。")
    if not _time_overlap_issue(time_rows):
        ok_lines.append("train/val/test 时间范围未重叠，符合时间顺序划分。")
    if not _feature_issue(feature_rows):
        ok_lines.append("node_x/link_x/service_x/sla_x 未发现 NaN 或 inf。")
    if not _path_size_issue(path_rows):
        ok_lines.append("节点数和链路数均在 v0 synthetic full 预期范围内。")

    lines = [
        "SPARTA synthetic full data audit",
        "",
        "Label ratios:",
        *(f"- {label}: {ratio:.2%}" for label, ratio in sorted(ratios.items())),
        "",
        "Passed checks:",
        *(f"- {line}" for line in ok_lines),
        "",
        "Potential issues:",
    ]
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- 未发现明显数据质量问题。")
    lines.extend(
        [
            "",
            "Interpretation note:",
            "- 该脚本只做数据质量审计，不代表模型效果结论。若 scenario 与 label 高度绑定，后续实验应增加更细粒度场景混合、跨场景泛化或按场景分层评估。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = ensure_dir(args.output_dir)

    splits = load_splits(dataset_dir)
    scenario_map = load_scenario_map(raw_dir)

    label_rows = label_distribution_rows(splits)
    attr_rows = attribution_distribution_rows(splits)
    scenario_rows = scenario_distribution_rows(splits, scenario_map)
    time_rows = time_range_rows(splits)
    feature_rows = feature_summary_rows(splits)
    path_rows = path_size_summary_rows(splits)

    write_csv_rows(output_dir / "label_distribution.csv", label_rows)
    write_csv_rows(output_dir / "attribution_distribution.csv", attr_rows)
    write_csv_rows(output_dir / "scenario_distribution.csv", scenario_rows)
    write_csv_rows(output_dir / "time_range.csv", time_rows)
    write_csv_rows(output_dir / "feature_summary.csv", feature_rows)
    write_csv_rows(output_dir / "path_size_summary.csv", path_rows)
    write_audit_report(output_dir / "audit_report.txt", label_rows, attr_rows, scenario_rows, time_rows, feature_rows, path_rows)
    print(f"Wrote synthetic full audit outputs to {output_dir}")


if __name__ == "__main__":
    main()
