from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import compute_attribution_majority_baseline
from src.utils.io import ensure_dir, load_pickle, read_csv_rows, write_csv_rows


SPLITS = ["train", "val", "test"]
LABELS = {0: "normal", 1: "risky", 2: "violated"}
SCENARIOS = ["normal", "burst", "node_overload", "link_congestion", "mixed"]
METRICS = {0: "delay", 1: "loss", 2: "cpu", 3: "queue", 4: "bandwidth"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/synthetic_full/dataset")
    parser.add_argument("--raw_dir", default="data/synthetic_full/raw_logs")
    parser.add_argument("--output_dir", default="results/synthetic_full/audit")
    return parser.parse_args()


def load_splits(dataset_dir: Path) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {}
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
    mapping: dict[tuple[int, int], str] = {}
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
            mapping.setdefault(key, row["scenario"])
    return mapping


def sample_scenario(sample: dict, scenario_map: dict[tuple[int, int], str]) -> str:
    if sample.get("scenario") not in [None, ""]:
        return str(sample["scenario"])
    key = (int(sample["time"]), int(sample["service_id"]))
    return scenario_map.get(key, "unknown")


def valid_attr_samples(samples: list[dict]) -> list[dict]:
    return [sample for sample in samples if int(sample.get("attr_mask", 0)) == 1]


def ratio(count: int, total: int) -> float:
    return count / total if total else math.nan


def label_distribution_rows(splits: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for split, samples in splits.items():
        counts = Counter(int(sample["risk_label"]) for sample in samples)
        total = len(samples)
        for label_id, label_name in LABELS.items():
            count = counts.get(label_id, 0)
            rows.append(
                {
                    "split": split,
                    "label_id": label_id,
                    "label": label_name,
                    "count": count,
                    "ratio": ratio(count, total),
                }
            )
    return rows


def scenario_distribution_rows(splits: dict[str, list[dict]], scenario_map: dict[tuple[int, int], str]) -> list[dict]:
    rows: list[dict] = []
    for split, samples in splits.items():
        counts = Counter(sample_scenario(sample, scenario_map) for sample in samples)
        total = len(samples)
        scenarios = sorted(set(SCENARIOS) | set(counts))
        for scenario in scenarios:
            count = counts.get(scenario, 0)
            rows.append(
                {
                    "split": split,
                    "scenario": scenario,
                    "count": count,
                    "ratio": ratio(count, total),
                    "present": int(count > 0),
                }
            )
    return rows


def label_by_scenario_rows(splits: dict[str, list[dict]], scenario_map: dict[tuple[int, int], str]) -> list[dict]:
    rows: list[dict] = []
    for split, samples in splits.items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        for sample in samples:
            grouped[sample_scenario(sample, scenario_map)].append(sample)
        for scenario in sorted(set(SCENARIOS) | set(grouped)):
            scenario_samples = grouped.get(scenario, [])
            total = len(scenario_samples)
            counts = Counter(int(sample["risk_label"]) for sample in scenario_samples)
            for label_id, label_name in LABELS.items():
                count = counts.get(label_id, 0)
                rows.append(
                    {
                        "split": split,
                        "scenario": scenario,
                        "label_id": label_id,
                        "label": label_name,
                        "count": count,
                        "ratio_within_scenario": ratio(count, total),
                    }
                )
    return rows


def attribution_distribution_rows(splits: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for split, samples in splits.items():
        valid = valid_attr_samples(samples)
        for field in ["risk_node", "risk_link", "risk_metric"]:
            counts = Counter(int(sample[field]) for sample in valid)
            total = len(valid)
            values = sorted(counts)
            if field == "risk_metric":
                values = sorted(set(METRICS) | set(values))
            for value in values:
                count = counts.get(value, 0)
                rows.append(
                    {
                        "split": split,
                        "field": field,
                        "value": value,
                        "value_name": METRICS.get(value, str(value)) if field == "risk_metric" else str(value),
                        "count": count,
                        "ratio": ratio(count, total),
                    }
                )
    return rows


def _csv_safe_baseline(row: dict) -> dict:
    out = dict(row)
    for key, value in list(out.items()):
        if isinstance(value, dict):
            out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return out


def attribution_majority_baseline_rows(splits: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    all_samples = [sample for split in SPLITS for sample in splits[split]]
    for split, samples in {**splits, "all": all_samples}.items():
        valid_row = compute_attribution_majority_baseline(samples, split=split, attr_mask_only=True)
        global_row = compute_attribution_majority_baseline(samples, split=split, attr_mask_only=False)
        valid_row.update(
            {
                "audit_global_metric_majority_acc": global_row["metric_majority_acc"],
                "audit_global_metric_majority_value": global_row["metric_majority_value"],
                "audit_global_metric_majority_name": global_row["metric_majority_name"],
                "audit_global_denominator": global_row["denominator"],
            }
        )
        rows.append(_csv_safe_baseline(valid_row))
    return rows


def split_label_ratios(label_rows: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for row in label_rows:
        out[row["split"]][row["label"]] = float(row["ratio"])
    return out


def split_scenario_ratios(scenario_rows: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for row in scenario_rows:
        out[row["split"]][row["scenario"]] = float(row["ratio"])
    return out


def split_shift_report_rows(label_rows: list[dict], scenario_rows: list[dict], attr_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    label_ratios = split_label_ratios(label_rows)
    scenario_ratios = split_scenario_ratios(scenario_rows)

    train_violated = label_ratios["train"].get("violated", 0.0)
    test_violated = label_ratios["test"].get("violated", 0.0)
    rows.append(
        {
            "check": "high_label_shift",
            "status": "fail" if test_violated - train_violated > 0.20 else "pass",
            "detail": f"test violated ratio - train violated ratio = {test_violated - train_violated:.4f}",
            "value": test_violated - train_violated,
            "threshold": 0.20,
        }
    )

    for label in LABELS.values():
        values = [label_ratios[split].get(label, 0.0) for split in SPLITS]
        spread = max(values) - min(values)
        rows.append(
            {
                "check": f"label_ratio_spread_{label}",
                "status": "fail" if spread > 0.20 else "pass",
                "detail": f"{label} ratio spread across train/val/test = {spread:.4f}",
                "value": spread,
                "threshold": 0.20,
            }
        )

    for split in SPLITS:
        missing = [scenario for scenario in SCENARIOS if scenario_ratios[split].get(scenario, 0.0) == 0.0]
        max_scenario = max(scenario_ratios[split].items(), key=lambda item: item[1]) if scenario_ratios[split] else ("", 0.0)
        rows.append(
            {
                "check": f"scenario_coverage_{split}",
                "status": "fail" if missing else "pass",
                "detail": "missing=" + "|".join(missing) if missing else "all scenarios present",
                "value": len(missing),
                "threshold": 0,
            }
        )
        rows.append(
            {
                "check": f"scenario_block_bias_{split}",
                "status": "fail" if max_scenario[1] > 0.45 else "pass",
                "detail": f"top scenario {max_scenario[0]} ratio = {max_scenario[1]:.4f}",
                "value": max_scenario[1],
                "threshold": 0.45,
            }
        )

    attr_grouped: dict[str, Counter] = {"risk_node": Counter(), "risk_link": Counter(), "risk_metric": Counter()}
    for row in attr_rows:
        attr_grouped[row["field"]][row["value"]] += int(row["count"])
    for field, check_name in [("risk_node", "node_attribution_concentration"), ("risk_link", "link_attribution_concentration")]:
        total = sum(attr_grouped[field].values())
        top_value, top_count = attr_grouped[field].most_common(1)[0] if total else ("", 0)
        top_ratio = top_count / total if total else math.nan
        rows.append(
            {
                "check": check_name,
                "status": "fail" if total and top_ratio > 0.50 else "pass",
                "detail": f"top {field}={top_value}, ratio={top_ratio:.4f}" if total else "no valid attribution samples",
                "value": top_ratio,
                "threshold": 0.50,
            }
        )

    metric_total = sum(attr_grouped["risk_metric"].values())
    for metric_id, metric_name in METRICS.items():
        metric_ratio = attr_grouped["risk_metric"].get(metric_id, 0) / metric_total if metric_total else 0.0
        rows.append(
            {
                "check": f"metric_coverage_{metric_name}",
                "status": "fail" if metric_ratio == 0.0 else "pass",
                "detail": f"{metric_name} ratio = {metric_ratio:.4f}",
                "value": metric_ratio,
                "threshold": ">0",
            }
        )
    metric_by_split: dict[str, Counter] = {split: Counter() for split in SPLITS}
    for row in attr_rows:
        if row["field"] == "risk_metric":
            metric_by_split[row["split"]][int(row["value"])] += int(row["count"])
    for split in SPLITS:
        split_total = sum(metric_by_split[split].values())
        for metric_id, metric_name in METRICS.items():
            split_metric_ratio = metric_by_split[split].get(metric_id, 0) / split_total if split_total else 0.0
            rows.append(
                {
                    "check": f"metric_coverage_{split}_{metric_name}",
                    "status": "fail" if split_metric_ratio == 0.0 else "pass",
                    "detail": f"{split} {metric_name} ratio = {split_metric_ratio:.4f}",
                    "value": split_metric_ratio,
                    "threshold": ">0",
                }
            )
        queue_ratio = metric_by_split[split].get(3, 0) / split_total if split_total else 0.0
        rows.append(
            {
                "check": f"queue_metric_min_ratio_{split}",
                "status": "fail" if queue_ratio < 0.05 else "pass",
                "detail": f"{split} queue ratio = {queue_ratio:.4f}",
                "value": queue_ratio,
                "threshold": 0.05,
            }
        )
    return rows


def report_passed(shift_rows: list[dict]) -> bool:
    return all(row["status"] == "pass" for row in shift_rows)


def load_run_manifest_for_report(output_path: Path) -> dict:
    manifest_path = output_path.parent.parent / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_report(output_path: Path, label_rows: list[dict], scenario_rows: list[dict], majority_rows: list[dict], shift_rows: list[dict]) -> None:
    label_ratios = split_label_ratios(label_rows)
    scenario_ratios = split_scenario_ratios(scenario_rows)
    passed = report_passed(shift_rows)
    failed = [row for row in shift_rows if row["status"] == "fail"]
    all_majority = next(row for row in majority_rows if row["split"] == "all")

    lines = [
        "SPARTA synthetic full 数据审计报告",
        "",
        f"结论：{'通过正式实验数据要求' if passed else '未通过正式实验数据要求'}。",
        "",
        "标签分布：",
    ]
    for split in SPLITS:
        ratios = label_ratios[split]
        lines.append(
            f"- {split}: normal={ratios.get('normal', 0.0):.2%}, "
            f"risky={ratios.get('risky', 0.0):.2%}, violated={ratios.get('violated', 0.0):.2%}"
        )

    lines.extend(["", "场景分布："])
    for split in SPLITS:
        parts = [f"{scenario}={scenario_ratios[split].get(scenario, 0.0):.2%}" for scenario in SCENARIOS]
        lines.append(f"- {split}: " + ", ".join(parts))

    lines.extend(
        [
            "",
            "归因 majority baseline：",
            f"- node_majority_acc={float(all_majority['node_majority_acc']):.2%}, "
            f"link_majority_acc={float(all_majority['link_majority_acc']):.2%}, "
            f"metric_majority_acc={float(all_majority['metric_majority_acc']):.2%}",
            "",
            "失败项：" if failed else "未发现失败项：",
        ]
    )
    if failed:
        lines.extend(f"- {row['check']}: {row['detail']}" for row in failed)
    else:
        lines.append("- 所有核心审计规则均通过。")

    lines.extend(
        [
            "",
            "中文总结：",
        ]
    )
    if passed:
        lines.append("当前 synthetic full 数据在标签分布、场景覆盖、归因分布和 split shift 上满足正式实验数据的最低要求。")
    else:
        lines.append(
            "当前 synthetic full 数据暂不建议作为正式实验数据直接使用。优先修复 split 间标签比例漂移、场景块偏置或归因类别过度集中问题，"
            "否则模型可能学到场景标签或固定瓶颈编号，而不是学习 SLA 风险随时间演化的规律。"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_v2(output_path: Path, label_rows: list[dict], scenario_rows: list[dict], majority_rows: list[dict], shift_rows: list[dict]) -> None:
    label_ratios = split_label_ratios(label_rows)
    scenario_ratios = split_scenario_ratios(scenario_rows)
    passed = report_passed(shift_rows)
    failed = [row for row in shift_rows if row["status"] == "fail"]
    metric_only_failures = bool(failed) and all(str(row["check"]).startswith("metric_coverage") for row in failed)
    all_majority = next(row for row in majority_rows if row["split"] == "all")
    manifest = load_run_manifest_for_report(output_path)
    effective_backend = manifest.get("effective_simulator_backend", "unknown")
    effective_backend_at_generation = manifest.get("effective_simulator_backend_at_generation", effective_backend)
    provenance_path = manifest.get("raw_log_generation_provenance", "")
    edgesimpy_backend_state = manifest.get("edgesimpy_backend_state", "")
    real_object_created = manifest.get("real_object_created", manifest.get("real_edgesimpy_objects_created", ""))
    real_simulation_ran = manifest.get("real_simulation_ran", "")

    lines = [
        "SPARTA synthetic full 数据审计报告",
        "",
        f"effective_simulator_backend: {effective_backend}",
        f"effective_simulator_backend_at_generation: {effective_backend_at_generation}",
        f"edgesimpy_backend_state: {edgesimpy_backend_state}",
        f"real_object_created: {real_object_created}",
        f"real_simulation_ran: {real_simulation_ran}",
        f"raw_log_generation_provenance: {provenance_path}",
        "",
        f"结论：{'通过正式实验数据要求' if passed else '未通过正式实验数据要求'}。",
        "",
        "标签分布：",
    ]
    for split in SPLITS:
        ratios = label_ratios[split]
        lines.append(
            f"- {split}: normal={ratios.get('normal', 0.0):.2%}, "
            f"risky={ratios.get('risky', 0.0):.2%}, violated={ratios.get('violated', 0.0):.2%}"
        )

    lines.extend(["", "场景分布："])
    for split in SPLITS:
        parts = [f"{scenario}={scenario_ratios[split].get(scenario, 0.0):.2%}" for scenario in SCENARIOS]
        lines.append(f"- {split}: " + ", ".join(parts))

    lines.extend(
        [
            "",
            "归因 majority baseline：",
            f"- node_majority_acc={float(all_majority['node_majority_acc']):.2%}, "
            f"link_majority_acc={float(all_majority['link_majority_acc']):.2%}, "
            f"metric_majority_acc={float(all_majority['metric_majority_acc']):.2%}",
            "",
            "失败项：" if failed else "未发现失败项：",
        ]
    )
    if failed:
        lines.extend(f"- {row['check']}: {row['detail']}" for row in failed)
    else:
        lines.append("- 所有核心审计规则均通过。")

    lines.extend(["", "中文总结："])
    if passed:
        lines.append("当前数据在标签分布、场景覆盖、归因分布和 split shift 上满足正式实验数据的最低要求。")
    elif metric_only_failures:
        lines.append(
            "标签分布和场景分布基本合格，当前失败主要来自 risk_metric 某类覆盖不足。"
            "优先修复数据生成器或 adapter 的风险指标归因覆盖，不应将该问题泛化为标签漂移或场景块偏置。"
        )
    else:
        lines.append(
            "当前数据暂不建议作为正式实验数据直接使用。优先修复 split 间标签比例漂移、场景块偏置或归因类别过度集中问题，"
            "否则模型可能学到场景标签或固定瓶颈编号，而不是 SLA 风险随时间演化的规律。"
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
    scenario_rows = scenario_distribution_rows(splits, scenario_map)
    label_scenario_rows = label_by_scenario_rows(splits, scenario_map)
    attr_rows = attribution_distribution_rows(splits)
    majority_rows = attribution_majority_baseline_rows(splits)
    shift_rows = split_shift_report_rows(label_rows, scenario_rows, attr_rows)

    write_csv_rows(output_dir / "label_distribution.csv", label_rows)
    write_csv_rows(output_dir / "scenario_distribution.csv", scenario_rows)
    write_csv_rows(output_dir / "label_by_scenario.csv", label_scenario_rows)
    write_csv_rows(output_dir / "attribution_distribution.csv", attr_rows)
    write_csv_rows(output_dir / "attribution_majority_baseline.csv", majority_rows)
    write_csv_rows(output_dir / "split_shift_report.csv", shift_rows)
    write_report_v2(output_dir / "audit_report.txt", label_rows, scenario_rows, majority_rows, shift_rows)
    print(f"Wrote enhanced synthetic full audit outputs to {output_dir}")


if __name__ == "__main__":
    main()
