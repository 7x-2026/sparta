from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.group_manager import create_group_dir, save_group_manifest, write_run_list
from src.utils.io import ensure_dir


MODELS = ["lstm", "transformer", "sparta"]
CLASSIFICATION_METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_acc",
    "risk_recall",
    "violation_recall",
    "auc",
]
ATTR_METRICS = ["node_attr_acc", "link_attr_acc", "metric_attr_acc"]
METRICS = CLASSIFICATION_METRICS + ATTR_METRICS
WINNER_METRICS = CLASSIFICATION_METRICS + ATTR_METRICS + ["overall_score"]
OVERALL_WEIGHTS = {
    "macro_f1": 0.30,
    "balanced_acc": 0.25,
    "auc": 0.20,
    "violation_recall": 0.15,
    "risk_recall": 0.10,
}

SUMMARY_COLUMNS = [
    "group_id",
    "run_id",
    "run_dir",
    "data_seed",
    "train_seed",
    "model",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_acc",
    "risk_recall",
    "violation_recall",
    "auc",
    "node_attr_acc",
    "link_attr_acc",
    "metric_attr_acc",
    "result_source_file",
    "checkpoint_path",
    "dataset_path",
]

MEAN_STD_COLUMNS = ["group_id", "model"] + [f"{metric}_{suffix}" for metric in METRICS for suffix in ["mean", "std"]]
SEED_WINNER_COLUMNS = [
    "run_id",
    "data_seed",
    "train_seed",
    "metric",
    "best_model",
    "best_value",
    "lstm_value",
    "transformer_value",
    "sparta_value",
    "sparta_rank",
    "baseline_has_no_attribution",
]
PER_SEED_WINNER_COLUMNS = ["group_id"] + SEED_WINNER_COLUMNS
WINNER_COUNT_COLUMNS = ["group_id", "metric", "lstm_wins", "transformer_wins", "sparta_wins", "total_seeds"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_ids", nargs="+", default=None)
    parser.add_argument("--run_list", default=None)
    parser.add_argument("--group_name", default="multiseed_synthetic_full")
    parser.add_argument("--output_root", default="run_groups")
    parser.add_argument("--output_dir", default=None, help="Use an existing group directory instead of creating a new one.")
    parser.add_argument("--output", default=None, help="Backward-compatible explicit path for multiseed_summary.csv.")
    parser.add_argument("--run_root", default="run")
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--dedupe", dest="dedupe", action="store_true", default=True)
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str], overwrite: bool = False) -> None:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])


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


def format_value(value: float | str) -> float | str:
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    return value


def parse_created_at(value: str) -> datetime:
    if not value:
        return datetime.min
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return datetime.min


def resolve_run_path(value: str, base: Path | None = None) -> Path:
    path = Path(value.strip())
    if path.is_absolute():
        return path
    root_candidate = ROOT / path
    if root_candidate.exists():
        return root_candidate
    if base is not None:
        base_candidate = base.parent / path
        if base_candidate.exists():
            return base_candidate
    return root_candidate


def run_dirs_from_run_list(run_list_path: Path) -> list[Path]:
    if not run_list_path.exists():
        raise FileNotFoundError(f"Missing run_list: {run_list_path}")
    lines = [line.strip() for line in run_list_path.read_text(encoding="utf-8").splitlines()]
    return [resolve_run_path(line, run_list_path) for line in lines if line and not line.startswith("#")]


def run_dirs_from_scan(run_root: Path, pattern: str | None) -> list[Path]:
    root = run_root if run_root.is_absolute() else ROOT / run_root
    if not root.exists():
        return []
    if not pattern:
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and pattern in path.name)


def select_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_ids:
        return [resolve_run_path(value) for value in args.run_ids]
    if args.run_list:
        return run_dirs_from_run_list(Path(args.run_list))
    return run_dirs_from_scan(Path(args.run_root), args.pattern)


def prepare_group_dir(args: argparse.Namespace) -> tuple[Path, bool]:
    if args.output_dir:
        group_dir = Path(args.output_dir)
        ensure_dir(group_dir)
        ensure_dir(group_dir / "logs")
        return group_dir, False
    if args.output:
        group_dir = Path(args.output).parent
        ensure_dir(group_dir)
        ensure_dir(group_dir / "logs")
        return group_dir, False
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return create_group_dir(output_root, args.group_name), True


def log_line(log_path: Path, text: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def log_many(log_path: Path, lines: list[str]) -> None:
    for line in lines:
        log_line(log_path, line)


def output_paths(args: argparse.Namespace, group_dir: Path) -> dict[str, Path]:
    if args.output:
        summary = Path(args.output)
        return {
            "summary": summary,
            "mean_std": summary.with_name("multiseed_mean_std.csv"),
            "per_seed_winner": summary.with_name("per_seed_winner_report.csv"),
            "metric_winner_counts": summary.with_name("metric_winner_counts.csv"),
        }
    return {
        "summary": group_dir / "multiseed_summary.csv",
        "mean_std": group_dir / "multiseed_mean_std.csv",
        "per_seed_winner": group_dir / "per_seed_winner_report.csv",
        "metric_winner_counts": group_dir / "metric_winner_counts.csv",
    }


def model_result_map(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        model = row.get("model") or row.get("method")
        if model:
            out[str(model)] = row
    return out


def collect_rows_for_run(run_dir: Path, group_id: str, log_path: Path) -> list[dict]:
    manifest_path = run_dir / "run_manifest.json"
    result_path = run_dir / "results" / "all_test_results.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    if not result_path.exists():
        raise FileNotFoundError(f"Missing all_test_results.csv: {result_path}")
    manifest = read_json(manifest_path)
    data_seed = manifest.get("data_seed", manifest.get("seed", ""))
    train_seed = manifest.get("train_seed", "")
    log_line(log_path, f"RUN {run_dir}: data_seed={data_seed}, train_seed={train_seed}")
    log_line(log_path, f"READ {result_path}")

    rows: list[dict] = []
    for result in read_csv_rows(result_path):
        row = {
            "group_id": group_id,
            "run_id": manifest.get("run_id", run_dir.name),
            "run_dir": str(run_dir),
            "data_seed": data_seed,
            "train_seed": train_seed,
            "model": result.get("model") or result.get("method") or "",
            "result_source_file": result.get("result_source_file", str(result_path)),
            "checkpoint_path": result.get("checkpoint_path", ""),
            "dataset_path": result.get("dataset_path", ""),
            "_created_at": manifest.get("created_at", ""),
            "_run_dir_path": str(run_dir),
        }
        for metric in METRICS:
            row[metric] = result.get(metric, "")
        rows.append(row)
    return rows


def dedupe_summary_rows(rows: list[dict], log_path: Path, enabled: bool) -> list[dict]:
    if not enabled:
        return rows
    kept: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (str(row.get("data_seed", "")), str(row.get("train_seed", "")), str(row.get("model", "")))
        prev = kept.get(key)
        if prev is None:
            kept[key] = row
            continue
        prev_time = parse_created_at(str(prev.get("_created_at", "")))
        row_time = parse_created_at(str(row.get("_created_at", "")))
        if row_time >= prev_time:
            kept[key] = row
            dropped = prev
            kept_row = row
        else:
            dropped = row
            kept_row = prev
        log_line(
            log_path,
            "WARNING: duplicated key "
            f"(data_seed={key[0]}, train_seed={key[1]}, model={key[2]})",
        )
        log_line(log_path, f"kept: {kept_row.get('_run_dir_path', '')}")
        log_line(log_path, f"dropped: {dropped.get('_run_dir_path', '')}")
    return list(kept.values())


def clean_summary_rows(rows: list[dict]) -> list[dict]:
    return [{key: row.get(key, "") for key in SUMMARY_COLUMNS} for row in rows]


def expected_summary_rows(rows: list[dict]) -> int:
    seed_pairs = {(str(row.get("data_seed", "")), str(row.get("train_seed", ""))) for row in rows}
    return len(seed_pairs) * len(MODELS)


def validate_summary_rows(rows: list[dict], strict: bool) -> None:
    if not strict:
        return
    expected = expected_summary_rows(rows)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} summary rows, got {len(rows)}")
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(str(row["data_seed"]), str(row["train_seed"]))].add(str(row["model"]))
    for key, models in grouped.items():
        missing = sorted(set(MODELS) - models)
        if missing:
            raise RuntimeError(f"Missing models for data_seed/train_seed={key}: {missing}")


def build_mean_std_rows(summary_rows: list[dict], group_id: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row.get("model", ""))].append(row)

    out_rows: list[dict] = []
    for model in sorted(grouped):
        out: dict = {"group_id": group_id, "model": model}
        model_rows = grouped[model]
        for metric in METRICS:
            values = [parse_float(row.get(metric)) for row in model_rows]
            values = [value for value in values if not math.isnan(value)]
            if values:
                out[f"{metric}_mean"] = statistics.mean(values)
                out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                out[f"{metric}_mean"] = "N/A"
                out[f"{metric}_std"] = "N/A"
        out_rows.append(out)
    return out_rows


def overall_score(row: dict) -> float:
    total = 0.0
    for metric, weight in OVERALL_WEIGHTS.items():
        value = parse_float(row.get(metric))
        if math.isnan(value):
            return math.nan
        total += weight * value
    return total


def rank_for_sparta(values: dict[str, float]) -> int | str:
    sparta_value = values.get("sparta", math.nan)
    if math.isnan(sparta_value):
        return ""
    ranked = sorted(((value, model) for model, value in values.items() if not math.isnan(value)), reverse=True)
    for idx, (_, model) in enumerate(ranked, start=1):
        if model == "sparta":
            return idx
    return ""


def value_for_model(model_rows: dict[str, dict], model: str, metric: str) -> float:
    if metric == "overall_score":
        return overall_score(model_rows.get(model, {}))
    return parse_float(model_rows.get(model, {}).get(metric))


def best_model_for_metric(model_rows: dict[str, dict], metric: str) -> tuple[str, float, dict[str, float], int | str, bool]:
    values = {model: value_for_model(model_rows, model, metric) for model in MODELS}
    if metric in ATTR_METRICS:
        return "sparta", values.get("sparta", math.nan), values, 1 if not math.isnan(values.get("sparta", math.nan)) else "", True
    candidates = [(value, model) for model, value in values.items() if not math.isnan(value)]
    if not candidates:
        return "", math.nan, values, "", False
    best_value, best_model = max(candidates, key=lambda item: item[0])
    return best_model, best_value, values, rank_for_sparta(values), False


def build_seed_winner_rows(summary_rows: list[dict], group_id: str) -> tuple[list[dict], dict[str, list[dict]]]:
    grouped: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in summary_rows:
        grouped[
            (
                str(row.get("run_id", "")),
                str(row.get("run_dir", "")),
                str(row.get("data_seed", "")),
                str(row.get("train_seed", "")),
            )
        ].append(row)

    per_seed_rows: list[dict] = []
    per_run_rows: dict[str, list[dict]] = {}
    for (run_id, run_dir, data_seed, train_seed), rows in sorted(grouped.items()):
        model_rows = model_result_map(rows)
        seed_rows: list[dict] = []
        for metric in WINNER_METRICS:
            best_model, best_value, values, sparta_rank, baseline_attr = best_model_for_metric(model_rows, metric)
            row = {
                "run_id": run_id,
                "data_seed": data_seed,
                "train_seed": train_seed,
                "metric": metric,
                "best_model": best_model,
                "best_value": format_value(best_value),
                "lstm_value": format_value(values.get("lstm", math.nan)),
                "transformer_value": format_value(values.get("transformer", math.nan)),
                "sparta_value": format_value(values.get("sparta", math.nan)),
                "sparta_rank": sparta_rank,
                "baseline_has_no_attribution": str(bool(baseline_attr)).lower(),
            }
            seed_rows.append(row)
            per_seed_rows.append({"group_id": group_id, **row})
        per_run_rows[run_dir] = seed_rows
    return per_seed_rows, per_run_rows


def build_winner_count_rows(per_seed_rows: list[dict], group_id: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[str(row["metric"])].append(row)

    rows: list[dict] = []
    for metric in WINNER_METRICS:
        metric_rows = grouped.get(metric, [])
        wins = defaultdict(int)
        for row in metric_rows:
            wins[str(row.get("best_model", ""))] += 1
        rows.append(
            {
                "group_id": group_id,
                "metric": metric,
                "lstm_wins": wins.get("lstm", 0),
                "transformer_wins": wins.get("transformer", 0),
                "sparta_wins": wins.get("sparta", 0),
                "total_seeds": len(metric_rows),
            }
        )
    return rows


def write_seed_winner_reports(per_run_rows: dict[str, list[dict]], overwrite: bool) -> None:
    for run_dir, rows in per_run_rows.items():
        write_csv(Path(run_dir) / "results" / "seed_winner_report.csv", rows, SEED_WINNER_COLUMNS, overwrite=overwrite)


def save_manifest(
    group_dir: Path,
    group_name: str,
    run_dirs: list[Path],
    paths: dict[str, Path],
    status: str,
    warnings: list[str],
    expected_rows: int,
) -> None:
    manifest = {
        "group_id": group_dir.name,
        "group_name": group_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dirs": [str(path).replace("\\", "/") for path in run_dirs],
        "num_runs": len(run_dirs),
        "expected_rows": expected_rows,
        "summary_file": paths["summary"].name,
        "mean_std_file": paths["mean_std"].name,
        "winner_file": paths["per_seed_winner"].name,
        "winner_count_file": paths["metric_winner_counts"].name,
        "status": status,
        "warnings": warnings,
    }
    save_group_manifest(group_dir, manifest)


def main() -> None:
    args = parse_args()
    group_dir, created_group = prepare_group_dir(args)
    paths = output_paths(args, group_dir)
    log_path = group_dir / "logs" / "summarize_multiseed.log"
    group_id = group_dir.name
    warnings: list[str] = []
    run_dirs = select_run_dirs(args)

    if not run_dirs:
        raise RuntimeError("No run directories provided. Use --run_ids or --run_list.")

    try:
        log_many(
            log_path,
            [
                f"group_id={group_id}",
                f"group_name={args.group_name}",
                "input_run_ids=" + "|".join(str(path) for path in run_dirs),
                f"created_group={created_group}",
                f"dedupe={args.dedupe}",
                f"strict={args.strict}",
            ],
        )
        write_run_list(group_dir, run_dirs)
        rows: list[dict] = []
        for run_dir in run_dirs:
            rows.extend(collect_rows_for_run(run_dir, group_id, log_path))
        before = len(rows)
        rows = dedupe_summary_rows(rows, log_path, args.dedupe)
        if len(rows) != before:
            warning = f"deduped summary rows from {before} to {len(rows)}"
            warnings.append(warning)
            log_line(log_path, f"WARNING: {warning}")
        validate_summary_rows(rows, args.strict)
        clean_rows = clean_summary_rows(rows)
        expected_rows = expected_summary_rows(clean_rows)
        per_seed_rows, per_run_rows = build_seed_winner_rows(clean_rows, group_id)
        winner_count_rows = build_winner_count_rows(per_seed_rows, group_id)

        write_seed_winner_reports(per_run_rows, args.overwrite)
        write_csv(paths["summary"], clean_rows, SUMMARY_COLUMNS, overwrite=args.overwrite)
        write_csv(paths["mean_std"], build_mean_std_rows(clean_rows, group_id), MEAN_STD_COLUMNS, overwrite=args.overwrite)
        write_csv(paths["per_seed_winner"], per_seed_rows, PER_SEED_WINNER_COLUMNS, overwrite=args.overwrite)
        write_csv(paths["metric_winner_counts"], winner_count_rows, WINNER_COUNT_COLUMNS, overwrite=args.overwrite)

        log_many(
            log_path,
            [
                f"summary_rows={len(clean_rows)}",
                f"expected_rows={expected_rows}",
                f"summary_file={paths['summary']}",
                f"mean_std_file={paths['mean_std']}",
                f"per_seed_winner_file={paths['per_seed_winner']}",
                f"metric_winner_counts_file={paths['metric_winner_counts']}",
            ],
        )
        save_manifest(group_dir, args.group_name, run_dirs, paths, "completed", warnings, expected_rows)
        print(f"[GROUP] {group_dir}")
        print(f"[SUMMARY] {paths['summary']}")
        print(f"[MEAN_STD] {paths['mean_std']}")
        print(f"[PER_SEED_WINNER] {paths['per_seed_winner']}")
        print(f"[METRIC_WINNER_COUNTS] {paths['metric_winner_counts']}")
    except Exception as exc:
        warnings.append(str(exc))
        save_manifest(group_dir, args.group_name, run_dirs, paths, "failed", warnings, 0)
        log_line(log_path, f"ERROR: {exc}")
        raise


if __name__ == "__main__":
    main()
