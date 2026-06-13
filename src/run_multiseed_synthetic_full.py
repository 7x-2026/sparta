from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.io import ensure_dir
from src.utils.group_manager import create_group_dir
from src.utils.run_group_manager import (
    init_run_group_structure,
    save_multiseed_manifest,
    save_run_list,
    update_multiseed_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_synthetic_full.yaml")
    parser.add_argument("--data_seeds", nargs="+", type=int, default=[42, 2025, 3407])
    parser.add_argument("--train_seed", type=int, default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--group_name", default="multiseed_synthetic_full")
    parser.add_argument("--group_root", default="run_groups")
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_relpath(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def group_root_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_experiment_name(config: dict, cli_name: str | None) -> str:
    if cli_name:
        return cli_name
    return config.get("experiment", {}).get("name") or config.get("project", {}).get("name", "synthetic_full")


def run_name_for_seed(group_experiment_name: str, seed: int) -> str:
    base = group_experiment_name[:-6] if group_experiment_name.endswith("_3seed") else group_experiment_name
    return f"{base}_seed{seed}"


def resolved_group_config(config: dict, experiment_name: str, data_seeds: list[int], train_seed: int) -> dict:
    out = dict(config)
    out["experiment"] = dict(config.get("experiment", {}))
    out["experiment"]["name"] = experiment_name
    out["experiment"]["data_seeds"] = list(data_seeds)
    out["experiment"]["train_seed"] = int(train_seed)
    out["run_group"] = {
        "enabled": True,
        "data_seeds": list(data_seeds),
        "train_seed": int(train_seed),
    }
    return out


def write_group_configs(group_dir: Path, config_path: Path, resolved_config: dict) -> None:
    config_dir = ensure_dir(group_dir / "configs")
    src = config_path
    if src.exists():
        shutil.copy2(src, config_dir / src.name)
    with (config_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False, allow_unicode=True)


def append_log(log_path: Path, text: str) -> None:
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")


def run_command(step: list[str], log_path: Path) -> subprocess.CompletedProcess:
    append_log(log_path, f"\n$ {' '.join(step)}\n")
    completed = subprocess.run(step, cwd=ROOT, text=True, capture_output=True)
    append_log(log_path, completed.stdout)
    append_log(log_path, completed.stderr)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, step, output=completed.stdout, stderr=completed.stderr)
    return completed


def parse_run_dir(output: str) -> Path:
    for line in output.splitlines():
        marker = "run_dir="
        if marker in line:
            return Path(line.split(marker, 1)[1].strip())
    raise RuntimeError("Could not find run_dir=... in run_synthetic_full_pipeline.py output")


def build_manifest(group_dir: Path, args: argparse.Namespace, experiment_name: str, train_seed: int, resolved_config_path: Path) -> dict:
    return {
        "group_id": group_dir.name,
        "experiment_name": experiment_name,
        "group_name": args.group_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(args.config),
        "resolved_config_path": str(resolved_config_path),
        "data_seeds": list(args.data_seeds),
        "train_seed": int(train_seed),
        "run_dirs": [],
        "summary_file": str(group_dir / "multiseed_summary.csv"),
        "mean_std_file": str(group_dir / "multiseed_mean_std.csv"),
        "winner_report_file": str(group_dir / "per_seed_winner_report.csv"),
        "winner_count_file": str(group_dir / "metric_winner_counts.csv"),
        "status": "running",
        "missing_outputs": [],
        "error": None,
    }


def run_single_seed(args: argparse.Namespace, seed: int, train_seed: int, group_experiment_name: str, log_path: Path) -> Path:
    step = [
        sys.executable,
        "src/run_synthetic_full_pipeline.py",
        "--config",
        args.config,
        "--data_seed",
        str(seed),
        "--train_seed",
        str(train_seed),
        "--experiment_name",
        run_name_for_seed(group_experiment_name, seed),
    ]
    if args.fast_dev_run:
        step.append("--fast_dev_run")
    completed = run_command(step, log_path)
    return parse_run_dir(completed.stdout)


def run_summary(group_dir: Path, log_path: Path, group_name: str, overwrite: bool) -> None:
    step = [
        sys.executable,
        "src/analysis/summarize_multiseed.py",
        "--run_list",
        str(group_dir / "run_list.txt"),
        "--output_dir",
        str(group_dir),
        "--group_name",
        group_name,
    ]
    if overwrite:
        step.append("--overwrite")
    run_command(step, log_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment_name = resolve_experiment_name(config, args.experiment_name)
    train_seed = int(args.train_seed if args.train_seed is not None else config.get("experiment", {}).get("train_seed", 0))
    group_dir = create_group_dir(group_root_path(args.group_root), args.group_name)
    init_run_group_structure(group_dir)
    log_path = group_dir / "logs" / "multiseed.log"
    resolved_config = resolved_group_config(config, experiment_name, list(args.data_seeds), train_seed)
    write_group_configs(group_dir, Path(args.config), resolved_config)

    manifest = build_manifest(group_dir, args, experiment_name, train_seed, group_dir / "configs" / "resolved_config.yaml")
    save_multiseed_manifest(group_dir, manifest)
    print(f"[GROUP] Created group dir: {safe_relpath(group_dir)}")

    run_dirs: list[str] = []
    try:
        for seed in args.data_seeds:
            run_dir = run_single_seed(args, seed, train_seed, experiment_name, log_path).resolve()
            rel_run_dir = safe_relpath(run_dir)
            run_dirs.append(rel_run_dir)
            save_run_list(group_dir, run_dirs)
            update_multiseed_manifest(group_dir, {"run_dirs": run_dirs})
            print(f"[RUN] data_seed={seed} saved to {rel_run_dir}")

        run_summary(group_dir, log_path, args.group_name, args.overwrite)
        update_multiseed_manifest(
            group_dir,
            {
                "run_dirs": run_dirs,
                "summary_file": str(group_dir / "multiseed_summary.csv"),
                "mean_std_file": str(group_dir / "multiseed_mean_std.csv"),
                "winner_report_file": str(group_dir / "per_seed_winner_report.csv"),
                "winner_count_file": str(group_dir / "metric_winner_counts.csv"),
                "status": "completed",
                "missing_outputs": [],
                "error": None,
            },
        )
        print(f"[SUMMARY] saved to {safe_relpath(group_dir / 'multiseed_summary.csv')}")
        print(f"[MEAN_STD] saved to {safe_relpath(group_dir / 'multiseed_mean_std.csv')}")
        print(f"[WINNER] saved to {safe_relpath(group_dir / 'per_seed_winner_report.csv')}")
        print(f"[WINNER_COUNTS] saved to {safe_relpath(group_dir / 'metric_winner_counts.csv')}")
    except Exception as exc:
        update_multiseed_manifest(group_dir, {"status": "failed", "run_dirs": run_dirs, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
