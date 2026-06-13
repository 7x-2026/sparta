from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.io import load_pickle
from src.utils.output_checker import REQUIRED_OUTPUTS_EVAL, REQUIRED_OUTPUTS_FULL, REQUIRED_OUTPUTS_GENERATE, check_outputs_or_report
from src.utils.run_manager import (
    build_initial_manifest,
    copy_config_to_run,
    create_run_dir,
    init_run_structure,
    load_run_manifest,
    save_run_manifest,
    update_run_manifest,
)


MODELS = ["lstm", "transformer", "sparta"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sparta_synthetic_full.yaml")
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--train_seed", type=int, default=None)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--resume_run", default=None)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def selected_mode(args: argparse.Namespace) -> str:
    flags = {
        "generate_only": args.generate_only,
        "audit_only": args.audit_only,
        "train_only": args.train_only,
        "eval_only": args.eval_only,
    }
    active = [name for name, enabled in flags.items() if enabled]
    if len(active) > 1:
        raise ValueError(f"Only one mode flag can be used at a time: {active}")
    return active[0] if active else "full"


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = copy.deepcopy(config)
    experiment_cfg = config.setdefault("experiment", {})
    if args.experiment_name:
        experiment_cfg["name"] = args.experiment_name
    if args.data_seed is not None:
        experiment_cfg["data_seed"] = int(args.data_seed)
        config["seed"] = int(args.data_seed)
    else:
        experiment_cfg.setdefault("data_seed", int(experiment_cfg.get("seed", config.get("seed", 42))))
    if args.train_seed is not None:
        experiment_cfg["train_seed"] = int(args.train_seed)
    else:
        experiment_cfg.setdefault("train_seed", int(experiment_cfg.get("seed", config.get("seed", 42))))
    return config


def apply_fast_dev_config(config: dict) -> dict:
    config = copy.deepcopy(config)
    data_cfg = config.setdefault("data", {})
    data_cfg["num_steps"] = 120
    data_cfg["num_services"] = 8
    data_cfg["num_users"] = 32
    config.setdefault("train", {})["epochs"] = 1
    config["train"]["batch_size"] = 16
    config.setdefault("validation", {})["min_samples_per_class"] = 1
    sim_cfg = config.setdefault("simulation", {})
    sim_cfg["allowed_label_ratio_range"] = {
        "normal": [0.20, 0.80],
        "risky": [0.02, 0.70],
        "violated": [0.02, 0.55],
    }
    split_check = sim_cfg.setdefault("split_balance_check", {})
    split_check["max_label_ratio_gap"] = 0.50
    split_check["max_scenario_ratio_per_split"] = 0.70
    split_check["max_top_node_attr_ratio"] = 0.90
    split_check["max_top_link_attr_ratio"] = 0.90
    split_check.setdefault("min_metric_ratio", {})["queue"] = 0.0
    sim_cfg["max_scenario_ratio_per_split"] = 0.70
    return config


def output_root_path(config: dict) -> Path:
    value = config.get("run", {}).get("output_root", "run")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def configure_run_paths(config: dict, run_dir: Path) -> dict:
    config = copy.deepcopy(config)
    run_dir = run_dir.resolve()
    dataset_dir = run_dir / "dataset"
    processed_dir = run_dir / "processed"
    raw_dir = run_dir / "raw_logs"
    checkpoint_dir = run_dir / "checkpoints"
    result_dir = run_dir / "results"
    audit_dir = run_dir / "audit"
    log_dir = run_dir / "logs"

    project_cfg = config.setdefault("project", {})
    project_cfg["root_dir"] = str(ROOT)
    project_cfg["output_dir"] = str(result_dir)
    project_cfg["checkpoint_dir"] = str(checkpoint_dir)
    project_cfg["log_dir"] = str(log_dir)

    data_cfg = config.setdefault("data", {})
    data_cfg["base_dir"] = str(run_dir)
    data_cfg["raw_logs_dir"] = str(raw_dir)
    data_cfg["processed_dir"] = str(processed_dir)
    data_cfg["dataset_dir"] = str(dataset_dir)
    data_cfg["train_path"] = str(dataset_dir / "train.pkl")
    data_cfg["val_path"] = str(dataset_dir / "val.pkl")
    data_cfg["test_path"] = str(dataset_dir / "test.pkl")
    data_cfg["metadata_path"] = str(dataset_dir / "metadata.json")

    paths_cfg = config.setdefault("paths", {})
    paths_cfg["raw_logs_dir"] = str(raw_dir)
    paths_cfg["processed_dir"] = str(processed_dir)
    paths_cfg["dataset_dir"] = str(dataset_dir)
    paths_cfg["checkpoint_dir"] = str(checkpoint_dir)
    paths_cfg["result_dir"] = str(result_dir)
    paths_cfg["audit_dir"] = str(audit_dir)
    paths_cfg["log_dir"] = str(log_dir)

    run_cfg = config.setdefault("run", {})
    run_cfg["run_id"] = run_dir.name
    run_cfg["resume_run"] = str(run_dir)
    run_cfg["output_root"] = str(run_dir.parent)
    return config


def data_seed(config: dict) -> int:
    experiment_cfg = config.get("experiment", {})
    return int(experiment_cfg.get("data_seed", experiment_cfg.get("seed", config.get("seed", 42))))


def train_seed(config: dict) -> int:
    experiment_cfg = config.get("experiment", {})
    return int(experiment_cfg.get("train_seed", experiment_cfg.get("seed", config.get("seed", 42))))


def persist_run_config_and_manifest(run_dir: Path, config_path: str | Path, config: dict, mode: str, new_run: bool) -> Path:
    copy_config_to_run(config_path, run_dir, config)
    manifest = build_initial_manifest(run_dir, config, config_path, mode)
    existing = load_run_manifest(run_dir)
    if existing and not new_run:
        completed = existing.get("completed_stages", [])
        manifest.update(existing)
        manifest.update(
            {
                "mode": mode,
                "status": "running",
                "command": " ".join(sys.argv),
                "experiment_name": config.get("experiment", {}).get("name", manifest["experiment_name"]),
                "data_seed": data_seed(config),
                "train_seed": train_seed(config),
                "seed": data_seed(config),
                "config_path": str(config_path),
                "resolved_config_path": str(run_dir / "config" / "resolved_config.yaml"),
                "dataset_dir": str(run_dir / "dataset"),
                "checkpoint_dir": str(run_dir / "checkpoints"),
                "result_dir": str(run_dir / "results"),
                "audit_dir": str(run_dir / "audit"),
                "completed_stages": completed,
                "error": None,
            }
        )
    save_run_manifest(run_dir, manifest)
    return run_dir / "config" / "resolved_config.yaml"


def prepare_run(args: argparse.Namespace, mode: str) -> tuple[Path, Path, dict]:
    if args.resume_run:
        run_dir = Path(args.resume_run).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        init_run_structure(run_dir)
        resolved_config = run_dir / "config" / "resolved_config.yaml"
        if not resolved_config.exists():
            raise FileNotFoundError(f"Missing resolved config for resume_run: {resolved_config}")
        config = load_config(resolved_config)
        config = apply_cli_overrides(config, args)
        if args.fast_dev_run:
            config = apply_fast_dev_config(config)
        config = configure_run_paths(config, run_dir)
        config_path = resolved_config
        resolved_config = persist_run_config_and_manifest(run_dir, config_path, config, mode, new_run=False)
        return run_dir, resolved_config, config

    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    if args.fast_dev_run:
        config = apply_fast_dev_config(config)
    experiment_name = config.get("experiment", {}).get("name") or config.get("project", {}).get("name", "synthetic_full")
    run_dir = create_run_dir(output_root_path(config), experiment_name).resolve()
    init_run_structure(run_dir)
    config = configure_run_paths(config, run_dir)
    resolved_config = persist_run_config_and_manifest(run_dir, args.config, config, mode, new_run=True)
    return run_dir, resolved_config, config


def run_step(label: str, step: list[str]) -> None:
    print(label, flush=True)
    print(f"[run_synthetic_full_pipeline] {' '.join(step)}", flush=True)
    subprocess.run(step, cwd=ROOT, check=True)


def mark_stage(run_dir: Path, stage: str) -> None:
    manifest = load_run_manifest(run_dir)
    completed = list(manifest.get("completed_stages", []))
    if stage not in completed:
        completed.append(stage)
    update_run_manifest(run_dir, {"completed_stages": completed, "status": "running", "error": None})


def finish_manifest(run_dir: Path, status: str = "completed", missing_outputs: list[str] | None = None) -> None:
    update_run_manifest(run_dir, {"status": status, "missing_outputs": missing_outputs or [], "error": None})


def fail_manifest(run_dir: Path, exc: BaseException) -> None:
    update_run_manifest(run_dir, {"status": "failed", "error": str(exc)})


def verify_attribution(config_path: str | Path) -> None:
    config = load_config(config_path)
    processed_dir = resolve_path(config, config["data"]["processed_dir"])
    labeled_path = processed_dir / "samples_labeled.pkl"
    samples = load_pickle(labeled_path)
    required = {"risk_label", "risk_node", "risk_link", "risk_metric", "attr_mask"}
    if not samples:
        raise RuntimeError(f"No labeled samples found in {labeled_path}")
    missing = required.difference(samples[0])
    if missing:
        raise RuntimeError(f"Missing attribution fields in {labeled_path}: {sorted(missing)}")
    print(f"[run_synthetic_full_pipeline] Attribution fields verified in {labeled_path}", flush=True)


def copy_processed_aliases(config_path: str | Path) -> None:
    config = load_config(config_path)
    processed_dir = resolve_path(config, config["data"]["processed_dir"])
    source = processed_dir / "samples_labeled.pkl"
    if not source.exists():
        raise FileNotFoundError(f"Missing labeled samples: {source}")
    for name in ["labeled_samples.pkl", "attributed_samples.pkl"]:
        shutil.copy2(source, processed_dir / name)


def copy_label_stats(config_path: str | Path) -> None:
    config = load_config(config_path)
    dataset_dir = resolve_path(config, config["data"]["dataset_dir"])
    metadata_path = dataset_dir / "metadata.json"
    if metadata_path.exists():
        shutil.copy2(metadata_path, dataset_dir / "label_stats.json")


def run_audit(run_dir: Path, config: dict) -> None:
    py = sys.executable
    dataset_dir = resolve_path(config, config["data"]["dataset_dir"])
    raw_dir = resolve_path(config, config["data"]["raw_logs_dir"])
    audit_dir = Path(config.get("paths", {}).get("audit_dir") or run_dir / "audit")
    run_step(
        "[6/6] Running audit...",
        [
            py,
            "src/analysis/audit_synthetic_full.py",
            "--dataset_dir",
            str(dataset_dir),
            "--raw_dir",
            str(raw_dir),
            "--output_dir",
            str(audit_dir),
        ],
    )
    mark_stage(run_dir, "audit")


def run_generate_pipeline(run_dir: Path, config_path: Path, config: dict) -> None:
    py = sys.executable
    run_step("[1/6] Generating raw logs...", [py, "src/simulation/synthetic_full_generator.py", "--config", str(config_path)])
    mark_stage(run_dir, "generate_raw_logs")
    run_step("[2/6] Building path graph...", [py, "src/preprocessing/build_path_graph.py", "--config", str(config_path)])
    mark_stage(run_dir, "build_path_graph")
    run_step("[3/6] Generating labels...", [py, "src/preprocessing/generate_labels.py", "--config", str(config_path)])
    print("[4/6] Generating attribution...", flush=True)
    verify_attribution(config_path)
    copy_processed_aliases(config_path)
    mark_stage(run_dir, "generate_labels_attribution")
    run_step("[5/6] Splitting dataset...", [py, "src/preprocessing/split_dataset.py", "--config", str(config_path)])
    copy_label_stats(config_path)
    mark_stage(run_dir, "split_dataset")
    run_audit(run_dir, config)
    checked = check_outputs_or_report(run_dir, REQUIRED_OUTPUTS_GENERATE)
    if checked["missing_outputs"]:
        update_run_manifest(run_dir, {"missing_outputs": checked["missing_outputs"]})
        raise RuntimeError(f"Generate pipeline missing outputs: {checked['missing_outputs']}")


def required_dataset_files(run_dir: Path) -> list[Path]:
    return [run_dir / "dataset" / f"{split}.pkl" for split in ["train", "val", "test"]]


def ensure_dataset_ready(run_dir: Path, config_path: Path) -> None:
    missing = [path for path in required_dataset_files(run_dir) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Dataset files not found. Please run: "
            f"python src/run_synthetic_full_pipeline.py --config {config_path} --generate_only"
        )


def ensure_checkpoints_ready(run_dir: Path) -> None:
    missing = [run_dir / "checkpoints" / f"{model}_best.pth" for model in MODELS if not (run_dir / "checkpoints" / f"{model}_best.pth").exists()]
    if missing:
        raise RuntimeError(f"Checkpoint files not found. Please run train_only first: {[str(path) for path in missing]}")


def run_train_pipeline(run_dir: Path, config_path: Path) -> None:
    ensure_dataset_ready(run_dir, config_path)
    py = sys.executable
    for model in MODELS:
        run_step(f"[train] Training {model}...", [py, "src/train.py", "--config", str(config_path), "--model", model])
    mark_stage(run_dir, "train")


def run_eval_pipeline(run_dir: Path, config_path: Path) -> None:
    ensure_dataset_ready(run_dir, config_path)
    ensure_checkpoints_ready(run_dir)
    py = sys.executable
    run_step("[eval] Evaluating checkpoints...", [py, "src/analysis/supplement_eval_outputs.py", "--run_dir", str(run_dir)])
    mark_stage(run_dir, "eval")
    checked = check_outputs_or_report(run_dir, REQUIRED_OUTPUTS_EVAL)
    if checked["missing_outputs"]:
        update_run_manifest(run_dir, {"missing_outputs": checked["missing_outputs"]})
        raise RuntimeError(f"Eval pipeline missing outputs: {checked['missing_outputs']}")


def run_audit_only(run_dir: Path, config: dict) -> None:
    ensure_dataset_ready(run_dir, run_dir / "config" / "resolved_config.yaml")
    run_audit(run_dir, config)


def main() -> None:
    args = parse_args()
    mode = selected_mode(args)
    run_dir, resolved_config, config = prepare_run(args, mode)
    print(f"[run_synthetic_full_pipeline] run_dir={run_dir}", flush=True)
    print(f"[run_synthetic_full_pipeline] data_seed={data_seed(config)} train_seed={train_seed(config)}", flush=True)

    try:
        if mode == "generate_only":
            run_generate_pipeline(run_dir, resolved_config, config)
            finish_manifest(run_dir)
            print("Generate-only pipeline finished.", flush=True)
            return
        if mode == "audit_only":
            run_audit_only(run_dir, config)
            finish_manifest(run_dir)
            print("Audit-only pipeline finished.", flush=True)
            return
        if mode == "train_only":
            run_train_pipeline(run_dir, resolved_config)
            finish_manifest(run_dir)
            print("Train-only pipeline finished.", flush=True)
            return
        if mode == "eval_only":
            run_eval_pipeline(run_dir, resolved_config)
            finish_manifest(run_dir)
            print("Eval-only pipeline finished.", flush=True)
            return

        run_generate_pipeline(run_dir, resolved_config, config)
        run_train_pipeline(run_dir, resolved_config)
        run_eval_pipeline(run_dir, resolved_config)
        checked = check_outputs_or_report(run_dir, sorted(set(REQUIRED_OUTPUTS_FULL + REQUIRED_OUTPUTS_EVAL + REQUIRED_OUTPUTS_GENERATE)))
        if checked["missing_outputs"]:
            update_run_manifest(run_dir, {"missing_outputs": checked["missing_outputs"]})
            raise RuntimeError(f"Full pipeline missing outputs: {checked['missing_outputs']}")
        finish_manifest(run_dir)
        print("SPARTA synthetic full pipeline complete", flush=True)
    except Exception as exc:
        fail_manifest(run_dir, exc)
        raise


if __name__ == "__main__":
    main()
