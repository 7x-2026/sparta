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
    effective_simulator_backend,
    init_run_structure,
    load_run_manifest,
    save_run_manifest,
    save_raw_log_generation_provenance,
    timestamp_now,
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
    parser.add_argument("--build_dataset_only", action="store_true")
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def selected_mode(args: argparse.Namespace) -> str:
    flags = {
        "generate_only": args.generate_only,
        "audit_only": args.audit_only,
        "train_only": args.train_only,
        "eval_only": args.eval_only,
        "build_dataset_only": args.build_dataset_only,
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


def data_source(config: dict) -> str:
    data_cfg = config.get("data", {})
    return str(data_cfg.get("source", data_cfg.get("mode", "synthetic_full")))


def simulator_backend(config: dict) -> str:
    if data_source(config) == "edgesimpy":
        return str(config.get("edgesimpy", {}).get("backend", "edgesimpy_stub"))
    return "synthetic_full"


def persist_run_config_and_manifest(run_dir: Path, config_path: str | Path, config: dict, mode: str, new_run: bool) -> Path:
    copy_config_to_run(config_path, run_dir, config)
    manifest = build_initial_manifest(run_dir, config, config_path, mode)
    existing = load_run_manifest(run_dir)
    if existing and not new_run:
        completed = existing.get("completed_stages", [])
        prior_effective_at_generation = existing.get("effective_simulator_backend_at_generation")
        prior_manifest_effective = existing.get("effective_simulator_backend")
        prior_provenance_inconsistent = bool(existing.get("provenance_inconsistent", False))
        provenance_inconsistent = prior_provenance_inconsistent or bool(
            prior_effective_at_generation
            and prior_manifest_effective
            and prior_manifest_effective != prior_effective_at_generation
        )
        manifest.update(existing)
        adapter_fallback = bool(
            existing.get(
                "edgesimpy_adapter_fallback_at_generation",
                existing.get("edgesimpy_adapter_fallback", False),
            )
        )
        effective_backend = prior_effective_at_generation or effective_simulator_backend(
            data_source(config),
            simulator_backend(config),
            adapter_fallback,
            bool(existing.get("real_object_created", existing.get("real_edgesimpy_objects_created", False))),
            bool(existing.get("real_simulation_ran", False)),
            existing.get("edgesimpy_installed"),
        )
        now = timestamp_now()
        manifest.update(
            {
                "mode": mode,
                "last_mode": mode,
                "status": "running",
                "command": " ".join(sys.argv),
                "last_command": " ".join(sys.argv),
                "last_updated_at": now,
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
                "data_source": data_source(config),
                "simulator_backend": simulator_backend(config),
                "effective_simulator_backend": effective_backend,
                "raw_log_schema_version": "v1",
                "edgesimpy_config": config.get("edgesimpy", {}),
                "provenance_inconsistent": provenance_inconsistent,
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
        if mode == "build_dataset_only":
            config = load_config(args.config)
        else:
            if not resolved_config.exists():
                raise FileNotFoundError(f"Missing resolved config for resume_run: {resolved_config}")
            config = load_config(resolved_config)
        config = apply_cli_overrides(config, args)
        if args.fast_dev_run:
            config = apply_fast_dev_config(config)
        config = configure_run_paths(config, run_dir)
        config_path = Path(args.config) if mode == "build_dataset_only" else resolved_config
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
    update_run_manifest(
        run_dir,
        {"status": status, "missing_outputs": missing_outputs or [], "error": None, "last_updated_at": timestamp_now()},
    )


def fail_manifest(run_dir: Path, exc: BaseException) -> None:
    update_run_manifest(run_dir, {"status": "failed", "error": str(exc), "last_updated_at": timestamp_now()})


def record_raw_log_generation_provenance(run_dir: Path, metadata: dict) -> None:
    manifest = load_run_manifest(run_dir)
    generation_created_at = timestamp_now()
    generation_mode = manifest.get("last_mode", manifest.get("mode"))
    effective_backend = metadata.get("effective_simulator_backend_at_generation") or metadata.get("effective_simulator_backend")
    if not effective_backend:
        effective_backend = effective_simulator_backend(
            str(metadata.get("data_source", "synthetic_full")),
            str(metadata.get("simulator_backend", metadata.get("requested_backend", "synthetic_full"))),
            bool(metadata.get("edgesimpy_adapter_fallback_at_generation", metadata.get("edgesimpy_adapter_fallback", False))),
            bool(metadata.get("real_object_created", metadata.get("real_edgesimpy_objects_created", False))),
            bool(metadata.get("real_simulation_ran", False)),
            metadata.get("edgesimpy_installed_at_generation", metadata.get("edgesimpy_installed")),
        )
    fallback_at_generation = bool(
        metadata.get("edgesimpy_adapter_fallback_at_generation", metadata.get("edgesimpy_adapter_fallback", False))
    )
    raw_log_files = metadata.get("raw_log_files")
    if raw_log_files is None:
        raw_log_files = {
            name: str(Path(metadata[name]).resolve())
            for name in ["node_log", "link_log", "service_log", "path_log", "sla_log"]
            if name in metadata
        }
    provenance = {
        "provenance_version": 1,
        "run_id": run_dir.name,
        "generation_command": " ".join(sys.argv),
        "generation_mode": generation_mode,
        "generation_created_at": generation_created_at,
        "data_source": metadata.get("data_source"),
        "requested_backend": metadata.get("requested_backend", metadata.get("simulator_backend")),
        "simulator_backend": metadata.get("simulator_backend"),
        "edgesimpy_backend_state": metadata.get("edgesimpy_backend_state"),
        "effective_simulator_backend_at_generation": effective_backend,
        "edgesimpy_adapter_fallback_at_generation": fallback_at_generation,
        "fallback_reason": metadata.get("fallback_reason"),
        "edgesimpy_installed_at_generation": metadata.get("edgesimpy_installed_at_generation", metadata.get("edgesimpy_installed")),
        "edgesimpy_import_error_at_generation": metadata.get(
            "edgesimpy_import_error_at_generation",
            metadata.get("edgesimpy_import_error"),
        ),
        "edgesimpy_module": metadata.get("edgesimpy_module"),
        "real_object_created": bool(metadata.get("real_object_created", metadata.get("real_edgesimpy_objects_created", False))),
        "real_simulation_ran": bool(metadata.get("real_simulation_ran", False)),
        "real_edgesimpy_objects_created": bool(metadata.get("real_object_created", metadata.get("real_edgesimpy_objects_created", False))),
        "created_object_types": metadata.get("created_object_types", []),
        "edgesimpy_object_counts": metadata.get("edgesimpy_object_counts", {}),
        "edgesimpy_smoke_error": metadata.get("edgesimpy_smoke_error"),
        "raw_log_generator_basis": metadata.get("raw_log_generator_basis"),
        "adapter_entry_function": metadata.get("adapter_entry_function"),
        "raw_log_generator_function": metadata.get("raw_log_generator_function"),
        "raw_log_schema_version": metadata.get("raw_log_schema_version", "v1"),
        "risk_injection": metadata.get("risk_injection", "none"),
        "cpu_precursor_enabled": metadata.get("cpu_precursor_enabled", False),
        "cpu_precursor_episode_count": metadata.get("cpu_precursor_episode_count", 0),
        "cpu_precursor_episodes_path": metadata.get("cpu_precursor_episodes_path"),
        "raw_log_files": raw_log_files,
        "python_executable": metadata.get("python_executable"),
        "python_version": metadata.get("python_version"),
        "conda_env": metadata.get("conda_env"),
    }
    provenance_path = save_raw_log_generation_provenance(run_dir, provenance, overwrite=False)
    update_run_manifest(
        run_dir,
        {
            **metadata,
            "effective_simulator_backend": effective_backend,
            "edgesimpy_adapter_fallback": fallback_at_generation,
            "generation_command": provenance["generation_command"],
            "generation_mode": generation_mode,
            "generation_created_at": generation_created_at,
            "effective_simulator_backend_at_generation": effective_backend,
            "edgesimpy_adapter_fallback_at_generation": fallback_at_generation,
            "edgesimpy_installed_at_generation": provenance["edgesimpy_installed_at_generation"],
            "edgesimpy_import_error_at_generation": provenance["edgesimpy_import_error_at_generation"],
            "raw_log_generation_provenance": str(provenance_path),
            "provenance_inconsistent": False,
            "last_updated_at": generation_created_at,
        },
    )


def verify_attribution(config_path: str | Path) -> None:
    config = load_config(config_path)
    processed_dir = resolve_path(config, config["data"]["processed_dir"])
    labeled_path = processed_dir / "samples_labeled.pkl"
    samples = load_pickle(labeled_path)
    required = {"risk_label", "risk_node", "risk_link", "risk_metric", "risk_metric_scores", "attr_mask"}
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


def run_raw_log_schema_check(run_dir: Path, config: dict) -> None:
    py = sys.executable
    raw_dir = resolve_path(config, config["data"]["raw_logs_dir"])
    run_step(
        "[schema] Checking raw log schema...",
        [
            py,
            "src/analysis/check_raw_log_schema.py",
            "--raw_dir",
            str(raw_dir),
        ],
    )
    mark_stage(run_dir, "check_raw_log_schema")


def run_raw_log_generation(run_dir: Path, config_path: Path, config: dict) -> None:
    source = data_source(config)
    if source == "synthetic_full":
        run_step("[1/6] Generating raw logs...", [sys.executable, "src/simulation/synthetic_full_generator.py", "--config", str(config_path)])
        raw_dir = resolve_path(config, config["data"]["raw_logs_dir"])
        record_raw_log_generation_provenance(
            run_dir,
            {
                "node_log": str(raw_dir / "node_log.csv"),
                "link_log": str(raw_dir / "link_log.csv"),
                "service_log": str(raw_dir / "service_log.csv"),
                "path_log": str(raw_dir / "path_log.csv"),
                "sla_log": str(raw_dir / "sla_log.csv"),
                "data_source": "synthetic_full",
                "requested_backend": "synthetic_full",
                "simulator_backend": "synthetic_full",
                "effective_simulator_backend": "synthetic_full",
                "effective_simulator_backend_at_generation": "synthetic_full",
                "raw_log_schema_version": "v1",
                "edgesimpy_installed": None,
                "edgesimpy_import_error": None,
                "edgesimpy_config": {},
                "edgesimpy_adapter_fallback": False,
                "edgesimpy_adapter_fallback_at_generation": False,
                "real_object_created": False,
                "real_simulation_ran": False,
                "real_edgesimpy_objects_created": False,
                "created_object_types": [],
                "edgesimpy_object_counts": {},
                "raw_log_generator_function": "src.simulation.synthetic_full_generator.generate_synthetic_full_logs",
                "raw_log_generator_basis": "synthetic_full_v1",
                "python_executable": sys.executable,
                "python_version": sys.version,
            },
        )
        return
    if source == "edgesimpy":
        print("[1/6] Generating raw logs...", flush=True)
        from src.simulation.edgesimpy_adapter import run_edgesimpy_adapter, run_edgesimpy_real_smoke, try_import_edgesimpy

        raw_dir = resolve_path(config, config["data"]["raw_logs_dir"])
        try:
            metadata = run_edgesimpy_adapter(config, raw_dir)
        except Exception:
            installed, module, import_error = try_import_edgesimpy()
            smoke = run_edgesimpy_real_smoke(config) if installed else {}
            real_object_created = bool(smoke.get("real_object_created", False))
            real_simulation_ran = bool(smoke.get("real_simulation_ran", False))
            if not installed:
                backend_state = "edgesimpy_import_failed"
            elif real_simulation_ran:
                backend_state = "edgesimpy_real_simulation_ran"
            elif real_object_created:
                backend_state = "edgesimpy_real_objects_created"
            else:
                backend_state = "edgesimpy_import_only"
            update_run_manifest(
                run_dir,
                {
                    "data_source": "edgesimpy",
                    "requested_backend": simulator_backend(config),
                    "simulator_backend": simulator_backend(config),
                    "effective_simulator_backend": effective_simulator_backend(
                        "edgesimpy",
                        simulator_backend(config),
                        False,
                        real_object_created,
                        real_simulation_ran,
                        bool(installed),
                    ),
                    "edgesimpy_backend_state": backend_state,
                    "edgesimpy_installed": bool(installed),
                    "edgesimpy_import_error": None if installed else import_error,
                    "edgesimpy_module": smoke.get("edgesimpy_module") or (getattr(module, "__file__", None) if module is not None else None),
                    "edgesimpy_adapter_fallback": False,
                    "real_object_created": real_object_created,
                    "real_simulation_ran": real_simulation_ran,
                    "real_edgesimpy_objects_created": real_object_created,
                    "created_object_types": smoke.get("created_object_types", []),
                    "edgesimpy_object_counts": smoke.get("edgesimpy_object_counts", {}),
                    "edgesimpy_smoke_error": smoke.get("error"),
                    "last_updated_at": timestamp_now(),
                },
            )
            raise
        record_raw_log_generation_provenance(run_dir, metadata)
        return
    raise ValueError(f"Unsupported data.source: {source}")


def run_generate_pipeline(run_dir: Path, config_path: Path, config: dict) -> None:
    py = sys.executable
    run_raw_log_generation(run_dir, config_path, config)
    mark_stage(run_dir, "generate_raw_logs")
    run_raw_log_schema_check(run_dir, config)
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


def backup_dataset(run_dir: Path) -> Path | None:
    dataset_dir = run_dir / "dataset"
    if not dataset_dir.exists():
        return None
    backup_dir = run_dir / "artifacts" / f"dataset_backup_{timestamp_now().replace(':', '').replace(' ', '_').replace('-', '')}"
    shutil.copytree(dataset_dir, backup_dir)
    return backup_dir


def run_build_dataset_pipeline(run_dir: Path, config_path: Path, config: dict) -> None:
    raw_dir = resolve_path(config, config["data"]["raw_logs_dir"])
    if not raw_dir.exists():
        raise FileNotFoundError(f"Cannot build dataset without existing raw logs: {raw_dir}")
    backup_dir = backup_dataset(run_dir)
    if backup_dir:
        update_run_manifest(run_dir, {"last_dataset_backup": str(backup_dir), "last_updated_at": timestamp_now()})
        print(f"[build_dataset_only] Backed up existing dataset to {backup_dir}", flush=True)
    py = sys.executable
    run_raw_log_schema_check(run_dir, config)
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
        raise RuntimeError(f"Build-dataset pipeline missing outputs: {checked['missing_outputs']}")


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
        if mode == "build_dataset_only":
            run_build_dataset_pipeline(run_dir, resolved_config, config)
            update_run_manifest(
                run_dir,
                {
                    "risk_metric_scores_use_future_information": True,
                    "used_as_supervision_only": True,
                    "used_as_model_input": False,
                    "last_updated_at": timestamp_now(),
                },
            )
            finish_manifest(run_dir)
            print("Build-dataset-only pipeline finished.", flush=True)
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
