from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml

from src.utils.io import ensure_dir


RUN_SUBDIRS = [
    "config",
    "raw_logs",
    "processed",
    "dataset",
    "checkpoints",
    "logs",
    "audit",
    "results",
    "artifacts",
]


def _safe_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return out or "experiment"


def create_run_dir(output_root: str | Path, experiment_name: str) -> Path:
    root = ensure_dir(output_root)
    today = datetime.now().strftime("%Y%m%d")
    safe_experiment = _safe_name(experiment_name)
    idx = 1
    while True:
        run_dir = root / f"{today}_{idx:03d}_{safe_experiment}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            idx += 1


def init_run_structure(run_dir: Path) -> dict[str, Path]:
    paths = {"run_dir": Path(run_dir)}
    for name in RUN_SUBDIRS:
        paths[name] = ensure_dir(Path(run_dir) / name)
    readme = paths["artifacts"] / "README.txt"
    if not readme.exists():
        readme.write_text("SPARTA synthetic full run artifacts.\n", encoding="utf-8")
    return paths


def copy_config_to_run(config_path: str | Path, run_dir: Path, resolved_config: dict) -> None:
    config_dir = ensure_dir(Path(run_dir) / "config")
    src = Path(config_path)
    if src.exists():
        dst = config_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    with (config_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, sort_keys=False, allow_unicode=True)


def _manifest_path(run_dir: Path) -> Path:
    return Path(run_dir) / "run_manifest.json"


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def raw_log_generation_provenance_path(run_dir: Path) -> Path:
    return Path(run_dir) / "artifacts" / "raw_log_generation_provenance.json"


def save_raw_log_generation_provenance(run_dir: Path, provenance: dict, overwrite: bool = False) -> Path:
    path = raw_log_generation_provenance_path(run_dir)
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Raw-log generation provenance already exists: {path}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    return path


def load_raw_log_generation_provenance(run_dir: Path) -> dict:
    path = raw_log_generation_provenance_path(run_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_run_manifest(run_dir: Path, manifest: dict) -> None:
    path = _manifest_path(run_dir)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_run_manifest(run_dir: Path) -> dict:
    path = _manifest_path(run_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_run_manifest(run_dir: Path, updates: dict) -> None:
    manifest = load_run_manifest(run_dir)
    manifest.update(updates)
    save_run_manifest(run_dir, manifest)


def check_required_outputs(run_dir: Path, required_files: list[str]) -> dict[str, list[str]]:
    existing: list[str] = []
    missing: list[str] = []
    for rel in required_files:
        target = Path(run_dir) / rel
        if target.exists():
            existing.append(rel)
        else:
            missing.append(rel)
    return {"existing_outputs": existing, "missing_outputs": missing}


def effective_simulator_backend(
    data_source: str,
    simulator_backend: str,
    adapter_fallback: bool = False,
    real_edgesimpy_objects_created: bool = False,
    real_simulation_ran: bool = False,
    edgesimpy_import_success: bool | None = None,
) -> str:
    if data_source == "synthetic_full":
        return "synthetic_full"
    if simulator_backend == "edgesimpy_stub":
        return "edgesimpy_stub"
    if simulator_backend == "edgesimpy" and real_edgesimpy_objects_created and real_simulation_ran and not adapter_fallback:
        return "edgesimpy_real"
    if simulator_backend == "edgesimpy" and adapter_fallback:
        return "edgesimpy_adapter_fallback"
    if simulator_backend == "edgesimpy" and edgesimpy_import_success is False:
        return "edgesimpy_import_failed"
    if simulator_backend == "edgesimpy" and real_edgesimpy_objects_created:
        return "edgesimpy_real_objects_created"
    if simulator_backend == "edgesimpy" and edgesimpy_import_success is True:
        return "edgesimpy_import_only"
    if simulator_backend == "edgesimpy":
        return "edgesimpy_unverified"
    return simulator_backend or data_source


def build_initial_manifest(run_dir: Path, config: dict, config_path: str | Path, mode: str) -> dict:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    experiment_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    edge_cfg = config.get("edgesimpy", {})
    experiment_name = experiment_cfg.get("name") or config.get("project", {}).get("name", "synthetic_full")
    data_seed = int(experiment_cfg.get("data_seed", experiment_cfg.get("seed", config.get("seed", 42))))
    train_seed = int(experiment_cfg.get("train_seed", experiment_cfg.get("seed", config.get("seed", 42))))
    data_source = data_cfg.get("source", data_cfg.get("mode", "synthetic_full"))
    simulator_backend = edge_cfg.get("backend", "edgesimpy_stub") if data_source == "edgesimpy" else "synthetic_full"
    effective_backend = effective_simulator_backend(str(data_source), str(simulator_backend), False, False)
    now = timestamp_now()
    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "created_at": now,
        "command": " ".join(sys.argv),
        "last_command": " ".join(sys.argv),
        "config_path": str(config_path),
        "resolved_config_path": str(run_dir / "config" / "resolved_config.yaml"),
        "run_dir": str(run_dir),
        "seed": data_seed,
        "data_seed": data_seed,
        "train_seed": train_seed,
        "mode": mode,
        "last_mode": mode,
        "last_updated_at": now,
        "status": "running",
        "dataset_dir": str(run_dir / "dataset"),
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "result_dir": str(run_dir / "results"),
        "audit_dir": str(run_dir / "audit"),
        "log_dir": str(run_dir / "logs"),
        "data_source": data_source,
        "simulator_backend": simulator_backend,
        "effective_simulator_backend": effective_backend,
        "raw_log_schema_version": "v1",
        "edgesimpy_installed": None,
        "edgesimpy_import_error": None,
        "edgesimpy_config": edge_cfg,
        "edgesimpy_backend_state": "not_generated",
        "edgesimpy_adapter_fallback": False,
        "real_object_created": False,
        "real_simulation_ran": False,
        "real_edgesimpy_objects_created": False,
        "created_object_types": [],
        "risk_injection": "none",
        "generation_command": None,
        "generation_mode": None,
        "generation_created_at": None,
        "effective_simulator_backend_at_generation": None,
        "edgesimpy_adapter_fallback_at_generation": None,
        "edgesimpy_installed_at_generation": None,
        "edgesimpy_import_error_at_generation": None,
        "raw_log_generation_provenance": None,
        "provenance_inconsistent": False,
        "models": ["lstm", "transformer", "sparta"],
        "completed_stages": [],
        "missing_outputs": [],
        "error": None,
    }
