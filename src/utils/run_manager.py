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


def build_initial_manifest(run_dir: Path, config: dict, config_path: str | Path, mode: str) -> dict:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    experiment_cfg = config.get("experiment", {})
    experiment_name = experiment_cfg.get("name") or config.get("project", {}).get("name", "synthetic_full")
    data_seed = int(experiment_cfg.get("data_seed", experiment_cfg.get("seed", config.get("seed", 42))))
    train_seed = int(experiment_cfg.get("train_seed", experiment_cfg.get("seed", config.get("seed", 42))))
    return {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "config_path": str(config_path),
        "resolved_config_path": str(run_dir / "config" / "resolved_config.yaml"),
        "run_dir": str(run_dir),
        "seed": data_seed,
        "data_seed": data_seed,
        "train_seed": train_seed,
        "mode": mode,
        "status": "running",
        "dataset_dir": str(run_dir / "dataset"),
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "result_dir": str(run_dir / "results"),
        "audit_dir": str(run_dir / "audit"),
        "log_dir": str(run_dir / "logs"),
        "models": ["lstm", "transformer", "sparta"],
        "completed_stages": [],
        "missing_outputs": [],
        "error": None,
    }
