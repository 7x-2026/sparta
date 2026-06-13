from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.utils.io import ensure_dir


RUN_GROUP_SUBDIRS = ["configs", "logs"]


def _safe_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return out or "experiment"


def create_run_group_dir(group_root: str | Path, experiment_name: str) -> Path:
    root = ensure_dir(group_root)
    today = datetime.now().strftime("%Y%m%d")
    safe_experiment = _safe_name(experiment_name)
    idx = 1
    while True:
        group_dir = root / f"{today}_{idx:03d}_{safe_experiment}"
        try:
            group_dir.mkdir(parents=True, exist_ok=False)
            return group_dir
        except FileExistsError:
            idx += 1


def init_run_group_structure(group_dir: Path) -> dict[str, Path]:
    group_dir = Path(group_dir)
    paths = {"group_dir": group_dir}
    for name in RUN_GROUP_SUBDIRS:
        paths[name] = ensure_dir(group_dir / name)
    return paths


def save_run_list(group_dir: Path, run_dirs: list[str | Path]) -> None:
    path = Path(group_dir) / "run_list.txt"
    lines = [str(run_dir).replace("\\", "/") for run_dir in run_dirs]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _manifest_path(group_dir: Path) -> Path:
    return Path(group_dir) / "multiseed_manifest.json"


def save_multiseed_manifest(group_dir: Path, manifest: dict) -> None:
    path = _manifest_path(group_dir)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def load_multiseed_manifest(group_dir: Path) -> dict:
    path = _manifest_path(group_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def update_multiseed_manifest(group_dir: Path, updates: dict) -> None:
    manifest = load_multiseed_manifest(group_dir)
    manifest.update(updates)
    save_multiseed_manifest(group_dir, manifest)
