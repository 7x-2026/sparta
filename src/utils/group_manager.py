from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.utils.io import ensure_dir


def _safe_name(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return out or "group"


def create_group_dir(output_root: str | Path, group_name: str) -> Path:
    """Create run_groups/YYYYMMDD_XXX_group_name without overwriting old groups."""
    root = ensure_dir(output_root)
    today = datetime.now().strftime("%Y%m%d")
    safe_group = _safe_name(group_name)
    idx = 1
    while True:
        group_dir = root / f"{today}_{idx:03d}_{safe_group}"
        try:
            group_dir.mkdir(parents=True, exist_ok=False)
            ensure_dir(group_dir / "logs")
            return group_dir
        except FileExistsError:
            idx += 1


def save_group_manifest(group_dir: Path, manifest: dict) -> None:
    path = Path(group_dir) / "group_manifest.json"
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def write_run_list(group_dir: Path, run_dirs: list[str | Path]) -> None:
    path = Path(group_dir) / "run_list.txt"
    lines = [str(run_dir).replace("\\", "/") for run_dir in run_dirs]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
