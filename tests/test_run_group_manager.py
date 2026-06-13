from __future__ import annotations

import json

from src.utils.run_group_manager import (
    create_run_group_dir,
    init_run_group_structure,
    save_multiseed_manifest,
    save_run_list,
    update_multiseed_manifest,
)
from src.utils.group_manager import create_group_dir, save_group_manifest, write_run_list


def test_run_group_manager_creates_incrementing_groups(tmp_path):
    first = create_run_group_dir(tmp_path, "synthetic_full_3seed")
    second = create_run_group_dir(tmp_path, "synthetic_full_3seed")
    assert first != second
    assert first.name.endswith("_001_synthetic_full_3seed")
    assert second.name.endswith("_002_synthetic_full_3seed")

    paths = init_run_group_structure(first)
    assert paths["configs"].is_dir()
    assert paths["logs"].is_dir()

    save_run_list(first, ["run/seed42", "run/seed2025", "run/seed3407"])
    save_multiseed_manifest(first, {"group_id": first.name, "status": "running", "run_dirs": []})
    update_multiseed_manifest(first, {"status": "completed", "run_dirs": ["run/seed42"]})

    assert (first / "run_list.txt").exists()
    assert (first / "multiseed_manifest.json").exists()
    manifest = json.loads((first / "multiseed_manifest.json").read_text(encoding="utf-8"))
    assert manifest["group_id"] == first.name
    assert manifest["status"] == "completed"
    assert manifest["run_dirs"] == ["run/seed42"]


def test_group_manager_creates_incrementing_group_dirs(tmp_path):
    first = create_group_dir(tmp_path, "multiseed_synthetic_full")
    second = create_group_dir(tmp_path, "multiseed_synthetic_full")
    assert first != second
    assert first.name.endswith("_001_multiseed_synthetic_full")
    assert second.name.endswith("_002_multiseed_synthetic_full")
    assert (first / "logs").is_dir()

    write_run_list(first, ["run/seed42", "run/seed2025", "run/seed3407"])
    save_group_manifest(first, {"group_id": first.name, "status": "completed"})
    assert (first / "run_list.txt").exists()
    assert (first / "group_manifest.json").exists()
