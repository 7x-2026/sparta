from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.run_manager import (
    load_run_manifest,
    raw_log_generation_provenance_path,
    save_run_manifest,
)


RAW_LOG_FILENAMES = [
    "node_log.csv",
    "link_log.csv",
    "service_log.csv",
    "path_log.csv",
    "sla_log.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: str | None, run_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = run_dir / path
    return path


def raw_log_paths(run_dir: Path, provenance: dict) -> dict[str, Path]:
    raw_files = provenance.get("raw_log_files") or {}
    paths: dict[str, Path] = {}
    for filename in RAW_LOG_FILENAMES:
        value = raw_files.get(filename)
        paths[filename] = resolve_path(value, run_dir) if value else run_dir / "raw_logs" / filename
    return paths


def check_provenance(run_dir: Path) -> tuple[str, list[str], bool]:
    manifest = load_run_manifest(run_dir)
    lines: list[str] = [f"run_dir: {run_dir}"]
    if not manifest:
        return "PROVENANCE NOT VERIFIED: run_manifest.json is missing.", lines, True

    manifest_provenance_path = resolve_path(manifest.get("raw_log_generation_provenance"), run_dir)
    default_provenance_path = raw_log_generation_provenance_path(run_dir)
    provenance_path = manifest_provenance_path if manifest_provenance_path and manifest_provenance_path.exists() else default_provenance_path

    lines.append(f"manifest_effective_simulator_backend: {manifest.get('effective_simulator_backend')}")
    lines.append(
        "manifest_effective_simulator_backend_at_generation: "
        f"{manifest.get('effective_simulator_backend_at_generation')}"
    )
    lines.append(f"provenance_path: {provenance_path}")

    if not provenance_path.exists():
        if manifest.get("effective_simulator_backend") == "edgesimpy_real":
            status = (
                "PROVENANCE NOT VERIFIED: manifest claims edgesimpy_real, "
                "but raw_log_generation_provenance.json is missing."
            )
        else:
            status = "PROVENANCE NOT VERIFIED: raw_log_generation_provenance.json is missing."
        return status, lines, True

    provenance = load_json(provenance_path)
    provenance_backend = provenance.get("effective_simulator_backend_at_generation")
    manifest_generation_backend = manifest.get("effective_simulator_backend_at_generation")
    manifest_current_backend = manifest.get("effective_simulator_backend")
    real_objects_created = bool(provenance.get("real_object_created", provenance.get("real_edgesimpy_objects_created", False)))
    real_simulation_ran = bool(provenance.get("real_simulation_ran", False))

    lines.append(f"provenance_effective_simulator_backend_at_generation: {provenance_backend}")
    lines.append(f"provenance_edgesimpy_backend_state: {provenance.get('edgesimpy_backend_state')}")
    lines.append(f"provenance_generation_mode: {provenance.get('generation_mode')}")
    lines.append(f"provenance_generation_created_at: {provenance.get('generation_created_at')}")
    lines.append(f"provenance_requested_backend: {provenance.get('requested_backend')}")
    lines.append(f"provenance_fallback_used: {provenance.get('edgesimpy_adapter_fallback_at_generation')}")
    lines.append(f"provenance_fallback_reason: {provenance.get('fallback_reason')}")
    lines.append(f"provenance_edgesimpy_installed: {provenance.get('edgesimpy_installed_at_generation')}")
    lines.append(f"provenance_edgesimpy_import_error: {provenance.get('edgesimpy_import_error_at_generation')}")
    lines.append(f"provenance_real_edgesimpy_objects_created: {real_objects_created}")
    lines.append(f"provenance_real_simulation_ran: {real_simulation_ran}")
    lines.append(f"provenance_created_object_types: {provenance.get('created_object_types', [])}")
    lines.append(f"provenance_edgesimpy_object_counts: {provenance.get('edgesimpy_object_counts')}")

    missing_raw_logs = [name for name, path in raw_log_paths(run_dir, provenance).items() if not path.exists()]
    if missing_raw_logs:
        lines.append(f"missing_raw_logs: {'|'.join(missing_raw_logs)}")
    else:
        lines.append("missing_raw_logs: none")

    mismatches: list[str] = []
    if manifest_generation_backend and provenance_backend and manifest_generation_backend != provenance_backend:
        mismatches.append(
            "manifest effective_simulator_backend_at_generation "
            f"({manifest_generation_backend}) != provenance ({provenance_backend})"
        )
    if manifest_current_backend and provenance_backend and manifest_current_backend != provenance_backend:
        mismatches.append(
            f"manifest effective_simulator_backend ({manifest_current_backend}) != provenance ({provenance_backend})"
        )
    if missing_raw_logs:
        mismatches.append(f"missing raw logs: {'|'.join(missing_raw_logs)}")
    if provenance_backend == "edgesimpy_real" and not real_objects_created:
        mismatches.append("provenance claims edgesimpy_real but real EdgeSimPy objects were not created")
    if provenance_backend == "edgesimpy_real" and not real_simulation_ran:
        mismatches.append("provenance claims edgesimpy_real but a real EdgeSimPy simulation step was not run")

    if mismatches:
        status = "PROVENANCE MISMATCH: " + "; ".join(mismatches)
        manifest["provenance_inconsistent"] = True
        save_run_manifest(run_dir, manifest)
        return status, lines, True

    if provenance_backend == "edgesimpy_real":
        status = "PROVENANCE VERIFIED: raw logs were generated by verified real EdgeSimPy objects."
    elif provenance_backend:
        status = f"PROVENANCE VERIFIED: effective_simulator_backend={provenance_backend}."
    else:
        status = "PROVENANCE NOT VERIFIED: effective backend is missing from provenance."
    return status, lines, False


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    report_dir = run_dir / "artifacts"
    report_dir.mkdir(parents=True, exist_ok=True)
    status, details, failed = check_provenance(run_dir)
    report_path = report_dir / "run_provenance_check_report.txt"
    report_path.write_text(status + "\n\n" + "\n".join(details) + "\n", encoding="utf-8")
    print(status)
    print(f"Wrote {report_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
