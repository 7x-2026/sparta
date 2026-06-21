from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.run_manager import load_run_manifest, save_run_manifest, timestamp_now


RAW_LOG_FILES = [
    "node_log.csv",
    "link_log.csv",
    "service_log.csv",
    "path_log.csv",
    "sla_log.csv",
]
DATASET_FILES = ["train.pkl", "val.pkl", "test.pkl"]
FINALIZED_STAGES = [
    "audit_synthetic_full",
    "audit_cpu_precursor",
    "pressure02_evidence_baseline",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely mark a manually audited SPARTA run as completed without regenerating data or training."
    )
    parser.add_argument("--run_dir", required=True)
    return parser.parse_args()


def check_file_exists(path: Path, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f"missing file: {path}")


def check_audit_report(run_dir: Path, failures: list[str]) -> None:
    path = run_dir / "audit" / "audit_report.txt"
    check_file_exists(path, failures)
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    if "通过正式实验数据要求" not in text and "未发现失败项" not in text:
        failures.append(
            "audit_report.txt does not contain required pass phrase: "
            "通过正式实验数据要求 or 未发现失败项"
        )


def check_cpu_precursor_audit(run_dir: Path, failures: list[str]) -> None:
    path = run_dir / "analysis" / "cpu_precursor_correlation.json"
    check_file_exists(path, failures)
    if not path.exists():
        return
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"failed to parse cpu_precursor_correlation.json: {exc}")
        return
    if payload.get("cpu_precursor_valid") is not True:
        failures.append("analysis/cpu_precursor_correlation.json has cpu_precursor_valid != true")


def check_manifest(manifest: dict, failures: list[str]) -> None:
    expected = {
        "effective_simulator_backend": "edgesimpy_real",
        "risk_injection": "cpu_precursor_v1",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            failures.append(f"run_manifest.json {key} expected {value!r}, got {manifest.get(key)!r}")
    for key in ["real_object_created", "real_simulation_ran"]:
        if manifest.get(key) is not True:
            failures.append(f"run_manifest.json {key} expected true, got {manifest.get(key)!r}")


def validate_run(run_dir: Path) -> tuple[dict, list[str]]:
    failures: list[str] = []
    if not run_dir.exists():
        return {}, [f"run_dir does not exist: {run_dir}"]

    for filename in RAW_LOG_FILES:
        check_file_exists(run_dir / "raw_logs" / filename, failures)
    for filename in DATASET_FILES:
        check_file_exists(run_dir / "dataset" / filename, failures)
    check_audit_report(run_dir, failures)
    check_cpu_precursor_audit(run_dir, failures)

    manifest_path = run_dir / "run_manifest.json"
    check_file_exists(manifest_path, failures)
    manifest = load_run_manifest(run_dir)
    if manifest_path.exists() and not manifest:
        failures.append(f"run_manifest.json is empty or invalid: {manifest_path}")
    if manifest:
        check_manifest(manifest, failures)
    return manifest, failures


def finalize_manifest(run_dir: Path, manifest: dict) -> dict:
    completed = list(manifest.get("completed_stages", []))
    for stage in FINALIZED_STAGES:
        if stage not in completed:
            completed.append(stage)
    now = timestamp_now()
    manifest.update(
        {
            "status": "completed",
            "last_mode": "generate_only_completed_after_manual_audit",
            "completed_stages": completed,
            "error": None,
            "finalized_at": now,
            "last_updated_at": now,
        }
    )
    save_run_manifest(run_dir, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest, failures = validate_run(run_dir)
    if failures:
        print("[finalize_run_status] Checks failed; run_manifest.json was not modified.", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    finalized = finalize_manifest(run_dir, manifest)
    print(f"[finalize_run_status] Finalized run: {run_dir}")
    print(f"status={finalized.get('status')}")
    print(f"last_mode={finalized.get('last_mode')}")
    print(f"finalized_at={finalized.get('finalized_at')}")


if __name__ == "__main__":
    main()
