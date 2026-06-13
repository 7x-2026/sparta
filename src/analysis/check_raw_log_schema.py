from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.log_schema import EXPORT_LOG_COLUMNS, LOG_FILENAMES, LOG_SCHEMA_VERSION
from src.utils.io import ensure_dir


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def _split_pipe(value: str) -> list[str]:
    return [part for part in str(value or "").split("|") if part]


def _is_bad_number(value: str) -> bool:
    if value in {None, ""}:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isnan(number) or math.isinf(number)


def _int_value(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def check_raw_log_schema(raw_dir: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    tables: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for filename in LOG_FILENAMES:
        path = raw_dir / filename
        if not path.exists():
            errors.append(f"missing_file: {filename}")
            continue
        columns, rows = _read_csv(path)
        expected = EXPORT_LOG_COLUMNS[filename]
        if columns != expected:
            errors.append(f"schema_mismatch: {filename}: expected={expected}, actual={columns}")
        tables[filename] = (columns, rows)

    for filename, (_, rows) in tables.items():
        for row_idx, row in enumerate(rows):
            for column, value in row.items():
                if _is_bad_number(value):
                    errors.append(f"bad_numeric_value: {filename}: row={row_idx}: column={column}: value={value}")
                    break

    for filename in ["node_log.csv", "link_log.csv", "service_log.csv", "path_log.csv"]:
        if filename not in tables:
            continue
        rows = tables[filename][1]
        times = [_int_value(row.get("time", "")) for row in rows]
        if any(time is None for time in times):
            errors.append(f"time_not_parseable: {filename}")
        elif times != sorted(times):
            errors.append(f"time_not_sorted: {filename}")

    service_ids = {
        _int_value(row.get("service_id", ""))
        for row in tables.get("service_log.csv", ([], []))[1]
        if _int_value(row.get("service_id", "")) is not None
    }
    path_service_ids = {
        _int_value(row.get("service_id", ""))
        for row in tables.get("path_log.csv", ([], []))[1]
        if _int_value(row.get("service_id", "")) is not None
    }
    sla_service_ids = {
        _int_value(row.get("service_id", ""))
        for row in tables.get("sla_log.csv", ([], []))[1]
        if _int_value(row.get("service_id", "")) is not None
    }
    if service_ids and not service_ids.issubset(sla_service_ids):
        errors.append("service_id_alignment_failed: service_log has ids missing from sla_log")
    if path_service_ids and not path_service_ids.issubset(sla_service_ids):
        errors.append("service_id_alignment_failed: path_log has ids missing from sla_log")
    if service_ids != path_service_ids:
        errors.append("service_id_alignment_warning: service_log and path_log service_id sets differ")

    link_ids: set[str] = set()
    for row in tables.get("link_log.csv", ([], []))[1]:
        src = row.get("src", "")
        dst = row.get("dst", "")
        if src and dst:
            link_ids.add(f"{src}-{dst}")
            link_ids.add(f"{dst}-{src}")

    for row_idx, row in enumerate(tables.get("path_log.csv", ([], []))[1]):
        path_nodes = _split_pipe(row.get("path_nodes", ""))
        path_links = _split_pipe(row.get("path_links", ""))
        if len(path_nodes) < 2:
            errors.append(f"path_nodes_not_parseable: row={row_idx}")
        if not path_links:
            errors.append(f"path_links_not_parseable: row={row_idx}")
        for link_id in path_links:
            if link_id not in link_ids:
                errors.append(f"path_link_missing_from_link_log: row={row_idx}: link={link_id}")
                break

    return not errors, errors


def write_report(raw_dir: Path, passed: bool, errors: list[str]) -> Path:
    report_path = raw_dir / "check_raw_log_schema_report.txt"
    ensure_dir(report_path.parent)
    lines = [
        f"raw_log_schema_version: {LOG_SCHEMA_VERSION}",
        f"raw_dir: {raw_dir}",
        f"status: {'passed' if passed else 'failed'}",
        "",
    ]
    if errors:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("No schema issues found.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    passed, errors = check_raw_log_schema(raw_dir)
    report_path = write_report(raw_dir, passed, errors)
    print(f"Wrote raw log schema report to {report_path}")
    if not passed:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print("[RAW LOG SCHEMA CHECK] passed")


if __name__ == "__main__":
    main()
