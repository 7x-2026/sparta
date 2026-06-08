from __future__ import annotations

from pathlib import Path

from src.utils.io import write_csv_rows


LOG_FILENAMES = [
    "node_log.csv",
    "link_log.csv",
    "service_log.csv",
    "path_log.csv",
    "sla_log.csv",
]


def export_logs(out_dir: str | Path, logs: dict[str, list[dict]]) -> None:
    out_dir = Path(out_dir)
    for filename in LOG_FILENAMES:
        rows = logs.get(filename, [])
        write_csv_rows(out_dir / filename, rows)
        print(f"Wrote {len(rows)} rows to {out_dir / filename}")
