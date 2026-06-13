from __future__ import annotations

from pathlib import Path

from src.simulation.log_schema import LOG_FILENAMES, export_fieldnames, normalize_log_row
from src.utils.io import write_csv_rows


def export_logs(out_dir: str | Path, logs: dict[str, list[dict]]) -> None:
    out_dir = Path(out_dir)
    for filename in LOG_FILENAMES:
        rows = [normalize_log_row(filename, row) for row in logs.get(filename, [])]
        write_csv_rows(out_dir / filename, rows, fieldnames=export_fieldnames(filename, rows))
        print(f"Wrote {len(rows)} rows to {out_dir / filename}")
