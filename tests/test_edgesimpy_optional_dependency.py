from __future__ import annotations

from pathlib import Path

from src.simulation.edgesimpy_adapter import run_edgesimpy_adapter, try_import_edgesimpy


def test_edgesimpy_import_is_delayed():
    installed, module, error = try_import_edgesimpy()
    if installed:
        assert module is not None
        assert error is None
    else:
        assert module is None
        assert error


def test_real_backend_reports_missing_optional_dependency(tmp_path):
    config = {
        "seed": 42,
        "experiment": {"name": "edgesimpy_real_missing_test", "data_seed": 42, "train_seed": 0},
        "data": {"source": "edgesimpy", "raw_logs_dir": str(tmp_path)},
        "edgesimpy": {"backend": "edgesimpy"},
    }
    installed, _, _ = try_import_edgesimpy()
    expected = "verified real EdgeSimPy objects" if installed else "pip install -r requirements-edgesimpy.txt"
    import pytest

    with pytest.raises(RuntimeError, match=expected):
        run_edgesimpy_adapter(config, Path(tmp_path))
