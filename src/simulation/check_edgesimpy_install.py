from __future__ import annotations

import platform
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.edgesimpy_adapter import find_edgesimpy_class, try_import_edgesimpy


KEY_CLASSES = ["Simulator", "EdgeServer", "Service", "User", "Application"]


def main() -> None:
    print(f"Python: {platform.python_version()}")
    installed, module, error = try_import_edgesimpy()
    if not installed:
        print("[EdgeSimPy Check Failed]")
        print("EdgeSimPy is not installed or not importable.")
        print("Please run:")
        print("pip install -r requirements-edgesimpy.txt")
        if error:
            print(f"import_error: {error}")
        raise SystemExit(1)

    print("[EdgeSimPy Check Passed]")
    print("status: edgesimpy_import_only")
    print(f"module: edge_sim_py")
    print(f"path: {getattr(module, '__file__', 'unknown')}")
    available = [name for name in KEY_CLASSES if find_edgesimpy_class(module, name) is not None]
    print(f"available_key_classes: {','.join(available) if available else 'unknown'}")
    print("class_signatures:")
    for class_name in KEY_CLASSES:
        cls = find_edgesimpy_class(module, class_name)
        if cls is None:
            print(f"- {class_name}: not found")
            continue
        try:
            signature = inspect.signature(cls)
        except Exception as exc:
            signature = f"<signature unavailable: {exc}>"
        print(f"- {class_name}: {cls.__module__}.{cls.__name__}{signature}")
    print("note: import success is not sufficient for effective_simulator_backend=edgesimpy_real.")


if __name__ == "__main__":
    main()
