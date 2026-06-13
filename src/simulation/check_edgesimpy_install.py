from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.edgesimpy_adapter import try_import_edgesimpy


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
    print(f"module: edge_sim_py")
    print(f"path: {getattr(module, '__file__', 'unknown')}")
    available = [name for name in ["Simulator", "EdgeServer", "Service", "User", "Application"] if hasattr(module, name)]
    print(f"available_key_classes: {','.join(available) if available else 'unknown'}")


if __name__ == "__main__":
    main()
