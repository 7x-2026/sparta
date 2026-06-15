from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulation.edgesimpy_adapter import find_edgesimpy_class, try_import_edgesimpy


KEY_CLASSES = ["Simulator", "EdgeServer", "Service", "User", "Application"]


def _local_source_root() -> Path:
    return ROOT / "simulators" / "EdgeSimPy" / "edge_sim_py"


def _source_signature(class_name: str) -> str | None:
    root = _local_source_root()
    if not root.exists():
        return None
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                init = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
                if init is None:
                    return f"{class_name}()"
                args = [arg.arg for arg in init.args.args if arg.arg != "self"]
                defaults = len(init.args.defaults)
                required = len(args) - defaults
                parts: list[str] = []
                for idx, arg in enumerate(args):
                    parts.append(arg if idx < required else f"{arg}=...")
                return f"{class_name}({', '.join(parts)})  # source: {path.relative_to(ROOT)}"
    return None


def main() -> None:
    installed, module, error = try_import_edgesimpy()
    if not installed or module is None:
        print("[EdgeSimPy API Inspect Failed]")
        print(f"import_error: {error}")
        print("local_source_signature_fallback:")
        for class_name in KEY_CLASSES:
            print(f"- {class_name}: {_source_signature(class_name) or 'not found'}")
        raise SystemExit(1)

    print("[EdgeSimPy API Inspect Passed]")
    print(f"module: edge_sim_py")
    print(f"path: {getattr(module, '__file__', 'unknown')}")
    top_level = [name for name in KEY_CLASSES if hasattr(module, name)]
    print(f"top_level_key_classes: {','.join(top_level) if top_level else 'none'}")
    print("class_signatures:")
    missing: list[str] = []
    for class_name in KEY_CLASSES:
        cls = find_edgesimpy_class(module, class_name)
        if cls is None:
            missing.append(class_name)
            print(f"- {class_name}: not found")
            continue
        try:
            signature = inspect.signature(cls)
        except Exception as exc:
            signature = f"<signature unavailable: {exc}>"
        print(f"- {class_name}: {cls.__module__}.{cls.__name__}{signature}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
