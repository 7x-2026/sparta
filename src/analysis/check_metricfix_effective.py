from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.io import ensure_dir


LOSS_KEYS = [
    "lambda_node",
    "lambda_link",
    "lambda_metric",
    "metric_loss_type",
    "metric_class_weighting",
    "metric_focal_gamma",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--metricfix_config", default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_path(value: str | Path | None, base: Path = ROOT) -> Path | None:
    if value in [None, ""]:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base / path).resolve()


def normalize_value(value):
    if isinstance(value, float):
        return round(value, 10)
    return value


def loss_subset(config: dict) -> dict:
    loss_cfg = config.get("loss", {})
    return {key: normalize_value(loss_cfg.get(key)) for key in LOSS_KEYS}


def normalize_variant(variant: str | None) -> str | None:
    if variant is None:
        return None
    text = str(variant).strip()
    if not text:
        return None
    if any(ch in text for ch in ["/", "\\", ":"]):
        raise ValueError(f"Invalid variant name: {variant}")
    return text


def sparta_artifact_name(variant: str | None) -> str:
    return f"sparta_{variant}" if variant else "sparta"


def compare_loss(actual: dict, expected: dict) -> dict:
    rows = {}
    for key in LOSS_KEYS:
        rows[key] = {
            "actual": actual.get(key),
            "expected": expected.get(key),
            "matches": actual.get(key) == expected.get(key),
        }
    return rows


def infer_metricfix_config_path(run_dir: Path, manifest: dict, args_config: str | None) -> Path:
    if args_config:
        path = resolve_path(args_config)
        if path is None:
            raise FileNotFoundError(args_config)
        return path
    seed = manifest.get("data_seed", manifest.get("seed"))
    if seed is None:
        run_name = run_dir.name
        for token in ["seed42", "seed2025", "seed3407"]:
            if token in run_name:
                seed = token.replace("seed", "")
                break
    if seed is None:
        raise RuntimeError("Cannot infer metricfix config path: missing data_seed in manifest/run name")
    return ROOT / "configs" / f"sparta_edgesimpy_real_strict_seed{seed}_metricfix.yaml"


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def latest_existing(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def checkpoint_metadata(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"exists": False}
    data = {"exists": True, "path": str(path), "mtime": path.stat().st_mtime, "size": path.stat().st_size}
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        data["epoch"] = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        data["best_metric"] = checkpoint.get("best_metric") if isinstance(checkpoint, dict) else None
        data["loss_config"] = loss_subset(ckpt_config) if ckpt_config else {}
    except Exception as exc:
        data["load_error"] = str(exc)
    return data


def path_same(a: str | Path | None, b: str | Path | None) -> bool:
    pa = resolve_path(a) if a not in [None, ""] else None
    pb = resolve_path(b) if b not in [None, ""] else None
    if pa is None or pb is None:
        return False
    try:
        return pa.resolve() == pb.resolve()
    except Exception:
        return str(pa) == str(pb)


def train_log_check(metricfix_cfg: dict, run_dir: Path, variant: str | None = None) -> dict:
    artifact = sparta_artifact_name(variant)
    if variant:
        log_dir = run_dir / "logs"
        csv_path = log_dir / f"train_log_{artifact}.csv"
        text_path = log_dir / f"{artifact}_train.log"
    else:
        log_dir = resolve_path(metricfix_cfg.get("project", {}).get("log_dir")) or (run_dir / "logs_metricfix")
        csv_path = log_dir / "train_log_sparta.csv"
        text_path = log_dir / "sparta_train.log"
    rows = read_csv_rows(csv_path)
    text = read_text_if_exists(text_path)
    fieldnames = list(rows[0].keys()) if rows else []
    metric_class_weights_values = [row.get("metric_class_weights", "") for row in rows if row.get("metric_class_weights", "")]
    gamma_text_tokens = ["metric_focal_gamma", "focal_gamma", "metric focal gamma", "metric_focal_gamma=2.0"]
    has_gamma = any(token in text for token in gamma_text_tokens) or any(
        any(token in str(row) for token in gamma_text_tokens) for row in rows
    )
    return {
        "log_dir": str(log_dir),
        "train_log_csv": str(csv_path),
        "sparta_train_log": str(text_path),
        "train_log_csv_exists": csv_path.exists(),
        "sparta_train_log_exists": text_path.exists(),
        "epochs_logged": len(rows),
        "has_variant": "variant" in fieldnames or f"variant={variant}" in text,
        "has_metric_loss_type": "metric_loss_type" in fieldnames or "metric_loss_type" in text,
        "has_loss_metric": "loss_metric" in fieldnames,
        "has_loss_risk": "loss_risk" in fieldnames,
        "has_loss_node": "loss_node" in fieldnames,
        "has_loss_link": "loss_link" in fieldnames,
        "has_metric_class_weights": "metric_class_weights" in fieldnames and bool(metric_class_weights_values),
        "metric_class_weights_example": metric_class_weights_values[0] if metric_class_weights_values else "",
        "has_metric_focal_gamma": has_gamma,
        "note": "metric_focal_gamma is expected in logs; older metricfix runs may lack this field even if focal loss was configured.",
    }


def eval_check(
    metricfix_cfg: dict,
    run_dir: Path,
    metricfix_ckpt: Path | None,
    original_ckpt: Path | None,
    variant: str | None = None,
) -> dict:
    artifact = sparta_artifact_name(variant)
    if variant:
        result_candidates = [run_dir / "results" / f"{artifact}_test_results.csv"]
    else:
        result_candidates = [
            (resolve_path(metricfix_cfg.get("project", {}).get("output_dir")) or (run_dir / "results_metricfix")) / "sparta_test_results.csv",
            run_dir / "results" / "sparta_test_results.csv",
        ]
    existing_results = [path for path in result_candidates if path.exists()]
    rows_by_path = {str(path): read_csv_rows(path) for path in existing_results}
    selected_path = None
    selected_row = None
    for path in existing_results:
        rows = rows_by_path[str(path)]
        if rows:
            row = rows[0]
            if metricfix_ckpt is not None and path_same(row.get("checkpoint_path"), metricfix_ckpt):
                selected_path = path
                selected_row = row
                break
    if selected_row is None and existing_results:
        selected_path = existing_results[0]
        selected_row = rows_by_path[str(selected_path)][0] if rows_by_path[str(selected_path)] else {}

    all_ckpts = [path for path in [metricfix_ckpt, original_ckpt] if path is not None]
    newest_ckpt = latest_existing(all_ckpts)
    selected_ckpt = selected_row.get("checkpoint_path") if selected_row else ""
    return {
        "result_candidates": [str(path) for path in result_candidates],
        "existing_results": [str(path) for path in existing_results],
        "selected_result_path": str(selected_path) if selected_path else "",
        "checkpoint_path_in_result": selected_ckpt,
        "metricfix_checkpoint_path": str(metricfix_ckpt) if metricfix_ckpt else "",
        "original_checkpoint_path": str(original_ckpt) if original_ckpt else "",
        "newest_sparta_best_checkpoint": str(newest_ckpt) if newest_ckpt else "",
        "eval_uses_metricfix_checkpoint": path_same(selected_ckpt, metricfix_ckpt),
        "eval_uses_newest_checkpoint": path_same(selected_ckpt, newest_ckpt),
        "eval_uses_original_checkpoint": path_same(selected_ckpt, original_ckpt),
    }


def main() -> None:
    args = parse_args()
    variant = normalize_variant(args.variant)
    artifact = sparta_artifact_name(variant)
    run_dir = resolve_path(args.run_dir)
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
    analysis_dir = ensure_dir(run_dir / "analysis")
    manifest_path = run_dir / "run_manifest.json"
    resolved_config_path = run_dir / "config" / "resolved_config.yaml"
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Missing resolved config: {resolved_config_path}")

    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    resolved_cfg = load_config(resolved_config_path)
    metricfix_config_path = infer_metricfix_config_path(run_dir, manifest, args.metricfix_config)
    if not metricfix_config_path.exists():
        raise FileNotFoundError(f"Missing metricfix config: {metricfix_config_path}")
    metricfix_cfg = load_config(metricfix_config_path)

    resolved_loss = loss_subset(resolved_cfg)
    metricfix_loss = loss_subset(metricfix_cfg)
    resolved_vs_metricfix = compare_loss(resolved_loss, metricfix_loss)
    resolved_matches_metricfix = all(item["matches"] for item in resolved_vs_metricfix.values())

    metricfix_checkpoint_dir = run_dir / "checkpoints" if variant else resolve_path(metricfix_cfg.get("project", {}).get("checkpoint_dir")) or (run_dir / "checkpoints_metricfix")
    original_checkpoint_dir = run_dir / "checkpoints"
    metricfix_ckpt = metricfix_checkpoint_dir / f"{artifact}_best.pth" if variant else metricfix_checkpoint_dir / "sparta_best.pth"
    original_ckpt = original_checkpoint_dir / "sparta_best.pth"
    metricfix_ckpt_meta = checkpoint_metadata(metricfix_ckpt)
    original_ckpt_meta = checkpoint_metadata(original_ckpt)
    metricfix_ckpt_loss = metricfix_ckpt_meta.get("loss_config", {})
    metricfix_ckpt_matches_metricfix = bool(metricfix_ckpt_loss) and all(
        metricfix_ckpt_loss.get(key) == metricfix_loss.get(key) for key in LOSS_KEYS
    )
    checkpoint_is_newer_than_old = bool(
        metricfix_ckpt.exists()
        and original_ckpt.exists()
        and metricfix_ckpt.stat().st_mtime > original_ckpt.stat().st_mtime
    )

    train_log = train_log_check(metricfix_cfg, run_dir, variant)
    eval_status = eval_check(metricfix_cfg, run_dir, metricfix_ckpt, original_ckpt, variant)

    errors: list[str] = []
    warnings: list[str] = []
    if not resolved_matches_metricfix and not variant:
        errors.append(
            "run/config/resolved_config.yaml does not match metricfix loss config. "
            "If training was launched with --resume_run, the pipeline may still be using the old resolved_config.yaml."
        )
    if not resolved_matches_metricfix and variant:
        warnings.append(
            "run/config/resolved_config.yaml does not match metricfix loss config; this is acceptable only if the "
            "variant checkpoint/log/result were produced from the metricfix config."
        )
    if not train_log["train_log_csv_exists"] and not train_log["sparta_train_log_exists"]:
        errors.append(f"Variant train log not found for artifact {artifact}.")
    if variant and not train_log["has_variant"]:
        errors.append(f"Variant train log does not record variant={variant}.")
    if not train_log["has_metric_loss_type"]:
        errors.append("Metricfix train log does not contain metric_loss_type.")
    if not train_log["has_loss_metric"]:
        errors.append("Metricfix train log does not contain loss_metric.")
    if not train_log["has_metric_class_weights"]:
        errors.append("Metricfix train log does not contain metric_class_weights.")
    if not train_log["has_metric_focal_gamma"]:
        errors.append("Metricfix train log does not contain metric_focal_gamma/focal_gamma.")
    if not metricfix_ckpt.exists():
        errors.append(f"Metricfix checkpoint not found: {metricfix_ckpt}")
    if metricfix_ckpt.exists() and original_ckpt.exists() and not checkpoint_is_newer_than_old:
        errors.append("Metricfix sparta_best.pth is not newer than the old checkpoint.")
    if metricfix_ckpt.exists() and not metricfix_ckpt_matches_metricfix:
        errors.append("Metricfix checkpoint embedded config does not match metricfix loss config.")
    if not eval_status["eval_uses_metricfix_checkpoint"]:
        errors.append(f"Evaluation result does not use the expected variant checkpoint: {metricfix_ckpt}.")
    if variant and eval_status["eval_uses_original_checkpoint"]:
        errors.append("Evaluation result still uses the original sparta_best.pth instead of the variant checkpoint.")
    if not eval_status["eval_uses_newest_checkpoint"]:
        warnings.append("Evaluation result does not point to the newest sparta_best.pth among old/metricfix checkpoints.")

    report = {
        "run_dir": str(run_dir),
        "variant": variant or "",
        "artifact": artifact,
        "manifest_path": str(manifest_path),
        "resolved_config_path": str(resolved_config_path),
        "metricfix_config_path": str(metricfix_config_path),
        "run_id": manifest.get("run_id", run_dir.name),
        "data_seed": manifest.get("data_seed", manifest.get("seed")),
        "resolved_config_experiment_name": resolved_cfg.get("experiment", {}).get("name"),
        "metricfix_config_experiment_name": metricfix_cfg.get("experiment", {}).get("name"),
        "resolved_loss_config": resolved_loss,
        "metricfix_loss_config": metricfix_loss,
        "resolved_vs_metricfix_loss": resolved_vs_metricfix,
        "resolved_config_matches_metricfix_loss": resolved_matches_metricfix,
        "train_log_check": train_log,
        "checkpoint_check": {
            "metricfix_checkpoint": metricfix_ckpt_meta,
            "old_checkpoint": original_ckpt_meta,
            "metricfix_checkpoint_is_newer_than_old": checkpoint_is_newer_than_old,
            "metricfix_checkpoint_matches_metricfix_loss": metricfix_ckpt_matches_metricfix,
        },
        "eval_check": eval_status,
        "effective": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
    out_path = analysis_dir / (f"{variant}_effective_check.json" if variant else "metricfix_effective_check.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    if errors:
        print("Metricfix effective check found errors:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    main()
