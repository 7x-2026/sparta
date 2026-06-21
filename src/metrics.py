from __future__ import annotations

import math
from collections import Counter

import torch

try:
    from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score
except Exception:  # pragma: no cover - optional runtime dependency
    sklearn_roc_auc_score = None

METRIC_NAMES = {0: "delay", 1: "loss", 2: "cpu", 3: "queue", 4: "bandwidth"}
RISK_LABELS = [0, 1, 2]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _binary_auc(y_true_binary: list[int], y_score: list[float]) -> float:
    pairs = sorted(zip(y_score, y_true_binary), key=lambda item: item[0])
    n_pos = sum(y_true_binary)
    n_neg = len(y_true_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("binary AUC requires both positive and negative samples")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _ovr_auc_fallback(y_true: list[int], y_score: list[list[float]], average: str) -> float:
    aucs: list[float] = []
    weights: list[int] = []
    for label in RISK_LABELS:
        binary = [1 if true == label else 0 for true in y_true]
        scores = [row[label] for row in y_score]
        aucs.append(_binary_auc(binary, scores))
        weights.append(sum(binary))
    if average == "macro":
        return sum(aucs) / len(aucs)
    if average == "weighted":
        total = sum(weights)
        return sum(auc * weight for auc, weight in zip(aucs, weights)) / total if total else math.nan
    raise ValueError(f"Unsupported AUC average={average}")


def _risk_score_matrix(y_score) -> list[list[float]] | None:
    if y_score is None:
        return None
    rows = []
    try:
        iterator = list(y_score)
    except TypeError:
        return None
    for row in iterator:
        if isinstance(row, torch.Tensor):
            row = row.detach().cpu().tolist()
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return None
        try:
            rows.append([float(row[0]), float(row[1]), float(row[2])])
        except (TypeError, ValueError):
            return None
    return rows


def compute_risk_auc_metrics(y_true, y_score=None) -> dict[str, float | str]:
    true = [int(x) for x in y_true]
    scores = _risk_score_matrix(y_score)
    out: dict[str, float | str] = {
        "auc": math.nan,
        "risk_auc_macro_ovr": math.nan,
        "risk_auc_weighted_ovr": math.nan,
        "risk_auc_binary": math.nan,
        "auc_error": "",
    }
    errors: list[str] = []
    if scores is None:
        out["auc_error"] = "risk probability y_score is missing or not shaped [N,3]"
        return out
    if len(scores) != len(true):
        out["auc_error"] = f"risk probability length mismatch: y_true={len(true)} y_score={len(scores)}"
        return out
    if set(true) != set(RISK_LABELS):
        errors.append(f"risk_auc_macro_ovr/weighted_ovr require all classes {RISK_LABELS}; present={sorted(set(true))}")
    else:
        for average, key in [("macro", "risk_auc_macro_ovr"), ("weighted", "risk_auc_weighted_ovr")]:
            try:
                if sklearn_roc_auc_score is not None:
                    value = sklearn_roc_auc_score(true, scores, multi_class="ovr", average=average, labels=RISK_LABELS)
                else:
                    value = _ovr_auc_fallback(true, scores, average)
                out[key] = float(value)
            except Exception as exc:
                errors.append(f"{key}: {exc}")
    try:
        binary_true = [1 if label in {1, 2} else 0 for label in true]
        binary_score = [row[1] + row[2] for row in scores]
        if sklearn_roc_auc_score is not None:
            out["risk_auc_binary"] = float(sklearn_roc_auc_score(binary_true, binary_score))
        else:
            out["risk_auc_binary"] = float(_binary_auc(binary_true, binary_score))
    except Exception as exc:
        errors.append(f"risk_auc_binary: {exc}")
    out["auc"] = out["risk_auc_macro_ovr"]
    out["auc_error"] = "; ".join(errors)
    return out


def _majority_for_samples(samples: list[dict], field: str) -> dict:
    if not samples:
        return {
            "majority_value": "",
            "majority_count": 0,
            "majority_acc": math.nan,
            "denominator": 0,
            "count_by_class": {},
        }
    counts = Counter(int(sample[field]) for sample in samples)
    majority_value, majority_count = counts.most_common(1)[0]
    return {
        "majority_value": majority_value,
        "majority_count": majority_count,
        "majority_acc": majority_count / len(samples),
        "denominator": len(samples),
        "count_by_class": {str(key): int(value) for key, value in sorted(counts.items())},
    }


def compute_attribution_majority_baseline(samples: list[dict], split: str = "test", attr_mask_only: bool = True) -> dict:
    """Compute attribution majority baselines with a consistent valid-attribution default.

    By default, normal samples are excluded because their attribution labels should be
    ignored (`attr_mask=0`, labels=-100). The return still includes including-normal
    values so audit reports can make the denominator explicit instead of mixing both
    conventions.
    """
    valid_samples = [sample for sample in samples if int(sample.get("attr_mask", 0)) == 1]
    baseline_samples = valid_samples if attr_mask_only else samples

    out = {
        "split": split,
        "attr_mask_only": bool(attr_mask_only),
        "total_samples": len(samples),
        "valid_attr_samples": len(valid_samples),
        "denominator": len(baseline_samples),
    }
    for field, prefix in [("risk_node", "node"), ("risk_link", "link"), ("risk_metric", "metric")]:
        selected = _majority_for_samples(baseline_samples, field)
        valid = _majority_for_samples(valid_samples, field)
        including = _majority_for_samples(samples, field)
        out.update(
            {
                f"{prefix}_majority_value": selected["majority_value"],
                f"{prefix}_majority_count": selected["majority_count"],
                f"{prefix}_majority_acc": selected["majority_acc"],
                f"{prefix}_majority_acc_valid_attr_only": valid["majority_acc"],
                f"{prefix}_majority_acc_including_normal": including["majority_acc"],
                f"{prefix}_majority_count_by_class": selected["count_by_class"],
                f"{prefix}_denominator": selected["denominator"],
            }
        )
        if field == "risk_metric":
            out["metric_majority_name"] = METRIC_NAMES.get(selected["majority_value"], str(selected["majority_value"])) if selected["majority_value"] != "" else ""
            out["metric_majority_count_by_class_named"] = {
                METRIC_NAMES.get(int(key), str(key)): value
                for key, value in selected["count_by_class"].items()
                if str(key).lstrip("-").isdigit()
            }
    return out


def compute_classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float | str]:
    true = [int(x) for x in y_true]
    pred = [int(x) for x in y_pred]
    total = len(true)
    accuracy = _safe_div(sum(int(t == p) for t, p in zip(true, pred)), total)
    f1s = []
    recalls = {}
    for cls in [0, 1, 2]:
        tp = sum(1 for t, p in zip(true, pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true, pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true, pred) if t == cls and p != cls)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        recalls[cls] = recall
        f1s.append(_safe_div(2 * precision * recall, precision + recall))
    out = {
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s),
        "risk_recall": recalls[1],
        "violation_recall": recalls[2],
    }
    out.update(compute_risk_auc_metrics(true, y_score))
    return out


def compute_attribution_metrics(labels: dict[str, list[int]], preds: dict[str, list[int]], has_attribution: bool = True) -> dict[str, float]:
    empty = {
        "node_attr_acc": math.nan,
        "link_attr_acc": math.nan,
        "metric_attr_acc": math.nan,
        "metric_attr_macro_f1": math.nan,
        "metric_attr_balanced_acc": math.nan,
        "metric_attr_per_class_recall_delay": math.nan,
        "metric_attr_per_class_recall_loss": math.nan,
        "metric_attr_per_class_recall_cpu": math.nan,
        "metric_attr_per_class_recall_queue": math.nan,
        "metric_attr_per_class_recall_bandwidth": math.nan,
    }
    if not has_attribution:
        return empty
    valid = [i for i, mask in enumerate(labels["attr_mask"]) if bool(mask)]
    if not valid:
        return empty

    def acc(label_key: str, pred_key: str) -> float:
        return sum(int(labels[label_key][i] == preds[pred_key][i]) for i in valid) / len(valid)

    metric_true = [int(labels["risk_metric"][i]) for i in valid]
    metric_pred = [int(preds["risk_metric"][i]) for i in valid]
    recalls: dict[int, float] = {}
    f1s: list[float] = []
    for cls in [0, 1, 2, 3, 4]:
        tp = sum(1 for true, pred in zip(metric_true, metric_pred) if true == cls and pred == cls)
        fp = sum(1 for true, pred in zip(metric_true, metric_pred) if true != cls and pred == cls)
        fn = sum(1 for true, pred in zip(metric_true, metric_pred) if true == cls and pred != cls)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        recalls[cls] = recall
        f1s.append(_safe_div(2 * precision * recall, precision + recall))

    return {
        "node_attr_acc": acc("risk_node", "risk_node"),
        "link_attr_acc": acc("risk_link", "risk_link"),
        "metric_attr_acc": acc("risk_metric", "risk_metric"),
        "metric_attr_macro_f1": sum(f1s) / len(f1s),
        "metric_attr_balanced_acc": sum(recalls.values()) / len(recalls),
        "metric_attr_per_class_recall_delay": recalls[0],
        "metric_attr_per_class_recall_loss": recalls[1],
        "metric_attr_per_class_recall_cpu": recalls[2],
        "metric_attr_per_class_recall_queue": recalls[3],
        "metric_attr_per_class_recall_bandwidth": recalls[4],
    }


@torch.no_grad()
def collect_predictions(model, dataloader, device: torch.device) -> tuple[dict, dict, bool]:
    model.eval()
    labels = {"risk_label": [], "risk_node": [], "risk_link": [], "risk_metric": [], "attr_mask": []}
    preds = {"risk_label": [], "risk_node": [], "risk_link": [], "risk_metric": [], "risk_probs": []}
    has_attr = False
    for batch in dataloader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(batch)
        risk_probs = torch.softmax(outputs["risk_logits"], dim=-1)
        preds["risk_label"].extend(outputs["risk_logits"].argmax(dim=-1).cpu().tolist())
        preds["risk_probs"].extend(risk_probs.cpu().tolist())
        labels["risk_label"].extend(batch["risk_label"].cpu().tolist())
        labels["risk_node"].extend(batch["risk_node"].cpu().tolist())
        labels["risk_link"].extend(batch["risk_link"].cpu().tolist())
        labels["risk_metric"].extend(batch["risk_metric"].cpu().tolist())
        labels["attr_mask"].extend(batch["attr_mask"].cpu().tolist())
        if all(key in outputs for key in ["node_logits", "link_logits", "metric_logits"]):
            has_attr = True
            preds["risk_node"].extend(outputs["node_logits"].argmax(dim=-1).cpu().tolist())
            preds["risk_link"].extend(outputs["link_logits"].argmax(dim=-1).cpu().tolist())
            preds["risk_metric"].extend(outputs["metric_logits"].argmax(dim=-1).cpu().tolist())
        else:
            n = batch["risk_label"].shape[0]
            preds["risk_node"].extend([-100] * n)
            preds["risk_link"].extend([-100] * n)
            preds["risk_metric"].extend([-100] * n)
    return labels, preds, has_attr
