from __future__ import annotations

import math

import torch


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
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
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s),
        "risk_recall": recalls[1],
        "violation_recall": recalls[2],
        "auc": math.nan,
    }


def compute_attribution_metrics(labels: dict[str, list[int]], preds: dict[str, list[int]], has_attribution: bool = True) -> dict[str, float]:
    if not has_attribution:
        return {"node_attr_acc": math.nan, "link_attr_acc": math.nan, "metric_attr_acc": math.nan}
    valid = [i for i, mask in enumerate(labels["attr_mask"]) if bool(mask)]
    if not valid:
        return {"node_attr_acc": math.nan, "link_attr_acc": math.nan, "metric_attr_acc": math.nan}

    def acc(label_key: str, pred_key: str) -> float:
        return sum(int(labels[label_key][i] == preds[pred_key][i]) for i in valid) / len(valid)

    return {
        "node_attr_acc": acc("risk_node", "risk_node"),
        "link_attr_acc": acc("risk_link", "risk_link"),
        "metric_attr_acc": acc("risk_metric", "risk_metric"),
    }


@torch.no_grad()
def collect_predictions(model, dataloader, device: torch.device) -> tuple[dict, dict, bool]:
    model.eval()
    labels = {"risk_label": [], "risk_node": [], "risk_link": [], "risk_metric": [], "attr_mask": []}
    preds = {"risk_label": [], "risk_node": [], "risk_link": [], "risk_metric": []}
    has_attr = False
    for batch in dataloader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        outputs = model(batch)
        preds["risk_label"].extend(outputs["risk_logits"].argmax(dim=-1).cpu().tolist())
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
