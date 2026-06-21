from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from src.models.common import MLP, masked_mean


METRIC_NAMES = ("delay", "loss", "cpu", "queue", "bandwidth")
DEFAULT_FEATURE_SCHEMA = {
    "node": [
        "cpu_util",
        "mem_util",
        "queue_len",
        "available_cpu",
        "available_mem",
        "node_type",
        "node_degree",
        "is_current_edge",
    ],
    "link": [
        "delay_ms",
        "bandwidth",
        "loss",
        "jitter_ms",
        "queue_delay_ms",
        "bandwidth_util",
        "is_current_link",
        "hop_position",
    ],
    "service": [
        "request_rate",
        "response_time",
        "service_type",
        "base_response_time",
        "dependency_count",
    ],
    "sla": [
        "max_delay",
        "max_loss",
        "min_bandwidth",
        "reliability_req",
        "cost_weight",
    ],
}
_WARNED_MISSING_FEATURES: set[str] = set()


def _schema_from_config(config: dict | None = None) -> dict[str, list[str]]:
    data_cfg = (config or {}).get("data", {})
    schema = {}
    for group, defaults in DEFAULT_FEATURE_SCHEMA.items():
        names = data_cfg.get(f"{group}_feature_names") or data_cfg.get(f"{group}_feat_names") or defaults
        schema[group] = [str(name) for name in names]
    return schema


def _feature_index(schema: dict[str, list[str]], group: str, aliases: list[str]) -> int | None:
    names = [name.lower() for name in schema.get(group, [])]
    alias_set = {alias.lower() for alias in aliases}
    for idx, name in enumerate(names):
        if name in alias_set or any(alias in name for alias in alias_set):
            return idx
    key = f"{group}:{'/'.join(aliases)}"
    if key not in _WARNED_MISSING_FEATURES:
        print(f"warning: metric evidence feature not found for {key}; filling zero evidence.", flush=True)
        _WARNED_MISSING_FEATURES.add(key)
    return None


def _masked_reduce(x: torch.Tensor, mask: torch.Tensor | None, index: int | None, mode: str = "mean") -> torch.Tensor:
    if index is None or index >= x.shape[-1]:
        return x.new_zeros(x.shape[0])
    values = x[..., index]
    if mask is None:
        if mode == "max":
            return values.amax(dim=tuple(range(1, values.dim())))
        if mode == "min":
            return values.amin(dim=tuple(range(1, values.dim())))
        return values.mean(dim=tuple(range(1, values.dim())))
    mask_t = mask.to(device=x.device, dtype=x.dtype)
    while mask_t.dim() < values.dim():
        mask_t = mask_t.unsqueeze(1)
    mask_t = mask_t.expand_as(values)
    if mode == "max":
        return values.masked_fill(mask_t <= 0, 0.0).amax(dim=tuple(range(1, values.dim())))
    if mode == "min":
        return values.masked_fill(mask_t <= 0, 1.0).amin(dim=tuple(range(1, values.dim())))
    denom = mask_t.sum(dim=tuple(range(1, values.dim()))).clamp_min(1.0)
    return (values * mask_t).sum(dim=tuple(range(1, values.dim()))) / denom


def _last_first_growth(x: torch.Tensor, mask: torch.Tensor | None, index: int | None) -> torch.Tensor:
    if index is None or index >= x.shape[-1] or x.shape[1] < 2:
        return x.new_zeros(x.shape[0])
    last = x[:, -1:, :, :]
    first = x[:, :1, :, :]
    growth = torch.clamp(last[..., index] - first[..., index], min=0.0)
    if mask is None:
        return growth.mean(dim=(1, 2))
    mask_t = mask[:, None, :].to(device=x.device, dtype=x.dtype)
    denom = mask_t.sum(dim=(1, 2)).clamp_min(1.0)
    return (growth * mask_t).sum(dim=(1, 2)) / denom


def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den.abs().clamp_min(1e-6)


def _normalize_evidence(evidence: torch.Tensor, norm: str = "zscore") -> torch.Tensor:
    norm = str(norm or "zscore").lower()
    if norm in {"none", "raw", "pressure01", "pressure02", "labelrule"}:
        return evidence
    if norm == "minmax":
        lo = evidence.amin(dim=0, keepdim=True)
        hi = evidence.amax(dim=0, keepdim=True)
        return (evidence - lo) / (hi - lo).clamp_min(1e-6)
    if norm == "zscore":
        mean = evidence.mean(dim=0, keepdim=True)
        std = evidence.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (evidence - mean) / std
    raise ValueError(f"Unsupported metric_evidence_norm: {norm}")


def _stat_value(evidence_stats: dict | None, key: str, default: float = 1.0) -> float:
    if not evidence_stats:
        return default
    value = evidence_stats.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    if value <= 1e-6:
        return default
    return value


def _clip01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _quantile_grid(values: torch.Tensor, steps: int = 101) -> list[float]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return [0.0 for _ in range(steps)]
    probs = torch.linspace(0.0, 1.0, steps=steps, device=flat.device)
    return [float(x) for x in torch.quantile(flat, probs).cpu().tolist()]


def _rank_from_stats(values: torch.Tensor, evidence_stats: dict | None, group: str, name: str) -> torch.Tensor:
    if not evidence_stats:
        return _clip01(values)
    grids = evidence_stats.get("rank_quantiles", {})
    grid = grids.get(group, {}).get(name) if isinstance(grids, dict) else None
    if not grid or len(grid) < 2:
        return _clip01(values)
    grid_tensor = values.new_tensor([float(v) for v in grid]).flatten()
    if grid_tensor.numel() < 2 or float((grid_tensor[-1] - grid_tensor[0]).abs().item()) <= 1e-8:
        return _clip01(values)
    flat = values.detach().reshape(-1).contiguous()
    idx = torch.bucketize(flat, grid_tensor.contiguous(), right=False).to(dtype=values.dtype)
    rank = idx / float(max(grid_tensor.numel() - 1, 1))
    return _clip01(rank.reshape_as(values))


def _nested_stat(evidence_stats: dict | None, group: str, name: str, key: str, default: float = 1.0) -> float:
    if not evidence_stats:
        return default
    group_stats = evidence_stats.get(group, {})
    if not isinstance(group_stats, dict):
        return default
    stats = group_stats.get(name, {})
    if not isinstance(stats, dict):
        return default
    try:
        value = float(stats.get(key, default))
    except (TypeError, ValueError):
        value = default
    return value if value > 1e-6 else default


def _metric_pressure_components(batch: dict[str, torch.Tensor], feature_schema: dict | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    schema = feature_schema or DEFAULT_FEATURE_SCHEMA
    node_x = batch["node_x"]
    link_x = batch["link_x"]
    service_x = batch["service_x"]
    sla_x = batch["sla_x"]
    node_mask = batch.get("node_mask")
    link_mask = batch.get("link_mask")

    node_cpu = _feature_index(schema, "node", ["cpu_util", "cpu"])
    node_available_cpu = _feature_index(schema, "node", ["available_cpu", "free_cpu"])
    node_queue = _feature_index(schema, "node", ["queue_len", "queue"])
    link_delay = _feature_index(schema, "link", ["delay_ms", "delay"])
    link_loss = _feature_index(schema, "link", ["loss", "packet_loss"])
    link_queue_delay = _feature_index(schema, "link", ["queue_delay_ms", "queue_delay"])
    link_bandwidth = _feature_index(schema, "link", ["bandwidth"])
    link_bandwidth_util = _feature_index(schema, "link", ["bandwidth_util", "bw_util"])
    service_response = _feature_index(schema, "service", ["response_time", "path_delay"])
    service_request = _feature_index(schema, "service", ["request_rate"])
    sla_max_delay = _feature_index(schema, "sla", ["max_delay"])
    sla_max_loss = _feature_index(schema, "sla", ["max_loss"])
    sla_min_bandwidth = _feature_index(schema, "sla", ["min_bandwidth"])

    response_ev = _masked_reduce(service_x, None, service_response, mode="mean")
    request_ev = _masked_reduce(service_x, None, service_request, mode="mean")
    max_delay = sla_x[:, sla_max_delay] if sla_max_delay is not None else torch.ones_like(response_ev)
    link_delay_ev = torch.maximum(
        _masked_reduce(link_x, link_mask, link_delay, mode="mean"),
        _masked_reduce(link_x, link_mask, link_queue_delay, mode="mean"),
    )
    queue_delay_ev = _masked_reduce(link_x, link_mask, link_queue_delay, mode="mean")
    response_delay_ratio = _safe_ratio(response_ev, max_delay)
    link_delay_ratio = _safe_ratio(link_delay_ev, max_delay)
    queue_delay_ratio = _safe_ratio(queue_delay_ev, max_delay)
    delay_evidence = torch.maximum(torch.maximum(response_delay_ratio, link_delay_ratio), queue_delay_ratio)

    link_loss_ev = _masked_reduce(link_x, link_mask, link_loss, mode="mean")
    max_loss = sla_x[:, sla_max_loss] if sla_max_loss is not None else torch.ones_like(link_loss_ev)
    # max_loss is stored as max_loss / 0.10 in the MVP feature schema, while
    # link loss is already a raw 0..1 loss rate.
    max_loss_raw = max_loss * 0.10 if sla_max_loss is not None else max_loss
    max_loss_ratio = _safe_ratio(link_loss_ev, max_loss_raw)
    loss_evidence = torch.maximum(link_loss_ev, max_loss_ratio)

    cpu_util_mean = _masked_reduce(node_x, node_mask, node_cpu, mode="mean")
    cpu_util_max = _masked_reduce(node_x, node_mask, node_cpu, mode="max")
    if node_available_cpu is None:
        available_cpu_ev = node_x.new_zeros(node_x.shape[0])
    else:
        available_cpu_pressure_mean = 1.0 - _masked_reduce(node_x, node_mask, node_available_cpu, mode="mean")
        available_cpu_pressure_max = 1.0 - _masked_reduce(node_x, node_mask, node_available_cpu, mode="min")
        available_cpu_ev = torch.maximum(available_cpu_pressure_mean, available_cpu_pressure_max)
    cpu_util_ev = torch.maximum(cpu_util_mean, cpu_util_max)
    cpu_evidence = torch.maximum(cpu_util_ev, available_cpu_ev)

    queue_len_ev = _masked_reduce(node_x, node_mask, node_queue, mode="mean")
    queue_growth_ev = _last_first_growth(node_x, node_mask, node_queue)
    # Queue pressure deliberately avoids absolute queue_len dominance. It should
    # reflect growth or explicit queueing delay, leaving high CPU utilization to
    # the CPU pressure dimension.
    queue_evidence = torch.maximum(queue_growth_ev, queue_delay_ratio)

    bandwidth_util_ev = _masked_reduce(link_x, link_mask, link_bandwidth_util, mode="mean")
    if link_bandwidth is None:
        bandwidth_pressure = link_x.new_zeros(link_x.shape[0])
        min_bandwidth_ratio = link_x.new_zeros(link_x.shape[0])
    else:
        available_bandwidth = _masked_reduce(link_x, link_mask, link_bandwidth, mode="mean")
        bandwidth_pressure = 1.0 - available_bandwidth
        if sla_min_bandwidth is not None:
            min_bandwidth_ratio = _safe_ratio(sla_x[:, sla_min_bandwidth], available_bandwidth)
            bandwidth_pressure = torch.maximum(bandwidth_pressure, min_bandwidth_ratio)
        else:
            min_bandwidth_ratio = link_x.new_zeros(link_x.shape[0])
    bandwidth_evidence = torch.maximum(bandwidth_util_ev, bandwidth_pressure)

    evidence = torch.stack(
        [delay_evidence, loss_evidence, cpu_evidence, queue_evidence, bandwidth_evidence],
        dim=-1,
    )
    evidence = torch.nan_to_num(evidence, nan=0.0, posinf=0.0, neginf=0.0)
    aux = {
        "queue_len": queue_len_ev,
        "queue_growth": queue_growth_ev,
        "queue_delay": queue_delay_ev,
        "queue_delay_ratio": queue_delay_ratio,
        "delay_ratio": delay_evidence,
        "loss_ratio": loss_evidence,
        "cpu_pressure": cpu_evidence,
        "bandwidth_pressure": bandwidth_evidence,
        "bandwidth_util": bandwidth_util_ev,
        "min_bandwidth_over_bandwidth": min_bandwidth_ratio,
    }
    return evidence, aux


def _pressure01_from_raw(raw: torch.Tensor, aux: dict[str, torch.Tensor], evidence_stats: dict | None = None) -> torch.Tensor:
    centers = []
    scales = []
    for idx, name in enumerate(METRIC_NAMES):
        default_key = {
            "delay": "delay_ratio_p95",
            "loss": "loss_ratio_p95",
            "cpu": "cpu_pressure_p95",
            "queue": "queue_pressure_p95",
            "bandwidth": "bandwidth_pressure_p95",
        }[name]
        scale = None
        center = 0.0
        if evidence_stats and isinstance(evidence_stats.get("evidence_raw"), dict):
            raw_stats = evidence_stats["evidence_raw"].get(name, {})
            if isinstance(raw_stats, dict):
                scale = raw_stats.get("p95")
                if name == "loss" and raw_stats.get("p90") is not None:
                    scale = raw_stats.get("p90")
                center = raw_stats.get("p50", 0.0)
        if scale is None:
            scale = _stat_value(evidence_stats, default_key, 1.0)
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = 1.0
        try:
            center = float(center)
        except (TypeError, ValueError):
            center = 0.0
        # pressure01 models excess pressure above the train median, scaled by
        # robust train spread. This prevents high-but-normal CPU utilization or
        # bandwidth pressure from dominating every sample.
        denom = scale - center
        centers.append(center)
        scales.append(denom if denom > 1e-6 else (scale if scale > 1e-6 else 1.0))
    center_tensor = raw.new_tensor(centers).unsqueeze(0)
    scale_tensor = raw.new_tensor(scales).unsqueeze(0)
    return _clip01((raw - center_tensor) / scale_tensor)


def _pressure02_from_raw(raw: torch.Tensor, aux: dict[str, torch.Tensor], evidence_stats: dict | None = None) -> torch.Tensor:
    delay = _rank_from_stats(raw[:, 0], evidence_stats, "evidence_raw", "delay")
    loss = _rank_from_stats(raw[:, 1], evidence_stats, "evidence_raw", "loss").pow(0.7)
    cpu = _rank_from_stats(raw[:, 2], evidence_stats, "evidence_raw", "cpu").pow(0.9)

    queue_rank = _rank_from_stats(aux.get("queue_len", raw[:, 3]), evidence_stats, "aux", "queue_len")
    queue_delay_rank = _rank_from_stats(aux.get("queue_delay", raw[:, 3]), evidence_stats, "aux", "queue_delay")
    queue_growth_rank = _rank_from_stats(aux.get("queue_growth", raw[:, 3]), evidence_stats, "aux", "queue_growth")
    queue_delay_p95 = _nested_stat(evidence_stats, "aux", "queue_delay_ratio", "p95", 1.0)
    queue_delay_ratio = _clip01(aux.get("queue_delay_ratio", raw[:, 3]) / queue_delay_p95)
    queue_len_excess = _clip01((queue_rank - 0.65) / 0.35)
    queue_delay_excess = _clip01((queue_delay_rank - 0.50) / 0.50)
    # Absolute queue length is intentionally weak evidence. Sustained queueing
    # delay and growth should dominate queue attribution, otherwise test-time
    # queue_len shifts collapse the baseline into predicting queue everywhere.
    queue = torch.maximum(
        torch.maximum(0.10 * queue_len_excess, 0.30 * queue_delay_excess),
        torch.maximum(1.20 * queue_growth_rank.pow(0.35), 0.40 * queue_delay_ratio),
    )
    queue = _clip01(queue)

    bandwidth_util_rank = _rank_from_stats(
        aux.get("bandwidth_util", raw[:, 4]), evidence_stats, "aux", "bandwidth_util"
    )
    min_bandwidth_rank = _rank_from_stats(
        aux.get("min_bandwidth_over_bandwidth", raw[:, 4]),
        evidence_stats,
        "aux",
        "min_bandwidth_over_bandwidth",
    )
    bandwidth_raw_rank = _rank_from_stats(raw[:, 4], evidence_stats, "evidence_raw", "bandwidth")
    bandwidth = torch.maximum(torch.maximum(bandwidth_util_rank, min_bandwidth_rank), bandwidth_raw_rank).pow(0.9)

    return _clip01(torch.stack([delay, loss, cpu, queue, bandwidth], dim=-1))


def build_metric_evidence(
    batch: dict[str, torch.Tensor],
    feature_schema: dict | None = None,
    norm: str = "zscore",
    evidence_stats: dict | None = None,
) -> torch.Tensor:
    norm = str(norm or "zscore").lower()
    if norm == "labelrule":
        from src.preprocessing.generate_labels import compute_metric_risk_scores

        info = compute_metric_risk_scores(batch)
        scores = info["scores"]
        if not torch.is_tensor(scores):
            scores = torch.as_tensor(scores, dtype=torch.float32)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        return scores.to(device=batch["node_x"].device, dtype=batch["node_x"].dtype)
    raw, aux = _metric_pressure_components(batch, feature_schema)
    if norm == "pressure01":
        return _pressure01_from_raw(raw, aux, evidence_stats)
    if norm == "pressure02":
        return _pressure02_from_raw(raw, aux, evidence_stats)
    return _normalize_evidence(raw, norm=norm)


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p75": float(torch.quantile(flat, 0.75).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
    }


@torch.no_grad()
def compute_metric_evidence_stats_from_loader(dataloader, feature_schema: dict | None = None, norm: str = "pressure01") -> dict:
    raw_parts: list[torch.Tensor] = []
    aux_parts: dict[str, list[torch.Tensor]] = {
        "queue_len": [],
        "queue_growth": [],
        "queue_delay": [],
        "queue_delay_ratio": [],
        "delay_ratio": [],
        "loss_ratio": [],
        "cpu_pressure": [],
        "bandwidth_pressure": [],
        "bandwidth_util": [],
        "min_bandwidth_over_bandwidth": [],
    }
    for batch in dataloader:
        batch = {key: value if torch.is_tensor(value) else value for key, value in batch.items()}
        raw, aux = _metric_pressure_components(batch, feature_schema)
        raw_parts.append(raw.detach().cpu())
        for key in aux_parts:
            aux_parts[key].append(aux[key].detach().cpu())
    raw_all = torch.cat(raw_parts, dim=0) if raw_parts else torch.zeros(0, 5)
    aux_all = {key: torch.cat(parts, dim=0) if parts else torch.zeros(0) for key, parts in aux_parts.items()}
    raw_stats = {name: _tensor_stats(raw_all[:, idx]) for idx, name in enumerate(METRIC_NAMES)}
    scale_stats = {"evidence_raw": raw_stats}
    norm = str(norm or "pressure01").lower()
    if norm == "pressure02":
        rank_quantiles = {
            "evidence_raw": {name: _quantile_grid(raw_all[:, idx]) for idx, name in enumerate(METRIC_NAMES)},
            "aux": {key: _quantile_grid(values) for key, values in aux_all.items()},
        }
        scale_stats = {"evidence_raw": raw_stats, "aux": {key: _tensor_stats(values) for key, values in aux_all.items()}, "rank_quantiles": rank_quantiles}
        pressure = _pressure02_from_raw(raw_all, aux_all, scale_stats)
        version = "pressure02_v5_growth_rank"
    else:
        pressure = _pressure01_from_raw(raw_all, aux_all, scale_stats)
        rank_quantiles = {}
        version = "pressure01_v3_excess"
    evidence_stats = {
        "version": version,
        "norm": norm,
        "queue_len_p95": _tensor_stats(aux_all["queue_len"])["p95"],
        "queue_growth_p95": _tensor_stats(aux_all["queue_growth"])["p95"],
        "queue_delay_p95": _tensor_stats(aux_all["queue_delay"])["p95"],
        "queue_delay_ratio_p95": _tensor_stats(aux_all["queue_delay_ratio"])["p95"],
        "delay_ratio_p95": _tensor_stats(aux_all["delay_ratio"])["p95"],
        "loss_ratio_p95": _tensor_stats(aux_all["loss_ratio"])["p95"],
        "cpu_pressure_p95": _tensor_stats(aux_all["cpu_pressure"])["p95"],
        "queue_pressure_p95": raw_stats["queue"]["p95"],
        "bandwidth_pressure_p95": _tensor_stats(aux_all["bandwidth_pressure"])["p95"],
        "aux": {key: _tensor_stats(values) for key, values in aux_all.items()},
        "evidence_raw": raw_stats,
        "rank_quantiles": rank_quantiles,
        "evidence": {},
    }
    for idx, name in enumerate(METRIC_NAMES):
        evidence_stats["evidence"][name] = _tensor_stats(pressure[:, idx])
    return evidence_stats


def save_metric_evidence_stats(path: str | Path, stats: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


def load_metric_evidence_stats(config: dict | None) -> dict | None:
    if not config:
        return None
    model_cfg = config.get("model", {})
    if isinstance(model_cfg.get("metric_evidence_stats"), dict):
        return model_cfg["metric_evidence_stats"]
    path = model_cfg.get("metric_evidence_stats_path")
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class TopologyEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_h: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        mask = node_mask.to(dtype=node_h.dtype, device=node_h.device)
        A = adj.to(device=node_h.device, dtype=node_h.dtype) * mask[:, :, None] * mask[:, None, :]
        eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype).unsqueeze(0)
        A = torch.clamp(A + eye * mask[:, :, None], 0.0, 1.0)
        deg = A.sum(dim=-1, keepdim=True).clamp_min(1.0)
        A = A / deg
        h = node_h * mask[:, :, None]
        for linear, norm in zip(self.layers, self.norms):
            msg = torch.bmm(A, h)
            update = self.dropout(torch.relu(linear(msg)))
            h = norm(h + update)
            h = h * mask[:, :, None]
        return h


class MetricEvidenceHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        alpha: float = 0.5,
        use_bias: bool = True,
        norm: str = "zscore",
        bias_type: str = "linear",
        detach: bool = False,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.use_bias = bool(use_bias)
        self.norm = str(norm or "zscore")
        self.bias_type = str(bias_type or "linear").lower()
        self.detach = bool(detach)
        self.evidence_mlp = MLP(5, hidden_dim, hidden_dim=hidden_dim, dropout=dropout, layers=2)
        self.metric_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, z_sla: torch.Tensor, metric_evidence: torch.Tensor) -> torch.Tensor:
        evidence = _normalize_evidence(metric_evidence, norm=self.norm)
        if self.detach:
            evidence = evidence.detach()
        evidence_emb = self.evidence_mlp(evidence)
        learned = self.metric_mlp(torch.cat([z_sla, evidence_emb], dim=-1))
        if not self.use_bias:
            return learned
        if self.bias_type == "logit":
            eps = 1e-4
            bias = torch.log((torch.clamp(evidence, eps, 1.0 - eps)) / (1.0 - torch.clamp(evidence, eps, 1.0 - eps)))
        elif self.bias_type == "linear":
            bias = evidence
        else:
            raise ValueError(f"Unsupported metric_evidence_bias_type: {self.bias_type}")
        return learned + self.alpha * bias


class SPARTA(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        data_cfg = config["data"]
        model_cfg = config["model"]
        self.input_window = int(data_cfg["input_window"])
        self.max_nodes = int(data_cfg["max_nodes"])
        self.max_links = int(data_cfg["max_links"])
        self.node_feat_dim = int(data_cfg["node_feat_dim"])
        self.link_feat_dim = int(data_cfg["link_feat_dim"])
        self.service_feat_dim = int(data_cfg["service_feat_dim"])
        self.sla_feat_dim = int(data_cfg["sla_feat_dim"])
        self.hidden_dim = int(model_cfg.get("hidden_dim", 32))
        self.num_classes = 3
        self.num_metrics = 5
        self.use_topology = bool(model_cfg.get("use_topology", True))
        self.use_sla_gate = bool(model_cfg.get("use_sla_gate", True))
        self.use_metric_evidence_head = bool(model_cfg.get("use_metric_evidence_head", False))
        self.metric_evidence_norm = str(model_cfg.get("metric_evidence_norm", "zscore"))
        self.metric_evidence_stats = load_metric_evidence_stats(config)
        self.feature_schema = _schema_from_config(config)
        dropout = float(model_cfg.get("dropout", 0.1))
        n_heads = int(model_cfg.get("n_heads", 2))
        if self.hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by n_heads={n_heads}")

        self.node_mlp = MLP(self.node_feat_dim, self.hidden_dim, dropout=dropout, layers=2)
        self.link_mlp = MLP(self.link_feat_dim, self.hidden_dim, dropout=dropout, layers=2)
        self.service_mlp = MLP(self.service_feat_dim, self.hidden_dim, dropout=dropout, layers=2)
        self.sla_mlp = MLP(self.sla_feat_dim, self.hidden_dim, dropout=dropout, layers=2)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=n_heads,
            dim_feedforward=int(model_cfg.get("ffn_dim", self.hidden_dim * 4)),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoder(temporal_layer, num_layers=int(model_cfg.get("temporal_layers", 1)))
        self.topology = TopologyEncoder(self.hidden_dim, int(model_cfg.get("topology_layers", 1)), dropout=dropout)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.sla_gate = MLP(self.hidden_dim, self.hidden_dim, hidden_dim=self.hidden_dim, dropout=dropout, layers=2)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        self.node_head = nn.Linear(self.hidden_dim, 1)
        self.link_head = nn.Linear(self.hidden_dim, 1)
        if self.use_metric_evidence_head:
            self.metric_head = MetricEvidenceHead(
                self.hidden_dim,
                dropout=dropout,
                alpha=float(model_cfg.get("metric_evidence_bias_alpha", 0.5)),
                use_bias=bool(model_cfg.get("metric_evidence_use_bias", True)),
                norm=self.metric_evidence_norm,
                bias_type=str(model_cfg.get("metric_evidence_bias_type", "linear")),
                detach=bool(model_cfg.get("metric_evidence_detach", False)),
            )
        else:
            self.metric_head = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.num_metrics))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        node_x = batch["node_x"]
        link_x = batch["link_x"]
        service_x = batch["service_x"]
        sla_x = batch["sla_x"]
        adj = batch["adj"]
        node_mask = batch["node_mask"]
        link_mask = batch["link_mask"]
        B, L, N, FN = node_x.shape
        B2, L2, E, FE = link_x.shape
        assert B2 == B and L2 == L
        assert L == self.input_window
        assert N == self.max_nodes
        assert E == self.max_links
        assert FN == self.node_feat_dim
        assert FE == self.link_feat_dim
        assert service_x.shape == (B, L, self.service_feat_dim)
        assert sla_x.shape == (B, self.sla_feat_dim)
        assert adj.shape == (B, N, N)
        assert node_mask.shape == (B, N)
        assert link_mask.shape == (B, E)

        node_h = self.node_mlp(node_x)
        link_h = self.link_mlp(link_x)
        service_h = self.service_mlp(service_x)
        sla_h = self.sla_mlp(sla_x)
        node_ctx = masked_mean(node_h, node_mask, dim=2)
        link_ctx = masked_mean(link_h, link_mask, dim=2)
        path_token = node_ctx + link_ctx + service_h
        temp_h = self.temporal(path_token)
        z_temp = temp_h[:, -1, :]

        node_topo_in = node_h.mean(dim=1)
        node_topo_h = self.topology(node_topo_in, adj, node_mask)
        z_topo = masked_mean(node_topo_h, node_mask, dim=1)
        z = self.fusion_norm(z_temp + z_topo) if self.use_topology else z_temp
        if self.use_sla_gate:
            gate = torch.sigmoid(self.sla_gate(sla_h))
            z_sla = z * (1.0 + gate)
        else:
            z_sla = z

        risk_logits = self.risk_head(z_sla)
        node_logits = self.node_head(node_topo_h).squeeze(-1)
        link_attr_h = link_h.mean(dim=1)
        link_logits = self.link_head(link_attr_h).squeeze(-1)
        metric_evidence = None
        if self.use_metric_evidence_head:
            metric_evidence = build_metric_evidence(
                batch,
                self.feature_schema,
                norm=self.metric_evidence_norm,
                evidence_stats=self.metric_evidence_stats,
            )
            metric_logits = self.metric_head(z_sla, metric_evidence)
        else:
            metric_logits = self.metric_head(z_sla)
        node_logits = node_logits.masked_fill(~node_mask, -1e9)
        link_logits = link_logits.masked_fill(~link_mask, -1e9)

        assert risk_logits.shape == (B, self.num_classes)
        assert node_logits.shape == (B, N)
        assert link_logits.shape == (B, E)
        assert metric_logits.shape == (B, self.num_metrics)
        outputs = {
            "risk_logits": risk_logits,
            "node_logits": node_logits,
            "link_logits": link_logits,
            "metric_logits": metric_logits,
            "z": z_sla,
        }
        if metric_evidence is not None:
            outputs["metric_evidence"] = metric_evidence
        return outputs
