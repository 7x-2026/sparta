# SPARTA 代码实现规格说明书：数据接口、模型流程、Forward/Backward、超参数与结果格式

> 项目名称：SPARTA  
> 论文题目建议：**SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services**  
> 本文件用途：给 AI/Codex/人工开发者作为**代码实现契约**使用，避免数据格式、张量维度、通道数、mask、loss 和评价指标不一致。  
> 核心原则：**先固定数据接口和模型接口，再写代码；所有模块输入输出必须在本文件约束下实现。**

---

## 0. 本文件解决什么问题

前面的实验报告已经确定了研究任务、项目目录和论文实验路线。本文件进一步补充更底层的实现细节，包括：

1. 数据从原始 trace 到训练样本的字段定义；
2. 每个 CSV / PKL 文件应包含哪些字段；
3. SPARTA 模型每个模块的输入输出 shape；
4. forward pass 的完整张量流；
5. backward pass 与 loss 计算；
6. normal 样本是否参与 attribution loss；
7. padding 和 mask 如何处理；
8. 每个模块默认用多少层、如何调参确定最优层数；
9. YAML 配置文件完整字段；
10. baseline 输入公平性；
11. 消融实验如何精确定义；
12. 结果 CSV 文件格式；
13. 评估指标公式；
14. 单元测试和 debug 模式；
15. 复现实验设置。

目标是让代码实现满足：

```text
数据 shape 对得上；
模型 forward 能跑通；
loss 能正常 backward；
baseline 和 SPARTA 输入公平；
结果能直接生成论文表格和图。
```

---

# 1. 统一符号、维度和默认配置

## 1.1 符号表

| 符号 | 含义 | 默认值 |
|---|---|---:|
| `B` | batch size | 64 |
| `L` | 输入历史窗口长度 input_window | 12 |
| `H` | 未来预测窗口 pred_horizon | 5 |
| `Np` | 每个 service-path subgraph 最大节点数 max_nodes | 8 |
| `Ep` | 每个 service-path subgraph 最大链路数 max_links | 12 |
| `Fn` | 节点特征维度 node_feat_dim | 8 |
| `Fe` | 链路特征维度 link_feat_dim | 8 |
| `Fs` | 服务特征维度 service_feat_dim | 5 |
| `Fc` | SLA 特征维度 sla_feat_dim | 5 |
| `D` | hidden_dim | 128 |
| `C` | 风险类别数 normal/risky/violated | 3 |
| `M` | 风险指标类别数 delay/loss/CPU/queue/bandwidth | 5 |

## 1.2 默认模型规模

第一篇 ICASSP 不建议模型太大。默认配置为：

```yaml
model:
  hidden_dim: 128
  temporal_layers: 2
  topology_layers: 2
  n_heads: 4
  dropout: 0.1
  embedding_layers: 2
  sla_gate_layers: 2
  risk_head_layers: 2
  attribution_head_layers: 1
```

理由：`L=12`、`Np=8`、`Ep=12`，输入窗口和路径子图都不大。过深模型会增加过拟合、调参和解释成本。ICASSP 论文重点应放在 service-path network signal、SLA-conditioned modeling、early detection 与 attribution 上，而不是堆模型深度。

## 1.3 层数如何确定

默认层数需要通过验证集小网格确定，不建议大规模搜索。

| 模块 | 搜索范围 | 默认值 | 选择指标 |
|---|---:|---:|---|
| Temporal Encoder layers | 1, 2, 3 | 2 | val Macro-F1 + Risk Recall |
| Topology Encoder layers | 1, 2 | 2 | val Macro-F1 + Attr-Acc |
| hidden_dim | 64, 128, 256 | 128 | val Macro-F1 |
| n_heads | 2, 4, 8 | 4 | val Macro-F1 |
| dropout | 0.1, 0.2, 0.3 | 0.1 | val Macro-F1 |
| input_window L | 6, 12, 24 | 12 | horizon performance |
| horizon H | 1, 3, 5, 10 | 5 | early detection |
| lambda_node/link/metric | 0.1, 0.3, 0.5 | 0.3/0.3/0.2 | joint performance |

选择规则：主指标看 `val_macro_f1`，辅助看 `val_risk_recall`，归因看 `val_attr_acc`。如果两个模型 Macro-F1 接近，优先选 Risk Recall 更高且模型更小的配置。

---

# 2. 原始数据到模型输入的完整流程

总体数据流：

```text
Alibaba / Google raw trace
        ↓
prepare_alibaba_trace.py / prepare_google_trace.py
        ↓
data/processed/alibaba_service_load.csv
data/processed/google_node_load.csv
        ↓
run_edgesimpy.py
        ↓
node_log.csv / link_log.csv / service_log.csv / path_log.csv / sla_log.csv
        ↓
build_path_graph.py
        ↓
path_graphs_w{L}.pkl
        ↓
generate_labels.py + generate_attribution.py
        ↓
sparta_w{L}_h{H}.pkl
        ↓
split_dataset.py
        ↓
train.pkl / val.pkl / test.pkl
        ↓
SPARTADataset
        ↓
batch dict
        ↓
SPARTA model training
```

---

# 3. 原始 trace 处理后格式

## 3.1 Alibaba trace 处理目标

Alibaba trace 不直接作为训练标签，而是用于生成更真实的服务请求率和服务响应趋势。

处理后文件：

```text
data/processed/alibaba_service_load.csv
```

字段：

```csv
time,service_type,request_rate,response_time_scale,dependency_count
```

| 字段 | 类型 | 范围/单位 | 用途 |
|---|---|---|---|
| `time` | int | time slot | 时间片 |
| `service_type` | int | 0/1/2 | 服务类型 |
| `request_rate` | float | normalized or raw | 控制请求到达 |
| `response_time_scale` | float | [0,+inf) | 控制服务处理时延基准 |
| `dependency_count` | int | >=0 | 控制服务复杂度 |

预处理伪代码：

```python
df = read_raw_alibaba()
df = group_by_time_and_service(df)
df["request_rate"] = normalize(qps)
df["response_time_scale"] = normalize(response_time)
df["dependency_count"] = count_dependencies(service_id)
save_csv(df)
```

## 3.2 Google trace 处理目标

Google trace 用于生成节点负载波动和突发模式。

处理后文件：

```text
data/processed/google_node_load.csv
```

字段：

```csv
time,node_group,cpu_scale,mem_scale,arrival_scale
```

| 字段 | 类型 | 范围 | 用途 |
|---|---|---|---|
| `time` | int | time slot | 时间片 |
| `node_group` | int | group id | 节点组 |
| `cpu_scale` | float | [0,1] | 控制 CPU 利用率 |
| `mem_scale` | float | [0,1] | 控制内存利用率 |
| `arrival_scale` | float | [0,1] | 控制任务到达强度 |

---

# 4. EdgeSimPy 输出日志字段契约

后续 preprocessing 只依赖这些 CSV，不直接依赖 EdgeSimPy 内部对象。

## 4.1 node_log.csv

路径：

```text
data/raw/{scenario}/node_log.csv
```

字段：

```csv
time,node_id,node_type,cpu_util,mem_util,queue_len,available_cpu,available_mem,node_degree,is_current_edge
```

## 4.2 link_log.csv

路径：

```text
data/raw/{scenario}/link_log.csv
```

字段：

```csv
time,src,dst,delay,bandwidth,loss,jitter,queue_delay,bandwidth_util,is_current_path,hop_position
```

## 4.3 service_log.csv

路径：

```text
data/raw/{scenario}/service_log.csv
```

字段：

```csv
time,service_id,user_id,service_type,request_rate,response_time,current_edge,cloud_node,service_priority,dependency_count
```

## 4.4 path_log.csv

路径：

```text
data/raw/{scenario}/path_log.csv
```

字段：

```csv
time,service_id,path_nodes,path_links,path_delay,path_loss,bottleneck_node,bottleneck_link,candidate_edges
```

其中 `path_nodes` 建议保存为字符串：

```text
1|3|7|cloud
```

`path_links` 保存为：

```text
1-3|3-7|7-cloud
```

## 4.5 sla_log.csv

路径：

```text
data/raw/{scenario}/sla_log.csv
```

字段：

```csv
service_id,service_type,max_delay,max_loss,min_bandwidth,reliability_req,cost_weight
```

---

# 5. 特征字典和维度固定

所有特征必须按固定顺序拼接，不能让不同脚本自己随意排列字段。

## 5.1 node_features

维度：

```text
Fn = 8
```

顺序固定：

```text
[
  cpu_util_norm,
  mem_util_norm,
  queue_len_norm,
  available_cpu_norm,
  available_mem_norm,
  node_type_norm,
  node_degree_norm,
  is_current_edge
]
```

shape：

```text
node_features: [L, Np, 8]
```

## 5.2 link_features

维度：

```text
Fe = 8
```

顺序固定：

```text
[
  delay_norm,
  bandwidth_norm,
  loss_norm,
  jitter_norm,
  queue_delay_norm,
  bandwidth_util,
  is_current_path,
  hop_position_norm
]
```

shape：

```text
link_features: [L, Ep, 8]
```

## 5.3 service_features

维度：

```text
Fs = 5
```

顺序固定：

```text
[
  request_rate_norm,
  response_time_norm,
  service_type_norm,
  service_priority_norm,
  dependency_count_norm
]
```

shape：

```text
service_features: [L, 5]
```

## 5.4 sla_features

维度：

```text
Fc = 5
```

顺序固定：

```text
[
  max_delay_norm,
  max_loss_norm,
  min_bandwidth_norm,
  reliability_req_norm,
  cost_weight_norm
]
```

shape：

```text
sla_features: [5]
```

---

# 6. 归一化方案

## 6.1 只用训练集统计归一化参数

严格要求：

```text
scaler 只能在 train set 上 fit；
val/test 使用 train scaler transform；
不能用全数据统计 min/max 或 mean/std。
```

## 6.2 连续特征归一化

默认使用 Min-Max normalization：

```text
x_norm = (x - x_min_train) / (x_max_train - x_min_train + eps)
```

适用于 CPU、memory、queue、delay、bandwidth、loss、jitter、request_rate、response_time、SLA 阈值等连续特征。

## 6.3 类别特征编码

第一版使用简单数值编码即可：

```text
node_type: cloud=0, edge=1, access=2, user=3 → node_type_norm = id / 3
service_type: latency=0, reliability=1, cost=2 → service_type_norm = id / 2
```

后续可以改成 embedding，但第一篇不必复杂化。

---

# 7. 样本构造契约

## 7.1 单个样本格式

每个样本保存为 Python dict：

```python
sample = {
    "node_x": np.ndarray,        # [L, Np, Fn]
    "link_x": np.ndarray,        # [L, Ep, Fe]
    "service_x": np.ndarray,     # [L, Fs]
    "sla_x": np.ndarray,         # [Fc]
    "adj": np.ndarray,           # [Np, Np]
    "node_mask": np.ndarray,     # [Np]
    "link_mask": np.ndarray,     # [Ep]
    "risk_label": int,           # 0 normal, 1 risky, 2 violated
    "risk_node": int,            # 0..Np-1, or -100 if ignored
    "risk_link": int,            # 0..Ep-1, or -100 if ignored
    "risk_metric": int,          # 0..4, or -100 if ignored
    "service_id": str/int,
    "time": int,
    "scenario_id": str
}
```

## 7.2 padding 规则

如果真实节点数 `n_real < Np`：

```text
node_x[:, n_real:Np, :] = 0
node_mask[n_real:Np] = 0
node_mask[:n_real] = 1
```

如果真实链路数 `e_real < Ep`：

```text
link_x[:, e_real:Ep, :] = 0
link_mask[e_real:Ep] = 0
link_mask[:e_real] = 1
```

如果真实节点数超过 `Np`，优先保留：

```text
当前服务路径节点 → 当前 edge → cloud → 最关键候选 edge → 邻近竞争节点
```

如果真实链路数超过 `Ep`，优先保留：

```text
当前 path links → 候选 path links → 邻近链路
```

## 7.3 normal 样本归因标签

normal 样本没有明确风险来源，规定：

```python
if risk_label == 0:
    risk_node = -100
    risk_link = -100
    risk_metric = -100
```

训练时使用：

```python
ignore_index = -100
```

normal 样本不参与 attribution loss。

---

# 8. Dataset 和 DataLoader 输出契约

`SPARTADataset.__getitem__()` 必须返回：

```python
{
    "node_x": torch.FloatTensor,       # [L, Np, Fn]
    "link_x": torch.FloatTensor,       # [L, Ep, Fe]
    "service_x": torch.FloatTensor,    # [L, Fs]
    "sla_x": torch.FloatTensor,        # [Fc]
    "adj": torch.FloatTensor,          # [Np, Np]
    "node_mask": torch.BoolTensor,     # [Np]
    "link_mask": torch.BoolTensor,     # [Ep]
    "risk_label": torch.LongTensor,    # scalar
    "risk_node": torch.LongTensor,     # scalar
    "risk_link": torch.LongTensor,     # scalar
    "risk_metric": torch.LongTensor,   # scalar
}
```

DataLoader collate 后 batch 为：

```python
batch = {
    "node_x": [B, L, Np, Fn],
    "link_x": [B, L, Ep, Fe],
    "service_x": [B, L, Fs],
    "sla_x": [B, Fc],
    "adj": [B, Np, Np],
    "node_mask": [B, Np],
    "link_mask": [B, Ep],
    "risk_label": [B],
    "risk_node": [B],
    "risk_link": [B],
    "risk_metric": [B],
}
```

---

# 9. SPARTA 模型结构契约

## 9.1 初始化参数

```python
SPARTA(
    node_feat_dim=8,
    link_feat_dim=8,
    service_feat_dim=5,
    sla_feat_dim=5,
    hidden_dim=128,
    num_classes=3,
    num_metrics=5,
    temporal_layers=2,
    topology_layers=2,
    n_heads=4,
    dropout=0.1,
    max_nodes=8,
    max_links=12,
    use_sla_gate=True,
    use_topology=True,
    use_attribution=True,
)
```

## 9.2 forward 输入

```python
outputs = model(
    node_x=batch["node_x"],          # [B,L,Np,Fn]
    link_x=batch["link_x"],          # [B,L,Ep,Fe]
    service_x=batch["service_x"],    # [B,L,Fs]
    sla_x=batch["sla_x"],            # [B,Fc]
    adj=batch["adj"],                # [B,Np,Np]
    node_mask=batch["node_mask"],    # [B,Np]
    link_mask=batch["link_mask"],    # [B,Ep]
)
```

## 9.3 forward 输出

必须返回 dict：

```python
outputs = {
    "risk_logits": torch.FloatTensor,     # [B, 3]
    "node_logits": torch.FloatTensor,     # [B, Np]
    "link_logits": torch.FloatTensor,     # [B, Ep]
    "metric_logits": torch.FloatTensor,   # [B, 5]
    "z": torch.FloatTensor,               # [B, D]
}
```

可选返回：

```python
outputs = {
    ...
    "node_repr": [B, Np, D],
    "link_repr": [B, Ep, D],
    "temporal_repr": [B, D],
    "topology_repr": [B, D],
    "sla_gate": [B, D],
}
```

---

# 10. Forward pass 完整 shape 流程

## 10.1 输入 batch

```text
node_x:      [B, L, Np, Fn] = [64, 12, 8, 8]
link_x:      [B, L, Ep, Fe] = [64, 12, 12, 8]
service_x:   [B, L, Fs]     = [64, 12, 5]
sla_x:       [B, Fc]        = [64, 5]
adj:         [B, Np, Np]    = [64, 8, 8]
node_mask:   [B, Np]        = [64, 8]
link_mask:   [B, Ep]        = [64, 12]
```

## 10.2 Heterogeneous Feature Embedding

```python
node_emb = node_mlp(node_x)          # [B, L, Np, D]
link_emb = link_mlp(link_x)          # [B, L, Ep, D]
service_emb = service_mlp(service_x) # [B, L, D]
sla_emb = sla_mlp(sla_x)             # [B, D]
```

输出：

```text
node_emb:    [B, L, Np, D] = [64, 12, 8, 128]
link_emb:    [B, L, Ep, D] = [64, 12, 12, 128]
service_emb: [B, L, D]     = [64, 12, 128]
sla_emb:     [B, D]        = [64, 128]
```

## 10.3 Masked Pooling

节点池化：

```python
node_mask_expand = node_mask[:, None, :, None]     # [B,1,Np,1]
node_sum = (node_emb * node_mask_expand).sum(dim=2) # [B,L,D]
node_count = node_mask.sum(dim=1).clamp(min=1)[:, None, None]
node_pool = node_sum / node_count                  # [B,L,D]
```

链路池化：

```python
link_mask_expand = link_mask[:, None, :, None]      # [B,1,Ep,1]
link_sum = (link_emb * link_mask_expand).sum(dim=2) # [B,L,D]
link_count = link_mask.sum(dim=1).clamp(min=1)[:, None, None]
link_pool = link_sum / link_count                   # [B,L,D]
```

## 10.4 Service-path Token Construction

```python
path_token = node_pool + link_pool + service_emb
```

输出：

```text
path_token: [B, L, D] = [64, 12, 128]
```

## 10.5 Temporal Encoder

```python
temporal_out = temporal_encoder(path_token)  # [B,L,D]
z_time = temporal_out[:, -1, :]              # [B,D]
```

## 10.6 Topology Encoder

使用最后一个时间片的节点嵌入：

```python
node_last = node_emb[:, -1, :, :]  # [B,Np,D]
```

简化 GCN：

```python
A_hat = normalize(adj + I)
node_topo = GCN(node_last, A_hat, node_mask)  # [B,Np,D]
z_topo = masked_mean(node_topo, node_mask)    # [B,D]
```

## 10.7 Temporal-Topology Fusion

```python
if use_topology:
    z = z_time + z_topo
else:
    z = z_time
```

输出：

```text
z: [B, D]
```

## 10.8 SLA Gate

```python
gate = sigmoid(sla_gate_mlp(sla_emb))  # [B,D]
z = z * gate + z
```

如果 `use_sla_gate=False`，则直接 `z=z`。

## 10.9 Risk Head

```python
risk_logits = risk_head(z)  # [B,3]
```

## 10.10 Node Attribution Head

```python
node_logits = node_attr_head(node_topo).squeeze(-1)  # [B,Np]
node_logits = node_logits.masked_fill(~node_mask, -1e9)
```

## 10.11 Link Attribution Head

```python
link_last = link_emb[:, -1, :, :]  # [B,Ep,D]
link_logits = link_attr_head(link_last).squeeze(-1)  # [B,Ep]
link_logits = link_logits.masked_fill(~link_mask, -1e9)
```

## 10.12 Metric Attribution Head

```python
metric_logits = metric_head(z)  # [B,5]
```

---

# 11. Backward pass 与 Loss 计算契约

## 11.1 标签输入

```python
risk_label:  [B]  # 0 normal, 1 risky, 2 violated
risk_node:   [B]  # 0..Np-1 or -100
risk_link:   [B]  # 0..Ep-1 or -100
risk_metric: [B]  # 0..4 or -100
```

## 11.2 normal 样本不参与 attribution loss

```python
attr_mask = risk_label != 0
```

如果 `risk_label == 0`：

```python
risk_node = -100
risk_link = -100
risk_metric = -100
```

CrossEntropyLoss 使用：

```python
ignore_index = -100
```

## 11.3 risk classification loss

默认先用 Weighted CE：

```python
risk_loss = CrossEntropyLoss(weight=class_weights)(risk_logits, risk_label)
```

如果类别极度不平衡，再切换 Focal Loss：

```python
risk_loss = FocalLoss(alpha=class_weights, gamma=2.0)(risk_logits, risk_label)
```

## 11.4 attribution loss

```python
node_loss = CrossEntropyLoss(ignore_index=-100)(node_logits, risk_node)
link_loss = CrossEntropyLoss(ignore_index=-100)(link_logits, risk_link)
metric_loss = CrossEntropyLoss(ignore_index=-100)(metric_logits, risk_metric)
```

## 11.5 total loss

```python
total_loss = (
    risk_loss
    + lambda_node * node_loss
    + lambda_link * link_loss
    + lambda_metric * metric_loss
)
```

默认：

```yaml
lambda_node: 0.3
lambda_link: 0.3
lambda_metric: 0.2
```

如果 attribution loss 影响分类效果，可调低为：

```yaml
lambda_node: 0.1
lambda_link: 0.1
lambda_metric: 0.1
```

## 11.6 backward 伪代码

```python
model.train()

for batch in train_loader:
    batch = move_to_device(batch)

    optimizer.zero_grad()

    outputs = model(
        node_x=batch["node_x"],
        link_x=batch["link_x"],
        service_x=batch["service_x"],
        sla_x=batch["sla_x"],
        adj=batch["adj"],
        node_mask=batch["node_mask"],
        link_mask=batch["link_mask"],
    )

    losses = compute_sparta_loss(outputs, batch, config)
    loss = losses["total_loss"]

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

---

# 12. 训练超参数协议

## 12.1 默认训练配置

```yaml
train:
  batch_size: 64
  epochs: 100
  optimizer: adamw
  lr: 0.0005
  weight_decay: 0.0001
  scheduler: reduce_on_plateau
  patience: 15
  grad_clip: 1.0
  seed: 42
  num_workers: 4
  device: cuda
  amp: false
```

## 12.2 Optimizer

默认：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-4
)
```

如果 loss 不稳定：

```text
lr 降到 1e-4；
batch size 降到 32；
grad_clip 保持 1.0。
```

## 12.3 Scheduler

推荐：

```python
ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)
```

监控：

```text
val_macro_f1
```

## 12.4 Early stopping

```yaml
early_stopping:
  monitor: val_macro_f1
  mode: max
  patience: 15
```

保存模型：

```text
checkpoints/sparta_best.pth
```

保存依据：`val_macro_f1` 最大。如果两个模型 Macro-F1 接近，选择 Risk Recall 更高的。

---

# 13. 类别不平衡处理

## 13.1 类别权重计算

```python
class_counts = [num_normal, num_risky, num_violated]
class_weights = total_samples / (num_classes * class_counts)
class_weights = class_weights / class_weights.mean()
```

## 13.2 WeightedSampler 可选

如果 risky / violated 太少，可以使用 WeightedRandomSampler。只在训练集使用，val/test 保持原始分布。

## 13.3 默认推荐

第一版：

```text
weighted CE + time-based split
```

第二版：

```text
focal loss + class weights
```

不建议一开始同时使用 focal loss 和 oversampling，可能过度补偿。

---

# 14. YAML 配置完整模板

文件：

```text
configs/sparta.yaml
```

内容：

```yaml
project:
  name: SPARTA
  task: sla_risk_detection
  seed: 42
  output_dir: results/sparta

data:
  dataset_path: data/sparta_dataset/sparta_w12_h5.pkl
  train_path: data/sparta_dataset/train.pkl
  val_path: data/sparta_dataset/val.pkl
  test_path: data/sparta_dataset/test.pkl
  input_window: 12
  pred_horizon: 5
  max_nodes: 8
  max_links: 12
  node_feat_dim: 8
  link_feat_dim: 8
  service_feat_dim: 5
  sla_feat_dim: 5
  num_classes: 3
  num_metrics: 5
  normalize: true
  scaler_path: data/processed/scaler.pkl
  split_type: time
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2

model:
  name: sparta
  hidden_dim: 128
  temporal_layers: 2
  topology_layers: 2
  embedding_layers: 2
  sla_gate_layers: 2
  risk_head_layers: 2
  attribution_head_layers: 1
  n_heads: 4
  dropout: 0.1
  activation: gelu
  use_sla_gate: true
  use_topology: true
  use_attribution: true

loss:
  risk_loss: weighted_ce
  use_focal_loss: false
  focal_gamma: 2.0
  lambda_node: 0.3
  lambda_link: 0.3
  lambda_metric: 0.2
  ignore_index: -100
  ignore_attribution_for_normal: true

train:
  batch_size: 64
  epochs: 100
  optimizer: adamw
  lr: 0.0005
  weight_decay: 0.0001
  scheduler: reduce_on_plateau
  scheduler_factor: 0.5
  scheduler_patience: 5
  early_stop_patience: 15
  grad_clip: 1.0
  num_workers: 4
  amp: false
  device: cuda
  save_best_by: val_macro_f1

eval:
  metrics:
    - accuracy
    - macro_f1
    - risk_recall
    - violation_recall
    - auc
    - node_attr_acc
    - link_attr_acc
    - metric_attr_acc
    - inference_time_ms
  topk_attribution:
    - 1
    - 3

logging:
  log_interval: 20
  save_predictions: true
  save_logits: false
  save_attention: false
```

---

# 15. Baseline 输入公平性协议

所有方法共用：

```text
相同 train/val/test split；
相同 input_window L；
相同 horizon H；
相同 risk labels；
相同特征归一化；
相同评价指标。
```

| 方法 | 输入格式 | 是否使用 SLA | 是否使用 topology | 是否归因 |
|---|---|---|---|---|
| RF | flatten all features | 是 | 否 | 否 |
| XGBoost | flatten all features | 是 | 否 | 否 |
| LSTM | path_token `[B,L,D]` | 可拼接 | 否 | 否 |
| GRU | path_token `[B,L,D]` | 可拼接 | 否 | 否 |
| TCN | `[B,D,L]` | 可拼接 | 否 | 否 |
| Transformer | path_token `[B,L,D]` | 可拼接 | 否 | 否 |
| PatchTST | multivariate path series | 可拼接 | 否 | 否 |
| STGCN | graph sequence | 可选 | 是 | 否 |
| SPARTA | path graph + SLA + masks | 是 | 是 | 是 |

---

# 16. 消融实验精确定义

| 消融版本 | 精确定义 |
|---|---|
| full SPARTA | service-path subgraph + temporal encoder + topology encoder + SLA gate + attribution heads |
| w/o SLA | 删除 SLA gate，`sla_x` 不进入模型，其余模块不变 |
| w/o topology | 删除 topology encoder，`z=z_time`，`adj` 不进入模型 |
| w/o attribution | 删除 node/link/metric attribution heads，loss 只包含 `risk_loss` |
| global graph | 不用 service-path subgraph，输入全局图或较大范围网络状态 |
| binary label | normal vs risky_or_violated，不使用三分类 |

---

# 17. Proactive intervention 规则契约

## 17.1 策略输入

```python
risk_pred = argmax(risk_logits)
risk_node = argmax(node_logits)
risk_link = argmax(link_logits)
risk_metric = argmax(metric_logits)
```

只有当：

```python
risk_pred in [1, 2]  # risky or violated
```

才触发干预。

## 17.2 规则

CPU / queue 风险：

```python
if risk_metric in ["CPU", "queue"]:
    target_edge = candidate_edge_with_lowest_cpu_queue()
    migrate_service(service_id, target_edge)
```

delay / bandwidth 风险：

```python
if risk_metric in ["delay", "bandwidth"]:
    target_path = candidate_path_with_lowest_delay()
    reroute_service(service_id, target_path)
```

loss 风险：

```python
if risk_metric == "loss":
    target_path = candidate_path_with_lowest_loss()
    reroute_service(service_id, target_path)
```

## 17.3 干预成本

迁移成本：

```text
migration_cost = service_state_size / available_bandwidth
```

换路成本：

```text
reroute_cost = fixed_path_switch_penalty
```

总成本：

```text
intervention_cost = migration_cost + reroute_cost
```

## 17.4 Proactive 实验对比

| 策略 | 说明 |
|---|---|
| No Intervention | 不处理 |
| Reactive | violated 后才迁移/换路 |
| Proactive-SPARTA | risky 阶段提前处理 |
| Oracle | 使用真实未来标签提前处理，可选上界 |

---

# 18. 评价指标公式

## 18.1 Macro-F1

```text
Macro-F1 = average(F1_normal, F1_risky, F1_violated)
```

## 18.2 Risk Recall

```text
Risk Recall = TP_risky / (TP_risky + FN_risky)
```

## 18.3 Violation Recall

```text
Violation Recall = TP_violated / (TP_violated + FN_violated)
```

## 18.4 Attribution Accuracy

只对 risky / violated 样本计算：

```text
Node-Attr-Acc = correct_node_prediction / num_risk_samples
Link-Attr-Acc = correct_link_prediction / num_risk_samples
Metric-Attr-Acc = correct_metric_prediction / num_risk_samples
```

## 18.5 Top-k Attribution Recall

```text
Top-k Attr Recall = whether ground-truth risk source appears in top-k predicted sources
```

建议报告 Top-1 和 Top-3。

## 18.6 SLA Violation Rate

```text
SLA Violation Rate = num_violated_services / num_total_services
```

## 18.7 Average Response Time

```text
Average Response Time = mean(response_time over services and time)
```

## 18.8 Intervention Cost

```text
Average Intervention Cost = total_intervention_cost / num_interventions
```

---

# 19. 结果文件格式

## 19.1 main_results.csv

路径：

```text
results/main_results.csv
```

字段：

```csv
method,accuracy,macro_f1,risk_recall,violation_recall,auc,node_attr_acc,link_attr_acc,metric_attr_acc,inference_time_ms,params
```

## 19.2 ablation_results.csv

```csv
variant,accuracy,macro_f1,risk_recall,violation_recall,node_attr_acc,link_attr_acc,metric_attr_acc
```

## 19.3 horizon_results.csv

```csv
method,horizon,macro_f1,risk_recall,violation_recall,auc
```

## 19.4 attribution_results.csv

```csv
method,node_top1,node_top3,link_top1,link_top3,metric_acc
```

## 19.5 proactive_results.csv

```csv
strategy,violation_rate,avg_response_time,intervention_cost,num_interventions
```

---

# 20. 训练日志格式

每个 epoch 输出：

```csv
epoch,train_loss,val_loss,val_accuracy,val_macro_f1,val_risk_recall,val_violation_recall,val_node_attr_acc,val_link_attr_acc,val_metric_attr_acc,lr
```

保存到：

```text
results/sparta/train_log.csv
```

---

# 21. 单元测试清单

建议创建：

```text
tests/test_dataset_shape.py
tests/test_model_forward.py
tests/test_loss_backward.py
tests/test_train_one_batch.py
tests/test_metrics.py
```

## 21.1 test_dataset_shape.py

检查：

```python
sample = dataset[0]
assert sample["node_x"].shape == (L, Np, Fn)
assert sample["link_x"].shape == (L, Ep, Fe)
assert sample["service_x"].shape == (L, Fs)
assert sample["sla_x"].shape == (Fc,)
assert sample["adj"].shape == (Np, Np)
assert sample["node_mask"].shape == (Np,)
assert sample["link_mask"].shape == (Ep,)
```

## 21.2 test_model_forward.py

检查：

```python
outputs = model(**batch_inputs)
assert outputs["risk_logits"].shape == (B, 3)
assert outputs["node_logits"].shape == (B, Np)
assert outputs["link_logits"].shape == (B, Ep)
assert outputs["metric_logits"].shape == (B, 5)
```

## 21.3 test_loss_backward.py

检查：

```python
loss = compute_loss(outputs, batch)
loss.backward()
```

确保：

```text
loss 非 NaN；
梯度存在；
padding 不参与 node/link attribution。
```

## 21.4 test_train_one_batch.py

检查一个 batch 能完整训练：

```python
for batch in train_loader:
    outputs = model(...)
    loss = compute_loss(...)
    loss.backward()
    optimizer.step()
    break
```

---

# 22. Debug 模式

配置文件：

```text
configs/sparta_debug.yaml
```

建议：

```yaml
data:
  debug_num_samples: 128
  input_window: 6
  pred_horizon: 3
  max_nodes: 6
  max_links: 8

model:
  hidden_dim: 32
  temporal_layers: 1
  topology_layers: 1
  n_heads: 2

train:
  batch_size: 8
  epochs: 2
  lr: 0.001
```

运行：

```bash
python src/train.py --config configs/sparta_debug.yaml
```

验收：

```text
2 个 epoch 能跑完；
loss 能下降或至少不为 NaN；
验证指标能输出；
checkpoint 能保存。
```

---

# 23. 复现实验设置

在 `src/utils/seed.py` 中写：

```python
def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

训练前必须调用：

```python
set_seed(config.project.seed)
```

建议使用 3 个 seed：

```text
42, 2024, 3407
```

论文中报告平均值和标准差。若时间紧，主实验用 3 seeds，消融用 1 seed。

---

# 24. 模型复杂度统计

建议输出：

```text
Params
Inference time
GPU memory
```

参数量：

```python
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

推理时间：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
outputs = model(...)
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

报告 `ms/sample` 或 `ms/batch`。

---

# 25. 可视化规范

## 25.1 horizon curve

输入：

```text
results/horizon_results.csv
```

输出：

```text
figures/horizon_curve.pdf
```

x 轴：

```text
H = 1, 3, 5, 10
```

y 轴：

```text
Risk Recall / Macro-F1
```

## 25.2 attribution case

展示：

```text
service-path subgraph
predicted risk node
true risk node
predicted risk link
true risk link
risk metric
```

颜色建议：

```text
红色：预测风险来源
蓝色：真实风险来源
橙色：预测与真实重合
灰色：普通节点/链路
```

---

# 26. 代码文件职责

## 26.1 src/models/sparta.py

必须包含：

```python
class MLPEmbedding(nn.Module)
class TopologyEncoder(nn.Module)
class SPARTA(nn.Module)
```

`SPARTA.forward()` 必须输出：

```python
{
    "risk_logits": ...,
    "node_logits": ...,
    "link_logits": ...,
    "metric_logits": ...,
    "z": ...
}
```

## 26.2 src/datasets/sparta_dataset.py

必须包含：

```python
class SPARTADataset(Dataset)
def collate_fn(batch)
```

职责：

```text
加载 pkl；
转 torch tensor；
处理 mask；
输出 batch dict。
```

## 26.3 src/losses/multitask_loss.py

必须包含：

```python
def compute_sparta_loss(outputs, batch, config):
    ...
```

返回：

```python
{
    "total_loss": total_loss,
    "risk_loss": risk_loss,
    "node_loss": node_loss,
    "link_loss": link_loss,
    "metric_loss": metric_loss,
}
```

## 26.4 src/train.py

职责：

```text
读取 config；
构建 dataset/dataloader；
构建 model；
构建 optimizer/scheduler；
训练循环；
验证；
保存 best checkpoint；
输出 train_log.csv。
```

## 26.5 src/evaluate.py

职责：

```text
加载 checkpoint；
跑 test set；
计算 metrics；
保存 prediction；
保存 main_results.csv。
```

## 26.6 src/visualize.py

职责：

```text
读取 results；
画 horizon curve；
画 attribution case；
画 confusion matrix；
保存 figures。
```

---

# 27. 最终代码实现顺序

建议 AI/Codex 按这个顺序写：

1. `configs/sparta_debug.yaml`
2. `src/preprocessing/generate_dummy_dataset.py`
3. `src/datasets/sparta_dataset.py`
4. `src/models/sparta.py`
5. `src/losses/multitask_loss.py`
6. `tests/test_dataset_shape.py`
7. `tests/test_model_forward.py`
8. `tests/test_loss_backward.py`
9. `src/train.py`
10. `src/evaluate.py`
11. `src/preprocessing/build_path_graph.py`
12. `src/preprocessing/generate_labels.py`
13. `src/simulation/run_edgesimpy.py`
14. baselines
15. visualization

特别建议先写 `generate_dummy_dataset.py`。它不依赖 EdgeSimPy，能先生成随机但 shape 正确的数据，确保模型、loss、训练流程能跑通。

---

# 28. Dummy dataset 规格

为了快速 debug，先生成：

```text
num_samples = 512
L = 12
Np = 8
Ep = 12
Fn = 8
Fe = 8
Fs = 5
Fc = 5
```

随机生成：

```python
node_x = np.random.rand(L, Np, Fn)
link_x = np.random.rand(L, Ep, Fe)
service_x = np.random.rand(L, Fs)
sla_x = np.random.rand(Fc)
adj = random symmetric matrix [Np,Np]
node_mask = all ones
link_mask = all ones
risk_label = random choice [0,1,2]
if risk_label == 0:
    risk_node = -100
    risk_link = -100
    risk_metric = -100
else:
    risk_node = random int [0,Np)
    risk_link = random int [0,Ep)
    risk_metric = random int [0,5)
```

用途：先验证代码，不用于论文实验。

---

# 29. 最终检查表

真正跑大实验前，逐项检查：

```text
[ ] Dataset sample shape 正确
[ ] DataLoader batch shape 正确
[ ] SPARTA forward 输出 shape 正确
[ ] node/link padding logits 被 mask 到 -1e9
[ ] normal 样本 attribution label = -100
[ ] normal 样本不参与 attribution loss
[ ] total_loss 能 backward
[ ] train one batch 能跑通
[ ] val metrics 能输出
[ ] checkpoint 能保存
[ ] evaluate.py 能加载 checkpoint
[ ] main_results.csv 能生成
[ ] horizon_results.csv 能生成
[ ] visualize.py 能画图
```

---

# 30. 总结

这份实现规格的核心目标是保证：

```text
数据格式固定；
模型输入输出固定；
forward shape 可追踪；
loss 可计算；
normal 样本归因不污染训练；
mask 不会出错；
baseline 输入公平；
结果文件能直接用于论文。
```

后续如果让 AI/Codex 写代码，必须要求它严格遵守本文件中的：

1. 特征顺序；
2. 张量 shape；
3. forward 输出 dict；
4. loss 计算方式；
5. YAML 字段；
6. CSV 结果格式；
7. 单元测试要求。
