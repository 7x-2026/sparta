# SPARTA 模型结构、层数选择与数据接口完整规格说明书

> 本文件用于解决代码实现中最容易出错的问题：**数据初始是什么样、处理后是什么样、模型每个阶段输入输出张量是什么形状、每个模块默认用多少层、如何通过验证集确定最优层数**。  
> 目标是让 AI / Codex / 代码助手可以直接按本规格写项目代码，最大限度避免维度、通道、mask、标签不匹配。

---

## 0. 核心结论

SPARTA 第一版不要追求复杂模型，先固定一个**可运行、可消融、可扩展**的结构。

推荐默认配置如下：

```yaml
# configs/sparta.yaml
model:
  name: sparta
  hidden_dim: 128
  node_feat_dim: 10
  link_feat_dim: 8
  service_feat_dim: 6
  sla_feat_dim: 5
  max_nodes: 8
  max_links: 12
  input_window: 12
  pred_horizon: 5

  embedding:
    node_mlp_layers: 2
    link_mlp_layers: 2
    service_mlp_layers: 2
    sla_mlp_layers: 2
    activation: gelu
    dropout: 0.1

  temporal_encoder:
    type: transformer
    num_layers: 2
    n_heads: 4
    ffn_dim: 512
    dropout: 0.1
    batch_first: true

  topology_encoder:
    type: gcn
    num_layers: 2
    dropout: 0.1
    residual: true
    layer_norm: true

  sla_gate:
    num_layers: 2
    hidden_dim: 64
    output_dim: 128
    activation: gelu
    gate_type: sigmoid_residual

  heads:
    risk_head_layers: 2
    node_attr_head_layers: 1
    link_attr_head_layers: 1
    metric_attr_head_layers: 2
    num_risk_classes: 3
    num_metric_classes: 5

loss:
  risk_loss: weighted_ce
  use_focal_loss: false
  lambda_node: 0.3
  lambda_link: 0.3
  lambda_metric: 0.2
  ignore_index: -100
```

这个配置的设计原则是：

1. **Temporal Encoder 用 2 层**：输入窗口默认只有 12 个时间片，2 层 Transformer 足够建模短期风险演化，3 层作为搜索上限。
2. **Topology Encoder 用 2 层**：service-path subgraph 很小，默认 `max_nodes=8`，2 层可以覆盖一跳和二跳关系，超过 3 层容易过平滑。
3. **SLA Gate 用 2 层 MLP**：它只是条件调制模块，不需要堆深。
4. **Embedding MLP 用 2 层**：用于把异构特征统一到 `hidden_dim=128`。
5. **Heads 尽量浅**：风险分类 head 用 2 层，归因 head 用 1–2 层，避免在 4 页会议论文里变成复杂分类器堆叠。

---

## 1. 每个模块层数怎么确定

### 1.1 不要一开始靠“拍脑袋”定最终层数

层数选择应该分两步：

第一步，给出一个**文献合理 + 代码稳定**的默认值；  
第二步，用验证集做小规模网格搜索，选择**效果好且最简单**的配置。

也就是说，论文里不要写“我们随便设置 2 层”，而要写：

> Following common lightweight Transformer and spatio-temporal graph modeling practice, we use a two-layer temporal encoder and a two-layer topology encoder as the default configuration. We further conduct a depth sensitivity analysis over 1–3 layers and select the model with the best validation macro-F1 and risk recall.

### 1.2 文献和工程依据

#### Temporal Encoder

PyTorch `TransformerEncoderLayer` 的核心参数包括 `d_model`、`nhead`、`dim_feedforward`、`dropout`、`batch_first` 等，其中 `d_model` 是输入最后一维特征数，`nhead` 是多头注意力头数。代码实现时必须保证 `hidden_dim % n_heads == 0`。

本项目输入窗口默认 `L=12`，不是长序列预测任务，因此不需要 6 层或更深的 Transformer。默认 2 层，搜索范围 1–3 层。

#### Topology Encoder

STGCN 经典结构使用两个 spatio-temporal convolution blocks，且每个 block 中包含 temporal gated convolution 与 spatial graph convolution。SPARTA 的 service-path subgraph 比交通全图更小，因此默认 2 层 topology encoder 是合理起点。

#### Time-series baselines

PatchTST 的核心思想是把时间序列切分为 patch token，再输入 Transformer；TimesNet 也被设计为通用时序分析骨干，支持 forecasting、imputation、classification、anomaly detection 等任务。因此在本项目里，PatchTST / TimesNet 适合作为强 baseline，而 SPARTA 的差异在于 service-path subgraph、SLA condition 和 attribution。

### 1.3 默认层数与搜索范围

| 模块 | 默认值 | 搜索范围 | 选择依据 | 过深风险 |
|---|---:|---:|---|---|
| Node/Link/Service Embedding MLP | 2 层 | 1–3 层 | 异构特征升维到统一 hidden_dim | 过拟合、参数冗余 |
| Temporal Encoder | 2 层 | 1–3 层 | L=12，短窗口时序建模 | 训练慢、过拟合 |
| Topology Encoder | 2 层 | 1–3 层 | 小型 service-path graph，覆盖 2-hop 关系 | graph oversmoothing |
| SLA Gate | 2 层 | 1–2 层 | 条件调制，不需要深 | gate 过强导致不稳定 |
| Risk Head | 2 层 | 1–2 层 | 分类任务稍复杂 | 过拟合 |
| Attribution Heads | 1 层 | 1–2 层 | 节点/链路打分任务简单 | 标签噪声被放大 |

### 1.4 用验证集确定最优层数

不要一次性大规模搜索。建议只做轻量网格：

```yaml
search_space:
  hidden_dim: [64, 128, 256]
  temporal_layers: [1, 2, 3]
  topology_layers: [1, 2, 3]
  n_heads: [2, 4, 8]
  dropout: [0.1, 0.2]
```

注意：只有当 `hidden_dim % n_heads == 0` 时才允许该组合。

模型选择指标不要只看 Accuracy，而是：

```text
score = 0.4 * MacroF1
      + 0.3 * RiskRecall
      + 0.2 * ViolationRecall
      + 0.1 * AttrAcc
```

如果两个模型得分差距小于 0.5% 或 1%，选择层数更少的模型。

推荐最终策略：

```text
先固定 hidden_dim=128, n_heads=4, dropout=0.1
只搜索 temporal_layers 和 topology_layers
如果 2+2 已经接近最优，就用 2+2
```

---

## 2. 数据从最初到训练输入的完整形态变化

数据流为：

```text
Alibaba / Google raw trace
        ↓
processed trace csv
        ↓
EdgeSimPy 仿真日志 csv
        ↓
service-path graph samples
        ↓
fixed-size tensor dataset
        ↓
DataLoader batch
        ↓
SPARTA forward
```

---

## 3. 原始 trace 数据格式

### 3.1 Alibaba trace 初始格式

原始 Alibaba trace 文件可能来自 microservices trace 或 cluster trace。不同版本字段不完全一致，所以第一步不要直接喂给模型，而是统一转换成内部格式。

统一输出为：

```csv
time,service_id,service_type,request_rate,base_response_time,dependency_count
0,svc_001,latency,0.42,0.18,3
1,svc_001,latency,0.51,0.20,3
2,svc_002,reliability,0.30,0.25,5
```

字段解释：

| 字段 | 类型 | 含义 |
|---|---|---|
| time | int | 统一时间片编号 |
| service_id | str/int | 服务编号 |
| service_type | str | latency / reliability / cost |
| request_rate | float | 归一化请求率，范围建议 [0,1] |
| base_response_time | float | 归一化服务基础响应时间 |
| dependency_count | int/float | 服务依赖数量或复杂度 |

输出路径：

```text
data/processed/alibaba_service_load.csv
```

### 3.2 Google trace 初始格式

Google trace 用于驱动节点负载或 workload burst。统一输出为：

```csv
time,node_group,cpu_scale,mem_scale,arrival_scale
0,edge_group_0,0.35,0.28,0.40
1,edge_group_0,0.42,0.30,0.47
2,edge_group_1,0.60,0.55,0.70
```

输出路径：

```text
data/processed/google_node_load.csv
```

---

## 4. EdgeSimPy 仿真输出日志格式

EdgeSimPy 不直接输出模型训练样本。它只输出云边环境的运行日志。

每个仿真场景生成一个目录：

```text
data/raw/smallworld_20n_300u/
├── node_log.csv
├── link_log.csv
├── service_log.csv
├── path_log.csv
└── sla_log.csv
```

### 4.1 node_log.csv

```csv
time,node_id,node_type,cpu_util,mem_util,queue_len,available_cpu,available_mem,request_load_on_node
0,e0,edge,0.42,0.35,0.10,0.58,0.65,0.31
0,e1,edge,0.38,0.30,0.08,0.62,0.70,0.25
```

固定转为节点特征维度：

```text
FN = 10
```

节点特征顺序必须固定：

| index | feature | 说明 |
|---:|---|---|
| 0 | cpu_util | CPU 利用率，0–1 |
| 1 | mem_util | 内存利用率，0–1 |
| 2 | queue_len_norm | 归一化队列长度 |
| 3 | available_cpu | 可用 CPU，0–1 |
| 4 | available_mem | 可用内存，0–1 |
| 5 | is_cloud | 云节点 one-hot |
| 6 | is_edge | 边缘节点 one-hot |
| 7 | is_access | 接入节点 one-hot |
| 8 | is_user | 用户节点 one-hot |
| 9 | request_load_on_node | 节点承载请求量 |

### 4.2 link_log.csv

```csv
time,src,dst,delay,bandwidth,loss,jitter,queue_delay,bandwidth_util,is_current_path,is_candidate_path
0,a0,e0,0.12,0.80,0.01,0.03,0.04,0.35,1,0
0,e0,c0,0.25,0.65,0.02,0.05,0.10,0.52,1,0
```

固定转为链路特征维度：

```text
FE = 8
```

链路特征顺序必须固定：

| index | feature | 说明 |
|---:|---|---|
| 0 | delay_norm | 归一化链路时延 |
| 1 | bandwidth_norm | 归一化带宽 |
| 2 | loss_norm | 归一化丢包率 |
| 3 | jitter_norm | 归一化 jitter |
| 4 | queue_delay_norm | 归一化排队时延 |
| 5 | bandwidth_util | 带宽利用率 |
| 6 | is_current_path | 是否当前服务路径 |
| 7 | is_candidate_path | 是否候选路径 |

### 4.3 service_log.csv

```csv
time,service_id,user_id,service_type,request_rate,response_time,current_edge,cloud_node,last_violation
0,s0,u0,latency,0.42,0.31,e0,c0,0
1,s0,u0,latency,0.51,0.36,e0,c0,0
```

固定转为服务特征维度：

```text
FS = 6
```

服务特征顺序：

| index | feature | 说明 |
|---:|---|---|
| 0 | request_rate_norm | 请求率 |
| 1 | response_time_norm | 响应时间 |
| 2 | is_latency_service | 服务类型 one-hot |
| 3 | is_reliability_service | 服务类型 one-hot |
| 4 | is_cost_service | 服务类型 one-hot |
| 5 | last_violation | 上一时间片是否违约 |

### 4.4 sla_log.csv

```csv
service_id,service_type,max_delay,max_loss,min_bandwidth,reliability_req,cost_weight
s0,latency,0.50,0.08,0.40,0.95,0.20
s1,reliability,0.70,0.03,0.35,0.99,0.30
```

固定 SLA 特征维度：

```text
FC = 5
```

SLA 特征顺序：

| index | feature | 说明 |
|---:|---|---|
| 0 | max_delay_norm | 最大允许时延 |
| 1 | max_loss_norm | 最大允许丢包率 |
| 2 | min_bandwidth_norm | 最小带宽需求 |
| 3 | reliability_req_norm | 可靠性要求 |
| 4 | cost_weight | 成本权重 |

### 4.5 path_log.csv

```csv
time,service_id,path_nodes,path_links,path_delay,path_loss,bottleneck_node,bottleneck_link
0,s0,"u0|a0|e0|c0","u0-a0|a0-e0|e0-c0",0.41,0.03,e0,e0-c0
```

用于构造 service-path subgraph。

---

## 5. Service-path subgraph 固定张量规格

为了避免不同样本节点数和链路数不同导致 DataLoader 报错，本项目采用固定 padding。

```yaml
max_nodes: 8
max_links: 12
input_window: 12
pred_horizon: 5
```

### 5.1 单个样本格式

每个样本保存为：

```python
sample = {
    "node_x": FloatTensor[L, N_MAX, FN],
    "link_x": FloatTensor[L, E_MAX, FE],
    "service_x": FloatTensor[L, FS],
    "sla_x": FloatTensor[FC],
    "adj": FloatTensor[N_MAX, N_MAX],
    "link_index": LongTensor[E_MAX, 2],
    "node_mask": BoolTensor[N_MAX],
    "link_mask": BoolTensor[E_MAX],
    "risk_label": LongTensor[],
    "risk_node_label": LongTensor[],
    "risk_link_label": LongTensor[],
    "risk_metric_label": LongTensor[],
    "sample_id": str
}
```

默认常量：

```python
L = 12
H = 5
N_MAX = 8
E_MAX = 12
FN = 10
FE = 8
FS = 6
FC = 5
```

因此单样本 shape 为：

```text
node_x:     [12, 8, 10]
link_x:     [12, 12, 8]
service_x:  [12, 6]
sla_x:      [5]
adj:        [8, 8]
link_index: [12, 2]
node_mask:  [8]
link_mask:  [12]
```

### 5.2 padding 规则

如果实际子图节点数小于 8：

```text
多余 node_x 全部填 0
node_mask 对应位置为 False
adj 对应行列为 0
risk_node_label 不允许指向 padding 节点
```

如果实际子图链路数小于 12：

```text
多余 link_x 全部填 0
link_mask 对应位置为 False
link_index 填 -1
risk_link_label 不允许指向 padding 链路
```

如果实际节点数超过 8：

保留优先级：

```text
user/access node > current edge > cloud > path nodes > candidate edges > competing nodes
```

如果实际链路数超过 12：

保留优先级：

```text
current path links > bottleneck links > candidate path links > neighboring links
```

---

## 6. 最终训练集文件格式

建议最终保存为 `.npz`，不要一开始用复杂嵌套 pickle。

文件：

```text
data/sparta_dataset/train.npz
data/sparta_dataset/val.npz
data/sparta_dataset/test.npz
```

每个 `.npz` 包含：

```python
node_x: np.float32 [S, 12, 8, 10]
link_x: np.float32 [S, 12, 12, 8]
service_x: np.float32 [S, 12, 6]
sla_x: np.float32 [S, 5]
adj: np.float32 [S, 8, 8]
link_index: np.int64 [S, 12, 2]
node_mask: bool [S, 8]
link_mask: bool [S, 12]
risk_label: np.int64 [S]
risk_node_label: np.int64 [S]
risk_link_label: np.int64 [S]
risk_metric_label: np.int64 [S]
```

标签编码：

```python
risk_label:
  0 = normal
  1 = risky
  2 = violated

risk_metric_label:
  0 = delay
  1 = loss
  2 = cpu
  3 = queue
  4 = bandwidth
```

对于 `normal` 样本，归因标签设置为：

```python
risk_node_label = -100
risk_link_label = -100
risk_metric_label = -100
```

并在 loss 中使用：

```python
ignore_index = -100
```

---

## 7. DataLoader batch 规格

`SPARTADataset.__getitem__()` 返回单样本 dict。

DataLoader collate 后 batch 为：

```python
batch = {
    "node_x": FloatTensor[B, 12, 8, 10],
    "link_x": FloatTensor[B, 12, 12, 8],
    "service_x": FloatTensor[B, 12, 6],
    "sla_x": FloatTensor[B, 5],
    "adj": FloatTensor[B, 8, 8],
    "link_index": LongTensor[B, 12, 2],
    "node_mask": BoolTensor[B, 8],
    "link_mask": BoolTensor[B, 12],
    "risk_label": LongTensor[B],
    "risk_node_label": LongTensor[B],
    "risk_link_label": LongTensor[B],
    "risk_metric_label": LongTensor[B]
}
```

模型 forward 的函数签名固定为：

```python
def forward(
    self,
    node_x,        # [B, L, N, FN]
    link_x,        # [B, L, E, FE]
    service_x,     # [B, L, FS]
    sla_x,         # [B, FC]
    adj,           # [B, N, N]
    node_mask=None,# [B, N]
    link_mask=None # [B, E]
):
    ...
```

返回：

```python
outputs = {
    "risk_logits": Tensor[B, 3],
    "node_logits": Tensor[B, N],
    "link_logits": Tensor[B, E],
    "metric_logits": Tensor[B, 5],
    "z": Tensor[B, D]
}
```

---

## 8. SPARTA 模型每个阶段输入输出 shape

默认：

```text
B = batch size
L = 12
N = 8
E = 12
FN = 10
FE = 8
FS = 6
FC = 5
D = 128
```

### 8.1 原始输入

```text
node_x:    [B, 12, 8, 10]
link_x:    [B, 12, 12, 8]
service_x: [B, 12, 6]
sla_x:     [B, 5]
adj:       [B, 8, 8]
```

### 8.2 Heterogeneous Feature Embedding

模块：

```python
NodeMLP:    10 → 128
LinkMLP:     8 → 128
ServiceMLP:  6 → 128
SlaMLP:      5 → 128
```

输出：

```text
node_h:    [B, 12, 8, 128]
link_h:    [B, 12, 12, 128]
service_h: [B, 12, 128]
sla_h:     [B, 128]
```

### 8.3 Service-Path Token Construction

masked mean pooling：

```python
node_ctx = masked_mean(node_h, mask=node_mask, dim=2)  # [B, L, D]
link_ctx = masked_mean(link_h, mask=link_mask, dim=2)  # [B, L, D]
```

输出：

```text
node_ctx:  [B, 12, 128]
link_ctx:  [B, 12, 128]
path_token = node_ctx + link_ctx + service_h
path_token: [B, 12, 128]
```

### 8.4 Temporal Encoder

输入：

```text
path_token: [B, 12, 128]
```

Transformer Encoder 设置：

```python
num_layers = 2
n_heads = 4
d_model = 128
ffn_dim = 512
batch_first = True
```

输出：

```text
temp_h: [B, 12, 128]
z_temp = temp_h[:, -1, :]
z_temp: [B, 128]
```

### 8.5 Topology Encoder

输入取节点 embedding 的时间平均或最后时间片：

```python
node_last = node_h[:, -1, :, :]       # [B, 8, 128]
# 或 node_mean = node_h.mean(dim=1)   # [B, 8, 128]
```

推荐第一版使用：

```python
node_topo_in = node_h.mean(dim=1)     # [B, 8, 128]
```

GCN layer：

```python
H^{k+1} = LayerNorm(H^k + ReLU(A_hat H^k W_k))
```

2 层后：

```text
node_topo_h: [B, 8, 128]
topo_pool:   [B, 128]
```

masked mean：

```python
topo_pool = masked_mean(node_topo_h, node_mask, dim=1)
```

### 8.6 Temporal + Topology Fusion

```python
z = LayerNorm(z_temp + topo_pool)
```

shape：

```text
z: [B, 128]
```

### 8.7 SLA Gate

SLA gate：

```python
gate = sigmoid(MLP(sla_x))  # [B, 128]
z_sla = z * (1 + gate)
```

shape：

```text
gate:  [B, 128]
z_sla: [B, 128]
```

为什么用 `1 + gate` 而不是只用 `gate`：

```text
z * gate 可能把主特征压得太小
z * (1 + gate) 是残差调制，更稳定
```

### 8.8 Risk Head

```python
risk_logits = RiskHead(z_sla)
```

shape：

```text
risk_logits: [B, 3]
```

### 8.9 Node Attribution Head

输入：

```text
node_topo_h: [B, 8, 128]
```

输出：

```python
node_logits = Linear(node_topo_h).squeeze(-1)
```

shape：

```text
node_logits: [B, 8]
```

mask padding 节点：

```python
node_logits = node_logits.masked_fill(~node_mask, -1e9)
```

### 8.10 Link Attribution Head

输入：

```python
link_attr_h = link_h.mean(dim=1)  # [B, 12, 128]
```

输出：

```python
link_logits = Linear(link_attr_h).squeeze(-1)
```

shape：

```text
link_logits: [B, 12]
```

mask padding 链路：

```python
link_logits = link_logits.masked_fill(~link_mask, -1e9)
```

### 8.11 Metric Attribution Head

输入：

```text
z_sla: [B, 128]
```

输出：

```text
metric_logits: [B, 5]
```

---

## 9. Loss 输入输出对应关系

模型输出：

```python
outputs = {
    "risk_logits": [B, 3],
    "node_logits": [B, 8],
    "link_logits": [B, 12],
    "metric_logits": [B, 5]
}
```

batch 标签：

```python
risk_label:        [B]
risk_node_label:   [B]
risk_link_label:   [B]
risk_metric_label: [B]
```

loss：

```python
loss_risk = CE(risk_logits, risk_label)
loss_node = CE(node_logits, risk_node_label, ignore_index=-100)
loss_link = CE(link_logits, risk_link_label, ignore_index=-100)
loss_metric = CE(metric_logits, risk_metric_label, ignore_index=-100)

loss = loss_risk + 0.3 * loss_node + 0.3 * loss_link + 0.2 * loss_metric
```

对于 normal 样本：

```python
risk_label = 0
risk_node_label = -100
risk_link_label = -100
risk_metric_label = -100
```

这样归因 loss 不会强迫 normal 样本找风险来源。

---

## 10. 代码文件必须实现的类和函数

### 10.1 `src/datasets/sparta_dataset.py`

必须实现：

```python
class SPARTADataset(torch.utils.data.Dataset):
    def __init__(self, npz_path: str):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        return {
            "node_x": ...,
            "link_x": ...,
            "service_x": ...,
            "sla_x": ...,
            "adj": ...,
            "link_index": ...,
            "node_mask": ...,
            "link_mask": ...,
            "risk_label": ...,
            "risk_node_label": ...,
            "risk_link_label": ...,
            "risk_metric_label": ...
        }
```

数据类型要求：

```python
node_x/link_x/service_x/sla_x/adj: torch.float32
link_index: torch.long
node_mask/link_mask: torch.bool
labels: torch.long
```

### 10.2 `src/models/sparta.py`

必须实现：

```python
class MLP(nn.Module):
    ...

class TopologyEncoder(nn.Module):
    ...

class SPARTA(nn.Module):
    def __init__(self, cfg):
        ...

    def forward(self, node_x, link_x, service_x, sla_x, adj, node_mask=None, link_mask=None):
        return {
            "risk_logits": risk_logits,
            "node_logits": node_logits,
            "link_logits": link_logits,
            "metric_logits": metric_logits,
            "z": z_sla
        }
```

### 10.3 `src/losses/multitask_loss.py`

必须实现：

```python
class SPARTALoss(nn.Module):
    def __init__(self, risk_class_weights=None, lambda_node=0.3, lambda_link=0.3, lambda_metric=0.2):
        ...

    def forward(self, outputs, batch):
        ...
        return loss, loss_dict
```

### 10.4 `src/train.py`

训练流程：

```python
for batch in train_loader:
    batch = move_to_device(batch)
    outputs = model(
        node_x=batch["node_x"],
        link_x=batch["link_x"],
        service_x=batch["service_x"],
        sla_x=batch["sla_x"],
        adj=batch["adj"],
        node_mask=batch["node_mask"],
        link_mask=batch["link_mask"]
    )
    loss, loss_dict = criterion(outputs, batch)
    loss.backward()
    optimizer.step()
```

### 10.5 `src/debug_check_shapes.py`

必须写一个 shape debug 脚本：

```python
batch = next(iter(loader))
outputs = model(...)
print(batch["node_x"].shape)
print(batch["link_x"].shape)
print(outputs["risk_logits"].shape)
print(outputs["node_logits"].shape)
print(outputs["link_logits"].shape)
print(outputs["metric_logits"].shape)
```

期望输出：

```text
node_x: torch.Size([B, 12, 8, 10])
link_x: torch.Size([B, 12, 12, 8])
service_x: torch.Size([B, 12, 6])
sla_x: torch.Size([B, 5])
risk_logits: torch.Size([B, 3])
node_logits: torch.Size([B, 8])
link_logits: torch.Size([B, 12])
metric_logits: torch.Size([B, 5])
```

---

## 11. shape assert 清单

在 `SPARTA.forward()` 开头加入：

```python
B, L, N, FN = node_x.shape
assert L == self.input_window
assert N == self.max_nodes
assert FN == self.node_feat_dim

B2, L2, E, FE = link_x.shape
assert B2 == B
assert L2 == L
assert E == self.max_links
assert FE == self.link_feat_dim

assert service_x.shape == (B, L, self.service_feat_dim)
assert sla_x.shape == (B, self.sla_feat_dim)
assert adj.shape == (B, N, N)
assert node_mask.shape == (B, N)
assert link_mask.shape == (B, E)
```

在输出前加入：

```python
assert risk_logits.shape == (B, 3)
assert node_logits.shape == (B, N)
assert link_logits.shape == (B, E)
assert metric_logits.shape == (B, 5)
```

这些 assert 能快速发现数据格式不一致。

---

## 12. 预处理脚本的数据契约

### 12.1 `prepare_alibaba_trace.py`

输入：

```text
data/traces/alibaba/raw files
```

输出：

```text
data/processed/alibaba_service_load.csv
```

必须字段：

```text
time,service_id,service_type,request_rate,base_response_time,dependency_count
```

### 12.2 `prepare_google_trace.py`

输出：

```text
data/processed/google_node_load.csv
```

必须字段：

```text
time,node_group,cpu_scale,mem_scale,arrival_scale
```

### 12.3 `run_edgesimpy.py`

输入：

```text
data/processed/alibaba_service_load.csv
可选 data/processed/google_node_load.csv
```

输出：

```text
node_log.csv
link_log.csv
service_log.csv
path_log.csv
sla_log.csv
```

### 12.4 `build_path_graph.py`

输入：

```text
node_log.csv
link_log.csv
service_log.csv
path_log.csv
sla_log.csv
```

输出中间文件：

```text
data/processed/path_graphs_w12.pkl
```

这里可以先存 list of sample dict，暂时不固定成 npz。

### 12.5 `generate_labels.py`

输入：

```text
path_graphs_w12.pkl
```

输出：

```text
path_graphs_w12_h5_labeled.pkl
```

必须添加：

```text
risk_label
risk_node_label
risk_link_label
risk_metric_label
```

### 12.6 `normalize.py`

输入：

```text
path_graphs_w12_h5_labeled.pkl
```

输出：

```text
path_graphs_w12_h5_norm.pkl
```

必须只用 train split 统计均值方差，不能用全数据统计，否则会数据泄漏。

### 12.7 `split_dataset.py`

输入：

```text
path_graphs_w12_h5_norm.pkl
```

输出：

```text
train.npz
val.npz
test.npz
```

划分原则：

```text
按时间顺序 60 / 20 / 20
不要随机打乱原始时间顺序
```

---

## 13. 训练配置文件完整模板

```yaml
seed: 42
save_dir: results/sparta_default

data:
  train_path: data/sparta_dataset/train.npz
  val_path: data/sparta_dataset/val.npz
  test_path: data/sparta_dataset/test.npz
  batch_size: 64
  num_workers: 4

model:
  name: sparta
  hidden_dim: 128
  input_window: 12
  pred_horizon: 5
  max_nodes: 8
  max_links: 12
  node_feat_dim: 10
  link_feat_dim: 8
  service_feat_dim: 6
  sla_feat_dim: 5

  embedding:
    node_mlp_layers: 2
    link_mlp_layers: 2
    service_mlp_layers: 2
    sla_mlp_layers: 2
    dropout: 0.1

  temporal_encoder:
    num_layers: 2
    n_heads: 4
    ffn_dim: 512
    dropout: 0.1

  topology_encoder:
    num_layers: 2
    dropout: 0.1
    residual: true
    layer_norm: true

  sla_gate:
    num_layers: 2
    hidden_dim: 64
    gate_type: sigmoid_residual

  heads:
    num_risk_classes: 3
    num_metric_classes: 5

loss:
  risk_loss: weighted_ce
  lambda_node: 0.3
  lambda_link: 0.3
  lambda_metric: 0.2
  ignore_index: -100

train:
  epochs: 80
  lr: 0.0005
  weight_decay: 0.0001
  optimizer: adamw
  scheduler: cosine
  early_stop_patience: 12
  grad_clip: 1.0

select_metric:
  main: macro_f1
  secondary: risk_recall
```

---

## 14. 模型层数调参实验怎么做

### 14.1 第一步：固定默认模型

先跑：

```text
Temporal=2, Topology=2, D=128, heads=4
```

确保代码和结果正常。

### 14.2 第二步：只搜索 Temporal 层数

保持其他不变：

| Exp | Temporal layers | Topology layers |
|---|---:|---:|
| T1 | 1 | 2 |
| T2 | 2 | 2 |
| T3 | 3 | 2 |

### 14.3 第三步：只搜索 Topology 层数

| Exp | Temporal layers | Topology layers |
|---|---:|---:|
| G1 | 2 | 1 |
| G2 | 2 | 2 |
| G3 | 2 | 3 |

### 14.4 第四步：必要时搜索 hidden_dim

| Exp | hidden_dim | n_heads |
|---|---:|---:|
| D64 | 64 | 4 |
| D128 | 128 | 4 |
| D256 | 256 | 4/8 |

注意：

```text
hidden_dim 必须能被 n_heads 整除
```

### 14.5 最终选择规则

选择验证集得分最高的配置：

```text
score = 0.4 MacroF1 + 0.3 RiskRecall + 0.2 ViolationRecall + 0.1 AttrAcc
```

如果更深模型只提升很小：

```text
提升 < 0.5% 或 < 1%
```

选择更浅模型。

论文中可以写：

```text
We use two temporal layers and two topology layers by default. A depth sensitivity study shows that deeper encoders bring marginal improvement but higher complexity; therefore, we select the 2-layer configuration for the final model.
```

---

## 15. Baseline 输入如何和 SPARTA 对齐

### 15.1 LSTM / GRU / Transformer / TCN 输入

这些模型不使用 graph 结构，只使用 path-level token。

输入：

```text
path_token_raw: [B, L, F_flat]
```

其中：

```text
F_flat = FN_node_summary + FE_link_summary + FS + FC
```

建议不要直接 flatten `[N,FN]`，而是使用统计 summary：

节点 summary：

```text
mean cpu, max cpu, mean queue, max queue, mean mem, max request_load
```

链路 summary：

```text
mean delay, max delay, mean loss, max loss, min bandwidth, max bandwidth_util
```

这样 baseline 更合理，维度更稳定。

### 15.2 STGCN 输入

输入全局或服务路径图：

```text
X: [B, L, N, C]
adj: [N, N]
```

如果使用全局图，`N` 是全网节点数；如果使用服务路径图，`N=8`。

建议主 baseline 使用：

```text
STGCN-global
```

消融中使用：

```text
SPARTA-global graph
SPARTA-service path graph
```

---

## 16. 训练前必须运行的检查脚本

### 16.1 检查数据分布

```bash
python src/debug/check_dataset_stats.py --path data/sparta_dataset/train.npz
```

必须输出：

```text
num_samples
risk_label distribution
risk_node_label valid ratio
risk_link_label valid ratio
risk_metric_label valid ratio
node_x min/max/mean/std
link_x min/max/mean/std
service_x min/max/mean/std
sla_x min/max/mean/std
```

### 16.2 检查 shape

```bash
python src/debug/check_shapes.py --config configs/sparta.yaml
```

必须通过所有 assert。

### 16.3 检查前向传播

```bash
python src/debug/check_forward.py --config configs/sparta.yaml
```

必须输出：

```text
risk_logits [B,3]
node_logits [B,8]
link_logits [B,12]
metric_logits [B,5]
loss ok
backward ok
```

---

## 17. 最小可运行代码顺序

AI 写代码时应按这个顺序：

```text
1. src/datasets/sparta_dataset.py
2. src/models/sparta.py
3. src/losses/multitask_loss.py
4. src/debug/check_shapes.py
5. src/train.py
6. src/evaluate.py
7. baseline models
8. preprocessing scripts
9. simulation scripts
10. visualization scripts
```

原因：

```text
先固定模型输入输出契约，再写复杂数据生成。
可以先用 random dummy data 测通模型，避免等仿真跑完才发现维度不对。
```

---

## 18. 用 dummy data 先跑通模型

在真实数据生成前，先写：

```python
def make_dummy_batch(B=4):
    return {
        "node_x": torch.randn(B, 12, 8, 10),
        "link_x": torch.randn(B, 12, 12, 8),
        "service_x": torch.randn(B, 12, 6),
        "sla_x": torch.randn(B, 5),
        "adj": torch.eye(8).unsqueeze(0).repeat(B, 1, 1),
        "node_mask": torch.ones(B, 8).bool(),
        "link_mask": torch.ones(B, 12).bool(),
        "risk_label": torch.randint(0, 3, (B,)),
        "risk_node_label": torch.randint(0, 8, (B,)),
        "risk_link_label": torch.randint(0, 12, (B,)),
        "risk_metric_label": torch.randint(0, 5, (B,)),
    }
```

如果 dummy batch 都跑不通，不要继续做仿真数据。

---

## 19. 论文中如何说明层数选择

建议写法：

```text
The hidden dimension is set to 128 and the number of attention heads is set to 4. Since the input history window is short and the service-path subgraph is small, we use a lightweight two-layer temporal encoder and a two-layer topology encoder. We also evaluate encoder depths from 1 to 3 layers on the validation set and observe that the two-layer configuration provides the best trade-off between risk detection performance and inference cost.
```

消融表可以放：

| Temporal layers | Topology layers | Macro-F1 | Risk Recall | Time |
|---:|---:|---:|---:|---:|
| 1 | 2 | - | - | - |
| 2 | 2 | best | best | moderate |
| 3 | 2 | +small / - | +small / - | slower |
| 2 | 1 | - | - | faster |
| 2 | 3 | +small / - | +small / - | slower |

---

## 20. 最终给 AI/Codex 的代码生成要求

可以把下面这段直接给代码 AI：

```text
请严格按照以下数据契约实现 SPARTA 项目。所有模型输入输出 shape 不允许自行更改。

输入 batch：
node_x: FloatTensor[B,12,8,10]
link_x: FloatTensor[B,12,12,8]
service_x: FloatTensor[B,12,6]
sla_x: FloatTensor[B,5]
adj: FloatTensor[B,8,8]
node_mask: BoolTensor[B,8]
link_mask: BoolTensor[B,12]
risk_label: LongTensor[B]
risk_node_label: LongTensor[B]
risk_link_label: LongTensor[B]
risk_metric_label: LongTensor[B]

模型输出：
risk_logits: FloatTensor[B,3]
node_logits: FloatTensor[B,8]
link_logits: FloatTensor[B,12]
metric_logits: FloatTensor[B,5]

SPARTA 默认结构：
- NodeMLP: 10→128, 2 layers
- LinkMLP: 8→128, 2 layers
- ServiceMLP: 6→128, 2 layers
- SlaMLP: 5→128, 2 layers
- Temporal Encoder: TransformerEncoder, 2 layers, d_model=128, n_heads=4, ffn_dim=512, dropout=0.1, batch_first=True
- Topology Encoder: 2-layer GCN with residual and layer norm
- SLA Gate: 2-layer MLP, output sigmoid gate [B,128], fusion z_sla=z*(1+gate)
- Risk Head: MLP 128→128→3
- Node Head: Linear 128→1 over node_topo_h, output [B,8]
- Link Head: Linear 128→1 over link_h.mean(dim=1), output [B,12]
- Metric Head: MLP 128→128→5

Loss:
L = CE(risk_logits,risk_label)
  + 0.3 CE(node_logits,risk_node_label,ignore_index=-100)
  + 0.3 CE(link_logits,risk_link_label,ignore_index=-100)
  + 0.2 CE(metric_logits,risk_metric_label,ignore_index=-100)

必须写 shape assert 和 dummy forward test，确保所有维度对齐后再写训练代码。
```

---

## 21. 最终确认

这份规格固定了三件事：

1. **数据格式固定**：从 CSV 到 NPZ，再到 DataLoader batch，每个字段都有 shape。
2. **模型接口固定**：SPARTA 的 forward 输入输出完全确定。
3. **层数选择方法固定**：默认 2 层 temporal + 2 层 topology，通过验证集深度敏感性实验确定最终配置。

这样后续 AI 按本文件写代码时，最常见的数据维度、通道数量、mask、标签、loss 不匹配问题可以大幅减少。

