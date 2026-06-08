# SPARTA Win11 最小可运行 MVP：Codex 搭建说明书

> 目标：在 Windows 11 本地先复现一个**最小可运行**的 SPARTA 算法闭环。  
> 范围：暂不接入 Alibaba / Google 真实 trace，暂不强依赖 EdgeSimPy，先用纯 Python synthetic 数据跑通：  
> `模拟 trace → 模拟云边日志 → service-path 样本 → train/val/test → SPARTA forward/backward → 训练 → 推理 → 权重保存 → 结果 CSV`。  
> 使用对象：Codex / VS Code / PyCharm 中的代码助手。  
> 核心要求：**完整、可运行、维度统一、命名统一、标签统一，不允许在代码中出现多套互相矛盾的数据接口。**

---

## 0. 本 MVP 的最终统一口径

原有 5 个说明文件中存在几处实现口径不一致。为了让 Codex 一次性写出能跑的代码，本文件将它们统一如下。

### 0.1 统一后的数据维度

本 MVP 采用 `SPARTA_Model_Data_Interface_Spec.md` 中更完整的一套特征维度，同时保留 Windows debug 的小规模设置。

| 名称 | debug 默认值 | full 默认值 | 说明 |
|---|---:|---:|---|
| `L` / `input_window` | 4 | 12 | 历史输入窗口 |
| `H` / `pred_horizon` | 2 | 5 | 未来预测窗口 |
| `max_nodes` | 6 | 8 | 每个 service-path 子图最大节点数 |
| `max_links` | 8 | 12 | 每个 service-path 子图最大链路数 |
| `node_feat_dim` | 10 | 10 | 节点特征维度，固定为 10 |
| `link_feat_dim` | 8 | 8 | 链路特征维度，固定为 8 |
| `service_feat_dim` | 6 | 6 | 服务特征维度，固定为 6 |
| `sla_feat_dim` | 5 | 5 | SLA 特征维度，固定为 5 |
| `num_classes` | 3 | 3 | normal / risky / violated |
| `num_metrics` | 5 | 5 | delay / loss / cpu / queue / bandwidth |

**禁止在 MVP 代码中再使用 `node_feat_dim=8` 或 `service_feat_dim=5`。** 这两个维度来自早期简化口径，本文件统一改为 `10/8/6/5`。

### 0.2 统一后的数据保存格式

中间样本可以用 `pkl`，最终训练集统一用 `npz`。

```text
中间文件：path_graphs.pkl、samples_labeled.pkl
最终训练文件：train.npz、val.npz、test.npz
```

这样做的原因是：`pkl` 适合保存包含 `node_ids/link_ids/sample_id` 的中间对象；`npz` 适合保存固定 shape 张量数组，Dataset 更简单，Windows 下也更稳定。

### 0.3 统一后的标签命名

最终训练数据和 batch 中统一使用以下字段名：

```python
risk_label
risk_node_label
risk_link_label
risk_metric_label
```

不要同时混用 `risk_node`、`risk_link`、`risk_metric`。如果历史代码中出现这些字段，必须统一重命名为带 `_label` 的版本。

### 0.4 统一后的 normal 样本归因标签

对于 normal 样本，归因标签统一使用 `-100`，并在 loss 中使用 `ignore_index=-100`。

```python
if risk_label == 0:
    risk_node_label = -100
    risk_link_label = -100
    risk_metric_label = -100
```

不要使用 `-1` 作为最终训练标签。`-1` 只能在临时调试中出现，但保存到 `train.npz/val.npz/test.npz` 前必须转换为 `-100`。

### 0.5 统一后的模型调用方式

为了让 SPARTA、LSTM、Transformer baseline 使用同一个训练器，所有模型的公共 forward 接口统一为：

```python
outputs = model(batch)
```

其中 `batch` 是 Dataset 返回并经 DataLoader collate 后的字典。SPARTA 内部再从 `batch` 中取出：

```python
node_x, link_x, service_x, sla_x, adj, node_mask, link_mask
```

SPARTA 输出统一为：

```python
{
    "risk_logits": Tensor[B, 3],
    "node_logits": Tensor[B, max_nodes],
    "link_logits": Tensor[B, max_links],
    "metric_logits": Tensor[B, 5],
    "z": Tensor[B, hidden_dim]
}
```

LSTM / Transformer baseline 只需输出：

```python
{
    "risk_logits": Tensor[B, 3]
}
```

baseline 不输出 attribution，评估时 attribution 指标写 `nan`。

---

## 1. Codex 总任务说明

请 Codex 严格按照本文件搭建一个 Windows 11 可运行的 SPARTA MVP 项目。不要添加复杂功能，不要接真实 trace，不要写强化学习，不要写复杂 GAT，不要下载第三方仿真器。第一阶段只需要保证下面命令能完整运行：

```powershell
python src/run_debug_pipeline.py --config configs/sparta_debug.yaml
```

运行成功后必须生成：

```text
data/debug/traces/debug_service_trace.csv
data/debug/traces/debug_node_trace.csv
data/debug/raw_logs/node_log.csv
data/debug/raw_logs/link_log.csv
data/debug/raw_logs/service_log.csv
data/debug/raw_logs/path_log.csv
data/debug/raw_logs/sla_log.csv
data/debug/processed/path_graphs.pkl
data/debug/processed/samples_labeled.pkl
data/debug/dataset/train.npz
data/debug/dataset/val.npz
data/debug/dataset/test.npz
checkpoints/debug/lstm_best.pth
checkpoints/debug/transformer_best.pth
checkpoints/debug/sparta_best.pth
results/debug/lstm_test_results.csv
results/debug/transformer_test_results.csv
results/debug/sparta_test_results.csv
logs/debug/train_log_lstm.csv
logs/debug/train_log_transformer.csv
logs/debug/train_log_sparta.csv
```

MVP 阶段不要求 SPARTA 指标一定高于 baseline，但必须满足：

```text
1. Dataset shape 全部正确；
2. SPARTA forward 输出四类 logits；
3. loss 可计算且不是 NaN；
4. backward 可执行；
5. checkpoint 可保存并可再次加载；
6. evaluate.py 可在 test set 上输出 CSV；
7. run_debug_pipeline.py 一键跑完整流程；
8. 所有测试脚本通过。
```

---

## 2. Windows 11 环境要求

### 2.1 项目路径

建议将项目放到英文路径，例如：

```text
D:\SPARTA
```

不要放到中文路径或含空格路径中，例如：

```text
D:\刘圳\第二篇论文\SPARTA
D:\My Project\SPARTA
```

### 2.2 Conda 环境

在 Anaconda Prompt 或 PowerShell 中执行：

```powershell
conda create -n sparta python=3.10 -y
conda activate sparta
```

CPU 版 PyTorch 足够跑通 debug：

```powershell
pip install torch torchvision torchaudio
```

安装基础依赖：

```powershell
pip install numpy pandas scipy scikit-learn networkx matplotlib tqdm pyyaml pytest
```

可选依赖：

```powershell
pip install einops
```

### 2.3 requirements.txt

Codex 需要创建：

```text
requirements.txt
```

内容：

```txt
torch
torchvision
torchaudio
numpy
pandas
scipy
scikit-learn
networkx
matplotlib
tqdm
pyyaml
pytest
einops
```

### 2.4 Windows 特别要求

在所有 DataLoader 中，debug 阶段必须设置：

```python
num_workers = 0
```

所有路径处理必须使用：

```python
from pathlib import Path
```

不要在代码中写 Linux 风格硬编码路径，不要写 `.sh` 脚本作为主流程。

---

## 3. 项目目录结构

Codex 必须创建如下目录结构：

```text
SPARTA/
├── configs/
│   ├── sparta_debug.yaml
│   ├── sparta.yaml
│   ├── lstm.yaml
│   └── transformer.yaml
├── data/
│   └── debug/
│       ├── traces/
│       ├── raw_logs/
│       ├── processed/
│       └── dataset/
├── src/
│   ├── simulation/
│   │   ├── make_debug_trace.py
│   │   └── run_synthetic_simulation.py
│   ├── preprocessing/
│   │   ├── build_path_graph.py
│   │   ├── generate_labels.py
│   │   ├── split_dataset.py
│   │   └── check_dataset_stats.py
│   ├── datasets/
│   │   └── sparta_dataset.py
│   ├── models/
│   │   ├── sparta.py
│   │   ├── lstm.py
│   │   └── transformer.py
│   ├── losses/
│   │   └── multitask_loss.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── seed.py
│   │   ├── io.py
│   │   └── shape_check.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_debug_pipeline.py
├── tests/
│   ├── test_dataset_shape.py
│   ├── test_model_forward.py
│   ├── test_loss_backward.py
│   └── test_train_one_batch.py
├── checkpoints/
│   └── debug/
├── results/
│   └── debug/
├── logs/
│   └── debug/
├── README.md
└── requirements.txt
```

MVP 阶段不创建或不使用：

```text
simulators/EdgeSimPy/
simulators/YAFS/
third_party/
data/traces/alibaba/
data/traces/google/
```

---

## 4. 配置文件

## 4.1 `configs/sparta_debug.yaml`

Codex 必须创建该文件，并以此作为首要运行配置。

```yaml
project:
  name: sparta_debug
  root_dir: .
  output_dir: results/debug
  checkpoint_dir: checkpoints/debug
  log_dir: logs/debug

seed: 42

data:
  mode: debug
  base_dir: data/debug
  traces_dir: data/debug/traces
  raw_logs_dir: data/debug/raw_logs
  processed_dir: data/debug/processed
  dataset_dir: data/debug/dataset

  input_window: 4
  pred_horizon: 2
  max_nodes: 6
  max_links: 8

  node_feat_dim: 10
  link_feat_dim: 8
  service_feat_dim: 6
  sla_feat_dim: 5

  num_classes: 3
  num_metrics: 5

  num_edge_nodes: 5
  num_access_nodes: 2
  num_users: 20
  num_services: 3
  num_steps: 220

  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2

simulation:
  topology: small_world
  enable_node_overload: true
  enable_link_congestion: true
  enable_burst: true
  delay_cap_ms: 300.0
  bandwidth_cap: 100.0
  queue_cap: 50.0
  request_cap: 100.0
  response_cap_ms: 400.0

model:
  name: sparta
  hidden_dim: 32
  temporal_layers: 1
  topology_layers: 1
  n_heads: 2
  ffn_dim: 128
  dropout: 0.1
  use_sla_gate: true
  use_topology: true
  use_attribution: true

loss:
  risk_loss: ce
  lambda_node: 0.3
  lambda_link: 0.3
  lambda_metric: 0.2
  ignore_index: -100

train:
  batch_size: 8
  epochs: 3
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  num_workers: 0
  device: auto
  best_metric: macro_f1
```

### 4.2 `configs/sparta.yaml`

正式小规模版本可以后续使用。MVP 跑通前可以先创建但不运行。

```yaml
project:
  name: sparta_full
  root_dir: .
  output_dir: results/sparta
  checkpoint_dir: checkpoints/sparta
  log_dir: logs/sparta

seed: 42

data:
  mode: full
  base_dir: data/sparta_dataset
  input_window: 12
  pred_horizon: 5
  max_nodes: 8
  max_links: 12
  node_feat_dim: 10
  link_feat_dim: 8
  service_feat_dim: 6
  sla_feat_dim: 5
  num_classes: 3
  num_metrics: 5
  train_path: data/sparta_dataset/train.npz
  val_path: data/sparta_dataset/val.npz
  test_path: data/sparta_dataset/test.npz

model:
  name: sparta
  hidden_dim: 128
  temporal_layers: 2
  topology_layers: 2
  n_heads: 4
  ffn_dim: 512
  dropout: 0.1
  use_sla_gate: true
  use_topology: true
  use_attribution: true

loss:
  risk_loss: weighted_ce
  lambda_node: 0.3
  lambda_link: 0.3
  lambda_metric: 0.2
  ignore_index: -100

train:
  batch_size: 64
  epochs: 100
  lr: 0.0005
  weight_decay: 0.0001
  grad_clip: 1.0
  patience: 15
  num_workers: 0
  device: auto
  best_metric: macro_f1
```

### 4.3 `configs/lstm.yaml` 和 `configs/transformer.yaml`

baseline 可以复用 debug 数据，只需替换模型名。为了减少重复，`train.py` 支持通过命令行 `--model lstm` 覆盖 config 中的 `model.name`。因此 baseline yaml 可以很简单：

```yaml
inherit: configs/sparta_debug.yaml
model:
  name: lstm
  hidden_dim: 32
  temporal_layers: 1
  dropout: 0.1
```

```yaml
inherit: configs/sparta_debug.yaml
model:
  name: transformer
  hidden_dim: 32
  temporal_layers: 1
  n_heads: 2
  ffn_dim: 128
  dropout: 0.1
```

如果 Codex 不方便实现 `inherit`，则可不使用这两个文件，直接运行：

```powershell
python src/train.py --config configs/sparta_debug.yaml --model lstm
python src/train.py --config configs/sparta_debug.yaml --model transformer
python src/train.py --config configs/sparta_debug.yaml --model sparta
```

---

## 5. 数据文件与字段契约

## 5.1 `debug_service_trace.csv`

路径：

```text
data/debug/traces/debug_service_trace.csv
```

字段：

```csv
time,service_id,service_type,request_rate,base_response_time_ms,dependency_count
```

字段含义：

| 字段 | 类型 | 说明 |
|---|---|---|
| `time` | int | 时间片 |
| `service_id` | int | 服务编号，0 到 `num_services-1` |
| `service_type` | str | `latency` / `reliability` / `cost` |
| `request_rate` | float | 请求率 |
| `base_response_time_ms` | float | 基础响应时间，单位 ms |
| `dependency_count` | int | 依赖复杂度，debug 阶段可随机为 1–5 |

生成逻辑：

```text
request_rate = base + sinusoidal fluctuation + Gaussian noise + burst
base_response_time_ms = service_type-dependent base + noise
dependency_count = small integer, debug 阶段固定或随机
```

必须故意制造少量 burst，否则数据全是 normal，训练没有意义。

---

## 5.2 `debug_node_trace.csv`

路径：

```text
data/debug/traces/debug_node_trace.csv
```

字段：

```csv
time,node_group,cpu_scale,mem_scale,arrival_scale
```

字段含义：

| 字段 | 类型 | 说明 |
|---|---|---|
| `time` | int | 时间片 |
| `node_group` | str | debug 阶段可为 `edge` |
| `cpu_scale` | float | 背景 CPU 负载缩放，0–1 |
| `mem_scale` | float | 背景内存负载缩放，0–1 |
| `arrival_scale` | float | 请求到达强度缩放 |

生成逻辑：

```text
cpu_scale = smooth curve + overload interval
mem_scale = cpu_scale 的平滑变体
arrival_scale = request burst interval
```

---

## 5.3 `node_log.csv`

路径：

```text
data/debug/raw_logs/node_log.csv
```

字段：

```csv
time,node_id,node_type,cpu_util,mem_util,queue_len,available_cpu,available_mem,request_load_on_node
```

节点类型：

```text
cloud
edge
access
user
```

数值范围：

| 字段 | 范围 |
|---|---|
| `cpu_util` | `[0,1]` |
| `mem_util` | `[0,1]` |
| `queue_len` | `>=0` |
| `available_cpu` | `[0,1]`，通常为 `1-cpu_util` |
| `available_mem` | `[0,1]`，通常为 `1-mem_util` |
| `request_load_on_node` | `>=0` |

---

## 5.4 `link_log.csv`

路径：

```text
data/debug/raw_logs/link_log.csv
```

字段：

```csv
time,src,dst,delay_ms,bandwidth,loss,jitter_ms,queue_delay_ms,bandwidth_util
```

数值范围：

| 字段 | 范围 |
|---|---|
| `delay_ms` | `>0` |
| `bandwidth` | `>0` |
| `loss` | `[0,1]` |
| `jitter_ms` | `>=0` |
| `queue_delay_ms` | `>=0` |
| `bandwidth_util` | `[0,1]` |

链路 ID 在代码内部统一表示为：

```python
f"{src}-{dst}"
```

无向图中查询链路时，需要同时兼容：

```python
src-dst
dst-src
```

---

## 5.5 `service_log.csv`

路径：

```text
data/debug/raw_logs/service_log.csv
```

字段：

```csv
time,service_id,user_id,service_type,request_rate,response_time_ms,current_edge,cloud_node,last_violation
```

说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `current_edge` | str | 当前承载该服务的边缘节点 ID |
| `cloud_node` | str | 云节点 ID，debug 阶段固定为 `cloud_0` |
| `last_violation` | int | 上一时间片是否 violated，0/1 |

---

## 5.6 `path_log.csv`

路径：

```text
data/debug/raw_logs/path_log.csv
```

字段：

```csv
time,service_id,path_nodes,path_links,path_delay_ms,path_loss,bottleneck_node,bottleneck_link
```

字符串格式：

```text
path_nodes: "user_0|access_0|edge_2|cloud_0"
path_links: "user_0-access_0|access_0-edge_2|edge_2-cloud_0"
```

如果某条路径实际不存在，synthetic simulator 必须重新选择可达路径，不能写空路径。

---

## 5.7 `sla_log.csv`

路径：

```text
data/debug/raw_logs/sla_log.csv
```

字段：

```csv
service_id,service_type,max_delay_ms,max_loss,min_bandwidth,reliability_req,cost_weight
```

debug 推荐值：

| service_type | `max_delay_ms` | `max_loss` | `min_bandwidth` | `reliability_req` | `cost_weight` |
|---|---:|---:|---:|---:|---:|
| `latency` | 100 | 0.05 | 20 | 0.99 | 0.2 |
| `reliability` | 150 | 0.02 | 15 | 0.999 | 0.3 |
| `cost` | 200 | 0.08 | 10 | 0.95 | 0.7 |

为了保证 debug 中存在 risky / violated，可以在合成数据中让部分响应时间超过 0.75×SLA 或 1.0×SLA。

---

## 6. 最终训练数组契约

`split_dataset.py` 最终保存：

```text
data/debug/dataset/train.npz
data/debug/dataset/val.npz
data/debug/dataset/test.npz
```

每个 `.npz` 必须包含以下 key：

```python
node_x              # float32 [S, L, max_nodes, 10]
link_x              # float32 [S, L, max_links, 8]
service_x           # float32 [S, L, 6]
sla_x               # float32 [S, 5]
adj                 # float32 [S, max_nodes, max_nodes]
node_mask           # bool    [S, max_nodes]
link_mask           # bool    [S, max_links]
risk_label          # int64   [S]
risk_node_label     # int64   [S]
risk_link_label     # int64   [S]
risk_metric_label   # int64   [S]
sample_time         # int64   [S]
service_id          # int64   [S]
```

debug 配置下 shape 为：

```text
node_x:            [S, 4, 6, 10]
link_x:            [S, 4, 8, 8]
service_x:         [S, 4, 6]
sla_x:             [S, 5]
adj:               [S, 6, 6]
node_mask:         [S, 6]
link_mask:         [S, 8]
risk_label:        [S]
risk_node_label:   [S]
risk_link_label:   [S]
risk_metric_label: [S]
```

正式配置下 shape 为：

```text
node_x:            [S, 12, 8, 10]
link_x:            [S, 12, 12, 8]
service_x:         [S, 12, 6]
sla_x:             [S, 5]
adj:               [S, 8, 8]
node_mask:         [S, 8]
link_mask:         [S, 12]
```

---

## 7. 特征字典

## 7.1 节点特征 `node_x[..., 10]`

固定顺序如下，不允许改动：

| index | feature | 范围 | 构造方式 |
|---:|---|---|---|
| 0 | `cpu_util` | `[0,1]` | 来自 node_log |
| 1 | `mem_util` | `[0,1]` | 来自 node_log |
| 2 | `queue_len_norm` | `[0,1]` | `queue_len / queue_cap` 后裁剪 |
| 3 | `available_cpu` | `[0,1]` | 来自 node_log |
| 4 | `available_mem` | `[0,1]` | 来自 node_log |
| 5 | `is_cloud` | 0/1 | node_type one-hot |
| 6 | `is_edge` | 0/1 | node_type one-hot |
| 7 | `is_access` | 0/1 | node_type one-hot |
| 8 | `is_user` | 0/1 | node_type one-hot |
| 9 | `request_load_norm` | `[0,1]` | `request_load_on_node / request_cap` 后裁剪 |

## 7.2 链路特征 `link_x[..., 8]`

固定顺序如下：

| index | feature | 范围 | 构造方式 |
|---:|---|---|---|
| 0 | `delay_norm` | `[0,1]` | `delay_ms / delay_cap_ms` |
| 1 | `bandwidth_norm` | `[0,1]` | `bandwidth / bandwidth_cap` |
| 2 | `loss_norm` | `[0,1]` | `loss` |
| 3 | `jitter_norm` | `[0,1]` | `jitter_ms / delay_cap_ms` |
| 4 | `queue_delay_norm` | `[0,1]` | `queue_delay_ms / delay_cap_ms` |
| 5 | `bandwidth_util` | `[0,1]` | 来自 link_log |
| 6 | `is_current_path` | 0/1 | path_log 中的当前路径链路 |
| 7 | `hop_position_norm` | `[0,1]` | 当前链路在 path_links 中的位置 / max_links |

## 7.3 服务特征 `service_x[..., 6]`

固定顺序如下：

| index | feature | 范围 | 构造方式 |
|---:|---|---|---|
| 0 | `request_rate_norm` | `[0,1]` | `request_rate / request_cap` |
| 1 | `response_time_norm` | `[0,1]` | `response_time_ms / response_cap_ms` |
| 2 | `is_latency_service` | 0/1 | service_type one-hot |
| 3 | `is_reliability_service` | 0/1 | service_type one-hot |
| 4 | `is_cost_service` | 0/1 | service_type one-hot |
| 5 | `last_violation` | 0/1 | 来自 service_log |

## 7.4 SLA 特征 `sla_x[5]`

固定顺序如下：

| index | feature | 范围 | 构造方式 |
|---:|---|---|---|
| 0 | `max_delay_norm` | `[0,1]` | `max_delay_ms / response_cap_ms` |
| 1 | `max_loss_norm` | `[0,1]` | `max_loss` |
| 2 | `min_bandwidth_norm` | `[0,1]` | `min_bandwidth / bandwidth_cap` |
| 3 | `reliability_req` | `[0,1]` | 来自 sla_log |
| 4 | `cost_weight` | `[0,1]` | 来自 sla_log |

---

## 8. 样本构造规则

### 8.1 样本时间定义

对于每个服务 `s` 和当前时间 `t`，一个样本使用过去 `L` 个时间片：

```text
input window = [t-L+1, ..., t]
future window = [t+1, ..., t+H]
```

只有当 `t >= L-1` 且 `t+H < num_steps` 时才能构造样本。

### 8.2 子图节点选择

debug 阶段每个 service-path 子图至少包含：

```text
user node
access node
current edge node
cloud node
```

如果不足 `max_nodes`，可加入其他 candidate edge nodes。若超过 `max_nodes`，保留优先级为：

```text
user/access node → current edge → cloud → current path nodes → candidate edges
```

### 8.3 子图链路选择

至少包含当前路径链路：

```text
user-access
access-edge
edge-cloud
```

若不足 `max_links`，padding。若超过 `max_links`，保留优先级为：

```text
current path links → bottleneck link → candidate path links → neighboring links
```

### 8.4 Padding 与 mask

如果实际节点数 `n_real < max_nodes`：

```python
node_x[:, n_real:max_nodes, :] = 0
node_mask[:n_real] = True
node_mask[n_real:] = False
```

如果实际链路数 `e_real < max_links`：

```python
link_x[:, e_real:max_links, :] = 0
link_mask[:e_real] = True
link_mask[e_real:] = False
```

风险归因标签不能指向 padding 位置。

### 8.5 邻接矩阵

`adj` shape 为：

```text
[max_nodes, max_nodes]
```

构造规则：

```python
adj[i, j] = 1 if node_i and node_j are connected by a selected link else 0
adj[i, i] = 1 for valid nodes
padding rows/cols = 0
```

---

## 9. 标签生成规则

### 9.1 风险标签 `risk_label`

对于样本 `(service_id=s, time=t)`，查看未来窗口 `[t+1, ..., t+H]`。

#### violated：标签为 2

未来窗口内任一条件成立：

```text
response_time_ms >= max_delay_ms
path_loss >= max_loss
```

#### risky：标签为 1

未来窗口没有 violated，但任一条件成立：

```text
response_time_ms >= 0.75 * max_delay_ms
path_loss >= 0.75 * max_loss
max CPU on path >= 0.85
max queue_len_norm on path >= 0.85
max bandwidth_util on path >= 0.85
```

#### normal：标签为 0

以上条件均不满足。

### 9.2 风险节点归因 `risk_node_label`

只对 risky / violated 样本计算。节点风险分数：

```text
node_score = 0.4 * cpu_util
           + 0.3 * queue_len_norm
           + 0.2 * request_load_norm
           + 0.1 * (1 - available_cpu)
```

在 future window 对路径节点计算最大或平均分数，取最高分节点对应的**局部 node index** 作为标签。

如果 `risk_label == 0`：

```python
risk_node_label = -100
```

### 9.3 风险链路归因 `risk_link_label`

只对 risky / violated 样本计算。链路风险分数：

```text
link_score = 0.4 * delay_norm
           + 0.25 * loss_norm
           + 0.25 * bandwidth_util
           + 0.10 * queue_delay_norm
```

取最高分链路对应的**局部 link index** 作为标签。

如果 `risk_label == 0`：

```python
risk_link_label = -100
```

### 9.4 风险指标归因 `risk_metric_label`

编码：

```text
0 = delay
1 = loss
2 = cpu
3 = queue
4 = bandwidth
```

计算：

```text
metric_delay = max(response_time_ms / max_delay_ms)
metric_loss = max(path_loss / max_loss)
metric_cpu = max(cpu_util_on_path / 0.85)
metric_queue = max(queue_len_norm_on_path / 0.85)
metric_bandwidth = max(bandwidth_util_on_path / 0.85)
```

取最大者作为 `risk_metric_label`。

如果 `risk_label == 0`：

```python
risk_metric_label = -100
```

---

## 10. 需要实现的 Python 文件

## 10.1 `src/utils/config.py`

职责：读取 YAML 配置，支持递归转 dict，不需要复杂依赖继承。

必须实现：

```python
def load_config(path: str) -> dict:
    """Load yaml config as nested dict."""
```

要求：

```text
1. 使用 yaml.safe_load；
2. 路径不存在时抛出清晰错误；
3. 返回普通 dict。
```

---

## 10.2 `src/utils/seed.py`

职责：固定随机种子。

必须实现：

```python
def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 10.3 `src/utils/io.py`

职责：统一创建目录、读写 pickle、保存 CSV。

必须实现：

```python
def ensure_dir(path) -> None:
    ...

def save_pickle(obj, path) -> None:
    ...

def load_pickle(path):
    ...

def save_json(obj, path) -> None:
    ...
```

所有保存文件前必须调用 `ensure_dir(Path(path).parent)`。

---

## 10.4 `src/simulation/make_debug_trace.py`

职责：生成 debug trace，不依赖外部数据。

输入：

```powershell
python src/simulation/make_debug_trace.py --config configs/sparta_debug.yaml
```

输出：

```text
data/debug/traces/debug_service_trace.csv
data/debug/traces/debug_node_trace.csv
```

必须实现：

```python
def make_service_trace(config: dict) -> pd.DataFrame:
    ...

def make_node_trace(config: dict) -> pd.DataFrame:
    ...

def main():
    ...
```

实现要求：

```text
1. 使用 seed 保证可复现；
2. 生成 num_steps × num_services 行 service trace；
3. 生成 num_steps 行 node trace；
4. 至少设置 2–3 个 burst/overload 区间；
5. request_rate、base_response_time_ms 不允许为负；
6. 保存 CSV 后打印路径和行数。
```

推荐异常区间：

```python
burst_intervals = [(50, 80), (140, 165)]
overload_intervals = [(90, 120), (170, 190)]
```

---

## 10.5 `src/simulation/run_synthetic_simulation.py`

职责：根据 debug trace 生成 5 个云边运行日志。

输入：

```powershell
python src/simulation/run_synthetic_simulation.py --config configs/sparta_debug.yaml
```

输出：

```text
node_log.csv
link_log.csv
service_log.csv
path_log.csv
sla_log.csv
```

必须实现：

```python
def build_topology(config: dict) -> nx.Graph:
    ...

def simulate_node_states(G: nx.Graph, node_trace: pd.DataFrame, config: dict) -> pd.DataFrame:
    ...

def simulate_link_states(G: nx.Graph, node_log: pd.DataFrame, config: dict) -> pd.DataFrame:
    ...

def simulate_services(
    G: nx.Graph,
    service_trace: pd.DataFrame,
    node_log: pd.DataFrame,
    link_log: pd.DataFrame,
    config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ...

def main():
    ...
```

拓扑要求：

```text
1 cloud node: cloud_0
num_edge_nodes edge nodes: edge_0 ... edge_n
num_access_nodes access nodes: access_0 ...
num_users user nodes: user_0 ...
```

边连接规则：

```text
user_i -- access_{i % num_access_nodes}
access_j -- several edge nodes
edge_i -- cloud_0
edge_i -- edge_{i+1} 可选，形成 small_world 近似结构
```

服务路径：

```text
user_id → access_node → current_edge → cloud_0
```

服务部署：

```text
current_edge = edge_{service_id % num_edge_nodes}
user_id = user_{service_id % num_users}
```

响应时间建议公式：

```text
response_time_ms =
    base_response_time_ms
    + 0.4 * path_delay_ms
    + 80 * current_edge_cpu
    + 2.0 * current_edge_queue_len
    + noise
```

链路状态建议公式：

```text
delay_ms = base_delay * (1 + bandwidth_util) + noise
loss = base_loss + 0.08 * max(0, bandwidth_util - 0.75) + noise
queue_delay_ms = delay_ms * bandwidth_util
```

必须保证：

```text
1. 至少一部分样本 normal；
2. 至少一部分样本 risky；
3. 至少一部分样本 violated；
4. label_stats 中三类都不能为 0。
```

如果初次生成后某类为 0，需要调整 synthetic 公式或 SLA 阈值，而不是改标签规则。

---

## 10.6 `src/preprocessing/build_path_graph.py`

职责：把 raw logs 转成固定 shape 的 service-path 样本。

输入：

```powershell
python src/preprocessing/build_path_graph.py --config configs/sparta_debug.yaml
```

输出：

```text
data/debug/processed/path_graphs.pkl
```

必须实现：

```python
def load_logs(raw_dir: str) -> dict:
    ...

def parse_path_nodes(path_str: str) -> list[str]:
    ...

def parse_path_links(path_str: str) -> list[tuple[str, str]]:
    ...

def build_node_features(
    path_nodes: list[str],
    time_window: list[int],
    logs: dict,
    config: dict
) -> tuple[np.ndarray, np.ndarray]:
    ...

def build_link_features(
    path_links: list[tuple[str, str]],
    time_window: list[int],
    logs: dict,
    config: dict
) -> tuple[np.ndarray, np.ndarray]:
    ...

def build_service_features(
    service_id: int,
    time_window: list[int],
    logs: dict,
    config: dict
) -> np.ndarray:
    ...

def build_sla_features(
    service_id: int,
    logs: dict,
    config: dict
) -> np.ndarray:
    ...

def build_adj_matrix(
    local_nodes: list[str],
    local_links: list[tuple[str, str]],
    config: dict
) -> np.ndarray:
    ...

def build_samples(logs: dict, config: dict) -> list[dict]:
    ...
```

每个 sample 必须包含：

```python
{
    "sample_id": str,
    "service_id": int,
    "time": int,
    "node_ids": list[str],
    "link_ids": list[str],
    "node_x": np.ndarray,      # [L, max_nodes, 10]
    "link_x": np.ndarray,      # [L, max_links, 8]
    "service_x": np.ndarray,   # [L, 6]
    "sla_x": np.ndarray,       # [5]
    "adj": np.ndarray,         # [max_nodes, max_nodes]
    "node_mask": np.ndarray,   # [max_nodes]
    "link_mask": np.ndarray    # [max_links]
}
```

必须在保存前检查 shape：

```python
assert sample["node_x"].shape == (L, max_nodes, 10)
assert sample["link_x"].shape == (L, max_links, 8)
assert sample["service_x"].shape == (L, 6)
assert sample["sla_x"].shape == (5,)
assert sample["adj"].shape == (max_nodes, max_nodes)
```

---

## 10.7 `src/preprocessing/generate_labels.py`

职责：根据 raw logs 和 `path_graphs.pkl` 生成风险标签和归因标签。

输入：

```powershell
python src/preprocessing/generate_labels.py --config configs/sparta_debug.yaml
```

输出：

```text
data/debug/processed/samples_labeled.pkl
```

必须实现：

```python
def assign_risk_label(service_id: int, time: int, logs: dict, sla_row: dict, horizon: int) -> int:
    ...

def compute_node_risk_score(sample: dict, future_times: list[int], logs: dict, config: dict) -> dict[int, float]:
    ...

def compute_link_risk_score(sample: dict, future_times: list[int], logs: dict, config: dict) -> dict[int, float]:
    ...

def assign_metric_label(sample: dict, future_times: list[int], logs: dict, sla_row: dict, config: dict) -> int:
    ...

def label_samples(samples: list[dict], logs: dict, config: dict) -> list[dict]:
    ...
```

输出 sample 在原字段基础上增加：

```python
risk_label: int
risk_node_label: int
risk_link_label: int
risk_metric_label: int
```

normal 样本必须使用：

```python
risk_node_label = -100
risk_link_label = -100
risk_metric_label = -100
```

---

## 10.8 `src/preprocessing/split_dataset.py`

职责：按时间划分并保存固定 shape 的 `.npz` 数据集。

输入：

```powershell
python src/preprocessing/split_dataset.py --config configs/sparta_debug.yaml
```

输出：

```text
data/debug/dataset/train.npz
data/debug/dataset/val.npz
data/debug/dataset/test.npz
data/debug/dataset/metadata.json
```

必须实现：

```python
def time_based_split(samples: list[dict], train_ratio: float, val_ratio: float):
    ...

def samples_to_arrays(samples: list[dict]) -> dict[str, np.ndarray]:
    ...

def save_npz(arrays: dict[str, np.ndarray], path: str) -> None:
    ...

def build_metadata(train, val, test, config) -> dict:
    ...
```

划分规则：

```text
1. 按 sample["time"] 升序；
2. 以唯一 time 为单位划分；
3. train: 前 60%;
4. val: 中间 20%;
5. test: 最后 20%;
6. 不允许随机打乱时间顺序。
```

`metadata.json` 至少包含：

```json
{
  "num_train": 0,
  "num_val": 0,
  "num_test": 0,
  "label_distribution_train": {},
  "label_distribution_val": {},
  "label_distribution_test": {},
  "shape": {
    "node_x": "...",
    "link_x": "...",
    "service_x": "...",
    "sla_x": "..."
  }
}
```

---

## 10.9 `src/preprocessing/check_dataset_stats.py`

职责：检查最终 `.npz` 数据是否正常。

命令：

```powershell
python src/preprocessing/check_dataset_stats.py --config configs/sparta_debug.yaml
```

必须输出：

```text
train/val/test 样本数
risk_label 分布
risk_node_label 有效比例
risk_link_label 有效比例
risk_metric_label 有效比例
node_x shape/min/max/mean/std
link_x shape/min/max/mean/std
service_x shape/min/max/mean/std
sla_x shape/min/max/mean/std
```

如果三类风险标签中某一类数量为 0，需要打印 warning。

---

## 10.10 `src/datasets/sparta_dataset.py`

职责：读取 `.npz` 并返回 PyTorch tensor。

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
            "node_mask": ...,
            "link_mask": ...,
            "risk_label": ...,
            "risk_node_label": ...,
            "risk_link_label": ...,
            "risk_metric_label": ...,
            "sample_time": ...,
            "service_id": ...
        }
```

数据类型要求：

```python
node_x/link_x/service_x/sla_x/adj: torch.float32
node_mask/link_mask: torch.bool
risk_label/risk_node_label/risk_link_label/risk_metric_label: torch.long
sample_time/service_id: torch.long
```

Dataset 中必须加入断言：

```python
assert node_x.ndim == 3
assert link_x.ndim == 3
assert service_x.ndim == 2
assert sla_x.ndim == 1
assert adj.ndim == 2
```

---

## 10.11 `src/models/sparta.py`

职责：实现 SPARTA 主模型。

必须实现的类：

```python
class MLP(nn.Module):
    ...

class SimpleGCNLayer(nn.Module):
    ...

class TopologyEncoder(nn.Module):
    ...

class SLAGate(nn.Module):
    ...

class SPARTA(nn.Module):
    ...
```

### 10.11.1 SPARTA forward 输入

公共接口：

```python
def forward(self, batch: dict) -> dict:
    ...
```

从 batch 中取：

```python
node_x:    [B, L, N, FN]
link_x:    [B, L, E, FE]
service_x: [B, L, FS]
sla_x:     [B, FC]
adj:       [B, N, N]
node_mask: [B, N]
link_mask: [B, E]
```

debug 下：

```text
node_x:    [B, 4, 6, 10]
link_x:    [B, 4, 8, 8]
service_x: [B, 4, 6]
sla_x:     [B, 5]
adj:       [B, 6, 6]
node_mask: [B, 6]
link_mask: [B, 8]
```

### 10.11.2 SPARTA 内部流程

#### Heterogeneous Feature Embedding

```python
node_h = node_mlp(node_x)          # [B,L,N,D]
link_h = link_mlp(link_x)          # [B,L,E,D]
service_h = service_mlp(service_x) # [B,L,D]
sla_h = sla_mlp(sla_x)             # [B,D]
```

#### Masked pooling

```python
node_ctx = masked_mean(node_h, node_mask, dim=2) # [B,L,D]
link_ctx = masked_mean(link_h, link_mask, dim=2) # [B,L,D]
path_token = node_ctx + link_ctx + service_h     # [B,L,D]
```

#### Temporal Encoder

```python
temp_h = transformer_encoder(path_token) # [B,L,D]
z_temp = temp_h[:, -1, :]                # [B,D]
```

#### Topology Encoder

```python
node_topo_in = node_h.mean(dim=1)                     # [B,N,D]
node_topo_h = topology_encoder(node_topo_in, adj, node_mask) # [B,N,D]
z_topo = masked_mean(node_topo_h, node_mask, dim=1)   # [B,D]
```

#### Fusion

```python
if use_topology:
    z = layer_norm(z_temp + z_topo)
else:
    z = z_temp
```

#### SLA Gate

```python
if use_sla_gate:
    gate = sigmoid(sla_gate(sla_h)) # [B,D]
    z_sla = z * (1 + gate)
else:
    z_sla = z
```

#### Heads

```python
risk_logits = risk_head(z_sla)                    # [B,3]
node_logits = node_head(node_topo_h).squeeze(-1)   # [B,N]
link_attr_h = link_h.mean(dim=1)                   # [B,E,D]
link_logits = link_head(link_attr_h).squeeze(-1)   # [B,E]
metric_logits = metric_head(z_sla)                 # [B,5]
```

mask padding：

```python
node_logits = node_logits.masked_fill(~node_mask, -1e9)
link_logits = link_logits.masked_fill(~link_mask, -1e9)
```

输出：

```python
return {
    "risk_logits": risk_logits,
    "node_logits": node_logits,
    "link_logits": link_logits,
    "metric_logits": metric_logits,
    "z": z_sla,
}
```

### 10.11.3 SPARTA shape assert

`forward()` 开头必须断言：

```python
B, L, N, FN = node_x.shape
B2, L2, E, FE = link_x.shape

assert B2 == B
assert L2 == L
assert service_x.shape == (B, L, self.service_feat_dim)
assert sla_x.shape == (B, self.sla_feat_dim)
assert adj.shape == (B, N, N)
assert node_mask.shape == (B, N)
assert link_mask.shape == (B, E)
assert FN == self.node_feat_dim
assert FE == self.link_feat_dim
```

输出前必须断言：

```python
assert risk_logits.shape == (B, self.num_classes)
assert node_logits.shape == (B, N)
assert link_logits.shape == (B, E)
assert metric_logits.shape == (B, self.num_metrics)
```

---

## 10.12 `src/models/lstm.py`

职责：实现 LSTM baseline。

公共接口：

```python
def forward(self, batch: dict) -> dict:
    ...
```

baseline 输入不是图，而是从 batch 中构造 path-level summary：

```python
node_summary = masked_mean(batch["node_x"], batch["node_mask"], dim=2)  # [B,L,10]
link_summary = masked_mean(batch["link_x"], batch["link_mask"], dim=2)  # [B,L,8]
sla_repeat = batch["sla_x"].unsqueeze(1).repeat(1, L, 1)                # [B,L,5]
x = torch.cat([node_summary, link_summary, batch["service_x"], sla_repeat], dim=-1)
```

输入维度：

```text
10 + 8 + 6 + 5 = 29
```

LSTM 输出：

```python
risk_logits = head(last_hidden) # [B,3]
return {"risk_logits": risk_logits}
```

不输出 attribution。

---

## 10.13 `src/models/transformer.py`

职责：实现 Vanilla Transformer baseline。

公共接口同 LSTM：

```python
def forward(self, batch: dict) -> dict:
    ...
```

输入同样是 29 维 path-level summary，经线性层映射到 `hidden_dim` 后送入 TransformerEncoder：

```python
x = input_proj(path_summary)       # [B,L,D]
h = transformer_encoder(x)         # [B,L,D]
risk_logits = head(h[:, -1, :])    # [B,3]
```

不要使用 topology encoder，不要使用 SLA gate，不要输出 attribution。

---

## 10.14 `src/losses/multitask_loss.py`

职责：实现 SPARTA 和 baseline 共用 loss。

必须实现：

```python
class SPARTALoss(nn.Module):
    def __init__(
        self,
        lambda_node: float = 0.3,
        lambda_link: float = 0.3,
        lambda_metric: float = 0.2,
        ignore_index: int = -100,
        risk_class_weights=None,
    ):
        ...

    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict]:
        ...
```

逻辑：

```python
risk_loss = F.cross_entropy(outputs["risk_logits"], batch["risk_label"], weight=class_weights)

if "node_logits" in outputs:
    node_loss = F.cross_entropy(outputs["node_logits"], batch["risk_node_label"], ignore_index=-100)
else:
    node_loss = zero_tensor

if "link_logits" in outputs:
    link_loss = F.cross_entropy(outputs["link_logits"], batch["risk_link_label"], ignore_index=-100)
else:
    link_loss = zero_tensor

if "metric_logits" in outputs:
    metric_loss = F.cross_entropy(outputs["metric_logits"], batch["risk_metric_label"], ignore_index=-100)
else:
    metric_loss = zero_tensor

loss = risk_loss + lambda_node * node_loss + lambda_link * link_loss + lambda_metric * metric_loss
```

注意：对于 LSTM / Transformer，outputs 中没有 attribution logits，此时归因损失必须为 0，不允许报错。

返回：

```python
return loss, {
    "loss": float(loss.detach().cpu()),
    "risk_loss": float(risk_loss.detach().cpu()),
    "node_loss": float(node_loss.detach().cpu()),
    "link_loss": float(link_loss.detach().cpu()),
    "metric_loss": float(metric_loss.detach().cpu()),
}
```

---

## 10.15 `src/metrics.py`

职责：计算分类和归因指标。

必须实现：

```python
def compute_classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    ...

def compute_attribution_metrics(all_labels: dict, all_preds: dict) -> dict:
    ...
```

分类指标：

```text
accuracy
macro_f1
risk_recall      # class 1 recall
violation_recall # class 2 recall
auc              # 若无法计算 multiclass AUC，返回 nan
```

归因指标只在 `risk_node_label != -100` 的样本上计算：

```text
node_attr_acc
link_attr_acc
metric_attr_acc
```

对于 baseline：

```text
node_attr_acc = nan
link_attr_acc = nan
metric_attr_acc = nan
```

---

## 10.16 `src/trainer.py`

职责：训练与验证循环。

必须实现：

```python
def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    ...

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, config) -> dict:
    ...

@torch.no_grad()
def validate(model, dataloader, loss_fn, device, config) -> dict:
    ...

def fit(model, train_loader, val_loader, loss_fn, optimizer, config, model_name: str) -> dict:
    ...
```

训练流程：

```python
model.train()
for batch in train_loader:
    batch = move_batch_to_device(batch, device)
    outputs = model(batch)
    loss, loss_dict = loss_fn(outputs, batch)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
```

保存 best：

```text
best_metric = val_macro_f1
checkpoint path = checkpoints/debug/{model_name}_best.pth
```

checkpoint 内容：

```python
{
    "model_state_dict": model.state_dict(),
    "config": config,
    "model_name": model_name,
    "best_metric": best_metric,
    "epoch": epoch,
}
```

每个 epoch 记录到：

```text
logs/debug/train_log_{model_name}.csv
```

列名：

```csv
epoch,train_loss,val_loss,val_accuracy,val_macro_f1,val_risk_recall,val_violation_recall,val_node_attr_acc,val_link_attr_acc,val_metric_attr_acc
```

---

## 10.17 `src/train.py`

职责：根据 config 和 model 名训练模型。

命令：

```powershell
python src/train.py --config configs/sparta_debug.yaml --model sparta
python src/train.py --config configs/sparta_debug.yaml --model lstm
python src/train.py --config configs/sparta_debug.yaml --model transformer
```

必须实现：

```python
def parse_args():
    ...

def build_model(config: dict, model_name: str):
    ...

def build_dataloaders(config: dict):
    ...

def main():
    ...
```

路径规则：

```python
train_path = Path(config["data"]["dataset_dir"]) / "train.npz"
val_path = Path(config["data"]["dataset_dir"]) / "val.npz"
```

device 规则：

```python
if config["train"]["device"] == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 10.18 `src/evaluate.py`

职责：加载 checkpoint，在 test set 上评估。

命令：

```powershell
python src/evaluate.py --config configs/sparta_debug.yaml --model sparta --checkpoint checkpoints/debug/sparta_best.pth
```

也要支持不传 checkpoint 时自动寻找：

```text
checkpoints/debug/{model_name}_best.pth
```

输出：

```text
results/debug/{model_name}_test_results.csv
```

CSV 列：

```csv
method,accuracy,macro_f1,risk_recall,violation_recall,auc,node_attr_acc,link_attr_acc,metric_attr_acc,inference_time_ms
```

推理时间：

```text
debug 阶段可用 time.perf_counter 统计整体平均 ms/sample。
```

---

## 10.19 `src/run_debug_pipeline.py`

职责：Windows 下一键跑通完整流程。

命令：

```powershell
python src/run_debug_pipeline.py --config configs/sparta_debug.yaml
```

内部步骤：

```python
steps = [
    ["python", "src/simulation/make_debug_trace.py", "--config", config_path],
    ["python", "src/simulation/run_synthetic_simulation.py", "--config", config_path],
    ["python", "src/preprocessing/build_path_graph.py", "--config", config_path],
    ["python", "src/preprocessing/generate_labels.py", "--config", config_path],
    ["python", "src/preprocessing/split_dataset.py", "--config", config_path],
    ["python", "src/preprocessing/check_dataset_stats.py", "--config", config_path],
    ["python", "src/train.py", "--config", config_path, "--model", "lstm"],
    ["python", "src/train.py", "--config", config_path, "--model", "transformer"],
    ["python", "src/train.py", "--config", config_path, "--model", "sparta"],
    ["python", "src/evaluate.py", "--config", config_path, "--model", "lstm"],
    ["python", "src/evaluate.py", "--config", config_path, "--model", "transformer"],
    ["python", "src/evaluate.py", "--config", config_path, "--model", "sparta"],
]
```

必须使用：

```python
subprocess.run(step, check=True)
```

不要使用 `.sh`，不要依赖 Linux shell。

---

## 11. 测试文件

## 11.1 `tests/test_dataset_shape.py`

目标：确认 Dataset 输出 shape。

检查：

```python
dataset = SPARTADataset("data/debug/dataset/train.npz")
sample = dataset[0]

assert sample["node_x"].shape == (4, 6, 10)
assert sample["link_x"].shape == (4, 8, 8)
assert sample["service_x"].shape == (4, 6)
assert sample["sla_x"].shape == (5,)
assert sample["adj"].shape == (6, 6)
assert sample["node_mask"].shape == (6,)
assert sample["link_mask"].shape == (8,)
```

---

## 11.2 `tests/test_model_forward.py`

目标：确认 SPARTA forward 输出 shape。

检查：

```python
batch = next(iter(loader))
model = SPARTA(config)
outputs = model(batch)

assert outputs["risk_logits"].shape == (B, 3)
assert outputs["node_logits"].shape == (B, 6)
assert outputs["link_logits"].shape == (B, 8)
assert outputs["metric_logits"].shape == (B, 5)
```

---

## 11.3 `tests/test_loss_backward.py`

目标：确认 loss 可以 backward。

检查：

```python
outputs = model(batch)
loss, loss_dict = loss_fn(outputs, batch)
assert torch.isfinite(loss)
loss.backward()
```

至少一个可训练参数必须有非 None 梯度。

---

## 11.4 `tests/test_train_one_batch.py`

目标：确认一个 batch 可完成训练。

检查：

```python
outputs = model(batch)
loss, _ = loss_fn(outputs, batch)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

不要求 loss 明显下降，只要求没有报错和 NaN。

---

## 12. README.md 最小内容

Codex 需要生成一个简短 README，包含：

```markdown
# SPARTA Windows MVP

## Setup
conda create -n sparta python=3.10 -y
conda activate sparta
pip install -r requirements.txt

## Run debug pipeline
python src/run_debug_pipeline.py --config configs/sparta_debug.yaml

## Run tests
pytest tests

## Expected outputs
- train/val/test npz
- checkpoints/debug/sparta_best.pth
- results/debug/sparta_test_results.csv
```

---

## 13. Codex 实现顺序

为了降低失败概率，请 Codex 按以下顺序写代码，不要一口气堆所有复杂逻辑。

### 第 1 批：基础框架

```text
requirements.txt
configs/sparta_debug.yaml
src/utils/config.py
src/utils/seed.py
src/utils/io.py
```

验收：

```powershell
python -c "from src.utils.config import load_config; print(load_config('configs/sparta_debug.yaml')['project']['name'])"
```

### 第 2 批：synthetic 数据

```text
src/simulation/make_debug_trace.py
src/simulation/run_synthetic_simulation.py
```

验收：

```powershell
python src/simulation/make_debug_trace.py --config configs/sparta_debug.yaml
python src/simulation/run_synthetic_simulation.py --config configs/sparta_debug.yaml
```

检查 7 个 CSV 是否存在。

### 第 3 批：样本构造与划分

```text
src/preprocessing/build_path_graph.py
src/preprocessing/generate_labels.py
src/preprocessing/split_dataset.py
src/preprocessing/check_dataset_stats.py
```

验收：

```powershell
python src/preprocessing/build_path_graph.py --config configs/sparta_debug.yaml
python src/preprocessing/generate_labels.py --config configs/sparta_debug.yaml
python src/preprocessing/split_dataset.py --config configs/sparta_debug.yaml
python src/preprocessing/check_dataset_stats.py --config configs/sparta_debug.yaml
```

检查 `train.npz/val.npz/test.npz` 是否存在，且三类标签都有样本。

### 第 4 批：Dataset、模型、loss

```text
src/datasets/sparta_dataset.py
src/models/sparta.py
src/models/lstm.py
src/models/transformer.py
src/losses/multitask_loss.py
```

验收：

```powershell
pytest tests/test_dataset_shape.py
pytest tests/test_model_forward.py
pytest tests/test_loss_backward.py
```

### 第 5 批：训练、评估、一键流程

```text
src/metrics.py
src/trainer.py
src/train.py
src/evaluate.py
src/run_debug_pipeline.py
```

验收：

```powershell
python src/run_debug_pipeline.py --config configs/sparta_debug.yaml
```

---

## 14. 最终验收标准

MVP 完成后，必须逐项检查：

```text
[ ] python src/run_debug_pipeline.py --config configs/sparta_debug.yaml 能跑完
[ ] data/debug/dataset/train.npz 存在
[ ] data/debug/dataset/val.npz 存在
[ ] data/debug/dataset/test.npz 存在
[ ] checkpoints/debug/lstm_best.pth 存在
[ ] checkpoints/debug/transformer_best.pth 存在
[ ] checkpoints/debug/sparta_best.pth 存在
[ ] results/debug/lstm_test_results.csv 存在
[ ] results/debug/transformer_test_results.csv 存在
[ ] results/debug/sparta_test_results.csv 存在
[ ] SPARTA 输出 risk_logits/node_logits/link_logits/metric_logits
[ ] baseline 输出 risk_logits
[ ] loss 不是 NaN
[ ] normal 样本归因标签为 -100
[ ] padding node/link 的 logits 被 mask 为 -1e9
[ ] pytest tests 全部通过
```

结果 CSV 至少包含：

```csv
method,accuracy,macro_f1,risk_recall,violation_recall,auc,node_attr_acc,link_attr_acc,metric_attr_acc,inference_time_ms
```

---

## 15. 常见错误与禁止事项

### 15.1 禁止混用特征维度

错误：

```text
node_feat_dim 有时为 8，有时为 10
service_feat_dim 有时为 5，有时为 6
```

正确：

```text
本 MVP 固定 node_feat_dim=10, link_feat_dim=8, service_feat_dim=6, sla_feat_dim=5
```

### 15.2 禁止混用标签名

错误：

```python
batch["risk_node"]
batch["risk_node_label"]
```

正确：

```python
batch["risk_node_label"]
```

### 15.3 禁止 normal 样本用 -1 进入 loss

错误：

```python
risk_node_label = -1
```

正确：

```python
risk_node_label = -100
ignore_index = -100
```

### 15.4 禁止随机划分时间序列样本

错误：

```python
train_test_split(samples, shuffle=True)
```

正确：

```text
按 sample_time 升序，以唯一 time 为单位做 60/20/20 划分。
```

### 15.5 禁止一开始接入 EdgeSimPy

MVP 阶段只使用：

```text
run_synthetic_simulation.py
```

后续正式实验再实现：

```text
run_edgesimpy.py
```

### 15.6 禁止让 baseline 使用 attribution

LSTM / Transformer 只做三分类，不做 node/link/metric 归因。

### 15.7 禁止只看 Accuracy 保存模型

保存 best checkpoint 应以：

```text
val_macro_f1
```

为主。debug 阶段也按 macro-F1 保存，避免类别不均衡时 accuracy 虚高。

---

## 16. 给 Codex 的直接提示词

可以把下面这段直接复制给 Codex：

```text
请在当前工作区搭建 SPARTA Windows MVP 项目。严格按照《SPARTA Win11 最小可运行 MVP：Codex 搭建说明书》实现，不要自行更改字段名、特征维度、标签编码或保存格式。

核心目标是在 Windows 11 上运行：
python src/run_debug_pipeline.py --config configs/sparta_debug.yaml

必须使用纯 Python synthetic/debug 数据，不要接入 EdgeSimPy，不要下载 Alibaba/Google trace。最终 train/val/test 使用 npz，固定字段为：
node_x [S,L,max_nodes,10]
link_x [S,L,max_links,8]
service_x [S,L,6]
sla_x [S,5]
adj [S,max_nodes,max_nodes]
node_mask [S,max_nodes]
link_mask [S,max_links]
risk_label [S]
risk_node_label [S]
risk_link_label [S]
risk_metric_label [S]

normal 样本的归因标签必须为 -100，并在 loss 中使用 ignore_index=-100。

SPARTA 公共接口为 outputs = model(batch)，输出必须包含：
risk_logits [B,3]
node_logits [B,max_nodes]
link_logits [B,max_links]
metric_logits [B,5]
z [B,D]

baseline LSTM 和 Transformer 也使用 outputs = model(batch)，但只输出 risk_logits。训练器必须兼容 SPARTA 和 baseline。

请按以下顺序实现：
1. configs 和 utils；
2. synthetic trace 和 synthetic simulation；
3. build_path_graph、generate_labels、split_dataset、check_dataset_stats；
4. Dataset、SPARTA、LSTM、Transformer、loss；
5. metrics、trainer、train、evaluate、run_debug_pipeline；
6. tests。

最终必须生成 checkpoints/debug/sparta_best.pth 和 results/debug/sparta_test_results.csv，并通过 pytest tests。
```

---

## 17. 跑通 MVP 后的下一步

MVP 跑通后，再逐步扩展：

```text
1. 将 debug L=4/H=2 改为 L=12/H=5；
2. 将 num_steps 从 220 扩大到 1000/5000；
3. 加入 GRU、TCN、STGCN baseline；
4. 增加 ablation：w/o SLA、w/o topology、w/o attribution；
5. 再接入 EdgeSimPy；
6. 最后再处理 Alibaba Microservices Trace 2021。
```

不要在 MVP 未跑通前开始真实 trace、复杂仿真和论文大规模实验。
