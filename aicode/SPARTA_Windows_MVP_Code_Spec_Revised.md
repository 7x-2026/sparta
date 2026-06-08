# SPARTA Windows MVP 代码实现规格说明书

> 目标：先在 Windows 本地把 SPARTA 实验代码流程跑通。  
> 原则：先不追求完整 EdgeSimPy + Alibaba trace 的正式实验，而是先用**纯 Python synthetic/debug 数据流**验证“数据生成 → 样本构造 → Dataset → 模型 forward/backward → 训练 → 评估”的完整闭环。  
> 适用对象：你自己、AI/Codex、后续代码助手。  
> 文件风格：尽量使用 `.py` 文件，不依赖 `.sh` 脚本，方便 Windows + PyCharm / VS Code 复现。

---

## 0.1 本修订版最终统一口径

本修订版专门解决 Codex 实现 MVP 时最容易出现的前后不一致问题。后续让 Codex 写代码时，必须以本节为最高优先级。

```text
1. Windows MVP 阶段统一使用 .pkl 文件，不使用 .npz；
2. debug 阶段不调用 normalize.py，所有 synthetic 特征在生成和构图阶段直接裁剪或缩放到合理范围；
3. normalize.py 只作为后续正式数据扩展占位文件，不能插入 debug pipeline；
4. normal 样本归因标签统一为 -100，而不是 -1；
5. attr_mask 仍然保留，normal 样本 attr_mask=0，risky/violated 样本 attr_mask=1；
6. checkpoint 命名统一为 checkpoints/debug/{model}_best.pth；
7. 结果文件命名统一为 results/debug/{model}_test_results.csv；
8. 一键 pipeline 必须训练并评估 lstm、transformer、sparta 三个模型；
9. LSTM 和 Transformer baseline 必须使用同一个 PathTokenEncoder 构造 path_token，不能直接调用 SPARTA 内部私有模块；
10. synthetic 数据必须保证 normal/risky/violated 三类都存在，否则训练前主动报错。
```

本文件的目标不是实现论文级完整系统，而是让 Codex 在 Win11 上搭建一个**确定能训练、能反向传播、能保存权重、能加载评估、能输出结果 CSV**的最小闭环。

---

## 0. 为什么先做 Windows MVP

你当前最重要的任务不是一上来跑正式 Alibaba / Google trace，也不是直接接 EdgeSimPy，而是先保证：

```text
1. 数据格式固定；
2. Dataset 能返回正确张量；
3. SPARTA 模型每一层输入输出 shape 对得上；
4. loss 能计算；
5. backward 能跑；
6. baseline 和 SPARTA 能训练；
7. evaluate 能输出结果 CSV。
```

所以第一阶段建议做一个 **MVP 版本**：

```text
synthetic trace
→ synthetic cloud-edge logs
→ service-path subgraph dataset
→ train/val/test split
→ baseline training
→ SPARTA training
→ evaluation
```

等 MVP 完整跑通后，再替换成：

```text
Alibaba trace / Google trace
→ EdgeSimPy simulation
→ 正式实验数据
```

这样可以大幅降低项目早期失败风险。

---

## 1. Windows 推荐环境

### 1.1 推荐路径

Windows 下建议把项目放在英文路径，避免中文路径和空格：

```text
D:\SPARTA
```

不要放在：

```text
D:\刘圳\我的项目\SPARTA
```

中文路径有时会导致第三方库、日志、pickle 读取出问题。

### 1.2 Conda 环境

PowerShell 或 Anaconda Prompt：

```powershell
conda create -n sparta python=3.10 -y
conda activate sparta
```

安装基础库：

```powershell
pip install numpy pandas scipy scikit-learn networkx matplotlib tqdm pyyaml xgboost einops
```

安装 PyTorch。如果你 Windows 本地有 NVIDIA GPU，可到 PyTorch 官网选择对应 CUDA。先跑通流程时 CPU 也可以：

```powershell
pip install torch torchvision torchaudio
```

如果你想装 CUDA 版，例如 cu121：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

检查环境：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 1.3 Windows 第一阶段不建议强依赖 EdgeSimPy

原因：

1. EdgeSimPy 可能有依赖版本问题；
2. Windows 下某些包安装可能更麻烦；
3. 你当前目标是先跑通 SPARTA 代码，不是验证仿真平台；
4. synthetic simulator 足够产生同格式日志。

因此第一阶段先写：

```text
src/simulation/run_synthetic_simulation.py
```

后续正式实验再写：

```text
src/simulation/run_edgesimpy.py
```

---

## 2. MVP 项目目录结构

Windows 第一阶段建议目录如下：

```text
SPARTA/
├── configs/
│   ├── sparta_debug.yaml
│   ├── sparta.yaml
│   ├── lstm.yaml
│   ├── transformer.yaml
│   └── stgcn.yaml
├── data/
│   ├── debug/
│   │   ├── traces/
│   │   ├── raw_logs/
│   │   ├── processed/
│   │   └── dataset/
│   ├── raw/
│   ├── processed/
│   └── sparta_dataset/
├── src/
│   ├── simulation/
│   │   ├── make_debug_trace.py
│   │   ├── run_synthetic_simulation.py
│   │   └── run_edgesimpy.py
│   ├── preprocessing/
│   │   ├── build_path_graph.py
│   │   ├── generate_labels.py
│   │   ├── split_dataset.py
│   │   ├── normalize.py
│   │   └── check_dataset_stats.py
│   ├── datasets/
│   │   └── sparta_dataset.py
│   ├── models/
│   │   ├── common.py
│   │   ├── sparta.py
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   ├── tcn.py
│   │   ├── transformer.py
│   │   └── stgcn.py
│   ├── losses/
│   │   ├── focal_loss.py
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
│   ├── test_train_one_batch.py
│   ├── test_checkpoint_load_eval.py
│   └── test_run_debug_pipeline.py
├── results/
├── checkpoints/
├── logs/
├── README.md
└── requirements.txt
```

第一阶段可以暂时不创建：

```text
third_party/
simulators/EdgeSimPy/
simulators/YAFS/
```

等 MVP 跑通后再加。

特别说明：`src/preprocessing/normalize.py` 在 MVP 阶段只保留为空间占位或后续扩展文件。debug 数据在生成阶段已经控制到合理范围，因此一键流程不调用 normalize.py。这样可以避免“先 split 还是先 normalize”的流程分歧。后续接真实 trace 时，再单独补充 train-only scaler。

---

## 3. MVP 总流程

### 3.1 一键流程

最终你应该能在 Windows 上运行：

```powershell
python src/run_debug_pipeline.py --config configs/sparta_debug.yaml
```

这个脚本依次执行：

```text
1. make_debug_trace.py
2. run_synthetic_simulation.py
3. build_path_graph.py
4. generate_labels.py
5. split_dataset.py
6. check_dataset_stats.py
7. train.py --model lstm
8. train.py --model transformer
9. train.py --model sparta
10. evaluate.py --model lstm
11. evaluate.py --model transformer
12. evaluate.py --model sparta
```

注意：debug pipeline 不调用 normalize.py。MVP 阶段的特征缩放在 `run_synthetic_simulation.py` 和 `build_path_graph.py` 内完成。

### 3.2 每一步输入输出总览

| 阶段 | 输入 | 输出 |
|---|---|---|
| make_debug_trace.py | 配置文件 | `debug_service_trace.csv`, `debug_node_trace.csv` |
| run_synthetic_simulation.py | debug trace | `node_log.csv`, `link_log.csv`, `service_log.csv`, `path_log.csv`, `sla_log.csv` |
| build_path_graph.py | raw logs | `path_graphs.pkl` |
| generate_labels.py | `path_graphs.pkl` | `samples_labeled.pkl` |
| split_dataset.py | `samples_labeled.pkl` | `train.pkl`, `val.pkl`, `test.pkl`, `metadata.json` |
| train.py | train/val pkl | `checkpoints/debug/{model}_best.pth` + 训练日志 |
| evaluate.py | test pkl + checkpoint | `results/debug/{model}_test_results.csv` |

---

## 4. 配置文件设计

### 4.1 `configs/sparta_debug.yaml`

AI 写代码时优先实现这个配置。

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
  input_window: 4
  pred_horizon: 2
  max_nodes: 6
  max_links: 8
  node_feat_dim: 8
  link_feat_dim: 8
  service_feat_dim: 5
  sla_feat_dim: 5
  num_edge_nodes: 5
  num_access_nodes: 2
  num_users: 20
  num_services: 2
  num_steps: 200
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  train_path: data/sparta_dataset/train.pkl
  val_path: data/sparta_dataset/val.pkl
  test_path: data/sparta_dataset/test.pkl
  metadata_path: data/sparta_dataset/metadata.json
  train_path: data/debug/dataset/train.pkl
  val_path: data/debug/dataset/val.pkl
  test_path: data/debug/dataset/test.pkl
  metadata_path: data/debug/dataset/metadata.json

simulation:
  topology: small_world
  enable_node_overload: true
  enable_link_congestion: true
  enable_burst: true

model:
  name: sparta
  hidden_dim: 32
  temporal_layers: 1
  topology_layers: 1
  n_heads: 2
  dropout: 0.1
  use_sla_gate: true
  use_topology: true
  use_attribution: true

loss:
  risk_loss: ce
  lambda_node: 0.3
  lambda_link: 0.3
  lambda_metric: 0.2
  ignore_attribution_for_normal: true
  ignore_index: -100
  ignore_index: -100

train:
  batch_size: 4
  epochs: 2
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  num_workers: 0
  device: auto
  best_metric: macro_f1
  checkpoint_name: "{model}_best.pth"
  checkpoint_name: "{model}_best.pth"

validation:
  require_all_classes: true
  min_risky_violated_ratio: 0.20
  min_samples_per_class: 5
```

### 4.2 `configs/sparta.yaml`

正式小规模实验可以用：

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
  node_feat_dim: 8
  link_feat_dim: 8
  service_feat_dim: 5
  sla_feat_dim: 5
  num_edge_nodes: 20
  num_access_nodes: 10
  num_users: 300
  num_services: 3
  num_steps: 10000
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2

model:
  name: sparta
  hidden_dim: 128
  temporal_layers: 2
  topology_layers: 2
  n_heads: 4
  dropout: 0.1
  use_sla_gate: true
  use_topology: true
  use_attribution: true

loss:
  risk_loss: focal
  focal_gamma: 2.0
  lambda_node: 0.3
  lambda_link: 0.3
  lambda_metric: 0.2
  ignore_attribution_for_normal: true

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

Windows 下 `num_workers` 建议先设为 `0`，避免 DataLoader 多进程问题。

---

## 5. 数据从初始到训练的完整格式

### 5.1 第一步：debug trace 原始数据

第一阶段不下载真实 Alibaba trace，而是生成两个 synthetic trace。

#### `debug_service_trace.csv`

路径：

```text
data/debug/traces/debug_service_trace.csv
```

字段：

```csv
time,service_id,service_type,request_rate,base_response_time
```

示例：

```csv
time,service_id,service_type,request_rate,base_response_time
0,0,latency,12,40.5
0,1,reliability,8,55.2
1,0,latency,14,43.1
1,1,reliability,10,58.4
```

含义：

| 字段 | 类型 | 含义 |
|---|---|---|
| time | int | 时间片 |
| service_id | int | 服务编号 |
| service_type | str | 服务类型 |
| request_rate | float | 请求到达率 |
| base_response_time | float | 基础响应时延 |

#### `debug_node_trace.csv`

路径：

```text
data/debug/traces/debug_node_trace.csv
```

字段：

```csv
time,node_group,cpu_scale,mem_scale,arrival_scale
```

示例：

```csv
time,node_group,cpu_scale,mem_scale,arrival_scale
0,edge,0.42,0.36,1.0
1,edge,0.47,0.39,1.1
2,edge,0.75,0.62,1.8
```

含义：

| 字段 | 类型 | 含义 |
|---|---|---|
| time | int | 时间片 |
| node_group | str | 节点组，debug 阶段可固定 edge |
| cpu_scale | float | CPU 负载缩放系数 |
| mem_scale | float | memory 负载缩放系数 |
| arrival_scale | float | 请求强度缩放系数 |

### 5.2 第二步：synthetic simulation 输出日志

`run_synthetic_simulation.py` 输入 debug trace，输出 5 个日志。

#### `node_log.csv`

路径：

```text
data/debug/raw_logs/node_log.csv
```

字段：

```csv
time,node_id,node_type,cpu_util,mem_util,queue_len,available_cpu,available_mem
```

要求：

| 字段 | 范围 |
|---|---|
| cpu_util | [0, 1] |
| mem_util | [0, 1] |
| queue_len | >= 0 |
| available_cpu | [0, 1] |
| available_mem | [0, 1] |

#### `link_log.csv`

路径：

```text
data/debug/raw_logs/link_log.csv
```

字段：

```csv
time,src,dst,delay,bandwidth,loss,jitter,queue_delay,bandwidth_util
```

要求：

| 字段 | 范围 |
|---|---|
| delay | > 0 |
| bandwidth | > 0 |
| loss | [0, 1] |
| jitter | >= 0 |
| queue_delay | >= 0 |
| bandwidth_util | [0, 1] |

#### `service_log.csv`

路径：

```text
data/debug/raw_logs/service_log.csv
```

字段：

```csv
time,service_id,user_id,service_type,request_rate,response_time,current_edge,cloud_node
```

#### `path_log.csv`

路径：

```text
data/debug/raw_logs/path_log.csv
```

字段：

```csv
time,service_id,path_nodes,path_links,path_delay,path_loss,bottleneck_node,bottleneck_link
```

其中 `path_nodes` 和 `path_links` 可以先用字符串保存：

```text
0|3|5|cloud
0-3|3-5|5-cloud
```

代码读取时再 split。

#### `sla_log.csv`

路径：

```text
data/debug/raw_logs/sla_log.csv
```

字段：

```csv
service_id,service_type,max_delay,max_loss,min_bandwidth,reliability_req,cost_weight
```

示例：

```csv
service_id,service_type,max_delay,max_loss,min_bandwidth,reliability_req,cost_weight
0,latency,100,0.05,10,0.99,0.2
1,reliability,150,0.02,5,0.999,0.3
```

### 5.3 第三步：path graph 中间数据

`build_path_graph.py` 输出：

```text
data/debug/processed/path_graphs.pkl
```

每个样本是一个 dict：

```python
sample = {
    "service_id": int,
    "time": int,
    "node_features": np.ndarray,     # [L, max_nodes, Fn]
    "link_features": np.ndarray,     # [L, max_links, Fe]
    "service_features": np.ndarray,  # [L, Fs]
    "sla_features": np.ndarray,      # [Fc]
    "adj_matrix": np.ndarray,        # [max_nodes, max_nodes]
    "node_mask": np.ndarray,         # [max_nodes]
    "link_mask": np.ndarray,         # [max_links]
    "node_ids": list,
    "link_ids": list
}
```

默认维度：

```text
L = 4 debug / 12 full
max_nodes = 6 debug / 8 full
max_links = 8 debug / 12 full
Fn = 8
Fe = 8
Fs = 5
Fc = 5
```

### 5.4 第四步：labeled samples

`generate_labels.py` 输出：

```text
data/debug/dataset/samples_labeled.pkl
```

每个 sample 增加：

```python
sample["risk_label"] = int       # 0 normal, 1 risky, 2 violated
sample["risk_node"] = int        # 0 ~ max_nodes-1 or -100
sample["risk_link"] = int        # 0 ~ max_links-1 or -100
sample["risk_metric"] = int      # 0~4 or -100
sample["attr_mask"] = int        # 0 for normal, 1 for risky/violated
```

规定：

```text
normal 样本：risk_node = -1, risk_link = -1, risk_metric = -1, attr_mask = 0
risky/violated 样本：risk_node/link/metric 必须有效，attr_mask = 1
```

### 5.5 第五步：train/val/test

`split_dataset.py` 输出：

```text
data/debug/dataset/train.pkl
data/debug/dataset/val.pkl
data/debug/dataset/test.pkl
data/debug/dataset/metadata.json
```

按时间划分：

```text
train: 前 60%
val: 中间 20%
test: 后 20%
```

不要随机划分。

---

## 6. 特征字典

### 6.1 node_features 固定顺序

`Fn = 8`

```text
0 cpu_util
1 mem_util
2 queue_len_norm
3 available_cpu
4 available_mem
5 node_type_id_norm
6 node_degree_norm
7 is_current_edge
```

说明：

| index | 字段 | 范围 | 说明 |
|---:|---|---|---|
| 0 | cpu_util | [0,1] | CPU 利用率 |
| 1 | mem_util | [0,1] | 内存利用率 |
| 2 | queue_len_norm | [0,1] | 队列长度归一化 |
| 3 | available_cpu | [0,1] | 可用 CPU |
| 4 | available_mem | [0,1] | 可用内存 |
| 5 | node_type_id_norm | [0,1] | access/edge/cloud 类型编码 |
| 6 | node_degree_norm | [0,1] | 节点度归一化 |
| 7 | is_current_edge | 0/1 | 是否当前服务部署边缘节点 |

### 6.2 link_features 固定顺序

`Fe = 8`

```text
0 delay_norm
1 bandwidth_norm
2 loss_norm
3 jitter_norm
4 queue_delay_norm
5 bandwidth_util
6 is_current_path
7 hop_position_norm
```

### 6.3 service_features 固定顺序

`Fs = 5`

```text
0 request_rate_norm
1 response_time_norm
2 service_type_id_norm
3 base_response_time_norm
4 dependency_count_norm
```

debug 阶段如果没有 `dependency_count`，填 0。

### 6.4 sla_features 固定顺序

`Fc = 5`

```text
0 max_delay_norm
1 max_loss_norm
2 min_bandwidth_norm
3 reliability_req_norm
4 cost_weight
```

---

## 7. 各 `.py` 文件具体要写什么

## 7.1 `src/simulation/make_debug_trace.py`

### 作用

生成最小 synthetic trace，不依赖外部数据。

### 输入

```text
configs/sparta_debug.yaml
```

### 输出

```text
data/debug/traces/debug_service_trace.csv
data/debug/traces/debug_node_trace.csv
```

### 必须实现的函数

```python
def make_service_trace(config: dict) -> pd.DataFrame:
    """Generate request_rate and base_response_time for each service at each time."""


def make_node_trace(config: dict) -> pd.DataFrame:
    """Generate cpu_scale, mem_scale, arrival_scale over time."""


def main():
    """Load config, generate traces, save csv files."""
```

### 生成逻辑建议

```text
request_rate = base + sinusoidal fluctuation + random noise + occasional burst
base_response_time = service_type-dependent base delay + noise
cpu_scale = smooth random curve + burst interval
```

debug 阶段要故意制造一些 risky/violated，否则模型没有东西学。生成后必须检查类别比例：

```text
normal/risky/violated 三类都必须存在；
risky + violated 至少占总样本的 20%；
每一类至少有 5 个样本；
如果不满足，run_debug_pipeline.py 应停止并提示降低 max_delay/max_loss 或增强 burst/congestion。
```

---

## 7.2 `src/simulation/run_synthetic_simulation.py`

### 作用

根据 debug trace 生成云边网络日志。

### 输入

```text
data/debug/traces/debug_service_trace.csv
data/debug/traces/debug_node_trace.csv
```

### 输出

```text
data/debug/raw_logs/node_log.csv
data/debug/raw_logs/link_log.csv
data/debug/raw_logs/service_log.csv
data/debug/raw_logs/path_log.csv
data/debug/raw_logs/sla_log.csv
```

### 必须实现的函数

```python
def build_topology(config: dict) -> nx.Graph:
    """Build a small cloud-edge topology."""


def simulate_node_states(G: nx.Graph, node_trace: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate node_log.csv."""


def simulate_link_states(G: nx.Graph, node_log: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate link_log.csv."""


def simulate_services(G: nx.Graph, service_trace: pd.DataFrame, node_log: pd.DataFrame,
                      link_log: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate service_log.csv, path_log.csv, sla_log.csv."""


def main():
    """Run synthetic simulation and save logs."""
```

### 拓扑建议

debug 版本：

```text
1 cloud node
5 edge nodes
2 access nodes
20 users
```

路径示例：

```text
user → access node → edge node → cloud
```

服务默认部署在某个 edge node，备用节点为邻近 edge nodes。

---

## 7.3 `src/preprocessing/build_path_graph.py`

### 作用

把 raw logs 转成固定 shape 的 service-path subgraph 样本。

### 输入

```text
node_log.csv
link_log.csv
service_log.csv
path_log.csv
sla_log.csv
```

### 输出

```text
data/debug/processed/path_graphs.pkl
```

### 必须实现的函数

```python
def load_logs(raw_dir: str) -> dict:
    """Load node/link/service/path/sla logs."""


def extract_service_path(service_id: int, time: int, logs: dict) -> dict:
    """Get path nodes and path links for a service at a given time."""


def build_node_features(path_nodes: list, time_window: list, logs: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return node_features [L, max_nodes, Fn] and node_mask [max_nodes]."""


def build_link_features(path_links: list, time_window: list, logs: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return link_features [L, max_links, Fe] and link_mask [max_links]."""


def build_service_features(service_id: int, time_window: list, logs: dict, config: dict) -> np.ndarray:
    """Return service_features [L, Fs]."""


def build_sla_features(service_id: int, logs: dict, config: dict) -> np.ndarray:
    """Return sla_features [Fc]."""


def build_adj_matrix(path_nodes: list, path_links: list, config: dict) -> np.ndarray:
    """Return adj [max_nodes, max_nodes]."""


def build_samples(logs: dict, config: dict) -> list[dict]:
    """Build all path graph samples."""
```

### 关键规则

1. 每个样本用过去 `L` 个时间片；
2. 节点数不足 `max_nodes` 时 padding；
3. 链路数不足 `max_links` 时 padding；
4. padding 节点/链路的特征全 0；
5. 用 `node_mask` 和 `link_mask` 标记真实位置；
6. 超出 `max_nodes/max_links` 时优先保留当前服务路径上的节点和链路。

---

## 7.4 `src/preprocessing/generate_labels.py`

### 作用

根据未来 `H` 个时间片生成 risk_label 和 attribution labels。

### 输入

```text
path_graphs.pkl
raw logs
```

### 输出

```text
data/debug/dataset/samples_labeled.pkl
```

### 必须实现的函数

```python
def assign_risk_label(service_id: int, time: int, logs: dict, sla: dict, horizon: int) -> int:
    """Return 0 normal, 1 risky, 2 violated."""


def compute_node_risk_score(path_nodes: list, future_window: list, logs: dict) -> dict:
    """Return node_id -> risk score."""


def compute_link_risk_score(path_links: list, future_window: list, logs: dict) -> dict:
    """Return link_id -> risk score."""


def assign_attribution(sample: dict, logs: dict, config: dict) -> tuple[int, int, int, int]:
    """Return risk_node, risk_link, risk_metric, attr_mask."""


def main():
    """Generate labels and save labeled samples."""
```

### 标签规则

```text
violated:
future response_time >= max_delay
or future path_loss >= max_loss

risky:
not violated, but response_time >= 0.75 * max_delay
or path_loss >= 0.75 * max_loss
or CPU >= 0.85
or queue_len increasing
or bandwidth_util >= 0.85

normal:
otherwise
```

### normal 样本规则

```python
if risk_label == 0:
    risk_node = -100
    risk_link = -100
    risk_metric = -100
    attr_mask = 0
else:
    attr_mask = 1
```

---

## 7.5 `src/preprocessing/split_dataset.py`

### 作用

按时间顺序划分 train/val/test。

### 输入

```text
samples_labeled.pkl
```

### 输出

```text
train.pkl
val.pkl
test.pkl
metadata.json
```

### 必须实现的函数

```python
def time_based_split(samples: list[dict], train_ratio: float, val_ratio: float) -> tuple[list, list, list]:
    """Split samples by time, not random shuffle."""


def build_metadata(samples: list[dict], config: dict) -> dict:
    """Build metadata including shape and class distribution."""
```

---

## 7.6 `src/datasets/sparta_dataset.py`

### 作用

PyTorch Dataset，固定输出 batch 数据。

### 返回格式

`__getitem__` 必须返回：

```python
{
    "node_x": torch.FloatTensor,       # [L, max_nodes, Fn]
    "link_x": torch.FloatTensor,       # [L, max_links, Fe]
    "service_x": torch.FloatTensor,    # [L, Fs]
    "sla_x": torch.FloatTensor,        # [Fc]
    "adj": torch.FloatTensor,          # [max_nodes, max_nodes]
    "node_mask": torch.BoolTensor,     # [max_nodes]
    "link_mask": torch.BoolTensor,     # [max_links]
    "risk_label": torch.LongTensor,    # scalar
    "risk_node": torch.LongTensor,     # scalar, -100 for normal
    "risk_link": torch.LongTensor,     # scalar, -100 for normal
    "risk_metric": torch.LongTensor,   # scalar, -100 for normal
    "attr_mask": torch.BoolTensor      # scalar
}
```

### 必须加入断言

```python
assert node_x.shape == (L, max_nodes, Fn)
assert link_x.shape == (L, max_links, Fe)
assert service_x.shape == (L, Fs)
assert sla_x.shape == (Fc,)
assert adj.shape == (max_nodes, max_nodes)
assert node_mask.shape == (max_nodes,)
assert link_mask.shape == (max_links,)
```

---

## 7.6.1 `src/models/common.py`

### 作用

提供 baseline 和 SPARTA 都可以复用的基础模块，尤其是 masked mean pooling 和 baseline 的 path token 构造。这样可以避免 LSTM、Transformer 和 SPARTA 各自定义一套输入处理逻辑。

### 必须实现

```python
def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    """Masked mean pooling. Supports node/link masks."""


class MLP(nn.Module):
    """Generic MLP used by feature projection modules."""


class PathTokenEncoder(nn.Module):
    """Build path_token [B,L,D] for LSTM and Transformer baselines."""
```

### `PathTokenEncoder` 输入输出

输入 batch：

```text
node_x: [B,L,N,Fn]
link_x: [B,L,E,Fe]
service_x: [B,L,Fs]
sla_x: [B,Fc]
node_mask: [B,N]
link_mask: [B,E]
```

输出：

```text
path_token: [B,L,D]
```

### 注意

`PathTokenEncoder` 只用于 baseline 的 path-level token 构造。SPARTA 可以复用 `masked_mean` 和 `MLP`，但 SPARTA 自己仍然要实现 topology encoder、SLA gate 和 attribution heads。

---

## 7.7 `src/models/sparta.py`

### 作用

实现 SPARTA 主模型。

### forward 输入

```python
def forward(self, batch):
    node_x = batch["node_x"]       # [B, L, N, Fn]
    link_x = batch["link_x"]       # [B, L, E, Fe]
    service_x = batch["service_x"] # [B, L, Fs]
    sla_x = batch["sla_x"]         # [B, Fc]
    adj = batch["adj"]             # [B, N, N]
    node_mask = batch["node_mask"] # [B, N]
    link_mask = batch["link_mask"] # [B, E]
```

### forward 输出

```python
{
    "risk_logits": Tensor[B, 3],
    "node_logits": Tensor[B, max_nodes],
    "link_logits": Tensor[B, max_links],
    "metric_logits": Tensor[B, 5]
}
```

### 模型内部 shape

```text
node_emb: [B, L, N, D]
link_emb: [B, L, E, D]
service_emb: [B, L, D]
sla_emb: [B, D]
path_token: [B, L, D]
temporal_out: [B, L, D]
z_time: [B, D]
topo_node: [B, N, D]
z_topo: [B, D]
z: [B, D]
```

### 必须处理 mask

1. node pooling 时忽略 padding node；
2. link pooling 时忽略 padding link；
3. node_logits 对 padding 位置赋值为 `-1e9`；
4. link_logits 对 padding 位置赋值为 `-1e9`。

---

## 7.8 `src/losses/multitask_loss.py`

### 作用

统一计算多任务 loss。

### 输入

```python
outputs = {
    "risk_logits": [B, 3],
    "node_logits": [B, N],
    "link_logits": [B, E],
    "metric_logits": [B, 5]
}

batch = {
    "risk_label": [B],
    "risk_node": [B],
    "risk_link": [B],
    "risk_metric": [B],
    "attr_mask": [B]
}
```

### 逻辑

MVP 阶段使用 `attr_mask` 和 `ignore_index=-100` 双保险。normal 样本不参与 node/link/metric 归因损失。

```python
ignore_index = config["loss"].get("ignore_index", -100)
ce_risk = torch.nn.CrossEntropyLoss()
ce_attr = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

risk_loss = ce_risk(outputs["risk_logits"], batch["risk_label"])

valid = batch["attr_mask"].bool()
if valid.sum() > 0:
    node_loss = ce_attr(outputs["node_logits"][valid], batch["risk_node"][valid])
    link_loss = ce_attr(outputs["link_logits"][valid], batch["risk_link"][valid])
    metric_loss = ce_attr(outputs["metric_logits"][valid], batch["risk_metric"][valid])
else:
    zero = outputs["risk_logits"].sum() * 0.0
    node_loss = zero
    link_loss = zero
    metric_loss = zero

total_loss = risk_loss + lambda_node * node_loss + lambda_link * link_loss + lambda_metric * metric_loss
```

注意：`else` 分支不能直接写 Python 数字 `0`，否则可能造成 device/dtype 不一致，也不利于日志统一。

### 输出

```python
{
    "loss": total_loss,
    "risk_loss": risk_loss,
    "node_loss": node_loss,
    "link_loss": link_loss,
    "metric_loss": metric_loss
}
```

---

## 7.9 `src/metrics.py`

### 必须实现

```python
def compute_classification_metrics(y_true, y_pred, y_score=None) -> dict:
    """Return accuracy, macro_f1, risk_recall, violation_recall, auc."""


def compute_attribution_metrics(batch, outputs) -> dict:
    """Return node_attr_acc, link_attr_acc, metric_attr_acc on attr_mask=1 samples."""
```

### 指标定义

```text
risk_recall = recall for class 1
violation_recall = recall for class 2
node_attr_acc = risk_node prediction accuracy on attr_mask=1
link_attr_acc = risk_link prediction accuracy on attr_mask=1
metric_attr_acc = risk_metric prediction accuracy on attr_mask=1
```

---

## 7.10 `src/trainer.py`

### 必须实现

```python
def train_one_epoch(model, dataloader, optimizer, loss_fn, device, config):
    ...


def validate(model, dataloader, loss_fn, device, config):
    ...


def fit(model, train_loader, val_loader, loss_fn, optimizer, scheduler, config):
    ...
```

### 训练流程

```text
model.train()
for batch in train_loader:
    move batch to device
    outputs = model(batch)
    loss_dict = loss_fn(outputs, batch)
    loss = loss_dict["loss"]
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_
    optimizer.step()
```

### 保存 best

所有模型使用统一 checkpoint 命名：

```text
checkpoints/debug/lstm_best.pth
checkpoints/debug/transformer_best.pth
checkpoints/debug/sparta_best.pth
```

保存内容必须包含：

```python
{
    "model_name": model_name,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "best_metric": best_metric_value,
    "config": config,
}
```

debug 阶段也必须保存 best，而不是只保存 last。选择指标：

```text
best_metric = val_macro_f1
```

---

## 7.11 `src/train.py`

### 作用

根据 config 训练指定模型。

### 命令

```powershell
python src/train.py --config configs/sparta_debug.yaml --model sparta
python src/train.py --config configs/sparta_debug.yaml --model lstm
python src/train.py --config configs/sparta_debug.yaml --model transformer
```

### 需要做的事

1. 读取 config；
2. 设置 seed；
3. 构建 Dataset / DataLoader；
4. 构建 model；
5. 构建 loss；
6. 构建 optimizer；
7. 调用 trainer.fit；
8. 按 `checkpoints/debug/{model}_best.pth` 保存 checkpoint；
9. 保存训练指标到 `results/debug/{model}_train_log.csv`。

训练脚本必须支持以下模型名：

```text
lstm
transformer
sparta
```

如果传入其他模型名，必须抛出清晰错误，不要静默 fallback。

---

## 7.12 `src/evaluate.py`

### 作用

加载 checkpoint，在 test set 上评估。

### 命令

```powershell
python src/evaluate.py --config configs/sparta_debug.yaml --model lstm --checkpoint checkpoints/debug/lstm_best.pth
python src/evaluate.py --config configs/sparta_debug.yaml --model transformer --checkpoint checkpoints/debug/transformer_best.pth
python src/evaluate.py --config configs/sparta_debug.yaml --model sparta --checkpoint checkpoints/debug/sparta_best.pth
```

### 输出

```text
results/debug/lstm_test_results.csv
results/debug/transformer_test_results.csv
results/debug/sparta_test_results.csv
```

列名：

```csv
method,accuracy,macro_f1,risk_recall,violation_recall,auc,node_attr_acc,link_attr_acc,metric_attr_acc,inference_time_ms
```

---

## 7.13 `src/run_debug_pipeline.py`

### 作用

Windows 下一键跑通最小流程。

### 命令

```powershell
python src/run_debug_pipeline.py --config configs/sparta_debug.yaml
```

### 内部顺序

```python
steps = [
    "python src/simulation/make_debug_trace.py --config configs/sparta_debug.yaml",
    "python src/simulation/run_synthetic_simulation.py --config configs/sparta_debug.yaml",
    "python src/preprocessing/build_path_graph.py --config configs/sparta_debug.yaml",
    "python src/preprocessing/generate_labels.py --config configs/sparta_debug.yaml",
    "python src/preprocessing/split_dataset.py --config configs/sparta_debug.yaml",
    "python src/preprocessing/check_dataset_stats.py --config configs/sparta_debug.yaml",
    "python src/train.py --config configs/sparta_debug.yaml --model lstm",
    "python src/train.py --config configs/sparta_debug.yaml --model transformer",
    "python src/train.py --config configs/sparta_debug.yaml --model sparta",
    "python src/evaluate.py --config configs/sparta_debug.yaml --model lstm --checkpoint checkpoints/debug/lstm_best.pth",
    "python src/evaluate.py --config configs/sparta_debug.yaml --model transformer --checkpoint checkpoints/debug/transformer_best.pth",
    "python src/evaluate.py --config configs/sparta_debug.yaml --model sparta --checkpoint checkpoints/debug/sparta_best.pth"
]
```

Windows 下建议用 `subprocess.run()`，不要用 shell 脚本。

---

## 8. Baseline MVP 要求

第一阶段不需要把所有 baseline 都写完，建议先写 3 个：

```text
1. LSTM
2. Vanilla Transformer
3. SPARTA
```

跑通后再加：

```text
4. GRU
5. TCN
6. STGCN
7. PatchTST
8. XGBoost / RandomForest
```

### 8.1 baseline 共享 `PathTokenEncoder`

LSTM 和 Vanilla Transformer baseline 不能直接复用 `SPARTA` 类内部的私有 embedding，也不能让每个 baseline 自己随意 flatten 输入。必须在 `src/models/common.py` 中实现一个共享模块：

```python
class PathTokenEncoder(nn.Module):
    def __init__(self, node_feat_dim, link_feat_dim, service_feat_dim, sla_feat_dim, hidden_dim):
        ...

    def forward(self, batch):
        # node_x: [B,L,N,Fn]
        # link_x: [B,L,E,Fe]
        # service_x: [B,L,Fs]
        # sla_x: [B,Fc]
        # node_mask: [B,N]
        # link_mask: [B,E]
        # return path_token: [B,L,D]
        ...
```

推荐构造方式：

```python
node_pool = masked_mean(node_x, node_mask, dim=2)       # [B,L,Fn]
link_pool = masked_mean(link_x, link_mask, dim=2)       # [B,L,Fe]
sla_expand = sla_x[:, None, :].expand(-1, L, -1)        # [B,L,Fc]
raw = torch.cat([node_pool, link_pool, service_x, sla_expand], dim=-1)
path_token = token_mlp(raw)                             # [B,L,D]
```

这样 baseline 与 SPARTA 使用同一批输入特征，但不使用 topology encoder、SLA gate 和 attribution heads，比较关系更清楚。

### 8.2 LSTM 输入

```text
path_token: [B, L, D]
```

输出：

```text
risk_logits: [B, 3]
```

LSTM 不做 attribution。

### 8.3 Transformer 输入

```text
path_token: [B, L, D]
```

输出：

```text
risk_logits: [B, 3]
```

Transformer 不使用 topology，不使用 SLA gate，不做 attribution。

### 8.4 SPARTA 输入

完整 batch。

输出：

```text
risk_logits, node_logits, link_logits, metric_logits
```

---

## 9. 测试文件

## 9.1 `tests/test_dataset_shape.py`

目标：确认 Dataset 输出 shape。

伪代码：

```python
def test_dataset_shape():
    dataset = SPARTADataset("data/debug/dataset/train.pkl", config)
    sample = dataset[0]
    assert sample["node_x"].shape == (L, max_nodes, Fn)
    assert sample["link_x"].shape == (L, max_links, Fe)
    assert sample["service_x"].shape == (L, Fs)
    assert sample["sla_x"].shape == (Fc,)
```

## 9.2 `tests/test_model_forward.py`

目标：确认 SPARTA forward 输出 shape。

```python
def test_model_forward():
    batch = make_fake_batch(B=2)
    model = SPARTA(config)
    outputs = model(batch)
    assert outputs["risk_logits"].shape == (2, 3)
    assert outputs["node_logits"].shape == (2, max_nodes)
    assert outputs["link_logits"].shape == (2, max_links)
    assert outputs["metric_logits"].shape == (2, 5)
```

## 9.3 `tests/test_loss_backward.py`

目标：确认 loss 能 backward。

```python
def test_loss_backward():
    batch = make_fake_batch(B=2)
    model = SPARTA(config)
    outputs = model(batch)
    loss_dict = loss_fn(outputs, batch)
    loss = loss_dict["loss"]
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
```

## 9.4 `tests/test_train_one_batch.py`

目标：确认一个 batch 能训练。

```python
def test_train_one_batch():
    batch = next(iter(train_loader))
    outputs = model(batch)
    loss = loss_fn(outputs, batch)["loss"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 9.5 `tests/test_checkpoint_load_eval.py`

目标：确认训练保存的 checkpoint 可以被 evaluate.py 正确加载。

```python
def test_checkpoint_load_eval():
    # 允许使用很小的数据和 1 个 epoch
    # 训练 sparta 后保存 checkpoints/debug/sparta_best.pth
    # 再加载 checkpoint，跑一次 test loader
    assert Path("checkpoints/debug/sparta_best.pth").exists()
    metrics = evaluate_once(...)
    assert "macro_f1" in metrics
```

这个测试用于防止“训练能跑，但评估阶段因为 checkpoint key、model_name 或 config 不一致而加载失败”。

## 9.6 `tests/test_run_debug_pipeline.py`

目标：确认一键 pipeline 至少能在最小配置下完整执行。为了让测试速度可控，可以覆盖配置：`num_steps=40, epochs=1, batch_size=2`。

```python
def test_run_debug_pipeline():
    result = subprocess.run(
        ["python", "src/run_debug_pipeline.py", "--config", "configs/sparta_debug.yaml", "--fast_dev_run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert Path("results/debug/sparta_test_results.csv").exists()
```

---

## 10. 结果文件最小要求

即使只跑 debug，也要输出：

```text
results/debug/lstm_test_results.csv
results/debug/transformer_test_results.csv
results/debug/sparta_test_results.csv
```

每个 CSV 至少包含：

```csv
method,accuracy,macro_f1,risk_recall,violation_recall,auc,node_attr_acc,link_attr_acc,metric_attr_acc,inference_time_ms
```

同时建议额外生成一个汇总文件：

```text
results/debug/all_test_results.csv
```

其中每一行对应一个模型，便于直接比较。对于 LSTM / Transformer，归因指标填 `nan`，不能填 0，因为 0 会被误解为模型归因能力很差，而实际上 baseline 根本没有归因输出。

---

## 11. 最小跑通验收标准

当以下条件全部满足，说明 Windows MVP 跑通：

```text
1. python src/run_debug_pipeline.py --config configs/sparta_debug.yaml 能完整运行；
2. data/debug/dataset/train.pkl、val.pkl、test.pkl 存在；
3. test_dataset_shape.py 通过；
4. test_model_forward.py 通过；
5. test_loss_backward.py 通过；
6. LSTM 能训练并输出 test_results.csv；
7. Transformer 能训练并输出 test_results.csv；
8. SPARTA 能训练并输出 test_results.csv；
9. SPARTA 输出 risk_logits/node_logits/link_logits/metric_logits；
10. checkpoints/debug/lstm_best.pth、transformer_best.pth、sparta_best.pth 均存在；
11. results/debug/lstm_test_results.csv、transformer_test_results.csv、sparta_test_results.csv 均存在；
12. test_checkpoint_load_eval.py 通过；
13. test_run_debug_pipeline.py 通过；
14. metadata.json 中 normal/risky/violated 三类均存在，且 risky+violated 占比不低于 20%。
```

这个阶段不要求 SPARTA 性能最好，只要求流程正确。

---

## 12. 跑通后再做什么

MVP 跑通后，再按以下顺序扩展：

```text
1. 把 L=4, H=2 改为 L=12, H=5；
2. 把 num_steps=200 改为 10000；
3. 增加 edge nodes / users / services；
4. 接入真实 Alibaba trace；
5. 替换 synthetic simulation 为 EdgeSimPy；
6. 增加更多 baseline；
7. 增加 ablation；
8. 增加 proactive intervention；
9. 生成论文结果表。
```

不要一开始就做第 5–9 步。先保证第 1 个 debug pipeline 能跑通。

---

## 13. 给 AI/Codex 的总代码任务提示词

可以直接把下面这段给 AI/Codex：

```text
请你根据 SPARTA Windows MVP 代码实现规格，在 Windows 友好的 Python 项目中实现一个最小可运行版本。不要使用 shell 脚本，尽量使用 .py 文件和 argparse。

项目目标是跑通以下流程：
make_debug_trace.py → run_synthetic_simulation.py → build_path_graph.py → generate_labels.py → split_dataset.py → check_dataset_stats.py → train.py(lstm/transformer/sparta) → evaluate.py(lstm/transformer/sparta)。

MVP 阶段统一使用 .pkl 文件，不使用 .npz；debug pipeline 不调用 normalize.py。

请先实现 debug 版本，不接入真实 Alibaba/Google trace，也不接 EdgeSimPy。使用 synthetic trace 生成 node_log.csv、link_log.csv、service_log.csv、path_log.csv、sla_log.csv。然后构造 service-path subgraph 样本，固定 padding 到 max_nodes 和 max_links，输出 train.pkl、val.pkl、test.pkl。

Dataset 每个样本必须返回：node_x [L,max_nodes,Fn], link_x [L,max_links,Fe], service_x [L,Fs], sla_x [Fc], adj [max_nodes,max_nodes], node_mask [max_nodes], link_mask [max_links], risk_label, risk_node, risk_link, risk_metric, attr_mask。normal 样本的 risk_node/risk_link/risk_metric 必须统一为 -100，attr_mask=0。

请实现 LSTM、Vanilla Transformer 和 SPARTA 三个模型。LSTM 和 Transformer 必须通过 src/models/common.py 中的 PathTokenEncoder 构造 path_token，不允许直接调用 SPARTA 内部模块。SPARTA forward 输出 risk_logits [B,3], node_logits [B,max_nodes], link_logits [B,max_links], metric_logits [B,5]。

请实现 multi-task loss：risk classification loss + attribution loss。normal 样本 attr_mask=0，不参与 node/link/metric attribution loss，同时 CrossEntropyLoss 使用 ignore_index=-100。

请实现 run_debug_pipeline.py，使其在 Windows 下可以一键运行整个流程。最后必须输出 results/debug/lstm_test_results.csv、results/debug/transformer_test_results.csv、results/debug/sparta_test_results.csv，并保存 checkpoints/debug/{model}_best.pth。

代码中必须加入 shape assert，并实现 tests/test_dataset_shape.py、tests/test_model_forward.py、tests/test_loss_backward.py、tests/test_train_one_batch.py、tests/test_checkpoint_load_eval.py、tests/test_run_debug_pipeline.py。
```

---

## 14. 最终建议

你现在不要马上要求 AI 写完整论文级代码。应该先让 AI 按这个 MVP 文件写出能跑通的最小版本。

第一阶段目标只有一个：

```text
Windows 本地一键跑通 debug pipeline。
```

只要这个跑通，后续接 Alibaba trace、EdgeSimPy、更多 baseline、消融实验就不会乱。
