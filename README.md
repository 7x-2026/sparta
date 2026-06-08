# SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection

> 本项目用于支撑 ICASSP 第一篇论文：  
> **SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services**  
> 中文题目：**面向云边服务的服务路径感知 SLA 风险早期检测与归因方法**

---

## 0. 项目定位

本项目不是传统的“边缘计算任务卸载”项目，也不是简单的“Transformer 做三分类”项目。

本项目的核心目标是：

> 将云边网络状态建模为 **service-path network signal**，针对每个云边服务构造服务路径子图，预测未来窗口内该服务是否进入 `normal / risky / violated` 状态，并进一步归因风险来源节点、风险链路和风险主导指标。

模型最终输出五类结果：

1. **risk label**：`normal / risky / violated`；
2. **risk probability**：三类风险概率；
3. **risk node**：最可能导致风险的节点；
4. **risk link**：最可能导致风险的链路；
5. **risk metric**：主导风险指标，例如 delay、loss、CPU、queue、bandwidth。

项目最终服务于 ICASSP 论文中的三个大创新点：

| 创新点 | 项目中对应模块 |
|---|---|
| 服务路径感知的网络信号建模 | `src/preprocessing/build_path_graph.py` |
| SLA 条件驱动的时空拓扑风险建模 | `src/models/sparta.py` |
| 面向主动干预的轻量风险归因与验证 | `src/evaluate.py`、`src/visualize.py`、`results/proactive_results.csv` |

---

## 1. 总体项目结构

推荐项目目录如下：

```text
SPARTA/
├── README.md
├── environment.yml
├── requirements.txt
├── configs/
│   ├── sparta.yaml
│   ├── lstm.yaml
│   ├── gru.yaml
│   ├── tcn.yaml
│   ├── transformer.yaml
│   ├── patchtst.yaml
│   └── stgcn.yaml
├── data/
│   ├── raw/
│   ├── traces/
│   │   ├── alibaba/
│   │   └── google/
│   ├── processed/
│   └── sparta_dataset/
├── simulators/
│   ├── EdgeSimPy/
│   └── YAFS/
├── third_party/
│   ├── Time-Series-Library/
│   ├── PatchTST/
│   └── STGCN-PyTorch/
├── src/
│   ├── simulation/
│   │   ├── run_edgesimpy.py
│   │   ├── inject_trace.py
│   │   └── export_logs.py
│   ├── preprocessing/
│   │   ├── prepare_alibaba_trace.py
│   │   ├── prepare_google_trace.py
│   │   ├── build_path_graph.py
│   │   ├── generate_labels.py
│   │   ├── generate_attribution.py
│   │   ├── normalize.py
│   │   └── split_dataset.py
│   ├── datasets/
│   │   └── sparta_dataset.py
│   ├── models/
│   │   ├── sparta.py
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   ├── tcn.py
│   │   ├── transformer.py
│   │   ├── patchtst_cls.py
│   │   └── stgcn_cls.py
│   ├── losses/
│   │   ├── focal_loss.py
│   │   └── multitask_loss.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── scripts/
│   ├── 00_setup.sh
│   ├── 01_download_code.sh
│   ├── 02_prepare_traces.sh
│   ├── 03_run_simulation.sh
│   ├── 04_build_dataset.sh
│   ├── 05_train_baselines.sh
│   ├── 06_train_sparta.sh
│   └── 07_evaluate_all.sh
├── results/
├── figures/
├── logs/
└── checkpoints/
```

---

## 2. 根目录文件说明

### 2.1 `README.md`

当前文件。作用是说明整个项目的研究目标、目录结构、数据流、训练流、每个代码文件职责，以及完整运行顺序。

### 2.2 `environment.yml`

用于创建 conda 环境。建议内容：

```yaml
name: sparta
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - networkx
  - matplotlib
  - tqdm
  - pyyaml
  - pytorch
  - pytorch-cuda=12.1
  - pip
  - pip:
      - xgboost
      - einops
```

创建环境：

```bash
conda env create -f environment.yml
conda activate sparta
```

### 2.3 `requirements.txt`

如果不使用 conda，也可以使用 pip：

```txt
numpy
pandas
scipy
scikit-learn
networkx
matplotlib
tqdm
pyyaml
xgboost
einops
torch
torchvision
torchaudio
```

安装：

```bash
pip install -r requirements.txt
```

---

## 3. `configs/`：实验配置目录

该目录存放所有模型和实验配置。不要把参数写死在代码里，所有可变参数都应写入 `.yaml`。

```text
configs/
├── sparta.yaml
├── lstm.yaml
├── gru.yaml
├── tcn.yaml
├── transformer.yaml
├── patchtst.yaml
└── stgcn.yaml
```

### 3.1 `configs/sparta.yaml`

SPARTA 主模型配置。负责定义数据路径、模型结构、损失权重、训练参数。

推荐内容：

```yaml
experiment:
  name: sparta_w12_h5
  seed: 42
  output_dir: results/sparta_w12_h5

data:
  dataset_path: data/sparta_dataset/sparta_w12_h5.pkl
  train_split: data/sparta_dataset/train.pkl
  val_split: data/sparta_dataset/val.pkl
  test_split: data/sparta_dataset/test.pkl
  window: 12
  horizon: 5
  num_classes: 3
  num_metrics: 5

model:
  name: sparta
  hidden_dim: 128
  num_layers: 2
  num_heads: 4
  dropout: 0.1
  use_sla_gate: true
  use_topology: true
  use_attribution: true
  topology_type: gcn

loss:
  type: multitask
  risk_loss: weighted_ce
  lambda_node: 0.3
  lambda_link: 0.3
  lambda_metric: 0.2
  class_weights: [1.0, 2.0, 3.0]

train:
  epochs: 80
  batch_size: 64
  lr: 0.0005
  weight_decay: 0.0001
  early_stop_patience: 12
  device: cuda
```

与其他文件的关系：

```text
configs/sparta.yaml
        ↓
src/train.py
        ↓
src/datasets/sparta_dataset.py
        ↓
src/models/sparta.py
        ↓
src/losses/multitask_loss.py
        ↓
checkpoints/sparta_best.pth
```

### 3.2 `configs/lstm.yaml`

LSTM baseline 配置。

应包含：

```yaml
model:
  name: lstm
  input_dim: 128
  hidden_dim: 128
  num_layers: 2
  dropout: 0.1
train:
  epochs: 80
  batch_size: 64
  lr: 0.001
```

### 3.3 `configs/gru.yaml`

GRU baseline 配置。结构与 `lstm.yaml` 基本一致，只是 `model.name=gru`。

### 3.4 `configs/tcn.yaml`

TCN baseline 配置。应包含卷积通道数、kernel size、层数等。

### 3.5 `configs/transformer.yaml`

Vanilla Transformer baseline 配置。注意该模型不使用 SLA gate、不使用 topology、不使用 attribution。

### 3.6 `configs/patchtst.yaml`

PatchTST classification baseline 配置。用于强时序 Transformer 对比。

### 3.7 `configs/stgcn.yaml`

STGCN baseline 配置。用于图时空模型对比。

---

## 4. `data/`：数据目录

```text
data/
├── raw/
├── traces/
│   ├── alibaba/
│   └── google/
├── processed/
└── sparta_dataset/
```

### 4.1 `data/traces/`

存放公开 trace 数据。

#### `data/traces/alibaba/`

放 Alibaba ClusterData。

建议下载：

```bash
git clone https://github.com/alibaba/clusterdata.git data/traces/alibaba/clusterdata
```

建议目录：

```text
data/traces/alibaba/
├── clusterdata/
├── microservices/
├── processed/
└── README.md
```

最终需要生成：

```text
data/traces/alibaba/processed/alibaba_service_load.csv
```

字段：

```csv
time,service_id,service_type,request_rate,base_response_time,dependency_count
```

#### `data/traces/google/`

放 Google ClusterData。

建议下载：

```bash
git clone https://github.com/google/cluster-data.git data/traces/google/cluster-data
```

最终需要生成：

```text
data/traces/google/processed/google_node_load.csv
```

字段：

```csv
time,node_group,cpu_scale,mem_scale,arrival_scale
```

---

### 4.2 `data/raw/`

存放 EdgeSimPy / YAFS 仿真输出的原始日志。

推荐结构：

```text
data/raw/
├── smallworld_20n_300u/
│   ├── node_log.csv
│   ├── link_log.csv
│   ├── service_log.csv
│   ├── path_log.csv
│   └── sla_log.csv
├── tree_20n_300u/
└── random_20n_300u/
```

每个仿真实验一个子目录，命名建议：

```text
{topology}_{edge_nodes}n_{users}u
```

#### `node_log.csv`

每个时间片每个节点的状态：

```csv
time,node_id,node_type,cpu_util,mem_util,queue_len,available_cpu,available_mem
```

#### `link_log.csv`

每个时间片每条链路的状态：

```csv
time,src,dst,delay,bandwidth,loss,jitter,queue_delay,bandwidth_util
```

#### `service_log.csv`

每个时间片每个服务的状态：

```csv
time,service_id,user_id,service_type,request_rate,response_time,current_edge,cloud_node
```

#### `path_log.csv`

每个时间片每个服务的路径信息：

```csv
time,service_id,path_nodes,path_links,path_delay,path_loss,bottleneck_node,bottleneck_link
```

#### `sla_log.csv`

每类服务的 SLA 条件：

```csv
service_id,service_type,max_delay,max_loss,min_bandwidth,reliability_req,cost_weight
```

---

### 4.3 `data/processed/`

存放中间处理结果。

```text
data/processed/
├── alibaba_service_load.csv
├── google_node_load.csv
├── path_graphs_w12.pkl
├── norm_stats.json
└── README.md
```

| 文件 | 来源 | 用途 |
|---|---|---|
| `alibaba_service_load.csv` | `prepare_alibaba_trace.py` | 注入服务请求率和响应时间基准 |
| `google_node_load.csv` | `prepare_google_trace.py` | 注入节点负载波动 |
| `path_graphs_w12.pkl` | `build_path_graph.py` | 存放窗口长度为 12 的服务路径子图 |
| `norm_stats.json` | `normalize.py` | 存放归一化参数 |

---

### 4.4 `data/sparta_dataset/`

存放最终训练、验证、测试数据。

```text
data/sparta_dataset/
├── sparta_w12_h5.pkl
├── train.pkl
├── val.pkl
├── test.pkl
├── label_stats.json
└── README.md
```

| 文件 | 作用 |
|---|---|
| `sparta_w12_h5.pkl` | 完整数据集，输入窗口 `L=12`，预测窗口 `H=5` |
| `train.pkl` | 按时间划分的训练集 |
| `val.pkl` | 按时间划分的验证集 |
| `test.pkl` | 按时间划分的测试集 |
| `label_stats.json` | normal/risky/violated、risk node/link/metric 的分布统计 |

---

## 5. `simulators/`：仿真平台目录

```text
simulators/
├── EdgeSimPy/
└── YAFS/
```

### 5.1 `simulators/EdgeSimPy/`

放 EdgeSimPy 官方仓库。

用途：

- 主仿真平台；
- 生成云边拓扑；
- 生成用户、服务、边缘节点；
- 模拟节点负载、链路状态、服务响应时间。

不要把论文核心代码写到这里，避免后续难以维护。你自己的仿真入口写在：

```text
src/simulation/run_edgesimpy.py
```

### 5.2 `simulators/YAFS/`

放 YAFS 官方仓库。

用途：

- 辅助仿真；
- 动态拓扑；
- 节点故障；
- 链路变化；
- 后续扩展 Computer Networks 期刊版。

ICASSP 第一篇可以先不主用 YAFS。

---

## 6. `third_party/`：第三方模型与 baseline 代码目录

```text
third_party/
├── Time-Series-Library/
├── PatchTST/
└── STGCN-PyTorch/
```

### 6.1 `third_party/Time-Series-Library/`

用途：

- 参考 TimesNet / Transformer 的训练逻辑；
- 可改成 baseline；
- 不建议作为主项目框架。

### 6.2 `third_party/PatchTST/`

用途：

- 作为强时序 Transformer baseline；
- 原始代码偏 forecasting，需要改 classification head；
- 改好的版本建议放在 `src/models/patchtst_cls.py`，不要直接改第三方仓库。

### 6.3 `third_party/STGCN-PyTorch/`

用途：

- 作为图时空 baseline；
- 原始交通预测输入改成云边图输入；
- 改好的版本建议放在 `src/models/stgcn_cls.py`。

---

## 7. `src/simulation/`：仿真代码

```text
src/simulation/
├── run_edgesimpy.py
├── inject_trace.py
└── export_logs.py
```

### 7.1 `run_edgesimpy.py`

#### 作用

主仿真入口，负责：

1. 构建云边拓扑；
2. 创建云节点、边缘节点、接入节点、用户；
3. 创建服务类型和 SLA 约束；
4. 调用 `inject_trace.py` 注入请求率和负载；
5. 运行仿真；
6. 调用 `export_logs.py` 导出日志。

#### 推荐命令

```bash
python src/simulation/run_edgesimpy.py \
  --num_edge_nodes 20 \
  --num_access_nodes 10 \
  --num_users 300 \
  --num_services 3 \
  --num_steps 10000 \
  --topology small_world \
  --trace_source alibaba \
  --output_dir data/raw/smallworld_20n_300u
```

#### 需要写的主要函数

```python
def parse_args():
    """Parse command line arguments."""


def build_topology(args):
    """Build tree/random/small_world cloud-edge topology."""


def create_nodes(args, graph):
    """Create cloud, edge, access, and user nodes."""


def create_services(args):
    """Create latency-sensitive, reliability-sensitive, and cost-sensitive services."""


def run_simulation(args):
    """Main simulation loop."""


def main():
    """Entry point."""
```

#### 输入

- `data/processed/alibaba_service_load.csv`
- `data/processed/google_node_load.csv`
- 命令行参数

#### 输出

- `node_log.csv`
- `link_log.csv`
- `service_log.csv`
- `path_log.csv`
- `sla_log.csv`

---

### 7.2 `inject_trace.py`

#### 作用

把 Alibaba / Google trace 转成仿真中的请求率、服务处理时间、节点负载扰动。

#### 需要写的主要函数

```python
def load_alibaba_trace(path):
    """Load alibaba_service_load.csv."""


def load_google_trace(path):
    """Load google_node_load.csv."""


def get_request_rate(time, service_type, alibaba_df):
    """Return request rate for a service type at a given time."""


def get_node_load_scale(time, node_group, google_df):
    """Return CPU/memory scale factor for a node group at a given time."""


def inject_workload(sim_state, time, trace_data):
    """Inject workload and resource fluctuations into simulator state."""
```

---

### 7.3 `export_logs.py`

#### 作用

统一导出仿真日志，保证字段固定，方便后续预处理。

#### 需要写的主要函数

```python
def export_node_log(records, output_dir):
    """Export node_log.csv."""


def export_link_log(records, output_dir):
    """Export link_log.csv."""


def export_service_log(records, output_dir):
    """Export service_log.csv."""


def export_path_log(records, output_dir):
    """Export path_log.csv."""


def export_sla_log(records, output_dir):
    """Export sla_log.csv."""
```

---

## 8. `src/preprocessing/`：预处理代码

```text
src/preprocessing/
├── prepare_alibaba_trace.py
├── prepare_google_trace.py
├── build_path_graph.py
├── generate_labels.py
├── generate_attribution.py
├── normalize.py
└── split_dataset.py
```

### 8.1 `prepare_alibaba_trace.py`

#### 作用

读取 Alibaba microservices / cluster trace，提取服务调用率、响应时间和服务依赖信息。

#### 输入

```text
data/traces/alibaba/clusterdata/
```

#### 输出

```text
data/processed/alibaba_service_load.csv
```

#### 输出字段

```csv
time,service_id,service_type,request_rate,base_response_time,dependency_count
```

#### 需要写的主要函数

```python
def read_raw_alibaba_files(raw_dir):
    """Read raw Alibaba trace files."""


def extract_call_rate(raw_df):
    """Extract service call rate."""


def extract_response_time(raw_df):
    """Extract base response time."""


def assign_service_type(df):
    """Assign latency/reliability/cost service type."""


def save_processed_trace(df, output_path):
    """Save processed CSV."""
```

---

### 8.2 `prepare_google_trace.py`

#### 作用

读取 Google ClusterData，提取节点负载和任务到达模式。

#### 输入

```text
data/traces/google/cluster-data/
```

#### 输出

```text
data/processed/google_node_load.csv
```

#### 输出字段

```csv
time,node_group,cpu_scale,mem_scale,arrival_scale
```

#### 需要写的主要函数

```python
def read_google_usage_files(raw_dir):
    """Read Google trace usage files."""


def aggregate_node_load(raw_df):
    """Aggregate CPU/memory usage."""


def normalize_load_pattern(df):
    """Normalize load pattern into scale factors."""


def save_processed_trace(df, output_path):
    """Save processed CSV."""
```

---

### 8.3 `build_path_graph.py`

#### 作用

从仿真日志中构造每个服务的 service-path subgraph。

这是论文第一个创新点对应的核心代码。

#### 输入

```text
data/raw/{experiment_name}/node_log.csv
data/raw/{experiment_name}/link_log.csv
data/raw/{experiment_name}/service_log.csv
data/raw/{experiment_name}/path_log.csv
data/raw/{experiment_name}/sla_log.csv
```

#### 输出

```text
data/processed/path_graphs_w12.pkl
```

#### 每个样本格式

```python
sample = {
    "service_id": int,
    "time": int,
    "node_ids": List[int],
    "link_ids": List[Tuple[int, int]],
    "node_features": np.ndarray,       # [L, Np, Fn]
    "link_features": np.ndarray,       # [L, Ep, Fe]
    "service_features": np.ndarray,    # [L, Fs]
    "sla_features": np.ndarray,        # [Fc]
    "adj_matrix": np.ndarray,          # [Np, Np]
}
```

#### 需要写的主要函数

```python
def load_logs(input_dir):
    """Load node/link/service/path/sla logs."""


def parse_path_nodes(path_str):
    """Parse path_nodes field."""


def parse_path_links(path_str):
    """Parse path_links field."""


def collect_node_features(node_log, node_ids, time_window):
    """Collect node features for a service-path subgraph."""


def collect_link_features(link_log, link_ids, time_window):
    """Collect link features for a service-path subgraph."""


def collect_service_features(service_log, service_id, time_window):
    """Collect service features."""


def build_adj_matrix(node_ids, link_ids):
    """Build adjacency matrix for service-path subgraph."""


def build_samples(args):
    """Loop over all service_id and time to build samples."""
```

---

### 8.4 `generate_labels.py`

#### 作用

根据未来窗口 `H` 内的服务状态生成风险等级标签。

#### 输入

```text
data/processed/path_graphs_w12.pkl
```

#### 输出

```text
data/processed/path_graphs_w12_h5_labeled.pkl
```

#### 标签规则

##### violated

未来 `H` 个时间片中任一条件成立：

```text
response_time >= max_delay
path_loss >= max_loss
service_failed == True
```

##### risky

未来 `H` 个时间片未 violated，但任一条件成立：

```text
response_time >= 0.75 * max_delay
path_loss >= 0.75 * max_loss
cpu_util >= 0.85
queue_len 连续 3 个时间片上升
bandwidth_util >= 0.85
```

##### normal

其他情况。

#### 需要写的主要函数

```python
def future_window(service_log, path_log, service_id, time, horizon):
    """Get future H time steps."""


def is_violated(future_df, sla):
    """Return whether future window is violated."""


def is_risky(future_df, sla):
    """Return whether future window is risky."""


def assign_risk_label(sample, logs, horizon):
    """Assign risk_label for a sample."""


def generate_labels(samples, logs, horizon):
    """Generate labels for all samples."""
```

---

### 8.5 `generate_attribution.py`

#### 作用

为 risky / violated 样本生成归因标签：

- `risk_node`
- `risk_link`
- `risk_metric`

这是论文第三个创新点的重要支撑。

#### 输入

```text
data/processed/path_graphs_w12_h5_labeled.pkl
```

#### 输出

```text
data/sparta_dataset/sparta_w12_h5.pkl
```

#### 推荐风险分数

节点风险：

```text
node_score = 0.4 * cpu_util + 0.3 * queue_norm + 0.3 * (1 - available_cpu_norm)
```

链路风险：

```text
link_score = 0.4 * delay_norm + 0.3 * loss_norm + 0.3 * bandwidth_util
```

#### 需要写的主要函数

```python
def compute_node_risk_score(node_features):
    """Compute node risk score."""


def compute_link_risk_score(link_features):
    """Compute link risk score."""


def compute_metric_risk_score(sample):
    """Select dominant risk metric."""


def assign_attribution(sample):
    """Assign risk_node, risk_link, and risk_metric."""


def generate_attribution(samples):
    """Generate attribution labels for all samples."""
```

---

### 8.6 `normalize.py`

#### 作用

归一化节点、链路、服务、SLA 特征。

#### 输入

```text
data/sparta_dataset/sparta_w12_h5.pkl
```

#### 输出

```text
data/sparta_dataset/sparta_w12_h5_norm.pkl
data/processed/norm_stats.json
```

#### 注意

只能用训练集计算均值和方差，不能用验证集和测试集，避免数据泄漏。

#### 需要写的主要函数

```python
def compute_norm_stats(train_samples):
    """Compute normalization statistics using only train samples."""


def normalize_sample(sample, stats):
    """Normalize a single sample."""


def normalize_dataset(samples, stats):
    """Normalize the dataset."""


def save_norm_stats(stats, path):
    """Save normalization statistics."""
```

---

### 8.7 `split_dataset.py`

#### 作用

按时间划分训练、验证、测试集。

#### 输入

```text
data/sparta_dataset/sparta_w12_h5_norm.pkl
```

#### 输出

```text
data/sparta_dataset/train.pkl
data/sparta_dataset/val.pkl
data/sparta_dataset/test.pkl
data/sparta_dataset/label_stats.json
```

#### 划分原则

必须按时间划分：

```text
train: 前 60%
val: 中间 20%
test: 后 20%
```

不能随机划分，否则会造成时间泄漏。

#### 需要写的主要函数

```python
def sort_samples_by_time(samples):
    """Sort samples by time."""


def split_by_time(samples, train_ratio=0.6, val_ratio=0.2):
    """Split samples by time order."""


def compute_label_stats(samples):
    """Compute label distribution."""


def save_splits(train, val, test, output_dir):
    """Save train/val/test splits."""
```

---

## 9. `src/datasets/sparta_dataset.py`：数据加载

### 9.1 作用

PyTorch Dataset，用于读取 `train.pkl`、`val.pkl`、`test.pkl`，返回模型所需张量。

### 9.2 输出给模型的 batch

```python
batch = {
    "node_x": Tensor[B, L, Np, Fn],
    "link_x": Tensor[B, L, Ep, Fe],
    "service_x": Tensor[B, L, Fs],
    "sla_x": Tensor[B, Fc],
    "adj": Tensor[B, Np, Np],
    "risk_label": Tensor[B],
    "risk_node": Tensor[B],
    "risk_link": Tensor[B],
    "risk_metric": Tensor[B],
    "node_mask": Tensor[B, Np],
    "link_mask": Tensor[B, Ep],
}
```

### 9.3 需要写的类和函数

```python
class SPARTADataset(torch.utils.data.Dataset):
    def __init__(self, pkl_path):
        """Load pkl samples."""

    def __len__(self):
        """Return number of samples."""

    def __getitem__(self, idx):
        """Return one sample."""


def sparta_collate_fn(batch):
    """Pad variable Np/Ep to batch max length."""


def build_dataloader(pkl_path, batch_size, shuffle):
    """Create DataLoader."""
```

---

## 10. `src/models/`：模型目录

```text
src/models/
├── sparta.py
├── lstm.py
├── gru.py
├── tcn.py
├── transformer.py
├── patchtst_cls.py
└── stgcn_cls.py
```

### 10.1 `sparta.py`

#### 作用

实现 SPARTA 主模型。

#### 输入

```python
node_x:    [B, L, Np, Fn]
link_x:    [B, L, Ep, Fe]
service_x: [B, L, Fs]
sla_x:     [B, Fc]
adj:       [B, Np, Np]
node_mask: [B, Np]
link_mask: [B, Ep]
```

#### 输出

```python
outputs = {
    "risk_logits": Tensor[B, 3],
    "node_logits": Tensor[B, Np],
    "link_logits": Tensor[B, Ep],
    "metric_logits": Tensor[B, 5],
}
```

#### 推荐类结构

```python
class MLP(nn.Module):
    """Generic MLP for node/link/service/SLA embedding."""


class SimpleGCNLayer(nn.Module):
    """Lightweight topology aggregation layer."""


class SLAGate(nn.Module):
    """SLA-conditioned gating module."""


class SPARTA(nn.Module):
    def __init__(self, config):
        """Initialize embeddings, temporal encoder, topology encoder, and heads."""

    def encode_node_link(self, node_x, link_x):
        """Encode node and link features."""

    def build_path_token(self, node_emb, link_emb, service_x):
        """Build [B,L,D] path-level token."""

    def forward(self, batch):
        """Forward pass."""
```

#### 模块连接

```text
node_x/link_x/service_x/sla_x/adj
        ↓
embedding
        ↓
path_token [B,L,D]
        ↓
Temporal Transformer
        ↓
Topology GCN / topology pooling
        ↓
SLA gate
        ↓
risk head + attribution heads
```

---

### 10.1.1 SPARTA 模型整体结构与端到端流程

SPARTA 的完整名称是 **Service-Path Aware Temporal Risk Attribution**。它不是一个普通的时序分类器，而是一个面向云边服务路径的多任务风险检测模型。模型输入是由 `build_path_graph.py` 构造出来的服务路径子图序列，输出是风险等级、风险概率、风险节点、风险链路和风险主导指标。

#### 10.1.1.1 总体结构图

```text
Alibaba / Google Trace
        ↓
EdgeSimPy 仿真生成 node/link/service/path/SLA 日志
        ↓
build_path_graph.py 构造 service-path subgraph
        ↓
SPARTADataset 读取 batch
        ↓
┌──────────────────────────────────────────────────────────────┐
│                         SPARTA Model                          │
├──────────────────────────────────────────────────────────────┤
│ 1. Heterogeneous Feature Embedding                             │
│    node_x, link_x, service_x, sla_x → hidden_dim               │
│                                                              │
│ 2. Service-Path Token Construction                             │
│    node_emb + link_emb + service_emb → path_token [B,L,D]      │
│                                                              │
│ 3. Temporal Encoder                                            │
│    path_token → Transformer/TCN temporal representation        │
│                                                              │
│ 4. Topology Encoder                                            │
│    node_emb + adj → topology-aware representation              │
│                                                              │
│ 5. SLA-Conditioned Fusion                                      │
│    temporal + topology + SLA gate → risk representation z      │
│                                                              │
│ 6. Multi-task Heads                                            │
│    z → risk_logits / node_logits / link_logits / metric_logits │
└──────────────────────────────────────────────────────────────┘
        ↓
multitask_loss.py 计算 risk + node + link + metric loss
        ↓
evaluate.py 输出 Macro-F1 / Risk Recall / Attr-Acc / AUC
        ↓
visualize.py 生成 attribution case 和 horizon curve
        ↓
proactive intervention 验证风险检测能否降低 violation rate
```

#### 10.1.1.2 输入层：模型真正吃什么数据

`SPARTA.forward(batch)` 接收的是 `SPARTADataset` 返回的 batch，而不是原始 CSV。一个 batch 至少包含：

```python
batch = {
    "node_x": Tensor[B, L, Np, Fn],      # 服务路径子图中的节点时间序列特征
    "link_x": Tensor[B, L, Ep, Fe],      # 服务路径子图中的链路时间序列特征
    "service_x": Tensor[B, L, Fs],       # 服务本身的请求率、响应时间等状态
    "sla_x": Tensor[B, Fc],              # 服务 SLA 条件向量
    "adj": Tensor[B, Np, Np],            # 服务路径子图邻接矩阵
    "node_mask": Tensor[B, Np],          # padding 后的有效节点 mask
    "link_mask": Tensor[B, Ep],          # padding 后的有效链路 mask
    "risk_label": Tensor[B],             # normal/risky/violated
    "risk_node": Tensor[B],              # 风险节点标签
    "risk_link": Tensor[B],              # 风险链路标签
    "risk_metric": Tensor[B],            # 风险主导指标标签
}
```

其中 `B` 是 batch size，`L` 是历史时间窗口长度，推荐初始值为 `12`；`Np` 是 padding 后的服务路径子图最大节点数；`Ep` 是 padding 后的最大链路数；`Fn/Fe/Fs/Fc` 分别是节点、链路、服务和 SLA 特征维度。

#### 10.1.1.3 第一层：Heterogeneous Feature Embedding

不同类型输入不能直接相加，必须先映射到统一维度 `D`。因此 `sparta.py` 中应包含四个 embedding 模块：

```python
self.node_mlp = MLP(node_feat_dim, hidden_dim)
self.link_mlp = MLP(link_feat_dim, hidden_dim)
self.service_mlp = MLP(service_feat_dim, hidden_dim)
self.sla_mlp = MLP(sla_feat_dim, hidden_dim)
```

输入输出关系：

```text
node_x    [B,L,Np,Fn] → node_emb    [B,L,Np,D]
link_x    [B,L,Ep,Fe] → link_emb    [B,L,Ep,D]
service_x [B,L,Fs]    → service_emb [B,L,D]
sla_x     [B,Fc]      → sla_emb     [B,D]
```

这一层对应论文中的 **heterogeneous network signal embedding**，作用是把节点、链路、服务和 SLA 统一成可融合的隐空间表示。

#### 10.1.1.4 第二层：Service-Path Token Construction

每个服务路径子图在每个时间片都要压缩成一个 path-level token，用于后续时间建模。推荐实现方式是 masked mean pooling：

```python
node_pool = masked_mean(node_emb, node_mask, dim=2)  # [B,L,D]
link_pool = masked_mean(link_emb, link_mask, dim=2)  # [B,L,D]
path_token = node_pool + link_pool + service_emb     # [B,L,D]
```

这一层的含义是：每个时间片的服务状态不是单一数值，而是由服务路径中的节点状态、链路状态和服务自身状态共同决定。

#### 10.1.1.5 第三层：Temporal Encoder

`path_token` 的形状为 `[B,L,D]`，表示过去 `L` 个时间片的服务路径状态。Temporal Encoder 用于学习风险随时间积累的过程，例如排队长度持续上升、链路时延持续恶化、请求率突增等。

推荐第一版使用 Transformer Encoder：

```python
self.temporal_encoder = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * 4,
        dropout=dropout,
        batch_first=True,
    ),
    num_layers=num_layers,
)
```

输出：

```python
temporal_out = self.temporal_encoder(path_token)  # [B,L,D]
z_temporal = temporal_out[:, -1, :]               # [B,D]
```

这一层对应论文中的 **early risk temporal modeling**。它的目标不是判断当前是否已经违约，而是根据历史窗口预测未来 `H` 个时间片内的风险。

#### 10.1.1.6 第四层：Topology Encoder

仅有时间序列还不够，因为 SLA 风险通常与服务路径拓扑相关。Topology Encoder 使用 `adj` 对服务路径子图中的节点进行轻量图聚合。

推荐第一版实现一个简单 GCN：

```python
node_last = node_emb[:, -1, :, :]                  # [B,Np,D]
topo_node = self.gcn(node_last, adj, node_mask)    # [B,Np,D]
z_topo = masked_mean(topo_node, node_mask, dim=1)  # [B,D]
```

如果后续实现能力足够，可以升级为 topology attention bias：

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(D) + alpha * A) V
```

但第一版 README 和代码建议以 GCN/topology pooling 为主，保证稳定可跑。

#### 10.1.1.7 第五层：SLA-Conditioned Fusion

不同业务的 SLA 不同，同一个网络状态对不同服务的风险含义也不同。例如，80ms delay 对普通同步服务可能 normal，但对实时控制服务可能 risky。因此必须显式引入 SLA 条件。

推荐使用 SLA gate：

```python
z = z_temporal + z_topo                         # [B,D]
gate = torch.sigmoid(self.sla_gate(sla_emb))    # [B,D]
z_sla = z * gate + z                            # [B,D]
```

这一层对应论文中的 **SLA-conditioned temporal-topology risk modeling**。如果消融 `w/o SLA`，就是去掉 `gate`，直接使用 `z` 做分类。

#### 10.1.1.8 第六层：Multi-task Heads

SPARTA 最终不是只输出风险类别，还要输出风险来源。建议在 `sparta.py` 中写四个 head：

```python
self.risk_head = nn.Linear(hidden_dim, 3)
self.node_head = nn.Linear(hidden_dim, 1)
self.link_head = nn.Linear(hidden_dim, 1)
self.metric_head = nn.Linear(hidden_dim, 5)
```

输出逻辑：

```python
risk_logits = self.risk_head(z_sla)                  # [B,3]
node_logits = self.node_head(topo_node).squeeze(-1)   # [B,Np]
link_logits = self.link_head(link_last).squeeze(-1)   # [B,Ep]
metric_logits = self.metric_head(z_sla)               # [B,5]
```

最终返回：

```python
return {
    "risk_logits": risk_logits,
    "node_logits": node_logits,
    "link_logits": link_logits,
    "metric_logits": metric_logits,
}
```

其中：

- `risk_logits` 用于预测 `normal / risky / violated`；
- `node_logits` 用于定位风险节点；
- `link_logits` 用于定位风险链路；
- `metric_logits` 用于判断风险主导指标。

#### 10.1.1.9 损失函数流程

`src/losses/multitask_loss.py` 负责把四个任务合并：

```text
L_total = L_risk + λ_node L_node + λ_link L_link + λ_metric L_metric
```

推荐初始权重：

```yaml
loss:
  risk_weight: 1.0
  node_weight: 0.3
  link_weight: 0.3
  metric_weight: 0.2
```

其中 `L_risk` 可以使用 class-weighted CE 或 focal loss，防止 normal 类数量过多导致模型忽视 risky / violated 类。

#### 10.1.1.10 训练阶段流程

训练时的完整流程如下：

```text
train.py
  ↓
读取 configs/sparta.yaml
  ↓
SPARTADataset 加载 train.pkl / val.pkl
  ↓
DataLoader 输出 batch
  ↓
SPARTA.forward(batch)
  ↓
multitask_loss(outputs, batch)
  ↓
反向传播更新参数
  ↓
验证集计算 Macro-F1 / Risk Recall
  ↓
保存 checkpoints/sparta_best.pth
```

推荐以 `Macro-F1 + Risk Recall` 作为保存最佳模型的主要依据，而不是只看 Accuracy。

#### 10.1.1.11 推理与评估阶段流程

评估时：

```text
evaluate.py
  ↓
加载 checkpoints/sparta_best.pth
  ↓
读取 test.pkl
  ↓
输出 risk prediction + attribution prediction
  ↓
计算 main_results.csv / attribution_results.csv / horizon_results.csv
```

需要输出的关键指标：

```text
Accuracy
Macro-F1
Risk Recall
Violation Recall
AUC
Node Attribution Accuracy
Link Attribution Accuracy
Metric Attribution Accuracy
Inference Time
```

#### 10.1.1.12 主动干预验证流程

主动干预不是 SPARTA 主模型的一部分，而是评估阶段用于证明预测结果有用的小实验。流程如下：

```text
SPARTA 预测某服务为 risky
        ↓
读取 node/link/metric attribution
        ↓
如果 risk node 过载 → 迁移到低负载邻近边缘节点
如果 risk link 拥塞 → 切换到低 delay/loss 备选路径
如果 risk metric 是 bandwidth → 选择剩余带宽更高路径
        ↓
重新计算未来窗口内 response_time / path_loss
        ↓
统计 proactive violation rate
        ↓
与 reactive 策略对比
```

输出文件：

```text
results/proactive_results.csv
```

推荐表头：

```csv
strategy,violation_rate,avg_response_time,intervention_cost,num_interventions
```

#### 10.1.1.13 SPARTA 和 baseline 的区别

| 模型 | 使用 service-path | 使用 topology | 使用 SLA condition | 输出 attribution |
|---|---:|---:|---:|---:|
| LSTM | 部分 | 否 | 否 | 否 |
| GRU | 部分 | 否 | 否 | 否 |
| TCN | 部分 | 否 | 否 | 否 |
| Vanilla Transformer | 部分 | 否 | 否 | 否 |
| PatchTST / TimesNet | 部分 | 否 | 否 | 否 |
| STGCN | 可用全局图 | 是 | 否 | 否 |
| **SPARTA** | **是** | **是** | **是** | **是** |

这张表可以直接放在 README，也可以在论文实验部分转化为方法对比说明。

#### 10.1.1.14 SPARTA 最小可运行版本

如果先做最小版本，`sparta.py` 至少实现以下组件：

```text
1. NodeMLP / LinkMLP / ServiceMLP / SLAMLP
2. Path token construction
3. TransformerEncoder
4. SimpleGCNLayer 或 topology pooling
5. SLA gate
6. risk_head
7. node_head / link_head / metric_head
8. forward(batch)
```

先不要实现复杂强化学习、复杂 GAT、多智能体调度或大模型意图理解。这些都不是第一篇 ICASSP 的必要内容。


### 10.2 `lstm.py`

LSTM baseline。

输入：

```python
path_token: [B, L, D]
```

输出：

```python
risk_logits: [B, 3]
```

推荐类：

```python
class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        pass

    def forward(self, batch):
        pass
```

LSTM baseline 不输出 attribution，评估时 Attr-Acc 记为 `N/A`。

### 10.3 `gru.py`

GRU baseline，与 `lstm.py` 类似，只是替换为 GRU。

### 10.4 `tcn.py`

TCN baseline，用一维卷积建模时间序列。

输入：

```python
[B, L, D] → transpose → [B, D, L]
```

输出：

```python
risk_logits: [B, 3]
```

### 10.5 `transformer.py`

Vanilla Transformer baseline。不能使用：

- service-path topology；
- SLA gate；
- attribution head。

只能用：

```text
path_token → TransformerEncoder → classification head
```

### 10.6 `patchtst_cls.py`

PatchTST classification baseline。

实现思路：

```text
time-series patching → Transformer encoder → pooling → classification head
```

### 10.7 `stgcn_cls.py`

STGCN baseline。

输入：

```python
node_x: [B, L, N, F]
adj: [B, N, N]
```

输出：

```python
risk_logits: [B, 3]
```

建议主 baseline 用全局图，从而突出 SPARTA 的 service-path subgraph 设计。

---

## 11. `src/losses/`：损失函数

```text
src/losses/
├── focal_loss.py
└── multitask_loss.py
```

### 11.1 `focal_loss.py`

处理 normal 类过多、risky/violated 类较少的问题。

推荐类：

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        pass

    def forward(self, logits, targets):
        pass
```

### 11.2 `multitask_loss.py`

SPARTA 的多任务损失。

输入：

```python
outputs = {
    "risk_logits": ...,
    "node_logits": ...,
    "link_logits": ...,
    "metric_logits": ...,
}

batch = {
    "risk_label": ...,
    "risk_node": ...,
    "risk_link": ...,
    "risk_metric": ...,
}
```

损失：

```text
L = L_risk + λ_node L_node + λ_link L_link + λ_metric L_metric
```

推荐类：

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_node=0.3, lambda_link=0.3, lambda_metric=0.2):
        pass

    def forward(self, outputs, batch):
        pass
```

---

## 12. 训练、评估与可视化

### 12.1 `src/train.py`

统一训练入口，支持 SPARTA 和所有 baseline。

推荐命令：

```bash
python src/train.py --config configs/sparta.yaml
```

需要写的主要函数：

```python
def load_config(path):
    """Load yaml config."""


def set_seed(seed):
    """Set random seed."""


def build_model(config):
    """Build model according to config.model.name."""


def build_loss(config):
    """Build loss function."""


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch."""


def validate(model, loader, criterion, device):
    """Validate model."""


def save_checkpoint(model, path, metrics):
    """Save best checkpoint."""


def main():
    """Training entry."""
```

### 12.2 `src/evaluate.py`

评估模型，生成论文表格所需结果。

推荐命令：

```bash
python src/evaluate.py \
  --checkpoint checkpoints/sparta_best.pth \
  --config configs/sparta.yaml \
  --split test \
  --output results/sparta_test_metrics.json
```

需要计算：

- Accuracy；
- Macro-F1；
- Risk Recall；
- Violation Recall；
- AUC；
- Node Attribution Accuracy；
- Link Attribution Accuracy；
- Metric Attribution Accuracy；
- Top-k Attribution Accuracy；
- Inference Time。

### 12.3 `src/visualize.py`

生成论文图表。

必须生成：

```text
figures/horizon_curve.pdf
figures/attribution_case.pdf
figures/confusion_matrix.pdf
figures/proactive_comparison.pdf
```

推荐函数：

```python
def plot_horizon_curve(horizon_results):
    """Plot Risk Recall or Macro-F1 under H=1/3/5/10."""


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""


def plot_attribution_case(sample, pred_node, pred_link, save_path):
    """Plot service-path subgraph and highlight risk node/link."""


def plot_proactive_results(results):
    """Plot proactive vs reactive results."""
```

---

## 13. `scripts/`：一键执行脚本

```text
scripts/
├── 00_setup.sh
├── 01_download_code.sh
├── 02_prepare_traces.sh
├── 03_run_simulation.sh
├── 04_build_dataset.sh
├── 05_train_baselines.sh
├── 06_train_sparta.sh
└── 07_evaluate_all.sh
```

### 13.1 `00_setup.sh`

创建环境。

```bash
#!/bin/bash
conda create -n sparta python=3.10 -y
conda activate sparta

pip install numpy pandas scipy scikit-learn networkx matplotlib tqdm pyyaml xgboost einops
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 13.2 `01_download_code.sh`

下载第三方代码。

```bash
#!/bin/bash

mkdir -p simulators third_party data/traces

git clone https://github.com/EdgeSimPy/EdgeSimPy.git simulators/EdgeSimPy
git clone https://github.com/acsicuib/YAFS.git simulators/YAFS

git clone https://github.com/thuml/Time-Series-Library.git third_party/Time-Series-Library
git clone https://github.com/yuqinie98/PatchTST.git third_party/PatchTST
git clone https://github.com/FelixOpolka/STGCN-PyTorch.git third_party/STGCN-PyTorch

git clone https://github.com/alibaba/clusterdata.git data/traces/alibaba/clusterdata
git clone https://github.com/google/cluster-data.git data/traces/google/cluster-data
```

### 13.3 `02_prepare_traces.sh`

处理 Alibaba / Google trace。

```bash
#!/bin/bash

python src/preprocessing/prepare_alibaba_trace.py \
  --raw_dir data/traces/alibaba/clusterdata \
  --output data/processed/alibaba_service_load.csv

python src/preprocessing/prepare_google_trace.py \
  --raw_dir data/traces/google/cluster-data \
  --output data/processed/google_node_load.csv
```

### 13.4 `03_run_simulation.sh`

运行 EdgeSimPy 仿真。

```bash
#!/bin/bash

python src/simulation/run_edgesimpy.py \
  --num_edge_nodes 20 \
  --num_access_nodes 10 \
  --num_users 300 \
  --num_services 3 \
  --num_steps 10000 \
  --topology small_world \
  --trace_source alibaba \
  --output_dir data/raw/smallworld_20n_300u
```

### 13.5 `04_build_dataset.sh`

构造最终数据集。

```bash
#!/bin/bash

python src/preprocessing/build_path_graph.py \
  --input_dir data/raw/smallworld_20n_300u \
  --window 12 \
  --output data/processed/path_graphs_w12.pkl

python src/preprocessing/generate_labels.py \
  --input data/processed/path_graphs_w12.pkl \
  --horizon 5 \
  --output data/processed/path_graphs_w12_h5_labeled.pkl

python src/preprocessing/generate_attribution.py \
  --input data/processed/path_graphs_w12_h5_labeled.pkl \
  --output data/sparta_dataset/sparta_w12_h5.pkl

python src/preprocessing/normalize.py \
  --input data/sparta_dataset/sparta_w12_h5.pkl \
  --output data/sparta_dataset/sparta_w12_h5_norm.pkl \
  --stats data/processed/norm_stats.json

python src/preprocessing/split_dataset.py \
  --input data/sparta_dataset/sparta_w12_h5_norm.pkl \
  --output_dir data/sparta_dataset
```

### 13.6 `05_train_baselines.sh`

训练所有 baseline。

```bash
#!/bin/bash

python src/train.py --config configs/lstm.yaml
python src/train.py --config configs/gru.yaml
python src/train.py --config configs/tcn.yaml
python src/train.py --config configs/transformer.yaml
python src/train.py --config configs/patchtst.yaml
python src/train.py --config configs/stgcn.yaml
```

### 13.7 `06_train_sparta.sh`

训练 SPARTA。

```bash
#!/bin/bash

python src/train.py --config configs/sparta.yaml
```

### 13.8 `07_evaluate_all.sh`

统一评估所有模型，生成结果表。

```bash
#!/bin/bash

python src/evaluate.py --config configs/lstm.yaml --checkpoint checkpoints/lstm_best.pth --output results/lstm.json
python src/evaluate.py --config configs/gru.yaml --checkpoint checkpoints/gru_best.pth --output results/gru.json
python src/evaluate.py --config configs/tcn.yaml --checkpoint checkpoints/tcn_best.pth --output results/tcn.json
python src/evaluate.py --config configs/transformer.yaml --checkpoint checkpoints/transformer_best.pth --output results/transformer.json
python src/evaluate.py --config configs/patchtst.yaml --checkpoint checkpoints/patchtst_best.pth --output results/patchtst.json
python src/evaluate.py --config configs/stgcn.yaml --checkpoint checkpoints/stgcn_best.pth --output results/stgcn.json
python src/evaluate.py --config configs/sparta.yaml --checkpoint checkpoints/sparta_best.pth --output results/sparta.json
```

---

## 14. `results/`：结果目录

```text
results/
├── main_results.csv
├── ablation_results.csv
├── horizon_results.csv
├── attribution_results.csv
├── proactive_results.csv
└── logs/
```

### 14.1 `main_results.csv`

主实验结果。

字段：

```csv
method,accuracy,macro_f1,risk_recall,violation_recall,auc,attr_acc,inference_time
```

### 14.2 `ablation_results.csv`

消融实验。

字段：

```csv
variant,macro_f1,risk_recall,violation_recall,node_attr_acc,link_attr_acc,metric_attr_acc
```

### 14.3 `horizon_results.csv`

提前预测窗口实验。

字段：

```csv
method,horizon,macro_f1,risk_recall,violation_recall
```

### 14.4 `attribution_results.csv`

归因结果。

字段：

```csv
method,node_attr_acc,link_attr_acc,metric_attr_acc,top3_node_acc,top3_link_acc
```

### 14.5 `proactive_results.csv`

主动干预验证。

字段：

```csv
strategy,violation_rate,avg_response_time,migration_cost,num_interventions
```

---

## 15. `figures/`：论文图表目录

```text
figures/
├── framework.pdf
├── service_path_subgraph.pdf
├── horizon_curve.pdf
├── attribution_case.pdf
├── confusion_matrix.pdf
└── proactive_comparison.pdf
```

### 15.1 `framework.pdf`

论文总框架图。

内容：

```text
trace-driven simulation
→ service-path subgraph
→ SPARTA
→ risk detection + attribution
→ proactive intervention
```

### 15.2 `service_path_subgraph.pdf`

展示服务路径子图。

必须包含：

- user；
- access node；
- current edge；
- candidate edge；
- cloud；
- path links；
- risk node；
- risk link。

### 15.3 `horizon_curve.pdf`

展示 `H=1/3/5/10` 时不同模型的 Risk Recall 或 Macro-F1。

### 15.4 `attribution_case.pdf`

展示具体样本的风险归因结果。

### 15.5 `confusion_matrix.pdf`

展示 normal / risky / violated 三分类混淆矩阵。

### 15.6 `proactive_comparison.pdf`

展示 Reactive vs Proactive-SPARTA 的违约率和平均响应时间。

---

## 16. `logs/` 与 `checkpoints/`

### 16.1 `logs/`

存放训练日志。

```text
logs/
├── sparta_train.log
├── lstm_train.log
├── transformer_train.log
└── stgcn_train.log
```

日志应记录：

- epoch；
- train loss；
- val loss；
- val macro-F1；
- val risk recall；
- best checkpoint path。

### 16.2 `checkpoints/`

存放模型权重。

```text
checkpoints/
├── sparta_best.pth
├── lstm_best.pth
├── gru_best.pth
├── tcn_best.pth
├── transformer_best.pth
├── patchtst_best.pth
└── stgcn_best.pth
```

---

## 17. 完整数据流

```text
Alibaba / Google trace
        ↓
prepare_alibaba_trace.py / prepare_google_trace.py
        ↓
data/processed/alibaba_service_load.csv
data/processed/google_node_load.csv
        ↓
inject_trace.py
        ↓
run_edgesimpy.py
        ↓
data/raw/{experiment}/node_log.csv
data/raw/{experiment}/link_log.csv
data/raw/{experiment}/service_log.csv
data/raw/{experiment}/path_log.csv
data/raw/{experiment}/sla_log.csv
        ↓
build_path_graph.py
        ↓
data/processed/path_graphs_w12.pkl
        ↓
generate_labels.py
        ↓
data/processed/path_graphs_w12_h5_labeled.pkl
        ↓
generate_attribution.py
        ↓
data/sparta_dataset/sparta_w12_h5.pkl
        ↓
normalize.py
        ↓
data/sparta_dataset/sparta_w12_h5_norm.pkl
        ↓
split_dataset.py
        ↓
train.pkl / val.pkl / test.pkl
        ↓
train.py
        ↓
checkpoints/*.pth
        ↓
evaluate.py
        ↓
results/*.csv
        ↓
visualize.py
        ↓
figures/*.pdf
```

---

## 18. 完整训练流

```text
configs/sparta.yaml
        ↓
train.py
        ↓
SPARTADataset
        ↓
SPARTA model
        ↓
MultiTaskLoss
        ↓
optimizer
        ↓
validation
        ↓
save checkpoints/sparta_best.pth
        ↓
evaluate.py
        ↓
results/sparta.json / main_results.csv
```

baseline 的训练流类似，只是模型和 loss 不同。

---

## 19. 最小运行顺序

第一次跑项目时，建议按下面顺序执行：

```bash
bash scripts/00_setup.sh
bash scripts/01_download_code.sh
bash scripts/02_prepare_traces.sh
bash scripts/03_run_simulation.sh
bash scripts/04_build_dataset.sh
bash scripts/05_train_baselines.sh
bash scripts/06_train_sparta.sh
bash scripts/07_evaluate_all.sh
```

如果某一步失败，先不要继续下一步。

---

## 20. 最小可交付版本

如果时间紧，最少需要完成以下文件：

```text
src/simulation/run_edgesimpy.py
src/simulation/export_logs.py
src/preprocessing/build_path_graph.py
src/preprocessing/generate_labels.py
src/preprocessing/generate_attribution.py
src/datasets/sparta_dataset.py
src/models/sparta.py
src/models/lstm.py
src/models/transformer.py
src/models/stgcn_cls.py
src/losses/multitask_loss.py
src/train.py
src/evaluate.py
src/visualize.py
configs/sparta.yaml
configs/lstm.yaml
configs/transformer.yaml
configs/stgcn.yaml
scripts/03_run_simulation.sh
scripts/04_build_dataset.sh
scripts/06_train_sparta.sh
```

最少需要完成以下实验：

1. LSTM baseline；
2. Transformer baseline；
3. STGCN baseline；
4. SPARTA；
5. w/o SLA；
6. w/o topology；
7. w/o attribution；
8. H=1/3/5 early detection；
9. proactive vs reactive。

---

## 21. 常见问题

### 21.1 是否必须用 YAFS？

不是。ICASSP 第一篇主用 EdgeSimPy 即可。YAFS 可留给扩展版。

### 21.2 是否必须用 Google trace？

不是。Alibaba trace 更贴近服务调用。Google trace 可用于补充节点负载波动。如果时间紧，先只用 Alibaba。

### 21.3 是否必须做 PatchTST 和 TimesNet？

不一定。至少要有：

- LSTM；
- Transformer；
- STGCN。

如果时间允许，加 PatchTST 或 TimesNet 会更有说服力。

### 21.4 是否必须做 proactive intervention？

强烈建议做。它不需要复杂算法，但能显著增强论文价值。

### 21.5 是否需要真实网络？

不需要。第一篇目标是 trace-driven simulation + network signal modeling，不是系统部署论文。

---

## 22. README 使用建议

这个 README 的作用不是只给读者看，更是给你自己、后续 AI 编程助手、论文写作助手和导师沟通使用。

推荐使用方式：

1. 先按 README 建好项目结构；
2. 每写一个文件，就在对应位置补充实际运行命令；
3. 每跑完一个实验，就更新 `results/` 和 `figures/`；
4. 论文写作时直接参考 `results/` 和 `figures/`；
5. 后续扩展 Computer Networks 期刊版时，保留此结构，只增加 YAFS、多拓扑和更完整主动干预。

---

## 23. 项目目标复述

最终项目应实现：

```text
trace-driven cloud-edge simulation
+ service-path subgraph construction
+ SLA-conditioned temporal-topology risk modeling
+ risk node/link/metric attribution
+ proactive intervention validation
```

只要这条主线不偏，项目就能支撑 ICASSP 第一篇论文。
