# ICASSP 稳妥投稿方案报告：面向云边服务的 SLA 风险主动检测与服务路径归因

## 0. 报告结论

如果目标是“尽量稳地冲 ICASSP”，不建议把论文写成传统网络优化或边缘计算调度论文，而应当将问题包装为：

> **云边网络状态是一类多变量时空信号，SLA 风险检测是一个面向网络信号的时序检测、分类与归因问题。**

推荐第一篇 ICASSP 论文题目：

> **SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services**

中文可写为：

> **面向云边服务的服务路径感知 SLA 风险早期检测与归因方法**

该题目适合投 ICASSP 的原因是：

1. 它不是普通边缘计算任务卸载，而是“网络信号的时空风险检测”；
2. ICASSP 接收机器学习、时序分析、图信号处理、检测分类、网络与分布式系统信号处理、边缘计算、IoT 等方向；
3. 论文篇幅较短，适合做一个“清晰问题 + 明确方法 + 充分实验”的第一篇练手 CCF B；
4. 你的 LSTM、Transformer、DETR、检测思维和可视化经验都能迁移；
5. 不需要真实 AI 数据中心和 GPU 集群，可以基于 EdgeSimPy / YAFS + Alibaba / Google trace 构建实验。

需要强调：没有任何会议可以保证“稳中”。这里的“稳”是指尽量降低选题不匹配、实验不完整、创新点不清楚带来的拒稿风险。

---

## 1. 为什么第一篇更建议投 ICASSP，而不是 Computer Networks

### 1.1 ICASSP 的优势

ICASSP 是会议，周期相对更可控，论文篇幅短，更适合作为博士第一篇练手论文。它更关注方法是否清楚、信号建模是否合理、实验是否扎实，而不一定要求构建完整网络系统。

你的课题如果写成：

> 云边服务 SLA 违约风险预测

可能显得偏网络；但如果写成：

> 多变量网络状态信号的早期风险检测与归因

就更贴近 ICASSP。

### 1.2 Computer Networks 的风险

Computer Networks 是 CCF B 网络期刊，方向更匹配，但它通常更看重网络系统性、网络机制、实验完整性和长期扩展。对于第一篇来说，如果只是“仿真 + Transformer 分类”，容易被质疑网络贡献不足。

因此建议：

- 第一篇：ICASSP，做“网络状态信号的早期风险检测与归因”；
- 第二篇：Computer Networks / PR，扩展成期刊版；
- 第三篇：INFOCOM / TMC / TON，做完整闭环编排。

---

## 2. ICASSP 版论文定位

### 2.1 推荐题目

英文题目：

> **SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services**

备用题目：

1. **Service-Path Aware Temporal Risk Detection for Cloud-Edge Network Signals**
2. **SLA-Aware Spatio-Temporal Network Signal Modeling for Proactive Cloud-Edge Service Risk Detection**
3. **Early Detection of SLA Violation Risks from Cloud-Edge Network Signals via Service-Path Attribution**

中文题目：

> **面向云边服务的服务路径感知 SLA 风险早期检测与归因方法**

### 2.2 投稿方向包装

建议不要投“纯通信网络调度”口径，而是投以下方向口径：

1. Machine Learning for Time Series Analysis；
2. Detection and Classification；
3. Graph Signal Processing；
4. Signal Processing for Networks and Distributed Systems；
5. Edge and Embedded Computing；
6. Internet of Things；
7. Network Resource Management；
8. Edge, Sensor and Ad-hoc Networks。

### 2.3 论文核心问题

给定云边网络过去一段时间内的节点、链路、服务请求和 SLA 约束状态，预测未来若干时间片内某个服务是否会出现 SLA 风险，并进一步判断风险主要来自哪一个节点、链路或指标。

形式化地说：

输入：

- 历史时间窗口：`X[t-L:t]`
- 服务路径子图：`G_s = (V_s, E_s)`
- SLA 条件向量：`c_s`
- 服务状态：`r_s`

输出：

- 风险等级：`normal / risky / violated`
- 风险概率：`p_risk`
- 风险来源节点：`risk node`
- 风险来源链路：`risk link`
- 风险主导指标：`delay / loss / CPU / queue / bandwidth`

### 2.4 论文核心创新句

可以写成：

> Unlike existing SLA prediction methods that directly classify global QoS records, we model cloud-edge service states as service-path network signals and perform early risk detection with lightweight temporal attribution.

---

## 3. 现有代码与平台选择

### 3.1 主仿真平台：EdgeSimPy

仓库地址：

```bash
https://github.com/EdgeSimPy/EdgeSimPy
```

EdgeSimPy 是 Python-based edge computing simulator，具有 edge servers、network devices、applications、user mobility、application composition、power consumption 等建模能力。它适合作为第一篇实验主平台。

#### 安装方式

建议使用 conda 单独建环境：

```bash
conda create -n sparta python=3.10 -y
conda activate sparta

git clone https://github.com/EdgeSimPy/EdgeSimPy.git
cd EdgeSimPy

pip install -r requirements.txt
pip install -e .
```

如果 `requirements.txt` 不完整，可以补充：

```bash
pip install numpy pandas scipy networkx matplotlib tqdm scikit-learn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 建议目录

```text
SPARTA/
├── simulators/
│   └── EdgeSimPy/
├── data/
│   ├── raw/
│   ├── traces/
│   ├── processed/
│   └── sparta_dataset/
├── src/
│   ├── simulation/
│   ├── preprocessing/
│   ├── models/
│   ├── baselines/
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
├── configs/
├── experiments/
├── results/
├── figures/
└── README.md
```

### 3.2 辅助仿真平台：YAFS

仓库地址：

```bash
https://github.com/acsicuib/YAFS
```

YAFS 是 Python-based fog computing simulator，适合动态拓扑、节点故障、服务放置、用户移动和运行时编排。

#### 安装方式

```bash
cd SPARTA/simulators
git clone https://github.com/acsicuib/YAFS.git
cd YAFS

pip install -r requirements.txt
pip install -e .
```

如果依赖冲突，可以单独创建环境：

```bash
conda create -n yafs python=3.8 -y
conda activate yafs
pip install networkx pandas numpy matplotlib simpy
```

### 3.3 后续可参考：EdgeAISim

仓库地址：

```bash
https://github.com/MuhammedGolec/EdgeAISIM
```

EdgeAISim 基于 EdgeSimPy 扩展，支持 AI-based resource management、task scheduling、service migration、network flow scheduling 等。第一篇不建议直接以它为主平台，但可以参考它的任务迁移、流调度和强化学习资源管理代码。第三篇做闭环编排时可以再引入。

---

## 4. 真实 trace 与数据来源

第一篇不应只用完全随机仿真数据，最好加入真实 trace 驱动负载，使实验更可信。

### 4.1 Alibaba Cluster Data

仓库地址：

```bash
https://github.com/alibaba/clusterdata
```

推荐优先使用：

1. microservices trace：服务调用率、响应时间、服务依赖；
2. cluster-trace-v2018：机器资源、batch workload、任务 DAG；
3. GPU trace：后续扩展 AI 服务时可用。

#### 下载方式

```bash
mkdir -p data/traces/alibaba
cd data/traces/alibaba

git clone https://github.com/alibaba/clusterdata.git
```

部分数据可能需要从仓库 README 给出的外部链接下载。下载后建议统一整理为：

```text
data/traces/alibaba/
├── clusterdata/
├── microservices/
├── cluster-trace-v2018/
└── processed/
```

### 4.2 Google Cluster Data

仓库地址：

```bash
https://github.com/google/cluster-data
```

Google cluster trace 可用于构造真实负载波动和资源利用率模式。

#### 下载方式

```bash
mkdir -p data/traces/google
cd data/traces/google

git clone https://github.com/google/cluster-data.git
```

Google trace 通常体量较大，第一篇不需要全部下载。建议只抽取一小部分 machine events、task events、task usage 等数据，用于模拟请求到达率和节点负载。

### 4.3 SLA Violation Classification 参考代码

仓库地址：

```bash
https://github.com/ReyhaneAskari/SLA_violation_classification
```

这个仓库使用 Google Cloud Cluster trace 做 SLA violation classification，可作为传统机器学习 baseline 和数据处理参考。

#### 下载方式

```bash
cd SPARTA
git clone https://github.com/ReyhaneAskari/SLA_violation_classification.git third_party/SLA_violation_classification
```

使用方式：

- 参考其 SLA violation 标签构造；
- 参考随机森林、朴素贝叶斯等传统 baseline；
- 注意不要直接复刻，需要改成云边服务路径风险检测任务。

---

## 5. 基线模型代码

### 5.1 Time-Series-Library / TimesNet

仓库地址：

```bash
https://github.com/thuml/Time-Series-Library
```

安装：

```bash
cd SPARTA/third_party
git clone https://github.com/thuml/Time-Series-Library.git
cd Time-Series-Library
pip install -r requirements.txt
```

用途：

- TimesNet baseline；
- Transformer baseline；
- 时间序列分类/预测框架参考；
- 训练脚本参考。

### 5.2 PatchTST

仓库地址：

```bash
https://github.com/yuqinie98/PatchTST
```

安装：

```bash
cd SPARTA/third_party
git clone https://github.com/yuqinie98/PatchTST.git
```

用途：

- 强时序 Transformer baseline；
- 对比 SPARTA 不是普通时序 Transformer。

### 5.3 STGCN PyTorch

仓库地址：

```bash
https://github.com/FelixOpolka/STGCN-PyTorch
```

安装：

```bash
cd SPARTA/third_party
git clone https://github.com/FelixOpolka/STGCN-PyTorch.git
```

用途：

- 图时空 baseline；
- 证明你的 service-path subgraph 比全局图 STGCN 更适合 SLA 风险检测。

---

## 6. 数据处理方案

### 6.1 总体流程

```text
Alibaba / Google trace
        ↓
请求率、服务响应时间、节点负载模式提取
        ↓
注入 EdgeSimPy / YAFS 云边仿真环境
        ↓
生成节点、链路、服务、SLA 状态日志
        ↓
构造 service-path subgraph
        ↓
生成 normal / risky / violated 标签
        ↓
生成风险归因标签：risk node / risk link / risk metric
        ↓
训练 SPARTA 与 baselines
```

### 6.2 需要生成的原始日志

每个时间片保存以下信息：

#### 节点日志

```csv
time,node_id,cpu_util,mem_util,queue_len,available_cpu,available_mem,node_type
```

#### 链路日志

```csv
time,src,dst,delay,bandwidth,loss,jitter,queue_delay
```

#### 服务日志

```csv
time,service_id,user_id,src_node,edge_node,cloud_node,request_rate,response_time,service_type
```

#### SLA 日志

```csv
service_id,max_delay,max_loss,min_bandwidth,reliability_req,cost_weight,service_priority
```

#### 路径日志

```csv
time,service_id,path_nodes,path_links,path_delay,path_loss,bottleneck_link,bottleneck_node
```

### 6.3 样本格式

每一个训练样本对应：

```python
sample = {
    "service_id": int,
    "time_window": [t-L, ..., t],
    "node_features": Tensor[L, Np, Fn],
    "link_features": Tensor[L, Ep, Fe],
    "service_features": Tensor[L, Fs],
    "sla_features": Tensor[Fc],
    "adj_matrix": Tensor[Np, Np],
    "risk_label": int,       # 0 normal, 1 risky, 2 violated
    "risk_node": int,        # attribution label
    "risk_link": int,        # attribution label
    "risk_metric": int       # delay/loss/cpu/queue/bandwidth
}
```

其中 `Np` 表示服务路径子图中的节点数量，不是全网节点数量。

### 6.4 标签生成规则

假设服务 `s` 的最大时延要求为 `D_s`，最大丢包率为 `L_s`，未来预测窗口为 `H`。

#### violated 标签

如果未来 `H` 个时间片中存在：

```text
response_time >= D_s
or path_loss >= L_s
or service_failed == True
```

则标记为：

```text
violated
```

#### risky 标签

如果未来 `H` 个时间片中没有明确 violated，但存在：

```text
response_time >= 0.75 * D_s
or path_loss >= 0.75 * L_s
or queue_len keeps increasing
or CPU load >= 0.85
or bandwidth utilization >= 0.85
```

则标记为：

```text
risky
```

#### normal 标签

其余样本为：

```text
normal
```

注意：为了避免标签过于人工，建议同时使用趋势条件，例如最近 3 个时间片 delay 连续上升，或者 queue 连续上升。

### 6.5 风险归因标签生成

风险归因标签可以利用仿真环境自动生成。

#### risk node

如果服务路径上某节点满足：

```text
CPU load highest
or queue length highest
or available resource lowest
```

且该节点对 response_time 上升贡献最大，则作为 risk node。

#### risk link

如果路径上某链路满足：

```text
delay highest
or loss highest
or bandwidth utilization highest
```

则作为 risk link。

#### risk metric

定义 5 类：

```text
0 delay
1 packet loss
2 CPU overload
3 queue congestion
4 bandwidth bottleneck
```

根据超阈值最明显的指标生成标签。

---

## 7. 方法设计：SPARTA

### 7.1 模型结构

```text
Service-Path Subgraph
        ↓
Node/Link/Service/SLA Embedding
        ↓
Temporal Encoder
        ↓
Service-Path Topology Attention
        ↓
SLA-conditioned Fusion
        ↓
Risk Detection Head
        ↓
Attribution Head
```

### 7.2 模块一：Service-Path Subgraph Construction

对每个服务请求，只保留与该服务相关的子图：

1. 用户接入节点；
2. 当前服务部署边缘节点；
3. 备选边缘节点；
4. 通向云端的路径节点；
5. 与该服务竞争资源的相邻服务节点；
6. 路径上的链路。

这样可以避免全局网络建模过重，也能突出网络贡献。

### 7.3 模块二：Temporal Encoder

主模型可使用轻量 Transformer Encoder：

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=hidden_dim,
    nhead=4,
    dim_feedforward=hidden_dim * 4,
    dropout=0.1,
    batch_first=True
)
```

输入为过去 `L` 个时间片的路径状态序列。

### 7.4 模块三：Topology Attention

根据服务路径子图构造拓扑矩阵：

```python
A_ij = 1 / (dist(i, j) + 1)
```

将拓扑矩阵作为 attention bias：

```python
attention_score = QK^T / sqrt(d) + alpha * A
```

这样模型不是单纯时序 Transformer，而是带网络路径结构约束的风险检测模型。

### 7.5 模块四：SLA-conditioned Fusion

将 SLA 向量编码后与时空特征融合：

```python
sla_emb = MLP(sla_features)
z = temporal_feature + gamma * sla_emb
```

不同服务的风险判断由 SLA 条件驱动。

### 7.6 模块五：Risk Detection Head

输出：

```text
normal / risky / violated
```

损失函数：

```text
L_cls = CrossEntropy(y_risk, pred_risk)
```

若类别不平衡，使用 class-weighted cross entropy 或 focal loss。

### 7.7 模块六：Attribution Head

输出：

1. risk node；
2. risk link；
3. risk metric。

损失函数：

```text
L_attr = L_node + L_link + L_metric
```

总损失：

```text
L = L_cls + λ1 L_node + λ2 L_link + λ3 L_metric
```

---

## 8. 代码结构建议

```text
SPARTA/
├── configs/
│   ├── sparta_icassp.yaml
│   ├── baseline_lstm.yaml
│   ├── baseline_transformer.yaml
│   └── baseline_stgcn.yaml
├── data/
│   ├── raw/
│   ├── traces/
│   ├── processed/
│   └── sparta_dataset/
├── src/
│   ├── simulation/
│   │   ├── run_edgesimpy.py
│   │   ├── run_yafs.py
│   │   └── inject_trace.py
│   ├── preprocessing/
│   │   ├── build_path_graph.py
│   │   ├── generate_labels.py
│   │   ├── normalize.py
│   │   └── split_dataset.py
│   ├── datasets/
│   │   └── sparta_dataset.py
│   ├── models/
│   │   ├── sparta.py
│   │   ├── lstm.py
│   │   ├── transformer.py
│   │   ├── tcn.py
│   │   └── stgcn.py
│   ├── losses/
│   │   ├── focal_loss.py
│   │   └── multi_task_loss.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── scripts/
│   ├── 01_download_traces.sh
│   ├── 02_run_simulation.sh
│   ├── 03_preprocess.sh
│   ├── 04_train_baselines.sh
│   └── 05_train_sparta.sh
├── results/
├── figures/
└── README.md
```

---

## 9. 最小可运行版本代码框架

### 9.1 环境文件

`environment.yml`

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

### 9.2 数据生成命令

```bash
python src/simulation/run_edgesimpy.py \
  --num_edge_nodes 20 \
  --num_users 300 \
  --num_services 3 \
  --num_steps 10000 \
  --topology small_world \
  --trace_source alibaba \
  --output data/raw/edgesimpy_smallworld_20n_300u.csv
```

### 9.3 预处理命令

```bash
python src/preprocessing/build_path_graph.py \
  --input data/raw/edgesimpy_smallworld_20n_300u.csv \
  --output data/processed/path_graphs.pkl

python src/preprocessing/generate_labels.py \
  --input data/processed/path_graphs.pkl \
  --horizon 5 \
  --output data/sparta_dataset/dataset_h5.pkl
```

### 9.4 训练 baseline

```bash
python src/train.py \
  --config configs/baseline_lstm.yaml

python src/train.py \
  --config configs/baseline_transformer.yaml

python src/train.py \
  --config configs/baseline_stgcn.yaml
```

### 9.5 训练 SPARTA

```bash
python src/train.py \
  --config configs/sparta_icassp.yaml
```

### 9.6 评估

```bash
python src/evaluate.py \
  --checkpoint results/sparta/best.pth \
  --dataset data/sparta_dataset/test.pkl \
  --metrics macro_f1 risk_recall violation_recall attribution_acc auc
```

---

## 10. 实验设计

### 10.1 主实验

比较方法：

1. Random Forest；
2. XGBoost；
3. LSTM；
4. GRU；
5. TCN；
6. Transformer；
7. PatchTST；
8. STGCN；
9. SPARTA。

指标：

1. Accuracy；
2. Macro-F1；
3. Risk Recall；
4. Violation Recall；
5. AUC；
6. Attribution Accuracy；
7. Inference Time。

### 10.2 提前预警实验

预测窗口：

```text
H = 1, 3, 5, 10
```

重点展示：

```text
Risk Recall@H
Violation Recall@H
Macro-F1@H
```

ICASSP 审稿人会关心这个任务是否真有早期检测价值，所以这部分必须做。

### 10.3 消融实验

| 模型变体 | 目的 |
|---|---|
| SPARTA w/o SLA embedding | 验证 SLA 条件是否有效 |
| SPARTA w/o topology attention | 验证服务路径拓扑是否有效 |
| SPARTA w/o attribution head | 验证归因任务是否提升表示学习 |
| SPARTA global graph | 验证 service-path subgraph 是否优于全图建模 |
| SPARTA binary labels | 验证 risky 中间状态是否有必要 |

### 10.4 鲁棒性实验

| 场景 | 目的 |
|---|---|
| 高负载突发 | 检查突发请求下的风险召回 |
| 链路拥塞 | 检查 link attribution 是否准确 |
| 节点过载 | 检查 node attribution 是否准确 |
| 节点故障 | 检查拓扑变化下模型鲁棒性 |
| 不同拓扑 | 检查泛化性 |

### 10.5 主动干预小实验

为了让论文不仅是“预测”，建议加一个很小的主动干预验证。

规则：

```text
如果 risk node 是边缘节点过载 → 将服务迁移到邻近低负载节点
如果 risk link 是链路拥塞 → 切换到备选路径
如果 risk metric 是 loss → 选择低丢包路径
```

对比：

1. Reactive：SLA 已经 violated 后再调整；
2. Proactive：SPARTA 预测 risky 后提前调整。

指标：

1. SLA violation rate；
2. average response time；
3. migration cost；
4. number of interventions。

这个实验非常加分，但篇幅要控制，放成一小节即可。

---

## 11. ICASSP 论文结构

ICASSP 一般篇幅有限，建议采用 4 页正文 + 1 页参考文献的写法。结构建议如下：

### 1. Introduction

写 3 段：

第一段：云边服务需要低时延和 SLA 保障，但网络状态动态变化。

第二段：现有 SLA 预测多基于全局 QoS 或资源指标，缺少服务路径视角，也难以解释风险来源。

第三段：本文提出 SPARTA，将网络状态建模为服务路径信号，进行早期风险检测与归因。

贡献点写 3 条：

1. 提出 service-path network signal formulation；
2. 设计 SLA-conditioned temporal topology attention；
3. 实现 early detection and attribution，并通过仿真 + trace 验证。

### 2. Problem Formulation

定义：

1. 云边网络图；
2. 服务路径子图；
3. SLA 条件；
4. early risk detection；
5. attribution labels。

### 3. Proposed Method

介绍：

1. Service-path subgraph；
2. Temporal encoder；
3. Topology attention；
4. SLA-conditioned fusion；
5. Risk and attribution heads。

### 4. Experiments

包括：

1. dataset generation；
2. baselines；
3. main results；
4. ablation；
5. early warning；
6. attribution；
7. proactive intervention。

### 5. Conclusion

强调该方法为后续云边网络主动编排提供风险感知基础。

---

## 12. 创新点评估

### 12.1 如果只做普通预测，创新性不足

如果论文只是：

```text
EdgeSimPy 数据 + Transformer + SLA 三分类
```

创新性不够，容易被认为是 routine application。

### 12.2 ICASSP 版本必须做到的创新

要提高命中率，至少要做到：

1. **服务路径信号建模**  
   不是全局网络状态分类，而是针对每个服务构造 service-path subgraph。

2. **SLA 条件风险检测**  
   不同服务的 SLA 不同，模型必须显式输入 SLA condition。

3. **提前风险检测**  
   预测未来窗口内的风险，而不是判断当前是否违约。

4. **轻量风险归因**  
   输出风险来源节点、链路或指标，为后续调度提供解释。

5. **主动干预验证**  
   用一个简单规则证明提前预警能减少 SLA violation rate。

### 12.3 创新等级判断

| 版本 | 内容 | 创新性 | 是否建议投 ICASSP |
|---|---|---|---|
| A | Transformer 三分类 | 偏弱 | 不建议 |
| B | SLA-aware early prediction | 一般 | 可试，但不稳 |
| C | Service-path + SLA-aware + early detection | 中等 | 可以 |
| D | C + attribution + proactive intervention | 中等偏强 | 最推荐 |

建议按照 D 版做，但论文中控制复杂度，不要写成大系统论文。

---

## 13. 时间规划

假设现在开始做，建议 12 周完成初稿。

### 第 1–2 周：环境与仿真

1. 跑通 EdgeSimPy；
2. 搭建 10–20 个边缘节点的拓扑；
3. 输出节点、链路、服务日志；
4. 整理 Alibaba / Google trace 的负载输入。

### 第 3–4 周：数据集构造

1. 构造 service-path subgraph；
2. 生成 normal / risky / violated 标签；
3. 生成 risk node / link / metric 标签；
4. 完成 train/val/test split。

### 第 5–6 周：baseline

1. Random Forest；
2. XGBoost；
3. LSTM；
4. Transformer；
5. STGCN / PatchTST。

### 第 7–8 周：SPARTA

1. 实现 temporal encoder；
2. 实现 topology attention；
3. 实现 SLA embedding；
4. 实现 attribution head；
5. 训练初版。

### 第 9–10 周：实验补全

1. 主实验；
2. 消融；
3. 提前预警；
4. 鲁棒性；
5. 归因；
6. 主动干预。

### 第 11–12 周：写论文

1. 写 introduction；
2. 画方法图；
3. 整理表格；
4. 写实验分析；
5. 压缩到 ICASSP 篇幅；
6. 找导师或同学预审。

---

## 14. 图表清单

### 图 1：Overall Framework

展示：

```text
Cloud-edge network signals
→ service-path subgraph
→ temporal topology encoder
→ SLA-conditioned fusion
→ risk detection + attribution
→ proactive action
```

### 图 2：Service-Path Subgraph

展示用户、接入边缘节点、候选边缘节点、云节点、路径链路、竞争服务。

### 表 1：Dataset Statistics

包括：

1. 拓扑规模；
2. 用户数量；
3. 服务类型；
4. 时间片数量；
5. normal/risky/violated 数量；
6. 风险来源分布。

### 表 2：Main Results

比较所有 baseline。

### 表 3：Ablation Study

展示各模块贡献。

### 图 3：Prediction Horizon

展示不同提前窗口下 Risk Recall / Macro-F1。

### 图 4：Attribution Visualization

展示风险节点、风险链路和注意力响应。

---

## 15. 投稿风险控制

### 风险 1：网络题不适合 ICASSP

应对：

不要强调“边缘计算调度”，而要强调“network signals, temporal detection, graph signal modeling, attribution”。

### 风险 2：创新不够

应对：

必须保留 service-path subgraph、SLA-conditioned fusion、risk attribution、early warning 这四个点。

### 风险 3：仿真数据说服力弱

应对：

引入 Alibaba / Google trace 作为负载驱动；使用多拓扑、多故障、多负载；公开代码和数据生成脚本。

### 风险 4：工作量过大

应对：

第一篇不做复杂强化学习，不做完整云边编排，只做早期检测、归因和轻量干预验证。

### 风险 5：篇幅不够

应对：

主文只讲核心方法和关键实验，更多实现细节放 GitHub 或 supplementary。

---

## 16. 推荐最终执行路线

最终建议采用：

### 论文题目

**SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services**

### 任务形式

多变量网络状态信号的早期风险检测与归因。

### 主平台

EdgeSimPy。

### 补充平台

YAFS。

### 真实 trace

Alibaba microservices / cluster trace + Google cluster trace。

### Baseline

Random Forest、XGBoost、LSTM、GRU、TCN、Transformer、PatchTST、STGCN。

### 创新点

1. service-path network signal formulation；
2. SLA-conditioned temporal topology attention；
3. early risk detection；
4. lightweight risk attribution；
5. proactive intervention validation。

### 投稿目标

ICASSP。

### 后续扩展

将 ICASSP 版本扩展为 Computer Networks / PR 期刊版，再进一步扩展为意图驱动云边闭环编排的 CCF A 工作。

---

## 17. 参考链接

1. ICASSP 2027 IEEE SPS event page  
   https://signalprocessingsociety.org/events/2027-ieee-international-conference-acoustics-speech-and-signal-processing-icassp

2. ICASSP 2026 paper topics  
   https://cmsworkshops.com/ICASSP2026/papers/paper_topics.php

3. CCF 推荐国际学术会议和期刊目录（2026）  
   https://ccf.atom.im/

4. EdgeSimPy  
   https://github.com/EdgeSimPy/EdgeSimPy

5. EdgeSimPy documentation  
   https://edgesimpy.github.io/

6. YAFS  
   https://github.com/acsicuib/YAFS

7. EdgeAISim  
   https://github.com/MuhammedGolec/EdgeAISIM

8. Alibaba Cluster Data  
   https://github.com/alibaba/clusterdata

9. Google Cluster Data  
   https://github.com/google/cluster-data

10. SLA Violation Classification  
    https://github.com/ReyhaneAskari/SLA_violation_classification

11. Time-Series-Library  
    https://github.com/thuml/Time-Series-Library

12. PatchTST  
    https://github.com/yuqinie98/PatchTST

13. STGCN PyTorch  
    https://github.com/FelixOpolka/STGCN-PyTorch
