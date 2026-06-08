# ICASSP 第一篇论文完整可执行报告：SPARTA —— 面向云边服务的 SLA 风险早期检测与服务路径归因

> 目标：在没有 AI 数据中心、没有 GPU 集群、主要依赖 AutoDL 租卡、代码能力一般但熟悉 LSTM/Transformer/DETR/检测思维的条件下，设计一篇**不空泛、可落地、具备 CCF B 会议工作量与创新性的 ICASSP 论文**。

---

## 0. 先给结论

### 0.1 推荐第一篇论文方向

**英文题目建议：**

**SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services**

**中文题目建议：**

**面向云边服务的服务路径感知 SLA 风险早期检测与归因方法**

### 0.2 论文定位

不要把论文写成：

> 边缘计算任务卸载 / 云边资源调度 / 网络优化算法

而要写成：

> **云边网络状态是一类多变量时空信号，SLA 风险检测是一个面向网络信号的早期检测、分类与归因问题。**

这更符合 ICASSP 的信号处理、机器学习、时序分析、检测分类、图信号建模口味。

### 0.3 是否能保证中？

不能保证任何会议一定录用。但这份方案可以保证三点：

1. **不是普通 Transformer 套壳**：核心是 service-path network signal formulation + SLA-conditioned temporal topology modeling + risk attribution；
2. **工作量达到 ICASSP 正常论文强度**：有仿真平台、真实 trace 驱动、多个 baseline、消融、早期预警、归因、主动干预验证；
3. **后续可扩展**：ICASSP 版本可自然扩展成 Computer Networks / PR / Neural Networks 等 CCF B 期刊版。

---

## 1. 为什么第一篇投 ICASSP 更合适

### 1.1 和你当前能力匹配

你已有基础包括：

- LSTM / Transformer：可直接用于网络状态时间序列建模；
- DETR 思想：可迁移成 risk query 或 attribution head；
- YOLO / 检测思维：可迁移成风险事件检测；
- CV 论文经验：可用于模型命名、消融实验、可视化和图表包装。

所以第一篇应该优先做：

> **风险检测 + 时序建模 + 归因解释**

而不是一开始做复杂资源调度或强化学习。

### 1.2 ICASSP 的适配口径

ICASSP 2026 的 session grid 中包含 MLSP（Machine Learning for Signal Processing）和 SPCOM（Signal Processing for Comms and Net）等方向；ICASSP 2027 的 IEEE SPS 页面显示会议地点为 Toronto，会议时间为 2027 年 5 月 16–21 日，并列出 paper submission deadline 为 2026 年 9 月 16 日。投稿前仍需再次核对官网，因为不同页面可能存在日期展示不一致的问题。

参考链接：

- ICASSP 2026 session grid: https://2026.ieeeicassp.org/session-grid/
- ICASSP 2027 IEEE SPS event page: https://signalprocessingsociety.org/events/2027-ieee-international-conference-acoustics-speech-and-signal-processing-icassp
- ICASSP 2027 official site: https://2027.ieeeicassp.org/

### 1.3 和 Computer Networks 的关系

Computer Networks 更适合做第二篇扩展期刊。第一篇 ICASSP 做短而清楚的早期风险检测与归因；后续扩展为：

- 更多拓扑；
- 更多真实 trace；
- 更完整主动干预；
- EdgeSimPy + YAFS 双平台；
- 更完整的服务迁移、路径切换和编排验证。

---

## 2. 论文核心问题定义

### 2.1 研究对象

云边协同服务场景中，一个服务请求从用户侧进入网络，经由接入节点、边缘节点、候选边缘节点，必要时到达云节点。服务质量受以下因素共同影响：

- 节点 CPU / memory / queue；
- 链路 delay / bandwidth / loss / jitter；
- 服务 request rate / response time；
- 服务路径 hop count / path delay / bottleneck；
- SLA 要求 max delay / max loss / min bandwidth / reliability。

### 2.2 输入

给定一个服务 `s` 在过去 `L` 个时间片内的 service-path subgraph：

```text
G_s(t-L:t) = (V_s, E_s, X_v, X_e, X_service, C_s)
```

其中：

- `V_s`：服务相关节点，包括 user access node、current edge node、candidate edge nodes、cloud node；
- `E_s`：服务路径链路；
- `X_v`：节点状态序列；
- `X_e`：链路状态序列；
- `X_service`：服务状态序列；
- `C_s`：SLA 条件向量。

### 2.3 输出

模型输出四类结果：

1. 风险等级：`normal / risky / violated`；
2. 风险概率：`p(normal), p(risky), p(violated)`；
3. 风险来源：`risk node / risk link`；
4. 风险主导指标：`delay / loss / CPU / queue / bandwidth`。

### 2.4 任务本质

这不是普通三分类，而是：

> **service-path network signal early detection and attribution**

即：面向服务路径网络信号的早期风险检测与归因。

---

## 3. 论文必须具备的 5 个创新点

### 创新点 1：Service-Path Network Signal Formulation

#### 内容

现有很多预测方法直接使用全局网络状态或单一 QoS 序列。本文不直接把全网状态丢进模型，而是围绕每个服务构造 service-path subgraph，把服务实际经过的节点、链路、候选边缘节点和竞争服务纳入局部子图。

#### 创新性评估

- 创新等级：中等偏强；
- 原因：它把问题从普通时间序列分类提升为服务路径条件下的网络信号建模；
- ICASSP 兼容性：强，符合 graph signal / structured time-series modeling；
- 和模型兼容性：强，可作为输入构造方式，不依赖复杂模型。

#### 必做程度

**必须做。** 如果不做这个，论文会退化成普通 Transformer 分类。

---

### 创新点 2：SLA-Conditioned Temporal Topology Attention

#### 内容

不同服务的 SLA 不同。低时延服务更关注 delay，高可靠服务更关注 loss 和 reliability，低成本服务更关注资源成本。模型需要把 SLA 条件显式输入，而不是只看网络状态。

本文设计：

```text
Temporal Encoder + Topology Attention + SLA-conditioned Fusion
```

#### 创新性评估

- 创新等级：中等；
- 原因：SLA 条件建模能说明模型不是泛化时序分类，而是面向服务质量保障；
- ICASSP 兼容性：强，可包装为 conditional signal modeling；
- 和模型兼容性：强，只需要 MLP 编码 SLA 向量，然后与时空特征融合。

#### 必做程度

**必须做。** 这是论文区别于普通时序模型的关键。

---

### 创新点 3：Early Risk Detection Instead of Current-State Classification

#### 内容

模型不是判断当前是否违约，而是给定过去 `L` 个时间片，预测未来 `H` 个时间片内是否出现风险。

例如：

```text
Input: t-12 到 t 的网络状态
Output: t+5 之前是否会 risky / violated
```

#### 创新性评估

- 创新等级：中等；
- 原因：early warning 比 current classification 更有网络管理价值；
- ICASSP 兼容性：强，符合 early detection / event prediction；
- 和模型兼容性：强，只需要调整标签生成方式。

#### 必做程度

**必须做。** 这直接决定论文是不是“有用”。

---

### 创新点 4：Lightweight Risk Attribution

#### 内容

模型不仅输出风险等级，还输出：

- 哪个节点导致风险；
- 哪条链路导致风险；
- 哪个指标导致风险。

这可以为后续主动干预提供依据。

#### 创新性评估

- 创新等级：中等偏强；
- 原因：多数短论文只做分类，归因能显著增强方法完整性；
- ICASSP 兼容性：强，可包装为 interpretable attribution for network signals；
- 和模型兼容性：中等，需要多任务 head 和归因标签生成。

#### 必做程度

**强烈建议做。** 如果删掉，论文创新会明显下降。

---

### 创新点 5：Proactive Intervention Validation

#### 内容

当模型预测风险时，执行一个简单规则：

- 风险节点过载 → 迁移到邻近低负载节点；
- 风险链路拥塞 → 切换备选路径；
- 丢包风险 → 选择低 loss 路径；
- 带宽瓶颈 → 选择剩余带宽更高路径。

比较：

- Reactive：已经 violated 后再处理；
- Proactive：risky 阶段提前处理。

#### 创新性评估

- 创新等级：中等；
- 原因：证明检测不是为了分类，而是能减少违约；
- ICASSP 兼容性：中等，放小实验即可，不能写成大系统；
- 和模型兼容性：强，不需要训练复杂调度器。

#### 必做程度

**建议做。** 它是提升命中率的加分项。

---

## 4. 最终方法：SPARTA

### 4.1 方法全称

**SPARTA: Service-Path Aware Temporal Risk Attribution**

### 4.2 整体流程

```text
Alibaba / Google Trace
        ↓
Trace-driven workload generation
        ↓
EdgeSimPy cloud-edge simulation
        ↓
Node / link / service / SLA logs
        ↓
Service-path subgraph construction
        ↓
SPARTA model
        ↓
Risk detection + risk attribution
        ↓
Proactive intervention validation
```

### 4.3 模型结构

```text
Service-path subgraph input
        ↓
Node / link / service embedding
        ↓
Temporal encoder
        ↓
Topology-aware attention bias
        ↓
SLA-conditioned fusion
        ↓
Risk detection head
        ↓
Attribution heads
```

---

## 5. 代码与平台：下载位置和如何改

### 5.1 主仿真平台：EdgeSimPy

#### 地址

```bash
https://github.com/EdgeSimPy/EdgeSimPy
```

EdgeSimPy 是 Python-based edge computing simulator，提供 edge servers、network devices、applications、user mobility、application composition、power consumption 等抽象，适合快速构造云边环境。

参考：

- GitHub: https://github.com/EdgeSimPy/EdgeSimPy
- Documentation: https://edgesimpy.github.io/

#### 下载

```bash
mkdir -p ~/SPARTA/simulators
cd ~/SPARTA/simulators

git clone https://github.com/EdgeSimPy/EdgeSimPy.git
cd EdgeSimPy
pip install -r requirements.txt
pip install -e .
```

#### 如何改

你不需要大改 EdgeSimPy 核心源码。建议在外部写 wrapper：

```text
SPARTA/src/simulation/run_edgesimpy.py
SPARTA/src/simulation/export_logs.py
SPARTA/src/simulation/inject_trace.py
```

需要增加的功能：

1. 每个时间片导出节点状态；
2. 每个时间片导出链路状态；
3. 每个服务导出 response time；
4. 保存服务路径；
5. 注入 Alibaba / Google trace 的 request rate；
6. 人为设置拥塞、故障、突发负载场景。

最小改法：

- 不动 EdgeSimPy 内部类；
- 在 simulation loop 里读取对象属性；
- 保存为 CSV。

---

### 5.2 辅助仿真平台：YAFS

#### 地址

```bash
https://github.com/acsicuib/YAFS
```

YAFS 支持动态拓扑、动态消息源、运行时 placement 和 orchestration，适合后续扩展期刊版或第三篇闭环编排。

参考：

- GitHub: https://github.com/acsicuib/YAFS
- Documentation: https://yafs.readthedocs.io/en/latest/

#### 下载

```bash
cd ~/SPARTA/simulators

git clone https://github.com/acsicuib/YAFS.git
cd YAFS
pip install -r requirements.txt
pip install -e .
```

#### 如何改

ICASSP 第一篇不建议主用 YAFS。只做补充：

1. 节点故障场景；
2. 链路动态变化场景；
3. 服务路径变更场景。

如果时间紧，YAFS 可以暂时不做，放在期刊扩展版。

---

### 5.3 Time-Series-Library / TimesNet

#### 地址

```bash
https://github.com/thuml/Time-Series-Library
```

Time-Series-Library 支持 long-term forecasting、short-term forecasting、imputation、anomaly detection、classification 等任务，可用于 TimesNet、Transformer 等 baseline。

参考：

- https://github.com/thuml/Time-Series-Library

#### 下载

```bash
mkdir -p ~/SPARTA/third_party
cd ~/SPARTA/third_party

git clone https://github.com/thuml/Time-Series-Library.git
cd Time-Series-Library
pip install -r requirements.txt
```

#### 如何改

不建议把你的主代码完全塞进它。建议做两种用途：

1. 直接跑 TimesNet / Transformer baseline；
2. 参考其数据加载格式，实现自己的 `SPARTADataset`。

你的主模型仍放在：

```text
SPARTA/src/models/sparta.py
```

---

### 5.4 PatchTST

#### 地址

```bash
https://github.com/yuqinie98/PatchTST
```

PatchTST 是 ICLR 2023 时间序列 Transformer 方法，可作为强 baseline。

#### 下载

```bash
cd ~/SPARTA/third_party

git clone https://github.com/yuqinie98/PatchTST.git
```

#### 如何改

PatchTST 原本偏 forecasting。你可以有两种用法：

1. 预测未来 response time，再根据 SLA 阈值转成 normal/risky/violated；
2. 把最后的 forecasting head 改成 classification head。

建议第一篇采用第 2 种，更直接。

---

### 5.5 STGCN PyTorch

#### 地址

```bash
https://github.com/FelixOpolka/STGCN-PyTorch
```

STGCN 原本用于交通预测，可作为图时空 baseline。

#### 下载

```bash
cd ~/SPARTA/third_party

git clone https://github.com/FelixOpolka/STGCN-PyTorch.git
```

#### 如何改

把交通节点替换成云边节点；把交通速度/流量替换成节点负载、链路 delay、queue 等特征。

STGCN baseline 用全局图即可，这样可以证明：

> SPARTA 的 service-path subgraph 比全局图建模更适合服务级 SLA 风险检测。

---

## 6. 数据集：下载位置和如何处理

### 6.1 Alibaba ClusterData

#### 地址

```bash
https://github.com/alibaba/clusterdata
```

其中 `cluster-trace-microservices-v2021` 包含大规模 microservices runtime metrics，官方说明包含 nearly twenty thousand microservices，来自 Alibaba production clusters，持续 12 小时，包括 call dependencies、response time、call rates 等。

参考：

- https://github.com/alibaba/clusterdata
- https://github.com/alibaba/clusterdata/blob/master/cluster-trace-microservices-v2021/README.md

#### 下载

```bash
mkdir -p ~/SPARTA/data/traces/alibaba
cd ~/SPARTA/data/traces/alibaba

git clone https://github.com/alibaba/clusterdata.git
```

部分大文件可能不直接在 git 仓库，需要根据 README 里的外部链接下载。

#### 如何处理

重点提取：

1. 服务调用率 call rate；
2. 响应时间 response time；
3. 服务依赖 dependency；
4. 时间变化趋势。

处理成：

```csv
time,service_id,request_rate,base_response_time,dependency_count
```

然后注入 EdgeSimPy：

- `request_rate` 控制用户请求到达；
- `base_response_time` 用作服务处理时间基准；
- `dependency_count` 用于设置服务复杂度或路径长度。

---

### 6.2 Google ClusterData

#### 地址

```bash
https://github.com/google/cluster-data
```

Google ClusterData 包含 Borg cluster workload traces，可用于模拟真实 workload pattern。

#### 下载

```bash
mkdir -p ~/SPARTA/data/traces/google
cd ~/SPARTA/data/traces/google

git clone https://github.com/google/cluster-data.git
```

#### 如何处理

重点提取：

1. task arrival；
2. resource usage；
3. CPU / memory usage patterns；
4. workload burst。

处理成：

```csv
time,node_group,cpu_pattern,mem_pattern,arrival_rate
```

用途：

- 驱动边缘节点负载；
- 构造突发流量；
- 证明负载模式不是完全随机生成。

---

## 7. 数据生成完整流程

### 7.1 项目目录

建议目录：

```text
SPARTA/
├── configs/
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
│   ├── preprocessing/
│   ├── datasets/
│   ├── models/
│   ├── losses/
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── scripts/
├── results/
├── figures/
└── README.md
```

### 7.2 第一步：trace 预处理

脚本：

```text
src/preprocessing/prepare_alibaba_trace.py
src/preprocessing/prepare_google_trace.py
```

输出：

```text
data/processed/alibaba_service_load.csv
data/processed/google_node_load.csv
```

格式：

```csv
time,service_type,request_rate,response_time_scale
```

```csv
time,node_group,cpu_scale,mem_scale,arrival_scale
```

### 7.3 第二步：EdgeSimPy 仿真

脚本：

```text
src/simulation/run_edgesimpy.py
```

命令：

```bash
python src/simulation/run_edgesimpy.py \
  --num_edge_nodes 20 \
  --num_users 300 \
  --num_services 3 \
  --num_steps 10000 \
  --topology small_world \
  --trace_source alibaba \
  --output_dir data/raw/smallworld_20n_300u
```

输出：

```text
data/raw/smallworld_20n_300u/node_log.csv
data/raw/smallworld_20n_300u/link_log.csv
data/raw/smallworld_20n_300u/service_log.csv
data/raw/smallworld_20n_300u/path_log.csv
data/raw/smallworld_20n_300u/sla_log.csv
```

### 7.4 第三步：构造 service-path subgraph

脚本：

```text
src/preprocessing/build_path_graph.py
```

命令：

```bash
python src/preprocessing/build_path_graph.py \
  --input_dir data/raw/smallworld_20n_300u \
  --window 12 \
  --output data/processed/path_graphs_w12.pkl
```

每个样本：

```python
{
  "node_features": [L, Np, Fn],
  "link_features": [L, Ep, Fe],
  "service_features": [L, Fs],
  "sla_features": [Fc],
  "adj_matrix": [Np, Np],
  "path_links": [Ep],
  "service_id": int,
  "time": int
}
```

### 7.5 第四步：生成风险标签

脚本：

```text
src/preprocessing/generate_labels.py
```

命令：

```bash
python src/preprocessing/generate_labels.py \
  --input data/processed/path_graphs_w12.pkl \
  --horizon 5 \
  --output data/sparta_dataset/sparta_w12_h5.pkl
```

标签规则：

#### violated

未来 `H` 个时间片中满足任一条件：

```text
response_time >= max_delay
path_loss >= max_loss
service_failed == True
```

#### risky

未来 `H` 个时间片未 violated，但满足任一条件：

```text
response_time >= 0.75 * max_delay
path_loss >= 0.75 * max_loss
CPU load >= 0.85
queue_len 连续上升
bandwidth_util >= 0.85
```

#### normal

其他情况。

### 7.6 第五步：生成归因标签

脚本：

```text
src/preprocessing/generate_attribution.py
```

规则：

- `risk_node`：路径中 CPU load、queue_len、available_cpu 最异常的节点；
- `risk_link`：路径中 delay、loss、bandwidth_util 最异常的链路；
- `risk_metric`：delay / loss / CPU / queue / bandwidth 中超阈值最明显的指标。

输出：

```python
risk_label: 0/1/2
risk_node: node index in subgraph
risk_link: link index in subgraph
risk_metric: 0 delay / 1 loss / 2 cpu / 3 queue / 4 bandwidth
```

---

## 8. SPARTA 具体模型实现

### 8.1 输入张量

```python
node_x:    [B, L, Np, Fn]
link_x:    [B, L, Ep, Fe]
service_x: [B, L, Fs]
sla_x:     [B, Fc]
adj:       [B, Np, Np]
```

建议初始参数：

```text
L = 12
H = 5
hidden_dim = 128
n_heads = 4
num_layers = 2
dropout = 0.1
```

### 8.2 Embedding

```python
node_emb = MLP(node_x)
link_emb = MLP(link_x)
service_emb = MLP(service_x)
sla_emb = MLP(sla_x)
```

融合方式：

```python
path_token = mean_pool(node_emb) + mean_pool(link_emb) + service_emb
```

### 8.3 Temporal Encoder

使用 Transformer Encoder：

```python
nn.TransformerEncoderLayer(
    d_model=128,
    nhead=4,
    dim_feedforward=512,
    dropout=0.1,
    batch_first=True
)
```

### 8.4 Topology Attention

构造拓扑距离矩阵：

```python
A_ij = 1 / (shortest_path_dist(i, j) + 1)
```

加入 attention bias：

```python
score = Q @ K.T / sqrt(d) + alpha * A
```

如果实现复杂，第一版可以先做简化：

```python
topo_feature = GCN(adj, node_feature)
z = temporal_feature + topo_feature
```

### 8.5 SLA-conditioned Fusion

```python
z = z + gamma * sla_emb
```

或者：

```python
gate = sigmoid(MLP(sla_emb))
z = z * gate + z
```

建议使用 gate，因为更容易写成“条件调制”。

### 8.6 Heads

#### 风险检测 head

```python
risk_logits = Linear(z, 3)
```

#### 节点归因 head

```python
node_logits = Linear(node_features, 1)  # [B, Np]
```

#### 链路归因 head

```python
link_logits = Linear(link_features, 1)  # [B, Ep]
```

#### 指标归因 head

```python
metric_logits = Linear(z, 5)
```

### 8.7 损失函数

```text
L = L_risk + λ1 L_node + λ2 L_link + λ3 L_metric
```

建议：

```text
λ1 = 0.3
λ2 = 0.3
λ3 = 0.2
```

风险分类使用 focal loss 或 weighted cross entropy，防止 normal 类过多。

---

## 9. Baseline 设计

### 9.1 必须做的 baseline

| 类型 | 方法 | 作用 |
|---|---|---|
| 传统机器学习 | Random Forest | 低门槛传统基线 |
| 传统机器学习 | XGBoost | 强传统基线 |
| 时序模型 | LSTM | 你熟悉，必须有 |
| 时序模型 | GRU | 轻量 recurrent baseline |
| 卷积时序 | TCN | 非 recurrent 时序基线 |
| Transformer | Vanilla Transformer | 证明不是普通 Transformer |
| 强时序模型 | PatchTST / TimesNet | 提升说服力 |
| 图时空模型 | STGCN | 证明图结构建模必要性 |
| 本文 | SPARTA | 主方法 |

### 9.2 baseline 输入统一

为了公平，所有深度模型使用相同输入窗口 `L=12`，预测同一 horizon `H=5`。

非图模型把 service-path features flatten 成多变量序列。

图模型使用全局拓扑或路径子图。

---

## 10. 实验设计

### 10.1 主结果实验

表格列：

```text
Method | Acc | Macro-F1 | Risk Recall | Violation Recall | AUC | Attr-Acc | Time(ms)
```

重点不是 Accuracy，而是：

- Macro-F1；
- Risk Recall；
- Violation Recall；
- Attribution Accuracy。

### 10.2 Early Detection 实验

预测窗口：

```text
H = 1, 3, 5, 10
```

展示：

```text
Risk Recall@H
Macro-F1@H
Violation Recall@H
```

这部分证明：模型能提前发现风险。

### 10.3 消融实验

| 变体 | 验证点 |
|---|---|
| w/o SLA | SLA 条件是否有效 |
| w/o topology | 拓扑结构是否有效 |
| w/o attribution | 归因任务是否促进学习 |
| global graph | service-path 子图是否更优 |
| binary label | risky 中间状态是否必要 |

### 10.4 鲁棒性实验

| 场景 | 目的 |
|---|---|
| high load burst | 突发请求下是否稳定 |
| link congestion | 链路风险是否能识别 |
| node overload | 节点风险是否能识别 |
| node failure | 拓扑变化是否鲁棒 |
| different topology | 泛化能力 |

### 10.5 主动干预验证

对比两种策略：

#### Reactive

```text
只有 violated 后才迁移 / 换路
```

#### Proactive-SPARTA

```text
SPARTA 预测 risky 后提前迁移 / 换路
```

指标：

```text
SLA violation rate
Average response time
Migration cost
Number of interventions
```

这个实验只需要一小节，但非常关键，因为它证明：

> SPARTA 的风险检测具有实际网络管理价值。

---

## 11. 论文结构：ICASSP 4 页写法

### 11.1 Abstract

四句话：

1. 云边服务 SLA 保障困难；
2. 现有方法忽视服务路径信号和风险来源；
3. 提出 SPARTA，进行早期检测与归因；
4. 实验证明优于 baseline，并降低 proactive intervention 下的 violation rate。

### 11.2 Introduction

三段即可：

第一段：云边服务质量保障背景。

第二段：现有 SLA prediction / time-series methods 的不足：

- global QoS records；
- no service-path formulation；
- no attribution；
- limited early warning validation。

第三段：本文提出 SPARTA，并列贡献。

贡献建议写：

```text
1. We formulate SLA violation early detection as service-path network signal modeling.
2. We propose an SLA-conditioned temporal topology attention model for risk detection.
3. We introduce lightweight node/link/metric attribution and validate proactive intervention.
```

### 11.3 Problem Formulation

定义：

- cloud-edge graph；
- service-path subgraph；
- SLA condition；
- risk labels；
- attribution labels。

### 11.4 Method

只写核心模块：

1. Service-path representation；
2. Temporal topology encoder；
3. SLA-conditioned fusion；
4. Risk and attribution heads。

### 11.5 Experiments

篇幅有限，必须集中展示：

1. Dataset setting；
2. Main comparison；
3. Ablation；
4. Early horizon；
5. Attribution / proactive intervention。

### 11.6 Conclusion

一句话：SPARTA 为后续云边服务主动编排提供了风险感知基础。

---

## 12. 图表清单

### 图 1：Overall Framework

必须有。展示：

```text
Trace-driven cloud-edge simulation
→ service-path subgraph
→ SPARTA encoder
→ risk detection + attribution
→ proactive action
```

### 图 2：Service-Path Subgraph

展示：

- user；
- access node；
- edge node；
- candidate edge；
- cloud；
- bottleneck link；
- competing services。

### 表 1：Dataset Statistics

包含：

```text
topology | edge nodes | users | services | samples | normal/risky/violated | risk node/link/metric distribution
```

### 表 2：Main Results

包含所有 baseline。

### 表 3：Ablation

只保留 4–5 行即可。

### 图 3：Prediction Horizon

展示 H=1/3/5/10 的曲线。

### 图 4：Attribution Visualization

展示风险链路和节点被模型定位。

---

## 13. 12 周执行计划

### 第 1 周：环境搭建

完成：

```bash
conda create -n sparta python=3.10 -y
conda activate sparta
pip install numpy pandas scipy scikit-learn networkx matplotlib tqdm pyyaml xgboost einops
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

下载：

- EdgeSimPy；
- Alibaba ClusterData；
- Time-Series-Library；
- PatchTST；
- STGCN。

### 第 2 周：跑通 EdgeSimPy 最小仿真

目标：

- 10 个 edge nodes；
- 100 users；
- 3 类 services；
- 输出 node/link/service/path logs。

### 第 3 周：接入 trace

目标：

- Alibaba trace 生成 request_rate；
- Google trace 生成 node load pattern；
- 注入 EdgeSimPy。

### 第 4 周：数据集生成

目标：

- service-path subgraph；
- normal/risky/violated 标签；
- risk node/link/metric 标签；
- train/val/test split。

### 第 5 周：传统 baseline

完成：

- Random Forest；
- XGBoost；
- LSTM；
- GRU。

### 第 6 周：强 baseline

完成：

- TCN；
- Vanilla Transformer；
- PatchTST 或 TimesNet；
- STGCN。

### 第 7 周：SPARTA 初版

完成：

- embedding；
- temporal encoder；
- SLA fusion；
- risk head。

### 第 8 周：SPARTA 完整版

完成：

- topology attention；
- attribution heads；
- multi-task loss。

### 第 9 周：主实验和消融

完成：

- main table；
- ablation table；
- H=1/3/5/10 early detection。

### 第 10 周：鲁棒性和主动干预

完成：

- load burst；
- link congestion；
- node overload；
- proactive vs reactive。

### 第 11 周：论文写作

完成：

- abstract；
- introduction；
- method；
- experiments；
- figures。

### 第 12 周：压缩与投稿版

完成：

- 4 页正文压缩；
- 图表重排；
- 参考文献；
- 代码 README；
- 给导师预审。

---

## 14. 是否达到 ICASSP 创新性和工作量

### 14.1 不达标版本

```text
EdgeSimPy + Transformer + 三分类
```

结论：不稳，像普通应用。

### 14.2 达标版本

```text
service-path subgraph + SLA-conditioned temporal topology attention + early detection
```

结论：具备 ICASSP 可投创新。

### 14.3 推荐版本

```text
service-path subgraph
+ SLA-conditioned temporal topology attention
+ early risk detection
+ lightweight attribution
+ proactive intervention validation
```

结论：这是推荐执行版本。创新性、工作量和后续扩展性都比较合理。

---

## 15. 与模型兼容性总表

| 模块 | 代码难度 | 创新贡献 | 是否兼容 SPARTA | 是否必须 |
|---|---:|---:|---:|---:|
| service-path subgraph | 中 | 高 | 强 | 必须 |
| SLA embedding | 低 | 中 | 强 | 必须 |
| temporal encoder | 低 | 中 | 强 | 必须 |
| topology attention | 中 | 高 | 强 | 必须 |
| attribution head | 中 | 高 | 强 | 强烈建议 |
| proactive intervention | 低 | 中 | 强 | 建议 |
| complex RL scheduling | 高 | 不适合第一篇 | 弱 | 不做 |
| LLM intent parsing | 中高 | 偏题 | 一般 | 不做 |

---

## 16. 最终执行建议

第一篇 ICASSP 不要追求“大而全”。真正稳的做法是：

1. 把问题包装成 network signal early detection；
2. 数据用 EdgeSimPy 仿真，但用 Alibaba / Google trace 驱动负载；
3. 方法用 SPARTA，强调 service-path、SLA-conditioned、early detection、attribution；
4. 实验必须有 baseline、消融、early horizon、归因、主动干预；
5. 不做复杂强化学习，不做完整闭环编排，不做大模型意图理解。

最终投稿题目建议：

**SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services**

这篇做成以后，不管 ICASSP 是否录用，都可以扩展成第二篇 CCF B 期刊：

**Proactive SLA Risk Detection and Attribution for Cloud-Edge Service Management**

目标可转向 Computer Networks / PR / Neural Networks。

---

## 17. 参考链接

1. ICASSP 2026 Session Grid: https://2026.ieeeicassp.org/session-grid/
2. ICASSP 2027 IEEE SPS Event Page: https://signalprocessingsociety.org/events/2027-ieee-international-conference-acoustics-speech-and-signal-processing-icassp
3. ICASSP 2027 Official Website: https://2027.ieeeicassp.org/
4. CCF 2026 Directory Mirror: https://ccf.atom.im/
5. EdgeSimPy GitHub: https://github.com/EdgeSimPy/EdgeSimPy
6. EdgeSimPy Docs: https://edgesimpy.github.io/
7. YAFS GitHub: https://github.com/acsicuib/YAFS
8. YAFS Docs: https://yafs.readthedocs.io/en/latest/
9. Alibaba ClusterData: https://github.com/alibaba/clusterdata
10. Alibaba Microservices Trace README: https://github.com/alibaba/clusterdata/blob/master/cluster-trace-microservices-v2021/README.md
11. Google ClusterData: https://github.com/google/cluster-data
12. Time-Series-Library: https://github.com/thuml/Time-Series-Library
13. PatchTST: https://github.com/yuqinie98/PatchTST
14. STGCN PyTorch: https://github.com/FelixOpolka/STGCN-PyTorch
