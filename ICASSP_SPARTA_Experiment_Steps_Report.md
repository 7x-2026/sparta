# ICASSP-SPARTA 实验执行步骤报告

> 论文目标：面向 ICASSP 的第一篇 CCF B 会议论文。  
> 论文题目建议：**SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services**  
> 中文题目建议：**面向云边服务的服务路径感知 SLA 风险早期检测与归因方法**  
> 核心定位：把云边网络状态建模为**服务路径网络信号**，完成 SLA 风险的**早期检测、风险归因和主动干预验证**。  
> 本报告用途：指导你或后续 AI/代码助手按步骤完成实验，不偏题、不乱加模块。

---

## 0. 实验总目标

本实验不是做传统边缘计算任务卸载，也不是单纯做 Transformer 三分类，而是完成以下任务：

给定某个云边服务在过去一段时间内的服务路径状态，包括节点负载、链路时延、丢包、队列长度、服务请求率、响应时间和 SLA 条件，预测该服务在未来一段时间内是否会进入：

- `normal`：服务质量正常；
- `risky`：存在 SLA 违约风险；
- `violated`：发生或即将发生 SLA 违约。

同时，模型还要输出风险来源：

- 风险节点 `risk node`；
- 风险链路 `risk link`；
- 风险主导指标 `risk metric`，例如 delay、loss、CPU、queue、bandwidth。

最终实验要证明：

1. **SPARTA 比传统 ML、LSTM、Transformer、PatchTST、STGCN 等 baseline 更适合该任务；**
2. **service-path subgraph、SLA condition、topology attention、attribution head 都有贡献；**
3. **SPARTA 能提前发现 SLA 风险，而不是只判断当前状态；**
4. **归因结果可以指导简单 proactive intervention，从而降低 SLA violation rate。**

---

## 1. 三个大创新点对应的实验任务

ICASSP 正文只有 4 页，创新点必须压缩成 3 个大点。实验也要围绕这 3 个点设计。

| 大创新点 | 实验中必须体现什么 |
|---|---|
| 创新点 1：服务路径感知的网络信号建模 | 构造 service-path subgraph，并与 global graph / flat sequence 对比 |
| 创新点 2：SLA 条件驱动的时空拓扑风险建模 | 实现 SLA-conditioned temporal-topology encoder，并做 w/o SLA、w/o topology 消融 |
| 创新点 3：面向主动干预的轻量风险归因与验证 | 输出 risk node/link/metric，并做 proactive vs reactive 对比 |

---

## 2. 推荐实验环境

### 2.1 硬件环境

最低要求：

- CPU：8 核以上；
- 内存：32 GB；
- GPU：RTX 3090 / RTX 4090 / A5000 均可；
- 显存：12 GB 以上即可；
- 存储：100 GB 以上。

AutoDL 租卡建议：

- 第一阶段数据生成和传统 baseline：CPU 或便宜 GPU 即可；
- 深度模型训练：3090 或 4090；
- 不需要多卡；
- 不需要真实 GPU 集群；
- 不需要 RDMA、RoCE、InfiniBand 等环境。

### 2.2 软件环境

建议使用 Linux，Ubuntu 20.04 / 22.04 均可。

推荐 Python 版本：

```bash
python=3.10
```

推荐 PyTorch：

```bash
torch >= 2.0
```

创建环境：

```bash
conda create -n sparta python=3.10 -y
conda activate sparta

pip install numpy pandas scipy scikit-learn networkx matplotlib tqdm pyyaml xgboost einops
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果 CUDA 版本不一致，可以根据 AutoDL 的 CUDA 版本调整 PyTorch 安装命令。

---

## 3. 项目目录结构

建议从一开始就按下面结构建项目，避免后期混乱。

```text
SPARTA/
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
└── README.md
```

---

## 4. 代码下载与用途

### 4.1 EdgeSimPy：主仿真平台

仓库：

```text
https://github.com/EdgeSimPy/EdgeSimPy
```

下载：

```bash
mkdir -p ~/SPARTA/simulators
cd ~/SPARTA/simulators

git clone https://github.com/EdgeSimPy/EdgeSimPy.git
cd EdgeSimPy
pip install -r requirements.txt
pip install -e .
```

用途：

- 构造云边拓扑；
- 设置边缘节点、云节点、用户、服务；
- 模拟服务请求、部署位置、资源负载；
- 导出节点、链路、服务、路径日志。

修改原则：

- 不建议直接重写 EdgeSimPy 源码；
- 建议在 `src/simulation/run_edgesimpy.py` 中写 wrapper；
- 只读取仿真对象属性并导出日志；
- 自己实现拥塞、节点过载、链路劣化等事件注入。

你需要从仿真中导出 5 类日志：

```text
node_log.csv
link_log.csv
service_log.csv
path_log.csv
sla_log.csv
```

---

### 4.2 YAFS：辅助仿真平台

仓库：

```text
https://github.com/acsicuib/YAFS
```

下载：

```bash
cd ~/SPARTA/simulators

git clone https://github.com/acsicuib/YAFS.git
cd YAFS
pip install -r requirements.txt
pip install -e .
```

用途：

- 做动态拓扑；
- 节点故障；
- 链路变化；
- 后续扩展期刊论文。

ICASSP 第一篇可选：

- 如果时间紧，YAFS 不作为主实验；
- 可以只在附加鲁棒性实验中使用；
- 也可以先不做，留给 Computer Networks 扩展版。

---

### 4.3 Time-Series-Library：时序模型 baseline

仓库：

```text
https://github.com/thuml/Time-Series-Library
```

下载：

```bash
mkdir -p ~/SPARTA/third_party
cd ~/SPARTA/third_party

git clone https://github.com/thuml/Time-Series-Library.git
cd Time-Series-Library
pip install -r requirements.txt
```

用途：

- Transformer baseline；
- TimesNet baseline；
- 时序分类 / 预测代码参考。

建议：

- 不要把主项目塞进 Time-Series-Library；
- 只参考其模型或训练结构；
- 你的主模型统一放在 `SPARTA/src/models/` 下。

---

### 4.4 PatchTST：强时序 baseline

仓库：

```text
https://github.com/yuqinie98/PatchTST
```

下载：

```bash
cd ~/SPARTA/third_party

git clone https://github.com/yuqinie98/PatchTST.git
```

用途：

- 作为强 Transformer 时序 baseline；
- 可改 classification head；
- 证明 SPARTA 不是普通时序 Transformer。

修改方式：

原版 PatchTST 偏 forecasting，建议改为 classification：

```text
PatchTST encoder output → mean pooling / CLS token → Linear(num_classes=3)
```

---

### 4.5 STGCN：图时空 baseline

仓库：

```text
https://github.com/FelixOpolka/STGCN-PyTorch
```

下载：

```bash
cd ~/SPARTA/third_party

git clone https://github.com/FelixOpolka/STGCN-PyTorch.git
```

用途：

- 图时空 baseline；
- 输入全局图或服务路径图；
- 证明服务路径子图建模比全局图/普通序列更有效。

修改方式：

交通流任务中的：

```text
node = road sensor
feature = traffic speed / flow
```

替换为：

```text
node = edge/cloud/access node
feature = CPU / memory / queue / delay aggregation
```

---

## 5. 数据下载与处理

### 5.1 Alibaba ClusterData

仓库：

```text
https://github.com/alibaba/clusterdata
```

下载：

```bash
mkdir -p ~/SPARTA/data/traces/alibaba
cd ~/SPARTA/data/traces/alibaba

git clone https://github.com/alibaba/clusterdata.git
```

重点使用：

```text
cluster-trace-microservices-v2021
```

可提取字段：

- 服务调用率；
- 响应时间；
- 服务依赖；
- 调用链变化；
- 负载波动。

处理输出：

```csv
time,service_id,service_type,request_rate,base_response_time,dependency_count
```

用途：

- `request_rate` 注入仿真，控制服务请求到达；
- `base_response_time` 用作服务处理时延基准；
- `dependency_count` 用于设置服务复杂度；
- trace 只用于驱动仿真，不直接作为最终标签。

---

### 5.2 Google ClusterData

仓库：

```text
https://github.com/google/cluster-data
```

下载：

```bash
mkdir -p ~/SPARTA/data/traces/google
cd ~/SPARTA/data/traces/google

git clone https://github.com/google/cluster-data.git
```

重点使用：

- task arrival；
- task usage；
- CPU usage；
- memory usage；
- workload burst。

处理输出：

```csv
time,node_group,cpu_scale,mem_scale,arrival_scale
```

用途：

- 驱动边缘节点负载；
- 构造突发流量；
- 让仿真负载不完全随机。

---

## 6. 仿真数据生成流程

### 6.1 基础拓扑配置

建议先做 3 类拓扑：

| 拓扑 | 作用 |
|---|---|
| tree | 云边层级结构 |
| random | 随机边缘连接 |
| small-world | 更接近局部连接 + 少量远程连接 |

节点规模：

| 设置 | 数量 |
|---|---:|
| 云节点 | 1 |
| 边缘节点 | 10 / 20 / 30 |
| 接入节点 | 10 / 20 |
| 用户 | 100 / 300 / 500 |
| 服务类型 | 3 |

服务类型：

| 服务类型 | SLA 特征 |
|---|---|
| latency-sensitive | 低时延 |
| reliability-sensitive | 低丢包 / 高可靠 |
| cost-sensitive | 低资源成本 |

### 6.2 仿真命令

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

### 6.3 必须导出的日志

#### node_log.csv

```csv
time,node_id,node_type,cpu_util,mem_util,queue_len,available_cpu,available_mem
```

#### link_log.csv

```csv
time,src,dst,delay,bandwidth,loss,jitter,queue_delay,bandwidth_util
```

#### service_log.csv

```csv
time,service_id,user_id,service_type,request_rate,response_time,current_edge,cloud_node
```

#### path_log.csv

```csv
time,service_id,path_nodes,path_links,path_delay,path_loss,bottleneck_node,bottleneck_link
```

#### sla_log.csv

```csv
service_id,service_type,max_delay,max_loss,min_bandwidth,reliability_req,cost_weight
```

---

## 7. Service-path subgraph 构造

### 7.1 构造对象

每个样本不是全网图，而是一个服务对应的路径子图：

```text
user access node
→ current edge node
→ candidate edge nodes
→ cloud node
→ path links
→ nearby competing services
```

### 7.2 输入窗口

建议：

```text
L = 12
```

即使用过去 12 个时间片作为输入。

预测窗口：

```text
H = 5
```

即预测未来 5 个时间片内是否发生风险。

后续做 horizon 实验：

```text
H = 1, 3, 5, 10
```

### 7.3 样本格式

```python
sample = {
    "node_features": Tensor[L, Np, Fn],
    "link_features": Tensor[L, Ep, Fe],
    "service_features": Tensor[L, Fs],
    "sla_features": Tensor[Fc],
    "adj_matrix": Tensor[Np, Np],
    "risk_label": int,
    "risk_node": int,
    "risk_link": int,
    "risk_metric": int
}
```

其中：

- `Np` 是路径子图节点数；
- `Ep` 是路径子图链路数；
- 不同样本的 `Np`、`Ep` 可通过 padding 到最大长度处理。

---

## 8. 标签生成规则

### 8.1 风险等级标签

#### violated

未来 `H` 个时间片中满足任一条件：

```text
response_time >= max_delay
path_loss >= max_loss
service_failed == True
```

则标签为：

```text
violated
```

#### risky

未来 `H` 个时间片中未 violated，但满足任一条件：

```text
response_time >= 0.75 * max_delay
path_loss >= 0.75 * max_loss
CPU load >= 0.85
queue_len 连续 3 个时间片上升
bandwidth_util >= 0.85
```

则标签为：

```text
risky
```

#### normal

其他样本：

```text
normal
```

### 8.2 风险归因标签

#### risk_node

路径子图中风险分数最高的节点：

```text
node_score = 0.4 * cpu_util + 0.3 * queue_norm + 0.3 * (1 - available_cpu_norm)
```

取最高者为 `risk_node`。

#### risk_link

路径子图中风险分数最高的链路：

```text
link_score = 0.4 * delay_norm + 0.3 * loss_norm + 0.3 * bandwidth_util
```

取最高者为 `risk_link`。

#### risk_metric

取超阈值最明显的指标：

```text
delay
loss
CPU
queue
bandwidth
```

建议保存为：

```text
0 delay
1 loss
2 CPU
3 queue
4 bandwidth
```

---

## 9. 数据集划分

必须按时间划分，不能随机打乱时间片，否则会产生时间泄漏。

建议：

```text
train: 前 60%
val: 中间 20%
test: 后 20%
```

如果样本来自不同拓扑，可以额外做跨拓扑测试：

```text
train: tree + random
test: small-world
```

这可以作为泛化实验。

---

## 10. 模型实现步骤

### 10.1 SPARTA 输入

```python
node_x:    [B, L, Np, Fn]
link_x:    [B, L, Ep, Fe]
service_x: [B, L, Fs]
sla_x:     [B, Fc]
adj:       [B, Np, Np]
```

### 10.2 Embedding 层

```python
node_emb = NodeMLP(node_x)
link_emb = LinkMLP(link_x)
service_emb = ServiceMLP(service_x)
sla_emb = SlaMLP(sla_x)
```

### 10.3 Path-level token 构造

```python
node_pool = mean_pool(node_emb, dim=2)
link_pool = mean_pool(link_emb, dim=2)
path_token = node_pool + link_pool + service_emb
```

此时：

```text
path_token: [B, L, D]
```

### 10.4 Temporal Encoder

```python
temporal_feature = TransformerEncoder(path_token)
```

输出：

```text
[B, L, D]
```

取最后时间片或 mean pooling：

```python
z_t = temporal_feature[:, -1, :]
```

### 10.5 Topology-aware feature

简化实现：

```python
topo_node = GCN(node_emb, adj)
topo_feature = mean_pool(topo_node)
```

然后融合：

```python
z = z_t + topo_feature
```

如果想更像 attention，可以实现：

```text
attention_score = QK^T / sqrt(d) + alpha * A
```

但第一版建议用 GCN/topology pooling，稳定易实现。

### 10.6 SLA-conditioned fusion

建议使用 gate：

```python
gate = sigmoid(MLP(sla_emb))
z = z * gate + z
```

### 10.7 输出 head

#### risk head

```python
risk_logits = Linear(z, 3)
```

#### node attribution head

```python
node_logits = Linear(node_repr, 1)
```

输出维度：

```text
[B, Np]
```

#### link attribution head

```python
link_logits = Linear(link_repr, 1)
```

输出维度：

```text
[B, Ep]
```

#### metric attribution head

```python
metric_logits = Linear(z, 5)
```

### 10.8 损失函数

```text
L = L_risk + 0.3 L_node + 0.3 L_link + 0.2 L_metric
```

其中：

- `L_risk`：weighted CE 或 focal loss；
- `L_node`：CE；
- `L_link`：CE；
- `L_metric`：CE。

---

## 11. Baseline 实验步骤

### 11.1 传统 ML

将每个样本 flatten：

```text
[L, Np, Fn] + [L, Ep, Fe] + [L, Fs] + [Fc]
→ 1D feature vector
```

方法：

- Random Forest；
- XGBoost。

### 11.2 LSTM / GRU

输入：

```text
path_token: [B, L, D]
```

输出：

```text
risk_label
```

### 11.3 TCN

输入：

```text
[B, D, L]
```

输出：

```text
risk_label
```

### 11.4 Vanilla Transformer

输入：

```text
path_token: [B, L, D]
```

不使用 SLA gate，不使用 topology feature。

### 11.5 PatchTST / TimesNet

输入多变量时间序列，输出 risk classification。

### 11.6 STGCN

输入全局图或 service-path 图，输出风险分类。

---

## 12. 评价指标

主指标：

```text
Accuracy
Macro-F1
Risk Recall
Violation Recall
AUC
Attribution Accuracy
Inference Time
```

重点汇报：

- Macro-F1；
- Risk Recall；
- Violation Recall；
- Attribution Accuracy。

不要只看 Accuracy，因为 normal 类可能占多数。

### 12.1 Risk Recall

```text
Risk Recall = risky 样本中被正确预测为 risky 的比例
```

### 12.2 Violation Recall

```text
Violation Recall = violated 样本中被正确识别的比例
```

### 12.3 Attribution Accuracy

```text
Attr-Acc = risk node/link/metric 是否预测正确
```

可分别报告：

```text
Node-Attr Acc
Link-Attr Acc
Metric-Attr Acc
```

如果篇幅不够，主文只报平均 Attr-Acc。

---

## 13. 必做实验清单

### 13.1 主实验

表格：

```text
Method | Acc | Macro-F1 | Risk Recall | Violation Recall | AUC | Attr-Acc | Time
```

方法：

- RF；
- XGBoost；
- LSTM；
- GRU；
- TCN；
- Transformer；
- PatchTST / TimesNet；
- STGCN；
- SPARTA。

### 13.2 消融实验

表格：

```text
Variant | Macro-F1 | Risk Recall | Violation Recall | Attr-Acc
```

变体：

- full SPARTA；
- w/o SLA；
- w/o topology；
- w/o attribution；
- global graph；
- binary label。

### 13.3 Early horizon 实验

```text
H = 1, 3, 5, 10
```

画图：

```text
Macro-F1@H
Risk Recall@H
Violation Recall@H
```

### 13.4 归因实验

报告：

```text
Node-Attr Acc
Link-Attr Acc
Metric-Attr Acc
```

配一张可视化图：

- 红色链路代表模型判断的风险链路；
- 橙色节点代表模型判断的风险节点；
- 标注真实风险来源。

### 13.5 主动干预实验

对比：

| 策略 | 说明 |
|---|---|
| Reactive | violated 后再处理 |
| Proactive-SPARTA | risky 阶段提前处理 |

指标：

- SLA violation rate；
- average response time；
- migration / rerouting cost；
- number of interventions。

干预规则：

```text
risk node overloading → migrate to neighbor edge with lowest CPU
risk link congestion → reroute to path with lower delay/loss
risk metric = bandwidth → choose path with higher residual bandwidth
risk metric = loss → choose path with lower packet loss
```

---

## 14. 推荐实验表格结果口径

注意：这里不是让你伪造结果，而是提前明确论文需要什么类型的结果。

你希望看到的结果趋势是：

1. SPARTA 的 Macro-F1 高于 Transformer、PatchTST、STGCN；
2. SPARTA 的 Risk Recall 提升明显；
3. w/o SLA 和 w/o topology 都会下降；
4. w/o attribution 不一定显著降低分类 Acc，但会降低可解释性和 proactive 效果；
5. proactive intervention 能降低 SLA violation rate，但 migration cost 会略增加。

如果实验结果不符合这些趋势，需要检查：

- 标签是否过于简单；
- risky 类是否太少；
- service-path subgraph 是否构造合理；
- baseline 是否输入不公平；
- normal 类是否过多导致类别不平衡。

---

## 15. 12 周执行计划

### 第 1 周：环境搭建

完成：

- 建立 conda 环境；
- 下载 EdgeSimPy；
- 下载 Alibaba / Google trace；
- 下载 baseline 代码；
- 建立项目目录。

验收标准：

```text
能正常 import torch、networkx、pandas；
EdgeSimPy 能跑示例；
项目目录完整。
```

### 第 2 周：EdgeSimPy 最小仿真

完成：

- 10 edge nodes；
- 100 users；
- 3 services；
- 1000 steps；
- 输出 node/link/service/path/sla log。

验收标准：

```text
5 个 csv 日志文件存在；
每个日志有 time 字段；
service_log 能看到 response_time。
```

### 第 3 周：trace 注入

完成：

- Alibaba request_rate 预处理；
- Google node_load 预处理；
- 注入仿真；
- 生成非随机负载。

验收标准：

```text
请求率随时间变化；
节点负载随 trace 波动；
能画出 request_rate 曲线。
```

### 第 4 周：数据集构造

完成：

- service-path subgraph；
- risk labels；
- attribution labels；
- train/val/test split。

验收标准：

```text
sparta_dataset.pkl 可加载；
normal/risky/violated 数量合理；
risk_node/link/metric 标签不为空。
```

### 第 5 周：传统和基础深度 baseline

完成：

- RF；
- XGBoost；
- LSTM；
- GRU。

验收标准：

```text
能输出 Acc/Macro-F1/Risk Recall。
```

### 第 6 周：强 baseline

完成：

- TCN；
- Transformer；
- PatchTST 或 TimesNet；
- STGCN。

验收标准：

```text
至少 7 个 baseline 可跑通。
```

### 第 7 周：SPARTA 初版

完成：

- embedding；
- temporal encoder；
- SLA gate；
- risk head。

验收标准：

```text
SPARTA 可训练；
risk classification 指标超过 LSTM。
```

### 第 8 周：SPARTA 完整版

完成：

- topology feature；
- attribution heads；
- multi-task loss。

验收标准：

```text
可输出 risk_label、risk_node、risk_link、risk_metric。
```

### 第 9 周：主实验 + 消融

完成：

- main results；
- ablation；
- H=1/3/5/10。

验收标准：

```text
主实验表格完整；
消融结果趋势合理；
horizon 曲线可画。
```

### 第 10 周：鲁棒性 + 主动干预

完成：

- load burst；
- link congestion；
- node overload；
- proactive vs reactive。

验收标准：

```text
proactive 能降低 violation rate；
可解释性图能画。
```

### 第 11 周：论文图表

完成：

- overall framework；
- service-path subgraph；
- main table；
- ablation table；
- horizon curve；
- attribution visualization。

验收标准：

```text
所有图表能放进 4 页正文。
```

### 第 12 周：论文初稿

完成：

- abstract；
- introduction；
- method；
- experiments；
- conclusion；
- 参考文献；
- 压缩到 ICASSP 格式。

---

## 16. 最小可行实验版本

如果时间非常紧，最低版本必须保留：

1. EdgeSimPy 仿真；
2. Alibaba trace 驱动负载；
3. service-path subgraph；
4. SLA-conditioned Transformer；
5. risk classification；
6. attribution head；
7. LSTM / Transformer / STGCN / PatchTST baseline；
8. w/o SLA、w/o topology、w/o attribution 消融；
9. H=1/3/5 early detection；
10. proactive vs reactive 小实验。

可以暂时不做：

- YAFS；
- Google trace；
- 复杂 RL；
- LLM intent parsing；
- 多平台系统部署。

---

## 17. 实验失败时的补救策略

### 17.1 SPARTA 提升不明显

检查：

- 是否类别极度不平衡；
- 是否 risky 标签规则太简单；
- baseline 是否也用了 topology 和 SLA；
- service-path subgraph 是否过大或过小；
- horizon 是否太短。

补救：

- 使用 focal loss；
- 重新平衡 normal/risky/violated；
- 增加趋势条件；
- 调整 risky 阈值从 0.75 到 0.7 或 0.8；
- 提高突发负载和拥塞场景比例。

### 17.2 Attribution 准确率低

检查：

- risk_node/link 标签是否合理；
- 多个节点同时异常是否导致标签不唯一；
- 是否应该用 Top-k accuracy。

补救：

- 报告 Top-3 attribution accuracy；
- 将 risk_node 和 risk_link 改为多标签；
- 简化风险场景，每次只注入一种主要故障。

### 17.3 Proactive 效果不明显

检查：

- 干预是否太晚；
- 迁移目标节点是否也过载；
- reroute 是否没有更优路径；
- risky 状态是否与 violation 之间间隔太短。

补救：

- 增大 horizon；
- 提前在 risky 阶段干预；
- 限制实验场景，让备选路径/节点可用；
- 报告 violation rate 与 response time 两个指标。

---

## 18. 最终实验交付物

完成实验后，你应该具备以下材料：

```text
1. data/sparta_dataset/sparta_w12_h5.pkl
2. results/main_results.csv
3. results/ablation_results.csv
4. results/horizon_results.csv
5. results/attribution_results.csv
6. results/proactive_results.csv
7. figures/framework.pdf
8. figures/service_path.pdf
9. figures/horizon_curve.pdf
10. figures/attribution_case.pdf
11. logs/train_sparta.log
12. checkpoints/sparta_best.pth
```

这些材料可以直接支撑 ICASSP 初稿。

---

## 19. 最终判断

如果你只做：

```text
EdgeSimPy + Transformer + normal/risky/violated 三分类
```

不够稳。

如果你按本报告完成：

```text
service-path subgraph
+ SLA-conditioned temporal-topology encoder
+ early risk detection
+ lightweight attribution
+ proactive intervention validation
```

则具备 ICASSP CCF B 会议论文的合理创新性和工作量。

本实验方案最大的优点是：

1. 不依赖真实 AI 数据中心；
2. 不依赖 GPU 集群；
3. 不需要复杂网络协议实现；
4. 能使用你已有 LSTM / Transformer / DETR / 检测思维；
5. 后续能扩展到 Computer Networks 或 PR 期刊版。

