# ICASSP-SPARTA 论文写作报告：初稿生成指南

> 论文题目建议：**SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services**  
> 中文题目建议：**面向云边服务的服务路径感知 SLA 风险早期检测与归因方法**  
> 投稿目标：ICASSP，CCF B 会议。  
> 本报告用途：为你或后续 AI 写作助手提供完整论文初稿生成框架，包括论文主线、每节写什么、贡献点怎么写、公式怎么写、图表怎么放、实验怎么解释。

---

## 0. 论文核心主线

这篇论文不能写成传统“边缘计算调度算法”，也不能写成“Transformer 做 SLA 三分类”。最合适的主线是：

> 云边服务的 SLA 风险并不是由单一指标瞬时触发，而是由服务路径上的节点负载、链路质量、请求波动和服务 SLA 约束共同作用形成。本文将云边服务状态建模为服务路径网络信号，提出 SPARTA，用于提前检测 SLA 风险，并归因到具体节点、链路和指标。最后通过 proactive intervention 验证风险检测具有实际网络管理价值。

这条主线包括 3 个大创新点：

1. **服务路径感知的网络信号建模**；
2. **SLA 条件驱动的时空拓扑风险建模**；
3. **面向主动干预的轻量风险归因与验证**。

---

## 1. 论文题目

### 1.1 推荐英文题目

```text
SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services
```

优点：

- 有方法名 SPARTA；
- 明确 service-path aware；
- 明确 temporal risk attribution；
- 明确 SLA violation early detection；
- 明确 cloud-edge services 场景。

### 1.2 备用英文题目

```text
Service-Path Aware Temporal Risk Attribution for Proactive SLA Violation Detection in Cloud-Edge Services
```

```text
SLA-Conditioned Service-Path Network Signal Modeling for Early Risk Detection in Cloud-Edge Services
```

```text
Early SLA Risk Detection and Attribution over Service-Path Network Signals in Cloud-Edge Systems
```

### 1.3 中文题目

```text
面向云边服务的服务路径感知 SLA 风险早期检测与归因方法
```

---

## 2. Abstract 写作模板

Abstract 控制在 150–180 词左右。ICASSP 论文摘要要短，不要写成期刊摘要。

### 2.1 Abstract 结构

建议 5 句话：

1. 背景：Cloud-edge services require proactive SLA assurance.
2. 问题：Existing methods ignore service-path structures and provide limited attribution.
3. 方法：We propose SPARTA.
4. 实验：Trace-driven simulations show improvement.
5. 意义：SPARTA enables proactive risk-aware service management.

### 2.2 英文 Abstract 初稿模板

```text
Cloud-edge services are increasingly required to satisfy stringent service-level agreements (SLAs) under dynamic network and workload conditions. Existing SLA prediction methods often rely on global QoS records or aggregated resource metrics, which overlook service-path structures and provide limited insight into the sources of potential violations. In this paper, we formulate SLA violation early detection as a service-path network signal modeling problem and propose SPARTA, a Service-Path Aware Temporal Risk Attribution framework. SPARTA constructs path-centric subgraphs for individual services, jointly encodes temporal network states, topology-aware dependencies, and service-specific SLA constraints, and predicts normal, risky, and violated states before actual SLA degradation occurs. Moreover, SPARTA introduces lightweight node-, link-, and metric-level attribution heads to identify the dominant risk sources. Trace-driven cloud-edge simulations show that SPARTA consistently improves macro-F1 and risk recall over classical machine learning, recurrent, Transformer-based, and spatio-temporal graph baselines. A simple proactive intervention study further demonstrates that attribution-guided early detection can reduce SLA violation rates.
```

### 2.3 中文解释

摘要里不要过多强调“边缘计算资源调度”，而要强调：

- service-path network signal；
- early detection；
- attribution；
- trace-driven simulation；
- proactive intervention。

---

## 3. Introduction 写作报告

Introduction 建议 4 段，控制在 0.7–0.8 页。

### 3.1 第一段：背景与问题重要性

目标：说明云边服务质量保障为什么重要。

要点：

- Cloud-edge services support latency-sensitive and reliability-critical applications；
- SLA assurance is challenging because workload and network states are dynamic；
- SLA violations are often caused by accumulated degradation along service paths.

英文写作模板：

```text
Cloud-edge computing has become an important paradigm for supporting latency-sensitive and reliability-critical applications, such as real-time video analytics, industrial sensing, and interactive mobile services. By placing computation and service functions closer to users, cloud-edge systems can reduce response latency and alleviate backbone traffic pressure. However, maintaining service-level agreements (SLAs) in such environments remains challenging, because edge servers, access links, user requests, and service dependencies evolve continuously over time. In practice, an SLA violation is rarely caused by a single instantaneous event; it is often the result of accumulated degradation along a service path, such as increasing queue length, link delay, packet loss, or edge-node overload.
```

### 3.2 第二段：现有方法不足

目标：指出已有 SLA prediction / QoS prediction / time-series model 的不足。

建议批评 3 点：

1. 使用全局 QoS 或资源指标，忽略 service path；
2. 只预测当前状态或未来 QoS 数值，缺少 early risk detection；
3. 缺少 risk attribution，不能指导后续干预。

英文写作模板：

```text
Existing studies on SLA or QoS prediction typically model historical measurements as flat time-series records or global resource states. Although recurrent networks, Transformers, and spatio-temporal models have shown promise in network performance prediction, directly applying them to cloud-edge SLA risk detection has two limitations. First, global representations may dilute the local dependencies that actually determine a service's quality, since each service is affected mainly by its own access node, serving edge node, candidate edge nodes, and the links along its path. Second, most prediction models only output a violation probability or a future QoS value, but they do not identify which node, link, or metric is responsible for the risk. Such black-box predictions are insufficient for proactive service management, where the system needs to know not only whether a risk will occur, but also where and why it may occur.
```

### 3.3 第三段：本文方法

目标：自然引出 SPARTA。

要点：

- service-path network signal formulation；
- SLA-conditioned temporal topology encoder；
- risk detection + attribution；
- proactive intervention validation。

英文写作模板：

```text
To address these issues, we propose SPARTA, a Service-Path Aware Temporal Risk Attribution framework for SLA violation early detection in cloud-edge services. Instead of representing the entire network as a global state vector, SPARTA constructs a path-centric subgraph for each service, which contains the related access node, edge nodes, cloud node, path links, service states, and SLA constraints. It then uses an SLA-conditioned temporal-topology encoder to capture historical network dynamics and topology-aware dependencies under service-specific quality requirements. In addition to predicting normal, risky, and violated states, SPARTA estimates node-, link-, and metric-level risk attribution, enabling the prediction results to support proactive interventions such as migration or rerouting.
```

### 3.4 第四段：贡献点

Introduction 末尾写 3 条贡献即可。

推荐写法：

```text
The main contributions of this paper are summarized as follows.

1. We formulate SLA violation early detection in cloud-edge services as a service-path network signal modeling problem. Instead of relying on global QoS records, each service is represented by a path-centric subgraph that captures its related nodes, links, service states, SLA conditions, and future risk horizon.

2. We propose SPARTA, an SLA-conditioned temporal-topology risk modeling framework. It jointly encodes historical network states, service-path topology, and service-specific SLA constraints to distinguish normal, risky, and violated states before actual SLA violations occur.

3. We introduce lightweight risk attribution heads to identify risk-related nodes, links, and dominant metrics. We further validate the usefulness of early detection through a simple proactive intervention strategy, showing that attribution-guided actions can reduce SLA violation rates.
```

---

## 4. Related Work 写作建议

ICASSP 4 页正文通常没有空间写完整 Related Work。建议：

- 不单独开大节；
- 在 Introduction 第二段中简要提；
- 或者在 Problem / Experiments 里用一句话说明 baseline 类别；
- 参考文献中覆盖 3 类即可。

### 4.1 参考文献类别

建议引用：

1. SLA / QoS prediction；
2. time-series / Transformer / PatchTST / TimesNet；
3. graph spatio-temporal modeling；
4. edge/fog simulation；
5. cloud / microservice trace。

### 4.2 Related Work 的一句话模板

```text
Prior works have investigated QoS prediction, SLA violation classification, and spatio-temporal traffic forecasting using machine learning or deep sequence models. Different from these studies, our focus is not only to predict service-level risk earlier, but also to model service-path-specific network signals and attribute the predicted risk to actionable network components.
```

---

## 5. Problem Formulation 写作报告

Problem Formulation 控制在 0.4–0.5 页。

### 5.1 定义云边网络图

英文模板：

```text
We model a cloud-edge system as a directed graph G=(V,E), where V contains cloud nodes, edge servers, access nodes, and users, and E denotes communication links. Each node v∈V is associated with time-varying resource states, such as CPU utilization, memory usage, queue length, and available capacity. Each link e∈E has dynamic network states, including delay, bandwidth, packet loss, jitter, and queueing delay.
```

### 5.2 定义服务路径子图

英文模板：

```text
For a service request s at time t, we extract a service-path subgraph G_s^t=(V_s^t,E_s^t), where V_s^t includes the user access node, the currently selected edge server, candidate edge servers, the cloud node, and neighboring nodes that compete for shared resources. E_s^t includes the links along the current or candidate service paths. This path-centric representation focuses on the network components that directly influence the SLA status of service s.
```

### 5.3 定义输入

可以写成公式：

```text
X_s^{t-L:t} = {X_v^{t-L:t}, X_e^{t-L:t}, X_m^{t-L:t}, c_s, A_s}
```

其中：

- `X_v` 是节点特征；
- `X_e` 是链路特征；
- `X_m` 是服务特征；
- `c_s` 是 SLA condition；
- `A_s` 是服务路径子图邻接矩阵。

英文模板：

```text
Given a historical window of length L, the input of service s is denoted as X_s^{t-L:t}={X_v, X_e, X_m, c_s, A_s}, where X_v, X_e, and X_m represent node, link, and service-state sequences, respectively; c_s denotes the service-specific SLA condition; and A_s is the adjacency matrix of the service-path subgraph.
```

### 5.4 定义输出

风险等级：

```text
y_s^t ∈ {normal, risky, violated}
```

归因标签：

```text
a_s^t = {a_node, a_link, a_metric}
```

英文模板：

```text
The goal is to predict the risk state y_s^{t+H}∈{normal,risky,violated} within a future horizon H, and to estimate attribution labels a_s={a_node,a_link,a_metric}, indicating the most risk-related node, link, and metric.
```

### 5.5 标签定义

英文模板：

```text
A sample is labeled as violated if its response time, path loss, or service availability violates the corresponding SLA threshold within the future horizon H. It is labeled as risky if no violation occurs but one or more indicators approach their SLA thresholds or exhibit continuously deteriorating trends. Otherwise, the sample is labeled as normal.
```

---

## 6. Method 写作报告

Method 建议分 3 小节，与 3 个创新点对应。

---

### 6.1 Service-Path Network Signal Construction

写作目标：说明输入不是全网状态，而是服务路径信号。

英文模板：

```text
The first component of SPARTA constructs service-path network signals. For each service request, we collect node states, link states, service states, and SLA constraints from its path-centric subgraph. Compared with global network representations, this construction reduces irrelevant background states and emphasizes the local components that determine service quality. The node and link sequences are padded to fixed sizes within a mini-batch, and the corresponding masks are used to avoid invalid positions during training.
```

可以加一句：

```text
This formulation is analogous to focusing on an object-centric region in detection tasks, but the region here is defined by the service path in a dynamic network graph.
```

如果觉得“object-centric”太 CV，可以删掉。

---

### 6.2 SLA-Conditioned Temporal-Topology Encoder

写作目标：讲 SPARTA 主模型。

#### 6.2.1 Embedding

英文模板：

```text
We first map heterogeneous inputs into a shared latent space. Node features, link features, and service features are projected using separate multilayer perceptrons. The SLA vector is also encoded into an SLA embedding, which contains service-specific constraints such as maximum delay, maximum loss, minimum bandwidth, reliability requirement, and cost preference.
```

公式：

```text
h_v = f_v(X_v), h_e = f_e(X_e), h_m = f_m(X_m), h_c = f_c(c_s)
```

#### 6.2.2 Temporal Encoder

英文模板：

```text
The temporal encoder captures the evolution of service-path states over the historical window. We aggregate node and link embeddings at each time step to obtain a path-level token sequence, which is then fed into a lightweight Transformer encoder.
```

公式：

```text
p_\tau = Pool(h_v^\tau) + Pool(h_e^\tau) + h_m^\tau
```

```text
z_{1:L} = Transformer(p_{1:L})
```

#### 6.2.3 Topology-aware Modeling

英文模板：

```text
To incorporate service-path topology, we compute topology-aware node representations using the adjacency matrix of the path-centric subgraph. In implementation, a lightweight graph aggregation layer is used to produce topology features, which are fused with the temporal representation.
```

公式：

```text
g_v = GCN(h_v, A_s)
```

```text
z = z_L + Pool(g_v)
```

如果你实现 attention bias，可以写：

```text
Attn(Q,K,V)=softmax(QK^T/sqrt(d)+αA_s)V
```

但如果实际只实现 GCN，就不要写 attention bias，避免方法和代码不一致。

#### 6.2.4 SLA-conditioned Fusion

英文模板：

```text
The encoded SLA condition modulates the temporal-topology representation through a gating mechanism. This allows the model to emphasize different risk factors under different service requirements.
```

公式：

```text
r = σ(W_c h_c)
```

```text
z_s = z ⊙ r + z
```

解释：

- 低时延服务会更关注 delay；
- 高可靠服务会更关注 loss；
- 低成本服务会更关注资源利用。

---

### 6.3 Risk Detection and Attribution

写作目标：讲输出 head 和 loss。

英文模板：

```text
The risk detection head predicts the probability of normal, risky, and violated states. To provide actionable explanations, SPARTA further introduces three lightweight attribution heads that estimate the most risk-related node, link, and metric.
```

风险分类：

```text
\hat{y}=softmax(W_y z_s)
```

归因：

```text
\hat{a}_{node}=softmax(W_n H_v)
```

```text
\hat{a}_{link}=softmax(W_l H_e)
```

```text
\hat{a}_{metric}=softmax(W_m z_s)
```

损失函数：

```text
L = L_risk + λ_n L_node + λ_l L_link + λ_m L_metric
```

英文解释：

```text
The multi-task objective encourages the shared representation to capture not only whether a risk will occur but also where the risk comes from. This design is important for proactive cloud-edge service management.
```

---

## 7. Experiments 写作报告

Experiments 是 ICASSP 审稿人判断论文是否可信的关键。建议 1.2–1.4 页。

### 7.1 Experimental Setup

#### 7.1.1 仿真环境

英文模板：

```text
We build a trace-driven cloud-edge simulation environment based on EdgeSimPy. The simulated system contains one cloud node, multiple edge servers, access nodes, users, and service types with different SLA requirements. Realistic workload patterns are injected from public cluster and microservice traces, while network states such as delay, loss, bandwidth utilization, and queueing delay are generated under different load, congestion, and failure scenarios.
```

#### 7.1.2 数据集统计

需要表 1：

```text
Topology | Edge Nodes | Users | Services | Samples | Normal | Risky | Violated
```

#### 7.1.3 Baselines

英文模板：

```text
We compare SPARTA with classical machine learning, sequence modeling, Transformer-based, and spatio-temporal graph baselines, including Random Forest, XGBoost, LSTM, GRU, TCN, vanilla Transformer, PatchTST or TimesNet, and STGCN.
```

#### 7.1.4 Metrics

英文模板：

```text
We report Accuracy, Macro-F1, Risk Recall, Violation Recall, AUC, Attribution Accuracy, and inference time. Macro-F1 and Risk Recall are emphasized because risky and violated samples are usually less frequent than normal samples.
```

---

### 7.2 Main Results

主结果表：

```text
Method | Acc | Macro-F1 | Risk Recall | Violation Recall | AUC | Attr-Acc | Time
```

写作逻辑：

1. SPARTA 总体最优；
2. 相比 LSTM/GRU，说明长距离时序和拓扑建模有效；
3. 相比 Transformer/PatchTST，说明 service-path + SLA + topology 有效；
4. 相比 STGCN，说明服务路径子图比全局图更适合服务级风险检测。

英文分析模板：

```text
SPARTA achieves the best overall performance across the main metrics. Compared with recurrent models, SPARTA better captures temporal dependencies under dynamic workload changes. Compared with vanilla Transformer and PatchTST, the improvement indicates that explicitly modeling service-path topology and SLA conditions is beneficial for service-level risk detection. Although STGCN incorporates graph structures, its global graph representation is less effective than the path-centric representation used by SPARTA, which focuses on the components directly related to each service.
```

---

### 7.3 Ablation Study

表格：

```text
Variant | Macro-F1 | Risk Recall | Violation Recall | Attr-Acc
```

变体：

- Full SPARTA；
- w/o SLA；
- w/o topology；
- w/o attribution；
- global graph；
- binary labels。

英文分析模板：

```text
Removing the SLA-conditioned fusion leads to a clear performance drop, suggesting that the same network state may imply different risks under different service requirements. Removing topology modeling also degrades performance, confirming that service-path dependencies are important for cloud-edge SLA risk detection. Without attribution heads, classification performance slightly decreases and attribution capability is lost, indicating that the auxiliary attribution task helps learn more informative risk representations. The binary-label variant performs worse in early warning, showing that the intermediate risky state is useful for proactive management.
```

---

### 7.4 Early Detection Horizon

图：

```text
H = 1, 3, 5, 10
```

指标：

- Macro-F1@H；
- Risk Recall@H；
- Violation Recall@H。

英文模板：

```text
We further evaluate different prediction horizons. As H increases, all methods experience performance degradation due to higher uncertainty. Nevertheless, SPARTA maintains higher Risk Recall and Macro-F1 than the baselines, demonstrating its ability to detect potential SLA risks earlier.
```

---

### 7.5 Attribution and Proactive Intervention

#### Attribution

英文模板：

```text
SPARTA provides node-, link-, and metric-level attribution. The attribution results show that the model can locate overloaded edge nodes, congested links, and dominant risk metrics with reasonable accuracy. This capability is essential for converting risk detection into actionable network management decisions.
```

#### Proactive Intervention

对比：

```text
Reactive
Proactive-SPARTA
```

英文模板：

```text
To verify whether early detection is actionable, we conduct a simple proactive intervention study. When SPARTA predicts a risky state, the system performs attribution-guided migration or rerouting according to lightweight rules. Compared with a reactive strategy that acts only after SLA violation, proactive intervention reduces the violation rate and average response time with a moderate increase in intervention cost.
```

注意：

- 这部分不要写成复杂调度算法；
- 只作为验证方法实用性的实验。

---

## 8. 图表设计报告

ICASSP 4 页最多放 4 个图表，建议如下。

### 8.1 图 1：Overall Framework

内容：

```text
Trace-driven cloud-edge simulation
→ service-path subgraph
→ SPARTA encoder
→ risk detection + attribution
→ proactive intervention
```

建议放在第 2 页顶部或第 1 页底部。

### 8.2 图 2：Service-path Subgraph 或 Attribution Case

如果篇幅紧，图 2 可以和图 1 合并。

内容：

- user；
- access node；
- current edge；
- candidate edge；
- cloud；
- risk link；
- risk node。

### 8.3 表 1：Main Results

必须有。

列：

```text
Method | Macro-F1 | Risk Recall | Violation Recall | AUC | Attr-Acc
```

可以不放 Accuracy，节省空间。

### 8.4 表 2：Ablation Study

必须有。

列：

```text
Variant | Macro-F1 | Risk Recall | Attr-Acc
```

### 8.5 图 3：Prediction Horizon

可选但强烈建议。

展示 H=1/3/5/10 下的 Risk Recall 曲线。

### 8.6 表 3：Proactive Intervention

如果篇幅够，放小表。

列：

```text
Strategy | Violation Rate | Avg. Delay | Cost
```

如果篇幅不够，把 proactive intervention 写在正文一段，不单独放表。

---

## 9. 论文 4 页篇幅分配

建议如下：

| 部分 | 页数 |
|---|---:|
| Abstract | 0.15 |
| Introduction | 0.75 |
| Problem Formulation | 0.45 |
| Method | 1.10 |
| Experiments | 1.35 |
| Conclusion | 0.10 |
| References | 单独参考文献页或压缩 |

注意：

- 不建议单独写 Related Work 大节；
- 不建议写太多公式；
- 不建议塞太多模块图；
- 不建议把 proactive intervention 写成大算法。

---

## 10. 论文初稿 Prompt

你可以把下面这段直接交给 AI，让它帮你生成英文初稿。

```text
Please draft an ICASSP-style 4-page conference paper titled “SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services”.

The paper should focus on cloud-edge SLA risk early detection from service-path network signals. Do not write it as a resource scheduling or task offloading paper. The main idea is to model each cloud-edge service as a service-path subgraph containing access nodes, edge servers, candidate edge servers, cloud node, path links, service states, and SLA constraints. The model predicts whether the service will be normal, risky, or violated within a future horizon, and also attributes the risk to a node, link, and dominant metric.

The three main contributions are:
1. Formulating SLA violation early detection as service-path network signal modeling, rather than global QoS time-series classification.
2. Proposing SPARTA, an SLA-conditioned temporal-topology encoder that jointly models historical network states, service-path topology, and service-specific SLA constraints.
3. Introducing lightweight risk attribution heads and validating proactive intervention based on attribution-guided migration or rerouting.

Write the paper with the following sections:
Abstract, Introduction, Problem Formulation, Proposed Method, Experiments, Conclusion.

In Introduction, explain that SLA violations in cloud-edge systems are caused by accumulated degradation along service paths, and existing global QoS prediction methods lack path-specific modeling and actionable attribution.

In Problem Formulation, define the cloud-edge graph G=(V,E), the service-path subgraph G_s, input X_s^{t-L:t}, SLA condition c_s, risk label y∈{normal,risky,violated}, and attribution labels including risk node, risk link, and risk metric.

In Method, describe:
- service-path network signal construction;
- heterogeneous node/link/service/SLA embedding;
- temporal encoder;
- topology-aware aggregation or topology attention;
- SLA-conditioned gating;
- risk detection head;
- node/link/metric attribution heads;
- multi-task loss.

In Experiments, describe trace-driven simulation using EdgeSimPy and public cluster/microservice traces, compare with Random Forest, XGBoost, LSTM, GRU, TCN, Transformer, PatchTST or TimesNet, and STGCN. Include main comparison, ablation, early detection horizon, attribution accuracy, and proactive intervention validation.

Use concise academic English suitable for ICASSP. Avoid overclaiming. Emphasize early detection, service-path modeling, and attribution.
```

---

## 11. 中文初稿 Prompt

如果你想先让 AI 写中文稿，再翻译英文，可以用这个。

```text
请帮我写一篇 ICASSP 风格的中文论文初稿，题目为《面向云边服务的服务路径感知 SLA 风险早期检测与归因方法》。论文不要写成传统边缘计算任务卸载或资源调度，而要写成“云边网络状态信号的早期风险检测与归因”问题。

论文核心思想是：云边服务的 SLA 违约通常不是单个指标瞬时异常造成的，而是由服务路径上的节点负载、链路时延、丢包、队列长度、请求波动和服务 SLA 约束共同作用产生。本文为每个服务构造 service-path subgraph，将节点、链路、服务状态和 SLA 条件编码为网络信号，并提出 SPARTA 模型进行 normal/risky/violated 三分类，同时输出 risk node、risk link 和 risk metric。最后通过 proactive intervention 验证早期风险检测可以降低 SLA violation rate。

论文三个创新点：
1. 将 SLA 违约早期检测定义为服务路径网络信号建模问题，而不是普通全局 QoS 时间序列分类；
2. 提出 SLA 条件驱动的时空拓扑风险建模框架 SPARTA，联合建模历史状态、服务路径拓扑和服务 SLA 约束；
3. 引入轻量级节点、链路、指标归因机制，并通过归因指导的主动迁移或换路验证检测结果的实际价值。

请按照 Abstract、Introduction、Problem Formulation、Proposed Method、Experiments、Conclusion 的结构写。内容要适合后续翻译成英文 ICASSP 论文。
```

---

## 12. 参考文献方向

你需要准备 20 篇左右参考文献，正文中可能只放 15–18 篇。

### 12.1 必须覆盖的方向

1. Edge computing / cloud-edge service management；
2. SLA / QoS prediction；
3. Time-series Transformer；
4. Spatio-temporal graph modeling；
5. Graph signal / network signal modeling；
6. Public cluster / microservice traces；
7. ICASSP 或 signal processing 相关方法。

### 12.2 参考文献占位

可先准备这些类别：

```text
[1] Edge computing survey / MEC survey
[2] Cloud-edge service placement or management
[3] SLA violation prediction with ML
[4] QoS prediction in cloud services
[5] LSTM/GRU for time-series prediction
[6] Transformer for time-series
[7] PatchTST
[8] TimesNet
[9] STGCN
[10] Graph attention / graph neural networks
[11] Google cluster trace
[12] Alibaba cluster trace
[13] Microservice trace
[14] EdgeSimPy
[15] YAFS
```

后续正式写作时要检索真实文献，不能只放占位。

---

## 13. 论文中不要写的内容

为了避免跑偏，下面这些不要写进 ICASSP 正文主线。

### 13.1 不要写成任务卸载论文

不要把题目写成：

```text
SLA-aware task offloading in edge computing
```

这会进入边缘计算调度老题，创新风险更高。

### 13.2 不要强调复杂强化学习

第一篇不要做：

```text
multi-agent reinforcement learning
deep reinforcement learning based orchestration
```

这会增加代码难度，也容易被要求和大量 RL baseline 对比。

### 13.3 不要写大模型意图理解

第一篇不要写：

```text
LLM-based intent parsing
natural language network intent understanding
```

这会偏离 ICASSP 第一篇主线，留给第二篇或第三篇。

### 13.4 不要过度包装主动干预

proactive intervention 只是验证风险检测有用，不是本文主算法。正文中控制在一小段即可。

---

## 14. 审稿人可能质疑与回应

### 14.1 质疑：这是不是普通 Transformer 分类？

回应要点：

- 输入不是普通全局时序，而是 service-path network signal；
- 模型显式融合 SLA condition；
- 模型输出 attribution；
- 实验有 service-path vs global graph 消融；
- 实验有 proactive intervention 验证。

### 14.2 质疑：仿真数据是否可信？

回应要点：

- 使用 trace-driven simulation；
- Alibaba / Google trace 用于 workload pattern；
- 多拓扑、多负载、多故障；
- 公开数据生成脚本；
- 重点研究方法有效性，而不是宣称完全替代真实运营商数据。

### 14.3 质疑：归因标签是否人工？

回应要点：

- 仿真环境中风险来源可由注入故障和超阈值指标自动生成；
- 归因用于辅助解释和干预，不作为唯一评价标准；
- 使用 top-k attribution 或 node/link/metric 三层评估减少标签噪声影响。

### 14.4 质疑：主动干预是否太简单？

回应要点：

- 本文重点不是提出复杂编排算法；
- proactive intervention 是验证 early detection 是否 actionable；
- 复杂优化和闭环编排留给后续工作。

---

## 15. 最终英文贡献句

最终论文中建议直接使用下面这 3 条贡献句。

```text
First, we formulate SLA violation early detection in cloud-edge services as a service-path network signal modeling problem, where each service is represented by a path-centric subgraph instead of global QoS records.

Second, we propose SPARTA, an SLA-conditioned temporal-topology risk modeling framework that jointly encodes historical node, link, and service states with service-specific SLA constraints for proactive risk detection.

Third, we introduce lightweight node-, link-, and metric-level attribution heads and validate that attribution-guided proactive intervention can reduce SLA violation rates.
```

---

## 16. 最终论文摘要压缩版

如果篇幅紧，可以用下面更短的摘要。

```text
Cloud-edge services often suffer from SLA violations caused by accumulated degradation along service paths, such as edge-node overload, link congestion, and request bursts. Existing SLA prediction methods usually rely on global QoS records and provide limited attribution for proactive management. This paper formulates SLA violation early detection as a service-path network signal modeling problem and proposes SPARTA, a Service-Path Aware Temporal Risk Attribution framework. SPARTA jointly encodes historical node, link, and service states, service-path topology, and service-specific SLA constraints to predict normal, risky, and violated states within a future horizon. It further identifies risk-related nodes, links, and dominant metrics through lightweight attribution heads. Trace-driven cloud-edge simulations show that SPARTA improves risk recall and macro-F1 over classical machine learning, sequence, Transformer-based, and spatio-temporal graph baselines. Attribution-guided proactive intervention also reduces SLA violation rates, demonstrating the practical value of early risk detection.
```

---

## 17. 最终写作执行顺序

建议按照这个顺序写，不要从头到尾硬写。

### 第一步：先写 Problem Formulation

把符号、任务、标签、输入输出定死。

### 第二步：写 Method

确保方法和代码一致。

### 第三步：写 Experiments

根据真实实验结果写分析。

### 第四步：写 Introduction

Introduction 要根据最终实验卖点来写。

### 第五步：写 Abstract 和 Conclusion

最后写，避免和正文不一致。

---

## 18. 最终判断

如果初稿按照本报告写，论文会呈现为：

```text
一个面向 ICASSP 的网络信号早期检测与归因方法论文
```

而不是：

```text
普通边缘计算调度论文
```

也不是：

```text
普通 Transformer 分类应用论文
```

这对第一篇 CCF B 会议投稿最重要。

最终主线保持：

```text
service-path network signal formulation
+ SLA-conditioned temporal-topology risk modeling
+ attribution-guided proactive validation
```

这三点就是全文的核心，不要再增加更多创新点。
