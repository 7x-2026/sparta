# SPARTA 顶层设计与 MVP 跑通后的渐进扩展路线

> 项目名称：SPARTA  
> 建议论文题目：Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services  
> 本文件用途：在 Windows MVP 跑通之后，作为后续逐步扩展 SPARTA 系统的**顶层设计、版本路线、接口治理和实验建设总纲**。  
> 适用对象：你本人、Codex、后续代码助手、论文实验整理。  
> 核心原则：**MVP 先不动；跑通后再按阶段扩展。每次只扩展一类能力，并用测试门禁确认不会破坏已有闭环。**

---

## 0. 先给结论：现有 5 个文件应该怎样分工

你现在的 5 个文件不是完全重复，而是处在不同层级。问题在于它们有些地方同时试图规定“当前 MVP”和“未来正式实验”，导致维度、文件格式、字段命名出现冲突。顶层设计需要把它们分层使用。

| 文件 | 建议定位 | 是否作为当前 Codex 实现最高优先级 | 使用方式 |
|---|---|---:|---|
| `SPARTA_Windows_MVP_Code_Spec_Revised.md` | 当前 Windows MVP 的实现契约 | 是 | 先按它把 debug pipeline 跑通，不要急着接真实 trace、EdgeSimPy 或复杂 baseline |
| `SPARTA_Win11_MVP_Codex_Build_Guide.md` | 较早的 Win11 Codex 搭建说明 | 否 | 只保留其中合理结构和测试思想；其中 `.npz`、`10/8/6/5` 等口径不要直接覆盖 Revised 版本 |
| `SPARTA_Model_Data_Interface_Spec.md` | 未来正式模型接口和层数选择参考 | 否 | MVP 后用于升级模型特征维度、层数敏感性、baseline 对齐、shape assert |
| `SPARTA_Code_Implementation_Spec.md` | 完整代码实现规格和实验结果格式参考 | 否 | MVP 后用于补齐 loss、评价指标、消融、proactive intervention、可视化和结果文件 |
| `SPARTA_Dataset_Processing_Spec.md` | 正式数据处理路线参考 | 否 | MVP 后用于接入 Alibaba trace、workload profile、EdgeSimPy 和正式样本构造 |

因此，当前正确工作方式是：

```text
第一步：严格按 Revised MVP 文件跑通最小闭环；
第二步：冻结 MVP 作为 v0.1；
第三步：按本顶层设计逐步引入正式数据、正式模型接口、正式 baseline 和论文实验；
第四步：每个阶段都保留可回退版本，不允许一次性大改所有模块。
```

---

## 1. 顶层设计的核心目标

SPARTA 不应该被实现成一个只会跑一组 synthetic debug 数据的脚本集合，而应该逐步演化成一个可复现实验系统。它的顶层目标包括四层。

第一层是工程闭环：从数据生成、样本构造、Dataset、模型训练、评估、checkpoint、结果 CSV 到测试脚本都能稳定运行。

第二层是研究闭环：模型不仅能分类 normal / risky / violated，还能给出风险节点、风险链路和风险指标归因，并能通过消融证明 service-path subgraph、temporal encoder、topology encoder、SLA gate 和 attribution heads 的必要性。

第三层是数据可信闭环：从纯 synthetic 数据逐步过渡到 Alibaba trace 驱动的 workload profile，再注入 EdgeSimPy 生成云边服务运行日志，避免论文中被质疑“数据完全随机、任务不真实”。

第四层是论文实验闭环：主结果、baseline、消融、horizon、depth sensitivity、attribution case、proactive intervention 都能由统一脚本生成，最终结果可以直接进入论文表格和图。

---

## 2. 系统总架构

SPARTA 顶层系统可以分成 8 个子系统。

```text
SPARTA/
├── 1. data_source_layer
│   ├── synthetic debug trace
│   ├── Alibaba microservice trace profiles
│   └── optional Google node workload profiles
│
├── 2. simulation_layer
│   ├── run_synthetic_simulation.py
│   └── run_edgesimpy.py
│
├── 3. log_layer
│   ├── node_log.csv
│   ├── link_log.csv
│   ├── service_log.csv
│   ├── path_log.csv
│   └── sla_log.csv
│
├── 4. sample_layer
│   ├── build_path_graph.py
│   ├── generate_labels.py
│   ├── normalize.py
│   └── split_dataset.py
│
├── 5. model_layer
│   ├── common.py / PathTokenEncoder
│   ├── sparta.py
│   ├── lstm.py / transformer.py / gru.py / tcn.py
│   ├── stgcn.py / patchtst.py
│   └── tree baselines: rf / xgboost
│
├── 6. training_eval_layer
│   ├── train.py
│   ├── evaluate.py
│   ├── trainer.py
│   ├── metrics.py
│   └── losses/
│
├── 7. experiment_layer
│   ├── run_debug_pipeline.py
│   ├── run_synthetic_full.py
│   ├── run_alibaba_edgesimpy.py
│   ├── run_baselines.py
│   ├── run_ablations.py
│   └── run_horizon_depth.py
│
└── 8. analysis_layer
    ├── collect_results.py
    ├── visualize_attribution.py
    ├── visualize_horizon.py
    ├── visualize_proactive.py
    └── paper_tables.py
```

所有子系统必须围绕一个原则设计：**上游可以替换，但下游接口不变**。也就是说，synthetic simulator 和 EdgeSimPy 都必须输出同一套 `node_log/link_log/service_log/path_log/sla_log`；Alibaba trace 和 synthetic trace 都必须最终进入同一套 workload profile 或 raw log schema；LSTM、Transformer 和 SPARTA 都必须通过统一训练器训练。

---

## 3. 版本路线图

### 3.1 v0.1：Windows Debug MVP

目标是跑通最小闭环。这个阶段不追求论文结果。

输入：

```text
synthetic trace
```

输出：

```text
train.pkl / val.pkl / test.pkl
checkpoints/debug/{model}_best.pth
results/debug/{model}_test_results.csv
```

必须满足：

```text
1. run_debug_pipeline.py 一键运行成功；
2. normal / risky / violated 三类都存在；
3. Dataset shape 正确；
4. SPARTA forward 输出 risk/node/link/metric 四类 logits；
5. loss 非 NaN，backward 成功；
6. LSTM / Transformer / SPARTA 都能保存和加载 checkpoint；
7. evaluate.py 能输出 CSV；
8. pytest 基础测试通过。
```

此阶段以 `SPARTA_Windows_MVP_Code_Spec_Revised.md` 为最高优先级。

### 3.2 v0.2：MVP 稳定化

目标是把“能跑”变成“稳定可复现”。此阶段仍然使用 synthetic/debug 数据。

新增内容：

```text
1. metadata.json 中记录数据配置、标签分布和样本 shape；
2. all_test_results.csv 汇总三个模型结果；
3. 训练日志固定字段：epoch, train_loss, val_loss, val_accuracy, val_macro_f1, val_risk_recall, val_violation_recall, lr；
4. checkpoint 加载测试；
5. fast_dev_run 模式；
6. shape_check.py 全局检查；
7. seed 固定与重复运行一致性检查。
```

验收标准：

```text
同一配置连续跑两次，不要求指标完全一致，但不应出现类别缺失、NaN、checkpoint 加载失败或 CSV 缺字段。
```

### 3.3 v0.3：Synthetic Full 规模扩展

目标是在不接真实 trace 和 EdgeSimPy 的前提下，把 debug 小样本扩展到接近正式实验规模，验证训练速度、显存、日志和结果收集是否稳定。

建议配置：

```yaml
data:
  input_window: 12
  pred_horizon: 5
  max_nodes: 8
  max_links: 12
  num_edge_nodes: 20
  num_access_nodes: 10
  num_users: 300
  num_services: 50
  num_steps: 720
  num_scenarios: 3

model:
  hidden_dim: 128
  temporal_layers: 2
  topology_layers: 2
  n_heads: 4

train:
  batch_size: 64
  epochs: 50
```

此阶段必须保持数据保存格式和字段命名与 v0.1 一致，避免为了扩规模再次引入接口分裂。

### 3.4 v1.0：正式数据接口升级

目标是将当前 MVP 的简化特征升级到正式论文更合理的特征空间，同时仍然允许旧 MVP 数据通过 adapter 读取。

推荐正式特征维度：

```text
node_feat_dim = 10
link_feat_dim = 8
service_feat_dim = 6
sla_feat_dim = 5
```

升级原因：

```text
node 侧从简单类型编码扩展为 cloud/edge/access/user one-hot，并加入 request_load_on_node；
service 侧从 service_type_id_norm 扩展为 latency/reliability/cost one-hot，并加入 last_violation；
link 和 SLA 维度保持稳定。
```

为了避免破坏 MVP，升级必须按以下顺序进行：

```text
1. 新增 configs/sparta_v1.yaml，不覆盖 sparta_debug.yaml；
2. 新增 feature_schema_version 字段；
3. Dataset 根据 metadata 中的 feature_schema_version 自动检查维度；
4. build_path_graph.py 支持 schema=v0 和 schema=v1；
5. SPARTA 初始化从 config 读取维度，不写死；
6. 测试文件分别覆盖 v0 和 v1 shape；
7. v1 跑通后，再把正式实验默认切到 10/8/6/5。
```

### 3.5 v1.1：Alibaba Trace Profile 接入

目标是把公开 trace 变成 workload profile，而不是直接把原始 trace 喂给模型。

处理流程：

```text
Alibaba raw trace
    ↓
prepare_alibaba_node.py
prepare_alibaba_ms_resource.py
prepare_alibaba_ms_rtqps.py
prepare_alibaba_callgraph.py
    ↓
alibaba_node_profile.csv
alibaba_ms_resource_profile.csv
alibaba_service_qos_profile.csv
alibaba_call_edges.csv
alibaba_service_dependency.csv
    ↓
build_workload_profiles.py
    ↓
service_load_profiles.csv
node_load_profiles.csv
service_dependency_edges.csv
```

第一篇论文建议优先使用 Alibaba Microservices Trace 2021。Google 数据只作为后续可选扩展，不应在第一篇里强行加入。

### 3.6 v1.2：EdgeSimPy 接入

目标是把 synthetic simulator 替换成 EdgeSimPy，但下游样本构造、Dataset、模型和训练脚本不变。

EdgeSimPy 必须输出与 synthetic simulator 一致的日志：

```text
node_log.csv
link_log.csv
service_log.csv
path_log.csv
sla_log.csv
```

禁止让 `build_path_graph.py` 直接依赖 EdgeSimPy 内部对象。它只能读取 CSV 日志。这样做是为了保证 simulator 可替换。

### 3.7 v1.3：Baseline 完整化

在 LSTM、Transformer、SPARTA 三个模型稳定后，再逐步加入：

```text
GRU
TCN
RF
XGBoost
STGCN-service-path
STGCN-global
PatchTST
TimesNet 或其他时序模型，可选
```

所有 baseline 必须满足公平性：相同 split、相同 L/H、相同风险标签、相同归一化、相同评价指标。无归因能力的 baseline 的 attribution 指标写 `nan`，不能写 0。

### 3.8 v1.4：Ablation、Horizon、Depth 实验

此阶段用于支撑论文核心结论。

必须完成：

```text
1. full SPARTA
2. w/o SLA gate
3. w/o topology encoder
4. w/o attribution heads
5. service-path graph vs global graph
6. L = 6 / 12 / 24
7. H = 1 / 3 / 5 / 10
8. temporal_layers = 1 / 2 / 3
9. topology_layers = 1 / 2 / 3
```

模型选择标准不只看 Accuracy，应优先看：

```text
Macro-F1
Risk Recall
Violation Recall
Attribution Accuracy
```

推荐综合分数：

```text
score = 0.4 * MacroF1
      + 0.3 * RiskRecall
      + 0.2 * ViolationRecall
      + 0.1 * AttrAcc
```

### 3.9 v1.5：Attribution Case 与 Proactive Intervention

此阶段用于增强论文故事性。

Attribution case 要回答：

```text
模型预测 risky/violated 后，是否能定位到具体节点、链路和风险指标？
这个归因是否和未来真实风险源一致？
```

Proactive intervention 要回答：

```text
如果在 risky 阶段提前迁移服务或重路由，是否能降低未来 SLA violation rate？
代价是否可接受？
```

对比策略：

```text
No Intervention
Reactive
Proactive-SPARTA
Oracle，可选
```

---

## 4. 统一接口治理原则

### 4.1 文件格式治理

当前 MVP 阶段统一使用 `.pkl`。这和 Revised MVP 文件一致，能够最大限度降低 Windows 下的数据读写复杂度。

正式阶段可以增加 `.npz` 导出，但 `.npz` 只能作为可选优化，不能同时让不同脚本一部分读 `.pkl`、一部分读 `.npz`。

推荐策略：

```text
v0.1-v1.2：pkl 为主格式；
v1.3 以后：如果样本量过大，可以新增 export_npz.py；
Dataset 必须同时能读取 pkl 和 npz，但训练配置只能指定一种格式；
论文实验中固定一种格式，避免复现混乱。
```

### 4.2 字段命名治理

当前 MVP 使用：

```python
risk_label
risk_node
risk_link
risk_metric
attr_mask
```

部分旧规格使用：

```python
risk_node_label
risk_link_label
risk_metric_label
```

顶层设计建议保留 MVP 的无 `_label` 命名作为主代码字段，同时在 Dataset 中兼容旧字段：

```python
if "risk_node" not in sample and "risk_node_label" in sample:
    sample["risk_node"] = sample["risk_node_label"]
```

对 Codex 的要求是：**训练代码内部只使用一种命名，不允许同一个 loss 同时访问两套字段。**

### 4.3 normal 样本归因治理

所有阶段统一规定：

```python
if risk_label == 0:
    risk_node = -100
    risk_link = -100
    risk_metric = -100
    attr_mask = 0
else:
    attr_mask = 1
```

训练时使用：

```python
ignore_index = -100
```

禁止将 `-1` 写入最终 train/val/test 文件。`-1` 只能作为临时内部值，保存前必须转换为 `-100`。

### 4.4 归一化治理

debug 阶段不调用 `normalize.py`，因为 synthetic 生成时已经缩放到合理范围。

正式 trace 阶段必须启用 train-only scaler：

```text
scaler 只能在 train split 上 fit；
val/test 只能 transform；
scaler 保存到 data/{dataset}/scaler.json；
metadata.json 中记录 scaler_path 和 feature_schema_version。
```

禁止用全数据统计 min/max，否则会引入时间泄漏。

### 4.5 配置治理

所有实验配置必须显式写出：

```yaml
project:
  name:
  output_dir:
  checkpoint_dir:
  log_dir:

data:
  mode:
  feature_schema_version:
  input_window:
  pred_horizon:
  max_nodes:
  max_links:
  node_feat_dim:
  link_feat_dim:
  service_feat_dim:
  sla_feat_dim:
  train_path:
  val_path:
  test_path:

model:
  name:
  hidden_dim:
  temporal_layers:
  topology_layers:
  n_heads:
  dropout:
  use_sla_gate:
  use_topology:
  use_attribution:

loss:
  risk_loss:
  lambda_node:
  lambda_link:
  lambda_metric:
  ignore_index:

train:
  batch_size:
  epochs:
  lr:
  weight_decay:
  grad_clip:
  num_workers:
  device:
  best_metric:
```

任何脚本都不应在代码里硬编码这些关键参数。

---

## 5. 数据层顶层设计

### 5.1 数据源分层

数据源分为三类。

第一类是 debug synthetic trace，用于跑通代码，不用于论文正式结论。

第二类是 trace-driven synthetic simulation，即从 Alibaba trace 提取 workload profile，再注入 synthetic simulator。这是 EdgeSimPy 接入前的过渡版本。

第三类是 trace-driven EdgeSimPy simulation，即从 Alibaba trace 提取 workload profile，再注入 EdgeSimPy。这是第一篇论文最推荐的正式数据路线。

### 5.2 统一日志接口

无论上游来自 synthetic simulator 还是 EdgeSimPy，下游只认以下五个日志：

```text
node_log.csv
link_log.csv
service_log.csv
path_log.csv
sla_log.csv
```

#### node_log.csv

正式建议字段：

```csv
time,node_id,node_type,cpu_util,mem_util,queue_len,available_cpu,available_mem,request_load_on_node,active_services,is_failed
```

MVP 可以只保留前 8 或 9 个字段，但正式 schema 中要保留可扩展字段。

#### link_log.csv

正式建议字段：

```csv
time,src,dst,delay_ms,bandwidth,loss,jitter_ms,queue_delay_ms,bandwidth_util,is_congested,is_failed
```

#### service_log.csv

正式建议字段：

```csv
time,service_id,user_id,service_type,request_rate,response_time_ms,current_edge,cloud_node,last_violation,dependency_count
```

#### path_log.csv

正式建议字段：

```csv
time,service_id,path_nodes,path_links,path_delay_ms,path_loss,path_bandwidth_min,bottleneck_node,bottleneck_link,candidate_edges
```

#### sla_log.csv

正式建议字段：

```csv
service_id,service_type,max_delay_ms,max_loss,min_bandwidth,reliability_req,cost_weight,priority
```

### 5.3 样本构造

每个样本对应一个服务 `s` 在当前时间 `t` 的历史窗口：

```text
input window = [t-L+1, ..., t]
future window = [t+1, ..., t+H]
```

样本保存字段：

```python
sample = {
    "sample_id": str,
    "scenario_id": str,
    "service_id": int,
    "time": int,
    "node_ids": list[str],
    "link_ids": list[str],
    "node_x": np.ndarray,       # [L, max_nodes, node_feat_dim]
    "link_x": np.ndarray,       # [L, max_links, link_feat_dim]
    "service_x": np.ndarray,    # [L, service_feat_dim]
    "sla_x": np.ndarray,        # [sla_feat_dim]
    "adj": np.ndarray,          # [max_nodes, max_nodes]
    "node_mask": np.ndarray,    # [max_nodes]
    "link_mask": np.ndarray,    # [max_links]
    "risk_label": int,
    "risk_node": int,
    "risk_link": int,
    "risk_metric": int,
    "attr_mask": int
}
```

### 5.4 标签定义

风险标签：

```text
0 = normal
1 = risky
2 = violated
```

指标归因：

```text
0 = delay
1 = loss
2 = cpu
3 = queue
4 = bandwidth
```

violated 条件：

```text
future response_time_ms >= max_delay_ms
or future path_loss >= max_loss
or future is_failed == 1
```

risky 条件：

```text
future response_time_ms >= 0.75 * max_delay_ms
or future path_loss >= 0.75 * max_loss
or max CPU on path >= 0.85
or max queue_len_norm on path >= 0.85
or max bandwidth_util on path >= 0.85
```

normal 条件：

```text
以上条件均不满足
```

---

## 6. 模型层顶层设计

### 6.1 SPARTA 主模型

SPARTA 由五个核心部分组成：

```text
Heterogeneous Feature Embedding
Temporal Encoder
Topology Encoder
SLA Gate
Risk + Attribution Heads
```

默认正式模型：

```yaml
model:
  hidden_dim: 128
  temporal_layers: 2
  topology_layers: 2
  n_heads: 4
  ffn_dim: 512
  dropout: 0.1
  use_sla_gate: true
  use_topology: true
  use_attribution: true
```

### 6.2 Forward 逻辑

```text
node_x/link_x/service_x/sla_x
        ↓
NodeMLP / LinkMLP / ServiceMLP / SlaMLP
        ↓
masked node/link pooling
        ↓
path_token = node_ctx + link_ctx + service_h
        ↓
Temporal Encoder → z_temp
        ↓
Topology Encoder → z_topo
        ↓
z = LayerNorm(z_temp + z_topo)
        ↓
z_sla = z * (1 + sigmoid(SLA gate))
        ↓
risk_logits, node_logits, link_logits, metric_logits
```

### 6.3 Loss

```text
L = CE(risk_logits, risk_label)
  + lambda_node   * CE(node_logits, risk_node, ignore_index=-100)
  + lambda_link   * CE(link_logits, risk_link, ignore_index=-100)
  + lambda_metric * CE(metric_logits, risk_metric, ignore_index=-100)
```

默认：

```yaml
lambda_node: 0.3
lambda_link: 0.3
lambda_metric: 0.2
```

如果分类效果被归因任务拖累，可先调成：

```yaml
lambda_node: 0.1
lambda_link: 0.1
lambda_metric: 0.1
```

### 6.4 Baseline 对齐

baseline 必须与 SPARTA 使用同一份样本数据，但不能使用 SPARTA 的私有模块。

LSTM / GRU / Transformer / TCN 使用统一 `PathTokenEncoder`：

```text
node_pool = masked_mean(node_x)
link_pool = masked_mean(link_x)
sla_expand = repeat(sla_x over L)
raw = concat(node_pool, link_pool, service_x, sla_expand)
path_token = MLP(raw)
```

STGCN 可以使用 service-path graph 或 global graph，但必须在结果表中标明。

RF / XGBoost 使用统计特征，不应直接 flatten 过高维张量造成不公平优势或维度爆炸。

---

## 7. 实验层顶层设计

### 7.1 主结果表

文件：

```text
results/main_results.csv
```

字段：

```csv
method,accuracy,macro_f1,risk_recall,violation_recall,auc,node_attr_acc,link_attr_acc,metric_attr_acc,inference_time_ms,params
```

### 7.2 消融表

文件：

```text
results/ablation_results.csv
```

字段：

```csv
variant,accuracy,macro_f1,risk_recall,violation_recall,node_attr_acc,link_attr_acc,metric_attr_acc,params
```

消融版本：

```text
full SPARTA
w/o SLA
w/o topology
w/o attribution
SPARTA-global graph
binary label，可选
```

### 7.3 Horizon 表

文件：

```text
results/horizon_results.csv
```

字段：

```csv
method,horizon,macro_f1,risk_recall,violation_recall,auc
```

推荐 horizon：

```text
H = 1, 3, 5, 10
```

### 7.4 Depth sensitivity 表

文件：

```text
results/depth_results.csv
```

字段：

```csv
temporal_layers,topology_layers,hidden_dim,macro_f1,risk_recall,violation_recall,node_attr_acc,link_attr_acc,metric_attr_acc,params,inference_time_ms
```

### 7.5 Attribution case 文件

文件：

```text
results/attribution_cases.csv
```

字段：

```csv
sample_id,scenario_id,service_id,time,true_risk_label,pred_risk_label,true_node,pred_node,true_link,pred_link,true_metric,pred_metric,response_time_future,path_loss_future
```

### 7.6 Proactive intervention 表

文件：

```text
results/proactive_results.csv
```

字段：

```csv
strategy,violation_rate,avg_response_time,intervention_cost,num_interventions,success_rate
```

---

## 8. 测试门禁设计

每个阶段都必须有测试门禁。

### 8.1 MVP 门禁

```text
test_dataset_shape.py
test_model_forward.py
test_loss_backward.py
test_train_one_batch.py
test_checkpoint_load_eval.py
test_run_debug_pipeline.py
```

### 8.2 正式数据门禁

```text
test_trace_profile_fields.py
test_edgesimpy_log_schema.py
test_train_only_scaler.py
test_time_based_split.py
test_label_distribution.py
```

### 8.3 实验门禁

```text
test_result_csv_schema.py
test_baseline_nan_attribution.py
test_ablation_config_flags.py
test_horizon_config_consistency.py
```

任何新增模块必须先通过对应门禁，再进入主分支。

---

## 9. Codex 后续分阶段任务拆分

### 9.1 MVP 跑通后第一个 Codex 任务

```text
请在不改变现有 MVP 数据接口的前提下，增强项目的实验稳定性：
1. 增加 metadata.json 的字段完整性；
2. 增加 results/debug/all_test_results.csv 汇总文件；
3. 增加 fast_dev_run 参数；
4. 增加 checkpoint 加载评估测试；
5. 确保 run_debug_pipeline.py 连续两次运行不会因为旧文件冲突失败。
不要接入真实 trace，不要改 node_feat_dim/service_feat_dim，不要切换 npz。
```

### 9.2 第二个 Codex 任务：Synthetic Full

```text
请在保留 debug pipeline 的前提下，新增 synthetic_full pipeline：
1. 新增 configs/sparta_synthetic_full.yaml；
2. 设置 L=12, H=5, max_nodes=8, max_links=12；
3. 支持多个 scenario_id；
4. 输出 results/synthetic_full/all_test_results.csv；
5. 不改变 debug 配置和 debug 输出路径。
```

### 9.3 第三个 Codex 任务：特征 schema v1

```text
请新增 feature_schema_version=v1，支持 node_feat_dim=10, service_feat_dim=6：
1. build_path_graph.py 支持 schema v0/v1；
2. Dataset 根据 metadata 校验 shape；
3. SPARTA 和 PathTokenEncoder 从 config 读取维度；
4. 新增 tests/test_schema_v1_shape.py；
5. 不删除 v0 兼容逻辑。
```

### 9.4 第四个 Codex 任务：Alibaba workload profile

```text
请新增 Alibaba trace profile 预处理模块，但不要直接训练模型：
1. prepare_alibaba_node.py；
2. prepare_alibaba_ms_resource.py；
3. prepare_alibaba_ms_rtqps.py；
4. prepare_alibaba_callgraph.py；
5. build_workload_profiles.py；
6. 每个脚本只输出中间 CSV 和统计报告；
7. 不接 EdgeSimPy，不改训练脚本。
```

### 9.5 第五个 Codex 任务：EdgeSimPy 替换 synthetic simulator

```text
请新增 run_edgesimpy.py，使其读取 workload_profiles 并输出与 synthetic simulator 完全同字段的 node_log/link_log/service_log/path_log/sla_log。
下游 build_path_graph.py、generate_labels.py、split_dataset.py 不允许改接口，只允许修 bug。
```

### 9.6 第六个 Codex 任务：论文实验脚本

```text
请新增实验调度脚本：
run_baselines.py、run_ablations.py、run_horizon.py、run_depth.py、collect_results.py。
所有脚本都从 configs/experiments/*.yaml 读取配置，输出统一 CSV，不要手写临时路径。
```

---

## 10. 当前 5 个文件的主要冲突与顶层处理方案

### 10.1 `.pkl` 与 `.npz` 冲突

有的文件建议最终训练集用 `.npz`，修订版 MVP 明确 Windows MVP 用 `.pkl`。顶层设计处理为：

```text
当前 MVP：只用 .pkl；
正式阶段：仍以 .pkl 为主；
大规模实验如有性能需要，再增加可选 .npz export；
禁止同时让一半脚本读 pkl、一半脚本读 npz。
```

### 10.2 `node_feat_dim=8` 与 `node_feat_dim=10` 冲突

早期实现规格和修订版 MVP 倾向 `8/8/5/5`，模型接口文件倾向 `10/8/6/5`。顶层设计处理为：

```text
MVP 不改，保持当前能跑的维度；
正式模型接口升级时新增 schema v1；
v1 目标维度为 10/8/6/5；
通过 adapter 兼容 v0，不允许直接破坏已有 MVP。
```

### 10.3 `risk_node` 与 `risk_node_label` 冲突

顶层设计处理为：

```text
主代码内部统一用 risk_node/risk_link/risk_metric；
Dataset 兼容 risk_node_label/risk_link_label/risk_metric_label；
loss 层只读统一后的字段。
```

### 10.4 `normalize.py` 是否进入 debug pipeline

顶层设计处理为：

```text
debug 不调用 normalize.py；
正式 trace 阶段必须启用 train-only scaler；
normalize.py 作为正式数据阶段模块，而不是 MVP 模块。
```

### 10.5 EdgeSimPy 是否现在接入

顶层设计处理为：

```text
当前不接；
先 synthetic full；
再 Alibaba workload profile；
最后 EdgeSimPy；
EdgeSimPy 只负责生成日志，不进入模型代码。
```

### 10.6 Google 数据是否现在加入

顶层设计处理为：

```text
第一篇不加入 Google；
Alibaba Microservices Trace 2021 作为主数据源；
Google 作为期刊扩展或鲁棒性实验。
```

---

## 11. 论文级最终实验包

当系统进入论文实验阶段，最终应形成以下文件集合。

```text
results/
├── main_results.csv
├── ablation_results.csv
├── horizon_results.csv
├── depth_results.csv
├── attribution_results.csv
├── proactive_results.csv
├── complexity_results.csv
└── all_tables.xlsx，可选
```

```text
figures/
├── fig_pipeline.pdf
├── fig_model_architecture.pdf
├── fig_horizon_curve.pdf
├── fig_attribution_case.pdf
├── fig_proactive_violation_rate.pdf
└── fig_confusion_matrix.pdf
```

```text
checkpoints/
├── sparta_full_best.pth
├── lstm_best.pth
├── transformer_best.pth
├── stgcn_best.pth
└── patchtst_best.pth
```

```text
logs/
├── train_log_sparta.csv
├── train_log_baselines.csv
├── experiment_config_snapshot.json
└── environment.txt
```

---

## 12. 最终执行顺序建议

如果你现在已经有 MVP `.md`，后续实际执行顺序应为：

```text
1. 让 Codex 按 Revised MVP 文件实现 v0.1；
2. 本地运行 run_debug_pipeline.py；
3. 如果出错，只修 MVP，不扩展功能；
4. MVP 跑通后，保存一份 tag 或 zip，命名 sparta_v0_1_mvp_working；
5. 按本顶层设计做 v0.2 稳定化；
6. 做 v0.3 synthetic full；
7. 做 v1.0 schema v1；
8. 做 v1.1 Alibaba profile；
9. 做 v1.2 EdgeSimPy；
10. 做 v1.3 baseline；
11. 做 v1.4 消融和 horizon；
12. 做 v1.5 attribution/proactive；
13. 最后整理论文表格和图。
```

最重要的是：**不要在 MVP 还没跑通时就修改维度、接真实数据、换保存格式、上 EdgeSimPy、加一堆 baseline。** 这些都是后续阶段的事情。

---

## 13. 给 Codex 的顶层总提示词

可以在 MVP 跑通后，把下面这段作为后续扩展的总提示词给 Codex：

```text
你现在维护的是 SPARTA 项目。请不要破坏已经跑通的 Windows MVP debug pipeline。后续所有扩展必须遵守以下顶层设计：

1. 当前 MVP 使用 .pkl、risk_node/risk_link/risk_metric、attr_mask 和 debug synthetic 数据流；不要随意切换到 .npz。
2. 新功能必须通过新配置或新 pipeline 增加，不允许覆盖 sparta_debug.yaml。
3. 上游数据源可以变化，但下游必须统一为 node_log.csv、link_log.csv、service_log.csv、path_log.csv、sla_log.csv。
4. 模型和 baseline 必须使用统一 Dataset 和统一训练器。
5. normal 样本归因标签必须为 -100，attr_mask=0。
6. 正式 trace 阶段必须使用 train-only scaler，禁止全数据归一化。
7. 所有新增实验必须输出统一 CSV，并保证结果可被 collect_results.py 汇总。
8. 每次只完成一个阶段：先稳定 MVP，再扩 synthetic full，再做 schema v1，再接 Alibaba profile，再接 EdgeSimPy，再补 baseline、消融和 proactive intervention。
9. 每次改动后必须运行 pytest 和对应 pipeline。

请先阅读 SPARTA_Top_Level_Design_After_MVP.md，然后根据当前阶段实现对应功能。不要一次性实现全部路线图。
```

---

## 14. 总结

当前最合理的策略不是继续堆更多细节文件，而是建立“分层文件体系”：

```text
Revised MVP 文件：当前实现最高优先级；
本顶层设计文件：MVP 跑通后的扩展路线最高优先级；
Model/Data Interface 文件：正式模型接口参考；
Dataset Processing 文件：正式数据接入参考；
Code Implementation 文件：loss、评估、消融、结果格式参考。
```

这样做的好处是：Codex 不会在 `.pkl/.npz`、`8/5 与 10/6`、`risk_node 与 risk_node_label`、`synthetic 与 EdgeSimPy` 之间来回摇摆。你也可以在每个阶段明确告诉 Codex：“只做这一层，不要碰下一层”。
