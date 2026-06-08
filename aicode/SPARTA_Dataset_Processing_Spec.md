# SPARTA 数据集处理详细规格说明书

> 适用项目：**SPARTA: Service-Path Aware Temporal Risk Attribution for SLA Violation Early Detection in Cloud-Edge Services**  
> 目标：把 Alibaba / Google 原始 trace 处理成 SPARTA 可训练的数据集，明确每个数据集刚下载时是什么样、需要经过哪些处理、处理后是什么样、数据量大概多少、每个处理脚本大概怎么写。  
> 本文件用于给 AI / Codex / 代码助手直接对照实现预处理代码，重点避免字段不统一、时间粒度不一致、标签规则不清楚、模型输入 shape 对不上等问题。

---

## 0. 总体结论

SPARTA 第一篇 ICASSP 论文不应该直接把 Alibaba / Google trace 当作最终训练数据。正确做法是：

```text
Alibaba / Google 原始 trace
        ↓
抽取真实负载模式、服务调用率、响应时间、节点资源利用率
        ↓
处理成统一的 workload profile CSV
        ↓
注入 EdgeSimPy 云边仿真环境
        ↓
生成云边网络日志 node_log/link_log/service_log/path_log/sla_log
        ↓
构造 service-path subgraph 样本
        ↓
生成 risk label 与 attribution label
        ↓
得到 SPARTA 训练数据 train/val/test.pkl
```

也就是说，真实 trace 的角色是：

1. 提供真实请求波动；
2. 提供真实服务响应时间分布；
3. 提供真实节点 CPU / memory 变化趋势；
4. 提供微服务依赖关系；
5. 提升仿真实验可信度。

最终模型训练用的数据不是原始 trace，而是处理后的：

```text
service-path subgraph samples
```

每个样本都包含：

```python
{
    "node_features": [L, Np, Fn],
    "link_features": [L, Ep, Fe],
    "service_features": [L, Fs],
    "sla_features": [Fc],
    "adj_matrix": [Np, Np],
    "node_mask": [Np],
    "link_mask": [Ep],
    "risk_label": int,
    "risk_node": int,
    "risk_link": int,
    "risk_metric": int
}
```

---

## 1. 数据源选择与优先级

### 1.1 主数据源：Alibaba Microservices Trace 2021

这是第一篇 ICASSP 最推荐使用的数据源。

原因：

1. 它包含微服务调用率 MCR 和响应时间 RT；
2. 它包含微服务资源使用情况；
3. 它包含服务调用图；
4. 它和“服务路径风险检测”这个论文故事最贴合。

官方说明中，Alibaba microservices trace 包含四部分：`node`、`MS_Resource_Table`、`MS_Metrics_Table / MS_MCR_RT_Table` 和 `MS_CallGraph_Table`。其中 node 记录 1300+ BM 节点 CPU / memory 利用率，MS resource 记录 90000+ containers 和 1300+ MS 的 CPU / memory，MS metrics 记录 MCR 和 RT，MS call graph 包含 2000 万级调用图样本。官方 README 还给出了目录大小：node 约 1.1Gi，MSCallGraph 约 25Gi，MSResource 约 16Gi，MSRTQps 约 19Gi。  
参考链接：`https://github.com/alibaba/clusterdata/blob/master/cluster-trace-microservices-v2021/README.md`

### 1.2 可选数据源：Alibaba Cluster Trace 2018

这个数据源不建议第一篇作为主数据源，因为体量很大，处理成本高。它适合做补充节点负载模式。

官方说明中，cluster-trace-v2018 约 4000 machines，持续 8 天，包括 6 张表：`machine_meta.csv`、`machine_usage.csv`、`container_meta.csv`、`container_usage.csv`、`batch_instance.csv`、`batch_task.csv`。压缩包约 49G，解压后约 280G。  
参考链接：`https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md`

第一篇 ICASSP 可以只取：

```text
machine_usage.csv
container_usage.csv
```

用于构造节点负载模式，不建议处理 batch DAG。

### 1.3 可选数据源：Google ClusterData 2019

Google 2019 trace 包含 2019 年 5 月 8 个 Borg cells 的 workload traces，并且官方说明其压缩体量约 2.4TiB，主要通过 Google BigQuery 使用。因此第一篇不建议本地下载完整数据。  
参考链接：`https://github.com/google/cluster-data/blob/master/ClusterData2019.md`

建议用法：

1. 如果你熟悉 BigQuery，只抽样聚合 CPU usage / task arrival；
2. 如果不熟悉，不要第一阶段使用；
3. 第一篇可以只用 Alibaba microservices trace，Google 作为后续扩展。

### 1.4 可选数据源：Google ClusterData 2011

Google 2011 trace 代表 29 天 Borg cell 信息，约 12.5k machines，压缩体量约 41GB。  
参考链接：`https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md`

它比 Google 2019 更适合本地下载，但与“微服务响应时间 / 服务调用图”关系不如 Alibaba microservices trace。第一篇不建议优先使用。

---

## 2. 第一阶段推荐数据组合

为了控制工作量，第一篇 ICASSP 推荐数据组合如下：

```text
必须使用：Alibaba Microservices Trace 2021
可选使用：Alibaba Cluster Trace 2018 的小规模 machine_usage 抽样
暂不使用：Google 2019 完整数据
暂不使用：Google 2011 完整数据
```

最小可行版本：

```text
Alibaba node
Alibaba MSResource
Alibaba MSRTQps
Alibaba MSCallGraph 小样本
        ↓
生成 alibaba_node_profile.csv
生成 alibaba_service_profile.csv
生成 alibaba_call_edges.csv
        ↓
注入 EdgeSimPy
        ↓
生成 SPARTA 训练样本
```

---

## 3. Alibaba Microservices Trace 原始数据说明

下载后原始目录建议放在：

```text
data/traces/alibaba/clusterdata/cluster-trace-microservices-v2021/
```

官方 fetch 后包含：

```text
node/Node.tar.gz
MSCallGraph/MSCallGraph_*.tar.gz
MSResource/MSResource_*.tar.gz
MSRTQps/MSRTQps_*.tar.gz
```

解压命令：

```bash
cd data/traces/alibaba/clusterdata/cluster-trace-microservices-v2021/node
for file in `ls *.tar.gz`; do tar -xzf $file; done

cd ../MSResource
for file in `ls *.tar.gz`; do tar -xzf $file; done

cd ../MSRTQps
for file in `ls *.tar.gz`; do tar -xzf $file; done

cd ../MSCallGraph
for file in `ls *.tar.gz`; do tar -xzf $file; done
```

---

## 4. Alibaba node 原始数据与处理

### 4.1 原始数据长什么样

官方 README 给出的 node 表字段为：

```text
timestamp
nodeid
cpu_utilization
memory_utilization
```

示例：

```csv
timestamp,nodeid,cpu_utilization,memory_utilization
1000,ff1fb31957db...,0.7236,0.7383
```

字段含义：

| 字段 | 含义 | 处理方式 |
|---|---|---|
| timestamp | 时间戳，毫秒级，范围约 0 到 43200000 | 转成分钟级 time_slot |
| nodeid | BM 节点 ID | 映射到 simulated node profile |
| cpu_utilization | 节点 CPU 利用率 | 保留，裁剪到 [0,1] |
| memory_utilization | 节点 memory 利用率 | 保留，裁剪到 [0,1] |

官方说明中，node 的记录间隔为 30s，即 30000 ms。

### 4.2 处理目标

处理成：

```text
data/processed/alibaba_node_profile.csv
```

输出字段：

```csv
time_slot,node_profile_id,cpu_scale,mem_scale,cpu_trend,mem_trend,burst_flag
```

解释：

| 字段 | 含义 |
|---|---|
| time_slot | 统一分钟级时间槽，0 到 719 |
| node_profile_id | 节点 profile 编号，不直接暴露原 nodeid |
| cpu_scale | CPU 负载缩放因子，范围 [0,1] |
| mem_scale | memory 负载缩放因子，范围 [0,1] |
| cpu_trend | 最近 3 个 time_slot 的 CPU 变化趋势 |
| mem_trend | 最近 3 个 time_slot 的 memory 变化趋势 |
| burst_flag | 是否处于高负载突发阶段 |

### 4.3 处理逻辑

1. 读取所有 node CSV；
2. 去除空值；
3. 对 `cpu_utilization`、`memory_utilization` 裁剪到 `[0,1]`；
4. 将 `timestamp` 转为 `time_slot = floor(timestamp / 60000)`；
5. 对每个 `nodeid, time_slot` 做均值聚合；
6. 选择记录完整度较高的 top N 个 nodeid；
7. 将原始 nodeid 重新编码为 `node_profile_id`；
8. 计算趋势字段；
9. 输出 CSV。

### 4.4 推荐数据量

第一篇不需要用全部 1300+ 节点。建议：

```text
选取 100 个 node profile
时间槽 720 个
输出约 72,000 行
```

### 4.5 脚本设计

文件：

```text
src/preprocessing/prepare_alibaba_node.py
```

命令：

```bash
python src/preprocessing/prepare_alibaba_node.py \
  --raw_dir data/traces/alibaba/clusterdata/cluster-trace-microservices-v2021/node \
  --out data/processed/alibaba_node_profile.csv \
  --slot_ms 60000 \
  --top_nodes 100
```

核心伪代码：

```python
def main(args):
    df = load_all_csv(args.raw_dir)
    df = clean_node_df(df)
    df["time_slot"] = (df["timestamp"] // args.slot_ms).astype(int)
    df = df.groupby(["nodeid", "time_slot"]).agg({
        "cpu_utilization": "mean",
        "memory_utilization": "mean"
    }).reset_index()
    selected_nodes = select_top_complete_nodes(df, top_n=args.top_nodes)
    df = df[df["nodeid"].isin(selected_nodes)]
    df["node_profile_id"] = encode_id(df["nodeid"])
    df = add_trend_features(df)
    df = add_burst_flag(df)
    df = df.rename(columns={
        "cpu_utilization": "cpu_scale",
        "memory_utilization": "mem_scale"
    })
    df.to_csv(args.out, index=False)
```

---

## 5. Alibaba MSResource 原始数据与处理

### 5.1 原始数据长什么样

官方 README 中 MS resource 表字段为：

```text
timestamp
msname
msinstanceid
nodeid
cpu_utilization
memory_utilization
```

示例：

```csv
timestamp,msname,msinstanceid,nodeid,cpu_utilization,memory_utilization
0,99f2e7b...,4d1cf6...,ecd8a8...,0.1299,0.6126
```

字段含义：

| 字段 | 含义 | 处理方式 |
|---|---|---|
| timestamp | 时间戳 | 转为 time_slot |
| msname | 微服务名称 | 映射为 service_profile_id |
| msinstanceid | 微服务实例 / container ID | 聚合时统计实例数量 |
| nodeid | 所在 BM 节点 | 可与 node 表 join |
| cpu_utilization | 实例 CPU 使用率 | 聚合 mean/max |
| memory_utilization | 实例 memory 使用率 | 聚合 mean/max |

官方说明中 MS resource 的记录间隔为 60s。

### 5.2 处理目标

处理成：

```text
data/processed/alibaba_ms_resource_profile.csv
```

输出字段：

```csv
time_slot,service_profile_id,node_profile_id,instance_count,ms_cpu_mean,ms_cpu_max,ms_mem_mean,ms_mem_max
```

### 5.3 处理逻辑

1. 读取 MSResource 原始文件；
2. 去除 `msname` 为空、`nodeid` 为空的记录；
3. 将 timestamp 转为分钟级 time_slot；
4. 只保留后续选中的 top service；
5. 按 `time_slot, msname, nodeid` 聚合；
6. 统计实例数量、CPU 均值、CPU 最大值、memory 均值、memory 最大值；
7. 将 msname 编码成 `service_profile_id`；
8. 将 nodeid 映射成 `node_profile_id`。

### 5.4 推荐数据量

建议第一篇选择：

```text
top_services = 50 或 100
node_profiles = 100
时间槽 = 720
输出约 50,000 到 200,000 行
```

具体行数取决于服务实例分布。

### 5.5 脚本设计

文件：

```text
src/preprocessing/prepare_alibaba_ms_resource.py
```

命令：

```bash
python src/preprocessing/prepare_alibaba_ms_resource.py \
  --raw_dir data/traces/alibaba/clusterdata/cluster-trace-microservices-v2021/MSResource \
  --selected_services data/processed/selected_services.json \
  --node_map data/processed/node_id_map.json \
  --out data/processed/alibaba_ms_resource_profile.csv \
  --slot_ms 60000
```

核心伪代码：

```python
def main(args):
    df = load_all_csv(args.raw_dir)
    df = df.dropna(subset=["timestamp", "msname", "nodeid"])
    df = df[~df["msname"].isin(["", "(?)", "NAN", "nan"])]
    df["time_slot"] = (df["timestamp"] // args.slot_ms).astype(int)
    selected = load_selected_services(args.selected_services)
    df = df[df["msname"].isin(selected)]
    df["service_profile_id"] = map_service_id(df["msname"])
    df["node_profile_id"] = map_node_id(df["nodeid"], args.node_map)
    out = df.groupby(["time_slot", "service_profile_id", "node_profile_id"]).agg(
        instance_count=("msinstanceid", "nunique"),
        ms_cpu_mean=("cpu_utilization", "mean"),
        ms_cpu_max=("cpu_utilization", "max"),
        ms_mem_mean=("memory_utilization", "mean"),
        ms_mem_max=("memory_utilization", "max"),
    ).reset_index()
    out.to_csv(args.out, index=False)
```

---

## 6. Alibaba MSRTQps 原始数据与处理

### 6.1 原始数据长什么样

官方 README 中 MS_MCR_RT 表字段为：

```text
timestamp
msname
msinstanceid
metrics
value
```

示例：

```csv
timestamp,msname,msinstanceid,metrics,value
6600000,1e5dd1...,0c14b7...,consumerRPC_RT,9.1414
```

官方说明中，metrics 包括：

```text
consumerRPC_MCR
providerRPC_MCR
HTTP_MCR
providerMQ_MCR
consumerMQ_MCR
consumerRPC_RT
providerRPC_RT
HTTP_RT
providerMQ_RT
consumerMQ_RT
```

其中：

- MCR 表示 call rate；
- RT 表示 response time，单位为 ms；
- RPC 的 RT 可能出现正负值，正负分别表示 UM / DM 侧的响应时间视角。

### 6.2 处理目标

处理成：

```text
data/processed/alibaba_service_qos_profile.csv
```

输出字段：

```csv
time_slot,service_profile_id,mcr_total,rt_mean_ms,rt_p95_ms,http_mcr,rpc_mcr,mq_mcr,http_rt_ms,rpc_rt_ms,mq_rt_ms
```

### 6.3 处理逻辑

1. 读取 MSRTQps 原始文件；
2. 去除无效 msname；
3. 转 time_slot；
4. 只保留 selected services；
5. 将 metrics pivot 成宽表；
6. MCR 类特征求和，得到 `mcr_total`；
7. RT 类特征取绝对值后求均值 / p95，得到 `rt_mean_ms`、`rt_p95_ms`；
8. 对极端值按 p99 裁剪；
9. 输出服务 QoS profile。

注意：

RPC RT 的正负值有具体语义。第一篇为了简化，可以使用 `abs(value)` 构造服务级响应时间强度；如果后续做期刊扩展，可以区分 UM_RT 与 DM_RT。

### 6.4 推荐数据量

建议第一篇：

```text
selected_services = 50 或 100
时间槽 = 720
输出约 36,000 到 72,000 行
```

即：

```text
50 services × 720 time slots = 36,000 rows
100 services × 720 time slots = 72,000 rows
```

### 6.5 脚本设计

文件：

```text
src/preprocessing/prepare_alibaba_ms_rtqps.py
```

命令：

```bash
python src/preprocessing/prepare_alibaba_ms_rtqps.py \
  --raw_dir data/traces/alibaba/clusterdata/cluster-trace-microservices-v2021/MSRTQps \
  --out data/processed/alibaba_service_qos_profile.csv \
  --slot_ms 60000 \
  --top_services 100
```

核心伪代码：

```python
def main(args):
    df = load_all_csv(args.raw_dir)
    df = clean_msname(df)
    df["time_slot"] = (df["timestamp"] // args.slot_ms).astype(int)

    # Select services by total MCR
    mcr_df = df[df["metrics"].str.endswith("_MCR")]
    service_rank = mcr_df.groupby("msname")["value"].sum().sort_values(ascending=False)
    selected = service_rank.head(args.top_services).index.tolist()
    save_json(selected, "data/processed/selected_services.json")

    df = df[df["msname"].isin(selected)]
    df["service_profile_id"] = map_service_id(df["msname"])

    # Split MCR / RT
    df_mcr = df[df["metrics"].str.endswith("_MCR")]
    df_rt = df[df["metrics"].str.endswith("_RT")].copy()
    df_rt["value"] = df_rt["value"].abs()
    df_rt["value"] = clip_by_quantile(df_rt["value"], q=0.99)

    mcr_wide = pivot_metrics(df_mcr)
    rt_wide = pivot_metrics(df_rt)
    out = merge_mcr_rt(mcr_wide, rt_wide)
    out["mcr_total"] = out[[c for c in out.columns if c.endswith("_MCR")]].sum(axis=1)
    out["rt_mean_ms"] = out[[c for c in out.columns if c.endswith("_RT")]].mean(axis=1)
    out["rt_p95_ms"] = compute_group_p95(df_rt)
    out.to_csv(args.out, index=False)
```

---

## 7. Alibaba MSCallGraph 原始数据与处理

### 7.1 原始数据长什么样

官方 README 中 MS_CallGraph 表字段包括：

```text
timestamp
traceid
rpcid
um
rpctype
interface
dm
rt
```

示例：

```csv
timestamp,traceid,rpcid,um,rpctype,interface,dm,rt
16397576,015101...,0.1.1.2.50,35114a...,rpc,af42b...,b65fd...,13
```

字段含义：

| 字段 | 含义 | 处理方式 |
|---|---|---|
| timestamp | 调用时间 | 转 time_slot |
| traceid | 一条调用图 ID | 可用于统计 call graph |
| rpcid | 调用边 ID | 可用于 deduplicate |
| um | upstream microservice | 作为调用源 |
| dm | downstream microservice | 作为调用目标 |
| rpctype | 调用类型 | one-hot / 统计 |
| interface | 接口 | 第一篇可不用 |
| rt | 调用响应时间 ms | 聚合 mean/p95 |

官方说明中 MSCallGraph 体量很大，且已经按 0.5% 采样。

### 7.2 处理目标

处理成两个文件：

```text
data/processed/alibaba_call_edges.csv
```

字段：

```csv
time_slot,src_service_id,dst_service_id,call_count,call_rt_mean,call_rt_p95,rpc_type_id
```

以及：

```text
data/processed/alibaba_service_dependency.csv
```

字段：

```csv
src_service_id,dst_service_id,total_call_count,avg_rt,p95_rt,dependency_weight
```

### 7.3 处理逻辑

1. 读取 MSCallGraph 原始文件；
2. 去除 `um` 或 `dm` 为空、`(?)`、`NAN` 的记录；
3. 只保留 selected services 相关调用；
4. timestamp 转 time_slot；
5. 对 `um, dm, time_slot` 聚合 call_count、rt_mean、rt_p95；
6. 构造 service dependency graph；
7. 输出边表。

### 7.4 推荐数据量

MSCallGraph 原始数据较大。第一篇建议：

```text
只处理 selected_services 相关记录
最多读取前 N 个文件或前 M 行
N = 10 到 20 个压缩文件
M = 5,000,000 到 20,000,000 行
```

最后输出规模建议：

```text
call_edges.csv: 50,000 到 500,000 行
service_dependency.csv: 500 到 5,000 条边
```

### 7.5 脚本设计

文件：

```text
src/preprocessing/prepare_alibaba_callgraph.py
```

命令：

```bash
python src/preprocessing/prepare_alibaba_callgraph.py \
  --raw_dir data/traces/alibaba/clusterdata/cluster-trace-microservices-v2021/MSCallGraph \
  --selected_services data/processed/selected_services.json \
  --out_edges data/processed/alibaba_call_edges.csv \
  --out_deps data/processed/alibaba_service_dependency.csv \
  --slot_ms 60000 \
  --max_files 20
```

核心伪代码：

```python
def main(args):
    selected = load_selected_services(args.selected_services)
    chunks = []
    for file in list_files(args.raw_dir)[:args.max_files]:
        df = read_csv(file)
        df = df.dropna(subset=["timestamp", "um", "dm"])
        df = df[~df["um"].isin(["", "(?)", "NAN", "nan"])]
        df = df[~df["dm"].isin(["", "(?)", "NAN", "nan"])]
        df = df[(df["um"].isin(selected)) | (df["dm"].isin(selected))]
        df["time_slot"] = (df["timestamp"] // args.slot_ms).astype(int)
        df["src_service_id"] = map_service_id(df["um"])
        df["dst_service_id"] = map_service_id(df["dm"])
        df["rt"] = df["rt"].abs()
        chunks.append(df[["time_slot", "src_service_id", "dst_service_id", "rpctype", "rt"]])

    df = pd.concat(chunks)
    edges = df.groupby(["time_slot", "src_service_id", "dst_service_id", "rpctype"]).agg(
        call_count=("rt", "count"),
        call_rt_mean=("rt", "mean"),
        call_rt_p95=("rt", lambda x: x.quantile(0.95))
    ).reset_index()

    deps = edges.groupby(["src_service_id", "dst_service_id"]).agg(
        total_call_count=("call_count", "sum"),
        avg_rt=("call_rt_mean", "mean"),
        p95_rt=("call_rt_p95", "mean")
    ).reset_index()
    deps["dependency_weight"] = normalize(deps["total_call_count"])

    edges.to_csv(args.out_edges, index=False)
    deps.to_csv(args.out_deps, index=False)
```

---

## 8. Alibaba 数据整合成 workload profile

前面四类数据处理完成后，需要合并成 EdgeSimPy 可以使用的输入 profile。

### 8.1 输入文件

```text
data/processed/alibaba_node_profile.csv
data/processed/alibaba_ms_resource_profile.csv
data/processed/alibaba_service_qos_profile.csv
data/processed/alibaba_call_edges.csv
data/processed/alibaba_service_dependency.csv
data/processed/selected_services.json
```

### 8.2 输出文件

```text
data/processed/workload_profiles/service_load_profiles.csv
data/processed/workload_profiles/node_load_profiles.csv
data/processed/workload_profiles/service_dependency_edges.csv
data/processed/workload_profiles/service_mapping.json
data/processed/workload_profiles/node_mapping.json
```

### 8.3 service_load_profiles.csv 字段

```csv
time_slot,service_profile_id,service_type,request_rate,base_response_time_ms,dependency_degree,rt_p95_ms,burst_flag
```

其中：

| 字段 | 来源 |
|---|---|
| time_slot | MSRTQps |
| service_profile_id | selected services |
| service_type | 根据 SLA 类型随机或规则分配 |
| request_rate | mcr_total |
| base_response_time_ms | rt_mean_ms |
| dependency_degree | service_dependency 中出入度 |
| rt_p95_ms | MSRTQps |
| burst_flag | request_rate 高于 p90 |

### 8.4 node_load_profiles.csv 字段

```csv
time_slot,node_profile_id,cpu_scale,mem_scale,cpu_trend,mem_trend,burst_flag
```

### 8.5 service_dependency_edges.csv 字段

```csv
src_service_id,dst_service_id,dependency_weight,avg_rt,p95_rt,total_call_count
```

### 8.6 脚本设计

文件：

```text
src/preprocessing/build_workload_profiles.py
```

命令：

```bash
python src/preprocessing/build_workload_profiles.py \
  --node_profile data/processed/alibaba_node_profile.csv \
  --ms_resource data/processed/alibaba_ms_resource_profile.csv \
  --service_qos data/processed/alibaba_service_qos_profile.csv \
  --call_edges data/processed/alibaba_call_edges.csv \
  --dependency data/processed/alibaba_service_dependency.csv \
  --out_dir data/processed/workload_profiles
```

核心逻辑：

```python
service_qos = read_csv(...)
resource = read_csv(...)
deps = read_csv(...)

service_profile = service_qos.merge(resource_agg, on=["time_slot", "service_profile_id"], how="left")
service_profile = add_dependency_degree(service_profile, deps)
service_profile = add_service_type(service_profile)
service_profile = add_burst_flag(service_profile, col="request_rate")
service_profile.to_csv("service_load_profiles.csv")

node_profile = read_csv(node_profile)
node_profile = fill_missing_time_slots(node_profile)
node_profile.to_csv("node_load_profiles.csv")
```

---

## 9. Google 数据处理规格（可选）

### 9.1 Google 2019

Google 2019 数据官方说明为 8 个 Borg cells、2019 年 5 月、压缩体量约 2.4TiB，并通过 BigQuery 提供。第一篇不建议完整使用。

如果使用，只抽样生成：

```text
data/processed/google_node_load.csv
```

字段：

```csv
time_slot,node_profile_id,cpu_scale,mem_scale,arrival_scale,usage_p95
```

建议 BigQuery 处理思路：

```sql
SELECT
  FLOOR(start_time / 60000000) AS time_slot,
  machine_id,
  AVG(cpu_usage) AS cpu_scale,
  AVG(memory_usage) AS mem_scale,
  COUNT(*) AS task_arrival
FROM task_usage
WHERE sampled_condition
GROUP BY time_slot, machine_id
```

注意：字段名需要根据 Google BigQuery 实际表结构调整。

### 9.2 Google 2011

Google 2011 可本地下载，压缩约 41GB，包含 29 天、约 12.5k machines 的 Borg trace。第一篇如果使用，只建议处理 `task_usage` 和 `task_events`。

处理输出：

```text
data/processed/google2011_node_profile.csv
```

字段：

```csv
time_slot,node_profile_id,cpu_scale,mem_scale,task_arrival,fail_count
```

### 9.3 Google 在第一篇中的建议角色

Google 数据不作为主数据源，只做可选补充：

1. 用于额外节点负载 pattern；
2. 用于鲁棒性实验；
3. 用于证明模型不依赖单一 trace。

如果时间紧，完全可以第一篇不使用 Google。

---

## 10. 将 workload profile 注入 EdgeSimPy

### 10.1 输入

```text
data/processed/workload_profiles/service_load_profiles.csv
data/processed/workload_profiles/node_load_profiles.csv
data/processed/workload_profiles/service_dependency_edges.csv
```

### 10.2 输出

EdgeSimPy 仿真日志：

```text
data/raw/{scenario_name}/node_log.csv
data/raw/{scenario_name}/link_log.csv
data/raw/{scenario_name}/service_log.csv
data/raw/{scenario_name}/path_log.csv
data/raw/{scenario_name}/sla_log.csv
```

### 10.3 注入逻辑

#### 服务请求率

```python
request_rate[t, service] = base_rate * (1 + alpha * normalized_mcr[t, service])
```

#### 服务基础响应时间

```python
service_base_rt[t, service] = base_response_time_ms[t, service]
```

#### 节点背景负载

```python
edge_node_cpu_load[t, node] = beta * cpu_scale[t, node_profile]
edge_node_mem_load[t, node] = beta * mem_scale[t, node_profile]
```

#### 服务依赖复杂度

```python
dependency_degree[service] = out_degree + in_degree
service_complexity = 1 + gamma * dependency_degree
```

#### 网络链路状态

链路 delay/loss/bandwidth 不是 Alibaba trace 直接提供的，需要仿真生成：

```python
link_delay = base_delay * (1 + congestion_factor)
link_loss = base_loss + random_noise + congestion_loss
bandwidth_util = traffic_on_link / link_capacity
```

### 10.4 脚本设计

文件：

```text
src/simulation/inject_trace.py
src/simulation/run_edgesimpy.py
src/simulation/export_logs.py
```

命令：

```bash
python src/simulation/run_edgesimpy.py \
  --service_profile data/processed/workload_profiles/service_load_profiles.csv \
  --node_profile data/processed/workload_profiles/node_load_profiles.csv \
  --dependency data/processed/workload_profiles/service_dependency_edges.csv \
  --num_edge_nodes 20 \
  --num_access_nodes 10 \
  --num_users 300 \
  --num_services 50 \
  --num_steps 720 \
  --topology small_world \
  --out_dir data/raw/smallworld_e20_s50
```

---

## 11. EdgeSimPy 输出日志规格

### 11.1 node_log.csv

字段：

```csv
time,node_id,node_type,cpu_util,mem_util,queue_len,available_cpu,available_mem,active_services,is_failed
```

行数估计：

```text
num_steps × num_nodes
```

例如：

```text
720 time slots × 31 nodes ≈ 22,320 rows
```

其中 31 nodes 可以是：

```text
1 cloud + 20 edge + 10 access
```

### 11.2 link_log.csv

字段：

```csv
time,src,dst,delay,bandwidth,loss,jitter,queue_delay,bandwidth_util,is_congested,is_failed
```

行数估计：

```text
num_steps × num_links
```

例如：

```text
720 × 80 links ≈ 57,600 rows
```

### 11.3 service_log.csv

字段：

```csv
time,service_id,service_type,user_group,current_edge,request_rate,response_time,base_response_time,is_failed
```

行数估计：

```text
num_steps × num_services
```

例如：

```text
720 × 50 services = 36,000 rows
```

### 11.4 path_log.csv

字段：

```csv
time,service_id,path_nodes,path_links,path_delay,path_loss,path_bandwidth_min,bottleneck_node,bottleneck_link,candidate_edges
```

`path_nodes`、`path_links` 用 JSON string 保存：

```text
"[0, 12, 5, 30]"
```

### 11.5 sla_log.csv

字段：

```csv
service_id,service_type,max_delay,max_loss,min_bandwidth,reliability_req,cost_weight,priority
```

SLA 可以按服务类型生成：

| service_type | max_delay | max_loss | min_bandwidth | reliability_req |
|---|---:|---:|---:|---:|
| latency_sensitive | 50 ms | 0.03 | high | 0.99 |
| reliability_sensitive | 100 ms | 0.01 | medium | 0.999 |
| cost_sensitive | 150 ms | 0.05 | low | 0.95 |

具体数值应根据仿真单位调整，不必声称来自真实 SLA。

---

## 12. 从仿真日志构造 service-path subgraph

### 12.1 输入文件

```text
node_log.csv
link_log.csv
service_log.csv
path_log.csv
sla_log.csv
```

### 12.2 输出文件

```text
data/processed/path_graphs_w12.pkl
```

### 12.3 时间窗口

默认：

```text
L = 12
H = 5
```

若 time_slot 为 1 分钟，则：

```text
L = 过去 12 分钟
H = 未来 5 分钟
```

如果用 30s time_slot，则：

```text
L = 过去 6 分钟
H = 未来 2.5 分钟
```

### 12.4 子图节点选择

每个 service-path subgraph 包含：

1. user access node；
2. current edge node；
3. cloud node；
4. path intermediate nodes；
5. top-k candidate edge nodes；
6. 共享 bottleneck link 的邻近节点。

默认固定：

```text
max_nodes = 8
max_links = 12
```

若节点数不足，padding；若超过，优先保留：

```text
access node > current edge > cloud > path nodes > candidate edges > neighbor nodes
```

### 12.5 node_features

固定顺序：

```text
0 cpu_util
1 mem_util
2 queue_len_norm
3 available_cpu_norm
4 available_mem_norm
5 active_services_norm
6 node_type_id_norm
7 is_current_edge
```

shape：

```text
[L, max_nodes, 8]
```

### 12.6 link_features

固定顺序：

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

shape：

```text
[L, max_links, 8]
```

### 12.7 service_features

固定顺序：

```text
0 request_rate_norm
1 response_time_norm
2 base_response_time_norm
3 service_type_id_norm
4 dependency_degree_norm
```

shape：

```text
[L, 5]
```

### 12.8 sla_features

固定顺序：

```text
0 max_delay_norm
1 max_loss_norm
2 min_bandwidth_norm
3 reliability_req_norm
4 cost_weight_norm
```

shape：

```text
[5]
```

### 12.9 adj_matrix

shape：

```text
[max_nodes, max_nodes]
```

构造：

```text
adj[i,j] = 1 if node i and node j are connected in service-path subgraph else 0
```

可选加 self-loop：

```text
adj[i,i] = 1
```

### 12.10 mask

```text
node_mask: [max_nodes]
link_mask: [max_links]
```

有效位置为 1，padding 为 0。

---

## 13. 风险标签生成

### 13.1 输入

```text
path_graphs_w12.pkl
service_log.csv
path_log.csv
sla_log.csv
```

### 13.2 输出

```text
data/sparta_dataset/sparta_w12_h5.pkl
```

### 13.3 标签定义

对于样本 `(service_id=s, time=t)`，查看未来窗口：

```text
[t+1, t+H]
```

#### violated

若未来窗口内任一时间满足：

```text
response_time >= max_delay
or path_loss >= max_loss
or is_failed == 1
```

则：

```text
risk_label = 2
```

#### risky

若不 violated，但满足：

```text
response_time >= 0.75 * max_delay
or path_loss >= 0.75 * max_loss
or max_cpu_on_path >= 0.85
or max_queue_on_path 连续上升
or max_bandwidth_util_on_path >= 0.85
```

则：

```text
risk_label = 1
```

#### normal

否则：

```text
risk_label = 0
```

### 13.4 类别比例控制

真实生成时，normal 可能太多。建议目标比例：

```text
normal: 50% 到 70%
risky: 15% 到 30%
violated: 5% 到 20%
```

如果 violated 太少，可以通过仿真注入更多：

1. link congestion；
2. node overload；
3. burst workload；
4. node failure；
5. path bottleneck。

---

## 14. 风险归因标签生成

### 14.1 risk_node

对未来窗口或当前窗口计算路径节点风险分数：

```text
node_score = 0.4 * cpu_util + 0.3 * queue_norm + 0.3 * (1 - available_cpu_norm)
```

取最高分节点作为 `risk_node`。

如果仿真中注入了明确故障节点，则优先使用注入节点作为 ground truth。

### 14.2 risk_link

链路风险分数：

```text
link_score = 0.4 * delay_norm + 0.3 * loss_norm + 0.3 * bandwidth_util
```

取最高分链路作为 `risk_link`。

如果仿真中注入了明确拥塞链路，则优先使用注入链路作为 ground truth。

### 14.3 risk_metric

定义：

```text
0 delay
1 loss
2 cpu
3 queue
4 bandwidth
```

规则：

```text
metric_score_delay = response_time / max_delay
metric_score_loss = path_loss / max_loss
metric_score_cpu = max_cpu_on_path / 0.85
metric_score_queue = max_queue_norm
metric_score_bandwidth = max_bandwidth_util / 0.85
```

取最大者。

### 14.4 normal 样本归因标签

normal 样本没有真实风险来源。建议：

```text
risk_node = -1
risk_link = -1
risk_metric = -1
```

训练时：

```text
normal 样本不参与 node/link/metric attribution loss
```

---

## 15. 最终 SPARTA 数据集格式

### 15.1 文件

```text
data/sparta_dataset/train.pkl
data/sparta_dataset/val.pkl
data/sparta_dataset/test.pkl
```

### 15.2 每个 pkl 内容

建议保存为 list of dict 或 PyTorch-friendly dict of arrays。

推荐 dict of arrays：

```python
{
    "node_x": np.ndarray,        # [N, L, max_nodes, 8]
    "link_x": np.ndarray,        # [N, L, max_links, 8]
    "service_x": np.ndarray,     # [N, L, 5]
    "sla_x": np.ndarray,         # [N, 5]
    "adj": np.ndarray,           # [N, max_nodes, max_nodes]
    "node_mask": np.ndarray,     # [N, max_nodes]
    "link_mask": np.ndarray,     # [N, max_links]
    "risk_label": np.ndarray,    # [N]
    "risk_node": np.ndarray,     # [N]
    "risk_link": np.ndarray,     # [N]
    "risk_metric": np.ndarray,   # [N]
    "meta": list                 # service_id, time, scenario_name 等
}
```

### 15.3 推荐样本量

ICASSP 第一篇建议：

```text
训练样本：80,000 到 200,000
验证样本：20,000 到 50,000
测试样本：20,000 到 50,000
```

最小可行：

```text
总样本 50,000 左右
```

较稳版本：

```text
总样本 150,000 到 300,000
```

样本量计算方式：

```text
num_samples ≈ (num_steps - L - H) × num_services × num_scenarios
```

例如：

```text
num_steps = 720
L = 12
H = 5
num_services = 50
num_scenarios = 6
num_samples ≈ 703 × 50 × 6 = 210,900
```

---

## 16. 数据划分

### 16.1 默认时间划分

每个 scenario 内部按时间划分：

```text
train: 前 60%
val: 中间 20%
test: 后 20%
```

不要随机打乱时间片，否则会时间泄漏。

### 16.2 可选泛化划分

跨拓扑测试：

```text
train: tree + random
val: random subset
test: small_world
```

跨负载测试：

```text
train: normal + moderate burst
test: severe burst
```

跨故障测试：

```text
train: node overload + link congestion
test: node failure
```

第一篇至少做时间划分；如果结果足够，可以加跨拓扑泛化。

---

## 17. 归一化处理

### 17.1 原则

只能用训练集统计归一化参数。

错误做法：

```text
用 train + val + test 统计 min/max 或 mean/std
```

正确做法：

```text
用 train 统计 scaler
val/test 使用 train scaler
```

### 17.2 连续特征

建议 min-max：

```python
x_norm = (x - train_min) / (train_max - train_min + 1e-8)
```

适用字段：

```text
delay
bandwidth
loss
jitter
queue_delay
queue_len
request_rate
response_time
base_response_time
max_delay
max_loss
min_bandwidth
```

### 17.3 已经在 [0,1] 的字段

```text
cpu_util
mem_util
bandwidth_util
available_cpu
available_mem
reliability_req
cost_weight
```

只需要裁剪：

```python
x = np.clip(x, 0, 1)
```

### 17.4 类别字段

第一篇为了简单，建议归一化成连续值或 one-hot：

```text
service_type_id_norm = service_type_id / num_service_types
node_type_id_norm = node_type_id / num_node_types
```

后续代码更规范时可以改成 embedding。

### 17.5 scaler 保存

保存：

```text
data/sparta_dataset/scaler.json
```

内容：

```json
{
  "delay": {"min": 1.0, "max": 120.0},
  "request_rate": {"min": 0.0, "max": 500.0},
  "response_time": {"min": 0.0, "max": 300.0}
}
```

---

## 18. 总控脚本顺序

建议提供一个总控脚本：

```text
scripts/02_prepare_datasets.sh
```

内容：

```bash
#!/usr/bin/env bash
set -e

python src/preprocessing/prepare_alibaba_node.py \
  --raw_dir data/traces/alibaba/clusterdata/cluster-trace-microservices-v2021/node \
  --out data/processed/alibaba_node_profile.csv \
  --slot_ms 60000 \
  --top_nodes 100

python src/preprocessing/prepare_alibaba_ms_rtqps.py \
  --raw_dir data/traces/alibaba/clusterdata/cluster-trace-microservices-v2021/MSRTQps \
  --out data/processed/alibaba_service_qos_profile.csv \
  --slot_ms 60000 \
  --top_services 100

python src/preprocessing/prepare_alibaba_ms_resource.py \
  --raw_dir data/traces/alibaba/clusterdata/cluster-trace-microservices-v2021/MSResource \
  --selected_services data/processed/selected_services.json \
  --node_map data/processed/node_id_map.json \
  --out data/processed/alibaba_ms_resource_profile.csv \
  --slot_ms 60000

python src/preprocessing/prepare_alibaba_callgraph.py \
  --raw_dir data/traces/alibaba/clusterdata/cluster-trace-microservices-v2021/MSCallGraph \
  --selected_services data/processed/selected_services.json \
  --out_edges data/processed/alibaba_call_edges.csv \
  --out_deps data/processed/alibaba_service_dependency.csv \
  --slot_ms 60000 \
  --max_files 20

python src/preprocessing/build_workload_profiles.py \
  --node_profile data/processed/alibaba_node_profile.csv \
  --ms_resource data/processed/alibaba_ms_resource_profile.csv \
  --service_qos data/processed/alibaba_service_qos_profile.csv \
  --call_edges data/processed/alibaba_call_edges.csv \
  --dependency data/processed/alibaba_service_dependency.csv \
  --out_dir data/processed/workload_profiles
```

仿真与样本构造：

```bash
python src/simulation/run_edgesimpy.py \
  --service_profile data/processed/workload_profiles/service_load_profiles.csv \
  --node_profile data/processed/workload_profiles/node_load_profiles.csv \
  --dependency data/processed/workload_profiles/service_dependency_edges.csv \
  --num_edge_nodes 20 \
  --num_access_nodes 10 \
  --num_users 300 \
  --num_services 50 \
  --num_steps 720 \
  --topology small_world \
  --out_dir data/raw/smallworld_e20_s50

python src/preprocessing/build_path_graph.py \
  --input_dir data/raw/smallworld_e20_s50 \
  --window 12 \
  --max_nodes 8 \
  --max_links 12 \
  --output data/processed/path_graphs_w12.pkl

python src/preprocessing/generate_labels.py \
  --input data/processed/path_graphs_w12.pkl \
  --horizon 5 \
  --output data/sparta_dataset/sparta_w12_h5.pkl

python src/preprocessing/generate_attribution.py \
  --input data/sparta_dataset/sparta_w12_h5.pkl \
  --output data/sparta_dataset/sparta_w12_h5_attr.pkl

python src/preprocessing/split_dataset.py \
  --input data/sparta_dataset/sparta_w12_h5_attr.pkl \
  --out_dir data/sparta_dataset \
  --split time \
  --train_ratio 0.6 \
  --val_ratio 0.2
```

---

## 19. 每个处理阶段的输入输出总表

| 阶段 | 输入 | 输出 | 脚本 |
|---|---|---|---|
| Alibaba node 处理 | node 原始表 | alibaba_node_profile.csv | prepare_alibaba_node.py |
| MS resource 处理 | MSResource 原始表 | alibaba_ms_resource_profile.csv | prepare_alibaba_ms_resource.py |
| MS RT/QPS 处理 | MSRTQps 原始表 | alibaba_service_qos_profile.csv | prepare_alibaba_ms_rtqps.py |
| MS call graph 处理 | MSCallGraph 原始表 | call_edges / dependency | prepare_alibaba_callgraph.py |
| profile 构造 | 以上 4 类 processed CSV | workload_profiles | build_workload_profiles.py |
| EdgeSimPy 仿真 | workload_profiles | node/link/service/path/sla logs | run_edgesimpy.py |
| 子图构造 | 仿真 logs | path_graphs_w12.pkl | build_path_graph.py |
| 风险标签 | path_graphs + future logs | risk labels | generate_labels.py |
| 归因标签 | risk samples + logs | attribution labels | generate_attribution.py |
| 数据划分 | full pkl | train/val/test.pkl | split_dataset.py |
| 归一化 | train/val/test | normalized pkl + scaler | normalize.py |

---

## 20. 给 AI/Codex 的实现要求

如果让 AI 写预处理代码，必须明确以下要求：

1. 所有脚本都支持 argparse；
2. 所有输出文件路径由参数指定；
3. 每个脚本运行后打印输入行数、输出行数、缺失比例；
4. 大文件读取必须支持 chunk；
5. 所有 ID 映射必须保存为 JSON；
6. 时间单位必须统一为 `time_slot`；
7. 所有连续特征必须有归一化记录；
8. 所有随机采样必须设置 seed；
9. 所有 pkl 必须能被 `SPARTADataset` 直接读取；
10. 每个阶段都要有最小单元测试。

---

## 21. 最小调试数据集

为了先让代码跑通，建议构造 debug 数据集。

配置：

```text
selected_services = 5
node_profiles = 10
num_steps = 120
num_edge_nodes = 5
num_access_nodes = 3
num_users = 50
max_nodes = 8
max_links = 12
```

预计样本数：

```text
(120 - 12 - 5) × 5 ≈ 515
```

生成：

```text
data/sparta_dataset/debug_train.pkl
data/sparta_dataset/debug_val.pkl
data/sparta_dataset/debug_test.pkl
```

用途：

1. 测试 Dataset；
2. 测试 DataLoader；
3. 测试模型 forward；
4. 测试 loss backward；
5. 测试 evaluate。

---

## 22. 正式实验推荐规模

ICASSP 第一篇推荐规模：

```text
selected_services = 50
node_profiles = 100
num_steps = 720
num_scenarios = 6
max_nodes = 8
max_links = 12
```

场景：

```text
tree_normal
tree_burst
random_normal
random_congestion
smallworld_normal
smallworld_failure
```

预计样本：

```text
(720 - 12 - 5) × 50 × 6 ≈ 210,900
```

划分：

```text
train: 126,000 左右
val: 42,000 左右
test: 42,000 左右
```

这对 ICASSP 已经足够。

---

## 23. 数据处理部分在论文中怎么写

论文中不需要写所有工程细节，但需要写清楚：

```text
We build a trace-driven cloud-edge simulation dataset. Microservice call rates, response times, and node utilization patterns are extracted from public Alibaba traces and injected into EdgeSimPy to generate cloud-edge service dynamics. For each service, we construct a path-centric subgraph and collect node, link, service, and SLA features over a historical window. Risk labels are generated according to future SLA states, and attribution labels are derived from injected or threshold-dominant risk sources.
```

中文含义：

```text
我们不是凭空随机生成数据，而是先从 Alibaba trace 提取真实负载和服务状态趋势，再注入云边仿真环境，最后生成服务路径级风险检测数据。
```

---

## 24. 最终建议

第一篇 ICASSP 数据处理不要贪多。最推荐路线：

```text
Alibaba microservices trace 2021
        ↓
node + MSResource + MSRTQps + MSCallGraph
        ↓
service_load_profiles + node_load_profiles + dependency_edges
        ↓
EdgeSimPy simulation
        ↓
node/link/service/path/sla logs
        ↓
service-path subgraph samples
        ↓
risk labels + attribution labels
        ↓
SPARTA train/val/test
```

Google 数据作为可选项，不要第一阶段强行加入。否则数据工程量会失控。

如果后续要扩展到 Computer Networks 期刊版，可以再加入：

1. Alibaba v2018 machine_usage；
2. Google 2011 或 2019 workload；
3. YAFS 动态拓扑；
4. 多 trace 跨域泛化实验。

---

## 25. 参考链接

1. Alibaba ClusterData GitHub  
   https://github.com/alibaba/clusterdata

2. Alibaba Microservices Trace 2021 README  
   https://github.com/alibaba/clusterdata/blob/master/cluster-trace-microservices-v2021/README.md

3. Alibaba Cluster Trace 2018 README  
   https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md

4. Google ClusterData GitHub  
   https://github.com/google/cluster-data

5. Google ClusterData 2019 README  
   https://github.com/google/cluster-data/blob/master/ClusterData2019.md

6. Google ClusterData 2011 README  
   https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md

