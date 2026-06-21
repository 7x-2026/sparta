# SPARTA Project Status README

Last updated: 2026-06-21

## 当前阶段

SPARTA 项目已经从 Windows Debug MVP 和 synthetic full 阶段，推进到最终实验结果整理阶段。

当前正式主线不是早期 synthetic full，也不是 EdgeSimPy adapter fallback，而是：

- 数据版本：`cpu_precursor_v1`
- 仿真来源：真实 EdgeSimPy strict
- 必须满足：`effective_simulator_backend=edgesimpy_real`
- Data seeds：`42 / 2025 / 3407`
- Baselines：`LSTM-CP`、`Transformer-CP`
- Main model：`SPARTA-CP`
- Final attribution version：`SPARTA-CP + Cal.`
- Calibration：对 SPARTA metric attribution logits 做 `macro_f1` calibration

正式三个 run 目录为：

```text
run/20260617_002_edgesimpy_real_strict_seed42_cpu_precursor_v1
run/20260617_001_edgesimpy_real_strict_seed2025_cpu_precursor_v1
run/20260617_001_edgesimpy_real_strict_seed3407_cpu_precursor_v1
```

这些 run 已完成：

- raw logs 生成
- dataset 构建
- audit
- LSTM / Transformer / SPARTA 训练与评估
- SPARTA attribution diagnosis
- macro_f1 metric logit calibration
- final 3-seed summary
- final model comparison summary

后续除非明确要求，不应重新仿真、重新训练或覆盖已有结果。

## 正式结果目录

SPARTA-CP 三 seed 汇总：

```text
run_groups/cpu_precursor_v1_final_3seed/
  summary.csv
  mean_std.csv
  final_report.md
```

最终模型对比汇总：

```text
run_groups/cpu_precursor_v1_final_model_comparison_3seed/
  model_comparison_summary.csv
  model_comparison_mean_std.csv
  model_comparison_report.md
```

最终模型对比包含：

1. `LSTM-CP`
2. `Transformer-CP`
3. `SPARTA-CP`
4. `SPARTA-CP + Cal.`

## Deprecated Exploration

以下版本是探索记录，不作为最终主表结果：

```text
metricfix
metricheadv2
metricheadv3
metricdistill
pressure01
pressure02-only
acc_plus_balanced calibration
balanced_acc calibration
synthetic_full-only
edgesimpy_adapter_fallback
```

这些内容保留用于实验可追溯，但不要误当成最终方案。

## 项目目录说明

### `aicode/`

早期设计文档、MVP spec、顶层设计参考。

主要用于理解需求来源，不是当前正式运行入口。

### `configs/`

所有实验配置文件。

当前正式配置重点是：

```text
sparta_edgesimpy_real_strict_seed42_cpu_precursor_v1.yaml
sparta_edgesimpy_real_strict_seed2025_cpu_precursor_v1.yaml
sparta_edgesimpy_real_strict_seed3407_cpu_precursor_v1.yaml
```

其他如 `metricfix`、`metricheadv2`、`metricdistill` 等是探索配置。

### `src/`

核心代码目录。

主要入口：

```text
src/run_synthetic_full_pipeline.py
src/train.py
src/evaluate.py
```

`run_synthetic_full_pipeline.py` 是统一 pipeline 入口。它现在不仅支持 synthetic full，也支持 EdgeSimPy strict / stub / adapter fallback，以及 `generate_only`、`audit_only`、`train_only`、`eval_only`、`build_dataset_only` 等模式。

`train.py` 和 `evaluate.py` 支持 `--resume_dataset_run` 和 `--variant`，因此可以在已有 run 的 dataset 上训练或评估新 variant，避免覆盖原始结果。

### `src/models/`

模型实现：

```text
common.py
lstm.py
transformer.py
sparta.py
```

当前最终主线没有继续修改模型结构。

### `src/datasets/`

Dataset 实现：

```text
sparta_dataset.py
```

当前不应随意修改 Dataset 格式。

### `src/losses/`

多任务 loss 与 focal loss：

```text
multitask_loss.py
focal_loss.py
```

这里保留了 metric attribution 加权、soft supervision 等探索能力。最终主线采用 `cpu_precursor_v1 + 原始 SPARTA + macro_f1 calibration`。

### `src/preprocessing/`

从 raw logs 到 model dataset 的构建：

```text
build_path_graph.py
generate_labels.py
generate_attribution.py
split_dataset.py
check_dataset_stats.py
```

注意：`normalize.py` 在 MVP 和当前主线中一般不调用。

### `src/simulation/`

数据生成与 EdgeSimPy 接入：

```text
edgesimpy_adapter.py
risk_injection.py
synthetic_full_generator.py
synthetic_scenarios.py
export_logs.py
```

当前最终数据版本的关键是 `cpu_precursor_v1` 风险注入。

### `src/analysis/`

审计、诊断、校准和汇总脚本。

最终主线重点文件包括：

```text
audit_synthetic_full.py
audit_cpu_precursor.py
diagnose_attribution.py
calibrate_metric_logits.py
summarize_cpu_precursor_final_3seed.py
summarize_final_model_comparison_3seed.py
```

其中：

- `calibrate_metric_logits.py` 做 SPARTA metric attribution 的后处理校准；
- `summarize_cpu_precursor_final_3seed.py` 汇总 SPARTA-CP final 3 seed；
- `summarize_final_model_comparison_3seed.py` 汇总 LSTM / Transformer / SPARTA / SPARTA+Cal 最终对比。

### `scripts/final_cpu_precursor_v1/`

最终实验 runbook 脚本目录。

推荐按编号理解流程：

```text
00_generate_data_3seed.ps1
01_audit_3seed.ps1
02_train_eval_lstm_3seed.ps1
03_train_eval_transformer_3seed.ps1
04_train_eval_sparta_3seed.ps1
05_calibrate_sparta_3seed.ps1
06_summarize_final_3seed.ps1
README_FINAL_RUNBOOK.md
```

这些脚本是最终主线的标准执行顺序。

### `run/`

每次实验的独立 run 目录。

典型结构：

```text
raw_logs/
dataset/
audit/
checkpoints/
results/
analysis/
logs/
run_manifest.json
```

这些是生成产物，通常不提交 git。

### `run_groups/`

跨 seed 或跨模型汇总结果目录。

当前最终结果主要在：

```text
run_groups/cpu_precursor_v1_final_3seed/
run_groups/cpu_precursor_v1_final_model_comparison_3seed/
```

### `docs/`

项目说明和实验状态文档。

当前包括：

```text
CODE_AUDIT_BEFORE_V0_4.md
EXPERIMENT_STATUS.md
PROJECT_STATUS_README.md
```

### `tests/`

pytest 测试目录，覆盖 pipeline、audit、metric evidence、calibration、final summarizer 等功能。

## 给后续 GPT 或合作者的注意事项

1. 不要把 `edgesimpy_adapter_fallback` 当成真实 EdgeSimPy 结果。
2. 最终主线必须确认 `effective_simulator_backend=edgesimpy_real`。
3. 不要默认重新运行仿真或训练，除非用户明确要求。
4. 不要覆盖已有 `run/` 和 `run_groups/` 结果。
5. 如果只是写论文、整理表格或分析结果，优先读取：

```text
run_groups/cpu_precursor_v1_final_model_comparison_3seed/
```

6. 如果要复现实验，按：

```text
scripts/final_cpu_precursor_v1/README_FINAL_RUNBOOK.md
```

以及对应编号脚本执行。

