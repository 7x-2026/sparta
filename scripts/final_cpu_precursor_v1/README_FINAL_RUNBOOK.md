# CPU Precursor v1 Final Runbook

## 最终主线

本项目最终实验主线固定为：

1. `cpu_precursor_v1` 数据生成与输入窗口 precursor 特征；
2. 三个 data seed：`42`、`2025`、`3407`；
3. 三个模型：`LSTM`、`Transformer`、`SPARTA`；
4. SPARTA 额外执行 `macro_f1` metric logit calibration；
5. 最终汇总目录：`run_groups/cpu_precursor_v1_final_3seed/`。

正式表格优先使用：

- 原始风险检测：`results/sparta_cpu_precursor_v1_test_results.csv` 中的 risk detection 指标；
- 原始 attribution：同一结果文件中的 node/link/metric attribution 指标；
- 校准后 metric attribution：`analysis/cpu_precursor_v1_metric_calibration_macro_f1.json`；
- 最终 3-seed 汇总：`run_groups/cpu_precursor_v1_final_3seed/summary.csv`、`mean_std.csv`、`final_report.md`。

## 正式版本

- FINAL：`cpu_precursor_v1 + SPARTA + macro_f1 metric logit calibration`
- BASELINES：`LSTM-CP`、`Transformer-CP`
- DATA：真实 EdgeSimPy strict backend 通过 smoke verification 后生成 raw logs，并在 adapter 层注入 `cpu_precursor_v1`
- SUMMARY：`src/analysis/summarize_cpu_precursor_final_3seed.py`

## Deprecated Exploration

以下版本保留用于实验记录和诊断，不作为最终主表结果：

- `metricfix`
- `metricheadv2`
- `metricdistill`
- `pressure01`
- `pressure02-only`
- `acc_plus_balanced` calibration
- `balanced_acc` calibration

## Conda 环境分工

- `sparta_edgesimpy`：只用于生成 raw logs。该环境需要能 import 本地/真实 EdgeSimPy。
- `sparta`：用于 audit、train、eval、calibration、summarize。该环境应包含 torch、sklearn、pandas/numpy 等训练与分析依赖。

如果在 `sparta_edgesimpy` 里执行 `--generate_only` 时，raw logs 和 dataset 已生成，但 audit 因 torch 缺失失败，可以切换到 `sparta` 环境继续执行 `01_audit_3seed.ps1`。

## 当前正式 Run 变量

```powershell
$RUN42   = 'run\20260617_002_edgesimpy_real_strict_seed42_cpu_precursor_v1'
$RUN2025 = 'run\20260617_001_edgesimpy_real_strict_seed2025_cpu_precursor_v1'
$RUN3407 = 'run\20260617_001_edgesimpy_real_strict_seed3407_cpu_precursor_v1'
```

如果重新生成数据，先在对应脚本顶部更新这三个变量，再继续后续步骤。

## 最简执行顺序

在 `sparta_edgesimpy` 环境：

```powershell
.\scripts\final_cpu_precursor_v1\00_generate_data_3seed.ps1
```

切换到 `sparta` 环境：

```powershell
.\scripts\final_cpu_precursor_v1\01_audit_3seed.ps1
.\scripts\final_cpu_precursor_v1\02_train_eval_lstm_3seed.ps1
.\scripts\final_cpu_precursor_v1\03_train_eval_transformer_3seed.ps1
.\scripts\final_cpu_precursor_v1\04_train_eval_sparta_3seed.ps1
.\scripts\final_cpu_precursor_v1\05_calibrate_sparta_3seed.ps1
.\scripts\final_cpu_precursor_v1\06_summarize_final_3seed.ps1
```

## 完整运行说明

1. 生成数据：运行 `00_generate_data_3seed.ps1`。这一步只在 `sparta_edgesimpy` 环境执行。
2. 审计数据：运行 `01_audit_3seed.ps1`，检查 synthetic full audit、CPU precursor audit、pressure02 evidence baseline 和 label rule replay。
3. 训练/评估 baseline：运行 `02_train_eval_lstm_3seed.ps1` 和 `03_train_eval_transformer_3seed.ps1`。
4. 训练/评估 SPARTA：运行 `04_train_eval_sparta_3seed.ps1`，同时生成 attribution diagnosis。
5. 校准 SPARTA：运行 `05_calibrate_sparta_3seed.ps1`，只生成 macro_f1 calibration 结果，不改 checkpoint。
6. 汇总最终结果：运行 `06_summarize_final_3seed.ps1`。

所有脚本均不会删除旧代码或移动已有 run 目录。训练脚本会按 variant 输出独立文件，例如 `sparta_cpu_precursor_v1_best.pth`、`sparta_cpu_precursor_v1_test_results.csv`。
