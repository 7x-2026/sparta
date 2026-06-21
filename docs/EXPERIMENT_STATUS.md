# Experiment Status

## FINAL

当前正式实验主线：

- 数据：`cpu_precursor_v1`
- Seeds：`42 / 2025 / 3407`
- 主模型：`SPARTA`
- 最终 metric attribution 后处理：`macro_f1 metric logit calibration`
- 汇总目录：`run_groups/cpu_precursor_v1_final_3seed/`

正式汇总文件：

- `run_groups/cpu_precursor_v1_final_3seed/summary.csv`
- `run_groups/cpu_precursor_v1_final_3seed/mean_std.csv`
- `run_groups/cpu_precursor_v1_final_3seed/final_report.md`

## BASELINES

最终 baseline：

- `LSTM-CP`
- `Transformer-CP`

Baseline 使用同一组 `cpu_precursor_v1` 数据和同一组 3 data seed，与 SPARTA 共享 run dataset。

## DEPRECATED

以下实验保留用于探索、诊断和审计记录，不作为最终主表结果：

- `metricfix`
- `metricheadv2`
- `metricdistill`
- `pressure01`
- `pressure02-only`
- `acc_plus_balanced calibration`
- `balanced_acc calibration`

这些版本可能用于解释为什么选择 `cpu_precursor_v1 + macro_f1 calibration`，但论文主表和最终对比不应直接引用它们作为最终方法结果。

## Environment Policy

- `sparta_edgesimpy`：只用于生成 raw logs 和验证真实 EdgeSimPy backend 可用性。
- `sparta`：用于 audit、train、eval、diagnosis、calibration、summarize。

如果 `sparta_edgesimpy` 环境因为缺少 torch 等训练依赖导致 audit 失败，但 raw logs 和 dataset 已生成，则切换到 `sparta` 环境继续运行审计和后续实验。

## Final Runbook

最终实验脚本集中在：

```text
scripts/final_cpu_precursor_v1/
```

按序执行：

1. `00_generate_data_3seed.ps1`
2. `01_audit_3seed.ps1`
3. `02_train_eval_lstm_3seed.ps1`
4. `03_train_eval_transformer_3seed.ps1`
5. `04_train_eval_sparta_3seed.ps1`
6. `05_calibrate_sparta_3seed.ps1`
7. `06_summarize_final_3seed.ps1`
