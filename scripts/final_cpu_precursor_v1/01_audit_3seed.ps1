# Environment: run in conda env `sparta`.
# Purpose: rerun final audits and evidence diagnostics for the 3 official cpu_precursor_v1 runs.

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RUN42 = 'run\20260617_002_edgesimpy_real_strict_seed42_cpu_precursor_v1'
$RUN2025 = 'run\20260617_001_edgesimpy_real_strict_seed2025_cpu_precursor_v1'
$RUN3407 = 'run\20260617_001_edgesimpy_real_strict_seed3407_cpu_precursor_v1'

function Invoke-FinalAudit {
    param([string]$RunDir)

    $DatasetDir = Join-Path $RunDir 'dataset'
    $RawDir = Join-Path $RunDir 'raw_logs'
    $AuditDir = Join-Path $RunDir 'audit'

    python src\analysis\audit_synthetic_full.py --dataset_dir $DatasetDir --raw_dir $RawDir --output_dir $AuditDir
    python src\analysis\audit_cpu_precursor.py --run_dir $RunDir
    python src\analysis\evaluate_metric_evidence_baseline.py --run_dir $RunDir --evidence_norm pressure02
    python src\analysis\replay_metric_label_rule.py --run_dir $RunDir
}

Invoke-FinalAudit -RunDir $RUN42
Invoke-FinalAudit -RunDir $RUN2025
Invoke-FinalAudit -RunDir $RUN3407
