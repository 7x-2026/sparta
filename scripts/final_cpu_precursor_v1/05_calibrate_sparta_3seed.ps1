# Environment: run in conda env `sparta`.
# Purpose: run macro_f1 metric logit calibration for SPARTA only.

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$VARIANT = 'cpu_precursor_v1'
$CALIBRATION = 'macro_f1'

$RUN42 = 'run\20260617_002_edgesimpy_real_strict_seed42_cpu_precursor_v1'
$RUN2025 = 'run\20260617_001_edgesimpy_real_strict_seed2025_cpu_precursor_v1'
$RUN3407 = 'run\20260617_001_edgesimpy_real_strict_seed3407_cpu_precursor_v1'

function Invoke-SpartaCalibration {
    param([string]$RunDir)

    python src\analysis\calibrate_metric_logits.py --run_dir $RunDir --variant $VARIANT --objective macro_f1 --calibration_name $CALIBRATION --grid_min -2.0 --grid_max 2.0 --grid_step 0.1
}

Invoke-SpartaCalibration -RunDir $RUN42
Invoke-SpartaCalibration -RunDir $RUN2025
Invoke-SpartaCalibration -RunDir $RUN3407
