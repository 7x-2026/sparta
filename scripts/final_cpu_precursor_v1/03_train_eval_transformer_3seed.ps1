# Environment: run in conda env `sparta`.
# Purpose: train and evaluate Transformer-CP baseline for the 3 official cpu_precursor_v1 runs.

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$VARIANT = 'cpu_precursor_v1'

$RUN42 = 'run\20260617_002_edgesimpy_real_strict_seed42_cpu_precursor_v1'
$RUN2025 = 'run\20260617_001_edgesimpy_real_strict_seed2025_cpu_precursor_v1'
$RUN3407 = 'run\20260617_001_edgesimpy_real_strict_seed3407_cpu_precursor_v1'

$CONFIG42 = 'configs\sparta_edgesimpy_real_strict_seed42_cpu_precursor_v1.yaml'
$CONFIG2025 = 'configs\sparta_edgesimpy_real_strict_seed2025_cpu_precursor_v1.yaml'
$CONFIG3407 = 'configs\sparta_edgesimpy_real_strict_seed3407_cpu_precursor_v1.yaml'

function Invoke-TrainEvalTransformer {
    param([string]$Config, [string]$RunDir)

    python src\train.py --config $Config --model transformer --resume_dataset_run $RunDir --variant $VARIANT
    python src\evaluate.py --config $Config --model transformer --resume_dataset_run $RunDir --variant $VARIANT
}

Invoke-TrainEvalTransformer -Config $CONFIG42 -RunDir $RUN42
Invoke-TrainEvalTransformer -Config $CONFIG2025 -RunDir $RUN2025
Invoke-TrainEvalTransformer -Config $CONFIG3407 -RunDir $RUN3407
