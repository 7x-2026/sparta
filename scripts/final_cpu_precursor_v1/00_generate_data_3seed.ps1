# Environment: run in conda env `sparta_edgesimpy`.
# Purpose: generate cpu_precursor_v1 raw logs and dataset for 3 data seeds.
# Note: if audit fails here because torch is missing, but dataset files exist,
# switch to conda env `sparta` and continue with 01_audit_3seed.ps1.

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$CONFIG42 = 'configs\sparta_edgesimpy_real_strict_seed42_cpu_precursor_v1.yaml'
$CONFIG2025 = 'configs\sparta_edgesimpy_real_strict_seed2025_cpu_precursor_v1.yaml'
$CONFIG3407 = 'configs\sparta_edgesimpy_real_strict_seed3407_cpu_precursor_v1.yaml'

python src\run_synthetic_full_pipeline.py --config $CONFIG42 --generate_only
python src\run_synthetic_full_pipeline.py --config $CONFIG2025 --generate_only
python src\run_synthetic_full_pipeline.py --config $CONFIG3407 --generate_only
