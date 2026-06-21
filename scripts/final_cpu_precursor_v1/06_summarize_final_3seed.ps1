# Environment: run in conda env `sparta`.
# Purpose: summarize the final cpu_precursor_v1 3-seed experiment.

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RUN42 = 'run\20260617_002_edgesimpy_real_strict_seed42_cpu_precursor_v1'
$RUN2025 = 'run\20260617_001_edgesimpy_real_strict_seed2025_cpu_precursor_v1'
$RUN3407 = 'run\20260617_001_edgesimpy_real_strict_seed3407_cpu_precursor_v1'
$OUTPUT_DIR = 'run_groups\cpu_precursor_v1_final_3seed'

python src\analysis\summarize_cpu_precursor_final_3seed.py --run_dirs $RUN42 $RUN2025 $RUN3407 --output_dir $OUTPUT_DIR

Write-Host "Final summary written to:"
Write-Host "  $OUTPUT_DIR\summary.csv"
Write-Host "  $OUTPUT_DIR\mean_std.csv"
Write-Host "  $OUTPUT_DIR\final_report.md"
