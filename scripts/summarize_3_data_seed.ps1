param(
    [string]$RunSeed42,
    [string]$RunSeed2025,
    [string]$RunSeed3407
)

$ErrorActionPreference = "Stop"

python src\analysis\summarize_multiseed.py `
  --run_ids $RunSeed42 $RunSeed2025 $RunSeed3407 `
  --group_name multiseed_synthetic_full `
  --output_root run_groups `
  --strict

if ($LASTEXITCODE -ne 0) {
    throw "SPARTA multiseed summarization failed."
}
