$ErrorActionPreference = "Stop"

python src\run_multiseed_synthetic_full.py --config configs\sparta_synthetic_full.yaml --data_seeds 42 2025 3407 --train_seed 0 --experiment_name synthetic_full_3seed --group_name multiseed_synthetic_full
if ($LASTEXITCODE -ne 0) {
    throw "SPARTA multiseed group run failed."
}
