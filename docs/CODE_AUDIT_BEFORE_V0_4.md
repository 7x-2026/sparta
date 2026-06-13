# CODE AUDIT BEFORE V0.4

Audit date: 2026-06-13

Scope: read-only structural audit of the current SPARTA repository before saving a stable version. This audit did not retrain models, regenerate data, modify model/loss code, connect EdgeSimPy, or change experiment outputs.

## 1. Executive Summary

1. The current project is functionally saveable as a controlled synthetic MVP: synthetic full generation, run directory management, 3 data seed runs, group-level multiseed summaries, and per-seed winner reports are present.
2. The main run-aware pipeline is centered on `src/run_synthetic_full_pipeline.py`, which correctly rewrites runtime outputs into the active `run/...` directory before calling downstream scripts.
3. Evaluation output is much stronger than the original MVP: per-model CSVs, `all_test_results.csv`, confusion matrices, prediction distributions, per-class metrics, attribution artifacts, and multiseed summaries are all represented in tests.
4. The largest structural issue is duplication: there are two evaluation paths (`src/evaluate.py` and `src/analysis/supplement_eval_outputs.py`), two group manager modules (`src/utils/group_manager.py` and `src/utils/run_group_manager.py`), and a growing orchestration script in `src/run_synthetic_full_pipeline.py`.
5. Several configs and placeholder files are legacy or future-stage stubs. They should not be deleted now, but they should be clearly marked after the stable version is saved.
6. Legacy fixed directories (`data/`, `results/`, `checkpoints/`, `logs/`) still exist and are still referenced by debug/default configs. The synthetic full run path overwrites these at runtime, but accidental direct script use can still write to fixed paths.
7. Multiseed output is now conceptually in the right place (`run_groups/...`), but backward-compatible options such as `--output` and `--output_dir` can still bypass automatic group creation if used manually.
8. EdgeSimPy readiness is moderate: the five CSV raw-log interface is already useful, but audit and labeling code still assume synthetic-specific fields such as `scenario`, `bottleneck_node`, `bottleneck_link`, and fixed topology sizes.
9. Recommendation: `safe_to_commit`, with no required code changes before saving the current version. Perform cleanup as a follow-up refactor after the stable commit.

Issue counts:

- Critical: 0
- Major: 7
- Minor: 10

Recommendation: `safe_to_commit`

## 2. Current Project Structure

Observed top-level structure:

| path | exists | role |
| --- | --- | --- |
| `configs/` | yes | YAML configs for debug, synthetic full, and future model variants |
| `src/` | yes | Main source package |
| `src/simulation/` | yes | Debug and synthetic full raw-log generators |
| `src/preprocessing/` | yes | Path graph building, labeling, splitting, dataset stats |
| `src/datasets/` | yes | MVP PyTorch dataset wrapper |
| `src/models/` | yes | SPARTA plus LSTM/Transformer baselines; several future placeholders |
| `src/losses/` | yes | Multitask loss and placeholder focal loss |
| `src/analysis/` | yes | Synthetic audit, supplemental eval outputs, multiseed summary |
| `src/utils/` | yes | Config, IO, run/group managers, output checking, seed helpers |
| `src/experiments/` | no | Not currently used |
| `scripts/` | yes | Mostly placeholder shell scripts plus multiseed PowerShell helpers |
| `tests/` | yes | Unit and integration tests for debug, synthetic full, eval, multiseed |
| `run/` | yes | Run-managed experiment outputs |
| `run_groups/` | yes | Group-managed multiseed summaries |
| `data/` | yes | Legacy/static data output area |
| `results/` | yes | Legacy/static result output area |
| `checkpoints/` | yes | Legacy/static checkpoint output area |
| `logs/` | yes | Legacy/static log output area |

Main directory responsibilities:

- `configs/`: experiment defaults. Current configs still mix stable MVP configs with old placeholders.
- `src/simulation/`: owns raw log generation. `synthetic_full_generator.py` is the controlled benchmark generator.
- `src/preprocessing/`: owns conversion from raw logs to path graph samples and labels.
- `src/datasets/`: owns loading `.pkl` samples into tensors.
- `src/models/`: owns model definitions only.
- `src/losses/`: owns loss computation only.
- `src/analysis/`: owns audit/evaluation summaries and multiseed reporting.
- `src/utils/`: owns low-level helpers and run/group directory management.
- `scripts/`: convenience wrappers, not core logic.
- `run/`: immutable-ish per-run artifacts.
- `run_groups/`: immutable-ish per-multiseed group artifacts.

Files with unclear or mixed responsibilities:

- `src/run_synthetic_full_pipeline.py`: orchestration plus config mutation plus output checks plus mode logic. It works, but it is large.
- `src/analysis/supplement_eval_outputs.py`: evaluation logic, artifact generation, metric fallbacks, and manifest mutation are all in one script.
- `src/analysis/summarize_multiseed.py`: summary generation, dedupe, winner reports, group creation, and per-run output mutation are all in one script.
- `src/utils/group_manager.py` and `src/utils/run_group_manager.py`: overlapping concepts.

Files to keep:

- `src/run_synthetic_full_pipeline.py`
- `src/run_multiseed_synthetic_full.py`
- `src/simulation/synthetic_full_generator.py`
- `src/simulation/synthetic_scenarios.py`
- `src/preprocessing/build_path_graph.py`
- `src/preprocessing/generate_labels.py`
- `src/preprocessing/split_dataset.py`
- `src/datasets/sparta_dataset.py`
- `src/models/sparta.py`
- `src/models/lstm.py`
- `src/models/transformer.py`
- `src/losses/multitask_loss.py`
- `src/trainer.py`
- `src/analysis/audit_synthetic_full.py`
- `src/analysis/supplement_eval_outputs.py`
- `src/analysis/summarize_multiseed.py`
- `src/utils/run_manager.py`
- `src/utils/group_manager.py`
- `src/utils/output_checker.py`
- Current tests under `tests/`

Files that may be legacy or future-stage residue:

- Empty model/config placeholders: `gru.py`, `patchtst_cls.py`, `stgcn_cls.py`, `tcn.py`, `gru.yaml`, `patchtst.yaml`, `stgcn.yaml`, `tcn.yaml`.
- Empty preprocessing placeholders: `generate_attribution.py`, `normalize.py`, `prepare_alibaba_trace.py`, `prepare_google_trace.py`.
- Empty scripts: `scripts/00_setup.sh` through `scripts/07_evaluate_all.sh`.
- Empty/future simulation placeholders: `inject_trace.py`, `run_edgesimpy.py`.
- Legacy direct configs: `configs/sparta.yaml`, model-only `lstm.yaml`, `transformer.yaml`.

Do not touch for now:

- All trained runs in `run/`
- All multiseed groups in `run_groups/`
- `data/`, `results/`, `checkpoints/`, `logs/` until after deciding whether to preserve debug artifacts
- Placeholder files until there is a documented cleanup commit

## 3. What Looks Good

1. The controlled synthetic pipeline has a clear five-CSV raw log interface: node, link, service, path, SLA.
2. The MVP uses `.pkl` dataset splits consistently: `train.pkl`, `val.pkl`, `test.pkl`.
3. Runtime synthetic full runs are isolated under `run/YYYYMMDD_XXX_experiment/`.
4. Multiseed summaries are isolated under `run_groups/YYYYMMDD_XXX_group/`.
5. `run_manifest.json` records key run metadata including seeds and output locations.
6. `group_manifest.json` and `run_list.txt` provide traceability for multiseed summaries.
7. `generate_only`, `train_only`, `eval_only`, and `audit_only` are explicitly modeled in the synthetic full pipeline.
8. Baseline models correctly avoid attribution heads.
9. Baseline attribution values are represented as `N/A` in evaluation outputs, avoiding false `nan` interpretation.
10. SPARTA output heads are explicit: risk, node, link, and metric logits.
11. Tests cover dataset shape, model forward, loss backward, training, checkpoint load/eval, synthetic full pipeline modes, run managers, and multiseed reporting.

## 4. Problems Found

### Critical

None found. No immediate blocker prevents saving a stable controlled synthetic MVP version.

### Major

1. Duplicate evaluation paths
   - Files: `src/evaluate.py`, `src/analysis/supplement_eval_outputs.py`
   - Risk: `evaluate.py` writes a simpler result set, while `supplement_eval_outputs.py` writes the richer official run outputs. A future user may call the wrong script and get incomplete metrics.
   - Fix timing: after commit.

2. Duplicate group management modules
   - Files: `src/utils/group_manager.py`, `src/utils/run_group_manager.py`
   - Risk: Both create group-like directories and manifests with different filenames (`group_manifest.json` vs `multiseed_manifest.json`).
   - Fix timing: after commit.

3. Legacy fixed-path defaults remain in configs
   - Files: `configs/sparta_debug.yaml`, `configs/sparta_synthetic_full.yaml`, `configs/sparta.yaml`
   - Risk: direct calls to lower-level scripts can still write to `data/`, `results/`, `checkpoints/`, or `logs/` instead of a run directory.
   - Fix timing: after commit, unless preparing a public release immediately.

4. `src/run_synthetic_full_pipeline.py` is too large for a stable long-term entry point
   - Risk: it mixes CLI parsing, config mutation, run creation, subprocess orchestration, mode checks, output validation, and manifest updates.
   - Fix timing: after commit.

5. Audit script is synthetic-specific and has output text encoding problems
   - File: `src/analysis/audit_synthetic_full.py`
   - Risk: report strings appear mojibake in source. It is still functional for CSV outputs, but human text reports are hard to read and the script is not ready as a generic `audit_dataset.py`.
   - Fix timing: after commit.

6. Summary script can mutate run directories
   - File: `src/analysis/summarize_multiseed.py`
   - Behavior: writes `results/seed_winner_report.csv` inside each seed run.
   - Risk: summarization is no longer purely read-only. This is intentional per current requirement, but should be documented as an analysis artifact write.
   - Fix timing: document now, refactor after commit if desired.

7. Synthetic-specific fields leak into downstream assumptions
   - Files: `build_path_graph.py`, `generate_labels.py`, `audit_synthetic_full.py`
   - Risk: EdgeSimPy logs may not naturally have `scenario`, `bottleneck_node`, `bottleneck_link`, or identical topology constraints.
   - Fix timing: before EdgeSimPy integration.

### Minor

1. Many placeholder files are empty and may confuse new contributors.
2. `configs/lstm.yaml` and `configs/transformer.yaml` are model-only and not full runnable configs.
3. `src/analysis/supplement_eval_outputs.py` has a `--backup_existing` argument with default behavior that always backs up; there is no explicit no-backup switch.
4. `src/analysis/summarize_multiseed.py` keeps compatibility flags such as `--output`, `--output_dir`, and `--pattern`, which are useful but can bypass recommended group creation.
5. `run_multiseed_synthetic_full.py` still uses `run_group_manager.py` for some manifest behavior while `summarize_multiseed.py` uses `group_manager.py`.
6. `src/visualize.py` is empty.
7. `src/losses/focal_loss.py` is empty while configs only use CE.
8. `class_weights` support exists in `SPARTALoss`, but `train.py` does not currently read risk class weights from config.
9. AUC exists in both `metrics.py` and `supplement_eval_outputs.py` pathways with different completeness.
10. `aicode/` and paper/report markdowns are valuable context but not part of runtime; they can distract if the repo is presented as a clean package.

## 5. Redundant or Suspicious Files

| file | issue | suggestion | safe_to_delete_now |
| --- | --- | --- | --- |
| `src/evaluate.py` | Simpler evaluation overlaps richer supplemental evaluation | Keep now; later fold into one official evaluator | false |
| `src/analysis/supplement_eval_outputs.py` | Official richer evaluator but name sounds like patch script | Keep now; later rename to `evaluate_run.py` | false |
| `src/utils/group_manager.py` | New group manager overlaps with old run group manager | Keep now; later merge APIs | false |
| `src/utils/run_group_manager.py` | Older group manager still used by `run_multiseed_synthetic_full.py` | Keep now; review manually | false |
| `src/preprocessing/generate_attribution.py` | Empty placeholder; attribution is currently in `generate_labels.py` | Mark deprecated or implement later | false |
| `src/preprocessing/normalize.py` | Empty placeholder; explicitly not used by MVP | Mark not used for MVP | false |
| `src/preprocessing/prepare_alibaba_trace.py` | Empty future trace placeholder | Keep as future placeholder or remove in cleanup commit | false |
| `src/preprocessing/prepare_google_trace.py` | Empty future trace placeholder | Keep as future placeholder or remove in cleanup commit | false |
| `src/simulation/run_edgesimpy.py` | Empty EdgeSimPy placeholder | Keep until EdgeSimPy MVP is scoped | false |
| `src/simulation/inject_trace.py` | Empty trace placeholder | Review manually | false |
| `src/models/gru.py` | Empty model placeholder | Mark future baseline or remove later | false |
| `src/models/patchtst_cls.py` | Empty model placeholder | Mark future baseline or remove later | false |
| `src/models/stgcn_cls.py` | Empty model placeholder | Mark future baseline or remove later | false |
| `src/models/tcn.py` | Empty model placeholder | Mark future baseline or remove later | false |
| `configs/gru.yaml` | Empty placeholder | Mark deprecated/future | false |
| `configs/patchtst.yaml` | Empty placeholder | Mark deprecated/future | false |
| `configs/stgcn.yaml` | Empty placeholder | Mark deprecated/future | false |
| `configs/tcn.yaml` | Empty placeholder | Mark deprecated/future | false |
| `scripts/00_setup.sh` through `07_evaluate_all.sh` | Empty scripts | Keep until replacing with real workflow docs | false |
| `src/visualize.py` | Empty placeholder | Review manually | false |

## 6. Hard-coded Path Risks

Known fixed-path references:

| location | path pattern | risk |
| --- | --- | --- |
| `configs/sparta_debug.yaml` | `data/debug`, `results/debug`, `checkpoints/debug`, `logs/debug` | Expected for debug MVP, but not run-isolated |
| `configs/sparta_synthetic_full.yaml` | `data/synthetic_full`, `results/synthetic_full`, `checkpoints/synthetic_full` | Runtime pipeline overrides these, but direct lower-level scripts do not |
| `configs/sparta.yaml` | `data/sparta_dataset`, `results/sparta`, `checkpoints/sparta` | Legacy/full config, not run-managed |
| `src/analysis/audit_synthetic_full.py` | default `data/synthetic_full/dataset`, `results/synthetic_full/audit` | Safe when called by pipeline with explicit args; risky when called directly |
| `src/evaluate.py` | writes under `project.output_dir` | Correct if config is resolved run config; otherwise fixed config paths |
| `src/analysis/supplement_eval_outputs.py` | writes under `run_dir/results` | Good for run-managed eval |
| `src/analysis/summarize_multiseed.py` | can use explicit `--output` or `--output_dir` | Useful compatibility path, but can bypass new group creation |

Assessment:

- Current synthetic full full-run path is safe because `configure_run_paths()` rewrites project/data/paths into the active `run/...` directory.
- Direct calls to lower-level scripts remain risky if passed non-resolved configs.
- The presence of legacy `data/`, `results/`, `checkpoints/`, and `logs/` directories is not a bug, but they should be documented as legacy/debug outputs.

## 7. Pipeline Mode Check

| mode | behavior | risk |
| --- | --- | --- |
| full | generate, audit, train, eval, output checks | Good; creates a new run by default |
| `generate_only` | raw logs, path graph, labels/attribution, split, audit | Good; does not train/eval |
| `audit_only` | requires dataset, runs audit only | Good; does not generate/train |
| `train_only` | requires train/val/test, trains all three models | Good; does not generate data |
| `eval_only` | requires dataset and checkpoints, runs supplemental evaluation | Good; does not train/generate |
| `resume_run` | reuses `run/config/resolved_config.yaml` | Good; watch that CLI overrides can mutate resolved config |

Findings:

1. The mode responsibilities are clear enough for v0.4.
2. `train_only` checks dataset existence before training.
3. `eval_only` checks dataset and checkpoints before evaluation.
4. `generate_only` still runs audit, which is expected by current requirements.
5. `audit_only` is cleanly separated.
6. Output checks guard against missing single-model result files.

Potential cleanup:

- Move mode functions into `src/pipelines/synthetic_full.py` after the stable commit.
- Move config path rewriting into a reusable function or class.

## 8. Result Output Check

Single-run outputs:

- `supplement_eval_outputs.py` requires and writes:
  - `results/lstm_test_results.csv`
  - `results/transformer_test_results.csv`
  - `results/sparta_test_results.csv`
  - `results/all_test_results.csv`
  - confusion matrix, prediction distribution, per-class precision/recall/F1
  - SPARTA attribution artifacts

Risk assessment:

1. `all_test_results.csv` is not generated alone in the official eval path. It is rebuilt from the three per-model files.
2. `evaluate.py` can still produce simpler result files. Avoid using it as the official synthetic full evaluator.
3. Baseline attribution values are `N/A`, which is correct.
4. SPARTA attribution metrics are numeric when valid attribution labels exist.

Multiseed outputs:

- Recommended current path:
  - `run_groups/.../multiseed_summary.csv`
  - `run_groups/.../multiseed_mean_std.csv`
  - `run_groups/.../per_seed_winner_report.csv`
  - `run_groups/.../metric_winner_counts.csv`
  - `run_groups/.../group_manifest.json`
  - `run_groups/.../logs/summarize_multiseed.log`
- Per-run winner:
  - `run/.../results/seed_winner_report.csv`

Risk assessment:

1. Current `summarize_multiseed.py` defaults to explicit `--run_ids` or `--run_list`; pattern scanning is retained but not default.
2. Dedupe logic exists and logs duplicate `(data_seed, train_seed, model)` keys.
3. Strict mode exists and checks expected seed/model row counts.
4. The script refuses to overwrite output files unless `--overwrite` is passed.
5. There is still a compatibility `--output` path. This should remain for tests/backward compatibility but should not be the recommended workflow.

## 9. Config Audit

| config_file | purpose | current_status | keep / merge / deprecated | reason |
| --- | --- | --- | --- | --- |
| `configs/sparta_debug.yaml` | Windows/debug MVP config | Working debug path, fixed directories | keep | Useful for quick MVP tests |
| `configs/sparta_synthetic_full.yaml` | Current synthetic full controlled benchmark | Active main config, run pipeline rewrites paths | keep | Primary config for v0.4 |
| `configs/sparta.yaml` | Older/full SPARTA config | Incomplete for current run-managed synthetic full | deprecated | Uses fixed paths and lacks run/seed split |
| `configs/lstm.yaml` | Model-only LSTM snippet | Not a runnable full config | merge/deprecated | Model is selected via CLI now |
| `configs/transformer.yaml` | Model-only Transformer snippet | Not a runnable full config | merge/deprecated | Model is selected via CLI now |
| `configs/gru.yaml` | Empty placeholder | Not active | deprecated | Future baseline only |
| `configs/patchtst.yaml` | Empty placeholder | Not active | deprecated | Future baseline only |
| `configs/stgcn.yaml` | Empty placeholder | Not active | deprecated | Future baseline only |
| `configs/tcn.yaml` | Empty placeholder | Not active | deprecated | Future baseline only |

Answers to requested config questions:

1. There is content overlap between `sparta_debug.yaml`, `sparta_synthetic_full.yaml`, and `sparta.yaml`.
2. Old configs still contain fixed `data/`, `results/`, and `checkpoints/` paths.
3. New synthetic full experiments go through `run/...` only when launched via `run_synthetic_full_pipeline.py`.
4. `sparta_synthetic_full.yaml` distinguishes `data_seed` and `train_seed`.
5. Synthetic full is clear; multiseed is CLI-driven rather than config-driven; EdgeSimPy config boundaries are not yet defined.
6. Empty configs are easy to misuse and should be documented or moved after commit.

## 10. Model and Loss Audit

Files checked:

- `src/models/sparta.py`
- `src/models/lstm.py`
- `src/models/transformer.py`
- `src/losses/multitask_loss.py`
- `src/train.py`
- `src/evaluate.py`
- `src/analysis/supplement_eval_outputs.py`

Findings:

1. Baselines do not use attribution heads. `lstm.py` and `transformer.py` return only `risk_logits`.
2. SPARTA returns `risk_logits`, `node_logits`, `link_logits`, and `metric_logits`.
3. Node/link logits are masked with `-1e9` for padding in SPARTA.
4. `SPARTALoss` only applies attribution loss when attribution logits are present and `attr_mask` is valid.
5. Loss weights `lambda_node`, `lambda_link`, and `lambda_metric` are configurable.
6. `risk_class_weights` support exists in the loss class but is not wired from config in `train.py`.
7. Best checkpoint selection uses `train.best_metric`, defaulting to `macro_f1`.
8. `train_seed` is used in `train.py` and sets deterministic CuDNN behavior.
9. Metric computation is duplicated between `metrics.py`, `evaluate.py`, and `supplement_eval_outputs.py`.
10. AUC is most complete in `supplement_eval_outputs.py`; `metrics.py` returns `nan` for AUC in the simpler path.
11. Confusion matrix, prediction distribution, and per-class metrics are centralized in `supplement_eval_outputs.py`, not `evaluate.py`.

Recommendation:

- Keep model/loss code unchanged before commit.
- After commit, decide whether `supplement_eval_outputs.py` should become the official `evaluate_run.py`.

## 11. Data Audit Script Review

Files:

- `src/analysis/audit_synthetic_full.py`
- `src/analysis/audit_dataset.py` does not exist.

Current support:

| check/output | supported |
| --- | --- |
| label distribution | yes |
| scenario distribution | yes |
| label by scenario | yes |
| attribution distribution | yes |
| attribution majority baseline | yes |
| split shift report | yes |
| risk node/link concentration | yes |
| risk metric coverage | yes |
| queue metric minimum ratio | yes |
| NaN/inf feature check | no, not in current enhanced audit |
| path size summary | no, not in current enhanced audit |
| generic dataset source | no, synthetic-specific |

Assessment:

- `audit_synthetic_full.py` is useful and should be kept.
- It should not be renamed before commit because current pipeline/tests expect it.
- Before EdgeSimPy, create a generic `audit_dataset.py` or add a data-source abstraction, then keep `audit_synthetic_full.py` as a wrapper.
- The report text in `audit_synthetic_full.py` appears mojibake and should be fixed after the stable commit.

## 12. EdgeSimPy Readiness

Answers:

1. A five-CSV raw-log interface exists and is a good integration boundary.
2. EdgeSimPy can likely replace only the raw-log generator if it exports compatible `node_log.csv`, `link_log.csv`, `service_log.csv`, `path_log.csv`, and `sla_log.csv`.
3. `build_path_graph.py`, `generate_labels.py`, and `split_dataset.py` can probably be reused if CSV schema compatibility is maintained.
4. Synthetic-specific fields may block EdgeSimPy:
   - `scenario`
   - `event_id`
   - `severity`
   - `affected_node`
   - `affected_link`
   - `bottleneck_node`
   - `bottleneck_link`
5. The current topology assumptions are fixed around synthetic full v0 dimensions and path candidate logic.
6. Before EdgeSimPy, add a small schema contract document for the five CSVs.
7. Consider adding `src/simulation/run_edgesimpy_mvp.py` later, but not before saving v0.4.
8. Keep the synthetic generator as a controlled benchmark even after EdgeSimPy is added.

Readiness rating: moderate. The boundary is promising, but schema assumptions must be made explicit.

## 13. Recommended Cleanup Plan

### Phase A: safe cleanup before commit

Goal: minimal risk. Do not refactor runtime code.

1. Save this audit report.
2. Ensure `.gitignore` excludes `run/`, `run_groups/`, `data/`, `results/`, `checkpoints/`, `logs/`, `__pycache__/`, and `.pytest_cache/`.
3. Record the latest test command and passing status in commit notes.
4. Do not delete placeholder files yet.
5. Do not move generated experiment outputs.

### Phase B: refactor after commit

Goal: reduce duplication without changing behavior.

1. Merge `group_manager.py` and `run_group_manager.py` into one stable group manager API.
2. Rename or replace `supplement_eval_outputs.py` with an official `evaluate_run.py`.
3. Decide whether `evaluate.py` remains as a lightweight evaluator or becomes a wrapper around the official evaluator.
4. Split `run_synthetic_full_pipeline.py` into:
   - config/run preparation
   - generation stage
   - audit stage
   - train stage
   - eval stage
5. Move `fast_dev_run` overrides into a test helper or explicit config overlay.
6. Mark empty placeholder configs/files with clear comments or remove them in a dedicated cleanup commit.
7. Fix mojibake in `audit_synthetic_full.py` report text.

### Phase C: EdgeSimPy preparation

Goal: keep synthetic full stable while adding a new raw-log backend.

1. Write `docs/RAW_LOG_SCHEMA_V0.md`.
2. Add a schema validator for the five CSV files.
3. Create `src/simulation/run_edgesimpy_mvp.py`.
4. Create or refactor `audit_dataset.py` to support both synthetic and EdgeSimPy sources.
5. Keep `synthetic_full_generator.py` as a controlled benchmark generator.
6. Make synthetic-only fields optional in downstream preprocessing.

## 14. Files That Should Not Be Touched Now

Do not touch before saving this version:

- `src/models/sparta.py`
- `src/models/lstm.py`
- `src/models/transformer.py`
- `src/losses/multitask_loss.py`
- `src/train.py`
- `src/trainer.py`
- `src/datasets/sparta_dataset.py`
- `src/preprocessing/build_path_graph.py`
- `src/preprocessing/generate_labels.py`
- `src/preprocessing/split_dataset.py`
- `src/simulation/synthetic_full_generator.py`
- Existing `run/` directories
- Existing `run_groups/` directories
- Existing `data/`, `results/`, `checkpoints/`, and `logs/` artifacts

## 15. Suggested Commit Message

Suggested commit message:

```text
stabilize synthetic full MVP run and multiseed reporting

- add run-managed synthetic full pipeline modes
- add 3 data seed run grouping and winner reports
- preserve baseline attribution metrics as N/A
- add audit and output checks for v0.4 readiness
```

Alternative shorter message:

```text
prepare controlled synthetic full MVP for v0.4
```

## 16. Final Recommendation

Recommendation: `safe_to_commit`

Reasoning:

- No critical structure issues were found.
- Current pipeline and reporting behavior are covered by tests.
- The known problems are mostly duplication, naming, and future extensibility concerns.
- Large refactors before saving the version would be riskier than committing the current stable state and cleaning up afterward.

