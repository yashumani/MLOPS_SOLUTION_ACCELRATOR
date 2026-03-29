# MLOps V3 Pipeline Quality & Completion Skill

> **Scope**: Pipeline mechanics, data flow, stage contracts, error handling, and operational correctness.  
> **Out of Scope**: Model accuracy, hyperparameter tuning, feature engineering strategy, algorithm selection.  
> **Goal**: Get every stage working correctly so the end user can iterate on model quality independently.

---

## 1. Pipeline Architecture (Verified Topology)

```
s1(Ingestion) → s2(Preparation) → s3(Preprocessing) → s4(Feature Engineering)
                     │                                        │
                     │                           ┌────────────┼────────────┐
                     │                           ▼            ▼            ▼
                     │                        s5a(PyCaret) s5b(FLAML) s5t(TimeSeries)
                     │                           │            │            │
                     │                           ▼            ▼            ✗ (unwired)
                     │                        s5z(Aggregate Baseline)
                     │                           │
                     ▼                           ▼
              s06(Variant Runner) ◄── s2.dataset_out (C2 fix: avoids double-preprocessing)
                     │
                     ▼
              s08(Optuna HPO) ◄── s06.champion
                     │
                     ▼
              s09(Aggregate PhaseC)
                     │
                     ▼
              s10(Final Evaluation) ◄── s5z + s06 + s09 champions
                     │
                     ▼
              s12(Model Registration) [NOT WIRED — manual]
```

---

## 2. Stage-by-Stage Contract Checklist

Use this checklist to verify each stage. A stage **passes** only when ALL items are ✅.

### s1 — Ingestion
- [ ] Reads dataset from `azureml://` datastore URI (config `dataset.path`)
- [ ] Writes `dataset_out/` with CSV + metadata
- [ ] Logs row count, column count, target distribution to MLflow
- [ ] Handles missing datastore gracefully (clear error message, not raw traceback)

### s2 — Preparation
- [ ] Reads `s1.dataset_out`
- [ ] Drops ID columns (per config `stages.preparation.drop_columns`)
- [ ] Handles missing values per config strategy
- [ ] Writes `dataset_out/` and `prep_report/`
- [ ] Logs preparation summary to MLflow

### s3 — Preprocessing
- [ ] Reads `s2.dataset_out` and `s2.prep_report`
- [ ] Applies encoding (label/onehot per config)
- [ ] Applies scaling (per config)
- [ ] Applies imputation (per config)
- [ ] **Does NOT apply SMOTE** (deferred to post-model-selection)
- [ ] Writes `dataset_out/` and `prep3_report/`
- [ ] Error handling: wraps encoding/scaling in try/except with traceback

### s4 — Feature Engineering
- [ ] Reads `s3.dataset_out`
- [ ] Applies feature selection per config
- [ ] Writes `dataset_out/` and `fe_report/`
- [ ] Logs selected features list to MLflow

### s5a — PyCaret Baseline
- [ ] Reads `s4.dataset_out`
- [ ] Calls `setup(preprocess=False)` — no double-preprocessing
- [ ] Calls `compare_models()` with configured fold count
- [ ] Saves best model as `.pkl` + manifest JSON + metrics JSON
- [ ] **Verify**: MLflow tracking URI fix (azureml→https) is present
- [ ] **Verify**: `log_models=False` or autolog suppression to avoid registry errors

### s5b — FLAML Baseline
- [ ] Reads `s4.dataset_out`
- [ ] Runs FLAML AutoML with configured `time_budget` + `metric`
- [ ] Saves best model + manifest + metrics
- [ ] **Verify**: MLflow tracking URI fix present
- [ ] Handles FLAML timeout gracefully (deadline buffer ≥ 360s)

### s5t — TimeSeries Baseline
- [ ] Reads `s4.dataset_out`
- [ ] Trains timeseries models (if `task_type == timeseries`)
- [ ] **KNOWN ISSUE**: Outputs are NOT wired to s5z. Results are silently lost.
- [ ] Status: **UNUSED** for classification/regression pipelines

### s5z — Aggregate Baseline
- [ ] Reads `s5a` and `s5b` manifests + models
- [ ] Compares baseline models on configured metric
- [ ] Selects Phase A champion
- [ ] Writes `champion_model/`, `champion_manifest/`, `aggregate_report/`
- [ ] **Does NOT** read s5t outputs (known gap — acceptable for non-timeseries tasks)

### s06 — Phase B Variant Runner
- [ ] Reads `s2.dataset_out` (NOT s4 — intentional C2 fix to avoid double-preprocessing)
- [ ] Reads config + variant recipes from uploaded code
- [ ] Applies variant preprocessing with `apply_smote=False`
- [ ] Trains models via PyCaret `compare_models()` on clean data
- [ ] **After model selection**: SMOTE retrain block resamples and re-fits best model
- [ ] Evaluates on holdout with `preprocess_holdout_aligned()` (no SMOTE in feature selection)
- [ ] Writes `leaderboard_csv`, `all_results_json`, `champion_manifest`, `champion_model`
- [ ] Shell `cp` commands in YAML: **fragile on zero results** (known W1 issue)
- [ ] Nested MLflow runs: one child per variant×engine

### s08 — Phase C HPO (Optuna)
- [ ] Reads `s06.champion_manifest` to identify best model type
- [ ] Runs Optuna trials with configured `n_trials` and `timeout`
- [ ] Saves optimized model + metrics JSON + study artifact
- [ ] `hpo_study` output declared but not consumed downstream (known W5 — acceptable)

### s09 — Aggregate Phase C
- [ ] Reads `s08.hpo_metrics_json` + `s08.optimized_model`
- [ ] Writes `aggregate_report/` + `champion_model/`

### s10 — Final Evaluation
- [ ] Reads baseline champion (s5z), Phase B champion (s06), Phase C champion (s09)
- [ ] Loads holdout split from dataset
- [ ] Evaluates ALL champions on same holdout using `balanced_accuracy_score`
- [ ] Uses `stratify=y` for train/test split
- [ ] Selects overall winner
- [ ] Writes `final_report/` + `final_champion_model/`
- [ ] Generates evaluation plots (confusion matrix, ROC, etc.)

### s12 — Model Registration
- [ ] **Status**: Component YAML exists but NOT loaded in `pipeline_builder.py`
- [ ] Manual step: requires explicit invocation after pipeline completes
- [ ] Registers final champion model in Azure ML registry

---

## 3. Known Issues Registry

### CRITICAL

| ID | Issue | File(s) | Fix |
|----|-------|---------|-----|
| C1 | s5t outputs not wired to s5z — timeseries results silently lost | `pipeline_builder.py`, `aggregate_baseline.yml` | Add timeseries inputs to aggregate component; wire in pipeline builder. **Acceptable to defer** if timeseries is not used. |
| C2 | Clustering config (`config_clustering_online_retail_azureml.yml`) references wrong dataset (College.csv) and wrong experiment name | `configs/config_clustering_online_retail_azureml.yml` | Update `experiment_name`, `blob_path`, dataset references |

### WARNING

| ID | Issue | File(s) | Fix |
|----|-------|---------|-----|
| W1 | s06 YAML `cp` commands crash on zero variant results | `components/s06_phaseb_variant_runner.yml` | Add `test -f ... && cp ... \|\| echo '{}' > ...` guards |
| W2 | Environment version tags in submit_pipeline.py say `:20` while YAMLs use `:23` | `pipelines/submit_pipeline.py` | Update tags (cosmetic only) |
| W3 | Config schema validates only 4 of 15+ config sections | `src/orchestration/config_schema.py` | Expand schema; add conditional `target_column` requirement |
| W4 | s5a/s5b missing MLflow tracking URI fix | `stage5_pycaret_train.py`, `stage5_flaml_train.py` | Add azureml→https URI conversion at script start |
| W5 | s08 `hpo_study` output declared but not consumed | `phasec_optuna_hpo.yml` | Informational — no action needed |
| W6 | No step script uses `sys.exit(1)` | All step scripts | Add structured error reporting |
| W7 | s06 YAML description says "Stage 4 dataset" but receives s2 output | `s06_phaseb_variant_runner.yml` | Update description to match actual wiring |

### INFO

| ID | Issue | File(s) | Fix |
|----|-------|---------|-----|
| I1 | 6 unused component YAMLs in `components/` | Various | Remove or archive dead components |
| I2 | s06 uses `--config_path` while all others use `--config` | `s06_phaseb_variant_runner.py` | Cosmetic — no action needed |
| I3 | stage3 has lowest error handling density (3 except in 420 lines) | `stage3_preprocessing.py` | Add try/except around encoding/scaling/imputation |
| I4 | Only 3 of 12 scripts use `traceback.format_exc()` | Most step scripts | Add traceback logging to catch blocks |
| I5 | stage1 `--dataset_in` declared but ignored (legacy) | `stage1_ingestion.py` | Remove dead parameter |

---

## 4. Exit Criteria — Pipeline Completion Definition

### Gate 1: Pipeline Mechanics (MUST PASS)

All items must be ✅ to consider the pipeline project complete.

| # | Criterion | How to Verify |
|---|-----------|---------------|
| 1.1 | **All 12 active steps complete without errors** in Azure ML | Submit pipeline for classification, regression, and clustering configs → all steps show "Completed" in Azure ML Studio |
| 1.2 | **Classification balanced_accuracy > 0.55** on holdout | Read `s10/named-outputs/final_report/final_report.json` → `balanced_accuracy` field. Value must be meaningfully above random chance (0.50). Target: ≥ 0.55. |
| 1.3 | **No SMOTE data leakage** | s06 logs show "⏭️ SMOTE deferred" and "🔄 SMOTE retrain" messages. PyCaret CV runs on clean data. |
| 1.4 | **Holdout evaluation uses stratified split** | s10 logs show `stratify=y` in train_test_split. Holdout class distribution matches training distribution. |
| 1.5 | **Phase B champion ≠ random chance** | s06 leaderboard shows no model with balanced_accuracy ≤ 0.50 as champion. |
| 1.6 | **Final champion model artifact exists and is loadable** | `s10/named-outputs/final_champion_model/` contains a valid `.pkl` file that loads with `joblib.load()`. |
| 1.7 | **Config-driven execution** | All dataset paths, task types, and hyperparams come from YAML config — zero hardcoded values in step scripts. |

### Gate 2: Multi-Task-Type Support (MUST PASS)

| # | Criterion | How to Verify |
|---|-----------|---------------|
| 2.1 | **Classification pipeline runs end-to-end** | Submit with `config_classification_telecom_churn_azureml.yml` → all steps complete |
| 2.2 | **Regression pipeline runs end-to-end** | Submit with `config_regression_college_azureml.yml` → all steps complete |
| 2.3 | **Clustering pipeline runs end-to-end** | Submit with a corrected clustering config → all steps complete |
| 2.4 | **Task-type isolation preserved** | Fixing regression does NOT break classification. Verified by re-running classification after regression fixes. |

### Gate 3: Operational Readiness (SHOULD PASS)

| # | Criterion | How to Verify |
|---|-----------|---------------|
| 3.1 | **Duplicate submission guard works** | Running `submit_pipeline.py` twice without `--force` → second run aborts |
| 3.2 | **`--stop_compute` flag works** | Submit with `--wait --stop_compute` → compute cluster scales to 0 after job completes |
| 3.3 | **Config validation catches invalid configs** | Submit with missing `target_column` for classification → clear error before job submission |
| 3.4 | **Pipeline outputs are downloadable** | Run `scripts/extract_job_results.py` on completed job → outputs downloaded successfully |

### Gate 4: Documentation (NICE TO HAVE)

| # | Criterion | How to Verify |
|---|-----------|---------------|
| 4.1 | **All component YAML descriptions match actual behavior** | Manual review of 18 YAML files |
| 4.2 | **Known issues documented** | This skill file exists and is up to date |
| 4.3 | **Unused components archived or removed** | No dead YAML files in `components/` |

---

## 5. Verification Commands

### Run Full Classification Pipeline
```bash
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id <sub> --resource_group <rg> --workspace_name <ws> \
  --compute mlopsv2computecluster --wait --stop_compute
```

### Check Final Evaluation Results
```bash
# After downloading outputs:
cat job_outputs_<name>/s10/named-outputs/final_report/final_report.json | python -m json.tool
# Look for: "balanced_accuracy": <value> — must be > 0.55
```

### Verify SMOTE Fix in Logs
```bash
# In Azure ML Studio → Job → s06 step → Outputs + logs → user_logs/std_log.txt
# Search for:
#   "⏭️ SMOTE deferred"     — confirms SMOTE NOT applied before model selection
#   "🔄 SMOTE retrain"       — confirms SMOTE applied AFTER model selection
#   "✅ Model retrained"     — confirms retrain completed
```

### Validate Stage Outputs Exist
```bash
# After downloading job outputs:
for step in s1 s2 s3 s4 s5a s5b s5z s06 s08 s09 s10; do
  echo -n "$step: "
  if [ -d "job_outputs_<name>/$step/named-outputs" ]; then
    ls "job_outputs_<name>/$step/named-outputs/" | wc -l
    echo " outputs"
  else
    echo "MISSING"
  fi
done
```

---

## 6. What the Pipeline Does vs. What the User Tunes

### Pipeline Responsibility (this skill)
- Data flows correctly between stages
- Outputs are written to correct paths
- Error handling prevents silent failures
- SMOTE/preprocessing applied at correct points (no leakage)
- MLflow tracking works without registry errors
- Config validation catches bad inputs early
- All three task types run without crashing

### User Responsibility (out of scope for this skill)
- Choosing the right metric to optimize (accuracy vs balanced_accuracy vs AUC)
- Selecting which variant recipes to include
- Setting time budgets for AutoML
- Choosing imputation/encoding/scaling strategies
- Tuning HPO search spaces
- Deciding on feature selection methods
- Selecting which models to include in `compare_models()`
- **Interpreting results and iterating on model quality**

---

## 7. Quick Reference: File Locations

| Purpose | File |
|---------|------|
| Pipeline wiring | `pipelines/pipeline_builder.py` |
| Job submission | `pipelines/submit_pipeline.py` |
| Config validation | `src/orchestration/config_schema.py` |
| Step scripts | `src/steps/stage*.py`, `src/steps/s06_*.py`, etc. |
| Component YAMLs | `components/*.yml` |
| Variant recipes | `configs/recipes/{task}/variant_search/*.yml` |
| Config files | `configs/config_*.yml` |
| Utilities | `src/utils/*.py` |
| Output extraction | `scripts/extract_job_results.py` |
| This quality skill | `docs/PIPELINE_QUALITY_SKILL.md` |

---

## 8. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-07 | SMOTE deferred to post-model-selection | Prevents data leakage into PyCaret's CV folds. Model selected on clean data, then retrained on SMOTE'd data. |
| 2026-03-07 | s06 reads s2 output (not s4) | C2 fix: s06 does its own preprocessing per variant recipe. Using s4 output would double-preprocess. |
| 2026-03-07 | s5t not wired to s5z | Acceptable for classification/regression. Timeseries support is a future enhancement. |
| 2026-03-07 | s12 not wired in pipeline | Model registration is a manual post-pipeline step. Keeps pipeline idempotent. |
| 2026-03-07 | `preprocess=False` in PyCaret | Data arrives pre-processed from s3/s4 (or s06 variant preprocessing). PyCaret must not re-encode/scale. |

---

## 9. When Is This Project DONE?

**The pipeline project is complete when:**

1. ✅ All Gate 1 criteria pass (pipeline mechanics)
2. ✅ All Gate 2 criteria pass (multi-task-type support)  
3. ✅ At least 3 of 4 Gate 3 criteria pass (operational readiness)

**Specifically, the single most important exit criterion is:**

> **Classification pipeline produces `balanced_accuracy > 0.55` on holdout data, with no SMOTE leakage, across a clean end-to-end run.**

Once this is achieved, the pipeline infrastructure is proven correct. Any further improvements are model-quality work (the user's domain), not pipeline-quality work.

**If balanced_accuracy is still ≈ 0.50 after the SMOTE leakage fix (`sincere_turnip_lb7t1dsgt0`):**
- The issue is NOT pipeline mechanics — it's the dataset/model combination
- Verify by checking if PyCaret's own CV metrics (logged in s06) are also ~0.50
- If CV metrics are reasonable (>0.60) but holdout is ~0.50, there's still a preprocessing mismatch
- If CV metrics are also ~0.50, the models genuinely can't predict this dataset well — that's a user/data-science problem, not a pipeline problem
