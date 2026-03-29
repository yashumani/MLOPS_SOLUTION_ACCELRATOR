# MLOps Solution Accelerator V3 — Complete Technical Reference

> **Generated from code inspection** — every claim below traces to a specific source file and line number.
> Last verified: 2025-03-29 against the live repository.

---

## Table of Contents

1. [Mental Model](#1-mental-model)
2. [File Inventory](#2-file-inventory)
3. [Data-Flow Trace](#3-data-flow-trace)
4. [Function-Level Breakdown](#4-function-level-breakdown)
5. [Variant System Deep Dive](#5-variant-system-deep-dive)
6. [Artifact Map](#6-artifact-map)
7. [Config as Control Panel](#7-config-as-control-panel)
8. [MLflow Tracking Hierarchy](#8-mlflow-tracking-hierarchy)
9. [Operational Guide](#9-operational-guide)
10. [Glossary](#10-glossary)

---

## 1. Mental Model

Imagine a cooking competition with three rounds, managed by a head judge who never tastes the food themselves.

**Round A (Baseline):** Every contestant (PyCaret and FLAML) cooks the same dish using the house recipe — identical ingredients, same prep. The judges score them on AUC (or R2, or silhouette). The best dish becomes the "baseline champion." This is Stages 5a/5b/5z.

**Round B (Variant Search):** Now the competition gets creative. A dataset profiler inspects the raw ingredients — how fresh, how varied, how balanced — and recommends up to 20 of 457 available recipes. Each recipe is a complete preprocessing pipeline: a specific imputation method, encoding scheme, scaling approach, imbalance handler, and feature selector. Every recipe is cooked by both PyCaret and FLAML. A cheap "taste test" (SGD proxy) eliminates obviously bad recipes before the expensive full cook. The best dish from Round B becomes the "Phase B champion." This is Stage 6 (s06).

**Round C (Hyperparameter Optimization):** The winning recipe from Round B gets fine-tuned. Optuna runs 50 trials, adjusting seasoning (learning rate, tree depth, regularization) to squeeze out the last percentage points. This is Stage 8/9.

**The Final Judging (Stage 10):** All three round champions compete on a holdout set that none of them have ever seen. The overall best model wins. SHAP explains why it won (which ingredients mattered most). An AIM Tournament ranks every candidate ever tested across multiple metrics. The winning model gets registered in the MLflow Model Registry (Stage 12) at "Staging" — a human must promote it to "Production."

The head judge is `submit_pipeline.py`: it never trains a model, never touches data, never computes a metric. It assembles the Azure ML pipeline DAG, submits it, and walks away. Everything else runs as Azure ML component jobs on remote compute.

---

## 2. File Inventory

### 2.1 Pipeline Infrastructure

| File | Lines | Purpose | Reads | Writes |
|------|-------|---------|-------|--------|
| `pipelines/submit_pipeline.py` | 755 | CLI entrypoint — parses args, selects variants, submits Azure ML job | Config YAML, recipe YAMLs | `.submit.lock`, `.last_submitted_job` |
| `pipelines/pipeline_builder.py` | 310 | Defines `@dsl.pipeline` DAG, loads 14 component YAMLs | 18 component YAMLs | — |
| `src/orchestration/config_schema.py` | 199 | JSON Schema validation for config files | — | — |

### 2.2 Step Scripts

| File | Lines | Step ID | Purpose | Reads | Writes |
|------|-------|---------|---------|-------|--------|
| `src/steps/stage0_data_validation.py` | 379 | S00 | Data quality gate (RED/YELLOW/GREEN) | Config, dataset CSV | `validation_results.json`, validated CSV |
| `src/steps/stage1_ingestion.py` | 1112 | S01 | Read-only EDA + recipe recommendations | Config, dataset CSV | `eda_report.json`, `column_statistics.csv`, `recipe_recommendations.json`, `time_series_detection.json`, raw CSV passthrough |
| `src/steps/stage2_preparation.py` | 351 | S02 | Imputation + statistical tests + high-cardinality filter | Config, raw CSV | Prepared CSV, `prep_report.json` |
| `src/steps/stage3_preprocessing.py` | 371 | S03 | Recipe-driven encoding + scaling + VIF | Config, prepared CSV, recipe YAML | Preprocessed CSV (all-numeric), `prep3_report.json` |
| `src/steps/stage4_feature_engineering.py` | 396 | S04 | Feature selection + PCA + imbalance detection | Config, preprocessed CSV, recipe YAML | Feature-engineered CSV, `imbalance_metadata.json` |
| `src/steps/stage5_pycaret_train.py` | 435 | S05a | PyCaret `compare_models` baseline | Config, FE CSV | `model.pkl`, `threshold_info.json`, leaderboard CSV, manifest JSON |
| `src/steps/stage5_flaml_train.py` | 364 | S05b | FLAML AutoML baseline (skips clustering) | Config, FE CSV | `model.pkl`, iterations CSV, manifest JSON |
| `src/steps/stage5_timeseries_train.py` | 388 | S05t | Statsmodels forecasting (conditional) | Config, FE CSV, `time_series_detection.json` | `model.pkl`, breakdown CSV, manifest JSON |
| `src/steps/aggregate_baseline.py` | 391 | S05z | Merge baselines → Phase A champion | Config, S05a+S05b manifests+models | Selection report JSON, champion model copy, stage signal |
| `src/steps/s06_phaseb_variant_runner.py` | 3139 | S06 | N variants × M engines with 3-round funnel | Config, **S02 prepared CSV**, variant YAMLs | `leaderboard.csv`, `champion_manifest.json`, `model.pkl`, holdout metrics, proxy/feasibility reports |
| `src/steps/s07_phase2_pipeline_attribution.py` | 715 | S07 | Post-training attribution analysis | S06 outputs (all_results, leaderboard, manifest) | `pipeline_attribution.json`, `decision_impact_table.csv`, `phase2_summary.md` |
| `src/steps/phaseb_pycaret_recipe.py` | 344 | — | Legacy per-recipe PyCaret training | Config, recipe YAML, CSV | Model, metrics, manifest |
| `src/steps/phaseb_flaml_recipe.py` | 485 | — | Legacy per-recipe FLAML training | Config, recipe YAML, CSV | Model, metrics, manifest |
| `src/steps/aggregate_phaseb.py` | 501 | — | Legacy 2-recipe × 2-engine aggregator | 4 manifests + 4 models | Report, champion copy, stage signal |
| `src/steps/phasec_optuna_hpo.py` | 802 | S09 | Optuna HPO on Phase B champion algo | Config, FE CSV, Phase B manifest | `model.pkl`, `study.pkl`, trials CSV, Optuna HTML plots |
| `src/steps/aggregate_phasec.py` | 220 | S09z | Pass-through aggregator for HPO | HPO metrics JSON, model | Report, champion copy |
| `src/steps/final_evaluation.py` | 1338 | S10 | Cross-phase holdout evaluation + SHAP + AIM tournament + ledger merge | Config, dataset, 3 phase models | Final report, champion copy, SHAP JSON, merged ledger, AIM tournament, model coverage |
| `src/steps/s12_model_registration.py` | 341 | S12 | MLflow Model Registry registration | Champion manifest, model, config name | `registry_info.json` |
| `src/steps/stage1_ingestion_v4.py` | ~500 | — | Alternate ingestion (NOT wired into pipeline) | — | — |

### 2.3 Utility Modules

| File | Lines | Purpose |
|------|-------|---------|
| `src/utils/model_universe.py` | 416 | Canonical model lists for PyCaret/FLAML per task type; breakdown builders |
| `src/utils/candidate_ledger.py` | 510 | 51-column candidate tracking ledger; never crashes |
| `src/utils/aim_tournament.py` | 414 | Multi-metric ranking, Pareto frontier, utility scoring |
| `src/utils/variant_recommender.py` | 224 | Dataset-aware variant scoring (0-100) and diversity selection |
| `src/utils/variant_planner.py` | 532 | Adaptive 3-round planner with proxy pruning |
| `src/utils/variant_schema.py` | 285 | `VariantConfig` dataclass + YAML loader + task validation |
| `src/utils/variant_selector.py` | 125 | File-level variant selection (alphabetical/random) |
| `src/utils/bundle_gating.py` | 397 | Data-signal-driven bundle enable/disable |
| `src/utils/dataset_profiler.py` | 329 | `DatasetProfile` + `DatasetProfiler` — missing rate, imbalance, cardinality, domain hints |
| `src/utils/preprocessing_cache.py` | 192 | In-memory LRU cache for preprocessed DataFrames; never caches SMOTE |
| `src/utils/recipe_selector.py` | 300 | Tier-based recipe selection with fallback chains |
| `src/utils/recipe_converter.py` | 402 | V1 JSON → V3 YAML recipe conversion (offline) |
| `src/utils/stage_signals.py` | 187 | 24-field `StageSignal` dataclass + atomic JSON writer |
| `src/utils/stage_registry.py` | 236 | 16-stage registry with canonical IDs and ordering |
| `src/utils/azureml_metrics_logger.py` | 258 | MLflow-based logger; outputs/ is source of truth |
| `src/utils/mlflow_helper.py` | 36 | Thin wrappers for execution ID, parent/child runs |
| `src/utils/jsonl_logger.py` | 133 | JSONL alternative to MLflow for command jobs |
| `src/utils/data_validator.py` | 23 | Target column validation + high-cardinality filter |
| `src/utils/eda_generator.py` | 150 | Correlation heatmap + Sweetviz report |
| `src/utils/azure_helper.py` | 10 | `get_ml_client()` — creates MLClient |
| `src/variant_search/variant_search_engine.py` | 609 | Grid/random/progressive variant generation with constraint validation |

### 2.4 Component YAMLs (18 files)

All under `components/`. All use `azureml:mlops-v3-unified:23` environment and code path `../`.

| YAML | Component Name | Version | Display Name |
|------|---------------|---------|-------------|
| `stage0_data_validation.yml` | `v3_stage0_data_validation` | 1 | V3 Stage 0 - Data Quality Gate |
| `stage1_ingestion.yml` | `v3_stage1_ingestion` | 9 | V3 Stage 1 - Ingestion (Read-only) |
| `stage2_preparation.yml` | `stage2_preparation` | 7 | V3 Stage 2 - Preparation |
| `stage3_preprocessing.yml` | `stage3_preprocessing` | 5 | V3 Stage 3 - Preprocessing |
| `stage4_feature_engineering.yml` | `stage4_feature_engineering` | 6 | V3 Stage 4 - Engineering |
| `stage5_pycaret_train.yml` | `v3_stage5_pycaret_train` | 5 | V3 Stage 5a - Baseline PyCaret |
| `stage5_flaml_train.yml` | `v3_stage5_flaml_train` | 7 | V3 Stage 5b - Baseline FLAML |
| `stage5_timeseries_train.yml` | `v3_stage5_timeseries_train` | 1 | V3 Stage 5t - Time-Series Forecasting |
| `aggregate_baseline.yml` | `v3_aggregate_baseline` | 4 | V3 Stage 5z - Aggregate Baseline |
| `s06_phaseb_variant_runner.yml` | `v3_phaseb_variant_runner` | 7 | s06 - V3 Phase B - Variant Runner |
| `phaseb_pycaret_recipe.yml` | `v3_phaseb_pycaret_recipe` | 9 | V3 Phase B - PyCaret Recipe |
| `phaseb_flaml_recipe.yml` | `v3_phaseb_flaml_recipe` | 10 | V3 Phase B - FLAML Recipe |
| `aggregate_phaseb.yml` | `v3_aggregate_phaseb` | 5 | V3 Phase B - Aggregate |
| `aggregate_phaseb_dynamic.yml` | `v3_aggregate_phaseb_dynamic` | 1 | V3 Phase B - Dynamic Aggregate |
| `phasec_optuna_hpo.yml` | `v3_phasec_optuna_hpo` | 9 | s08 - V3 Phase C - Optuna HPO |
| `aggregate_phasec.yml` | `v3_aggregate_phasec` | 6 | s09 - V3 Phase C - Aggregate |
| `final_evaluation.yml` | `v3_final_evaluation` | 9 | s10 - V3 Final Evaluation |
| `s12_model_registration.yml` | `v3_s12_model_registration` | 1 | V3 Stage 12 - Model Registry |

### 2.5 Configuration Files

| File | Task Type | Dataset | Target Column | Key Differences |
|------|-----------|---------|---------------|-----------------|
| `configs/config_classification_telecom_churn_azureml.yml` | classification | telecom_churn | `churn` | Production, boruta, delimiter=`,` |
| `configs/config_classification_bank_marketing_azureml.yml` | classification | bank_marketing | `y` | Production, boruta, **delimiter=`;`** |
| `configs/config_regression_college_azureml.yml` | regression | college | `Grad.Rate` | Production, **mutual_info** (no target for boruta) |
| `configs/config_clustering_online_retail_azureml.yml` | clustering | online_retail | *(none)* | Engines: **pycaret only**, PCA threshold=50, **encoding=latin-1** |
| `configs/config_classification_telecom_churn_local.yml` | classification | telecom_churn | `churn` | Diagnostic, FLAML disabled, **local_path** |
| `configs/config_classification_telecom_churn_test_s06.yml` | classification | telecom_churn | `churn` | Test config, **max_variants=5**, hardcoded 3 recipes |

### 2.6 Recipe Library (457 files)

| Directory | Count | Generated By |
|-----------|-------|-------------|
| `configs/recipes/classification/variant_search/` | 210 | `VariantSearchEngine` grid_sample |
| `configs/recipes/classification/v1_generated/` | 44 | V1→V3 recipe converter |
| `configs/recipes/classification/` (root) | 3 | Manual |
| `configs/recipes/regression/variant_search/` | 80 | `VariantSearchEngine` grid_sample |
| `configs/recipes/regression/v1_generated/` | 43 | V1→V3 recipe converter |
| `configs/recipes/regression/` (root) | 4 | Manual |
| `configs/recipes/clustering/variant_search/` | 40 | `VariantSearchEngine` grid_sample |
| `configs/recipes/clustering/v1_generated/` | 25 | V1→V3 recipe converter |
| `configs/recipes/clustering/` (root) | 2 | Manual |
| `configs/recipes/` (root) | 3 | Manual |
| **Total** | **457** | |

### 2.7 Environments

| File | Purpose |
|------|---------|
| `environments/unified_conda.yml` | Conda spec: Python 3.10.14, 7 conda + 21 pip packages |
| `environments/azureml_unified_env.yml` | Azure ML env definition v23, base image `openmpi4.1.0-ubuntu20.04` |

### 2.8 Operational Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/extract_job_results.py` | 230 | Download job metadata + child steps + SMOTE/FLAML checks |
| `scripts/validate_aim_tournament.py` | 181 | 8-section validation of AIM tournament outputs |
| `scripts/validate_candidate_ledger.py` | 146 | 7-section validation of candidate ledger integrity |
| `scripts/monitor_pipeline.sh` | ~50 | `az ml job stream` with metric threshold checks |

### 2.9 Documents

| File | Purpose |
|------|---------|
| `docs/AIM_TOURNAMENT.md` | AIM tournament scoring methodology |
| `docs/LEDGER.md` | Candidate ledger specification |
| `docs/PHASE_B_VARIANT_RUNNER_ARCHITECTURE.md` | Phase B variant runner design |
| `docs/PRODUCTION_REPO_LAYOUT.md` | Production repository layout |
| `docs/VARIANT_SEARCH_GUIDE.md` | Variant search usage guide |
| `docs/PIPELINE_QUALITY_SKILL.md` | Pipeline quality skill reference |
| `docs/MLOPS-v3-blueprint.CSV` | Pipeline blueprint spreadsheet |

### 2.10 Tests

| File | Lines | Purpose |
|------|-------|---------|
| `tests/validate_critical_fixes.py` | 191 | 3 pre-submission tests: recipe diversity, FLAML artifact parsing, SMOTE application |

---

## 3. Data-Flow Trace

### Pipeline DAG

```
S01 → S02 → S03 → S04 → ┬─ S05a ─┐
                         ├─ S05b ─┤→ S05z → S06 → S09 → S09z → S10 → S12
                         └─ S05t ─┘
```

**Critical design decision**: S06 receives **S02's output** (prepared data), NOT S04's. This prevents double-preprocessing — S06 applies its own variant-specific preprocessing internally (`apply_variant_preprocessing` at s06 L280-600).

### Per-Stage Trace

#### Stage 1 — Ingestion (`stage1_ingestion.py`)

| | |
|---|---|
| **ENTERS** | Raw CSV from Azure ML datastore (`azureml://` URI), Config YAML |
| **KEY OPS** | `pd.read_csv()` with configurable `sep`; column type detection; missing value analysis; target distribution analysis; Pearson correlation matrix; Shapiro-Wilk normality tests; time-series auto-detection (confidence ≥ 0.60); intelligent recipe recommendations; Sweetviz HTML report |
| **EXITS** | Raw CSV unchanged (read-only passthrough) |
| **ARTIFACTS** | `eda_report.json`, `column_statistics.csv`, `data_quality_report.json`, `target_analysis.json`, `correlation_matrix.csv`, `recipe_recommendations.json`, `time_series_detection.json`, Sweetviz HTML, visualization PNGs |

#### Stage 2 — Preparation (`stage2_preparation.py`)

| | |
|---|---|
| **ENTERS** | Raw CSV from S01 |
| **KEY OPS** | Statistical tests (Shapiro-Wilk, KS normality, IQR outliers, Pearson); strategy-driven imputation (KNN/Iterative/ffill/group_median/median — **NO mean**, stakeholder requirement at L143); high-cardinality filtering (max_unique=100) |
| **EXITS** | Prepared CSV (imputed, high-cardinality columns dropped) |
| **ARTIFACTS** | `prep_report.json` (statistical tests + imputation summary) |

#### Stage 3 — Preprocessing (`stage3_preprocessing.py`)

| | |
|---|---|
| **ENTERS** | Prepared CSV from S02, optional recipe YAML |
| **KEY OPS** | Recipe-driven encoding (label/onehot/target/catboost); recipe-driven scaling (none/standard/robust/quantile/yeo_johnson/adaptive); binary preservation (one-hot columns NOT scaled); VIF multicollinearity detection (VIF>10 flagged); SMOTE **explicitly deferred** to Phase B (leakage prevention) |
| **EXITS** | All-numeric preprocessed CSV |
| **ARTIFACTS** | `prep3_report.json` (encoding/scaling summary, VIF report) |

#### Stage 4 — Feature Engineering (`stage4_feature_engineering.py`)

| | |
|---|---|
| **ENTERS** | Preprocessed CSV from S03, optional recipe YAML |
| **KEY OPS** | ID column detection (regex + cardinality ratio); feature selection (none/boruta with 3-level fallback/mutual_info/variance); PCA auto-trigger at >100 features (95% variance retention); zero-feature guard (raises `ValueError`); imbalance detection → `imbalance_metadata.json` |
| **EXITS** | Final feature-engineered CSV (used by S05a, S05b, S05t, S09) |
| **ARTIFACTS** | `fe_report.json`, `imbalance_metadata.json` |

#### Stage 5a — PyCaret Baseline (`stage5_pycaret_train.py`)

| | |
|---|---|
| **ENTERS** | FE CSV from S04, Config |
| **KEY OPS** | Imbalance detection + double-SMOTE guard (ratio ≥ 0.8 = skip `fix_imbalance`); adaptive folds (3 for >50K rows, 5 otherwise); `compare_models(sort="AUC"/"R2"/"Silhouette")` over MODEL_UNIVERSE; optimal F1 threshold tuning via grid search [0.10, 0.90] step=0.01; candidate ledger row |
| **EXITS** | PyCaret model pickle, metrics JSON, manifest JSON |
| **ARTIFACTS** | `model.pkl`, `threshold_info.json`, `stage5a_baseline_pycaret_leaderboard.csv`, `stage5a_baseline_pycaret_top10.json`, `s05a_model_breakdown.csv`, `s05a_candidates.csv`+`.parquet` |

#### Stage 5b — FLAML Baseline (`stage5_flaml_train.py`)

| | |
|---|---|
| **ENTERS** | FE CSV from S04, Config |
| **KEY OPS** | Honest holdout split (80/20, stratified for classification); `AutoML.fit(metric="roc_auc"/"r2")` with `estimator_list` from MODEL_UNIVERSE; time-budgeted with `+360s` cross_val_predict buffer; **skips clustering** (writes `.skipped` sentinel) |
| **EXITS** | FLAML model (joblib), metrics JSON, manifest JSON |
| **ARTIFACTS** | `model.pkl`, `stage5b_baseline_flaml_iterations.csv`, `s05b_model_breakdown.csv`, `s05b_candidates.csv`+`.parquet` |

#### Stage 5t — Timeseries Baseline (`stage5_timeseries_train.py`)

| | |
|---|---|
| **ENTERS** | FE CSV from S04, `time_series_detection.json` from S01 |
| **KEY OPS** | Signal gate: only runs if `is_time_series=True` (confidence ≥ 0.60); temporal 80/20 split (NO shuffle); fits 6 models (ARIMA, SARIMA, Holt-Winters, SES, Theta, Naive); evaluates MAE/RMSE/MAPE; minimum 30 data points required |
| **EXITS** | Best forecasting model (joblib), manifest JSON |
| **ARTIFACTS** | `model.pkl`, `model_breakdown_s05t.csv`, `stage5t_forecasting_summary.json`, `s05t_candidates.csv`+`.parquet` |

#### Stage 5z — Aggregate Baseline (`aggregate_baseline.py`)

| | |
|---|---|
| **ENTERS** | S05a manifest+model, S05b manifest+model |
| **KEY OPS** | Multi-fallback score extraction (6+ fallback paths per engine); champion selection (higher-is-better: AUC/R2/silhouette); handles skipped engines; recursive model copy to champion directory |
| **EXITS** | Phase A champion model, selection report JSON |
| **ARTIFACTS** | `{report_out}`, champion model files, `baseline_stage_signal.json`, `s05z_candidates.csv`+`.parquet` |

#### Stage 6 — Phase B Variant Runner (`s06_phaseb_variant_runner.py`)

| | |
|---|---|
| **ENTERS** | **S02 prepared CSV** (NOT S04), variant YAML paths, engine list, Config |
| **KEY OPS** | Holdout split (80/20, stratified); data fingerprinting (SHA-256); **3-round funnel**: Round 0 (transform-only feasibility — eliminates feature explosion >20K or 10x), Round 1 (SGD proxy on ≤5K rows — eliminates scores < threshold), Round 2 (full training); per-variant preprocessing via `apply_variant_preprocessing()` (18 imputation × 5 outlier × 3 encoding × 5 scaling × 4 imbalance × 3 feature selection); **SMOTE retrain** post model selection (label encoders saved on model via `_smote_label_encoders`); preprocessing cache (LRU, hash-based); nested MLflow runs per variant×engine; `preprocess_holdout_aligned()` for leakage-free holdout eval; deadline guards (5 per variant×engine); checkpoint manager for resume |
| **EXITS** | Champion model, leaderboard CSV, manifest JSON, holdout metrics |
| **ARTIFACTS** | `leaderboard.csv`, `champion_manifest.json`, `all_results.json`, `model.pkl`, `holdout_data.csv`, `holdout_metrics.json`, `phaseb_eval_data.csv`, `round0_feasibility_report.csv`, `round1_proxy_leaderboard.csv`, `elimination_report.json`, `resume_state.json`, `s06_candidates.csv`+`.parquet` |

#### Stage 9 — Phase C HPO (`phasec_optuna_hpo.py`)

| | |
|---|---|
| **ENTERS** | FE CSV from S04, Phase B champion manifest, Config |
| **KEY OPS** | Reads Phase B champion algorithm; recipe-aware preprocessing (replicates Phase B encoding+scaling); 6 algorithm-specific search spaces (XGBoost, LightGBM, CatBoost, RandomForest, LogisticRegression, Ridge); Optuna MedianPruner; clustering OOM guard (silhouette sample cap 10K, float32 downcast); 5-fold CV with `balanced_accuracy_score` for classification |
| **EXITS** | HPO-tuned model, study pickle, metrics JSON |
| **ARTIFACTS** | `model.pkl`, `label_encoder.pkl`, `study.pkl`, `phasec_optuna_trials.csv`, `phasec_optuna_best_params.json`, `phasec_optuna_optimization_history.html`, `phasec_optuna_param_importance.html`, `phasec_hpo_stage_signal.json`, `s08_candidates.csv`+`.parquet` |

#### Stage 9z — Aggregate Phase C (`aggregate_phasec.py`)

| | |
|---|---|
| **ENTERS** | HPO metrics JSON, optimized model |
| **KEY OPS** | Thin pass-through: copies report, copies model, validates, emits stage signal |
| **EXITS** | HPO champion model, report |
| **ARTIFACTS** | `phasec_aggregate_report.json`, `phasec_champion_summary.json`, `phasec_stage_signal.json`, `s09_candidates.csv`+`.parquet` |

#### Stage 10 — Final Evaluation (`final_evaluation.py`)

| | |
|---|---|
| **ENTERS** | FE CSV, baseline model (S05z), Phase B model (S06), Phase C model (S09z), Config |
| **KEY OPS** | Holdout split (80/20, stratified); loads models+encoders+thresholds from all 3 phases; evaluates via `eval_model()` with optimal threshold support for imbalanced binary classification; Phase B uses variant-preprocessed holdout (`phaseb_eval_data.csv`); feature-aligned evaluation (matches `feature_names_in_`); SHAP explainability (TreeExplainer → KernelExplainer fallback, top-20 features); cross-phase champion selection (max `primary_metric`); merges ALL upstream candidate ledger CSVs; AIM Tournament (rank-percentile → Pareto → utility); model coverage report; performance visualizations (3 PNGs) |
| **EXITS** | Final champion model, comprehensive report JSON |
| **ARTIFACTS** | Final report, champion model+encoder, `shap_feature_importance.json`, `all_stages_metrics.json`, 3 comparison PNGs, Sweetviz HTML, `final_phase_comparison.csv`, `variant_rankings.csv`, `final_champion_summary.json`, `all_candidates.csv`+`.parquet`+`_summary.json`+`_README.md`, AIM tournament CSVs, `model_coverage_*.csv`, `all_models_breakdown.csv`, `final_stage_signal.json`, `s10_candidates.csv`+`.parquet` |

#### Stage 12 — Model Registration (`s12_model_registration.py`)

| | |
|---|---|
| **ENTERS** | Champion manifest JSON, champion model directory, config name |
| **KEY OPS** | Validates manifest JSON (required fields); finds `.pkl` artifact; `mlflow.sklearn.log_model()` → `mlflow.register_model()`; adds metadata tags (task_type, algorithm, metrics, recipe); transitions to `"Staging"` stage (NOT Production — requires human approval); model named `{dataset}_{task}_mlops` |
| **EXITS** | Registry info JSON |
| **ARTIFACTS** | `registry_info.json` (model_name, version, stage, algorithm, metrics) |

---

## 4. Function-Level Breakdown

### 4.1 `pipelines/submit_pipeline.py` (755 lines)

| Function | Lines | Signature | Purpose |
|----------|-------|-----------|---------|
| `_acquire_lock()` | L27 | `() → bool` | File-based `.submit.lock`; returns False if lock held by another process |
| `_release_lock()` | L55 | `() → None` | Removes lock file |
| `_check_active_jobs()` | L60 | `(ml_client, experiment_name) → bool` | Queries Azure ML for running/queued jobs in same experiment |
| `_azure_from_local_config()` | L107 | `(config) → dict` | Extracts azureml.{sub,rg,ws,compute} from config dict |
| `derive_experiment_name()` | L124 | `(config) → str` | Returns `experiment_name` or `f"{dataset}_{task}_v3"` fallback |
| `derive_display_name()` | L135 | `(config, execution_id) → str` | Returns `f"{experiment}_{execution_id}"` |
| `filter_variants_by_imputation_preset()` | L157 | `(variant_paths, preset, task_type) → list` | Filters variants by imputation method based on `imputation_preset` config |
| `main()` | L185 | `() → None` | Full orchestration: parse 18 args, load config, validate schema, select variants (3 paths: legacy tier-based / intelligent scored / bundle gating), build pipeline, submit to Azure ML |

**3 Variant Selection Paths in `main()`:**
1. **Legacy tier-based** (L320-345): `recipe_selector.select_recipes_for_tier()` → tier + count
2. **Intelligent scored** (L350-420): `variant_selector.select_variants()` → scored + diversity
3. **Bundle gating** (L425-470): `bundle_gating.compute_data_signals()` → `select_enabled_bundles()` → `resolve_variant_paths()`

### 4.2 `pipelines/pipeline_builder.py` (310 lines)

| Item | Lines | Purpose |
|------|-------|---------|
| Component loading | L7–22 | Loads 14 component YAMLs via `load_component()` |
| `full_pipeline()` | L24–152 | Production `@dsl.pipeline` with 5 params, 19 outputs |
| `full_pipeline_v2()` | L155–310 | Phase 1 `@dsl.pipeline` with planner params (adds 5 params) |

### 4.3 `src/steps/stage0_data_validation.py` (379 lines)

| Function/Class | Lines | Purpose |
|----------------|-------|---------|
| `DataValidator.__init__()` | L42 | Stores config |
| `DataValidator.validate_schema()` | L50 | Column existence, type checks |
| `DataValidator.validate_row_count()` | L68 | Minimum row check |
| `DataValidator.validate_missing()` | L82 | Max missing percentage check |
| `DataValidator.validate_target()` | L98 | Target column class distribution |
| `DataValidator.validate_duplicates()` | L115 | Duplicate row detection |
| `DataValidator.validate_dtypes()` | L130 | Data type consistency |
| `DataValidator.validate_cardinality()` | L148 | High-cardinality column detection |
| `DataValidator.run_all()` | L165 | Runs all 8 validators, produces RED/YELLOW/GREEN verdict |
| `main()` | L210–379 | Argparse, config load, MLflow URI fix, DataValidator.run_all(), artifact writes |

### 4.4 `src/steps/stage1_ingestion.py` (1112 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `compute_column_types()` | L45 | Classifies columns as numeric/categorical/datetime/boolean |
| `generate_basic_statistics()` | L89 | Per-column mean/median/std/min/max/unique/missing |
| `generate_data_quality_report()` | L155 | Missing rate, duplicate rate, constant columns, zero-variance |
| `generate_target_analysis()` | L210 | Class distribution (classification), basic stats (regression) |
| `generate_correlation_analysis()` | L270 | Pearson correlation matrix, top-N correlated pairs |
| `generate_normality_tests()` | L340 | Shapiro-Wilk / KS tests per numeric column (sampled at 5000) |
| `detect_potential_issues()` | L400 | ID columns, high-cardinality, constant columns, leakage |
| `generate_comprehensive_eda()` | L451 | Orchestrates all above sub-functions into single EDA report |
| `detect_time_series()` | L586 | DateTime detection, frequency inference, confidence scoring (≥0.60) |
| `generate_intelligent_recipe_recommendations()` | L694 | Data-driven imputation/encoding/scaling/imbalance recommendations |
| `generate_visualization_suite()` | L800 | Matplotlib visualizations: distributions, correlations, target analysis |
| `_save_artifacts()` | L900 | Writes all artifact files atomically |
| `main()` | L950–1112 | Orchestrates S01: parse args, read CSV (sep=delimiter), generate EDA, detect time series, generate recommendations, save artifacts, passthrough dataset |

### 4.5 `src/steps/stage2_preparation.py` (351 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `perform_statistical_tests()` | L23–72 | Shapiro-Wilk/KS normality, IQR outlier detection, Pearson correlation |
| `prep_dataframe()` | L75–172 | Full imputation pipeline: KNN (n=5), Iterative (max_iter=10), ffill, group_median, median (default). Filters high-cardinality (>100 unique). **NO mean** — stakeholder requirement |
| `main()` | L175–351 | Parse args, load config, read CSV, `perform_statistical_tests()`, `prep_dataframe()`, write CSV + report |

### 4.6 `src/steps/stage3_preprocessing.py` (371 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `compute_vif()` | L21–43 | Variance Inflation Factor per feature. VIF > 10 = problematic |
| `preprocess()` | L46–192 | Full preprocessing: recipe-driven encoding, recipe-driven scaling, binary preservation (one-hot columns skipped), SMOTE deferred |
| `main()` | L195–371 | Parse args, load config+recipe, read CSV, `preprocess()`, `compute_vif()`, write CSV + report |

### 4.7 `src/steps/stage4_feature_engineering.py` (396 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `feature_engineer()` | L48–244 | ID column detection (regex `^(id|.*_id|index)$` + cardinality > 0.9), feature selection (none/boruta/mutual_info/variance), PCA at >100 features, zero-feature guard |
| `main()` | L247–396 | Parse args, load config+recipe, read CSV, `feature_engineer()`, write imbalance_metadata, write CSV + report |

**Boruta 3-level fallback** (inside `feature_engineer()`):
1. `confirmed_` features (default Boruta output)
2. `tentative_` features (if confirmed gives zero)
3. `feature_importances_` from fitted estimator (if tentative gives zero)
4. Final fallback: `mutual_info_classif`/`mutual_info_regression`

### 4.8 `src/steps/stage5_pycaret_train.py` (435 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `get_primary_metric()` | L33–35 | Returns `"AUC"`, `"R2"`, or `"Silhouette"` |
| `_optimal_threshold_f1()` | L38–46 | Grid-search [0.10, 0.90] step=0.01 to maximize F1 for given pos_label |
| `main()` | L49–435 | Config load, MLflow URI fix, dataset read, task-branched training (classification: fix_imbalance + compare_models + threshold tuning; regression: compare_models; clustering: create_model("kmeans") + sklearn silhouette/davies_bouldin), model save, MLflow log, candidate ledger |

### 4.9 `src/steps/stage5_flaml_train.py` (364 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `main()` | L23–364 | Config load, MLflow URI fix, clustering early-exit (writes skip artifacts), holdout split (80/20, stratified), `AutoML.fit(metric="roc_auc"/"r2")`, explicit holdout evaluation, model save (joblib), iterations table from `config_history`, model breakdown, MLflow log, candidate ledger |

### 4.10 `src/steps/stage5_timeseries_train.py` (388 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `_fit_arima()` | L50–53 | ARIMA wrapper, order=(1,1,1) |
| `_fit_sarima()` | L56–60 | SARIMAX wrapper, seasonal_order=(1,1,1,12) |
| `_fit_exponential_smoothing()` | L63–68 | Holt-Winters (additive trend + seasonal) |
| `_fit_ses()` | L71–74 | Simple Exponential Smoothing |
| `_fit_theta()` | L77–80 | Theta model |
| `_fit_naive()` | L83–90 | Seasonal naive forecaster (inner class `_NaiveResult`) |
| `_evaluate_forecast()` | L101–108 | Computes MAE, RMSE, MAPE |
| `MODEL_HANDLERS` | L93–100 | Dict mapping model names → handler functions |
| `main()` | L114–388 | Signal check (`time_series_detection.json`), dataset prep (datetime parse, sort, freq), min-30-rows check, temporal 80/20 split, model loop, champion save, MLflow log, candidate ledger |

### 4.11 `src/steps/aggregate_baseline.py` (391 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `validate_and_log_outputs()` | L14–36 | Validates output files exist, returns validation dict |
| `get_primary_metric()` | L39–55 | Returns `"AUC"`, `"R2"`, or `"silhouette_score"` |
| `load_json()` | L58–62 | Safe JSON loader |
| `select_champion()` | L65–131 | Multi-fallback champion selection: PyCaret (leaderboard → best_metric → metrics_dict → top_model_metrics), FLAML (primary_metric → best_metric), clustering (silhouette_score directly) |
| `main()` | L134–391 | Load manifests, select champion, validate + copy model folder, emit stage signal, write candidate ledger |

### 4.12 `src/steps/s06_phaseb_variant_runner.py` (3139 lines)

**Group A — Setup / Dataclasses (L1-182)**

| Item | Lines | Purpose |
|------|-------|---------|
| `VariantResult` dataclass | L56–68 | Stable contract: 10 fields (variant_id, engine, algorithm, metrics, runtime, timed_out, failed, failure_reason, leakage_risk, n_features) |
| `ChampionManifest` dataclass | L71–88 | Locked schema: 15 fields (variant_id/path, engine, algorithm, metric name/value, preprocessing_config, feature_engineering_config, data_fingerprint, code_version, timestamp, leakage_risk, task_type) |
| `CheckpointManager` class | L95–128 | Resume state: `_load()`, `is_completed()`, `mark_completed()`, `_flush()`, `get_progress()` |
| `set_deterministic_seed()` | L135–140 | Seeds `random` and `numpy` |
| `compute_data_fingerprint()` | L147–160 | SHA-256 of first 1000 rows CSV |
| `get_code_version()` | L163–170 | Git commit SHA[:8] or timestamp fallback |
| `atomic_write()` | L173–182 | Write to .tmp then `os.replace` |

**Group B — Defensive Guards (L188-275)**

| Function | Lines | Purpose |
|----------|-------|---------|
| `validate_metrics()` | L188–199 | Guards NaN/inf → replaces with 0.0 |
| `check_leakage_risk()` | L202–210 | Returns risk level; upgrades to "medium" for target encoding |
| `deadline_guard()` | L213–217 | Returns True if wall-clock deadline exceeded |
| `get_primary_metric()` | L220–230 | Returns `"AUC"`, `"R2"`, or `"silhouette_score"` |
| `get_metric_columns_for_task()` | L233–245 | Task-specific metric column lists |
| `is_lower_better()` | L248–252 | Checks if metric is error-type (rmse, mae, etc.) |
| `safe_float()` | L255–264 | Coerce to float, return `-inf` for None/NaN |
| `get_result_score()` | L267–278 | Unified scoring — negates lower-better metrics |

**Group C — Preprocessing (L280-1060)**

| Function | Lines | Purpose |
|----------|-------|---------|
| `apply_variant_preprocessing()` | L280–600 | Full pipeline: 18 imputation methods, 5 outlier handlers, 3 encoding methods (+fallback), 5 scaling methods (+fallback), 4 imbalance methods (SMOTE/ADASYN/SMOTEENN/SMOTETomek — deferred by default via `apply_smote=False`), 3 feature selection methods |
| `preprocess_holdout_aligned()` | L700–1060 | Training-aligned holdout preprocessing: fits ALL transformers on training data, transforms holdout using training statistics. **Never applies SMOTE to holdout.** |

**Group D — Signal Rounds (L1065-1380)**

| Function | Lines | Purpose |
|----------|-------|---------|
| `run_round0_feasibility()` | L1065–1130 | Transform-only feasibility check. Detects zero features, feature explosion (>20K or 10x). Returns `(df_or_None, report_dict)` |
| `run_round1_proxy()` | L1135–1250 | Cheap SGD proxy on ≤5000 rows. Classification: `SGDClassifier(log_loss, early_stopping)` → balanced_accuracy. Regression: `SGDRegressor` → R2. Clustering: `KMeans` → silhouette. Status `"warning"` for suspiciously low scores |

**Group E — Engine Runners (L1380-1770)**

| Function | Lines | Purpose |
|----------|-------|---------|
| `train_pycaret_variant()` | L1380–1620 | PyCaret with `preprocess=False` (data already processed), 3-fold CV, MODEL_UNIVERSE enforcement, adaptive budget (600s for >50K rows). **SMOTE retrain** (L1480-1560): label-encodes remaining non-numeric, applies imblearn SMOTE, refits best model on resampled data, saves `_smote_label_encoders` dict on model object |
| `train_flaml_variant()` | L1623–1770 | FLAML with min budget floor (120s), skips clustering, time-aware CV (≥300s→3-fold, ≥120s→2-fold, <120s→skip), tracks trials from `config_history` |

**Group F — MLflow Integration (L1770-1960)**

| Function | Lines | Purpose |
|----------|-------|---------|
| `run_variant_with_nested_mlflow()` | L1770–1960 | One variant×engine: nested MLflow run, reuses cached preprocessed data, 5 deadline checkpoints, logs individual FLAML trials as nested child runs |

**Group G — Main Orchestration (L1960-3139)**

17 argparse arguments (L1970-2000). Main flow:
1. Parse args, resolve variant paths (L2001-2040)
2. Load config, wire FLAML min budget (L2050-2068)
3. Deterministic seeding + checkpoint init (L2068-2077)
4. Load dataset + holdout split 80/20 stratified (L2081-2102)
5. Data fingerprinting + preprocessing cache (L2108-2118)
6. Adaptive planner mode if enabled (L2121-2200)
7. MLflow setup (L2205-2232)
8. **Variant loop** (L2250-2500): load → validate → Round 0 → cache check → Round 1 → per-engine training → champion tracking
9. **Post-processing** (L2501-2700): filter usable, build leaderboard, select champion, build ChampionManifest
10. Candidate ledger + signal artifacts (L2700-2880)
11. Champion model save (L2885-2915)
12. Holdout evaluation via `preprocess_holdout_aligned()` (L2920-3050)
13. Final MLflow logging + cleanup (L3050-3139)

### 4.13 `src/steps/s07_phase2_pipeline_attribution.py` (715 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| 4 dataclasses | L30–92 | `PipelineRecord`, `DimensionAttribution`, `ChampionComparison`, `ConfidenceSignals` |
| `load_phase1_artifacts()` | L100–121 | Loads all_results.json, leaderboard.csv, champion_manifest.json |
| `extract_model_family()` | L128–154 | Maps algorithm → family (xgboost, lightgbm, random_forest, linear, etc.) |
| `normalize_results_to_records()` | L157–210 | Converts raw results to `PipelineRecord` with dimension parsing |
| `compute_dimension_attribution()` | L218–263 | Per-choice attribution: avg deviation from median, normalized impact_score [-1,1] |
| `compute_all_attributions()` | L266–294 | Runs attribution across 7 dimensions: imputation, encoding, scaling, imbalance, feature_selection, engine, model_family |
| `compare_champion_vs_runnerup()` | L301–354 | Identifies changed/unchanged decisions between #1 and #2 |
| `compute_confidence_signals()` | L361–410 | CV, top-3 gap, sensitivity analysis → confidence level (high/medium/low) |
| `write_*()` | L417–582 | 5 writer functions for JSON/CSV/Markdown outputs |
| `main()` | L590–715 | Orchestrates full attribution analysis |

### 4.14 `src/steps/phasec_optuna_hpo.py` (802 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `_safe_disable_autolog()` | L27–39 | Disables MLflow autolog, URI fix, local registry |
| `objective()` (nested in main) | ~L345–590 | Optuna trial: per-algorithm search spaces (XGB: n_estimators/max_depth/learning_rate/subsample/colsample. LGB: +num_leaves. CatBoost: iterations/depth/l2_leaf_reg. RF: min_samples_split/leaf/max_features. LR/Ridge: C log-scale) |
| `main()` | L42–802 | Config load, clustering KMeans/DBSCAN branch, Phase B manifest read, algorithm mapping, recipe-aware preprocessing (replicates Phase B encoding+scaling), Optuna study with MedianPruner, final model train, artifact export, MLflow log, stage signal, candidate ledger |

### 4.15 `src/steps/final_evaluation.py` (1338 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `_safe_disable_autolog()` | L40–52 | Standard MLflow safety |
| `collect_all_stage_metrics()` | L55–116 | Scans MLflow experiment, categorizes runs by stage |
| `generate_performance_visualizations()` | L119–183 | 3 PNGs: baseline comparison, Phase B recipes, phase-level |
| `generate_comprehensive_sweetviz_report()` | L186–213 | Sweetviz HTML (sampled at 10K rows) |
| `validate_and_log_outputs()` | L216–237 | Standard output validation |
| `validate_input_paths()` | L240–276 | Validates config, dataset, 3 model paths |
| `load_model_and_encoder()` | L284–326 | Loads model.pkl + label_encoder.pkl + threshold_info.json |
| `get_primary_metric()` | L329–341 | Returns `"balanced_accuracy"`, `"r2"`, or `"silhouette_score"` |
| `eval_model()` | L344–530 | Full evaluation: classification (accuracy, balanced_accuracy, precision, recall, F1, ROC AUC with threshold support), regression (R2, MAE, RMSE, MSE), clustering (silhouette, davies_bouldin, calinski_harabasz) |
| `log_metrics_to_mlflow()` | L533–587 | Logs all phase metrics |
| `validate_champion_output()` | L590–613 | Validates champion model.pkl non-empty |
| `main()` | L616–1338 | Full orchestration: validate → load → evaluate 3 phases → SHAP → champion select → ledger merge → AIM tournament → model coverage → visualizations → stage signal |

### 4.16 `src/steps/s12_model_registration.py` (341 lines)

| Class/Function | Lines | Purpose |
|----------------|-------|---------|
| `ModelRegistry.__init__()` | L50–62 | MlflowClient init, azureml://→https://, dataset name extraction |
| `ModelRegistry._extract_dataset_name()` | L64–71 | Config name → dataset name (e.g., `config_classification_telecom_churn_azureml` → `telecom_churn`) |
| `ModelRegistry.register_champion_model()` | L73–166 | Loads pkl, logs sklearn model, registers version, transitions to Staging |
| `ModelRegistry._find_model_artifact()` | L168–185 | Finds .pkl/.joblib in directory |
| `ModelRegistry._add_model_metadata()` | L187–231 | Sets tags on model version (task_type, algorithm, metrics, recipe) |
| `_write_skip_output()` | L234–243 | Placeholder JSON on skip |
| `main()` | L246–341 | 4 validation gates → `ModelRegistry` → register → save registry_info. **4 skip paths:** manifest not found, invalid JSON, missing algorithm, empty model |

### 4.17 Key Utility Modules

#### `model_universe.py` (416 lines)

| Item | Lines | Purpose |
|------|-------|---------|
| `MODEL_UNIVERSE` | L31–114 | Dict: 14 classification_pycaret, 9 classification_flaml, 23 regression_pycaret, 7 regression_flaml, 8 clustering_pycaret, 6 forecasting_statsmodels → **67 total model entries** |
| `FAST_MODELS` | L125–144 | Curated subset excluding slow O(n²) models |
| `get_model_list()` | L203–212 | Canonical list for task+engine |
| `build_pycaret_breakdown()` | L258–299 | PyCaret `pull()` → breakdown DataFrame |
| `build_flaml_breakdown()` | L302–361 | FLAML `config_history` → breakdown DataFrame |

#### `candidate_ledger.py` (510 lines)

| Item | Lines | Purpose |
|------|-------|---------|
| `ALL_COLUMNS` | L80 | 51 canonical columns across 6 groups: IDENTITY(9), INPUT(4), CLASSIFICATION_METRICS(6), REGRESSION_METRICS(4), CLUSTERING_METRICS(3), OUTPUT(15), SIGNAL(7), PROVENANCE(2), TOURNAMENT(14) |
| `make_row()` | L99–131 | Creates populated ledger row |
| `normalize_metrics()` | L134–164 | Maps mixed-case metric names to canonical column names |
| `merge_ledgers()` | L262–295 | Merges multiple stage CSVs into consolidated file |
| `build_summary()` | L298–334 | JSON summary from merged ledger |
| `build_readme_md()` | L337–402 | Human-readable Markdown |

#### `aim_tournament.py` (414 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `METRIC_CATALOG` | L30–53 | Per-task metric definitions with direction and weight |
| `compute_all_metrics()` | L68–121 | Full metric vector via sklearn |
| `add_rank_columns()` | L155–167 | Dense ranking per metric |
| `find_pareto_frontier()` | L174–203 | True non-dominated sorting |
| `add_utility_scores()` | L210–242 | Rank-percentile utility scoring |
| `run_aim_tournament()` | L323–414 | Main entry: rank → Pareto → utility → top-K → report |

#### `variant_recommender.py` (224 lines)

| Method | Lines | Purpose |
|--------|-------|---------|
| `score_variant_relevance()` | L30–111 | Scores 0-100 across 5 dimensions (imputation 25pt, encoding 20pt, scaling 15pt, imbalance 25pt, feature_selection 15pt) + leakage penalty + alignment bonus. **Clamped** to [0, 100] after bonuses (R6 fix). |
| `select_top_variants()` | L113–142 | Scores all, filters by threshold, applies diversity boost |
| `_apply_diversity_boost()` | L144–176 | Greedy diversity via unique (imputation, encoding, feature_selection, imbalance) tuples; +5 bonus |

#### `variant_planner.py` (532 lines)

| Function | Lines | Purpose |
|----------|-------|---------|
| `EdaPriors` dataclass | L23–42 | Data priors: missing_rate, imbalance_ratio, high_cardinality, outlier_prevalence, skewness, n_rows, n_features. `from_eda_report()` classmethod |
| `score_variant_relevance()` | L86–215 | Core scorer: base=50, 5 dimensions (more granular than recommender — recognizes ~15 imputation methods) |
| `diverse_sample()` | L227–262 | Greedy furthest-first Hamming distance sampling |
| `build_variant_plan()` | L310–411 | Main entry: score → Round 0 filter → diverse sample → proxy prune → shortlist + budget. **Zero-result fallback** (R5 fix) |

#### `dataset_profiler.py` (329 lines)

| Class/Method | Lines | Purpose |
|--------------|-------|---------|
| `DatasetProfile` | L15–189 | Rich profile: n_rows, n_features, n_numeric, n_categorical, target_type, missing_rate, imbalance_ratio, outlier_prevalence, high_cardinality, correlations, domain_hints. `recommend_preprocessing_strategies()` with memoization |
| `DatasetProfiler.profile_dataset()` | L203–298 | Full analysis: target type, missing rate, class imbalance, IQR outliers, cardinality (>100), correlation structure, multicollinearity (>0.85), domain hints |
| `DatasetProfiler._detect_domain_hints()` | L300–329 | Heuristic: finance/time_series/academic/healthcare column name keywords |

---

## 5. Variant System Deep Dive

### 5.1 What Is a Variant?

A **variant** (also called a **recipe**) is a YAML file that specifies a complete preprocessing + feature engineering pipeline configuration. It encodes exactly 6 decisions:

1. **Imputation method** — how to handle missing values (18 options in s06)
2. **Encoding method** — how to convert categorical columns to numeric (label/onehot/target)
3. **Scaling method** — how to normalize feature magnitudes (none/standard/robust/minmax/yeo_johnson/quantile)
4. **Imbalance handling** — how to address class imbalance (none/smote/adasyn/smoteenn/smotetomek)
5. **Outlier handling** — how to address extreme values (none/iqr_removal/iqr_capping/zscore/winsorize/isolation_forest)
6. **Feature selection** — how to reduce dimensionality (none/correlation/variance/mutual_info/boruta/rfe)

Each combination produces a different "view" of the dataset, which may or may not be better for a given ML algorithm. Phase B's job is to find which view works best.

### 5.2 Variant YAML Structure

```yaml
# configs/recipes/classification/variant_search/variant_01ace0cb3ddd.yml
recipe_name: 01ace0cb3ddd
version: '1.0'
description: 'Pipeline Variant: mean+onehot+none+correlation'
task_type: classification

stage3_preprocessing:
  imputation:
    method: mean
    # Optional: n_neighbors (KNN), max_iter (Iterative), fill_value (constant)
  encoding:
    categorical_method: onehot    # label | onehot | target
  scaling:
    method: none                  # none | standard | robust | minmax | yeo_johnson | quantile
  imbalance_handling:
    method: none                  # none | smote | adasyn | smoteenn | smotetomek

stage4_feature_engineering:
  feature_selection:
    method: correlation           # none | correlation | variance | mutual_info
    threshold: 0.85               # Correlation threshold for feature selection

variant_metadata:
  variant_id: 01ace0cb3ddd        # SHA1[:12] deterministic hash of config
  leakage_risk: none              # none | low | medium | high
  estimated_runtime_sec: 30       # Heuristic runtime estimate
  generation_mode: grid_sample    # grid_sample | random_sample | progressive | locked
```

### 5.3 The 457-Recipe Library

The library was generated by `src/variant_search/variant_search_engine.py` using constrained grid sampling:

| Dimension | Options | Count |
|-----------|---------|-------|
| **Imputation** | mean, median, drop, knn, iterative, mode, forward_fill, backward_fill, interpolate_linear, constant, zero_fill, trimmed_mean, winsorized_mean, random_sample, numeric_mean_cat_mode, numeric_median_cat_mode | 16 |
| **Encoding** | label, onehot, target, (catboost) | 4 |
| **Scaling** | none, standard, robust, minmax, yeo_johnson, quantile | 6 |
| **Imbalance** | none, smote, adasyn, smoteenn, smotetomek | 5 |
| **Outlier** | none, iqr_removal, iqr_capping, zscore, winsorize, isolation_forest | 6 |
| **Feature Selection** | none, correlation, variance, mutual_info, boruta, rfe, select_k_best | 7 |

Full grid = 16 × 4 × 6 × 5 × 6 × 7 = **80,640** possible combinations. The engine generates a constrained sample after applying task-type-specific rules:

- **Classification**: SMOTE/ADASYN allowed. 210 variant_search + 44 v1_generated + 3 manual = **257**
- **Regression**: SMOTE/ADASYN blocked. 80 variant_search + 43 v1_generated + 4 manual = **127**
- **Clustering**: SMOTE blocked, target encoding blocked. 40 variant_search + 25 v1_generated + 2 manual = **67**
- **Cross-task root recipes**: 3, **Grand total**: **457** (verified on disk)

### 5.4 Leakage Risk Scoring

Defined in `variant_search_engine.py` L411-431 and `s06_phaseb_variant_runner.py` L202-210:

| Risk Level | Trigger | Implication |
|------------|---------|-------------|
| **CRITICAL** | — | Blocked at generation time |
| **HIGH** | Target encoding or CatBoost encoding | Uses target variable during transformation — requires careful CV |
| **MEDIUM** | SMOTE/ADASYN | Creates synthetic samples from target distribution |
| **LOW** | RFE or Boruta feature selection | Uses target for selection (but not transformation) |
| **NONE** | All other combinations | No target leakage risk |

### 5.5 Intelligent Variant Selection (Three Paths)

**Path 1 — Scored Selection** (production default):
`submit_pipeline.py` L350-420 → `variant_selector.select_variants()` → `variant_recommender.VariantRecommender.score_variant_relevance()`:
1. `DatasetProfiler.profile_dataset()` analyzes the dataset
2. `DatasetProfile.recommend_preprocessing_strategies()` returns data-driven recommendations
3. Each variant scored 0-100 across 5 dimensions (imputation 25pt, encoding 20pt, scaling 15pt, imbalance 25pt, feature_selection 15pt)
4. Leakage penalty subtracted; alignment bonus added
5. Score clamped to [0, 100] (R6 fix)
6. Diversity boost ensures unique (imputation, encoding, feature_selection, imbalance) tuples
7. Top 20 selected (configurable via `phases.phase_b.max_variants`)

**Path 2 — Adaptive Planner** (if `--enable_planner`):
`s06_phaseb_variant_runner.py` L2121-2200 → `variant_planner.build_variant_plan()`:
1. Builds `EdaPriors` from actual dataset statistics
2. Scores ALL variants via `score_variant_relevance()` (base=50, more granular than recommender)
3. Round 0 filtering (feasibility check)
4. Diverse sampling via `diverse_sample()` (greedy furthest-first Hamming)
5. Round 1 proxy pruning below threshold
6. Returns `VariantPlan` with shortlist + budget allocation
7. **Zero-result fallback**: if pruning eliminates all, returns top-3 by score (R5 fix)

**Path 3 — Bundle Gating** (if bundles configured):
`submit_pipeline.py` L425-470 → `bundle_gating.py`:
1. `compute_data_signals()` computes ~20 statistical signals from the dataset
2. `select_enabled_bundles()` evaluates gating rules (operator dispatch: <, >, ==, etc.)
3. `resolve_variant_paths()` collects + deduplicates variant paths from enabled bundles

### 5.6 The 3-Round Funnel (s06 core architecture)

```
All Selected Variants (e.g., 20)
        │
  Round 0: Feasibility (transform-only)
  ├── Feature explosion? (>20K or 10x) → ELIMINATE
  ├── Zero features after selection? → ELIMINATE
  └── Transform succeeds? → PASS
        │
  Round 1: Proxy (SGD on ≤5K rows, ~10s each)
  ├── Classification: SGDClassifier → balanced_accuracy < threshold → ELIMINATE
  ├── Regression: SGDRegressor → R2 < threshold → ELIMINATE
  └── Clustering: KMeans → silhouette < threshold → ELIMINATE
        │
  Round 2: Full Training (PyCaret + FLAML, ~300-600s each)
  ├── Per-engine training with MODEL_UNIVERSE enforcement
  ├── Per-variant × per-engine nested MLflow runs
  └── Champion tracking (best score overall)
        │
  Post-Processing: Leaderboard + Champion Selection
```

**Preprocessing cache** (`preprocessing_cache.py`): Hash-based LRU cache. Same variant preprocessing config across different engines → cache hit. `compute_key()` excludes imbalance handling from hash (target-dependent). Never caches SMOTE/ADASYN results. Returns DataFrame **copies** to prevent mutation.

**Checkpoint manager**: Each variant×engine completion immediately flushed to `resume_state.json`. On step restart, completed combinations are skipped. Enables resume after Azure ML spot-instance preemption.

### 5.7 SMOTE Retrain Logic

Inside `train_pycaret_variant()` at s06 L1480-1560:

1. Check variant's `imbalance_handling.method` (smote/adasyn/smoteenn/smotetomek)
2. **During preprocessing**: `apply_smote=False` — SMOTE is deferred to prevent data leakage in CV
3. **After PyCaret selects best model**: SMOTE is applied to full training data
4. Any remaining non-numeric columns are label-encoded
5. imblearn `SMOTE()`/`ADASYN()`/`SMOTEENN()`/`SMOTETomek()` applied
6. Best model refitted on SMOTE-resampled data
7. Label encoders saved as `model._smote_label_encoders` dict on model object (R2 fix)
8. Metrics logged: `smote_retrained=True`, `smote_rows_after=N`

### 5.8 Holdout Evaluation in s06

At s06 L2920-3050:

1. `preprocess_holdout_aligned(df_train_raw, df_holdout_raw, variant, target_column)` fits all transformers on training data, applies to holdout
2. **Never applies SMOTE to holdout** (holdout must reflect real distribution)
3. Aligns holdout columns to model's `feature_names_in_`
4. Computes holdout metrics
5. Saves `phaseb_eval_data.csv` (variant-preprocessed holdout for S10 to reuse)
6. Updates `champion_manifest.json` with holdout metrics

---

## 6. Artifact Map

### 6.1 Complete Artifact Index

Every artifact written by the pipeline, organized by stage:

#### Stage 0 — Data Validation
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| `validation_results.json` | JSON | `stage0_data_validation.py` | Human review (gating decision) |
| validated dataset (passthrough) | CSV | S00 | S01 |

#### Stage 1 — Ingestion
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| `eda_report.json` | JSON | S01 `generate_comprehensive_eda()` | Human review, recipe recommendations |
| `column_statistics.csv` | CSV | S01 | Human review |
| `data_quality_report.json` | JSON | S01 | Human review |
| `target_analysis.json` | JSON | S01 | Human review |
| `correlation_matrix.csv` | CSV | S01 | Human review |
| `recipe_recommendations.json` | JSON | S01 `generate_intelligent_recipe_recommendations()` | S02 (imputation strategy = `from_stage1`) |
| `time_series_detection.json` | JSON | S01 `detect_time_series()` | **S05t** (signal gate: only runs if `is_time_series=True`) |
| Sweetviz HTML report | HTML | S01 | Human review |
| Visualization PNGs | PNG | S01 | Human review |
| dataset passthrough (raw CSV) | CSV | S01 | S02 |

#### Stage 2 — Preparation
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| prepared CSV | CSV | S02 `prep_dataframe()` | S03, **S06** (Phase B gets S02 data!) |
| `prep_report.json` | JSON | S02 | Human review, S03 (VIF comparison) |

#### Stage 3 — Preprocessing
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| preprocessed CSV (all-numeric) | CSV | S03 `preprocess()` | S04 |
| `prep3_report.json` | JSON | S03 | Human review |

#### Stage 4 — Feature Engineering
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| feature-engineered CSV | CSV | S04 `feature_engineer()` | S05a, S05b, S05t, **S09** (Phase C HPO) |
| `imbalance_metadata.json` | JSON | S04 | S05a (double-SMOTE guard) |
| `fe_report.json` | JSON | S04 | Human review |

#### Stage 5a — PyCaret Baseline
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| `model.pkl` | PyCaret pickle | S05a | S05z → S10 |
| `threshold_info.json` | JSON | S05a `_optimal_threshold_f1()` | **S10** (optimal threshold for holdout eval) |
| `stage5a_baseline_pycaret_leaderboard.csv` | CSV | S05a | Human review, S10 |
| `stage5a_baseline_pycaret_top10.json` | JSON | S05a | Human review |
| `s05a_model_breakdown.csv` | CSV | S05a `build_pycaret_breakdown()` | S10 `all_models_breakdown.csv` merge |
| `s05a_candidates.csv` + `.parquet` | CSV/Parquet | S05a `write_stage_table()` | S10 `merge_ledgers()` |
| manifest JSON | JSON | S05a | S05z `select_champion()` |
| metrics JSON | JSON | S05a | S05z |

#### Stage 5b — FLAML Baseline
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| `model.pkl` | joblib | S05b | S05z → S10 |
| `.skipped` sentinel | text | S05b (clustering only) | S05z (handles skipped engine) |
| `stage5b_baseline_flaml_iterations.csv` | CSV | S05b | Human review |
| `s05b_model_breakdown.csv` | CSV | S05b `build_flaml_breakdown()` | S10 merge |
| `s05b_candidates.csv` + `.parquet` | CSV/Parquet | S05b | S10 `merge_ledgers()` |
| manifest JSON, metrics JSON | JSON | S05b | S05z |

#### Stage 5t — Timeseries Baseline
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| `model.pkl` | joblib | S05t | (not consumed in current pipeline DAG) |
| `.skipped` sentinel | text | S05t (if not time-series) | S05z |
| `model_breakdown_s05t.csv` | CSV | S05t | S10 merge |
| `s05t_candidates.csv` + `.parquet` | CSV/Parquet | S05t | S10 `merge_ledgers()` |

#### Stage 5z — Aggregate Baseline
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| Phase A champion model (recursive copy) | folder | S05z | **S10** (baseline_model) |
| selection report JSON | JSON | S05z | Human review |
| `baseline_stage_signal.json` | JSON | S05z `write_stage_signal()` | Human review |
| `s05z_candidates.csv` + `.parquet` | CSV/Parquet | S05z | S10 `merge_ledgers()` |

#### Stage 6 — Phase B Variant Runner
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| `leaderboard.csv` | CSV | S06 (atomic write) | **S10** (variant rankings), S07 |
| `champion_manifest.json` | JSON | S06 `ChampionManifest` | **S09** (algorithm for HPO), **S10**, **S12** |
| `all_results.json` | JSON | S06 `VariantResult` list | **S07** (attribution), **S10** (variant rankings) |
| `model.pkl` | joblib | S06 (atomic write) | **S10** (phaseb_model) |
| `holdout_data.csv` | CSV | S06 | (internal holdout reference) |
| `holdout_metrics.json` | JSON | S06 | Human review, S10 |
| `phaseb_eval_data.csv` | CSV | S06 `preprocess_holdout_aligned()` | **S10** (variant-preprocessed holdout for Phase B model) |
| `round0_feasibility_report.csv` | CSV | S06 | Human review |
| `round1_proxy_leaderboard.csv` | CSV | S06 | Human review |
| `elimination_report.json` | JSON | S06 | Human review, S07 |
| `resume_state.json` | JSON | S06 `CheckpointManager` | S06 (on restart) |
| `variant_plan.json` | JSON | S06 (if planner enabled) | Human review |
| `s06_candidates.csv` + `.parquet` | CSV/Parquet | S06 | S10 `merge_ledgers()` |

#### Stage 7 — Phase 2 Attribution
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| `pipeline_attribution.json` | JSON | S07 | Human review |
| `decision_impact_table.csv` | CSV | S07 | Human review |
| `winner_vs_runnerup.json` | JSON | S07 | Human review |
| `pipeline_confidence_report.json` | JSON | S07 | Human review |
| `phase2_summary.md` | Markdown | S07 | Human review |

#### Stage 9 — Phase C HPO
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| `model.pkl` | joblib | S09 | S09z → S10 |
| `label_encoder.pkl` | joblib | S09 | S09z → S10 |
| `study.pkl` | pickle | S09 | Human review |
| `phasec_optuna_trials.csv` | CSV | S09 | Human review |
| `phasec_optuna_best_params.json` | JSON | S09 | Human review |
| `phasec_optuna_optimization_history.html` | HTML | S09 (Optuna) | Human review |
| `phasec_optuna_param_importance.html` | HTML | S09 (Optuna) | Human review |
| `phasec_hpo_stage_signal.json` | JSON | S09 | Human review |
| `s08_candidates.csv` + `.parquet` | CSV/Parquet | S09 | S10 `merge_ledgers()` |

#### Stage 9z — Aggregate Phase C
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| Phase C champion model (recursive copy) | folder | S09z | **S10** (phasec_model) |
| `phasec_aggregate_report.json` | JSON | S09z | Human review |
| `phasec_champion_summary.json` | JSON | S09z | Human review |
| `s09_candidates.csv` + `.parquet` | CSV/Parquet | S09z | S10 `merge_ledgers()` |

#### Stage 10 — Final Evaluation
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| Final champion model | folder | S10 | **S12** |
| Final report JSON | JSON | S10 | Human review, **S12** (manifest) |
| `shap_feature_importance.json` | JSON | S10 | Human review |
| `all_stages_metrics.json` | JSON | S10 `collect_all_stage_metrics()` | Human review |
| `baseline_models_comparison.png` | PNG | S10 | Human review |
| `phaseb_recipes_comparison.png` | PNG | S10 | Human review |
| `phase_comparison.png` | PNG | S10 | Human review |
| `final_dataset_sweetviz_report.html` | HTML | S10 | Human review |
| `final_phase_comparison.csv` | CSV | S10 | Human review |
| `variant_rankings.csv` | CSV | S10 | Human review |
| `final_champion_summary.json` | JSON | S10 | Human review |
| `all_candidates.csv` + `.parquet` | CSV/Parquet | S10 `merge_ledgers()` | **AIM tournament**, validation scripts |
| `all_candidates_summary.json` | JSON | S10 `build_summary()` | Human review |
| `all_candidates_README.md` | Markdown | S10 `build_readme_md()` | Human review |
| AIM tournament files | CSV/JSON | S10 `run_aim_tournament()` | Human review |
| `model_coverage_*.csv` | CSV | S10 | Human review |
| `all_models_breakdown.csv` | CSV | S10 | Human review |
| `final_stage_signal.json` | JSON | S10 | Human review |
| `s10_candidates.csv` + `.parquet` | CSV/Parquet | S10 | (terminal stage) |

#### Stage 12 — Model Registration
| Artifact | Format | Writer | Consumer |
|----------|--------|--------|----------|
| `registry_info.json` | JSON | S12 `ModelRegistry` | Human review / CI/CD |
| MLflow Model Registry entry | MLflow | S12 | Production deployment |

### 6.2 Key Artifact Deep Dives

#### Deep Dive 1: `champion_manifest.json` (s06)

Written by `ChampionManifest` dataclass (s06 L71-88). This is the **most critical hand-off artifact** — consumed by S09 (algorithm selection), S10 (champion comparison), and S12 (model registration).

```json
{
  "variant_id": "01ace0cb3ddd",
  "variant_path": "configs/recipes/classification/variant_search/variant_01ace0cb3ddd.yml",
  "engine": "pycaret",
  "algorithm": "xgboost",
  "primary_metric_name": "AUC",
  "primary_metric_value": 0.9234,
  "preprocessing_config": {
    "imputation": {"method": "mean"},
    "encoding": {"categorical_method": "onehot"},
    "scaling": {"method": "none"},
    "imbalance_handling": {"method": "none"}
  },
  "feature_engineering_config": {
    "feature_selection": {"method": "correlation", "threshold": 0.85}
  },
  "data_fingerprint": "sha256:abc123...",
  "code_version": "a1b2c3d4",
  "timestamp": "2025-03-29T12:00:00Z",
  "leakage_risk": "none",
  "task_type": "classification"
}
```

**Consumers and key field mappings:**
- S09 reads `algorithm` to select HPO search space
- S10 reads `primary_metric_value` for cross-phase comparison. Fallback keys: `variant_path`, `primary_metric_value` (R4 fix)
- S12 reads `algorithm`, `task_type` for model naming and tags. Normalizes `task` → `task_type`, derives phase from `selection.key` (R4 fix)

#### Deep Dive 2: `all_candidates.csv` (s10)

Written by `merge_ledgers()` at s10 L880+. The **merged candidate ledger** — the single source of truth for every model ever tried across all stages.

**51 canonical columns** (from `candidate_ledger.py` L80):
- **Identity (9):** dataset_id, task_type, preset, pipeline_version, stage, step_name, engine, candidate_id, run_id, timestamp_utc
- **Input (4):** recipe_name, recipe_hash, params_json, pipeline_dims_json
- **Classification metrics (6):** accuracy, roc_auc, f1, precision, recall, logloss
- **Regression metrics (4):** r2, rmse, mae, mse
- **Clustering metrics (3):** silhouette, davies_bouldin, calinski_harabasz
- **Output (15):** primary_metric_name, primary_metric_value, and per-metric values
- **Signal (7):** candidate_rank, delta_vs_baseline_best, is_stage_best, is_final_champion, status, failure_reason, compute_time_sec
- **Provenance (2):** source_path, artifacts_json
- **Tournament (14):** rank_*, utility_score, pareto_optimal (added by AIM tournament)

**Source stage CSVs merged:** `s05a_candidates.csv`, `s05b_candidates.csv`, `s05z_candidates.csv`, `s06_candidates.csv`, `s08_candidates.csv`, `s09_candidates.csv`, `s10_candidates.csv`

#### Deep Dive 3: `leaderboard.csv` (s06)

Written atomically at s06 L2580+. Contains one row per variant×engine combination that produced usable results.

**Columns:** variant_id, engine, algorithm, primary_metric (AUC/R2/silhouette), + all task-specific metrics, runtime_sec, timed_out, failed, leakage_risk, n_features, score (unified higher-is-better)

**Filtering logic:** Only includes results where `metric > 0.01` and `not timed_out`. If all results are filtered out, falls back to all valid (non-failed) results.

#### Deep Dive 4: `StageSignal` (emitted by all aggregate steps)

24-field dataclass (`stage_signals.py` L42-95):
```json
{
  "stage_name": "aggregate_baseline",
  "stage_id": "S05z",
  "task_type": "classification",
  "candidates_in": 2,
  "candidates_out": 1,
  "best_score": 0.92,
  "best_metric_name": "AUC",
  "delta_vs_baseline": null,
  "failure_rate": 0.0,
  "compute_time_sec": 45.2,
  "recommendation": "proceed",
  "recommendation_reason": "Champion model score 0.92 exceeds minimum threshold"
}
```

Written atomically (temp + `os.replace`). Never throws on write failure.

#### Deep Dive 5: `threshold_info.json` (s05a)

Written by s05a L227-234 for classification tasks only:
```json
{
  "optimal_threshold": 0.42,
  "best_f1_at_threshold": 0.87,
  "default_threshold": 0.50,
  "pos_label": 1,
  "method": "grid_search_f1",
  "search_range": [0.10, 0.90],
  "search_step": 0.01
}
```

**Consumer:** S10 `load_model_and_encoder()` at L284-326 loads this file and applies optimal threshold during holdout evaluation, improving F1/recall on imbalanced data.

---

## 7. Config as Control Panel

### 7.1 Top-Level Keys

| Key | Type | Required | Purpose |
|-----|------|----------|---------|
| `experiment_name` | string | Yes | Azure ML experiment name. Determines MLflow experiment grouping |
| `preset` | enum | Yes | `production` or `diagnostic`. Controls pipeline behavior |
| `task_type` | enum | Yes | `classification`, `regression`, `clustering`, or `forecasting` |

### 7.2 Dataset Block

| Key | Type | Required | Purpose |
|-----|------|----------|---------|
| `dataset.name` | string | Yes | Human-readable dataset identifier |
| `dataset.target_column` | string | Required for classification/regression | Column to predict. Validated by `config_schema.py` cross-field check |
| `dataset.blob_path` | string | Yes (Azure) | Path within Azure ML datastore |
| `dataset.datastore_name` | string | Yes (Azure) | Azure ML datastore name (e.g., `mlops_blob`) |
| `dataset.local_path` | string | Yes (local) | Local CSV path (diagnostic preset only) |
| `dataset.delimiter` | string | No (default `,`) | CSV delimiter. **Critical for bank_marketing** (uses `;`) |
| `dataset.encoding` | string | No | File encoding (e.g., `latin-1` for online_retail) |

### 7.3 Stage Blocks

#### `stage1` — Ingestion Controls

| Key | Type | Default | Purpose |
|-----|------|---------|---------|
| `min_rows` | int | 1000 | Minimum dataset rows (below = RED gate) |
| `max_missing_pct` | int | 50 | Maximum missing value percentage |
| `classification_min_samples_per_class` | int | 30 | Minimum samples per class (classification) |
| `regression_target_min_variance` | float | 0.01 | Minimum target variance (regression) |
| `generate_sweetviz` | bool | true | Generate Sweetviz HTML EDA report |
| `eda_sample_size` | int | 10000 | Max rows for EDA sampling |

#### `stage2` — Preparation Controls

| Key | Type | Purpose |
|-----|------|---------|
| `imputation_strategy` | string | `from_stage1` = use Stage 1 recommendations; or explicit method name |
| `statistical_tests_enabled` | bool | Enable Shapiro-Wilk, KS, IQR tests |

#### `stage3` — Preprocessing Controls

| Key | Type | Purpose |
|-----|------|---------|
| `adaptive_scaling` | bool | Auto-select scaling based on data distribution |
| `multicollinearity_check` | bool | Enable VIF computation |

#### `stage4` — Feature Engineering Controls

| Key | Type | Purpose |
|-----|------|---------|
| `selection_method` | enum | `boruta` (classification), `mutual_info` (regression-friendly), `variance` (clustering — no target needed) |
| `apply_pca_threshold` | int | Feature count triggering PCA (100 for classification/regression, 50 for clustering) |
| `pca_variance_retained` | float | PCA variance retention (0.95) |
| `imbalance_detection` | bool | Generate `imbalance_metadata.json` |

### 7.4 Azure ML Block

| Key | Type | Purpose |
|-----|------|---------|
| `azureml.subscription_id` | string | Azure subscription |
| `azureml.resource_group` | string | Azure resource group |
| `azureml.workspace_name` | string | Azure ML workspace |
| `azureml.compute_target` | string | Compute cluster name |
| `azureml.environment_name_preprocessing` | string | Azure ML env for preprocessing steps |
| `azureml.environment_name_training` | string | Azure ML env for training steps |

### 7.5 Phase B Block (Intelligent)

| Key | Type | Default | Purpose |
|-----|------|---------|---------|
| `phases.phase_b.enable_profiling` | bool | true | Run `DatasetProfiler` for variant scoring |
| `phases.phase_b.library_dir` | string | — | Variant recipe directory (task-specific) |
| `phases.phase_b.max_variants` | int | 20 | Maximum variants to select |
| `phases.phase_b.selection_strategy` | string | `scored` | `scored` (intelligent) or `alphabetical` |
| `phases.phase_b.min_relevance_score` | float | 30.0 | Minimum variant score threshold |
| `phases.phase_b.diversity_boost` | bool | true | Enable diversity in variant selection |
| `phases.phase_b.runtime_budget_sec` | int | 180 | Per-variant runtime budget |
| `phases.phase_b.time_budget_per_variant` | int | 600 | s06 training time per variant |
| `phases.phase_b.engines` | list | `[pycaret, flaml]` | ML engines to use. **Clustering: `[pycaret]` only** |
| `phases.phase_b.imputation_preset` | string | `auto` | Filter variants by imputation method |

#### Planner Sub-Block

| Key | Type | Default | Purpose |
|-----|------|---------|---------|
| `phases.phase_b.planner.enabled` | bool | false | Enable adaptive planner in s06 |
| `phases.phase_b.planner.round1_max_variants` | int | 40 | Max variants for Round 1 proxy |
| `phases.phase_b.planner.round2_max_variants` | int | 10 | Max variants for Round 2 full training |
| `phases.phase_b.planner.proxy_prune_threshold` | float | 0.50 | Proxy score below which variants are eliminated |
| `phases.phase_b.planner.cache_enabled` | bool | true | Enable preprocessing cache |

### 7.6 Phase B Block (Legacy)

| Key | Type | Purpose |
|-----|------|---------|
| `phases.phase_b_recipes.library` | string | `variant_search` or `v1_generated` |
| `phases.phase_b_recipes.tier` | string | `progressive`, `balanced_performance`, etc. |
| `phases.phase_b_recipes.max_recipes` | int | Recipe count |
| `phases.phase_b_recipes.runtime_budget_sec` | int | Per-recipe budget |

### 7.7 Phase C Block

| Key | Type | Default | Purpose |
|-----|------|---------|---------|
| `phases.phase_c_hpo.n_trials` | int | 50 | Optuna trial count |
| `phases.phase_c_hpo.timeout_seconds` | int | 3600 | Optuna overall timeout |

### 7.8 Task-Type Configuration Differences

| Config Key | Classification | Regression | Clustering |
|------------|---------------|------------|------------|
| `target_column` | Required | Required | **Omitted** |
| `stage4.selection_method` | `boruta` | `mutual_info` | `variance` |
| `stage4.apply_pca_threshold` | 100 | 100 | **50** |
| `stage4.imbalance_detection` | true | true | **false** |
| `phases.phase_b.engines` | `[pycaret, flaml]` | `[pycaret, flaml]` | **`[pycaret]`** |
| `phases.phase_b.library_dir` | `classification/variant_search` | `regression/variant_search` | `clustering/v1_generated` |
| SMOTE allowed? | Yes | **No** | **No** |
| Target encoding allowed? | Yes | Yes | **No** |

---

## 8. MLflow Tracking Hierarchy

### 8.1 Run Nesting Structure

```
Pipeline Run (Azure ML managed)
│
├── S01: v3_stage1_ingestion
│   └── (no MLflow runs — file-based artifacts only)
│
├── S02: stage2_preparation
│   └── (no MLflow runs — file-based artifacts only)
│
├── S03: stage3_preprocessing
│   └── (no MLflow runs — file-based artifacts only)
│
├── S04: stage4_feature_engineering
│   └── (no MLflow runs — file-based artifacts only)
│
├── S05a: v3_stage5_pycaret_train
│   └── MLflow Run: "s05a_baseline_pycaret" (via create_metrics_logger)
│       ├── Params: task_type, target_column, best_model_name, n_models_compared
│       └── Metrics: AUC, accuracy, f1, precision, recall, kappa, MCC (classification)
│                    R2, RMSE, MAE (regression)
│                    silhouette_score, davies_bouldin_score (clustering)
│
├── S05b: v3_stage5_flaml_train
│   └── MLflow Run: "s05b_baseline_flaml" (via create_metrics_logger)
│       ├── Params: task_type, target_column, metric_optimized, time_budget
│       └── Metrics: (same as S05a per task type)
│
├── S05t: v3_stage5_timeseries_train
│   └── MLflow Run: "s05t_baseline_timeseries"
│       ├── Params: task_type, n_models_tested, best_model_name
│       └── Metrics: MAE, RMSE, MAPE
│
├── S05z: v3_aggregate_baseline
│   └── (stage signal only — no MLflow run)
│
├── S06: v3_phaseb_variant_runner
│   └── Step Run (uses existing Azure ML run context if available)
│       ├── Nested Run: "variant_{id}_{engine}"  (per variant×engine)
│       │   ├── Params: imputation, encoding, scaling, imbalance, feature_selection,
│       │   │           leakage_risk, variant_id, engine
│       │   ├── Metrics: (task-specific metrics per model)
│       │   └── Nested Run: "flaml_trial_{i}" (FLAML only, per trial)
│       │       └── Metrics: val_loss, estimator, time
│       ├── Summary Params: n_variants, n_engines, cache_hit_rate, planner_mode
│       └── Artifacts: leaderboard.csv, champion_manifest.json
│
├── S07: (NOT wired into Azure ML pipeline DAG — runs offline)
│
├── S09: v3_phasec_optuna_hpo
│   └── MLflow Run: "s08_phasec_hpo" (via create_metrics_logger)
│       ├── Params: algorithm, n_trials, best_params (JSON)
│       └── Metrics: best_score, n_trials_completed
│
├── S09z: v3_aggregate_phasec
│   └── (stage signal only)
│
├── S10: v3_final_evaluation
│   └── MLflow Run: "s10_final_evaluation" (via create_metrics_logger)
│       ├── Params: task_type, champion_phase, champion_algorithm
│       ├── Metrics: baseline_{metric}, phase_b_{metric}, phase_c_{metric},
│       │           champion_{metric}, delta_vs_baseline
│       └── Artifacts: final report, comparison CSVs, SHAP JSON
│
└── S12: v3_s12_model_registration
    └── MLflow Run: model logging + registry
        ├── Model: sklearn model logged via mlflow.sklearn.log_model()
        ├── Tags: task_type, algorithm, primary_metric, recipe, phase
        └── Stage: "Staging" (auto-promoted; human promotes to Production)
```

### 8.2 MLflow URI Fix Pattern

Every step that uses MLflow includes this fix (e.g., s05a L56-59):

```python
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
if mlflow_uri.startswith("azureml://"):
    https_uri = mlflow_uri.replace("azureml://", "https://")
    mlflow.set_tracking_uri(https_uri)
```

**Reason:** Azure ML sets `MLFLOW_TRACKING_URI=azureml://eastus2.api.azureml.ms/mlflow/...` which MLflow's model registry does not support. Converting to `https://` resolves `Model registry functionality is unavailable` errors.

Additionally, steps set a local model registry to prevent remote registry calls during training:
```python
os.environ["MLFLOW_REGISTRY_URI"] = "file:///tmp/mlflow-registry"
```

### 8.3 Exact Metric Names Logged

| Stage | Classification | Regression | Clustering |
|-------|---------------|------------|------------|
| S05a | `AUC`, `accuracy`, `f1`, `precision`, `recall`, `kappa`, `MCC`, `pr_auc` | `R2`, `RMSE`, `MAE` | `silhouette_score`, `davies_bouldin_score` |
| S05b | `accuracy`, `f1`, `precision`, `recall`, `kappa`, `MCC`, `roc_auc`, `pr_auc` | `r2`, `rmse`, `mae`, `mse` | (skipped) |
| S06 | Per variant: `AUC`/`balanced_accuracy`, `accuracy`, `f1`, `precision`, `recall` | `R2`, `RMSE`, `MAE` | `silhouette_score`, `davies_bouldin_score` |
| S09 | `best_score` (balanced_accuracy_score) | `best_score` (r2_score) | `best_score` (silhouette_score) |
| S10 | `baseline_balanced_accuracy`, `phase_b_balanced_accuracy`, `phase_c_balanced_accuracy`, `champion_balanced_accuracy` | `baseline_r2`, `phase_b_r2`, `phase_c_r2`, `champion_r2` | `baseline_silhouette`, `phase_b_silhouette`, `phase_c_silhouette`, `champion_silhouette` |

---

## 9. Operational Guide

### 9.1 Submitting a Pipeline

#### Standard Submission (All 4 Task Configs)

```bash
# Classification — Telecom Churn
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --wait --stop_compute

# Classification — Bank Marketing
python pipelines/submit_pipeline.py \
  --config configs/config_classification_bank_marketing_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --wait --stop_compute

# Regression — College
python pipelines/submit_pipeline.py \
  --config configs/config_regression_college_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --wait --stop_compute

# Clustering — Online Retail
python pipelines/submit_pipeline.py \
  --config configs/config_clustering_online_retail_azureml.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --wait --stop_compute
```

#### Forced Re-Submission

```bash
python pipelines/submit_pipeline.py ... --force
```

The `--force` flag bypasses both duplicate-submission guards:
1. **Lock file** (`.submit.lock`): Prevents concurrent submissions (uses `fcntl.flock`)
2. **Active job check**: Queries Azure ML for running/queued jobs in the same experiment; aborts if found

#### Variant Selection CLI Args

| Arg | Purpose |
|-----|---------|
| `--variant_selection_method` | Override variant selection: `scored`, `alphabetical`, `random` |
| `--variant_bundle` | Path to pre-selected variant bundle YAML |
| `--max_variants` | Override max variants from config |
| `--enable_planner` | Enable 3-round adaptive planner in S06 |

#### Other CLI Args

| Arg | Purpose |
|-----|---------|
| `--wait` | Block until pipeline completes (polls every 60s) |
| `--stop_compute` | Stop compute cluster after completion (cost control) |
| `--experiment_name` | Override experiment name from config |
| `--tags` | Add custom tags to the Azure ML job |
| `--display_name` | Override default job display name |
| `--use_v2_pipeline` | Use `full_pipeline_v2()` (Phase 1 with planner) |

### 9.2 Monitoring a Running Pipeline

#### In Azure ML Studio
1. Navigate to **Experiments** → select experiment name
2. Click on the running job
3. View the **DAG** to see which steps have completed, are running, or failed
4. Click on individual steps to view:
   - **Outputs + logs**: stdout/stderr, system logs
   - **Metrics**: MLflow-logged metrics (appears in S05-S12)
   - **Child runs**: For S06, each variant×engine is a nested run

#### Via CLI (`monitor_pipeline.sh`)

```bash
# Note: monitor_pipeline.sh hardcodes RESOURCE_GROUP=mlops-accelerator-rg
# For current workspace (resource_group=mvpv1), modify the script or use az ml directly
az ml job show --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query "status"
```

#### Via extract_job_results.py

```bash
python scripts/extract_job_results.py \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --job_name <job_name> \
  --output_dir ./job_results/
```

Downloads all step outputs to local filesystem for inspection.

### 9.3 Reading Results

After pipeline completion, the key results are in the **S10** (final_evaluation) step outputs:

```
job_results/<job_name>/final_evaluation/
├── final_champion_summary.json    # Champion info: algorithm, scores, phase
├── all_candidates.csv             # Complete candidate ledger (51 columns)
├── all_candidates_summary.json    # Condensed summary
├── all_stages_metrics.json        # Metrics from every stage
├── final_phase_comparison.csv     # A vs B vs C comparison table
├── variant_rankings.csv           # Ranked variant results from Phase B
├── all_models_breakdown.csv       # Every model tried across all stages
├── *_comparison.png               # Visual comparisons
└── shap_feature_importance.json   # Feature importance via SHAP
```

The **champion model** is in:
```
job_results/<job_name>/final_evaluation/champion_model/
└── model.pkl
```

The **registered model** is viewable in Azure ML Studio → **Models** tab.

### 9.4 Common Failure Patterns and Fixes

#### 1. "Model registry functionality is unavailable"
- **Cause**: `MLFLOW_TRACKING_URI` set to `azureml://` (not supported by MLflow registry)
- **Fix**: Already applied in all training steps — URI converted to `https://`
- **Verify**: Check step logs for `🔗 MLflow tracking URI converted to HTTPS`

#### 2. "No path specified" or "Input path is None"
- **Cause**: Azure ML didn't mount the datastore properly, or config path is wrong
- **Fix**: Verify `dataset.blob_path` in config matches actual datastore path

#### 3. Out of Memory on S06
- **Cause**: Too many variants with large datasets
- **Fix**: Reduce `phases.phase_b.max_variants` or increase compute SKU

#### 4. FLAML "hard kill" (step terminated without results)
- **Cause**: FLAML training exceeds Azure ML step timeout
- **Fix**: FLAML training uses `time_budget = max_seconds - 360` buffer (already implemented)

#### 5. PyCaret SMOTE crash on regression
- **Cause**: PyCaret regression doesn't support `fix_imbalance`
- **Fix**: SMOTE guard in `s06_phaseb_variant_runner.py` skips SMOTE for non-classification

#### 6. Zero features after feature engineering
- **Cause**: Boruta or correlation filter removed all features
- **Fix**: S04 has zero-feature guard — falls back to top 2 by importance

#### 7. NFS delay on submission (~12 minutes)
- **Cause**: `ml_client.jobs.create_or_update()` slow on NFS-mounted file systems
- **Info**: This is expected behavior, not a failure. Wait for completion.

#### 8. Duplicate submission blocked
- **Cause**: Lock file exists from previous run, or active job found
- **Fix**: Use `--force` flag, or wait for active job to complete

#### 9. Phase B champion worse than baseline
- **Cause**: Double-preprocessing (PyCaret re-encoding already-encoded data)
- **Fix**: S06 produces `phaseb_eval_data.csv` with aligned preprocessing; use `preprocess=False` in PyCaret

#### 10. "KeyError: target_column" in clustering
- **Cause**: Clustering has no target column but code assumes one
- **Fix**: All steps check `task_type == "clustering"` and skip target-dependent logic

### 9.5 Validation Scripts

```bash
# Validate AIM tournament logic (unit test)
python scripts/validate_aim_tournament.py

# Validate candidate ledger schema (unit test)
python scripts/validate_candidate_ledger.py

# Validate critical fixes across codebase
python tests/validate_critical_fixes.py
```

### 9.6 Cost Control

1. **`--stop_compute`**: Stops the compute cluster after job completion (saves ~$2-5/hr depending on SKU)
2. **`--wait`**: Required with `--stop_compute` (can't stop compute without waiting for job completion)
3. **Variant pruning**: S06's 3-round funnel eliminates 60-80% of variants in Round 0 feasibility check, saving compute
4. **Preprocessing cache**: S06 caches encoded/scaled data with LRU, avoiding redundant computation
5. **Deadline guards**: Every training loop in S06 checks remaining time and terminates gracefully

### 9.7 Environment Management

The unified environment `azureml:mlops-v3-unified:23` is used by all components:

```yaml
# environments/unified_conda.yml
name: mlops-v3-unified
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10.14
  - pip:
    - pycaret==3.3.2
    - flaml==2.2.0
    - scikit-learn==1.4.2
    - optuna>=3.4
    - xgboost>=2.0
    - lightgbm>=4.0
    - catboost>=1.2
    - mlflow
    - shap
    - sweetviz
    - azureml-mlflow
    - azureml-fsspec          # Required for azureml:// URI reading
    - boruta                  # Feature selection
```

If you need a new version, update `environments/azureml_unified_env.yml` and register via:
```bash
az ml environment create --file environments/azureml_unified_env.yml \
  --resource-group mvpv1 --workspace-name mlops-accelerator
```

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| **AIM Tournament** | Adaptive Integrated Metric tournament. Multi-metric ranking system (`aim_tournament.py`) that computes rank-percentile utility scores across balanced_accuracy, f1, recall, precision, AUC, etc. Uses Pareto frontier to identify non-dominated candidates. |
| **Baseline** | Phase A result. The best model from `compare_models()` (PyCaret) or AutoML (FLAML) using default preprocessing (Stages 0-4). |
| **Boruta** | Feature selection algorithm used in S04. Three-level fallback: (1) standard Boruta, (2) Boruta with reduced iterations, (3) correlation-based fallback. |
| **Bundle** | Pre-selected collection of variant recipes stored in `configs/variant_bundles/`. Selected based on dataset gating signals (`bundle_gating.py`). |
| **Candidate** | Any model trained across any stage/phase. Tracked in the candidate ledger with 51 columns. |
| **Candidate Ledger** | CSV file tracking every model candidate across all stages. 51 canonical columns defined in `candidate_ledger.py`. Merged in S10 from per-stage CSVs. |
| **Champion** | Best model within a phase. Phase A champion (S05z), Phase B champion (S06), Phase C champion (S09z), Final champion (S10). |
| **Champion Manifest** | JSON artifact from S06 describing the Phase B champion: algorithm, metrics, preprocessing config, variant path. Critical hand-off to S09 and S12. |
| **Checkpoint** | Resume state saved by S06's `CheckpointManager`. Enables resuming variant processing after failure. |
| **Component** | Azure ML component definition (YAML in `components/`). Defines inputs, outputs, command, and environment for each pipeline step. |
| **Diagnostic Preset** | Config `preset: diagnostic` — uses smaller subsets, fewer trials to speed up testing. |
| **Double-Preprocessing** | Anti-pattern where PyCaret re-applies encoding/scaling to already-preprocessed data. Prevented by `preprocess=False` flag and S06's aligned holdout. |
| **Engine** | ML framework used for training: `pycaret` or `flaml`. Some configs support both, clustering uses `pycaret` only. |
| **Execution ID** | UUID generated by `submit_pipeline.py` and propagated to all steps via the config. Used for tracing. |
| **FLAML** | Fast and Lightweight AutoML. Used in S05b (baseline) and as an engine option in S06 (Phase B). |
| **Funnel (3-Round)** | S06's variant processing pipeline: Round 0 (feasibility — DataFrame shape/NaN check → eliminates ~20%), Round 1 (SGD proxy on 5000 rows, 30s budget → eliminates ~50%), Round 2 (full training with deadline guard). |
| **Gating Signal** | Statistical dataset property used by `bundle_gating.py` to select variant bundles. ~20 signals including missing_rate, imbalance_ratio, n_features, etc. |
| **HPO** | Hyperparameter Optimization. Phase C uses Optuna with algorithm-specific search spaces (S09). |
| **Holdout** | S06 reserves 20% of data for holdout evaluation. S05b uses honest 80/20 split pre-FLAML. S10 does final holdout with stratified split. |
| **Imbalance Metadata** | JSON from S04 recording class distribution stats. Used by S05a to guard against double-SMOTE. |
| **Leakage Risk** | Score (0.0 = none, 0.5 = moderate, 1.0 = high) assigned by `variant_search_engine.py` to flag variants that might cause data leakage (e.g., target encoding + no regularization). |
| **Ledger** | See Candidate Ledger. |
| **Model Universe** | Complete set of ML models defined in `model_universe.py`. 67 total entries across classification (31), regression (23), clustering (13). Organized by complexity tier (fast, medium, slow). |
| **Nested Run** | MLflow child run under a parent step run. S06 creates one nested run per variant×engine combination. |
| **Pareto Frontier** | Set of non-dominated candidates where no single candidate is better in ALL metrics. Computed in `aim_tournament.py`. |
| **Phase A** | Baseline training (S05a + S05b + S05t → S05z). Uses default preprocessing, all MODEL_UNIVERSE models. |
| **Phase B** | Variant exploration (S06). Tests N recipe variants × engines. Each variant applies different preprocessing. |
| **Phase C** | Hyperparameter optimization (S09 → S09z). Takes Phase B champion's algorithm, optimizes hyperparameters via Optuna. |
| **Planner** | `variant_planner.py`. Adaptive 3-round planner that pre-selects variants based on EDA priors and dataset size. |
| **PyCaret** | Low-code AutoML library. Used in S05a (baseline) and as an engine option in S06/S09. |
| **Recipe** | YAML file defining a preprocessing pipeline configuration. 457 recipes in `configs/recipes/`. Each specifies imputation, encoding, scaling, imbalance handling, and feature selection methods. |
| **Recipe Converter** | `recipe_converter.py`. Converts legacy V1 JSON recipe format to V3 YAML format. |
| **SMOTE** | Synthetic Minority Over-sampling Technique. Applied for imbalanced classification in S06. Uses `_smote_label_encoders` for safe SMOTE on encoded data. Classification only. |
| **Stage Signal** | 24-field JSON emitted by aggregate steps. Records candidates_in/out, best_score, recommendation ("proceed"/"stop"). |
| **Sweetviz** | Automated EDA report generator. S01 optionally generates Sweetviz HTML. S10 generates final dataset report. |
| **Tier** | Categorization of recipe sets by computational cost/coverage: `progressive`, `balanced_performance`, `top5_essentials`, `exhaustive`, etc. |
| **Variant** | A specific preprocessing configuration (recipe) to test in Phase B. Each variant specifies imputation, encoding, scaling, imbalance, and feature selection strategies. |
| **Variant Search Space** | Full combinatorial grid of preprocessing options: 18 imputation × 5 outlier × 3 encoding × 5 scaling × 4 imbalance × 3 feature selection = 80,640 possible variants. |
| **VIF** | Variance Inflation Factor. Computed in S03 to detect multicollinearity. Features with VIF > 10 are flagged. |

---

*Document generated from actual code inspection of mlops-solution-accelerator-v3/. All line numbers, function names, and artifact schemas verified against source files as of document generation date.*
