# MLOps V3 Pipeline — Session History

> Complete chronicle of all development sessions for the V3 pipeline.
> Each session documents objectives, changes, outcomes, and open issues carried forward.

---

## Session 1: 12-Item Production Audit

**Date**: Initial V3 pipeline stabilization
**Branch**: `v3-production`
**Focus**: End-to-end pipeline audit — 12 critical fixes

### Objectives
- Audit the complete 13-step pipeline for production readiness
- Fix all blocking issues preventing end-to-end execution

### Changes Made
| # | Issue | Fix | Files |
|---|-------|-----|-------|
| T01 | SMOTE applied to regression/clustering | Task-type guard — skip SMOTE for non-classification | `stage3_preprocessing.py` |
| T02 | AUC metric fails for multiclass | Use `roc_auc_score(multi_class='ovr', average='weighted')` | `final_evaluation.py` |
| T03 | Holdout split not stratified | Add `stratify=y` for classification | `final_evaluation.py` |
| T04 | Feature selection drops target | Exclude target column from SelectKBest | `stage4_feature_engineering.py` |
| T05 | PyCaret uses accuracy instead of balanced_accuracy | Set `sort='balanced_accuracy'` in compare_models | `stage5_pycaret_train.py` |
| T06 | FLAML time budget too short | Add +360s deadline buffer | `stage5_flaml_train.py` |
| T07 | ID columns leak into model | Drop nameOrig, nameDest, transactionID pre-encoding | `stage3_preprocessing.py` |
| T08 | Phase B runs all 457 recipes blindly | Integrate variant_recommender for top-20 selection | `s06_phaseb_variant_runner.py` |
| T09 | Phase C search space model-agnostic | Add per-model search spaces (LightGBM, XGBoost, RF, etc.) | `phasec_optuna_hpo.py` |
| T10 | MLflow azureml:// URI breaks model registry | Convert to https:// in all step scripts | Multiple step scripts |
| T11 | Config validation too loose | Strict schema validation in config_schema.py | `config_schema.py` |
| T12 | No execution_id propagation | Add execution_id to all step scripts for tracing | Multiple step scripts |

### Outcome
- 12/12 fixes applied
- Pipeline executes end-to-end for classification (telecom_churn)
- Commit on `v3-production` branch

---

## Session 2: 15-Dataset Scaling

**Date**: Post-audit scaling
**Branch**: `v3-production`
**Focus**: Scale from 1 dataset to 15 across all 3 task types

### Objectives
- Create config YAMLs for 15 datasets (5 classification, 5 regression, 5 clustering)
- Upload datasets to Azure ML datastore
- Submit all 15 pipelines and track results

### Changes Made
- Created 15 config YAML files in `configs/`
- Fixed column sanitization for LightGBM/XGBoost special character sensitivity
- Fixed delimiter handling for KDDCup09 (semicolon-separated)
- Added `eda_sample_size` for large datasets

### Dataset Results
| Task Type | Dataset | Status |
|-----------|---------|--------|
| Classification | bank_marketing | ✅ Completed |
| Classification | credit_card_fraud | ✅ Completed |
| Classification | airlines_delay | ✅ Completed |
| Classification | diabetes | ❌ Failed (NaN in stage4) |
| Classification | telco_churn_ibm | ✅ Completed |
| Regression | workers_comp | ✅ Completed |
| Regression | house_sales | ✅ Completed |
| Regression | nyc_taxi | ✅ Completed |
| Regression | medical_charges | ❌ Failed (NaN in HPO) |
| Regression | kddcup09 | ❌ Failed (NaN + special chars) |
| Clustering | credit_default | ❌ Failed (T20 recipe issue) |
| Clustering | online_retail_ii | ❌ Failed (T20 recipe issue) |
| Clustering | atp1d | ❌ Failed (T20 recipe issue) |
| Clustering | kidney_disease | ❌ Failed (T20 recipe issue) |
| Clustering | churn_uplift | ❌ Failed (T20 recipe issue) |

### Outcome
- 6/15 completed successfully on first run
- Identified 3 major failure categories: NaN propagation, column naming, clustering recipe gaps
- Failures carried to Sessions 3-6

---

## Session 3: T15-T19 Hardening

**Date**: Post-scaling fixes
**Branch**: `v3-production`
**Commit**: `d01fae8`
**Focus**: Fix 5 critical issues found during 15-dataset scaling

### Changes Made
| # | Issue | Fix | Files |
|---|-------|-----|-------|
| T15 | json.dump crashes on numpy types | Add NumpyEncoder/default=str to critical paths | `aggregate_baseline.py`, `aggregate_phaseb.py`, `aggregate_phasec.py` |
| T16 | Stage3 column sanitization incomplete | Regex `[\[\]<>{},:"\'\\\\ ]` → underscore, deduplicate | `stage3_preprocessing.py` |
| T17 | Stage4 NaN after ID column removal | Median (numeric) / mode (categorical) imputation guard | `stage4_feature_engineering.py` |
| T18 | Phase C HPO column names unsanitized | Add `_sanitize_columns()` helper + NaN guard | `phasec_optuna_hpo.py` |
| T19 | Phase B/C quality gates missing | Add balanced_accuracy threshold checks | `phaseb_pycaret_recipe.py`, `phaseb_flaml_recipe.py` |

### Outcome
- All 5 fixes applied and committed
- Resubmitted diabetes, medical_charges, kddcup09 → all passed
- T18 note: Phase C HPO still lacks "keep baseline if HPO hurts" comparison

---

## Session 4: T20 Clustering Recovery

**Date**: Post-T15-T19
**Branch**: `v3-production`
**Focus**: Fix all 5 clustering datasets

### Objectives
- Root cause analysis of clustering pipeline failures
- Fix recipe loading and clustering-specific logic

### Changes Made
| # | Issue | Fix | Files |
|---|-------|-----|-------|
| T20a | Clustering recipes missing from configs/recipes/clustering/ | Generated 77 clustering recipes | `configs/recipes/clustering/v1_generated/` |
| T20b | Phase B variant runner hardcoded classification recipe path | Task-type routing for recipe directory | `s06_phaseb_variant_runner.py` |
| T20c | PyCaret clustering uses `create_model` not `compare_models` | Conditional branch for clustering | `stage5_pycaret_train.py` |
| T20d | Silhouette score as default metric for clustering | Task-type metric selection | `phasec_optuna_hpo.py`, `final_evaluation.py` |

### Outcome
- All 5 clustering configs resubmitted
- credit_default ✅, atp1d ✅, kidney_disease ✅
- online_retail_ii and churn_uplift hit OOM (Session 5)

---

## Session 5: OOM Fix

**Date**: Post-clustering recovery
**Branch**: `v3-production`
**Focus**: Fix Out-of-Memory errors on large clustering datasets

### Root Cause
- `silhouette_score()` computes O(n²) pairwise distance matrix
- online_retail_ii (541K rows) and churn_uplift (64K rows) exceed memory

### Changes Made
- Added sampling before silhouette_score: `min(n_rows, 10000)` random sample
- Applied to `phasec_optuna_hpo.py` and `final_evaluation.py`
- Added VIF computation sampling for datasets > 50K rows

### Outcome
- online_retail_ii ✅ Completed
- churn_uplift ✅ Completed
- All 15 datasets now pass through at least stage s10

---

## Session 6: KDDCup09 NaN Fix

**Date**: Post-OOM fix
**Branch**: `v3-production`
**Focus**: Fix persistent NaN issues in KDDCup09 regression dataset

### Root Cause
- KDDCup09 has >90% missing values in many columns
- Standard median imputation left residual NaN when ALL values in a column were NaN
- FLAML and HPO crashed on remaining NaN

### Changes Made
- Added zero-fill fallback after median/mode imputation: `df.fillna(0)`
- Applied at stage3 (source fix) and stage4 + phasec (defensive fix)
- Added pre-training NaN assertion: `assert not df.isnull().any().any()`

### Outcome
- KDDCup09 ✅ Completed end-to-end
- All 15 datasets now complete through final evaluation
- s12 model registration still SKIPPED (known issue O1)

---

## Session 7: Drift Detection (s13)

**Date**: Feature development
**Branch**: `drift-detection-v1`
**Commit**: `e299ff69`
**Focus**: Production-ready drift monitoring as pipeline step s13

### Changes Made
- **New file**: `src/steps/s13_drift_monitor.py` (656 lines)
  - PSI (Population Stability Index) with self-check validation
  - Evidently AI comparison report (if library available)
  - Concept drift detection via prediction distribution analysis
  - Stability scoring with retraining recommendations
  - MLflow logging of all drift metrics and HTML reports

- **New file**: `src/utils/drift_detector.py` (334 lines)
  - `compute_psi()` with histogram-based bucketing
  - `compute_baseline_statistics()` for reference snapshot
  - `compute_stability_score()` weighted aggregation
  - Constants: PSI thresholds (0.1 low, 0.25 high)

- **New file**: `components/s13_drift_monitor.yml`
  - Inputs: config, production_data, baseline_data, baseline_model, threshold
  - Outputs: drift_report, drift_metrics, retraining_recommendation

- **New file**: `docs/DRIFT_DETECTION.md` (14 sections)
  - Architecture, PSI methodology, concept drift, stability scoring
  - Integration guide, troubleshooting, API reference

- **Updated**: 14 component YAMLs with `mode: upload` + version bumps
- **Added**: 456 variant recipe configs (classification/regression/clustering)
- **Added**: `scripts/resubmit_6_failed.py`

### Outcome
- Complete drift detection system ready for integration
- Pushed to `origin/drift-detection-v1`

---

## Session 8: FastAPI Pipeline API

**Date**: API development
**Branch**: Files created but NOT committed to drift-detection-v1
**Focus**: RESTful API for pipeline management

### Changes Made
- **New directory**: `api/`
  - `app.py` — FastAPI application with CORS, health check
  - `routes/pipeline_routes.py` — Pipeline CRUD endpoints
  - `routes/job_routes.py` — Job monitoring endpoints
  - `routes/config_routes.py` — Config management endpoints
  - `services/azure_ml_service.py` — Azure ML SDK v2 integration
  - `services/config_service.py` — Config file management
  - `models/schemas.py` — Pydantic request/response models

- **New file**: `api_requirements.txt` — FastAPI, uvicorn, azure-ai-ml dependencies
- **New file**: `.env.example` — Environment variable template
- **New file**: `scripts/run_api.sh` — API launch script

### API Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/api/v1/pipelines/submit` | Submit pipeline job |
| GET | `/api/v1/jobs` | List recent jobs |
| GET | `/api/v1/jobs/{job_name}` | Job details |
| GET | `/api/v1/jobs/{job_name}/steps` | Job child steps |
| GET | `/api/v1/configs` | List configs |
| GET | `/api/v1/configs/{name}` | Read config |

### Outcome
- API functional but NOT committed to any branch
- Files exist as untracked in working directory

---

## Session 9: Drift Detection Push

**Date**: Branch management
**Branch**: `drift-detection-v1`
**Commit**: `e299ff69`
**Focus**: Push drift detection to remote

### Changes Made
- Pushed drift-detection-v1 branch to `origin/drift-detection-v1`
- Verified commit `e299ff69` at remote HEAD

### Outcome
- Branch available at `origin/drift-detection-v1`
- Ready for PR to main when approved

---

## Open Issues (Carried Forward)

| ID | Priority | Issue | Status |
|----|----------|-------|--------|
| O1 | HIGH | s12 model registration SKIPPED — artifact path format invalid for MLflow registry | Open |
| T18 | MEDIUM | Phase C HPO lacks "keep baseline if HPO hurts" comparison | Open |
| O2 | LOW | ~40% of json.dump calls in non-critical utils lack NumpyEncoder/default=str | Open |

## Branch State Summary

| Branch | HEAD Commit | Status |
|--------|-------------|--------|
| `v3-production` | `d01fae8` (T15-T19 fixes) | Clean, pushed |
| `drift-detection-v1` | `e299ff69` (drift detection) | 12 modified files (T15-T19 fixes) + untracked API/configs |
| `main` | Default branch | Upstream |

## Uncommitted Changes on drift-detection-v1
- **Modified (12 files)**: T15-T19 hardening fixes from v3-production (aggregate_baseline, aggregate_phaseb, aggregate_phasec, final_evaluation, phaseb_flaml_recipe, phaseb_pycaret_recipe, phasec_optuna_hpo, s13_drift_monitor, stage1_ingestion, stage3_preprocessing, stage5_flaml_train, stage5_pycaret_train)
- **Untracked**: api/ directory, api_requirements.txt, .env.example, scripts/run_api.sh, 5 clustering config YAMLs
