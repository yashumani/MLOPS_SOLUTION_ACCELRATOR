# MLOps Solution Accelerator — V3 Production Blueprint

> **Version:** V3.0  
> **Date:** January 28, 2026  
> **Repository:** SAVYMINDS/YS_MVP · Branch: `v3-production`  
> **Azure ML Workspace:** `mlops-accelerator` (RG: `mvpv1`, Sub: `93044a08-...`)  
> **Compute:** `mlopsv2computecluster` · **Environment:** `mlops-v3-unified:20`

---

## Executive Summary

The V3 Production Blueprint documents the complete Data Science Lifecycle implemented across the MLOps Solution Accelerator V3 pipeline. It covers **12 orchestrated Azure ML pipeline steps** (with 2 additional component definitions — `s0_data_validation` and `s12_model_registration` — defined but not yet wired), **4 task types** (Classification, Regression, Clustering, Forecasting), and **67 registered models** spanning 6 engine categories. The pipeline is fully config-driven, recipe-based, and submitted exclusively through Azure ML with zero local execution.

The variant library contains **330 variant-search recipes** (210 classification, 80 regression, 40 clustering), **112 V1-generated recipes**, **3 enterprise-tier recipes**, and **8 baseline/standalone recipes** — totaling **~453 preprocessing variants** across all task types.

---

## Blueprint Table — Data Science Lifecycle Stages

### Row 1 — Data Ingestion & Validation

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Data Ingestion & Validation |
| **Pipeline Step(s)** | `s01_ingestion` (stage0_data_validation.py → stage1_ingestion.py) |
| **Possible Libraries & Tools** | pandas, numpy, PyYAML, MLflow, pandera, Great Expectations, Azure ML Datastores, azureml-fsspec |
| **Type of Problem Supported** | Classification, Regression, Clustering, Forecasting (auto-detected) |
| **Libraries & Methods Actually Used** | pandas (read_csv with configurable delimiter), numpy, PyYAML (config loader), MLflow (artifact logging), pandera (schema validation), matplotlib/seaborn (EDA plots). Functions: `load_config()`, `build_dataset_uri()`, `validate_data_quality()` with RED/YELLOW/GREEN quality gates, `detect_time_series()` with 7 heuristics (sorted datetime index, autocorrelation, frequency detection, seasonal decomposition, trend stationarity, temporal column ratio, temporal ordering) |
| **Datasets Used** | telecom_churn.csv (243K×45, Telecom), california_housing.csv (20K×9, Real Estate), wholesale_customers.csv (440×8, Retail), bank_marketing.csv (Finance), college.csv (Education), online_retail.csv (E-Commerce) |
| **Domain / Industry** | Telecommunications, Real Estate, Retail, Finance, Education, E-Commerce |
| **Key Performance Metrics** | Data quality gates (row count, null percentage, duplicate rate, column type validation), schema validation pass/fail, time-series detection confidence score, EDA report generation status |

---

### Row 2 — Data Cleaning & Preparation

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Data Cleaning & Preparation |
| **Pipeline Step(s)** | `s02_preparation` (stage2_preparation.py) |
| **Possible Libraries & Tools** | pandas, numpy, scikit-learn (SimpleImputer, KNNImputer, IterativeImputer), category_encoders, sweetviz, feature-engine |
| **Type of Problem Supported** | Classification, Regression, Clustering, Forecasting |
| **Libraries & Methods Actually Used** | pandas (dropna, fillna, astype, drop_duplicates, get_dummies), numpy, sweetviz (EDA report generation), category_encoders (TargetEncoder, BinaryEncoder, OrdinalEncoder). Functions: EDA directory propagation via `--eda_dir` formal arg, missing value profiling, cardinality analysis, duplicate removal, type inference and coercion |
| **Datasets Used** | All configured datasets (passed from s01 output) |
| **Domain / Industry** | Domain-agnostic — applied uniformly across all configured datasets |
| **Key Performance Metrics** | Missing value count before/after, duplicate rows removed, column type conversion count, EDA report artifacts logged to MLflow |

---

### Row 3 — Preprocessing & Recipe-Driven Transforms

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Preprocessing & Recipe-Driven Transforms |
| **Pipeline Step(s)** | `s03_preprocessing` (stage3_preprocessing.py) |
| **Possible Libraries & Tools** | scikit-learn (StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer, LabelEncoder, OrdinalEncoder, OneHotEncoder), imblearn (SMOTE, ADASYN, BorderlineSMOTE), statsmodels (VIF), category_encoders (TargetEncoder) |
| **Type of Problem Supported** | Classification (with imbalance handling), Regression (with outlier handling), Clustering (scaling only), Forecasting (passthrough) |
| **Libraries & Methods Actually Used** | sklearn.preprocessing (StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer, LabelEncoder), imblearn.over_sampling (SMOTE, ADASYN), statsmodels.stats.outliers_influence (variance_inflation_factor for VIF multicollinearity detection). Functions: `detect_multicollinearity()` (VIF > 10 threshold), `preprocess()` with recipe-driven pipeline supporting: imputation (mean, median, knn, iterative, none), encoding (onehot, target, label, ordinal, binary, none), scaling (standard, robust, minmax, quantile, power, none), imbalance handling (smote, adasyn, borderline_smote, none — classification only), outlier handling (iqr, zscore, percentile — regression only) |
| **Datasets Used** | All configured datasets (passed from s02 output) |
| **Domain / Industry** | Configurable per domain — finance uses robust scaling, academic uses standard scaling (domain-aware recommendations from DatasetProfiler) |
| **Key Performance Metrics** | VIF multicollinearity scores, imbalance ratio before/after SMOTE, feature count before/after encoding, scaling method applied, outlier count removed |

---

### Row 4 — Feature Engineering & Selection

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Feature Engineering & Selection |
| **Pipeline Step(s)** | `s04_feature_engineering` (stage4_feature_engineering.py) |
| **Possible Libraries & Tools** | scikit-learn (SelectKBest, mutual_info_classif, mutual_info_regression, VarianceThreshold, f_classif, f_regression, RFE), sklearn.decomposition (PCA), Boruta, feature-engine |
| **Type of Problem Supported** | Classification, Regression, Clustering, Forecasting |
| **Libraries & Methods Actually Used** | sklearn.feature_selection (SelectKBest with mutual_info_classif/mutual_info_regression, VarianceThreshold), sklearn.decomposition (PCA). Functions: `detect_imbalance()`, `feature_engineer()` with recipe-driven selection supporting: correlation filter, variance threshold, mutual information (classification/regression-aware), PCA dimensionality reduction, Boruta wrapper (optional). High-cardinality protection: features with >100 unique values dropped before one-hot encoding |
| **Datasets Used** | All configured datasets (passed from s03 output) |
| **Domain / Industry** | Domain-agnostic — selection strategy driven by recipe YAML configuration |
| **Key Performance Metrics** | Features selected count, features dropped count, variance explained (PCA), mutual information scores, feature importance rankings |

---

### Row 5 — Baseline Model Training (Phase A — PyCaret)

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Baseline Model Training — Phase A (PyCaret Engine) |
| **Pipeline Step(s)** | `s05a_baseline_pycaret` (stage5_pycaret_train.py) |
| **Possible Libraries & Tools** | PyCaret (classification, regression, clustering), scikit-learn, XGBoost, LightGBM, CatBoost |
| **Type of Problem Supported** | Classification (14 models), Regression (23 models), Clustering (8 models) |
| **Libraries & Methods Actually Used** | PyCaret 3.x (`setup()`, `compare_models()`, `pull()`, `save_model()`, `predict_model()`), scikit-learn (classification_report, confusion_matrix, roc_auc_score, average_precision_score). Imbalance-aware training with automatic threshold optimization (`_optimal_threshold_f1`). **Classification Models (14):** lr, knn, nb, dt, ridge, rf, qda, ada, gbc, lda, et, xgboost, lightgbm, catboost. **Regression Models (23):** lr, lasso, ridge, en, lar, llar, omp, br, ard, par, ransac, tr, huber, kr, knn, dt, rf, et, ada, gbr, xgboost, lightgbm, catboost. **Clustering Models (8):** kmeans, ap, meanshift, sc, hclust, dbscan, optics, birch. NOTE: SVM, RBF SVM, MLP intentionally REMOVED from classification for runtime safety |
| **Datasets Used** | Engineered dataset from s04 output |
| **Domain / Industry** | Multi-domain — model selection automatic per task type |
| **Key Performance Metrics** | Classification: AUC, Accuracy, Precision, Recall, F1, Kappa, MCC, Log Loss, optimal F1 threshold. Regression: R2, MAE, MSE, RMSE, MAPE. Clustering: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index. All models tracked via `model_breakdown_s05a.csv` and Candidate Ledger |

---

### Row 6 — Baseline Model Training (Phase A — FLAML)

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Baseline Model Training — Phase A (FLAML Engine) |
| **Pipeline Step(s)** | `s05b_baseline_flaml` (stage5_flaml_train.py) |
| **Possible Libraries & Tools** | FLAML (AutoML), scikit-learn, XGBoost, LightGBM, CatBoost |
| **Type of Problem Supported** | Classification (9 models), Regression (7 models). Clustering: SKIPPED (FLAML does not support clustering) |
| **Libraries & Methods Actually Used** | FLAML AutoML (`AutoML()`, `.fit()` with configurable `time_budget`), scikit-learn (accuracy_score, r2_score, roc_auc_score). Per-estimator timeout protection via multiprocessing guard. **Classification Models (9):** lgbm, xgboost, xgb_limitdepth, catboost, rf, extra_tree, lrl1, lrl2, kneighbor. **Regression Models (7):** lgbm, xgboost, xgb_limitdepth, catboost, rf, extra_tree, kneighbor. **Clustering:** Explicitly skipped with graceful skip artifacts written |
| **Datasets Used** | Engineered dataset from s04 output |
| **Domain / Industry** | Multi-domain — automatic model selection with time-budget constraints |
| **Key Performance Metrics** | Classification: AUC, Accuracy, F1. Regression: R2, MAE, MSE. Time budget utilization, best estimator selected, training time per model. All models tracked via `model_breakdown_s05b.csv` and Candidate Ledger |

---

### Row 7 — Time-Series Forecasting Training (Phase A — statsmodels)

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Time-Series / Forecasting Training — Phase A (statsmodels) |
| **Pipeline Step(s)** | `s05t_baseline_timeseries` (stage5_timeseries_train.py) |
| **Possible Libraries & Tools** | statsmodels (ARIMA, SARIMAX, ExponentialSmoothing, SimpleExpSmoothing, ThetaModel), Prophet, NeuralProphet |
| **Type of Problem Supported** | Forecasting only (conditionally executed when `detect_time_series()` returns True) |
| **Libraries & Methods Actually Used** | statsmodels.tsa.arima.model (ARIMA), statsmodels.tsa.statespace.sarimax (SARIMAX), statsmodels.tsa.holtwinters (ExponentialSmoothing, SimpleExpSmoothing), statsmodels.tsa.forecasting.theta (ThetaModel), custom Seasonal Naive wrapper. **Forecasting Models (6):** arima, sarima, exponential_smoothing, ses, theta, naive. Temporal train/test split (no random shuffle). Auto-detected seasonal periods |
| **Datasets Used** | Time-series datasets detected at s01 ingestion stage |
| **Domain / Industry** | Any domain with temporal ordering — Finance, Retail, IoT, Weather |
| **Key Performance Metrics** | MAE, RMSE, MAPE, SMAPE per model. Best model selection by lowest MAE. All models tracked via `model_breakdown_s05t.csv` and Candidate Ledger |

---

### Row 8 — Baseline Aggregation

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Baseline Aggregation — Phase A Results Merge |
| **Pipeline Step(s)** | `s05z_aggregate_baseline` (aggregate step) |
| **Possible Libraries & Tools** | pandas, MLflow, custom AIM-Tournament engine |
| **Type of Problem Supported** | Classification, Regression, Clustering, Forecasting (aggregates all Phase A engines) |
| **Libraries & Methods Actually Used** | pandas (concat, merge, sort_values), MLflow (artifact logging), JSON manifest merging. Collects results from s05a (PyCaret), s05b (FLAML), and s05t (TimeSeries). Produces unified baseline leaderboard. Naming uses `z` suffix to appear last alphabetically in Azure ML Studio |
| **Datasets Used** | Phase A output manifests and metrics from all three engines |
| **Domain / Industry** | Domain-agnostic — unified ranking across engines |
| **Key Performance Metrics** | Unified leaderboard ranked by primary metric (AUC/R2/Silhouette/MAE), engine coverage report, model count per engine, best model identification |

---

### Row 9 — Phase B Variant Exploration (Intelligent Preprocessing Search)

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Phase B — Variant Exploration with Intelligent Recommendation |
| **Pipeline Step(s)** | `s06_phaseb_variant_runner` (s06_phaseb_variant_runner.py, 2050 lines) |
| **Possible Libraries & Tools** | PyCaret, FLAML, scikit-learn, imblearn, Optuna, custom DatasetProfiler, VariantRecommender, VariantPlanner, PreprocessingCache |
| **Type of Problem Supported** | Classification (210 variant search library), Regression (80 variant search library), Clustering (40 variant search library) — intelligent selection per run |
| **Libraries & Methods Actually Used** | Industrial-grade variant execution engine with: `DatasetProfiler` (dataset statistical profiling), `VariantRecommender` (variant scoring 0-100 with diversity boost), `VariantPlanner` (EdaPriors, VariantPlan, score_variant_relevance, diverse_sample, compute_preprocessing_hash), `PreprocessingCache` (dedup identical preprocessing), `VariantSchema` (load_variant, validate_variant_for_task). Features: per-variant checkpointing (resume capability), per-variant time budget enforcement, stable ChampionManifest contract, normalized VariantResult schema, failure tolerance, **two-round selection** (round 1: up to 40 variants from planner, round 2: top 10 after proxy pruning at 0.50 threshold), **`preprocess=False`** in PyCaret `setup()` to prevent double-preprocessing, **`flaml_min_budget=120`** seconds per FLAML estimator. Runs N variants × M engines in single Azure ML step with nested MLflow runs |
| **Datasets Used** | Engineered dataset from s04, plus variant YAML recipes in `configs/recipes/{task}/variant_search/` — 210 classification, 80 regression, 40 clustering (330 total). Variants are passed via `--variants_list` (comma-separated string). Additionally: 112 V1-generated recipes, 3 enterprise lightning-fast recipes (classification), 8 baseline/standalone recipes |
| **Domain / Industry** | Data-driven — profiler recommends strategies based on dataset characteristics: imbalance ratio, missing rate, outlier prevalence, multicollinearity, domain hints |
| **Key Performance Metrics** | Per-variant: primary metric (AUC/R2/Silhouette), runtime_sec, timed_out flag, leakage_risk score. Aggregate: leaderboard.csv, all_results.json, champion_manifest.json, champion_model.pkl. Variant coverage: intelligently selected from 330 total variant-search recipes via two-round planner (round 1 max 40 → proxy prune → round 2 max 10). **`--stop_compute`** flag available on `submit_pipeline.py` for automatic compute shutdown after job completion |

---

### Row 10 — Phase B Aggregation

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Phase B Aggregation — Variant Results Merge |
| **Pipeline Step(s)** | `s08_aggregate_phaseb` (aggregate step) |
| **Possible Libraries & Tools** | pandas, MLflow, JSON |
| **Type of Problem Supported** | Classification, Regression, Clustering |
| **Libraries & Methods Actually Used** | pandas (DataFrame operations), MLflow (artifact logging). Merges all Phase B variant×engine results into unified leaderboard. Identifies Phase B champion. Logs variant selection report |
| **Datasets Used** | Phase B output leaderboards and champion manifests |
| **Domain / Industry** | Domain-agnostic |
| **Key Performance Metrics** | Phase B champion metric value, variant count completed vs failed, best variant_id, preprocessing configuration of champion |

---

### Row 11 — Phase C Hyperparameter Optimization (Optuna)

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Phase C — Hyperparameter Optimization |
| **Pipeline Step(s)** | `s09_phasec_optuna_hpo` (phasec_optuna_hpo.py, 687 lines) |
| **Possible Libraries & Tools** | Optuna, scikit-learn, XGBoost, LightGBM, CatBoost, MLflow |
| **Type of Problem Supported** | Classification, Regression (HPO on champion model from Phase B) |
| **Libraries & Methods Actually Used** | Optuna (create_study, TPESampler, MedianPruner), scikit-learn (cross_val_score, train_test_split, make_scorer, accuracy_score, r2_score), MLflow (nested trial logging). Configurable `n_trials` (default 50) and `test_size` from config. Dataset inspection with size/content validation before loading. Explicit autolog disable to preserve MLflow hierarchy. Local model registry fallback (`file:///tmp/mlflow-registry`) to avoid azureml:// registry errors |
| **Datasets Used** | Engineered dataset from s04 (same as training steps) |
| **Domain / Industry** | Domain-agnostic — search space configured per model type in config YAML |
| **Key Performance Metrics** | Best trial metric value, number of completed trials, pruned trials count, best hyperparameter set, optimization time, cross-validation scores per trial |

---

### Row 12 — Phase C Aggregation

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Phase C Aggregation — HPO Results Merge |
| **Pipeline Step(s)** | `s10_aggregate_phasec` (aggregate step) |
| **Possible Libraries & Tools** | pandas, MLflow, Optuna study export |
| **Type of Problem Supported** | Classification, Regression |
| **Libraries & Methods Actually Used** | pandas, MLflow artifact logging, Optuna study serialization. Merges HPO results with Phase B champion to determine if HPO improved performance |
| **Datasets Used** | Phase C study outputs and optimized model |
| **Domain / Industry** | Domain-agnostic |
| **Key Performance Metrics** | HPO improvement delta over Phase B champion, best hyperparameters, final champion metric value |

---

### Row 13 — Final Evaluation & Champion Selection

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Final Evaluation & Champion Selection |
| **Pipeline Step(s)** | `s11_final_evaluation` (final_evaluation.py, 1125 lines) |
| **Possible Libraries & Tools** | scikit-learn (all metrics), MLflow, matplotlib, seaborn, pandas, custom AIM-Tournament, Candidate Ledger |
| **Type of Problem Supported** | Classification, Regression, Clustering, Forecasting |
| **Libraries & Methods Actually Used** | sklearn.metrics (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error, silhouette_score, davies_bouldin_score, calinski_harabasz_score), matplotlib/seaborn (evaluation plots — confusion matrix, ROC curve, feature importance, residual plots), MLflow (final artifact logging). Functions: `collect_all_stage_metrics()` (scans all 15 stages via MLflow client), `run_aim_tournament()` (multi-metric ranking with Pareto frontier), `merge_ledgers()`, `build_summary()`, `build_readme_md()`. Outputs: `all_candidates.csv`, champion model .pkl, champion metrics JSON, evaluation plots, coverage report, stage signal summaries |
| **Datasets Used** | Original dataset + all stage outputs and manifests |
| **Domain / Industry** | Multi-domain — evaluation metrics selected per task type |
| **Key Performance Metrics** | Classification: AUC, Accuracy, Precision, Recall, F1, MCC, Log Loss. Regression: R2, MAE, MSE, RMSE, MAPE. Clustering: Silhouette, Davies-Bouldin, Calinski-Harabasz. Forecasting: MAE, RMSE, MAPE, SMAPE. Cross-phase champion comparison (Phase A vs B vs C) |

---

### Row 14 — Model Registration & Deployment Readiness

| Dimension | Detail |
|-----------|--------|
| **Data Science Lifecycle Stage** | Model Registration & Deployment Readiness |
| **Pipeline Step(s)** | `s12_model_registration` (component YAML defined) |
| **Possible Libraries & Tools** | Azure ML SDK v2 (Model Registration), MLflow Model Registry, Azure ML Managed Endpoints |
| **Type of Problem Supported** | Classification, Regression, Clustering, Forecasting |
| **Libraries & Methods Actually Used** | Azure ML SDK v2 (model registration to workspace), MLflow (model packaging with signature and input example). Champion model from final_evaluation registered with full lineage: execution_id, config hash, dataset fingerprint, training metrics, preprocessing recipe |
| **Datasets Used** | Champion model artifact from s11 final evaluation |
| **Domain / Industry** | Domain-agnostic — registration metadata includes domain tags |
| **Key Performance Metrics** | Registration success/fail, model version, model size, artifact URI, deployment readiness score |

---

## Cross-Cutting Frameworks & Utilities

### Signal Framework
| Item | Detail |
|------|--------|
| **Components** | `stage_signals.py`, `stage_registry.py` |
| **Purpose** | Inter-stage communication via typed signals (StageSignal dataclass) |
| **Coverage** | Wired into all 12 wired pipeline stages (s01–s10) |
| **Signals Tracked** | Stage status (pass/fail/skip), quality gates, timing, artifact paths |

### Candidate Ledger
| Item | Detail |
|------|--------|
| **Component** | `candidate_ledger.py` |
| **Purpose** | Filesystem-based candidate tracking across all training stages |
| **Functions** | `make_row()`, `normalize_metrics()`, `write_candidate_artifacts()`, `write_stage_table()`, `merge_ledgers()`, `build_summary()`, `build_readme_md()`, `sha256_file()` |
| **Output** | `all_candidates.csv` — unified view of every model trained across Phase A/B/C |

### AIM-Tournament
| Item | Detail |
|------|--------|
| **Component** | `aim_tournament.py` |
| **Purpose** | Multi-metric ranking with Pareto frontier for champion selection |
| **Method** | Weighted multi-objective scoring, Pareto-optimal identification, tiebreaker logic |

### Model Universe
| Item | Detail |
|------|--------|
| **Component** | `model_universe.py` (417 lines) |
| **Total Models** | 67 model entries across 6 categories |
| **Breakdown** | Classification PyCaret: 14 · Classification FLAML: 9 · Regression PyCaret: 23 · Regression FLAML: 7 · Clustering PyCaret: 8 · Forecasting statsmodels: 6 |
| **Functions** | `get_model_list()`, `get_forecasting_models()`, `build_coverage_report()`, `write_model_coverage()`, `build_pycaret_breakdown()`, `build_flaml_breakdown()`, `write_model_breakdown()` |

### Bundle Gating
| Item | Detail |
|------|--------|
| **Component** | `bundle_gating.py` |
| **Purpose** | Signal-based bundle decisions — controls which variant bundles execute based on data characteristics |

### Dataset Profiler & Variant Recommendation
| Item | Detail |
|------|--------|
| **Components** | `dataset_profiler.py`, `variant_recommender.py`, `variant_planner.py`, `variant_selector.py`, `variant_schema.py` |
| **Purpose** | Intelligent variant selection: Profile dataset → Score task-specific variants (210/80/40) by relevance (0-100) → Two-round selection (round 1 max 40, round 2 max 10 after proxy pruning) → Execute |
| **Scoring Weights** | Imputation 25%, Encoding 20%, Scaling 15%, Imbalance 25%, Feature Selection 15% |

---

## Pipeline Orchestration Summary

| Item | Detail |
|------|--------|
| **Orchestration** | Azure ML `@dsl.pipeline` — single level, no nested pipelines |
| **Submission** | `pipelines/submit_pipeline.py` (canonical entrypoint) |
| **Pipeline Builder** | `src/orchestration/pipeline_builder.py` (dynamic assembly) |
| **Pipeline Definition** | `pipelines/full_pipeline.py` |
| **Step Count** | 12 wired component steps per run (s01→s02→s03→s04→s5a∥s5b∥s5t→s5z→s06→s08→s09→s10). Components `s0_data_validation` and `s12_model_registration` defined but not yet wired into the pipeline DAG |
| **Child Jobs** | 12 child jobs observed per successful run |
| **Execution ID** | `sha1(dataset|task|preset|timestamp|config_hash)[:12]` — deterministic, propagated to all steps |
| **Config-Driven** | All parameters from YAML in `configs/` — zero hardcoded values |
| **Environments** | `environments/` — single unified conda environment (`mlops-v3-unified`) used by all steps; s06 variant runner may use `:23` for extended dependencies |

---

## Validated Production Runs

| Job Name | Duration | Task | Variants | Engines | Status |
|----------|----------|------|----------|---------|--------|
| `stoic_knee_kn5111100k` | 5h 18m | Classification (telecom_churn) | 20 | pycaret + flaml | Completed (canonical) |
| `coral_onion_d3cg9dzz7s` | 4h 59m | Classification (telecom_churn) | 20 | pycaret + flaml | Completed |
| `wheat_feijoa_7458ytkdyq` | — | Classification (telecom_churn) | — | pycaret + flaml | Submitted (latest) |
| `amiable_glass_343413dpkt` | — | Regression (california_housing) | — | pycaret + flaml | Submitted (latest) |
| `keen_yak_2v8ptp3kg4` | — | Clustering (wholesale_customers) | — | pycaret | Submitted (latest) |

---

## Comparison: V1 Blueprint → V3 Blueprint

| Dimension | V1 (Old) | V3 (Current) |
|-----------|----------|--------------|
| Pipeline Steps | 5-6 manual stages | 12 wired Azure ML components (+ 2 defined but unwired) |
| Task Types | Classification only | Classification, Regression, Clustering, Forecasting |
| Training Engines | PyCaret only | PyCaret + FLAML + statsmodels (3 engines) |
| Model Count | ~10 classification models | 67 models across 6 categories |
| Preprocessing | Hardcoded transforms | 330 variant-search recipes (210 classification, 80 regression, 40 clustering), 112 V1-generated, 3 enterprise + 8 baseline — ~453 total variants, intelligently selected per run via two-round planner |
| HPO | Grid/Random search | Optuna TPE with MedianPruner (50 trials default) |
| Tracking | Basic MLflow | Candidate Ledger + AIM-Tournament + Signal Framework + model_breakdown CSVs |
| Execution | Local + Azure mixed | Azure ML only — zero local execution |
| Config | Scattered parameters | Unified YAML configs with schema validation |
| Variant Selection | Manual | DatasetProfiler → VariantRecommender → VariantPlanner (scored 0-100, two-round selection with proxy pruning at 0.50 threshold) |

---

*Generated from codebase audit of `mlops-solution-accelerator-v3/` on January 28, 2026. Last updated: January 30, 2026.*
