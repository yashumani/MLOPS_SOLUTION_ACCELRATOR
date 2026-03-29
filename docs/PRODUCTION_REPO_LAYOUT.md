# Production Repository Layout — V3 Bounded-Tournament Pipeline

**Last updated**: 2026-03-13

---

## 1  Production-Critical Directories

These directories are required for Azure ML pipeline submission and execution.

```
mlops-solution-accelerator-v3/
├── components/                 # Azure ML component YAMLs (18 total — immutable I/O contracts)
│   ├── stage0_data_validation.yml        # ⚠ NOT wired in pipeline
│   ├── stage1_ingestion.yml
│   ├── stage2_preparation.yml
│   ├── stage3_preprocessing.yml
│   ├── stage4_feature_engineering.yml
│   ├── stage5_pycaret_train.yml
│   ├── stage5_flaml_train.yml
│   ├── stage5_timeseries_train.yml
│   ├── aggregate_baseline.yml
│   ├── s06_phaseb_variant_runner.yml
│   ├── phaseb_pycaret_recipe.yml         # Legacy (kept for reference)
│   ├── phaseb_flaml_recipe.yml           # Legacy (kept for reference)
│   ├── aggregate_phaseb.yml
│   ├── aggregate_phaseb_dynamic.yml
│   ├── phasec_optuna_hpo.yml
│   ├── aggregate_phasec.yml
│   ├── final_evaluation.yml
│   └── s12_model_registration.yml        # ⚠ NOT wired in pipeline
│
├── configs/                    # Task configs (6 files) + recipe library
│   ├── config_classification_telecom_churn_azureml.yml
│   ├── config_classification_telecom_churn_local.yml
│   ├── config_classification_telecom_churn_test_s06.yml
│   ├── config_classification_bank_marketing_azureml.yml
│   ├── config_regression_college_azureml.yml
│   ├── config_clustering_online_retail_azureml.yml
│   ├── variant_bundles/        # Pre-selected variant bundles
│   │   ├── classification/
│   │   ├── regression/
│   │   └── clustering/
│   └── recipes/                # 457 variant recipe YAMLs
│       ├── recipe_knn_onehot_minmax.yml
│       ├── recipe_library.yml
│       ├── recipe_smote_target_standard.yml
│       ├── classification/
│       │   ├── variant_search/           # 210 auto-generated variants
│       │   ├── v1_generated/             # 44 structured variants
│       │   └── enterprise_lightning_fast/ # 3 fast-track variants
│       ├── regression/
│       │   ├── variant_search/           # 80 auto-generated variants
│       │   └── v1_generated/             # 43 structured variants
│       └── clustering/
│           ├── variant_search/           # 40 auto-generated variants
│           └── v1_generated/             # 25 structured variants
│
├── environments/               # Single unified Azure ML environment
│   ├── unified_conda.yml               # Conda env (PyCaret + FLAML + all deps)
│   └── azureml_unified_env.yml         # Azure ML environment definition
│
├── pipelines/                  # Orchestration (IMMUTABLE entrypoints)
│   ├── submit_pipeline.py      # ❌ DO NOT MODIFY — canonical submission (--stop_compute flag)
│   └── pipeline_builder.py     # ❌ DO NOT MODIFY — @dsl.pipeline + dynamic assembly
│
├── src/
│   ├── steps/                  # Step scripts (19 .py + __init__.py = 20 files)
│   │   ├── __init__.py
│   │   ├── stage0_data_validation.py       # ⚠ NOT wired
│   │   ├── stage1_ingestion.py
│   │   ├── stage1_ingestion_v4.py          # V4 ingestion (alternate)
│   │   ├── stage2_preparation.py
│   │   ├── stage3_preprocessing.py
│   │   ├── stage4_feature_engineering.py
│   │   ├── stage5_pycaret_train.py
│   │   ├── stage5_flaml_train.py
│   │   ├── stage5_timeseries_train.py
│   │   ├── aggregate_baseline.py
│   │   ├── s06_phaseb_variant_runner.py    # Variant exploration engine
│   │   ├── s07_phase2_pipeline_attribution.py
│   │   ├── phaseb_pycaret_recipe.py        # Legacy (kept for reference)
│   │   ├── phaseb_flaml_recipe.py          # Legacy (kept for reference)
│   │   ├── aggregate_phaseb.py
│   │   ├── phasec_optuna_hpo.py
│   │   ├── aggregate_phasec.py
│   │   ├── final_evaluation.py
│   │   └── s12_model_registration.py       # ⚠ NOT wired
│   │
│   ├── utils/                  # Shared utilities (20 files)
│   │   ├── azureml_metrics_logger.py
│   │   ├── stage_signals.py    # StageSignal framework
│   │   ├── stage_registry.py   # Canonical stage numbering
│   │   ├── data_validator.py
│   │   ├── eda_generator.py
│   │   ├── azure_helper.py
│   │   ├── mlflow_helper.py
│   │   ├── aim_tournament.py
│   │   ├── bundle_gating.py
│   │   ├── candidate_ledger.py
│   │   ├── dataset_profiler.py
│   │   ├── jsonl_logger.py
│   │   ├── model_universe.py
│   │   ├── preprocessing_cache.py
│   │   ├── recipe_converter.py
│   │   ├── recipe_selector.py
│   │   ├── variant_planner.py
│   │   ├── variant_recommender.py
│   │   ├── variant_schema.py
│   │   └── variant_selector.py
│   │
│   ├── orchestration/          # Config validation
│   │   └── config_schema.py
│   │
│   └── variant_search/         # Variant search engine
│       ├── __init__.py
│       └── variant_search_engine.py
│
├── scripts/                    # Operational scripts (NOT uploaded to Azure ML)
│   ├── extract_job_results.py
│   ├── validate_aim_tournament.py
│   ├── validate_candidate_ledger.py
│   └── monitor_pipeline.sh
│
├── tests/                      # Validation tests (NOT uploaded to Azure ML)
│   ├── validate_critical_fixes.py
│   └── analyze_pipeline_run.py
│
└── docs/                       # Architecture decisions & guides
    ├── PRODUCTION_REPO_LAYOUT.md
    ├── AIM_TOURNAMENT.md
    ├── LEDGER.md
    ├── MLOPS-v3-blueprint.CSV
    ├── PHASE_B_VARIANT_RUNNER_ARCHITECTURE.md
    ├── PIPELINE_QUALITY_SKILL.md
    ├── VARIANT_SEARCH_GUIDE.md
    └── blueprints/
```

---

## 2  Dev-Only / Archive Directories

These directories should **not** be uploaded with pipeline code or included in production conda environments.

| Directory / File                      | Disposition       | Notes                                      |
|---------------------------------------|-------------------|--------------------------------------------|
| `archive/`                            | Keep local only   | V1/V2 backups, old experiments             |
| `tests/`                              | Keep local only   | Unit tests (run pre-submission, not in job) |
| `*.csv` (tracking files at root)      | Archive           | `ALL_JOBS_MASTER_TRACKING.csv` etc.        |

| `Logs/`, `tmp_logs/`                  | `.gitignore`      | Runtime artifacts                          |
| `sdk v2 sample/`                      | Archive           | Reference only                             |

---

## 3  Upload Scope for Azure ML Jobs

When `submit_pipeline.py` runs, it uploads the **code directory** to Azure ML. The uploaded tree should include:

```
components/       ✅  (component YAMLs)
configs/          ✅  (task configs + recipes)
src/              ✅  (step scripts + utils)
pipelines/        ✅  (pipeline definitions)
environments/     ✅  (conda env files)
```

The following should be **excluded** via `.amlignore`:

```
# .amlignore
archive/
tests/
scripts/
docs/
Logs/
tmp_logs/
*.csv
*.md
__pycache__/
.git/
.github/
```

---

## 4  Naming Conventions

| Entity              | Pattern                                                    | Example                                    |
|---------------------|------------------------------------------------------------|--------------------------------------------|
| Step DSL key        | `s{N}` (single) or `s{N}{a/b/t/z}` (parallel+aggregate)  | `s5a`, `s5t`, `s5z`, `s06`                 |
| Step canonical ID   | `S{NN}` or `S{NN}{a/b/t/z}`                              | `S05t`, `S05z`, `S08`, `S10`               |
| Signal file         | `{stage}_stage_signal.json`                                | `baseline_stage_signal.json`               |
| Config file         | `config_{task}_{dataset}_azureml.yml`                      | `config_classification_telecom_churn_azureml.yml` |
| Recipe file         | `recipe_{strategy}.yml` or `variant_*.yml`                 | `recipe_smote_target_standard.yml`         |
| Component YAML      | matches step script name                                   | `stage5_pycaret_train.yml`                 |

---

## 5  Immutable Contracts

These interfaces **must not change** without explicit approval:

1. **Component YAML I/O**: inputs/outputs defined in `components/*.yml`
2. **Step CLI args**: `argparse` arguments in each step script
3. **Pipeline entrypoint**: `pipelines/submit_pipeline.py` CLI interface
4. **Execution ID**: Generated in `pipeline_builder.py`, propagated unchanged
5. **Signal file format**: `StageSignal` dataclass fields in `src/utils/stage_signals.py`
