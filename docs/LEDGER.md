# Candidate Ledger — V3 Unified Candidate Tracking

## Overview

The **Candidate Ledger** is a filesystem-first system that records every model candidate
evaluated across the entire V3 pipeline — baseline, Phase B variants, Phase C HPO trials,
and final evaluation — in a single consolidated CSV/Parquet table.

**Design principle**: No MLflow dependency. Treat filesystem outputs as the source of truth.

---

## Output Files

After a pipeline run, `outputs/` contains:

| File | Description |
|------|-------------|
| `s05a_candidates.csv` | PyCaret baseline candidates |
| `s05b_candidates.csv` | FLAML baseline candidates |
| `s05z_candidates.csv` | Aggregate baseline (pycaret vs flaml) |
| `s06_candidates.csv` | Phase B variant × engine results |
| `s07z_candidates.csv` | Phase B aggregate |
| `s08_candidates.csv` | Phase C HPO trial rows |
| `s09_candidates.csv` | Phase C aggregate passthrough |
| `s10_candidates.csv` | Final evaluation comparison (3 rows) |
| **`all_candidates.csv`** | **Merged ledger from all stages** |
| `all_candidates.parquet` | Parquet version (if pyarrow available) |
| `all_candidates_summary.json` | Machine-readable summary |
| `all_candidates_README.md` | Human-readable summary |

---

## Schema (30 columns)

### Identity Columns
| Column | Type | Description |
|--------|------|-------------|
| `dataset_id` | str | Dataset file stem or config name |
| `task_type` | str | classification / regression / clustering |
| `preset` | str | diagnostic / production |
| `pipeline_version` | str | Always "v3" |
| `stage` | str | baseline / phase_b / phase_c / final |
| `step_name` | str | s05a / s05b / s05z / s06 / s07z / s08 / s09 / s10 |
| `engine` | str | pycaret / flaml / optuna / phase name |
| `candidate_id` | str | Unique within stage (e.g. `flaml_xgboost`, `trial_42`) |
| `run_id` | str | Azure ML run ID (if available) |
| `timestamp_utc` | str | ISO 8601 UTC timestamp |

### Input Columns
| Column | Type | Description |
|--------|------|-------------|
| `recipe_name` | str | Recipe/variant ID |
| `recipe_hash` | str | SHA-256 of recipe YAML (provenance) |
| `params_json` | str | JSON-encoded hyperparameters |
| `pipeline_dims_json` | str | Preprocessing dimensions JSON |

### Output Columns (Metrics)
| Column | Type | Description |
|--------|------|-------------|
| `primary_metric_name` | str | Canonical metric name for task type |
| `primary_metric_value` | float | Value of primary metric |
| `accuracy` | float | Classification accuracy |
| `roc_auc` | float | ROC AUC |
| `f1` | float | F1 score |
| `precision` | float | Precision |
| `recall` | float | Recall |
| `logloss` | float | Log-loss |
| `rmse` | float | Root mean squared error |
| `mae` | float | Mean absolute error |
| `r2` | float | R² score |
| `mse` | float | Mean squared error |
| `silhouette` | float | Silhouette score (clustering) |
| `davies_bouldin` | float | Davies-Bouldin index (clustering) |
| `calinski_harabasz` | float | Calinski-Harabasz index (clustering) |

### Signal Columns
| Column | Type | Description |
|--------|------|-------------|
| `candidate_rank` | int | Rank within stage (1 = best) |
| `delta_vs_baseline_best` | float | Improvement over baseline best |
| `is_stage_best` | bool | True if best within stage |
| `is_final_champion` | bool | True for overall winner |
| `status` | str | ok / failed / timed_out / skipped |
| `failure_reason` | str | Error message if status ≠ ok |
| `compute_time_sec` | float | Wall-clock time for candidate |

### Provenance Columns
| Column | Type | Description |
|--------|------|-------------|
| `source_path` | str | Source script relative path |
| `artifacts_json` | str | JSON-encoded artifact paths |

---

## Metric Normalization

The `normalize_metrics()` function maps mixed-case engine-specific metric names
to canonical column names:

```python
# Classification
"Accuracy" / "accuracy" → accuracy + primary_metric_value
"AUC" / "roc_auc"       → roc_auc
"F1" / "f1"             → f1

# Regression
"R2" / "r2"                 → r2 + primary_metric_value
"RMSE" / "rmse"             → rmse
"MAE" / "mae"               → mae

# Clustering
"silhouette_score" / "Silhouette" → silhouette + primary_metric_value
"davies_bouldin_score"            → davies_bouldin
```

---

## Usage Examples

### Load merged ledger in pandas
```python
import pandas as pd
df = pd.read_csv("outputs/all_candidates.csv")

# Show champion
champ = df[df["is_final_champion"] == True]
print(champ[["stage", "engine", "candidate_id", "primary_metric_value"]])

# Compare stages
print(df.groupby("stage")["primary_metric_value"].describe())
```

### Load Parquet (faster for large ledgers)
```python
df = pd.read_parquet("outputs/all_candidates.parquet")
```

### Validate after a run
```bash
python scripts/validate_candidate_ledger.py outputs/
```

---

## Failure Tolerance

Every ledger write is wrapped in `try/except` — a ledger failure **never** crashes
the pipeline. If a stage ledger cannot be written, a warning is printed and
execution continues normally.

---

## Integration Points

| Step | Stage Script | CSV Output | Rows |
|------|---|---|---|
| s5a | `stage5_pycaret_train.py` | `s05a_candidates.csv` | 1 (best model) |
| s5b | `stage5_flaml_train.py` | `s05b_candidates.csv` | 1 (best model or skipped) |
| s5t | `stage5_timeseries_train.py` | (optional) | 1 if time-series |
| s5z | `aggregate_baseline.py` | `s05z_candidates.csv` | 2-3 (pycaret + flaml + timeseries) |
| s06 | `s06_phaseb_variant_runner.py` | `s06_candidates.csv` | N (variant × engine) |
| s08 | `phasec_optuna_hpo.py` | `s08_candidates.csv` | N (all HPO trials) |
| s09 | `s07_phase2_pipeline_attribution.py` | `s09_candidates.csv` | 1 (passthrough) |
| s10 | `final_evaluation.py` | `s10_candidates.csv` + **`all_candidates.csv`** | 3 + merged |

> **Note**: The legacy `aggregate_phaseb.py` / `s07z_candidates.csv` step is superseded
> by `s06_phaseb_variant_runner.py`, which produces its own leaderboard and champion
> selection internally. Phase B aggregation is now embedded in s06.
