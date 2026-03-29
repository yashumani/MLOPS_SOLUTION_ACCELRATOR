# AIM-Tournament: Adaptive Indicator-gated Multi-metric Tournament

## Overview

AIM-Tournament upgrades the V3 pipeline from single-metric champion selection
to a **multi-metric, statistically-gated** approach.  The key components are:

1. **Bundle Gating** — data-driven variant selection before training
2. **Multi-metric Evaluation** — every candidate is scored on all task-relevant metrics
3. **Pareto Frontier** — non-dominated solution identification across metrics
4. **Utility Ranking** — weighted rank-percentile aggregation for tie-breaking
5. **Model Universe** — explicit, enforceable model lists per engine

## Architecture

```
Dataset → Data Signals → Bundle Gating → Enabled Bundles
                                              ↓
                                    Variant Paths (N variants)
                                              ↓
                            Pipeline: Stage 5 (baseline) / Stage 6 (variants)
                                              ↓
                                    Candidate Ledger (all_candidates.csv)
                                              ↓
                            AIM-Tournament (final_evaluation.py)
                                  ↓         ↓         ↓
                            Per-metric   Pareto    Utility
                            Top-K CSVs   Frontier  Ranking
```

## Signals Computed

Signals are computed from the raw dataset in `src/utils/bundle_gating.py`:

| Signal | Description | Used For |
|--------|-------------|----------|
| `n_rows` | Total row count | Sample size gating |
| `n_features` | Feature count (excl. target) | Dimensionality gating |
| `missing_rate` | Mean missing value rate | Imputation selection |
| `max_cardinality` | Highest cardinality in categoricals | Encoding selection |
| `high_cardinality_cols` | Count of cols with >50 unique values | Encoding selection |
| `sparsity` | Fraction of zero values | Sparse-aware models |
| `mean_abs_skewness` | Average absolute skewness | Scaling selection |
| `max_abs_skewness` | Maximum absolute skewness | Outlier handling |
| `mean_kurtosis` | Average kurtosis | Distribution shape |
| `outlier_rate` | IQR-based outlier fraction | Outlier handling |
| `max_correlation` | Highest pairwise correlation | Feature selection |
| `high_corr_pairs` | Count of pairs with r > 0.85 | Collinearity gating |
| `imbalance_ratio` | min_class / max_class | (Classification) SMOTE |
| `minority_fraction` | min_class / n_total | (Classification) Sampling |
| `n_classes` | Number of classes | Multi-class handling |
| `sample_bucket` | tiny / small / medium / large | Runtime planning |

## Bundle Gating Rules

Bundles are defined in `configs/variant_bundles/<task_type>/*.yml`.
Each bundle has a `default_enabled` flag and optional `gating_rules`.

A bundle is **enabled** if:
- `default_enabled: true` (always on), **or**
- ALL of its `gating_rules` are satisfied against computed signals

Example rule:
```yaml
gating_rules:
  - signal: "imbalance_ratio"
    operator: "<"
    threshold: 0.3
    reason: "Class imbalance detected"
```

Decisions are logged to `outputs/signals/bundle_decisions.json`.

## Multi-metric Ranking

### Metrics per Task Type

| Classification | Regression | Clustering |
|---------------|------------|------------|
| accuracy | r2 | silhouette |
| roc_auc | rmse | davies_bouldin |
| f1 | mae | calinski_harabasz |
| precision | | |
| recall | | |
| logloss | | |

### Per-metric Top-K Tables

After the full pipeline, `outputs/topk/` contains:
- `top_10_accuracy.csv`
- `top_10_roc_auc.csv`
- `top_10_f1.csv`
- ... (one per metric)

### Pareto Frontier

A candidate is **Pareto-optimal** (non-dominated) if no other candidate is
strictly better in ALL metrics simultaneously.

- Algorithm: pairwise dominance check (exact, O(N²M))
- Output: `outputs/pareto_frontier.csv`, `outputs/pareto_summary.json`

### Utility Score

For final ranking, a weighted rank-percentile approach is used:

1. For each metric, compute dense rank (1 = best)
2. Convert rank to percentile: `1 - (rank - 1) / (N - 1)`
3. Multiply by metric weight and sum
4. Normalize to [0, 1]

Default weights can be overridden in config. If `primary_metric` is set,
that metric's weight is boosted by 50% (then re-normalised).

## Model Universe

`src/utils/model_universe.py` defines `MODEL_UNIVERSE` — a single dict
mapping `{task_type}_{engine}` to model ID lists.

- PyCaret: `include=` parameter in `compare_models()`
- FLAML: `estimator_list=` parameter in `automl.fit()`

Model coverage is reported in `outputs/model_coverage.json`.

## How to Interpret `outputs/all_candidates.csv`

### Identity Columns
- `candidate_id`, `stage`, `step_name`, `engine` — who is this candidate

### Metric Columns
- `accuracy`, `f1`, `precision`, `recall`, `roc_auc`, `logloss` (classification)
- `r2`, `rmse`, `mae` (regression)
- `silhouette`, `davies_bouldin`, `calinski_harabasz` (clustering)

### Tournament Columns (added by AIM-Tournament)
- `rank_<metric>` — per-metric dense rank (1 = best)
- `utility_score` — weighted rank-percentile aggregate
- `utility_rank` — rank by utility score
- `pareto_optimal` — True if non-dominated across all metrics

### Finding the Champion

```python
import pandas as pd

df = pd.read_csv("outputs/all_candidates_ranked.csv")

# Pareto-optimal candidates
pareto = df[df["pareto_optimal"] == True]
print(pareto[["candidate_id", "engine", "utility_score"]].sort_values("utility_score", ascending=False))

# Best by utility score
best = df.loc[df["utility_score"].idxmax()]
print(f"Champion: {best['candidate_id']} (utility={best['utility_score']:.4f})")
```

## Validation

After a pipeline run completes, validate outputs:

```bash
python scripts/validate_aim_tournament.py outputs/
```

This checks:
- `all_candidates.csv` exists and has required metric columns
- Per-metric top-K CSVs exist in `outputs/topk/`
- Pareto frontier report exists
- Bundle gating signals were recorded
- Model coverage report exists

## Backward Compatibility

- All AIM-Tournament artifacts are **additive** — existing outputs are unchanged
- `all_candidates.csv` retains its original schema; tournament columns are added
  to `all_candidates_ranked.csv`
- Bundle gating is **optional** — pass `--bundles_dir` to enable
- If AIM-Tournament fails, the pipeline continues normally
