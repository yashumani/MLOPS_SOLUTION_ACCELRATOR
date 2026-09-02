# Drift Detection And Retraining Control Guide

> **Branch:** `codex_ys/mlops-pipeline-correctness`
> **Step IDs:** `s13` (drift monitor) and `s14` (retrain decision gate)
> **Feature Status:** Implemented and wired in source; current exact-source Azure pipeline acceptance is pending.
> **Current posture:** Drift alerts are non-blocking by default. `s13` emits drift and baseline artifacts, `s14` emits retrain decision artifacts, and Azure ML submission remains behind an explicit controller gate. The external controller resolves approved baselines and builds canonical submissions, while model promotion remains manual.

## Configured Thresholds

The standalone drift config is `configs/drift_config.yaml`.

| Drift type | Method | Threshold |
|---|---|---:|
| Feature drift | PSI mean across features | `0.15` |
| Prediction drift | Kolmogorov-Smirnov statistic | `0.10` |
| Concept drift | Accuracy drop versus baseline | `0.05` |
| Label drift | Chi-square complement | `0.10` |

Alert dispatch is best-effort and non-fatal. Missing Teams or email environment variables should not fail `s13`.


## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Position (DAG)](#2-pipeline-position-dag)
3. [Component I/O Contract](#3-component-io-contract)
4. [Execution Flow](#4-execution-flow)
5. [Algorithms & Metrics](#5-algorithms--metrics)
   - [PSI (Population Stability Index)](#51-psi-population-stability-index)
   - [Evidently Comparison Drift](#52-evidently-comparison-drift)
   - [Concept Drift](#53-concept-drift)
   - [Stability Score](#54-stability-score)
   - [Retraining Cadence](#55-retraining-cadence)
6. [Configuration](#6-configuration)
7. [Output Artifacts](#7-output-artifacts)
8. [MLflow Reporting](#8-mlflow-reporting)
9. [Baseline Chaining Across Runs](#9-baseline-chaining-across-runs)
10. [Standalone Drift Library](#10-standalone-drift-library)
11. [Test Coverage](#11-test-coverage)
12. [File Inventory](#12-file-inventory)
13. [Architecture Diagrams](#13-architecture-diagrams)
14. [Known Gaps & Future Work](#14-known-gaps--future-work)

---

## 1. Overview

Drift detection in the MLOps V3 pipeline is implemented as **step s13 (Drift Monitor)** with a separate **step s14 (Retrain Decision Gate)**. These are **training-time** steps, not a real-time production monitor. Their purpose is to:

1. **Establish a baseline profile** of the feature-engineered training data for future production monitoring.
2. **Validate the PSI drift detector** via a self-check (train/test split on the same data — expects PSI ≈ 0).
3. **Recommend a retraining cadence** (quarterly → weekly) based on dataset characteristics (size, complexity, volatility, imbalance).
4. **Compare against a previous baseline** (if provided) using Evidently and concept drift checks.
5. **Emit an auto-retrain policy decision** (`observe_only`, `refresh_baseline`, `candidate_retrain`, `promote_candidate`, or `blocked`) as a dedicated `s14` artifact for the external controller.

`s13` consumes the feature-engineered dataset from `s04` and the final evaluation report from `s10`, runs drift analysis, and outputs a drift report JSON plus a baseline artifact folder for chaining into future runs. `s14` consumes those `s13` outputs, applies the auto-retrain policy, and writes operator-facing decision artifacts without submitting another pipeline run.

---

## 2. Pipeline Position (DAG)

```
s01 → s02 → s03 → s04 ─┬─ [s05a, s05b] → s05z
(ingestion)  (prep)  (preproc)  (feat_eng)     (baselines)  (aggregate)
                        │
                        └──→ s06 → s08 → s09 → s10 → s12 → s13 → s14
                          (PhaseB) (HPO) (agg) (final_eval) (registry) (DRIFT) (RETRAIN DECISION)
                                                            ↑
                                                         TERMINAL STEP
```

**Key data flows into s13:**

| Input | Source Step | Content |
|-------|-----------|---------|
| `dataset_in` | **s04** (feature engineering) | Feature-engineered CSV — the processed training dataset |
| `final_report` | **s10** (final evaluation) | Champion model info, metrics, algorithm, phase selection |
| `baseline_in` *(optional)* | Previous pipeline run's `drift_baseline` output | Reference distributions, feature stats, reference CSV |

**s14 is the last step** — it depends on s13 and emits pipeline-level decision artifacts. `s13` outputs (`drift_report`, `drift_baseline`) remain pipeline-level outputs used for monitoring and chaining into subsequent runs.

---

## 3. Component I/O Contract

### Component YAML: `components/s13_drift_monitor.yml`

**Inputs:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `config_name` | `string` | Yes | Config YAML filename (e.g., `config_classification_telecom_churn_azureml.yml`) |
| `dataset_in` | `uri_file` | Yes | Processed dataset CSV from s04 feature engineering |
| `final_report` | `uri_file` | Yes | Final evaluation report JSON from s10 |
| `registry_info` | `uri_file` | No | Model registration info JSON from s12 |
| `baseline_in` | `uri_folder` | No | Previous run's drift baseline folder (enables comparison drift) |

**Outputs:**

| Name | Type | Description |
|------|------|-------------|
| `drift_report` | `uri_file` | JSON evidence with PSI scores, stability assessment, comparison drift results, cadence, alerts, and warnings; no policy decision or submission result |
| `drift_baseline` | `uri_folder` | Folder containing `feature_baseline.json`, `reference_distributions.json`, `reference_data.csv` |

### CLI Arguments (s13_drift_monitor.py)

```
python src/steps/s13_drift_monitor.py \
  --config_name <config.yml> \
  --dataset_in <path/to/dataset.csv> \
  --final_report <path/to/final_report.json> \
  [--registry_info <path/to/registry_info.json>] \
  [--baseline_in <path/to/previous_baseline_folder>] \
  --drift_report <output/drift_report.json> \
  --drift_baseline <output/drift_baseline/>
```

### Component YAML: `components/s14_retrain_decision.yml`

**Inputs:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `config_name` | `string` | Yes | Config YAML filename |
| `drift_report` | `uri_file` | Yes | Drift analysis report JSON from s13 |
| `candidate_baseline` | `uri_folder` | Yes | Drift baseline folder emitted by s13 for possible future approval |
| `final_report` | `uri_file` | No | Final evaluation report JSON from s10 |
| `registry_info` | `uri_file` | No | Model registration info JSON from s12 |
| `drift_policy_config` | `string` | Yes | Repository-relative policy YAML consumed by s14 |
| `trigger` | `string` | No | Decision trigger label, default `pipeline_s14` |

**Outputs:**

| Name | Type | Description |
|------|------|-------------|
| `retrain_decision` | `uri_file` | Auto-retrain decision JSON for operator review |
| `decision_ledger_record` | `uri_file` | Ledger-shaped decision JSON to append after approval or controller processing |

`s14` does not submit Azure ML jobs and does not approve baselines. It emits evidence for the external controller and manual promotion workflow.

---

## 4. Execution Flow

When s13 runs, it follows this sequence:

### Step 1: Load Configuration
- Reads the pipeline config YAML to extract `task_type`, `target_column`, `dataset_name`.
- Uses the workspace-provided `azureml://` MLflow tracking URI unchanged with `azureml-mlflow` and disables conflicting autolog behavior where required.

### Step 2: Load Upstream Artifacts
- Loads `final_report` JSON (champion model info, primary metric, algorithm).
- Optionally loads `registry_info` JSON (model registration status).

### Step 3: Load Dataset
- Reads the feature-engineered CSV from s04.
- Auto-detects delimiter (comma, tab, semicolon).
- Separates features (`X`) from target (`y`) based on task type.
  - **Classification/Regression:** Drops `target_column` from `X`, keeps it as `y`.
  - **Clustering:** No target column — `X` = full dataset, `y` = None.

### Step 4: Train/Test Split (Self-Check)
- Splits the training-only monitoring input **80/20** with `random_state=42` for detector smoke testing. This is not the Stage 2 locked test and is not model-selection evidence.
- For classification: uses `stratify=y`.
- Reference set = 80% (train), Test set = 20% (test).

### Step 5: PSI Self-Check
- Computes **per-feature PSI** between the reference and test splits.
- Since both splits come from the same distribution, PSI should be ≈ 0.
- This validates that the PSI detector is working correctly and there are no pathological features.
- Flags features where PSI ≥ 0.1 (green threshold).
- Self-check status: `PASS` if overall PSI < 0.1, else `WARN`.

### Step 6: Baseline Statistics
- Computes per-feature statistics for the full dataset:
  - **Numeric:** mean, std, min, max, quantiles (p5, p25, p50, p75, p95), missing rate.
  - **Categorical:** value counts (top 50 categories), n_unique, missing rate.

### Step 7: Feature Volatility
- Computes the **mean coefficient of variation** (CV = std / |mean|) across all numeric features.
- Higher CV → more volatile features → less stable model.

### Step 8: Stability Score & Cadence
- Combines 5 weighted components into a **stability score (0–100)**.
- Maps the score to a **retraining cadence recommendation** (quarterly → weekly).

### Step 9: Comparison Drift (Optional)
If a `baseline_in` folder from a previous run is provided:
- **Evidently Feature Drift:** Runs `DataDriftPreset` comparing the previous reference data against the current dataset. Detects per-column statistical drift using appropriate tests (KS for numeric, chi-squared for categorical).
- **Concept Drift:** Compares the champion model's primary metric from the current run against the previous run's metric. Flags if the drop exceeds 5%.

### Step 10: Generate Evidence Outputs
- Writes `drift_report.json` with all analysis results.
- Writes `drift_baseline/` folder with:
  - `feature_baseline.json` — per-feature statistics + champion metadata.
  - `reference_distributions.json` — histogram bin edges/counts per numeric feature, category counts per categorical feature.
  - `reference_data.csv` — the 80% reference split (used by Evidently in the next run's comparison).

### Step 11: MLflow Logging
- Logs metrics: `overall_psi`, `max_feature_psi`, `stability_score`, `recommended_days`, etc.
- Logs params: `detector`, `cadence`, `self_check_status`, `dataset_name`, `baseline_comparison`.
- If comparison drift was run: logs `evidently_dataset_drift`, `evidently_drifted_share`, `concept_drift_detected`, `concept_drift_drop`.

---

## 5. Algorithms & Metrics

### 5.1 PSI (Population Stability Index)

PSI measures how much a feature's distribution has shifted between two datasets.

**Formula:**

$$
PSI = \sum_{i=1}^{n} (P_i^{test} - P_i^{ref}) \times \ln\left(\frac{P_i^{test}}{P_i^{ref}}\right)
$$
Focused drift and orchestration tests cover the standalone drift library plus auto-retrain policy, ledger, and controller planning:

Focused drift and orchestration tests cover the standalone drift library plus auto-retrain policy, ledger, and controller planning:
Where $P_i^{ref}$ and $P_i^{test}$ are proportions of observations in bin $i$ for the reference and test sets.

| `test_auto_retrain_policy.py` | Tests policy outcomes for stable reports, missing baselines, drift signals, and promotion gating |
| `test_auto_retrain_decision_ledger.py` | Tests append/load behavior and latest approved baseline URI resolution |
| `test_auto_retrain_controller.py` | Tests controller baseline resolution, canonical command generation, and submitted job-name parsing |
**Implementation (`src/utils/drift_detector.py`):**

- **Numeric features:** 10 equal-width bins from reference distribution. Bin edges extended to capture out-of-range test values. Laplace smoothing ($\epsilon = 10^{-6}$) prevents log(0).
- **Categorical features:** Each unique category is a bin. Union of categories from both sets.

**Thresholds (industry standard):**

| PSI Range | Classification | Color |
|-----------|---------------|-------|
| < 0.1 | No significant drift | 🟢 GREEN |
| 0.1 – 0.25 | Moderate drift (warning) | 🟡 YELLOW |
| > 0.25 | Significant drift (action required) | 🔴 RED |

### 5.2 Evidently Comparison Drift

When a previous baseline is provided, Evidently's `DataDriftPreset` runs statistical tests on every feature:

- **Numeric features:** Kolmogorov-Smirnov test (by default).
- **Categorical features:** Chi-squared test (by default).
- **Dataset-level:** Flags overall dataset drift if >50% of features are individually drifted.

Outputs include:
- `dataset_drift` (bool) — overall drift flag.
- `share_of_drifted_columns` (float) — proportion of features that drifted.
- `drifted_columns` (list) — individual drifted columns with drift scores and test names.

### 5.3 Concept Drift

Concept drift occurs when the relationship between features and the target changes — even if the feature distributions haven't.

**Detection method:** Compare the champion model's primary metric between the current and previous run.

```
drop = baseline_metric - current_metric
detected = drop > 0.05
```

- **Classification:** Compares `balanced_accuracy`.
- **Regression:** Compares `r2_score`.
- **Threshold:** 5% absolute drop triggers concept drift alert.

### 5.4 Stability Score

A composite score (0–100) that estimates how stable the dataset is and how likely it is to drift in production.

**Components:**

| Component | Weight | Scoring Logic |
|-----------|--------|---------------|
| **Self-check PSI** | 40% | $100 \times (1 - \frac{mean\_PSI}{0.25})$, clamped to [0, 100] |
| **Dataset size** | 20% | $\frac{\log_{10}(n_{rows}) - 2}{4} \times 100$, clamped to [0, 100] |
| **Feature complexity** | 20% | $100 \times (1 - \min(\frac{n_{features}}{n_{rows}} \times 100, 1))$, clamped to [0, 100] |
| **Class balance** | 10% | $imbalance\_ratio \times 100$ (min/max class ratio). Neutral (75) for clustering/regression |
| **Feature volatility** | 10% | $100 \times (1 - \min(\frac{mean\_CV}{2}, 1))$, clamped to [0, 100] |

**Final score:**

$$
stability = 0.40 \times PSI_{score} + 0.20 \times Size_{score} + 0.20 \times Complexity_{score} + 0.10 \times Balance_{score} + 0.10 \times Volatility_{score}
$$

### 5.5 Retraining Cadence

The stability score maps directly to a retraining cadence recommendation:

| Score Range | Cadence | Days | Rationale |
|-------------|---------|------|-----------|
| ≥ 80 | Quarterly | 90 | Stable dataset with low drift risk. Large data volume and consistent distributions. |
| ≥ 60 | Monthly | 30 | Moderate complexity with some feature volatility. |
| ≥ 40 | Biweekly | 14 | Notable feature complexity or volatility detected. |
| < 40 | Weekly | 7 | High drift risk due to small dataset, high volatility, or class imbalance. |

---

## 6. Configuration

### Pipeline Config (in `configs/config_*.yml`)

Drift behavior is controlled by the pipeline config's `drift_monitoring` section:

```yaml
drift_monitoring:
  enabled: true
  psi_green: 0.1          # Below this = stable (no drift)
  psi_yellow: 0.25        # Below this = warning, above = significant drift
  concept_drift_threshold: 0.05  # Accuracy drop threshold for concept drift
  cadence_override: null   # Override auto-cadence with fixed interval (days)
```

### Standalone Drift Config (`configs/drift_config.yaml`)

For the standalone drift library (`src/drift_detection/`):

```yaml
drift_methods:
  feature: "psi"                     # PSI for feature distributions
  prediction: "ks"                   # KS test for prediction column
  concept: "accuracy_threshold"      # Accuracy-drop for concept drift
  label: "chi_square"                # Chi-squared for label distribution

thresholds:
  feature_drift: 0.15
  prediction_drift: 0.10
  concept_drift_accuracy_drop: 0.05
  label_drift: 0.10

schedule:
  frequency: "daily"
  time: "02:00"                      # UTC

actions:
  on_drift_detected: "trigger_full_pipeline" # Legacy action label; active s13 emits evidence only
  alert_channels: [email, mlflow_dashboard]

auto_retrain:
  enabled: false                    # External-controller compatibility; s13/s14 never submit
  mode: "dry_run"                   # Controller planning mode, not an s13 execution mode
  config_path: null                 # Controller input when that compatibility path is used
  compute: null                     # Controller input; submit_pipeline.py may use env fallback
  drift_baseline_in: null           # Optional previous s13 drift_baseline uri_folder
  force: false                      # Keep duplicate-submission guards enabled by default

artifact_paths:
  baseline_dir: "outputs/drift_baseline"
  reports_dir: "outputs/drift_reports"
  logs_dir: "outputs/drift_logs"

column_mapping:
  prediction_column: "prediction"
  target_column: null                # Overridden at runtime
  id_column: null
```

---

## 7. Output Artifacts

### drift_report.json

```json
{
  "execution_id": "s13_telecom_churn_1704067200",
  "config_name": "config_classification_telecom_churn_azureml.yml",
  "task_type": "classification",
  "dataset_name": "telecom_churn",
  "n_rows": 7043,
  "n_features": 19,
  "target_column": "Churn",
  "detector": "psi",
  "smoke_test": {
    "method": "train_test_split_80_20_seed_42",
    "overall_psi": 0.001234,
    "max_feature_psi": 0.008567,
    "max_feature_name": "TotalCharges",
    "drifted_features": [],
    "n_drifted": 0,
    "status": "PASS"
  },
  "feature_psi_scores": {
    "TotalCharges": 0.008567,
    "MonthlyCharges": 0.005432,
    "tenure": 0.003210,
    ...
  },
  "stability_assessment": {
    "stability_score": 82,
    "components": {
      "self_check_psi": {"raw": 0.001234, "score": 99.5, "weight": 0.40},
      "dataset_size": {"raw": 7043, "score": 73.2, "weight": 0.20},
      "feature_complexity": {"raw": 0.002698, "score": 99.7, "weight": 0.20},
      "class_balance": {"raw": 0.3652, "score": 36.5, "weight": 0.10},
      "feature_volatility": {"raw": 0.8421, "score": 57.9, "weight": 0.10}
    },
    "recommended_cadence": "quarterly",
    "recommended_days": 90,
    "rationale": "Stable dataset with low drift risk..."
  },
  "champion_info": {
    "algorithm": "GradientBoostingClassifier",
    "primary_metric": 0.912,
    "phase": "phase_b_pycaret",
    "registered": true,
    "model_name": "telecom_churn_champion",
    "model_version": "3"
  },
  "comparison_drift": {
    "available": false,
    "baseline_status": "not_provided",
    "baseline_metadata": {}
  },
  "warnings": [],
  "runtime_seconds": 12.5
}
```

### drift_baseline/ folder

| File | Content |
|------|---------|
| `feature_baseline.json` | Per-feature statistics (mean, std, min, max, quantiles, missing rate), champion metric & algorithm, PSI bin count, reference split method |
| `reference_distributions.json` | For each numeric feature: histogram bin edges + counts. For each categorical feature: category → count mapping. Used for PSI recomputation in production. |
| `reference_data.csv` | The 80% reference split of the feature-engineered data. Used by Evidently in the next run's comparison drift check. |

---

## 8. MLflow Reporting

All drift metrics and params are logged to the step's MLflow run within the parent pipeline run.

### Metrics

| Metric | Description |
|--------|-------------|
| `overall_psi` | Mean PSI across all features (self-check) |
| `max_feature_psi` | Highest individual feature PSI (self-check) |
| `stability_score` | Composite stability score (0–100) |
| `recommended_days` | Retraining cadence in days |
| `n_features_monitored` | Number of features profiled |
| `n_drifted_features` | Number of features with PSI ≥ 0.1 |
| `evidently_dataset_drift` | 1 if Evidently detected overall dataset drift, 0 otherwise (only when baseline provided) |
| `evidently_drifted_share` | Share of drifted columns per Evidently (only when baseline provided) |
| `concept_drift_detected` | 1 if concept drift detected, 0 otherwise (only when baseline provided) |
| `concept_drift_drop` | Absolute metric drop from baseline (only when baseline provided) |

### Parameters

| Param | Description |
|-------|-------------|
| `detector` | Always `"psi"` |
| `cadence` | Recommended cadence name (quarterly/monthly/biweekly/weekly) |
| `self_check_status` | `"PASS"` or `"WARN"` |
| `dataset_name` | Name of the dataset |
| `baseline_comparison` | `"true"` or `"false"` |

---

## 9. Baseline Chaining Across Runs

To enable comparison drift (detecting distribution shift between training runs), pass the previous run's reusable `drift_baseline` output URI to the next run via `--drift_baseline_in`:

```bash
# First run (no baseline — self-check only)
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id <sub> --resource_group <rg> --workspace_name <ws> \
  --compute <AZURE_COMPUTE> --wait

# Second run (with baseline from first run)
python pipelines/submit_pipeline.py \
  --config configs/config_classification_telecom_churn_azureml.yml \
  --subscription_id <sub> --resource_group <rg> --workspace_name <ws> \
  --compute <AZURE_COMPUTE> --wait \
  --drift_baseline_in <azureml_uri_folder_for_previous_drift_baseline>
```

When `--drift_baseline_in` is provided, `submit_pipeline.py` wraps the URI as an Azure ML `uri_folder` input and passes it as the `baseline_in` input to s13. This enables:

- **Evidently comparison drift** between the previous reference data and the current dataset.
- **Concept drift** detection by comparing model metrics across runs.

Without `--drift_baseline_in`, s13 runs in **self-check only mode** — no comparison drift is performed, and `comparison_drift.available` will be `false` in the report.

The FastAPI submit endpoint also accepts `baseline_job`. It resolves `outputs.drift_baseline.path` from the previous Azure ML job and passes that path to the pipeline. If a previous job has a downloadable `drift_baseline` artifact but Azure ML does not expose a reusable output path, the API returns `baseline_output_path_unavailable` from `/api/v1/pipelines/baseline/capture` and rejects baseline-chained submit with HTTP 400. This prevents accidental unchained validation runs.

Historical Azure finding (May 2026): Azure ML job metadata can omit `outputs.drift_baseline.path`, while `az ml job download --output-name drift_baseline` can expose the underlying datastore URI in its download output. Treat any recovered URI as evidence for that exact historical job only.

Historical second-cycle proof completed on 2026-05-16 with regression job `loyal_owl_0h0rz9krcn`. That earlier revision combined policy/trigger fields inside `drift_report`; the active contract no longer does so. The job remains baseline-comparison history, not current-source S13/S14 or deployment proof.

Current classification, regression, and clustering canonical SDK dry-runs emit separate `s13` and `s14` artifacts. An exact-source Azure canary attempted on 2026-08-02 was rejected before job creation by `ReadOnlyDisabledSubscription`; no current-revision Azure runtime claim is valid until that external blocker is cleared and outputs are downloaded.

The controller entrypoint is `scripts/run_auto_retrain_controller.py`. It resolves the latest approved baseline from the JSONL decision ledger, checks for active jobs, builds the canonical `pipelines/submit_pipeline.py` command, and appends a pending decision record only after a successful submit.

---

## 10. Standalone Drift Library

In addition to the pipeline step, the codebase includes a **standalone drift detection library** at `src/drift_detection/`. This is an importable compatibility package and is **separate from and not imported by** the pipeline step. Its legacy `PipelineTrigger` execution path is not an approved operational submitter; active submissions require an S14 decision and the external controller.

### Library Components

| Module | Class/Function | Purpose |
|--------|---------------|---------|
| `drift_config.py` | `DriftConfig` | Typed dataclass config (6 nested dataclasses) loaded from `configs/drift_config.yaml` |
| `baseline_capture.py` | `BaselineCapture` | Capture, save (parquet + JSON), and load reference baselines |
| `drift_checker.py` | `DriftChecker` | 4 drift checks via Evidently: feature, prediction, concept, label |
| `drift_checker.py` | `DriftResult` | Dataclass for individual check results (type, detected, score, details) |
| `pipeline_trigger.py` | `PipelineTrigger` | Evaluate drift results → determine if retraining should trigger |
| `report_generator.py` | `ReportGenerator` | Generate timestamped Evidently HTML drift reports |
| `synthetic_data_generator.py` | `generate_drifted_data()` | Generate test data with none/mild/severe drift presets |

### Library Usage Pattern

```python
from drift_detection import DriftConfig, BaselineCapture, DriftChecker, PipelineTrigger

# Load config
config = DriftConfig.from_yaml("configs/drift_config.yaml")

# Capture baseline from training data
baseline = BaselineCapture(config)
baseline.capture(train_df, target_column="Churn", prediction_column="prediction")
baseline.save("outputs/drift_baseline")

# Check drift on new production data
checker = DriftChecker(config, baseline)
results = checker.run_all_checks(prod_df)
# Returns: [DriftResult(type="feature", detected=True, score=0.18, ...), ...]

# Produce a legacy trigger recommendation only; do not dispatch from this path
trigger = PipelineTrigger(config)
action = trigger.evaluate(results)
# Returns: {"should_trigger": True, "action": "trigger_full_pipeline", "execution": {...}}
```

### Drift Types in Library

| Drift Type | Method | What It Detects |
|------------|--------|----------------|
| **Feature drift** | PSI via Evidently `DataDriftPreset` | Input feature distribution shift |
| **Prediction drift** | KS test via Evidently `ColumnDriftMetric` | Model output distribution shift |
| **Concept drift** | Accuracy drop comparison | Degradation in model-target relationship |
| **Label drift** | Chi-squared via Evidently `ColumnDriftMetric` | Target variable distribution shift |

---

## 11. Test Coverage

Four test files in `tests/test_drift_detection/`:

| Test File | What It Tests |
|-----------|--------------|
| `test_synthetic_data.py` | `generate_drifted_data()`: shape preserved, distribution shift correct for presets, deterministic seeds, edge cases |
| `test_baseline_capture.py` | `BaselineCapture`: capture stores reference, stats computed correctly, save/load round-trip (parquet + JSON), feature_columns excludes prediction/target |
| `test_drift_checker.py` | `DriftChecker` + `DriftResult`: all 4 drift types detected correctly, `run_all_checks()` returns 4 results, DriftResult fields populated |
| `test_pipeline_trigger.py` | `PipelineTrigger`: no trigger when no drift, trigger when drift detected (action = trigger_full_pipeline), dry_run flag, trigger_history tracking, save_trigger_log JSON output |
| `test_auto_retrain_policy.py` | Pure policy outcomes for stable reports, missing baseline, severe feature drift, concept drift, and explicitly allowed promotion |
| `test_auto_retrain_decision_ledger.py` | Append/load behavior and latest approved baseline URI resolution |
| `test_auto_retrain_controller.py` | Controller baseline resolution, canonical command generation, and submitted job-name parsing |

---

## 12. File Inventory

### Pipeline Step (runs in Azure ML)

| File | Lines | Role |
|------|-------|------|
| `src/steps/s13_drift_monitor.py` | ~650 | Main step: PSI self-check, Evidently comparison, stability scoring, MLflow logging |
| `components/s13_drift_monitor.yml` | ~40 | Azure ML component definition (v2, non-deterministic) |
| `src/utils/drift_detector.py` | ~335 | Core PSI computation, baseline stats, stability score, retraining cadence |
| `src/orchestration/auto_retrain_policy.py` | ~240 | Pure auto-retrain policy decision layer; no Azure side effects |
| `src/orchestration/auto_retrain_decision_ledger.py` | ~170 | Append-only decision ledger helpers for baseline URI lineage and controller audit records |
| `src/orchestration/auto_retrain_controller.py` | ~230 | Controller planning utilities: approved-baseline resolution, canonical submit command construction, pending decision record generation |
| `scripts/run_auto_retrain_controller.py` | ~170 | CLI wrapper for dry-run/submit controller cycles with Azure active-job checks |

### Standalone Library

| File | Lines | Role |
|------|-------|------|
| `src/drift_detection/__init__.py` | 28 | Package API: exports 7 public symbols |
| `src/drift_detection/drift_config.py` | ~155 | Typed dataclasses for YAML config |
| `src/drift_detection/baseline_capture.py` | ~210 | Capture/persist/load reference baselines |
| `src/drift_detection/drift_checker.py` | ~310 | 4 drift checks via Evidently |
| `src/drift_detection/pipeline_trigger.py` | ~140 | Evaluate drift results → trigger retraining |
| `src/drift_detection/report_generator.py` | ~95 | Generate Evidently HTML drift reports |
| `src/drift_detection/synthetic_data_generator.py` | ~175 | Generate test data with controlled drift |

### Configuration

| File | Role |
|------|------|
| `configs/drift_config.yaml` | Standalone drift config: methods, thresholds, schedule, actions |
| `config/sample_config.yaml` | Pipeline config with `drift_monitoring` section |
| `config/production_config.yaml` | Production pipeline config with `drift_monitoring` section |

### Tests

| File | Role |
|------|------|
| `tests/test_drift_detection/test_synthetic_data.py` | Tests synthetic data generation |
| `tests/test_drift_detection/test_baseline_capture.py` | Tests baseline capture/save/load |
| `tests/test_drift_detection/test_drift_checker.py` | Tests all 4 drift types |
| `tests/test_drift_detection/test_pipeline_trigger.py` | Tests trigger evaluation |

---

## 13. Architecture Diagrams

### Pipeline Step Flow

```
                    ┌────────────────────────────────────────────────────────────┐
                    │  s13_drift_monitor.py                                      │
                    │                                                            │
  dataset_in ──────►│  1. Load config (task_type, target_column)                │
  (from s04)        │  2. Load final_report (champion info, metrics)             │
                    │  3. Load dataset CSV → split X / y                        │
  final_report ────►│  4. Train/Test split (80/20, seed=42)                     │
  (from s10)        │  5. ┌──── PSI Self-Check ────────────────────┐            │
                    │     │  compute_feature_psi(X_ref, X_test)    │            │
                    │     │  → per-feature PSI scores               │            │
                    │     │  → self_check: PASS/WARN               │            │
                    │     └────────────────────────────────────────┘            │
                    │  6. compute_baseline_statistics(X)                        │
                    │  7. compute_feature_volatility(X)                         │
                    │  8. compute_stability_score() → score 0-100              │
                    │  9. determine_retraining_cadence() → quarterly/weekly     │
                    │                                                            │
  baseline_in ─────►│  10. If baseline provided:                                │
  (optional,        │      ├── Evidently DataDriftPreset(ref vs current)        │
  from prev run)    │      └── Concept drift (metric comparison)               │
                    │                                                            │
                    │  11. Write drift_report.json                              │──► drift_report
                    │  12. Write drift_baseline/                                │──► drift_baseline
                    │      ├── feature_baseline.json                            │
                    │      ├── reference_distributions.json                     │
                    │      └── reference_data.csv                               │
                    │  13. Log to MLflow (metrics + params)                     │
                    └────────────────────────────────────────────────────────────┘
```

### Baseline Chaining

```
  Run 1 (no baseline)                  Run 2 (with baseline from Run 1)
  ════════════════════                 ════════════════════════════════

  s04 → ... → s10 → s12 → s13         s04 → ... → s10 → s12 → s13
                      │                                     │
                      ▼                                     ▼
                drift_baseline/ ──────────────────► baseline_in
                ├── feature_baseline.json                   │
                ├── reference_distributions.json            ├── Evidently comparison
                └── reference_data.csv                      ├── Concept drift check
                                                            └── drift_report (with comparison results)
```

### Standalone Library Flow

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  src/drift_detection/                                               │
  │                                                                     │
  │  DriftConfig.from_yaml("configs/drift_config.yaml")                │
  │        │                                                            │
  │        ▼                                                            │
  │  BaselineCapture.capture(train_df)                                 │
  │        │                                                            │
  │        ├──► .save("outputs/drift_baseline")                        │
  │        │    ├── reference_data.parquet                              │
  │        │    └── baseline_stats.json                                │
  │        │                                                            │
  │        ▼                                                            │
  │  DriftChecker.run_all_checks(prod_df)                              │
  │        ├── Feature drift  (PSI via Evidently DataDriftPreset)      │
  │        ├── Prediction drift (KS test via ColumnDriftMetric)        │
  │        ├── Concept drift  (accuracy drop comparison)               │
  │        └── Label drift    (chi-squared via ColumnDriftMetric)      │
  │              │                                                      │
  │              ▼                                                      │
  │  PipelineTrigger.evaluate(results)  [standalone compatibility]     │
  │        └── recommendation only; active submission requires         │
  │            s14 policy output + external controller                 │
  │                                                                     │
  │  ReportGenerator.generate(results) → Evidently HTML report         │
  │  generate_drifted_data() → Synthetic test data (none/mild/severe)  │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 14. Known Gaps & Future Work

1. **Two parallel implementations:** The pipeline step (`s13` + `drift_detector.py`) and the standalone library (`src/drift_detection/`) overlap in functionality but are not integrated. The step does not import from the library. A future refactor should unify them so the step delegates to the library.

2. **Legacy standalone trigger path:** `PipelineTrigger._execute_trigger()` can still construct a submission in the standalone library, but that violates the active S13/S14/controller ownership contract. Keep it disabled, do not use it operationally, and retire or route it through an immutable S14 decision before release.

3. **Blueprint CSV is historical:** The `docs/MLOPS-v3-blueprint.CSV` predates the current `s13 -> s14` terminal flow and should not override `PIPELINE_STAGES.md` or `pipeline_builder.py`.

4. **Prediction drift & label drift only in standalone library:** The pipeline step only checks feature drift (PSI) and concept drift (metric comparison). Prediction drift (KS test on model outputs) and label drift (chi-squared on target) are only available in the standalone library.

5. **Schedule truth is external:** May 2026 schedule names are historical records, not proof that schedules are currently enabled or correctly configured. Operators must query live Azure schedule state; any active schedule must preserve the S13 evidence -> S14 decision -> external controller ownership boundary and manual promotion.

6. **Evidently API compatibility:** The code uses `evidently.legacy.*` fallback imports, indicating compatibility with Evidently 0.5.x while using the 0.4.x API surface. This should be updated when the environment stabilizes on a single Evidently version.

7. **Clustering-specific drift:** For clustering tasks (`task_type == "clustering"`), drift detection works on features only (no target, no concept drift, no label drift). Feature PSI and stability scoring still apply. Class balance component defaults to neutral (75) for clustering.

---

## 15. API Integration (post-V3 production)

The drift report produced by `s13` is now consumed by the FastAPI layer at:

```
GET /api/v1/pipelines/jobs/{job_name}/drift
Header: X-API-Key: <key>
```

Implementation: [api/services/pipeline_service.py](../api/services/pipeline_service.py) → `get_job_drift(job_name)` (lines ~727–905).

### Producer → Consumer field mapping

| s13 producer key | API `DriftResponse` field | Notes |
|------------------|---------------------------|-------|
| `feature_psi_scores: {feature: float}` | `features: list[DriftResultItem]` | sorted desc by PSI; severity assigned by threshold |
| `self_check.status` (`PASS`/`WARN`) | `overall_drift_detected` | when no comparison baseline available |
| `self_check.drifted_features` | `drifted_columns` | fallback when Evidently absent |
| `comparison_drift.evidently.dataset_drift` | `overall_drift_detected` | takes precedence when `comparison_drift.available == true` |
| `comparison_drift.evidently.drifted_columns` | `drifted_columns` | preferred source |
| `stability_assessment.stability_score` | `stability_score` | passed through |
| (resolved) | `drift_type` | `comparison` → `self_check` → `psi` |

The API joins the producer-side `s13` `drift_report` with the separate `s14` `retrain_decision` artifact and verifies their execution/decision identity. `s13` never emits or dispatches a retraining policy result.

### Severity thresholds (PSI)

| Range | Severity | `drift_detected` |
|-------|----------|------------------|
| `≥ 0.25` | `severe` | `true` |
| `≥ 0.10` | `moderate` | `true` |
| else | `none` | `false` |

### Verified example response (`happy_owl_sfmkgs2jrd`, regression_insurance_v3)

```jsonc
{
  "job_name": "happy_owl_sfmkgs2jrd",
  "overall_drift_detected": false,
  "stability_score": 65.0,
  "drift_type": "self_check",
  "drifted_columns": [],
  "features": [
    {"feature": "age", "psi": 0.027702, "drift_detected": false, "severity": "none"},
    {"feature": "bmi", "psi": 0.025250, "drift_detected": false, "severity": "none"},
    "... 9 more ..."
  ],
  "evidently_report_path": null,
  "studio_url": "https://ml.azure.com/runs/happy_owl_sfmkgs2jrd?wsid=..."
}
```

### Bugs caught & fixed during integration

| # | Symptom | Root cause | Fix |
|---|---------|------------|-----|
| 1 | API returned wrong/empty fields | Parser written against pre-V3 keys (`psi_scores` instead of `feature_psi_scores`) | Rewrote parser with 3-level fallback chain + nested-dict per-feature value handling |
| 2 | Even after fix #1, all jobs returned `features: []` | `ml_client.jobs.download(output_name="drift_report")` writes a file literally named `drift_report` (no `.json` suffix). Parser's `tmp.rglob("*.json")` matched zero files | Now globs `*.json` first, then any extension-less file with parseable JSON content. Documented as a cautionary precedent for any new endpoint downloading named outputs. |

See [POST_V3_PRODUCTION_REPORT.md](POST_V3_PRODUCTION_REPORT.md) for the full launch report and [FASTAPI_INTEGRATION.md](FASTAPI_INTEGRATION.md) for the full API surface.
