# V3 Imputation Strategy Expansion Blueprint

**Date:** 2026-02-23  
**Status:** Blueprint Validated — On-disk library at 445 variants; Tier 1 methods ready for integration  
**Scope:** Expand imputation methods from 6 → 33 across all task types  

---

## 1. Current State Audit — The Problem

Across **445 variant YAML files** (330 variant_search + 112 v1_generated + 3 enterprise, as of v3-production branch), the pipeline uses only **6 imputation methods**:

| Method | Count | % of All Variants | What It Does |
|--------|------:|-------------------:|--------------|
| `mean` | 205 | 60.8% | Arithmetic average of available values |
| `knn` | 44 | 13.1% | K-Nearest Neighbors (uniform or distance-weighted) |
| `mode` | 42 | 12.5% | Most frequent value (categorical) |
| `median` | 29 | 8.6% | Middle value (robust to outliers) |
| `iterative` | 9 | 2.7% | MICE with Bayesian Ridge (one estimator) |
| `drop` | 7 | 2.1% | Drop rows with any missing value |

**By Task Type:**

| Task Type | mean | median | mode | knn | iterative | drop |
|-----------|-----:|-------:|-----:|----:|----------:|-----:|
| Classification | 183 | 6 | 16 | 16 | 2 | 2 |
| Regression | 15 | 16 | 16 | 21 | 6 | 2 |
| Clustering | 7 | 6 | 10 | — | — | 3 |

### What This Means

Every variant defaults to one of **three simple statistical aggregates** (mean, median, mode) or one of **three ML-based methods** (KNN, iterative, drop). This is the equivalent of a carpenter who only owns a hammer and three sizes of screwdriver.

A lead data scientist working with messy industrial data — telecom CDRs with dropped records, healthcare patient files with months of missing labs, financial transaction logs with system outages — would never restrict themselves to these 6 options. The missing value problem is the **single most impactful preprocessing decision** for model quality, yet we give it the least diversity.

---

## 2. What's Missing — Thinking Like a Lead Data Scientist

### 2.1 The Real-World Missing Data Taxonomy

Missing data isn't one problem. It's at least **five different problems**, each requiring a different family of solutions:

| Missing Data Pattern | Real-World Example | Current Coverage | Gap |
|---------------------|--------------------|:----------------:|:---:|
| **Random scatter** (MCAR) | Sensor glitch drops 3% of readings | ✅ mean/median | Adequate for low rates |
| **Correlated gaps** (MAR) | Income missing more for younger users | ⚠️ KNN only | Need group-aware, regression-based |
| **Structural absence** (MNAR) | Customer didn't fill "income" because unemployed | ❌ Nothing | Need indicator + model-based |
| **Temporal gaps** | 2 weeks of missing sensor data between good reads | ❌ Nothing | Need interpolation, forward/backward fill |
| **Block missingness** | Entire column missing for one data source | ❌ Only drop | Need conditional imputation, matrix methods |

### 2.2 Industry-Specific Analysis

**A lead data scientist doesn't pick imputation methods from a menu — they pick based on the DATA GENERATION PROCESS.**

---

#### 🔷 TELECOM (Churn, Network, CDR Data)

**Data Reality:** Call detail records have timestamps. When a cell tower goes down, you get blocks of missing data. Customer survey fields are sparse. Network metrics have temporal patterns.

**What We Should Use:**
| Method | Why | Example |
|--------|-----|---------|
| **Forward Fill (ffill)** | Last known plan/usage carries forward until changed | Customer's `monthly_plan` was $49 last month, missing this month = still $49 |
| **Rolling Mean** (window=7/30) | Smooth over recent history | `avg_call_duration` missing → use 7-day rolling average |
| **Group Mean** (by segment) | Customers within same plan behave similarly | `data_usage` missing → impute with mean of same `plan_type` |
| **Zero Fill** | No call/SMS = zero usage, not missing | `international_calls` missing in a given period = 0 calls |
| **Indicator + Mean** | Flag that the value was missing (model can learn the pattern) | `is_missing_tenure` = 1, then fill `tenure` with median |

---

#### 🔷 HEALTHCARE (Patient Records, Clinical Trials, Lab Results)

**Data Reality:** Lab tests are ordered selectively (a healthy patient doesn't get a cancer marker test — that's MNAR). Time between visits can be months. FDA requires specific imputation audit trails.

**What We Should Use:**
| Method | Why | Example |
|--------|-----|---------|
| **LOCF** (Last Observation Carried Forward) | Clinical trial gold standard | Patient's `blood_pressure` at visit 3 missing → carry forward visit 2 value |
| **NOCB** (Next Observation Carried Backward) | Complement to LOCF | New patient, first lab missing → use value from their second visit |
| **Predictive Mean Matching (PMM)** | Imputed values are always from actual observed distribution | `cholesterol` imputed as 245 (an actual observed value), not 244.7 (fake value) |
| **Group Median** (by condition) | Patients with same diagnosis have similar ranges | `hemoglobin` missing → median for patients with same `diagnosis_code` |
| **Multiple Imputation** (Rubin's) | Generate 5-10 imputed datasets, pool results | Required for regulatory submissions — single imputation understates uncertainty |
| **Indicator + Constant** | "Test not ordered" is informative | `tumor_marker` missing → flag as "not_ordered" + fill with normal range midpoint |

---

#### 🔷 FINANCE (Credit Scoring, Fraud, Transactions)

**Data Reality:** Missing income on a credit application is likely MNAR (applicant chose not to disclose = red flag). Transaction gaps during system outages need temporal interpolation. Fraud features are often sparse.

**What We Should Use:**
| Method | Why | Example |
|--------|-----|---------|
| **Indicator + Separate Model** | Missing-ness IS a feature (non-disclosure = risk) | `income_missing` = 1 → model learns non-disclosure correlates with default |
| **Winsorized Mean** | Trim extreme values before averaging (robust to fraud outliers) | `transaction_amount` → trim top/bottom 5%, then mean |
| **Group Mean by Risk Segment** | Borrowers in same segment behave similarly | `debt_to_income` missing → mean within same `risk_grade` |
| **Zero Fill** | No transaction = zero, not missing | `num_chargebacks` missing = 0 chargebacks |
| **EWMA** (Exponential Weighted Moving Average) | Recent behavior matters more than old | `monthly_spending` missing → EWMA with α=0.3 (recent months weighted higher) |
| **Regression Imputation** | Use correlated features to predict missing | `income` ≈ f(`job_title`, `zip_code`, `age`) with noise |

---

#### 🔷 RETAIL / E-COMMERCE (Sales, Customer Behavior, Inventory)

**Data Reality:** Sales data is seasonal (missing December data can't be filled with January averages). Customer browsing data is sparse. Inventory records have batch gaps.

**What We Should Use:**
| Method | Why | Example |
|--------|-----|---------|
| **Seasonal Decomposition** | Respect seasonal patterns | `december_sales` missing → use trend + last December's seasonal component |
| **Rolling Median** (window=4 weeks) | Robust to promotional spikes | `weekly_units_sold` → 4-week rolling median |
| **Hot Deck** | Replace with actual value from similar record | Customer A missing `avg_basket_size` → copy from Customer B with same `segment`, `region` |
| **Group Mode** (by category) | Product preferences cluster within categories | `preferred_payment` missing → mode within same `customer_segment` |
| **Linear Interpolation** | Smooth between known data points | Monthly `revenue` known for Jan and Mar, Feb missing → interpolate |

---

#### 🔷 MANUFACTURING / IoT (Sensor Data, Quality Metrics)

**Data Reality:** Sensors fail in bursts. Readings between known values should be smooth. Some sensors only report on change (missing = unchanged).

**What We Should Use:**
| Method | Why | Example |
|--------|-----|---------|
| **Linear Interpolation** | Sensor readings are physically continuous | `temperature` at 10:00 = 72°, 10:10 = 74° → 10:05 ≈ 73° |
| **Spline Interpolation** | Smoother than linear for physical processes | `vibration_frequency` follows smooth curves |
| **Forward Fill** | "No change reported" = same as last reading | `machine_status` missing = still running |
| **Rolling Mean** (window=10) | Smooth over sensor noise | `pressure_reading` → 10-sample rolling average |
| **Domain Constant** | Known physical limits | `temperature` missing → fill with operating range midpoint (25°C) |

---

#### 🔷 CUSTOMER SERVICE / HR / SURVEYS

**Data Reality:** Surveys have voluntary fields (MNAR). Likert scales cluster. Employee records have structured gaps (new hire has no performance history).

**What We Should Use:**
| Method | Why | Example |
|--------|-----|---------|
| **Hot Deck** | Preserve actual response distributions | `satisfaction_score` → copy from respondent with same `department` + `tenure_band` |
| **Predictive Mean Matching** | Imputed Likert values stay on scale (1-5) | `engagement_score` imputed as 4 (real value), not 3.7 (impossible on scale) |
| **Group Mode** (by department) | Survey responses cluster by organizational unit | `work_life_balance` missing → mode within `department` |
| **Constant (midpoint)** | Neutral assumption for Likert | Missing `satisfaction` → fill with 3 (midpoint of 1-5 scale) |
| **Indicator + Drop** | "Declined to answer" is informative | `salary_satisfaction` missing → flag + exclude from that variable's model |

---

## 3. Proposed Imputation Method Expansion

### Tier 1 — High-Priority (Immediate Value, Proven Methods)

These methods are well-established in sklearn/scipy, require minimal implementation effort, and cover the biggest gaps:

| # | Method Key | Full Name | Library | Numeric | Categorical | Industry Fit |
|---|-----------|-----------|---------|:-------:|:-----------:|-------------|
| 1 | `ffill` | Forward Fill | pandas | ✅ | ✅ | Telecom, IoT, Healthcare |
| 2 | `bfill` | Backward Fill | pandas | ✅ | ✅ | Healthcare (NOCB), IoT |
| 3 | `interpolate_linear` | Linear Interpolation | pandas | ✅ | — | IoT, Manufacturing, Finance |
| 4 | `zero` | Zero Fill | pandas | ✅ | — | Finance (count features), Retail |
| 5 | `constant` | Domain Constant Fill | pandas | ✅ | ✅ | All (with config parameter) |
| 6 | `indicator_mean` | Missing Indicator + Mean | pandas + sklearn | ✅ | — | Finance (MNAR), Healthcare |
| 7 | `indicator_median` | Missing Indicator + Median | pandas + sklearn | ✅ | — | Finance (MNAR), Healthcare |
| 8 | `group_mean` | Group-Based Mean | pandas groupby | ✅ | — | All (requires grouping column config) |
| 9 | `group_median` | Group-Based Median | pandas groupby | ✅ | — | Healthcare, Finance, Telecom |
| 10 | `group_mode` | Group-Based Mode | pandas groupby | — | ✅ | Retail, Customer Service, HR |

### Tier 2 — Medium-Priority (Stronger Methods, Moderate Complexity)

These require slightly more implementation but significantly expand capabilities for advanced use cases:

| # | Method Key | Full Name | Library | Numeric | Categorical | Industry Fit |
|---|-----------|-----------|---------|:-------:|:-----------:|-------------|
| 11 | `rolling_mean` | Rolling Window Mean | pandas | ✅ | — | Telecom, IoT, Finance |
| 12 | `rolling_median` | Rolling Window Median | pandas | ✅ | — | Retail, Manufacturing |
| 13 | `ewma` | Exponential Weighted Moving Average | pandas | ✅ | — | Finance, Telecom |
| 14 | `interpolate_spline` | Spline Interpolation | scipy | ✅ | — | Manufacturing, IoT |
| 15 | `winsorized_mean` | Winsorized Mean (trim extremes) | scipy | ✅ | — | Finance (fraud), Healthcare |
| 16 | `trimmed_mean` | Trimmed Mean (remove tails) | scipy | ✅ | — | Finance, Manufacturing QC |
| 17 | `hot_deck` | Hot Deck (copy from similar record) | custom | ✅ | ✅ | Surveys, HR, Retail |
| 18 | `pmm` | Predictive Mean Matching | custom | ✅ | — | Healthcare, Clinical Trials |
| 19 | `regression` | Regression-Based Imputation | sklearn | ✅ | — | Finance (credit), Healthcare |
| 20 | `stochastic_regression` | Regression + Random Noise | sklearn | ✅ | — | Healthcare (uncertainty), Research |

### Tier 3 — Advanced (Research-Grade, High Complexity)

These are for state-of-the-art performance on very challenging missing data patterns:

| # | Method Key | Full Name | Library | Numeric | Categorical | Industry Fit |
|---|-----------|-----------|---------|:-------:|:-----------:|-------------|
| 21 | `iterative_rf` | MICE with Random Forest | sklearn | ✅ | — | Any high-missing dataset |
| 22 | `iterative_et` | MICE with Extra Trees | sklearn | ✅ | — | Any high-missing dataset |
| 23 | `iterative_gb` | MICE with Gradient Boosting | sklearn | ✅ | — | Finance, Telecom |
| 24 | `matrix_factorization` | Low-Rank Matrix Completion | sklearn (NMF) | ✅ | — | Sparse data, recommendations |
| 25 | `multiple_imputation` | Rubin's Multiple Imputation (pool N datasets) | custom | ✅ | ✅ | Clinical trials, regulatory |
| 26 | `missforest` | Random Forest-Based (MissForest algorithm) | custom/missingpy | ✅ | ✅ | Any complex dataset |
| 27 | `em_imputation` | Expectation-Maximization | custom | ✅ | — | Research, mixed-type data |

---

## 4. YAML Configuration Design

### Current Format (limited):
```yaml
stage3_preprocessing:
  imputation:
    method: "mean"
```

### Proposed Expanded Format:
```yaml
stage3_preprocessing:
  imputation:
    method: "group_median"          # Core method
    # --- Group-based parameters ---
    group_column: "auto"            # "auto" = highest-correlated categorical, or explicit column name
    fallback_method: "median"       # If group has < min_group_size records
    min_group_size: 5               # Minimum records in group to trust group statistic
    # --- Window-based parameters (rolling/ewma) ---
    window_size: 7                  # For rolling_mean, rolling_median
    ewma_alpha: 0.3                 # For EWMA (0.1=smooth, 0.9=reactive)
    ewma_span: null                 # Alternative to alpha (span in periods)
    # --- Interpolation parameters ---
    interpolation_order: 3          # For spline (cubic=3)
    interpolation_limit: 5          # Max consecutive NaN to interpolate
    interpolation_direction: "both" # "forward", "backward", "both"
    # --- Trimming parameters ---
    trim_percentile: 0.05           # For winsorized_mean / trimmed_mean
    # --- Indicator parameters ---
    add_missing_indicator: true     # For indicator_* methods
    indicator_suffix: "_was_missing" # Column naming
    # --- Regression parameters ---
    regression_estimator: "ridge"   # "ridge", "lasso", "rf", "gbm"
    add_noise: false                # true for stochastic_regression
    noise_scale: 0.1                # Residual noise multiplier
    # --- MICE variant parameters ---
    max_iter: 10                    # For iterative_* methods
    mice_estimator: "BayesianRidge" # "BayesianRidge", "RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor"
    # --- Hot deck parameters ---
    hot_deck_matching: "knn"        # "knn", "random", "sequential"
    hot_deck_k: 5                   # Number of donor candidates
    # --- Multiple imputation ---
    n_imputations: 5                # For multiple_imputation (Rubin's)
    pool_method: "mean"             # How to combine: "mean", "median"
    # --- Domain constant ---
    fill_value: null                # For "constant" method — null means infer from domain
    fill_value_numeric: 0           # Default for numeric columns
    fill_value_categorical: "unknown" # Default for categorical columns
```

> **Backward Compatibility:** All new parameters have sensible defaults. Existing variants with `method: "mean"` continue to work unchanged. Only new variants specify the extended parameters.

---

## 5. Variant Expansion Impact

### Current: 6 methods → 445 variants (on disk)
### Proposed: 33 methods → Est. 650-800+ variants

**New Variants by Industry Template:**

| Industry Template | Key New Methods | New Variants (Est.) |
|-------------------|----------------|--------------------:|
| **Telecom** | ffill, rolling_mean, group_mean, ewma, zero, indicator_mean | ~30 |
| **Healthcare** | ffill (LOCF), bfill (NOCB), pmm, group_median, multiple_imputation, indicator_median | ~35 |
| **Finance** | indicator_mean, winsorized_mean, ewma, group_mean, regression, zero | ~30 |
| **Retail / E-Commerce** | rolling_median, interpolate_linear, hot_deck, group_mode, zero | ~25 |
| **Manufacturing / IoT** | interpolate_linear, interpolate_spline, rolling_mean, ffill, constant | ~25 |
| **Customer Service / HR** | hot_deck, pmm, group_mode, constant, indicator_median | ~20 |
| **General / Cross-Industry** | iterative_rf, iterative_gb, missforest, trimmed_mean, stochastic_regression | ~35 |

**Total New Variants: ~200**  
**Grand Total with Existing: ~645** (445 current + ~200 new)

---

## 6. Method Decision Matrix — When to Use What

This is the decision logic the **DatasetProfiler** and **VariantRecommender** should use:

### By Missing Rate

| Missing Rate | Recommended Methods | Avoid |
|:------------:|---------------------|-------|
| < 1% | `mean`, `median`, `mode`, `zero` | Anything complex (overhead > benefit) |
| 1–5% | `mean`, `median`, `knn`, `group_mean`, `indicator_mean` | `drop` (wastes too much data above 3%) |
| 5–15% | `group_median`, `knn`, `iterative`, `regression`, `hot_deck`, `indicator_*` | `mean` alone (biased at this rate) |
| 15–30% | `iterative_rf`, `pmm`, `missforest`, `multiple_imputation`, `group_*` | `mean`, `median` alone (too much bias) |
| > 30% | `multiple_imputation`, `iterative_gb`, `missforest`, plus `indicator_*` | `drop` (loses > 30% of data), simple fill |

### By Missing Pattern

| Pattern | Detection Signal | Recommended Methods |
|---------|-----------------|---------------------|
| **MCAR** (random scatter) | Little's test non-significant | Any simple method works |
| **MAR** (correlated with observed) | Missing rate varies by subgroup | `group_*`, `regression`, `knn`, `iterative_*` |
| **MNAR** (related to missing value itself) | Missing rate correlates with target | `indicator_*` (flag is critical), `pmm`, `multiple_imputation` |
| **Temporal gaps** | Data has time column + sequential NaN blocks | `ffill`, `bfill`, `interpolate_*`, `rolling_*`, `ewma` |
| **Block missing** (entire column for subset) | > 50% missing in some columns, 0% in others | `conditional` (different method per column), `matrix_factorization` |
| **Monotone** (cumulative drop-off) | Missing increases over time/rows | `ffill` (LOCF), `multiple_imputation` |

### By Data Type

| Data Type | Best Methods | Notes |
|-----------|-------------|-------|
| **Continuous numeric** | `mean`, `median`, `knn`, `regression`, `interpolate_*`, `ewma` | Most methods apply |
| **Ordinal/Likert** | `pmm`, `group_mode`, `median`, `hot_deck` | Must stay on scale (1-5); avoid mean (gives 3.7) |
| **Categorical (low card)** | `mode`, `group_mode`, `hot_deck`, `constant` | Never use mean/median |
| **Categorical (high card)** | `constant("unknown")`, `group_mode`, `hot_deck` | Adding "unknown" category is often best |
| **Binary (0/1)** | `mode`, `group_mode`, `indicator_*` | Imputed value should be 0 or 1, not 0.4 |
| **Count data** | `zero`, `median`, `group_median` | Counts are non-negative integers |
| **Datetime** | `ffill`, `interpolate_linear` | Treat as numeric (epoch) for interpolation |
| **Percentage/ratio** | `winsorized_mean`, `trimmed_mean`, `group_median` | Must stay in [0, 1] range |

---

## 7. DatasetProfiler Expansion Recommendations

The current profiler (in `src/utils/dataset_profiler.py`) uses only 3 tiers of recommendation:

```
< 5% missing  → ["mean"]
5-20% missing → ["mean", "median", "knn"]
> 20% missing → ["knn", "iterative", "median"]
```

### Proposed Enhanced Logic:

```
STEP 1: Detect missing rate
STEP 2: Detect missing pattern (MCAR/MAR/MNAR)
STEP 3: Detect temporal structure (has date/time columns?)
STEP 4: Detect domain signals (column names, value ranges)
STEP 5: Combine all signals into ranked recommendation list
```

| Condition | Imputation Recommendations (ranked) |
|-----------|-------------------------------------|
| **Low missing (<5%) + No time column** | `mean`, `median`, `zero` (for counts), `constant` |
| **Low missing (<5%) + Time column present** | `ffill`, `interpolate_linear`, `mean` |
| **Moderate (5-15%) + Categorical-heavy** | `group_mode`, `hot_deck`, `knn`, `indicator_mean` |
| **Moderate (5-15%) + Numeric-heavy** | `group_median`, `knn`, `regression`, `indicator_median` |
| **Moderate (5-15%) + Time column** | `rolling_mean`, `ewma`, `interpolate_linear`, `ffill` |
| **High (15-30%) + Any** | `iterative_rf`, `pmm`, `missforest`, `group_median`, `indicator_mean` |
| **Very High (>30%)** | `multiple_imputation`, `iterative_gb`, `missforest`, `indicator_*` |
| **MNAR detected** | `indicator_mean`, `indicator_median` (flag is mandatory), then `pmm`, `regression` |
| **Finance domain** | `winsorized_mean`, `indicator_mean`, `ewma`, `group_mean`, `zero` |
| **Healthcare domain** | `ffill` (LOCF), `pmm`, `group_median`, `multiple_imputation` |
| **Telecom domain** | `ffill`, `rolling_mean`, `group_mean`, `ewma`, `zero` |
| **IoT/Sensor domain** | `interpolate_linear`, `interpolate_spline`, `rolling_mean`, `ffill` |

---

## 8. Implementation Phases

### Phase 1 — Tier 1 Methods (10 new methods) 
**Effort:** Low — all use pandas/sklearn builtins  
**Variant Impact:** +80 new variants  
**What to do:**
- Add `ffill`, `bfill`, `interpolate_linear`, `zero`, `constant` to `s06_phaseb_variant_runner.py` (the production Phase B step)
- Add `indicator_mean`, `indicator_median` with flag column logic
- Add `group_mean`, `group_median`, `group_mode` with auto-detect grouping column
- Generate new variant YAML files via `configs/generate_variant_library.py` and/or `src/variant_search/variant_search_engine.py`
- Update `DatasetProfiler` (`src/utils/dataset_profiler.py`) to recommend new methods
- Update `VariantRecommender` (`src/utils/variant_recommender.py`) scoring weights to include new method families

### Phase 2 — Tier 2 Methods (10 new methods)
**Effort:** Medium — some custom logic, scipy dependency  
**Variant Impact:** +80 new variants  
**What to do:**
- Add `rolling_mean`, `rolling_median`, `ewma` with configurable windows
- Add `interpolate_spline` via scipy
- Add `winsorized_mean`, `trimmed_mean` via scipy.stats
- Add `hot_deck` with KNN-based donor matching
- Add `pmm` with predictive matching
- Add `regression`, `stochastic_regression` with configurable estimator
- Generate industry-specific variant YAML files

### Phase 3 — Tier 3 Methods (7 new methods)
**Effort:** High — custom implementations or new dependencies  
**Variant Impact:** +40 new variants  
**What to do:**
- Add MICE variants: `iterative_rf`, `iterative_et`, `iterative_gb`
- Add `matrix_factorization` for sparse datasets
- Add `multiple_imputation` with pooling logic
- Add `missforest` (can implement from scratch using sklearn RF)
- Add `em_imputation` for Gaussian mixture EM 

---

## 9. Quick Reference: All 27 Methods at a Glance

| # | Method Key | Type | Handles Numeric | Handles Categorical | Complexity | sklearn/pandas Built-in |
|---|-----------|------|:---:|:---:|:---:|:---:|
| 1 | `mean` | Statistical | ✅ | — | ⭐ | ✅ |
| 2 | `median` | Statistical | ✅ | — | ⭐ | ✅ |
| 3 | `mode` | Statistical | — | ✅ | ⭐ | ✅ |
| 4 | `drop` | Deletion | ✅ | ✅ | ⭐ | ✅ |
| 5 | `knn` | Model-Based | ✅ | — | ⭐⭐ | ✅ |
| 6 | `iterative` | Model-Based | ✅ | — | ⭐⭐ | ✅ |
| 7 | `ffill` | Temporal | ✅ | ✅ | ⭐ | ✅ |
| 8 | `bfill` | Temporal | ✅ | ✅ | ⭐ | ✅ |
| 9 | `interpolate_linear` | Temporal | ✅ | — | ⭐ | ✅ |
| 10 | `interpolate_spline` | Temporal | ✅ | — | ⭐⭐ | scipy |
| 11 | `rolling_mean` | Window-Based | ✅ | — | ⭐ | ✅ |
| 12 | `rolling_median` | Window-Based | ✅ | — | ⭐ | ✅ |
| 13 | `ewma` | Window-Based | ✅ | — | ⭐ | ✅ |
| 14 | `zero` | Constant | ✅ | — | ⭐ | ✅ |
| 15 | `constant` | Constant | ✅ | ✅ | ⭐ | ✅ |
| 16 | `indicator_mean` | Hybrid | ✅ | — | ⭐ | ✅ |
| 17 | `indicator_median` | Hybrid | ✅ | — | ⭐ | ✅ |
| 18 | `group_mean` | Group-Based | ✅ | — | ⭐⭐ | ✅ |
| 19 | `group_median` | Group-Based | ✅ | — | ⭐⭐ | ✅ |
| 20 | `group_mode` | Group-Based | — | ✅ | ⭐⭐ | ✅ |
| 21 | `winsorized_mean` | Robust Statistical | ✅ | — | ⭐⭐ | scipy |
| 22 | `trimmed_mean` | Robust Statistical | ✅ | — | ⭐⭐ | scipy |
| 23 | `hot_deck` | Donor-Based | ✅ | ✅ | ⭐⭐⭐ | custom |
| 24 | `pmm` | Donor-Based | ✅ | — | ⭐⭐⭐ | custom |
| 25 | `regression` | Model-Based | ✅ | — | ⭐⭐ | sklearn |
| 26 | `stochastic_regression` | Model-Based | ✅ | — | ⭐⭐⭐ | sklearn |
| 27 | `iterative_rf` | Advanced MICE | ✅ | — | ⭐⭐⭐ | sklearn |
| 28 | `iterative_et` | Advanced MICE | ✅ | — | ⭐⭐⭐ | sklearn |
| 29 | `iterative_gb` | Advanced MICE | ✅ | — | ⭐⭐⭐ | sklearn |
| 30 | `missforest` | Ensemble | ✅ | ✅ | ⭐⭐⭐ | custom |
| 31 | `multiple_imputation` | Statistical | ✅ | ✅ | ⭐⭐⭐⭐ | custom |
| 32 | `matrix_factorization` | Decomposition | ✅ | — | ⭐⭐⭐⭐ | sklearn |
| 33 | `em_imputation` | Statistical | ✅ | — | ⭐⭐⭐⭐ | custom |

**Current coverage: 6 of 33 (18%)**  
**After expansion: 33 of 33 (100%)**

---

## 10. Real-World Impact Example

### Scenario: Telecom Churn Dataset (7,043 rows, 19 features, 2.8% missing)

**Current behavior:** All 445 variants use mean/median/knn/mode for imputation. They all produce nearly identical imputed datasets. The "diversity" from 337 variants is an illusion — the model sees essentially the same preprocessed data through slightly different encodings/scaling, but the fundamental missing value treatment is identical.

**After expansion:** 
- Variant A: `mean` imputation (baseline)
- Variant B: `ffill` (respects temporal order of customer tenure)
- Variant C: `group_median` by `Contract` type (Monthly/1yr/2yr customers imputed differently)
- Variant D: `indicator_mean` (adds `TotalCharges_was_missing` flag — model learns "missing charges" correlates with new customers)
- Variant E: `ewma` (recent billing patterns weighted more)
- Variant F: `zero` for `TechSupport`, `OnlineBackup` (no service = 0, not mean)

**Each variant now produces a genuinely DIFFERENT preprocessed dataset**, leading to models that learn different patterns. The AIM-Tournament can then select the imputation strategy that actually works best for the data — not just pick between 6 flavors of "fill with a number."

---

## 11. Summary

The current pipeline treats imputation as an afterthought — 6 basic methods covering only simple statistical fills and one ML method. A lead data scientist working across industries would find this library inadequate for any non-trivial missing data scenario.

The expansion to **33 methods across 7 families** (Statistical, Temporal, Window-Based, Constant, Group-Based, Donor-Based, Model-Based) transforms imputation from a checkbox into a **genuine competitive dimension** in the variant search space.

**Key insight:** The value isn't just in having more methods — it's in having methods that match the **data generation process**. Forward fill makes physical sense for sensor data. Group median makes clinical sense for patient data. Missing indicators make risk-modeling sense for credit data. The current 6 methods ignore all of this domain intelligence.

### Production Integration Notes

- **Variant runner:** New imputation methods must be added to `src/steps/s06_phaseb_variant_runner.py`, which is the production Phase B step (component `v3_phaseb_variant_runner`, version 7, environment `mlops-v3-unified:23`).
- **Variant generation:** Use `configs/generate_variant_library.py` (CLI) or `src/variant_search/variant_search_engine.py` to produce YAML files with expanded imputation dimensions.
- **Current on-disk library:** 445 variants across classification (257), regression (123), clustering (65). See `configs/recipes/{task_type}/` directories.
- **Scoring integration:** `src/utils/variant_recommender.py` scores imputation relevance at **25% weight** — the highest single dimension. Expanding the method set directly increases scoring discrimination.

**Next step:** Confirm scope (Tier 1 only, or Tiers 1+2, or all three), then generate YAML variants and add imputation dispatch logic to s06.
