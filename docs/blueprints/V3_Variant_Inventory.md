# V3 Variant Inventory — MLOps Solution Accelerator

> **Purpose:** Complete explanation of every variant category in the V3 pipeline — what exists today and what is planned for future expansion.
>
> **Last Updated:** January 2026

---

## Table of Contents

1. [What Is a Variant?](#1-what-is-a-variant)
2. [Section A — Existing Variants](#section-a--existing-variants)
   - [A1. Classification Variant Search (210 Variants)](#a1-classification-variant-search-210-variants)
   - [A2. Classification Enterprise Lightning Fast (3 Variants)](#a2-classification-enterprise-lightning-fast-3-variants)
   - [A3. Classification V1 Legacy Recipes (44 Variants)](#a3-classification-v1-legacy-recipes-44-variants)
   - [A4. Classification Baseline and Standalone Recipes (3 Recipes)](#a4-classification-baseline-and-standalone-recipes-3-recipes)
   - [A5. Regression Variant Search (80 Variants)](#a5-regression-variant-search-80-variants)
   - [A6. Regression Baseline and Standalone Recipes (3 Recipes)](#a6-regression-baseline-and-standalone-recipes-3-recipes)
   - [A7. Regression V1 Legacy Recipes (43 Variants)](#a7-regression-v1-legacy-recipes-43-variants)
   - [A8. Clustering Baseline and Standalone Recipes (2 Recipes)](#a8-clustering-baseline-and-standalone-recipes-2-recipes)
   - [A9. Clustering V1 Legacy Recipes (25 Variants)](#a9-clustering-v1-legacy-recipes-25-variants)
   - [A10. Clustering Variant Search (40 Variants)](#a10-clustering-variant-search-40-variants)
   - [A11. Top-Level Recipe Library (3 Named Recipes)](#a11-top-level-recipe-library-3-named-recipes)
   - [Existing Variants — Summary Table](#existing-variants--summary-table)
3. [Section B — Planned Variants](#section-b--planned-variants)
   - [B1. Forecasting / Time-Series Variants (New)](#b1-forecasting--time-series-variants-new)
   - [B2. Regression Enterprise Lightning Fast (New)](#b2-regression-enterprise-lightning-fast-new)
   - [B3. Clustering Enterprise Lightning Fast (New)](#b3-clustering-enterprise-lightning-fast-new)
   - [B4. Classification Variant Search Expansion](#b4-classification-variant-search-expansion)
   - [B5. Regression Variant Search Expansion](#b5-regression-variant-search-expansion)
   - [B6. Clustering State-of-the-Art V1 Tier (New)](#b6-clustering-state-of-the-art-v1-tier-new)
   - [Planned Variants — Summary Table](#planned-variants--summary-table)

---

## 1. What Is a Variant?

A **variant** (also called a **recipe**) is a YAML configuration file that defines a complete preprocessing pipeline for a single Azure ML training run. Each variant specifies a unique combination of preprocessing methods across several dimensions such as imputation, encoding, scaling, imbalance handling, feature selection, and outlier handling.

During Phase B of the V3 pipeline, the **DatasetProfiler** analyzes the incoming dataset, then the **VariantRecommender** scores all available variants for relevance (0–100) and selects the top 20 most appropriate variants for that specific dataset. This intelligent selection means only a data-relevant subset of variants is trained in any single pipeline run, keeping compute costs manageable while covering the most promising preprocessing strategies.

**Variant YAML files live under:** `configs/recipes/{task_type}/`

Each variant is identified by a 12-character hexadecimal ID (e.g., `variant_01ace0cb3ddd.yml`) for automated variants, or by a human-readable name (e.g., `baseline_recipe.yml`) for curated and baseline recipes.

---

## Section A — Existing Variants

---

### A1. Classification Variant Search (210 Variants)

**Location:** `configs/recipes/classification/variant_search/`
**Count:** 210 YAML files
**Generation Mode:** Grid sample — systematically generated from a combinatorial grid of preprocessing options

These are the core automated classification variants. Each file represents a distinct preprocessing pipeline assembled from five dimensions:

#### Preprocessing Dimensions

| Dimension | Methods Available | Count per Method |
|-----------|-------------------|-----------------|
| **Imputation** | mean, median, knn | Expanded from original single-method to 3 methods |
| **Encoding** | onehot, label | ~105 each |
| **Scaling** | standard, robust, minmax, quantile, yeo_johnson, none | ~35 each |
| **Imbalance Handling** | smoteenn, smoketomek, none | ~70 each |
| **Feature Selection** | correlation, variance, none | Distributed across variants |

#### Dimension Details

**Imputation (3 methods — mean, median, knn):**
The 210 classification variants now span three imputation strategies. Mean imputation replaces missing numerical values with the column average (fast and stable). Median imputation is robust to skewed distributions. KNN imputation preserves local feature structure using neighbor-based fill.

**Encoding (2 methods):**
- **One-Hot Encoding (onehot):** Creates binary columns for each categorical level. Best for low-to-medium cardinality features. Used in ~105 variants.
- **Label Encoding (label):** Assigns an integer to each categorical level. Compact and tree-friendly. Used in ~105 variants.

**Scaling (6 methods):**
- **Standard:** Centers to zero mean, unit variance. The most common default for linear models.
- **Robust:** Uses median and IQR instead of mean/std. Tolerant of outliers.
- **MinMax:** Rescales features to a fixed [0, 1] range. Useful for neural networks and distance-based models.
- **Quantile:** Maps feature values to a uniform or normal distribution. Handles extreme skew.
- **Yeo-Johnson (yeo_johnson):** Power transform that stabilizes variance and reduces skewness. Works with both positive and negative values.
- **None:** No scaling applied. Appropriate for tree-based algorithms that are scale-invariant.

Each method appears in approximately 35 variants (35 × 6 = 210).

**Imbalance Handling (3 methods):**
- **SMOTE-ENN (smoteenn):** Combines synthetic oversampling (SMOTE) with Edited Nearest Neighbors cleaning. Generates synthetic minority samples, then removes noisy samples near the decision boundary. Used in ~70 variants.
- **SMOTE-Tomek (smoketomek):** Combines SMOTE with Tomek Link cleaning. Generates synthetic minority samples, then removes ambiguous pairs that are nearest neighbors from different classes. Used in ~70 variants.
- **None:** No resampling. Relies on the model's native class-weight support. Used in ~70 variants.

**Feature Selection (3 methods):**
- **Correlation (correlation):** Removes features with pairwise Pearson correlation above a threshold (0.95). The most common selection method.
- **Variance (variance):** Drops features with near-zero variance.
- **None:** Keeps all features.

#### Leakage Risk Scoring

Every classification variant carries a leakage risk tag:
- **None (~70 variants):** Preprocessing methods that do not use target-column information. Safe by construction.
- **Medium (~140 variants):** Methods such as SMOTE-ENN or SMOTE-Tomek that use target values to generate synthetic samples. Require proper train/test separation to avoid data leakage.

High-leakage variants (those using target encoding or other direct target-dependent transforms) are excluded from the grid entirely.

#### Runtime and Selection

- **Estimated runtime:** 30 seconds per variant
- **Selection strategy:** The VariantRecommender scores each variant 0–100 based on dataset characteristics, then picks the top 20 per run
- **Search mode:** Progressive — if the top-ranked variants all belong to the same preprocessing family, diversity boosting injects alternatives from underrepresented dimensions

---

### A2. Classification Enterprise Lightning Fast (3 Variants)

**Location:** `configs/recipes/classification/enterprise_lightning_fast/`
**Count:** 3 YAML files
**Generation Mode:** Hand-curated with academic research citations

These are premium, research-backed recipes designed for production classification workloads. They use advanced preprocessing methods not found in the standard variant search grid. Each recipe includes citations to the peer-reviewed papers that validate the chosen methods.

#### Enterprise Lightning 01
**File:** `enterprise_lightning_01.yml`
**Pipeline:** MICE Iterative Imputation → Target Encoding → Quantile Scaling → Borderline SMOTE → Permutation Feature Selection
**Expected Runtime:** 180 seconds

- **MICE Imputation (Iterative with BayesianRidge):** Multiple Imputation by Chained Equations. Models each missing feature as a function of the other features, producing more realistic imputed values than simple mean or median. Uses 10 iterations with Bayesian Ridge regression as the estimator.
- **Target Encoding:** Replaces categorical levels with the mean of the target variable per category. Captures target-category relationships without creating hundreds of binary columns. Best suited for high-cardinality categoricals.
- **Quantile Scaling:** Transforms features to follow a normal distribution via quantile mapping. Handles extreme skewness that standard scaling cannot.
- **Borderline SMOTE:** Oversamples only the minority class examples that lie near the decision boundary. More focused than standard SMOTE because it targets the region where the classifier needs the most help.
- **Permutation Feature Selection:** Measures each feature's importance by shuffling it and observing the drop in model accuracy. Data-driven and model-agnostic.

**Research Citations:** van Buuren (2011) for MICE imputation; Breiman (2001) for permutation importance.

#### Enterprise Lightning 02
**File:** `enterprise_lightning_02.yml`
**Pipeline:** KNN Weighted Imputation → CatBoost Encoding → Yeo-Johnson Transform → SMOTE-ENN → Boruta Feature Selection
**Expected Runtime:** 240 seconds

- **KNN Weighted Imputation:** Uses the K nearest neighbors (K=5, distance-weighted) to fill missing values. Preserves local feature relationships better than global-mean imputation.
- **CatBoost Encoding:** An information-rich encoding that uses target statistics with ordered boosting to prevent data leakage. Produces a single numerical column per categorical feature. Particularly effective for high-cardinality categoricals with natural ordering.
- **Yeo-Johnson Transform:** A parametric power transformation that handles both positive and negative values. Reduces skewness and stabilizes variance in a statistically principled way.
- **SMOTE-ENN:** The same oversampling + cleaning combination described in the variant search, but here combined with more sophisticated upstream preprocessing.
- **Boruta Feature Selection:** A wrapper method that creates shadow (permuted) copies of all features, trains a random forest on real + shadow features, and keeps only the features that consistently beat their shadow counterparts. More thorough than correlation or variance methods.

**Research Citations:** Troyanskaya et al. (2001) for KNN imputation; Yeo & Johnson (2000) for the power transform; Kursa & Rudnicki (2010) for Boruta.

#### Enterprise Lightning 03
**File:** `enterprise_lightning_03.yml`
**Pipeline:** Median Imputation → Isolation Forest Outlier Detection → Robust Scaling → ADASYN → One-Hot Encoding → Mutual Information Feature Selection
**Expected Runtime:** 200 seconds

- **Median Imputation:** Replaces missing values with the column median. More robust to outliers than mean imputation.
- **Isolation Forest Outlier Detection:** Identifies and removes anomalous rows by measuring how easily a random forest can isolate each sample. Outliers are easier to isolate and receive higher anomaly scores. Uses a contamination threshold of 0.05 (5%).
- **Robust Scaling:** Scales using median and interquartile range, making it naturally resistant to the remaining outliers after Isolation Forest filtering.
- **ADASYN (Adaptive Synthetic Sampling):** Like SMOTE, but generates more synthetic samples in regions where the minority class density is lower. Adapts to the difficulty of different regions of the feature space.
- **Mutual Information Feature Selection:** Measures the statistical dependency between each feature and the target. Non-linear and model-agnostic. Selects the top-K features (K=20) ranked by mutual information score.

**Research Citations:** Liu et al. (2008) for Isolation Forest; Kraskov et al. (2004) for mutual information estimation; Huber (1964) for robust statistics.

---

### A3. Classification V1 Legacy Recipes (44 Variants)

**Location:** `configs/recipes/classification/v1_generated/`
**Count:** 44 YAML files across 5 performance tiers
**Generation Mode:** Ported from V1 MLOps Solution Accelerator

These recipes were migrated from the original V1 codebase. They are organized into five performance tiers, each targeting a different trade-off between speed and preprocessing depth. Every recipe carries V1 metadata including a compatibility score, expected quality tier, and maximum runtime.

#### Tiers

| Tier | Recipes | Typical Runtime | Level Range | Character |
|------|---------|----------------|-------------|-----------|
| **Lightning Fast** | 5 | ~12 seconds | 0 | Minimal preprocessing — drop missing, label-encode, no scaling, no selection. Fastest possible baseline. |
| **Quick Exploration** | 9 | ~20–30 seconds | 10–20 | Light preprocessing — simple imputation and encoding with one or two additional steps. Good for rapid experimentation. |
| **Balanced Performance** | 15 | ~40–60 seconds | 20–40 | Moderate preprocessing — combines imputation, encoding, and either scaling or feature selection. The largest tier, covering the widest range of strategies. |
| **High Performance** | 10 | ~60–75 seconds | 40–60 | Heavy preprocessing — multi-step pipelines with advanced imputation, scaling, and selection. Designed for datasets where quality matters more than speed. |
| **State-of-the-Art** | 5 | ~89 seconds | 60 | Full preprocessing — KNN imputation, target encoding, robust scaling, chi-squared feature selection. Maximum preprocessing depth. |

#### V1 Metadata Fields

Each V1 recipe includes:
- **compatibility_score:** A float indicating how universally applicable the recipe is (higher = more general).
- **expected_quality:** The performance tier name (e.g., "lightning_fast", "state-of-the-art").
- **max_runtime_seconds:** Estimated maximum wall-clock time for the preprocessing pipeline.
- **level:** An integer (0–60) representing the preprocessing complexity level.
- **original_name:** The human-readable label from V1 (e.g., "state-of-the-art: knn+target+robust+chi2").

These V1 recipes serve as a reference library and provide backward compatibility with experiments originally run on the V1 pipeline. They are not actively selected by the VariantRecommender but can be manually specified in phase configuration.

---

### A4. Classification Baseline and Standalone Recipes (3 Recipes)

**Location:** `configs/recipes/classification/`

#### Baseline Recipe
**File:** `baseline_recipe.yml`
**Pipeline:** Mean Imputation → One-Hot Encoding → Standard Scaling → Variance Feature Selection

The baseline recipe uses the simplest, most universally applicable preprocessing combination. It establishes a performance floor that every variant must beat to justify its additional complexity. Each dimension uses the most conservative option: mean imputation is numerically stable, one-hot encoding avoids information leakage, standard scaling is the most widely understood normalization, and variance-based feature selection removes only truly constant features.

#### Standalone Recipes

**File:** `recipe_smote_target_standard.yml`
**Pipeline:** KNN Imputation → SMOTE → Target Encoding → Standard Scaling → Mutual Information Feature Selection

This recipe pairs SMOTE oversampling with target encoding, a combination specifically designed for imbalanced classification datasets with high-cardinality categorical features. KNN imputation preserves local structure while SMOTE addresses class imbalance. Mutual information feature selection ensures only informative features survive.

**File:** `recipe_knn_onehot_minmax.yml`
**Pipeline:** KNN Imputation → One-Hot Encoding → MinMax Scaling → Variance Feature Selection

This recipe is optimized for distance-based algorithms (KNN classifiers, SVM, neural networks) where MinMax scaling ensures all features contribute equally and one-hot encoding avoids ordinal assumptions.

---

### A5. Regression Variant Search (80 Variants)

**Location:** `configs/recipes/regression/variant_search/`
**Count:** 80 YAML files
**Generation Mode:** Curated grid — systematically expanded from an initial hand-curated set of 30 to a broader combinatorial grid

The regression variant search uses a different design philosophy than classification. Instead of exhaustively sampling a five-dimensional grid, the 80 variants were curated to cover the most relevant preprocessing combinations for continuous-target prediction. The library was expanded from the initial 30 variants to 80 as part of the January 2026 variant expansion (Task 3). A key difference: **there is no imbalance handling** (a classification-only concept) and a new dimension—**outlier handling**—is added because outliers disproportionately affect regression loss functions.

#### Preprocessing Dimensions

| Dimension | Methods Available | Count Distribution |
|-----------|-------------------|-------------------|
| **Imputation** | mean (~24), median (~22), knn (~18), iterative (~16) | Broader than classification — four methods |
| **Encoding** | onehot (~32), label (~27), target (~21) | Adds target encoding |
| **Scaling** | standard (~32), robust (~26), minmax (~16), none (~6) | Weighted toward regression-appropriate methods |
| **Feature Selection** | none (~34), correlation (~19), variance (~14), mutual_info (~13) | Mutual info added; none is most common |
| **Outlier Handling** | iqr (~16), winsorize (~16), zscore (~8), none (~40) | Regression-specific dimension |

#### Dimension Details — Regression-Specific

**Imputation — Four Methods:**
Unlike classification (mean only), regression variants distribute across four imputation strategies. This reflects the fact that regression targets are sensitive to how missing values distort feature distributions:
- **Mean (9):** Fast, stable, most common.
- **Median (8):** Robust to skewed distributions and outliers.
- **KNN (7):** Uses neighbor structure to fill gaps. Preserves local patterns.
- **Iterative (6):** MICE-style chained equations. Most statistically sound but slowest.

**Encoding — Three Methods:**
- **One-Hot (12):** Safe default. Creates binary indicators.
- **Label (10):** Compact. Works well with tree-based regressors.
- **Target (8):** Replaces categories with smoothed target means. Powerful for regression because the encoding directly relates to the predicted variable. Requires careful cross-validation to prevent leakage.

**Outlier Handling — Regression-Specific Dimension:**
Outliers can drastically distort regression loss functions (MSE, RMSE). This dimension offers three removal or transformation strategies:
- **IQR (6 variants):** Removes rows where feature values fall beyond 1.5 × the interquartile range. Simple and distribution-free.
- **Winsorize (6 variants):** Clips extreme values to the 5th and 95th percentiles instead of removing them. Keeps all rows, reduces extreme influence.
- **Zscore (3 variants):** Flags values beyond 3 standard deviations as outliers. Assumes roughly normal feature distributions.
- **None (15 variants):** No outlier treatment. Relies on robust models (tree ensembles) that are naturally tolerant.

#### Leakage Risk

All 80 regression variants are tagged with **leakage_risk: none**. Since there is no SMOTE or ADASYN (imbalance handling is absent), and target encoding variants are handled with proper cross-fold encoding, the overall leakage risk is low.

---

### A6. Regression Baseline and Standalone Recipes (3 Recipes)

**Location:** `configs/recipes/regression/`

#### Baseline Recipe
**File:** `baseline_recipe.yml`
**Pipeline:** Mean Imputation → One-Hot Encoding → Standard Scaling → Variance Feature Selection

Identical in spirit to the classification baseline. Establishes the simplest possible regression preprocessing pipeline.

#### Standalone Outlier Recipes

**File:** `recipe_outlier_iqr_standard.yml`
**Pipeline:** KNN Imputation → IQR Outlier Removal (1.5× multiplier) → Standard Scaling → Mutual Information Feature Selection (K=20)

Designed for datasets with known heavy-tailed distributions. The IQR method removes rows with extreme values, then KNN imputation fills gaps left by the removal process. Mutual information selects the top 20 most informative features.

**File:** `recipe_winsorize_robust.yml`
**Pipeline:** Median Imputation → Winsorize (5th–95th percentile) → Robust Scaling → Variance Feature Selection (threshold 0.01)

An alternative outlier strategy that clips rather than removes extreme values. Paired with robust scaling (median/IQR-based) and median imputation to create a fully outlier-resistant pipeline. Preserves every row, which is valuable when data volume is limited.

---

### A7. Regression V1 Legacy Recipes (43 Variants)

**Location:** `configs/recipes/regression/v1_generated/`
**Count:** 43 YAML files across 5 tiers

| Tier | Recipes |
|------|---------|
| Lightning Fast | 4 |
| Quick Exploration | 9 |
| Balanced Performance | 15 |
| High Performance | 10 |
| State-of-the-Art | 5 |

Same tiered structure as classification V1 recipes but adapted for regression tasks. Includes all five performance tiers, offering a full spectrum from minimal to maximal preprocessing depth. These serve as backward-compatible reference recipes from the V1 pipeline.

---

### A8. Clustering Baseline and Standalone Recipes (2 Recipes)

**Location:** `configs/recipes/clustering/`

#### Baseline Recipe
**File:** `baseline_recipe.yml`
**Pipeline:** Mean Imputation → One-Hot Encoding → Standard Scaling → Variance Feature Selection

The clustering baseline notes that **scaling is critical for clustering because most clustering algorithms (K-Means, DBSCAN, hierarchical) rely on distance metrics.** Unscaled features with larger ranges dominate the distance calculation and bias cluster assignments. The recipe also includes a placeholder for dimensionality reduction (method: none) indicating that PCA or similar transforms can be added for high-dimensional datasets.

**Key difference from classification/regression baselines:** No imbalance handling (unsupervised learning has no target class). No outlier handling dimension (not yet added but relevant — outliers strongly affect centroid-based clustering).

#### Standalone Recipe
**File:** `recipe_knn_onehot_minmax.yml`
**Pipeline:** KNN Imputation → One-Hot Encoding → MinMax Scaling → Variance Feature Selection

MinMax scaling is particularly appropriate for clustering because it bounds all features to [0, 1], giving each feature equal weight in distance calculations. KNN imputation preserves local structure, which matters for proximity-based clustering.

---

### A9. Clustering V1 Legacy Recipes (25 Variants)

**Location:** `configs/recipes/clustering/v1_generated/`
**Count:** 25 YAML files across 4 tiers

| Tier | Recipes |
|------|---------|
| Lightning Fast | 5 |
| Quick Exploration | 5 |
| Balanced Performance | 10 |
| High Performance | 5 |

**Notable gap:** Clustering does NOT have a **State-of-the-Art** tier. Classification and regression both have 5 state-of-the-art recipes each, but clustering stops at the High Performance tier. This is one of the gaps addressed in the Planned Variants section.

---

### A10. Clustering Variant Search (40 Variants)

**Location:** `configs/recipes/clustering/variant_search/`
**Count:** 40 YAML files
**Generation Mode:** Curated grid — systematically generated in the January 2026 expansion (Task 3)
**Status:** ✅ IMPLEMENTED — This was previously listed as a planned item (Section B1). The 40 variants were created as part of the V3 variant expansion and now exist on disk.

Clustering variant search uses a curated grid approach similar to regression. Because there is no target variable, the preprocessing decisions directly shape how clusters form. Scaling, dimensionality reduction, and distance-metric awareness are the dominant concerns.

#### Preprocessing Dimensions

| Dimension | Methods Available | Count Distribution |
|-----------|-------------------|-----------------------|
| **Imputation** | mean, median, knn | ~13 each |
| **Encoding** | onehot, label | ~20 each |
| **Scaling** | standard, robust, minmax, none | ~10 each |
| **Feature Selection** | variance, correlation, none | Distributed |

#### Design Notes

- **No imbalance handling** — not applicable to unsupervised tasks
- **No target encoding** — no target variable exists
- **Scaling is critical** — most clustering algorithms (K-Means, DBSCAN, hierarchical) rely on distance metrics; unscaled features with larger ranges dominate distance calculations
- **All 40 variants** use `preprocess=False` in PyCaret setup (preprocessing is handled by the variant pipeline, not by PyCaret internally)

#### Leakage Risk

All 40 clustering variants are tagged with **leakage_risk: none**. Without a target variable, there is no mechanism for target leakage.

---

### A11. Top-Level Recipe Library (3 Named Recipes)

**Location:** `configs/recipes/recipe_library.yml` and `configs/recipes/`

The recipe library serves as an index of named, task-agnostic baseline recipes that can be referenced from pipeline configuration files:

| Recipe Name | File | Pipeline |
|-------------|------|----------|
| **baseline** | `baseline_recipe.yml` | Mean → One-Hot → Standard → Variance |
| **smote_target_standard** | `recipe_smote_target_standard.yml` | KNN → SMOTE → Target → Standard → Mutual Info |
| **knn_onehot_minmax** | `recipe_knn_onehot_minmax.yml` | KNN → One-Hot → MinMax → Variance |

These top-level recipes exist for historical compatibility and for quick experimentation. They can be referenced by name in phase configuration without specifying a full file path.

---

### Existing Variants — Summary Table

| Task Type | Category | Count | Generation | Intelligent Selection |
|-----------|----------|-------|------------|----------------------|
| Classification | Variant Search | 210 | Grid sample | Yes — top 20 per run |
| Classification | Enterprise Lightning | 3 | Hand-curated | Yes — always included |
| Classification | V1 Legacy | 44 | Ported from V1 | No — manual only |
| Classification | Baseline + Standalone | 3 | Hand-curated | Baseline always included |
| **Classification Total** | | **260** | | |
| Regression | Variant Search | 80 | Curated grid | Yes — top 20 per run |
| Regression | Baseline + Standalone | 4 | Hand-curated | Baseline always included |
| Regression | V1 Legacy | 43 | Ported from V1 | No — manual only |
| **Regression Total** | | **127** | | |
| Clustering | Variant Search | 40 | Curated grid | Yes — top 20 per run |
| Clustering | Baseline + Standalone | 2 | Hand-curated | Baseline always included |
| Clustering | V1 Legacy | 25 | Ported from V1 | No — manual only |
| **Clustering Total** | | **67** | | |
| Shared (top-level) | Shared recipes | 3 | Cross-task | Used as defaults |
| Forecasting | — | 0 | — | — |
| **Grand Total** | | **457** | | |

---

## Section B — Planned Variants

---

### ~~B1. Clustering Variant Search~~ — ✅ COMPLETED

> **This initiative has been implemented.** See **Section A10 — Clustering Variant Search (40 Variants)** for the full inventory.
> 40 curated variants were created in `configs/recipes/clustering/variant_search/` covering imputation (mean, median, knn), encoding (onehot, label), scaling (standard, robust, minmax, none), dimensionality reduction (pca, none), and feature selection (variance, correlation, none).

~~**Status:** Not yet created~~  →  **Status:** ✅ Implemented (40 variants)
~~**Target Count:** 40–60~~ → **Actual Count:** 40 curated variants
~~**Priority:** High~~ → **Priority:** Complete

Clustering requires a fundamentally different variant design than classification or regression because there is no target variable. The preprocessing decisions directly shape how clusters form, making scaling, dimensionality reduction, and distance-metric awareness the dominant concerns.

<details>
<summary><em>Original planned dimensions (click to expand — superseded by A10 implementation)</em></summary>

| Dimension | Methods | Rationale |
|-----------|---------|-----------|
| **Imputation** | mean, median, knn, iterative | Same as regression — four methods to cover different missing data patterns |
| **Encoding** | onehot, label, ordinal, binary | Target encoding is NOT available (no target variable). Binary and ordinal encoding are lightweight alternatives to one-hot for distance metrics. |
| **Scaling** | standard, robust, minmax, quantile, power, none | Full scaling spectrum — critical for clustering because distance metrics drive cluster formation |
| **Dimensionality Reduction** | pca, t-sne, umap, none | NEW dimension — essential for clustering in high-dimensional spaces. PCA is linear and fast, t-SNE captures non-linear manifold structure, UMAP preserves both local and global topology. |
| **Feature Selection** | variance, correlation, mutual_info_unsupervised, none | Mutual information adapted for unsupervised settings (using proxy measures or cluster-free filter methods) |
| **Outlier Handling** | isolation_forest, lof, dbscan_noise, none | NEW dimension — outliers heavily distort centroid-based clusters (K-Means). Isolation Forest and Local Outlier Factor (LOF) detect anomalies without a target. DBSCAN's noise-point labeling can serve as a pre-filter. |

#### Design Philosophy

- **Curated generation** (like regression, not grid-sampled like classification) — every combination will be manually verified to make sense for unsupervised learning
- **Dimensionality reduction is the priority** — for datasets with more than 50 features, PCA or UMAP can dramatically improve cluster quality
- **Distance-metric awareness** — each variant should document which distance metric it is optimized for (Euclidean, Manhattan, cosine)
- **No imbalance handling** — not applicable to unsupervised tasks

</details>

---

### B1. Forecasting / Time-Series Variants (New)

**Status:** No variant files exist anywhere for forecasting
**Target Count:** 20–30 curated variants
**Priority:** High — time-series forecasting was added to the V3 model universe (6 statsmodels models) but has zero preprocessing variants

Forecasting variants require entirely different preprocessing dimensions because the data is temporally ordered and the preprocessing choices affect stationarity, seasonality decomposition, and autoregressive feature creation.

#### Planned Dimensions

| Dimension | Methods | Rationale |
|-----------|---------|-----------|
| **Missing Value Handling** | forward_fill, interpolate_linear, interpolate_spline, seasonal_fill, none | Time-series imputation must respect temporal order — mean/median destroys time structure. Forward fill carries the last known value forward. Interpolation preserves trends. Seasonal fill uses same-period historical values. |
| **Stationarity Transform** | differencing, log_transform, box_cox, none | Many forecasting models (ARIMA, SARIMAX) require stationary input. Differencing removes trends, log transform stabilizes variance, Box-Cox is a generalized power transform. |
| **Seasonal Decomposition** | additive, multiplicative, stl, none | Separates the time series into trend, seasonal, and residual components. Additive assumes constant seasonal amplitude, multiplicative assumes proportional amplitude, STL (Seasonal and Trend decomposition using Loess) is the most flexible. |
| **Lag Feature Engineering** | auto_lag, fixed_window, rolling_stats, none | Creates autoregressive features. Auto-lag uses partial autocorrelation to select optimal lag counts. Fixed-window creates a standard set (e.g., 7, 14, 30-day lags). Rolling statistics add moving averages, standard deviations, and min/max over configurable windows. |
| **Calendar Features** | day_of_week, month, holiday, fourier, none | Exogenous time-based features. Day-of-week and month capture basic periodicity. Holiday flags add event awareness. Fourier terms represent complex seasonal patterns as sine/cosine pairs. |
| **Scaling** | standard, robust, minmax, none | Some forecasting models (Prophet, neural-network based) benefit from scaling; ARIMA/SARIMAX typically do not. |

#### Design Philosophy

- All variants must preserve temporal order — no random shuffling, no row-removal that creates gaps
- Train/test splits are always chronological (most recent N periods for test)
- Variants should specify their target model family compatibility (e.g., "ARIMA-compatible: requires stationarity" vs "Prophet-compatible: handles trend internally")
- Leakage risk is especially important — future data must never leak into feature engineering

---

### B2. Regression Enterprise Lightning Fast (New)

**Status:** Enterprise-tier recipes exist only for classification. Regression has no equivalent.
**Target Count:** 3–5 curated variants
**Priority:** Medium

Planned recipes will port the enterprise-lightning philosophy (research-backed, citation-supported, advanced methods) to regression:

#### Planned Recipes

**Enterprise Regression 01 — Robust Outlier Pipeline:**
Iterative (MICE) Imputation → Isolation Forest Outlier Removal → Robust Scaling → CatBoost Encoding → Recursive Feature Elimination

Designed for datasets with heavy-tailed distributions and significant outlier contamination. The combination of Isolation Forest pre-filtering with robust statistics creates a defense-in-depth strategy against extreme values.

**Enterprise Regression 02 — High-Cardinality Pipeline:**
KNN Imputation → Target Encoding (with cross-fold regularization) → Quantile Transform → Mutual Information Selection → Winsorize (1st–99th percentile)

Optimized for regression datasets with many categorical features with high cardinality (zip codes, product IDs). Target encoding converts these efficiently while cross-fold regularization prevents leakage.

**Enterprise Regression 03 — Feature-Rich Pipeline:**
Median Imputation → One-Hot Encoding → Yeo-Johnson Power Transform → Boruta Feature Selection → IQR Outlier Handling

Designed for datasets with many (100+) features. Boruta rigorously identifies the truly informative subset, while the power transform and outlier handling normalize the feature space before regression modeling.

Each planned recipe will include full research citations and estimated runtime benchmarks.

---

### B3. Clustering Enterprise Lightning Fast (New)

**Status:** No enterprise-tier recipes exist for clustering
**Target Count:** 3–5 curated variants
**Priority:** Medium

#### Planned Recipes

**Enterprise Clustering 01 — High-Dimensional Discovery:**
KNN Imputation → Robust Scaling → PCA Dimensionality Reduction (95% variance) → Isolation Forest Outlier Removal

For datasets with 50+ features where direct clustering produces poor results due to the curse of dimensionality. PCA reduces the feature space while preserving variance, and outlier removal prevents extreme points from anchoring false clusters.

**Enterprise Clustering 02 — Mixed-Type Pipeline:**
Median Imputation → Binary Encoding (categoricals) → MinMax Scaling → UMAP Dimensionality Reduction → Variance Feature Selection

For datasets with a mix of numerical and categorical features. Binary encoding is compact and distance-friendly. UMAP preserves both local and global structure in the reduced space.

**Enterprise Clustering 03 — Robust Distance Pipeline:**
Iterative Imputation → One-Hot Encoding → Quantile Transform → Correlation Feature Selection → LOF Outlier Removal

For datasets where standard Euclidean distance is unreliable. Quantile normalization ensures features follow the same distribution shape, making distance comparisons more meaningful. LOF detects outliers based on local density.

---

### B4. Classification Variant Search Expansion

**Status:** Current grid covers 210 variants with expanded imputation (mean, median, knn) and 3 imbalance methods
**Target Delta:** +40–80 additional variants
**Priority:** Medium

#### Planned Expansions

**Imputation Expansion:**
The current 210 variants now cover mean, median, and knn imputation. The expansion will add:
- **Iterative (MICE):** For complex multi-feature imputation patterns where feature correlations matter

This would extend imputation coverage from 3 methods to 4 methods.

**Imbalance Handling Expansion:**
Current methods (smoteenn, smoketomek, none) will be joined by:
- **Standard SMOTE:** The classic oversampling without cleaning
- **ADASYN:** Adaptive density-based oversampling that focuses on harder regions
- **Borderline SMOTE:** Targets only boundary minority samples

**Encoding Expansion:**
Current methods (onehot, label) will be joined by:
- **Target Encoding:** With proper cross-fold regularization to prevent leakage
- **Ordinal Encoding:** For features with natural ordering
- **Binary Encoding:** For high-cardinality categoricals (compact representation)

**Feature Selection Expansion:**
Current methods (correlation, variance, none) will be joined by:
- **Mutual Information:** Non-linear dependency measure
- **PCA:** Dimensionality reduction as feature compression
- **Boruta:** Wrapper-based rigorous selection

---

### B5. Regression Variant Search Expansion

**Status:** Current library has 80 curated variants with good dimension diversity (expanded from 30 in January 2026)
**Target Delta:** +10–20 additional variants
**Priority:** Low

#### Planned Expansions

**Scaling Expansion:**
- **Quantile Transform:** For heavily skewed regression features
- **Yeo-Johnson:** Power transform for mixed positive/negative values
- **Power Transform (Box-Cox):** For strictly positive features

**Feature Selection Expansion:**
- **Boruta:** Wrapper-based importance for regression
- **Recursive Feature Elimination (RFE):** Model-driven sequential removal
- **PCA:** Linear dimensionality reduction

**Outlier Handling Expansion:**
- **Isolation Forest:** Tree-based anomaly detection
- **LOF (Local Outlier Factor):** Density-based for non-uniform distributions
- **Cook's Distance:** Regression-specific — measures each observation's influence on the fitted model

---

### B6. Clustering State-of-the-Art V1 Tier (New)

**Status:** Clustering V1 Legacy has only 4 tiers (missing State-of-the-Art). Classification and regression both have 5 tiers including State-of-the-Art.
**Target Count:** 5 recipes
**Priority:** Low

The planned State-of-the-Art tier for clustering will include full-depth preprocessing pipelines with:
- KNN or iterative imputation
- Combined encoding (binary for high-cardinality, one-hot for low-cardinality)
- Robust or quantile scaling
- PCA or UMAP dimensionality reduction
- Advanced outlier handling (Isolation Forest or LOF)

This completes the V1 tier parity across all three original task types.

---

### Planned Variants — Summary Table

| Initiative | Task Type | Estimated Count | Priority | Key Innovation |
|------------|-----------|----------------|----------|----------------|
| ~~Clustering Variant Search~~ | ~~Clustering~~ | ~~40–60~~ | ~~High~~ | ✅ **COMPLETED** — see Section A10 (40 variants implemented) |
| Forecasting Variants | Forecasting | 20–30 | High | Temporal preprocessing (stationarity, lag features, seasonal decomposition) |
| Regression Enterprise Lightning | Regression | 3–5 | Medium | Research-backed outlier and high-cardinality pipelines |
| Clustering Enterprise Lightning | Clustering | 3–5 | Medium | High-dimensional discovery and mixed-type pipelines |
| Classification Grid Expansion | Classification | 40–80 | Medium | Iterative imputation, ADASYN/SMOTE, encoding diversity |
| Regression Variant Expansion | Regression | 10–20 | Low | Advanced scaling, selection, and outlier methods |
| Clustering V1 State-of-the-Art | Clustering | 5 | Low | V1 tier parity |
| **Total Planned** | | **~81–145** | | |

**Combined Inventory After Expansion:** 457 existing + 81–145 planned = **538–602 total variants** across all task types.

---

*This document reflects the V3 MLOps Solution Accelerator variant system as of March 13, 2026.*
*Last updated: March 13, 2026 — Corrected Regression Baseline+Standalone count (3→4), added Shared top-level recipes (3), updated Grand Total (453→457).*
