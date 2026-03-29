## V3 Phase B: Intelligent Variant Recommendation System (Updated 2026-01-30)

### Architecture Overview

Phase B has been refactored from hardcoded 2-variant architecture to an **intelligent recommendation system** that:
1. Profiles the dataset to understand its characteristics
2. Recommends relevant preprocessing strategies based on data quality and domain
3. Scores and selects the most relevant variants from the per-task library
4. Runs all selected variants in a SINGLE Azure ML step (`s06`) using nested MLflow runs

**Variant Library (actual counts):**
| Task | variant_search | v1_generated | enterprise_lightning_fast | Total |
|------|---------------|-------------|--------------------------|-------|
| Classification | 210 | 44 (5 sub-tiers) | 3 | 257 |
| Regression | 80 | 43 (5 sub-tiers) | — | 123 |
| Clustering | 40 | 25 (4 sub-tiers) | — | 65 |
| **Grand Total** | **330** | **112** | **3** | **~445** |

**Key Philosophy:** Don't blindly test all variants. Use data-driven profiling to prune the search space intelligently.

---

### Phase B Components

#### 1. Dataset Profiler (`src/utils/dataset_profiler.py`)

**Purpose:** Analyze dataset characteristics and generate preprocessing recommendations.

**Key Classes:**
- `DatasetProfile`: Data class capturing statistical properties, quality issues, correlation structure, and domain signals
- `DatasetProfiler`: Analyzer that generates profiles from pandas DataFrames

**Profiling Dimensions:**
```python
profile = DatasetProfile(
    n_rows, n_features, n_numeric, n_categorical,  # Basic dimensions
    missing_rate, imbalance_ratio, outlier_prevalence,  # Quality issues
    feature_correlation_mean, multicollinearity_detected,  # Correlation structure
    domain_hints=["finance", "time_series", "academic"]  # Domain signals
)
```

**Recommendation Rules:**
- **Missing rate > 20%** → Recommend KNN/iterative imputation
- **Imbalance ratio < 0.3** → Recommend SMOTE variants (classification only)
- **Finance domain or outliers > 10%** → Recommend robust scaling
- **High feature correlation** → Recommend feature selection (correlation/variance)
- **High-cardinality categoricals** → Recommend target encoding

**Example Usage:**
```python
profiler = DatasetProfiler(task_type="classification")
profile = profiler.profile_dataset(df, target_column="churn")
recommendations = profile.recommend_preprocessing_strategies()

# Returns:
{
    "imputation": ["knn", "iterative"],
    "encoding": ["target", "onehot"],
    "scaling": ["robust", "minmax"],
    "imbalance_handling": ["smote", "smoteenn"],
    "feature_selection": ["correlation", "variance"],
    "priority_scores": {"imbalance_handling": 0.95, "imputation": 0.8, ...},
    "reasoning": ["Severe class imbalance (0.28) → SMOTE critical", ...]
}
```

---

#### 2. Variant Recommender (`src/utils/variant_recommender.py`)

**Purpose:** Score and select the most relevant variants for a dataset.

**Key Class:** `VariantRecommender`

**Scoring Algorithm:**
```python
def score_variant_relevance(variant: VariantConfig) -> (score, reasoning):
    score = 0.0
    
    # Weight each dimension by profile priority scores
    if variant.imputation in recommendations["imputation"]:
        score += 25.0 * priority_scores["imputation"]
    
    if variant.encoding in recommendations["encoding"]:
        score += 20.0 * priority_scores["encoding"]
    
    if variant.scaling in recommendations["scaling"]:
        score += 15.0 * priority_scores["scaling"]
    
    if variant.imbalance_handling in recommendations["imbalance_handling"]:
        score += 25.0 * priority_scores["imbalance_handling"]
    
    if variant.feature_selection in recommendations["feature_selection"]:
        score += 15.0 * priority_scores["feature_selection"]
    
    # Penalty for leakage risk
    if variant.leakage_risk == "high":
        score *= 0.5
    
    return score  # Max = 100
```

**Selection Strategy:**
- Sort variants by relevance score (descending)
- Apply diversity boost to ensure coverage across preprocessing dimensions
- Select top N variants (default: 20)
- Filter out variants below minimum threshold (default: 30/100)

**Example:**
```python
recommender = VariantRecommender(profile, all_variants)
selected = recommender.select_top_variants(max_variants=20)

# Returns list of (variant, score, reasoning) tuples:
[
    (variant_01ace0cb3ddd, 85.0, ["+21.2 imputation matches", "+18.0 encoding matches", ...]),
    (variant_fe89f53fba40, 78.5, ["+23.8 imbalance handling matches", ...]),
    ...
]
```

---

#### 3. Variant Schema (`src/utils/variant_schema.py`)

**Purpose:** Typed data structures for variant configurations.

**Key Classes:**
- `VariantConfig`: Complete variant specification with stage3/stage4 configs
- `Stage3PreprocessingConfig`: Imputation, encoding, scaling, imbalance, outlier handling
- `Stage4FeatureEngineeringConfig`: Feature selection methods
- `VariantMetadata`: Leakage risk, runtime estimates, generation mode

**Validation Rules:**
- **Classification:** Can use all preprocessing methods (including SMOTE)
- **Regression:** DISALLOW imbalance handling, target encoding
- **Clustering:** DISALLOW target encoding, imbalance handling

**Example:**
```python
variant = load_variant("configs/recipes/classification/variant_search/variant_01ace0cb3ddd.yml")
validate_variant_for_task(variant, "classification")  # Passes

validate_variant_for_task(variant, "regression")  # May raise ValueError if SMOTE present
```

---

#### 4. Variant Runner Step (`src/steps/s06_phaseb_variant_runner.py`, ~2 393 lines)

**Purpose:** Single Azure ML step that processes N variants with nested MLflow tracking.

**Component YAML** (`components/s06_phaseb_variant_runner.yml`, v7):
| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `config_name` | string | — | Config YAML filename |
| `variants_list` | string | — | Comma-separated variant paths |
| `engine_list` | string | `"pycaret,flaml"` | Engines to run |
| `dataset_in` | uri_file | — | Processed dataset from Stage 4 |
| `time_budget_per_variant` | integer | `300` | Max training time per variant (sec) |
| `flaml_min_budget` | integer | `120` | Floor for FLAML time budget |
| `planner_enabled` | boolean | `false` | Enable V3-Proposed adaptive planner |
| `round1_max_variants` | integer | `40` | Max variants for Round 1 proxy |
| `round2_max_variants` | integer | `10` | Max variants for Round 2 full |
| `proxy_prune_threshold` | number | `0.50` | Proxy metric pruning threshold |
| `cache_enabled` | boolean | `true` | Enable preprocessing cache |

> **Critical:** PyCaret `setup()` is called with `preprocess=False` (lines 801, 873, 946)
> because the variant runner applies all preprocessing _before_ engine training.
> Removing this flag would cause double-preprocessing.

**Architecture (simplified):**
```python
# Main loop
for variant in selected_variants:
    for engine in ["pycaret", "flaml"]:
        with mlflow.start_run(nested=True):
            # 1. Log variant config as MLflow params
            mlflow.log_params({
                "variant_id": variant.variant_id,
                "engine": engine,
                "imputation": variant.stage3_preprocessing.imputation.method,
                "encoding": variant.stage3_preprocessing.encoding.categorical_method,
                ...
            })
            
            # 2. Apply preprocessing (variant recipe)
            df_processed = apply_variant_preprocessing(df, variant)
            
            # 3. Train model (PyCaret with preprocess=False, or FLAML)
            model, metrics = train_model(df_processed, engine, variant)
            
            # 4. Log metrics + model
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            results.append((variant, engine, model, metrics))

# Select champion
champion = max(results, key=lambda r: r[3]["primary_metric"])
save_champion(champion)
```

**Outputs:**
- `leaderboard.csv`: All variant×engine results sorted by primary metric
- `all_results.json`: Complete details for all runs
- `champion_manifest.json`: Champion variant configuration and metrics
- `champion_model.pkl`: Serialized champion model

**MLflow Hierarchy:**
```
Parent Pipeline Run
└─ Phase B Variant Runner (child run)
    ├─ variant_01ace0cb3ddd_pycaret (nested run)
    │   └─ Params: {variant_id, imputation, encoding, ...}
    │   └─ Metrics: {primary_metric, runtime_sec, ...}
    ├─ variant_01ace0cb3ddd_flaml (nested run)
    ├─ variant_01e9778f8033_pycaret (nested run)
    └─ variant_01e9778f8033_flaml (nested run)
```

---

### Configuration Schema Updates

**New Phase B config block:**
```yaml
phases:
  phase_b:
    # Dataset profiling
    enable_profiling: true
    profiling_output_path: "outputs/dataset_profile.json"
    
    # Variant selection
    library_dir: "configs/recipes/classification/variant_search"
    max_variants: 20  # How many variants to test (per task: 257 classification, 123 regression, 65 clustering)
    selection_strategy: "scored"  # scored (intelligent) | alphabetical | random_seeded
    min_relevance_score: 30.0  # Minimum score threshold (0-100)
    diversity_boost: true  # Ensure diverse preprocessing strategies
    
    # Runtime constraints
    runtime_budget_sec: 180  # Filter variants by estimated runtime
    time_budget_per_variant: 300  # Max time per variant training
    
    # Engines
    engines: ["pycaret", "flaml"]
```

---

### Updated Pipeline Builder Pattern

**Old (Hardcoded):**
```python
# pipelines/pipeline_builder.py - OLD
s6a = phaseb_pycaret(config_name=config_name, dataset_in=s4.outputs.dataset_out, recipe_name=recipe1_name)
s6b = phaseb_flaml(config_name=config_name, dataset_in=s4.outputs.dataset_out, recipe_name=recipe1_name)
s7a = phaseb_pycaret(config_name=config_name, dataset_in=s4.outputs.dataset_out, recipe_name=recipe2_name)
s7b = phaseb_flaml(config_name=config_name, dataset_in=s4.outputs.dataset_out, recipe_name=recipe2_name)

s08z = agg_phaseb(
    config_name=config_name,
    r1_pycaret_manifest=s6a.outputs.manifest_json,
    r1_pycaret_model=s6a.outputs.best_model,
    r1_flaml_manifest=s6b.outputs.manifest_json,
    r1_flaml_model=s6b.outputs.best_model,
    r2_pycaret_manifest=s7a.outputs.manifest_json,
    r2_pycaret_model=s7a.outputs.best_model,
    r2_flaml_manifest=s7b.outputs.manifest_json,
    r2_flaml_model=s7b.outputs.best_model,
)
```

**New (Single Step):**
```python
# pipelines/pipeline_builder.py - NEW
s06 = variant_runner(
    config_name=config_name,
    dataset_in=s4.outputs.dataset_out,
    variants_list=variants_list_str,   # Comma-separated variant paths (string)
    engine_list="pycaret,flaml",
    time_budget_per_variant=300,
    flaml_min_budget=120,
    planner_enabled=False,
    round1_max_variants=40,
    round2_max_variants=10,
    proxy_prune_threshold=0.50,
    cache_enabled=True
)

# Outputs already include champion + leaderboard
leaderboard = s06.outputs.leaderboard_csv
champion_manifest = s06.outputs.champion_manifest
champion_model = s06.outputs.champion_model
```

---

### Submission Script Pattern

**Updated `pipelines/submit_pipeline.py`:**
```python
# 1. Load config
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# 2. Load dataset and profile
df = pd.read_csv(dataset_path)
profiler = DatasetProfiler(task_type=cfg["task_type"])
profile = profiler.profile_dataset(df, cfg["dataset"]["target_column"])

# 3. Load all variants for this task type
task_type = cfg["task_type"]   # classification | regression | clustering
variant_dir = ROOT / "configs" / "recipes" / task_type / "variant_search"
all_variants = [load_variant(p) for p in variant_dir.glob("*.yml")]

# 4. Score and select relevant variants
recommender = VariantRecommender(profile, all_variants)
selected = recommender.select_top_variants(max_variants=20)

print("=== VARIANT SELECTION REPORT ===")
print(recommender.generate_selection_report(selected))

# 5. Build comma-separated variant path string (s06 input type: string)
variants_list_str = ",".join([str(v.path) for v, _, _ in selected])

# 6. Build pipeline job with variants_list input (string, NOT uri_file)
job = full_pipeline(
    config_name=config_name,
    dataset_folder=dataset_folder_uri,
    variants_list=variants_list_str     # Comma-separated string
)
```

> **Note:** The `variants_list` input is a **plain string**, not a JSON file
> or `Input(type="uri_file")`. The s06 component parses it with
> `args.variants_list.split(",")`.

---

### Anti-Patterns (FORBIDDEN)

❌ **DO NOT test all variants blindly** without profiling:
```python
# BAD: Wastes compute on irrelevant variants
# Classification has 210 variant_search + 44 v1_generated = 254 recipes!
variants = glob("configs/recipes/classification/variant_search/*.yml")
```

✅ **DO profile dataset and select relevant variants**:
```python
# GOOD: Intelligent selection based on data characteristics
profile = profiler.profile_dataset(df, target_column)
recommender = VariantRecommender(profile, all_variants)
selected = recommender.select_top_variants(max_variants=20)  # Top 20 only
```

---

❌ **DO NOT hardcode preprocessing strategies**:
```python
# BAD: Assumes SMOTE always needed
if task_type == "classification":
    use_smote = True
```

✅ **DO use data-driven recommendations**:
```python
# GOOD: Check if imbalance actually exists
recommendations = profile.recommend_preprocessing_strategies()
if "smote" in recommendations["imbalance_handling"]:
    use_smote = True
```

---

❌ **DO NOT skip dataset profiling in Stage 0-1**:
```python
# BAD: Directly jump to training without understanding data
df = pd.read_csv(dataset_path)
train_model(df)
```

✅ **DO profile first, then train**:
```python
# GOOD: Profile → Recommend → Select → Train
profile = profiler.profile_dataset(df, target_column)
recommendations = profile.recommend_preprocessing_strategies()
print(profile.generate_profile_summary())
selected_variants = recommender.select_top_variants(max_variants=20)
train_with_variants(selected_variants)
```

---

### Data-Driven Recommendation Examples

**Example 1: Balanced Dataset (Finance Domain)**
```
Dataset Profile:
- Imbalance ratio: 0.55 (balanced)
- Missing rate: 3%
- Outlier prevalence: 12%
- Domain: finance

Recommendations:
✅ Imputation: mean (low missing rate)
✅ Scaling: robust, minmax (outlier-resistant for finance)
❌ SMOTE: none (balanced classes)
✅ Feature selection: correlation, variance

Selected Variants: 15/257 (classification)
- Excludes all SMOTE variants (45 eliminated)
- Prioritizes robust scaling variants
```

**Example 2: Imbalanced Dataset (Academic Domain)**
```
Dataset Profile:
- Imbalance ratio: 0.22 (severe imbalance)
- Missing rate: 18%
- Domain: academic

Recommendations:
✅ Imputation: knn, iterative (high missing rate)
✅ Scaling: standard (academic domain)
✅ SMOTE: smote, smoteenn, smotetomek (critical for imbalance)
✅ Feature selection: mutual_info

Selected Variants: 18/257 (classification)
- Prioritizes SMOTE variants (score boost)
- Advanced imputation methods ranked higher
```

**Example 3: High-Dimensional Dataset**
```
Dataset Profile:
- n_features: 250
- feature_correlation_max: 0.92 (multicollinearity)

Recommendations:
✅ Feature selection: correlation, variance, rfe (critical)
✅ Imputation: mean (if low missing)
✅ Scaling: standard

Selected Variants: 12/257 (classification)
- Heavy weighting on feature selection methods
- Excludes variants without feature selection (38 eliminated)
```

---

### Testing and Validation

**Stage 0-1: Profile Output Validation**
```bash
# Profiling should generate human-readable summary
python src/steps/stage1_ingestion.py --config configs/config.yml --dataset_in ...

# Expected output:
=== DATASET PROFILE SUMMARY ===
Dimensions: 7043 rows × 19 features
Quality Issues:
  - Missing rate: 2.8%
  - Imbalance ratio: 0.27
Recommendations:
  • Severe class imbalance (0.27) → SMOTE variants critical
  • Low missing rate → Simple imputation sufficient
PRIORITY DIMENSIONS:
  • imbalance_handling: 0.95/1.0
  • feature_selection: 0.7/1.0
```

**Phase B: Variant Selection Validation**
```bash
# Selection report should show scoring logic
=== VARIANT SELECTION REPORT ===
Selected: 18 variants from 257 available (classification)

TOP RECOMMENDATIONS:
1. Variant 01ace0cb3ddd (Score: 89.5/100)
   Config: mean+onehot+none+smote+correlation
   +23.8 imbalance handling matches recommendation
   +21.2 imputation matches recommendation
   +15.0 feature selection matches recommendation

2. Variant fe89f53fba40 (Score: 85.2/100)
   Config: mean+smoteenn+onehot+none+correlation
   +23.8 imbalance handling matches recommendation
   +18.0 encoding matches recommendation
```

---

### Migration Path (Old → New Phase B)

**Step 1:** Keep old Phase B working (no breaking changes)
```yaml
# configs/config_classification.yml
phases:
  phase_b_mode: "legacy"  # Use old s6/s7 hardcoded steps
  phase_b_recipes:
    library: variant_search
    max_recipes: 2
```

**Step 2:** Enable new variant runner in parallel
```yaml
phases:
  phase_b_mode: "variant_runner"  # Use new single-step runner
  phase_b:
    enable_profiling: true
    max_variants: 20
    selection_strategy: "scored"
```

**Step 3:** Compare results and switch default
- Run both pipelines on same dataset
- Compare champion metrics, runtime, cost
- If new system performs better, make it default

---

### Performance Metrics

**Old Architecture (2 hardcoded variants):**
- Variants tested: 2 of 257 classification variants (0.8% coverage)
- Phase B steps: 4 (s6a, s6b, s7a, s7b)
- Phase B runtime: ~15 minutes
- Cost: 4 compute hours

**New Architecture (s06 variant runner, 20 selected):**
- Variants tested: 20 of 257 classification (~7.8%, intelligently selected)
- Phase B steps: 1 (`s06_phaseb_variant_runner` with nested MLflow runs)
- Phase B runtime: ~25 minutes (parallelizable internally)
- Cost: 1 compute hour (single long-running step)
- Benefit: 10× more variants tested with data-driven selection

**New Architecture with Planner Mode (2-round adaptive):**
- Round 1: up to 40 variants with proxy training (lightweight)
- Prune at threshold 0.50 → Round 2: up to 10 variants with full training
- Total runtime: ~35 minutes (proxy round is fast)
- Benefit: Tests more variants per compute dollar; prunes losers early

---

### Summary

**Key Takeaways:**
1. **Profile first, train later:** Always understand dataset before selecting preprocessing
2. **Intelligent pruning:** Use recommendations to select 10–20 relevant variants from 257+ (classification)
3. **Single-step execution:** Run all variants in one Azure ML step (`s06`) with nested MLflow runs
4. **Data-driven decisions:** No hardcoded preprocessing assumptions
5. **Scalable architecture:** Can easily increase to 50–100 variants without pipeline changes
6. **`preprocess=False`:** PyCaret `setup()` never re-preprocesses data already transformed by the variant recipe
7. **Planner Mode (V3-Proposed):** 2-round adaptive execution for large search spaces
8. **`flaml_min_budget`:** Floor of 120 s prevents FLAML from timing out at 100%

**Benefits:**
- ✅ Reduces blind grid search waste
- ✅ Improves preprocessing relevance
- ✅ Simplifies pipeline architecture (1 step vs 4+ steps)
- ✅ Better MLflow organization (nested runs)
- ✅ Domain-aware recommendations (finance vs academic)
- ✅ Task-aware variant library (classification / regression / clustering each have their own recipes)

---

### Appendix: Planner Mode (V3-Proposed Adaptive Execution)

The s06 component supports an **optional two-round adaptive planner** enabled via
`planner_enabled=true`. Planner Mode is useful when the variant library is very large
(e.g., 200+ classification variants) and you want to explore broadly without wasting
compute on losers.

**Round 1 — Proxy Training (broad, lightweight)**
| Parameter | Default | Purpose |
|-----------|---------|----------|
| `round1_max_variants` | 40 | Max variants in Round 1 |
| `proxy_prune_threshold` | 0.50 | Metric threshold to advance to Round 2 |
| `cache_enabled` | true | Cache preprocessed DataFrames across variants |

- Each variant is trained with a reduced time budget (fast proxy).
- Variants whose primary metric falls below `proxy_prune_threshold` are **pruned**.

**Round 2 — Full Training (narrow, high-budget)**
| Parameter | Default | Purpose |
|-----------|---------|----------|
| `round2_max_variants` | 10 | Max survivors from Round 1 |
| `time_budget_per_variant` | 300 | Full training budget per surviving variant |
| `flaml_min_budget` | 120 | Floor for FLAML; prevents 100 % timeout |

- Only the top-k survivors from Round 1 proceed.
- They receive the full `time_budget_per_variant` for thorough model search.

**Workflow diagram:**
```
All variants (e.g., 210)
  → Recommender prunes to round1_max (40)
    → Round 1 proxy training
      → Prune at threshold 0.50
        → Round 2 full training (10 survivors)
          → Champion selection
```

**Config snippet:**
```yaml
phases:
  phase_b:
    planner_enabled: true
    round1_max_variants: 40
    round2_max_variants: 10
    proxy_prune_threshold: 0.50
    cache_enabled: true
    time_budget_per_variant: 300
    flaml_min_budget: 120
```

**When to use Planner Mode:**
- ✅ Large variant library (>50 variants for the task type)
- ✅ Expensive engine (FLAML auto-ML with long time budgets)
- ✅ Cost-sensitive environments (Azure spot / low-priority clusters)
- ❌ Skip for small variant sets (<20) — overhead exceeds benefit
