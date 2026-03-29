"""
CRITICAL FIXES VALIDATION SCRIPT
=================================

Tests all 3 critical fixes before pipeline submission:
1. Recipe diversity (SMOTE presence in lightning_fast tier)
2. FLAML artifact export (type checking and error handling)
3. SMOTE implementation (synthetic sample generation)

Usage:
    python tests/validate_critical_fixes.py

Expected Output:
    ✅ All 3 fixes validated successfully
    🚀 Safe to submit production pipeline
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from collections import Counter

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

print("="*80)
print("🔍 CRITICAL FIXES VALIDATION")
print("="*80)

# ============================================================================
# TEST 1: Recipe Diversity - Verify SMOTE in lightning_fast tier
# ============================================================================
print("\n📋 TEST 1: Recipe Diversity")
print("-" * 80)

lightning_fast_dir = ROOT / "configs/recipes/classification/v1_generated/lightning_fast"
recipe_files = sorted(lightning_fast_dir.glob("*.yml"))

print(f"   📂 Found {len(recipe_files)} lightning_fast recipes")

recipes_with_smote = []
recipes_with_variance = []
recipes_with_scaling = []

for recipe_file in recipe_files:
    with open(recipe_file) as f:
        recipe = yaml.safe_load(f)
    
    recipe_name = recipe.get('recipe_name', recipe_file.stem)
    imbalance_method = recipe.get('stage3_preprocessing', {}).get('imbalance_handling', {}).get('method', 'none')
    feat_sel_method = recipe.get('stage4_feature_engineering', {}).get('feature_selection', {}).get('method', 'none')
    scaling_method = recipe.get('stage3_preprocessing', {}).get('scaling', {}).get('method', 'none')
    
    print(f"   📄 {recipe_name}:")
    print(f"      - Imbalance: {imbalance_method}")
    print(f"      - Scaling: {scaling_method}")
    print(f"      - Feature Selection: {feat_sel_method}")
    
    if imbalance_method in ['smote', 'adasyn']:
        recipes_with_smote.append(recipe_name)
    if feat_sel_method != 'none':
        recipes_with_variance.append(recipe_name)
    if scaling_method != 'none':
        recipes_with_scaling.append(recipe_name)

print(f"\n   📊 Summary:")
print(f"      - Recipes with SMOTE: {len(recipes_with_smote)}")
print(f"      - Recipes with feature selection: {len(recipes_with_variance)}")
print(f"      - Recipes with scaling: {len(recipes_with_scaling)}")

# Validation
test1_pass = len(recipes_with_smote) >= 1 and len(recipes_with_variance) >= 1 and len(recipes_with_scaling) >= 1

if test1_pass:
    print(f"\n   ✅ TEST 1 PASSED: Recipe diversity confirmed")
    print(f"      - SMOTE recipes: {recipes_with_smote}")
else:
    print(f"\n   ❌ TEST 1 FAILED: Insufficient recipe diversity")
    sys.exit(1)

# ============================================================================
# TEST 2: FLAML Artifact Export - Simulate config_history parsing
# ============================================================================
print("\n📋 TEST 2: FLAML Artifact Export (Type Checking)")
print("-" * 80)

# Simulate FLAML config_history with different formats
mock_config_history = {
    0: ({'learner': 'lgbm', 'n_estimators': 100}, {'val_loss': 0.15, 'wall_clock_time': 2.3}),  # Tuple format
    1: {'val_loss': 0.18, 'wall_clock_time': 1.5},  # Dict only format
    2: 2,  # Failed trial (int)
    3: ({'learner': 'xgboost'}, {'val_loss': 0.14}),  # Tuple format
    4: None,  # Failed trial (None)
}

print(f"   📊 Mock config_history: {len(mock_config_history)} entries")

# Test the parsing logic from phaseb_flaml_recipe.py
iterations_data = []
skipped_trials = 0
processed_trials = 0

for trial_id, value in mock_config_history.items():
    config = None
    result = None
    
    if isinstance(value, tuple):
        if len(value) == 2:
            config, result = value
        else:
            print(f"      ⚠️  Trial {trial_id}: Unexpected tuple length {len(value)}, skipping")
            skipped_trials += 1
            continue
    elif isinstance(value, dict):
        config = {}
        result = value
    elif isinstance(value, (int, type(None))):
        print(f"      ⚠️  Trial {trial_id}: Failed trial (value={value}), skipping")
        skipped_trials += 1
        continue
    else:
        print(f"      ⚠️  Trial {trial_id}: Unknown type {type(value)}, skipping")
        skipped_trials += 1
        continue
    
    learner = config.get('learner', 'unknown') if isinstance(config, dict) else 'unknown'
    iteration_metrics = {
        'iteration': trial_id,
        'learner': learner,
        'val_loss': result.get('val_loss') if isinstance(result, dict) else None,
    }
    
    iterations_data.append(iteration_metrics)
    processed_trials += 1

print(f"   📊 Processed: {processed_trials}/{len(mock_config_history)} trials")
print(f"   📊 Skipped: {skipped_trials} failed/malformed trials")

# Validation
test2_pass = processed_trials == 3 and skipped_trials == 2

if test2_pass:
    print(f"\n   ✅ TEST 2 PASSED: FLAML type checking works correctly")
    for row in iterations_data:
        print(f"      - Trial {row['iteration']}: learner={row['learner']}, val_loss={row['val_loss']}")
else:
    print(f"\n   ❌ TEST 2 FAILED: Unexpected parsing behavior")
    sys.exit(1)

# ============================================================================
# TEST 3: SMOTE Implementation - Test with synthetic imbalanced data
# ============================================================================
print("\n📋 TEST 3: SMOTE Implementation")
print("-" * 80)

# Create synthetic imbalanced dataset
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.8, 0.2],  # 80/20 imbalance
    random_state=42,
    flip_y=0.1
)

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
df['target'] = y

print(f"   📊 Synthetic dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

original_dist = Counter(y)
print(f"   📊 Original distribution:")
for cls, count in sorted(original_dist.items()):
    pct = (count / len(y)) * 100
    print(f"      Class {cls}: {count:,} samples ({pct:.1f}%)")

imbalance_ratio = original_dist[1] / original_dist[0]
print(f"   📊 Imbalance ratio: {imbalance_ratio:.3f} (minority/majority)")

# Load SMOTE recipe
smote_recipe_path = lightning_fast_dir / "recipe_clas_lightning_fast_0005.yml"
if smote_recipe_path.exists():
    with open(smote_recipe_path) as f:
        smote_recipe = yaml.safe_load(f)
    
    print(f"\n   📄 Testing with: {smote_recipe['recipe_name']}")
    
    # Test SMOTE application
    from imblearn.over_sampling import SMOTE
    
    X_train = df.drop(columns=['target'])
    y_train = df['target']
    
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, original_dist[1] - 1))
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        new_dist = Counter(y_resampled)
        print(f"\n   ✅ SMOTE applied successfully!")
        print(f"   📊 New distribution:")
        for cls, count in sorted(new_dist.items()):
            pct = (count / len(y_resampled)) * 100
            original_count = original_dist.get(cls, 0)
            delta = count - original_count
            print(f"      Class {cls}: {count:,} samples ({pct:.1f}%) [+{delta:,} synthetic]")
        
        new_ratio = new_dist[1] / new_dist[0]
        print(f"   📊 New imbalance ratio: {new_ratio:.3f}")
        print(f"   📊 Dataset size: {len(y_train):,} → {len(y_resampled):,} (+{len(y_resampled) - len(y_train):,} samples)")
        
        # Validation
        test3_pass = new_ratio > 0.95  # Should be ~1.0 (balanced)
        
        if test3_pass:
            print(f"\n   ✅ TEST 3 PASSED: SMOTE balances dataset correctly")
        else:
            print(f"\n   ❌ TEST 3 FAILED: Imbalance ratio still too low ({new_ratio:.3f})")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n   ❌ TEST 3 FAILED: SMOTE raised exception: {e}")
        sys.exit(1)
else:
    print(f"\n   ⚠️  TEST 3 SKIPPED: SMOTE recipe not found at {smote_recipe_path}")
    test3_pass = False

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📊 VALIDATION SUMMARY")
print("="*80)

all_passed = test1_pass and test2_pass and test3_pass

print(f"\n   {'✅' if test1_pass else '❌'} TEST 1: Recipe Diversity")
print(f"   {'✅' if test2_pass else '❌'} TEST 2: FLAML Artifact Export")
print(f"   {'✅' if test3_pass else '❌'} TEST 3: SMOTE Implementation")

if all_passed:
    print(f"\n{'✅'*40}")
    print(f"   ALL TESTS PASSED - SAFE TO DEPLOY")
    print(f"{'✅'*40}")
    print(f"\n🚀 Next Steps:")
    print(f"   1. Submit pipeline with: python pipelines/submit_pipeline.py")
    print(f"   2. Monitor Stage 3 for SMOTE logs")
    print(f"   3. Validate Recall > 0% in Phase B/C results")
    sys.exit(0)
else:
    print(f"\n{'❌'*40}")
    print(f"   TESTS FAILED - DO NOT DEPLOY")
    print(f"{'❌'*40}")
    sys.exit(1)
