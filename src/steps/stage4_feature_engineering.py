import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import mlflow

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.azureml_metrics_logger import create_metrics_logger, ensure_outputs_dir, safe_write_json
from utils.eda_generator import generate_correlation_heatmap, generate_sweetviz_report, load_config


def load_csv(path: str, delimiter: str = ",") -> pd.DataFrame:
    """Load CSV with specified delimiter (critical for semicolon-delimited datasets)."""
    return pd.read_csv(path, sep=delimiter)


def detect_imbalance(y: pd.Series, task_type: str) -> dict:
    """
    Detect class imbalance for classification tasks.
    Recommends SMOTE for severe imbalance (<0.2), class weights for moderate (<0.3).
    """
    imbalance_data = {"is_imbalanced": False, "ratio": 1.0, "recommendation": "none"}
    
    if task_type == "classification" and y is not None:
        value_counts = y.value_counts()
        imbalance_ratio = value_counts.min() / value_counts.max()
        
        imbalance_data = {
            "is_imbalanced": bool(imbalance_ratio < 0.3),
            "ratio": float(imbalance_ratio),
            "class_distribution": {str(k): int(v) for k, v in value_counts.items()},
            "recommendation": "none"
        }
        
        if imbalance_ratio < 0.2:
            imbalance_data["recommendation"] = "smote"  # Severe imbalance
            print(f"      🚨 Severe imbalance detected (ratio={imbalance_ratio:.3f}) - SMOTE recommended")
        elif imbalance_ratio < 0.3:
            imbalance_data["recommendation"] = "class_weights"  # Moderate imbalance
            print(f"      ⚠️  Moderate imbalance detected (ratio={imbalance_ratio:.3f}) - Class weights recommended")
        else:
            print(f"      ✓ Balanced dataset (ratio={imbalance_ratio:.3f})")
    
    return imbalance_data


def feature_engineer(df: pd.DataFrame, target_col: str | None, task_type: str, cfg: dict, feature_selection_config: dict = None) -> tuple[pd.DataFrame, list, dict, dict]:
    """
    Recipe-driven feature engineering with support for multiple selection methods.
    Falls back to Boruta/mutual_info if no recipe provided.
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, VarianceThreshold
    from sklearn.decomposition import PCA
    
    print("\n🛠️  Stage 4: Recipe-Driven Feature Engineering")
    
    if feature_selection_config is None:
        feature_selection_config = {}
    
    # 1. Separate target
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
        print(f"   🎯 Target separated: '{target_col}'")
    else:
        X = df
        print(f"   ⚠️  No target column found (clustering task?)")
    
    initial_features = X.shape[1]
    print(f"   🔢 Initial features: {initial_features}")
    
    # 1b. 🔥 FIX (A4): Drop likely ID / row-identifier columns before feature selection
    #     These columns (e.g. customer_id) have near-unique values and leak no predictive signal,
    #     but can dominate tree splits and inflate one-hot encoding dimensions.
    import re
    _ID_NAME_PATTERN = re.compile(r'(^id$|_id$|^row_?id|^customer_?id|^transaction_?id|^user_?id|^index$)', re.IGNORECASE)
    id_cols_detected = []
    for col in X.columns:
        # Name-based detection
        if _ID_NAME_PATTERN.search(col):
            id_cols_detected.append((col, "name_pattern"))
            continue
        # Cardinality-based detection (numeric cols with >95% unique values)
        if X[col].dtype in ("int64", "float64", "int32", "float32"):
            cardinality_ratio = X[col].nunique() / max(len(X), 1)
            if cardinality_ratio > 0.95:
                id_cols_detected.append((col, f"high_cardinality({cardinality_ratio:.3f})"))
    
    if id_cols_detected:
        drop_names = [c for c, _ in id_cols_detected]
        X = X.drop(columns=drop_names)
        for col_name, reason in id_cols_detected:
            print(f"   🗑️  Dropped ID column '{col_name}' (reason: {reason})")
        print(f"   🔢 Features after ID removal: {X.shape[1]}")
    
    # 2b. 🔥 FIX: Guard against NaN values before feature selection
    #     SelectKBest and Boruta do NOT accept NaN natively. If stage3 left
    #     residual NaN (e.g., columns not fully imputed), impute them here.
    if X.isnull().any().any():
        _nan_cols = X.columns[X.isnull().any()].tolist()
        _nan_total = X.isnull().sum().sum()
        print(f"   ⚠️  Found {_nan_total} NaN values across {len(_nan_cols)} columns — imputing before feature selection")

        # Step 1: Drop columns that are 100% NaN (zero information, median=NaN)
        _all_nan_cols = [c for c in _nan_cols if X[c].isnull().all()]
        if _all_nan_cols:
            X = X.drop(columns=_all_nan_cols)
            print(f"   🗑️  Dropped {len(_all_nan_cols)} columns that are 100% NaN: {_all_nan_cols[:10]}{'...' if len(_all_nan_cols) > 10 else ''}")
            # Refresh the list to only partially-NaN columns
            _nan_cols = [c for c in _nan_cols if c not in _all_nan_cols]

        # Step 2: Impute remaining partially-NaN columns
        for _nc in _nan_cols:
            if X[_nc].dtype in ('float64', 'float32', 'int64', 'int32'):
                _med = X[_nc].median()
                X[_nc] = X[_nc].fillna(_med if _med == _med else 0)  # NaN != NaN
            else:
                _mode = X[_nc].mode()
                X[_nc] = X[_nc].fillna(_mode.iloc[0] if len(_mode) > 0 else 0)

        # Step 3: Final safety — if ANY NaN still remain, fill with 0
        if X.isnull().any().any():
            _remaining = X.isnull().sum().sum()
            print(f"   ⚠️  {_remaining} NaN values survived imputation — filling with 0")
            X = X.fillna(0)

        print(f"   ✅ NaN cleanup complete — {X.shape[1]} features, 0 NaN remain")
    
    # 2. Recipe-based Feature Selection
    selection_method = feature_selection_config.get("method", cfg.get("stage4", {}).get("selection_method", "boruta"))
    kept_cols = X.columns.tolist()
    
    print(f"   🔍 Feature selection method from recipe: {selection_method}")
    
    if selection_method == "none":
        print(f"      ✓ No feature selection (recipe specifies 'none')")
        kept_cols = X.columns.tolist()
    
    elif selection_method == "boruta" and y is not None and len(X) > 50:
        # 🔥 BORUTA Feature Selection (Recipe or Stakeholder Requirement)
        print(f"   🎯 Running Boruta feature selection...")
        
        # Get recipe parameters or use defaults
        max_iter = feature_selection_config.get("params", {}).get("max_iter", 100)
        n_estimators = feature_selection_config.get("params", {}).get("n_estimators", "auto")
        
        print(f"      Parameters: max_iter={max_iter}, n_estimators={n_estimators}")
        
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            # 🔥 SUBSAMPLE for large datasets — Boruta is O(n * max_iter * n_estimators)
            # Running on 243K rows with 100 iterations is extremely slow / may fail
            BORUTA_MAX_ROWS = 50_000
            X_boruta, y_boruta = X, y
            if len(X) > BORUTA_MAX_ROWS:
                print(f"      ℹ️  Subsampling {BORUTA_MAX_ROWS:,} / {len(X):,} rows for Boruta (performance)")
                sample_idx = X.sample(n=BORUTA_MAX_ROWS, random_state=42).index
                X_boruta = X.loc[sample_idx]
                y_boruta = y.loc[sample_idx]
            
            # Choose estimator based on task
            if task_type == "classification":
                estimator = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=42)
            else:
                estimator = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
            
            # Run Boruta with recipe parameters
            boruta_selector = BorutaPy(
                estimator=estimator,
                n_estimators=n_estimators,
                max_iter=max_iter,
                random_state=42
            )
            boruta_selector.fit(X_boruta.values, y_boruta.values)
            
            # Extract selected features (confirmed + tentative)
            kept_mask = boruta_selector.support_
            kept_cols = X.columns[kept_mask].tolist()
            
            # 🔥 Also include tentatively selected features if confirmed is empty/small
            if len(kept_cols) < 5 and hasattr(boruta_selector, "support_weak_"):
                tentative_mask = boruta_selector.support_weak_
                tentative_cols = X.columns[tentative_mask].tolist()
                if tentative_cols:
                    kept_cols = list(set(kept_cols + tentative_cols))
                    print(f"      ℹ️  Added {len(tentative_cols)} tentative features (total: {len(kept_cols)})")
            
            # 🔥 VALIDATION: Prevent zero features
            if len(kept_cols) == 0:
                print(f"      ⚠️  Boruta selected ZERO features! Falling back to mutual_info")
                # Use the underlying estimator's feature_importances_ if available
                try:
                    # BorutaPy stores the fitted estimator as .estimator_
                    _estimator = getattr(boruta_selector, "estimator_", None)
                    if _estimator is None:
                        raise AttributeError("BorutaPy has no estimator_ attribute")
                    importances = _estimator.feature_importances_
                    top_k = min(50, len(importances))
                    top_indices = np.argsort(importances)[-top_k:][::-1]
                    kept_cols = X.columns[top_indices].tolist()
                    print(f"      ℹ️  Used estimator importances as fallback: {len(kept_cols)} features")
                except (AttributeError, Exception) as imp_err:
                    print(f"      ⚠️  Importance fallback failed ({imp_err}), falling through to mutual_info")
                    selection_method = "mutual_info"  # trigger mutual_info block below
                    # 🔥 CRITICAL: Do NOT subset X to empty kept_cols — leave X
                    # intact so mutual_info works on the full feature set.
                    kept_cols = X.columns.tolist()
            
            X = X[kept_cols]
            
            print(f"      ✓ Boruta selected {len(kept_cols)}/{initial_features} features ({len(kept_cols)/initial_features*100:.1f}%)")
        except Exception as e:
            print(f"      ⚠️  Boruta failed: {e}, falling back to Mutual Info")
            selection_method = "mutual_info"
    
    if selection_method == "mutual_info" and y is not None:
        # Mutual Information (recipe or fallback)
        print(f"   📊 Using Mutual Information feature selection...")
        
        # Get k_features from recipe or default to 50
        k_features = feature_selection_config.get("params", {}).get("k_features", min(50, X.shape[1]))
        k = max(1, min(k_features, X.shape[1]))  # 🔥 FIX: Ensure k >= 1 (never 0)
        
        print(f"      Parameters: k={k} features")
        
        score_func = mutual_info_classif if task_type == "classification" else mutual_info_regression
        selector = SelectKBest(score_func=score_func, k=k)
        X_sel = selector.fit_transform(X, y)
        kept_cols = X.columns[selector.get_support(indices=True)].tolist()
        X = pd.DataFrame(X_sel, columns=kept_cols)
        print(f"      ✓ Selected top {len(kept_cols)} features by mutual information")
    
    if selection_method == "variance" or (selection_method == "boruta" and y is None):
        # Variance Threshold (recipe-based or fallback)
        threshold_val = feature_selection_config.get("params", {}).get("threshold", 0.01)
        print(f"   📏 Using variance threshold ({threshold_val})...")
        
        selector = VarianceThreshold(threshold=threshold_val)
        X_sel = selector.fit_transform(X)
        kept_cols = X.columns[selector.get_support(indices=True)].tolist()
        
        # 🔥 VALIDATION: Prevent zero features
        if len(kept_cols) == 0:
            print(f"      ⚠️  Variance threshold removed ALL features! Keeping all original features")
            kept_cols = X.columns.tolist()
        else:
            X = pd.DataFrame(X_sel, columns=kept_cols)
            print(f"      ✓ Kept {len(kept_cols)}/{initial_features} features (variance > {threshold_val})")
    
    # 3. PCA for High Dimensionality (Stakeholder Requirement)
    pca_applied = False
    pca_components = 0
    pca_threshold = cfg.get("stage4", {}).get("apply_pca_threshold", 100)
    
    if X.shape[1] > pca_threshold:
        print(f"   🔬 High dimensionality detected ({X.shape[1]} features > {pca_threshold})")
        print(f"   🔄 Applying PCA (retaining 95% variance)...")
        
        pca_variance = cfg.get("stage4", {}).get("pca_variance_retained", 0.95)
        pca = PCA(n_components=pca_variance, random_state=42)
        X_pca = pca.fit_transform(X)
        
        pca_components = X_pca.shape[1]
        pca_applied = True
        
        # Create PCA feature names
        pca_cols = [f"PC{i+1}" for i in range(pca_components)]
        X = pd.DataFrame(X_pca, columns=pca_cols)
        kept_cols = pca_cols
        
        print(f"      ✓ PCA reduced {initial_features} → {pca_components} components ({pca_variance*100}% variance)")
    
    # 4. Imbalance Detection (for training scripts)
    imbalance_metadata = detect_imbalance(y, task_type)
    
    # 🔥 FINAL VALIDATION: Ensure we have features before reattaching target
    if X.shape[1] == 0:
        raise ValueError(
            f"❌ CRITICAL: Feature engineering resulted in ZERO features! "
            f"Original features: {initial_features}, kept_cols: {kept_cols}. "
            f"This will cause failures in downstream stages."
        )
    
    # 5. Reattach target
    df_out = X.copy()
    if y is not None:
        df_out[target_col] = y.values
        print(f"   ✅ Target reattached")
    
    print(f"   🎯 Final output: {df_out.shape[1]} columns (target + {X.shape[1]} features)")
    
    pca_metadata = {"applied": pca_applied, "n_components": pca_components}
    
    return df_out, kept_cols, pca_metadata, imbalance_metadata


def generate_report(kept_cols: list, pca_metadata: dict, imbalance_metadata: dict) -> dict:
    return {
        "kept_feature_count": len(kept_cols),
        "kept_features_sample": kept_cols[:20],
        "pca_metadata": pca_metadata,
        "imbalance_metadata": imbalance_metadata
    }


def save_outputs(df: pd.DataFrame, report: dict, report_dir: str, dataset_out: str, delimiter: str = ","):
    """Save outputs with preserved delimiter (critical for inter-step consistency)."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(report_dir) / "feature_engineering_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save imbalance metadata separately for training scripts to consume
    with open(Path(report_dir) / "imbalance_metadata.json", "w") as f:
        json.dump(report["imbalance_metadata"], f, indent=2)
    
    out_path = Path(dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep=delimiter, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--recipe_name", required=False, default=None)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--dataset_out", required=True)
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔧 STAGE 4: FEATURE ENGINEERING (Selection)")
    print("="*80)
    
    # Load config
    cfg = load_config(args.config)
    task_type = cfg.get("task_type", "classification")
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")
    
    # Load recipe if provided
    recipe = None
    feature_selection_config = {}
    if args.recipe_name:
        import yaml
        root = Path(__file__).resolve().parents[2]
        recipe_path = root / "configs" / "recipes" / args.recipe_name
        if recipe_path.is_file():
            with open(recipe_path, "r") as f:
                recipe = yaml.safe_load(f)
            feature_selection_config = recipe.get("stage4_feature_engineering", {}).get("feature_selection", {})
            print(f"✅ Loaded recipe: {recipe.get('recipe_name', args.recipe_name)}")
            print(f"   📋 Feature selection: {feature_selection_config.get('method', 'none')}")
        else:
            print(f"⚠️  Recipe file not found: {recipe_path}, using defaults")

    df = load_csv(args.dataset_in, delimiter=delimiter)
    print(f"📊 Input shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    df2, kept, pca_metadata, imbalance_metadata = feature_engineer(df, target_col, task_type, cfg, feature_selection_config)
    print(f"✅ After feature engineering: {df2.shape[0]:,} rows × {df2.shape[1]} columns")
    print(f"   🎯 Final feature count: {len(kept)}")
    if recipe:
        print(f"   📋 Applied recipe: {recipe.get('recipe_name', 'unknown')}")
        print(f"   📋 Feature selection method: {feature_selection_config.get('method', 'none')}")
    
    report = generate_report(kept, pca_metadata, imbalance_metadata)
    save_outputs(df2, report, args.report_dir, args.dataset_out, delimiter=delimiter)
    
    # 🎯 Multi-Stage EDA: Final feature set quality check
    print("\n🔍 Generating Stage 4 EDA (Final Feature Set)...")
    job_outputs_dir = Path("outputs")
    job_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Correlation heatmap (FINAL: selected features only)
    heatmap_path = job_outputs_dir / "stage4_correlation_heatmap.png"
    generate_correlation_heatmap(df2, heatmap_path, "Stage 4 - Final Features")
    
    # 2. Sweetviz HTML report
    sweetviz_path = job_outputs_dir / "stage4_sweetviz_report.html"
    generate_sweetviz_report(df2, sweetviz_path, "Stage 4 - Final Features", target_col, cfg)

    # Start MLflow run
    # Create dual logger for MLflow and Azure ML
    logger = create_metrics_logger(
        run_name="s04_feature_engineering",
        tags={"pipeline": "v3_mlops", "phase": "preprocessing", "step": "s04"}
    )

    # Log feature engineering params and metrics to MLflow
    try:
        logger.log_param("feature_selector", "VarianceThreshold")
        logger.log_param("variance_threshold", 1e-6)
        logger.log_metric("kept_feature_count", int(report["kept_feature_count"]))
        logger.log_metric("rows_after_fe", int(df2.shape[0]))
        logger.log_metric("cols_after_fe", int(df2.shape[1]))
        try:
            logger.log_dict(report, "feature_engineering_report.json")
        except Exception as artifact_err:
            print(f"⚠️  MLflow artifact logging failed (non-fatal): {artifact_err}")
    except Exception as e:
        print(f"⚠️  MLflow logging failed (stage4): {e}")

    # End logging
    logger.end_run()
    
    print("\n" + "="*80)
    print("✅ Stage 4 Feature Engineering completed successfully")
    print(f"📊 EDA outputs saved to: outputs/")
    print(f"   - fe_report.json (feature engineering summary)")
    print(f"   - stage4_correlation_heatmap.png (🔥 FINAL feature correlations)")
    print(f"   - stage4_top_correlations.csv (top 50 pairs)")
    print(f"   - stage4_sweetviz_report.html (final dataset viz)")
    print(f"💡 Azure ML Studio: Navigate to Outputs + logs → outputs/")
    print(f"🎯 This is the FINAL dataset that will be used for model training!")
    print("="*80)


if __name__ == "__main__":
    main()
