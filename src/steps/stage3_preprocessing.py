import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.azureml_metrics_logger import create_metrics_logger, ensure_outputs_dir, safe_write_json
from utils.eda_generator import generate_correlation_heatmap, generate_sweetviz_report, load_config


def load_csv(path: str, delimiter: str = ",") -> pd.DataFrame:
    """Load CSV with specified delimiter (critical for semicolon-delimited datasets)."""
    return pd.read_csv(path, sep=delimiter)


def detect_multicollinearity(df: pd.DataFrame) -> dict:
    """
    VIF (Variance Inflation Factor) analysis for multicollinearity detection.
    VIF > 10 indicates problematic multicollinearity.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    print("   🔍 Detecting multicollinearity (VIF analysis)...")
    vif_data = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1 and len(df) > 10:
        try:
            for i, col in enumerate(numeric_cols[:50]):  # Limit to 50 features for performance
                vif = variance_inflation_factor(df[numeric_cols].values, i)
                vif_data[col] = float(vif) if not np.isinf(vif) else 999.9
            
            high_vif = {k: v for k, v in vif_data.items() if v > 10}
            if high_vif:
                print(f"      ⚠️  {len(high_vif)} features with VIF > 10 (multicollinearity)")
        except Exception as e:
            print(f"      ⚠️  VIF calculation failed: {e}")
    
    return {"vif_scores": vif_data}


def preprocess(df: pd.DataFrame, target_col: str | None, test_results: dict = None, recipe_preprocessing: dict = None, task_type: str = "classification") -> tuple[pd.DataFrame, dict]:
    """
    Recipe-driven preprocessing with encoding and scaling from recipe specifications.
    Falls back to adaptive defaults if no recipe provided.
    Supports classification, regression, and clustering task types.
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, QuantileTransformer, PowerTransformer
    
    print("\n🔧 Stage 3: Recipe-Driven Preprocessing")
    
    if recipe_preprocessing is None:
        recipe_preprocessing = {}
    
    # 1. Separate target
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])
        print(f"   🎯 Target separated: '{target_col}' (will reattach after)")
    
    # 2. CRITICAL: Apply imbalance handling BEFORE encoding (requires numeric features + target)
    imbalance_config = recipe_preprocessing.get("imbalance_handling", {})
    imbalance_method = imbalance_config.get("method", "none")
    
    print(f"\n   ⚖️  IMBALANCE HANDLING: method={imbalance_method}")
    
    # SMOTE/ADASYN must NOT be applied here (before train/test split) — causes data leakage.
    # Synthetic samples generated from the full dataset leak test-set information into training.
    # Imbalance handling is deferred to Phase B (s06), where it is applied within CV folds.
    if imbalance_method in ["smote", "adasyn"] and y is not None:
        print(f"      ⏩ {imbalance_method.upper()} deferred to Phase B (applied within CV folds to prevent data leakage)")
        print(f"      ✓ Proceeding with original class distribution")
    elif imbalance_method == "none":
        print(f"      ✓ No imbalance handling (recipe specifies 'none')")
    else:
        if y is None:
            print(f"      ⚠️  Cannot apply {imbalance_method} - target column not available")
        else:
            print(f"      ✓ No imbalance handling applied")
    
    # 3. CRITICAL: Store original numeric columns BEFORE encoding
    original_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n   🔢 Original numeric features: {len(original_numeric_cols)}")
    
    # 4. Recipe-based encoding (default: onehot with drop_first)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoding_config = recipe_preprocessing.get("encoding", {})
    encoding_method = encoding_config.get("categorical_method", "onehot")
    
    print(f"\n   🏷️  ENCODING: method={encoding_method}")
    if cat_cols:
        print(f"      📋 Categorical features to encode: {cat_cols[:5]}{'...' if len(cat_cols) > 5 else ''}")
        print(f"      📊 Total categorical features: {len(cat_cols)}")
        
        if encoding_method == "label":
            # Label encoding
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            print(f"      ✓ Label encoded: {len(cat_cols)} features")
        
        elif encoding_method == "onehot":
            # One-hot encoding with drop_first
            handle_unknown = encoding_config.get("handle_unknown", "error")
            df = pd.get_dummies(df, drop_first=True)
            print(f"      ✓ One-hot encoded: {df.shape[1]} features (handle_unknown={handle_unknown})")
        
        elif encoding_method in ["target", "catboost"]:
            # Target encoding (requires target - skip if not available)
            if y is not None:
                from category_encoders import TargetEncoder
                te = TargetEncoder(cols=cat_cols)
                df[cat_cols] = te.fit_transform(df[cat_cols], y)
                print(f"      ✓ Target encoded: {len(cat_cols)} features")
            else:
                print(f"      ⚠️  Target encoding requires target column, falling back to onehot")
                df = pd.get_dummies(df, drop_first=True)
        else:
            # Default fallback
            df = pd.get_dummies(df, drop_first=True)
            print(f"      ✓ Default one-hot encoding applied")
    else:
        print(f"   ✓ No categorical features to encode")
    
    # 5. Recipe-based Scaling - ONLY scale original numeric columns (NOT binary one-hot)
    scaling_config = recipe_preprocessing.get("scaling", {})
    scaling_method = scaling_config.get("method", "adaptive")
    
    print(f"\n   ⚙️  SCALING: method={scaling_method}")
    print(f"      📊 Original numeric columns (before encoding): {len(original_numeric_cols)}")
    
    numeric_cols_to_scale = [col for col in original_numeric_cols if col in df.columns]
    print(f"      📊 Numeric columns to scale (after encoding): {len(numeric_cols_to_scale)}")
    
    if scaling_method == "none":
        print(f"      ✓ No scaling applied (recipe specifies 'none')")
    
    elif scaling_method == "standard":
        scaler = StandardScaler()
        df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
        print(f"      ✓ StandardScaler applied to {len(numeric_cols_to_scale)} features")
    
    elif scaling_method == "robust":
        quantile_range = scaling_config.get("quantile_range", [25.0, 75.0])
        scaler = RobustScaler(quantile_range=tuple(quantile_range))
        df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
        print(f"      ✓ RobustScaler applied to {len(numeric_cols_to_scale)} features")
    
    elif scaling_method == "quantile":
        n_quantiles = scaling_config.get("n_quantiles", 1000)
        output_distribution = scaling_config.get("output_distribution", "uniform")
        scaler = QuantileTransformer(n_quantiles=min(n_quantiles, len(df)), output_distribution=output_distribution)
        df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
        print(f"      ✓ QuantileTransformer applied to {len(numeric_cols_to_scale)} features")
    
    elif scaling_method == "yeo_johnson":
        scaler = PowerTransformer(method='yeo-johnson', standardize=scaling_config.get("standardize", True))
        df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])
        print(f"      ✓ Yeo-Johnson transform applied to {len(numeric_cols_to_scale)} features")
    
    elif scaling_method == "adaptive" or not scaling_method:
        # Adaptive scaling based on statistical tests (original behavior)
        scalers_used = {"robust": 0, "standard": 0}
        for col in numeric_cols_to_scale:
            use_robust = False
            if test_results:
                is_normal = test_results.get("normality_tests", {}).get(col, {}).get("is_normal", True)
                outlier_pct = test_results.get("outlier_analysis", {}).get(col, {}).get("outlier_percentage", 0)
                if not is_normal or outlier_pct > 10:
                    use_robust = True
            
            if use_robust:
                scaler = RobustScaler()
                df[[col]] = scaler.fit_transform(df[[col]])
                scalers_used["robust"] += 1
            else:
                scaler = StandardScaler()
                df[[col]] = scaler.fit_transform(df[[col]])
                scalers_used["standard"] += 1
        print(f"      ✓ Adaptive scaling: {scalers_used['robust']} Robust, {scalers_used['standard']} Standard")
    
    print(f"      ✓ Binary features preserved (NOT scaled)")
    
    # 5. Multicollinearity detection
    multicollinearity = detect_multicollinearity(df)
    
    # 6. Reattach target
    if y is not None:
        df[target_col] = y.values
        print(f"   ✅ Target reattached")
    
    return df, multicollinearity


def generate_report(df: pd.DataFrame, multicollinearity: dict) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "sample_cols": df.columns[:10].tolist(),
        "multicollinearity_analysis": multicollinearity
    }


def save_outputs(df: pd.DataFrame, report: dict, report_dir: str, dataset_out: str, delimiter: str = ","):
    """Save outputs with preserved delimiter (critical for inter-step consistency)."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(report_dir) / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2)
    out_path = Path(dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep=delimiter, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--prep_report", required=False, default=None)
    parser.add_argument("--recipe_name", required=False, default=None)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--dataset_out", required=True)
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("⚙️  STAGE 3: PREPROCESSING (Encoding + Scaling)")
    print("="*80)
    
    # Load config
    cfg = load_config(args.config)
    task_type = cfg.get("task_type", "classification")
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")
    
    # Load recipe if provided
    recipe = None
    recipe_preprocessing = {}
    if args.recipe_name:
        import yaml
        root = Path(__file__).resolve().parents[2]
        recipe_path = root / "configs" / "recipes" / args.recipe_name
        if recipe_path.is_file():
            with open(recipe_path, "r") as f:
                recipe = yaml.safe_load(f)
            recipe_preprocessing = recipe.get("stage3_preprocessing", {})
            print(f"✅ Loaded recipe: {recipe.get('recipe_name', args.recipe_name)}")
            print(f"   📋 Encoding: {recipe_preprocessing.get('encoding', {}).get('categorical_method', 'default')}")
            print(f"   📋 Scaling: {recipe_preprocessing.get('scaling', {}).get('method', 'default')}")
        else:
            print(f"⚠️  Recipe file not found: {recipe_path}, using defaults")

    df = load_csv(args.dataset_in, delimiter=delimiter)
    print(f"📊 Input shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Load Stage 2 statistical test results from prep_report folder
    test_results = {}
    if args.prep_report:
        test_path = Path(args.prep_report) / "prep_report.json"
        if test_path.exists():
            import json
            with open(test_path) as f:
                loaded = json.load(f)
                test_results = loaded.get("statistical_tests", {})
            print(f"   ✅ Loaded Stage 2 statistical tests from prep_report for adaptive scaling")
            
            # Log test summary
            normality_count = len(test_results.get("normality_tests", {}))
            outlier_count = len(test_results.get("outlier_analysis", {}))
            print(f"      📊 {normality_count} normality tests, {outlier_count} outlier analyses")
        else:
            print(f"   ⚠️  prep_report.json not found at {test_path}, using StandardScaler")
    else:
        print(f"   ⚠️  No prep_report provided, using StandardScaler")
    
    df2, multicollinearity = preprocess(df, target_col, test_results, recipe_preprocessing, task_type=task_type)
    print(f"✅ After preprocessing: {df2.shape[0]:,} rows × {df2.shape[1]} columns")
    print(f"   🎯 All features are now numeric (encoded & scaled)")
    if recipe:
        print(f"   📋 Applied recipe: {recipe.get('recipe_name', 'unknown')}")
    
    report = generate_report(df2, multicollinearity)
    save_outputs(df2, report, args.report_dir, args.dataset_out, delimiter=delimiter)
    
    # 🎯 Multi-Stage EDA: Track transformation impact (ALL FEATURES NUMERIC!)
    print("\n🔍 Generating Stage 3 EDA (Post-Preprocessing)...")
    job_outputs_dir = Path("outputs")
    job_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Correlation heatmap (CRITICAL: all features now numeric!)
    heatmap_path = job_outputs_dir / "stage3_correlation_heatmap.png"
    generate_correlation_heatmap(df2, heatmap_path, "Stage 3 - Post-Preprocessing")
    
    # 2. Sweetviz HTML report
    sweetviz_path = job_outputs_dir / "stage3_sweetviz_report.html"
    generate_sweetviz_report(df2, sweetviz_path, "Stage 3 - Post-Preprocessing", target_col, cfg)

    # Start MLflow run
    # Create dual logger for MLflow and Azure ML
    logger = create_metrics_logger(
        run_name="s03_preprocessing",
        tags={"pipeline": "v3_mlops", "phase": "preprocessing", "step": "s03"}
    )

    # 🔥 ENTERPRISE-LEVEL LOGGING: Complete transparency of preprocessing strategy
    try:
        # Dataset dimensions
        logger.log_metric("input_rows", int(df.shape[0]))
        logger.log_metric("input_cols", int(df.shape[1]))
        logger.log_metric("output_rows", int(df2.shape[0]))
        logger.log_metric("output_cols", int(df2.shape[1]))
        logger.log_metric("features_added", int(df2.shape[1] - df.shape[1]))

        # Encoding / scaling info from recipe (safe access)
        enc_method = recipe_preprocessing.get("encoding", {}).get("categorical_method", "onehot") if recipe_preprocessing else "onehot"
        scl_method = recipe_preprocessing.get("scaling", {}).get("method", "adaptive") if recipe_preprocessing else "adaptive"
        logger.log_param("categorical_encoding", enc_method)
        logger.log_param("scaling_strategy", scl_method)

        # Statistical tests integration
        if test_results:
            normality_tests = test_results.get("normality_tests", {})
            outlier_analysis = test_results.get("outlier_analysis", {})
            logger.log_metric("features_with_normality_tests", len(normality_tests))
            logger.log_metric("features_with_outlier_analysis", len(outlier_analysis))
            logger.log_metric("normal_distributions_count", sum(1 for v in normality_tests.values() if isinstance(v, dict) and v.get("is_normal")))
            logger.log_metric("features_with_outliers", sum(1 for v in outlier_analysis.values() if isinstance(v, dict) and v.get("has_outliers")))
        else:
            logger.log_param("statistical_tests_used", "none_defaulted_to_standard")

        # Multicollinearity detection
        vif_scores = multicollinearity.get("vif_scores", {})
        if vif_scores:
            high_vif_count = sum(1 for v in vif_scores.values() if v > 10)
            logger.log_metric("features_with_vif", len(vif_scores))
            logger.log_metric("high_multicollinearity_features", high_vif_count)
            max_vif = max(vif_scores.values()) if vif_scores else 0
            logger.log_metric("max_vif_score", float(min(max_vif, 999.9)))

        # Log full report as artifact (writes to outputs/ first)
        logger.log_dict(report, "prep3_report.json")
        logger.log_dict({"encoding": enc_method, "scaling": scl_method, "statistical_tests_applied": bool(test_results)}, "preprocessing_strategy.json")

        print(f"\n✅ MLflow Logging Complete")

    except Exception as e:
        print(f"⚠️  MLflow logging failed (stage3, non-fatal): {e}")

    # End logging
    logger.end_run()
    
    print("\n" + "="*80)
    print("✅ Stage 3 Preprocessing completed successfully")
    print(f"📊 EDA outputs saved to: outputs/")
    print(f"   - prep3_report.json (preprocessing summary)")
    print(f"   - stage3_correlation_heatmap.png (🔥 encoded feature correlations)")
    print(f"   - stage3_top_correlations.csv (top 50 pairs)")
    print(f"   - stage3_sweetviz_report.html (scaled distribution viz)")
    print(f"💡 Azure ML Studio: Navigate to Outputs + logs → outputs/")
    print("="*80)


if __name__ == "__main__":
    main()
