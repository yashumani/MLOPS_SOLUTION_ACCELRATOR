import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.azureml_metrics_logger import create_metrics_logger

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.data_validator import drop_high_cardinality
from utils.eda_generator import generate_correlation_heatmap, generate_sweetviz_report, load_config


def load_csv(path: str, delimiter: str = ",") -> pd.DataFrame:
    """Load CSV with specified delimiter (critical for semicolon-delimited datasets)."""
    return pd.read_csv(path, sep=delimiter)


def perform_statistical_tests(df: pd.DataFrame, num_cols: list, target_col: str, task_type: str) -> dict:
    """
    Statistical validation before preprocessing.
    Tests: Normality (Shapiro-Wilk/KS), Outliers (IQR), Target correlation
    """
    from scipy.stats import shapiro, kstest, pearsonr
    
    results = {
        "normality_tests": {},
        "outlier_analysis": {},
        "correlation_tests": {}
    }
    
    print("📊 Running statistical tests...")
    
    for col in num_cols:
        # Normality test (Shapiro-Wilk for n<5000, KS for larger)
        if len(df) < 5000:
            try:
                stat, p_value = shapiro(df[col].dropna())
                test_name = "shapiro"
            except:
                p_value = 0.01
                test_name = "shapiro_failed"
        else:
            try:
                stat, p_value = kstest(df[col].dropna(), 'norm')
                test_name = "ks"
            except:
                p_value = 0.01
                test_name = "ks_failed"
        
        is_normal = p_value > 0.05
        results["normality_tests"][col] = {
            "test": test_name,
            "p_value": float(p_value),
            "is_normal": bool(is_normal),
            "recommendation": "standard_scaling" if is_normal else "robust_scaling"
        }
        
        # IQR-based outlier detection
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_pct = (outliers / len(df)) * 100
        
        results["outlier_analysis"][col] = {
            "outlier_count": int(outliers),
            "outlier_percentage": float(outlier_pct),
            "has_outliers": bool(outlier_pct > 5)
        }
    
    # Correlation with target (for feature importance preview)
    if task_type in ["classification", "regression"] and target_col and target_col in df.columns:
        for col in num_cols[:20]:  # Top 20 to avoid timeout
            try:
                corr, p_val = pearsonr(df[col].dropna(), df[target_col].dropna())
                results["correlation_tests"][col] = {
                    "correlation": float(corr),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05)
                }
            except:
                continue
    
    print(f"   ✓ Normality tests: {sum(1 for r in results['normality_tests'].values() if r['is_normal'])}/{len(num_cols)} normal")
    print(f"   ✓ Outlier detection: {sum(1 for r in results['outlier_analysis'].values() if r['has_outliers'])} features with outliers")
    
    return results


def prep_dataframe(df: pd.DataFrame, target_col: str, recommendations: dict, task_type: str):
    """
    Advanced preparation with statistical testing and stakeholder-compliant imputation.
    NO AVERAGES - Uses robust estimators (median/KNN/iterative).
    """
    print("\n🧹 Stage 2: Advanced Data Preparation")
    
    # 1. Remove rows with missing target (can't train on unknown labels)
    if target_col and target_col in df.columns:
        initial_rows = len(df)
        df = df[df[target_col].notna()].copy()
        removed = initial_rows - len(df)
        if removed > 0:
            print(f"   🗑️  Removed {removed} rows with missing target ({removed/initial_rows*100:.1f}%)")
    
    # 2. Identify columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target from numeric imputation
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    # 3. Advanced Numeric Imputation (NO MEAN/AVERAGE - stakeholder requirement)
    print(f"   🔢 Imputing {len([c for c in num_cols if df[c].isna().any()])} numeric features...")
    strategy = recommendations.get("imputation_numeric", "median")
    
    if strategy == "knn" and len(num_cols) > 0:
        # KNN Imputer (weighted neighbors)
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df[num_cols] = imputer.fit_transform(df[num_cols])
        print(f"      ✓ KNN imputation (k=5, weighted)")
    elif strategy == "iterative" and len(num_cols) > 0:
        # Iterative Imputer (Bayesian Ridge)
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(random_state=42, max_iter=10)
        df[num_cols] = imputer.fit_transform(df[num_cols])
        print(f"      ✓ Iterative imputation (Bayesian Ridge)")
    elif strategy == "ffill" and len(num_cols) > 0:
        # Forward fill then median for any remaining leading NaNs
        for col in num_cols:
            if df[col].isna().any():
                df[col] = df[col].ffill()
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
        print(f"      ✓ Forward-fill imputation (ffill + median fallback)")
    elif strategy == "group_median" and len(num_cols) > 0:
        # Group-based median using the lowest-cardinality categorical column
        group_col = None
        if cat_cols:
            group_col = min(cat_cols, key=lambda c: df[c].nunique())
        if group_col and df[group_col].nunique() <= 50:
            for col in num_cols:
                if df[col].isna().any():
                    group_medians = df.groupby(group_col)[col].transform("median")
                    df[col] = df[col].fillna(group_medians)
                    if df[col].isna().any():
                        df[col] = df[col].fillna(df[col].median())
            print(f"      ✓ Group-median imputation (grouped by '{group_col}', median fallback)")
        else:
            for col in num_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].median())
            print(f"      ✓ Median imputation (group_median fallback: no suitable group column)")
    else:
        # Median (ROBUST default - no mean!)
        for col in num_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        print(f"      ✓ Median imputation (robust to outliers)")
    
    # 4. Categorical Imputation (Most Frequent via sklearn)
    if cat_cols:
        print(f"   🏷️  Imputing {len([c for c in cat_cols if df[c].isna().any()])} categorical features...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = imputer.fit_transform(df[cat_cols])
        print(f"      ✓ Most frequent imputation (deterministic)")
    
    # 5. Statistical Testing
    test_results = perform_statistical_tests(df, num_cols, target_col, task_type)
    
    # 6. High-cardinality filtering
    df, dropped = drop_high_cardinality(df, cat_cols, max_unique=100)
    if dropped:
        print(f"   🗑️  Dropped {len(dropped)} high-cardinality features (>100 unique)")
    
    return df, dropped, test_results


def generate_report(df: pd.DataFrame, dropped: list[str], test_results: dict) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "na_total": int(df.isna().sum().sum()),
        "cat_cols": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "num_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
        "dropped_high_cardinality": dropped,
        "dropped_high_cardinality_count": len(dropped),
        "statistical_tests": test_results
    }


def save_outputs(df: pd.DataFrame, report: dict, report_dir: str, dataset_out: str, delimiter: str = ","):
    """Save outputs with preserved delimiter (critical for inter-step consistency)."""
    # V4 Pattern: Write to outputs/ folder for Studio visibility
    job_outputs_dir = Path("outputs")
    job_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save preparation report to outputs/
    prep_report_path = job_outputs_dir / "prep_report.json"
    with open(prep_report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Also save to component output parameter for backwards compatibility
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(report_dir) / "prep_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save dataset to component output with preserved delimiter
    out_path = Path(dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep=delimiter, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--dataset_out", required=True)
    parser.add_argument("--eda_dir", required=False, default=None,
                        help="Formal Stage 1 EDA output folder (preferred over filesystem hack)")
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("📦 STAGE 2: DATA PREPARATION (Imputation + Filtering)")
    print("="*80)
    
    # Load config
    cfg = load_config(args.config)
    task_type = cfg.get("task_type", "classification")
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")

    df = load_csv(args.dataset_in, delimiter=delimiter)
    print(f"📊 Input shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Load Stage 1 recommendations (formal eda_dir input > filesystem fallback)
    recommendations = {}
    ts_detection = {}
    eda_dir = None
    if args.eda_dir and Path(args.eda_dir).exists():
        eda_dir = Path(args.eda_dir)
        print(f"   ✓ Using formal Stage 1 EDA folder: {eda_dir}")
    else:
        # Legacy fallback: infer path from dataset_in location
        eda_dir_fallback = Path(args.dataset_in).parent.parent / "s1"
        if eda_dir_fallback.exists():
            eda_dir = eda_dir_fallback
            print(f"   ⚠️  Using filesystem fallback for Stage 1 signals: {eda_dir}")

    if eda_dir:
        rec_path = eda_dir / "recipe_recommendations.json"
        if rec_path.exists():
            with open(rec_path) as f:
                recommendations = json.load(f)
            print(f"   ✓ Loaded Stage 1 recommendations: {recommendations.get('imputation_numeric', 'median')} imputation")
        else:
            print(f"   ⚠️  No recipe_recommendations.json in {eda_dir}")

        ts_path = eda_dir / "time_series_detection.json"
        if ts_path.exists():
            with open(ts_path) as f:
                ts_detection = json.load(f)
            if ts_detection.get("is_time_series"):
                print(f"   🕐 TIME-SERIES signal from Stage 1: column={ts_detection.get('time_column')}, confidence={ts_detection.get('confidence')}")
            else:
                print(f"   🕐 Time-series: NOT detected (confidence={ts_detection.get('confidence', 0)})")
    else:
        print(f"   ⚠️  No Stage 1 EDA folder found, using defaults")
    
    df2, dropped, test_results = prep_dataframe(df, target_col, recommendations, task_type)
    print(f"✅ After preparation: {df2.shape[0]:,} rows × {df2.shape[1]} columns")
    
    report = generate_report(df2, dropped, test_results)

    # Propagate time-series signal into the prep report
    if ts_detection:
        report["time_series_detection"] = ts_detection

    save_outputs(df2, report, args.report_dir, args.dataset_out, delimiter=delimiter)
    
    # 🎯 Multi-Stage EDA: Track preprocessing impact
    print("\n🔍 Generating Stage 2 EDA (Post-Preparation)...")
    job_outputs_dir = Path("outputs")
    job_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Correlation heatmap
    heatmap_path = job_outputs_dir / "stage2_correlation_heatmap.png"
    generate_correlation_heatmap(df2, heatmap_path, "Stage 2 - Post-Preparation")
    
    # 2. Sweetviz HTML report
    sweetviz_path = job_outputs_dir / "stage2_sweetviz_report.html"
    generate_sweetviz_report(df2, sweetviz_path, "Stage 2 - Post-Preparation", target_col, cfg)

    # Start MLflow run
    # Create dual logger for MLflow and Azure ML
    logger = create_metrics_logger(
        run_name="s02_preparation",
        tags={"pipeline": "v3_mlops", "phase": "preprocessing", "step": "s02"}
    )

    # Log preparation parameters and metrics to MLflow
    try:
        actual_numeric_strategy = recommendations.get("imputation_numeric", "median")
        logger.log_param("impute_numeric", actual_numeric_strategy)
        logger.log_param("impute_categorical", "most_frequent")
        logger.log_param("high_cardinality_threshold", 100)
        logger.log_metric("rows", int(report["rows"]))
        logger.log_metric("cols", int(report["cols"]))
        logger.log_metric("na_total", int(report["na_total"]))
        logger.log_metric("num_cols_count", len(report["num_cols"]))
        logger.log_metric("cat_cols_count", len(report["cat_cols"]))
        logger.log_metric("high_cardinality_dropped", int(report["dropped_high_cardinality_count"]))
        try:
            logger.log_dict(report, "prep_report.json")
        except Exception as artifact_err:
            print(f"⚠️  MLflow artifact logging failed (non-fatal): {artifact_err}")
    except Exception as e:
        print(f"⚠️  MLflow logging failed (stage2): {e}")

    # End logging
    logger.end_run()
    
    print("\n" + "="*80)
    print("✅ Stage 2 Preparation completed successfully")
    print(f"📊 EDA outputs saved to: outputs/")
    print(f"   - prep_report.json (preparation summary)")
    print(f"   - stage2_correlation_heatmap.png (🔥 feature relationships)")
    print(f"   - stage2_top_correlations.csv (top 50 pairs)")
    print(f"   - stage2_sweetviz_report.html (interactive viz)")
    print(f"💡 Azure ML Studio: Navigate to Outputs + logs → outputs/")
    print("="*80)


if __name__ == "__main__":
    main()
