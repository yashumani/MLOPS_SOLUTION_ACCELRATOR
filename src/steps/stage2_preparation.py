import argparse
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Add src to path for imports (single canonical insertion at front of sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.azureml_metrics_logger import create_metrics_logger
from utils.data_validator import drop_high_cardinality
from utils.eda_generator import generate_correlation_heatmap, generate_sweetviz_report, load_config
from utils.holdout_partition import (
    HOLDOUT_PARTITION,
    ROW_ID_COLUMN,
    SPLIT_COLUMN,
    TRAIN_PARTITION,
    ensure_holdout_partition,
)
from orchestration.contracts import SplitManifest, canonical_hash


def load_csv(path: str, delimiter: str = ",") -> pd.DataFrame:
    """Load CSV with specified delimiter (critical for semicolon-delimited datasets)."""
    return pd.read_csv(path, sep=delimiter)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value)]


def resolve_protected_columns(cfg: dict, target_col: str | None) -> list[str]:
    """Columns that S02 must not impute, statistically test, or drop."""
    protected: set[str] = set()
    if target_col:
        protected.add(str(target_col))

    dataset_cfg = cfg.get("dataset") or {}
    stages_cfg = cfg.get("stages") or {}
    stage2_cfg = cfg.get("stage2") or stages_cfg.get("stage2") or {}
    for section in (dataset_cfg, stage2_cfg):
        for key in (
            "id_column",
            "id_columns",
            "entity_id_column",
            "entity_id_columns",
            "protected_column",
            "protected_columns",
            "passthrough_columns",
        ):
            protected.update(_as_list(section.get(key)))
    return sorted(protected)


def resolve_excluded_feature_columns(cfg: dict, target_col: str | None) -> list[str]:
    """Columns that must never be exposed to preprocessing or model fitting."""

    dataset_cfg = cfg.get("dataset") or {}
    excluded = set(_as_list(dataset_cfg.get("id_columns")))
    excluded.update(_as_list(dataset_cfg.get("excluded_columns")))
    if target_col and str(target_col) in excluded:
        raise ValueError(
            f"Target column {target_col!r} cannot be listed in dataset.id_columns "
            "or dataset.excluded_columns"
        )
    return sorted(excluded)


def drop_excluded_feature_columns(
    frame: pd.DataFrame,
    excluded_columns: list[str],
) -> pd.DataFrame:
    """Drop configured identifiers/leakage columns and fail on schema drift."""

    missing = sorted(set(excluded_columns).difference(map(str, frame.columns)))
    if missing:
        raise ValueError(
            "Configured dataset exclusion columns are absent: " + ", ".join(missing)
        )
    return frame.drop(columns=excluded_columns)


def extract_raw_train_and_holdout(
    raw_partitioned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve the exact Stage 2 row partition and holdout identity."""

    required = {SPLIT_COLUMN, ROW_ID_COLUMN}
    missing = sorted(required.difference(raw_partitioned.columns))
    if missing:
        raise ValueError(
            "Raw partition is missing identity columns: " + ", ".join(missing)
        )
    train_rows = raw_partitioned[SPLIT_COLUMN].eq(TRAIN_PARTITION)
    holdout_rows = raw_partitioned[SPLIT_COLUMN].eq(HOLDOUT_PARTITION)
    raw_train = raw_partitioned.loc[train_rows].drop(
        columns=[SPLIT_COLUMN, ROW_ID_COLUMN]
    )
    raw_holdout = raw_partitioned.loc[holdout_rows].drop(columns=[SPLIT_COLUMN])
    holdout_ids = raw_holdout[ROW_ID_COLUMN].astype(str)
    train_ids = raw_partitioned.loc[train_rows, ROW_ID_COLUMN].astype(str)
    if not holdout_ids.is_unique:
        raise ValueError("Raw locked-test row identities must be unique")
    if set(train_ids).intersection(holdout_ids):
        raise ValueError("Raw train and locked-test row identities must be disjoint")
    return raw_train, raw_holdout


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
            except (ValueError, TypeError) as _e:
                logger.warning("shapiro failed for %s: %s", col, _e)
                p_value = 0.01
                test_name = "shapiro_failed"
        else:
            try:
                stat, p_value = kstest(df[col].dropna(), 'norm')
                test_name = "ks"
            except (ValueError, TypeError) as _e:
                logger.warning("kstest failed for %s: %s", col, _e)
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
            except (ValueError, TypeError) as _e:
                logger.warning("pearsonr failed for %s vs %s: %s", col, target_col, _e)
                continue
    
    print(f"   ✓ Normality tests: {sum(1 for r in results['normality_tests'].values() if r['is_normal'])}/{len(num_cols)} normal")
    print(f"   ✓ Outlier detection: {sum(1 for r in results['outlier_analysis'].values() if r['has_outliers'])} features with outliers")
    
    return results


def prep_dataframe(
    df: pd.DataFrame,
    target_col: str,
    recommendations: dict,
    task_type: str,
    protected_columns: list[str] | None = None,
    holdout_fraction: float = 0.2,
    random_seed: int = 42,
    split_strategy: str = "random",
    time_column: str | None = None,
):
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

    df = ensure_holdout_partition(
        df,
        target_col=target_col,
        task_type=task_type,
        holdout_fraction=holdout_fraction,
        random_seed=random_seed,
        split_strategy=split_strategy,
        time_column=time_column,
    )
    split_assignment = df.pop(SPLIT_COLUMN)
    row_identity = df.pop(ROW_ID_COLUMN)
    train_rows = split_assignment.eq(TRAIN_PARTITION)
    train_df = df.loc[train_rows]
    print(
        "   🔒 Canonical split assigned before learned preparation: "
        f"train={int(train_rows.sum()):,}, holdout={int((~train_rows).sum()):,}"
    )
    
    # 2. Identify feature columns while preserving target/ID passthrough columns.
    protected_existing = {col for col in (protected_columns or []) if col in df.columns}
    if protected_existing:
        print(f"   🛡️  Protected passthrough columns: {sorted(protected_existing)}")
    feature_df = df.drop(columns=list(protected_existing), errors="ignore")
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 3. Advanced Numeric Imputation (NO MEAN/AVERAGE - stakeholder requirement)
    print(f"   🔢 Imputing {len([c for c in num_cols if df[c].isna().any()])} numeric features...")
    strategy = recommendations.get("imputation_numeric", "median")
    
    if strategy == "knn" and len(num_cols) > 0:
        # KNN Imputer (weighted neighbors)
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        imputer.fit(train_df[num_cols])
        df[num_cols] = imputer.transform(df[num_cols])
        print(f"      ✓ KNN imputation (k=5, weighted)")
    elif strategy == "iterative" and len(num_cols) > 0:
        # Iterative Imputer (Bayesian Ridge)
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(random_state=42, max_iter=10)
        imputer.fit(train_df[num_cols])
        df[num_cols] = imputer.transform(df[num_cols])
        print(f"      ✓ Iterative imputation (Bayesian Ridge)")
    elif strategy == "ffill" and len(num_cols) > 0:
        # Forward fill then median for any remaining leading NaNs
        for col in num_cols:
            if df[col].isna().any():
                train_values = df.loc[train_rows, col].ffill()
                train_median = train_values.median()
                df.loc[train_rows, col] = train_values.fillna(train_median)
                df.loc[~train_rows, col] = df.loc[~train_rows, col].fillna(
                    train_median
                )
        print(f"      ✓ Forward-fill imputation (ffill + median fallback)")
    elif strategy == "group_median" and len(num_cols) > 0:
        # Group-based median using the lowest-cardinality categorical column
        group_col = None
        if cat_cols:
            group_col = min(cat_cols, key=lambda c: train_df[c].nunique())
        if group_col and train_df[group_col].nunique() <= 50:
            for col in num_cols:
                if df[col].isna().any():
                    group_medians = train_df.groupby(group_col)[col].median()
                    mapped_medians = df[group_col].map(group_medians)
                    df[col] = df[col].fillna(mapped_medians)
                    if df[col].isna().any():
                        df[col] = df[col].fillna(train_df[col].median())
            print(f"      ✓ Group-median imputation (grouped by '{group_col}', median fallback)")
        else:
            for col in num_cols:
                if df[col].isna().any():
                    df[col] = df[col].fillna(train_df[col].median())
            print(f"      ✓ Median imputation (group_median fallback: no suitable group column)")
    else:
        # Median (ROBUST default - no mean!)
        for col in num_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(train_df[col].median())
        print(f"      ✓ Median imputation (robust to outliers)")
    
    # 4. Categorical Imputation (Most Frequent via sklearn)
    if cat_cols:
        print(f"   🏷️  Imputing {len([c for c in cat_cols if df[c].isna().any()])} categorical features...")
        from sklearn.impute import SimpleImputer
        df[cat_cols] = df[cat_cols].replace({None: np.nan})
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.fit(df.loc[train_rows, cat_cols])
        df[cat_cols] = imputer.transform(df[cat_cols])
        print(f"      ✓ Most frequent imputation (deterministic)")
    
    # 5. Statistical Testing
    test_results = perform_statistical_tests(
        df.loc[train_rows],
        num_cols,
        target_col,
        task_type,
    )
    
    # 6. High-cardinality filtering
    stage2_max_unique = int(recommendations.get("high_cardinality_max", 100) or 100)
    _, dropped = drop_high_cardinality(
        df.loc[train_rows],
        cat_cols,
        max_unique=stage2_max_unique,
    )
    df = df.drop(columns=dropped, errors="ignore")
    if dropped:
        print(f"   🗑️  Dropped {len(dropped)} high-cardinality features (>100 unique)")
    
    df[SPLIT_COLUMN] = split_assignment.values
    df[ROW_ID_COLUMN] = row_identity.values
    return df, dropped, test_results


def generate_report(
    df: pd.DataFrame,
    dropped: list[str],
    test_results: dict,
    protected_columns: list[str] | None = None,
    high_cardinality_max: int = 100,
) -> dict:
    protected_existing = [col for col in (protected_columns or []) if col in df.columns]
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "na_total": int(df.isna().sum().sum()),
        "cat_cols": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "num_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
        "protected_columns": protected_existing,
        "protected_columns_count": len(protected_existing),
        "protected_column_policy": "protected columns are skipped during S02 imputation, statistical tests, and high-cardinality drops",
        "high_cardinality_max": int(high_cardinality_max),
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
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--raw_train_out", required=True)
    parser.add_argument("--raw_holdout_out", required=True)
    parser.add_argument("--split_manifest_out", required=True)
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
    protected_columns = resolve_protected_columns(cfg, target_col)
    excluded_feature_columns = resolve_excluded_feature_columns(cfg, target_col)

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

    stage2_cfg = cfg.get("stage2") or (cfg.get("stages") or {}).get("stage2") or {}
    high_cardinality_max = int(stage2_cfg.get("high_cardinality_max", 100) or 100)
    recommendations["high_cardinality_max"] = high_cardinality_max
    
    split_source = df.copy()
    if target_col and target_col in split_source.columns:
        split_source = split_source[split_source[target_col].notna()].copy()
    raw_partitioned = ensure_holdout_partition(
        split_source,
        target_col=target_col,
        task_type=task_type,
        holdout_fraction=float(cfg.get("holdout_fraction", 0.2)),
        random_seed=int(cfg.get("random_seed", 42)),
        split_strategy=str(cfg.get("holdout_split_strategy", "random")),
        time_column=cfg.get("holdout_time_column"),
    )

    df2, dropped, test_results = prep_dataframe(
        df,
        target_col,
        recommendations,
        task_type,
        protected_columns=protected_columns,
        holdout_fraction=float(cfg.get("holdout_fraction", 0.2)),
        random_seed=int(cfg.get("random_seed", 42)),
        split_strategy=str(cfg.get("holdout_split_strategy", "random")),
        time_column=cfg.get("holdout_time_column"),
    )
    df2 = drop_excluded_feature_columns(df2, excluded_feature_columns)
    print(f"✅ After preparation: {df2.shape[0]:,} rows × {df2.shape[1]} columns")
    analysis_df = df2.drop(
        columns=[SPLIT_COLUMN, ROW_ID_COLUMN],
        errors="ignore",
    )

    report = generate_report(
        analysis_df,
        dropped,
        test_results,
        protected_columns,
        high_cardinality_max=high_cardinality_max,
    )

    # Propagate time-series signal into the prep report
    if ts_detection:
        report["time_series_detection"] = ts_detection

    save_outputs(df2, report, args.report_dir, args.dataset_out, delimiter=delimiter)

    # S06 receives only this raw training partition. S10 receives the exact raw
    # locked holdout directly from Stage 2 and applies the selected bundle once.
    train_rows = df2[SPLIT_COLUMN].eq(TRAIN_PARTITION)
    holdout_rows = df2[SPLIT_COLUMN].eq(HOLDOUT_PARTITION)
    train_frame = df2.loc[train_rows].drop(
        columns=[SPLIT_COLUMN, ROW_ID_COLUMN],
        errors="ignore",
    )
    train_destination = Path(args.train_out)
    train_destination.parent.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(train_destination, index=False, sep=delimiter)
    raw_train_frame, raw_holdout_frame = extract_raw_train_and_holdout(
        raw_partitioned
    )
    raw_train_frame = drop_excluded_feature_columns(
        raw_train_frame,
        excluded_feature_columns,
    )
    raw_holdout_frame = drop_excluded_feature_columns(
        raw_holdout_frame,
        excluded_feature_columns,
    )
    if len(raw_train_frame) != len(train_frame):
        raise RuntimeError(
            "Raw and prepared training partitions have different row counts"
        )
    raw_train_destination = Path(args.raw_train_out)
    raw_train_destination.parent.mkdir(parents=True, exist_ok=True)
    raw_train_frame.to_csv(
        raw_train_destination,
        index=False,
        sep=delimiter,
    )
    if len(raw_holdout_frame) != int(holdout_rows.sum()):
        raise RuntimeError(
            "Raw and prepared locked-test partitions have different row counts"
        )
    raw_holdout_destination = Path(args.raw_holdout_out)
    raw_holdout_destination.parent.mkdir(parents=True, exist_ok=True)
    raw_holdout_frame.to_csv(
        raw_holdout_destination,
        index=False,
        sep=delimiter,
    )
    dataset_cfg = cfg.get("dataset") or {}
    split_manifest = SplitManifest(
        task_type=task_type,
        strategy=str(cfg.get("holdout_split_strategy", "random")),
        random_seed=int(cfg.get("random_seed", 42)),
        train_count=int(train_rows.sum()),
        validation_count=0,
        test_count=int(holdout_rows.sum()),
        train_ids_hash=canonical_hash(
            df2.loc[train_rows, ROW_ID_COLUMN].astype(str).tolist()
        ),
        validation_ids_hash=canonical_hash([]),
        test_ids_hash=canonical_hash(
            df2.loc[holdout_rows, ROW_ID_COLUMN].astype(str).tolist()
        ),
        data_version=(
            f"{dataset_cfg.get('name', 'dataset')}@"
            f"{dataset_cfg.get('version', 'unversioned')}:"
            f"{dataset_cfg.get('blob_path', '')}:"
            f"{dataset_cfg.get('content_sha256') or 'content-unverified'}"
        ),
        locked_test=True,
        group_column=dataset_cfg.get("group_column"),
        time_column=dataset_cfg.get("time_column"),
    )
    split_destination = Path(args.split_manifest_out)
    split_destination.parent.mkdir(parents=True, exist_ok=True)
    split_destination.write_text(
        split_manifest.to_json(indent=2),
        encoding="utf-8",
    )
    
    # 🎯 Multi-Stage EDA: Track preprocessing impact
    print("\n🔍 Generating Stage 2 EDA (Post-Preparation)...")
    job_outputs_dir = Path("outputs")
    job_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Correlation heatmap
    heatmap_path = job_outputs_dir / "stage2_correlation_heatmap.png"
    generate_correlation_heatmap(analysis_df, heatmap_path, "Stage 2 - Post-Preparation")
    
    # 2. Sweetviz HTML report
    sweetviz_path = job_outputs_dir / "stage2_sweetviz_report.html"
    generate_sweetviz_report(analysis_df, sweetviz_path, "Stage 2 - Post-Preparation", target_col, cfg)

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
        logger.log_param("high_cardinality_threshold", high_cardinality_max)
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
