import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import mlflow

# Add src to path for imports (single canonical insertion at front of sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.azureml_metrics_logger import create_metrics_logger, ensure_outputs_dir, safe_write_json
from utils.eda_generator import generate_correlation_heatmap, generate_sweetviz_report, load_config
from utils.holdout_partition import (
    ROW_ID_COLUMN,
    SPLIT_COLUMN,
    TRAIN_PARTITION,
    ensure_holdout_partition,
)


def load_csv(path: str, delimiter: str = ",") -> pd.DataFrame:
    """Load CSV with specified delimiter (critical for semicolon-delimited datasets)."""
    return pd.read_csv(path, sep=delimiter)


def detect_multicollinearity(df: pd.DataFrame) -> dict:
    """
    VIF (Variance Inflation Factor) analysis for multicollinearity detection.
    VIF > 10 indicates problematic multicollinearity.
    """
    print("   🔍 Detecting multicollinearity (VIF analysis)...")
    vif_data = {}
    try:
        variance_inflation_factor = importlib.import_module(
            "statsmodels.stats.outliers_influence"
        ).variance_inflation_factor
    except ImportError as exc:
        print(f"      ⚠️  VIF analysis skipped; statsmodels is not available: {exc}")
        return {"vif_scores": vif_data, "status": "skipped", "reason": "statsmodels_unavailable"}

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


def build_preprocessing_anomaly_report(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    target_col: str | None,
    recipe_name: str | None,
    recipe_path: str | None,
    recipe_found: bool,
    multicollinearity: dict | None = None,
    test_results: dict | None = None,
) -> dict:
    excluded = [SPLIT_COLUMN, ROW_ID_COLUMN]
    if target_col:
        excluded.append(target_col)
    feature_df = output_df.drop(columns=excluded, errors="ignore")
    non_numeric_features = feature_df.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    missing_after = output_df.isna().sum()
    missing_columns_after = {
        col: int(count)
        for col, count in missing_after.items()
        if int(count) > 0
    }
    numeric_feature_df = feature_df.select_dtypes(include=[np.number, "bool"])
    infinite_columns = []
    if not numeric_feature_df.empty:
        infinite_counts = np.isinf(numeric_feature_df.to_numpy()).sum(axis=0)
        infinite_columns = [
            col
            for col, count in zip(numeric_feature_df.columns.tolist(), infinite_counts)
            if int(count) > 0
        ]

    anomalies = []
    if non_numeric_features:
        anomalies.append({"level": "error", "code": "non_numeric_features_after_s03", "columns": non_numeric_features})
    if missing_columns_after:
        anomalies.append({"level": "error", "code": "missing_values_after_s03", "columns": missing_columns_after})
    if infinite_columns:
        anomalies.append({"level": "error", "code": "infinite_values_after_s03", "columns": infinite_columns})
    if output_df.columns.duplicated().any():
        duplicates = output_df.columns[output_df.columns.duplicated()].tolist()
        anomalies.append({"level": "error", "code": "duplicate_columns_after_s03", "columns": duplicates})
    if len(input_df) != len(output_df):
        anomalies.append({"level": "warning", "code": "row_count_changed_in_s03", "before": int(len(input_df)), "after": int(len(output_df))})

    multicollinearity = multicollinearity or {}
    vif_scores = multicollinearity.get("vif_scores") or {}
    high_vif = {
        col: float(value)
        for col, value in vif_scores.items()
        if isinstance(value, (int, float)) and value > 10
    }
    if high_vif:
        anomalies.append({"level": "warning", "code": "high_multicollinearity_vif", "columns": high_vif})

    test_results = test_results or {}
    outlier_analysis = test_results.get("outlier_analysis") or {}
    high_outlier_columns = {
        col: details
        for col, details in outlier_analysis.items()
        if isinstance(details, dict) and details.get("has_outliers")
    }
    if high_outlier_columns:
        anomalies.append({"level": "warning", "code": "stage2_outlier_signals", "columns": high_outlier_columns})

    normality = test_results.get("normality_tests") or {}
    skew_or_non_normal = {
        col: details
        for col, details in normality.items()
        if isinstance(details, dict) and details.get("is_normal") is False
    }
    if skew_or_non_normal:
        anomalies.append({"level": "warning", "code": "non_normal_or_skewed_features", "columns": skew_or_non_normal})

    has_errors = any(item.get("level") == "error" for item in anomalies)
    has_warnings = any(item.get("level") == "warning" for item in anomalies)
    status = "fail" if has_errors else "warn" if has_warnings else "pass"
    return {
        "status": status,
        "recipe": {
            "name": recipe_name,
            "path": recipe_path,
            "found": recipe_found,
        },
        "input_shape": {"rows": int(input_df.shape[0]), "cols": int(input_df.shape[1])},
        "output_shape": {"rows": int(output_df.shape[0]), "cols": int(output_df.shape[1])},
        "target_column": target_col,
        "non_numeric_feature_count": len(non_numeric_features),
        "missing_columns_after_count": len(missing_columns_after),
        "infinite_columns_after_count": len(infinite_columns),
        "multicollinearity_summary": {
            "features_with_vif": len(vif_scores),
            "high_vif_count": len(high_vif),
            "high_vif_columns": high_vif,
        },
        "outlier_summary": {
            "features_checked": len(outlier_analysis),
            "features_with_outliers": len(high_outlier_columns),
            "columns": high_outlier_columns,
        },
        "distribution_summary": {
            "normality_tests": len(normality),
            "non_normal_or_skewed_count": len(skew_or_non_normal),
            "columns": skew_or_non_normal,
        },
        "anomalies": anomalies,
    }


def resolve_recipe_path(root: Path, recipe_name: str, task_type: str) -> Path | None:
    recipes_dir = root / "configs" / "recipes"
    candidates = [
        recipes_dir / recipe_name,
        recipes_dir / task_type / recipe_name,
    ]
    if recipe_name in {"recipe_baseline.yml", "baseline_recipe.yml"}:
        candidates.append(recipes_dir / task_type / "baseline_recipe.yml")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _one_hot_encode_train_only(
    df: pd.DataFrame,
    cat_cols: list[str],
    train_rows: pd.Series,
) -> pd.DataFrame:
    """One-hot encode using only categories observed in training rows."""
    encoded_parts = []
    for col in cat_cols:
        categories = sorted(
            df.loc[train_rows, col].astype(str).dropna().unique().tolist()
        )
        categorical = pd.Categorical(
            df[col].astype(str),
            categories=categories,
        )
        encoded_parts.append(
            pd.get_dummies(categorical, prefix=col, drop_first=True)
            .set_axis(df.index)
        )
    return pd.concat([df.drop(columns=cat_cols), *encoded_parts], axis=1)


def preprocess(
    df: pd.DataFrame,
    target_col: str | None,
    test_results: dict = None,
    recipe_preprocessing: dict = None,
    task_type: str = "classification",
    holdout_fraction: float = 0.2,
    random_seed: int = 42,
    split_strategy: str = "random",
    time_column: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Recipe-driven preprocessing with encoding and scaling from recipe specifications.
    Falls back to adaptive defaults if no recipe provided.
    Supports classification, regression, and clustering task types.
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
    
    print("\n🔧 Stage 3: Recipe-Driven Preprocessing")
    
    if recipe_preprocessing is None:
        recipe_preprocessing = {}

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
    print(
        "   Split assigned before learned preprocessing: "
        f"train={int(train_rows.sum()):,}, holdout={int((~train_rows).sum()):,}"
    )
    
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
            # Freeze category mappings on training rows. Holdout-only values
            # use -1 and cannot influence the mapping.
            for col in cat_cols:
                train_values = df.loc[train_rows, col].astype(str)
                classes = sorted(train_values.dropna().unique().tolist())
                mapping = {value: index for index, value in enumerate(classes)}
                df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
            print(f"      ✓ Label encoded: {len(cat_cols)} features")
        
        elif encoding_method == "onehot":
            # Build columns from training categories only. Unknown holdout
            # values become an all-zero category.
            handle_unknown = encoding_config.get("handle_unknown", "error")
            df = _one_hot_encode_train_only(df, cat_cols, train_rows)
            print(f"      ✓ One-hot encoded: {df.shape[1]} features (handle_unknown={handle_unknown})")
        
        elif encoding_method in ["target", "catboost"]:
            # Fit target statistics on training rows only.
            if y is not None:
                try:
                    TargetEncoder = importlib.import_module("category_encoders").TargetEncoder
                    te = TargetEncoder(cols=cat_cols)
                    te.fit(df.loc[train_rows, cat_cols], y.loc[train_rows])
                    df[cat_cols] = te.transform(df[cat_cols])
                    print(f"      ✓ Target encoded: {len(cat_cols)} features")
                except ImportError as exc:
                    print(f"      ⚠️  Target encoding unavailable ({exc}), falling back to onehot")
                    df = _one_hot_encode_train_only(df, cat_cols, train_rows)
            else:
                print(f"      ⚠️  Target encoding requires target column, falling back to onehot")
                df = _one_hot_encode_train_only(df, cat_cols, train_rows)
        else:
            # Default fallback
            df = _one_hot_encode_train_only(df, cat_cols, train_rows)
            print(f"      ✓ Default one-hot encoding applied")
    else:
        print(f"   ✓ No categorical features to encode")

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(np.uint8)
        print(f"   🔢 Converted {len(bool_cols)} boolean indicator columns to uint8")
    
    # 4b. 🔥 FIX: Sanitize column names to be compatible with LightGBM, XGBoost, CatBoost
    #     pd.get_dummies() can create names with [, ], <, >, {, }, etc. from categorical values.
    #     These characters crash LightGBM ("special JSON characters") and XGBoost.
    import re as _re
    _orig_cols = list(df.columns)
    _clean_map = {}
    for _col in _orig_cols:
        _clean = _re.sub(r'[\[\]<>{},:"\'\\\\ ]', '_', str(_col))
        _clean = _re.sub(r'_+', '_', _clean).strip('_')
        _clean_map[_col] = _clean
    # Handle duplicate names after sanitization
    _seen = {}
    for _orig, _clean in list(_clean_map.items()):
        if _clean in _seen:
            _seen[_clean] += 1
            _clean_map[_orig] = f"{_clean}_{_seen[_clean]}"
        else:
            _seen[_clean] = 0
    _renamed = sum(1 for o, c in _clean_map.items() if o != c)
    if _renamed > 0:
        df.columns = [_clean_map[c] for c in df.columns]
        # Also update original_numeric_cols to reflect new names
        original_numeric_cols = [_clean_map.get(c, c) for c in original_numeric_cols]
        print(f"   🔧 Sanitized {_renamed} column names (removed special chars for LightGBM/XGBoost compat)")
    
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
        scaler.fit(df.loc[train_rows, numeric_cols_to_scale])
        df[numeric_cols_to_scale] = scaler.transform(df[numeric_cols_to_scale])
        print(f"      ✓ StandardScaler applied to {len(numeric_cols_to_scale)} features")
    
    elif scaling_method == "robust":
        quantile_range = scaling_config.get("quantile_range", [25.0, 75.0])
        scaler = RobustScaler(quantile_range=tuple(quantile_range))
        scaler.fit(df.loc[train_rows, numeric_cols_to_scale])
        df[numeric_cols_to_scale] = scaler.transform(df[numeric_cols_to_scale])
        print(f"      ✓ RobustScaler applied to {len(numeric_cols_to_scale)} features")
    
    elif scaling_method == "quantile":
        n_quantiles = scaling_config.get("n_quantiles", 1000)
        output_distribution = scaling_config.get("output_distribution", "uniform")
        scaler = QuantileTransformer(
            n_quantiles=min(n_quantiles, int(train_rows.sum())),
            output_distribution=output_distribution,
        )
        scaler.fit(df.loc[train_rows, numeric_cols_to_scale])
        df[numeric_cols_to_scale] = scaler.transform(df[numeric_cols_to_scale])
        print(f"      ✓ QuantileTransformer applied to {len(numeric_cols_to_scale)} features")
    
    elif scaling_method == "yeo_johnson":
        scaler = PowerTransformer(method='yeo-johnson', standardize=scaling_config.get("standardize", True))
        scaler.fit(df.loc[train_rows, numeric_cols_to_scale])
        df[numeric_cols_to_scale] = scaler.transform(df[numeric_cols_to_scale])
        print(f"      ✓ Yeo-Johnson transform applied to {len(numeric_cols_to_scale)} features")
    
    elif scaling_method == "adaptive" or not scaling_method:
        # Select the scaler from training-only distribution statistics.
        scalers_used = {"robust": 0, "standard": 0}
        for col in numeric_cols_to_scale:
            train_values = df.loc[train_rows, col].dropna()
            q1 = train_values.quantile(0.25)
            q3 = train_values.quantile(0.75)
            iqr = q3 - q1
            outlier_pct = 0.0
            if len(train_values) and iqr > 0:
                outlier_pct = float(
                    (
                        (train_values < q1 - 1.5 * iqr)
                        | (train_values > q3 + 1.5 * iqr)
                    ).mean()
                    * 100
                )
            skew = float(train_values.skew()) if len(train_values) > 2 else 0.0
            use_robust = abs(skew) > 1.0 or outlier_pct > 10
            
            if use_robust:
                scaler = RobustScaler()
                scaler.fit(df.loc[train_rows, [col]])
                df[[col]] = scaler.transform(df[[col]])
                scalers_used["robust"] += 1
            else:
                scaler = StandardScaler()
                scaler.fit(df.loc[train_rows, [col]])
                df[[col]] = scaler.transform(df[[col]])
                scalers_used["standard"] += 1
        print(f"      ✓ Adaptive scaling: {scalers_used['robust']} Robust, {scalers_used['standard']} Standard")
    
    print(f"      ✓ Binary features preserved (NOT scaled)")
    
    # 5. Multicollinearity detection
    multicollinearity = detect_multicollinearity(df.loc[train_rows])
    
    # 6. Reattach target
    if y is not None:
        df[target_col] = y.values
        print(f"   ✅ Target reattached")
    df[SPLIT_COLUMN] = split_assignment.values
    df[ROW_ID_COLUMN] = row_identity.values
    
    return df, multicollinearity


def generate_report(df: pd.DataFrame, multicollinearity: dict, anomaly_report: dict | None = None) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "sample_cols": df.columns[:10].tolist(),
        "multicollinearity_analysis": multicollinearity,
        "preprocessing_anomaly_report": anomaly_report or {},
    }


def save_outputs(
    df: pd.DataFrame,
    report: dict,
    report_dir: str,
    dataset_out: str,
    delimiter: str = ",",
    anomaly_report: dict | None = None,
):
    """Save outputs with preserved delimiter (critical for inter-step consistency)."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    safe_write_json(Path(report_dir) / "preprocessing_report.json", report)
    safe_write_json(Path(report_dir) / "prep3_report.json", report)
    if anomaly_report is not None:
        safe_write_json(Path(report_dir) / "preprocessing_anomaly_report.json", anomaly_report)
        safe_write_json(Path(report_dir) / "anomaly_report.json", anomaly_report)

    outputs_dir = ensure_outputs_dir()
    safe_write_json(outputs_dir / "prep3_report.json", report)
    if anomaly_report is not None:
        safe_write_json(outputs_dir / "preprocessing_anomaly_report.json", anomaly_report)
        safe_write_json(outputs_dir / "anomaly_report.json", anomaly_report)

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
    recipe_path = None
    recipe_found = False
    if args.recipe_name:
        import yaml
        root = Path(__file__).resolve().parents[2]
        recipe_path = resolve_recipe_path(root, args.recipe_name, task_type)
        if recipe_path is not None:
            with open(recipe_path, "r") as f:
                recipe = yaml.safe_load(f)
            recipe_preprocessing = recipe.get("stage3_preprocessing", {})
            recipe_found = True
            print(f"✅ Loaded recipe: {recipe.get('recipe_name', args.recipe_name)}")
            print(f"   📋 Encoding: {recipe_preprocessing.get('encoding', {}).get('categorical_method', 'default')}")
            print(f"   📋 Scaling: {recipe_preprocessing.get('scaling', {}).get('method', 'default')}")
        else:
            raise FileNotFoundError(f"Recipe file not found under configs/recipes: {args.recipe_name}")

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
            print(
                "   ✅ Loaded Stage 2 statistical tests for diagnostics "
                "(learned preprocessing uses training rows only)"
            )
            
            # Log test summary
            normality_count = len(test_results.get("normality_tests", {}))
            outlier_count = len(test_results.get("outlier_analysis", {}))
            print(f"      📊 {normality_count} normality tests, {outlier_count} outlier analyses")
        else:
            print(f"   ⚠️  prep_report.json not found at {test_path}, using StandardScaler")
    else:
        print(f"   ⚠️  No prep_report provided, using StandardScaler")
    
    df2, multicollinearity = preprocess(
        df,
        target_col,
        test_results,
        recipe_preprocessing,
        task_type=task_type,
        holdout_fraction=float(cfg.get("holdout_fraction", 0.2)),
        random_seed=int(cfg.get("random_seed", 42)),
        split_strategy=str(cfg.get("holdout_split_strategy", "random")),
        time_column=cfg.get("holdout_time_column"),
    )
    print(f"✅ After preprocessing: {df2.shape[0]:,} rows × {df2.shape[1]} columns")
    print(f"   🎯 All model features are now numeric (encoded & scaled)")
    if recipe:
        print(f"   📋 Applied recipe: {recipe.get('recipe_name', 'unknown')}")
    
    anomaly_report = build_preprocessing_anomaly_report(
        df,
        df2,
        target_col,
        args.recipe_name,
        str(recipe_path) if recipe_path else None,
        recipe_found,
        multicollinearity=multicollinearity,
        test_results=test_results,
    )
    if anomaly_report.get("status") == "fail":
        print(f"⚠️  Structured S03 anomaly report status=fail: {anomaly_report.get('anomalies')}")
    analysis_df = df2.drop(
        columns=[SPLIT_COLUMN, ROW_ID_COLUMN],
        errors="ignore",
    )
    report = generate_report(analysis_df, multicollinearity, anomaly_report)
    save_outputs(df2, report, args.report_dir, args.dataset_out, delimiter=delimiter, anomaly_report=anomaly_report)
    if anomaly_report.get("status") == "fail":
        raise RuntimeError("S03 preprocessing anomaly gate failed; see preprocessing_anomaly_report.json")
    
    # 🎯 Multi-Stage EDA: Track transformation impact (ALL FEATURES NUMERIC!)
    print("\n🔍 Generating Stage 3 EDA (Post-Preprocessing)...")
    job_outputs_dir = Path("outputs")
    job_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Correlation heatmap (CRITICAL: all features now numeric!)
    heatmap_path = job_outputs_dir / "stage3_correlation_heatmap.png"
    generate_correlation_heatmap(analysis_df, heatmap_path, "Stage 3 - Post-Preprocessing")
    
    # 2. Sweetviz HTML report
    sweetviz_path = job_outputs_dir / "stage3_sweetviz_report.html"
    generate_sweetviz_report(analysis_df, sweetviz_path, "Stage 3 - Post-Preprocessing", target_col, cfg)

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
        logger.log_metric("output_cols", int(analysis_df.shape[1]))
        logger.log_metric("features_added", int(analysis_df.shape[1] - df.shape[1]))

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
