import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import featuretools as ft
from ydata_profiling import ProfileReport
from sklearn.impute import SimpleImputer

# Ensure the src directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data_ingest import ingest_dataframe


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def save_artifact(obj, name: str) -> Path:
    """Saves an object to the artifacts directory."""
    path = ARTIFACTS_DIR / name
    if name.endswith(".json"):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    elif name.endswith(".parquet"):
        if isinstance(obj, pd.DataFrame):
            obj.to_parquet(path, index=False)
        else:
            raise TypeError(f"Object for .parquet must be a pandas DataFrame. Got {type(obj)}")
    elif name.endswith(".html"):
        if hasattr(obj, 'to_file'):
            obj.to_file(path)
        else:
            raise TypeError(f"Object for .html must have a 'to_file' method. Got {type(obj)}")
    else:
        raise ValueError(f"Unknown format for artifact: {name}")
    print(f"Artifact saved: {path}")
    return path


def clean_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing values in numerical columns using the median."""
    df_copy = df.copy()
    num_cols = df_copy.select_dtypes(include=[np.number]).columns
    if not num_cols.empty:
        imputer = SimpleImputer(strategy="median")
        df_copy[num_cols] = imputer.fit_transform(df_copy[num_cols])
    return df_copy


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes categorical columns."""
    df_copy = df.copy()
    cat_cols = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df_copy = pd.get_dummies(df_copy, columns=cat_cols, drop_first=True, dummy_na=False)
    return df_copy


def generate_eda(df: pd.DataFrame, title: str = "EDA Report") -> Path:
    """Generates an EDA report using YData Profiling."""
    profile = ProfileReport(df, title=title, minimal=True, explorative=True)
    return save_artifact(profile, "eda_report.html")


def generate_features_dfs(df: pd.DataFrame, entity_id: str = "main_data", index_col: str = "index_col_for_dfs") -> pd.DataFrame:
    """
    Generates features using Featuretools Deep Feature Synthesis.
    Expects df to have a unique index column named by index_col.
    """
    es = ft.EntitySet(id=entity_id)
    if index_col not in df.columns:
        raise ValueError(f"Index column '{index_col}' not found in DataFrame for Featuretools.")

    es = es.add_dataframe(
        dataframe_name="main_table",
        dataframe=df.copy(),
        index=index_col,
    )
    feature_matrix, _ = ft.dfs(entityset=es, target_dataframe_name="main_table")
    return feature_matrix


def run_pipeline(input_path: str, target_col: str, use_dfs: bool = False):
    """Main function to run the preparation pipeline."""
    print("Starting data preparation pipeline...")

    df = ingest_dataframe(input_path) # From data_ingest.py
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    INDEX_COL_DFS = "index_col_for_dfs"
    df = df.reset_index(drop=True)
    df[INDEX_COL_DFS] = df.index

    # Initial deduplication
    df = df.drop_duplicates(subset=[col for col in df.columns if col != INDEX_COL_DFS], keep='first')
    df = df.reset_index(drop=True) # Reset index again after potential row drops
    df[INDEX_COL_DFS] = df.index # Recreate the index column
    print(f"After initial deduplication: {df.shape[0]} rows, {df.shape[1]} columns")

    # === Minor EDA Prints - Stage 1 (After Load & Initial Dedup) ===
    print("\n--- Initial Data Overview (Post-Load & Dedup) ---")
    print("\n[INFO] DataFrame Info:")
    df.info(verbose=True, show_counts=True) # More detailed info
    print("\n[INFO] Descriptive Statistics (Top 5 rows shown for brevity if large):")
    with pd.option_context('display.max_rows', 5, 'display.max_columns', None): # Limit rows for describe
        print(df.describe(include='all'))
    print("\n[INFO] Missing Values per Column:")
    print(df.isnull().sum()[df.isnull().sum() > 0]) # Only show columns with missing values
    if target_col in df.columns:
        print(f"\n[INFO] Initial Target Column ('{target_col}') Value Counts:")
        print(df[target_col].value_counts(dropna=False).sort_index().to_string())
    print("--------------------------------------------------")
    # === End Minor EDA Prints - Stage 1 ===

    # Step 3: Basic cleaning (Numerical Imputation and Categorical Encoding)
    df_cleaned = clean_numerical(df)
    df_cleaned = encode_categoricals(df_cleaned)
    print(f"\nAfter cleaning and encoding: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

    # === Minor EDA Prints - Stage 2 (After Cleaning & Encoding) ===
    print("\n--- Cleaned Data Overview (Post-Imputation & Encoding) ---")
    print("\n[INFO] DataFrame Info (Cleaned):")
    df_cleaned.info(verbose=True, show_counts=True)
    print("\n[INFO] Descriptive Statistics (Cleaned - Top 5 rows shown):")
    with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
        print(df_cleaned.describe(include='all'))
    print("\n[INFO] Missing Values per Column (Cleaned):")
    print(df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0]) # Should ideally be empty
    if target_col in df_cleaned.columns:
        print(f"\n[INFO] Target Column ('{target_col}') Value Counts (Cleaned):")
        print(df_cleaned[target_col].value_counts(dropna=False).sort_index().to_string())
    print("-------------------------------------------------------")
    # === End Minor EDA Prints - Stage 2 ===

    # Step 4: Save cleaned data
    prepared_path = save_artifact(df_cleaned, "prepared.parquet")

    # Step 5: Generate EDA report on the cleaned data
    print("\nGenerating comprehensive EDA report (this may take a moment)...")
    eda_path = generate_eda(df_cleaned, title=f"EDA Report for {Path(input_path).name} (Target: {target_col})")

    # Step 6: Optional - Featuretools DFS
    dfs_matrix_path_str = None
    if use_dfs:
        print("\nGenerating features with Featuretools DFS...")
        feature_matrix_dfs = generate_features_dfs(df_cleaned, index_col=INDEX_COL_DFS)
        dfs_matrix_path = save_artifact(feature_matrix_dfs.reset_index(), "featuretools_matrix.parquet")
        dfs_matrix_path_str = str(dfs_matrix_path)
        print(f"Featuretools matrix generated: {feature_matrix_dfs.shape[0]} rows, {feature_matrix_dfs.shape[1]} columns")

    # Step 7: Write manifest
    manifest = {
        "input_path": input_path,
        "target_column": target_col,
        "prepared_data_path": str(prepared_path),
        "eda_report_path": str(eda_path),
        "dfs_feature_matrix_path": dfs_matrix_path_str,
        "index_column_dfs": INDEX_COL_DFS,
        "data_shape_prepared": df_cleaned.shape,
        "data_shape_dfs": pd.read_parquet(dfs_matrix_path_str).shape if dfs_matrix_path_str else None,
        "target_info_after_prep": df_cleaned[target_col].value_counts().to_dict() if target_col in df_cleaned else "Target not found post-prep"
    }
    save_artifact(manifest, "prep_manifest.json")
    print("\nData preparation pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset (CSV)")
    parser.add_argument("--target", type=str, required=True, help="Name of target column")
    parser.add_argument("--dfs", action="store_true", help="Enable Featuretools Deep Feature Synthesis")
    args = parser.parse_args()

    run_pipeline(args.input, args.target, args.dfs)