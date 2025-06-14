# src/prep_pipeline.py
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
import os
import json
import argparse
import logging
import shutil
from pathlib import Path
import sys
from typing import Optional, List
import itertools

# Import pandera with the recommended alias
import pandera.pandas as pa
from pandera import DataFrameSchema

# For custom EDA PDF
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import io

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper, EDA, and Validation Functions ---
def clear_or_create_directory(directory_path: str):
    """Clears all files and subdirectories within the specified directory or creates it."""
    path = Path(directory_path)
    if path.exists():
        logger.info(f"Clearing contents of directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists and is empty: {path}")

def generate_profiling_report(df: pd.DataFrame, output_path: str, title: str):
    """Generates an EDA report using ydata-profiling."""
    try:
        from ydata_profiling import ProfileReport
        logger.info("Generating comprehensive EDA report with ydata-profiling...")
        profile = ProfileReport(df, title=title, explorative=True, minimal=True) # Use minimal to save memory
        profile.to_file(output_path)
        logger.info(f"EDA report (ydata-profiling) saved to {output_path}")
    except ImportError:
        logger.warning("ydata-profiling not found. Skipping ydata-profiling report. Install with: pip install ydata-profiling")
    except Exception as e:
        logger.error(f"Error generating ydata-profiling report: {e}", exc_info=True)

def generate_custom_eda_pdf(df_for_eda: pd.DataFrame, output_path: str, input_file_name: str, target_column: Optional[str]):
     logger.info(f"Starting custom EDA PDF generation: {output_path}")
     # This is a placeholder for the detailed plotting functions.
     with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27)); fig.clf()
        fig.text(0.5, 0.6, 'Custom Exploratory Data Analysis Report', ha='center', va='center', fontsize=24)
        fig.text(0.5, 0.5, f"Dataset: {input_file_name}", ha='center', va='center', fontsize=18)
        pdf.savefig(fig); plt.close(fig)
     logger.info(f"Custom EDA PDF saved to {output_path}")


# --- Advanced Preprocessing Functions ---
def handle_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Handles outliers using the IQR method by capping them."""
    logger.info("Handling outliers using IQR capping...")
    df_out = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((df_out[col] < lower_bound) | (df_out[col] > upper_bound)).sum()
            if outliers_count > 0:
                logger.info(f"Capping {outliers_count} outliers in column '{col}'.")
                df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)
    return df_out

def create_interaction_features(df: pd.DataFrame, numerical_cols: List[str], n_features=5) -> pd.DataFrame:
    """Creates polynomial interaction features for the top N most variant numerical features."""
    logger.info("Creating interaction and polynomial features...")
    df_out = df.copy()
    
    if len(numerical_cols) < 2:
        logger.warning("Not enough numerical columns to create interaction features. Skipping.")
        return df_out

    top_n_features = df_out[numerical_cols].std().nlargest(min(n_features, len(numerical_cols))).index.tolist()
    if not top_n_features:
        logger.warning("Could not determine top variant features. Skipping interaction feature creation.")
        return df_out
        
    logger.info(f"Selected top {len(top_n_features)} variant features for interactions: {top_n_features}")

    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interactions = poly.fit_transform(df_out[top_n_features])
    
    interaction_feature_names = poly.get_feature_names_out(top_n_features)
    interactions_df = pd.DataFrame(interactions, columns=interaction_feature_names, index=df_out.index)
    
    interactions_df = interactions_df.drop(columns=top_n_features)
    
    df_out = pd.concat([df_out, interactions_df], axis=1)
    logger.info(f"Added {interactions_df.shape[1]} new interaction features. New total features: {df_out.shape[1]}")
    return df_out


def main(args):
    # --- DYNAMIC ARTIFACT PATHS ---
    ARTIFACTS_PATH = args.artifacts_path
    PREPARED_DATA_FILE = os.path.join(ARTIFACTS_PATH, "prepared.parquet")
    TRAIN_COLUMNS_FILE = os.path.join(ARTIFACTS_PATH, "train_columns.json")
    PREP_MANIFEST_FILE = os.path.join(ARTIFACTS_PATH, "prep_manifest.json")
    INFERRED_SCHEMA_FILE = os.path.join(ARTIFACTS_PATH, "inferred_schema.py")
    EDA_HTML_FILE = os.path.join(ARTIFACTS_PATH, "eda_report.html")
    CUSTOM_EDA_PDF_FILE = os.path.join(ARTIFACTS_PATH, "custom_eda_report.pdf")
    
    logger.info(f"Starting data preparation. Artifacts will be saved to: {ARTIFACTS_PATH}")
    clear_or_create_directory(ARTIFACTS_PATH)

    logger.info(f"Loading data from: {args.input}")
    try:
        df_raw = pd.read_csv(args.input)
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True); sys.exit(1)
    
    logger.info("Dynamically inferring data schema using Pandera...")
    try:
        inferred_schema = pa.infer_schema(df_raw)
        validated_df = inferred_schema.validate(df_raw, lazy=True)
        logger.info("Dynamic schema validation and type coercion successful.")
        with open(INFERRED_SCHEMA_FILE, "w") as f:
            f.write(inferred_schema.to_script())
    except Exception as e:
        logger.error(f"Schema inference/validation failed: {e}", exc_info=True); sys.exit(1)

    df = validated_df.drop_duplicates()
    
    if args.eda:
        generate_profiling_report(df, EDA_HTML_FILE, title=f"EDA Report for {Path(args.input).name}")
    if args.custom_eda_pdf:
        generate_custom_eda_pdf(df, CUSTOM_EDA_PDF_FILE, Path(args.input).name, args.target)

    y = None
    if args.target and args.target in df.columns:
        y = df.pop(args.target)
        logger.info(f"Target column '{args.target}' separated from features.")
    
    X = df.copy()

    potential_id_cols = [col for col in X.columns if 'id' in col.lower() and X[col].nunique() > 0.95 * len(X)]
    if potential_id_cols:
        X = X.drop(columns=potential_id_cols)
        logger.info(f"Dropped potential high-cardinality identifier columns: {potential_id_cols}")

    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if args.handle_outliers:
        X = handle_outliers_iqr(X, numerical_cols)
    
    if numerical_cols:
        if args.imputation_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        elif args.imputation_strategy == 'iterative':
            imputer = IterativeImputer(max_iter=10, random_state=42)
        else:
            imputer = SimpleImputer(strategy="mean")
        logger.info(f"Using {args.imputation_strategy} imputer for numerical features...")
        X[numerical_cols] = imputer.fit_transform(X[numerical_cols])
    
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        logger.info("Categorical features imputed with mode.")

    if args.create_interactions:
        X = create_interaction_features(X, numerical_cols)

    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dummy_na=False)
        logger.info(f"Categorical features one-hot encoded. New feature shape for X: {X.shape}")
    
    df_processed = X.copy()
    if y is not None:
        df_processed[args.target] = y

    try:
        df_processed.to_parquet(PREPARED_DATA_FILE, index=False)
        with open(TRAIN_COLUMNS_FILE, 'w') as f:
            json.dump(X.columns.tolist(), f)
        
        manifest = {
            "input_file": args.input,
            "target_column": args.target,
            "imputation_strategy": args.imputation_strategy,
            "outlier_handling_enabled": args.handle_outliers,
            "interaction_features_created": args.create_interactions
        }
        with open(PREP_MANIFEST_FILE, 'w') as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"All artifacts for this recipe saved successfully to '{ARTIFACTS_PATH}'")

    except Exception as e:
        logger.error(f"Error saving artifacts: {e}", exc_info=True)
        return

    logger.info("Data preparation pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Data Preparation & Validation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", type=str, required=False, default=None, help="Name of the target column.")
    parser.add_argument("--artifacts_path", type=str, default="artifacts", help="Path to save output artifacts.")
    parser.add_argument("--eda", action="store_true", help="Generate a comprehensive EDA report.")
    parser.add_argument("--custom_eda_pdf", action="store_true", help="Generate a custom EDA PDF report.")
    parser.add_argument("--imputation_strategy", type=str, default="mean", choices=["mean", "knn", "iterative"], help="Strategy for numerical imputation.")
    parser.add_argument("--handle_outliers", action="store_true", help="Enable outlier handling by capping with IQR method.")
    parser.add_argument("--create_interactions", action="store_true", help="Enable creation of new interaction/polynomial features.")
    
    args = parser.parse_args()
    main(args)