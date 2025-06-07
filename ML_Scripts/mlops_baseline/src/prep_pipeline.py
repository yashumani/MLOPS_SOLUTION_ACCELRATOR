# src/prep_pipeline.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
# LabelEncoder is not typically used in prep_pipeline unless the target itself needs it before splitting,
# which is usually handled in the training script. Removing if not strictly needed here.
# from sklearn.preprocessing import LabelEncoder 
import os
import json
import argparse
import logging
import shutil
from pathlib import Path
import sys

# For custom EDA PDF
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the artifacts directory
ARTIFACTS_PATH = "artifacts"
PREPARED_DATA_FILE = os.path.join(ARTIFACTS_PATH, "prepared.parquet")
TRAIN_COLUMNS_FILE = os.path.join(ARTIFACTS_PATH, "train_columns.json")
PREP_MANIFEST_FILE = os.path.join(ARTIFACTS_PATH, "prep_manifest.json")
EDA_REPORT_HTML_FILE = os.path.join(ARTIFACTS_PATH, "eda_report.html")
CUSTOM_EDA_PDF_FILE = os.path.join(ARTIFACTS_PATH, "custom_eda_report.pdf")

def clear_or_create_directory(directory_path):
    """
    Clears all files and subdirectories within the specified directory if it exists,
    or creates the directory if it doesn't exist.
    """
    path = Path(directory_path)
    if path.exists():
        logger.info(f"Clearing contents of directory: {path}")
        for item in path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        logger.info(f"Directory {path} cleared.")
    else:
        logger.info(f"Directory {path} does not exist. Creating it.")
        path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists and is empty: {path}")

def generate_profiling_report(df, output_path, title="Data Profiling Report"):
    """
    Generates an EDA report using ydata-profiling.
    """
    try:
        from ydata_profiling import ProfileReport
        logger.info("Generating comprehensive EDA report with ydata-profiling (this may take a moment)...")
        profile = ProfileReport(df, title=title, explorative=True, minimal=False) # Use minimal=False for more detail
        profile.to_file(output_path)
        logger.info(f"EDA report (ydata-profiling) saved to {output_path}")
    except ImportError:
        logger.warning("ydata-profiling library not found. Skipping ydata-profiling EDA report generation. Please install with: pip install ydata-profiling")
    except Exception as e:
        logger.error(f"Error generating ydata-profiling EDA report: {e}", exc_info=True)

def plot_histograms_to_pdf(df, numerical_cols, pdf_pages):
    """Plots histograms for numerical columns and saves them to a PDF."""
    if not numerical_cols:
        logger.info("No numerical columns to plot histograms for.")
        return
    logger.info("Generating histograms for numerical features...")
    for col in numerical_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}', fontsize=15)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            pdf_pages.savefig()
            plt.close()
        else:
            logger.warning(f"Column {col} not found in DataFrame for histogram.")
    logger.info("Histograms generated.")

def plot_boxplots_to_pdf(df, numerical_cols, categorical_target_col, pdf_pages):
    """Plots boxplots for numerical columns, optionally grouped by a categorical target."""
    if not numerical_cols:
        logger.info("No numerical columns to plot boxplots for.")
        return
    logger.info("Generating boxplots for numerical features...")
    for col in numerical_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            if categorical_target_col and categorical_target_col in df.columns and df[categorical_target_col].nunique() < 10: # Only group if target is categorical and has few unique values
                sns.boxplot(x=categorical_target_col, y=col, data=df)
                plt.title(f'Box Plot of {col} by {categorical_target_col}', fontsize=15)
            else:
                sns.boxplot(y=df[col])
                plt.title(f'Box Plot of {col}', fontsize=15)
            plt.xlabel(categorical_target_col if categorical_target_col and categorical_target_col in df.columns else '', fontsize=12)
            plt.ylabel(col, fontsize=12)
            plt.tight_layout()
            pdf_pages.savefig()
            plt.close()
        else:
            logger.warning(f"Column {col} not found in DataFrame for boxplot.")
    logger.info("Boxplots generated.")


def plot_correlation_heatmap_to_pdf(df, numerical_cols, pdf_pages):
    """Plots a correlation heatmap for numerical columns."""
    if not numerical_cols or len(numerical_cols) < 2:
        logger.info("Not enough numerical columns to plot a correlation heatmap.")
        return
    logger.info("Generating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features', fontsize=15)
    plt.tight_layout()
    pdf_pages.savefig()
    plt.close()
    logger.info("Correlation heatmap generated.")

def plot_categorical_counts_to_pdf(df, categorical_cols, pdf_pages):
    """Plots count plots for categorical columns."""
    if not categorical_cols:
        logger.info("No categorical columns to plot count plots for.")
        return
    logger.info("Generating count plots for categorical features...")
    for col in categorical_cols:
        if col in df.columns:
            if df[col].nunique() < 30: # Avoid plotting for very high cardinality categoricals
                plt.figure(figsize=(10, 6))
                sns.countplot(y=df[col], order = df[col].value_counts().index)
                plt.title(f'Count Plot of {col}', fontsize=15)
                plt.xlabel('Count', fontsize=12)
                plt.ylabel(col, fontsize=12)
                plt.tight_layout()
                pdf_pages.savefig()
                plt.close()
            else:
                logger.info(f"Skipping count plot for high cardinality categorical column: {col}")
        else:
            logger.warning(f"Column {col} not found in DataFrame for count plot.")
    logger.info("Count plots generated.")

def generate_custom_eda_pdf(df_for_eda, numerical_cols_eda, categorical_cols_eda, target_col_eda, output_path):
    """Generates a PDF with custom EDA plots."""
    logger.info(f"Starting custom EDA PDF generation: {output_path}")
    with PdfPages(output_path) as pdf:
        # Title Page
        fig = plt.figure(figsize=(11.69, 8.27)) # A4 landscape
        fig.clf()
        fig.text(0.5, 0.5, 'Custom EDA Report', ha='center', va='center', fontsize=24, weight='bold')
        fig.text(0.5, 0.4, f"Dataset: Overview", ha='center', va='center', fontsize=18)
        pdf.savefig()
        plt.close()

        plot_histograms_to_pdf(df_for_eda, numerical_cols_eda, pdf)
        plot_boxplots_to_pdf(df_for_eda, numerical_cols_eda, target_col_eda, pdf) # Pass target for potential grouping
        plot_correlation_heatmap_to_pdf(df_for_eda, numerical_cols_eda, pdf)
        plot_categorical_counts_to_pdf(df_for_eda, categorical_cols_eda, pdf)
        
        # Pairplot (select a few important columns to avoid overly large/slow plots)
        if len(numerical_cols_eda) > 1:
            sample_cols_for_pairplot = numerical_cols_eda[:min(len(numerical_cols_eda), 5)] # Max 5 columns for pairplot
            if target_col_eda and target_col_eda in df_for_eda.columns and df_for_eda[target_col_eda].nunique() < 10:
                sample_cols_for_pairplot_with_target = list(set(sample_cols_for_pairplot + [target_col_eda]))
                if len(sample_cols_for_pairplot_with_target) > 1:
                    logger.info(f"Generating pairplot for: {sample_cols_for_pairplot_with_target}")
                    pair_plot_fig = sns.pairplot(df_for_eda[sample_cols_for_pairplot_with_target], hue=target_col_eda, diag_kind='kde')
                    pdf.savefig(pair_plot_fig.fig)
                    plt.close()
            elif len(sample_cols_for_pairplot) > 1:
                logger.info(f"Generating pairplot for: {sample_cols_for_pairplot}")
                pair_plot_fig = sns.pairplot(df_for_eda[sample_cols_for_pairplot], diag_kind='kde')
                pdf.savefig(pair_plot_fig.fig)
                plt.close()

    logger.info(f"Custom EDA PDF saved to {output_path}")


def main(input_file, target_column_name, generate_profiling=False, generate_custom_pdf=False): # Added generate_custom_pdf
    logger.info("Starting data preparation pipeline...")

    clear_or_create_directory(ARTIFACTS_PATH)

    logger.info(f"Loading data from: {input_file}")
    try:
        df = pd.read_csv(input_file)
        initial_rows, initial_cols = df.shape
        logger.info(f"Successfully loaded DataFrame. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Error: Input file not found at {input_file}")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return

    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    df_for_eda = df.copy() # Use a copy for EDA before extensive manipulation

    if 'index_col_for_dfs' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'index_col_for_dfs'}, inplace=True)
        logger.info("Added 'index_col_for_dfs' as an index column.")

    df_deduplicated = df.drop_duplicates()
    logger.info(f"After initial deduplication: {df_deduplicated.shape[0]} rows, {df_deduplicated.shape[1]} columns")
    
    logger.info("\n--- Initial Data Overview (Post-Load & Dedup) ---")
    logger.info("\n[INFO] DataFrame Info:")
    # Capture info to a string buffer to log it, as df.info() prints to stdout by default
    import io
    buffer = io.StringIO()
    df_deduplicated.info(buf=buffer)
    logger.info(buffer.getvalue())
    
    logger.info("\n[INFO] Descriptive Statistics (Top 5 rows shown for brevity if large):")
    try:
        display_rows = 5 if len(df_deduplicated) > 10 else len(df_deduplicated)
        with pd.option_context('display.max_columns', None): # Show all columns for describe
            logger.info(f"\n{df_deduplicated.describe(include='all').head(display_rows if display_rows > 0 else None).to_string()}")
    except Exception as e:
        logger.warning(f"Could not generate full descriptive statistics: {e}")

    missing_values_before = df_deduplicated.isnull().sum()
    missing_values_before = missing_values_before[missing_values_before > 0]
    logger.info(f"\n[INFO] Missing Values per Column (Before Imputation):\n{missing_values_before.to_string() if not missing_values_before.empty else 'No missing values found.'}")

    if target_column_name and target_column_name in df_deduplicated.columns:
        logger.info(f"\n[INFO] Initial Target Column ('{target_column_name}') Value Counts:\n{df_deduplicated[target_column_name].value_counts().to_string() if not df_deduplicated[target_column_name].value_counts().empty else 'Target column is empty or has no values.'}")
    else:
        logger.info(f"Target column '{target_column_name}' not specified or not found for initial value counts.")
    logger.info("-" * 50)

    # Store original numerical and categorical columns for custom EDA before OHE
    original_numerical_cols = df_deduplicated.select_dtypes(include=np.number).columns.tolist()
    original_categorical_cols = df_deduplicated.select_dtypes(include=['object', 'category']).columns.tolist()
    # Exclude target and index from features for EDA plotting if they are in these lists
    if target_column_name and target_column_name in original_numerical_cols:
        original_numerical_cols.remove(target_column_name)
    if target_column_name and target_column_name in original_categorical_cols:
        original_categorical_cols.remove(target_column_name)
    if 'index_col_for_dfs' in original_numerical_cols:
        original_numerical_cols.remove('index_col_for_dfs')


    # --- Step 3: Preprocessing ---
    X = df_deduplicated.copy()
    y_series_for_eda = None # For passing to custom EDA if target exists

    if target_column_name and target_column_name in X.columns:
        y_series_for_eda = X[target_column_name].copy() # For EDA before it's dropped
        X = X.drop(columns=[target_column_name])
        logger.info(f"Target column '{target_column_name}' separated from features.")
    elif target_column_name:
        logger.warning(f"Target column '{target_column_name}' specified but not found in DataFrame. Proceeding without target separation.")
    else:
        logger.info("No target column specified. All columns will be treated as features for preprocessing (relevant for unsupervised tasks).")


    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove index_col_for_dfs from features if it's still there
    if 'index_col_for_dfs' in numerical_cols: numerical_cols.remove('index_col_for_dfs')
    if 'index_col_for_dfs' in categorical_cols: categorical_cols.remove('index_col_for_dfs')
    if 'index_col_for_dfs' in X.columns:
        X = X.drop(columns=['index_col_for_dfs'])
        logger.info("'index_col_for_dfs' dropped from feature set X.")


    logger.info(f"Numerical columns for processing: {numerical_cols}")
    logger.info(f"Categorical columns for processing: {categorical_cols}")

    if numerical_cols:
        num_imputer = SimpleImputer(strategy="mean")
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
        logger.info("Numerical features imputed with mean.")
    
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent") # Changed to most_frequent for categoricals
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        logger.info("Categorical features imputed with mode.")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dummy_na=False)
        logger.info(f"Categorical features one-hot encoded. New feature shape: {X.shape}")
    
    # df_processed will contain only features (X) for saving to prepared.parquet
    # The target column (y_series_for_eda) will be added back to prepared.parquet if it exists
    df_processed = X.copy()
    if y_series_for_eda is not None:
        df_processed[target_column_name] = y_series_for_eda

    logger.info(f"\nAfter cleaning and encoding: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")
    
    logger.info("\n--- Cleaned Data Overview (Post-Imputation & Encoding) ---")
    buffer_cleaned = io.StringIO()
    df_processed.info(buf=buffer_cleaned)
    logger.info(buffer_cleaned.getvalue())
    
    missing_values_after = df_processed.isnull().sum()
    missing_values_after = missing_values_after[missing_values_after > 0]
    logger.info(f"\n[INFO] Missing Values per Column (Cleaned):\n{missing_values_after.to_string() if not missing_values_after.empty else 'No missing values found after processing.'}")
    logger.info("-" * 50)

    # --- Step 4: Save Processed Data and Manifest ---
    try:
        df_processed.to_parquet(PREPARED_DATA_FILE, index=False)
        logger.info(f"Processed data saved to {PREPARED_DATA_FILE}")

        # train_columns.json should contain only the feature columns (X.columns)
        with open(TRAIN_COLUMNS_FILE, 'w') as f:
            json.dump(X.columns.tolist(), f) # X here is after OHE and dropping target
        logger.info(f"Feature column names (for model training) saved to {TRAIN_COLUMNS_FILE}. Count: {len(X.columns)}")

        manifest = {
            "input_file": input_file,
            "target_column": target_column_name, # This is the original target name
            "original_shape": (initial_rows, initial_cols),
            "deduplicated_shape": df_deduplicated.shape,
            "processed_feature_shape": X.shape, # Shape of X before adding target back
            "final_prepared_shape": df_processed.shape, # Shape of prepared.parquet
            "numerical_features_processed": numerical_cols,
            "categorical_features_original": original_categorical_cols, # Before OHE
            "train_columns_file": TRAIN_COLUMNS_FILE,
            "prepared_data_file": PREPARED_DATA_FILE
        }
        with open(PREP_MANIFEST_FILE, 'w') as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"Preparation manifest saved to {PREP_MANIFEST_FILE}")

    except Exception as e:
        logger.error(f"Error saving artifacts: {e}", exc_info=True)
        return

    # --- Step 5: Optional EDA Report Generation ---
    if generate_profiling:
        generate_profiling_report(df_deduplicated, EDA_REPORT_HTML_FILE, title=f"YData Profiling Report for {Path(input_file).name}")
    
    if generate_custom_pdf:
        # For custom EDA, use df_deduplicated which has original categoricals and target
        # Identify numerical and categorical columns from df_deduplicated for EDA
        eda_numerical_cols = df_deduplicated.select_dtypes(include=np.number).columns.tolist()
        eda_categorical_cols = df_deduplicated.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclude index and potentially the target if it's numerical but we want to use it for hue
        if 'index_col_for_dfs' in eda_numerical_cols: eda_numerical_cols.remove('index_col_for_dfs')
        
        # Target column for EDA grouping/hue (can be None)
        eda_target_col_for_hue = target_column_name if target_column_name in df_deduplicated.columns else None
        if eda_target_col_for_hue in eda_numerical_cols and df_deduplicated[eda_target_col_for_hue].nunique() > 20: # If target is numerical with many values, don't use for hue in boxplots
            eda_target_col_for_hue_for_boxplot = None
        else:
            eda_target_col_for_hue_for_boxplot = eda_target_col_for_hue

        generate_custom_eda_pdf(df_deduplicated, eda_numerical_cols, eda_categorical_cols, eda_target_col_for_hue_for_boxplot, CUSTOM_EDA_PDF_FILE)

    logger.info("Data preparation pipeline finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation Pipeline for MLOps Project")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--target", type=str, required=False, default=None, 
                        help="Name of the target column. If not provided, all columns are treated as features (e.g., for clustering).")
    parser.add_argument("--eda", action="store_true", help="Generate a comprehensive EDA report using ydata-profiling (HTML).")
    parser.add_argument("--custom_eda_pdf", action="store_true", help="Generate a custom EDA report with Matplotlib/Seaborn plots (PDF).")

    args = parser.parse_args()
    main(args.input, args.target, args.eda, args.custom_eda_pdf)