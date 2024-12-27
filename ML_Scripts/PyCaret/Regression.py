import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pycaret.regression import setup, compare_models, pull

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/BostonHousing.csv"
TARGET_COLUMN = "medv"  # Replace with the target column name in your dataset
LOG_FILE_PATH = "pycaret_logs.log"  # Path to save logs

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    filemode="w",
)

# Step 1: Data Loading
def load_data(filepath):
    """
    Load dataset from a specified file path.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        logging.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print("File not found. Please check the filepath.")
        logging.error("File not found. Please check the filepath.")
        return None

# Step 2: Data Cleaning
def clean_data(df):
    """
    Clean the dataset by handling missing values and duplicate rows.
    """
    print("\n--- Data Cleaning ---")
    logging.info("Starting data cleaning.")

    # Handle missing values
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    logging.info(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    df = df.fillna(df.median())
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
    logging.info(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows. Removing...")
        logging.info(f"Found {duplicates} duplicate rows. Removing...")
        df = df.drop_duplicates()

    print(f"Dataset shape after cleaning: {df.shape}")
    logging.info(f"Dataset shape after cleaning: {df.shape}")
    return df

# Step 3: EDA
def perform_eda(df, target_column):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.
    """
    print("\n--- Exploratory Data Analysis ---")
    logging.info("Starting EDA.")

    # Summary Statistics
    print("\nDataset Overview:")
    print(df.info())
    logging.info(f"Dataset Info: \n{df.info()}")

    print("\nStatistical Summary:")
    print(df.describe())
    logging.info(f"Statistical Summary: \n{df.describe()}")

    # Target Variable Distribution
    print("\nTarget Variable Distribution:")
    sns.histplot(data=df, x=target_column, kde=True)
    plt.title("Target Variable Distribution")
    plt.show()

    # Correlation Heatmap
    print("\nCorrelation Heatmap:")
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()
    logging.info("EDA completed successfully.")

# Step 4: PyCaret Workflow
def run_pycaret_workflow(df, target_column):
    """
    Use PyCaret for automated feature engineering and regression model comparison.
    """
    print("\n--- PyCaret Workflow ---")
    logging.info("Starting PyCaret Workflow.")

    try:
        # Set up PyCaret
        print("\nSetting up PyCaret...")
        setup(data=df, target=target_column, session_id=123, verbose=False, feature_selection=True)
        logging.info("PyCaret setup completed.")

        # Compare models
        print("\nComparing models...")
        best_model = compare_models()
        logging.info("Model comparison completed.")

        # Save model comparison results
        results = pull()
        print("\nModel Comparison Results:")
        print(results)
        results.to_csv("model_comparison_results.csv", index=False)
        logging.info("Model comparison results saved to 'model_comparison_results.csv'.")

        return best_model
    except Exception as e:
        logging.error(f"An error occurred during PyCaret Workflow: {e}")
        print(f"An error occurred during PyCaret Workflow: {e}")
        return None

# Main Workflow
def main():
    # Load dataset
    df = load_data(DATA_PATH)
    if df is not None:
        # Clean data
        df = clean_data(df)
        
        # Perform EDA
        perform_eda(df, TARGET_COLUMN)
        
        # Run PyCaret workflow
        print("\nRunning PyCaret workflow for automated model selection and training...")
        best_model = run_pycaret_workflow(df, TARGET_COLUMN)
        if best_model:
            print(f"\nBest Model Selected by PyCaret: {best_model}")
            logging.info(f"Best Model Selected by PyCaret: {best_model}")
        else:
            print("\nPyCaret workflow failed.")
            logging.error("PyCaret workflow failed.")

if __name__ == "__main__":
    main()
