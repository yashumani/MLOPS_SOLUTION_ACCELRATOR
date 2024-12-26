import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Define dataset details
DATASET = "ericamohadjei/trending-public-datasets"
DOWNLOAD_PATH = "data/"

# Create data directory if it doesn't exist
os.makedirs(DOWNLOAD_PATH, exist_ok=True)

# Download dataset
api.dataset_download_files(
    dataset=DATASET,
    path=DOWNLOAD_PATH,
    unzip=True
)

# Function to load CSV files
def load_csv_files(path):
    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not files:
        print("No CSV files found in the directory.")
        return None
    dataframes = {}
    for file in files:
        try:
            df = pd.read_csv(os.path.join(path, file))
            dataframes[file] = df
            print(f"Loaded {file}: {df.shape} rows, {df.shape[1]} columns.")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return dataframes

# Load all CSV files in the directory
dataframes = load_csv_files(DOWNLOAD_PATH)

# Perform basic EDA
def perform_eda(df):
    print("\n--- Basic EDA ---")
    print("Columns:", df.columns)
    print("Dataset Shape:", df.shape)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSample Data:")
    print(df.head())
    print("\nData Summary:")
    print(df.describe())

# Run EDA for each dataset
if dataframes:
    for file, df in dataframes.items():
        print(f"\nEDA for {file}")
        perform_eda(df)
