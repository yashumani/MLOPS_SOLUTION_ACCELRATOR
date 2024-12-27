import pandas as pd
import numpy as np
import featuretools as ft
from tsfresh import extract_features
from featurewiz import featurewiz
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/BostonHousing.csv"
TARGET_COLUMN = "medv"  # Replace with the target column name in your dataset

# Step 1: Data Collection
def load_data(filepath):
    """
    Load dataset from a specified file path.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print("File not found. Please check the filepath.")
        return None

# Step 2: Data Cleaning
def clean_data(df):
    """
    Clean the dataset by handling missing values and removing outliers.
    """
    print("\n--- Data Cleaning ---")
    
    # Handling missing values
    print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    df = df.fillna(df.median())
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    # Removing outliers
    for col in df.select_dtypes(include=[np.number]).columns:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    print(f"Dataset shape after outlier removal: {df.shape}")
    return df

# Step 3: Exploratory Data Analysis (EDA)
def perform_eda(df):
    """
    Perform exploratory data analysis to uncover insights.
    """
    print("\n--- EDA ---")
    print("Dataset Overview:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())

    print("\nTarget Variable Distribution:")
    sns.histplot(data=df, x=TARGET_COLUMN, kde=True)
    plt.title("Target Variable Distribution")
    plt.show()

    # Correlation heatmap
    corr_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# Step 4: Feature Engineering with FeatureTools
def feature_engineering_with_featuretools(df):
    """
    Generate features using FeatureTools.
    """
    print("\n--- Feature Engineering with FeatureTools ---")
    entity_set = ft.EntitySet(id="boston_data")
    entity_set = entity_set.entity_from_dataframe(entity_id="data", dataframe=df, index="index")
    feature_matrix, feature_defs = ft.dfs(entityset=entity_set, target_entity="data", agg_primitives=["mean", "sum", "max", "min", "std"], trans_primitives=["multiply_numeric"])
    print("Feature matrix shape:", feature_matrix.shape)
    return feature_matrix

# Step 5: Feature Engineering with TSFresh
def feature_engineering_with_tsfresh(df):
    """
    Generate features using TSFresh.
    """
    print("\n--- Feature Engineering with TSFresh ---")
    df["id"] = range(len(df))
    df["time"] = df.index
    extracted_features = extract_features(df, column_id="id", column_sort="time", default_fc_parameters="efficient")
    print("TSFresh features shape:", extracted_features.shape)
    return extracted_features

# Step 6: Feature Engineering with Featurewiz
def feature_engineering_with_featurewiz(df):
    """
    Generate features using Featurewiz.
    """
    print("\n--- Feature Engineering with Featurewiz ---")
    features, train_df = featurewiz(df, target=TARGET_COLUMN, corr_limit=0.7, verbose=2)
    print("Featurewiz selected features:", features)
    return train_df

# Main Workflow
def main():
    # Load dataset
    df = load_data(DATA_PATH)
    if df is not None:
        df = df.reset_index()  # Add index column for feature engineering
        df = clean_data(df)
        perform_eda(df)

        # Feature Engineering
        print("\nPerforming Feature Engineering with FeatureTools...")
        feature_matrix = feature_engineering_with_featuretools(df)

        print("\nPerforming Feature Engineering with TSFresh...")
        tsfresh_features = feature_engineering_with_tsfresh(df)

        print("\nPerforming Feature Engineering with Featurewiz...")
        featurewiz_features = feature_engineering_with_featurewiz(df)

        # Combine features for final dataset
        final_features = pd.concat([feature_matrix, tsfresh_features, featurewiz_features], axis=1, join="inner")
        print("\nFinal Feature Set Shape:", final_features.shape)

if __name__ == "__main__":
    main()
