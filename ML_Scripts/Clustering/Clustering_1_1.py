import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.clustering import setup as clu_setup, create_model as clu_create_model, assign_model as clu_assign_model, evaluate_model as clu_evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import logging
import os
import sweetviz as sv
import optuna
from flaml import AutoML

# Configure logging
log_path = 'C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Clustering/Reports/logs.log'
logging.basicConfig(
    filename=log_path, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/datasets/supermarket_sales.csv"
REPORTS_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Clustering/Reports"

# Ensure the reports directory exists
os.makedirs(REPORTS_PATH, exist_ok=True)

# Step 1: Data Collection
def load_data(path):
    logging.info("Loading dataset...")
    df = pd.read_csv(path)
    logging.info("Dataset loaded successfully.")
    return df

# Step 2: Exploratory Data Analysis (EDA)
def perform_eda(df, title="EDA"):
    logging.info(f"--- {title} ---")
    
    # Identify and drop unique value (ID) columns
    unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    if unique_cols:
        logging.info(f"Dropping unique value columns: {unique_cols}")
        df = df.drop(columns=unique_cols)
    
    logging.info("Generating Sweetviz report...")
    report = sv.analyze(df, pairwise_analysis='off')
    report.show_html(os.path.join(REPORTS_PATH, f"{title}_report.html"), open_browser=False)
    logging.info("Sweetviz report generated successfully.")

# Step 3: Data Cleaning
def clean_data(df):
    logging.info("--- Data Cleaning ---")
    logging.info("Initial dataset shape: %s", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()
    logging.info("Shape after removing duplicates: %s", df.shape)

    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    logging.info("Categorical columns: %s", list(categorical_cols))
    logging.info("Numeric columns: %s", list(numeric_cols))

    # Drop columns with high percentage of missing values
    missing_threshold = 0.5
    df = df.loc[:, df.isnull().mean() < missing_threshold]
    logging.info("Shape after dropping columns with high missing values: %s", df.shape)

    # Encode categorical variables
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        logging.info("Shape after encoding categorical variables: %s", df.shape)

    # Advanced imputation using Iterative Imputer for numeric columns
    for col in numeric_cols:
        if col in df.columns:  # Check if the column still exists after encoding
            try:
                min_val = df[col].min(skipna=True)
                max_val = df[col].max(skipna=True)

                if not np.isfinite(min_val) or not np.isfinite(max_val):
                    raise ValueError(f"Invalid bounds for column {col}: min_val={min_val}, max_val={max_val}")

                imputer = IterativeImputer(min_value=float(min_val), max_value=float(max_val))
                df[[col]] = imputer.fit_transform(df[[col]])
            except Exception as e:
                logging.warning(f"Skipping imputation for column {col} due to error: {e}")

    # Remove outliers using percentiles for numeric columns
    for col in numeric_cols:
        if col in df.columns:  # Check if the column still exists after encoding
            lower_percentile = df[col].quantile(0.01)
            upper_percentile = df[col].quantile(0.99)
            df = df[(df[col] >= lower_percentile) & (df[col] <= upper_percentile)]
    date_cols = df.select_dtypes(include=['datetime64', 'object']).columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_Day'] = df[col].dt.day
            df[f'{col}_Month'] = df[col].dt.month
            df[f'{col}_DayOfWeek'] = df[col].dt.dayofweek
            df = df.drop(columns=[col])
        except Exception as e:
            logging.warning(f"Skipping column {col} for datetime extraction: {e}")
    # Calculate new features based on numeric columns
    if len(numeric_cols) > 1:
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                df[f'{col1}_div_{col2}'] = df[col1] / df[col2]
                df[f'{col1}_mul_{col2}'] = df[col1] * df[col2]
        df['AverageSpending'] = df['Total'] / df['Quantity']

    logging.info("Shape after feature engineering: %s", df.shape)
    return df

# Step 4: Feature Engineering


def validate_numeric_data(df):
    logging.info("Validating numeric data...")
    numeric_df = df.select_dtypes(include=["number"]).dropna()
    if numeric_df.empty:
        raise ValueError("Dataset does not contain any numeric columns.")
    return numeric_df

def feature_engineering(df):
    logging.info("--- Feature Engineering ---")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 1:
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                df[f"{col1}_div_{col2}"] = df[col2].apply(lambda x: x if x != 0 else np.nan)  # Avoid division by zero
                df[f"{col1}_mul_{col2}"] = df[col1] * df[col2]
        if "Total" in df.columns and "Quantity" in df.columns:
            df["AverageSpending"] = df["Total"].div(df["Quantity"].replace(0, np.nan))  # Avoid division by zero
    return df

# Step 5: Model Building using PyCaret
def build_model_pycaret(df):
    logging.info("--- Building PyCaret Model ---")
    # Validate dataframe contains only numeric columns
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError("No numeric columns available for clustering.")
    
    clu_setup(data=numeric_df, session_id=123, verbose=False)
    model = clu_create_model('kmeans')
    clustered_df = clu_assign_model(model)
    return model, clustered_df

# Step 6: Hyperparameter Tuning using Optuna
def tune_hyperparameters(df):
    def objective(trial):
        n_clusters = trial.suggest_int('n_clusters', 2, 10)
        model = clu_create_model('kmeans', num_clusters=n_clusters)
        clustered_df = clu_assign_model(model)
        score = silhouette_score(df, clustered_df['Cluster'])
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    best_params = study.best_params

    # Retrain the model with best parameters
    model = clu_create_model('kmeans', num_clusters=best_params['n_clusters'])
    clustered_df = clu_assign_model(model)
    return model, clustered_df, best_params

# Step 7: Model Building using FLAML
def build_model_flaml(df):
    logging.info("--- Building FLAML Model ---")

    # Ensure the input is numeric-only
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError("No numeric columns available for clustering.")

    # FLAML does not natively support unsupervised clustering; implement a custom clustering wrapper
    from sklearn.cluster import KMeans

    try:
        def objective(trial):
            n_clusters = trial.suggest_int('n_clusters', 2, 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(numeric_df)
            labels = kmeans.labels_
            score = silhouette_score(numeric_df, labels)
            return score

        # Use Optuna as the tuner for FLAML-like behavior
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        n_clusters = best_params['n_clusters']

        # Retrain the best KMeans model
        best_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        best_kmeans.fit(numeric_df)
        clustered_df = numeric_df.copy()
        clustered_df['Cluster'] = best_kmeans.labels_

        logging.info(f"FLAML (Optuna KMeans) predicted {n_clusters} clusters.")
        return best_kmeans, clustered_df

    except Exception as e:
        logging.error(f"FLAML Model failed: {e}")
        raise


# Function to visualize model performance
def visualize_model_performance(df, clustered_df, title):
    logging.info(f"Visualizing {title} Model Performance")

    # Reduce dimensionality with PCA
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    clustered_df['PCA1'] = df_pca[:, 0]
    clustered_df['PCA2'] = df_pca[:, 1]

    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=clustered_df, x='PCA1', y='PCA2', hue='Cluster')
    plt.title(f'{title} Model: Cluster Plot')
    plt.savefig(os.path.join(REPORTS_PATH, f"{title.lower()}_cluster_plot.png"))
    plt.close()

    # Save clustering results with cluster labels
    clustered_df.to_csv(os.path.join(REPORTS_PATH, f"{title.lower()}_clustered_data.csv"), index=False)

    # Calculate and log cluster evaluation metrics
    silhouette_avg = silhouette_score(df, clustered_df['Cluster'])
    calinski_harabasz = calinski_harabasz_score(df, clustered_df['Cluster'])
    davies_bouldin = davies_bouldin_score(df, clustered_df['Cluster'])

    logging.info(f"{title} Silhouette Score: {silhouette_avg}")
    logging.info(f"{title} Calinski-Harabasz Index: {calinski_harabasz}")
    logging.info(f"{title} Davies-Bouldin Index: {davies_bouldin}")

    return silhouette_avg, calinski_harabasz, davies_bouldin

# Main Function
# Main Function
def main():
    try:
        # Load dataset
        df = load_data(DATA_PATH)

        # Perform EDA before cleaning
        perform_eda(df, title="EDA Before Cleaning")

        # Clean the data
        df = clean_data(df)

        # Perform EDA after cleaning
        perform_eda(df, title="EDA After Cleaning")

        # Feature Engineering
        df = feature_engineering(df)

        # Validate numeric data
        numeric_df = validate_numeric_data(df)

        # Build the clustering model using PyCaret
        model_pycaret, clustered_df_pycaret = build_model_pycaret(numeric_df)

        # Hyperparameter Tuning using Optuna
        model_optuna, clustered_df_optuna, best_params = tune_hyperparameters(numeric_df)

        # Build the clustering model using FLAML
        model_flaml, clustered_df_flaml = build_model_flaml(numeric_df)

        # Visualize the performance of the clustering models
        pycaret_scores = visualize_model_performance(numeric_df, clustered_df_pycaret, "PyCaret")
        flaml_scores = visualize_model_performance(numeric_df, clustered_df_flaml, "FLAML")

        # Compare the models
        logging.info("Comparing PyCaret and FLAML models...")
        if pycaret_scores[0] > flaml_scores[0]:
            best_model = "PyCaret"
        else:
            best_model = "FLAML"

        # Log and print the results
        logging.info("Clustering Model built successfully.")
        logging.info(f"Best Hyperparameters (PyCaret): {best_params}")
        logging.info(f"Best Model: {best_model}")
        logging.info(f"PyCaret Scores: Silhouette={pycaret_scores[0]}, Calinski-Harabasz={pycaret_scores[1]}, Davies-Bouldin={pycaret_scores[2]}")
        logging.info(f"FLAML Scores: Silhouette={flaml_scores[0]}, Calinski-Harabasz={flaml_scores[1]}, Davies-Bouldin={flaml_scores[2]}")

        print("\nClustering Model built successfully.")
        print(f"Best Hyperparameters (PyCaret): {best_params}")
        print(f"Best Model: {best_model}")
        print(f"PyCaret Scores: Silhouette={pycaret_scores[0]}, Calinski-Harabasz={pycaret_scores[1]}, Davies-Bouldin={pycaret_scores[2]}")
        print(f"FLAML Scores: Silhouette={flaml_scores[0]}, Calinski-Harabasz={flaml_scores[1]}, Davies-Bouldin={flaml_scores[2]}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    main()