import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.clustering import setup as clu_setup, create_model as clu_create_model, assign_model as clu_assign_model
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import IterativeImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import numpy as np
import logging
import os
import sweetviz as sv
import optuna

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
    unique_cols = df.columns[df.nunique() == len(df)]
    if not unique_cols.empty:
        logging.info(f"Dropping unique value columns: {unique_cols.tolist()}")
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
    
    # Identify and drop unique value columns
    unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    if unique_cols:
        logging.info(f"Dropping unique value columns: {unique_cols}")
        df = df.drop(columns=unique_cols)
    logging.info("Shape after dropping unique value columns: %s", df.shape)
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    logging.info("Categorical columns: %s", list(categorical_cols))
    logging.info("Numeric columns: %s", list(numeric_cols))
    
    # Drop columns with high percentage of missing values
    df = df.loc[:, df.isnull().mean() < 0.5]
    logging.info("Shape after dropping columns with high missing values: %s", df.shape)
    
    # Encode categorical variables
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        logging.info("Shape after encoding categorical variables: %s", df.shape)
    
    # Impute missing values for numeric columns
    if len(numeric_cols) > 0:
        imputer = IterativeImputer()
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        logging.info("Shape after imputing missing values: %s", df.shape)
    
    # Remove outliers using percentiles for numeric columns
    for col in numeric_cols:
        if col in df.columns:
            lower_percentile = df[col].quantile(0.01)
            upper_percentile = df[col].quantile(0.99)
            df[col] = np.clip(df[col], lower_percentile, upper_percentile)
    logging.info("Shape after removing outliers: %s", df.shape)
    
    # Extract date features
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
    
    logging.info("Final dataset shape after cleaning: %s", df.shape)
    return df

# Step 4: Feature Engineering
def feature_engineering(df):
    logging.info("--- Feature Engineering ---")
    numeric_cols = df.select_dtypes(include=["number"]).columns
    
    if len(numeric_cols) > 1:
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        numeric_data = df[numeric_cols]
        poly_features = poly.fit_transform(numeric_data)
        
        # Create a DataFrame with the new polynomial features
        poly_feature_names = poly.get_feature_names_out(numeric_cols)
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        
        # Drop original numeric columns and concatenate the new polynomial features
        df = df.drop(columns=numeric_cols)
        df = pd.concat([df, poly_df], axis=1)
    
    return df

# Step 5: Model Building using PyCaret
def build_model_pycaret(df):
    logging.info("--- Building PyCaret Model ---")
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
    model = clu_create_model('kmeans', num_clusters=best_params['n_clusters'])
    clustered_df = clu_assign_model(model)
    return model, clustered_df, best_params

# Step 7: Model Building using FLAML
def build_model_flaml(df):
    logging.info("--- Building FLAML Model ---")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        raise ValueError("No numeric columns available for clustering.")
    def objective(trial):
        n_clusters = trial.suggest_int('n_clusters', 2, 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(numeric_df)
        labels = kmeans.labels_
        score = silhouette_score(numeric_df, labels)
        return score
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    n_clusters = best_params['n_clusters']
    best_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    best_kmeans.fit(numeric_df)
    clustered_df = numeric_df.copy()
    clustered_df['Cluster'] = best_kmeans.labels_
    logging.info(f"FLAML (Optuna KMeans) predicted {n_clusters} clusters.")
    return best_kmeans, clustered_df

# Function to visualize model performance
def visualize_model_performance(df, clustered_df, title):
    logging.info(f"Visualizing {title} Model Performance")
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    clustered_df['PCA1'] = df_pca[:, 0]
    clustered_df['PCA2'] = df_pca[:, 1]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=clustered_df, x='PCA1', y='PCA2', hue='Cluster')
    plt.title(f'{title} Model: Cluster Plot')
    plt.savefig(os.path.join(REPORTS_PATH, f"{title.lower()}_cluster_plot.png"))
    plt.close()
    clustered_df.to_csv(os.path.join(REPORTS_PATH, f"{title.lower()}_clustered_data.csv"), index=False)
    silhouette_avg = silhouette_score(df, clustered_df['Cluster'])
    calinski_harabasz = calinski_harabasz_score(df, clustered_df['Cluster'])
    davies_bouldin = davies_bouldin_score(df, clustered_df['Cluster'])
    logging.info(f"{title} Silhouette Score: {silhouette_avg}")
    logging.info(f"{title} Calinski-Harabasz Index: {calinski_harabasz}")
    logging.info(f"{title} Davies-Bouldin Index: {davies_bouldin}")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', data=clustered_df)
    plt.title(f'{title} Model: Cluster Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.savefig(os.path.join(REPORTS_PATH, f"{title.lower()}_cluster_distribution.png"))
    plt.close()
    cluster_centers = clustered_df.groupby('Cluster').mean()
    cluster_centers.to_csv(os.path.join(REPORTS_PATH, f"{title.lower()}_cluster_centers.csv"))
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_centers, annot=True, cmap='coolwarm')
    plt.title(f'{title} Model: Cluster Centers')
    plt.savefig(os.path.join(REPORTS_PATH, f"{title.lower()}_cluster_centers.png"))
    plt.close()
    profile_report = sv.analyze(clustered_df, pairwise_analysis='off')
    profile_report.show_html(os.path.join(REPORTS_PATH, f"{title.lower()}_cluster_profiles.html"), open_browser=False)
    return silhouette_avg, calinski_harabasz, davies_bouldin

# Main Function
def main():
    try:
        print("Starting the clustering process...")
        logging.info("Starting the clustering process...")
        
        # Step 1: Load Data
        print("Step 1: Loading the dataset...")
        df = load_data(DATA_PATH)
        print("Dataset loaded successfully.")
        
        # Step 2: Perform EDA before cleaning
        print("Step 2: Performing EDA before cleaning...")
        perform_eda(df, title="EDA Before Cleaning")
        print("EDA report generated before cleaning.")
        
        # Step 3: Clean Data
        print("Step 3: Cleaning the dataset...")
        df = clean_data(df)
        print("Dataset cleaned successfully.")
        
        # Step 4: Perform EDA after cleaning
        print("Step 4: Performing EDA after cleaning...")
        perform_eda(df, title="EDA After Cleaning")
        print("EDA report generated after cleaning.")
        
        # Step 5: Feature Engineering
        print("Step 5: Performing feature engineering...")
        df = feature_engineering(df)
        print("Feature engineering completed.")
        
        # Step 6: Model Building using PyCaret
        print("Step 6: Building clustering model using PyCaret...")
        numeric_df = df.select_dtypes(include=["number"]).dropna()
        if numeric_df.empty:
            raise ValueError("Dataset does not contain any numeric columns.")
        model_pycaret, clustered_df_pycaret = build_model_pycaret(numeric_df)
        print("PyCaret model built successfully.")
        
        # Step 7: Hyperparameter Tuning using Optuna
        print("Step 7: Tuning hyperparameters using Optuna...")
        model_optuna, clustered_df_optuna, best_params = tune_hyperparameters(numeric_df)
        print(f"Optuna hyperparameter tuning completed. Best parameters: {best_params}")
        
        # Step 8: Model Building using FLAML
        print("Step 8: Building clustering model using FLAML...")
        model_flaml, clustered_df_flaml = build_model_flaml(numeric_df)
        print("FLAML model built successfully.")
        
        # Step 9: Visualize Model Performance
        print("Step 9: Visualizing model performance for PyCaret...")
        pycaret_scores = visualize_model_performance(numeric_df, clustered_df_pycaret, "PyCaret")
        print("PyCaret model performance visualized.")
        
        print("Step 9: Visualizing model performance for FLAML...")
        flaml_scores = visualize_model_performance(numeric_df, clustered_df_flaml, "FLAML")
        print("FLAML model performance visualized.")
        
        # Compare Models
        print("Comparing PyCaret and FLAML models...")
        logging.info("Comparing PyCaret and FLAML models...")
        best_model = "PyCaret" if pycaret_scores[0] > flaml_scores[0] else "FLAML"
        logging.info("Clustering Model built successfully.")
        logging.info(f"Best Hyperparameters (PyCaret): {best_params}")
        logging.info(f"Best Model: {best_model}")
        logging.info(f"PyCaret Scores: Silhouette={pycaret_scores[0]}, Calinski-Harabasz={pycaret_scores[1]}, Davies-Bouldin={pycaret_scores[2]}")
        logging.info(f"FLAML Scores: Silhouette={flaml_scores[0]}, Calinski-Harabasz={flaml_scores[1]}, Davies-Bouldin={flaml_scores[2]}")
        
        # Print final results
        print("\nClustering Model built successfully.")
        print(f"Best Hyperparameters (PyCaret): {best_params}")
        print(f"Best Model: {best_model}")
        print(f"PyCaret Scores: Silhouette={pycaret_scores[0]}, Calinski-Harabasz={pycaret_scores[1]}, Davies-Bouldin={pycaret_scores[2]}")
        print(f"FLAML Scores: Silhouette={flaml_scores[0]}, Calinski-Harabasz={flaml_scores[1]}, Davies-Bouldin={flaml_scores[2]}")
        
        # Recommendations
        print("\nRecommendations:")
        if best_model == "PyCaret":
            print("The PyCaret model performed better based on the Silhouette Score. It is recommended to use the PyCaret model for clustering.")
        else:
            print("The FLAML model performed better based on the Silhouette Score. It is recommended to use the FLAML model for clustering.")
        print("Further tuning and validation can be performed to ensure the robustness of the selected model.")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

