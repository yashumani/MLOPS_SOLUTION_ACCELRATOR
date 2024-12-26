import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import jarque_bera
from pandas_profiling import ProfileReport

# Define paths
DATA_PATH = "c:/Users/yashu/Desktop/SAVYMINDS/ML Ops/YS_MVP/data/BostonHousing.csv"
REPORTS_PATH = "c:/Users/yashu/Desktop/SAVYMINDS/ML Ops/YS_MVP/Reports"
TARGET_COLUMN = "medv"  # Target column for prediction

# Create Reports directory if it doesn't exist
os.makedirs(REPORTS_PATH, exist_ok=True)

# Load dataset
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print("File not found. Please check the filepath.")
        return None

# Data Cleaning
def clean_data(df):
    """Clean and preprocess the dataset."""
    print("\n--- Data Cleaning ---")
    print(f"Initial dataset shape: {df.shape}")
    
    # Handle missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Missing values detected: {missing_values}. Removing rows with missing values.")
        df = df.dropna()
    
    # Outlier removal using Z-scores
    from scipy.stats import zscore
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[(np.abs(zscore(df[numeric_cols])) < 3).all(axis=1)]
    print(f"After removing outliers: {df.shape}")
    
    return df

# EDA
def perform_eda(df):
    """Perform exploratory data analysis."""
    print("\n--- Exploratory Data Analysis ---")
    
    # Descriptive statistics
    print("\n### Dataset Summary ###")
    print(df.describe())
    
    # Correlation heatmap
    print("\n### Correlation Heatmap ###")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.savefig(os.path.join(REPORTS_PATH, "Correlation_Heatmap.png"))
    plt.show()
    
    # Target variable distribution
    print("\n### Target Variable Distribution ###")
    plt.figure(figsize=(8, 6))
    sns.histplot(df[TARGET_COLUMN], kde=True, bins=30)
    plt.title(f"Distribution of {TARGET_COLUMN}")
    plt.xlabel(TARGET_COLUMN)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(REPORTS_PATH, "Target_Variable_Distribution.png"))
    plt.show()
    
    # Pandas profiling
    print("\n### Generating Profiling Report ###")
    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    profile.to_file(os.path.join(REPORTS_PATH, "Linear_Regression_EDA_Report.html"))

# Preprocessing
def preprocess_data(df, target_column, degree=2, n_components=0.95):
    """Preprocess the dataset."""
    # Select numeric features
    num_df = df.select_dtypes(include=[np.number])
    X = num_df.drop(columns=[target_column])
    y = num_df[target_column]
    
    # Polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Variance Inflation Factor (VIF)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = [f"x{i}" for i in range(X_poly.shape[1])]
    vif_data["VIF"] = [variance_inflation_factor(X_poly, i) for i in range(X_poly.shape[1])]
    print("\n--- VIF Analysis ---")
    print(vif_data[vif_data["VIF"] > 10])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    return X_scaled, y

# Train and Evaluate Model
def train_and_evaluate(X, y, model_type="linear", alpha=1.0):
    """Train and evaluate the model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Select model
    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "lasso":
        model = Lasso(alpha=alpha)
    else:
        model = LinearRegression()
    
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Model Performance ---")
    print(f"Model Type: {model_type.capitalize()}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    
    return model, mse, r2

# Narrative Generator
def generate_narrative(mse, r2, cross_val_scores, stats_summary):
    """Generate narratives based on model performance."""
    print("\n--- Key Observations ---")
    print(f"Model Performance:")
    print(f"- The R-squared value on the test set is {r2:.4f}, indicating that the model explains a moderate-to-high proportion of the variance.")
    print(f"- The Mean Squared Error (MSE) is {mse:.4f}, indicating the average squared difference between actual and predicted values.")
    
    print("\nRecommendations for Improvement:")
    if r2 < 0.7:
        print("- Consider adding more features or improving feature selection.")
    print("- Use Ridge or Lasso regression to handle multicollinearity.")
    print("- Perform PCA to reduce dimensionality and improve generalization.")

# Main workflow
def main():
    print("Boston Housing Price Prediction")
    
    # Load dataset
    df = load_data(DATA_PATH)
    if df is not None:
        # Clean data
        df = clean_data(df)
        
        # Perform EDA
        perform_eda(df)
        
        # Preprocess data
        X, y = preprocess_data(df, TARGET_COLUMN)
        
        # Train and evaluate model
        model, mse, r2 = train_and_evaluate(X, y, model_type="ridge", alpha=1.0)
        
        # Generate narrative
        generate_narrative(mse, r2, None, None)

if __name__ == "__main__":
    main()
