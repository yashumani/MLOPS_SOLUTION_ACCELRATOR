import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Define paths
DATA_PATH = "data/BostonHousing.csv"  # Update path if necessary
TARGET_COLUMN = "MEDV"  # Target column for prediction

# Load dataset
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print("File not found. Please check the filepath.")
        return None

# Preprocess and feature engineering
def preprocess_data(df, target_column, degree=2):
    # Select numerical columns
    num_df = df.select_dtypes(include=[np.number])
    
    # Handle missing values
    num_df = num_df.dropna()
    
    # Extract features and target
    X = num_df.drop(columns=[target_column])
    y = num_df[target_column]
    
    # Add polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    print(f"Processed data: Features shape: {X_scaled.shape}, Target shape: {y.shape}")
    return X_scaled, y

# Train and evaluate model
def train_and_evaluate(X, y, model_type="linear", alpha=1.0):
    # Split data
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
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Performance ---")
    print(f"Model Type: {model_type.capitalize()}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=20)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()
    
    return model

# Detailed analysis with statsmodels
def statsmodels_analysis(X, y):
    # Add constant for intercept
    X = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X).fit()
    print("\n--- Statsmodels Summary ---")
    print(model.summary())

# Prepare data for Linear Regression
def prepare_linear_regression_data(df, target_column):
    """Prepare data for Linear Regression."""
    print("\n--- Preparing Data for Linear Regression ---")
    
    # Select numerical columns only
    num_df = df.select_dtypes(include=[np.number])
    print(f"Numerical columns: {num_df.columns.tolist()}")
    
    # Check if target column exists
    if target_column not in num_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    
    # Drop rows with missing values
    num_df = num_df.dropna()
    print(f"Dataset after dropping missing values: {num_df.shape}")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
    
    # Split features and target
    X = num_df.drop(columns=[target_column])
    y = num_df[target_column]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y

# Train and evaluate Linear Regression model
def perform_linear_regression(X, y):
    """Perform Linear Regression and evaluate the model."""
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Linear Regression Results ---")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Visualize predictions vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
    plt.title("Predictions vs Actual Values")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()

# Main workflow
if __name__ == "__main__":
    print("Processing Boston Housing Dataset...")
    df = load_data(DATA_PATH)
    
    if df is not None:
        # Prepare data and perform Linear Regression
        try:
            X, y = preprocess_data(df, TARGET_COLUMN, degree=2)
            
            # Train and evaluate multiple models
            linear_model = train_and_evaluate(X, y, model_type="linear")
            ridge_model = train_and_evaluate(X, y, model_type="ridge", alpha=0.5)
            lasso_model = train_and_evaluate(X, y, model_type="lasso", alpha=0.1)
            
            # Detailed analysis
            statsmodels_analysis(X, y)
        except Exception as e:
            print(f"Error during processing: {e}")
