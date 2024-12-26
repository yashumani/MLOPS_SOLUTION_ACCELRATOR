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
from pandas_profiling import ProfileReport

# Define paths
DATA_PATH = "c:/Users/yashu/Desktop/SAVYMINDS/ML Ops/YS_MVP/data/BostonHousing.csv"  # Update path if necessary
TARGET_COLUMN = "medv"  # Target column for prediction

# Load dataset
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
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
    
    print(f"\n--- Model Performance ---")
    print(f"Model Type: {model_type.capitalize()}")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Narrative on model performance
    if r2 > 0.8:
        print("The model explains a high proportion of the variance in the target variable. This indicates a strong predictive power.")
    elif r2 > 0.5:
        print("The model explains a moderate proportion of the variance in the target variable. There is room for improvement.")
    else:
        print("The model explains a low proportion of the variance in the target variable. Consider improving the model or using different features.")
    
    if mse < 10:
        print("The Mean Squared Error is low, indicating that the model's predictions are close to the actual values.")
    else:
        print("The Mean Squared Error is high, indicating that the model's predictions are not very accurate.")
    
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
    
    # Narrative on statsmodels analysis
    print("\n--- Detailed Analysis ---")
    print("The statsmodels summary provides detailed information about the regression model.")
    print("Key metrics to consider:")
    print("1. R-squared: Indicates the proportion of variance explained by the model.")
    print("2. P-values: Indicates the significance of each feature. Features with p-values less than 0.05 are considered significant.")
    print("3. Coefficients: Indicates the impact of each feature on the target variable.")
    print("4. Standard Errors: Indicates the variability of the coefficient estimates.")

# Main workflow
def main():
    print("Boston Housing Price Prediction")
    
    # Load dataset
    df = load_data(DATA_PATH)
    if df is not None:
        print(df.head())
        
        # EDA
        print("### Dataset Information")
        print(df.describe())
        
        print("### Correlation Heatmap")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        plt.show()
        
        print("### Pair Plot")
        sns.pairplot(df)
        plt.show()
        
        print("### Target Variable Distribution")
        plt.figure(figsize=(8, 6))
        sns.histplot(df[TARGET_COLUMN], kde=True, bins=30)
        plt.title(f"Distribution of {TARGET_COLUMN}")
        plt.xlabel(TARGET_COLUMN)
        plt.ylabel("Frequency")
        plt.show()
        
        # Pandas Profiling Report
        print("### Pandas Profiling Report")
        profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        profile.to_file("pandas_profiling_report.html")
    
        # Preprocess data
        degree = 2
        X, y = preprocess_data(df, TARGET_COLUMN, degree)
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
        # Train and evaluate model
        model_type = "linear"
        alpha = 1.0
        model = train_and_evaluate(X, y, model_type, alpha)
        print(f"Trained {model_type.capitalize()} model.")
    
        # Detailed analysis with statsmodels
        statsmodels_analysis(X, y)

if __name__ == "__main__":
    main()
