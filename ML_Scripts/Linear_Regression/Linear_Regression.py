import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/BostonHousing.csv"
TARGET_COLUMN = "medv"
DEGREE = 2
ALPHA = 1.0

# Create Reports directory if it doesn't exist
REPORTS_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Linear_Regression/Reports"
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
    print("\n--- Data Cleaning ---")
    print(f"Initial dataset shape: {df.shape}")
    
    # Handle missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Missing values detected: {missing_values}. Removing rows with missing values.")
        df = df.dropna()
    
    # Outlier removal using percentiles
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(f"After removing outliers: {df.shape}")
    
    return df

# EDA
def perform_eda(df, target_column, pdf_pages):
    print("\n--- Exploratory Data Analysis ---")
    
    # Descriptive statistics
    print("\n### Dataset Summary ###")
    summary = df.describe()
    print(summary)
    summary.to_csv(os.path.join(REPORTS_PATH, "Dataset_Summary.csv"))
    
    # Correlation heatmap
    print("\n### Correlation Heatmap ###")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.savefig(os.path.join(REPORTS_PATH, "Correlation_Heatmap.png"))
    pdf_pages.savefig()
    plt.close()
    
    # Target variable distribution
    print("\n### Target Variable Distribution ###")
    plt.figure(figsize=(8, 6))
    sns.histplot(df[target_column], kde=True, bins=30)
    plt.title(f"Distribution of {target_column}")
    plt.xlabel(target_column)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(REPORTS_PATH, "Target_Variable_Distribution.png"))
    pdf_pages.savefig()
    plt.close()

# Preprocessing
def preprocess_data(df, target_column, degree=2):
    num_df = df.select_dtypes(include=[np.number])
    X = num_df.drop(columns=[target_column])
    y = num_df[target_column]
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = [f"x{i}" for i in range(X_poly.shape[1])]
    vif_data["VIF"] = [variance_inflation_factor(X_poly, i) if variance_inflation_factor(X_poly, i) < np.inf else np.nan for i in range(X_poly.shape[1])]
    print("\n--- VIF Analysis ---")
    print(vif_data[vif_data["VIF"] > 10])
    vif_data.to_csv(os.path.join(REPORTS_PATH, "VIF_Analysis.csv"))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    return X_scaled, y

# Plot actual vs. predicted values
def plot_actual_vs_predicted(y_test, y_pred, model_type, pdf_pages):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color="green", label="Actual response, $y_i$", zorder=3)
    plt.scatter(range(len(y_pred)), y_pred, color="red", label="Predicted response, $f(x_i)$", zorder=3)
    plt.plot(range(len(y_pred)), y_pred, color="black", label="Estimated regression line, $f(x) = b_0 + b_1x$")
    for i in range(len(y_test)):
        plt.vlines(x=i, ymin=y_test.iloc[i], ymax=y_pred[i], colors="gray", linestyles="--", lw=1, zorder=2)
    plt.title(f"Actual vs Predicted Values ({model_type.capitalize()})")
    plt.xlabel("Sample Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(REPORTS_PATH, f"{model_type}_Actual_vs_Predicted_Improved.png"))
    pdf_pages.savefig()
    plt.close()

# Plot training vs validation error
def plot_training_vs_validation_error(train_errors, val_errors, model_type, pdf_pages):
    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, label='Training Error', marker='o')
    plt.plot(val_errors, label='Validation Error', marker='o')
    plt.title(f'Training vs Validation Error ({model_type.capitalize()})')
    plt.xlabel('Fold')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(REPORTS_PATH, f"{model_type}_Training_vs_Validation_Error.png"))
    pdf_pages.savefig()
    plt.close()

# Train and Evaluate Model
def train_and_evaluate(X, y, model_type="linear", alpha=1.0, pdf_pages=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if model_type == "ridge":
        model = Ridge(alpha=alpha)
    elif model_type == "lasso":
        model = Lasso(alpha=alpha)
    elif model_type == "elasticnet":
        model = ElasticNet(alpha=alpha)
    else:
        model = LinearRegression()
    
    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)) or np.any(np.isinf(X_train)) or np.any(np.isinf(y_train)):
        raise ValueError("Training data contains NaNs or infinite values.")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))
    
    print(f"\n--- Model Performance ---")
    print(f"Model Type: {model_type.capitalize()}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")
    print(f"Cross-validation R-squared scores: {cv_scores}")
    
    plot_actual_vs_predicted(y_test, y_pred, model_type, pdf_pages)
    
    train_errors = []
    val_errors = []
    y_train_np = y_train.to_numpy()
    for train_index, val_index in KFold(n_splits=5, shuffle=True, random_state=42).split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train_np[train_index], y_train_np[val_index]
        model.fit(X_train_fold, y_train_fold)
        train_errors.append(mean_squared_error(y_train_fold, model.predict(X_train_fold)))
        val_errors.append(mean_squared_error(y_val_fold, model.predict(X_val_fold)))
    
    plot_training_vs_validation_error(train_errors, val_errors, model_type, pdf_pages)
    
    return model, mse, r2, cv_scores

# Narrative Generator
def generate_narrative(mse, r2, cross_val_scores):
    print("\n--- Key Observations ---")
    print(f"Model Performance:")
    print(f"- The R-squared value on the test set is {r2:.4f}, indicating that the model explains a {'high' if r2 > 0.7 else 'moderate' if r2 > 0.5 else 'low'} proportion of the variance.")
    print(f"- The Mean Squared Error (MSE) is {mse:.4f}, indicating the average squared difference between actual and predicted values.")
    print(f"- Cross-validation R-squared scores: {cross_val_scores}")
    
    print("\nRecommendations for Improvement:")
    if r2 < 0.5:
        print("- The model performance is relatively low. Consider adding more features, improving feature selection, or using more complex models.")
    elif r2 < 0.7:
        print("- The model performance is moderate. Consider fine-tuning the model parameters, adding interaction terms, or using ensemble methods.")
    else:
        print("- The model performance is good. Ensure that the model is not overfitting by validating on a separate test set.")
    
    print("- Use Ridge, Lasso, or Elastic Net regression to handle multicollinearity if VIF values are high.")
    print("- Perform PCA to reduce dimensionality and improve generalization if the number of features is large.")
    print("- Regularly check for data quality issues such as missing values or outliers.")
    print("- Consider using cross-validation to ensure the model's robustness and generalizability.")
    
    print("\nUser Interaction Tips:")
    print("- Always visualize the results to understand the model's behavior better.")
    print("- Communicate the model's performance metrics clearly to stakeholders.")
    print("- Be prepared to iterate on the model based on feedback and new data.")
    print("- Document the entire modeling process for reproducibility and future reference.")

# Main workflow
def main():
    print("Automated Machine Learning Workflow")
    
    df = load_data(DATA_PATH)
    if df is not None:
        df = clean_data(df)
        
        with PdfPages(os.path.join(REPORTS_PATH, "Exploratory_Data_Analysis.pdf")) as pdf_pages:
            perform_eda(df, TARGET_COLUMN, pdf_pages)
            X, y = preprocess_data(df, TARGET_COLUMN, degree=DEGREE)
            
            results = {}
            models = ["linear", "ridge", "lasso", "elasticnet"]
            for model_type in models:
                model, mse, r2, cv_scores = train_and_evaluate(X, y, model_type=model_type, alpha=ALPHA, pdf_pages=pdf_pages)
                results[model_type] = {"model": model, "mse": mse, "r2": r2, "cross_val_scores": cv_scores}
            
            print("\n--- Model Comparison ---")
            for model_type, metrics in results.items():
                print(f"Model: {model_type.capitalize()}")
                print(f"Mean Squared Error: {metrics['mse']:.4f}")
                print(f"R-squared: {metrics['r2']:.4f}")
                print("-" * 30)
            
            best_model_type = input("Enter the model type to use for prediction (linear, ridge, lasso, elasticnet): ")
            best_model = results[best_model_type]["model"]
            print(f"Selected Model: {best_model_type.capitalize()}")
            
            generate_narrative(results[best_model_type]["mse"], results[best_model_type]["r2"], results[best_model_type]["cross_val_scores"])
            
if __name__ == "__main__":
    main()

def main():
    print("Automated Machine Learning Workflow")
    
    df = load_data(DATA_PATH)
    if df is not None:
        df = clean_data(df)
        
        with PdfPages(os.path.join(REPORTS_PATH, "Exploratory_Data_Analysis.pdf")) as pdf_pages:
            perform_eda(df, TARGET_COLUMN, pdf_pages)
            X, y = preprocess_data(df, TARGET_COLUMN, degree=DEGREE)
            
            results = {}
            models = ["linear", "ridge", "lasso", "elasticnet"]
            for model_type in models:
                model, mse, r2, cv_scores = train_and_evaluate(X, y, model_type=model_type, alpha=ALPHA, pdf_pages=pdf_pages)
                results[model_type] = {"model": model, "mse": mse, "r2": r2, "cross_val_scores": cv_scores}
            
            print("\n--- Model Comparison ---")
            for model_type, metrics in results.items():
                print(f"Model: {model_type.capitalize()}")
                print(f"Mean Squared Error: {metrics['mse']:.4f}")
                print(f"R-squared: {metrics['r2']:.4f}")
                print("-" * 30)
            
            best_model_type = input("Enter the model type to use for prediction (linear, ridge, lasso, elasticnet): ")
            best_model = results[best_model_type]["model"]
            print(f"Selected Model: {best_model_type.capitalize()}")
            
            generate_narrative(results[best_model_type]["mse"], results[best_model_type]["r2"], results[best_model_type]["cross_val_scores"])
            
            predictions_df = pd.DataFrame({"Actual": y, "Predicted": best_model.predict(X)})
            predictions_df.to_csv(os.path.join(REPORTS_PATH, f"{best_model_type}_Predictions.csv"), index=False)
            print(f"Predictions saved to {os.path.join(REPORTS_PATH, f'{best_model_type}_Predictions.csv')}")

if __name__ == "__main__":
    main()
