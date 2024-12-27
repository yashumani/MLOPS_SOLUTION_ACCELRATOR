import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import setup, compare_models, pull, create_model, tune_model, predict_model
from flaml import AutoML
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import logging
from fpdf import FPDF
import os

# Use Agg backend for matplotlib
plt.switch_backend('Agg')

# Configure logging
logging.basicConfig(filename='logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/BostonHousing.csv"
TARGET_COLUMN = "medv"  # Replace with the target column name in your dataset

# PDF class
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Linear Regression Analysis Report', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_image(self, image_path, x=None, y=None, w=0, h=0):
        self.image(image_path, x, y, w, h)
        self.ln(10)

# Function to save plots
def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)

# Function to generate PDF report
def generate_pdf_report():
    pdf = PDF()
    pdf.add_page()

    # Add EDA section
    pdf.chapter_title("Exploratory Data Analysis (EDA)")
    pdf.chapter_body("This section includes the initial exploration of the dataset, including visualizations and statistical summaries.")
    pdf.add_image('eda_histograms.png', w=180)
    pdf.add_image('eda_heatmap.png', w=180)

    # Add Data Cleaning section
    pdf.chapter_title("Data Cleaning")
    pdf.chapter_body("This section includes the steps taken to clean the dataset, such as removing duplicates and handling missing values.")

    # Add PyCaret results
    pdf.chapter_title("PyCaret Linear Regression")
    pdf.chapter_body("This section includes the results of the PyCaret linear regression model, including the best model and its performance metrics.")
    pdf.add_image('pycaret_true_vs_pred.png', w=180)
    pdf.add_image('pycaret_residuals.png', w=180)

    # Add FLAML results
    pdf.chapter_title("FLAML Linear Regression")
    pdf.chapter_body("This section includes the results of the FLAML linear regression model, including the best model and its performance metrics.")
    pdf.add_image('flaml_true_vs_pred.png', w=180)
    pdf.add_image('flaml_residuals.png', w=180)

    # Save PDF
    pdf.output('Linear_Regression_Report.pdf')

    # Open the PDF file
    os.startfile('Linear_Regression_Report.pdf')

# Step 1: Exploratory Data Analysis (EDA)
def perform_eda(df):
    logging.info("--- Exploratory Data Analysis (EDA) ---")
    logging.info("Dataset Head:\n%s", df.head())
    logging.info("Dataset Info:\n%s", df.info())
    logging.info("Dataset Description:\n%s", df.describe())
    
    # Visualize distributions of features
    fig, ax = plt.subplots(figsize=(20, 15))
    df.hist(bins=30, figsize=(20, 15), layout=(5, 3), ax=ax)
    plt.tight_layout()
    save_plot(fig, 'eda_histograms.png')
    
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation Heatmap')
    save_plot(fig, 'eda_heatmap.png')

# Step 2: Data Cleaning
def clean_data(df):
    logging.info("--- Data Cleaning ---")
    logging.info("Initial dataset shape: %s", df.shape)
    # Remove duplicates
    df = df.drop_duplicates()
    logging.info("Shape after removing duplicates: %s", df.shape)
    # Fill missing values with median
    df = df.fillna(df.median())
    logging.info("Shape after filling missing values: %s", df.shape)
    
    # Remove outliers using percentiles
    for col in df.columns:
        lower_percentile = df[col].quantile(0.01)
        upper_percentile = df[col].quantile(0.99)
        df = df[(df[col] >= lower_percentile) & (df[col] <= upper_percentile)]
        logging.info("Shape after removing outliers in %s: %s", col, df.shape)
    
    logging.info("Dataset shape after cleaning: %s", df.shape)
    return df

# Step 3: PyCaret Linear Regression with Hyperparameter Tuning
def pycaret_linear_regression(train_df, test_df):
    logging.info("--- PyCaret Linear Regression ---")
    logging.info("Setting up PyCaret...")
    # Setup PyCaret
    setup(data=train_df, target=TARGET_COLUMN, session_id=123, verbose=False)
    logging.info("Comparing models to find the best one...")
    # Compare models to find the best one
    best_model = compare_models()
    logging.info("Best Model from PyCaret (Pre-Tuning):\n%s", best_model)
    
    logging.info("Tuning the best model...")
    # Tune the best model
    tuned_model = tune_model(best_model, optimize="R2")
    logging.info("Tuned Model from PyCaret:\n%s", tuned_model)
    
    logging.info("Predicting on test set...")
    # Predict on test set
    predictions = predict_model(tuned_model, data=test_df)
    r2 = r2_score(test_df[TARGET_COLUMN], predictions['prediction_label'])
    rmse = np.sqrt(mean_squared_error(test_df[TARGET_COLUMN], predictions['prediction_label']))
    
    logging.info("PyCaret R²: %s", r2)
    logging.info("PyCaret RMSE: %s", rmse)
    
    return tuned_model, r2, rmse

# Step 4: FLAML Linear Regression with Hyperparameter Tuning
def flaml_linear_regression(train_df, test_df):
    logging.info("--- FLAML Linear Regression ---")
    # Split data into features and target
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    
    logging.info("Initializing AutoML...")
    # Initialize AutoML
    automl = AutoML()
    
    logging.info("Fitting the AutoML model...")
    # Fit the AutoML model
    automl.fit(X_train=X_train, y_train=y_train, task="regression", time_budget=300, use_ray=False)
    logging.info("Best Model from FLAML (Pre-Tuning):\n%s", automl.best_estimator)
    logging.info("Best Config: %s", automl.best_config)
    logging.info("Best Loss: %s", automl.best_loss)
    
    logging.info("Predicting on test set...")
    # Predict on test set
    predictions = automl.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    logging.info("FLAML R²: %s", r2)
    logging.info("FLAML RMSE: %s", rmse)
    
    return automl, r2, rmse

# Function to compare the best models from PyCaret and FLAML
def compare_best_models(pycaret_model, flaml_model, test_df):
    logging.info("--- Final Model Comparison ---")
    # Predict on test set using PyCaret model
    pycaret_predictions = predict_model(pycaret_model, data=test_df)
    pycaret_r2 = r2_score(test_df[TARGET_COLUMN], pycaret_predictions['prediction_label'])
    pycaret_rmse = np.sqrt(mean_squared_error(test_df[TARGET_COLUMN], pycaret_predictions['prediction_label']))
    
    # Predict on test set using FLAML model
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    flaml_predictions = flaml_model.predict(X_test)
    flaml_r2 = r2_score(y_test, flaml_predictions)
    flaml_rmse = np.sqrt(mean_squared_error(y_test, flaml_predictions))
    
    logging.info("PyCaret Model: %s", pycaret_model)
    logging.info("PyCaret R²: %s, RMSE: %s", pycaret_r2, pycaret_rmse)
    logging.info("FLAML Model: %s", flaml_model.best_estimator)
    logging.info("FLAML R²: %s, RMSE: %s", flaml_r2, flaml_rmse)
    
    # Determine the better model
    if pycaret_r2 > flaml_r2:
        logging.info("Winner: PyCaret")
        return pycaret_model
    else:
        logging.info("Winner: FLAML")
        return flaml_model

# Function to visualize model performance
def visualize_model_performance(model, test_df, model_name):
    logging.info("Visualizing %s Model Performance", model_name)
    if model_name == "PyCaret":
        predictions = predict_model(model, data=test_df)
        y_true = test_df[TARGET_COLUMN]
        y_pred = predictions['prediction_label']
    else:
        X_test = test_df.drop(columns=[TARGET_COLUMN])
        y_true = test_df[TARGET_COLUMN]
        y_pred = model.predict(X_test)
    
    # Scatter plot of true vs predicted values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'{model_name} Model: True vs Predicted Values')
    save_plot(fig, f'{model_name.lower()}_true_vs_pred.png')
    
    # Residual plot
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel('Residuals')
    ax.set_title(f'{model_name} Model: Residuals Distribution')
    save_plot(fig, f'{model_name.lower()}_residuals.png')

# Main Function
def main():
    logging.info("Loading dataset...")
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    logging.info("Dataset loaded successfully.")
    
    logging.info("Performing EDA...")
    # Perform EDA
    perform_eda(df)
    
    logging.info("Cleaning the dataset...")
    # Clean the dataset
    df = clean_data(df)
    
    logging.info("Performing EDA after cleaning...")
    # Perform EDA after cleaning
    perform_eda(df)
    
    logging.info("Splitting dataset into training and testing subsets...")
    # Split dataset into training and testing subsets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    logging.info("Dataset split completed.")
    
    logging.info("Running PyCaret Linear Regression...")
    # PyCaret Linear Regression
    pycaret_model, pycaret_r2, pycaret_rmse = pycaret_linear_regression(train_df, test_df)
    
    logging.info("Running FLAML Linear Regression...")
    # FLAML Linear Regression
    flaml_model, flaml_r2, flaml_rmse = flaml_linear_regression(train_df, test_df)
    
    logging.info("--- Initial Model Comparison ---")
    logging.info("PyCaret Model: %s", pycaret_model)
    logging.info("PyCaret R²: %s, RMSE: %s", pycaret_r2, pycaret_rmse)
    logging.info("FLAML Model: %s", flaml_model.best_estimator)
    logging.info("FLAML R²: %s, RMSE: %s", flaml_r2, flaml_rmse)
    
    # Compare the best models from PyCaret and FLAML
    best_model = compare_best_models(pycaret_model, flaml_model, test_df)
    
    # Visualize the performance of the best model
    model_name = "PyCaret" if best_model == pycaret_model else "FLAML"
    visualize_model_performance(best_model, test_df, model_name)

    # Generate PDF report
    generate_pdf_report()

if __name__ == "__main__":
    main()
