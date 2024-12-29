import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import setup as clf_setup, compare_models as clf_compare_models, tune_model as clf_tune_model, predict_model as clf_predict_model
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import logging
import os

# Configure logging
log_path = 'C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Classification/Reports/logs.log'
logging.basicConfig(
    filename=log_path, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/Titanic.csv"
TARGET_COLUMN = "survived"  # Ensure this matches the exact column name in your dataset
REPORTS_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Classification/Reports"

# Ensure the reports directory exists
os.makedirs(REPORTS_PATH, exist_ok=True)

# Step 1: Exploratory Data Analysis (EDA)
def perform_eda(df, target_column_lower, title="EDA"):
    logging.info(f"--- {title} ---")
    logging.info("Dataset Head:\n%s", df.head())
    logging.info("Dataset Info:")
    df.info()  # Logging info directly doesn’t capture info output
    logging.info("Dataset Description:\n%s", df.describe())

    # Visualize distributions of features
    plt.figure(figsize=(20, 15))
    df.hist(bins=30, figsize=(20, 15), layout=(5, 3))
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_PATH, f"{title}_distributions.png"))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(REPORTS_PATH, f"{title}_correlation_heatmap.png"))
    plt.close()

    # EDA for target column
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_column_lower], kde=True)
    plt.title(f'Distribution of {TARGET_COLUMN}')
    plt.xlabel(TARGET_COLUMN)
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(REPORTS_PATH, f"{title}_{TARGET_COLUMN}_distribution.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[target_column_lower])
    plt.title(f'Boxplot of {TARGET_COLUMN}')
    plt.xlabel(TARGET_COLUMN)
    plt.savefig(os.path.join(REPORTS_PATH, f"{title}_{TARGET_COLUMN}_boxplot.png"))
    plt.close()

# Step 2: Data Cleaning
def clean_data(df):
    logging.info("--- Data Cleaning ---")
    logging.info("Initial dataset shape: %s", df.shape)

    # Remove rows with missing target values
    if TARGET_COLUMN in df.columns:
        missing_target_count = df[TARGET_COLUMN].isna().sum()
        if missing_target_count > 0:
            logging.info("Removing %s rows with missing target values.", missing_target_count)
            df = df[df[TARGET_COLUMN].notna()]

    # Remove duplicates
    df = df.drop_duplicates()
    logging.info("Shape after removing duplicates: %s", df.shape)

    # Fill missing values with median for numeric columns
    df = df.fillna(df.median(numeric_only=True))
    logging.info("Shape after filling missing values: %s", df.shape)

    # Remove outliers using percentiles for numeric columns
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            lower_percentile = df[col].quantile(0.01)
            upper_percentile = df[col].quantile(0.99)
            df = df[(df[col] >= lower_percentile) & (df[col] <= upper_percentile)]
            logging.info("Shape after removing outliers in %s: %s", col, df.shape)

    logging.info("Dataset shape after cleaning: %s", df.shape)
    return df

# Preprocess data
def preprocess_data(train_df, test_df, target_column_lower):
    # Identify numeric and categorical columns
    numeric_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove the target column from features if present
    if target_column_lower in numeric_features:
        numeric_features.remove(target_column_lower)
    if target_column_lower in categorical_features:
        categorical_features.remove(target_column_lower)

    # Debug: Print features for validation
    logging.info("Numeric Features: %s", numeric_features)
    logging.info("Categorical Features: %s", categorical_features)
    print(f"Numeric Features: {numeric_features}")
    print(f"Categorical Features: {categorical_features}")

    # Define a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'
    )

    # Separate features and target
    X_train = train_df.drop(columns=[target_column_lower])
    y_train = train_df[target_column_lower]
    X_test = test_df.drop(columns=[target_column_lower])
    y_test = test_df[target_column_lower]

    # Validate column existence
    print(f"Columns in X_train before preprocessing: {X_train.columns}")
    print(f"Columns in X_test before preprocessing: {X_test.columns}")

    # Apply transformations
    try:
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
    except ValueError as e:
        logging.error("Error in ColumnTransformer: %s", e)
        raise

    # Convert preprocessed data back to DataFrame with original column names
    X_train_columns = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    X_train = pd.DataFrame(X_train, columns=X_train_columns)
    X_test = pd.DataFrame(X_test, columns=X_train_columns)

    return X_train, X_test, y_train, y_test

# Step 3: PyCaret Classification with Hyperparameter Tuning
def pycaret_classification(train_df, test_df, X_train, X_test, y_train, y_test):
    logging.info("--- PyCaret Classification ---")
    logging.info("Setting up PyCaret...")

    # Setup PyCaret
    clf_setup(data=train_df, target=TARGET_COLUMN, session_id=123, verbose=False, preprocess=True)
    logging.info("Comparing models to find the best one...")

    # Compare models to find the best one
    best_model = clf_compare_models()
    logging.info("Best Model from PyCaret (Pre-Tuning):\n%s", best_model)

    logging.info("Tuning the best model...")
    # Tune the best model
    tuned_model = clf_tune_model(best_model, optimize="Accuracy")
    logging.info("Tuned Model from PyCaret:\n%s", tuned_model)

    logging.info("Predicting on test set...")
    # Predict on test set
    # Remove rows with missing target values before prediction
    test_data = pd.concat([X_test, y_test], axis=1).dropna(subset=[TARGET_COLUMN])
    test_data.columns = X_train.columns.tolist() + [TARGET_COLUMN]  # Ensure test data has the same columns as train data
    predictions = clf_predict_model(tuned_model, data=test_data)
    accuracy = accuracy_score(test_df[TARGET_COLUMN], predictions['prediction_label'])
    report = classification_report(test_df[TARGET_COLUMN], predictions['prediction_label'])

    logging.info("PyCaret Accuracy: %s", accuracy)
    logging.info("PyCaret Classification Report:\n%s", report)

    return tuned_model, accuracy, report

# Step 4: FLAML Classification with Hyperparameter Tuning
def flaml_classification(train_df, test_df):
    logging.info("--- FLAML Classification ---")

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
    automl.fit(X_train=X_train, y_train=y_train, task="classification", time_budget=300)
    logging.info("Best Model from FLAML (Pre-Tuning):\n%s", automl.best_estimator)
    logging.info("Best Config: %s", automl.best_config)
    logging.info("Best Loss: %s", automl.best_loss)

    # Ensure consistent filtering of X_test and y_test
    valid_indices = X_test.index
    y_test = y_test.loc[valid_indices].dropna()

    logging.info("Predicting on test set...")
    # Predict on test set
    predictions = automl.predict(X_test)
    predictions = predictions[:len(y_test)]  # Ensure lengths match

    # Remove NaN values from predictions and corresponding y_test values
    valid_indices = ~np.isnan(predictions)
    predictions = predictions[valid_indices]
    y_test = y_test[valid_indices]

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    logging.info("FLAML Accuracy: %s", accuracy)
    logging.info("FLAML Classification Report:\n%s", report)

    return automl, accuracy, report

# Function to visualize model performance
def visualize_model_performance(model, test_df, model_name, train_columns):
    logging.info("Visualizing %s Model Performance", model_name)

    # Ensure test_df has the same columns as the training data
    test_df_aligned = test_df.reindex(columns=train_columns, fill_value=0)

    if model_name == "PyCaret":
        predictions = clf_predict_model(model, data=test_df_aligned)
        y_true = test_df_aligned[TARGET_COLUMN]
        y_pred = predictions['prediction_label']
    else:
        X_test = test_df_aligned.drop(columns=[TARGET_COLUMN])
        y_true = test_df_aligned[TARGET_COLUMN]
        y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Model: Confusion Matrix')
    plt.savefig(os.path.join(REPORTS_PATH, f"{model_name}_confusion_matrix.png"))
    plt.close()

def main():
    logging.info("Loading dataset...")

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    logging.info("Dataset loaded successfully.")

    # Ensure column names are in the correct case
    df.columns = df.columns.str.strip().str.lower()
    target_column_lower = TARGET_COLUMN.lower()
    
    # Check if the target column exists
    if target_column_lower not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in the dataset.")
    
    logging.info("Performing EDA before cleaning...")
    # Perform EDA before cleaning
    perform_eda(df, target_column_lower, title="EDA Before Cleaning")

    logging.info("Cleaning the dataset...")
    # Clean the dataset
    df = clean_data(df)

    # Remove rows with missing target values
    df = df.dropna(subset=[target_column_lower])

    logging.info("Columns after cleaning: %s", df.columns)
    print(df.columns)

    logging.info("Performing EDA after cleaning...")
    # Perform EDA after cleaning
    perform_eda(df, target_column_lower, title="EDA After Cleaning")

    logging.info("Splitting dataset into training and testing subsets...")
    # Split dataset into training and testing subsets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
    logging.info("Dataset split completed.")
    logging.info("Columns in train_df: %s", train_df.columns)
    logging.info("Columns in test_df: %s", test_df.columns)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df, target_column_lower)

    # Combine preprocessed features with targets
    train_df_clean = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_df_clean = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    # PyCaret Classification
    logging.info("Running PyCaret Classification...")
    pycaret_model, pycaret_accuracy, pycaret_report = pycaret_classification(
        train_df_clean, test_df_clean, X_train, X_test, y_train, y_test
    )

    # FLAML Classification
    logging.info("Running FLAML Classification...")
    flaml_model, flaml_accuracy, flaml_report = flaml_classification(train_df_clean, test_df_clean)

    # Model Comparison
    logging.info("--- Model Comparison ---")
    logging.info("PyCaret Accuracy: %s", pycaret_accuracy)
    logging.info("FLAML Accuracy: %s", flaml_accuracy)

    # Print results
    print("\n--- Model Comparison ---")
    print(f"PyCaret Accuracy: {pycaret_accuracy}")
    print(f"FLAML Accuracy: {flaml_accuracy}")

    # Visualize model performance
    visualize_model_performance(pycaret_model, test_df_clean, "PyCaret", train_df_clean.columns)
    visualize_model_performance(flaml_model, test_df_clean, "FLAML", train_df_clean.columns)

    # Determine best model
    best_model_name = "PyCaret" if pycaret_accuracy > flaml_accuracy else "FLAML"
    logging.info("Best Model: %s", best_model_name)
    print(f"\nBest Model: {best_model_name}")

if __name__ == "__main__":
    main()