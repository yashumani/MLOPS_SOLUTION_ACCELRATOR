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
import optuna
import sweetviz as sv

# Configure logging
log_path = 'C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Classification/Reports/logs.log'
logging.basicConfig(
    filename=log_path, 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
DATA_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/data/datasets/BreastCancer.csv"
TARGET_COLUMN = "Class".lower()
REPORTS_PATH = "C:/Users/yashu/Desktop/SAVYMINDS/MLOps/YS_MVP/ML_Scripts/Classification/Reports"

# Ensure the reports directory exists
os.makedirs(REPORTS_PATH, exist_ok=True)

def perform_eda(df, target_column, title="EDA"):
    logging.info(f"--- {title} ---")
    logging.info("Generating EDA report using Sweetviz...")

    # Generate Sweetviz report
    report = sv.analyze(df, target_feat=target_column)
    report.show_html(os.path.join(REPORTS_PATH, f"{title}_report.html"))

def clean_data(df, target_column):
    logging.info("--- Data Cleaning ---")
    logging.info("Initial dataset shape: %s", df.shape)

    # Remove rows with missing target values
    if target_column in df.columns:
        missing_target_count = df[target_column].isna().sum()
        if missing_target_count > 0:
            logging.info("Removing %s rows with missing target values.", missing_target_count)
            df = df[df[target_column].notna()]

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

def preprocess_data(train_df, test_df, target_column):
    # Identify numeric and categorical columns
    numeric_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove the target column from features if present
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Debug: Print features for validation
    logging.info("Numeric Features: %s", numeric_features)
    logging.info("Categorical Features: %s", categorical_features)

    # Fit OneHotEncoder separately
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(train_df[categorical_features])

    # Define a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', ohe, categorical_features)
        ],
        remainder='drop'
    )

    # Separate features and target
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # Fit the preprocessor
    preprocessor.fit(X_train)

    # Apply transformations
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Convert preprocessed data back to DataFrame with original column names
    X_train_columns = numeric_features + list(ohe.get_feature_names_out())
    X_train = pd.DataFrame(X_train, columns=X_train_columns)
    X_test = pd.DataFrame(X_test, columns=X_train_columns)

    return X_train, X_test, y_train, y_test

# Step 3: PyCaret Classification with Hyperparameter Tuning
# PyCaret Classification with Optuna Tuning
def pycaret_classification(train_df, test_df, X_train, X_test, y_train, y_test, target_column):
    logging.info("--- PyCaret Classification ---")
    logging.info("Setting up PyCaret...")

    # Setup PyCaret
    clf_setup(data=train_df, target=target_column, session_id=123, verbose=True, preprocess=True)
    logging.info("Comparing models to find the best one...")

    # Compare models
    try:
        best_model = clf_compare_models(fold=5, turbo=False)
        if not best_model:
            raise ValueError("No models were returned by clf_compare_models.")
    except Exception as e:
        logging.error("Error during model comparison: %s", e)
        raise

    # Define Optuna objective for tuning
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
        }
        try:
            tuned_model = clf_tune_model(
                best_model, 
                custom_grid={"n_estimators": [params["n_estimators"]], "max_depth": [params["max_depth"]]}, 
                optimize="Accuracy", 
                n_iter=1  # Single iteration as we're manually specifying grid
            )
            predictions = clf_predict_model(tuned_model, data=test_df)
            accuracy = accuracy_score(test_df[target_column], predictions["prediction_label"])
            return accuracy
        except Exception as e:
            logging.error("Error during model tuning: %s", e)
            return 0.0


    # Optimize using Optuna
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=20)  # Reduce trials for testing

    best_trial = study.best_trial
    logging.info("Best Trial Params: %s", best_trial.params)
    logging.info("Best Trial Accuracy: %s", best_trial.value)

    # Tune the best model with the best parameters
    tuned_model = clf_tune_model(
        best_model, 
        custom_grid={"n_estimators": [best_trial.params["n_estimators"]], "max_depth": [best_trial.params["max_depth"]]}, 
        optimize="Accuracy", 
        n_iter=1,
        search_library="scikit-learn"
    )
    logging.info("Tuned Model from PyCaret:\n%s", tuned_model)

    # Predict on test set
    test_data = pd.concat([X_test, y_test], axis=1).dropna(subset=[target_column])
    test_data = test_data.reindex(columns=X_train.columns.tolist() + [target_column], fill_value=0)
    predictions = clf_predict_model(tuned_model, data=test_data)
    accuracy = accuracy_score(test_df[target_column], predictions["prediction_label"])
    report = classification_report(test_df[target_column], predictions["prediction_label"])

    logging.info("PyCaret Accuracy: %s", accuracy)
    logging.info("PyCaret Classification Report:\n%s", report)

    return tuned_model, accuracy, report



# Step 4: FLAML Classification with Hyperparameter Tuning
def flaml_classification(train_df, test_df, target_column):
    logging.info("--- FLAML Classification ---")

    # Split data into features and target
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    logging.info("Initializing AutoML...")
    # Initialize AutoML
    automl = AutoML()

    logging.info("Fitting the AutoML model...")
    # Fit the AutoML model with custom search space
    automl.fit(
        X_train=X_train, 
        y_train=y_train, 
        task="classification", 
        time_budget=300,
        custom_hp={
            'n_estimators': {'domain': (10, 1000), 'init_value': 100},
            'max_depth': {'domain': (3, 15), 'init_value': 6},
            'learning_rate': {'domain': (1e-4, 1e-1), 'init_value': 0.01, 'log': True}
        }
    )
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
def visualize_model_performance(model, test_df, model_name, train_columns, target_column):
    logging.info("Visualizing %s Model Performance", model_name)

    # Ensure test_df has the same columns as the training data
    test_df_aligned = test_df.reindex(columns=train_columns, fill_value=0)

    if model_name == "PyCaret":
        predictions = clf_predict_model(model, data=test_df_aligned)
        y_true = test_df_aligned[target_column]
        y_pred = predictions['prediction_label']
    else:
        X_test = test_df_aligned.drop(columns=[target_column])
        y_true = test_df_aligned[target_column]
        y_pred = model.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(REPORTS_PATH, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # Classification report
    report = classification_report(y_true, y_pred)
    logging.info(f"{model_name} Classification Report:\n{report}")

def main():
    logging.info("Starting execution...")
    start_time = pd.Timestamp.now()

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    logging.info("Dataset loaded successfully.")

    # Ensure consistent column naming
    df.columns = df.columns.str.strip().str.lower()
    target_column = TARGET_COLUMN.lower()

    # Validate the target column
    if target_column not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in the dataset. Available columns: {df.columns.tolist()}")


    logging.info("Performing EDA before cleaning...")
    perform_eda(df, target_column, title="EDA Before Cleaning")

    logging.info("Cleaning the dataset...")
    df = clean_data(df, target_column)
    df = df.dropna(subset=[target_column])

    logging.info("Performing EDA after cleaning...")
    perform_eda(df, target_column, title="EDA After Cleaning")

    # Ensure valid classes for stratified split
    class_counts = df[target_column].value_counts()
    if class_counts.min() < 2:
        logging.warning("Some classes have fewer than 2 samples. Consider oversampling or removing these classes.")
        df = df[df[target_column].isin(class_counts[class_counts >= 2].index)]

    logging.info("Splitting dataset into training and testing subsets...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=123, stratify=df[target_column]
    )
    logging.info("Dataset split completed.")
    logging.info("Class distributions - Train: %s, Test: %s", 
                 train_df[target_column].value_counts().to_dict(),
                 test_df[target_column].value_counts().to_dict())

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df, target_column)

    # Combine preprocessed features with targets
    train_df_clean = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_df_clean = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

    # PyCaret Classification
    logging.info("Running PyCaret Classification...")
    pycaret_model, pycaret_accuracy, pycaret_report = pycaret_classification(
        train_df_clean, test_df_clean, X_train, X_test, y_train, y_test, target_column
    )

    # FLAML Classification
    logging.info("Running FLAML Classification...")
    flaml_model, flaml_accuracy, flaml_report = flaml_classification(train_df_clean, test_df_clean, target_column)

    # Model Comparison
    logging.info("--- Model Comparison ---")
    logging.info("PyCaret Accuracy: %s", pycaret_accuracy)
    logging.info("FLAML Accuracy: %s", flaml_accuracy)

    print("\n--- Model Comparison ---")
    print(f"PyCaret Accuracy: {pycaret_accuracy}")
    print(f"FLAML Accuracy: {flaml_accuracy}")

    # Visualize model performance
    visualize_model_performance(pycaret_model, test_df_clean, "PyCaret", train_df_clean.columns, target_column)
    visualize_model_performance(flaml_model, test_df_clean, "FLAML", train_df_clean.columns, target_column)

    # Determine best model
    best_model_name = "PyCaret" if pycaret_accuracy > flaml_accuracy else "FLAML"
    logging.info("Best Model: %s", best_model_name)
    print(f"\nBest Model: {best_model_name}")

    end_time = pd.Timestamp.now()
    logging.info("Execution completed in %s", end_time - start_time)


if __name__ == "__main__": 
    main()
