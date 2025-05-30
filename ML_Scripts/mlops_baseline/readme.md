# MLOps Baseline Project: Simplified Overview

Hi Vinay,

This document provides a high-level summary of our MLOps Baseline project. The goal was to build a foundational, automated pipeline for developing, tracking, and deploying machine learning models efficiently.

## What We've Built: An End-to-End ML Pipeline

We've created a system that takes a dataset and a prediction target, and then automatically handles:

1.  **Data Preparation (`src/prep_pipeline.py`):**
    * Loads data (e.g., CSV files like `titanic.csv` or `college.csv`).
    * Cleans it (handles duplicates, fills missing numerical values using mean imputation).
    * Prepares features for models (e.g., converts text categories into numbers using one-hot encoding).
    * Saves the processed data and a list of features (`train_columns.json`) needed for consistent predictions.

2.  **Model Training & Experimentation (`src/train_pipeline.py`):**
    * **Task-Aware:** Can be run for "classification" (like predicting 'Survived' on Titanic) or "regression" (like predicting 'Grad.Rate' on College data) by changing a command-line flag (`--task_type`).
    * **Model Variety:** Trains a "garden" of different models:
        * For Classification: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVC, K-Nearest Neighbors.
        * (Regression models like Ridge, RandomForestRegressor, etc., are defined and can be trained by switching the task type).
    * **Smart Tuning (Optuna):** Automatically finds the best settings (hyperparameters) for each model to maximize its performance (e.g., F1-score for classification).
    * **MLflow Integration (Key for MLOps):**
        * **Tracks Everything:** Every training run, its parameters, and performance scores are logged in MLflow. This means we have a full history and can compare models easily.
        * **Model Registry:** Saves all trained models in a central place (MLflow Model Registry), versioning them.
        * **Best Model Staging:** Automatically identifies the best performing model on test data and promotes it to a "Staging" area in MLflow, ready for potential deployment.
    * **Local Artifacts:** Saves the trained models, scalers (for feature normalization), and the label encoder (for classification targets) in the `artifacts/` folder for the API to use.

3.  **Model Serving API (`model_serving_api.py`):**
    * **FastAPI Server:** A high-performance API that makes our trained models available for predictions.
    * **Automatic Loading:** When the API starts, it automatically finds and loads all the trained models and necessary preprocessing tools (scalers, `train_columns.json`, label encoder) from the `artifacts/` folder. It knows if a model is for classification or regression based on its filename.
    * **Prediction Endpoint (`/predict`):**
        * Takes new, raw data as input (you tell it which model alias to use).
        * Internally, it performs the *exact same* preprocessing steps (one-hot encoding, scaling, handling of missing columns, fixing NaN/infinity values from scaling, and special name cleaning for LightGBM/CatBoost) that were done during training. This ensures consistency.
        * Returns the model's prediction.
    * **Helper Endpoints:**
        * `/health`: Shows if the API is running and how many models are loaded.
        * `/available_models`: Lists all model aliases ready for use.
        * `/docs`: An interactive page to see and test all API functions.

4.  **API Testing (`src/test_api.py`):**
    * **Automated Checks (Pytest):** We have a suite of tests that automatically check if the API is working correctly – from loading models to making predictions and handling various types of input.
    * **Current Status:** **All 15 functional tests are passing** for the classification task with the Titanic dataset (the 16th test was a temporary diagnostic one that can be removed). This confirms the API loads models, preprocesses input, and returns predictions as expected.

## Current Status & What You Can See Today:

* **Fully Functional for Classification (Titanic Dataset):** The entire pipeline from data prep to API serving and testing is working end-to-end for the Titanic classification task.
* **Adaptable for Regression:** The `train_pipeline.py` is built to handle regression by changing the `--task_type` flag. The API is also designed to load and serve regression models.
* **MLflow Tracking & Registry Operational:** All experiments and models are being logged and registered.

**How to See It In Action (Demonstration Steps):**

1.  **Environment Setup:**
    * Ensure Python and Conda are installed.
    * Create and activate the `mlops_env_1` conda environment:
        ```bash
        conda create -n mlops_env_1 python=3.10 
        conda activate mlops_env_1
        pip install -r requirements.txt
        ```
2.  **Clean Previous Artifacts (Important for a fresh demo):**
    * Manually empty the `mlops_baseline/artifacts/` directory.
    * (Optional, for MLflow) Delete the `mlops_baseline/mlruns/` directory and `mlops_baseline/mlflow.db` if you want a completely fresh MLflow history for the demo.
3.  **Run Data Preparation (Titanic Classification):**
    ```bash
    python src/prep_pipeline.py --input data/titanic.csv --target Survived
    ```
    * *Observe: `artifacts/` folder gets populated (`prepared.parquet`, `train_columns.json`).*
4.  **Run Model Training (Titanic Classification):**
    ```bash
    python src/train_pipeline.py --task_type classification
    ```
    * *Observe: Console output showing model training, Optuna progress. `artifacts/` gets model files. Check MLflow UI (`mlflow ui` in a new terminal) for new runs, registered models, and the "Staging" model.*
5.  **Start the API Server:**
    ```bash
    uvicorn model_serving_api:app --reload --port 8000
    ```
    * *Observe: Server startup logs showing models being loaded from `artifacts/`.*
6.  **Run API Tests (in a new terminal, ensure Uvicorn server is stopped first for `TestClient`):**
    * Stop the Uvicorn server (Ctrl+C in its terminal).
    * Run:
        ```bash
        pytest src/test_api.py -v -s
        ```
    * *Observe: Test results (should be 15 or 16 passing if `test_lifespan_execution_flag` is removed/kept).*
7.  **Manual API Interaction (Restart Uvicorn server if stopped for pytest):**
    * If Uvicorn was stopped, restart it: `uvicorn model_serving_api:app --reload --port 8000`
    * Open `http://127.0.0.1:8000/docs` in your browser to see the API and try the `/predict` endpoint with sample Titanic data.

## Next Steps & Cloud Integration Path

This baseline is a strong foundation. Future enhancements include:

* **More Sophisticated Feature Engineering:** Especially for high-cardinality features in `prep_pipeline.py`.
* **Data & Model Monitoring:** Implementing checks for data drift and model performance degradation over time.
* **Cloud Deployment:**
    * **Containerize (Docker):** Package the FastAPI for cloud deployment.
    * **Remote Artifact Storage:** Use S3/GCS/Azure Blob for storing models and other artifacts.
    * **Remote MLflow Server:** Set up a centralized MLflow server.
    * **Cloud Serving Platforms:** Deploy the containerized API to services like AWS Lambda, Google Cloud Run, Azure App Service, or Kubernetes (EKS, GKE, AKS).
    * **CI/CD Pipelines:** Automate the entire workflow from code commit to deployment using tools like GitHub Actions, Jenkins, etc.

This setup provides a robust starting point for building and deploying various ML models in a structured and automated way.
