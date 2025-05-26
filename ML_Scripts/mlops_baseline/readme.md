# Model Garden: End-to-End AutoML Pipeline with MLOps + FastAPI Deployment

## Project Objective

To build a modular and automated machine learning platform that allows users to input a dataset, select a prediction target, and execute a full MLOps lifecycle. This includes data ingestion & validation, automated preprocessing, exploratory data analysis (EDA), AutoML-driven model training and tuning across a variety of regressors (Random Forest, XGBoost, Ridge, LightGBM, CatBoost, SVR, KNeighbors, ElasticNet), hyperparameter optimization with Optuna, model evaluation, comprehensive experiment tracking and model registry with MLflow (including promotion to "Staging"), and model serving via a FastAPI-based REST API.

This project simulates a self-service "model garden" experience where users can obtain production-ready models with minimal manual intervention.

## Key Features & Capabilities

* **Automated Data Preparation (`prep_pipeline.py`):**
    * Ingests CSV data.
    * Validates data schema using Pandera (`data_ingest.py`).
    * Performs data cleaning: deduplication and median imputation for numerical features.
    * Encodes categorical features using one-hot encoding.
    * Generates a comprehensive EDA report using YData Profiling (`eda_report.html`).
    * (Optional) Performs Deep Feature Synthesis (DFS) using Featuretools.
    * Outputs `prepared.parquet` (or `featuretools_matrix.parquet`) and `prep_manifest.json`.
* **AutoML Training & Tuning (`train_pipeline.py`):**
    * Consumes data from the preparation pipeline.
    * Supports a configurable "Model Garden" (`MODEL_CONFIG`) including:
        * RandomForestRegressor
        * XGBRegressor
        * Ridge Regression
        * LGBMRegressor
        * CatBoostRegressor
        * Support Vector Regressor (SVR)
        * KNeighborsRegressor
        * ElasticNet
    * Performs model-specific conditional feature scaling (e.g., `StandardScaler`).
    * Automated hyperparameter optimization for each model using Optuna.
    * **MLflow Integration:**
        * Tracks all Optuna studies and individual trials (parameters, metrics, tags).
        * Logs final retrained models, their parameters, evaluation metrics (RMSE), and scaler artifacts.
        * Registers each trained model family to the MLflow Model Registry.
        * Automatically promotes the overall best-performing model to the "Staging" alias.
    * Saves final model (`.joblib`), scaler (`.joblib` if used), and `train_columns.json` locally in the `artifacts/` directory.
* **Model Serving (`model_serving_api.py`):**
    * FastAPI application for serving trained models.
    * Dynamically loads all available models and their scalers from the `artifacts/` directory at startup.
    * **Endpoints:**
        * `/health`: API health check and count of loaded models.
        * `/available_models`: Lists all model aliases ready for prediction.
        * `/predict`: Accepts a model alias and raw features (JSON payload). Performs necessary preprocessing (one-hot encoding, scaling, column alignment) and returns the prediction.
* **API Testing (`test_api.py`):**
    * Automated tests using `pytest` and FastAPI's `TestClient`.
    * Covers health checks, available models, and prediction endpoints for all loaded models.
* **Modularity & Extensibility:** Designed to easily add new models to the training pipeline via `MODEL_CONFIG`.

## Project Structure

mlops_baseline/
├── artifacts/            # Stores generated models, scalers, data, reports, manifests
│   ├── eda_report.html
│   ├── prep_manifest.json
│   ├── prepared.parquet
│   ├── featuretools_matrix.parquet (optional)
│   ├── train_columns.json
│   ├── randomforest_model.joblib
│   ├── xgboost_model.joblib
│   ├── ridge_model.joblib
│   ├── ridge_scaler.joblib
│   └── ... (other model and scaler files)
├── data/
│   └── college.csv       # Sample dataset
├── src/
│   ├── data_ingest.py    # Data validation schema and ingestion
│   ├── prep_pipeline.py  # Data preparation and EDA generation
│   ├── train_pipeline.py # Model training, tuning, and registration
│   └── test_api.py       # API tests
├── model_serving_api.py  # FastAPI application for model serving
├── requirements.txt      # Project dependencies
├── README.md             # This file
└── mlruns/               # MLflow tracking data (if using local tracking)


## Tech Stack & Requirements

* Python 3.9+
* Pandas, NumPy
* Scikit-learn
* XGBoost
* LightGBM
* CatBoost
* Optuna (for hyperparameter optimization)
* MLflow (for experiment tracking and model registry)
* FastAPI (for model serving API)
* Uvicorn (ASGI server for FastAPI)
* Pandera (for data validation)
* YData Profiling (for EDA reports)
* Featuretools (optional, for Deep Feature Synthesis)
* Joblib (for model serialization)
* Pytest (for API testing)

See `requirements.txt` for a full list of dependencies and versions.

## Setup Instructions

1.  **Clone the Repository (if applicable)**
    ```bash
    git clone <repository_url>
    cd mlops_baseline
    ```

2.  **Create Conda Environment:**
    It's recommended to use a Conda environment.
    ```bash
    conda create -n mlops_env_1 python=3.10  # Or your preferred Python 3.9+
    conda activate mlops_env_1
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **MLflow Setup:**
    * **Local Tracking:** MLflow will automatically use a local `mlruns` directory for tracking if no tracking URI is set.
    * **Remote Tracking Server (Optional):** If you have an MLflow tracking server, set the `MLFLOW_TRACKING_URI` environment variable:
        ```bash
        export MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
        ```
    * You can view the MLflow UI by running `mlflow ui` in your terminal (from the directory containing `mlruns` or if `MLFLOW_TRACKING_URI` is set).

## Usage Instructions (Workflow)

Execute the pipelines in the following order:

**1. Data Preparation (`prep_pipeline.py`)**

* This script ingests, validates, cleans, encodes data, and generates an EDA report.
* **Arguments:**
    * `--input`: Path to the input CSV dataset (e.g., `data/college.csv`).
    * `--target`: Name of the target column (e.g., `Grad.Rate`).
    * `--dfs` (optional flag): Enable Deep Feature Synthesis with Featuretools.

* **Example (without DFS):**
    ```bash
    python src/prep_pipeline.py --input data/college.csv --target Grad.Rate
    ```
* **Example (with DFS):**
    ```bash
    python src/prep_pipeline.py --input data/college.csv --target Grad.Rate --dfs
    ```
* **Outputs:** Artifacts will be saved in the `artifacts/` directory (`prepared.parquet`, `eda_report.html`, `prep_manifest.json`, and `featuretools_matrix.parquet` if `--dfs` is used).

**2. Model Training & Selection (`train_pipeline.py`)**

* This script trains and tunes all models defined in `MODEL_CONFIG`, logs to MLflow, registers models, and saves artifacts.
* **Arguments:**
    * `--use_dfs` (optional flag): If specified, the pipeline will attempt to load features from `artifacts/featuretools_matrix.parquet` (generated by `prep_pipeline.py --dfs`). Otherwise, it uses `artifacts/prepared.parquet`.

* **Example (using default prepared data):**
    ```bash
    python src/train_pipeline.py
    ```
* **Example (using DFS features):**
    ```bash
    python src/train_pipeline.py --use_dfs
    ```
* **Outputs:**
    * MLflow: Experiments updated with new runs, models registered, winner promoted.
    * `artifacts/`: Updated with `<model_alias>_model.joblib`, `<model_alias>_scaler.joblib` (if applicable), and `train_columns.json`.
    * Console: Leaderboard and status messages.

**3. Run the Model Serving API (`model_serving_api.py`)**

* This starts the FastAPI server. Ensure the `artifacts/` directory contains the trained models, scalers, and `train_columns.json`.
    ```bash
    uvicorn model_serving_api:app --reload --port 8000
    ```
* The API will be available at `http://127.0.0.1:8000`.
* Check startup logs to see which models and scalers were loaded.

**4. Test the API (`test_api.py`)**

* Ensure the FastAPI server (from step 3) is **NOT** running if you are using `pytest` with `TestClient`, as `TestClient` runs the app in memory. If your tests are designed to hit a running server, then keep it running. (The provided `test_api.py` uses `TestClient`).
* Run the tests from the project root:
    ```bash
    pytest src/test_api.py
    ```
* **Key Endpoints to interact with manually (e.g., via browser or `curl` if API is running):**
    * `http://127.0.0.1:8000/docs`: Interactive API documentation (Swagger UI).
    * `http://127.0.0.1:8000/health`: Health status.
    * `http://127.0.0.1:8000/available_models`: List of loaded model aliases.
    * `POST http://127.0.0.1:8000/predict`: Send JSON payload with `model_alias` and `features`.

## Extending the Model Garden (Adding New Models)

To add a new machine learning model to the training pipeline:

1.  **Open `src/train_pipeline.py`**.
2.  **Import the Model Class:** Add the import statement for your new model (e.g., `from sklearn.ensemble import AdaBoostRegressor`).
3.  **Define an Optuna Space Function:** Create a new function similar to `get_rf_optuna_space` that defines the hyperparameter search space for your new model using Optuna's `trial.suggest_*` methods.
    ```python
    def get_adaboost_optuna_space(trial: optuna.trial.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True)
        }
    ```
4.  **Add to `MODEL_CONFIG`:** Add a new entry to the `MODEL_CONFIG` dictionary:
    ```python
    MODEL_CONFIG = {
        # ... existing model entries ...
        "AdaBoost": { # Choose a unique alias
            "model_class": AdaBoostRegressor, # The imported class
            "space_func": get_adaboost_optuna_space, # Reference to the space function
            "fixed_params": {"random_state": RND_STATE},
            "requires_scaling": False # Or True, if your model needs scaling
        },
    }
    ```
5.  **Update `requirements.txt`** if the new model comes from a new library and reinstall dependencies.
6.  Re-run `src/train_pipeline.py`. The new model will be automatically included in the training and tuning process.
7.  Restart `model_serving_api.py` to load the newly trained model.
8.  Update `EXPECTED_MODEL_ALIASES` in `src/test_api.py` and re-run tests.

## MLflow Usage

* **Experiment Tracking:** All training runs, Optuna trials, parameters, and metrics are logged to MLflow. By default, this uses a local `mlruns` directory.
* **Model Registry:** Final trained models are registered with names like `college_grad_rate_RandomForest`, `college_grad_rate_XGBoost`, etc. The best overall model (based on test set RMSE) is promoted to the "Staging" alias.
* **Viewing UI:** Navigate to the directory containing `mlruns` (usually your project root) and run `mlflow ui`. Open your browser to `http://127.0.0.1:5000`.

