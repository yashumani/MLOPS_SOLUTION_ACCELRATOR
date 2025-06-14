import sys
import re
import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import Dict, Any, List, Union, Optional

# --- Logger Setup ---
logger = logging.getLogger("model_serving_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- Artifacts Configuration ---
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TRAIN_COLUMNS_FILE = ARTIFACTS_DIR / "train_columns.json"

# --- Global Variables for Loaded Objects ---
MODELS: Dict[str, Any] = {}
SCALERS: Dict[str, Any] = {}
TRAIN_COLS: List[str] = []
LABEL_ENCODERS: Dict[str, Any] = {}

# --- Pydantic Models for API ---
class PredictionRequest(BaseModel):
    model_alias: str = Field(..., json_schema_extra={"example": "catboostclassifier_classification"})
    features: Dict[str, Union[str, float, int, None]] = Field(
        ...,
        json_schema_extra={"example": {"Pclass": 3, "Sex": "male", "Age": 29.0, "SibSp": 0, "Parch": 0, "Fare": 7.90, "Embarked": "S"}}
    )
    class Config:
        protected_namespaces = ()

class PredictionResponse(BaseModel):
    model_alias_used: str
    prediction: float
    class Config:
        protected_namespaces = ()

class ModelData:
    """Container for a loaded model and its metadata."""
    def __init__(self, model: Any, task_type: str, alias: str, clustering_features: Optional[List[str]] = None):
        self.model = model
        self.task_type = task_type
        self.alias = alias
        self.clustering_features = clustering_features

# --- Helper Function ---
def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitizes column names for compatibility. This version ensures consistency
    between training and prediction.
    """
    df_copy = df.copy()
    
    # 1. Replace all non-alphanumeric characters with a single underscore
    sanitized_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in df_copy.columns]
    
    # 2. Handle duplicate names created by sanitization by appending a count
    seen = {}
    final_cols = []
    for col in sanitized_cols:
        if col in seen:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_cols.append(col)
            
    df_copy.columns = final_cols
    return df_copy

# --- API Lifespan Events: Load Models on Startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles loading all necessary artifacts when the API starts."""
    global MODELS, SCALERS, TRAIN_COLS, LABEL_ENCODERS
    logger.info("LIFESPAN: Application startup sequence initiated.")

    if not ARTIFACTS_DIR.exists():
        logger.error(f"FATAL: Artifacts directory not found at {ARTIFACTS_DIR}. Cannot start API.")
        return

    # --- Load Global Training Columns (Master Feature List from Prep Pipeline) ---
    if TRAIN_COLUMNS_FILE.exists():
        try:
            with open(TRAIN_COLUMNS_FILE, 'r') as f:
                TRAIN_COLS = json.load(f)
            logger.info(f"LIFESPAN: Successfully loaded train_columns.json ({len(TRAIN_COLS)} columns)")
        except Exception as e:
            logger.error(f"LIFESPAN: Error loading {TRAIN_COLUMNS_FILE}: {e}", exc_info=True)
    else:
        logger.warning(f"LIFESPAN: {TRAIN_COLUMNS_FILE} not found. API may have issues processing features.")

    # --- Load Task-Specific Label Encoders ---
    for task in ["classification", "regression"]:
        le_path = ARTIFACTS_DIR / f"label_encoder_{task}.joblib"
        if le_path.exists():
            try:
                LABEL_ENCODERS[task] = joblib.load(le_path)
                logger.info(f"LIFESPAN: Successfully loaded label encoder for '{task}' task.")
            except Exception as e:
                logger.error(f"LIFESPAN: Error loading label encoder {le_path}: {e}", exc_info=True)

    # --- Load All Available Models and Their Specific Artifacts ---
    model_files = list(ARTIFACTS_DIR.glob("*_model.joblib"))
    logger.info(f"LIFESPAN: Found {len(model_files)} model file(s) to process.")

    for model_path in model_files:
        try:
            filename = model_path.name
            parts = filename.rsplit('_', 2)
            if len(parts) < 3 or parts[-1] != "model.joblib":
                logger.warning(f"LIFESPAN: Skipping file with unexpected name format: {filename}")
                continue
            
            task_type = parts[-2]
            model_base_name = parts[0]
            full_alias = f"{model_base_name}_{task_type}"

            if task_type not in ["classification", "regression", "clustering"]:
                logger.warning(f"LIFESPAN: Skipping model {filename} with unknown task_type '{task_type}'.")
                continue
            
            model = joblib.load(model_path)
            
            clustering_features_list = None
            if task_type == "clustering":
                clustering_features_filename = f"{model_base_name}_clustering_features.json"
                clustering_features_filepath = ARTIFACTS_DIR / clustering_features_filename
                if clustering_features_filepath.exists():
                    with open(clustering_features_filepath, 'r') as f:
                        clustering_features_list = json.load(f)
                    logger.info(f"LIFESPAN: Loaded {len(clustering_features_list)} specific features for clustering model {full_alias}.")
                else:
                    logger.warning(f"LIFESPAN: Clustering features file '{clustering_features_filename}' not found for model {full_alias}.")
            
            MODELS[full_alias] = ModelData(model=model, task_type=task_type, alias=full_alias, clustering_features=clustering_features_list)
            
            scaler_filename = f"{model_base_name}_{task_type}_scaler.joblib"
            scaler_path = ARTIFACTS_DIR / scaler_filename
            if scaler_path.exists():
                SCALERS[full_alias] = joblib.load(scaler_path)
            
            logger.info(f"LIFESPAN: Successfully loaded model '{full_alias}' (Task: {task_type}).")

        except Exception as e:
            logger.error(f"LIFESPAN: Failed to load model or artifacts from {model_path.name}: {e}", exc_info=True)
    
    available_models_str = ", ".join(sorted(list(MODELS.keys()))) if MODELS else "None"
    logger.info(f"API ready. Available models: [{available_models_str}]")
    
    yield # API is now running
    
    logger.info("LIFESPAN: Application shutdown sequence initiated.")
    MODELS.clear(); SCALERS.clear(); TRAIN_COLS.clear(); LABEL_ENCODERS.clear()
    logger.info("LIFESPAN: Cleared loaded artifacts.")

app = FastAPI(lifespan=lifespan, title="MLOps Model Garden API", version="2.0.0")

@app.get("/health", summary="Check API Health", response_model=Dict[str, Any])
async def health_check_endpoint():
    """Returns the operational status and list of loaded models."""
    return {"status": "ok", "available_models_count": len(MODELS), "loaded_model_aliases": sorted(list(MODELS.keys()))}

@app.get("/available_models", summary="List Available Model Aliases", response_model=Dict[str, List[str]])
async def get_available_models_endpoint():
    """Provides a list of all model aliases that are loaded and ready for prediction."""
    return {"available_model_aliases": sorted(list(MODELS.keys()))}

@app.post("/predict", response_model=PredictionResponse, summary="Get Prediction from a Model")
async def predict(req: PredictionRequest = Body(...)):
    """
    Receives raw features and a model alias, preprocesses the data consistently
    with the training pipeline, and returns the model's prediction.
    """
    model_alias_requested = req.model_alias.lower()
    logger.info(f"PREDICT: Request received for model_alias: {model_alias_requested}")

    if not TRAIN_COLS:
        logger.error("PREDICT: TRAIN_COLS is not loaded. Cannot process features.")
        raise HTTPException(status_code=500, detail="Training column info unavailable. API may not have initialized correctly.")
    if model_alias_requested not in MODELS:
        logger.warning(f"PREDICT: Model alias '{model_alias_requested}' not found. Available models: {list(MODELS.keys())}")
        raise HTTPException(status_code=404, detail=f"Model alias '{model_alias_requested}' not found.")

    model_data: ModelData = MODELS[model_alias_requested]
    model_to_use = model_data.model
    
    try:
        # --- PREPROCESSING PIPELINE ---
        # 1. Create DataFrame from incoming raw features
        df_features = pd.DataFrame([req.features])

        # 2. One-Hot Encode categorical features present in the input
        cat_cols = df_features.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            df_one_hot = pd.get_dummies(df_features, columns=cat_cols, drop_first=True, dummy_na=False)
        else: 
            df_one_hot = df_features.copy()
        
        # 3. Align with global TRAIN_COLS from training to create a consistent feature space
        df_processed = pd.DataFrame(columns=TRAIN_COLS, index=[0])
        for col in df_one_hot.columns:
            if col in df_processed.columns:
                df_processed[col] = df_one_hot[col].values
        df_processed.fillna(0.0, inplace=True)
        df_processed = df_processed.astype(float)
        logger.info(f"PREDICT: Input aligned to {len(TRAIN_COLS)} features.")

        # 4. Select a specific subset of features if required (e.g., for clustering)
        df_for_predict = df_processed.copy()
        if model_data.task_type == "clustering" and model_data.clustering_features:
            logger.info(f"PREDICT: Selecting {len(model_data.clustering_features)} features for clustering model {model_alias_requested}.")
            try:
                df_for_predict = df_for_predict[model_data.clustering_features]
            except KeyError as ke:
                missing_cols = list(set(model_data.clustering_features) - set(df_for_predict.columns))
                logger.error(f"PREDICT: Key error when selecting clustering features. Missing: {missing_cols}.")
                raise HTTPException(status_code=400, detail=f"Input data missing required features for clustering model: {missing_cols}")
        
        # 5. Apply scaling if a scaler exists for this model
        if model_alias_requested in SCALERS:
            scaler_to_use = SCALERS[model_alias_requested]
            logger.info(f"PREDICT: Applying scaler for model: {model_alias_requested}.")
            scaled_np = scaler_to_use.transform(df_for_predict) 
            df_for_predict = pd.DataFrame(scaled_np, columns=df_for_predict.columns, index=df_for_predict.index)
            logger.info("PREDICT: Scaling complete.")

            # Handle potential NaNs/Infs produced by scaler (e.g., from division by zero std dev)
            if np.any(np.isinf(df_for_predict.values)) or np.any(np.isnan(df_for_predict.values)):
                logger.warning(f"PREDICT: NaNs or Infs detected after scaling. Replacing with 0.")
                df_for_predict = df_for_predict.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 6. Sanitize feature names if model requires it (must match training sanitization)
        if model_data.model.__class__.__name__.startswith(("LGBM", "CatBoost")):
            logger.info(f"PREDICT: Sanitizing feature names for {model_alias_requested}.")
            df_for_predict = sanitize_feature_names(df_for_predict)

        # 7. Make Prediction
        logger.info(f"PREDICT: Making prediction with {model_alias_requested} on final data with shape {df_for_predict.shape}.")
        if not hasattr(model_to_use, 'predict'):
            logger.error(f"PREDICT: Model {model_alias_requested} does not support .predict() for new instances.")
            raise HTTPException(status_code=501, detail=f"Prediction not supported for model type {model_data.model.__class__.__name__}.")

        prediction_array = model_to_use.predict(df_for_predict)
        prediction_value = prediction_array[0]
        logger.info(f"PREDICT: Prediction successful. Raw result: {prediction_value}")
        
        return PredictionResponse(model_alias_used=model_alias_requested, prediction=float(prediction_value))
    except Exception as e:
        logger.error(f"PREDICT: Unhandled error during prediction for model {model_alias_requested}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("model_serving_api:app", host="0.0.0.0", port=8000, reload=True)
