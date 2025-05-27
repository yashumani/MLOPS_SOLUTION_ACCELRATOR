import sys
import re
import json # For loading train_columns.json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import Dict, Any, List, Union

# --- Logger Setup ---
logger = logging.getLogger("model_serving_api")
# Set basicConfig level to INFO or DEBUG for regular Uvicorn runs
# The pytest -s flag will show logs based on handler settings.
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

# --- Pydantic Models (Corrected for Pydantic V2 examples) ---
class PredictionRequest(BaseModel):
    model_alias: str = Field(..., json_schema_extra={"example": "xgbclassifier_classification"})
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
    def __init__(self, model: Any, task_type: str, alias: str):
        self.model = model
        self.task_type = task_type
        self.alias = alias

# --- Sanitization Function ---
def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in df_copy.columns]
    return df_copy

# --- Loading Functions with Enhanced (Error-Level for Debug) Logging ---
def load_model_data():
    global MODELS, SCALERS
    # Using logger.error for debug visibility in pytest -s
    logger.error(f"LIFESPAN/LOAD_MODELS: Current CWD for loading: {Path.cwd()}")
    logger.error(f"LIFESPAN/LOAD_MODELS: ARTIFACTS_DIR = {ARTIFACTS_DIR}, Exists: {ARTIFACTS_DIR.exists()}")
    MODELS.clear(); SCALERS.clear()
    
    model_files = list(ARTIFACTS_DIR.glob("*_model.joblib"))
    logger.error(f"LIFESPAN/LOAD_MODELS: Found model files: {[f.name for f in model_files]}")
    if not model_files: logger.error("LIFESPAN/LOAD_MODELS: No model files found in artifacts directory!"); return

    loaded_aliases = []
    for model_path in model_files:
        try:
            filename = model_path.name
            parts = filename.rsplit('_', 2)
            if len(parts) < 3 or parts[-1] != "model.joblib":
                logger.error(f"LIFESPAN/LOAD_MODELS: Skipping file with unexpected name format: {filename}"); continue
            
            task_type = parts[-2]
            model_base_alias = parts[0]
            full_alias = f"{model_base_alias}_{task_type}"

            if task_type not in ["classification", "regression"]:
                logger.error(f"LIFESPAN/LOAD_MODELS: Skipping model {filename} with unknown task_type '{task_type}'."); continue
            
            model = joblib.load(model_path)
            MODELS[full_alias] = ModelData(model=model, task_type=task_type, alias=full_alias)
            loaded_aliases.append(full_alias)
            logger.error(f"LIFESPAN/LOAD_MODELS: Successfully loaded model: {full_alias}")

            scaler_filename = f"{model_base_alias}_{task_type}_scaler.joblib"
            scaler_path = ARTIFACTS_DIR / scaler_filename
            if scaler_path.exists():
                SCALERS[full_alias] = joblib.load(scaler_path)
                logger.error(f"LIFESPAN/LOAD_MODELS: Successfully loaded scaler for: {full_alias}")
            else:
                logger.error(f"LIFESPAN/LOAD_MODELS: No scaler found for: {full_alias}")
        except Exception as e:
            logger.error(f"LIFESPAN/LOAD_MODELS: Error loading model or scaler for {model_path.name}: {e}", exc_info=True)
    
    if not loaded_aliases: logger.error("LIFESPAN/LOAD_MODELS: No models were successfully loaded into MODELS dict.")
    else: logger.error(f"LIFESPAN/LOAD_MODELS: Finished. MODELS populated with: {list(MODELS.keys())}")

def load_shared_artifacts():
    global TRAIN_COLS, LABEL_ENCODERS
    logger.error(f"LIFESPAN/LOAD_SHARED: Current CWD for loading: {Path.cwd()}") # DEBUG CWD
    logger.error(f"LIFESPAN/LOAD_SHARED: TRAIN_COLUMNS_FILE = {TRAIN_COLUMNS_FILE}, Exists: {TRAIN_COLUMNS_FILE.exists()}") # DEBUG
    TRAIN_COLS.clear(); LABEL_ENCODERS.clear()

    if TRAIN_COLUMNS_FILE.exists():
        try:
            with open(TRAIN_COLUMNS_FILE, 'r') as f: TRAIN_COLS = json.load(f)
            logger.error(f"LIFESPAN/LOAD_SHARED: Successfully loaded train_columns.json ({len(TRAIN_COLS)} cols)") # DEBUG
            if not TRAIN_COLS: logger.error("LIFESPAN/LOAD_SHARED: train_columns.json is empty!") # DEBUG
        except Exception as e:
            logger.error(f"LIFESPAN/LOAD_SHARED: Error loading {TRAIN_COLUMNS_FILE}: {e}", exc_info=True); TRAIN_COLS = []
    else:
        logger.error(f"LIFESPAN/LOAD_SHARED: {TRAIN_COLUMNS_FILE} not found."); TRAIN_COLS = []

    for task in ["classification", "regression"]:
        le_path = ARTIFACTS_DIR / f"label_encoder_{task}.joblib"
        logger.error(f"LIFESPAN/LOAD_SHARED: Checking label encoder: {le_path}, Exists: {le_path.exists()}") # DEBUG
        if le_path.exists():
            try:
                LABEL_ENCODERS[task] = joblib.load(le_path)
                logger.error(f"LIFESPAN/LOAD_SHARED: Successfully loaded label encoder for {task}") # DEBUG
            except Exception as e: logger.error(f"LIFESPAN/LOAD_SHARED: Error loading label encoder {le_path}: {e}", exc_info=True)
        else:
            logger.error(f"LIFESPAN/LOAD_SHARED: Label encoder for {task} not found at {le_path}.") # DEBUG
    logger.error(f"LIFESPAN/LOAD_SHARED: Finished. TRAIN_COLS length: {len(TRAIN_COLS)}, LABEL_ENCODERS: {list(LABEL_ENCODERS.keys())}") # DEBUG

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.error("LIFESPAN: Application startup sequence initiated (TestClient/Uvicorn).") # DEBUG
    logger.error(f"LIFESPAN: ARTIFACTS_DIR is {ARTIFACTS_DIR}") # DEBUG
    logger.error(f"LIFESPAN: TRAIN_COLUMNS_FILE is {TRAIN_COLUMNS_FILE}") # DEBUG
    
    load_model_data()
    load_shared_artifacts()
    
    logger.error(f"LIFESPAN: Loading sequence complete. MODELS keys: {list(MODELS.keys())}, TRAIN_COLS count: {len(TRAIN_COLS)}") # DEBUG
    available_models_str = ", ".join(sorted(list(MODELS.keys()))) if MODELS else "None"
    logger.error(f"API ready (from lifespan). Available models: [{available_models_str}]") # DEBUG
    yield
    logger.error("LIFESPAN: Application shutdown sequence initiated.") # DEBUG
    MODELS.clear(); SCALERS.clear(); TRAIN_COLS.clear(); LABEL_ENCODERS.clear()
    logger.error("LIFESPAN: Cleared loaded artifacts.") # DEBUG

app = FastAPI(lifespan=lifespan, title="ML Model Serving API", version="1.0.0")

@app.get("/health", summary="Check API Health", response_model=Dict[str, Any])
async def health_check_endpoint():
    return {"status": "ok", "available_models_count": len(MODELS), "loaded_model_aliases": sorted(list(MODELS.keys()))}

@app.get("/available_models", summary="List Available Model Aliases", response_model=Dict[str, List[str]])
async def get_available_models_endpoint():
    return {"available_model_aliases": sorted(list(MODELS.keys()))}

@app.post("/predict", response_model=PredictionResponse, summary="Get Prediction from a Model")
async def predict(req: PredictionRequest = Body(...)):
    model_alias_requested = req.model_alias.lower()
    logger.info(f"PREDICT endpoint: Received request for model_alias: {model_alias_requested}") # Keep this as INFO
    logger.info(f"PREDICT endpoint: Current TRAIN_COLS count: {len(TRAIN_COLS)}")
    logger.info(f"PREDICT endpoint: Current MODELS loaded: {list(MODELS.keys())}")

    if not TRAIN_COLS:
        logger.error("PREDICT endpoint: TRAIN_COLS is empty. Cannot process features.")
        raise HTTPException(status_code=500, detail="Training column info unavailable.")
    if model_alias_requested not in MODELS:
        logger.warning(f"PREDICT endpoint: Model alias '{model_alias_requested}' not found. Available: {list(MODELS.keys())}")
        raise HTTPException(status_code=404, detail=f"Model alias '{model_alias_requested}' not found. Available: {', '.join(MODELS.keys())}")

    model_data: ModelData = MODELS[model_alias_requested]
    model_to_use = model_data.model
    
    try:
        df_features = pd.DataFrame([req.features])
        cat_cols = df_features.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols: df_one_hot = pd.get_dummies(df_features, columns=cat_cols, drop_first=True, dummy_na=False)
        else: df_one_hot = df_features.copy()

        df_aligned = pd.DataFrame(columns=TRAIN_COLS)
        for col in TRAIN_COLS:
            if col in df_one_hot.columns: df_aligned[col] = df_one_hot[col]
            else: df_aligned[col] = 0
        
        try: df_aligned = df_aligned.astype(float)
        except ValueError as ve: logger.error(f"Type conversion error for aligned DataFrame: {ve}", exc_info=True); raise HTTPException(status_code=400, detail=f"Feature type conversion error: {ve}")

        df_for_predict = df_aligned.copy()
        if model_alias_requested in SCALERS:
            scaler_to_use = SCALERS[model_alias_requested]
            logger.info(f"PREDICT endpoint: Applying scaler for model: {model_alias_requested}")
            scaled_np = scaler_to_use.transform(df_for_predict) 
            df_for_predict = pd.DataFrame(scaled_np, columns=df_for_predict.columns)

        if model_alias_requested.startswith(("lgbm", "catboost")):
            logger.info(f"PREDICT endpoint: Sanitizing feature names for {model_alias_requested}")
            df_for_predict = sanitize_feature_names(df_for_predict)

        prediction_array = model_to_use.predict(df_for_predict)
        prediction_value = prediction_array[0]
        return PredictionResponse(model_alias_used=model_alias_requested, prediction=float(prediction_value))
    except Exception as e:
        logger.error(f"PREDICT endpoint: Error during prediction for model {model_alias_requested}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing prediction request: {str(e)}")

if __name__ == "__main__":
    # When running directly, set log level to INFO for model_serving_api
    # but the basicConfig above might already handle this.
    # If you want to override uvicorn's default logging, you can pass log_config.
    uvicorn.run("model_serving_api:app", host="0.0.0.0", port=8000, reload=True)