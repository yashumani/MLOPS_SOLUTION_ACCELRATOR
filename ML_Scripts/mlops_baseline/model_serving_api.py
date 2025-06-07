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

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    model_alias: str = Field(..., json_schema_extra={"example": "kmeans_clustering"}) # Example for clustering
    features: Dict[str, Union[str, float, int, None]] = Field(
        ...,
        json_schema_extra={"example": {"BALANCE": 40.90, "PURCHASES": 95.4, "CREDIT_LIMIT": 1000.0}} # Example for credit card
    )
    class Config:
        protected_namespaces = ()

class PredictionResponse(BaseModel):
    model_alias_used: str
    prediction: float # For clustering, this will be the cluster ID
    class Config:
        protected_namespaces = ()

class ModelData:
    def __init__(self, model: Any, task_type: str, alias: str, clustering_features: Optional[List[str]] = None):
        self.model = model
        self.task_type = task_type
        self.alias = alias
        self.clustering_features = clustering_features

# --- Sanitization Function ---
def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in df_copy.columns]
    df_copy.columns = [re.sub(r"_+", "_", col) for col in df_copy.columns]
    df_copy.columns = [col.strip("_") for col in df_copy.columns]
    cols = pd.Series(df.columns)
    if cols.duplicated().any():
        logger.warning(f"Duplicate column names found after sanitization, attempting to rename: {cols[cols.duplicated()].unique().tolist()}")
        # A more robust way to handle duplicates created by sanitization
        seen = {}
        new_columns = []
        for col_name in df.columns:
            if col_name in seen:
                seen[col_name] += 1
                new_columns.append(f"{col_name}_{seen[col_name]}")
            else:
                seen[col_name] = 0
                new_columns.append(col_name)
        df_copy.columns = new_columns
    return df_copy

# --- Loading Functions ---
def load_model_data():
    global MODELS, SCALERS
    logger.info(f"LIFESPAN: Current CWD for loading models: {Path.cwd()}")
    logger.info(f"LIFESPAN: Looking for model artifacts in: {ARTIFACTS_DIR}, Exists: {ARTIFACTS_DIR.exists()}")
    MODELS.clear(); SCALERS.clear()

    model_files = list(ARTIFACTS_DIR.glob("*_model.joblib"))
    logger.info(f"LIFESPAN: Found model files: {[f.name for f in model_files]}" if model_files else "LIFESPAN: No model files found.")
    if not model_files:
        logger.warning("LIFESPAN: No model files (*_model.joblib) found in artifacts directory.")
        return

    loaded_aliases = []
    for model_path in model_files:
        try:
            filename = model_path.name
            parts = filename.rsplit('_', 2)
            if len(parts) < 3 or parts[-1] != "model.joblib":
                logger.warning(f"LIFESPAN: Skipping file with unexpected name format: {filename}"); continue
            
            task_type = parts[-2]
            model_base_name = parts[0] # e.g., "kmeans" from "kmeans_clustering_model.joblib"
            full_alias = f"{model_base_name}_{task_type}" # e.g., "kmeans_clustering"

            if task_type not in ["classification", "regression", "clustering"]:
                logger.warning(f"LIFESPAN: Skipping model {filename} with unknown task_type '{task_type}'."); continue
            
            model = joblib.load(model_path)
            
            clustering_features_list = None
            if task_type == "clustering":
                # Adjusted to match the filename saved by clustering_trainer.py
                clustering_features_filename = f"{model_base_name}_clustering_features.json"
                clustering_features_filepath = ARTIFACTS_DIR / clustering_features_filename
                if clustering_features_filepath.exists():
                    with open(clustering_features_filepath, 'r') as f:
                        clustering_features_list = json.load(f)
                    logger.info(f"LIFESPAN: Loaded clustering features for {full_alias} from {clustering_features_filename}")
                else:
                    logger.warning(f"LIFESPAN: Clustering features file '{clustering_features_filename}' not found for {full_alias}. This model might require specific features for prediction.")
            
            MODELS[full_alias] = ModelData(model=model, task_type=task_type, alias=full_alias, clustering_features=clustering_features_list)
            loaded_aliases.append(full_alias)
            logger.info(f"LIFESPAN: Successfully loaded model: {full_alias}")

            scaler_filename = f"{model_base_name}_{task_type}_scaler.joblib"
            scaler_path = ARTIFACTS_DIR / scaler_filename
            if scaler_path.exists():
                SCALERS[full_alias] = joblib.load(scaler_path)
                logger.info(f"LIFESPAN: Successfully loaded scaler for: {full_alias}")
            else:
                logger.info(f"LIFESPAN: No scaler found for: {full_alias}")
        except Exception as e:
            logger.error(f"LIFESPAN: Error loading model or scaler for {model_path.name}: {e}", exc_info=True)
    
    if not loaded_aliases: logger.warning("LIFESPAN: No models were successfully loaded into MODELS dict.")
    else: logger.info(f"LIFESPAN: Finished loading models. MODELS populated with: {list(MODELS.keys())}")


def load_shared_artifacts():
    global TRAIN_COLS, LABEL_ENCODERS
    logger.info(f"LIFESPAN: Current CWD for loading shared artifacts: {Path.cwd()}")
    logger.info(f"LIFESPAN: TRAIN_COLUMNS_FILE = {TRAIN_COLUMNS_FILE}, Exists: {TRAIN_COLUMNS_FILE.exists()}")
    TRAIN_COLS = []
    LABEL_ENCODERS.clear()

    if TRAIN_COLUMNS_FILE.exists():
        try:
            with open(TRAIN_COLUMNS_FILE, 'r') as f: TRAIN_COLS = json.load(f)
            logger.info(f"LIFESPAN: Successfully loaded train_columns.json ({len(TRAIN_COLS)} cols)")
            if not TRAIN_COLS: logger.warning("LIFESPAN: train_columns.json is empty!")
        except Exception as e:
            logger.error(f"LIFESPAN: Error loading {TRAIN_COLUMNS_FILE}: {e}", exc_info=True); TRAIN_COLS = []
    else:
        logger.warning(f"LIFESPAN: {TRAIN_COLUMNS_FILE} not found."); TRAIN_COLS = []

    for task in ["classification", "regression"]: 
        le_path = ARTIFACTS_DIR / f"label_encoder_{task}.joblib"
        logger.info(f"LIFESPAN: Checking label encoder: {le_path}, Exists: {le_path.exists()}")
        if le_path.exists():
            try:
                LABEL_ENCODERS[task] = joblib.load(le_path)
                logger.info(f"LIFESPAN: Successfully loaded label encoder for {task}")
            except Exception as e: logger.error(f"LIFESPAN: Error loading label encoder {le_path}: {e}", exc_info=True)
        else:
            logger.info(f"LIFESPAN: Label encoder for {task} task not found at {le_path}.")
    logger.info(f"LIFESPAN: Finished loading shared artifacts. TRAIN_COLS length: {len(TRAIN_COLS)}, LABEL_ENCODERS: {list(LABEL_ENCODERS.keys())}")


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    logger.info("LIFESPAN: Application startup sequence initiated.")
    load_model_data()
    load_shared_artifacts()
    logger.info(f"LIFESPAN: Loading sequence complete. MODELS keys: {list(MODELS.keys())}, TRAIN_COLS count: {len(TRAIN_COLS)}")
    available_models_str = ", ".join(sorted(list(MODELS.keys()))) if MODELS else "None"
    logger.info(f"API ready. Available models: [{available_models_str}]")
    yield
    logger.info("LIFESPAN: Application shutdown sequence initiated.")
    MODELS.clear(); SCALERS.clear(); TRAIN_COLS.clear(); LABEL_ENCODERS.clear()
    logger.info("LIFESPAN: Cleared loaded artifacts.")

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
    logger.info(f"PREDICT: Request for model_alias: {model_alias_requested}")

    if not TRAIN_COLS:
        logger.error("PREDICT: TRAIN_COLS is empty. Cannot process features.")
        raise HTTPException(status_code=500, detail="Training column information not available. API may not have initialized correctly.")
    if model_alias_requested not in MODELS:
        logger.warning(f"PREDICT: Model alias '{model_alias_requested}' not found. Available: {list(MODELS.keys())}")
        raise HTTPException(status_code=404, detail=f"Model alias '{model_alias_requested}' not found. Available models: {', '.join(MODELS.keys())}")

    model_data: ModelData = MODELS[model_alias_requested]
    model_to_use = model_data.model
    
    try:
        # 1. Create DataFrame from incoming features
        df_features = pd.DataFrame([req.features])
        logger.debug(f"PREDICT: Input features DataFrame shape: {df_features.shape}")

        # 2. One-hot encode categorical features based on input data
        cat_cols = df_features.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            logger.debug(f"PREDICT: Applying OneHotEncoding to: {cat_cols}")
            df_one_hot = pd.get_dummies(df_features, columns=cat_cols, drop_first=True, dummy_na=False)
        else: 
            df_one_hot = df_features.copy()

        # 3. Align columns with TRAIN_COLS (master list of features from prep_pipeline)
        # This df_aligned will be used for scaling IF the model is not clustering
        # or if clustering model doesn't have specific features defined.
        df_aligned = pd.DataFrame(columns=TRAIN_COLS)
        for col in TRAIN_COLS:
            if col in df_one_hot.columns: 
                df_aligned[col] = df_one_hot[col]
            else: 
                df_aligned[col] = 0.0 # Fill missing columns with 0
        
        try: 
            df_aligned = df_aligned.astype(float) # Convert all to float before further processing
        except ValueError as ve: 
            logger.error(f"PREDICT: Type conversion error for aligned DataFrame: {ve}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Feature type conversion error: {ve}")
        logger.debug(f"PREDICT: DataFrame aligned to TRAIN_COLS and cast to float. Shape: {df_aligned.shape}")

        # Prepare the final DataFrame for the model
        df_for_predict = df_aligned.copy()

        # 4. Feature Selection specifically for Clustering Models (if applicable)
        # This should happen BEFORE scaling if the scaler was fit on these specific features.
        if model_data.task_type == "clustering" and model_data.clustering_features:
            logger.info(f"PREDICT: Model {model_alias_requested} is clustering. Selecting specific features for model input.")
            try:
                # Ensure only existing columns are selected to avoid KeyErrors if TRAIN_COLS was too broad
                cols_to_select = [col for col in model_data.clustering_features if col in df_for_predict.columns]
                if len(cols_to_select) != len(model_data.clustering_features):
                    missing_from_aligned = list(set(model_data.clustering_features) - set(cols_to_select))
                    logger.warning(f"PREDICT: Some clustering_features for {model_alias_requested} not found in aligned data: {missing_from_aligned}. Using available ones.")
                
                df_for_predict = df_for_predict[cols_to_select]
                logger.info(f"PREDICT: Selected {len(df_for_predict.columns)} features for clustering model {model_alias_requested}.")
            except KeyError as ke:
                logger.error(f"PREDICT: Key error when selecting clustering features for {model_alias_requested}. Missing from df_aligned: {ke}. Expected: {model_data.clustering_features}. Available: {df_aligned.columns.tolist()}")
                raise HTTPException(status_code=400, detail=f"Input data missing required features for clustering model {model_alias_requested} after initial processing.")
        
        # 5. Apply scaling if a scaler exists for this model
        if model_alias_requested in SCALERS:
            scaler_to_use = SCALERS[model_alias_requested]
            logger.info(f"PREDICT: Applying scaler for model: {model_alias_requested}. Features for scaling: {df_for_predict.columns.tolist()}")
            
            # Scaler was fit on the same set of features (and order) as df_for_predict now has.
            # For clustering, scaler was fit *after* CUST_ID (etc.) was dropped.
            # For supervised, scaler was fit on all X_train features.
            if not df_for_predict.empty:
                scaled_np = scaler_to_use.transform(df_for_predict) 
                df_for_predict = pd.DataFrame(scaled_np, columns=df_for_predict.columns, index=df_for_predict.index)
                logger.info(f"PREDICT: Scaling complete for {model_alias_requested}.")

                # Handle potential NaNs/Infs produced by scaler
                if np.any(np.isinf(df_for_predict.values)) or np.any(np.isnan(df_for_predict.values)):
                    logger.warning(f"PREDICT: NaNs or Infs detected after scaling for {model_alias_requested}. Replacing with 0.")
                    df_for_predict = df_for_predict.replace([np.inf, -np.inf], np.nan)
                    df_for_predict = df_for_predict.fillna(0)
            else:
                logger.warning(f"PREDICT: df_for_predict is empty before scaling for {model_alias_requested}. Skipping scaling.")
        
        # 6. Feature name sanitization for relevant models (LGBM, CatBoost)
        # This should happen on df_for_predict which now has the final set of features for the model
        if model_data.model.__class__.__name__.startswith(("LGBM", "CatBoost")):
            logger.info(f"PREDICT: Sanitizing feature names for {model_alias_requested} (model type: {model_data.model.__class__.__name__})")
            df_for_predict = sanitize_feature_names(df_for_predict)

        # 7. Make prediction
        if df_for_predict.empty:
            logger.error(f"PREDICT: df_for_predict is empty before prediction for {model_alias_requested}. Cannot make prediction.")
            raise HTTPException(status_code=400, detail=f"Feature processing resulted in empty data for model {model_alias_requested}.")

        logger.info(f"PREDICT: Making prediction with {model_alias_requested} on data with shape {df_for_predict.shape} and columns {df_for_predict.columns.tolist()}")
        
        # Handle models that might not have a 'predict' method for new instances (like some clustering algos if not wrapped by pyfunc)
        if model_data.task_type == "clustering" and not hasattr(model_to_use, 'predict'):
            if hasattr(model_to_use, 'fit_predict') and hasattr(model_to_use, 'labels_'):
                # This is a tricky case for new, unseen data.
                # For now, we'll assume predict is available or model is not suitable for this type of API call.
                logger.error(f"PREDICT: Model {model_alias_requested} is a clustering model without a direct 'predict' method for new instances. This scenario needs specific handling.")
                raise HTTPException(status_code=501, detail=f"Prediction on new instances not directly supported for {model_alias_requested}.")
            else:
                logger.error(f"PREDICT: Clustering model {model_alias_requested} has no standard predict or fit_predict method.")
                raise HTTPException(status_code=501, detail=f"Unsupported clustering model type for prediction: {model_alias_requested}.")

        prediction_array = model_to_use.predict(df_for_predict)
        prediction_value = prediction_array[0] 
        logger.info(f"PREDICT: Raw prediction for {model_alias_requested}: {prediction_value}")
        
        return PredictionResponse(
            model_alias_used=model_alias_requested,
            prediction=float(prediction_value) 
        )
    except Exception as e:
        logger.error(f"PREDICT: Error during prediction process for model {model_alias_requested}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing prediction request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("model_serving_api:app", host="0.0.0.0", port=8000, reload=True)