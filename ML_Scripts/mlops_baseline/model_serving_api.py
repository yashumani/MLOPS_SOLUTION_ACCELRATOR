"""
FastAPI inference service for the Model Garden.

 - Loads all available models and scalers from ./artifacts at startup.
 - Accepts JSON with model alias and features -> returns single-value prediction.
 - Applies consistent preprocessing including one-hot encoding and model-specific scaling.

Run locally:
    uvicorn model_serving_api:app --reload --port 8000
"""

from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict # <--- Import ConfigDict
import logging
from contextlib import asynccontextmanager # <--- Import asynccontextmanager

# ─── Setup Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TRAIN_COLS_PATH = ARTIFACTS_DIR / "train_columns.json"

# ─── Load artifacts once at startup (Global scope for lifespan access) ────────
AVAILABLE_MODELS = {}
AVAILABLE_SCALERS = {} # To store scalers, keyed by model_alias
TRAIN_COLS = None

logger.info(f"Looking for model artifacts in: {ARTIFACTS_DIR}")

if not ARTIFACTS_DIR.exists():
    logger.error(f"Artifacts directory not found: {ARTIFACTS_DIR}")
    # This would ideally prevent the app from starting or make it unhealthy
    # For now, it will log and continue, but endpoints might fail.
else:
    for model_path in ARTIFACTS_DIR.glob("*_model.joblib"):
        try:
            model_alias = model_path.name.replace("_model.joblib", "")
            logger.info(f"Loading model for alias: {model_alias} from {model_path}")
            AVAILABLE_MODELS[model_alias] = joblib.load(model_path)

            scaler_path = ARTIFACTS_DIR / f"{model_alias}_scaler.joblib"
            if scaler_path.exists():
                logger.info(f"Loading scaler for alias: {model_alias} from {scaler_path}")
                AVAILABLE_SCALERS[model_alias] = joblib.load(scaler_path)
            else:
                AVAILABLE_SCALERS[model_alias] = None
                logger.info(f"No scaler found for alias: {model_alias}")
        except Exception as e:
            logger.error(f"Error loading model or scaler for {model_path.name}: {e}", exc_info=True)

if not AVAILABLE_MODELS:
    logger.warning("🔴 No models were loaded from the artifacts directory.")

try:
    with open(TRAIN_COLS_PATH, 'r') as f:
        TRAIN_COLS = json.load(f)
    logger.info(f"Successfully loaded training columns from {TRAIN_COLS_PATH}")
except FileNotFoundError:
    logger.error(f"🔴 Missing train_columns.json at {TRAIN_COLS_PATH} – run train_pipeline.py first")
except json.JSONDecodeError:
    logger.error(f"🔴 Error decoding JSON from {TRAIN_COLS_PATH}.")

# ─── Lifespan Event Handler ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app_instance: FastAPI): # app_instance is the FastAPI app
    # Code to run on startup
    logger.info("Lifespan event: Application startup sequence initiated.")
    if not TRAIN_COLS:
        logger.critical("CRITICAL: TRAIN_COLS not loaded. Prediction endpoint will not function correctly.")
    if not AVAILABLE_MODELS:
        logger.warning("WARNING: No models loaded at startup. Prediction endpoint will not be able to serve any models.")
    else:
        logger.info(f"API ready. Available models: {list(AVAILABLE_MODELS.keys())}")
    
    yield # This is where the application runs

    # Code to run on shutdown (if any)
    logger.info("Lifespan event: Application shutdown sequence initiated.")
    # Example: print("Lifespan shutdown: Cleaning up resources...")

# ─── API setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Model Garden Prediction Service",
    description="Serves predictions from various models trained by the AutoML pipeline.",
    version="1.0.0",
    lifespan=lifespan # <--- Use the lifespan manager
)

class PredictionRequest(BaseModel):
    model_alias: str
    features: dict

    # Fix for Pydantic "model_" namespace warning
    model_config = ConfigDict(protected_namespaces=())


class PredictionResponse(BaseModel):
    model_alias_used: str
    prediction: float

    # Fix for Pydantic "model_" namespace warning
    model_config = ConfigDict(protected_namespaces=())


@app.get("/health", summary="Check API Health")
def health():
    """Returns the operational status of the API and number of loaded models."""
    return {"status": "ok", "available_models_count": len(AVAILABLE_MODELS)}

@app.get("/available_models", summary="List Available Models")
def get_available_models():
    """Returns a list of model aliases that can be used for prediction."""
    if not AVAILABLE_MODELS:
        return {"message": "No models currently loaded or available."}
    return {"available_model_aliases": list(AVAILABLE_MODELS.keys())}

@app.post("/predict", response_model=PredictionResponse, summary="Get Model Prediction")
def predict(req: PredictionRequest):
    """
    Receives input features and a model alias, returns the model's prediction.
    Preprocessing (one-hot encoding, scaling if applicable, column alignment)
    is performed automatically based on how the specified model was trained.
    """
    if not TRAIN_COLS:
        logger.error("Prediction failed: TRAIN_COLS is not available (was not loaded at startup).")
        raise HTTPException(status_code=500, detail="Server configuration error: Training columns not loaded.")

    model_alias_requested = req.model_alias.lower()

    if model_alias_requested not in AVAILABLE_MODELS:
        logger.warning(f"Prediction failed: Requested model alias '{model_alias_requested}' not found. Available: {list(AVAILABLE_MODELS.keys())}")
        raise HTTPException(
            status_code=404,
            detail=f"Model alias '{model_alias_requested}' not found. Available models: {', '.join(AVAILABLE_MODELS.keys())}"
        )

    model_to_use = AVAILABLE_MODELS[model_alias_requested]
    scaler_to_use = AVAILABLE_SCALERS.get(model_alias_requested)

    try:
        df = pd.DataFrame([req.features])
        logger.debug(f"Raw input DataFrame for {model_alias_requested}: \n{df.head()}")

        index_cols_to_drop = [c for c in ["index", "Index", "index_col_for_dfs"] if c in df.columns]
        if index_cols_to_drop:
            df = df.drop(columns=index_cols_to_drop)
            logger.debug(f"DataFrame after dropping index columns: \n{df.head() if not df.empty else 'DataFrame is empty'}")

        df_encoded = pd.get_dummies(df, dummy_na=False)
        logger.debug(f"DataFrame after one-hot encoding: \n{df_encoded.head() if not df_encoded.empty else 'DataFrame is empty'}")
        
        # Align columns: Create a DataFrame with all TRAIN_COLS, then fill
        # This ensures correct order and handles missing/extra columns from input
        df_aligned = pd.DataFrame(columns=TRAIN_COLS, index=df_encoded.index, dtype=float) # Initialize with float for numeric
        
        common_cols = df_aligned.columns.intersection(df_encoded.columns)
        df_aligned[common_cols] = df_encoded[common_cols]
        df_aligned = df_aligned.fillna(0) # Fill any columns not in df_encoded (newly added from TRAIN_COLS) with 0

        # Ensure final df only has columns from TRAIN_COLS and in that order
        df_processed = df_aligned[TRAIN_COLS]
        logger.debug(f"DataFrame after aligning with TRAIN_COLS: \n{df_processed.head() if not df_processed.empty else 'DataFrame is empty'}")
        
        if scaler_to_use:
            logger.info(f"Applying scaler for model: {model_alias_requested}")
            df_scaled_np = scaler_to_use.transform(df_processed)
            df_processed = pd.DataFrame(df_scaled_np, columns=df_processed.columns, index=df_processed.index)
            logger.debug(f"DataFrame after scaling for {model_alias_requested}: \n{df_processed.head() if not df_processed.empty else 'DataFrame is empty'}")
        
        logger.info(f"Making prediction with model: {model_alias_requested}")
        prediction_array = model_to_use.predict(df_processed)
        prediction_value = float(prediction_array[0])

        logger.info(f"Prediction successful for {model_alias_requested}: {prediction_value}")
        return PredictionResponse(model_alias_used=model_alias_requested, prediction=prediction_value)

    except HTTPException:
        raise 
    except Exception as e:
        logger.error(f"Error during prediction for model {model_alias_requested}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing prediction request: {str(e)}")

# To run (save as model_serving_api.py):
# uvicorn model_serving_api:app --reload --port 8000