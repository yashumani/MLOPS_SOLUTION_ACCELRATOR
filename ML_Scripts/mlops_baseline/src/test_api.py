import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import logging
from typing import Dict, Any, List

# --- Path Setup ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from model_serving_api import app
except ImportError as e:
    print(f"Error during initial imports: {e}")
    sys.exit(1)

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pytest Fixture for TestClient ---
@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance for the API tests."""
    logger.info("Setting up TestClient for API testing.")
    with TestClient(app) as c:
        yield c
    logger.info("TestClient torn down.")

# --- Configuration for Clustering Test ---

# Sample features from the Credit Card dataset (raw format, before any preprocessing)
SAMPLE_VALID_FEATURES = {
    "BALANCE": 1500.50,
    "BALANCE_FREQUENCY": 1.0,
    "PURCHASES": 850.75,
    "ONEOFF_PURCHASES": 400.0,
    "INSTALLMENTS_PURCHASES": 450.75,
    "CASH_ADVANCE": 0.0,
    "PURCHASES_FREQUENCY": 0.8,
    "ONEOFF_PURCHASES_FREQUENCY": 0.3,
    "PURCHASES_INSTALLMENTS_FREQUENCY": 0.5,
    "CASH_ADVANCE_FREQUENCY": 0.0,
    "CASH_ADVANCE_TRX": 0,
    "PURCHASES_TRX": 15,
    "CREDIT_LIMIT": 4000.0,
    "PAYMENTS": 1000.0,
    "MINIMUM_PAYMENTS": 250.0,
    "PRC_FULL_PAYMENT": 0.1,
    "TENURE": 12
    # NOTE: CUST_ID is intentionally omitted as it's not a feature the user provides.
}

# Expected model aliases based on your successful clustering run
EXPECTED_MODEL_ALIASES = sorted([
    'kmeans_clustering',
    'dbscan_clustering',
    'agglomerativeclustering_clustering',
    'gaussianmixture_clustering'
])
EXPECTED_MODEL_COUNT = len(EXPECTED_MODEL_ALIASES)

# --- Helper Function for Logging Failures ---
def log_response_details(response, model_alias="N/A"):
    logger.error(f"Test failed for model/alias: {model_alias}")
    logger.error(f"Response Status Code: {response.status_code}")
    try:
        logger.error(f"Response JSON: {response.json()}")
    except Exception:
        logger.error(f"Response Text: {response.text}")

# --- API Tests ---

def test_health_check(client: TestClient):
    """Test the /health endpoint for a 200 OK and correct model count."""
    logger.info("Running test_health_check...")
    response = client.get("/health")
    if response.status_code != 200:
        log_response_details(response)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "ok"
    assert "available_models_count" in response_json
    assert response_json["available_models_count"] == EXPECTED_MODEL_COUNT, \
        f"Expected {EXPECTED_MODEL_COUNT} models, but API reported {response_json['available_models_count']}"

def test_available_models(client: TestClient):
    """Test the /available_models endpoint to ensure it returns the expected list of models."""
    logger.info("Running test_available_models...")
    response = client.get("/available_models")
    if response.status_code != 200:
        log_response_details(response)
    assert response.status_code == 200
    response_json = response.json()
    assert "available_model_aliases" in response_json
    assert sorted(response_json["available_model_aliases"]) == EXPECTED_MODEL_ALIASES, \
        f"API aliases do not match expected. API: {sorted(response_json['available_model_aliases'])}, Expected: {EXPECTED_MODEL_ALIASES}"

@pytest.mark.parametrize("model_alias_to_test", EXPECTED_MODEL_ALIASES)
def test_predict_valid_models(model_alias_to_test: str, client: TestClient):
    """Test the /predict endpoint for all loaded clustering models with valid data."""
    logger.info(f"Running test_predict_valid_models for: {model_alias_to_test}")

    # Skip models that do not have a .predict() method for new instances
    if "dbscan" in model_alias_to_test or "agglomerative" in model_alias_to_test:
        pytest.skip(f"Skipping prediction test for {model_alias_to_test} as it may not support predicting on new single instances.")

    payload = {"model_alias": model_alias_to_test, "features": SAMPLE_VALID_FEATURES}
    response = client.post("/predict", json=payload)
    
    if response.status_code != 200:
        log_response_details(response, model_alias_to_test)
    
    assert response.status_code == 200, f"Prediction failed for model: {model_alias_to_test}"
    response_json = response.json()
    
    assert response_json["model_alias_used"] == model_alias_to_test
    assert "prediction" in response_json
    
    prediction = response_json["prediction"]
    assert isinstance(prediction, float)
    # For clustering, the prediction is a cluster ID, which should be a whole number
    assert prediction == int(prediction), f"Prediction for clustering should be a whole number, but got {prediction}"
    assert prediction >= 0, f"Cluster ID should be non-negative, but got {prediction}"
    logger.info(f"Successfully received cluster prediction '{int(prediction)}' for model '{model_alias_to_test}'")

def test_predict_invalid_model_alias(client: TestClient):
    """Test that requesting a non-existent model returns a 404 error."""
    logger.info("Running test_predict_invalid_model_alias...")
    payload = {"model_alias": "non_existent_model_xyz", "features": SAMPLE_VALID_FEATURES}
    response = client.post("/predict", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json().get("detail", "").lower()