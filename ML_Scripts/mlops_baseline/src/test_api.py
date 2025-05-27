import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import logging

# Configure basic logging for the test script itself
logging.basicConfig(level=logging.INFO) # Changed to INFO for less noise, can be DEBUG
logger = logging.getLogger(__name__)

# Adjust path to import app from model_serving_api.py
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_serving_api import app, ARTIFACTS_DIR as api_artifacts_dir, TRAIN_COLUMNS_FILE as api_train_cols_file
    # Also import the paths used by the API to check them directly
except ImportError as e:
    logger.error(f"Error importing app from model_serving_api: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    raise

client = TestClient(app)

# === Sample valid features for the Titanic dataset ===
SAMPLE_VALID_FEATURES = {
    "Pclass": 3, "Sex": "male", "Age": 29.7,
    "SibSp": 0, "Parch": 0, "Fare": 7.90, "Embarked": "S"
}

# === Expected model aliases for classification models ===
EXPECTED_MODEL_ALIASES = sorted([
    'catboostclassifier_classification',
    'kneighborsclassifier_classification',
    'lgbmclassifier_classification',
    'logisticregression_classification',
    'randomforestclassifier_classification',
    'svc_classification',
    'xgbclassifier_classification'
])
EXPECTED_MODEL_COUNT = len(EXPECTED_MODEL_ALIASES)

# Helper to print response details on failure
def log_response_details(response, model_alias_to_test="N/A"):
    logger.error(f"Test failed for model: {model_alias_to_test}")
    logger.error(f"Response Status Code: {response.status_code}")
    try: logger.error(f"Response JSON: {response.json()}")
    except Exception: logger.error(f"Response Text: {response.text}")

# === NEW DIAGNOSTIC TEST ===
def test_artifact_paths_exist():
    logger.info(f"Checking artifact paths from model_serving_api.py context:")
    logger.info(f"API's ARTIFACTS_DIR: {api_artifacts_dir}")
    logger.info(f"API's TRAIN_COLUMNS_FILE: {api_train_cols_file}")
    
    assert api_artifacts_dir.exists(), f"ARTIFACTS_DIR defined in API does not exist: {api_artifacts_dir}"
    assert api_train_cols_file.exists(), f"TRAIN_COLUMNS_FILE defined in API does not exist: {api_train_cols_file}"
    
    # Check for at least one model file
    model_files_found = list(api_artifacts_dir.glob("*_classification_model.joblib"))
    logger.info(f"Found classification model files in API's ARTIFACTS_DIR: {model_files_found}")
    assert len(model_files_found) > 0, f"No '*_classification_model.joblib' files found in {api_artifacts_dir}"
    assert len(model_files_found) == EXPECTED_MODEL_COUNT, \
        f"Expected {EXPECTED_MODEL_COUNT} model files, found {len(model_files_found)} in {api_artifacts_dir}"
# === END NEW DIAGNOSTIC TEST ===

def test_health_check():
    logger.info("Running test_health_check")
    response = client.get("/health")
    if response.status_code != 200: log_response_details(response)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "ok"
    assert "available_models_count" in response_json
    assert response_json["available_models_count"] == EXPECTED_MODEL_COUNT, \
        f"Expected {EXPECTED_MODEL_COUNT} models, API reported {response_json['available_models_count']}"

def test_available_models():
    logger.info("Running test_available_models")
    response = client.get("/available_models")
    if response.status_code != 200: log_response_details(response)
    assert response.status_code == 200
    response_json = response.json()
    assert "available_model_aliases" in response_json
    assert sorted(response_json["available_model_aliases"]) == EXPECTED_MODEL_ALIASES, \
        f"API aliases: {sorted(response_json['available_model_aliases'])}, Expected: {EXPECTED_MODEL_ALIASES}"

@pytest.mark.parametrize("model_alias_to_test", EXPECTED_MODEL_ALIASES)
def test_predict_valid_models(model_alias_to_test: str):
    logger.info(f"Running test_predict_valid_models for: {model_alias_to_test}")
    payload = {"model_alias": model_alias_to_test, "features": SAMPLE_VALID_FEATURES}
    response = client.post("/predict", json=payload)
    if response.status_code != 200: log_response_details(response, model_alias_to_test)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["model_alias_used"] == model_alias_to_test
    assert "prediction" in response_json; assert isinstance(response_json["prediction"], float)
    assert response_json["prediction"] in [0.0, 1.0]

def test_predict_invalid_model_alias():
    logger.info("Running test_predict_invalid_model_alias")
    payload = {"model_alias": "non_existent_model_xyz", "features": SAMPLE_VALID_FEATURES}
    response = client.post("/predict", json=payload)
    # This might now return 404 if TRAIN_COLS loads, or 500 if TRAIN_COLS is still the primary issue
    if response.status_code not in [404, 500]: log_response_details(response, "non_existent_model_xyz")
    
    if response.status_code == 500 and "Training column info unavailable" in response.text:
        pytest.fail("Predict invalid alias failed because training columns were not loaded first.")
    
    assert response.status_code == 404
    response_json = response.json()
    assert "detail" in response_json
    assert "Model alias 'non_existent_model_xyz' not found" in response_json["detail"]

def test_predict_missing_features_key():
    logger.info("Running test_predict_missing_features_key")
    payload = {"model_alias": EXPECTED_MODEL_ALIASES[0] if EXPECTED_MODEL_ALIASES else "any_model"}
    response = client.post("/predict", json=payload)
    if response.status_code != 422: log_response_details(response, payload["model_alias"])
    assert response.status_code == 422

def test_predict_features_not_a_dict():
    logger.info("Running test_predict_features_not_a_dict")
    payload = {"model_alias": EXPECTED_MODEL_ALIASES[0] if EXPECTED_MODEL_ALIASES else "any_model", "features": "not_a_dict"}
    response = client.post("/predict", json=payload)
    if response.status_code != 422: log_response_details(response, payload["model_alias"])
    assert response.status_code == 422

def test_predict_missing_required_feature_in_dict():
    logger.info("Running test_predict_missing_required_feature_in_dict")
    incomplete_features = SAMPLE_VALID_FEATURES.copy()
    if "Age" in incomplete_features: del incomplete_features["Age"]
    
    model_to_test_with = "randomforestclassifier_classification"
    if model_to_test_with not in EXPECTED_MODEL_ALIASES and EXPECTED_MODEL_ALIASES:
        model_to_test_with = EXPECTED_MODEL_ALIASES[0]

    payload = {"model_alias": model_to_test_with, "features": incomplete_features}
    response = client.post("/predict", json=payload)
    if response.status_code != 200: log_response_details(response, model_to_test_with)
    
    if response.status_code == 500 and "Training column info unavailable" in response.text:
        pytest.fail("Predict missing feature failed because training columns were not loaded first.")
        
    assert response.status_code == 200
    response_json = response.json(); assert "prediction" in response_json
    assert isinstance(response_json["prediction"], float)
    assert response_json["prediction"] in [0.0, 1.0]

def test_predict_feature_wrong_type_in_dict():
    logger.info("Running test_predict_feature_wrong_type_in_dict")
    features_with_wrong_type = SAMPLE_VALID_FEATURES.copy()
    features_with_wrong_type["Pclass"] = "should_be_numeric_not_string"
    
    model_to_test_with = "randomforestclassifier_classification"
    if model_to_test_with not in EXPECTED_MODEL_ALIASES and EXPECTED_MODEL_ALIASES:
        model_to_test_with = EXPECTED_MODEL_ALIASES[0]

    payload = {"model_alias": model_to_test_with, "features": features_with_wrong_type}
    response = client.post("/predict", json=payload)
    if response.status_code != 200: log_response_details(response, model_to_test_with)

    if response.status_code == 500 and "Training column info unavailable" in response.text:
        pytest.fail("Predict wrong type failed because training columns were not loaded first.")

    assert response.status_code == 200
    response_json = response.json(); assert "prediction" in response_json
    assert isinstance(response_json["prediction"], float)
    assert response_json["prediction"] in [0.0, 1.0]