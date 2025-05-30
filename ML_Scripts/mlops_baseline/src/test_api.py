import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import logging

# Configure basic logging for the test script itself
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

# Adjust path to import app from model_serving_api.py
try:
    # Assuming test_api.py is in src/ and model_serving_api.py is in the project root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_serving_api import app, ARTIFACTS_DIR as api_artifacts_dir, TRAIN_COLUMNS_FILE as api_train_cols_file
except ImportError as e:
    logger.error(f"Error importing app from model_serving_api: {e}")
    logger.error(f"Current sys.path: {sys.path}")
    raise

# --- Pytest Fixture for TestClient ---
@pytest.fixture(scope="module")
def client():
    logger.info("Creating TestClient instance using fixture.")
    with TestClient(app) as c:
        logger.info("TestClient instance created and lifespan startup should have run.")
        yield c
    logger.info("TestClient lifespan shutdown should have run.")

# === Sample valid features for the Titanic dataset ===
SAMPLE_VALID_FEATURES = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 29.7, 
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.90,
    "Embarked": "S"
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
    logger.error(f"Test failed for model/alias: {model_alias_to_test}")
    logger.error(f"Response Status Code: {response.status_code}")
    try:
        logger.error(f"Response JSON: {response.json()}")
    except Exception:
        logger.error(f"Response Text: {response.text}")

# === DIAGNOSTIC TEST (can be kept or removed) ===
def test_artifact_paths_exist():
    logger.info("Running test_artifact_paths_exist (checks paths from model_serving_api.py's perspective)")
    logger.info(f"API's ARTIFACTS_DIR: {api_artifacts_dir}")
    logger.info(f"API's TRAIN_COLUMNS_FILE: {api_train_cols_file}")
    
    assert api_artifacts_dir.exists(), f"ARTIFACTS_DIR defined in API does not exist: {api_artifacts_dir}"
    assert api_train_cols_file.exists(), f"TRAIN_COLUMNS_FILE defined in API does not exist: {api_train_cols_file}"
    
    model_files_found = list(api_artifacts_dir.glob("*_classification_model.joblib"))
    logger.info(f"Found classification model files in API's ARTIFACTS_DIR: {[f.name for f in model_files_found]}")
    assert len(model_files_found) > 0, f"No '*_classification_model.joblib' files found in {api_artifacts_dir}"
    assert len(model_files_found) == EXPECTED_MODEL_COUNT, \
        f"Expected {EXPECTED_MODEL_COUNT} model files, found {len(model_files_found)} in {api_artifacts_dir}"

# === REGULAR API TESTS ===
def test_health_check(client):
    logger.info("Running test_health_check")
    response = client.get("/health")
    if response.status_code != 200: log_response_details(response)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "ok"
    assert "available_models_count" in response_json
    assert response_json["available_models_count"] == EXPECTED_MODEL_COUNT, \
        f"Expected {EXPECTED_MODEL_COUNT} models, API reported {response_json['available_models_count']}"

def test_available_models(client):
    logger.info("Running test_available_models")
    response = client.get("/available_models")
    if response.status_code != 200: log_response_details(response)
    assert response.status_code == 200
    response_json = response.json()
    assert "available_model_aliases" in response_json
    assert sorted(response_json["available_model_aliases"]) == EXPECTED_MODEL_ALIASES, \
        f"API aliases: {sorted(response_json['available_model_aliases'])}, Expected: {EXPECTED_MODEL_ALIASES}"

@pytest.mark.parametrize("model_alias_to_test", EXPECTED_MODEL_ALIASES)
def test_predict_valid_models(model_alias_to_test: str, client):
    logger.info(f"Running test_predict_valid_models for: {model_alias_to_test}")
    payload = {"model_alias": model_alias_to_test, "features": SAMPLE_VALID_FEATURES}
    response = client.post("/predict", json=payload)
    if response.status_code != 200: log_response_details(response, model_alias_to_test)
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["model_alias_used"] == model_alias_to_test
    assert "prediction" in response_json; assert isinstance(response_json["prediction"], float)
    assert response_json["prediction"] in [0.0, 1.0]

def test_predict_invalid_model_alias(client):
    logger.info("Running test_predict_invalid_model_alias")
    payload = {"model_alias": "non_existent_model_xyz", "features": SAMPLE_VALID_FEATURES}
    response = client.post("/predict", json=payload)
    if response.status_code != 404: log_response_details(response, "non_existent_model_xyz")
    assert response.status_code == 404
    response_json = response.json()
    assert "detail" in response_json
    assert "Model alias 'non_existent_model_xyz' not found" in response_json["detail"]

def test_predict_missing_features_key(client):
    logger.info("Running test_predict_missing_features_key")
    payload = {"model_alias": EXPECTED_MODEL_ALIASES[0] if EXPECTED_MODEL_ALIASES else "any_model"}
    response = client.post("/predict", json=payload)
    if response.status_code != 422: log_response_details(response, payload["model_alias"])
    assert response.status_code == 422

def test_predict_features_not_a_dict(client):
    logger.info("Running test_predict_features_not_a_dict")
    payload = {"model_alias": EXPECTED_MODEL_ALIASES[0] if EXPECTED_MODEL_ALIASES else "any_model", "features": "not_a_dict"}
    response = client.post("/predict", json=payload)
    if response.status_code != 422: log_response_details(response, payload["model_alias"])
    assert response.status_code == 422

def test_predict_missing_required_feature_in_dict(client):
    logger.info("Running test_predict_missing_required_feature_in_dict")
    incomplete_features = SAMPLE_VALID_FEATURES.copy()
    if "Age" in incomplete_features: del incomplete_features["Age"]
    
    model_to_test_with = "randomforestclassifier_classification" # Or any other valid one
    if EXPECTED_MODEL_ALIASES: # Ensure list is not empty
        if model_to_test_with not in EXPECTED_MODEL_ALIASES:
            model_to_test_with = EXPECTED_MODEL_ALIASES[0]
    else: # Fallback if EXPECTED_MODEL_ALIASES is somehow empty
        model_to_test_with = "some_model_if_expected_list_is_empty"


    payload = {"model_alias": model_to_test_with, "features": incomplete_features}
    response = client.post("/predict", json=payload)
    if response.status_code != 200: log_response_details(response, model_to_test_with)
    assert response.status_code == 200
    response_json = response.json(); assert "prediction" in response_json
    assert isinstance(response_json["prediction"], float)
    assert response_json["prediction"] in [0.0, 1.0]

def test_predict_feature_wrong_type_in_dict(client):
    logger.info("Running test_predict_feature_wrong_type_in_dict")
    features_with_wrong_type = SAMPLE_VALID_FEATURES.copy()
    features_with_wrong_type["Pclass"] = "should_be_numeric_not_string"
    
    model_to_test_with = "randomforestclassifier_classification" # Or any other valid one
    if EXPECTED_MODEL_ALIASES: # Ensure list is not empty
        if model_to_test_with not in EXPECTED_MODEL_ALIASES:
            model_to_test_with = EXPECTED_MODEL_ALIASES[0]
    else: # Fallback
        model_to_test_with = "some_model_if_expected_list_is_empty"

    payload = {"model_alias": model_to_test_with, "features": features_with_wrong_type}
    response = client.post("/predict", json=payload)
    if response.status_code != 200: log_response_details(response, model_to_test_with)
    assert response.status_code == 200
    response_json = response.json(); assert "prediction" in response_json
    assert isinstance(response_json["prediction"], float)
    assert response_json["prediction"] in [0.0, 1.0]