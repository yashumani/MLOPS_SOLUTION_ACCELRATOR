import sys
from pathlib import Path
import pytest # You might need to install pytest: pip install pytest
from fastapi.testclient import TestClient

# Adjust path to import app from model_serving_api.py in the parent directory
# This assumes test_api.py is in src/ and model_serving_api.py is in the project root.
# If your project structure is different or your test runner handles paths, adjust as needed.
try:
    from model_serving_api import app
except ImportError:
    # Simple path adjustment if running 'python src/test_api.py' from project root
    # or if model_serving_api is not directly on PYTHONPATH for the test runner.
    # More robust solutions involve proper packaging or PYTHONPATH configuration.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_serving_api import app


client = TestClient(app)

# Sample valid features based on college.csv (excluding the target 'Grad.Rate')
# Use feature names exactly as they appear in the raw CSV before any encoding
SAMPLE_VALID_FEATURES = {
    "Private": "Yes",
    "Apps": 1660,
    "Accept": 1232,
    "Enroll": 721,
    "Top10perc": 23,
    "Top25perc": 52,
    "F.Undergrad": 2885,
    "P.Undergrad": 537,
    "Outstate": 7440,
    "Room.Board": 3300,
    "Books": 450,
    "Personal": 2200,
    "PhD": 70,
    "Terminal": 78,
    "S.F.Ratio": 18.1,
    "perc.alumni": 12,
    "Expend": 7041
}

# Based on your startup log: ['randomforest', 'rf', 'ridge', 'xgboost', 'xgb']
EXPECTED_MODEL_ALIASES = sorted(['randomforest', 'rf', 'ridge', 'xgboost', 'xgb', 'catboost', 'elasticnet', 'kneighbors', 'lightgbm', 'svr'])
EXPECTED_MODEL_COUNT = len(EXPECTED_MODEL_ALIASES)


def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "ok"
    # Check if available_models_count is present and matches expected
    # This depends on whether your /health endpoint actually returns this count.
    # Based on the code I provided, it does.
    assert "available_models_count" in response_json
    assert response_json["available_models_count"] == EXPECTED_MODEL_COUNT


def test_available_models():
    """Test the /available_models endpoint."""
    response = client.get("/available_models")
    assert response.status_code == 200
    response_json = response.json()
    assert "available_model_aliases" in response_json
    # Sort both lists to ensure order doesn't affect the assertion
    assert sorted(response_json["available_model_aliases"]) == EXPECTED_MODEL_ALIASES


@pytest.mark.parametrize("model_alias_to_test", EXPECTED_MODEL_ALIASES)
def test_predict_valid_models(model_alias_to_test: str):
    """Test /predict endpoint with valid model aliases and valid features."""
    payload = {
        "model_alias": model_alias_to_test,
        "features": SAMPLE_VALID_FEATURES
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Failed for model: {model_alias_to_test}. Response: {response.text}"
    response_json = response.json()
    assert response_json["model_alias_used"] == model_alias_to_test
    assert "prediction" in response_json
    assert isinstance(response_json["prediction"], float)


def test_predict_invalid_model_alias():
    """Test /predict endpoint with an invalid model alias."""
    payload = {
        "model_alias": "non_existent_model_123",
        "features": SAMPLE_VALID_FEATURES
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 404 # Not Found
    response_json = response.json()
    assert "detail" in response_json
    assert "Model alias 'non_existent_model_123' not found" in response_json["detail"]


def test_predict_missing_features_key():
    """Test /predict endpoint with the 'features' key missing from payload."""
    payload = {
        "model_alias": "randomforest"
        # 'features' key is missing
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity (Pydantic validation error)


def test_predict_features_not_a_dict():
    """Test /predict endpoint where 'features' is not a dictionary."""
    payload = {
        "model_alias": "randomforest",
        "features": "this_is_not_a_dict"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity


def test_predict_missing_required_feature_in_dict():
    """
    Test /predict endpoint where a feature might be missing from the features dict.
    Note: The current API design with `features: dict` is very flexible and might not
    fail Pydantic validation for missing keys within the dict itself.
    The model's behavior or one-hot encoding + alignment logic would handle it (e.g., defaulting to 0).
    This test primarily ensures the API accepts a dict, even if it's incomplete from a data perspective.
    A more robust test would check if the prediction output changes as expected or if specific
    logic for missing feature imputation (if any) is triggered.
    For now, we just check if it processes without Pydantic error if 'features' is a dict.
    """
    incomplete_features = SAMPLE_VALID_FEATURES.copy()
    del incomplete_features["Apps"] # Remove a feature

    payload = {
        "model_alias": "randomforest",
        "features": incomplete_features
    }
    response = client.post("/predict", json=payload)
    # Expecting 200 because the API will one-hot encode and then align columns,
    # filling missing ones with 0.
    assert response.status_code == 200
    response_json = response.json()
    assert "prediction" in response_json
    assert isinstance(response_json["prediction"], float)


def test_predict_feature_wrong_type_in_dict():
    """
    Test /predict endpoint where a feature has an unexpected data type.
    Note: Pydantic validates the structure of PredictionRequest (model_alias: str, features: dict).
    It does not validate the types *within* the 'features' dict itself.
    Pandas, when creating the DataFrame (pd.DataFrame([req.features])), might attempt type conversion.
    If a conversion fails later (e.g. during one-hot encoding or model prediction if a string is
    passed where a number is expected and can't be coerced), it could lead to a 500 error or
    unexpected behavior. This test sends a plausible "wrong type".
    """
    features_with_wrong_type = SAMPLE_VALID_FEATURES.copy()
    features_with_wrong_type["Apps"] = "this_should_be_a_number" # Invalid type for 'Apps'

    payload = {
        "model_alias": "randomforest",
        "features": features_with_wrong_type
    }
    response = client.post("/predict", json=payload)
    # This might result in a 200 if pandas/model can handle/coerce it,
    # or a 500 if an unhandled error occurs during processing.
    # Given the current API structure, if pd.get_dummies encounters this,
    # it might create a column like "Apps_this_should_be_a_number".
    # The subsequent alignment would drop this and use 0 for expected numeric "Apps" columns.
    # So, a 200 is plausible.
    # If you had strict type checking for individual features *before* model processing,
    # this could be a 422.
    assert response.status_code == 200 # Assuming it processes due to robust column alignment
    response_json = response.json()
    assert "prediction" in response_json
    assert isinstance(response_json["prediction"], float)


if __name__ == "__main__":
    # This allows running the tests with 'python src/test_api.py'
    # You'll need to have pytest installed and in your PATH, or run specific functions.
    # For a better experience, run with 'pytest' command from the project root.
    # Example: pytest src/test_api.py
    
    # For a simple programmatic run without pytest CLI:
    print("Running API tests...")
    test_health_check()
    print("✓ Health check passed")
    test_available_models()
    print("✓ Available models passed")
    for alias in EXPECTED_MODEL_ALIASES:
        test_predict_valid_models(alias)
        print(f"✓ Predict valid model ({alias}) passed")
    test_predict_invalid_model_alias()
    print("✓ Predict invalid model alias passed")
    test_predict_missing_features_key()
    print("✓ Predict missing features key passed")
    test_predict_features_not_a_dict()
    print("✓ Predict features not a dict passed")
    test_predict_missing_required_feature_in_dict()
    print("✓ Predict with incomplete features dict passed")
    test_predict_feature_wrong_type_in_dict()
    print("✓ Predict with feature wrong type passed")
    print("\nAll basic tests executed.")