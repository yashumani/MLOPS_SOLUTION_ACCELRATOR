"""Regression tests for the persisted Phase C preprocessing contract."""

from __future__ import annotations

import inspect
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.steps import phasec_optuna_hpo
from src.steps.phasec_optuna_hpo import build_phasec_preprocessor
from src.utils.fitted_variant_preprocessor import FittedVariantPreprocessor
from src.utils.model_bundle import (
    ModelBundle,
    capture_input_schema,
    load_model_bundle,
    save_model_bundle,
)


def _training_data():
    features = pd.DataFrame(
        {
            "category": ["a", "b", "a", "b", "a", "b"],
            "numeric": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    target = np.array([0, 1, 0, 1, 0, 1])
    return features, target


def test_final_fit_error_captures_representation_without_target_values(tmp_path):
    class RejectingSampler:
        def fit_resample(self, X, y):
            raise ValueError("resampling rejected")

    error_path = tmp_path / "error.txt"
    phasec_optuna_hpo._final_fit_worker(
        str(tmp_path / "model.joblib"), str(error_path), object(),
        np.zeros((2, 1)), pd.Series(["private-label-a", "private-label-b"]),
        RejectingSampler(),
    )
    error = error_path.read_text()
    assert "resampling rejected" in error
    assert '"container": "pandas.core.series.Series"' in error
    assert '"array_dtype": "object"' in error
    assert "private-label" not in error


@pytest.mark.parametrize("sampler_name", ["smotetomek", "smoteenn"])
@pytest.mark.parametrize("string_labels", [False, True])
def test_final_fit_detaches_read_only_classification_target(tmp_path, sampler_name, string_labels):
    from imblearn.combine import SMOTEENN, SMOTETomek

    values = np.array(["negative"] * 8 + ["positive"] * 8) if string_labels else np.repeat([0, 1], 8)
    backing = np.frombuffer(values.tobytes(), dtype=values.dtype)
    target = pd.Series(backing, copy=False)
    features = np.column_stack((np.repeat([0.0, 10.0], 8), np.arange(16) / 100))
    sampler = {"smotetomek": SMOTETomek, "smoteenn": SMOTEENN}[sampler_name](random_state=42)
    result_path, error_path = tmp_path / "model.joblib", tmp_path / "error.txt"

    phasec_optuna_hpo._final_fit_worker(
        str(result_path), str(error_path), LogisticRegression(random_state=42),
        features, target, sampler,
    )

    assert not error_path.exists(), error_path.read_text() if error_path.exists() else ""
    restored = joblib.load(result_path)
    np.testing.assert_array_equal(restored.predict(features), values)
    np.testing.assert_array_equal(target.to_numpy(), values)
    assert not backing.flags.writeable


def test_final_fit_preserves_read_only_continuous_target(tmp_path):
    from sklearn.linear_model import LinearRegression

    values = np.array([0.25, 1.75, 3.25, 4.75])
    backing = np.frombuffer(values.tobytes(), dtype=values.dtype)
    target = pd.Series(backing, copy=False)
    features = np.arange(4).reshape(-1, 1)
    result_path, error_path = tmp_path / "model.joblib", tmp_path / "error.txt"
    phasec_optuna_hpo._final_fit_worker(
        str(result_path), str(error_path), LinearRegression(), features, target, None,
    )
    assert not error_path.exists(), error_path.read_text() if error_path.exists() else ""
    np.testing.assert_allclose(joblib.load(result_path).predict(features), values)
    assert target.dtype == values.dtype
    assert not backing.flags.writeable


def test_final_fit_preserves_targetless_clustering(tmp_path):
    from sklearn.cluster import KMeans

    features = np.array([[0.0], [0.1], [10.0], [10.1]])
    result_path, error_path = tmp_path / "model.joblib", tmp_path / "error.txt"
    phasec_optuna_hpo._final_fit_worker(
        str(result_path), str(error_path), KMeans(n_clusters=2, n_init=1, random_state=42),
        features, None, None,
    )
    assert not error_path.exists(), error_path.read_text() if error_path.exists() else ""
    assert len(set(joblib.load(result_path).predict(features))) == 2


def test_phasec_scaler_is_fitted_only_on_training_rows():
    train, target = _training_data()
    holdout = pd.DataFrame({"category": ["unseen"], "numeric": [100.0]})
    preprocessor = build_phasec_preprocessor(
        train,
        encoding="onehot",
        scaling="standard",
        random_seed=42,
    )

    transformed_train = preprocessor.fit_transform(train, target)
    transformed_holdout = preprocessor.transform(holdout)
    output_names = list(
        preprocessor.named_steps["columns"].get_feature_names_out()
    )
    numeric_index = output_names.index("remainder__numeric")

    assert transformed_train[:, numeric_index].mean() == pytest.approx(0.0)
    assert transformed_holdout[0, numeric_index] > 20
    assert transformed_train.shape[1] == transformed_holdout.shape[1]
    assert np.isfinite(transformed_holdout).all()


def test_phasec_model_roundtrip_applies_fitted_preprocessing(tmp_path):
    train, target = _training_data()
    raw_holdout = pd.DataFrame(
        {"category": ["a", "unseen"], "numeric": [1.5, 20.0]}
    )
    preprocessor = build_phasec_preprocessor(
        train,
        encoding="onehot",
        scaling="robust",
        random_seed=42,
    )
    transformed_train = preprocessor.fit_transform(train, target)
    estimator = LogisticRegression(random_state=42).fit(
        transformed_train,
        target,
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ]
    )
    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    restored = joblib.load(model_path)
    predictions = restored.predict(raw_holdout)

    assert len(predictions) == len(raw_holdout)
    assert hasattr(restored, "predict_proba")
    assert np.isfinite(restored.predict_proba(raw_holdout)).all()


def test_phasec_main_has_no_bare_estimator_output_fallback():
    source = inspect.getsource(phasec_optuna_hpo.main)

    assert '"model.pkl"' not in source
    assert "train_test_split(" not in source
    assert "evaluate_candidate(" in source
    assert "remaining_hpo_seconds" in source
    assert "cross_val_score(" not in source
    assert "save_model_bundle(phasec_bundle, model_dir)" in source
    assert "candidate_id=tuned_candidate_id" in source


def test_phasec_bundle_replays_complete_recipe_from_raw_boundary(tmp_path):
    train, target = _training_data()
    raw_inference = pd.DataFrame(
        {"category": ["unseen"], "numeric": [10.0]}
    )
    recipe = {
        "recipe_name": "phaseb-recipe",
        "stage3_preprocessing": {
            "imputation": {"method": "median"},
            "encoding": {"categorical_method": "onehot"},
            "scaling": {"method": "standard"},
            "imbalance_handling": {"method": "none"},
            "outlier_handling": {"method": "none"},
        },
        "stage4_feature_engineering": {
            "feature_selection": {"method": "none"},
        },
    }
    preprocessor = FittedVariantPreprocessor(recipe, random_seed=42)
    transformed = preprocessor.fit_transform(train, target)
    estimator = LogisticRegression(random_state=42).fit(
        transformed,
        target,
    )
    bundle = ModelBundle(
        estimator=estimator,
        preprocessing=preprocessor,
        task_type="classification",
        candidate_id="phasec:phaseb-1:logistic:tuned:abc",
        input_schema=capture_input_schema(train),
        recipe=recipe,
    )
    save_model_bundle(bundle, tmp_path)

    restored = load_model_bundle(tmp_path)

    assert restored.recipe == recipe
    assert len(restored.predict(raw_inference)) == 1
    assert restored.input_schema["column_order"] == list(train.columns)
