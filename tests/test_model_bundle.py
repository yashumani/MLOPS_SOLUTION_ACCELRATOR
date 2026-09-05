from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.model_bundle import (
    BUNDLE_FILE_NAME,
    BUNDLE_MANIFEST_NAME,
    ModelBundle,
    _model_state_sha256,
    capture_input_schema,
    load_model_bundle,
    save_model_bundle,
)
from src.steps.final_evaluation import eval_model


class _MixedFeaturePreprocessor:
    def __getstate__(self) -> dict[str, object]:
        return {}

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        country = frame["country"].map({"US": 0.0, "UK": 1.0}).to_numpy()
        return np.column_stack((frame["quantity"].to_numpy(), country))


def _bundle():
    raw = pd.DataFrame({"feature": [-2.0, -1.0, 1.0, 2.0]})
    target = np.asarray([0, 0, 1, 1])
    preprocessing = StandardScaler().fit(raw)
    estimator = LogisticRegression(random_state=42).fit(
        preprocessing.transform(raw),
        target,
    )
    bundle = ModelBundle(
        estimator=estimator,
        preprocessing=preprocessing,
        task_type="classification",
        candidate_id="candidate-1",
        input_schema=capture_input_schema(raw),
        recipe={"encoding": "none", "scaling": "standard"},
        selection_metrics={"balanced_accuracy": 1.0},
        final_test_metrics={"balanced_accuracy": 1.0},
        environment={"name": "mlops-v3-unified", "version": "23"},
        lineage={
            "execution_id": "execution-1",
            "parent_run_id": "parent-1",
            "candidate_run_id": "child-1",
            "model_version": "7",
        },
        dependencies=("scikit-learn", "pandas"),
        signature={"inputs": ["feature"], "outputs": ["prediction"]},
        input_example={"feature": 0.0},
    )
    return raw, bundle


def test_bundle_roundtrip_predicts_from_raw_input(tmp_path):
    raw, bundle = _bundle()
    manifest = save_model_bundle(bundle, tmp_path)
    restored = load_model_bundle(tmp_path)

    assert restored.predict(raw).tolist() == bundle.predict(raw).tolist()
    assert manifest["bundle_id"] == restored.bundle_id
    assert manifest["artifact_file"] == BUNDLE_FILE_NAME
    assert (tmp_path / BUNDLE_MANIFEST_NAME).is_file()


def test_bundle_enforces_raw_schema():
    _, bundle = _bundle()
    with pytest.raises(ValueError, match="missing columns"):
        bundle.predict(pd.DataFrame({"wrong": [1.0]}))


def test_bundle_load_fails_when_artifact_hash_is_tampered(tmp_path):
    _, bundle = _bundle()
    save_model_bundle(bundle, tmp_path)
    manifest_path = tmp_path / BUNDLE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="artifact hash"):
        load_model_bundle(tmp_path)


def test_bundle_identity_binds_fitted_estimator_state():
    raw, first = _bundle()
    preprocessing = StandardScaler().fit(raw)
    second_estimator = LogisticRegression(random_state=42).fit(
        preprocessing.transform(raw),
        np.asarray([1, 1, 0, 0]),
    )
    second = ModelBundle(
        estimator=second_estimator,
        preprocessing=preprocessing,
        task_type=first.task_type,
        candidate_id=first.candidate_id,
        input_schema=first.input_schema,
        recipe=first.recipe,
        selection_metrics=first.selection_metrics,
        final_test_metrics=first.final_test_metrics,
        environment=first.environment,
        lineage=first.lineage,
        dependencies=first.dependencies,
        signature=first.signature,
        input_example=first.input_example,
    )

    assert first.bundle_id != second.bundle_id
    assert (
        first.metadata()["model_state_sha256"]
        != second.metadata()["model_state_sha256"]
    )


def test_bundle_detects_fitted_state_mutation():
    _, bundle = _bundle()
    bundle.estimator.coef_[0, 0] += 1.0

    with pytest.raises(ValueError, match="fitted model state changed"):
        bundle.assert_integrity()


@pytest.mark.parametrize("schema_version", [3, 4])
def test_legacy_bundle_keeps_identity_after_roundtrip(tmp_path, schema_version):
    raw, bundle = _bundle()
    legacy = replace(bundle, bundle_schema_version=schema_version)
    expected_id = legacy.bundle_id
    save_model_bundle(legacy, tmp_path)
    restored = load_model_bundle(tmp_path)

    assert restored.bundle_schema_version == schema_version
    assert restored.bundle_id == expected_id
    np.testing.assert_array_equal(restored.predict(raw), legacy.predict(raw))
    restored.estimator.coef_[0, 0] += 1.0
    with pytest.raises(ValueError, match="fitted model state changed"):
        restored.assert_integrity()


def test_bundle_rejects_unknown_hash_schema():
    _, bundle = _bundle()
    with pytest.raises(ValueError, match="Unsupported ModelBundle schema"):
        replace(bundle, bundle_schema_version=99)


def test_schema_five_hashes_class_references_without_ignoring_fitted_values():
    assert _model_state_sha256({"class": float, "state": [1.0]}, None) != (
        _model_state_sha256({"class": int, "state": [1.0]}, None)
    )
    assert _model_state_sha256({"class": float, "state": [1.0]}, None) != (
        _model_state_sha256({"class": float, "state": [2.0]}, None)
    )
    with pytest.raises(TypeError, match="state must be hashable"):
        _model_state_sha256(float, None, schema_version=4)
    with pytest.raises(TypeError, match="state must be hashable"):
        _model_state_sha256(float.__add__, None)


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_flaml_catboost_recipe_bundle_roundtrip_and_refit(tmp_path, task_type):
    from flaml.automl.model import CatBoostEstimator
    from src.utils.fitted_variant_preprocessor import FittedVariantPreprocessor

    raw = pd.DataFrame({
        "feature [units]": np.linspace(-3.0, 3.0, 80),
        "category": ["a", "b"] * 40,
    })
    target = pd.Series([0, 1] * 40) if task_type == "classification" else raw.iloc[:, 0] ** 2
    preprocessing = FittedVariantPreprocessor({
        "task_type": task_type,
        "stage3_preprocessing": {
            "encoding": {"categorical_method": "onehot"},
            "scaling": {"method": "standard"},
        },
    })
    transformed = preprocessing.fit_transform(raw, target)
    estimator = CatBoostEstimator(
        task=task_type, n_estimators=5, thread_count=1, random_seed=42,
        allow_writing_files=False,
    )
    estimator.fit(transformed, target, use_best_model=False)
    bundle = ModelBundle(
        estimator=estimator, preprocessing=preprocessing, task_type=task_type,
        candidate_id=f"flaml-catboost-{task_type}", input_schema=capture_input_schema(raw),
    )
    prediction = bundle.predict(raw)
    save_model_bundle(bundle, tmp_path)
    restored = load_model_bundle(tmp_path)
    assert restored.bundle_schema_version == 5
    assert restored.bundle_id == bundle.bundle_id
    np.testing.assert_allclose(restored.predict(raw), prediction)
    changed_target = 1 - target if task_type == "classification" else -target
    restored.estimator.fit(transformed, changed_target, use_best_model=False)
    with pytest.raises(ValueError, match="fitted model state changed"):
        restored.assert_integrity()


def test_schema_four_hash_ignores_aliases_but_detects_value_changes():
    shared = np.array([1.0, 2.0])
    aliased = {"first": shared, "second": shared}
    copied = {"first": shared.copy(), "second": shared.copy()}
    assert _model_state_sha256(aliased, None) == _model_state_sha256(copied, None)
    assert _model_state_sha256(aliased, None, schema_version=3) != (
        _model_state_sha256(copied, None, schema_version=3)
    )
    copied["second"][0] += 1.0
    assert _model_state_sha256(aliased, None) != _model_state_sha256(copied, None)


def test_schema_four_hash_handles_cycles_and_rejects_opaque_state():
    left, right = [], []
    left.append(left)
    right.append(right)
    assert _model_state_sha256(left, None) == _model_state_sha256(right, None)
    right.append(1)
    assert _model_state_sha256(left, None) != _model_state_sha256(right, None)
    with pytest.raises(TypeError, match="state must be hashable"):
        _model_state_sha256(object(), None)


def test_cython_loss_hash_binds_reduction_parameters():
    from sklearn._loss._loss import CyPinballLoss

    assert _model_state_sha256(CyPinballLoss(0.2), None) == (
        _model_state_sha256(CyPinballLoss(0.2), None)
    )
    assert _model_state_sha256(CyPinballLoss(0.2), None) != (
        _model_state_sha256(CyPinballLoss(0.8), None)
    )


@pytest.mark.parametrize("family", ["gradient_boosting", "catboost", "lightgbm"])
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_boosted_tree_bundle_roundtrip_preserves_identity_and_detects_refit(
    tmp_path, family, task_type
):
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    classification = task_type == "classification"
    if family == "gradient_boosting":
        cls = GradientBoostingClassifier if classification else GradientBoostingRegressor
        estimator = cls(n_estimators=5, random_state=42)
    elif family == "catboost":
        cls = CatBoostClassifier if classification else CatBoostRegressor
        estimator = cls(
            iterations=5, depth=3, random_seed=42, thread_count=1,
            verbose=False, allow_writing_files=False,
        )
    else:
        cls = LGBMClassifier if classification else LGBMRegressor
        estimator = cls(n_estimators=5, num_leaves=5, random_state=42, n_jobs=1, verbosity=-1)
    raw = pd.DataFrame(np.random.RandomState(42).normal(size=(80, 4)))
    raw.columns = [f"feature_{index}" for index in range(4)]
    continuous_target = raw.iloc[:, 0] * 2 + raw.iloc[:, 1]
    target = (continuous_target > 0).astype(int) if classification else continuous_target
    preprocessing = StandardScaler().fit(raw)
    transformed = preprocessing.transform(raw)
    estimator.fit(transformed, target)
    bundle = ModelBundle(
        estimator=estimator,
        preprocessing=preprocessing,
        task_type=task_type,
        candidate_id=f"{family}-{task_type}",
        input_schema=capture_input_schema(raw),
    )
    expected_id = bundle.bundle_id
    expected_predictions = bundle.predict(raw)
    save_model_bundle(bundle, tmp_path)
    restored = load_model_bundle(tmp_path)

    np.testing.assert_allclose(restored.predict(raw), expected_predictions)
    np.testing.assert_allclose(restored.predict(raw), expected_predictions)
    assert restored.bundle_id == expected_id

    changed_target = 1 - target if classification else -target
    restored.estimator.fit(transformed, changed_target)
    with pytest.raises(ValueError, match="fitted model state changed"):
        restored.assert_integrity()


def test_bundle_roundtrip_decodes_classification_predictions(tmp_path):
    raw = pd.DataFrame({"feature": [-2.0, -1.0, 1.0, 2.0]})
    original_target = np.asarray(["stay", "stay", "churn", "churn"])
    target_decoder = LabelEncoder().fit(original_target)
    encoded_target = target_decoder.transform(original_target)
    preprocessing = StandardScaler().fit(raw)
    estimator = LogisticRegression(random_state=42).fit(
        preprocessing.transform(raw),
        encoded_target,
    )
    bundle = ModelBundle(
        estimator=estimator,
        preprocessing=preprocessing,
        target_decoder=target_decoder,
        task_type="classification",
        candidate_id="phasec-string-labels",
        input_schema=capture_input_schema(raw),
        labels=tuple(target_decoder.classes_),
    )

    save_model_bundle(bundle, tmp_path)
    restored = load_model_bundle(tmp_path)

    assert set(restored.predict(raw)) == {"stay", "churn"}
    assert restored.metadata()["target_decoder_type"] == "LabelEncoder"
    assert restored.bundle_id == bundle.bundle_id


def test_bundle_identity_binds_target_decoder_state():
    raw, _ = _bundle()
    preprocessing = StandardScaler().fit(raw)
    estimator = LogisticRegression(random_state=42).fit(
        preprocessing.transform(raw),
        np.asarray([0, 0, 1, 1]),
    )
    decoder_a = LabelEncoder().fit(["churn", "stay"])
    decoder_b = LabelEncoder().fit(["left", "right"])
    first = ModelBundle(
        estimator=estimator,
        preprocessing=preprocessing,
        target_decoder=decoder_a,
        task_type="classification",
        candidate_id="decoder-bound",
        input_schema=capture_input_schema(raw),
    )
    second = ModelBundle(
        estimator=estimator,
        preprocessing=preprocessing,
        target_decoder=decoder_b,
        task_type="classification",
        candidate_id="decoder-bound",
        input_schema=capture_input_schema(raw),
    )

    assert first.bundle_id != second.bundle_id


def test_locked_test_evaluation_uses_decoded_target_domain():
    raw = pd.DataFrame({"feature": [-2.0, -1.0, 1.0, 2.0]})
    target = np.asarray(["stay", "stay", "churn", "churn"])
    target_decoder = LabelEncoder().fit(target)
    preprocessing = StandardScaler().fit(raw)
    estimator = LogisticRegression(random_state=42).fit(
        preprocessing.transform(raw),
        target_decoder.transform(target),
    )
    bundle = ModelBundle(
        estimator=estimator,
        preprocessing=preprocessing,
        target_decoder=target_decoder,
        task_type="classification",
        candidate_id="decoded-evaluation",
        input_schema=capture_input_schema(raw),
    )

    metrics = eval_model(bundle, raw, pd.Series(target), "classification")

    assert metrics is not None
    assert metrics["accuracy"] == 1.0


def test_clustering_evaluation_uses_bundle_raw_input_contract():
    raw = pd.DataFrame(
        {
            "quantity": [1.0, 1.2, 9.8, 10.0],
            "country": ["US", "US", "UK", "UK"],
        }
    )
    preprocessing = _MixedFeaturePreprocessor()
    transformed = preprocessing.transform(raw)
    estimator = KMeans(n_clusters=2, random_state=42, n_init=10).fit(transformed)
    bundle = ModelBundle(
        estimator=estimator,
        preprocessing=preprocessing,
        task_type="clustering",
        candidate_id="cluster-raw-contract",
        input_schema=capture_input_schema(raw),
    )

    metrics = eval_model(bundle, raw, None, "clustering")

    assert metrics is not None
    assert metrics["n_clusters"] == 2
    assert metrics["silhouette_score"] > 0.0


def test_random_forest_bundle_hash_is_stable_across_save_load(tmp_path):
    raw = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    estimator = RandomForestClassifier(
        n_estimators=3,
        random_state=7,
    ).fit(raw, [0, 0, 1, 1])
    bundle = ModelBundle(
        estimator=estimator,
        task_type="classification",
        candidate_id="rf-candidate",
        input_schema=capture_input_schema(raw),
    )
    expected_id = bundle.bundle_id
    expected_predictions = bundle.predict(raw).tolist()

    save_model_bundle(bundle, tmp_path)
    restored = load_model_bundle(tmp_path)

    assert restored.bundle_id == expected_id
    assert restored.predict(raw).tolist() == expected_predictions
