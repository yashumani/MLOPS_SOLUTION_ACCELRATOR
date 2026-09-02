from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.model_bundle import (
    BUNDLE_FILE_NAME,
    BUNDLE_MANIFEST_NAME,
    ModelBundle,
    capture_input_schema,
    load_model_bundle,
    save_model_bundle,
)
from src.steps.final_evaluation import eval_model


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
