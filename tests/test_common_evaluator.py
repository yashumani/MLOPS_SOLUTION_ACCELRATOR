from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.common_evaluator import (
    CandidateEvidence,
    EvaluationSpec,
    deterministic_cv_splits,
    evaluate_candidate,
    select_best_evidence,
)


SUCCESS_PROCESS_TIMEOUT_SECONDS = 45


def _classification_data():
    features = pd.DataFrame(
        {
            "x": np.linspace(-3.0, 3.0, 60),
            "noise": np.tile([0.0, 1.0, 2.0], 20),
        }
    )
    target = pd.Series(([0] * 30) + ([1] * 30))
    return features, target


def _candidate():
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(random_state=17)),
        ]
    )


def test_cv_split_detaches_read_only_target_buffer() -> None:
    backing = np.asarray(([0] * 10) + ([1] * 10))
    backing.flags.writeable = False
    target = pd.Series(backing, copy=False)

    splits = deterministic_cv_splits(
        np.zeros((len(target), 1)),
        target,
        task_type="classification",
        folds=5,
        seed=17,
    )

    assert target.to_numpy(copy=False).flags.writeable is False
    assert len(splits) == 5
    assert all(len(train) == 16 and len(validation) == 4 for train, validation in splits)


def test_engines_share_exact_fold_assignment_and_metric_contract():
    X, y = _classification_data()
    spec = EvaluationSpec(
        task_type="classification",
        seed=17,
        folds=5,
        timeout_seconds=SUCCESS_PROCESS_TIMEOUT_SECONDS,
        execution_id="execution-1",
    )
    pycaret = evaluate_candidate(
        _candidate(),
        X,
        y,
        candidate_id="pycaret-logistic",
        engine="pycaret",
        spec=spec,
    )
    flaml = evaluate_candidate(
        _candidate(),
        X,
        y,
        candidate_id="flaml-logistic",
        engine="flaml",
        spec=spec,
    )

    assert pycaret.status == flaml.status == "success"
    assert pycaret.split_fingerprint == flaml.split_fingerprint
    assert pycaret.primary_metric == flaml.primary_metric == "balanced_accuracy"
    assert pycaret.selection_score == flaml.selection_score
    assert pycaret.execution_id == flaml.execution_id == "execution-1"


class _SlowClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, delay=0.02):
        self.delay = delay

    def fit(self, X, y):
        time.sleep(self.delay)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.repeat(self.classes_[0], len(X))


class _SerialOnlyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if self.n_jobs != 1:
            raise ValueError("nested parallelism was not disabled")
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.resize(self.classes_, len(X))


class _NoiseAwareClusterer(BaseEstimator, ClusterMixin):
    def fit_predict(self, X, y=None):
        return np.asarray([-1, -1, 0, 0, 1, 1])


def test_isolated_evaluator_disables_nested_estimator_process_pools():
    X, y = _classification_data()

    evidence = evaluate_candidate(
        _SerialOnlyClassifier(),
        X,
        y,
        candidate_id="serial-only",
        engine="pycaret",
        spec=EvaluationSpec(
            task_type="classification",
            folds=3,
            timeout_seconds=SUCCESS_PROCESS_TIMEOUT_SECONDS,
        ),
    )

    assert evidence.status == "success"
    assert evidence.completed_folds == 3


def test_timeout_is_process_enforced_and_cannot_win():
    X, y = _classification_data()
    started_at = time.monotonic()
    evidence = evaluate_candidate(
        _SlowClassifier(delay=30.0),
        X,
        y,
        candidate_id="slow",
        engine="flaml",
        spec=EvaluationSpec(
            task_type="classification",
            folds=5,
            timeout_seconds=0.2,
        ),
    )

    observed_elapsed = time.monotonic() - started_at

    assert evidence.status == "timeout"
    assert evidence.failure_reason == "candidate_engine_timeout"
    assert evidence.completed_folds == 0
    assert evidence.censored is True
    assert evidence.elapsed_seconds <= observed_elapsed
    assert observed_elapsed - evidence.elapsed_seconds < 0.5
    assert observed_elapsed < 5.0
    assert select_best_evidence([evidence]) is None


def test_censored_success_evidence_cannot_win():
    censored = CandidateEvidence(
        candidate_id="censored",
        engine="pycaret",
        task_type="classification",
        status="success",
        primary_metric="balanced_accuracy",
        selection_score=0.99,
        censored=True,
    )
    complete = CandidateEvidence(
        candidate_id="complete",
        engine="flaml",
        task_type="classification",
        status="success",
        primary_metric="balanced_accuracy",
        selection_score=0.75,
    )

    assert censored.selectable is False
    assert select_best_evidence([censored, complete])["candidate_id"] == "complete"


def test_non_pycaret_clustering_is_explicitly_unsupported():
    evidence = evaluate_candidate(
        object(),
        np.asarray([[0.0], [1.0], [2.0]]),
        None,
        candidate_id="flaml-cluster",
        engine="flaml",
        spec=EvaluationSpec(task_type="clustering"),
    )

    assert evidence.status == "skipped_unsupported"
    assert evidence.failure_reason == "clustering_is_pycaret_only"
    assert evidence.selectable is False


def test_clustering_scores_fitted_preprocessing_feature_space():
    frame = pd.DataFrame(
        {
            "segment": ["a", "a", "a", "b", "b", "b"],
            "value": [0.0, 0.1, 0.2, 9.8, 9.9, 10.0],
        }
    )
    preprocessing = ColumnTransformer(
        [
            (
                "segment",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["segment"],
            )
        ],
        remainder="passthrough",
    )
    candidate = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("estimator", KMeans(n_clusters=2, random_state=42, n_init=5)),
        ]
    )

    evidence = evaluate_candidate(
        candidate,
        frame,
        None,
        candidate_id="cluster-candidate",
        engine="pycaret",
        spec=EvaluationSpec(
            task_type="clustering",
            seed=42,
            timeout_seconds=SUCCESS_PROCESS_TIMEOUT_SECONDS,
        ),
    )

    assert evidence.status == "success"
    assert evidence.metrics["silhouette_score"] > 0.8


def test_clustering_penalizes_noise_fraction():
    evidence = evaluate_candidate(
        _NoiseAwareClusterer(),
        np.asarray([[50.0], [60.0], [0.0], [0.1], [10.0], [10.1]]),
        None,
        candidate_id="noise-aware",
        engine="pycaret",
        spec=EvaluationSpec(
            task_type="clustering",
            timeout_seconds=SUCCESS_PROCESS_TIMEOUT_SECONDS,
        ),
    )

    assert evidence.status == "success"
    assert evidence.metrics["clustered_fraction"] == pytest.approx(4 / 6)
    assert evidence.metrics["silhouette_score"] < 4 / 6
