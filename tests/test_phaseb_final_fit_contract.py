"""The published Phase B estimator must honor its evaluated sampling recipe."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from steps import s06_phaseb_variant_runner as runner
from utils.common_evaluator import build_training_resampler


def _data():
    values, target = make_classification(
        n_samples=120, n_features=4, n_redundant=0, n_informative=3,
        weights=[0.8, 0.2], random_state=7,
    )
    return pd.DataFrame(values, columns=list("abcd")), pd.Series(target)


@pytest.mark.parametrize("method", ["smote", "adasyn", "smoteenn", "smotetomek"])
def test_final_fit_matches_recipe_without_mutating_discovery_model(method):
    features, target = _data()
    original_features = features.copy(deep=True)
    original_target = target.copy(deep=True)
    recipe = {"stage3_preprocessing": {"imbalance_handling": {"method": method}}}
    discovery = LogisticRegression(C=0.7, random_state=17).fit(features, target)
    original_coef = discovery.coef_.copy()
    fitted, evidence = runner.fit_resampled_phaseb_estimator(
        discovery, features, target, build_training_resampler(recipe, 42)
    )
    expected_x, expected_y = build_training_resampler(recipe, 42).fit_resample(
        features, target
    )
    expected = clone(discovery).fit(expected_x, expected_y)
    assert fitted is not discovery
    assert fitted.get_params() == discovery.get_params()
    np.testing.assert_allclose(fitted.coef_, expected.coef_)
    np.testing.assert_allclose(discovery.coef_, original_coef)
    pd.testing.assert_frame_equal(features, original_features)
    pd.testing.assert_series_equal(target, original_target)
    assert evidence["training_rows_before"] == len(features)
    assert evidence["training_rows_after"] == len(expected_x)
    assert evidence["holdout_used"] is False


def test_resampling_failure_is_not_silently_replaced_with_search_model():
    features, target = _data()

    class BrokenSampler:
        def fit_resample(self, *_args):
            raise ValueError("insufficient minority samples")

    with pytest.raises(ValueError, match="insufficient minority samples"):
        runner.fit_resampled_phaseb_estimator(
            LogisticRegression(), features, target, BrokenSampler()
        )


def test_recipe_final_fit_serializes_through_existing_hard_timeout():
    features, target = _data()
    recipe = {"stage3_preprocessing": {"imbalance_handling": {"method": "smote"}}}
    fitted, evidence = runner.run_with_hard_timeout(
        runner.fit_resampled_phaseb_estimator,
        LogisticRegression(C=0.7, random_state=17), features, target,
        build_training_resampler(recipe, 42), timeout_seconds=30,
    )
    assert fitted.predict(features.iloc[:5]).shape == (5,)
    assert evidence["training_rows_after"] > len(features)


@pytest.mark.parametrize("engine", ["flaml", "pycaret"])
@pytest.mark.parametrize("outcome", ["success", "sampler_failure", "timeout"])
def test_candidate_returns_only_recipe_fitted_model_with_original_deadline(
    monkeypatch, engine, outcome,
):
    clock = [100.0]
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    monkeypatch.setattr(runner, "check_leakage_risk", lambda _: "none")
    monkeypatch.setattr(
        runner.mlflow, "start_run",
        lambda **_: nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="child"))),
    )
    for method in ("set_tag", "log_params", "log_metrics"):
        monkeypatch.setattr(runner.mlflow, method, lambda *_args, **_kwargs: None)
    logged = []
    monkeypatch.setattr(runner.mlflow, "log_dict", lambda *args: logged.append(args))
    features, target = _data()
    frame = features.assign(target=target)
    discovery = LogisticRegression(C=0.7, random_state=17).fit(features, target)
    recipe = {"stage3_preprocessing": {"imbalance_handling": {"method": "smote"}}}
    variant = SimpleNamespace(
        variant_id="sampled-recipe", to_dict=lambda: recipe,
        stage3_preprocessing=SimpleNamespace(
            imputation=SimpleNamespace(method="median"),
            encoding=SimpleNamespace(categorical_method="none"),
            scaling=SimpleNamespace(method="none"),
            imbalance_handling=SimpleNamespace(method="smote"),
        ),
        stage4_feature_engineering=SimpleNamespace(
            feature_selection=SimpleNamespace(method="none")
        ),
    )
    candidate = runner.CandidateRecord(
        task_type="classification", recipe_id="sampled-recipe", recipe_hash="recipe-hash",
        engine=engine, algorithm="search", parameters={}, split_id="split",
        data_version="data-v1", code_sha="code-sha", environment_hash="env-hash",
    )
    observed = {}

    def worker(function, *args, timeout_seconds):
        if function is runner.fit_resampled_phaseb_estimator:
            observed["fit_deadline"] = clock[0] + timeout_seconds
            pd.testing.assert_frame_equal(args[1], features)
            pd.testing.assert_series_equal(args[2], frame["target"])
            clock[0] += 2
            if outcome == "sampler_failure":
                raise ValueError("insufficient minority samples")
            if outcome == "timeout":
                raise runner.HardDeadlineExceeded("recipe final fit timed out")
            return function(*args)
        clock[0] += timeout_seconds
        return discovery, {"algorithm": "lr", "primary_metric": 0.75}, False

    def evaluate(*_args, spec, **_kwargs):
        observed["cv_deadline"] = clock[0] + spec.timeout_seconds
        clock[0] += 10
        return SimpleNamespace(
            selectable=True, status="success", metrics={"balanced_accuracy": 0.75},
            selection_score=0.75, to_dict=lambda: {"selectable": True},
        )

    monkeypatch.setattr(runner, "run_with_hard_timeout", worker)
    monkeypatch.setattr(runner, "evaluate_candidate", evaluate)
    result, fitted = runner.run_variant_with_nested_mlflow(
        variant, frame, engine, "target", "classification", time_budget=120,
        attempt_deadline=220, execution_id="execution", search_candidate=candidate,
        random_seed=42, cv_folds=3, mlflow_parent_run_id="parent",
        df_preprocessed=frame,
    )
    assert observed["cv_deadline"] < 220
    assert observed["fit_deadline"] == pytest.approx(220)
    if outcome != "success":
        assert result.failed
        assert fitted is None
        assert result.timed_out is (outcome == "timeout")
        assert "Recipe final fitting failed" in result.failure_reason
        assert result.candidate_record["status"] != "success"
        assert logged == []
    else:
        assert not result.failed
        assert fitted is not discovery
        assert result.metrics["recipe_final_fit"]["holdout_used"] is False
        assert logged[0][1] == "recipe_final_fit.json"
        assert result.metrics["common_evaluator"]["selectable"] is True
