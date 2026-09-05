from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import sys

import pandas as pd
import pytest

from steps import s06_phaseb_variant_runner as runner


@pytest.mark.parametrize("remaining", [0.5, 30, 120, 300, 600, 900])
def test_search_worker_and_evaluation_reserves_stay_inside_ceiling(remaining):
    engine, worker = runner.candidate_training_budgets(remaining)
    assert 0 < runner.engine_search_remaining(engine, 0) < engine < worker
    assert worker == pytest.approx(min(remaining, 600) * 0.60)
    assert worker < remaining
    assert runner.engine_search_remaining(engine, engine) == 0


@pytest.mark.parametrize("remaining", [0, -1, float("inf"), float("nan")])
def test_invalid_candidate_budget_fails_closed(remaining):
    with pytest.raises(runner.HardDeadlineExceeded):
        runner.candidate_training_budgets(remaining)


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_pycaret_search_uses_fractional_minutes_and_reserves_completion(
    monkeypatch, task_type
):
    clock = [100.0]
    observed = []
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    module = ModuleType(f"pycaret.{task_type}")

    def setup(**_kwargs):
        clock[0] += 5

    def compare_models(**kwargs):
        observed.append(kwargs["budget_time"])
        clock[0] += kwargs["budget_time"] * 60
        return object()

    module.setup = setup
    module.compare_models = compare_models
    module.add_metric = lambda *_args, **_kwargs: None
    module.pull = lambda: pd.DataFrame(
        {"Balanced Accuracy": [0.75], "R2": [0.75]}
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    variant = SimpleNamespace(
        stage3_preprocessing=SimpleNamespace(imbalance_handling=None)
    )
    model, _metrics, timed_out = runner.train_pycaret_variant(
        pd.DataFrame({"feature": [1, 2], "target": [0, 1]}),
        variant, "target", task_type, time_budget=60,
    )
    assert model is not None
    assert not timed_out
    assert observed == [pytest.approx(43 / 60)]
    assert clock[0] < 160


def test_flaml_search_returns_before_engine_deadline(monkeypatch):
    clock = [100.0]
    observed = {}
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    module = ModuleType("flaml")

    class AutoML:
        best_validation_score = 0.75
        best_estimator = "rf"
        config_history = {}
        model = object()

        def fit(self, **kwargs):
            observed.update(kwargs)
            clock[0] += kwargs["time_budget"] + 2

    module.AutoML = AutoML
    monkeypatch.setitem(sys.modules, "flaml", module)
    model, _metrics, timed_out = runner.train_flaml_variant(
        pd.DataFrame({"feature": [1, 2], "target": [0, 1]}),
        object(), "target", "regression", time_budget=60,
    )
    assert model is AutoML.model
    assert not timed_out
    assert observed["time_budget"] == pytest.approx(48)
    assert clock[0] < 160


@pytest.mark.parametrize("engine", ["pycaret", "flaml"])
def test_worker_consuming_all_its_budget_leaves_time_for_common_cv(
    monkeypatch, engine
):
    clock = [100.0]
    observed = {}
    monkeypatch.setattr(runner.time, "time", lambda: clock[0])
    monkeypatch.setattr(runner, "check_leakage_risk", lambda _: "none")
    monkeypatch.setattr(
        runner.mlflow, "start_run",
        lambda **_: nullcontext(SimpleNamespace(info=SimpleNamespace(run_id="child"))),
    )
    monkeypatch.setattr(runner.mlflow, "set_tag", lambda *_args: None)
    monkeypatch.setattr(runner.mlflow, "log_params", lambda *_args: None)
    monkeypatch.setattr(runner, "build_fold_local_pipeline", lambda *_args, **_: object())
    monkeypatch.setattr(runner, "FittedVariantPreprocessor", lambda *_args, **_: object())

    def train(_function, *_args, timeout_seconds):
        observed["engine_budget"] = _args[4]
        observed["worker_budget"] = timeout_seconds
        clock[0] += timeout_seconds
        return object(), {"algorithm": "rf", "primary_metric": 0.75}, False

    def evaluate(*_args, spec, **_kwargs):
        observed["evaluation_budget"] = spec.timeout_seconds
        # Fail closed to avoid unrelated MLflow model publication in this test.
        return SimpleNamespace(
            selectable=False, status="failed", metrics={}, censored=False,
            failure_reason="test evaluation failure", to_dict=lambda: {},
        )

    monkeypatch.setattr(runner, "run_with_hard_timeout", train)
    monkeypatch.setattr(runner, "evaluate_candidate", evaluate)
    variant = SimpleNamespace(
        variant_id="test-recipe",
        stage3_preprocessing=SimpleNamespace(
            imputation=SimpleNamespace(method="median"),
            encoding=SimpleNamespace(categorical_method="none"),
            scaling=SimpleNamespace(method="none"), imbalance_handling=None,
        ),
        stage4_feature_engineering=SimpleNamespace(
            feature_selection=SimpleNamespace(method="none")
        ),
        to_dict=lambda: {},
    )
    candidate = runner.CandidateRecord(
        task_type="regression", recipe_id="test-recipe", recipe_hash="recipe-hash",
        engine=engine, algorithm="search", parameters={}, split_id="split",
        data_version="data-v1", code_sha="code-sha", environment_hash="env-hash",
    )
    frame = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    result, model = runner.run_variant_with_nested_mlflow(
        variant, frame, engine, "target", "regression", time_budget=120,
        attempt_deadline=220, execution_id="execution", search_candidate=candidate,
        random_seed=42, cv_folds=3, mlflow_parent_run_id="parent",
        df_preprocessed=frame,
    )
    assert observed["engine_budget"] < observed["worker_budget"]
    assert observed["worker_budget"] == pytest.approx(72)
    assert observed["evaluation_budget"] == pytest.approx(48)
    assert result.failed
    assert result.failure_reason == "test evaluation failure"
    assert model is None
