"""Regression coverage for task, p-value and unavailable-evidence semantics."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.drift_detection.baseline_capture import BaselineCapture
from src.drift_detection.drift_checker import DriftChecker, DriftResult
from src.drift_detection.drift_config import DriftConfig
from src.drift_detection.pipeline_trigger import PipelineTrigger


def _checker(frame, task="classification", metric=None):
    config = DriftConfig(task_type=task, concept_metric=metric)
    config.column_mapping.target_column = "target"
    baseline = BaselineCapture(config)
    baseline.capture(frame)
    return DriftChecker(config, baseline)


def _frame():
    return pd.DataFrame({
        "feature": np.arange(100, dtype=float),
        "target": [0, 1] * 50,
        "prediction": [0, 1] * 50,
    })


@pytest.mark.parametrize("task", ["classification", "regression", "clustering"])
def test_identical_observations_do_not_create_prediction_drift(task):
    frame = _frame()
    result = _checker(frame, task).check_prediction_drift(frame.copy())
    assert result.status == "evaluated"
    assert result.drift_detected is False
    assert result.details["stattest_name"]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_identical_observations_do_not_create_label_drift(task):
    frame = _frame()
    result = _checker(frame, task).check_label_drift(frame.copy())
    assert result.status == "evaluated"
    assert result.drift_detected is False


@pytest.mark.parametrize("drift_type", ["prediction", "label"])
@pytest.mark.parametrize(
    "score,detected,method",
    [(0.9, False, "K-S p_value"), (0.001, True, "K-S p_value"),
     (0.5, False, "distance with configured threshold")],
)
def test_statistical_engine_owns_score_direction(monkeypatch, drift_type, score, detected, method):
    frame = _frame()
    checker = _checker(frame, "regression")
    supplied = {}

    class Report:
        def __init__(self, metrics):
            supplied["metrics"] = metrics

        def run(self, **kwargs):
            supplied["mapping"] = kwargs["column_mapping"]

        def as_dict(self):
            return {"metrics": [{"result": {
                "drift_score": score, "drift_detected": detected,
                "stattest_name": method, "stattest_threshold": 0.1,
            }}]}

    monkeypatch.setattr("src.drift_detection.drift_checker.Report", Report)
    monkeypatch.setattr(
        "src.drift_detection.drift_checker.ColumnDriftMetric",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    result = getattr(checker, f"check_{drift_type}_drift")(frame)
    assert result.drift_detected is detected
    assert result.drift_score == score
    assert supplied["metrics"][0].stattest_threshold == 0.1
    assert result.details["stattest_name"] == method


@pytest.mark.parametrize("check", ["prediction", "concept", "label"])
def test_missing_columns_are_unavailable_not_successful_negative(check):
    result = getattr(_checker(_frame()), f"check_{check}_drift")(
        pd.DataFrame({"feature": [1, 2]})
    )
    assert result.status == "unavailable"
    assert result.reason
    assert result.drift_detected is False


def test_missing_labels_do_not_prevent_feature_or_prediction_checks():
    frame = _frame()
    results = {r.drift_type: r for r in _checker(frame).run_all_checks(frame.drop(columns="target"))}
    assert results["feature"].status == "evaluated"
    assert results["prediction"].status == "evaluated"
    assert results["concept"].status == "unavailable"
    assert results["label"].status == "unavailable"


def test_empty_feature_window_is_unavailable():
    frame = _frame()
    result = _checker(frame).check_feature_drift(frame.iloc[:0])
    assert result.status == "unavailable"
    assert result.reason == "empty_observation_window"


@pytest.mark.parametrize("metric", ["r2", "mae", "mse", "rmse"])
def test_regression_performance_degradation_uses_metric_direction(metric):
    frame = pd.DataFrame({"feature": range(30), "target": np.arange(30, dtype=float)})
    frame["prediction"] = frame["target"] + 0.01
    checker = _checker(frame, "regression", metric)
    degraded = frame.copy()
    degraded["prediction"] += 10
    result = checker.check_concept_drift(degraded)
    assert result.status == "evaluated"
    assert result.drift_detected is True
    assert result.details["metric_name"] == metric
    assert result.details["degradation"] > 0
    expected_direction = "maximize" if metric == "r2" else "minimize"
    assert result.details["metric_direction"] == expected_direction
    improved = frame.copy()
    improved["prediction"] = improved["target"]
    assert checker.check_concept_drift(improved).drift_detected is False


def test_clustering_does_not_invent_supervised_concept_or_label_drift():
    frame = _frame()
    checker = _checker(frame, "clustering")
    for check in (checker.check_concept_drift, checker.check_label_drift):
        result = check(frame)
        assert result.status == "not_applicable"
        assert result.reason == "unlabeled_task"
        assert result.drift_detected is False


def test_string_class_labels_are_valid_baseline_and_prediction_inputs():
    frame = _frame().replace({"target": {0: "no", 1: "yes"}, "prediction": {0: "no", 1: "yes"}})
    checker = _checker(frame)
    assert checker.check_prediction_drift(frame).drift_detected is False
    result = checker.check_concept_drift(frame)
    assert result.status == "evaluated"
    assert result.details["metric_name"] == "balanced_accuracy"
    assert result.drift_detected is False


@pytest.mark.parametrize("problem", ["constant_target", "missing_value", "single_row", "infinite_prediction"])
def test_unreliable_regression_metric_is_explicitly_unavailable(problem):
    frame = pd.DataFrame({"feature": range(10), "target": np.arange(10, dtype=float)})
    frame["prediction"] = frame["target"]
    checker = _checker(frame, "regression")
    current = frame.copy()
    if problem == "constant_target":
        current["target"] = 1.0
    elif problem == "missing_value":
        current.loc[0, "target"] = np.nan
    elif problem == "single_row":
        current = current.iloc[:1]
    else:
        current.loc[0, "prediction"] = np.inf
    result = checker.check_concept_drift(current)
    assert result.status == "unavailable"
    assert result.reason
    assert result.drift_detected is False


def test_zero_threshold_does_not_flag_identical_performance():
    frame = _frame()
    checker = _checker(frame)
    checker.config.thresholds.concept_drift_accuracy_drop = 0.0
    assert checker.check_concept_drift(frame).drift_detected is False


def test_unavailable_evidence_survives_trigger_summary(caplog):
    checker = _checker(_frame())
    results = checker.run_all_checks(pd.DataFrame({"feature": np.arange(100, dtype=float)}))
    summary = PipelineTrigger(checker.config, dry_run=True).evaluate(results)
    assert summary["assessment_complete"] is False
    assert summary["should_trigger"] is False
    assert summary["execution"]["reason"] == "incomplete_evidence"
    assert summary["check_statuses"]["concept"]["status"] == "unavailable"
    assert "No drift detected across all checks" not in caplog.text


def test_unavailable_result_cannot_trigger_even_with_stale_boolean():
    result = DriftResult(drift_type="concept", drift_detected=True, status="unavailable")
    summary = PipelineTrigger(DriftConfig(), dry_run=True).evaluate([result])
    assert summary["should_trigger"] is False
    assert summary["assessment_complete"] is False


def test_yaml_loads_task_and_metric_without_reinterpreting_threshold(tmp_path):
    path = tmp_path / "drift.yaml"
    path.write_text("task_type: regression\nconcept_metric: mae\nthresholds:\n  concept_drift_accuracy_drop: 2.5\n")
    config = DriftConfig.from_yaml(str(path))
    assert config.task_type == "regression"
    assert config.concept_metric == "mae"
    assert config.get_threshold("concept") == 2.5


@pytest.mark.parametrize(
    "task,metric", [("forecasting", None), ("classification", "rmse"), ("clustering", "accuracy")],
)
def test_invalid_task_metric_contract_is_rejected(task, metric):
    with pytest.raises(ValueError):
        DriftConfig(task_type=task, concept_metric=metric)
