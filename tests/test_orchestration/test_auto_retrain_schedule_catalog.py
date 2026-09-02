"""Tests for the auto-retrain rotation schedule catalog."""

from __future__ import annotations

from orchestration.auto_retrain_schedule_catalog import (
    PLANNED_AUTO_RETRAIN_SCHEDULES,
    build_planned_schedules_table,
)


def test_catalog_contains_the_three_rotation_schedules() -> None:
    rows = [schedule.as_dict() for schedule in PLANNED_AUTO_RETRAIN_SCHEDULES]

    assert [row["task_type"] for row in rows] == ["regression", "classification", "clustering"]
    assert {row["schedule_name"] for row in rows} == {
        "auto-retrain-regression-college-daily",
        "auto-retrain-classification-telecom-churn-daily",
        "auto-retrain-clustering-online-retail-daily",
    }
    assert all(row["decision_mode"] == "candidate_retrain" for row in rows)
    assert all(row["promotion_mode"] == "manual" for row in rows)


def test_planned_schedules_table_marks_current_regression_row() -> None:
    table = build_planned_schedules_table(
        current_task_type="regression",
        current_dataset_name="college",
        current_config_name="config_regression_college_azureml.yml",
        current_schedule_name="auto-retrain-regression-college-daily",
        decision={"outcome": "candidate_retrain", "severity": "severe", "should_submit": True},
        input_baseline_uri="azureml://baseline/",
        promotion_status="manual_pending",
    )

    assert table["summary"] == {
        "total_planned_schedules": 3,
        "current_rows": 1,
        "candidate_retrains_pending": 1,
        "baselines_approved": 0,
    }
    current_rows = [row for row in table["rows"] if row["is_current"]]
    assert len(current_rows) == 1
    assert current_rows[0]["task_type"] == "regression"
    assert current_rows[0]["dataset_name"] == "college"
    assert current_rows[0]["outcome"] == "candidate_retrain"
    assert current_rows[0]["severity"] == "severe"
    assert current_rows[0]["should_submit"] is True
    assert current_rows[0]["input_baseline_uri"] == "azureml://baseline/"


def test_planned_schedules_table_marks_current_row_without_schedule_name() -> None:
    table = build_planned_schedules_table(
        current_task_type="classification",
        current_dataset_name="telecom_churn",
        current_config_name="config_classification_telecom_churn_azureml.yml",
        current_schedule_name=None,
        decision={"outcome": "observe_only", "severity": "none", "should_submit": False},
        input_baseline_uri=None,
        promotion_status="manual_pending",
    )

    current_rows = [row for row in table["rows"] if row["is_current"]]
    assert len(current_rows) == 1
    assert current_rows[0]["schedule_name"] == "auto-retrain-classification-telecom-churn-daily"
    assert current_rows[0]["outcome"] == "observe_only"
    assert current_rows[0]["should_submit"] is False