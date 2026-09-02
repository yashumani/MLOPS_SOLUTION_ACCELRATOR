"""Planned auto-retrain schedule catalog for the three-task rotation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PlannedAutoRetrainSchedule:
    """One planned auto-retrain schedule row."""

    task_type: str
    dataset_name: str
    config_name: str
    schedule_name: str
    cadence: str
    cadence_days: int
    decision_mode: str = "candidate_retrain"
    promotion_mode: str = "manual"
    enabled_expected: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PLANNED_AUTO_RETRAIN_SCHEDULES: tuple[PlannedAutoRetrainSchedule, ...] = (
    PlannedAutoRetrainSchedule(
        task_type="regression",
        dataset_name="college",
        config_name="config_regression_college_azureml.yml",
        schedule_name="auto-retrain-regression-college-daily",
        cadence="daily",
        cadence_days=1,
    ),
    PlannedAutoRetrainSchedule(
        task_type="classification",
        dataset_name="telecom_churn",
        config_name="config_classification_telecom_churn_azureml.yml",
        schedule_name="auto-retrain-classification-telecom-churn-daily",
        cadence="daily",
        cadence_days=1,
    ),
    PlannedAutoRetrainSchedule(
        task_type="clustering",
        dataset_name="online_retail_clustering",
        config_name="config_clustering_online_retail_azureml.yml",
        schedule_name="auto-retrain-clustering-online-retail-daily",
        cadence="daily",
        cadence_days=1,
    ),
)


def build_planned_schedules_table(
    *,
    current_task_type: str,
    current_dataset_name: str,
    current_config_name: str,
    current_schedule_name: str | None,
    decision: dict[str, Any],
    input_baseline_uri: str | None,
    promotion_status: str,
) -> dict[str, Any]:
    """Build the planned schedule table embedded in s14 artifacts."""
    normalized_config = _normalize_config_name(current_config_name)
    outcome = str(decision.get("outcome") or "observe_only")
    severity = str(decision.get("severity") or "none")
    should_submit = bool(decision.get("should_submit"))
    rows: list[dict[str, Any]] = []

    for schedule in PLANNED_AUTO_RETRAIN_SCHEDULES:
        row = schedule.as_dict()
        is_current = _is_current_schedule(
            schedule=schedule,
            current_task_type=current_task_type,
            current_dataset_name=current_dataset_name,
            current_config_name=normalized_config,
            current_schedule_name=current_schedule_name,
        )
        row.update(
            {
                "is_current": is_current,
                "outcome": outcome if is_current else None,
                "severity": severity if is_current else None,
                "should_submit": should_submit if is_current else False,
                "promotion_status": promotion_status if is_current else None,
                "input_baseline_uri": input_baseline_uri if is_current else None,
            }
        )
        rows.append(row)

    current_rows = [row for row in rows if row["is_current"]]
    candidate_retrains_pending = sum(
        1
        for row in current_rows
        if row.get("outcome") in {"candidate_retrain", "promote_candidate"}
        and row.get("promotion_status") == "manual_pending"
    )
    baselines_approved = sum(
        1 for row in current_rows if row.get("promotion_status") in {"approved", "baseline_approved", "production"}
    )
    return {
        "schema_version": "1.0",
        "rows": rows,
        "summary": {
            "total_planned_schedules": len(rows),
            "current_rows": len(current_rows),
            "candidate_retrains_pending": candidate_retrains_pending,
            "baselines_approved": baselines_approved,
        },
    }


def _is_current_schedule(
    *,
    schedule: PlannedAutoRetrainSchedule,
    current_task_type: str,
    current_dataset_name: str,
    current_config_name: str,
    current_schedule_name: str | None,
) -> bool:
    if current_schedule_name and current_schedule_name == schedule.schedule_name:
        return True
    return (
        schedule.task_type == current_task_type
        and schedule.dataset_name == current_dataset_name
        and _normalize_config_name(schedule.config_name) == current_config_name
    )


def _normalize_config_name(config_name: str) -> str:
    return config_name if config_name.endswith(".yml") else f"{config_name}.yml"