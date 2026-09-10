"""Task-aware drift evidence against a persisted observation baseline."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from .baseline_capture import BaselineCapture
from .drift_config import DriftConfig
from .evidently_compat import (
    ColumnDriftMetric,
    ColumnMapping,
    DataDriftPreset,
    DatasetDriftMetric,
    Report,
)

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """A statistical result, or an explicit reason the check was not evaluated."""

    drift_detected: bool = False
    drift_type: str = ""
    drift_score: float = 0.0
    drifted_columns: List[str] = field(default_factory=list)
    timestamp: str = ""
    evidently_report_path: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    status: Literal["evaluated", "unavailable", "not_applicable"] = "evaluated"
    reason: str = ""


class DriftChecker:
    """Compare windows from the same monitored model against its baseline.

    The caller owns model/window identity validation. Training-candidate scores
    from different models are not production concept-drift evidence.
    """

    def __init__(self, config: DriftConfig, baseline: BaselineCapture) -> None:
        self.config = config
        self.baseline = baseline

    def run_all_checks(self, current_df: pd.DataFrame) -> List[DriftResult]:
        results = [
            self.check_feature_drift(current_df),
            self.check_prediction_drift(current_df),
            self.check_concept_drift(current_df),
            self.check_label_drift(current_df),
        ]
        drifted = [r.drift_type for r in results if r.status == "evaluated" and r.drift_detected]
        unavailable = [r.drift_type for r in results if r.status == "unavailable"]
        if drifted:
            logger.warning("Drift detected in: %s", drifted)
        if unavailable:
            logger.warning("Drift assessment incomplete; unavailable checks: %s", unavailable)
        elif not drifted:
            logger.info("No drift detected in evaluated, applicable checks.")
        return results

    def check_feature_drift(self, current_df: pd.DataFrame) -> DriftResult:
        columns = self.baseline.feature_columns
        missing = sorted(set(columns) - set(current_df.columns))
        if not columns:
            return self._unavailable("feature", "no_feature_columns")
        if missing:
            return self._unavailable("feature", "missing_feature_columns", columns=missing)
        if current_df.empty:
            return self._unavailable("feature", "empty_observation_window")
        method = self.config.get_method("feature")
        options = {
            "stattest": None if method == "auto" else method,
            "stattest_threshold": self._threshold("feature"),
        }
        report = Report(metrics=[DataDriftPreset(**options), DatasetDriftMetric(**options)])
        report.run(
            reference_data=self.baseline.reference_df[columns],
            current_data=current_df[columns],
            column_mapping=ColumnMapping(target=None, prediction=None),
        )
        drifted_columns = []
        dataset_drifted = False
        drift_share = 0.0
        column_count = 0
        details = {}
        for metric_result in report.as_dict().get("metrics", []):
            data = metric_result.get("result", {})
            if "dataset_drift" in data:
                dataset_drifted = bool(data["dataset_drift"])
                drift_share = float(data.get("share_of_drifted_columns", 0.0) or 0.0)
                details["dataset_drift"] = dataset_drifted
            for name, info in data.get("drift_by_columns", {}).items():
                column_count += 1
                drifted = bool(info.get("drift_detected", False))
                if drifted:
                    drifted_columns.append(name)
                details[name] = {
                    "score": info.get("drift_score"),
                    "drifted": drifted,
                    "stattest_name": info.get("stattest_name"),
                    "threshold": info.get("stattest_threshold"),
                }
        if not details:
            return self._unavailable("feature", "missing_statistical_result")
        if drift_share == 0.0 and column_count:
            drift_share = len(set(drifted_columns)) / column_count
        details.update(drift_share=drift_share, score_type="drifted_feature_share")
        return DriftResult(
            drift_detected=bool(dataset_drifted or drifted_columns),
            drift_type="feature", drift_score=round(drift_share, 6),
            drifted_columns=sorted(set(drifted_columns)), timestamp=_utc_now(),
            details=details,
        )

    def check_prediction_drift(self, current_df: pd.DataFrame) -> DriftResult:
        return self._check_column_drift(
            current_df, "prediction", self.config.column_mapping.prediction_column,
        )

    def check_concept_drift(self, current_df: pd.DataFrame) -> DriftResult:
        """Compare supervised performance, not exact equality for regression."""
        if self.config.task_type == "clustering":
            return self._unavailable("concept", "unlabeled_task", status="not_applicable")
        pred_col = self.config.column_mapping.prediction_column
        tgt_col = self.config.column_mapping.target_column
        ref = self.baseline.reference_df
        if not tgt_col or any(
            column not in frame.columns
            for frame in (ref, current_df) for column in (pred_col, tgt_col)
        ):
            return self._unavailable("concept", "missing_target_or_prediction")
        details = {"reference_rows": len(ref), "current_rows": len(current_df)}
        if min(len(ref), len(current_df)) < 2:
            return self._unavailable("concept", "insufficient_rows", **details)
        if any(frame[[pred_col, tgt_col]].isna().any().any() for frame in (ref, current_df)):
            return self._unavailable("concept", "missing_target_or_prediction_values", **details)
        metric = self.config.concept_metric or (
            "balanced_accuracy" if self.config.task_type == "classification" else "r2"
        )
        scorers = {
            "accuracy": accuracy_score,
            "balanced_accuracy": balanced_accuracy_score,
            "f1_weighted": lambda y, p: f1_score(y, p, average="weighted", zero_division=0),
            "r2": r2_score,
            "mae": mean_absolute_error,
            "mse": mean_squared_error,
            "rmse": lambda y, p: math.sqrt(mean_squared_error(y, p)),
        }
        if metric == "r2" and any(frame[tgt_col].nunique() < 2 for frame in (ref, current_df)):
            return self._unavailable("concept", "constant_target_r2_undefined", **details)
        if metric == "balanced_accuracy" and any(
            frame[tgt_col].nunique() < 2 for frame in (ref, current_df)
        ):
            return self._unavailable("concept", "insufficient_target_classes", **details)
        try:
            reference_score = float(scorers[metric](ref[tgt_col], ref[pred_col]))
            current_score = float(scorers[metric](current_df[tgt_col], current_df[pred_col]))
        except (ValueError, TypeError):
            return self._unavailable("concept", "invalid_metric_inputs", metric_name=metric, **details)
        if not all(math.isfinite(value) for value in (reference_score, current_score)):
            return self._unavailable("concept", "non_finite_metric", metric_name=metric, **details)
        lower_is_better = metric in {"mae", "mse", "rmse"}
        drop = current_score - reference_score if lower_is_better else reference_score - current_score
        threshold = self._threshold("concept")
        details.update({
            "metric_name": metric,
            "metric_direction": "minimize" if lower_is_better else "maximize",
            "reference_score": reference_score,
            "current_score": current_score,
            "degradation": drop,
            "threshold": threshold,
            "evidence_type": "supervised_performance_degradation",
        })
        if metric in {"accuracy", "balanced_accuracy"}:
            details.update(
                reference_accuracy=reference_score,
                current_accuracy=current_score, accuracy_drop=drop,
            )
        detected = drop > threshold
        return DriftResult(
            drift_detected=detected, drift_type="concept",
            drift_score=round(drop, 6),
            drifted_columns=[pred_col, tgt_col] if detected else [],
            timestamp=_utc_now(), details=details,
        )

    def check_label_drift(self, current_df: pd.DataFrame) -> DriftResult:
        if self.config.task_type == "clustering":
            return self._unavailable("label", "unlabeled_task", status="not_applicable")
        return self._check_column_drift(
            current_df, "label", self.config.column_mapping.target_column,
        )

    @staticmethod
    def _unavailable(drift_type, reason, *, status="unavailable", **details) -> DriftResult:
        return DriftResult(
            drift_type=drift_type, timestamp=_utc_now(), status=status,
            reason=reason, details=details,
        )

    def _threshold(self, drift_type: str) -> float:
        threshold = self.config.get_threshold(drift_type)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Drift threshold must be finite and non-negative")
        return threshold

    def _check_column_drift(self, current_df, drift_type, column) -> DriftResult:
        ref = self.baseline.reference_df
        if not column or column not in current_df.columns or column not in ref.columns:
            return self._unavailable(drift_type, "missing_column", column=column)
        if current_df[column].dropna().empty or ref[column].dropna().empty:
            return self._unavailable(drift_type, "no_observed_values", column=column)
        method = self.config.get_method(drift_type)
        method = {"auto": None, "chi_square": "chisquare"}.get(method, method)
        threshold = self._threshold(drift_type)
        report = Report(metrics=[ColumnDriftMetric(
            column_name=column, stattest=method, stattest_threshold=threshold,
        )])
        numeric = self.config.task_type == "regression"
        report.run(
            reference_data=ref[[column]], current_data=current_df[[column]],
            column_mapping=ColumnMapping(
                target=None, prediction=None,
                numerical_features=[column] if numeric else [],
                categorical_features=[] if numeric else [column],
            ),
        )
        for result in report.as_dict().get("metrics", []):
            data = result.get("result", {})
            if "drift_detected" not in data or data.get("column_name", column) != column:
                continue
            try:
                score = float(data["drift_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(score):
                continue
            # Evidently owns score direction: p-values and distances differ.
            detected = bool(data["drift_detected"])
            return DriftResult(
                drift_type=drift_type, drift_score=score, drift_detected=detected,
                drifted_columns=[column] if detected else [], timestamp=_utc_now(),
                details={
                    "stattest_name": data.get("stattest_name"),
                    "threshold": data.get("stattest_threshold", threshold),
                    "reference_rows": len(ref), "current_rows": len(current_df),
                    "reference_observed_rows": int(ref[column].notna().sum()),
                    "current_observed_rows": int(current_df[column].notna().sum()),
                },
            )
        return self._unavailable(drift_type, "missing_statistical_result", column=column)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
