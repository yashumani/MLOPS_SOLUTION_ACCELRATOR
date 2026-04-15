"""Run drift checks against a persisted baseline and return results.

Supports four drift types using Evidently:

1. **Feature drift** — ``DataDriftPreset`` with PSI statistic.
2. **Prediction drift** — ``ValueDrift`` (or column-level check) on the
   prediction column using the Kolmogorov–Smirnov test.
3. **Concept drift** — compares model quality metrics (accuracy drop).
4. **Label drift** — chi-squared test on the target column distribution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
)
from evidently.legacy.report import Report

from .baseline_capture import BaselineCapture
from .drift_config import DriftConfig

logger = logging.getLogger(__name__)


# ── Result dataclass ────────────────────────────────────────────

@dataclass
class DriftResult:
    """Container for a single drift-check result.

    Attributes
    ----------
    drift_detected : bool
        ``True`` if the score exceeds its threshold.
    drift_type : str
        One of ``feature``, ``prediction``, ``concept``, ``label``.
    drift_score : float
        Numeric score returned by the statistical test.
    drifted_columns : list[str]
        Names of columns that individually drifted.
    timestamp : str
        ISO-8601 UTC timestamp of when the check ran.
    evidently_report_path : str
        File path to the saved HTML report (empty if not saved).
    details : dict
        Extra per-column or per-metric detail.
    """

    drift_detected: bool = False
    drift_type: str = ""
    drift_score: float = 0.0
    drifted_columns: List[str] = field(default_factory=list)
    timestamp: str = ""
    evidently_report_path: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# ── Checker ─────────────────────────────────────────────────────

class DriftChecker:
    """Compare current production data against a stored baseline.

    Parameters
    ----------
    config : DriftConfig
        Global drift-detection settings.
    baseline : BaselineCapture
        A loaded baseline instance (call ``BaselineCapture.load()`` first).

    Example
    -------
    >>> cfg = DriftConfig.from_yaml()
    >>> bl  = BaselineCapture.load(cfg)
    >>> checker = DriftChecker(cfg, bl)
    >>> results = checker.run_all_checks(production_df)
    >>> for r in results:
    ...     print(r.drift_type, r.drift_detected)
    """

    def __init__(self, config: DriftConfig, baseline: BaselineCapture) -> None:
        self.config = config
        self.baseline = baseline

    # ── Public API ──────────────────────────────────────────────

    def run_all_checks(self, current_df: pd.DataFrame) -> List[DriftResult]:
        """Execute all four drift checks and return a list of results.

        Parameters
        ----------
        current_df : pd.DataFrame
            Production/inference data to evaluate.

        Returns
        -------
        list[DriftResult]
        """
        results: List[DriftResult] = [
            self.check_feature_drift(current_df),
            self.check_prediction_drift(current_df),
            self.check_concept_drift(current_df),
            self.check_label_drift(current_df),
        ]
        drifted = [r for r in results if r.drift_detected]
        if drifted:
            logger.warning("Drift detected in: %s", [r.drift_type for r in drifted])
        else:
            logger.info("No drift detected across all checks.")
        return results

    # ── Individual checks ───────────────────────────────────────

    def check_feature_drift(self, current_df: pd.DataFrame) -> DriftResult:
        """Feature drift using Evidently ``DataDriftPreset`` (PSI by default).

        Parameters
        ----------
        current_df : pd.DataFrame

        Returns
        -------
        DriftResult
        """
        ts = _utc_now()
        col_mapping = self._column_mapping()

        report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
        report.run(
            reference_data=self.baseline.reference_df,
            current_data=current_df,
            column_mapping=col_mapping,
        )

        report_dict = report.as_dict()
        drifted_cols: List[str] = []
        max_score: float = 0.0
        details: Dict[str, Any] = {}

        for metric_result in report_dict.get("metrics", []):
            result_data = metric_result.get("result", {})
            # DatasetDriftMetric stores dataset_drift
            if "dataset_drift" in result_data:
                dataset_drifted = result_data.get("dataset_drift", False)
                drift_share = result_data.get("share_of_drifted_columns", 0.0)
                details["dataset_drift"] = dataset_drifted
                details["drift_share"] = drift_share
            # DataDriftPreset stores per-column info in drift_by_columns
            if "drift_by_columns" in result_data:
                for col_name, col_info in result_data["drift_by_columns"].items():
                    col_score = col_info.get("drift_score", 0.0)
                    if col_info.get("drift_detected", False):
                        drifted_cols.append(col_name)
                    max_score = max(max_score, col_score)
                    details[col_name] = {
                        "score": col_score,
                        "drifted": col_info.get("drift_detected", False),
                    }

        threshold = self.config.get_threshold("feature")
        detected = max_score >= threshold or bool(drifted_cols)

        return DriftResult(
            drift_detected=detected,
            drift_type="feature",
            drift_score=round(max_score, 6),
            drifted_columns=drifted_cols,
            timestamp=ts,
            details=details,
        )

    def check_prediction_drift(self, current_df: pd.DataFrame) -> DriftResult:
        """Prediction-column drift using KS test via Evidently.

        Parameters
        ----------
        current_df : pd.DataFrame

        Returns
        -------
        DriftResult
        """
        ts = _utc_now()
        pred_col = self.config.column_mapping.prediction_column

        if pred_col not in current_df.columns or pred_col not in self.baseline.reference_df.columns:
            logger.info("Prediction column %r missing — skipping prediction drift check.", pred_col)
            return DriftResult(drift_type="prediction", timestamp=ts)

        col_mapping = self._column_mapping()

        report = Report(metrics=[ColumnDriftMetric(column_name=pred_col)])
        report.run(
            reference_data=self.baseline.reference_df,
            current_data=current_df,
            column_mapping=col_mapping,
        )

        report_dict = report.as_dict()
        score: float = 0.0
        detected: bool = False

        for metric_result in report_dict.get("metrics", []):
            result_data = metric_result.get("result", {})
            score = result_data.get("drift_score", 0.0)
            detected = result_data.get("drift_detected", False)

        threshold = self.config.get_threshold("prediction")
        detected = detected or score >= threshold

        return DriftResult(
            drift_detected=detected,
            drift_type="prediction",
            drift_score=round(score, 6),
            drifted_columns=[pred_col] if detected else [],
            timestamp=ts,
        )

    def check_concept_drift(self, current_df: pd.DataFrame) -> DriftResult:
        """Concept drift by comparing accuracy between baseline and current.

        This is a simple metric-comparison approach: if the accuracy on the
        current data drops by more than the configured threshold relative
        to the baseline, concept drift is flagged.

        Parameters
        ----------
        current_df : pd.DataFrame
            Must contain both ``prediction`` and ``target`` columns.

        Returns
        -------
        DriftResult
        """
        ts = _utc_now()
        pred_col = self.config.column_mapping.prediction_column
        tgt_col = self.config.column_mapping.target_column

        if not tgt_col or tgt_col not in current_df.columns or pred_col not in current_df.columns:
            logger.info("Target or prediction column missing — skipping concept drift check.")
            return DriftResult(drift_type="concept", timestamp=ts)

        ref = self.baseline.reference_df
        if tgt_col not in ref.columns or pred_col not in ref.columns:
            logger.info("Baseline lacks target/prediction — skipping concept drift check.")
            return DriftResult(drift_type="concept", timestamp=ts)

        ref_acc = float((ref[pred_col] == ref[tgt_col]).mean())
        cur_acc = float((current_df[pred_col] == current_df[tgt_col]).mean())
        drop = ref_acc - cur_acc

        threshold = self.config.get_threshold("concept")
        detected = drop >= threshold

        return DriftResult(
            drift_detected=detected,
            drift_type="concept",
            drift_score=round(drop, 6),
            drifted_columns=[pred_col, tgt_col] if detected else [],
            timestamp=ts,
            details={
                "reference_accuracy": round(ref_acc, 6),
                "current_accuracy": round(cur_acc, 6),
                "accuracy_drop": round(drop, 6),
            },
        )

    def check_label_drift(self, current_df: pd.DataFrame) -> DriftResult:
        """Label/target distribution drift using chi-squared via Evidently.

        Parameters
        ----------
        current_df : pd.DataFrame

        Returns
        -------
        DriftResult
        """
        ts = _utc_now()
        tgt_col = self.config.column_mapping.target_column

        if not tgt_col or tgt_col not in current_df.columns:
            logger.info("Target column %r missing — skipping label drift check.", tgt_col)
            return DriftResult(drift_type="label", timestamp=ts)

        if tgt_col not in self.baseline.reference_df.columns:
            logger.info("Baseline lacks target column — skipping label drift check.")
            return DriftResult(drift_type="label", timestamp=ts)

        col_mapping = self._column_mapping()

        report = Report(metrics=[ColumnDriftMetric(column_name=tgt_col)])
        report.run(
            reference_data=self.baseline.reference_df,
            current_data=current_df,
            column_mapping=col_mapping,
        )

        report_dict = report.as_dict()
        score: float = 0.0
        detected: bool = False

        for metric_result in report_dict.get("metrics", []):
            result_data = metric_result.get("result", {})
            score = result_data.get("drift_score", 0.0)
            detected = result_data.get("drift_detected", False)

        threshold = self.config.get_threshold("label")
        detected = detected or score >= threshold

        return DriftResult(
            drift_detected=detected,
            drift_type="label",
            drift_score=round(score, 6),
            drifted_columns=[tgt_col] if detected else [],
            timestamp=ts,
        )

    # ── Helpers ─────────────────────────────────────────────────

    def _column_mapping(self) -> ColumnMapping:
        """Build Evidently ``ColumnMapping`` from config."""
        return ColumnMapping(
            prediction=self.config.column_mapping.prediction_column,
            target=self.config.column_mapping.target_column,
            id=self.config.column_mapping.id_column,
        )


def _utc_now() -> str:
    """ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()
