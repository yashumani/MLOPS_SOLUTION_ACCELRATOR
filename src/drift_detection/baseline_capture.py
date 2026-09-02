"""Capture and persist a baseline (reference) dataset for drift detection.

The ``BaselineCapture`` class accepts a training ``DataFrame``, computes
reference statistics with Evidently's ``DataDriftPreset``, and serialises
everything to disk so ``DriftChecker`` can load it later.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .drift_config import DriftConfig
from .evidently_compat import ColumnMapping, DataDriftPreset, Report

logger = logging.getLogger(__name__)


class BaselineCapture:
    """Compute and persist a reference baseline from training data.

    Parameters
    ----------
    config : DriftConfig
        Drift-detection configuration (thresholds, paths, column mapping).

    Example
    -------
    >>> cfg = DriftConfig.from_yaml()
    >>> bc = BaselineCapture(cfg)
    >>> bc.capture(train_df)
    >>> bc.save()
    """

    def __init__(self, config: DriftConfig) -> None:
        self.config = config
        self._reference_df: Optional[pd.DataFrame] = None
        self._prediction_stats: Dict[str, Any] = {}
        self._label_stats: Dict[str, Any] = {}
        self._feature_stats: Dict[str, Any] = {}
        self._capture_timestamp: Optional[str] = None

    # ── Public API ──────────────────────────────────────────────

    def capture(self, training_df: pd.DataFrame) -> None:
        """Compute reference statistics from the training set.

        Parameters
        ----------
        training_df : pd.DataFrame
            The training dataset used to establish the baseline.
        """
        if training_df.empty:
            raise ValueError("Cannot capture baseline from an empty DataFrame.")

        self._reference_df = training_df.copy()
        self._capture_timestamp = datetime.now(timezone.utc).isoformat()

        self._compute_feature_stats(training_df)
        self._compute_prediction_stats(training_df)
        self._compute_label_stats(training_df)

        logger.info(
            "Baseline captured — %d rows, %d columns, ts=%s",
            len(training_df),
            training_df.shape[1],
            self._capture_timestamp,
        )

    def save(self, output_dir: Optional[str] = None) -> Path:
        """Persist the reference dataset and statistics to disk.

        Parameters
        ----------
        output_dir : str or None
            Directory to write artefacts to.  Defaults to
            ``config.artifact_paths.baseline_dir``.

        Returns
        -------
        Path
            The directory that was written.
        """
        if self._reference_df is None:
            raise RuntimeError("Call .capture() before .save().")

        out = Path(output_dir or self.config.artifact_paths.baseline_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Reference data
        ref_path = out / "reference_data.parquet"
        self._reference_df.to_parquet(ref_path, index=False)
        logger.info("Reference data  → %s", ref_path)

        # 2. Stats payload
        stats = {
            "capture_timestamp": self._capture_timestamp,
            "n_rows": len(self._reference_df),
            "n_columns": self._reference_df.shape[1],
            "feature_stats": self._feature_stats,
            "prediction_stats": self._prediction_stats,
            "label_stats": self._label_stats,
        }
        stats_path = out / "baseline_stats.json"
        with open(stats_path, "w") as fh:
            json.dump(stats, fh, indent=2, default=str)
        logger.info("Baseline stats  → %s", stats_path)

        return out

    @classmethod
    def load(cls, config: DriftConfig, baseline_dir: Optional[str] = None) -> "BaselineCapture":
        """Restore a previously-saved baseline from disk.

        Parameters
        ----------
        config : DriftConfig
            Drift-detection configuration.
        baseline_dir : str or None
            Override directory.  Defaults to ``config.artifact_paths.baseline_dir``.

        Returns
        -------
        BaselineCapture
            A hydrated instance ready for comparison.
        """
        base = Path(baseline_dir or config.artifact_paths.baseline_dir)
        ref_path = base / "reference_data.parquet"
        stats_path = base / "baseline_stats.json"

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference data not found: {ref_path}")

        instance = cls(config)
        instance._reference_df = pd.read_parquet(ref_path)

        if stats_path.exists():
            with open(stats_path, "r") as fh:
                stats = json.load(fh)
            instance._capture_timestamp = stats.get("capture_timestamp")
            instance._feature_stats = stats.get("feature_stats", {})
            instance._prediction_stats = stats.get("prediction_stats", {})
            instance._label_stats = stats.get("label_stats", {})
        else:
            logger.warning("Stats file missing at %s — partial load", stats_path)

        logger.info("Baseline loaded from %s (%d rows)", base, len(instance._reference_df))
        return instance

    @property
    def reference_df(self) -> pd.DataFrame:
        """Return the stored reference DataFrame (raises if not loaded)."""
        if self._reference_df is None:
            raise RuntimeError("No baseline captured or loaded.")
        return self._reference_df

    @property
    def feature_columns(self) -> List[str]:
        """Feature column names (excludes prediction/target/id)."""
        exclude = {
            self.config.column_mapping.prediction_column,
            self.config.column_mapping.target_column,
            self.config.column_mapping.id_column,
        }
        return [c for c in self.reference_df.columns if c not in exclude]

    # ── Evidently report (for callers needing raw report) ──────

    def build_evidently_report(self, current_df: pd.DataFrame) -> Report:
        """Run an Evidently ``DataDriftPreset`` against the baseline.

        Parameters
        ----------
        current_df : pd.DataFrame
            Production data to compare.

        Returns
        -------
        evidently.report.Report
        """
        col_mapping = ColumnMapping(
            prediction=self.config.column_mapping.prediction_column,
            target=self.config.column_mapping.target_column,
            id=self.config.column_mapping.id_column,
        )

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_df, current_data=current_df, column_mapping=col_mapping)
        return report

    # ── Internal helpers ────────────────────────────────────────

    def _compute_feature_stats(self, df: pd.DataFrame) -> None:
        """Compute per-feature summary statistics."""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self._feature_stats = {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "missing_rate": float(df[col].isna().mean()),
            }
            for col in numeric_cols
        }

    def _compute_prediction_stats(self, df: pd.DataFrame) -> None:
        """Compute prediction-column distribution stats."""
        pred_col = self.config.column_mapping.prediction_column
        if pred_col and pred_col in df.columns:
            self._prediction_stats = {
                "mean": float(df[pred_col].mean()),
                "std": float(df[pred_col].std()),
                "value_counts": df[pred_col].value_counts().to_dict(),
            }

    def _compute_label_stats(self, df: pd.DataFrame) -> None:
        """Compute target/label-column distribution stats."""
        tgt_col = self.config.column_mapping.target_column
        if tgt_col and tgt_col in df.columns:
            self._label_stats = {
                "value_counts": df[tgt_col].value_counts().to_dict(),
                "unique_count": int(df[tgt_col].nunique()),
                "missing_rate": float(df[tgt_col].isna().mean()),
            }
