"""Generate Evidently HTML reports and persist them as artefacts.

Wraps Evidently's ``Report`` so the rest of the codebase only needs to
call ``ReportGenerator.generate()`` to get a full data-drift report
saved to the configured artefact directory.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

from .drift_config import DriftConfig

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Create and save Evidently HTML drift reports.

    Parameters
    ----------
    config : DriftConfig
        Drift-detection configuration (used for column mapping and paths).

    Example
    -------
    >>> gen = ReportGenerator(cfg)
    >>> path = gen.generate(reference_df, current_df)
    >>> print(f"Report at {path}")
    """

    def __init__(self, config: DriftConfig) -> None:
        self.config = config

    def generate(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        output_dir: Optional[str] = None,
        report_name: Optional[str] = None,
    ) -> Path:
        """Build an Evidently drift report and save it as HTML.

        Parameters
        ----------
        reference_df : pd.DataFrame
            Baseline / training data.
        current_df : pd.DataFrame
            Production / inference data.
        output_dir : str or None
            Override directory.  Defaults to ``config.artifact_paths.reports_dir``.
        report_name : str or None
            Custom file name (without extension).  Defaults to a timestamped
            name like ``drift_report_20250101T120000.html``.

        Returns
        -------
        Path
            Absolute path to the saved HTML file.
        """
        col_mapping = ColumnMapping(
            prediction=self.config.column_mapping.prediction_column,
            target=self.config.column_mapping.target_column,
            id=self.config.column_mapping.id_column,
        )

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=col_mapping,
        )

        out = Path(output_dir or self.config.artifact_paths.reports_dir)
        out.mkdir(parents=True, exist_ok=True)

        ts_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        name = report_name or f"drift_report_{ts_slug}"
        html_path = out / f"{name}.html"

        report.save_html(str(html_path))
        logger.info("Evidently report saved → %s", html_path)

        return html_path
