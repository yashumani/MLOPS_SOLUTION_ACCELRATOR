"""Evidently imports shared across supported runtime versions."""

try:
    # Evidently 0.4.x, pinned by the active Azure ML unified environment.
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
    from evidently.pipeline.column_mapping import ColumnMapping
    from evidently.report import Report
except ImportError:
    # Evidently >=0.5 exposes the previous report API under ``legacy``.
    from evidently.legacy.metric_preset import DataDriftPreset
    from evidently.legacy.metrics import ColumnDriftMetric, DatasetDriftMetric
    from evidently.legacy.pipeline.column_mapping import ColumnMapping
    from evidently.legacy.report import Report

__all__ = [
    "ColumnDriftMetric",
    "ColumnMapping",
    "DataDriftPreset",
    "DatasetDriftMetric",
    "Report",
]
