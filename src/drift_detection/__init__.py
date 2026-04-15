"""Drift detection package for MLOps V3.

Public API
----------
- ``DriftConfig``         — typed configuration from YAML
- ``BaselineCapture``     — capture & persist reference baselines
- ``DriftChecker``        — run 4 drift checks (feature/prediction/concept/label)
- ``DriftResult``         — dataclass for individual check results
- ``PipelineTrigger``     — evaluate results and trigger retraining
- ``ReportGenerator``     — Evidently HTML report artefacts
- ``generate_drifted_data`` — synthetic test data with controlled drift
"""

from .baseline_capture import BaselineCapture
from .drift_checker import DriftChecker, DriftResult
from .drift_config import DriftConfig
from .pipeline_trigger import PipelineTrigger
from .report_generator import ReportGenerator
from .synthetic_data_generator import generate_drifted_data

__all__ = [
    "BaselineCapture",
    "DriftChecker",
    "DriftConfig",
    "DriftResult",
    "PipelineTrigger",
    "ReportGenerator",
    "generate_drifted_data",
]
