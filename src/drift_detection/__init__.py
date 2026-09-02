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

from .drift_config import DriftConfig
from .pipeline_trigger import PipelineTrigger


def __getattr__(name: str):
    """Lazily load optional Evidently-dependent helpers.

    Azure ML pipeline step s13 imports ``drift_detection.drift_config`` and
    ``drift_detection.pipeline_trigger``. Eager imports here would pull in
    Evidently-only modules and break environments where trigger evaluation is
    needed but Evidently is not installed.
    """
    if name == "BaselineCapture":
        from .baseline_capture import BaselineCapture

        return BaselineCapture
    if name in {"DriftChecker", "DriftResult"}:
        from .drift_checker import DriftChecker, DriftResult

        return {"DriftChecker": DriftChecker, "DriftResult": DriftResult}[name]
    if name == "ReportGenerator":
        from .report_generator import ReportGenerator

        return ReportGenerator
    if name == "generate_drifted_data":
        from .synthetic_data_generator import generate_drifted_data

        return generate_drifted_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaselineCapture",
    "DriftChecker",
    "DriftConfig",
    "DriftResult",
    "PipelineTrigger",
    "ReportGenerator",
    "generate_drifted_data",
]
