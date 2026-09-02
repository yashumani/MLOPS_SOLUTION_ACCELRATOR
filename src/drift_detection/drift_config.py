"""Drift detection configuration — thresholds, methods, and scheduling.

Loads ``configs/drift_config.yaml`` and exposes typed dataclasses so every
other module can import a single ``DriftConfig`` instead of parsing YAML
independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Default config path, relative to repository root
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "drift_config.yaml"


@dataclass
class DriftMethods:
    """Statistical method to use for each drift type."""

    feature: str = "psi"
    prediction: str = "ks"
    concept: str = "accuracy_threshold"
    label: str = "chi_square"


@dataclass
class DriftThresholds:
    """Numeric thresholds that trigger drift alerts."""

    feature_drift: float = 0.15
    prediction_drift: float = 0.10
    concept_drift_accuracy_drop: float = 0.05
    label_drift: float = 0.10


@dataclass
class DriftSchedule:
    """When drift checks run."""

    frequency: str = "daily"
    time: str = "02:00"


@dataclass
class DriftActions:
    """What happens when drift is detected."""

    on_drift_detected: str = "trigger_full_pipeline"
    alert_channels: List[str] = field(default_factory=lambda: ["email", "mlflow_dashboard"])


@dataclass
class AutoRetrainConfig:
    """How drift-triggered retraining is submitted.

    Automatic submission is intentionally disabled by default. Operators must
    opt in with ``enabled: true`` and ``mode: submit`` before the trigger will
    invoke the Azure ML submission entrypoint.
    """

    enabled: bool = False
    mode: str = "dry_run"
    config_path: Optional[str] = None
    subscription_id: Optional[str] = None
    resource_group: Optional[str] = None
    workspace_name: Optional[str] = None
    compute: Optional[str] = None
    experiment_name: Optional[str] = None
    display_name_prefix: str = "auto_retrain"
    drift_baseline_in: Optional[str] = None
    wait: bool = False
    stop_compute: bool = False
    force: bool = False
    timeout_seconds: int = 1800
    extra_args: List[str] = field(default_factory=list)


@dataclass
class ArtifactPaths:
    """Where drift artifacts are stored."""

    baseline_dir: str = "outputs/drift_baseline"
    reports_dir: str = "outputs/drift_reports"
    logs_dir: str = "outputs/drift_logs"


@dataclass
class ColumnMapping:
    """Column name conventions for production data."""

    prediction_column: str = "prediction"
    target_column: Optional[str] = None
    id_column: Optional[str] = None


@dataclass
class DriftConfig:
    """Top-level configuration aggregating all drift-detection settings.

    Usage::

        cfg = DriftConfig.from_yaml()           # loads default config
        cfg = DriftConfig.from_yaml("my.yaml")  # loads custom file
        print(cfg.thresholds.feature_drift)
    """

    methods: DriftMethods = field(default_factory=DriftMethods)
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    schedule: DriftSchedule = field(default_factory=DriftSchedule)
    actions: DriftActions = field(default_factory=DriftActions)
    auto_retrain: AutoRetrainConfig = field(default_factory=AutoRetrainConfig)
    artifact_paths: ArtifactPaths = field(default_factory=ArtifactPaths)
    column_mapping: ColumnMapping = field(default_factory=ColumnMapping)

    # ── Factory ─────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "DriftConfig":
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str or None
            Path to the YAML config file.  Falls back to the repository
            default at ``configs/drift_config.yaml``.

        Returns
        -------
        DriftConfig
        """
        config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
        if not config_path.exists():
            logger.warning("Config file not found at %s — using defaults", config_path)
            return cls()

        with open(config_path, "r") as fh:
            raw: Dict = yaml.safe_load(fh) or {}

        auto_retrain_raw = raw.get("auto_retrain", {}) or {}
        auto_retrain_fields = AutoRetrainConfig.__dataclass_fields__
        unknown_auto_retrain = set(auto_retrain_raw) - set(auto_retrain_fields)
        if unknown_auto_retrain:
            logger.warning(
                "Ignoring unknown auto_retrain config keys: %s",
                sorted(unknown_auto_retrain),
            )

        return cls(
            methods=DriftMethods(**raw.get("drift_methods", {})),
            thresholds=DriftThresholds(**raw.get("thresholds", {})),
            schedule=DriftSchedule(**raw.get("schedule", {})),
            actions=DriftActions(**raw.get("actions", {})),
            auto_retrain=AutoRetrainConfig(**{
                key: value
                for key, value in auto_retrain_raw.items()
                if key in auto_retrain_fields
            }),
            artifact_paths=ArtifactPaths(**raw.get("artifact_paths", {})),
            column_mapping=ColumnMapping(**raw.get("column_mapping", {})),
        )

    # ── Helpers ─────────────────────────────────────────────────

    def get_threshold(self, drift_type: str) -> float:
        """Return the numeric threshold for a given drift type.

        Parameters
        ----------
        drift_type : str
            One of ``feature``, ``prediction``, ``concept``, ``label``.

        Returns
        -------
        float
        """
        mapping = {
            "feature": self.thresholds.feature_drift,
            "prediction": self.thresholds.prediction_drift,
            "concept": self.thresholds.concept_drift_accuracy_drop,
            "label": self.thresholds.label_drift,
        }
        if drift_type not in mapping:
            raise ValueError(f"Unknown drift type: {drift_type!r}. Choose from {list(mapping)}")
        return mapping[drift_type]

    def get_method(self, drift_type: str) -> str:
        """Return the statistical method for a given drift type.

        Parameters
        ----------
        drift_type : str
            One of ``feature``, ``prediction``, ``concept``, ``label``.

        Returns
        -------
        str
        """
        mapping = {
            "feature": self.methods.feature,
            "prediction": self.methods.prediction,
            "concept": self.methods.concept,
            "label": self.methods.label,
        }
        if drift_type not in mapping:
            raise ValueError(f"Unknown drift type: {drift_type!r}. Choose from {list(mapping)}")
        return mapping[drift_type]
