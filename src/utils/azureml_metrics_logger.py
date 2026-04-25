"""
Azure ML Metrics Logger Utility - SDK v2 Compatible (Hardened V5)

Provides MLflow-based logging for Azure ML SDK v2 component jobs.
Azure ML automatically captures MLflow metrics in component runs.

KEY DESIGN PRINCIPLES:
1. Convert azureml:// tracking URIs to https:// before MLflow operations
2. ALWAYS write artifacts to outputs/ first (source of truth for Azure ML Studio)
3. NEVER let logging failures crash the pipeline step
4. Start nested runs if an active run already exists
5. MLflow artifacts are best-effort only (azureml:// scheme unsupported)

Usage:
    from utils.azureml_metrics_logger import create_metrics_logger

    logger = create_metrics_logger(run_name="s01_ingestion", tags={"step": "s01"})
    logger.log_metric("accuracy", 0.95)
    logger.log_param("model_type", "xgboost")
    logger.write_json_to_outputs("report.json", {"rows": 1000})
    logger.end_run()
"""

import mlflow
from typing import Any, Dict, Optional
import os
import logging
import warnings
import json
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OUTPUTS_DIR = Path("outputs")

_SUPPORTED_ARTIFACT_SCHEMES = {"", "file", "http", "https", "databricks",
                                "databricks-uc", "s3", "gs", "wasbs", "dbfs"}


def normalize_mlflow_tracking_uri() -> None:
    """Normalize Azure ML tracking URIs for MLflow client compatibility."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if uri.startswith("azureml://"):
        https_uri = uri.replace("azureml://", "https://", 1)
        os.environ["MLFLOW_TRACKING_URI"] = https_uri
        mlflow.set_tracking_uri(https_uri)
        print("🔗 MLflow tracking URI converted to HTTPS")


def _artifact_logging_supported() -> bool:
    """Return True if the current MLflow artifact URI scheme is known-safe."""
    try:
        uri = mlflow.get_artifact_uri() or ""
        scheme = uri.split("://")[0].lower() if "://" in uri else ""
        return scheme in _SUPPORTED_ARTIFACT_SCHEMES
    except Exception:
        return False


def ensure_outputs_dir() -> Path:
    """Create and return the outputs/ directory (Azure ML Studio source of truth)."""
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUTS_DIR


def safe_write_json(path: Path, obj: Any) -> bool:
    """Write JSON to *path*; never throw."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        return True
    except Exception as exc:
        print(f"⚠️  safe_write_json({path}): {exc}")
        return False


def safe_write_csv(path: Path, df: "pd.DataFrame") -> bool:
    """Write DataFrame CSV to *path*; never throw."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return True
    except Exception as exc:
        print(f"⚠️  safe_write_csv({path}): {exc}")
        return False


def safe_copy(src: Path, dst: Path) -> bool:
    """Copy file from src to dst; never throw."""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return True
    except Exception as exc:
        print(f"⚠️  safe_copy({src} → {dst}): {exc}")
        return False


def safe_dict_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely call .get() on *obj*; return *default* if obj is not a dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


# ---------------------------------------------------------------------------
# Logger class
# ---------------------------------------------------------------------------

class AzureMLMetricsLogger:
    """Logger that uses MLflow for Azure ML SDK v2 component jobs.

    Azure ML SDK v2 automatically captures MLflow metrics and artifacts
    when ``enableMLflowTracking: true`` is set in component YAML.

    This logger:
      * Starts a NEW run, or a NESTED run if one is already active.
      * Logs metrics / params via safe wrappers (never throws).
      * Always writes artifacts to ``outputs/`` (Azure ML Studio).
      * Attempts ``mlflow.log_artifact`` only when the scheme is supported.
    """

    def __init__(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        self.run_name = run_name
        self.tags = tags or {}
        self.mlflow_active = False
        self._owns_run = False          # True if *we* started the run
        self.metrics_buffer: Dict[str, float] = {}
        self.params_buffer: Dict[str, str] = {}

        # Suppress noisy MLflow logs
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

        # Ensure outputs/ always exists
        ensure_outputs_dir()

        try:
            # Disable autologging that might interfere
            try:
                normalize_mlflow_tracking_uri()
                mlflow.autolog(disable=True)
            except Exception:
                pass

            active = mlflow.active_run()
            if active is not None:
                # Another run is already open → start nested
                mlflow.start_run(run_name=run_name, nested=True)
                self._owns_run = True
            else:
                mlflow.start_run(run_name=run_name)
                self._owns_run = True

            if self.tags:
                mlflow.set_tags(self.tags)
            self.mlflow_active = True
            print(f"✅ MLflow run started: {run_name} (nested={active is not None})")
        except Exception as exc:
            print(f"⚠️  MLflow run setup failed (non-critical): {exc}")

    # ------------------------------------------------------------------
    # Metric / Param helpers
    # ------------------------------------------------------------------

    def log_metric(self, key: str, value: float) -> None:
        """Log a numeric metric. Never throws."""
        try:
            val = float(value)
            if self.mlflow_active:
                mlflow.log_metric(key, val)
            self.metrics_buffer[key] = val
        except Exception as exc:
            print(f"⚠️  log_metric('{key}'): {exc}")

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter. Never throws."""
        try:
            val_str = str(value)[:500]  # MLflow param limit
            if self.mlflow_active:
                mlflow.log_param(key, val_str)
            self.params_buffer[key] = val_str
        except Exception as exc:
            print(f"⚠️  log_param('{key}'): {exc}")

    # ------------------------------------------------------------------
    # Artifact helpers — outputs/ first, MLflow second (best-effort)
    # ------------------------------------------------------------------

    def log_dict(self, dictionary: Dict[str, Any], artifact_file: str) -> None:
        """Write dict as JSON to outputs/ and optionally to MLflow artifacts."""
        out_path = _OUTPUTS_DIR / artifact_file
        safe_write_json(out_path, dictionary)

        if self.mlflow_active and _artifact_logging_supported():
            try:
                mlflow.log_artifact(str(out_path))
            except Exception as exc:
                print(f"⚠️  mlflow.log_artifact('{artifact_file}'): {exc}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Copy file to outputs/ and optionally log to MLflow artifacts."""
        src = Path(local_path)
        if src.exists():
            safe_copy(src, _OUTPUTS_DIR / src.name)

        if self.mlflow_active and _artifact_logging_supported():
            try:
                mlflow.log_artifact(local_path, artifact_path or "")
            except Exception as exc:
                print(f"⚠️  mlflow.log_artifact('{local_path}'): {exc}")

    # ------------------------------------------------------------------
    # Convenience writers (always land in outputs/)
    # ------------------------------------------------------------------

    def write_json_to_outputs(self, name: str, obj: Any) -> Path:
        """Write *obj* as JSON into outputs/<name>. Returns the path."""
        p = _OUTPUTS_DIR / name
        safe_write_json(p, obj)
        return p

    def write_csv_to_outputs(self, name: str, df: "pd.DataFrame") -> Path:
        """Write DataFrame CSV into outputs/<name>. Returns the path."""
        p = _OUTPUTS_DIR / name
        safe_write_csv(p, df)
        return p

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def end_run(self) -> None:
        """End the MLflow run we started. Never throws."""
        try:
            if self.mlflow_active and self._owns_run:
                mlflow.end_run()
                print(f"✅ MLflow run ended: {self.run_name}")
        except Exception as exc:
            print(f"⚠️  end_run(): {exc}")
        finally:
            self.mlflow_active = False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_metrics_logger(
    run_name: str,
    tags: Optional[Dict[str, str]] = None
) -> AzureMLMetricsLogger:
    """Create a hardened metrics logger.

    Args:
        run_name: Display name for the MLflow run.
        tags: Optional tags dict.

    Returns:
        AzureMLMetricsLogger instance.
    """
    return AzureMLMetricsLogger(run_name=run_name, tags=tags)
