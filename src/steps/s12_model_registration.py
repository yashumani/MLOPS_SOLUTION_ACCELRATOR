"""
Stage 12: Model Registry Integration

Registers champion model from final evaluation to MLflow Model Registry.
Creates versioned model artifact with full lineage, metadata, and stage promotion.

Features:
- Model versioning (auto-increments version on each registration)
- Stage management ("Staging" for validation, "Production" after approval)
- Rich metadata (dataset, task_type, algorithm, metrics, recipe)
- Model signature (input/output schema for serving validation)
- Artifact tracking (links model to source experiment run)

Exit Codes:
- 0: Model successfully registered
- 1: Registration failed (manifest not found, model invalid)
"""

import argparse
import json
import logging
import os
import time as _time_mod
from pathlib import Path
from typing import Dict, Any

import mlflow
import numpy as np
import yaml
from mlflow.tracking import MlflowClient

# T11/T12: Import metrics logger + safe autolog disable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.azureml_metrics_logger import create_metrics_logger

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """T16: Handle numpy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return v if np.isfinite(v) else None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _safe_disable_autolog():
    """T12: Disable autolog + convert azureml:// tracking URI to https://."""
    try:
        mlflow.autolog(disable=True)
    except Exception as e:
        logger.debug("mlflow.autolog(disable=True) failed: %s", e)
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if uri.startswith("azureml://"):
        mlflow.set_tracking_uri(uri.replace("azureml://", "https://"))
        logger.info("🔗 MLflow tracking URI converted to HTTPS")


class ModelRegistry:
    """
    MLflow Model Registry integration for champion model tracking.
    
    Handles model versioning, stage transitions, and metadata management
    for production model lifecycle.
    """
    
    def __init__(self, config_name: str, cfg: Dict[str, Any] = None):
        self.config_name = config_name
        # K11 fix: prefer cfg['dataset']['name'] (and cfg['registry']['model_name'])
        # over fragile filename parsing. cfg may be None for legacy callers.
        self.cfg = cfg or {}
        
        # 🔥 FIX: Convert azureml:// to https:// BEFORE creating MlflowClient
        # The MlflowClient() constructor captures the tracking URI at creation time.
        # We must also update the env var so that mlflow.sklearn.log_model() internals
        # (which read MLFLOW_TRACKING_URI directly) also get the corrected URI.
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        if mlflow_uri.startswith("azureml://"):
            https_uri = mlflow_uri.replace("azureml://", "https://")
            os.environ["MLFLOW_TRACKING_URI"] = https_uri
            mlflow.set_tracking_uri(https_uri)
            logger.info(f"🔗 MLflow tracking URI converted to HTTPS for model registry")
        
        self.client = MlflowClient()
        
        # Extract dataset name from cfg first, fall back to filename parsing
        self.dataset_name = self._extract_dataset_name(config_name)
        # K11: also expose explicit registry model name override if cfg provides one.
        self.model_name_override = (self.cfg.get("registry", {}) or {}).get("model_name")
    
    def _extract_dataset_name(self, config_name: str) -> str:
        """Resolve dataset name with the following precedence:
        1) cfg['dataset']['name']           (explicit, preferred)
        2) cfg['dataset']['display_name']   (alternate explicit field)
        3) parsed from config filename       (legacy fallback)
        4) 'unknown'                         (last resort)
        """
        ds_cfg = (self.cfg.get("dataset") or {}) if hasattr(self, "cfg") else {}
        for key in ("name", "display_name", "dataset_name"):
            v = ds_cfg.get(key)
            if v and isinstance(v, str) and v.strip():
                logger.info(f"📒 K11: dataset name from cfg['dataset']['{key}'] = {v}")
                return v.strip()
        # Filename fallback (legacy behaviour)
        # config_classification_telecom_churn_azureml.yml -> telecom_churn
        parts = config_name.replace(".yml", "").split("_")
        if len(parts) >= 3:
            inferred = "_".join(parts[2:]).replace("_azureml", "").replace("_local", "")
            logger.info(f"📒 K11: dataset name parsed from filename = {inferred}")
            return inferred
        logger.warning(f"⚠️ K11: could not resolve dataset name from cfg or filename '{config_name}'")
        return "unknown"
    
    def register_champion_model(
        self,
        manifest: Dict[str, Any],
        model_path: Path
    ) -> Dict[str, Any]:
        """
        Register champion model to MLflow Model Registry.
        
        Args:
            manifest: Champion manifest from final evaluation
            model_path: Path to champion model artifact directory
            
        Returns:
            Registry info dict with model name, version, stage
        """
        # DESIGN DECISION (R4 audit 2026-02): s12 receives s10's final_report
        # (not s06's ChampionManifest).  The report uses "task" instead of
        # "task_type" and splits metrics by phase.  We normalise here so the
        # registry logic works regardless of which manifest schema is passed.
        task_type = manifest.get("task_type") or manifest.get("task", "unknown")
        algorithm = manifest.get("algorithm", "unknown")
        
        # If algorithm not at root, try to infer from selection phase metrics
        if algorithm == "unknown":
            selection = manifest.get("selection", {})
            phase_key = selection.get("key", "")
            phase_metrics = manifest.get(f"{phase_key}_metrics", {})
            algorithm = phase_metrics.get("algorithm", "unknown")
        
        # Model name: {dataset}_{task}_mlops
        model_name = f"{self.dataset_name}_{task_type}_mlops"
        # K11: allow explicit override from cfg['registry']['model_name'].
        if getattr(self, "model_name_override", None):
            model_name = self.model_name_override
            logger.info(f"📛 K11: using cfg-provided registry model name = {model_name}")
        
        logger.info(f"📦 Registering model: {model_name}")
        logger.info(f"Algorithm: {algorithm}, Metrics: {manifest.get('metrics', {})}")
        
        # Find model artifact (PyCaret saves as .pkl, FLAML as .pkl or model/)
        model_file = self._find_model_artifact(model_path)
        
        if not model_file:
            logger.error(f"❌ No model artifact found in {model_path}")
            raise FileNotFoundError(f"Model artifact not found in {model_path}")
        
        logger.info(f"Found model artifact: {model_file}")
        
        # Register model with MLflow
        try:
            # Load model from disk — required by mlflow.sklearn.log_model()
            import joblib
            logger.info(f"Loading model from {model_file}")
            sk_model = joblib.load(str(model_file))
            logger.info(f"Model loaded: {type(sk_model).__name__}")

            # 🔥 FIX: Ensure an active MLflow run exists.
            # Azure ML sets enableMLflowTracking=true but the auto-started run
            # may not survive the HTTPS URI conversion.  Without an active run,
            # the old code fell back to register_model(file://...) which Azure ML
            # rejects with INVALID_PARAMETER_VALUE.  Starting a run guarantees
            # log_model() creates a proper azureml://artifacts/... URI.
            active_run = mlflow.active_run()
            if not active_run:
                active_run = mlflow.start_run(run_name=f"s12_register_{model_name}")
                logger.info(f"🆕 Started MLflow run: {active_run.info.run_id}")
            
            run_id = active_run.info.run_id
            logger.info(f"Logging model to run {run_id}")

            # Log model artifact into the active run AND register in one call.
            # mlflow.sklearn.log_model with registered_model_name creates an
            # azureml://artifacts/... URI that the model registry accepts.
            try:
                mlflow.sklearn.log_model(
                    sk_model=sk_model,
                    artifact_path="champion_model",
                    registered_model_name=model_name
                )
            except Exception as log_err:
                # Fallback for non-sklearn models (e.g. LightGBM, XGBoost native)
                logger.warning(f"sklearn.log_model failed ({log_err}), trying pyfunc")
                mlflow.pyfunc.log_model(
                    artifact_path="champion_model",
                    loader_module="mlflow.sklearn",
                    data_path=str(model_file),
                    registered_model_name=model_name
                )
            
            # Get latest version
            latest_versions = self.client.get_latest_versions(model_name, stages=["None"])
            if not latest_versions:
                # Retry with "Staging" — some backends auto-promote
                latest_versions = self.client.get_latest_versions(model_name)
            if not latest_versions:
                logger.error(f"❌ Model registration failed - no versions found")
                raise RuntimeError("Model registration failed")
            
            model_version = latest_versions[0].version
            logger.info(f"✅ Model registered as version {model_version}")
            
            # Add metadata tags
            self._add_model_metadata(model_name, model_version, manifest)
            
            # Transition to Staging stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Staging",
                archive_existing_versions=False
            )
            logger.info(f"📈 Model promoted to 'Staging' stage")
            
            # Build registry info
            registry_info = {
                "model_name": model_name,
                "version": model_version,
                "stage": "Staging",
                "algorithm": algorithm,
                "task_type": task_type,
                "metrics": manifest.get("metrics", {}),
                "dataset": self.dataset_name,
                "config": self.config_name
            }
            
            return registry_info
        
        except Exception as e:
            logger.error(f"❌ Model registration failed: {str(e)}")
            raise
    
    def _find_model_artifact(self, model_path: Path) -> Path:
        """Find model artifact file in directory."""
        # Look for .pkl files (PyCaret, FLAML)
        pkl_files = list(model_path.glob("*.pkl"))
        if pkl_files:
            return pkl_files[0]
        
        # Look for model subdirectory (FLAML sometimes creates model/)
        model_dir = model_path / "model"
        if model_dir.exists():
            pkl_files = list(model_dir.glob("*.pkl"))
            if pkl_files:
                return pkl_files[0]
        
        # Look for .joblib files (scikit-learn)
        joblib_files = list(model_path.glob("*.joblib"))
        if joblib_files:
            return joblib_files[0]
        
        return None
    
    def _add_model_metadata(
        self,
        model_name: str,
        model_version: str,
        manifest: Dict[str, Any]
    ):
        """Add metadata tags to registered model version."""
        # Normalise field names: s10 report uses "task"/"selection" vs s06's
        # "task_type"/"phase"/"recipe".
        selection = manifest.get("selection", {})
        champion_phase = manifest.get("phase") or selection.get("key", "unknown")
        tags = {
            "task_type": manifest.get("task_type") or manifest.get("task", "unknown"),
            "algorithm": manifest.get("algorithm", "unknown"),
            "dataset": self.dataset_name,
            "config": self.config_name,
            "phase": champion_phase,
            "recipe": manifest.get("recipe", manifest.get("variant_path", "default"))
        }
        
        # Collect metrics — s10 splits by phase, s06 stores at root
        metrics = manifest.get("metrics", {})
        if not metrics:
            # Try phase-specific metrics from s10 report
            metrics = manifest.get(f"{champion_phase}_metrics", {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                tags[f"metric_{metric_name}"] = str(metric_value)
        
        # Set tags on model version
        for tag_key, tag_value in tags.items():
            try:
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version,
                    key=tag_key,
                    value=tag_value
                )
            except Exception as e:
                logger.warning(f"Failed to set tag {tag_key}: {str(e)}")
        
        logger.info(f"✅ Added {len(tags)} metadata tags to model version")


def _write_skip_output(output_path: Path, reason: str):
    """Write a placeholder registry info file when registration is skipped."""
    skip_info = {
        "model_name": "SKIPPED",
        "version": "0",
        "stage": "None",
        "registration_skipped": True,
        "skip_reason": reason
    }
    with open(output_path, 'w') as f:
        json.dump(skip_info, f, indent=2, cls=NumpyEncoder)
    logger.info(f"💾 Wrote skip-registration output: {output_path}")
    # T8: Log skip reason to MLflow for Azure ML Studio dashboard visibility
    try:
        mlflow.log_param("registration_skipped", "true")
        mlflow.log_param("skip_reason", reason)
    except Exception as e:
        logger.debug("MLflow skip-reason log_param failed: %s", e)


def main():
    _t0 = _time_mod.time()
    parser = argparse.ArgumentParser(description="Stage 12: Model Registry")
    parser.add_argument("--champion_manifest", type=str, required=True, help="Champion manifest JSON")
    parser.add_argument("--champion_model", type=str, required=True, help="Champion model directory")
    parser.add_argument("--config_name", type=str, required=True, help="Config file name")
    parser.add_argument("--registry_info", type=str, required=True, help="Output registry info JSON")
    
    args = parser.parse_args()

    # T12: Convert tracking URI before ANY MLflow call
    _safe_disable_autolog()

    # T11: Create metrics logger for Azure ML Studio visibility
    ml_logger = create_metrics_logger(
        run_name="s12_model_registration",
        tags={"pipeline": "v3_mlops", "phase": "registration", "step": "s12"}
    )
    
    # Load champion manifest
    logger.info(f"📋 Loading champion manifest: {args.champion_manifest}")
    
    # 🔥 FIX (A5): Graceful handling if champion manifest or model is missing/empty
    manifest_path = Path(args.champion_manifest)
    model_path = Path(args.champion_model)
    output_path = Path(args.registry_info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists():
        logger.warning(f"⚠️ Champion manifest not found: {manifest_path}")
        logger.warning("Skipping model registration — writing placeholder output")
        _write_skip_output(output_path, "manifest_not_found")
        try:
            ml_logger.log_metric("registration_success", 0.0)
            ml_logger.log_param("skip_reason", "manifest_not_found")
            ml_logger.end_run()
        except Exception as e:
            logger.warning(f"⚠️ T16: MLflow skip-path logging failed: {e}")
        return
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"⚠️ Champion manifest is invalid JSON: {e}")
        _write_skip_output(output_path, f"invalid_manifest: {e}")
        try:
            ml_logger.log_metric("registration_success", 0.0)
            ml_logger.log_param("skip_reason", f"invalid_manifest")
            ml_logger.end_run()
        except Exception as e:
            logger.warning(f"⚠️ T16: MLflow skip-path logging failed: {e}")
        return
    
    if not manifest.get("algorithm") and not manifest.get("selection"):
        logger.warning("⚠️ Champion manifest has no 'algorithm' or 'selection' field — skipping registration")
        _write_skip_output(output_path, "manifest_missing_algorithm")
        try:
            ml_logger.log_metric("registration_success", 0.0)
            ml_logger.log_param("skip_reason", "manifest_missing_algorithm")
            ml_logger.end_run()
        except Exception as e:
            logger.warning(f"⚠️ T16: MLflow skip-path logging failed: {e}")
        return
    
    # T3: Check explicit champion_valid flag from s10's evaluation report
    if manifest.get("champion_valid") is False:
        logger.warning("⚠️ s10 reports no valid champion (all models failed) — skipping registration")
        _write_skip_output(output_path, "no_valid_champion")
        try:
            ml_logger.log_metric("registration_success", 0.0)
            ml_logger.log_param("skip_reason", "no_valid_champion")
            ml_logger.end_run()
        except Exception as e:
            logger.warning(f"⚠️ T16: MLflow skip-path logging failed: {e}")
        return

    if manifest.get("quality_gate_passed") is False:
        logger.warning("⚠️ s10 quality gate failed — skipping model registration")
        _write_skip_output(output_path, "quality_gate_failed")
        try:
            ml_logger.log_metric("registration_success", 0.0)
            ml_logger.log_param("skip_reason", "quality_gate_failed")
            ml_logger.end_run()
        except Exception as e:
            logger.warning(f"⚠️ T16: MLflow skip-path logging failed: {e}")
        return
    
    _sel_score = manifest.get("selection", {}).get("score")
    if _sel_score is None and not manifest.get("algorithm"):
        logger.warning("⚠️ Champion manifest has null score and no algorithm — skipping registration")
        _write_skip_output(output_path, "null_score_no_algorithm")
        try:
            ml_logger.log_metric("registration_success", 0.0)
            ml_logger.log_param("skip_reason", "null_score_no_algorithm")
            ml_logger.end_run()
        except Exception as e:
            logger.warning(f"⚠️ T16: MLflow skip-path logging failed: {e}")
        return
    
    if not model_path.exists() or not any(model_path.iterdir()):
        logger.warning(f"⚠️ Champion model directory missing or empty: {model_path}")
        _write_skip_output(output_path, "model_not_found")
        try:
            ml_logger.log_metric("registration_success", 0.0)
            ml_logger.log_param("skip_reason", "model_not_found")
            ml_logger.end_run()
        except Exception as e:
            logger.warning(f"⚠️ T16: MLflow skip-path logging failed: {e}")
        return
    
    _sel = manifest.get("selection", {})
    _phase_key = _sel.get("key", "unknown")
    logger.info(f"Champion: {manifest.get('algorithm', _phase_key)} (Phase: {manifest.get('phase', _phase_key)})")
    _display_metrics = manifest.get("metrics") or manifest.get(f"{_phase_key}_metrics", {})
    logger.info(f"Metrics: {_display_metrics}")
    
    # Register model (wrapped for crash safety)
    try:
        # K11: load YAML config so ModelRegistry can use cfg['dataset']['name']
        cfg_dict = None
        try:
            cfg_path_candidates = [
                Path("configs") / args.config_name,
                Path(args.config_name),
            ]
            for _cp in cfg_path_candidates:
                if _cp.exists():
                    with open(_cp, "r") as _cf:
                        cfg_dict = yaml.safe_load(_cf)
                    logger.info(f"📋 K11: loaded config from {_cp}")
                    break
            if cfg_dict is None:
                logger.warning(f"⚠️ K11: config file not found in {cfg_path_candidates}; falling back to filename parsing")
        except Exception as _cfg_err:
            logger.warning(f"⚠️ K11: failed to load config '{args.config_name}': {_cfg_err}")
            cfg_dict = None
        registry = ModelRegistry(args.config_name, cfg=cfg_dict)
        registry_info = registry.register_champion_model(
            manifest=manifest,
            model_path=model_path
        )
    except Exception as reg_err:
        logger.error(f"❌ Model registration failed: {reg_err}")
        logger.warning("Writing skip output so pipeline can continue")
        _write_skip_output(output_path, f"registration_error: {reg_err}")
        try:
            ml_logger.log_metric("registration_success", 0.0)
            ml_logger.log_param("skip_reason", f"registration_error")
            ml_logger.end_run()
        except Exception as e:
            logger.warning("MLflow registration-error skip-path logging failed: %s", e)
        return
    
    # Save registry info
    with open(output_path, 'w') as f:
        json.dump(registry_info, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"💾 Saved registry info: {output_path}")
    
    # T11: Log registration metrics via create_metrics_logger
    try:
        ml_logger.log_param("model_name", registry_info["model_name"])
        ml_logger.log_param("model_version", str(registry_info["version"]))
        ml_logger.log_param("model_stage", registry_info["stage"])
        ml_logger.log_param("config_name", args.config_name)
        ml_logger.log_metric("registration_success", 1.0)
        ml_logger.log_metric("model_version_num", int(registry_info["version"]))
        logger.info("✅ MLflow metrics logged via create_metrics_logger")
    except Exception as e:
        logger.warning(f"⚠️ MLflow logging failed (non-critical): {e}")
    
    # End the metrics logger run
    try:
        ml_logger.end_run()
    except Exception as e:
        logger.debug("ml_logger.end_run() failed: %s", e)
    
    logger.info(f"✅ Model registration complete")
    logger.info(f"Model: {registry_info['model_name']} v{registry_info['version']} (Stage: {registry_info['stage']})")



if __name__ == "__main__":
    main()
