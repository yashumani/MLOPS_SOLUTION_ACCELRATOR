"""
Stage 12: Model Registry Integration

Registers champion model from final evaluation to MLflow Model Registry.
Creates a run-bound model version with full lineage and normalized metadata.

Features:
- Model versioning (exact version bound to the source run)
- Lifecycle management (MLflow stage or Azure ML lifecycle tag)
- Rich metadata (dataset, task_type, algorithm, metrics, recipe)
- Model signature (input/output schema for serving validation)
- Artifact tracking (links model to source experiment run)

Exit Codes:
- 0: Model successfully registered
- 1: Registration failed (manifest not found, model invalid)
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# T11/T12: Import metrics logger + safe autolog disable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.azureml_metrics_logger import create_metrics_logger
from utils.model_bundle import load_model_bundle
from orchestration.contracts import (
    ContractValidationError,
    ExecutionManifest,
    QualityDecision,
)
from orchestration.config_compiler import compile_config
from orchestration.execution_identity import validate_execution_manifest_binding

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
_MLFLOW_SKLEARN_LOG_MODEL = mlflow.sklearn.log_model
_MODEL_BUNDLE_CODE_PATH = Path(__file__).resolve().parent.parent / "utils"


def _model_bundle_code_paths() -> list[str]:
    """Return package roots required to load a ModelBundle outside this repo."""
    if not _MODEL_BUNDLE_CODE_PATH.is_dir():
        raise RuntimeError(
            f"ModelBundle code path does not exist: {_MODEL_BUNDLE_CODE_PATH}"
        )
    return [str(_MODEL_BUNDLE_CODE_PATH)]


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
    """Disable autologging without mutating Azure ML tracking identity."""
    try:
        mlflow.autolog(disable=True)
    except Exception as e:
        logger.debug("mlflow.autolog(disable=True) failed: %s", e)


def resolve_quality_decision(manifest: Dict[str, Any]) -> str:
    """Return S10's one authoritative pass/warn/block decision."""
    quality = manifest.get("quality_decision")
    if int(manifest.get("schema_version") or 0) >= 2:
        if not isinstance(quality, dict):
            raise ContractValidationError(
                "Schema-v2 final report requires an immutable QualityDecision"
            )
        return QualityDecision.from_dict(quality).decision
    if isinstance(quality, dict):
        decision = str(quality.get("decision", "")).lower()
    else:
        decision = str(quality or "").lower()
    if decision in {"pass", "warn", "block"}:
        return decision
    if (
        "quality_decision" not in manifest
        and "quality_gate_passed" not in manifest
    ):
        # Missing policy evidence fails closed to registration-with-warning:
        # preserve the exact evaluated bundle, but never promote/alias it.
        return "warn"
    if manifest.get("quality_gate_passed") is True:
        return "pass"
    if manifest.get("quality_gate_passed") is False:
        # Legacy manifests treated a failed quality gate as a registration
        # block. Only an explicit schema-v2 QualityDecision may opt into warn.
        return "block"
    return "warn"


def validate_quality_decision_bundle(
    manifest: Dict[str, Any],
    exact_bundle: Any,
) -> QualityDecision | None:
    """Bind a schema-v2 policy decision to the exact registered bundle."""
    if int(manifest.get("schema_version") or 0) < 2:
        return None
    quality = QualityDecision.from_dict(manifest["quality_decision"])
    if quality.candidate_id != exact_bundle.candidate_id:
        raise ContractValidationError(
            "QualityDecision candidate_id does not match ModelBundle"
        )
    if quality.evaluated_bundle_hash != exact_bundle.bundle_id:
        raise ContractValidationError(
            "QualityDecision evaluated_bundle_hash does not match ModelBundle"
        )
    selected_candidate = (
        (manifest.get("selection") or {}).get("candidate_id")
    )
    if selected_candidate and selected_candidate != quality.candidate_id:
        raise ContractValidationError(
            "QualityDecision candidate_id does not match frozen selection"
        )
    expected_registration = quality.decision in {"pass", "warn"}
    if quality.registration_allowed != expected_registration:
        raise ContractValidationError(
            "QualityDecision registration policy is inconsistent"
        )
    return quality


def validate_registration_execution_binding(
    manifest: Dict[str, Any],
    exact_bundle: Any,
    execution_manifest: ExecutionManifest,
) -> None:
    """Refuse registration when final evidence is not from one exact execution."""
    report_lineage = manifest.get("lineage") or {}
    bundle_lineage = exact_bundle.lineage or {}
    for owner, lineage in (
        ("final report", report_lineage),
        ("ModelBundle", bundle_lineage),
    ):
        if lineage.get("execution_id") != execution_manifest.execution_id:
            raise ContractValidationError(
                f"{owner} execution_id does not match ExecutionManifest"
            )
        if lineage.get("config_hash") != execution_manifest.config_hash:
            raise ContractValidationError(
                f"{owner} config_hash does not match ExecutionManifest"
            )
        if lineage.get("code_sha") != execution_manifest.code_sha:
            raise ContractValidationError(
                f"{owner} code_sha does not match ExecutionManifest"
            )
    embedded = manifest.get("execution_manifest") or {}
    if embedded.get("execution_id") != execution_manifest.execution_id:
        raise ContractValidationError(
            "Final report embedded ExecutionManifest identity does not match"
        )


def _detect_model_flavor(model):
    """Agent 7: pick the correct MLflow flavor module by inspecting model class.

    Returns one of mlflow.sklearn, mlflow.lightgbm, mlflow.xgboost, mlflow.catboost.
    Falls back to mlflow.sklearn for unknown / wrapped sklearn estimators.
    """
    try:
        mod = (type(model).__module__ or "").lower()
    except Exception:
        mod = ""
    if "lightgbm" in mod:
        try:
            import mlflow.lightgbm as _mlf
            return _mlf
        except Exception:
            pass
    if "xgboost" in mod:
        try:
            import mlflow.xgboost as _mlf
            return _mlf
        except Exception:
            pass
    if "catboost" in mod:
        try:
            import mlflow.catboost as _mlf
            return _mlf
        except Exception:
            pass
    return mlflow.sklearn


def _log_exact_model_bundle(bundle: Any, model_name: str) -> Any:
    """Log and register the exact evaluated bundle through MLflow."""
    raw_example = bundle.input_example
    if raw_example is None:
        raise ContractValidationError(
            "Exact ModelBundle requires a representative input_example"
        )
    if isinstance(raw_example, dict):
        input_example = pd.DataFrame([raw_example])
    else:
        input_example = pd.DataFrame(list(raw_example))
    if input_example.empty:
        raise ContractValidationError(
            "Exact ModelBundle input_example cannot be empty"
        )
    predictions = bundle.predict(input_example)
    signature = infer_signature(input_example, predictions)
    return _MLFLOW_SKLEARN_LOG_MODEL(
        sk_model=bundle,
        artifact_path="champion_model",
        registered_model_name=model_name,
        signature=signature,
        input_example=input_example,
        code_paths=_model_bundle_code_paths(),
        serialization_format="cloudpickle",
    )


def _is_unsupported_azureml_artifact_repository(error: Exception) -> bool:
    """Return whether MLflow lacks the Azure ML artifact repository plugin."""
    message = str(error).lower()
    return (
        "could not find a registered artifact repository" in message
        and "azureml://" in message
    )


def _positive_model_version(version: Any) -> str:
    """Return an Azure ML numeric model version or fail closed."""
    normalized = str(version).strip() if version is not None else ""
    if not normalized.isdigit() or int(normalized) <= 0:
        raise RuntimeError(
            "Azure ML SDK registration did not return a positive integer "
            f"model version: {version!r}"
        )
    return normalized


_AZUREML_WORKSPACE_ENV = (
    "AZUREML_ARM_SUBSCRIPTION",
    "AZUREML_ARM_RESOURCEGROUP",
    "AZUREML_ARM_WORKSPACE_NAME",
)


def _has_any_azureml_context_signal() -> bool:
    """Return whether the process presents itself as an Azure ML job."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    return (
        any(os.getenv(name) for name in _AZUREML_WORKSPACE_ENV)
        or bool(os.getenv("AZUREML_RUN_ID"))
        or bool(os.getenv("AZUREML_RUN_TOKEN"))
        or tracking_uri.startswith("azureml://")
    )


def _get_azureml_workspace_context() -> Dict[str, str] | None:
    """Return complete AML workspace context, failing on partial job identity."""
    values = {
        name: os.getenv(name)
        for name in _AZUREML_WORKSPACE_ENV
    }
    if not _has_any_azureml_context_signal():
        return None

    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            "Incomplete Azure ML workspace context; refusing model "
            "serialization fallback. Missing: "
            + ", ".join(missing)
        )
    return {
        "subscription_id": str(values["AZUREML_ARM_SUBSCRIPTION"]),
        "resource_group": str(values["AZUREML_ARM_RESOURCEGROUP"]),
        "workspace_name": str(values["AZUREML_ARM_WORKSPACE_NAME"]),
    }


class _NoOpRegistrationMetricsLogger:
    """Keep registration independent of optional MLflow job telemetry."""

    def log_metric(self, *_args, **_kwargs) -> None:
        pass

    def log_param(self, *_args, **_kwargs) -> None:
        pass

    def end_run(self) -> None:
        pass


def _create_registration_metrics_logger():
    """Use MLflow telemetry only when it cannot affect AML registration."""
    if _has_any_azureml_context_signal():
        logger.info(
            "Azure ML job context detected; model output and tags are the "
            "authoritative registration telemetry"
        )
        return _NoOpRegistrationMetricsLogger()
    return create_metrics_logger(
        run_name="s12_model_registration",
        tags={
            "pipeline": "v3_mlops",
            "phase": "registration",
            "step": "s12",
        },
    )


def _infer_model_algorithm(model: Any) -> str:
    """Resolve the concrete estimator class, including sklearn Pipelines."""
    estimator = model
    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        estimator = named_steps.get("estimator") or list(named_steps.values())[-1]
    return type(estimator).__name__


def _resolve_registration_metadata(
    manifest: Dict[str, Any],
    model: Any = None,
) -> Dict[str, Any]:
    """Normalize S06 and S10 manifest metadata once for every backend."""
    selection = manifest.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    phase = manifest.get("phase") or selection.get("key") or "unknown"
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        phase_metrics = manifest.get(f"{phase}_metrics")
        metrics = phase_metrics if isinstance(phase_metrics, dict) else {}

    algorithm = manifest.get("algorithm") or metrics.get("algorithm")
    if not algorithm and model is not None:
        algorithm = _infer_model_algorithm(model)

    task_type = manifest.get("task_type") or manifest.get("task")
    task_type = (
        task_type.strip()
        if isinstance(task_type, str) and task_type.strip()
        else "unknown"
    )
    return {
        "task_type": task_type,
        "algorithm": algorithm or "unknown",
        "phase": phase,
        "metrics": metrics,
        "recipe": manifest.get("recipe")
        or manifest.get("variant_path")
        or "default",
    }


class _AzureMLOBOCredentialAdapter:
    """Make the legacy AML OBO credential compatible with Azure Core."""

    def __init__(self, credential: Any):
        self._credential = credential

    def get_token(self, *scopes: str, **kwargs: Any):
        # azure-ai-ml 1.17 forwards Azure Core token options to
        # requests.Session.request(), which accepts none of them.
        return self._credential.get_token(*scopes)


def _create_azureml_sdk_client(
    *,
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
):
    """Create an Azure ML client using the job's delegated or managed identity."""
    from azure.ai.ml import MLClient
    from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
    from azure.identity import ChainedTokenCredential, DefaultAzureCredential

    credentials = []
    if os.getenv("OBO_ENDPOINT"):
        credentials.append(
            _AzureMLOBOCredentialAdapter(
                AzureMLOnBehalfOfCredential()
            )
        )
    credentials.append(
        DefaultAzureCredential(exclude_interactive_browser_credential=True)
    )
    credential = (
        credentials[0]
        if len(credentials) == 1
        else ChainedTokenCredential(*credentials)
    )
    return MLClient(
        credential,
        subscription_id,
        resource_group,
        workspace_name,
    )


def _resolve_azureml_job_input_uri(
    ml_client: Any,
    *,
    run_id: str,
    input_name: str,
) -> str:
    """Resolve a server-owned AML job input to its remote artifact URI."""
    job = ml_client.jobs.get(run_id)
    inputs = getattr(job, "inputs", None)
    if not isinstance(inputs, dict) or input_name not in inputs:
        raise RuntimeError(
            f"Azure ML job {run_id!r} does not declare input {input_name!r}"
        )

    job_input = inputs[input_name]
    if isinstance(job_input, dict):
        input_uri = job_input.get("path") or job_input.get("uri")
        input_type = job_input.get("type") or job_input.get("jobInputType")
    else:
        input_uri = getattr(job_input, "path", None) or getattr(
            job_input,
            "uri",
            None,
        )
        input_type = getattr(job_input, "type", None)

    if input_type and str(input_type).lower() != "uri_folder":
        raise RuntimeError(
            f"Azure ML job input {input_name!r} must be a uri_folder, "
            f"got {input_type!r}"
        )
    if not isinstance(input_uri, str) or not input_uri.startswith("azureml://"):
        raise RuntimeError(
            f"Azure ML job input {input_name!r} must resolve to a remote "
            "azureml:// URI"
        )
    return input_uri


def _find_run_bound_azureml_model(
    ml_client: Any,
    *,
    model_name: str,
    run_id: str,
) -> Any:
    """Find a single model version previously created for this exact run."""
    from azure.core.exceptions import ResourceNotFoundError

    try:
        versions = list(ml_client.models.list(name=model_name))
    except ResourceNotFoundError:
        versions = []
    candidates = [
        model
        for model in versions
        if str(
            (getattr(model, "properties", None) or {}).get(
                "source_run_id",
                "",
            )
        )
        == str(run_id)
        or str(
            (getattr(model, "tags", None) or {}).get("source_run_id", "")
        )
        == str(run_id)
    ]
    if len(candidates) > 1:
        raise RuntimeError(
            "Azure ML model registration is ambiguous: "
            f"{len(candidates)} versions map to source run {run_id!r}"
        )
    if not candidates:
        return None

    candidate_version = _positive_model_version(
        getattr(candidates[0], "version", None)
    )
    return ml_client.models.get(
        name=model_name,
        version=candidate_version,
    )


def _model_uri_fingerprint(model_uri: str) -> str:
    """Return a stable fingerprint for the immutable model source URI."""
    normalized_uri = _normalize_azureml_uri(model_uri)
    return hashlib.sha256(normalized_uri.encode("utf-8")).hexdigest()


def _normalize_azureml_uri(model_uri: str) -> str:
    """Normalize AML resource casing while preserving case-sensitive paths."""
    normalized_uri = model_uri.rstrip("/")
    marker = "/paths/"
    marker_index = normalized_uri.lower().find(marker)
    if marker_index < 0:
        return normalized_uri.lower()
    resource_end = marker_index + len(marker)
    return (
        normalized_uri[:resource_end].lower()
        + normalized_uri[resource_end:]
    )


def _validate_run_bound_azureml_model(
    model: Any,
    *,
    run_id: str,
    model_uri: str,
) -> str:
    """Validate exact version, immutable lineage, tags, and artifact identity."""
    version = _positive_model_version(getattr(model, "version", None))
    expected_run_id = str(run_id)
    expected_uri = model_uri.rstrip("/")
    expected_fingerprint = _model_uri_fingerprint(expected_uri)
    tags = getattr(model, "tags", None) or {}
    properties = getattr(model, "properties", None) or {}
    registered_uri = str(getattr(model, "path", "") or "").rstrip("/")

    if str(properties.get("source_run_id", "")) != expected_run_id:
        raise RuntimeError(
            "Azure ML model immutable source-run ownership does not match: "
            f"expected={expected_run_id!r}"
        )
    if str(tags.get("source_run_id", "")) != expected_run_id:
        raise RuntimeError(
            "Azure ML model source-run tag does not match immutable ownership"
        )
    if (
        str(properties.get("source_model_uri_sha256", ""))
        != expected_fingerprint
    ):
        raise RuntimeError(
            "Azure ML model immutable source-URI fingerprint does not match"
        )
    if _normalize_azureml_uri(registered_uri) != _normalize_azureml_uri(
        expected_uri
    ):
        raise RuntimeError(
            "Azure ML model artifact URI does not match canonical job input: "
            f"expected={expected_uri!r}, returned={registered_uri!r}"
        )
    return version


def _create_azureml_model_asset(**kwargs):
    """Build a custom Azure ML model asset without eager SDK imports."""
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model

    return Model(type=AssetTypes.CUSTOM_MODEL, **kwargs)


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

        self.azureml_workspace_context = _get_azureml_workspace_context()
        self.client = MlflowClient()
        
        # Extract dataset name from cfg first, fall back to filename parsing
        self.dataset_name = self._extract_dataset_name(config_name)
        # K11: also expose explicit registry model name override if cfg provides one.
        registry_cfg = self.cfg.get("registry", {}) or {}
        self.model_name_override = registry_cfg.get("model_name")
        self.requested_promotion_aliases = tuple(
            str(alias).strip()
            for alias in (registry_cfg.get("pass_aliases") or [])
            if str(alias).strip()
        )
    
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
        model_path: Path,
        execution_manifest: ExecutionManifest,
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
        metadata = _resolve_registration_metadata(manifest)
        task_type = metadata["task_type"]
        quality_decision = resolve_quality_decision(manifest)
        if quality_decision == "block":
            raise ValueError("Blocked QualityDecision cannot be registered")
        promotion_allowed = quality_decision == "pass"
        
        # Model name: {dataset}_{task}_mlops
        model_name = f"{self.dataset_name}_{task_type}_mlops"
        # K11: allow explicit override from cfg['registry']['model_name'].
        if getattr(self, "model_name_override", None):
            model_name = self.model_name_override
            logger.info(f"📛 K11: using cfg-provided registry model name = {model_name}")
        
        logger.info(f"📦 Registering model: {model_name}")
        
        # Find model artifact (PyCaret saves as .pkl, FLAML as .pkl or model/)
        model_file = self._find_model_artifact(model_path)
        
        if not model_file:
            logger.error(f"❌ No model artifact found in {model_path}")
            raise FileNotFoundError(f"Model artifact not found in {model_path}")
        
        logger.info(f"Found model artifact: {model_file}")
        # Register the exact serialized bundle with MLflow. Azure ML jobs use
        # the workspace MLflow registry through azureml-mlflow; they never
        # register the mounted job-input folder as a separate model asset.
        try:
            logger.info(f"Loading model from {model_file}")
            exact_bundle = load_model_bundle(model_file)
            validate_quality_decision_bundle(manifest, exact_bundle)
            validate_registration_execution_binding(
                manifest,
                exact_bundle,
                execution_manifest,
            )
            logger.info(f"Model loaded: {type(exact_bundle).__name__}")
            metadata = _resolve_registration_metadata(
                manifest,
                exact_bundle.estimator,
            )
            logger.info(
                f"Algorithm: {metadata['algorithm']}, "
                f"Metrics: {metadata['metrics']}"
            )

            active_run = mlflow.active_run()
            if not active_run:
                azureml_run_id = os.getenv("AZUREML_RUN_ID")
                active_run = (
                    mlflow.start_run(run_id=azureml_run_id)
                    if azureml_run_id
                    else mlflow.start_run(
                        run_name=f"s12_register_{model_name}"
                    )
                )
                logger.info(f"🆕 Attached MLflow run: {active_run.info.run_id}")
            run_id = active_run.info.run_id
            logger.info(f"Logging exact ModelBundle to run {run_id}")
            registration_backend = "mlflow"
            log_model_result = _log_exact_model_bundle(
                exact_bundle,
                model_name,
            )
            model_version = self._resolve_registered_model_version(
                log_model_result=log_model_result,
                model_name=model_name,
                run_id=run_id,
            )
            logger.info(f"✅ Model registered as version {model_version}")
            
            self._add_model_metadata(
                model_name,
                model_version,
                manifest,
                metadata,
                execution_manifest=execution_manifest,
                registration_run_id=run_id,
            )
            if promotion_allowed:
                logger.info(
                    "Passing model registered without a stage or alias; "
                    "promotion remains an explicit operator action"
                )
                stage_backend = "manual_promotion_required"
            else:
                stage_backend = "quality_warning_no_promotion"
                logger.info("⚠️ Warning model registered without promotion")
            stage = "None"
            requested_aliases = (
                list(getattr(self, "requested_promotion_aliases", ()))
                if promotion_allowed
                else []
            )
            
            # Build registry info
            registry_info = {
                "model_name": model_name,
                "version": model_version,
                "model_uri": f"models:/{model_name}/{model_version}",
                "stage": stage,
                "lifecycle_stage": "Unassigned",
                "quality_decision": quality_decision,
                "promotion_allowed": promotion_allowed,
                "promotion_mode": "manual",
                "promotion_performed": False,
                "requested_promotion_aliases": requested_aliases,
                "algorithm": metadata["algorithm"],
                "task_type": task_type,
                "metrics": metadata["metrics"],
                "dataset": self.dataset_name,
                "config": self.config_name,
                "registration_run_id": run_id,
                "execution_id": execution_manifest.execution_id,
                "config_hash": execution_manifest.config_hash,
                "code_sha": execution_manifest.code_sha,
                "recipe_catalog_hash": execution_manifest.recipe_catalog_hash,
                "dataset_content_sha256": str(
                    execution_manifest.dataset.get("content_sha256") or ""
                ),
                "registration_backend": registration_backend,
                "stage_backend": stage_backend,
            }
            
            return registry_info

        except Exception as e:
            logger.error(f"❌ Model registration failed: {str(e)}")
            raise

    def _resolve_registered_model_version(
        self,
        log_model_result: Any,
        model_name: str,
        run_id: str,
    ) -> str:
        """Resolve only the model version created by the active run."""
        returned_version = getattr(
            log_model_result,
            "registered_model_version",
            None,
        )
        if returned_version in (None, ""):
            raise RuntimeError(
                "MLflow log_model did not return the exact registered model "
                f"version for model={model_name!r}, run_id={run_id!r}"
            )
        return str(returned_version)

    def _register_with_azureml_sdk(
        self,
        *,
        model_name: str,
        model_path: Path,
        manifest: Dict[str, Any],
        metadata: Dict[str, Any],
        run_id: str,
    ) -> str:
        """Register directly when the runtime lacks azureml-mlflow artifacts."""
        workspace_context = _get_azureml_workspace_context()
        if workspace_context is None:
            raise RuntimeError(
                "Azure ML SDK registration requires Azure ML workspace "
                "context: "
                + ", ".join(_AZUREML_WORKSPACE_ENV)
            )

        ml_client = _create_azureml_sdk_client(
            **workspace_context,
        )
        remote_model_uri = _resolve_azureml_job_input_uri(
            ml_client,
            run_id=run_id,
            input_name="champion_model",
        )
        logger.info(
            "Binding model asset to canonical AML input %s (mounted at %s)",
            remote_model_uri,
            model_path,
        )
        tags = self._build_model_metadata_tags(manifest, metadata)
        tags.update(
            {
                "source_run_id": str(run_id),
                "lifecycle_stage": "Unassigned",
                "registration_backend": "azureml_sdk",
            }
        )
        properties = {
            "source_run_id": str(run_id),
            "source_model_uri_sha256": _model_uri_fingerprint(
                remote_model_uri
            ),
        }
        registered = _find_run_bound_azureml_model(
            ml_client,
            model_name=model_name,
            run_id=run_id,
        )
        if registered is None:
            registered = ml_client.models.create_or_update(
                _create_azureml_model_asset(
                    path=remote_model_uri,
                    name=model_name,
                    description="MLOps V3 final champion model",
                    tags=tags,
                    properties=properties,
                )
            )
        else:
            logger.info(
                "Reusing exact Azure ML model version %s for source run %s",
                getattr(registered, "version", None),
                run_id,
            )

        return _validate_run_bound_azureml_model(
            registered,
            run_id=run_id,
            model_uri=remote_model_uri,
        )
    
    def _find_model_artifact(self, model_path: Path) -> Path | None:
        """Return only the canonical champion model artifact."""
        bundle_file = model_path / "model_bundle.pkl"
        if bundle_file.is_file() and bundle_file.stat().st_size > 0:
            return bundle_file
        return None
    
    def _add_model_metadata(
        self,
        model_name: str,
        model_version: str,
        manifest: Dict[str, Any],
        metadata: Dict[str, Any],
        execution_manifest: ExecutionManifest | None = None,
        registration_run_id: str | None = None,
    ):
        """Add metadata tags to registered model version."""
        tags = self._build_model_metadata_tags(
            manifest,
            metadata,
            execution_manifest=execution_manifest,
            registration_run_id=registration_run_id,
        )
        failures = []

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
                logger.error(f"Failed to set required tag {tag_key}: {str(e)}")
                failures.append(tag_key)

        if failures:
            raise RuntimeError(
                "Failed to persist required model metadata tags: "
                + ", ".join(failures)
            )

        logger.info(f"✅ Added {len(tags)} metadata tags to model version")

    def _build_model_metadata_tags(
        self,
        manifest: Dict[str, Any],
        metadata: Dict[str, Any] = None,
        execution_manifest: ExecutionManifest | None = None,
        registration_run_id: str | None = None,
    ) -> Dict[str, str]:
        """Build string tags shared by MLflow and Azure ML SDK registration."""
        metadata = metadata or _resolve_registration_metadata(manifest)
        tags = {
            "task_type": metadata["task_type"],
            "algorithm": metadata["algorithm"],
            "dataset": self.dataset_name,
            "config": self.config_name,
            "phase": metadata["phase"],
            "recipe": metadata["recipe"],
            "quality_decision": resolve_quality_decision(manifest),
            "promotion_allowed": str(
                resolve_quality_decision(manifest) == "pass"
            ).lower(),
            "promotion_mode": "manual",
            "promotion_performed": "false",
            "lifecycle_stage": "Unassigned",
        }
        if execution_manifest is not None:
            tags.update(
                {
                    "execution_id": execution_manifest.execution_id,
                    "config_hash": execution_manifest.config_hash,
                    "code_sha": execution_manifest.code_sha,
                    "recipe_catalog_hash": execution_manifest.recipe_catalog_hash,
                    "dataset_content_sha256": str(
                        execution_manifest.dataset.get("content_sha256") or ""
                    ),
                }
            )
        if registration_run_id:
            tags["registration_run_id"] = str(registration_run_id)
        bundle = manifest.get("model_bundle") or {}
        if isinstance(bundle, dict) and bundle.get("bundle_id"):
            tags["model_bundle_id"] = bundle["bundle_id"]

        for metric_name, metric_value in metadata["metrics"].items():
            if isinstance(metric_value, (int, float)):
                tags[f"metric_{metric_name}"] = str(metric_value)
        return {key: str(value) for key, value in tags.items()}


def _write_skip_output(output_path: Path, reason: str):
    """Write registry info for an explicit policy-controlled skip."""
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


def _write_failure_output(output_path: Path, reason: str):
    """Write diagnostic registry output before failing the component."""
    failure_info = {
        "model_name": "FAILED",
        "version": "0",
        "stage": "None",
        "registration_failed": True,
        "failure_reason": reason,
    }
    with open(output_path, "w") as handle:
        json.dump(failure_info, handle, indent=2, cls=NumpyEncoder)
    logger.info(f"💾 Wrote registration-failure diagnostic: {output_path}")


def _fail_registration_contract(
    output_path: Path,
    reason: str,
    ml_logger: Any,
) -> None:
    """Record an invalid-input diagnostic and fail the Azure ML component."""
    logger.error(f"❌ Model registration contract failed: {reason}")
    _write_failure_output(output_path, reason)
    try:
        ml_logger.log_metric("registration_success", 0.0)
        ml_logger.log_param("failure_reason", reason)
        ml_logger.end_run()
    except Exception as error:
        logger.warning(
            "MLflow registration-contract failure logging failed: %s",
            error,
        )
    raise RuntimeError(f"Model registration contract failed: {reason}")


def main():
    parser = argparse.ArgumentParser(description="Stage 12: Model Registry")
    parser.add_argument("--champion_manifest", type=str, required=True, help="Champion manifest JSON")
    parser.add_argument("--champion_model", type=str, required=True, help="Champion model directory")
    parser.add_argument("--config_name", type=str, required=True, help="Config file name")
    parser.add_argument(
        "--execution_manifest",
        type=str,
        default="",
        help="Validated immutable execution manifest from Phase B",
    )
    parser.add_argument("--registry_info", type=str, required=True, help="Output registry info JSON")
    
    args = parser.parse_args()

    # Disable automatic model logging without changing AML tracking identity.
    _safe_disable_autolog()

    ml_logger = _create_registration_metrics_logger()
    
    # Load champion manifest
    logger.info(f"📋 Loading champion manifest: {args.champion_manifest}")
    
    manifest_path = Path(args.champion_manifest)
    model_path = Path(args.champion_model)
    output_path = Path(args.registry_info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not manifest_path.exists():
        _fail_registration_contract(
            output_path,
            "manifest_not_found",
            ml_logger,
        )
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        _fail_registration_contract(
            output_path,
            f"invalid_manifest: {e}",
            ml_logger,
        )

    if not isinstance(manifest, dict):
        _fail_registration_contract(
            output_path,
            "invalid_manifest_root",
            ml_logger,
        )

    selection = manifest.get("selection")
    if selection is not None and not isinstance(selection, dict):
        _fail_registration_contract(
            output_path,
            "invalid_selection_metadata",
            ml_logger,
        )
    
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

    quality_decision = resolve_quality_decision(manifest)
    if quality_decision == "block":
        logger.warning("⚠️ s10 QualityDecision=block — skipping model registration")
        skip_reason = (
            "quality_gate_failed"
            if "quality_decision" not in manifest
            and manifest.get("quality_gate_passed") is False
            else "quality_decision_block"
        )
        _write_skip_output(output_path, skip_reason)
        try:
            ml_logger.log_metric("registration_success", 0.0)
            ml_logger.log_param("skip_reason", skip_reason)
            ml_logger.end_run()
        except Exception as e:
            logger.warning(f"⚠️ T16: MLflow skip-path logging failed: {e}")
        return
    logger.info(
        "QualityDecision=%s; registration allowed, promotion_allowed=%s",
        quality_decision,
        quality_decision == "pass",
    )

    task_type = manifest.get("task_type") or manifest.get("task")
    if not isinstance(task_type, str) or not task_type.strip():
        _fail_registration_contract(
            output_path,
            "invalid_task_metadata",
            ml_logger,
        )

    if not manifest.get("algorithm") and not manifest.get("selection"):
        _fail_registration_contract(
            output_path,
            "manifest_missing_algorithm_or_selection",
            ml_logger,
        )
    
    _sel_score = (selection or {}).get("score")
    if _sel_score is None and not manifest.get("algorithm"):
        _fail_registration_contract(
            output_path,
            "null_score_no_algorithm",
            ml_logger,
        )
    
    if not model_path.is_dir() or not any(model_path.iterdir()):
        _fail_registration_contract(
            output_path,
            "model_not_found",
            ml_logger,
        )

    try:
        config_path = next(
            path
            for path in (
                Path("configs") / Path(args.config_name).name,
                Path(args.config_name),
            )
            if path.is_file()
        )
        raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        compiled_cfg = compile_config(raw_cfg, source_name=config_path.name)
        execution_manifest = validate_execution_manifest_binding(
            args.execution_manifest,
            compiled_cfg,
        )
        report_execution_id = (manifest.get("lineage") or {}).get(
            "execution_id"
        )
        if report_execution_id != execution_manifest.execution_id:
            raise ContractValidationError(
                "Final report execution_id does not match ExecutionManifest"
            )
    except Exception as error:
        _fail_registration_contract(
            output_path,
            f"execution_identity_invalid: {error}",
            ml_logger,
        )
    
    _sel = manifest.get("selection", {})
    _phase_key = _sel.get("key", "unknown")
    logger.info(f"Champion: {manifest.get('algorithm', _phase_key)} (Phase: {manifest.get('phase', _phase_key)})")
    _display_metrics = manifest.get("metrics") or manifest.get(f"{_phase_key}_metrics", {})
    logger.info(f"Metrics: {_display_metrics}")
    
    # Register model (wrapped for crash safety)
    try:
        # K11: load YAML config so ModelRegistry can use cfg['dataset']['name']
        registry = ModelRegistry(args.config_name, cfg=compiled_cfg)
        registry_info = registry.register_champion_model(
            manifest=manifest,
            model_path=model_path,
            execution_manifest=execution_manifest,
        )
    except Exception as reg_err:
        logger.error(f"❌ Model registration failed: {reg_err}")
        logger.warning("Writing diagnostic output before failing the component")
        _write_failure_output(
            output_path,
            f"registration_error: {reg_err}",
        )
        try:
            ml_logger.log_metric("registration_success", 0.0)
            ml_logger.log_param("failure_reason", "registration_error")
            ml_logger.end_run()
        except Exception as e:
            logger.warning("MLflow registration-error skip-path logging failed: %s", e)
        raise RuntimeError("Model registration failed") from reg_err
    
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
        if str(registry_info["version"]).isdigit():
            ml_logger.log_metric(
                "model_version_num",
                int(registry_info["version"]),
            )
        logger.info("✅ MLflow metrics logged via create_metrics_logger")
    except Exception as e:
        logger.warning(f"⚠️ MLflow logging failed (non-critical): {e}")
    
    # End the metrics logger run
    try:
        ml_logger.end_run()
    except Exception as e:
        logger.debug("ml_logger.end_run() failed: %s", e)
    
    logger.info("✅ Model registration complete")
    logger.info(f"Model: {registry_info['model_name']} v{registry_info['version']} (Stage: {registry_info['stage']})")



if __name__ == "__main__":
    main()
