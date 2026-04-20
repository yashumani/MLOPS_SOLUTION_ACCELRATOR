"""Pipeline submission, monitoring, and output retrieval service."""

import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from azure.ai.ml import Input, MLClient

from api.core.azure_ml import get_ml_client
from api.core.config import settings
from api.schemas.pipeline import (
    DriftResponse,
    DriftResultItem,
    JobListResponse,
    JobStatus,
    JobSummary,
    MetricsResponse,
    ModelMetric,
    OutputInfo,
    OutputListResponse,
    ResubmitRequest,
    StepStatus,
    SubmitRequest,
    SubmitResponse,
)

# Repo root so we can import pipeline_builder and read configs
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _derive_experiment_name(config_name: str) -> str:
    """Derive reusable experiment name from config filename.

    config_classification_telecom_churn_azureml → classification_telecom_churn_v3
    """
    normalized = config_name.replace("config_", "").replace("_azureml", "").replace("_local", "")
    return f"{normalized}_v3"


def _derive_display_name(experiment_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{experiment_name}_{timestamp}_{unique_id}"


def _studio_url(job_name: str) -> str:
    return (
        f"https://ml.azure.com/runs/{job_name}"
        f"?wsid=/subscriptions/{settings.azure_subscription_id}"
        f"/resourcegroups/{settings.azure_resource_group}"
        f"/workspaces/{settings.azure_workspace_name}"
    )


def _load_config_yaml(config_name: str) -> dict:
    path = _REPO_ROOT / "configs" / f"{config_name}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_name}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def submit_pipeline(req: SubmitRequest) -> SubmitResponse:
    """Submit an Azure ML pipeline job mirroring submit_pipeline.py logic."""
    ml_client = get_ml_client()
    cfg = _load_config_yaml(req.config_name)

    # Ensure pipeline_builder is importable
    pipelines_dir = str(_REPO_ROOT / "pipelines")
    if pipelines_dir not in sys.path:
        sys.path.insert(0, pipelines_dir)
    from pipeline_builder import full_pipeline  # type: ignore[import-untyped]

    datastore_name = (cfg.get("dataset") or {}).get("datastore_name", "mlops_blob")
    dataset_folder_uri = (
        f"azureml://subscriptions/{settings.azure_subscription_id}"
        f"/resourcegroups/{settings.azure_resource_group}"
        f"/workspaces/{settings.azure_workspace_name}"
        f"/datastores/{datastore_name}/paths/"
    )

    config_filename = f"{req.config_name}.yml"
    pipeline_kwargs: dict = dict(
        config_name=config_filename,
        dataset_folder=Input(path=dataset_folder_uri, type="uri_folder"),
    )

    # Optional drift baseline
    if req.baseline_job:
        try:
            baseline_job_obj = ml_client.jobs.get(req.baseline_job)
            outputs = baseline_job_obj.outputs or {}
            if "drift_baseline" in outputs:
                asset_id = getattr(outputs["drift_baseline"], "path", None)
                if asset_id:
                    pipeline_kwargs["drift_baseline_in"] = Input(path=asset_id, type="uri_folder")
        except Exception:
            pass  # proceed without baseline

    job = full_pipeline(**pipeline_kwargs)

    compute = req.compute or settings.compute_target
    job.settings.default_compute = compute
    if req.force_rerun:
        job.settings.force_rerun = True

    experiment_name = _derive_experiment_name(req.config_name)
    display_name = _derive_display_name(experiment_name)
    job.experiment_name = experiment_name
    job.display_name = display_name

    # Tags
    dataset_meta = cfg.get("dataset") or {}
    tags = {
        "dataset": dataset_meta.get("name", "unknown"),
        "task": cfg.get("task_type", "unknown"),
        "pipeline_version": "v3",
        "source": "api",
    }
    tags.update(req.tags)
    job.tags = tags

    submitted = ml_client.jobs.create_or_update(job)

    return SubmitResponse(
        job_name=submitted.name,
        experiment_name=experiment_name,
        display_name=display_name,
        status=submitted.status or "Submitted",
        studio_url=_studio_url(submitted.name),
    )


# ---------------------------------------------------------------------------
# List / Get / Cancel
# ---------------------------------------------------------------------------


def list_jobs(
    experiment_name: str | None = None,
    status_filter: str | None = None,
    max_results: int = 50,
) -> JobListResponse:
    ml_client = get_ml_client()

    kwargs: dict = {}
    if experiment_name:
        kwargs["experiment_name"] = experiment_name

    jobs: list[JobSummary] = []
    for j in ml_client.jobs.list(**kwargs):
        if status_filter and (j.status or "").lower() != status_filter.lower():
            continue
        jobs.append(
            JobSummary(
                job_name=j.name,
                experiment_name=getattr(j, "experiment_name", None),
                display_name=getattr(j, "display_name", None),
                status=j.status or "Unknown",
                start_time=getattr(j, "creation_context", None)
                and getattr(j.creation_context, "created_at", None),
            )
        )
        if len(jobs) >= max_results:
            break

    return JobListResponse(jobs=jobs, total=len(jobs))


def get_job(job_name: str) -> JobStatus:
    ml_client = get_ml_client()
    j = ml_client.jobs.get(job_name)

    # Fetch child step statuses
    steps: list[StepStatus] = []
    try:
        for child in ml_client.jobs.list(parent_job_name=job_name):
            steps.append(
                StepStatus(
                    name=getattr(child, "display_name", None) or child.name,
                    status=child.status or "Unknown",
                    start_time=getattr(child, "creation_context", None)
                    and getattr(child.creation_context, "created_at", None),
                    end_time=None,
                )
            )
    except Exception:
        pass

    return JobStatus(
        job_name=j.name,
        experiment_name=getattr(j, "experiment_name", None),
        display_name=getattr(j, "display_name", None),
        status=j.status or "Unknown",
        start_time=getattr(j, "creation_context", None)
        and getattr(j.creation_context, "created_at", None),
        end_time=None,
        tags=dict(j.tags) if j.tags else {},
        steps=steps,
    )


def cancel_job(job_name: str) -> JobStatus:
    ml_client = get_ml_client()
    ml_client.jobs.cancel(job_name)
    return get_job(job_name)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def list_outputs(job_name: str) -> OutputListResponse:
    ml_client = get_ml_client()
    j = ml_client.jobs.get(job_name)
    outputs_dict = j.outputs or {}
    items = [
        OutputInfo(name=k, type=getattr(v, "type", None))
        for k, v in outputs_dict.items()
    ]
    return OutputListResponse(job_name=job_name, outputs=items)


def download_output(job_name: str, output_name: str) -> Path:
    """Download a specific job output to a temp directory and return the path."""
    ml_client = get_ml_client()
    tmp = Path(tempfile.mkdtemp(prefix=f"mlops_output_{job_name}_"))
    ml_client.jobs.download(job_name, download_path=str(tmp), output_name=output_name)
    return tmp


# ---------------------------------------------------------------------------
# Metrics (Phase 0a)
# ---------------------------------------------------------------------------


def get_job_metrics(job_name: str) -> MetricsResponse:
    """Retrieve per-model MLflow metrics logged by training steps."""
    ml_client = get_ml_client()
    parent = ml_client.jobs.get(job_name)
    task_type = (parent.tags or {}).get("task", None)

    models: list[ModelMetric] = []
    try:
        for child in ml_client.jobs.list(parent_job_name=job_name):
            child_detail = ml_client.jobs.get(child.name)
            # Only training/aggregate steps log model metrics
            display = getattr(child_detail, "display_name", "") or child.name
            tags = dict(child_detail.tags) if child_detail.tags else {}

            # Try to retrieve nested per-model runs
            try:
                for grandchild in ml_client.jobs.list(parent_job_name=child.name):
                    gc_detail = ml_client.jobs.get(grandchild.name)
                    gc_name = getattr(gc_detail, "display_name", "") or grandchild.name
                    gc_tags = dict(gc_detail.tags) if gc_detail.tags else {}

                    # Collect numeric metrics from tags (MLflow logs appear as tags)
                    metrics: dict[str, float] = {}
                    for k, v in gc_tags.items():
                        try:
                            metrics[k] = float(v)
                        except (ValueError, TypeError):
                            continue

                    if metrics:
                        models.append(
                            ModelMetric(
                                model_name=gc_name,
                                engine=gc_tags.get("engine"),
                                phase=tags.get("phase", display),
                                metrics=metrics,
                                is_champion=gc_tags.get("champion", "").lower() == "true",
                            )
                        )
            except Exception:
                pass

    except Exception:
        pass

    return MetricsResponse(job_name=job_name, task_type=task_type, models=models)


# ---------------------------------------------------------------------------
# Drift (Phase 0b)
# ---------------------------------------------------------------------------


def get_job_drift(job_name: str) -> DriftResponse:
    """Retrieve drift detection results from a completed pipeline job."""
    ml_client = get_ml_client()

    # Attempt to download drift output
    try:
        tmp = Path(tempfile.mkdtemp(prefix=f"mlops_drift_{job_name}_"))
        ml_client.jobs.download(
            job_name, download_path=str(tmp), output_name="drift_report"
        )
        # Look for drift result JSON
        import json

        for f in tmp.rglob("*.json"):
            try:
                data = json.loads(f.read_text())
                features = []
                for feat in data.get("feature_drift", []):
                    psi = float(feat.get("psi", 0))
                    severity = (
                        "severe" if psi >= 0.25
                        else "moderate" if psi >= 0.1
                        else "none"
                    )
                    features.append(
                        DriftResultItem(
                            feature=feat.get("feature", "unknown"),
                            psi=psi,
                            drift_detected=psi >= 0.1,
                            severity=severity,
                        )
                    )
                return DriftResponse(
                    job_name=job_name,
                    overall_drift_detected=data.get("drift_detected", False),
                    stability_score=data.get("stability_score"),
                    drift_type=data.get("drift_type"),
                    drifted_columns=data.get("drifted_columns", []),
                    features=features,
                    evidently_report_path=data.get("evidently_report_path"),
                )
            except (json.JSONDecodeError, KeyError):
                continue
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass

    return DriftResponse(job_name=job_name)


# ---------------------------------------------------------------------------
# Resubmit (Phase 0d)
# ---------------------------------------------------------------------------


def resubmit_pipeline(req: ResubmitRequest) -> SubmitResponse:
    """Resubmit a pipeline job using the same configuration as the original."""
    ml_client = get_ml_client()
    original = ml_client.jobs.get(req.job_name)
    tags = dict(original.tags) if original.tags else {}

    config_name = tags.get("config_name", "")
    if not config_name:
        # Infer from experiment name
        exp = getattr(original, "experiment_name", "") or ""
        config_name = f"config_{exp.replace('_v3', '')}_azureml"

    submit_req = SubmitRequest(
        config_name=config_name,
        force_rerun=req.force_rerun,
        tags={"resubmit_from": req.job_name},
    )
    return submit_pipeline(submit_req)
