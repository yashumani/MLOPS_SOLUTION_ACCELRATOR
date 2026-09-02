"""Pipeline submission, monitoring, and output retrieval service."""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from azure.ai.ml import MLClient
try:
    from azure.ai.ml.constants import ListViewType
except Exception:  # pragma: no cover - older SDKs may not expose this enum
    ListViewType = None  # type: ignore[assignment]

from api.core.azure_ml import get_ml_client
from api.core.config import settings
from api.services.auto_retrain_service import (
    load_config_metadata,
    validate_baseline_job,
)
from api.services.submission_request_store import (
    SubmissionRequestStoreError,
    create_request_record,
    get_request_record,
    update_request_record,
)
from api.schemas.pipeline import (
    BaselineCaptureRequest,
    BaselineCaptureResponse,
    DriftResponse,
    DriftResultItem,
    ExperimentNode,
    ExperimentTreeResponse,
    JobListResponse,
    JobStatus,
    JobSummary,
    LocalOutputFileInfo,
    LocalOutputsResponse,
    MetricsResponse,
    ModelMetric,
    OutputContentResponse,
    OutputFileInfo,
    OutputInfo,
    OutputListResponse,
    PipelineSummaryResponse,
    ResubmitRequest,
    StepStatus,
    SubmitRequest,
    SubmitResponse,
)
from api.utils.azure_links import build_studio_url

# Repo root so we can import pipeline_builder and read configs
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS_DIR = _REPO_ROOT / "configs"
_LOCAL_OUTPUTS_DIR = _REPO_ROOT / "outputs"
_CANONICAL_SUBMITTER = _REPO_ROOT / "pipelines" / "submit_pipeline.py"
_SAFE_CONFIG_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_SUBMIT_ERROR_LIMIT = 2000
_PROTECTED_SUBMISSION_TAGS = {
    "compiled_config_hash",
    "config_name",
    "dataset",
    "environment",
    "execution_id",
    "parent_config_hash",
    "parent_execution_id",
    "parent_source_identity",
    "pipeline_version",
    "preset",
    "recipe_catalog_hash",
    "revision_reason",
    "source",
    "source_decision_id",
    "source_identity",
    "submission_request_id",
    "submission_revision_kind",
    "task",
}


@dataclass(frozen=True)
class ReplaySubmissionContext:
    """Immutable parent identity and explicit semantics for a replay submission."""

    revision_kind: str
    parent_execution_id: str
    parent_config_hash: str
    parent_source_identity: str
    source_decision_id: str | None = None
    revision_reason: str | None = None

# In-memory cache for terminal-state job metrics/summaries.
# Outputs are immutable once a job is Completed/Failed/Canceled.
# Key: (function_name, job_name); Value: response object.
_TERMINAL_STATES = {"Completed", "Failed", "Canceled", "CancelRequested"}
_response_cache: dict[tuple[str, str], object] = {}

# Warm cache for the experiment→jobs tree, populated at startup and refreshed
# periodically by the background warmer in api.main.lifespan.
_logger = logging.getLogger(__name__)
_experiments_cache: dict = {
    "data": None,            # ExperimentTreeResponse | None
    "fetched_at": None,      # datetime | None (UTC)
    "max_per_experiment": None,  # int | None
    "duration_s": None,      # float | None
}
_experiments_lock = threading.Lock()

_STAGE_ALIASES = {
    "s01": "s1",
    "s02": "s2",
    "s03": "s3",
    "s04": "s4",
    "s05a": "s5a",
    "s05b": "s5b",
    "s05t": "s5t",
    "s05z": "s5z",
    "s6": "s06",
    "s08": "s08",
    "s8": "s08",
    "s09": "s09",
    "s9": "s09",
}

_STAGE_KEYWORDS = {
    "ingestion": "s1",
    "preparation": "s2",
    "preprocessing": "s3",
    "feature_engineering": "s4",
    "pycaret": "s5a",
    "flaml": "s5b",
    "timeseries": "s5t",
    "forecasting": "s5t",
    "aggregate_baseline": "s5z",
    "baseline_aggregate": "s5z",
    "variant_runner": "s06",
    "phaseb": "s06",
    "phasec_hpo": "s08",
    "optuna": "s08",
    "aggregate_phasec": "s09",
    "phasec_aggregate": "s09",
    "final_evaluation": "s10",
    "model_registration": "s12",
    "drift_monitor": "s13",
    "drift": "s13",
}

_CANONICAL_STAGE_LABELS = {
    "s1": "S01 Ingestion",
    "s2": "S02 Preparation",
    "s3": "S03 Preprocessing",
    "s4": "S04 Feature Engineering",
    "s5a": "S05a PyCaret Baseline",
    "s5b": "S05b FLAML Baseline",
    "s5t": "S05t Time-Series Baseline",
    "s5z": "S05z Baseline Aggregate",
    "s06": "S06 Phase B Variant Runner",
    "s08": "S08 Phase C Optuna HPO",
    "s09": "S09 Phase C Aggregate",
    "s10": "S10 Final Evaluation",
    "s12": "S12 Model Registration",
    "s13": "S13 Drift Monitor",
}

_OUTPUT_STAGE_FALLBACKS: dict[str, tuple[str, ...]] = {
    "eda_report": ("s1",),
    "prep_report": ("s2",),
    "prep3_report": ("s3",),
    "fe_report": ("s4",),
    "baseline_pycaret_metrics": ("s5a",),
    "baseline_flaml_metrics": ("s5b",),
    "baseline_aggregate_report": ("s5z",),
    "baseline_champion_model": ("s5z",),
    "phaseb_leaderboard": ("s06",),
    "phaseb_all_results": ("s06",),
    "phaseb_champion_manifest": ("s06",),
    "phaseb_champion_model": ("s06",),
    # s09 depends on s08, so this output confirms both for historical jobs
    # where Azure ML child step metadata is missing.
    "phasec_aggregate_report": ("s08", "s09"),
    "phasec_champion_model": ("s09",),
    "final_report": ("s10",),
    "final_champion_model": ("s10",),
    "registry_info": ("s12",),
    "drift_report": ("s13",),
    "drift_baseline": ("s13",),
}


def _infer_stage_key(*names: str | None) -> str | None:
    """Infer the canonical DSL stage key from Azure child job identifiers."""
    for raw in names:
        text = (raw or "").lower()
        if not text:
            continue
        for token in re.findall(r"s\d{1,2}[a-z]?", text):
            normalized = _STAGE_ALIASES.get(token, token)
            if normalized in {
                "s1", "s2", "s3", "s4", "s5a", "s5b", "s5t",
                "s5z", "s06", "s08", "s09", "s10", "s12", "s13",
            }:
                return normalized
        compact = text.replace("-", "_").replace(" ", "_")
        for keyword, stage_key in _STAGE_KEYWORDS.items():
            if keyword in compact:
                return stage_key
    return None


def _stage_display_name(stage_key: str | None, fallback: str | None = None) -> str | None:
    if stage_key:
        return _CANONICAL_STAGE_LABELS.get(stage_key, fallback or stage_key)
    return fallback


def _append_inferred_steps_from_outputs(
    steps: list[StepStatus],
    outputs: dict[str, Any],
) -> None:
    """Backfill completed stages from parent named outputs for old job records."""
    seen = {step.stage_key for step in steps if step.stage_key}
    for output_name, stage_keys in _OUTPUT_STAGE_FALLBACKS.items():
        if output_name not in outputs:
            continue
        for stage_key in stage_keys:
            if stage_key in seen:
                continue
            steps.append(
                StepStatus(
                    name=f"{stage_key}_inferred_from_{output_name}",
                    display_name=(
                        f"{_stage_display_name(stage_key)} "
                        f"(inferred from {output_name})"
                    ),
                    stage_key=stage_key,
                    status="Completed",
                    is_inferred=True,
                )
            )
            seen.add(stage_key)


def refresh_experiments_cache(max_per_experiment: int) -> "ExperimentTreeResponse":
    """Re-fetch the experiment tree and atomically update the warm cache.

    Returns the freshly fetched ExperimentTreeResponse. Safe to call from
    a thread executor; uses an internal lock so concurrent refreshes coalesce.
    """
    t0 = time.monotonic()
    data = list_experiments(max_results_per_experiment=max_per_experiment)
    duration = time.monotonic() - t0
    with _experiments_lock:
        _experiments_cache["data"] = data
        _experiments_cache["fetched_at"] = datetime.utcnow()
        _experiments_cache["max_per_experiment"] = max_per_experiment
        _experiments_cache["duration_s"] = duration
    _logger.info(
        "warmed experiments cache: %d exp / %d jobs in %.1fs (max_per_experiment=%d)",
        data.total_experiments,
        data.total_jobs,
        duration,
        max_per_experiment,
    )
    return data


def get_cached_experiments() -> tuple["ExperimentTreeResponse | None", dict]:
    """Return (data, meta) from the warm cache without triggering a refresh.

    meta keys: fetched_at (datetime|None), age_seconds (int|None),
    max_per_experiment (int|None), duration_s (float|None).
    """
    with _experiments_lock:
        data = _experiments_cache["data"]
        fetched_at = _experiments_cache["fetched_at"]
        max_per_experiment = _experiments_cache["max_per_experiment"]
        duration_s = _experiments_cache["duration_s"]
    age = None
    if fetched_at is not None:
        age = int((datetime.utcnow() - fetched_at).total_seconds())
    return data, {
        "fetched_at": fetched_at,
        "age_seconds": age,
        "max_per_experiment": max_per_experiment,
        "duration_s": duration_s,
    }


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


# _studio_url removed — use build_studio_url(ml_client, job_name) instead


def _load_config_yaml(config_name: str) -> dict:
    if not _SAFE_CONFIG_NAME.fullmatch(config_name):
        raise ValueError(f"Invalid config name: {config_name!r}")
    path = (_CONFIGS_DIR / f"{config_name}.yml").resolve()
    if _CONFIGS_DIR.resolve() not in path.parents:
        raise ValueError(f"Invalid config path for: {config_name!r}")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_name}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_name}")
    try:
        from src.orchestration.config_schema import validate_config

        validate_config(cfg)
    except Exception as exc:
        raise ValueError(f"Config validation failed for {config_name}: {exc}") from exc
    return cfg


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def _resolve_baseline_uri(
    baseline_job: str | None,
    *,
    config_name: str,
) -> str | None:
    """Resolve only a completed, identity-matched baseline job output."""
    if not baseline_job:
        return None

    config_path = (_CONFIGS_DIR / f"{config_name}.yml").resolve()
    metadata = load_config_metadata(config_path)
    baseline_uri, _, _ = validate_baseline_job(
        config_path=config_path,
        metadata=metadata,
        baseline_job_name=baseline_job,
        requested_uri=None,
    )
    return baseline_uri


def _append_cli_option(command: list[str], flag: str, value: str | None) -> None:
    if value:
        command.extend((flag, value))


def _build_canonical_submit_command(
    req: SubmitRequest,
    *,
    result_path: Path,
    baseline_uri: str | None,
    replay_context: ReplaySubmissionContext | None = None,
    internal_tags: dict[str, str] | None = None,
) -> tuple[list[str], str, str]:
    """Build the canonical submitter command used by sync and async API paths."""
    config_path = (_CONFIGS_DIR / f"{req.config_name}.yml").resolve()
    experiment_name = _derive_experiment_name(req.config_name)
    display_name = _derive_display_name(experiment_name)

    command = [
        sys.executable,
        str(_CANONICAL_SUBMITTER),
        "--config",
        str(config_path),
        "--experiment_name",
        experiment_name,
        "--display_name",
        display_name,
        "--result_json",
        str(result_path),
    ]
    _append_cli_option(
        command,
        "--subscription_id",
        settings.azure_subscription_id,
    )
    _append_cli_option(
        command,
        "--resource_group",
        settings.azure_resource_group,
    )
    _append_cli_option(
        command,
        "--workspace_name",
        settings.azure_workspace_name,
    )
    _append_cli_option(
        command,
        "--compute",
        req.compute or settings.compute_target,
    )
    _append_cli_option(command, "--drift_baseline_in", baseline_uri)

    if replay_context is not None:
        command.extend(
            (
                "--submission_revision_kind",
                replay_context.revision_kind,
                "--parent_execution_id",
                replay_context.parent_execution_id,
                "--parent_config_hash",
                replay_context.parent_config_hash,
                "--parent_source_identity",
                replay_context.parent_source_identity,
            )
        )
        if replay_context.revision_kind in {"exact_replay", "decision_retrain"}:
            command.extend(
                (
                    "--expected_execution_id",
                    replay_context.parent_execution_id,
                    "--expected_config_hash",
                    replay_context.parent_config_hash,
                    "--expected_source_identity",
                    replay_context.parent_source_identity,
                )
            )
        _append_cli_option(
            command,
            "--source_decision_id",
            replay_context.source_decision_id,
        )
        _append_cli_option(
            command,
            "--revision_reason",
            replay_context.revision_reason,
        )

    if req.force_rerun:
        command.append("--force_rerun")

    protected = sorted(_PROTECTED_SUBMISSION_TAGS.intersection(req.tags))
    if protected:
        raise ValueError(
            "Request tags cannot override protected submission metadata: "
            + ", ".join(protected)
        )
    api_tags = {"source": "api", **req.tags, **dict(internal_tags or {})}
    command.extend(
        (
            "--tags_json",
            json.dumps(api_tags, separators=(",", ":"), sort_keys=True),
        )
    )
    return command, experiment_name, display_name


def _canonical_failure_detail(completed: subprocess.CompletedProcess[str]) -> str:
    """Return bounded, operator-useful output without echoing the full process log."""
    output = (completed.stderr or completed.stdout or "").strip()
    if not output:
        return "canonical submitter returned no diagnostic output"
    return output[-_SUBMIT_ERROR_LIMIT:]


def _load_canonical_submit_result(
    result_path: Path,
    *,
    experiment_name: str,
    display_name: str,
) -> SubmitResponse:
    if not result_path.is_file():
        raise RuntimeError(
            "Canonical submitter completed without a structured result. "
            "Submission state is unknown; inspect the submitter logs before retrying."
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Canonical submitter returned an unreadable structured result. "
            "Submission state is unknown; inspect the submitter logs before retrying."
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Canonical submitter result must be a JSON object.")

    expected = {
        "experiment_name": experiment_name,
        "display_name": display_name,
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            raise RuntimeError(
                f"Canonical submitter result field {field!r} did not match the request."
            )

    required_fields = ("job_name", "status", "studio_url")
    missing = [field for field in required_fields if not payload.get(field)]
    if missing:
        raise RuntimeError(
            "Canonical submitter result is missing required fields: "
            + ", ".join(missing)
        )
    return SubmitResponse(**payload)


def submit_pipeline(
    req: SubmitRequest,
    *,
    replay_context: ReplaySubmissionContext | None = None,
    internal_tags: dict[str, str] | None = None,
) -> SubmitResponse:
    """Submit through the canonical CLI so every caller shares its safety guards."""
    _load_config_yaml(req.config_name)
    if not _CANONICAL_SUBMITTER.is_file():
        raise FileNotFoundError(
            f"Canonical pipeline submitter not found: {_CANONICAL_SUBMITTER}"
        )

    baseline_uri = _resolve_baseline_uri(
        req.baseline_job,
        config_name=req.config_name,
    )
    with tempfile.TemporaryDirectory(prefix="mlops-api-submit-") as temp_dir:
        result_path = Path(temp_dir) / "submission-result.json"
        command, experiment_name, display_name = _build_canonical_submit_command(
            req,
            result_path=result_path,
            baseline_uri=baseline_uri,
            replay_context=replay_context,
            internal_tags=internal_tags,
        )
        completed = subprocess.run(
            command,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Canonical pipeline submission failed "
                f"(exit code {completed.returncode}): "
                f"{_canonical_failure_detail(completed)}"
            )
        return _load_canonical_submit_result(
            result_path,
            experiment_name=experiment_name,
            display_name=display_name,
        )


# ---------------------------------------------------------------------------
# Async submit (Phase 4) - durable fire-and-poll request state
# ---------------------------------------------------------------------------

# Azure ML job creation can take 5–60s wall time on cold compute. The async
# submit endpoint hands the work off to a background thread so the HTTP request
# returns in <100ms. Clients poll /submit/status/{request_id} until job_name
# appears, then switch to the standard /pipelines/jobs/{job_name} endpoints.

_submit_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="submit-async")


def _new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:12]}"


def _submit_worker(request_id: str, req: SubmitRequest) -> None:
    """Background worker that performs the blocking Azure ML submission."""
    try:
        result = submit_pipeline(
            req,
            internal_tags={"submission_request_id": request_id},
        )
        update_request_record(
            request_id,
            {
                "status": "submitted",
                "job_name": result.job_name,
                "experiment_name": result.experiment_name,
                "display_name": result.display_name,
                "studio_url": result.studio_url,
                "completed_at": datetime.utcnow().isoformat() + "Z",
            },
        )
    except Exception as exc:  # noqa: BLE001 — record any failure for the poller
        try:
            update_request_record(
                request_id,
                {
                    "status": "failed",
                    "error": str(exc),
                    "completed_at": datetime.utcnow().isoformat() + "Z",
                },
            )
        except SubmissionRequestStoreError:
            _logger.exception(
                "Async submission failed and durable request state could not be updated: %s",
                request_id,
            )


def submit_pipeline_async(req: SubmitRequest) -> dict:
    """Enqueue a pipeline submission and return immediately with a request_id."""
    request_id = _new_request_id()
    record = {
        "request_id": request_id,
        "status": "pending",
        "config_name": req.config_name,
        "submitted_at": datetime.utcnow().isoformat() + "Z",
        "job_name": None,
        "experiment_name": None,
        "display_name": None,
        "studio_url": None,
        "error": None,
        "completed_at": None,
    }
    create_request_record(record)
    _submit_executor.submit(_submit_worker, request_id, req)
    return dict(record)


def _pending_request_is_stale(record: dict[str, Any]) -> bool:
    try:
        threshold = max(
            0,
            int(os.environ.get("MLOPS_SUBMISSION_RECONCILE_AFTER_SECONDS", "300")),
        )
    except ValueError:
        threshold = 300
    raw_submitted_at = str(record.get("submitted_at") or "")
    try:
        submitted_at = datetime.fromisoformat(raw_submitted_at.replace("Z", "+00:00"))
        if submitted_at.tzinfo is None:
            submitted_at = submitted_at.replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    age_seconds = (datetime.now(timezone.utc) - submitted_at).total_seconds()
    return age_seconds >= threshold


def _reconcile_stale_submit_request(record: dict[str, Any]) -> dict[str, Any]:
    """Recover job identity after an API restart without submitting again."""

    request_id = str(record["request_id"])
    checked_at = datetime.now(timezone.utc).isoformat()
    try:
        ml_client = get_ml_client()
        matches = []
        for job in ml_client.jobs.list():
            tags = dict(getattr(job, "tags", None) or {})
            if tags.get("submission_request_id") == request_id:
                matches.append(job)
        if len(matches) == 1:
            job = matches[0]
            return update_request_record(
                request_id,
                {
                    "status": "submitted",
                    "job_name": str(job.name),
                    "experiment_name": getattr(job, "experiment_name", None),
                    "display_name": getattr(job, "display_name", None),
                    "studio_url": build_studio_url(ml_client, str(job.name)),
                    "completed_at": checked_at,
                    "reconciled_at": checked_at,
                    "error": None,
                },
            )
        detail = (
            "multiple Azure jobs carry this request ID"
            if len(matches) > 1
            else "no Azure job carrying this request ID was found"
        )
    except Exception as exc:  # noqa: BLE001 - preserve uncertainty instead of retrying a write
        detail = f"Azure reconciliation failed: {type(exc).__name__}: {exc}"

    return update_request_record(
        request_id,
        {
            "status": "reconciliation_required",
            "error": (
                f"Async submission state is uncertain because {detail}. "
                "Inspect Azure ML and the canonical submitter logs before retrying."
            ),
            "reconciled_at": checked_at,
        },
    )


def get_submit_request(request_id: str) -> dict | None:
    """Return durable async request state, or None if the ID is unknown."""
    try:
        record = get_request_record(request_id)
        if (
            record is not None
            and record.get("status") == "pending"
            and _pending_request_is_stale(record)
        ):
            return _reconcile_stale_submit_request(record)
        return record
    except SubmissionRequestStoreError as exc:
        if "Invalid submission request ID" in str(exc):
            return None
        raise


# ---------------------------------------------------------------------------
# List / Get / Cancel
# ---------------------------------------------------------------------------


def list_jobs(
    experiment_name: str | None = None,
    status_filter: str | None = None,
    max_results: int = 50,
) -> JobListResponse:
    ml_client = get_ml_client()

    jobs: list[JobSummary] = []
    for j in ml_client.jobs.list():
        if experiment_name and getattr(j, "experiment_name", None) != experiment_name:
            continue
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
                studio_url=build_studio_url(ml_client, j.name),
            )
        )
        if len(jobs) >= max_results:
            break

    return JobListResponse(jobs=jobs, total=len(jobs))


def list_experiments(max_results_per_experiment: int = 100) -> ExperimentTreeResponse:
    """Return all jobs grouped by experiment for hierarchical UI pickers.

    Each experiment node lists its jobs (most recent first) so the UI can
    render a two-level tree: experiment → display_name (job).
    """
    ml_client = get_ml_client()

    # Single pass over the workspace job list, group by experiment.
    grouped: dict[str, list[JobSummary]] = {}
    last_activity: dict[str, datetime] = {}

    for j in ml_client.jobs.list():
        exp = getattr(j, "experiment_name", None) or "(no experiment)"
        if len(grouped.get(exp, [])) >= max_results_per_experiment:
            continue
        created = (
            getattr(j, "creation_context", None)
            and getattr(j.creation_context, "created_at", None)
        )
        summary = JobSummary(
            job_name=j.name,
            experiment_name=exp,
            display_name=getattr(j, "display_name", None) or j.name,
            status=j.status or "Unknown",
            start_time=created,
            studio_url=build_studio_url(ml_client, j.name),
        )
        grouped.setdefault(exp, []).append(summary)
        if created and (exp not in last_activity or created > last_activity[exp]):
            last_activity[exp] = created

    # Sort jobs in each experiment by start_time desc (most recent first)
    nodes: list[ExperimentNode] = []
    for exp, jobs in grouped.items():
        jobs.sort(key=lambda x: x.start_time or datetime.min, reverse=True)
        nodes.append(
            ExperimentNode(
                experiment_name=exp,
                job_count=len(jobs),
                last_activity=last_activity.get(exp),
                jobs=jobs,
            )
        )
    nodes.sort(key=lambda n: n.last_activity or datetime.min, reverse=True)

    return ExperimentTreeResponse(
        experiments=nodes,
        total_experiments=len(nodes),
        total_jobs=sum(n.job_count for n in nodes),
    )


def get_job(job_name: str) -> JobStatus:
    ml_client = get_ml_client()
    j = ml_client.jobs.get(job_name)

    # Fetch child step statuses
    steps: list[StepStatus] = []
    try:
        list_kwargs: dict[str, Any] = {"parent_job_name": job_name}
        if ListViewType is not None:
            list_kwargs["list_view_type"] = ListViewType.ALL
        for child in ml_client.jobs.list(**list_kwargs):
            child_name = child.name
            azure_display_name = getattr(child, "display_name", None)
            properties = getattr(child, "properties", None) or {}
            module_name = properties.get("azureml.moduleName")
            stage_key = _infer_stage_key(azure_display_name, child_name, module_name)
            steps.append(
                StepStatus(
                    name=child_name,
                    display_name=_stage_display_name(stage_key, azure_display_name),
                    stage_key=stage_key,
                    status=child.status or "Unknown",
                    start_time=getattr(child, "creation_context", None)
                    and getattr(child.creation_context, "created_at", None),
                    end_time=None,
                )
            )
    except Exception:
        _logger.exception("failed to list child steps for job %s", job_name)

    _append_inferred_steps_from_outputs(steps, j.outputs or {})

    return JobStatus(
        job_name=j.name,
        experiment_name=getattr(j, "experiment_name", None),
        display_name=getattr(j, "display_name", None),
        status=j.status or "Unknown",
        start_time=getattr(j, "creation_context", None)
        and getattr(j.creation_context, "created_at", None),
        end_time=None,
        studio_url=build_studio_url(ml_client, j.name),
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
# Output content preview (for UI rendering)
# ---------------------------------------------------------------------------

_TEXT_EXTENSIONS = {
    ".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".md", ".html",
    ".yml", ".yaml", ".py", ".ipynb",
}
_BINARY_EXTENSIONS = {".pkl", ".joblib", ".bin", ".onnx", ".pt", ".pb", ".h5", ".parquet"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}

_TEXT_PREVIEW_BYTES = 200_000  # ~200KB cap
_CSV_PREVIEW_ROWS = 200


def _classify_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".json" or ext == ".jsonl":
        return "json"
    if ext == ".csv" or ext == ".tsv":
        return "csv"
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in {".html", ".htm"}:
        return "html"
    if ext in {".md", ".markdown"}:
        return "markdown"
    if ext in {".yml", ".yaml"}:
        return "yaml"
    if ext in _BINARY_EXTENSIONS:
        return "binary"
    if ext in _TEXT_EXTENSIONS:
        return "text"
    # No / unknown extension — sniff the first bytes
    if not ext:
        try:
            with open(path, "rb") as fh:
                head = fh.read(2048)
            stripped = head.lstrip()
            if stripped[:1] in (b"{", b"["):
                return "json"
            # ASCII-printable test
            if head and sum(1 for b in head if 9 <= b <= 13 or 32 <= b <= 126) / len(head) > 0.85:
                return "text"
        except OSError:
            pass
    return "binary"


def get_output_content(job_name: str, output_name: str) -> OutputContentResponse:
    """Download an output and return its parsed content for UI rendering.

    Picks the most useful file in the output and returns:
      - json_content for JSON files
      - csv_preview (first N rows as list of dicts) for CSV
      - text_preview (truncated) for text/yaml/html/md
    """
    import csv
    import json

    ml_client = get_ml_client()
    tmp = Path(tempfile.mkdtemp(prefix=f"mlops_preview_{job_name}_"))
    files_info: list[OutputFileInfo] = []
    json_content: Any | None = None
    text_preview: str | None = None
    csv_preview: list[dict] | None = None
    primary_file: str | None = None
    truncated = False

    try:
        ml_client.jobs.download(
            job_name, download_path=str(tmp), output_name=output_name
        )
    except Exception as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"Failed to download output '{output_name}': {exc}")

    try:
        # Collect file metadata
        all_files = sorted(
            [f for f in tmp.rglob("*") if f.is_file()],
            key=lambda p: (p.suffix.lower() != ".json", p.suffix.lower() != ".csv", str(p)),
        )

        for f in all_files:
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            files_info.append(OutputFileInfo(
                name=f.name,
                relative_path=str(f.relative_to(tmp)),
                size_bytes=size,
                kind=_classify_file(f),
            ))

        # Pick primary file: prefer JSON, then CSV, then text
        primary: Path | None = None
        for kind in ("json", "csv", "yaml", "markdown", "html", "text"):
            for f in all_files:
                if _classify_file(f) == kind:
                    primary = f
                    break
            if primary:
                break

        if primary:
            primary_file = str(primary.relative_to(tmp))
            kind = _classify_file(primary)
            try:
                if kind == "json":
                    if primary.suffix.lower() == ".jsonl":
                        rows = []
                        with open(primary) as fh:
                            for i, line in enumerate(fh):
                                if i >= _CSV_PREVIEW_ROWS:
                                    truncated = True
                                    break
                                line = line.strip()
                                if line:
                                    try:
                                        rows.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        pass
                        json_content = rows
                    else:
                        with open(primary) as fh:
                            json_content = json.load(fh)
                elif kind == "csv":
                    delim = "\t" if primary.suffix.lower() == ".tsv" else ","
                    rows: list[dict] = []
                    with open(primary, newline="") as fh:
                        reader = csv.DictReader(fh, delimiter=delim)
                        for i, row in enumerate(reader):
                            if i >= _CSV_PREVIEW_ROWS:
                                truncated = True
                                break
                            rows.append(row)
                    csv_preview = rows
                elif kind in ("text", "yaml", "markdown", "html"):
                    raw = primary.read_bytes()
                    if len(raw) > _TEXT_PREVIEW_BYTES:
                        raw = raw[:_TEXT_PREVIEW_BYTES]
                        truncated = True
                    text_preview = raw.decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 — preview is best-effort
                pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return OutputContentResponse(
        job_name=job_name,
        output_name=output_name,
        files=files_info,
        json_content=json_content,
        text_preview=text_preview,
        csv_preview=csv_preview,
        primary_file=primary_file,
        truncated=truncated,
    )


def list_local_outputs(max_depth: int = 4, max_files: int = 500) -> LocalOutputsResponse:
    """Return a read-only inventory of the repo-local outputs/ folder."""
    max_depth = max(1, min(max_depth, 10))
    max_files = max(1, min(max_files, 2_000))

    if not _LOCAL_OUTPUTS_DIR.exists():
        return LocalOutputsResponse(files=[], total=0, truncated=False)

    items: list[LocalOutputFileInfo] = []
    truncated = False
    base = _LOCAL_OUTPUTS_DIR.resolve()

    for path in sorted(base.rglob("*"), key=lambda p: str(p).lower()):
        try:
            rel = path.relative_to(base)
        except ValueError:
            continue
        if any(part.startswith("__pycache__") for part in rel.parts):
            continue
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        if len(items) >= max_files:
            truncated = True
            break
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append(
            LocalOutputFileInfo(
                relative_path=rel.as_posix(),
                name=path.name,
                is_dir=path.is_dir(),
                size_bytes=None if path.is_dir() else stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime),
                kind="directory" if path.is_dir() else _classify_file(path),
                depth=depth,
            )
        )

    return LocalOutputsResponse(
        files=items,
        total=len(items),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Pipeline summary (combined aggregate reports)
# ---------------------------------------------------------------------------


def get_pipeline_summary(job_name: str) -> PipelineSummaryResponse:
    """Return all four aggregate JSON reports for a job in one structured payload."""
    ml_client = get_ml_client()
    parent = ml_client.jobs.get(job_name)
    task_type = (parent.tags or {}).get("task")
    status = getattr(parent, "status", None)
    available = list((parent.outputs or {}).keys())

    cache_key = ("summary", job_name)
    if str(status or "") in _TERMINAL_STATES and cache_key in _response_cache:
        return _response_cache[cache_key]  # type: ignore[return-value]

    report_names = [
        "baseline_aggregate_report",
        "phaseb_aggregate_report",
        "phasec_aggregate_report",
        "final_report",
    ]

    reports: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        def _fetch(output_name: str) -> tuple[str, Any]:
            if output_name not in available:
                return output_name, None
            target = tmp_root / output_name
            try:
                ml_client.jobs.download(
                    name=job_name, output_name=output_name, download_path=str(target)
                )
                return output_name, _read_json_report(target)
            except Exception:
                return output_name, None

        with ThreadPoolExecutor(max_workers=4) as pool:
            for name, data in pool.map(_fetch, report_names):
                reports[name] = data

    final = reports.get("final_report") or {}
    selection = final.get("selection") or {}

    response = PipelineSummaryResponse(
        job_name=job_name,
        task_type=task_type,
        status=status,
        champion_phase=selection.get("key"),
        champion_score=selection.get("score"),
        baseline_aggregate=reports.get("baseline_aggregate_report"),
        phaseb_aggregate=reports.get("phaseb_aggregate_report"),
        phasec_aggregate=reports.get("phasec_aggregate_report"),
        final_report=final or None,
        available_outputs=available,
    )
    if str(status or "") in _TERMINAL_STATES:
        _response_cache[cache_key] = response
    return response


# ---------------------------------------------------------------------------
# Metrics (Phase 0a)
# ---------------------------------------------------------------------------


def _read_json_report(download_dir: Path) -> dict | None:
    """Find and load the first JSON file under download_dir."""
    import json

    if not download_dir.exists():
        return None
    for f in download_dir.rglob("*"):
        if f.is_file():
            try:
                with open(f) as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                continue
    return None


def get_job_metrics(job_name: str) -> MetricsResponse:
    """Retrieve per-phase metrics from the pipeline's aggregate JSON reports.

    The V3 pipeline writes per-phase scores to named outputs:
      - baseline_aggregate_report (phase A)
      - phaseb_aggregate_report (phase B)
      - phasec_aggregate_report (phase C)
      - final_report (holdout metrics per phase + champion selection)
    """
    ml_client = get_ml_client()
    parent = ml_client.jobs.get(job_name)
    task_type = (parent.tags or {}).get("task", None)
    job_status = str(getattr(parent, "status", "") or "")

    cache_key = ("metrics", job_name)
    if job_status in _TERMINAL_STATES and cache_key in _response_cache:
        return _response_cache[cache_key]  # type: ignore[return-value]

    models: list[ModelMetric] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)

        def _download(output_name: str) -> dict | None:
            target = tmp_root / output_name
            try:
                ml_client.jobs.download(
                    name=job_name, output_name=output_name, download_path=str(target)
                )
            except Exception:
                return None
            return _read_json_report(target)

        # Parallel download: 4 named outputs concurrently (~4x faster)
        names = [
            "baseline_aggregate_report",
            "phaseb_aggregate_report",
            "phasec_aggregate_report",
            "final_report",
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(_download, names))
        baseline_agg = results[0] or {}
        phaseb_agg = results[1] or {}
        phasec_agg = results[2] or {}
        final_rep = results[3] or {}

        champion_phase = (final_rep.get("selection") or {}).get("key")

        # Phase A (baseline)
        baseline_metrics = final_rep.get("baseline_metrics") or {}
        baseline_engine = (baseline_agg.get("selection") or {}).get("source")
        if baseline_metrics or baseline_agg:
            score = (baseline_agg.get("selection") or {}).get("score")
            if score is not None and "score" not in baseline_metrics:
                baseline_metrics = {**baseline_metrics, "score": float(score)}
            models.append(
                ModelMetric(
                    model_name=f"baseline ({baseline_engine})" if baseline_engine else "baseline",
                    engine=baseline_engine,
                    phase="baseline",
                    metrics={k: float(v) for k, v in baseline_metrics.items()
                             if isinstance(v, (int, float))},
                    is_champion=(champion_phase == "baseline"),
                )
            )

        # Phase B
        phaseb_metrics = final_rep.get("phaseb_metrics") or {}
        phaseb_sel = (phaseb_agg.get("selection") or {})
        phaseb_key = phaseb_sel.get("key", "")
        phaseb_engine = None
        if "_" in phaseb_key:
            phaseb_engine = phaseb_key.rsplit("_", 1)[-1]
        if phaseb_metrics or phaseb_agg:
            if phaseb_sel.get("score") is not None and "score" not in phaseb_metrics:
                phaseb_metrics = {**phaseb_metrics, "score": float(phaseb_sel["score"])}
            models.append(
                ModelMetric(
                    model_name=f"phaseB ({phaseb_key})" if phaseb_key else "phaseB",
                    engine=phaseb_engine,
                    phase="phaseB",
                    metrics={k: float(v) for k, v in phaseb_metrics.items()
                             if isinstance(v, (int, float))},
                    is_champion=(champion_phase == "phaseb"),
                )
            )

        # Phase C
        phasec_metrics = final_rep.get("phasec_metrics") or {}
        if phasec_metrics or phasec_agg:
            phasec_sel = (phasec_agg.get("selection") or {})
            if phasec_sel.get("score") is not None and "score" not in phasec_metrics:
                phasec_metrics = {**phasec_metrics, "score": float(phasec_sel["score"])}
            models.append(
                ModelMetric(
                    model_name=f"phaseC ({phasec_sel.get('key','hpo')})",
                    engine=None,
                    phase="phaseC",
                    metrics={k: float(v) for k, v in phasec_metrics.items()
                             if isinstance(v, (int, float))},
                    is_champion=(champion_phase == "phasec"),
                )
            )

    response = MetricsResponse(job_name=job_name, task_type=task_type, models=models)
    if job_status in _TERMINAL_STATES:
        _response_cache[cache_key] = response
    return response


# ---------------------------------------------------------------------------
# Drift (Phase 0b)
# ---------------------------------------------------------------------------


_DRIFT_IDENTITY_FIELDS = (
    "execution_id",
    "config_hash",
    "config_revision",
    "candidate_id",
    "model_bundle_id",
    "data_fingerprint",
    "dataset_version",
    "source_sha",
    "environment_hash",
    "split_fingerprint",
)


def _download_json_output(
    ml_client: MLClient,
    job_name: str,
    output_name: str,
    download_root: Path,
) -> dict[str, Any] | None:
    """Download one named output into an isolated folder and parse its JSON object."""
    output_dir = download_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        ml_client.jobs.download(
            job_name,
            download_path=str(output_dir),
            output_name=output_name,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download {output_name}: {exc}") from exc

    candidates = sorted(
        (path for path in output_dir.rglob("*") if path.is_file()),
        key=lambda path: (path.suffix.lower() != ".json", str(path)),
    )
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict):
            return payload

    _logger.warning(
        "%s download had no parseable JSON object; files=%s",
        output_name,
        [str(path.relative_to(output_dir)) for path in candidates],
    )
    return None


def _join_s14_retrain_decision(
    drift_report: dict[str, Any],
    s14_payload: dict[str, Any] | None,
    *,
    missing_status: str,
) -> tuple[dict[str, Any], str | None]:
    """Return an S14 decision only when it is identity-bound to S13 evidence."""
    base = {
        "available": False,
        "source_stage": "s14_retrain_decision",
        "source_output": "retrain_decision",
        "join_status": missing_status,
    }
    if not s14_payload:
        return base, (
            "S14 retrain_decision output is not available; no policy decision is reported."
        )

    s13_identity = drift_report.get("identity") or {}
    s14_identity = s14_payload.get("identity") or {}
    if not isinstance(s13_identity, dict) or not isinstance(s14_identity, dict):
        base["join_status"] = "identity_unverified"
        return base, "S13/S14 identity metadata is malformed; the S14 decision was withheld."

    s13_execution_id = s13_identity.get("execution_id")
    source = s14_payload.get("source") or {}
    s14_execution_id = s14_identity.get("execution_id") or (
        source.get("drift_execution_id") if isinstance(source, dict) else None
    )
    if not s13_execution_id or not s14_execution_id:
        base["join_status"] = "identity_unverified"
        return base, (
            "S13/S14 execution identity is missing; the S14 decision was withheld."
        )

    mismatched_fields = [
        field
        for field in _DRIFT_IDENTITY_FIELDS
        if s13_identity.get(field) not in (None, "")
        and s14_identity.get(field) not in (None, "")
        and str(s13_identity[field]) != str(s14_identity[field])
    ]
    for field in ("config_name", "task_type", "dataset_name"):
        if (
            drift_report.get(field) not in (None, "")
            and s14_payload.get(field) not in (None, "")
            and str(drift_report[field]) != str(s14_payload[field])
        ):
            mismatched_fields.append(field)
    if str(s13_execution_id) != str(s14_execution_id):
        mismatched_fields.append("execution_id")
    if mismatched_fields:
        mismatch_names = sorted(set(mismatched_fields))
        base.update({
            "join_status": "identity_mismatch",
            "mismatched_fields": mismatch_names,
        })
        return base, (
            "S13/S14 identity mismatch; the S14 decision was withheld: "
            + ", ".join(mismatch_names)
        )

    decision = s14_payload.get("retrain_decision") or s14_payload.get("decision") or {}
    if not isinstance(decision, dict) or not decision.get("outcome"):
        base["join_status"] = "malformed_decision"
        return base, "S14 output has no valid retrain decision; policy status was withheld."
    artifact_decision_id = s14_payload.get("decision_id")
    contract_decision_id = decision.get("decision_id")
    if not artifact_decision_id or not contract_decision_id:
        base["join_status"] = "decision_identity_unverified"
        return base, "S14 decision identity is missing; policy status was withheld."
    if str(artifact_decision_id) != str(contract_decision_id):
        base["join_status"] = "decision_identity_mismatch"
        return base, "S14 decision identity is inconsistent; policy status was withheld."

    return {
        **decision,
        "available": True,
        "source_stage": str(s14_payload.get("stage") or "s14_retrain_decision"),
        "source_output": "retrain_decision",
        "join_status": "matched",
        "matched_execution_id": str(s13_execution_id),
        "decision_id": str(contract_decision_id),
        "policy": s14_payload.get("policy") or {},
    }, None


def _effective_psi_thresholds(
    auto_retrain_decision: dict[str, Any],
) -> tuple[float, float]:
    """Use the policy recorded by S14, with legacy PSI bands as a fallback."""
    effective = (auto_retrain_decision.get("policy") or {}).get("effective") or {}
    try:
        moderate = float(effective.get("moderate_feature_psi", 0.10))
        severe = float(effective.get("severe_feature_psi", 0.25))
    except (TypeError, ValueError):
        return 0.10, 0.25
    if moderate < 0 or severe <= moderate:
        return 0.10, 0.25
    return moderate, severe


def get_job_drift(job_name: str) -> DriftResponse:
    """Join S13 drift evidence with the identity-matched S14 policy decision."""
    ml_client = get_ml_client()
    parent = ml_client.jobs.get(job_name)
    available = list((parent.outputs or {}).keys())

    if "drift_report" not in available:
        return DriftResponse(
            job_name=job_name,
            studio_url=build_studio_url(ml_client, job_name),
        )

    cache_key = ("drift", job_name)
    status = str(getattr(parent, "status", "") or "")
    if status in _TERMINAL_STATES and cache_key in _response_cache:
        return _response_cache[cache_key]  # type: ignore[return-value]

    tmp = Path(tempfile.mkdtemp(prefix=f"mlops_drift_{job_name}_"))
    try:
        report = _download_json_output(ml_client, job_name, "drift_report", tmp)
        if not report:
            return DriftResponse(
                job_name=job_name,
                studio_url=build_studio_url(ml_client, job_name),
            )

        s14_payload: dict[str, Any] | None = None
        s14_download_warning: str | None = None
        if "retrain_decision" in available:
            try:
                s14_payload = _download_json_output(
                    ml_client,
                    job_name,
                    "retrain_decision",
                    tmp,
                )
            except RuntimeError as exc:
                s14_download_warning = str(exc)
        missing_status = "pending" if status not in _TERMINAL_STATES else "missing_output"
        auto_retrain_decision, join_warning = _join_s14_retrain_decision(
            report,
            s14_payload,
            missing_status=missing_status,
        )

        self_check = report.get("self_check") or report.get("smoke_test") or {}
        stability = report.get("stability_assessment") or {}
        comparison = report.get("comparison_drift") or {}
        if not isinstance(self_check, dict):
            self_check = {}
        if not isinstance(stability, dict):
            stability = {}
        if not isinstance(comparison, dict):
            comparison = {}

        self_check_psi_scores = (
            report.get("feature_psi_scores")
            or report.get("psi_scores")
            or report.get("psi_per_feature")
            or {}
        )
        comparison_psi_scores = (
            comparison.get("feature_psi_scores")
            or comparison.get("comparison_feature_psi_scores")
            or comparison.get("psi_scores")
            or {}
        )
        if not isinstance(self_check_psi_scores, dict):
            self_check_psi_scores = {}
        if not isinstance(comparison_psi_scores, dict):
            comparison_psi_scores = {}
        if not self_check_psi_scores:
            _logger.warning(
                "drift_report missing per-feature PSI; top-level keys=%s",
                list(report.keys()),
            )

        comparison_available = bool(comparison.get("available"))
        psi_scores = (
            comparison_psi_scores
            if comparison_available and comparison_psi_scores
            else self_check_psi_scores
        )
        moderate_threshold, severe_threshold = _effective_psi_thresholds(
            auto_retrain_decision
        )
        features: list[DriftResultItem] = []
        for feat_name, raw in psi_scores.items():
            if isinstance(raw, dict):
                raw = next(
                    (
                        raw.get(key)
                        for key in ("psi", "score", "value")
                        if raw.get(key) is not None
                    ),
                    None,
                )
            try:
                psi_value = float(raw)
            except (TypeError, ValueError):
                continue
            severity = (
                "severe"
                if psi_value >= severe_threshold
                else "moderate"
                if psi_value >= moderate_threshold
                else "none"
            )
            features.append(
                DriftResultItem(
                    feature=str(feat_name),
                    psi=psi_value,
                    drift_detected=psi_value >= moderate_threshold,
                    severity=severity,
                )
            )
        features.sort(key=lambda item: item.psi, reverse=True)

        baseline_status = comparison.get("baseline_status")
        if not baseline_status:
            baseline_status = (
                "comparison_ready" if comparison_available else "not_available"
            )
        baseline_metadata = comparison.get("baseline_metadata") or {}
        evidently = comparison.get("evidently") or {}
        concept = comparison.get("concept_drift") or {}
        if not isinstance(baseline_metadata, dict):
            baseline_metadata = {}
        if not isinstance(evidently, dict):
            evidently = {}
        if not isinstance(concept, dict):
            concept = {}

        if comparison_available:
            overall = bool(
                evidently.get("dataset_drift")
                or concept.get("detected")
                or any(feature.drift_detected for feature in features)
            )
            drift_type = "comparison"
        elif self_check:
            overall = self_check.get("status") == "WARN"
            drift_type = "self_check"
        else:
            overall = any(feature.drift_detected for feature in features)
            drift_type = "psi"

        drifted_cols = [
            feature.feature for feature in features if feature.drift_detected
        ]
        ev_cols = evidently.get("drifted_columns") or []
        if isinstance(ev_cols, list):
            for item in ev_cols:
                if isinstance(item, dict):
                    name = item.get("column") or item.get("feature")
                    if name:
                        drifted_cols.append(str(name))
                elif isinstance(item, str):
                    drifted_cols.append(item)
        if not drifted_cols and not comparison_available:
            sc_cols = self_check.get("drifted_features") or []
            if isinstance(sc_cols, list):
                for item in sc_cols:
                    if isinstance(item, dict):
                        name = item.get("feature") or item.get("column")
                        if name:
                            drifted_cols.append(str(name))
                    elif isinstance(item, str):
                        drifted_cols.append(item)
        drifted_cols = list(dict.fromkeys(drifted_cols))

        s13_evidence = {
            "contract_type": "S13DriftEvidence",
            "source_stage": "s13_drift_monitor",
            "ownership": "evidence_only",
            "identity": report.get("identity") or {},
            "self_check": {
                **self_check,
                "feature_psi_scores": self_check_psi_scores,
            },
            "comparison": {
                "available": comparison_available,
                "baseline_status": baseline_status,
                "feature_psi_scores": comparison_psi_scores,
                "evidently": evidently,
                "concept_drift": concept,
            },
        }
        warnings = [str(item) for item in report.get("warnings") or []]
        for warning in (s14_download_warning, join_warning):
            if warning and warning not in warnings:
                warnings.append(warning)
        if comparison_available and not comparison_psi_scores:
            warnings.append(
                "S13 comparison is available but has no comparison PSI scores; "
                "the feature table uses self-check PSI."
            )

        response = DriftResponse(
            job_name=job_name,
            task_type=report.get("task_type"),
            dataset_name=report.get("dataset_name"),
            overall_drift_detected=overall,
            stability_score=stability.get("stability_score"),
            drift_type=drift_type,
            recommended_cadence=stability.get("recommended_cadence"),
            recommended_days=stability.get("recommended_days"),
            cadence_rationale=stability.get("rationale"),
            comparison_available=comparison_available,
            baseline_status=baseline_status,
            baseline_metadata=baseline_metadata,
            auto_retrain_decision=auto_retrain_decision,
            auto_retrain_trigger=s13_evidence,
            warnings=warnings,
            drifted_columns=drifted_cols,
            features=features,
            evidently_report_path=report.get("evidently_report_path"),
            studio_url=build_studio_url(ml_client, job_name),
        )
        if status in _TERMINAL_STATES:
            _response_cache[cache_key] = response
        return response
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Resubmit (Phase 0d)
# ---------------------------------------------------------------------------


def _original_revision_identity(tags: dict[str, Any]) -> dict[str, str]:
    fields = {
        "execution_id": "execution_id",
        "config_hash": "compiled_config_hash",
        "source_identity": "source_identity",
    }
    identity = {
        field: str(tags.get(tag_name) or "").strip()
        for field, tag_name in fields.items()
    }
    missing = [field for field, value in identity.items() if not value]
    if missing:
        raise ValueError(
            "Original job lacks immutable replay identity tags: "
            + ", ".join(missing)
            + ". It cannot be replayed or linked as a new revision safely."
        )
    return identity


def _original_config_name(original: Any, tags: dict[str, Any]) -> str:
    raw_name = str(tags.get("config_name") or "").strip()
    if not raw_name:
        experiment_name = str(getattr(original, "experiment_name", "") or "")
        if not experiment_name:
            raise ValueError(
                "Original job has no config_name tag or experiment identity"
            )
        experiment_stem = (
            experiment_name[:-3]
            if experiment_name.endswith("_v3")
            else experiment_name
        )
        raw_name = f"config_{experiment_stem}_azureml"
    if raw_name.endswith(".yml"):
        raw_name = raw_name[:-4]
    if not _SAFE_CONFIG_NAME.fullmatch(raw_name):
        raise ValueError(f"Original job has an invalid config identity: {raw_name!r}")
    return raw_name


def resubmit_pipeline(req: ResubmitRequest) -> SubmitResponse:
    """Replay an exact revision or explicitly branch current inputs as a new one."""
    ml_client = get_ml_client()
    original = ml_client.jobs.get(req.job_name)
    tags = dict(original.tags) if original.tags else {}
    identity = _original_revision_identity(tags)
    config_name = _original_config_name(original, tags)

    revision_reason = str(req.revision_reason or "").strip()
    if req.revision_mode == "new_revision":
        if not revision_reason:
            raise ValueError("new_revision requires a non-empty revision_reason")
    elif revision_reason:
        raise ValueError("revision_reason is valid only for new_revision")

    replay_context = ReplaySubmissionContext(
        revision_kind=req.revision_mode,
        parent_execution_id=identity["execution_id"],
        parent_config_hash=identity["config_hash"],
        parent_source_identity=identity["source_identity"],
        revision_reason=revision_reason or None,
    )

    submit_req = SubmitRequest(
        config_name=config_name,
        force_rerun=req.force_rerun,
        tags={"resubmit_from": req.job_name},
    )
    return submit_pipeline(submit_req, replay_context=replay_context)


# ---------------------------------------------------------------------------
# Baseline capture (Phase 0c)
# ---------------------------------------------------------------------------


def capture_baseline(req: BaselineCaptureRequest) -> BaselineCaptureResponse:
    """Extract drift baseline artifacts from a completed pipeline job."""
    ml_client = get_ml_client()
    j = ml_client.jobs.get(req.job_name)

    baseline_path: str | None = None
    output_present = False
    outputs = j.outputs or {}
    if "drift_baseline" in outputs:
        output_present = True
        baseline_path = getattr(outputs["drift_baseline"], "path", None)

    if baseline_path:
        status = "captured"
    elif output_present:
        status = "baseline_output_path_unavailable"
    else:
        status = "no_baseline_output"

    return BaselineCaptureResponse(
        job_name=req.job_name,
        baseline_path=baseline_path,
        output_present=output_present,
        status=status,
        studio_url=build_studio_url(ml_client, req.job_name),
    )
