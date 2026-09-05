import argparse
import atexit
import json
import logging
import os
import re
import signal
import hashlib
import subprocess
import time
import traceback
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import PipelineJob, Environment
from azure.ai.ml._restclient.runhistory.models import QueryParams
from azure.core.exceptions import (
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
import yaml

# Direct CLI execution puts ``pipelines/`` on sys.path, not the repository's
# import roots. Bootstrap both before importing mixed ``src.*`` and historical
# top-level ``orchestration``/``utils`` modules.
_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
_BOOTSTRAP_SRC_ROOT = _BOOTSTRAP_REPO_ROOT / "src"
for _import_root in (_BOOTSTRAP_REPO_ROOT, _BOOTSTRAP_SRC_ROOT):
    _import_root_text = str(_import_root)
    if _import_root_text not in sys.path:
        sys.path.insert(0, _import_root_text)

from utils.azure_helper import get_ml_client, resolve_credential_mode

# Module logger — used for non-fatal warnings instead of bare except: pass
logger = logging.getLogger("submit_pipeline")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# K2: schema validation gate — HARD FAIL if the validator cannot be imported.
# The K2 schema check is a security gate (catches missing target_column,
# unknown task_type, etc.) and MUST run before any Azure work.
try:
    from src.orchestration.config_schema import validate_config as _validate_config  # type: ignore
except Exception as _e:  # pragma: no cover - validator must be present in repo
    print(f"❌ K2: config validator unavailable ({_e}). Refusing to submit without schema gate.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Duplicate-submission prevention helpers
# ---------------------------------------------------------------------------
# Operator state (lock file, audit log, last-job pointer) lives under
# $MLOPS_STATE_DIR (default ~/.mlops). NEVER inside the repo: ``git clean -fdx``
# would otherwise wipe audit history and the lock file would land in commits.
_USER_STATE_DIR = Path(os.environ.get("MLOPS_STATE_DIR", Path.home() / ".mlops"))
_LOCK_DIR = _USER_STATE_DIR / "locks"
_LOCK_DIR.mkdir(parents=True, exist_ok=True)
_LOCK_FILE = _LOCK_DIR / ".submit.lock"
_LAST_JOB_FILE = _USER_STATE_DIR / "last_submitted_job.json"

_LOCK_MAX_AGE_SEC = 4 * 60 * 60                      # 4 hours hard ceiling — protects
                                                     # against PID-recycling false hits.

# Audit trail for --force submissions (security-relevant; keep alongside lock file)
_FORCE_AUDIT_FILE = _LOCK_DIR / ".force_submit_audit.jsonl"
_CANONICAL_TAG_KEYS = {
    "compiled_config_hash",
    "config_name",
    "dataset",
    "environment",
    "execution_id",
    "parent_config_hash",
    "parent_execution_id",
    "parent_source_identity",
    "pipeline_version",
    "task",
    "preset",
    "recipe_catalog_hash",
    "revision_reason",
    "source_decision_id",
    "source_identity",
    "submission_revision_kind",
}
_SUBMISSION_REVISION_KINDS = {
    "original",
    "exact_replay",
    "decision_retrain",
    "new_revision",
}


def _resolve_drift_baseline_input(ml_client: MLClient, job: PipelineJob) -> None:
    """Bind a completed job's baseline to storage, preserving its lineage URI."""
    baseline = job.inputs.get("drift_baseline_in")
    # PipelineInput.type describes its declaration, which may be untyped.
    # result() returns the supplied Input (or None for an omitted optional input).
    result = getattr(baseline, "result", None)
    value = result() if callable(result) else baseline
    path = getattr(value, "path", None)
    if not isinstance(path, str) or not path.startswith("azureml://jobs/"):
        return
    match = re.fullmatch(
        r"azureml://jobs/([A-Za-z0-9][A-Za-z0-9_.-]*)/outputs/drift_baseline(?:/paths/?)?",
        path,
    )
    if match is None:
        raise ValueError("Baseline job URI must identify the complete drift_baseline output")
    if not isinstance(value, Input) or value.type != "uri_folder":
        raise ValueError("Baseline input must have type uri_folder")
    job_name = match.group(1)
    producer = ml_client.jobs.get(job_name)
    if producer.name != job_name or str(producer.status).lower() != "completed":
        raise ValueError("Baseline producer must be the requested completed job")
    outputs = getattr(producer, "outputs", None)
    output = outputs.get("drift_baseline") if isinstance(outputs, Mapping) else None
    output_type = output.get("type") if isinstance(output, Mapping) else getattr(output, "type", None)
    if output_type != "uri_folder":
        raise ValueError("Baseline producer must declare a uri_folder drift_baseline output")
    resolved = output.get("path") if isinstance(output, Mapping) else getattr(output, "path", None)
    if not resolved:
        # Pinned azure-ai-ml 1.34.1 uses this same resolver in jobs.download.
        # ARM can omit pipeline output paths; never guess a storage location.
        resolver = getattr(ml_client.jobs, "_get_named_output_uri", None)
        if not callable(resolver):
            raise RuntimeError("Azure ML SDK cannot resolve the baseline named output")
        locations = resolver(job_name, output_names="drift_baseline")
        resolved = locations.get("drift_baseline") if isinstance(locations, Mapping) else None
    datastore_uri = (
        r"azureml://(?:subscriptions/[^/?#]+/resourcegroups/[^/?#]+/"
        r"workspaces/[^/?#]+/)?datastores/[^/?#]+/paths/[^?#]+"
    )
    if not isinstance(resolved, str) or re.fullmatch(datastore_uri, resolved) is None:
        raise ValueError("Baseline output did not resolve to an Azure ML datastore URI")
    baseline.path = resolved


def _pid_is_alive(pid: int) -> bool:
    """Check process liveness without sending a destructive Windows signal."""
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        synchronize = 0x00100000
        wait_timeout = 0x00000102
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        handle = kernel32.OpenProcess(
            process_query_limited_information | synchronize,
            False,
            int(pid),
        )
        if not handle:
            error = ctypes.get_last_error()
            return error == 5  # Access denied means the process exists.
        try:
            return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError as exc:
        logger.warning(
            "Process liveness probe failed (%s); treating lock as live",
            exc,
        )
        return True


def _acquire_lock() -> bool:
    """Try to create a lock file.  Return True if acquired.

    Hardening notes:
    * ``os.kill(pid, 0)`` returning ``PermissionError`` (EPERM) means the PID
      exists but is owned by a different user — the lock is GENUINE, never stale.
    * Only ``ProcessLookupError`` (ESRCH) means the process is truly gone.
    * A hard ``_LOCK_MAX_AGE_SEC`` ceiling protects against PID recycling on
      long-lived shared machines.
    """
    if _LOCK_FILE.exists():
        try:
            lock_data = json.loads(_LOCK_FILE.read_text())
            lock_pid = lock_data.get("pid")
            lock_ts  = lock_data.get("ts", 0)
            lock_expires = lock_data.get("expires", 0)
            now_ts = datetime.now().timestamp()
            age = now_ts - lock_ts

            # TTL: if the lock is past its declared expiry, treat as stale
            past_ttl = lock_expires and now_ts > lock_expires
            past_age = age >= _LOCK_MAX_AGE_SEC

            if lock_pid and not past_ttl and not past_age:
                if _pid_is_alive(int(lock_pid)):
                    return False
            # Lock is stale (expired by TTL/age, or process gone) – remove it
            _LOCK_FILE.unlink(missing_ok=True)
        except (json.JSONDecodeError, OSError) as _exc:
            logger.warning("Could not parse existing lock file (%s); reclaiming", _exc)
            _LOCK_FILE.unlink(missing_ok=True)

    now = datetime.now()
    lock_payload = json.dumps({
        "pid": os.getpid(),
        "ts": now.timestamp(),
        "started": now.isoformat(),
        "expires": now.timestamp() + _LOCK_MAX_AGE_SEC,
        "user": os.getenv("USER", "unknown"),
    })
    try:
        fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as lock_file:
        lock_file.write(lock_payload)
        lock_file.flush()
        os.fsync(lock_file.fileno())
    return True


def _release_lock():
    """Remove the lock file (safe to call multiple times)."""
    _LOCK_FILE.unlink(missing_ok=True)


def _handle_submit_signal(signum, _frame):
    """Release the submit lock before process interruption exits."""
    _release_lock()
    raise SystemExit(128 + signum)


def _check_active_jobs(ml_client: MLClient, experiment_name: str) -> list:
    """Return active jobs, failing closed if the control plane cannot be queried."""
    terminal_statuses = {"completed", "failed", "canceled", "cancelled"}
    active_status_filter = (
        "Status ne 'Completed' and Status ne 'Failed' and Status ne 'Canceled'"
    )
    connection_timeout_seconds = 10
    read_timeout_seconds = 30
    page_size = 100
    retryable_errors = (
        ConnectionError,
        TimeoutError,
        ServiceRequestError,
        ServiceResponseError,
    )

    def _field(value, name: str):
        if isinstance(value, dict):
            return value.get(name)
        return getattr(value, name, None)

    def _active_job_payload(job, *, experiment_scoped: bool):
        properties = _field(job, "properties")
        job_experiment = _field(properties, "experiment_name") or _field(
            job,
            "experiment_name",
        )
        if not experiment_scoped and job_experiment != experiment_name:
            return None

        job_status = _field(properties, "status") or _field(job, "status")
        normalized_status = str(job_status or "Unknown")
        if normalized_status.casefold() in terminal_statuses:
            return None

        return {
            "name": _field(job, "name") or _field(job, "run_id") or "unknown",
            "status": normalized_status,
            "display_name": _field(properties, "display_name")
            or _field(job, "display_name")
            or "",
        }

    def _list_experiment_runs():
        """Query only non-terminal runs for this experiment through Run History."""
        job_operations = ml_client.jobs
        run_history = getattr(job_operations, "_runs_operations", None)
        runs_operation = getattr(run_history, "_operation", None)
        operation_scope = getattr(job_operations, "_operation_scope", None)
        subscription_id = getattr(job_operations, "_subscription_id", None)
        workspace_name = getattr(job_operations, "_workspace_name", None)
        resource_group_name = getattr(
            operation_scope,
            "resource_group_name",
            None,
        )
        if not all(
            (
                runs_operation,
                subscription_id,
                resource_group_name,
                workspace_name,
            )
        ):
            return None

        active = []
        continuation_token = None
        seen_tokens = set()
        while True:
            try:
                response = runs_operation.get_by_query_by_experiment_name(
                    subscription_id,
                    resource_group_name,
                    workspace_name,
                    experiment_name,
                    body=QueryParams(
                        filter=active_status_filter,
                        continuation_token=continuation_token,
                        top=page_size,
                    ),
                    connection_timeout=connection_timeout_seconds,
                    read_timeout=read_timeout_seconds,
                )
            except ResourceNotFoundError as exc:
                message = str(exc).casefold()
                missing_experiment = (
                    f"experiment {experiment_name}".casefold() in message
                    and "not found" in message
                    and f"workspace {workspace_name}".casefold() in message
                )
                if missing_experiment:
                    return []
                raise
            resources = getattr(response, "value", None)
            if resources is None:
                raise RuntimeError("Run History returned no run collection")
            for job in resources:
                payload = _active_job_payload(job, experiment_scoped=True)
                if payload is not None:
                    active.append(payload)

            next_token = getattr(response, "continuation_token", None)
            if not next_token:
                return active
            if next_token in seen_tokens:
                raise RuntimeError(
                    "Run History repeated a continuation token while checking "
                    f"experiment {experiment_name!r}"
                )
            seen_tokens.add(next_token)
            continuation_token = next_token

    def _list_job_resources():
        job_operations = ml_client.jobs
        rest_client = getattr(
            job_operations,
            "service_client_01_2024_preview",
            None,
        )
        operation_scope = getattr(job_operations, "_operation_scope", None)
        workspace_name = getattr(job_operations, "_workspace_name", None)
        if rest_client and operation_scope and workspace_name:
            try:
                return rest_client.jobs.list(
                    operation_scope.resource_group_name,
                    workspace_name,
                    connection_timeout=connection_timeout_seconds,
                    read_timeout=read_timeout_seconds,
                )
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
                return rest_client.jobs.list(
                    operation_scope.resource_group_name,
                    workspace_name,
                )
        return job_operations.list()

    last_error = None
    for attempt in range(1, 4):
        try:
            experiment_runs = _list_experiment_runs()
            if experiment_runs is not None:
                return experiment_runs

            active = []
            for job in _list_job_resources():
                payload = _active_job_payload(job, experiment_scoped=False)
                if payload is not None:
                    active.append(payload)
            return active
        except Exception as exc:
            last_error = exc
            if not isinstance(exc, retryable_errors):
                break
            if attempt < 3:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Could not query active jobs for experiment {experiment_name!r}; "
        "refusing submission because duplicate state is unknown."
    ) from last_error


def _write_submission_result(result_path: str | None, payload: dict) -> None:
    """Atomically write the machine-readable result requested by API callers."""
    if not result_path:
        return
    destination = Path(result_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)


# ---------------------------------------------------------------------------
# Repository root + dataset traversal guard
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = (REPO_ROOT / "data").resolve()

# Bounded reads — Phase-1 profiling never needs more than statistics, and the
# submit host is shared / NFS-mounted so we MUST cap memory and IO.
MAX_LOCAL_CSV_BYTES = 500 * 1024 * 1024              # 500 MB hard cap
PROFILE_NROWS = 50_000                               # rows used for profiling only

# Variant safety caps — Azure ML pipeline parameter has a ~2 KB string limit
MAX_CATALOG_RECIPES_PER_RUN = 1000
MAX_VARIANT_LIST_CHARS = 1800


def _safe_join_data_path(blob_path: str) -> Path:
    """Resolve ``blob_path`` under ``DATA_ROOT`` and refuse traversal.

    Refuses absolute paths, paths containing ``..`` segments, and any resolved
    location that escapes ``DATA_ROOT``. Raises ``ValueError`` on any violation.
    """
    if not blob_path or not isinstance(blob_path, str):
        raise ValueError("blob_path must be a non-empty string")
    candidate = Path(blob_path)
    if (
        candidate.is_absolute()
        or PurePosixPath(blob_path).is_absolute()
        or PureWindowsPath(blob_path).is_absolute()
    ):
        raise ValueError(f"blob_path must be relative, got absolute path: {blob_path!r}")
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"blob_path traversal blocked (contains '..'): {blob_path!r}")
    resolved = (DATA_ROOT / candidate).resolve()
    if not str(resolved).startswith(str(DATA_ROOT) + os.sep) and resolved != DATA_ROOT:
        raise ValueError(f"blob_path traversal blocked (escapes DATA_ROOT): {blob_path!r}")
    return resolved


def _check_csv_size_within_cap(local_path: Path, max_bytes: int = MAX_LOCAL_CSV_BYTES) -> None:
    """Refuse to read CSVs larger than ``max_bytes`` on the submit host."""
    size = local_path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"Local dataset {local_path.name} is {size / 1024 / 1024:.1f} MB, "
            f"exceeds {max_bytes / 1024 / 1024:.0f} MB cap. "
            "Profiling/gating must run as a remote step instead."
        )
    logger.info("Local dataset %s = %.1f MB (within %.0f MB cap)",
                local_path.name, size / 1024 / 1024, max_bytes / 1024 / 1024)


def _record_force_audit(args, user: str) -> str:
    """Durably append the audit reservation required before ``--force``."""
    reason = str(getattr(args, "force_reason", "") or "").strip()
    if not reason:
        raise ValueError("--force requires a non-empty --force_reason")
    audit_id = str(uuid.uuid4())
    record = {
        "audit_id": audit_id,
        "timestamp": datetime.now().isoformat(),
        "user": user,
        "pid": os.getpid(),
        "reason": reason,
        "config": getattr(args, "config", None),
        "experiment_name": getattr(args, "experiment_name", None),
        "display_name": getattr(args, "display_name", None),
        "compute": getattr(args, "compute", None),
    }
    with open(_FORCE_AUDIT_FILE, "a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(record, sort_keys=True) + "\n")
        audit_file.flush()
        os.fsync(audit_file.fileno())
    return audit_id


# Recipe selector — sys.path was already prepared at module top for the K2 import.
from src.orchestration.contracts import (
    CandidateRecord,
    ExecutionManifest,
    canonical_hash,
    dataset_version_identity,
)
from src.orchestration.config_compiler import compile_config
from src.utils.recipe_catalog import (
    RecipeCatalog,
    RecipeCatalogEntry,
    catalog_evidence,
    compile_recipe_catalog,
    select_catalog_entries,
)

# Import variant selection components (Phase 1)
try:
    from src.utils.dataset_profiler import DatasetProfiler
    from src.utils.variant_recommender import VariantRecommender
    from src.utils.variant_selector import select_variants, load_and_validate_variants
    from src.utils.variant_schema import load_variant
    PHASE1_AVAILABLE = True
except ImportError:
    PHASE1_AVAILABLE = False
    print("⚠️ Phase 1 components not found, falling back to legacy architecture")

# Import bundle gating (AIM-Tournament)
try:
    from src.utils.bundle_gating import (
        compute_data_signals, load_bundle_catalog,
        select_enabled_bundles, resolve_variant_paths,
        write_gating_artifacts,
    )
    BUNDLES_AVAILABLE = True
except ImportError:
    BUNDLES_AVAILABLE = False

# Pipeline builder import — component YAMLs are loaded once at import time.
# To pick up component-YAML edits, restart the process (do NOT importlib.reload
# inside a long-lived submitter — it has historically masked stale-component bugs).
from pipelines.pipeline_builder import (
    _COMPONENT_MANIFEST,
    full_pipeline,
    full_pipeline_v2,
)


def _component_environment_identities() -> dict[str, str]:
    """Read the exact pinned environment identity for every active component."""

    identities: dict[str, str] = {}
    for component_name, source in sorted(_COMPONENT_MANIFEST.items()):
        payload = yaml.safe_load(Path(source).read_text(encoding="utf-8")) or {}
        environment = payload.get("environment")
        if not isinstance(environment, str) or not environment.strip():
            raise RuntimeError(
                f"Active component {component_name!r} has no immutable environment"
            )
        identities[component_name] = environment.strip()
    if "variant_runner" not in identities:
        raise RuntimeError("Active component manifest is missing variant_runner")
    return identities


def _normalize_azureml_environment(value: str) -> str:
    normalized = str(value or "").strip()
    return normalized[len("azureml:") :] if normalized.startswith("azureml:") else normalized


def _compiled_round1_cap(phase_b_config: dict) -> int:
    """Return the single effective recipe/runner cap from compiled policy."""

    return min(
        int(phase_b_config["max_variants"]),
        int(phase_b_config["planner"]["round1_max_variants"]),
    )


def _configure_pipeline_job_settings(
    job,
    *,
    default_compute: str,
    default_datastore: str,
    force_rerun: bool,
) -> None:
    """Bind shared pipeline settings before the job is submitted."""

    if not str(default_datastore or "").strip():
        raise ValueError("default_datastore must be non-empty")
    job.settings.default_compute = default_compute
    job.settings.default_datastore = default_datastore
    if force_rerun:
        job.settings.force_rerun = True


def _production_data_identity_verified(cfg: dict) -> bool:
    """Return whether a production config binds an expected content digest."""

    if str(cfg.get("preset") or "production") != "production":
        return True
    digest = str((cfg.get("dataset") or {}).get("content_sha256") or "")
    return len(digest) == 64


def _compute_upload_source_manifest(
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Hash exactly the files Azure ML's ``.amlignore`` upload will include."""

    from azure.ai.ml._utils._asset_utils import get_ignore_file

    root = Path(repo_root).resolve()
    ignore = get_ignore_file(root)
    files: list[dict[str, object]] = []
    total_bytes = 0
    for current_root, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_root)
        for name in sorted(directory_names):
            path = current / name
            if path.is_symlink() and not ignore.is_file_excluded(str(path)):
                raise RuntimeError(
                    "Upload-eligible directory symlinks are unsupported: "
                    f"{path.relative_to(root).as_posix()}"
                )
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not ignore.is_file_excluded(str(current / name))
            and not (current / name).is_symlink()
        )
        for name in sorted(file_names):
            path = current / name
            if ignore.is_file_excluded(str(path)):
                continue
            if path.is_symlink():
                raise RuntimeError(
                    "Upload-eligible file symlinks are unsupported: "
                    f"{path.relative_to(root).as_posix()}"
                )
            relative = path.relative_to(root).as_posix()
            payload = path.read_bytes()
            size = len(payload)
            total_bytes += size
            files.append(
                {
                    "path": relative,
                    "size": size,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    files.sort(key=lambda item: str(item["path"]))
    source_sha = canonical_hash(
        {
            "schema_version": "1.0",
            "files": files,
        }
    )
    override = os.environ.get("MLOPS_SOURCE_SHA")
    if override and override.strip() != source_sha:
        raise RuntimeError(
            "MLOPS_SOURCE_SHA does not match the .amlignore-filtered upload bytes"
        )
    return {
        "schema_version": "1.0",
        "ignore_file": (
            ".amlignore" if (root / ".amlignore").is_file() else ".gitignore"
        ),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "source_sha256": source_sha,
        "files": files,
    }


def _compute_source_identity(repo_root: Path = REPO_ROOT) -> str:
    """Compatibility wrapper returning the exact upload package digest."""

    return str(_compute_upload_source_manifest(repo_root)["source_sha256"])


def _validate_revision_cli_contract(args: argparse.Namespace) -> None:
    """Validate explicit replay/new-revision arguments before resolving inputs."""

    kind = str(args.submission_revision_kind or "original")
    if kind not in _SUBMISSION_REVISION_KINDS:
        raise ValueError(f"Unsupported submission revision kind: {kind!r}")

    parent = {
        "parent_execution_id": args.parent_execution_id,
        "parent_config_hash": args.parent_config_hash,
        "parent_source_identity": args.parent_source_identity,
    }
    expected = {
        "expected_execution_id": args.expected_execution_id,
        "expected_config_hash": args.expected_config_hash,
        "expected_source_identity": args.expected_source_identity,
    }
    replay_values = {
        **parent,
        **expected,
        "source_decision_id": args.source_decision_id,
        "revision_reason": args.revision_reason,
    }
    if kind == "original":
        unexpected = sorted(name for name, value in replay_values.items() if value)
        if unexpected:
            raise ValueError(
                "Original submissions may not carry replay metadata: "
                + ", ".join(unexpected)
            )
        return

    missing_parent = sorted(name for name, value in parent.items() if not value)
    if missing_parent:
        raise ValueError(
            f"{kind} requires immutable parent identity: "
            + ", ".join(missing_parent)
        )

    if kind in {"exact_replay", "decision_retrain"}:
        missing_expected = sorted(
            name for name, value in expected.items() if not value
        )
        if missing_expected:
            raise ValueError(
                f"{kind} requires exact identity expectations: "
                + ", ".join(missing_expected)
            )
        if args.revision_reason:
            raise ValueError(f"revision_reason is not valid for {kind}")
        if kind == "decision_retrain" and not args.source_decision_id:
            raise ValueError("decision_retrain requires source_decision_id")
        if kind == "exact_replay" and args.source_decision_id:
            raise ValueError("source_decision_id is valid only for decision_retrain")
        return

    unexpected_expected = sorted(
        name for name, value in expected.items() if value
    )
    if unexpected_expected:
        raise ValueError(
            "new_revision must not claim exact identity expectations: "
            + ", ".join(unexpected_expected)
        )
    if args.source_decision_id:
        raise ValueError(
            "An existing retrain decision cannot authorize changed source/config; "
            "produce a new S14 decision from the new revision"
        )
    reason = str(args.revision_reason or "").strip()
    if not reason:
        raise ValueError("new_revision requires a non-empty revision_reason")
    if len(reason) > 256:
        raise ValueError("revision_reason must be at most 256 characters")


def _validate_submission_revision_identity(
    *,
    revision_kind: str,
    execution_manifest: ExecutionManifest,
    config_hash: str,
    source_identity: str,
    parent_execution_id: str | None,
    parent_config_hash: str | None,
    parent_source_identity: str | None,
    expected_execution_id: str | None,
    expected_config_hash: str | None,
    expected_source_identity: str | None,
) -> None:
    """Fail closed when replay inputs do not represent the claimed revision."""

    if revision_kind == "original":
        return

    current = {
        "execution_id": execution_manifest.execution_id,
        "config_hash": config_hash,
        "source_identity": source_identity,
    }
    parent = {
        "execution_id": str(parent_execution_id or ""),
        "config_hash": str(parent_config_hash or ""),
        "source_identity": str(parent_source_identity or ""),
    }
    if revision_kind in {"exact_replay", "decision_retrain"}:
        expected = {
            "execution_id": str(expected_execution_id or ""),
            "config_hash": str(expected_config_hash or ""),
            "source_identity": str(expected_source_identity or ""),
        }
        mismatches = sorted(
            field for field, value in current.items() if value != expected[field]
        )
        parent_mismatches = sorted(
            field for field, value in parent.items() if value != expected[field]
        )
        if parent_mismatches:
            raise ValueError(
                "Replay parent identity does not match its expected revision: "
                + ", ".join(parent_mismatches)
            )
        if mismatches:
            next_action = (
                "produce a fresh S14 decision from the current revision"
                if revision_kind == "decision_retrain"
                else "resubmit explicitly with revision_mode='new_revision' and a reason"
            )
            raise ValueError(
                f"{revision_kind} rejected because current immutable inputs changed: "
                + ", ".join(mismatches)
                + f"; {next_action}"
            )
        return

    changed_fields = sorted(
        field for field, value in current.items() if value != parent[field]
    )
    if not changed_fields:
        raise ValueError(
            "new_revision rejected because config, source, and execution identity are "
            "unchanged; use exact_replay"
        )


def _build_execution_manifest(
    cfg: dict,
    selected_entries: tuple[RecipeCatalogEntry, ...],
    catalog: RecipeCatalog,
    *,
    code_sha: str,
    environment: str,
    component_environments: dict[str, str] | None = None,
    round1_max_variants: int | None = None,
    round2_max_variants: int | None = None,
    proxy_prune_threshold: float | None = None,
) -> tuple[ExecutionManifest, tuple[CandidateRecord, ...]]:
    """Bind compiled config, recipes, engines, budgets, code, and environment."""

    task_type = cfg["task_type"]
    phase_b = cfg["phases"]["phase_b"]
    engines = tuple(phase_b["engines"])
    split_id = canonical_hash(cfg["split"])
    data_version = dataset_version_identity(cfg["dataset"])
    resolved_environments = dict(component_environments or {})
    training_environment = resolved_environments.get("variant_runner", environment)
    if _normalize_azureml_environment(training_environment) != (
        _normalize_azureml_environment(environment)
    ):
        raise ValueError(
            "Configured training environment does not match the pinned "
            f"variant-runner environment: {environment!r} != {training_environment!r}"
        )
    environment_hash = canonical_hash({"environment": training_environment})
    environment_hashes = {
        "training": environment_hash,
        **{
            f"component:{name}": canonical_hash({"environment": identity})
            for name, identity in sorted(resolved_environments.items())
        },
    }
    records = tuple(
        CandidateRecord(
            task_type=task_type,
            recipe_id=entry.recipe_id,
            recipe_hash=entry.semantic_hash,
            engine=engine,
            algorithm="engine_search",
            parameters=entry.normalized_recipe,
            split_id=split_id,
            data_version=data_version,
            code_sha=code_sha,
            environment_hash=environment_hash,
        )
        for entry in selected_entries
        for engine in engines
    )
    planner = phase_b["planner"]
    manifest = ExecutionManifest(
        config_hash=cfg["compiled_config_hash"],
        task_type=task_type,
        dataset=cfg["dataset"],
        split_policy=cfg["split"],
        engines=engines,
        recipe_paths=tuple(entry.path for entry in selected_entries),
        recipe_ids=tuple(entry.recipe_id for entry in selected_entries),
        candidate_ids=tuple(record.candidate_id for record in records),
        budgets={
            "round1_max_variants": (
                round1_max_variants
                if round1_max_variants is not None
                else planner["round1_max_variants"]
            ),
            "round2_max_variants": (
                round2_max_variants
                if round2_max_variants is not None
                else planner["round2_max_variants"]
            ),
            "proxy_prune_threshold": (
                proxy_prune_threshold
                if proxy_prune_threshold is not None
                else planner["proxy_prune_threshold"]
            ),
            "candidate_engine_timeout_seconds": phase_b[
                "time_budget_per_variant"
            ],
            "phase_b_timeout_seconds": phase_b["phase_timeout_seconds"],
            "hpo_trials": cfg["phases"]["phase_c_hpo"]["n_trials"],
            "hpo_timeout_seconds": cfg["phases"]["phase_c_hpo"][
                "timeout_seconds"
            ],
        },
        code_sha=code_sha,
        environment_hashes=environment_hashes,
        recipe_catalog_hash=catalog.catalog_hash,
    )
    return manifest, records


def _persist_execution_artifacts(
    manifest: ExecutionManifest,
    candidates: tuple[CandidateRecord, ...],
    catalog: RecipeCatalog,
    selected_entries: tuple[RecipeCatalogEntry, ...],
    source_manifest: dict[str, object],
) -> tuple[Path, Path, Path]:
    """Atomically persist immutable submission inputs outside the repository."""

    manifest_dir = _USER_STATE_DIR / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{manifest.execution_id}.json"
    catalog_path = manifest_dir / f"{manifest.execution_id}.recipe_catalog.json"
    source_path = manifest_dir / f"{manifest.execution_id}.source_manifest.json"
    manifest_payload = {
        **manifest.to_dict(),
        "candidate_records": [record.to_dict() for record in candidates],
    }
    candidate_catalog_payload = {
        **catalog_evidence(catalog, selected_entries),
        "execution_id": manifest.execution_id,
        "recipe_catalog_hash": manifest.recipe_catalog_hash,
        "recipe_paths": list(manifest.recipe_paths),
        "recipe_ids": list(manifest.recipe_ids),
        "candidate_ids": list(manifest.candidate_ids),
        "candidate_records": [record.to_dict() for record in candidates],
    }
    for destination, payload in (
        (manifest_path, manifest_payload),
        (catalog_path, candidate_catalog_payload),
        (source_path, source_manifest),
    ):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(destination)
    return manifest_path, catalog_path, source_path


def _sdk_local_input_path(path: Path) -> str:
    """Return an absolute filesystem path that Azure ML SDK uploads as input."""

    resolved = path.expanduser().resolve(strict=True)
    return str(resolved)


def _sdk_local_file_input(path: Path, *, datastore: str) -> Input:
    """Route a local control artifact to the configured working datastore."""

    if not str(datastore or "").strip():
        raise ValueError("datastore must be non-empty for local file inputs")
    return Input(
        path=_sdk_local_input_path(path),
        type="uri_file",
        datastore=datastore,
    )


def _azure_from_local_config(cfg):
    """Extract azureml connection defaults from an already-loaded config dict."""
    if not cfg or not isinstance(cfg, dict):
        return None, None, None
    azure_cfg = cfg.get("azureml") or cfg.get("azure_ml") or {}
    return (
        azure_cfg.get("subscription_id"),
        azure_cfg.get("resource_group"),
        azure_cfg.get("workspace_name"),
    )


def derive_experiment_name(config_path: str) -> str:
    """Derive generic reusable experiment name from config filename.
    
    Example: config_classification_telecom_churn_azureml.yml → classification_telecom_churn_v3
    """
    config_stem = Path(config_path).stem
    normalized = config_stem.replace("config_", "").replace("_azureml", "").replace("_local", "")
    return f"{normalized}_v3"


def derive_display_name(experiment_name: str) -> str:
    """Generate unique display name for this job submission.
    
    Format: {experiment_name}_{timestamp}_{random_id}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{experiment_name}_{timestamp}_{unique_id}"


# ---------------------------------------------------------------------------
# Imputation preset filter — filters selected variants by imputation family
# ---------------------------------------------------------------------------
IMPUTATION_PRESET_MAP = {
    "auto": None,  # No filter — pass all variants through
    "statistical": ["mean", "median", "mode", "trimmed_mean", "winsorized_mean"],
    "ml_based": ["knn", "iterative"],
    "removal": ["drop"],
    "pandas_native": ["forward_fill", "backward_fill", "interpolate_linear", "constant", "zero_fill"],
    "composite": ["numeric_mean_cat_mode", "numeric_median_cat_mode"],
    "sampling": ["random_sample"],
    "advanced": None,  # All Tier 1 methods — same as "auto" until Tier 2+ is added
}


def filter_variants_by_imputation_preset(
    variant_paths: list,
    preset: str,
) -> list:
    """Filter variant paths to only those whose imputation method matches the preset.

    Args:
        variant_paths: List of absolute or relative paths to variant YAML files.
        preset: One of the keys in IMPUTATION_PRESET_MAP.

    Returns:
        Filtered list of paths. If preset is "auto" or "advanced", all paths are kept.
    """
    allowed = IMPUTATION_PRESET_MAP.get(preset)
    if allowed is None:
        # "auto" / "advanced" — no filtering
        return variant_paths

    filtered = []
    for p in variant_paths:
        try:
            variant = load_variant(str(p))
            method = variant.stage3_preprocessing.imputation.method.lower().strip()
            if method in allowed:
                filtered.append(p)
        except Exception as exc:
            # If a variant YAML is malformed, skip it with a warning
            logger.warning("Could not read variant %s for imputation filter: %s", p, exc)
    return filtered


def main():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Submit V3 pipeline with proper experiment/display naming",
        allow_abbrev=False,
    )
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--subscription_id", required=False, help="Azure subscription ID")
    parser.add_argument("--resource_group", required=False, help="Azure resource group")
    parser.add_argument("--workspace_name", required=False, help="Azure ML workspace name")
    parser.add_argument("--compute", required=False, default=None,
                        help="Compute target (default: $AZURE_COMPUTE env var; required if unset)")
    parser.add_argument(
        "--experiment_name",
        required=False,
        help="Reusable experiment name (auto-derived from config if not provided)",
    )
    parser.add_argument(
        "--display_name",
        required=False,
        help="Unique job display name (auto-generated with timestamp if not provided)",
    )
    parser.add_argument("--wait", action="store_true", help="Wait for job to complete")
    parser.add_argument("--stop_compute", action="store_true",
                        help="Stop compute cluster after job completes (requires --wait)")
    parser.add_argument("--use_phase1", action="store_true", help="Use Phase 1 intelligent variant runner (NEW)")
    # V3-Proposed Planner flags
    parser.add_argument("--enable_planner", action="store_true", help="Enable V3-Proposed adaptive planner mode")
    parser.add_argument("--round1_max_variants", type=int, default=None, help="Optional lower Round 1 cap (schema maximum: 40)")
    parser.add_argument("--round2_max_variants", type=int, default=None, help="Optional lower Round 2 cap (schema maximum: 8)")
    parser.add_argument("--proxy_prune_threshold", type=float, default=None, help="Optional proxy threshold override")
    parser.add_argument("--disable_cache", action="store_true", help="Disable preprocessing cache")
    parser.add_argument("--bundles_dir", required=False, default=None,
                        help="Path to variant_bundles/<task> directory for AIM-Tournament bundle gating")
    parser.add_argument("--drift_baseline_in", required=False, default=None,
                        help="Optional previous s13 drift_baseline uri_folder for baseline comparison")
    parser.add_argument(
        "--force_rerun",
        action="store_true",
        help="Disable Azure ML component caching without bypassing submission guards",
    )
    parser.add_argument(
        "--submission_revision_kind",
        choices=sorted(_SUBMISSION_REVISION_KINDS),
        default="original",
        help="Immutable revision semantics for original, replay, or retrain submissions",
    )
    parser.add_argument("--parent_execution_id", default=None)
    parser.add_argument("--parent_config_hash", default=None)
    parser.add_argument("--parent_source_identity", default=None)
    parser.add_argument("--expected_execution_id", default=None)
    parser.add_argument("--expected_config_hash", default=None)
    parser.add_argument("--expected_source_identity", default=None)
    parser.add_argument("--source_decision_id", default=None)
    parser.add_argument("--revision_reason", default=None)
    parser.add_argument(
        "--tags_json",
        default=None,
        help="Optional JSON object of additional string job tags",
    )
    parser.add_argument(
        "--result_json",
        default=None,
        help="Optional path for a structured submission result used by API callers",
    )
    parser.add_argument("--imputation_preset", required=False, default=None,
                        choices=["auto", "statistical", "ml_based", "removal",
                                 "pandas_native", "composite", "sampling", "advanced"],
                        help="Filter variants by imputation family (overrides config value)")
    parser.add_argument("--force", action="store_true",
                        help="Skip duplicate-submission guards (lock file + active-job check). "
                             "AUDITED: appends to ~/.mlops/locks/.force_submit_audit.jsonl")
    parser.add_argument(
        "--force_reason",
        default=None,
        help="Required operator reason when --force bypasses submission guards",
    )
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose tracebacks and debug-only diagnostics (URIs, etc.)")
    parser.add_argument("--env_version", default=None,
                        help="Azure ML environment tag (default: read from environments/azureml_unified_env.yml)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build the pipeline job and print its YAML — do NOT submit to Azure ML")
    args = parser.parse_args()

    if args.imputation_preset is not None:
        parser.error(
            "--imputation_preset is retired; recipe feasibility and diversity are "
            "enforced inside the canonical S06 funnel"
        )
    if args.force and not str(args.force_reason or "").strip():
        parser.error("--force requires a non-empty --force_reason")
    if args.force_reason and not args.force:
        parser.error("--force_reason is valid only together with --force")
    try:
        _validate_revision_cli_contract(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # ----- Load and validate config ONCE up front (H1) ---------------------
    config_path = args.config
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    try:
        with open(config_path, "r") as _f:
            cfg = yaml.safe_load(_f) or {}
    except yaml.YAMLError as _ye:
        logger.error("Config %s is not valid YAML: %s", config_path, _ye)
        raise SystemExit(2) from _ye
    if not isinstance(cfg, dict):
        logger.error("Config %s did not parse to a mapping (got %s)", config_path, type(cfg).__name__)
        raise SystemExit(2)

    # K2 schema gate — fail fast BEFORE any Azure work
    if _validate_config is not None:
        try:
            cfg = compile_config(
                cfg,
                source_name=Path(config_path).name,
            )
            logger.info("K2: config schema validation passed for %s", config_path)
            print(
                "✅ K2: schema-v2 compile passed for "
                f"{config_path} (hash={cfg['compiled_config_hash'][:12]})"
            )
        except Exception as _ve:
            print(f"❌ K2: config schema validation FAILED: {_ve}")
            raise SystemExit(2) from _ve

    if not _production_data_identity_verified(cfg):
        message = (
            "Production submissions require dataset.content_sha256. "
            "Generate the canonical dataframe fingerprint in Azure compute and "
            "add it to the versioned dataset config."
        )
        if args.dry_run:
            print(f"WARNING: {message}")
        else:
            print(f"Refusing submission: {message}", file=sys.stderr)
            raise SystemExit(2)

    extra_tags: dict[str, str] = {}
    if args.tags_json:
        try:
            parsed_tags = json.loads(args.tags_json)
        except json.JSONDecodeError as exc:
            raise SystemExit("--tags_json must contain valid JSON") from exc
        if not isinstance(parsed_tags, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in parsed_tags.items()
        ):
            raise SystemExit("--tags_json must be a JSON object of string values")
        protected = sorted(_CANONICAL_TAG_KEYS.intersection(parsed_tags))
        if protected:
            raise SystemExit(
                "--tags_json cannot override canonical tags: "
                + ", ".join(protected)
            )
        extra_tags = parsed_tags

    # If CLI context missing, fall back to azureml block in cfg
    if not args.subscription_id or not args.resource_group or not args.workspace_name:
        sub, rg, ws = _azure_from_local_config(cfg)
        args.subscription_id = args.subscription_id or sub
        args.resource_group = args.resource_group or rg
        args.workspace_name = args.workspace_name or ws

    # Then env-var fallback (CLI > config > env). NEVER hardcode.
    args.subscription_id = args.subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID")
    args.resource_group  = args.resource_group  or os.environ.get("AZURE_RESOURCE_GROUP")
    args.workspace_name  = args.workspace_name  or os.environ.get("AZURE_WORKSPACE_NAME")
    args.compute         = args.compute         or os.environ.get("AZURE_COMPUTE")

    _missing_ctx = [n for n, v in (
        ("subscription_id (--subscription_id / AZURE_SUBSCRIPTION_ID)", args.subscription_id),
        ("resource_group (--resource_group / AZURE_RESOURCE_GROUP)",   args.resource_group),
        ("workspace_name (--workspace_name / AZURE_WORKSPACE_NAME)",   args.workspace_name),
        ("compute (--compute / AZURE_COMPUTE)",                         args.compute),
    ) if not v]
    if _missing_ctx:
        print("❌ Missing Azure context: " + ", ".join(_missing_ctx), file=sys.stderr)
        print("   See .env.example for the full list of required variables.", file=sys.stderr)
        sys.exit(2)

    # FIXED: Use config filename only (from uploaded code directory)
    # Avoid workspaceblobstore upload by passing filename as string parameter
    config_name = Path(config_path).name
    print(f"✅ Using config filename: {config_name} (from uploaded code/configs directory)")

    # Resolve env_version (CLI > config > default)
    env_version = (
        args.env_version
        or (cfg.get("azureml") or cfg.get("azure_ml") or {}).get("environment")
        or "mlops-v3-unified:33"
    )
    component_environments = _component_environment_identities()
    pinned_training_environment = component_environments["variant_runner"]
    if _normalize_azureml_environment(env_version) != (
        _normalize_azureml_environment(pinned_training_environment)
    ):
        raise SystemExit(
            "Configured/CLI environment does not match component YAMLs: "
            f"{env_version!r} != {pinned_training_environment!r}. "
            "Update and validate the component environment mapping as one revision."
        )

    # Derive experiment name (reusable, generic)
    if not args.experiment_name:
        args.experiment_name = derive_experiment_name(config_path)
    
    # Derive display name (unique per submission, with timestamp)
    if not args.display_name:
        args.display_name = derive_display_name(args.experiment_name)
    
    print("\n" + "="*80)
    print("NAMING CONFIGURATION")
    print("="*80)
    print(f"📊 Experiment name (reusable):  {args.experiment_name}")
    print(f"🎯 Display name (unique):       {args.display_name}")
    print("="*80 + "\n")
    datastore_name = (cfg.get("dataset") or {}).get("datastore_name", "mlops_blob")
    default_datastore = (cfg.get("azureml") or {}).get(
        "default_datastore", datastore_name
    )

    # Dataset folder URI (Azure ML will mount it). The full URI exposes the
    # subscription ID; only print it when --debug is set.
    dataset_folder_uri = (
        f"azureml://subscriptions/{args.subscription_id}"
        f"/resourcegroups/{args.resource_group}"
        f"/workspaces/{args.workspace_name}"
        f"/datastores/{datastore_name}/paths/"
    )
    print(f"Using datastore: {datastore_name}")
    print(f"Using pipeline output datastore: {default_datastore}")
    if args.debug:
        print(f"Dataset folder URI: {dataset_folder_uri}")
    else:
        logger.debug("Dataset folder URI: %s", dataset_folder_uri)

    # Compile the complete task catalog before any Azure client or job is created.
    # Invalid sources remain untouched but are quarantined in the evidence artifact;
    # semantic duplicates cannot consume more than one shortlist slot.
    task_type = cfg["task_type"]
    phase_b_config = cfg["phases"]["phase_b"]
    try:
        recipes_base_dir = REPO_ROOT / "configs" / "recipes"
        recipe_catalog = compile_recipe_catalog(recipes_base_dir, task_type)
        selected_catalog_entries = select_catalog_entries(
            recipe_catalog,
            library=phase_b_config["library"],
            tier=phase_b_config["tier"],
            # Do not discard data-blind recipes here. S06 profiles every
            # eligible semantic recipe before applying the <=40 Round 1 cap.
            max_variants=None,
            runtime_budget_sec=phase_b_config["runtime_budget_sec"],
        )
        all_selected_recipes = [
            entry.path for entry in selected_catalog_entries
        ]
        print(
            "✅ Recipe catalog compiled before Azure: "
            f"checked={recipe_catalog.checked_count}, "
            f"valid={recipe_catalog.valid_count}, "
            f"unique={recipe_catalog.unique_count}, "
            f"quarantined={len(recipe_catalog.quarantined)}, "
            f"selected={len(selected_catalog_entries)}"
        )
        for i, r in enumerate(all_selected_recipes, 1):
            print(f"   [{i}] {r}")
    except Exception as e:
        print(f"❌ Recipe catalog compile/selection failed: {e}")
        raise SystemExit(2) from e

    variants_list_str = ",".join(all_selected_recipes)
    if len(all_selected_recipes) > MAX_CATALOG_RECIPES_PER_RUN:
        raise SystemExit(
            f"Refusing to submit: {len(all_selected_recipes)} variants exceed cap of "
            f"{MAX_CATALOG_RECIPES_PER_RUN} catalog recipes."
        )
    # Canonical schema-v2 transports the complete catalog as a uri_file.  Keep
    # the string only as bounded compatibility for older direct callers.
    if len(variants_list_str) >= MAX_VARIANT_LIST_CHARS:
        variants_list_str = ""

    # Legacy submit-host bundle gating is not a canonical selection path. Data
    # profiling and ranking occur inside S06 against Azure-resolved training data.
    bundle_gated_variants = None  # Will be set if bundle gating succeeds

    if args.bundles_dir:
        raise SystemExit(
            "--bundles_dir is incompatible with the canonical schema-v2 funnel; "
            "compile the task recipe catalog instead"
        )
    if False and args.bundles_dir and BUNDLES_AVAILABLE:
        print("\n" + "="*80)
        print("AIM-TOURNAMENT: BUNDLE GATING")
        print("="*80)
        try:
            import pandas as _bg_pd
            _bg_task = cfg.get("task_type", "classification")
            _bg_target = cfg.get("dataset", {}).get("target_column")
            _bg_blob = cfg.get("dataset", {}).get("blob_path", "")
            try:
                _bg_local = _safe_join_data_path(_bg_blob)
            except ValueError as _path_err:
                logger.error("Bundle gating refused dataset path %r: %s", _bg_blob, _path_err)
                raise

            if _bg_local.exists():
                _check_csv_size_within_cap(_bg_local)
                _bg_df = _bg_pd.read_csv(_bg_local, nrows=PROFILE_NROWS)
                signals = compute_data_signals(_bg_df, _bg_target, _bg_task)
                print(f"\n📡 Data signals computed ({len(signals)} signals)")
                for _sk, _sv in sorted(signals.items()):
                    print(f"   {_sk}: {_sv}")

                catalog = load_bundle_catalog(args.bundles_dir)
                enabled, decisions = select_enabled_bundles(signals, catalog)

                bundle_gated_variants = resolve_variant_paths(enabled, str(REPO_ROOT))
                write_gating_artifacts(signals, decisions, str(REPO_ROOT / "outputs" / "signals"))

                print(f"\n✅ Bundle gating: {len(enabled)}/{len(catalog)} bundles enabled → {len(bundle_gated_variants)} variants")
            else:
                print(f"⚠️ Local dataset not found at {_bg_local} — skipping bundle gating")
        except Exception as _bg_err:
            logger.warning("Bundle gating failed (non-fatal): %s", _bg_err)
            if args.debug:
                import traceback
                traceback.print_exc()

    # ============================================================================
    # PHASE 1: INTELLIGENT VARIANT SELECTION (NEW ARCHITECTURE)
    # ============================================================================
    variants_json_path = None
    # The schema-v2 compiled funnel is the only production graph.  --use_phase1
    # remains accepted as a compatibility no-op for existing callers.
    use_phase1_pipeline = True
    
    if False:  # Legacy submit-host profiling path intentionally retired.
        print("\n" + "="*80)
        print("PHASE 1: INTELLIGENT VARIANT RECOMMENDATION SYSTEM")
        print("="*80)
        
        try:
            phase_b_config = cfg.get("phases", {}).get("phase_b", {})
            if not phase_b_config:
                print("⚠️ No phase_b config found, using defaults")
                phase_b_config = {
                    "enable_profiling": True,
                    "library_dir": f"configs/recipes/{task_type}/variant_search",
                    "max_variants": 20,
                    "selection_strategy": "scored",
                    "min_relevance_score": 30.0,
                    "diversity_boost": True,
                    "runtime_budget_sec": 180,
                    "time_budget_per_variant": 600,
                    "engines": ["pycaret", "flaml"]
                }
            
            # 1. Load and profile dataset
            print("\n📊 STEP 1: Dataset Profiling")
            print("-" * 80)
            
            # Construct local dataset path for profiling
            # In Azure ML job, this would be mounted, but for submission we need local path
            dataset_blob_path = cfg.get("dataset", {}).get("blob_path", "")
            try:
                local_dataset_path = _safe_join_data_path(dataset_blob_path)
            except ValueError as _path_err:
                logger.error("Phase 1 refused dataset path %r: %s", dataset_blob_path, _path_err)
                raise
            
            if not local_dataset_path.exists():
                print(f"⚠️ Local dataset not found at {local_dataset_path}")
                print("⚠️ Falling back to alphabetical variant selection (no profiling)")
                phase_b_config["selection_strategy"] = "alphabetical"
                phase_b_config["enable_profiling"] = False
            
            profile = None
            if phase_b_config.get("enable_profiling", True) and local_dataset_path.exists():
                import pandas as pd
                _check_csv_size_within_cap(local_dataset_path)
                df = pd.read_csv(local_dataset_path, nrows=PROFILE_NROWS)
                target_column = cfg.get("dataset", {}).get("target_column")
                
                profiler = DatasetProfiler(task_type=task_type)
                profile = profiler.profile_dataset(df, target_column)
                
                print(profile.generate_profile_summary())
                recommendations = profile.recommend_preprocessing_strategies()
                
                print("\n🎯 PREPROCESSING RECOMMENDATIONS:")
                for dim, strategies in recommendations.items():
                    if dim != "priority_scores" and dim != "reasoning":
                        print(f"  • {dim}: {', '.join(strategies)}")
                
                print("\n💡 REASONING:")
                for reason in recommendations.get("reasoning", []):
                    print(f"  • {reason}")
            
            # 2. Load all variants and score
            print("\n\n🔍 STEP 2: Variant Scoring and Selection")
            print("-" * 80)
            
            library_dir = Path(__file__).resolve().parents[1] / phase_b_config.get("library_dir", f"configs/recipes/{task_type}/variant_search")
            all_variant_paths = sorted(library_dir.glob("variant_*.yml"))
            
            print(f"Found {len(all_variant_paths)} variants in library: {library_dir.name}")
            
            # Select variants based on strategy
            selection_strategy = phase_b_config.get("selection_strategy", "scored")
            max_variants = phase_b_config.get("max_variants", 20)
            runtime_budget = phase_b_config.get("runtime_budget_sec")
            
            if selection_strategy == "scored" and profile:
                # Intelligent selection using dataset profiling
                all_variants = [load_variant(str(p)) for p in all_variant_paths]
                recommender = VariantRecommender(profile, all_variants)
                selected = recommender.select_top_variants(
                    max_variants=max_variants,
                    min_relevance_score=phase_b_config.get("min_relevance_score", 30.0),
                    diversity_boost=phase_b_config.get("diversity_boost", True)
                )
                
                # Generate selection report
                report = recommender.generate_selection_report(selected)
                print(report)
                
                selected_paths = [str(v.metadata.source_path) for v, _, _ in selected]
            else:
                # Fallback: alphabetical or random selection
                selected_paths = select_variants(
                    task_type=task_type,
                    library_dir=str(library_dir),
                    max_variants=max_variants,
                    selection_strategy=selection_strategy,
                    runtime_budget_sec=runtime_budget
                )
                print(f"✅ Selected {len(selected_paths)} variants using '{selection_strategy}' strategy")
            
            print(f"\n✅ Selected {len(selected_paths)} variants for Phase B pipeline")
            
            # ── Imputation preset filter ──────────────────────────────────
            # CLI flag overrides config; default is "auto" (no filter).
            imputation_preset = (
                args.imputation_preset
                or phase_b_config.get("imputation_preset", "auto")
            )
            if imputation_preset and imputation_preset != "auto":
                pre_filter_count = len(selected_paths)
                selected_paths = filter_variants_by_imputation_preset(
                    selected_paths, imputation_preset
                )
                print(
                    f"🔬 Imputation preset '{imputation_preset}': "
                    f"{pre_filter_count} → {len(selected_paths)} variants "
                    f"(allowed methods: {IMPUTATION_PRESET_MAP.get(imputation_preset)})"
                )
                if not selected_paths:
                    print("⚠️ No variants matched the imputation preset — falling back to 'auto' (all methods)")
                    selected_paths = filter_variants_by_imputation_preset(
                        [str(v.metadata.source_path) for v, _, _ in selected]
                        if selection_strategy == "scored" and profile
                        else select_variants(
                            task_type=task_type,
                            library_dir=str(library_dir),
                            max_variants=max_variants,
                            selection_strategy=selection_strategy,
                            runtime_budget_sec=runtime_budget,
                        ),
                        "auto",
                    )
            else:
                print(f"🔬 Imputation preset: 'auto' (no filter applied)")

            # CRITICAL FIX: Convert absolute paths to relative paths from repo root
            repo_root = Path(__file__).resolve().parents[1]
            relative_paths = []
            for p in selected_paths:
                path_obj = Path(p)
                if path_obj.is_absolute():
                    try:
                        # Convert to path relative to repo root
                        rel_path = path_obj.relative_to(repo_root)
                        relative_paths.append(str(rel_path))
                    except ValueError:
                        # If path is not under repo_root, try to extract just the filename pattern
                        # This handles cases where path structure differs between environments
                        print(f"⚠️ Warning: Could not make path relative: {path_obj}")
                        relative_paths.append(str(path_obj.name))
                else:
                    # Already relative
                    relative_paths.append(str(path_obj))
            
            print(f"📝 Converted to relative paths (sample): {relative_paths[0] if relative_paths else 'N/A'}")
            print("="*80 + "\n")
            
            # Store as comma-separated string
            variants_list_str = ",".join(relative_paths)
            if len(relative_paths) > MAX_CATALOG_RECIPES_PER_RUN:
                raise SystemExit(
                    f"Refusing to submit: Phase 1 selected {len(relative_paths)} variants "
                    f"(cap {MAX_CATALOG_RECIPES_PER_RUN})."
                )
            if len(variants_list_str) >= MAX_VARIANT_LIST_CHARS:
                variants_list_str = ""
            
        except Exception as e:
            logger.error("Phase 1 variant selection failed: %s", e)
            print(f"\n❌ Phase 1 variant selection failed: {e}")
            print("⚠️ Falling back to legacy pipeline\n")
            use_phase1_pipeline = False
            if args.debug:
                import traceback
                traceback.print_exc()

    # Build pipeline job (config filename passed as string, no upload needed)
    if use_phase1_pipeline and 'variants_list_str' in locals():
        # ADVANCED: Use Phase 1 intelligent variant runner with planner mode
        print("🚀 Using Phase 1 intelligent variant runner pipeline (advanced planner)\n")
        
        engine_list = ",".join(phase_b_config.get("engines", ["pycaret", "flaml"]))
        time_budget_per_variant = phase_b_config.get("time_budget_per_variant", 600)
        
        # V3-Proposed Planner settings (from CLI or config)
        planner_config = phase_b_config.get("planner", {})
        planner_enabled = True
        compiled_round1_cap = _compiled_round1_cap(phase_b_config)
        round1_max = (
            args.round1_max_variants
            if args.round1_max_variants is not None
            else compiled_round1_cap
        )
        round2_max = (
            args.round2_max_variants
            if args.round2_max_variants is not None
            else planner_config["round2_max_variants"]
        )
        proxy_threshold = (
            args.proxy_prune_threshold
            if args.proxy_prune_threshold is not None
            else planner_config["proxy_prune_threshold"]
        )
        if not 1 <= round1_max <= compiled_round1_cap:
            raise SystemExit(
                "--round1_max_variants may lower but not exceed the compiled "
                f"effective cap ({compiled_round1_cap})"
            )
        if not 1 <= round2_max <= min(8, round1_max):
            raise SystemExit(
                "--round2_max_variants must be between 1 and min(8, Round 1)"
            )
        cache_enabled = not args.disable_cache and planner_config.get("cache_enabled", True)
        
        if planner_enabled:
            print("="*80)
            print("V3-PROPOSED PLANNER MODE ENABLED")
            print("="*80)
            print(f"  Round 1 max variants: {round1_max}")
            print(f"  Round 2 max variants: {round2_max}")
            print(f"  Proxy prune threshold: {proxy_threshold}")
            print(f"  Preprocessing cache: {'ENABLED' if cache_enabled else 'DISABLED'}")
            print("="*80 + "\n")

        source_manifest = _compute_upload_source_manifest()
        code_identity = str(source_manifest["source_sha256"])
        execution_manifest, candidate_records = _build_execution_manifest(
            cfg,
            selected_catalog_entries,
            recipe_catalog,
            code_sha=code_identity,
            environment=env_version,
            component_environments=component_environments,
            round1_max_variants=round1_max,
            round2_max_variants=round2_max,
            proxy_prune_threshold=proxy_threshold,
        )
        execution_manifest_path, catalog_evidence_path, source_manifest_path = (
            _persist_execution_artifacts(
                execution_manifest,
                candidate_records,
                recipe_catalog,
                selected_catalog_entries,
                source_manifest,
            )
        )
        print(
            "🔒 Execution manifest frozen: "
            f"{execution_manifest.execution_id[:16]} "
            f"(catalog={recipe_catalog.catalog_hash[:16]})"
        )
        logger.info("Execution manifest: %s", execution_manifest_path)
        logger.info("Recipe catalog evidence: %s", catalog_evidence_path)
        logger.info("Upload source manifest: %s", source_manifest_path)
        execution_manifest_input = _sdk_local_file_input(
            execution_manifest_path,
            datastore=default_datastore,
        )
        candidate_catalog_input = _sdk_local_file_input(
            catalog_evidence_path,
            datastore=default_datastore,
        )

        drift_baseline_input = Input(path=args.drift_baseline_in, type="uri_folder") if args.drift_baseline_in else None
        job = full_pipeline_v2(
            config_name=config_name,
            dataset_folder=Input(path=dataset_folder_uri, type="uri_folder"),
            execution_manifest=execution_manifest_input,
            candidate_catalog=candidate_catalog_input,
            variants_list="",
            engine_list=engine_list,
            time_budget_per_variant=time_budget_per_variant,
            phaseb_time_budget_sec=phase_b_config["phase_timeout_seconds"],
            drift_baseline_in=drift_baseline_input,
            drift_baseline_uri=args.drift_baseline_in or "",
            # V3-Proposed Planner parameters
            planner_enabled=planner_enabled,
            round1_max_variants=round1_max,
            round2_max_variants=round2_max,
            proxy_prune_threshold=proxy_threshold,
            cache_enabled=cache_enabled
        )
    else:
        # DEFAULT: Use production pipeline with ALL selected variants
        # The variant runner processes every recipe in a single step
        engine_list_str = ",".join(phase_b_config["engines"])
        
        # Read time_budget_per_variant from config (fallback 600s)
        _pb_cfg = cfg.get("phases", {}).get("phase_b", {}) if 'cfg' in dir() else {}
        _time_budget = _pb_cfg.get("time_budget_per_variant", 600)
        print(f"🚀 Using production pipeline with {len(all_selected_recipes)} variants × engines={engine_list_str}, time_budget={_time_budget}s\n")
        planner_config = phase_b_config["planner"]
        round1_max = _compiled_round1_cap(phase_b_config)
        round2_max = planner_config["round2_max_variants"]
        proxy_threshold = planner_config["proxy_prune_threshold"]
        source_manifest = _compute_upload_source_manifest()
        code_identity = str(source_manifest["source_sha256"])
        execution_manifest, candidate_records = _build_execution_manifest(
            cfg,
            selected_catalog_entries,
            recipe_catalog,
            code_sha=code_identity,
            environment=env_version,
            component_environments=component_environments,
            round1_max_variants=round1_max,
            round2_max_variants=round2_max,
            proxy_prune_threshold=proxy_threshold,
        )
        (
            execution_manifest_path,
            catalog_evidence_path,
            source_manifest_path,
        ) = _persist_execution_artifacts(
            execution_manifest,
            candidate_records,
            recipe_catalog,
            selected_catalog_entries,
            source_manifest,
        )
        execution_manifest_input = _sdk_local_file_input(
            execution_manifest_path,
            datastore=default_datastore,
        )
        candidate_catalog_input = _sdk_local_file_input(
            catalog_evidence_path,
            datastore=default_datastore,
        )
        drift_baseline_input = Input(path=args.drift_baseline_in, type="uri_folder") if args.drift_baseline_in else None
        job = full_pipeline(
            config_name=config_name,
            dataset_folder=Input(path=dataset_folder_uri, type="uri_folder"),
            execution_manifest=execution_manifest_input,
            candidate_catalog=candidate_catalog_input,
            variants_list="",
            engine_list=engine_list_str,
            time_budget_per_variant=_time_budget,
            phaseb_time_budget_sec=phase_b_config["phase_timeout_seconds"],
            drift_baseline_in=drift_baseline_input,
            drift_baseline_uri=args.drift_baseline_in or "",
        )
    
    _configure_pipeline_job_settings(
        job,
        default_compute=args.compute,
        default_datastore=default_datastore,
        force_rerun=args.force_rerun,
    )
    credential_mode = resolve_credential_mode()
    if credential_mode == "azureml_obo":
        from azure.ai.ml.entities import UserIdentityConfiguration

        job.identity = UserIdentityConfiguration()

    # 🚀 Set display names for Phase B step (variant runner)
    if not use_phase1_pipeline:
        # Default pipeline: Set display name for variant runner step
        try:
            if hasattr(job, 'jobs') and 's06' in job.jobs:
                variant_count = len(all_selected_recipes)
                job.jobs['s06'].display_name = f"s06_phaseb_variant_runner__{variant_count}_variants"
                print(f"✅ Set display name: s06_phaseb_variant_runner__{variant_count}_variants")
        except Exception as e:
            logger.warning("Could not set display name (non-critical): %s", e)
    else:
        # Phase 1 pipeline: Set display name for variant runner step
        try:
            if hasattr(job, 'jobs') and 's06' in job.jobs:
                job.jobs['s06'].display_name = f"s06_phaseb_variant_runner__intelligent"
                print(f"✅ Set display name for intelligent variant runner")
        except Exception as e:
            logger.warning("Could not set display name (non-critical): %s", e)

    try:
        _validate_submission_revision_identity(
            revision_kind=args.submission_revision_kind,
            execution_manifest=execution_manifest,
            config_hash=cfg["compiled_config_hash"],
            source_identity=code_identity,
            parent_execution_id=args.parent_execution_id,
            parent_config_hash=args.parent_config_hash,
            parent_source_identity=args.parent_source_identity,
            expected_execution_id=args.expected_execution_id,
            expected_config_hash=args.expected_config_hash,
            expected_source_identity=args.expected_source_identity,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    job.experiment_name = args.experiment_name
    job.display_name = args.display_name

    # Add job-level tags for dataset/task/preset and pipeline version
    dataset_tag = (cfg.get('dataset') or {}).get('name') or 'unknown'
    task_tag = cfg.get('task_type') or 'unknown'
    preset_tag = cfg.get('preset') or 'unknown'
    job.tags = {
        'config_name': config_name,
        'dataset': dataset_tag,
        'task': task_tag,
        'preset': preset_tag,
        'pipeline_version': 'v3',
        'environment': env_version,
        'execution_id': execution_manifest.execution_id,
        'compiled_config_hash': cfg['compiled_config_hash'],
        'recipe_catalog_hash': recipe_catalog.catalog_hash,
        'source_identity': code_identity,
        'submission_revision_kind': args.submission_revision_kind,
    }
    if args.submission_revision_kind != "original":
        job.tags.update(
            {
                "parent_execution_id": args.parent_execution_id,
                "parent_config_hash": args.parent_config_hash,
                "parent_source_identity": args.parent_source_identity,
            }
        )
    if args.source_decision_id:
        job.tags["source_decision_id"] = args.source_decision_id
    if args.revision_reason:
        job.tags["revision_reason"] = str(args.revision_reason).strip()
    job.tags.update(extra_tags)
    if args.force:
        job.tags['force_submit'] = 'true'
        job.tags['force_submitted_by'] = os.getenv('USER', 'unknown')

    # Run the same SDK graph validation used by create_or_update even for dry
    # runs, so optional/required binding errors cannot hide in YAML rendering.
    job._validate(raise_error=True)

    # If Azure ML context provided, submit; else print YAML
    if args.dry_run:
        print("\n🔍 --dry_run: emitting pipeline job (NOT submitting)\n")
        print(job)
        return

    if args.subscription_id and args.resource_group and args.workspace_name:
        # ---------- Duplicate-submission guard: lock file ----------
        if not args.force:
            if not _acquire_lock():
                try:
                    lock_info = json.loads(_LOCK_FILE.read_text())
                except Exception as _le:
                    logger.warning("Could not read lock file %s: %s", _LOCK_FILE, _le)
                    lock_info = {}
                print("\n" + "="*80)
                print("🚫  DUPLICATE SUBMISSION BLOCKED")
                print("="*80)
                print(f"Another submit_pipeline.py is already running (PID {lock_info.get('pid')}, "
                      f"started {lock_info.get('started', '?')}, user {lock_info.get('user', '?')}).")
                print(f"If that process is dead, delete the lock file:")
                print(f"   rm {_LOCK_FILE}")
                print(f"Or use --force to submit anyway (audited).")
                print("="*80 + "\n")
                sys.exit(1)
            # Ensure lock is released on exit / signals
            atexit.register(_release_lock)
            signal.signal(signal.SIGTERM, _handle_submit_signal)
            signal.signal(signal.SIGINT, _handle_submit_signal)
        else:
            _force_user = os.getenv("USER") or os.getenv("USERNAME") or "unknown"
            print("\n" + "="*80)
            print(f"⚠️  SECURITY NOTICE: --force bypassed all submission guards")
            print(f"   user={_force_user}  pid={os.getpid()}  time={datetime.now().isoformat()}")
            print("="*80 + "\n")
            force_audit_id = _record_force_audit(args, _force_user)
            job.tags["force_audit_id"] = force_audit_id

        ml_client = get_ml_client(
            args.subscription_id,
            args.resource_group,
            args.workspace_name,
            credential_mode=credential_mode,
        )

        # ---------- Duplicate-submission guard: active-job check ----------
        if not args.force:
            active_jobs = _check_active_jobs(ml_client, args.experiment_name)
            if active_jobs:
                print("\n" + "="*80)
                print("⚠️   ACTIVE JOBS DETECTED in experiment: " + args.experiment_name)
                print("="*80)
                for aj in active_jobs:
                    print(f"   • {aj['name']}  [{aj['status']}]  {aj['display_name']}")
                print("\nA pipeline is already running. Submitting again will create a duplicate.")
                print("Use --force to submit anyway.")
                print("="*80 + "\n")
                _release_lock()
                sys.exit(1)

        _resolve_drift_baseline_input(ml_client, job)

        # Environment version from component YAMLs
        print(f"Note: Using environment {env_version} from component YAMLs (includes azureml-core, sweetviz, etc.)\n")

        print("🚀 Submitting pipeline to Azure ML (this may take several minutes on NFS)...")
        submitted = ml_client.jobs.create_or_update(job)
        studio_url = (
            f"https://ml.azure.com/runs/{submitted.name}"
            f"?wsid=/subscriptions/{args.subscription_id}"
            f"/resourceGroups/{args.resource_group}"
            "/providers/Microsoft.MachineLearningServices"
            f"/workspaces/{args.workspace_name}"
        )
        _write_submission_result(
            args.result_json,
            {
                "job_name": submitted.name,
                "experiment_name": args.experiment_name,
                "display_name": args.display_name,
                "status": submitted.status or "Submitted",
                "studio_url": studio_url,
            },
        )
        print(f"✅ Submitted job: {submitted.name}")
        # H2: do NOT leak subscription/rg/workspace IDs in the URL by default.
        if args.debug:
            print(f"🌐 Web View: https://ml.azure.com/runs/{submitted.name}?wsid=/subscriptions/{args.subscription_id}/resourcegroups/{args.resource_group}/workspaces/{args.workspace_name}")
        else:
            print(f"🌐 Web View: https://ml.azure.com/runs/{submitted.name}")

        # Write marker file for easy status checks later
        try:
            _USER_STATE_DIR.mkdir(parents=True, exist_ok=True)
            _LAST_JOB_FILE.write_text(json.dumps({
                "name": submitted.name,
                "display_name": args.display_name,
                "experiment": args.experiment_name,
                "submitted_at": datetime.now().isoformat(),
                "config": config_name,
            }, indent=2))
            print(f"📝 Job name saved to {_LAST_JOB_FILE}")
        except OSError as _me:
            logger.warning("Could not write last-job marker %s: %s", _LAST_JOB_FILE, _me)

        # Release lock after successful submission
        _release_lock()

        if args.wait:
            print("\n⏳ Waiting for pipeline to complete...")
            try:
                ml_client.jobs.stream(submitted.name)
            finally:
                if args.stop_compute:
                    compute_name = args.compute
                    print(f"\n🛑 Stopping compute cluster '{compute_name}'...")
                    try:
                        ml_client.compute.begin_stop(compute_name).wait()
                        print(f"✅ Compute cluster '{compute_name}' stopped successfully.")
                    except Exception as stop_err:
                        print(f"⚠️ Could not stop compute '{compute_name}': {stop_err}")
        elif args.stop_compute:
            print("⚠️ --stop_compute requires --wait. Compute will not be stopped.")
    else:
        # Local dry-run: emit job yaml for inspection
        print(job)


if __name__ == "__main__":
    main()
