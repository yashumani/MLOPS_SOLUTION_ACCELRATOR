#!/usr/bin/env python3
"""Submit an isolated registered-model inference smoke job for a matrix run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from azure.ai.ml import MLClient, Output, command
from azure.ai.ml.entities import UserIdentityConfiguration
from azure.identity import AzureCliCredential

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _azure_ctx import (  # noqa: E402
    MissingAzureContextError,
    get_state_dir,
    load_azure_context,
)


ROOT = Path(__file__).resolve().parents[1]
SCORE_ROOT = Path(__file__).resolve().parent / "registered_model_inference_smoke"
DEFAULT_ENVIRONMENT = "mlops-v3-unified:33"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
SCENARIO_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")


def _require_sha256(name: str, value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"registry_info.{name} must be a SHA-256 digest")
    return normalized


def validate_registry_info(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("registry_info must be a JSON object")
    model_name = str(payload.get("model_name") or "").strip()
    version = str(payload.get("version") or "").strip()
    if MODEL_NAME_PATTERN.fullmatch(model_name) is None:
        raise ValueError("registry_info.model_name is invalid")
    if not model_name.startswith("mlops-v3-qualification-"):
        raise ValueError("registry_info must reference a qualification model")
    if not version.isdigit() or int(version) < 1:
        raise ValueError("registry_info.version must be a positive integer")
    expected_model_uri = f"models:/{model_name}/{version}"
    if payload.get("model_uri") != expected_model_uri:
        raise ValueError("registry_info.model_uri does not bind the exact version")
    if payload.get("promotion_mode") != "manual":
        raise ValueError("registry_info.promotion_mode must be manual")
    if payload.get("promotion_performed") is not False:
        raise ValueError("registry_info reports an unapproved promotion")
    if payload.get("lifecycle_stage") != "Unassigned" or payload.get("stage") != "None":
        raise ValueError("registry_info model must remain unassigned")

    validated = dict(payload)
    validated["execution_id"] = _require_sha256(
        "execution_id",
        payload.get("execution_id"),
    )
    validated["code_sha"] = _require_sha256("code_sha", payload.get("code_sha"))
    validated["dataset_content_sha256"] = _require_sha256(
        "dataset_content_sha256",
        payload.get("dataset_content_sha256"),
    )
    validated["model_name"] = model_name
    validated["version"] = version
    return validated


def _load_downloaded_registry_info(download_root: Path) -> dict[str, Any]:
    candidates = [
        path
        for path in download_root.rglob("*")
        if path.is_file() and path.name in {"registry_info", "registry_info.json"}
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one downloaded registry_info artifact, found "
            f"{len(candidates)}"
        )
    return validate_registry_info(
        json.loads(candidates[0].read_text(encoding="utf-8"))
    )


def _git_identity() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    if run("status", "--porcelain"):
        raise RuntimeError("Registered-model smoke submission requires a clean worktree")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
    }


def _environment_id(value: str) -> str:
    normalized = str(value).strip()
    return normalized if normalized.startswith("azureml:") else f"azureml:{normalized}"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _default_result_path(parent_job: str) -> Path:
    return get_state_dir() / "registered_model_smokes" / f"{parent_job}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-job", required=True)
    parser.add_argument("--environment", default=DEFAULT_ENVIRONMENT)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args(argv)
    if JOB_NAME_PATTERN.fullmatch(args.parent_job) is None:
        print("Smoke submission preflight failed: invalid parent job name", file=sys.stderr)
        return 2

    try:
        context = load_azure_context()
        git_identity = _git_identity()
    except (MissingAzureContextError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Smoke submission preflight failed: {exc}", file=sys.stderr)
        return 2
    if git_identity["branch"] in {"main", "master"} or git_identity[
        "branch"
    ].startswith("release/"):
        print(
            "Smoke submission preflight failed: protected branches require "
            "explicit production approval",
            file=sys.stderr,
        )
        return 2

    client = MLClient(
        AzureCliCredential(),
        context.subscription_id,
        context.resource_group,
        context.workspace_name,
    )
    try:
        parent = client.jobs.get(args.parent_job)
        if parent.status != "Completed":
            raise RuntimeError(
                f"Parent pipeline must be Completed, got {parent.status!r}"
            )
        parent_tags = {str(key): str(value) for key, value in (parent.tags or {}).items()}
        scenario_id = parent_tags.get("qualification_scenario", "")
        if SCENARIO_PATTERN.fullmatch(scenario_id) is None:
            raise RuntimeError("Parent job is not a governed qualification scenario")
        if not parent_tags.get("qualification_matrix"):
            raise RuntimeError("Parent job has no qualification_matrix tag")

        with tempfile.TemporaryDirectory(prefix="registered-model-smoke-") as temp:
            download_root = Path(temp)
            client.jobs.download(
                args.parent_job,
                download_path=download_root,
                output_name="registry_info",
            )
            registry_info = _load_downloaded_registry_info(download_root)

        if parent_tags.get("source_identity") != registry_info["code_sha"]:
            raise RuntimeError(
                "Parent source_identity does not match registered model code_sha"
            )
    except Exception as exc:  # noqa: BLE001 - convert SDK/contract failures to CLI exit
        print(f"Smoke submission preflight failed: {exc}", file=sys.stderr)
        return 2

    short_commit = git_identity["commit"][:8]
    display_name = f"{short_commit}-registered-smoke-{scenario_id}"
    task_type = str(registry_info.get("task_type") or "unknown")
    smoke_job = command(
        code=str(SCORE_ROOT),
        command=(
            "python score.py "
            "--model-uri '${{inputs.model_uri}}' "
            "--expected-execution-id '${{inputs.execution_id}}' "
            "--expected-code-sha '${{inputs.code_sha}}' "
            "--expected-dataset-sha '${{inputs.dataset_sha}}' "
            "--output-dir '${{outputs.evidence}}'"
        ),
        inputs={
            "model_uri": registry_info["model_uri"],
            "execution_id": registry_info["execution_id"],
            "code_sha": registry_info["code_sha"],
            "dataset_sha": registry_info["dataset_content_sha256"],
        },
        outputs={"evidence": Output(type="uri_folder")},
        environment=_environment_id(args.environment),
        compute=context.compute,
        identity=UserIdentityConfiguration(),
        experiment_name=f"qualification_registered_inference_{task_type}",
        display_name=display_name,
        tags={
            "qualification_matrix": parent_tags["qualification_matrix"],
            "qualification_scenario": scenario_id,
            "qualification_parent_job": args.parent_job,
            "qualification_check": "registered_model_inference",
            "source_identity": registry_info["code_sha"],
            "source_git_commit": git_identity["commit"],
            "execution_id": registry_info["execution_id"],
            "model_name": registry_info["model_name"],
            "model_version": registry_info["version"],
        },
    )

    try:
        created = client.jobs.create_or_update(smoke_job)
        if args.wait:
            client.jobs.stream(created.name)
            created = client.jobs.get(created.name)
    except Exception as exc:  # noqa: BLE001 - SDK error is preserved in evidence
        print(f"Smoke submission failed: {exc}", file=sys.stderr)
        return 1

    result = {
        "schema_version": "1.0",
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_name": created.name,
        "status": created.status,
        "display_name": display_name,
        "parent_job": args.parent_job,
        "scenario_id": scenario_id,
        "model_uri": registry_info["model_uri"],
        "execution_id": registry_info["execution_id"],
        "code_sha": registry_info["code_sha"],
        "source_git_commit": git_identity["commit"],
        "environment": _environment_id(args.environment),
        "compute": context.compute,
    }
    result_path = (args.result_json or _default_result_path(args.parent_job)).resolve()
    _write_json_atomic(result_path, result)
    print(json.dumps(result, indent=2))
    print(f"Submission evidence: {result_path}")
    return 0 if not args.wait or created.status == "Completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
