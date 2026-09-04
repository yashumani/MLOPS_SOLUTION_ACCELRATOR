#!/usr/bin/env python3
"""Collect one terminal qualification wave and its registered-model smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for _import_root in (ROOT / "src", Path(__file__).resolve().parent):
    _import_root_text = str(_import_root)
    if _import_root_text not in sys.path:
        sys.path.insert(0, _import_root_text)

from _azure_ctx import load_azure_context  # noqa: E402
from submit_registered_model_smoke import main as submit_smoke  # noqa: E402
from utils.azure_helper import get_ml_client  # noqa: E402


REQUIRED_OUTPUTS = (
    "execution_manifest",
    "split_manifest",
    "quality_decision",
    "final_report",
    "registry_info",
    "drift_report",
    "retrain_decision",
    "decision_ledger_record",
)
SCENARIO_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _single_json(root: Path, *, label: str) -> dict[str, Any]:
    files = [path for path in root.rglob("*") if path.is_file()]
    if len(files) != 1:
        raise RuntimeError(f"{label} must contain exactly one file; found {len(files)}")
    return _read_json(files[0])


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _status_text(value: object) -> str:
    return str(getattr(value, "value", value) or "")


def collect_wave(
    client: Any,
    *,
    submission_path: Path,
    monitor_path: Path,
    asset_audit_path: Path,
    output_dir: Path,
    environment: str,
    output_datastore: str,
) -> Path:
    submission = _read_json(submission_path)
    monitor = _read_json(monitor_path)
    asset_audit = _read_json(asset_audit_path)
    if submission.get("complete") is not True:
        raise RuntimeError("Qualification submission manifest is incomplete")
    if monitor.get("state") != "passed":
        raise RuntimeError("Qualification monitor is not terminal-passing")
    if asset_audit.get("all_passed") is not True:
        raise RuntimeError("Qualification data asset audit did not pass")
    records = submission.get("submissions")
    if not isinstance(records, list) or not 1 <= len(records) <= 2:
        raise RuntimeError("A qualification wave must contain one or two scenarios")
    if submission.get("requested_count") != len(records) or submission.get(
        "accepted_count"
    ) != len(records):
        raise RuntimeError("Qualification submission counts are inconsistent")
    git = submission.get("git")
    if not isinstance(git, dict):
        raise RuntimeError("Qualification source identity is missing")
    git_commit = str(git.get("commit") or "").lower()
    archive_sha256 = str(git.get("archive_sha256") or "").lower()
    if (
        GIT_SHA_PATTERN.fullmatch(git_commit) is None
        or SHA256_PATTERN.fullmatch(archive_sha256) is None
        or git.get("provenance") != "verified_azure_archive"
        or git.get("dirty") is not False
        or not str(git.get("branch") or "").strip()
    ):
        raise RuntimeError(
            "Qualification wave requires a clean checksum-verified Azure archive"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios: list[dict[str, str]] = []
    runtime_sources: set[str] = set()
    scenario_ids: set[str] = set()
    parent_jobs: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or record.get("accepted") is not True:
            raise RuntimeError("Qualification wave contains an unaccepted submission")
        scenario_id = str(record.get("scenario_id") or "")
        job_payload = record.get("job") or {}
        parent_job = str(job_payload.get("job_name") or "")
        if SCENARIO_PATTERN.fullmatch(scenario_id) is None or not parent_job:
            raise RuntimeError("Qualification submission identity is incomplete")
        if scenario_id in scenario_ids or parent_job in parent_jobs:
            raise RuntimeError("Qualification submission identities must be unique")
        scenario_ids.add(scenario_id)
        parent_jobs.add(parent_job)
        parent = client.jobs.get(parent_job)
        if _status_text(parent.status).lower() != "completed":
            raise RuntimeError(
                f"Qualification parent {parent_job} is not Completed: {parent.status}"
            )
        parent_tags = {
            str(key): str(value) for key, value in (parent.tags or {}).items()
        }
        if parent_tags.get("qualification_scenario") != scenario_id or not parent_tags.get(
            "qualification_matrix"
        ):
            raise RuntimeError(
                f"Qualification parent {parent_job} has inconsistent scenario tags"
            )

        scenario_root = output_dir / "scenarios" / scenario_id
        pipeline_root = scenario_root / "pipeline"
        for output_name in REQUIRED_OUTPUTS:
            client.jobs.download(
                name=parent_job,
                output_name=output_name,
                download_path=pipeline_root,
            )
        execution = _single_json(
            pipeline_root / "named-outputs" / "execution_manifest",
            label=f"{scenario_id} execution manifest",
        )
        runtime_source = str(execution.get("code_sha") or "")
        if SHA256_PATTERN.fullmatch(runtime_source) is None:
            raise RuntimeError(f"{scenario_id} runtime source identity is invalid")
        if parent_tags.get("source_identity") != runtime_source:
            raise RuntimeError(f"{scenario_id} parent source identity does not match")
        runtime_sources.add(runtime_source)

        smoke_submission = scenario_root / "smoke-submission.json"
        smoke_exit = submit_smoke(
            [
                "--parent-job",
                parent_job,
                "--environment",
                environment,
                "--output-datastore",
                output_datastore,
                "--result-json",
                str(smoke_submission),
                "--wait",
            ]
        )
        if smoke_exit != 0:
            raise RuntimeError(
                f"Registered-model smoke submission failed for {scenario_id}"
            )
        smoke_record = _read_json(smoke_submission)
        if smoke_record.get("status") != "Completed":
            raise RuntimeError(f"Registered-model smoke did not complete: {scenario_id}")
        smoke_root = scenario_root / "smoke-evidence"
        client.jobs.download(
            name=str(smoke_record["job_name"]),
            output_name="evidence",
            download_path=smoke_root,
        )
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "parent_job": parent_job,
                "pipeline_evidence_dir": _relative(pipeline_root, output_dir),
                "registered_model_smoke_submission": _relative(
                    smoke_submission, output_dir
                ),
                "registered_model_smoke_evidence": _relative(
                    smoke_root, output_dir
                ),
            }
        )

    if len(runtime_sources) != 1:
        raise RuntimeError(
            "Qualification wave does not share one runtime source identity"
        )
    manifest = {
        "schema_version": "1.0",
        "monitor_summaries": [_relative(monitor_path, output_dir)],
        "data_asset_audit": _relative(asset_audit_path, output_dir),
        "release_candidate": {
            "git_commit": git_commit,
            "runtime_source_sha256": next(iter(runtime_sources)),
        },
        "source_archive_sha256": archive_sha256,
        "scenarios": scenarios,
    }
    manifest_path = output_dir / "qualification-evidence-manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-json", type=Path, required=True)
    parser.add_argument("--monitor-summary", type=Path, required=True)
    parser.add_argument("--data-asset-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment", default="mlops-v3-unified:33")
    parser.add_argument("--output-datastore", default="mlops_blob")
    args = parser.parse_args(argv)
    try:
        context = load_azure_context()
        client = get_ml_client(
            context.subscription_id,
            context.resource_group,
            context.workspace_name,
        )
        manifest = collect_wave(
            client,
            submission_path=args.submission_json.resolve(),
            monitor_path=args.monitor_summary.resolve(),
            asset_audit_path=args.data_asset_audit.resolve(),
            output_dir=args.output_dir.resolve(),
            environment=args.environment,
            output_datastore=args.output_datastore,
        )
        print(json.dumps({"status": "passed", "manifest": str(manifest)}))
        return 0
    except Exception as exc:  # noqa: BLE001 - failure is preserved in job output
        print(
            f"Qualification evidence collection failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
