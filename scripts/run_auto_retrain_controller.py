#!/usr/bin/env python
"""Consume an explicit S14 decision and submit a V3 auto-retrain candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration.auto_retrain_controller import (  # noqa: E402
    AutoRetrainControllerError,
    AutoRetrainControllerRequest,
    AzureSubmissionContext,
    build_controller_plan,
    build_pending_decision_record,
    parse_submitted_job_name,
)
from orchestration.auto_retrain_decision_ledger import (  # noqa: E402
    DecisionReservationConflict,
    append_decision_record,
    reserve_candidate_submission,
)


ACTIVE_STATUSES = {
    "Running",
    "Queued",
    "Preparing",
    "Starting",
    "NotStarted",
    "Provisioning",
    "CancelRequested",
}


def _env(name: str) -> str | None:
    return os.environ.get(name) or None


def _default_ledger_path() -> Path:
    override = _env("MLOPS_AUTO_RETRAIN_LEDGER")
    if override:
        return Path(override)
    return ROOT / "outputs" / "auto_retrain_decisions.jsonl"


def _active_jobs(context: AzureSubmissionContext, experiment_name: str) -> list[dict]:
    command = [
        "az",
        "ml",
        "job",
        "list",
        "--resource-group",
        context.resource_group,
        "--workspace-name",
        context.workspace_name,
        "--max-results",
        "100",
        "-o",
        "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AutoRetrainControllerError(
            "Could not query Azure ML jobs before submission: " + (result.stderr or result.stdout)
        )
    jobs = json.loads(result.stdout or "[]")
    return [
        {
            "name": job.get("name"),
            "status": job.get("status"),
            "display_name": job.get("display_name"),
        }
        for job in jobs
        if job.get("experiment_name") == experiment_name and job.get("status") in ACTIVE_STATUSES
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V3 auto-retrain controller")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument(
        "--decision",
        required=True,
        help="S14 retrain_decision JSON (or its containing output folder)",
    )
    parser.add_argument("--ledger", default=str(_default_ledger_path()), help="Auto-retrain JSONL ledger path")
    parser.add_argument("--mode", choices=["dry_run", "submit"], default="dry_run")
    parser.add_argument("--trigger", default="manual_controller")
    parser.add_argument("--schedule-name", default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--display-name", default=None)
    parser.add_argument("--subscription-id", default=_env("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=_env("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace-name", default=_env("AZURE_WORKSPACE_NAME"))
    parser.add_argument("--compute", default=_env("AZURE_COMPUTE"))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--skip-active-check", action="store_true")
    parser.add_argument("--force-submit", action="store_true", help="Pass --force to submit_pipeline.py")
    parser.add_argument("--force-reason", default=None, help="Required audit reason when --force-submit is used")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra arg to append to submit_pipeline.py")
    args = parser.parse_args()

    if args.force_submit and not (args.force_reason or "").strip():
        print("--force-reason is required when --force-submit is used", file=sys.stderr)
        return 2
    if args.mode == "submit" and args.skip_active_check and not (args.force_reason or "").strip():
        print("--force-reason is required when --skip-active-check is used in submit mode", file=sys.stderr)
        return 2

    missing = [
        name
        for name, value in (
            ("subscription_id", args.subscription_id),
            ("resource_group", args.resource_group),
            ("workspace_name", args.workspace_name),
            ("compute", args.compute),
        )
        if not value
    ]
    if missing:
        print("Missing Azure context: " + ", ".join(missing), file=sys.stderr)
        return 2

    context = AzureSubmissionContext(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        compute=args.compute,
    )
    request = AutoRetrainControllerRequest(
        config_path=Path(args.config),
        ledger_path=Path(args.ledger),
        decision_path=Path(args.decision),
        azure_context=context,
        mode=args.mode,
        trigger=args.trigger,
        schedule_name=args.schedule_name,
        experiment_name=args.experiment_name,
        display_name=args.display_name,
        python_executable=args.python_executable,
        force_submit=args.force_submit,
        force_reason=args.force_reason,
        skip_active_check=args.skip_active_check,
        extra_args=tuple(args.extra_arg),
    )

    try:
        plan = build_controller_plan(request)
        print("AUTO-RETRAIN CONTROLLER PLAN")
        print(f"  config={plan.metadata.config_stem}")
        print(f"  task_type={plan.metadata.task_type}")
        print(f"  dataset={plan.metadata.dataset_name}")
        print(f"  s14_decision_id={plan.decision_payload['decision_id']}")
        source_decision = plan.decision_payload["retrain_decision"]
        print(f"  s14_outcome={source_decision.get('outcome')}")
        print(f"  experiment={plan.experiment_name}")
        print(f"  display_name={plan.display_name}")
        print("  baseline_in=provided")
        print(f"  command={plan.command_text}")

        if not args.skip_active_check:
            active_jobs = _active_jobs(context, plan.experiment_name)
            if active_jobs and not args.force_submit:
                print("Active jobs already exist for this experiment; refusing duplicate submission.")
                for job in active_jobs:
                    print(f"  {job['name']} [{job['status']}] {job.get('display_name') or ''}")
                return 3

        if args.mode == "dry_run":
            record = build_pending_decision_record(plan)
            print("Dry run only; no Azure ML job submitted and no ledger record written.")
            print("PENDING_DECISION_RECORD")
            print(json.dumps(record.as_dict(), indent=2, sort_keys=True))
            return 0

        reservation = build_pending_decision_record(
            plan,
            promotion_status="submitting",
        )
        reserve_candidate_submission(args.ledger, reservation)
        print(f"Submission reservation appended to {args.ledger}")
        try:
            result = subprocess.run(
                list(plan.command),
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                check=False,
                timeout=args.timeout_seconds,
            )
        except BaseException as exc:
            append_decision_record(
                args.ledger,
                replace(
                    reservation,
                    promotion_status="submission_failed",
                    metadata={
                        **reservation.metadata,
                        "submission_error": f"{type(exc).__name__}: {exc}",
                    },
                ),
            )
            raise
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            append_decision_record(
                args.ledger,
                replace(
                    reservation,
                    promotion_status="submission_failed",
                    metadata={
                        **reservation.metadata,
                        "submit_returncode": result.returncode,
                    },
                ),
            )
            return result.returncode

        candidate_job_name = parse_submitted_job_name(result.stdout)
        record = replace(
            reservation,
            candidate_job_name=candidate_job_name,
            promotion_status="manual_pending",
        )
        append_decision_record(args.ledger, record)
        print(f"Decision record appended to {args.ledger}")
        return 0
    except (AutoRetrainControllerError, DecisionReservationConflict) as exc:
        print(f"Controller failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
