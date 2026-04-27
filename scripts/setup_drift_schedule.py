"""Create / update an Azure ML schedule for the drift-only pipeline.

Phase 5 of the production hardening plan. The schedule fires the existing
``pipelines/submit_pipeline.py`` against a target config on a recurrence:

  * Default cadence: hourly (first week post-launch)
  * Steady-state:    daily   (after first week)

Switch cadence with ``--cadence {hourly,daily}``. The script is idempotent:
a schedule with the same name is updated in place via ``begin_create_or_update``.

Per the locked-in defaults (#1 alert-only retraining for first 30 days), the
schedule submits a ``--drift-only`` pipeline; retraining is NOT auto-triggered.
Alerts are dispatched from inside ``s13_drift_monitor.py`` via ``utils.alerts``.

Env vars (required, via scripts/_azure_ctx.py):
  AZURE_SUBSCRIPTION_ID
  AZURE_RESOURCE_GROUP
  AZURE_WORKSPACE_NAME
  AZURE_COMPUTE

Optional env vars (alerting; passed through to schedule env):
  TEAMS_WEBHOOK_URL
  ACS_CONNECTION_STRING / ACS_SENDER_ADDRESS / DRIFT_ALERT_RECIPIENTS
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from _azure_ctx import (  # noqa: E402
    AzureContext,
    MissingAzureContextError,
    load_azure_context,
)

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (
        JobSchedule,
        RecurrencePattern,
        RecurrenceTrigger,
    )
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        ManagedIdentityCredential,
    )
except ImportError as exc:  # pragma: no cover — surfaced at runtime
    print(f"❌ azure-ai-ml not available: {exc}", file=sys.stderr)
    sys.exit(2)


CADENCE_PRESETS = {
    "hourly": {"frequency": "hour", "interval": 1, "minutes": [0]},
    "daily": {"frequency": "day", "interval": 1, "hours": [2], "minutes": [0]},
}


def _ml_client(ctx: AzureContext) -> MLClient:
    cred = ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential())
    return MLClient(
        credential=cred,
        subscription_id=ctx.subscription_id,
        resource_group_name=ctx.resource_group,
        workspace_name=ctx.workspace_name,
    )


def _build_trigger(cadence: str, start_in_minutes: int) -> RecurrenceTrigger:
    preset = CADENCE_PRESETS[cadence]
    start_time = datetime.now(timezone.utc) + timedelta(minutes=start_in_minutes)
    pattern_kwargs = {"minutes": preset["minutes"]}
    if "hours" in preset:
        pattern_kwargs["hours"] = preset["hours"]
    return RecurrenceTrigger(
        frequency=preset["frequency"],
        interval=preset["interval"],
        start_time=start_time,
        time_zone="UTC",
        schedule=RecurrencePattern(**pattern_kwargs),
    )


def _load_pipeline_job(config_name: str, ctx: AzureContext):
    """Build the pipeline job object the schedule will submit on each tick."""
    repo_root = ROOT.parent  # mlops-solution-accelerator-v3/
    sys.path.insert(0, str(repo_root / "pipelines"))
    from pipeline_builder import full_pipeline  # type: ignore[import-untyped]

    config_path = repo_root / "configs" / f"{config_name}.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    job = full_pipeline(config_name=config_name)
    job.settings.default_compute = ctx.compute
    job.experiment_name = f"{config_name.replace('config_', '').replace('_azureml', '')}_drift"
    job.display_name = f"drift-monitor::{config_name}"
    job.tags = {"purpose": "drift_monitor", "schedule": "true"}
    return job


def main() -> int:
    parser = argparse.ArgumentParser(description="Create/update drift-monitor schedule")
    parser.add_argument("--name", required=True, help="Schedule name (idempotent)")
    parser.add_argument("--config", required=True,
                        help="Config filename without .yml (e.g. config_classification_telco_churn_azureml)")
    parser.add_argument("--cadence", choices=list(CADENCE_PRESETS), default="hourly",
                        help="Recurrence cadence (default: hourly)")
    parser.add_argument("--start-in-minutes", type=int, default=5,
                        help="Minutes from now to first execution (default: 5)")
    parser.add_argument("--disable", action="store_true",
                        help="Disable an existing schedule instead of creating it")
    args = parser.parse_args()

    try:
        ctx = load_azure_context()
    except MissingAzureContextError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    client = _ml_client(ctx)

    if args.disable:
        print(f"⏸  Disabling schedule: {args.name}")
        client.schedules.begin_disable(args.name).result()
        print("✅ Disabled")
        return 0

    print(f"📅 Building drift schedule: {args.name} ({args.cadence})")
    pipeline_job = _load_pipeline_job(args.config, ctx)
    trigger = _build_trigger(args.cadence, args.start_in_minutes)

    schedule = JobSchedule(
        name=args.name,
        trigger=trigger,
        create_job=pipeline_job,
        description=f"Drift-only pipeline ({args.cadence}) for {args.config}",
        tags={"cadence": args.cadence, "config": args.config, "owner": "mlops"},
    )

    print(f"📤 Submitting schedule …")
    poller = client.schedules.begin_create_or_update(schedule)
    result = poller.result()
    print(f"✅ Schedule '{result.name}' active (cadence={args.cadence})")
    print(f"   First run starts ~{args.start_in_minutes} min from now (UTC)")
    print(f"   To switch cadence later: rerun with --cadence daily")
    return 0


if __name__ == "__main__":
    sys.exit(main())
