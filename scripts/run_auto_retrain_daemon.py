#!/usr/bin/env python3
"""Run the external candidate controller on the approved API/state host."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration import operational_state as state
from orchestration.auto_retrain_controller import AzureSubmissionContext
from orchestration.automated_retrain_controller import WatchTarget, discover_completed_runs, process_source_job
from orchestration.auto_retrain_schedule_catalog import PLANNED_AUTO_RETRAIN_SCHEDULES


def load_targets(path: Path) -> list[WatchTarget]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("Controller manifest schema_version must be 1.0")
    entries = payload.get("targets")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 30:
        raise ValueError("Controller requires between 1 and 30 explicit targets")
    targets = []
    for entry in entries:
        name = entry["config"]
        if not isinstance(name, str) or Path(name).name != name or Path(name).suffix not in {".yml", ".yaml"}:
            raise ValueError("Target config must be a reviewed configs/ filename")
        config = (ROOT / "configs" / name).resolve()
        if config.parent != (ROOT / "configs").resolve() or not config.is_file():
            raise ValueError("Target config must exist inside the canonical configs directory")
        experiment = entry["experiment_name"]
        if not isinstance(experiment, str) or not experiment.strip() or len(experiment) > 256:
            raise ValueError("Target experiment_name is required")
        targets.append(WatchTarget(config, experiment))
    if len({target.config_path for target in targets}) != len(targets):
        raise ValueError("Duplicate controller config target")
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-age-seconds", type=int, default=86400)
    parser.add_argument("--max-runs", type=int, default=200)
    args = parser.parse_args()
    try:
        if not 60 <= args.interval_seconds <= 3600 or not 60 <= args.max_age_seconds <= 86400 or not 1 <= args.max_runs <= 1000:
            raise ValueError("Invalid polling, freshness, or scan bound")
        if not args.execute and not args.once:
            raise ValueError("Use --once for a dry run; continuous mode requires --execute")
        targets = load_targets(args.manifest)
        required = ("AZURE_TENANT_ID", "AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME", "AZURE_COMPUTE", "MLOPS_OPERATIONAL_STATE_DB", "MLOPS_AUTO_RETRAIN_LEDGER", "MLOPS_STATE_DIR")
        missing = [name for name in required if not os.environ.get(name, "").strip()]
        if missing:
            raise ValueError("Missing controller settings: " + ", ".join(missing))
        state.configure_database(os.environ["MLOPS_OPERATIONAL_STATE_DB"])
        ledger = Path(os.environ["MLOPS_AUTO_RETRAIN_LEDGER"])
        if not ledger.is_absolute() or not Path(os.environ["MLOPS_STATE_DIR"]).is_absolute():
            raise ValueError("Controller state and ledger paths must be absolute and shared with the API")
        context = AzureSubmissionContext(*(os.environ[name] for name in required[1:5]))
        binding = {"tenant_id": os.environ["AZURE_TENANT_ID"], "subscription_id": context.subscription_id, "resource_group": context.resource_group, "workspace_name": context.workspace_name}
        with state.transaction() as connection:
            if state.get_document(connection, "configuration", "workspace") != binding:
                raise ValueError("Initialize the API database for this tenant/workspace before the controller")
        from azure.ai.ml import MLClient
        from azure.identity import ManagedIdentityCredential
        # The deployed service never falls back to a developer's cached CLI identity.
        credential = ManagedIdentityCredential(client_id=os.environ.get("AZURE_CLIENT_ID") or None)
        client = MLClient(credential, context.subscription_id, context.resource_group, context.workspace_name)
        while True:
            for schedule in PLANNED_AUTO_RETRAIN_SCHEDULES:
                live = client.schedules.get(schedule.schedule_name)
                if live.is_enabled is not False:
                    raise RuntimeError(f"Legacy schedule is not disabled: {schedule.schedule_name}")
            now = datetime.now(timezone.utc)
            discovered = [(target, discover_completed_runs(client, context, target.experiment_name, now=now, max_age_seconds=args.max_age_seconds, max_runs=args.max_runs)) for target in targets]
            for target, jobs in discovered:
                for job_name in jobs:
                    report = process_source_job(client, target, job_name, context=context, ledger=ledger, execute=args.execute, now=now, max_age_seconds=args.max_age_seconds)
                    print(json.dumps(report, sort_keys=True), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Controller stopped: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
