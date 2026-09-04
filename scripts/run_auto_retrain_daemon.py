#!/usr/bin/env python3
"""Initialize or run the external candidate controller on its approved state host."""

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
from utils.azure_helper import get_ml_client, resolve_credential_mode


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


def validate_state_paths() -> Path:
    root = Path(os.environ["MLOPS_STATE_DIR"]).expanduser()
    ledger = Path(os.environ["MLOPS_AUTO_RETRAIN_LEDGER"]).expanduser()
    database = state.database_path()
    if not root.is_absolute() or not ledger.is_absolute() or database is None:
        raise ValueError("Controller state paths must be absolute")
    root = root.resolve()
    ledger = ledger.resolve()
    if any(path == root or not path.is_relative_to(root) for path in (ledger, database)):
        raise ValueError("Controller database and ledger must be contained in MLOPS_STATE_DIR")
    if ledger == database:
        raise ValueError("Controller database and legacy ledger must have distinct paths")
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--initialize-state", action="store_true", help="Verify Azure access, bind an empty state database, and exit without submission")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-age-seconds", type=int, default=86400)
    parser.add_argument("--max-runs", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        if args.initialize_state and (args.manifest or args.execute or args.once):
            raise ValueError("--initialize-state cannot be combined with a manifest, --execute, or --once")
        if not args.initialize_state and args.manifest is None:
            raise ValueError("A reviewed --manifest is required to run the controller")
        if not 60 <= args.interval_seconds <= 3600 or not 60 <= args.max_age_seconds <= 86400 or not 1 <= args.max_runs <= 1000:
            raise ValueError("Invalid polling, freshness, or scan bound")
        if not args.initialize_state and not args.execute and not args.once:
            raise ValueError("Use --once for a dry run; continuous mode requires --execute")
        targets = [] if args.initialize_state else load_targets(args.manifest)
        required = ("AZURE_TENANT_ID", "AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME", "AZURE_COMPUTE", "MLOPS_OPERATIONAL_STATE_DB", "MLOPS_AUTO_RETRAIN_LEDGER", "MLOPS_STATE_DIR")
        missing = [name for name in required if not os.environ.get(name, "").strip()]
        if missing:
            raise ValueError("Missing controller settings: " + ", ".join(missing))
        state.configure_database(os.environ["MLOPS_OPERATIONAL_STATE_DB"])
        ledger = validate_state_paths()
        context = AzureSubmissionContext(*(os.environ[name] for name in required[1:5]))
        binding = {"tenant_id": os.environ["AZURE_TENANT_ID"], "subscription_id": context.subscription_id, "resource_group": context.resource_group, "workspace_name": context.workspace_name}
        if not args.initialize_state:
            state.bind_workspace(binding)
        credential_mode = resolve_credential_mode(
            os.environ.get("MLOPS_CONTROLLER_CREDENTIAL_MODE")
            or os.environ.get("MLOPS_AZURE_CREDENTIAL_MODE")
            or "managed_identity"
        )
        if credential_mode not in {"managed_identity", "azureml_obo"}:
            raise ValueError(
                "Controller credential mode must be managed_identity or azureml_obo"
            )
        # The deployed controller never falls back to a developer's cached CLI identity.
        client = get_ml_client(
            context.subscription_id,
            context.resource_group,
            context.workspace_name,
            credential_mode=credential_mode,
        )
        if args.initialize_state:
            client.workspaces.get(context.workspace_name)
            initialized = state.bind_workspace(binding, initialize=True)
            print(json.dumps({"status": "initialized" if initialized else "already_initialized", "workspace": binding}, sort_keys=True), flush=True)
            return 0
        while True:
            for schedule in PLANNED_AUTO_RETRAIN_SCHEDULES:
                live = client.schedules.get(schedule.schedule_name)
                if live.is_enabled is not False:
                    raise RuntimeError(f"Legacy schedule is not disabled: {schedule.schedule_name}")
            now = datetime.now(timezone.utc)
            discovered = [(target, discover_completed_runs(client, context, target.experiment_name, now=now, max_age_seconds=args.max_age_seconds, max_runs=args.max_runs)) for target in targets]
            for target, jobs in discovered:
                for job_name in jobs:
                    report = process_source_job(client, target, job_name, context=context, ledger=ledger, execute=args.execute, now=now, max_age_seconds=args.max_age_seconds, credential_mode=credential_mode)
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
