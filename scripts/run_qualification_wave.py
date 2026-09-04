#!/usr/bin/env python3
"""Run one bounded Azure-only qualification wave through final evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for _import_root in (ROOT, ROOT / "src", Path(__file__).resolve().parent):
    _import_root_text = str(_import_root)
    if _import_root_text not in sys.path:
        sys.path.insert(0, _import_root_text)

from _azure_ctx import load_azure_context  # noqa: E402
from audit_qualification_data_assets import main as audit_assets  # noqa: E402
from batch_submit_all import main as submit_batch  # noqa: E402
from collect_qualification_evidence import collect_wave  # noqa: E402
from monitor_batch import main as monitor_batch  # noqa: E402
from utils.azure_helper import get_ml_client  # noqa: E402
from verify_qualification_evidence import verify_qualification_evidence  # noqa: E402


SCENARIO_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", required=True)
    parser.add_argument("--datastore-canary-job", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment", default="mlops-v3-unified:33")
    parser.add_argument("--output-datastore", default="mlops_blob")
    parser.add_argument("--poll-minutes", type=float, default=2.0)
    parser.add_argument("--max-hours", type=float, default=8.0)
    args = parser.parse_args(argv)

    output = args.output_dir.resolve()
    summary_path = output / "qualification-wave-summary.json"
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "azureml_run_id": os.environ.get("AZUREML_RUN_ID"),
        "status": "running",
        "scenarios": list(args.scenario),
    }
    try:
        if not os.environ.get("AZUREML_RUN_ID"):
            raise RuntimeError("Qualification waves must execute inside Azure ML")
        if not 1 <= len(args.scenario) <= 2 or len(set(args.scenario)) != len(
            args.scenario
        ):
            raise ValueError("A wave requires one or two distinct scenarios")
        if any(SCENARIO_PATTERN.fullmatch(value) is None for value in args.scenario):
            raise ValueError("Qualification scenario name is invalid")
        if not 0.25 <= args.poll_minutes <= 15:
            raise ValueError("--poll-minutes must be between 0.25 and 15")
        if not 1 <= args.max_hours <= 24:
            raise ValueError("--max-hours must be between 1 and 24")
        output.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(summary_path, summary)

        context = load_azure_context()
        client = get_ml_client(
            context.subscription_id,
            context.resource_group,
            context.workspace_name,
        )
        compute = client.compute.get(context.compute)
        maximum_nodes = getattr(compute, "max_instances", None)
        if not isinstance(maximum_nodes, int) or maximum_nodes < 2:
            raise RuntimeError(
                "Qualification orchestration requires cluster max_instances >= 2"
            )
        summary["compute_capacity"] = {
            "name": context.compute,
            "max_instances": maximum_nodes,
        }
        _write_json_atomic(summary_path, summary)

        asset_audit = output / "azure-data-asset-audit.json"
        audit_exit = audit_assets(["--output-json", str(asset_audit)])
        if audit_exit != 0:
            raise RuntimeError("Live qualification data asset audit failed")
        summary["data_asset_audit"] = str(asset_audit)
        _write_json_atomic(summary_path, summary)

        submission = output / "qualification-submissions.json"
        submit_args = [
            "--execute",
            "--datastore-canary-job",
            args.datastore_canary_job,
            "--result-json",
            str(submission),
        ]
        for scenario in args.scenario:
            submit_args.extend(("--scenario", scenario))
        submit_exit = submit_batch(submit_args)
        if submit_exit != 0:
            raise RuntimeError("Qualification parent submission failed")
        summary["submission_manifest"] = str(submission)
        _write_json_atomic(summary_path, summary)

        monitor_dir = output / "monitor"
        monitor_exit = monitor_batch(
            [
                "--submissions",
                str(submission),
                "--output-dir",
                str(monitor_dir),
                "--interval",
                str(args.poll_minutes),
                "--max-hours",
                str(args.max_hours),
            ]
        )
        if monitor_exit != 0:
            raise RuntimeError(
                f"Qualification parent monitoring failed with exit {monitor_exit}"
            )
        monitor_summary = monitor_dir / "monitor-summary.json"
        summary["monitor_summary"] = str(monitor_summary)
        _write_json_atomic(summary_path, summary)

        evidence_manifest = collect_wave(
            client,
            submission_path=submission,
            monitor_path=monitor_summary,
            asset_audit_path=asset_audit,
            output_dir=output,
            environment=args.environment,
            output_datastore=args.output_datastore,
        )
        evidence_report = verify_qualification_evidence(evidence_manifest)
        report_path = output / "qualification-evidence-report.json"
        _write_json_atomic(report_path, evidence_report)
        if evidence_report.get("state") != "passed":
            raise RuntimeError("Qualification wave evidence verification failed")
        summary.update(
            {
                "status": "passed",
                "evidence_manifest": str(evidence_manifest),
                "evidence_report": str(report_path),
                "accepted_scenario_count": evidence_report.get(
                    "accepted_scenario_count"
                ),
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - report exact remote failure
        summary["status"] = "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"Qualification wave failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(summary_path, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
