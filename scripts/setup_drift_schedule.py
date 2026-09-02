"""Fail-closed compatibility entrypoint for the retired training DAG schedule.

Azure ML ``JobSchedule`` stores a static job template. A static training DAG
cannot consume a fresh S14 policy decision on every tick, so this legacy setup
entrypoint must not create or update a recurring training pipeline.

Schedule an external controller invocation instead. That controller must receive
the explicit ``retrain_decision.json`` emitted by S14 and delegates the actual
submission to the canonical ``pipelines/submit_pipeline.py`` entrypoint.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts" / "run_auto_retrain_controller.py"
CADENCE_CHOICES = ("hourly", "daily", "weekly")


def _controller_command_preview(args: argparse.Namespace) -> list[str]:
    """Return the external-controller shape operators should schedule."""
    config_path = Path(args.config)
    if not config_path.suffix:
        config_path = ROOT / "configs" / f"{args.config}.yml"
    return [
        sys.executable,
        str(CONTROLLER),
        "--config",
        str(config_path),
        "--decision",
        "<fresh-s14-retrain-decision.json>",
        "--ledger",
        "<shared-auto-retrain-decision-ledger.jsonl>",
        "--mode",
        "submit",
        "--trigger",
        f"schedule:{args.cadence}",
        "--schedule-name",
        args.name,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retired AML training schedule setup; prints the external controller boundary"
    )
    parser.add_argument("--name", required=True, help="External schedule/controller name")
    parser.add_argument("--config", required=True, help="V3 config path or config stem")
    parser.add_argument("--cadence", choices=CADENCE_CHOICES, default="daily")
    parser.add_argument(
        "--schedule-mode",
        choices=["candidate_retrain"],
        default="candidate_retrain",
        help="Retained only for legacy CLI compatibility",
    )
    parser.add_argument(
        "--drift-baseline-in",
        default=None,
        help="Retired: the controller resolves the approved baseline from its ledger",
    )
    parser.add_argument(
        "--start-in-minutes",
        type=int,
        default=5,
        help="Retained only for legacy CLI compatibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the required external-controller command shape",
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="No Azure mutation is performed; disable old schedules through approved operations",
    )
    args = parser.parse_args()

    print(
        "Direct Azure ML training-DAG schedules are disabled: a static schedule "
        "cannot enforce a fresh S14 decision."
    )
    print("Schedule the external controller after S14 produces its decision artifact:")
    print(f"  {shlex.join(_controller_command_preview(args))}")
    if args.disable:
        print(
            "This compatibility entrypoint does not mutate Azure schedules; "
            "use approved Azure ML operations to disable the named legacy schedule.",
            file=sys.stderr,
        )
        return 2
    if args.dry_run:
        return 0
    print(
        "No schedule was created or updated. Re-run with --dry-run to inspect "
        "the controller boundary.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
