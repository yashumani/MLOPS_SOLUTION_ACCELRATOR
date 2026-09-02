#!/usr/bin/env python3
"""Submit the governed 15-scenario qualification matrix through one entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _azure_ctx import (  # noqa: E402
    MissingAzureContextError,
    get_state_dir,
    load_azure_context,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs" / "qualification" / "industry_matrix_execution_catalog.yml"
SUBMITTER = ROOT / "pipelines" / "submit_pipeline.py"
TASK_TYPES = ("classification", "regression", "clustering")
MATRIX_ID = "industry-qualification-20260902"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    task_type: str
    industry: str
    config_path: str
    dataset_content_sha256: str
    dataset_schema_sha256: str


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def _scenario_from_record(record: dict[str, Any]) -> Scenario:
    required = (
        "scenario_id",
        "task_type",
        "industry",
        "config_path",
        "dataset_content_sha256",
        "dataset_schema_sha256",
    )
    missing = [name for name in required if not str(record.get(name) or "").strip()]
    if missing:
        raise ValueError(f"Qualification scenario is missing fields: {', '.join(missing)}")
    scenario = Scenario(**{name: str(record[name]).strip() for name in required})
    if scenario.task_type not in TASK_TYPES:
        raise ValueError(
            f"Unsupported task type {scenario.task_type!r} for {scenario.scenario_id}"
        )
    for name, value in (
        ("dataset_content_sha256", scenario.dataset_content_sha256),
        ("dataset_schema_sha256", scenario.dataset_schema_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{scenario.scenario_id} has invalid {name}")
    return scenario


def load_execution_catalog(path: Path = DEFAULT_CATALOG) -> list[Scenario]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    records = payload.get("configs")
    if not isinstance(records, list):
        raise ValueError("Qualification execution catalog must contain a configs list")

    scenarios = [_scenario_from_record(record) for record in records]
    if len(scenarios) != 15 or payload.get("scenario_count") != 15:
        raise ValueError("Qualification execution catalog must contain exactly 15 scenarios")
    identifiers = [scenario.scenario_id for scenario in scenarios]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Qualification execution catalog contains duplicate scenario IDs")

    actual_counts: dict[str, int] = {}
    for task_type in TASK_TYPES:
        task_scenarios = [item for item in scenarios if item.task_type == task_type]
        actual_counts[task_type] = len(task_scenarios)
        industries = {item.industry for item in task_scenarios}
        if len(task_scenarios) != 5 or len(industries) != 5:
            raise ValueError(
                f"{task_type} must contain five scenarios from five distinct industries"
            )
    if payload.get("task_counts") != actual_counts:
        raise ValueError("Qualification catalog task_counts do not match its scenarios")

    for scenario in scenarios:
        config_path = (ROOT / scenario.config_path).resolve()
        try:
            config_path.relative_to((ROOT / "configs").resolve())
        except ValueError as exc:
            raise ValueError(
                f"Scenario config escapes the repository configs root: {scenario.config_path}"
            ) from exc
        if not config_path.is_file():
            raise ValueError(f"Scenario config does not exist: {scenario.config_path}")
    return scenarios


def select_scenarios(
    scenarios: Iterable[Scenario],
    *,
    task_types: set[str] | None = None,
    scenario_ids: set[str] | None = None,
) -> list[Scenario]:
    selected = [
        scenario
        for scenario in scenarios
        if (not task_types or scenario.task_type in task_types)
        and (not scenario_ids or scenario.scenario_id in scenario_ids)
    ]
    if scenario_ids:
        missing = sorted(scenario_ids - {item.scenario_id for item in selected})
        if missing:
            raise ValueError(f"Unknown qualification scenarios: {', '.join(missing)}")
    if not selected:
        raise ValueError("No qualification scenarios matched the requested filters")
    return selected


def build_submission_command(
    scenario: Scenario,
    *,
    result_path: Path,
    context: Any,
    wait: bool = False,
    force_rerun: bool = False,
) -> list[str]:
    tags = {
        "qualification_matrix": MATRIX_ID,
        "qualification_scenario": scenario.scenario_id,
        "qualification_industry": scenario.industry,
    }
    command = [
        sys.executable,
        str(SUBMITTER),
        "--config",
        str(ROOT / scenario.config_path),
        *context.as_cli_args(),
        "--tags_json",
        json.dumps(tags, sort_keys=True),
        "--result_json",
        str(result_path),
    ]
    if wait:
        command.append("--wait")
    if force_rerun:
        command.append("--force_rerun")
    return command


def _git_identity() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _default_result_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return get_state_dir() / "qualification_submissions" / f"{stamp}.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or submit the exact 15-scenario qualification catalog. "
            "The default is read-only; --execute is required to submit."
        )
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--task-type", action="append", choices=TASK_TYPES)
    parser.add_argument("--scenario", action="append", dest="scenario_ids")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Disable component reuse without bypassing duplicate-submission guards.",
    )
    parser.add_argument("--continue-on-submission-error", action="store_true")
    parser.add_argument("--result-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        catalog_path = args.catalog.resolve()
        scenarios = select_scenarios(
            load_execution_catalog(catalog_path),
            task_types=set(args.task_type or []),
            scenario_ids=set(args.scenario_ids or []),
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Qualification catalog error: {exc}", file=sys.stderr)
        return 2

    print(f"Qualification matrix: {len(scenarios)} governed scenario(s)")
    for index, scenario in enumerate(scenarios, start=1):
        print(
            f"  [{index:02d}] {scenario.task_type:<14} "
            f"{scenario.industry:<24} {scenario.scenario_id}"
        )
    if not args.execute:
        print("Read-only plan complete. Pass --execute to submit through the canonical guard.")
        return 0

    try:
        context = load_azure_context()
        git_identity = _git_identity()
    except (MissingAzureContextError, OSError, subprocess.SubprocessError) as exc:
        print(f"Submission preflight failed: {exc}", file=sys.stderr)
        return 2
    if git_identity["dirty"]:
        print(
            "Submission preflight failed: qualification requires a clean Git worktree",
            file=sys.stderr,
        )
        return 2

    result_path = (args.result_json or _default_result_path()).resolve()
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "matrix_id": MATRIX_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_path": str(catalog_path),
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "git": git_identity,
        "azure": {
            "subscription_id": context.subscription_id,
            "resource_group": context.resource_group,
            "workspace_name": context.workspace_name,
            "compute": context.compute,
        },
        "requested_count": len(scenarios),
        "submissions": [],
    }

    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="mlops-qualification-matrix-") as tmp:
        temporary_root = Path(tmp)
        for index, scenario in enumerate(scenarios, start=1):
            submission_result = temporary_root / f"{index:02d}.json"
            command = build_submission_command(
                scenario,
                result_path=submission_result,
                context=context,
                wait=args.wait,
                force_rerun=args.force_rerun,
            )
            print(f"[{index:02d}/{len(scenarios):02d}] submitting {scenario.scenario_id}")
            completed = subprocess.run(command, cwd=ROOT, check=False)
            payload: dict[str, Any] = {}
            if submission_result.is_file():
                payload = json.loads(submission_result.read_text(encoding="utf-8"))
            accepted = completed.returncode == 0 and bool(payload.get("job_name"))
            record = {
                **asdict(scenario),
                "accepted": accepted,
                "submit_exit_code": completed.returncode,
                "job": payload,
            }
            summary["submissions"].append(record)
            _write_json_atomic(result_path, summary)
            if not accepted:
                exit_code = 1
                print(f"Submission failed for {scenario.scenario_id}", file=sys.stderr)
                if not args.continue_on_submission_error:
                    break

    summary["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["accepted_count"] = sum(
        1 for item in summary["submissions"] if item["accepted"]
    )
    summary["complete"] = (
        summary["accepted_count"] == summary["requested_count"] and exit_code == 0
    )
    _write_json_atomic(result_path, summary)
    print(f"Submission evidence: {result_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
