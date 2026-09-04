#!/usr/bin/env python3
"""Monitor a governed Azure ML qualification submission manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

TERMINAL_STATUSES = {
    "Completed",
    "Failed",
    "CanceledOrFailed",
    "Canceled",
    "NotResponding",
}
FAILURE_STATUSES = {
    "Failed",
    "CanceledOrFailed",
    "Canceled",
    "NotResponding",
}


class SubmissionManifestError(ValueError):
    """Raised when submission evidence cannot safely drive monitoring."""


@dataclass(frozen=True)
class Submission:
    label: str
    job_id: str


@dataclass(frozen=True)
class SubmissionManifest:
    source: str
    expected: int
    jobs: tuple[Submission, ...]
    submission_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class PollResult:
    label: str
    job_id: str
    status: str
    query_error: str | None = None


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_expected(value: Any, *, source: str) -> int:
    if isinstance(value, bool):
        raise SubmissionManifestError(f"{source} expected count must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SubmissionManifestError(
            f"{source} expected count must be a positive integer"
        ) from exc
    if parsed < 1:
        raise SubmissionManifestError(f"{source} expected count must be a positive integer")
    return parsed


def _validate_unique_jobs(jobs: Iterable[Submission]) -> tuple[Submission, ...]:
    validated = tuple(jobs)
    identifiers = [item.job_id for item in validated]
    if len(set(identifiers)) != len(identifiers):
        raise SubmissionManifestError("Submission manifest contains duplicate job IDs")
    return validated


def _load_json_manifest(path: Path, expected_override: int | None) -> SubmissionManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SubmissionManifestError(f"Cannot read submission JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SubmissionManifestError("Submission JSON must contain an object")
    records = payload.get("submissions")
    if not isinstance(records, list):
        raise SubmissionManifestError("Submission JSON must contain a submissions list")

    declared_expected = _positive_expected(
        payload.get("requested_count"),
        source="Submission JSON requested_count",
    )
    if expected_override is not None and expected_override != declared_expected:
        raise SubmissionManifestError(
            "--expected does not match submission JSON requested_count "
            f"({expected_override} != {declared_expected})"
        )
    expected = expected_override or declared_expected

    jobs: list[Submission] = []
    failures: list[str] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise SubmissionManifestError(
                f"Submission record {index} must be an object"
            )
        label = str(
            record.get("scenario_id")
            or record.get("config_path")
            or f"submission-{index}"
        ).strip()
        accepted = record.get("accepted") is True
        job = record.get("job")
        job_id = str(job.get("job_name") or "").strip() if isinstance(job, dict) else ""
        if accepted and not job_id:
            raise SubmissionManifestError(
                f"Accepted submission {label!r} has no job.job_name"
            )
        if not accepted:
            failures.append(label)
            continue
        jobs.append(Submission(label=label, job_id=job_id))

    if len(records) > expected:
        raise SubmissionManifestError(
            f"Submission JSON has {len(records)} records but expected {expected}"
        )
    return SubmissionManifest(
        source="canonical_json",
        expected=expected,
        jobs=_validate_unique_jobs(jobs),
        submission_failures=tuple(failures),
    )


def _load_tsv_manifest(path: Path, expected_override: int | None) -> SubmissionManifest:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "job_id" not in reader.fieldnames:
                raise SubmissionManifestError("Legacy TSV must contain a job_id column")
            rows = list(reader)
    except SubmissionManifestError:
        raise
    except OSError as exc:
        raise SubmissionManifestError(f"Cannot read submission TSV {path}: {exc}") from exc

    jobs: list[Submission] = []
    for index, row in enumerate(rows, start=1):
        job_id = str(row.get("job_id") or "").strip()
        if not job_id:
            continue
        label = str(
            row.get("scenario_id")
            or row.get("config")
            or row.get("wave")
            or f"submission-{index}"
        ).strip()
        jobs.append(Submission(label=label, job_id=job_id))
    expected = expected_override or len(jobs)
    expected = _positive_expected(expected, source="TSV expected")
    if len(jobs) > expected:
        raise SubmissionManifestError(
            f"Submission TSV has {len(jobs)} jobs but expected {expected}"
        )
    return SubmissionManifest(
        source="legacy_tsv",
        expected=expected,
        jobs=_validate_unique_jobs(jobs),
    )


def load_submission_manifest(
    path: Path,
    *,
    expected_override: int | None = None,
) -> SubmissionManifest:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SubmissionManifestError(f"Submission manifest does not exist: {resolved}")
    if resolved.suffix.lower() == ".json":
        return _load_json_manifest(resolved, expected_override)
    if resolved.suffix.lower() in {".tsv", ".txt"}:
        return _load_tsv_manifest(resolved, expected_override)
    raise SubmissionManifestError(
        "Submission manifest must be canonical .json or legacy .tsv"
    )


def get_ml_client(subscription_id: str, resource_group: str, workspace_name: str):
    from utils.azure_helper import get_ml_client as create_ml_client

    return create_ml_client(
        subscription_id,
        resource_group,
        workspace_name,
    )


def poll_jobs(client: Any, jobs: Iterable[Submission]) -> list[PollResult]:
    results: list[PollResult] = []
    for submission in jobs:
        try:
            job = client.jobs.get(submission.job_id)
            status = str(job.status or "Unknown")
            results.append(
                PollResult(
                    label=submission.label,
                    job_id=submission.job_id,
                    status=status,
                )
            )
        except Exception as exc:  # noqa: BLE001 - query failure is evidence, not success
            results.append(
                PollResult(
                    label=submission.label,
                    job_id=submission.job_id,
                    status="QueryError",
                    query_error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results


def evaluate_monitor_state(
    manifest: SubmissionManifest,
    results: Iterable[PollResult],
) -> str:
    observed = tuple(results)
    if manifest.submission_failures:
        return "submission_failed"
    if len(manifest.jobs) < manifest.expected:
        return "waiting_for_submissions"
    if len(observed) != len(manifest.jobs):
        return "query_incomplete"
    if any(item.query_error for item in observed):
        return "query_error"
    if not all(item.status in TERMINAL_STATUSES for item in observed):
        return "running"
    if any(item.status in FAILURE_STATUSES for item in observed):
        return "failed"
    return "passed"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _append_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_monitor_summary(
    manifest: SubmissionManifest,
    results: Iterable[PollResult],
    *,
    state: str,
) -> dict[str, Any]:
    observed = tuple(results)
    counts: dict[str, int] = {}
    for item in observed:
        counts[item.status] = counts.get(item.status, 0) + 1
    return {
        "schema_version": "1.0",
        "observed_at_utc": utcnow(),
        "state": state,
        "manifest_source": manifest.source,
        "expected_count": manifest.expected,
        "accepted_submission_count": len(manifest.jobs),
        "submission_failures": list(manifest.submission_failures),
        "status_counts": counts,
        "jobs": [asdict(item) for item in observed],
    }


def _format_poll(summary: dict[str, Any]) -> list[str]:
    lines = [
        "=" * 72,
        f"POLL {summary['observed_at_utc']}",
        (
            f"state={summary['state']} "
            f"submitted={summary['accepted_submission_count']}/"
            f"{summary['expected_count']}"
        ),
    ]
    for item in summary["jobs"]:
        line = f"{item['label']}\t{item['job_id']}\t{item['status']}"
        if item.get("query_error"):
            line += f"\t{item['query_error']}"
        lines.append(line)
    if summary["submission_failures"]:
        lines.append(
            "submission_failures=" + ",".join(summary["submission_failures"])
        )
    return lines


def monitor(
    *,
    submissions_path: Path,
    expected_override: int | None,
    output_dir: Path,
    interval_seconds: float,
    max_seconds: float,
    once: bool,
    client_factory: Callable[[], Any],
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "monitor_status.log"
    summary_path = output_dir / "monitor-summary.json"
    done_path = output_dir / "BATCH_DONE.txt"
    failure_path = output_dir / "FAILURES.txt"
    timeout_path = output_dir / "BATCH_TIMEOUT.txt"
    for sentinel in (done_path, failure_path, timeout_path):
        sentinel.unlink(missing_ok=True)
    deadline = monotonic() + max_seconds
    client: Any | None = None

    while True:
        try:
            manifest = load_submission_manifest(
                submissions_path,
                expected_override=expected_override,
            )
        except SubmissionManifestError as exc:
            print(f"Submission manifest error: {exc}", file=sys.stderr)
            return 2

        if client is None:
            try:
                client = client_factory()
            except Exception as exc:  # noqa: BLE001 - authentication failure is terminal
                print(
                    f"Azure ML client initialization failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                return 2

        results = poll_jobs(client, manifest.jobs)
        state = evaluate_monitor_state(manifest, results)
        summary = build_monitor_summary(manifest, results, state=state)
        _write_json_atomic(summary_path, summary)
        poll_lines = _format_poll(summary)
        _append_log(log_path, poll_lines)
        print("\n".join(poll_lines), flush=True)

        if state == "passed":
            _write_text_atomic(
                done_path,
                f"BATCH_DONE {summary['observed_at_utc']}\n"
                f"Total:{len(results)} Completed:{len(results)} Failed:0\n",
            )
            return 0
        if state in {"submission_failed", "failed"}:
            failed_lines = [
                f"submission\t{label}\tSubmissionFailed"
                for label in manifest.submission_failures
            ]
            failed_lines.extend(
                f"job\t{item.job_id}\t{item.status}"
                for item in results
                if item.status in FAILURE_STATUSES
            )
            _write_text_atomic(failure_path, "\n".join(failed_lines) + "\n")
            _write_text_atomic(
                done_path,
                f"BATCH_DONE {summary['observed_at_utc']}\n"
                f"Total:{len(results)} Failed:{len(failed_lines)}\n",
            )
            return 1
        if once:
            return 4
        if monotonic() >= deadline:
            timeout_summary = dict(summary)
            timeout_summary["state"] = "timed_out"
            timeout_summary["timed_out_at_utc"] = utcnow()
            _write_json_atomic(summary_path, timeout_summary)
            _write_text_atomic(
                timeout_path,
                f"BATCH_TIMEOUT {timeout_summary['timed_out_at_utc']}\n"
                f"LastState:{state}\n",
            )
            print("Monitor timed out before authoritative terminal evidence", file=sys.stderr)
            return 3
        sleep(interval_seconds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor governed Azure ML qualification submissions"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--submissions",
        type=Path,
        help="Canonical batch_submit_all.py JSON or legacy TSV manifest",
    )
    source.add_argument(
        "--tsv",
        dest="submissions",
        type=Path,
        help="Deprecated alias for a legacy TSV manifest",
    )
    parser.add_argument(
        "--expected",
        type=int,
        help="Expected job count; canonical JSON already declares this value",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interval", type=float, default=15.0, help="Poll interval in minutes")
    parser.add_argument("--max-hours", "--max_hours", type=float, default=8.0)
    parser.add_argument("--once", action="store_true", help="Poll once and return 4 if nonterminal")
    parser.add_argument(
        "--sub",
        "--subscription-id",
        dest="subscription_id",
        default=os.environ.get("AZURE_SUBSCRIPTION_ID"),
    )
    parser.add_argument(
        "--rg",
        "--resource-group",
        dest="resource_group",
        default=os.environ.get("AZURE_RESOURCE_GROUP"),
    )
    parser.add_argument(
        "--ws",
        "--workspace-name",
        dest="workspace_name",
        default=os.environ.get("AZURE_WORKSPACE_NAME"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    missing = [
        name
        for name, value in (
            ("--sub/AZURE_SUBSCRIPTION_ID", args.subscription_id),
            ("--rg/AZURE_RESOURCE_GROUP", args.resource_group),
            ("--ws/AZURE_WORKSPACE_NAME", args.workspace_name),
        )
        if not value
    ]
    if missing:
        print(f"Missing Azure context: {', '.join(missing)}", file=sys.stderr)
        return 2
    if args.expected is not None and args.expected < 1:
        print("--expected must be positive", file=sys.stderr)
        return 2
    if args.interval < 0 or args.max_hours < 0:
        print("--interval and --max-hours cannot be negative", file=sys.stderr)
        return 2

    submissions_path = args.submissions.resolve()
    output_dir = (args.output_dir or submissions_path.parent).resolve()
    return monitor(
        submissions_path=submissions_path,
        expected_override=args.expected,
        output_dir=output_dir,
        interval_seconds=args.interval * 60,
        max_seconds=args.max_hours * 3600,
        once=args.once,
        client_factory=lambda: get_ml_client(
            args.subscription_id,
            args.resource_group,
            args.workspace_name,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
