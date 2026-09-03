from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "monitor_batch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("qualification_batch_monitor", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, submissions: list[dict], *, requested: int = 2) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "requested_count": requested,
                "submissions": submissions,
            }
        ),
        encoding="utf-8",
    )


def _accepted(scenario: str, job_name: str) -> dict:
    return {
        "scenario_id": scenario,
        "accepted": True,
        "job": {"job_name": job_name},
    }


class _Jobs:
    def __init__(self, statuses: dict[str, object]) -> None:
        self._statuses = statuses

    def get(self, job_id: str):
        value = self._statuses[job_id]
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(status=value)


class _Client:
    def __init__(self, statuses: dict[str, object]) -> None:
        self.jobs = _Jobs(statuses)


def test_loads_partial_canonical_manifest_without_claiming_completion(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = tmp_path / "wave.json"
    _write_manifest(path, [_accepted("scenario-a", "job-a")])

    manifest = module.load_submission_manifest(path)
    results = module.poll_jobs(_Client({"job-a": "Completed"}), manifest.jobs)

    assert manifest.source == "canonical_json"
    assert manifest.expected == 2
    assert [item.job_id for item in manifest.jobs] == ["job-a"]
    assert module.evaluate_monitor_state(manifest, results) == "waiting_for_submissions"


def test_canonical_manifest_preserves_submission_failure(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "wave.json"
    _write_manifest(
        path,
        [
            _accepted("scenario-a", "job-a"),
            {
                "scenario_id": "scenario-b",
                "accepted": False,
                "submit_exit_code": 1,
                "job": {},
            },
        ],
    )

    manifest = module.load_submission_manifest(path)
    results = module.poll_jobs(_Client({"job-a": "Running"}), manifest.jobs)

    assert manifest.submission_failures == ("scenario-b",)
    assert module.evaluate_monitor_state(manifest, results) == "submission_failed"


def test_rejects_accepted_submission_without_job_name(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "wave.json"
    _write_manifest(
        path,
        [{"scenario_id": "scenario-a", "accepted": True, "job": {}}],
        requested=1,
    )

    with pytest.raises(module.SubmissionManifestError, match="job.job_name"):
        module.load_submission_manifest(path)


def test_rejects_duplicate_job_ids(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "wave.json"
    _write_manifest(
        path,
        [
            _accepted("scenario-a", "same-job"),
            _accepted("scenario-b", "same-job"),
        ],
    )

    with pytest.raises(module.SubmissionManifestError, match="duplicate job IDs"):
        module.load_submission_manifest(path)


def test_rejects_expected_override_that_disagrees_with_json(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "wave.json"
    _write_manifest(path, [_accepted("scenario-a", "job-a")], requested=1)

    with pytest.raises(module.SubmissionManifestError, match="does not match"):
        module.load_submission_manifest(path, expected_override=2)


@pytest.mark.parametrize(
    ("status", "expected_state"),
    [
        ("Completed", "passed"),
        ("Running", "running"),
        ("Failed", "failed"),
        ("Canceled", "failed"),
        ("NotResponding", "failed"),
    ],
)
def test_evaluates_authoritative_job_statuses(
    tmp_path: Path,
    status: str,
    expected_state: str,
) -> None:
    module = _load_module()
    path = tmp_path / "wave.json"
    _write_manifest(path, [_accepted("scenario-a", "job-a")], requested=1)
    manifest = module.load_submission_manifest(path)
    results = module.poll_jobs(_Client({"job-a": status}), manifest.jobs)

    assert module.evaluate_monitor_state(manifest, results) == expected_state


def test_query_error_is_not_treated_as_terminal(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "wave.json"
    _write_manifest(path, [_accepted("scenario-a", "job-a")], requested=1)
    manifest = module.load_submission_manifest(path)
    results = module.poll_jobs(
        _Client({"job-a": ConnectionError("temporary reset")}),
        manifest.jobs,
    )

    assert results[0].status == "QueryError"
    assert "temporary reset" in results[0].query_error
    assert module.evaluate_monitor_state(manifest, results) == "query_error"


def test_monitor_returns_nonzero_and_writes_failure_evidence(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "wave.json"
    output = tmp_path / "evidence"
    _write_manifest(path, [_accepted("scenario-a", "job-a")], requested=1)

    exit_code = module.monitor(
        submissions_path=path,
        expected_override=None,
        output_dir=output,
        interval_seconds=0,
        max_seconds=1,
        once=False,
        client_factory=lambda: _Client({"job-a": "NotResponding"}),
    )

    assert exit_code == 1
    assert "NotResponding" in (output / "FAILURES.txt").read_text(encoding="utf-8")
    summary = json.loads((output / "monitor-summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "failed"


def test_monitor_timeout_returns_nonzero_and_records_last_state(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "wave.json"
    output = tmp_path / "evidence"
    _write_manifest(path, [_accepted("scenario-a", "job-a")], requested=1)

    exit_code = module.monitor(
        submissions_path=path,
        expected_override=None,
        output_dir=output,
        interval_seconds=0,
        max_seconds=0,
        once=False,
        client_factory=lambda: _Client({"job-a": "Running"}),
    )

    assert exit_code == 3
    assert "LastState:running" in (output / "BATCH_TIMEOUT.txt").read_text(
        encoding="utf-8"
    )
    summary = json.loads((output / "monitor-summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "timed_out"


def test_monitor_removes_stale_terminal_sentinels_before_polling(
    tmp_path: Path,
) -> None:
    module = _load_module()
    path = tmp_path / "wave.json"
    output = tmp_path / "evidence"
    output.mkdir()
    _write_manifest(path, [_accepted("scenario-a", "job-a")], requested=1)
    for name in ("BATCH_DONE.txt", "FAILURES.txt", "BATCH_TIMEOUT.txt"):
        (output / name).write_text("stale\n", encoding="utf-8")

    exit_code = module.monitor(
        submissions_path=path,
        expected_override=None,
        output_dir=output,
        interval_seconds=0,
        max_seconds=1,
        once=True,
        client_factory=lambda: _Client({"job-a": "Running"}),
    )

    assert exit_code == 4
    for name in ("BATCH_DONE.txt", "FAILURES.txt", "BATCH_TIMEOUT.txt"):
        assert not (output / name).exists()


def test_monitor_passes_only_after_all_expected_jobs_complete(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "wave.json"
    output = tmp_path / "evidence"
    _write_manifest(
        path,
        [
            _accepted("scenario-a", "job-a"),
            _accepted("scenario-b", "job-b"),
        ],
    )

    exit_code = module.monitor(
        submissions_path=path,
        expected_override=None,
        output_dir=output,
        interval_seconds=0,
        max_seconds=1,
        once=False,
        client_factory=lambda: _Client(
            {"job-a": "Completed", "job-b": "Completed"}
        ),
    )

    assert exit_code == 0
    assert "Failed:0" in (output / "BATCH_DONE.txt").read_text(encoding="utf-8")
    summary = json.loads((output / "monitor-summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "passed"


def test_legacy_tsv_remains_supported(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "submissions.tsv"
    path.write_text(
        "wave\tconfig\tjob_id\n1\tconfig-a.yml\tjob-a\n",
        encoding="utf-8",
    )

    manifest = module.load_submission_manifest(path, expected_override=1)

    assert manifest.source == "legacy_tsv"
    assert manifest.jobs[0].label == "config-a.yml"
    assert manifest.jobs[0].job_id == "job-a"
