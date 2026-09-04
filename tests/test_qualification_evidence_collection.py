from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import collect_qualification_evidence as collector


SOURCE_SHA = "b" * 64
SCENARIO = "classification-healthcare-heart-disease"
PARENT = "qualification-parent"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class FakeJobs:
    def get(self, name: str):
        assert name == PARENT
        return SimpleNamespace(
            status="Completed",
            tags={
                "source_identity": SOURCE_SHA,
                "qualification_scenario": SCENARIO,
                "qualification_matrix": "industry-qualification-20260902",
            },
        )

    def download(self, *, name, output_name, download_path):
        root = Path(download_path) / "named-outputs" / output_name
        payload = {"code_sha": SOURCE_SHA} if output_name == "execution_manifest" else {}
        if name == "registered-smoke":
            assert output_name == "evidence"
            payload = {"status": "passed"}
        _write(root / "evidence.json", payload)


def test_collect_wave_builds_relative_evidence_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "inputs"
    output = tmp_path / "wave"
    submission = source / "submission.json"
    monitor = output / "monitor" / "monitor-summary.json"
    audit = output / "azure-data-asset-audit.json"
    _write(
        submission,
        {
            "complete": True,
            "requested_count": 1,
            "accepted_count": 1,
            "git": {
                "commit": "a" * 40,
                "branch": "codex_ys/release-candidate",
                "dirty": False,
                "provenance": "verified_azure_archive",
                "archive_sha256": "c" * 64,
            },
            "submissions": [
                {
                    "scenario_id": SCENARIO,
                    "accepted": True,
                    "job": {"job_name": PARENT},
                }
            ],
        },
    )
    _write(monitor, {"state": "passed"})
    _write(audit, {"all_passed": True})

    def submit_smoke(arguments):
        result = Path(arguments[arguments.index("--result-json") + 1])
        _write(result, {"status": "Completed", "job_name": "registered-smoke"})
        return 0

    monkeypatch.setattr(collector, "submit_smoke", submit_smoke)
    manifest_path = collector.collect_wave(
        SimpleNamespace(jobs=FakeJobs()),
        submission_path=submission,
        monitor_path=monitor,
        asset_audit_path=audit,
        output_dir=output,
        environment="mlops-v3-unified:33",
        output_datastore="mlops_blob",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["release_candidate"] == {
        "git_commit": "a" * 40,
        "runtime_source_sha256": SOURCE_SHA,
    }
    assert manifest["source_archive_sha256"] == "c" * 64
    assert manifest["scenarios"][0]["parent_job"] == PARENT
    assert not Path(manifest["scenarios"][0]["pipeline_evidence_dir"]).is_absolute()
