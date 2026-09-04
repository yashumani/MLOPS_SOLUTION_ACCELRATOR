from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import merge_qualification_waves as merger


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _wave(root: Path, index: int, *, commit: str = "a" * 40) -> Path:
    wave = root / f"wave-{index:02d}"
    _write(wave / "monitor.json", {"state": "passed"})
    _write(
        wave / "asset-audit.json",
        {
            "schema_version": "1.0",
            "asset_version": "1",
            "asset_count": 11,
            "scenario_count": 15,
            "all_passed": True,
            "errors": [],
            "assets": [{"identity": "shared"}],
        },
    )
    (wave / "scenario" / "pipeline").mkdir(parents=True)
    _write(wave / "scenario" / "smoke.json", {"status": "Completed"})
    (wave / "scenario" / "evidence").mkdir()
    _write(
        wave / "qualification-evidence-manifest.json",
        {
            "schema_version": "1.0",
            "monitor_summaries": ["monitor.json"],
            "data_asset_audit": "asset-audit.json",
            "release_candidate": {
                "git_commit": commit,
                "runtime_source_sha256": "b" * 64,
            },
            "source_archive_sha256": "c" * 64,
            "scenarios": [
                {
                    "scenario_id": f"scenario-{index:02d}",
                    "parent_job": f"parent-{index:02d}",
                    "pipeline_evidence_dir": "scenario/pipeline",
                    "registered_model_smoke_submission": "scenario/smoke.json",
                    "registered_model_smoke_evidence": "scenario/evidence",
                }
            ],
        },
    )
    return wave


def test_merge_builds_one_relative_complete_matrix_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waves = [_wave(tmp_path, index) for index in range(1, 16)]

    def verify(path: Path, *, require_complete_matrix: bool = False):
        if require_complete_matrix:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert len(payload["scenarios"]) == 15
            assert all(
                not Path(item["pipeline_evidence_dir"]).is_absolute()
                for item in payload["scenarios"]
            )
            return {"state": "passed", "release_matrix_accepted": True}
        return {"state": "passed"}

    monkeypatch.setattr(merger, "verify_qualification_evidence", verify)

    manifest, report = merger.merge_waves(waves, tmp_path / "combined")

    assert manifest.is_file()
    assert report.is_file()
    assert json.loads(report.read_text(encoding="utf-8"))[
        "release_matrix_accepted"
    ] is True


def test_merge_rejects_mixed_release_commits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waves = [_wave(tmp_path, index) for index in range(1, 16)]
    payload = json.loads(
        (waves[-1] / "qualification-evidence-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    payload["release_candidate"]["git_commit"] = "d" * 40
    _write(waves[-1] / "qualification-evidence-manifest.json", payload)
    monkeypatch.setattr(
        merger,
        "verify_qualification_evidence",
        lambda *_args, **_kwargs: {"state": "passed"},
    )

    with pytest.raises(RuntimeError, match="one frozen source identity"):
        merger.merge_waves(waves, tmp_path / "combined")
