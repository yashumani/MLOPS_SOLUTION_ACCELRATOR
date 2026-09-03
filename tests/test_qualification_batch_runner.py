from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "batch_submit_all.py"


def _load_module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("qualification_batch_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeSchedules:
    def __init__(self, *, enabled_name: str | None = None) -> None:
        self.enabled_name = enabled_name

    def get(self, name: str) -> SimpleNamespace:
        return SimpleNamespace(
            is_enabled=name == self.enabled_name,
            provisioning_state="Succeeded",
        )


class _FakeJobs:
    def __init__(
        self,
        *,
        marker_created_at: datetime,
        status: str = "Completed",
        write_artifact: bool = True,
        write_marker: bool = True,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.marker_created_at = marker_created_at
        self.status = status
        self.write_artifact = write_artifact
        self.write_marker = write_marker
        self.tags = tags or {
            "evidence_scope": "platform-recovery",
            "shared_datastore_change_required": "true",
        }
        self.download_calls: list[str | None] = []

    def get(self, _name: str) -> SimpleNamespace:
        return SimpleNamespace(status=self.status, tags=self.tags)

    def download(
        self,
        _name: str,
        *,
        download_path: Path,
        output_name: str | None = None,
    ) -> None:
        self.download_calls.append(output_name)
        root = Path(download_path)
        if output_name is None and self.write_artifact:
            artifact = root / "artifacts" / "logs" / "user_log.txt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("canary completed", encoding="utf-8")
        if output_name == "probe" and self.write_marker:
            marker = root / "named-outputs" / "probe" / "workspace_datastore_probe.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "status": "workspace_datastore_write_succeeded",
                        "created_at": self.marker_created_at.isoformat(),
                        "run_id": "datastore-canary",
                    }
                ),
                encoding="utf-8",
            )


def _fake_client(
    *,
    jobs: _FakeJobs,
    enabled_name: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        schedules=_FakeSchedules(enabled_name=enabled_name),
        jobs=jobs,
    )


def test_release_catalog_has_exact_five_industries_per_task() -> None:
    module = _load_module()
    scenarios = module.load_execution_catalog()

    assert len(scenarios) == 15
    for task_type in module.TASK_TYPES:
        selected = [item for item in scenarios if item.task_type == task_type]
        assert len(selected) == 5
        assert len({item.industry for item in selected}) == 5


def test_selection_rejects_unknown_scenario() -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="Unknown qualification scenarios"):
        module.select_scenarios(
            module.load_execution_catalog(),
            scenario_ids={"missing-scenario"},
        )


def test_submission_command_preserves_canonical_guards(tmp_path: Path) -> None:
    module = _load_module()
    scenario = module.load_execution_catalog()[0]
    context = SimpleNamespace(
        as_cli_args=lambda: [
            "--subscription_id",
            "sub",
            "--resource_group",
            "rg",
            "--workspace_name",
            "ws",
            "--compute",
            "cluster",
        ]
    )

    command = module.build_submission_command(
        scenario,
        result_path=tmp_path / "result.json",
        context=context,
    )

    assert str(module.SUBMITTER) in command
    assert "--force" not in command
    assert "--force_rerun" not in command
    assert "--result_json" in command
    assert "--tags_json" in command


def test_execute_requires_datastore_canary_job() -> None:
    module = _load_module()
    scenario_id = module.load_execution_catalog()[0].scenario_id

    assert module.main(["--execute", "--scenario", scenario_id]) == 2


def test_live_release_gates_verify_schedules_and_both_datastores(
    tmp_path: Path,
) -> None:
    module = _load_module()
    now = datetime(2026, 9, 3, 17, 30, tzinfo=timezone.utc)
    jobs = _FakeJobs(marker_created_at=now - timedelta(minutes=2))

    evidence = module.verify_live_release_gates(
        _fake_client(jobs=jobs),
        datastore_canary_job="datastore-canary",
        download_root=tmp_path,
        now_utc=now,
    )

    assert evidence["state"] == "passed"
    assert all(
        item["is_enabled"] is False for item in evidence["legacy_schedules"]
    )
    assert evidence["datastore_canary"]["status"] == "Completed"
    assert evidence["datastore_canary"]["default_artifact_file_count"] == 1
    assert jobs.download_calls == [None, "probe"]


def test_live_release_gates_reject_enabled_legacy_schedule(tmp_path: Path) -> None:
    module = _load_module()
    now = datetime(2026, 9, 3, 17, 30, tzinfo=timezone.utc)
    jobs = _FakeJobs(marker_created_at=now)

    with pytest.raises(module.ReleaseGateError, match="is_enabled=True"):
        module.verify_live_release_gates(
            _fake_client(
                jobs=jobs,
                enabled_name=module.LEGACY_SCHEDULE_NAMES[0],
            ),
            datastore_canary_job="datastore-canary",
            download_root=tmp_path,
            now_utc=now,
        )

    assert jobs.download_calls == []


def test_live_release_gates_reject_incomplete_canary(tmp_path: Path) -> None:
    module = _load_module()
    now = datetime(2026, 9, 3, 17, 30, tzinfo=timezone.utc)

    with pytest.raises(module.ReleaseGateError, match="not 'Completed'"):
        module.verify_live_release_gates(
            _fake_client(jobs=_FakeJobs(marker_created_at=now, status="Failed")),
            datastore_canary_job="datastore-canary",
            download_root=tmp_path,
            now_utc=now,
        )


def test_live_release_gates_reject_wrong_canary_identity(tmp_path: Path) -> None:
    module = _load_module()
    now = datetime(2026, 9, 3, 17, 30, tzinfo=timezone.utc)

    with pytest.raises(module.ReleaseGateError, match="identity tags"):
        module.verify_live_release_gates(
            _fake_client(
                jobs=_FakeJobs(
                    marker_created_at=now,
                    tags={"evidence_scope": "unrelated-job"},
                )
            ),
            datastore_canary_job="datastore-canary",
            download_root=tmp_path,
            now_utc=now,
        )


def test_live_release_gates_require_default_artifact_download(tmp_path: Path) -> None:
    module = _load_module()
    now = datetime(2026, 9, 3, 17, 30, tzinfo=timezone.utc)

    with pytest.raises(module.ReleaseGateError, match="returned no files"):
        module.verify_live_release_gates(
            _fake_client(
                jobs=_FakeJobs(marker_created_at=now, write_artifact=False)
            ),
            datastore_canary_job="datastore-canary",
            download_root=tmp_path,
            now_utc=now,
        )


def test_live_release_gates_require_probe_output_marker(tmp_path: Path) -> None:
    module = _load_module()
    now = datetime(2026, 9, 3, 17, 30, tzinfo=timezone.utc)

    with pytest.raises(module.ReleaseGateError, match="found 0"):
        module.verify_live_release_gates(
            _fake_client(
                jobs=_FakeJobs(marker_created_at=now, write_marker=False)
            ),
            datastore_canary_job="datastore-canary",
            download_root=tmp_path,
            now_utc=now,
        )


def test_live_release_gates_require_fresh_probe_output(tmp_path: Path) -> None:
    module = _load_module()
    now = datetime(2026, 9, 3, 17, 30, tzinfo=timezone.utc)

    with pytest.raises(module.ReleaseGateError, match="stale"):
        module.verify_live_release_gates(
            _fake_client(
                jobs=_FakeJobs(marker_created_at=now - timedelta(hours=25))
            ),
            datastore_canary_job="datastore-canary",
            download_root=tmp_path,
            now_utc=now,
        )


def test_main_records_release_gate_failure_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    scenario_id = module.load_execution_catalog()[0].scenario_id
    now = datetime.now(timezone.utc)
    jobs = _FakeJobs(marker_created_at=now)
    client = _fake_client(
        jobs=jobs,
        enabled_name=module.LEGACY_SCHEDULE_NAMES[0],
    )
    context = SimpleNamespace(
        subscription_id="sub",
        resource_group="rg",
        workspace_name="ws",
        compute="cluster",
    )
    monkeypatch.setattr(module, "load_azure_context", lambda: context)
    monkeypatch.setattr(
        module,
        "_git_identity",
        lambda: {"commit": "a" * 40, "branch": "feature/test", "dirty": False},
    )
    monkeypatch.setattr(module, "_create_ml_client", lambda _context: client)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "canonical submitter ran before release gates passed"
        ),
    )
    result_path = tmp_path / "submission.json"

    exit_code = module.main(
        [
            "--execute",
            "--scenario",
            scenario_id,
            "--datastore-canary-job",
            "datastore-canary",
            "--result-json",
            str(result_path),
        ]
    )

    evidence = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert evidence["release_gates"]["state"] == "blocked"
    assert evidence["accepted_count"] == 0
    assert evidence["submissions"] == []


def test_shell_entrypoints_delegate_without_force_bypass() -> None:
    for name in (
        "submit_all_15.sh",
        "submit_15_parallel.sh",
        "submit_all_dryrun.sh",
    ):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "batch_submit_all.py" in source
        assert "--force" not in source
        assert "config_classification_cardiac_arrest" not in source

    dry_run = (ROOT / "scripts" / "submit_all_dryrun.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ "$argument" == "--execute" ]]' in dry_run
