"""Regression tests for API reuse of the canonical pipeline submitter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _set_api_context(monkeypatch, pipeline_service) -> None:
    monkeypatch.setattr(
        pipeline_service.settings,
        "azure_subscription_id",
        "test-subscription",
    )
    monkeypatch.setattr(
        pipeline_service.settings,
        "azure_resource_group",
        "test-resource-group",
    )
    monkeypatch.setattr(
        pipeline_service.settings,
        "azure_workspace_name",
        "test-workspace",
    )
    monkeypatch.setattr(
        pipeline_service.settings,
        "compute_target",
        "default-compute",
    )


def test_api_submit_invokes_canonical_submitter_and_parses_result(
    monkeypatch,
) -> None:
    from api.schemas.pipeline import SubmitRequest
    from api.services import pipeline_service

    _set_api_context(monkeypatch, pipeline_service)
    monkeypatch.setattr(
        pipeline_service,
        "_load_config_yaml",
        lambda _config_name: {"task_type": "classification"},
    )
    monkeypatch.setattr(
        pipeline_service,
        "get_ml_client",
        lambda: pytest.fail("API must not construct an MLClient for normal submission"),
    )
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        result_path = Path(command[command.index("--result_json") + 1])
        experiment_name = command[command.index("--experiment_name") + 1]
        display_name = command[command.index("--display_name") + 1]
        result_path.write_text(
            json.dumps(
                {
                    "job_name": "canonical-job-123",
                    "experiment_name": experiment_name,
                    "display_name": display_name,
                    "status": "Submitted",
                    "studio_url": "https://ml.azure.com/runs/canonical-job-123",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="submitted", stderr="")

    monkeypatch.setattr(pipeline_service.subprocess, "run", fake_run)

    result = pipeline_service.submit_pipeline(
        SubmitRequest(
            config_name="config_classification_telecom_churn_azureml",
            compute="request-compute",
            force_rerun=True,
            tags={"request_id": "request-123"},
        )
    )

    command = captured["command"]
    assert command[0] == sys.executable
    assert Path(command[1]).name == "submit_pipeline.py"
    assert command[command.index("--compute") + 1] == "request-compute"
    assert command[command.index("--subscription_id") + 1] == "test-subscription"
    assert command[command.index("--resource_group") + 1] == "test-resource-group"
    assert command[command.index("--workspace_name") + 1] == "test-workspace"
    assert "--force_rerun" in command
    assert "--force" not in command
    assert json.loads(command[command.index("--tags_json") + 1]) == {
        "request_id": "request-123",
        "source": "api",
    }
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["check"] is False
    assert result.job_name == "canonical-job-123"
    assert result.status == "Submitted"


def test_api_submit_fails_when_canonical_submitter_fails(monkeypatch) -> None:
    from api.schemas.pipeline import SubmitRequest
    from api.services import pipeline_service

    _set_api_context(monkeypatch, pipeline_service)
    monkeypatch.setattr(
        pipeline_service,
        "_load_config_yaml",
        lambda _config_name: {"task_type": "classification"},
    )
    monkeypatch.setattr(
        pipeline_service.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            3,
            stdout="",
            stderr="active-job state unavailable; refusing submission",
        ),
    )

    with pytest.raises(RuntimeError, match="refusing submission"):
        pipeline_service.submit_pipeline(
            SubmitRequest(
                config_name="config_classification_telecom_churn_azureml"
            )
        )


def test_api_submit_rejects_missing_structured_result(monkeypatch) -> None:
    from api.schemas.pipeline import SubmitRequest
    from api.services import pipeline_service

    _set_api_context(monkeypatch, pipeline_service)
    monkeypatch.setattr(
        pipeline_service,
        "_load_config_yaml",
        lambda _config_name: {"task_type": "classification"},
    )
    monkeypatch.setattr(
        pipeline_service.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="submitted without result",
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="Submission state is unknown"):
        pipeline_service.submit_pipeline(
            SubmitRequest(
                config_name="config_classification_telecom_churn_azureml"
            )
        )


def test_canonical_active_job_query_fails_closed(monkeypatch) -> None:
    from pipelines import submit_pipeline

    class BrokenJobs:
        def list(self):
            raise ConnectionError("control plane unavailable")

    class FakeClient:
        jobs = BrokenJobs()

    monkeypatch.setattr(submit_pipeline.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="refusing submission"):
        submit_pipeline._check_active_jobs(FakeClient(), "test-experiment")


def test_canonical_active_job_query_uses_raw_resources() -> None:
    from types import SimpleNamespace

    from pipelines import submit_pipeline

    resources = [
        SimpleNamespace(
            name="completed-job",
            properties=SimpleNamespace(
                experiment_name="test-experiment",
                status="Completed",
                display_name="completed",
            ),
        ),
        SimpleNamespace(
            name="running-job",
            properties=SimpleNamespace(
                experiment_name="test-experiment",
                status="Running",
                display_name="running",
            ),
        ),
        SimpleNamespace(
            name="other-job",
            properties=SimpleNamespace(
                experiment_name="other-experiment",
                status="Running",
                display_name="other",
            ),
        ),
        SimpleNamespace(
            name="finalizing-job",
            properties=SimpleNamespace(
                experiment_name="test-experiment",
                status="Finalizing",
                display_name="finalizing",
            ),
        ),
    ]

    class RawJobs:
        def list(self, resource_group, workspace):
            assert resource_group == "mvpv1"
            assert workspace == "mlops-accelerator"
            return iter(resources)

    class PublicJobs:
        service_client_01_2024_preview = SimpleNamespace(jobs=RawJobs())
        _operation_scope = SimpleNamespace(resource_group_name="mvpv1")
        _workspace_name = "mlops-accelerator"

        def list(self):
            raise AssertionError("full SDK job hydration must not be used")

    active = submit_pipeline._check_active_jobs(
        SimpleNamespace(jobs=PublicJobs()),
        "test-experiment",
    )

    assert active == [
        {
            "name": "running-job",
            "status": "Running",
            "display_name": "running",
        },
        {
            "name": "finalizing-job",
            "status": "Finalizing",
            "display_name": "finalizing",
        },
    ]


def test_canonical_active_job_query_uses_bounded_run_history() -> None:
    from types import SimpleNamespace

    from pipelines import submit_pipeline

    calls = []

    class RunHistoryOperation:
        def get_by_query_by_experiment_name(
            self,
            subscription_id,
            resource_group,
            workspace,
            experiment,
            *,
            body,
            connection_timeout,
            read_timeout,
        ):
            calls.append(body.continuation_token)
            assert subscription_id == "subscription-id"
            assert resource_group == "mvpv1"
            assert workspace == "mlops-accelerator"
            assert experiment == "test-experiment"
            assert body.top == 100
            assert "Status ne 'Completed'" in body.filter
            assert connection_timeout == 10
            assert read_timeout == 30
            if body.continuation_token is None:
                return SimpleNamespace(
                    value=[
                        SimpleNamespace(
                            run_id="completed-job",
                            status="Completed",
                            display_name="completed",
                        ),
                        SimpleNamespace(
                            run_id="running-job",
                            status="Running",
                            display_name="running",
                        ),
                    ],
                    continuation_token="next-page",
                )
            assert body.continuation_token == "next-page"
            return SimpleNamespace(
                value=[
                    SimpleNamespace(
                        run_id="finalizing-job",
                        status="Finalizing",
                        display_name="finalizing",
                    ),
                    SimpleNamespace(
                        run_id="failed-job",
                        status="Failed",
                        display_name="failed",
                    ),
                ],
                continuation_token=None,
            )

    class RawJobs:
        def list(self, *_args, **_kwargs):
            raise AssertionError("workspace-wide job listing must not be used")

    jobs = SimpleNamespace(
        _runs_operations=SimpleNamespace(_operation=RunHistoryOperation()),
        _subscription_id="subscription-id",
        _operation_scope=SimpleNamespace(resource_group_name="mvpv1"),
        _workspace_name="mlops-accelerator",
        service_client_01_2024_preview=SimpleNamespace(jobs=RawJobs()),
    )

    active = submit_pipeline._check_active_jobs(
        SimpleNamespace(jobs=jobs),
        "test-experiment",
    )

    assert calls == [None, "next-page"]
    assert active == [
        {
            "name": "running-job",
            "status": "Running",
            "display_name": "running",
        },
        {
            "name": "finalizing-job",
            "status": "Finalizing",
            "display_name": "finalizing",
        },
    ]


def test_canonical_active_job_query_allows_missing_experiment() -> None:
    from types import SimpleNamespace

    from azure.core.exceptions import ResourceNotFoundError

    from pipelines import submit_pipeline

    class MissingExperimentOperation:
        def get_by_query_by_experiment_name(
            self,
            _subscription_id,
            _resource_group,
            workspace,
            experiment,
            **_kwargs,
        ):
            raise ResourceNotFoundError(
                message=f"Experiment {experiment} not found in workspace {workspace}"
            )

    jobs = SimpleNamespace(
        _runs_operations=SimpleNamespace(
            _operation=MissingExperimentOperation()
        ),
        _subscription_id="subscription-id",
        _operation_scope=SimpleNamespace(resource_group_name="mvpv1"),
        _workspace_name="mlops-accelerator",
    )

    assert submit_pipeline._check_active_jobs(
        SimpleNamespace(jobs=jobs),
        "new-experiment",
    ) == []


def test_canonical_active_job_query_retries_transient_failure(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from pipelines import submit_pipeline

    calls = 0

    class FlakyJobs:
        def list(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("transient reset")
            return iter(
                [
                    SimpleNamespace(
                        name="running-job",
                        experiment_name="test-experiment",
                        status="Running",
                        display_name="running",
                    )
                ]
            )

    monkeypatch.setattr(submit_pipeline.time, "sleep", lambda _seconds: None)
    active = submit_pipeline._check_active_jobs(
        SimpleNamespace(jobs=FlakyJobs()),
        "test-experiment",
    )

    assert calls == 2
    assert active[0]["name"] == "running-job"


def test_canonical_active_job_query_does_not_retry_permanent_failure(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from pipelines import submit_pipeline

    calls = 0

    class ForbiddenJobs:
        def list(self):
            nonlocal calls
            calls += 1
            raise PermissionError("forbidden")

    monkeypatch.setattr(
        submit_pipeline.time,
        "sleep",
        lambda _seconds: pytest.fail("permanent failure must not be retried"),
    )
    with pytest.raises(RuntimeError, match="refusing submission"):
        submit_pipeline._check_active_jobs(
            SimpleNamespace(jobs=ForbiddenJobs()),
            "test-experiment",
        )

    assert calls == 1


def test_canonical_result_writer_emits_structured_json(tmp_path) -> None:
    from pipelines import submit_pipeline

    result_path = tmp_path / "submission.json"
    payload = {
        "job_name": "job-123",
        "experiment_name": "experiment",
        "display_name": "display",
        "status": "Submitted",
        "studio_url": "https://ml.azure.com/runs/job-123",
    }

    submit_pipeline._write_submission_result(str(result_path), payload)

    assert json.loads(result_path.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize(
    "protected_tag",
    (
        "source",
        "dataset",
        "task",
        "preset",
        "pipeline_version",
        "environment",
    ),
)
def test_api_rejects_protected_submission_tags(
    monkeypatch,
    tmp_path,
    protected_tag,
) -> None:
    from api.schemas.pipeline import SubmitRequest
    from api.services import pipeline_service

    _set_api_context(monkeypatch, pipeline_service)

    with pytest.raises(ValueError, match="protected submission metadata"):
        pipeline_service._build_canonical_submit_command(
            SubmitRequest(
                config_name="config_classification_telecom_churn_azureml",
                tags={protected_tag: "forged"},
            ),
            result_path=tmp_path / "result.json",
            baseline_uri=None,
        )


def test_canonical_submitter_protects_lineage_tag_keys() -> None:
    from pipelines import submit_pipeline

    assert submit_pipeline._CANONICAL_TAG_KEYS == {
        "compiled_config_hash",
        "config_name",
        "dataset",
        "environment",
        "execution_id",
        "parent_config_hash",
        "parent_execution_id",
        "parent_source_identity",
        "pipeline_version",
        "preset",
        "recipe_catalog_hash",
        "revision_reason",
        "source_decision_id",
        "source_identity",
        "submission_revision_kind",
        "task",
    }


def test_canonical_submitter_direct_cli_bootstraps_import_roots() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "pipelines/submit_pipeline.py",
            "--help",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--dry_run" in result.stdout


def test_upload_identity_fails_closed_for_included_symlink(tmp_path) -> None:
    from pipelines import submit_pipeline

    (tmp_path / ".amlignore").write_text("", encoding="utf-8")
    target = tmp_path / "source.py"
    target.write_text("print('exact bytes')\n", encoding="utf-8")
    link = tmp_path / "linked.py"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")

    with pytest.raises(RuntimeError, match="symlinks are unsupported"):
        submit_pipeline._compute_upload_source_manifest(tmp_path)
