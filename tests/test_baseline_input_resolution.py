from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from azure.ai.ml import Input, Output
from azure.ai.ml.entities import PipelineJob

from pipelines import submit_pipeline


JOB_URI = "azureml://jobs/baseline-job/outputs/drift_baseline/paths/"
DATASTORE_URI = "azureml://datastores/mlops_blob/paths/azureml/component/drift_baseline/"


def _job(path=JOB_URI):
    return PipelineJob(inputs={
        "drift_baseline_in": Input(type="uri_folder", path=path, mode="ro_mount"),
        "drift_baseline_uri": JOB_URI,
    })


def _client(output=None):
    producer = SimpleNamespace(
        name="baseline-job", status="Completed",
        outputs={"drift_baseline": output or Output(type="uri_folder")},
    )
    return SimpleNamespace(jobs=SimpleNamespace(
        get=Mock(return_value=producer),
        _get_named_output_uri=Mock(return_value={"drift_baseline": DATASTORE_URI}),
    ))


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("mapping", [False, True])
def test_resolves_storage_binding_without_changing_lineage_or_mode(declared, mapping):
    output = {"type": "uri_folder", "path": DATASTORE_URI if declared else None}
    client = _client(output if mapping else Output(**output))
    job = _job()

    submit_pipeline._resolve_drift_baseline_input(client, job)

    assert job.inputs["drift_baseline_in"].path == DATASTORE_URI
    assert job.inputs["drift_baseline_in"].mode == "ro_mount"
    assert job.inputs["drift_baseline_uri"] == JOB_URI
    client.jobs.get.assert_called_once_with("baseline-job")
    if declared:
        client.jobs._get_named_output_uri.assert_not_called()
    else:
        client.jobs._get_named_output_uri.assert_called_once_with(
            "baseline-job", output_names="drift_baseline",
        )


def test_accepts_workspace_qualified_datastore_output():
    path = (
        "azureml://subscriptions/sub/resourcegroups/rg/workspaces/ws/"
        "datastores/store/paths/baseline/"
    )
    client = _client(Output(type="uri_folder", path=path))
    job = _job()
    submit_pipeline._resolve_drift_baseline_input(client, job)
    assert job.inputs["drift_baseline_in"].path == path


@pytest.mark.parametrize("path", [DATASTORE_URI, "azureml:approved-baseline:1", "/tmp/baseline"])
def test_non_job_baselines_do_not_call_azure(path):
    client = _client()
    job = _job(path)
    submit_pipeline._resolve_drift_baseline_input(client, job)
    assert job.inputs["drift_baseline_in"].path == path
    client.jobs.get.assert_not_called()
    client.jobs._get_named_output_uri.assert_not_called()


def test_absent_baseline_does_not_call_azure():
    client = _client()
    submit_pipeline._resolve_drift_baseline_input(client, PipelineJob())
    client.jobs.get.assert_not_called()


@pytest.mark.parametrize("path", [
    "azureml://jobs/../outputs/drift_baseline/paths/",
    "azureml://jobs/baseline-job/outputs/other/paths/",
    JOB_URI + "subset/",
    JOB_URI + "?token=secret",
    JOB_URI + "#fragment",
])
def test_rejects_invalid_job_output_references(path):
    client = _client()
    with pytest.raises(ValueError, match="complete drift_baseline"):
        submit_pipeline._resolve_drift_baseline_input(client, _job(path))
    client.jobs.get.assert_not_called()


@pytest.mark.parametrize("status", ["Running", "Failed", "Canceled", None])
def test_rejects_incomplete_producer(status):
    client = _client()
    client.jobs.get.return_value.status = status
    with pytest.raises(ValueError, match="requested completed job"):
        submit_pipeline._resolve_drift_baseline_input(client, _job())
    client.jobs._get_named_output_uri.assert_not_called()


def test_rejects_mismatched_producer():
    client = _client()
    client.jobs.get.return_value.name = "another-job"
    with pytest.raises(ValueError, match="requested completed job"):
        submit_pipeline._resolve_drift_baseline_input(client, _job())


@pytest.mark.parametrize("outputs", [None, {}, {"drift_baseline": Output(type="uri_file")}])
def test_rejects_missing_or_wrong_output(outputs):
    client = _client()
    client.jobs.get.return_value.outputs = outputs
    with pytest.raises(ValueError, match="declare a uri_folder"):
        submit_pipeline._resolve_drift_baseline_input(client, _job())
    client.jobs._get_named_output_uri.assert_not_called()


def test_rejects_wrong_input_type():
    job = _job()
    job.inputs["drift_baseline_in"].type = "uri_file"
    with pytest.raises(ValueError, match="input must have type"):
        submit_pipeline._resolve_drift_baseline_input(_client(), job)


@pytest.mark.parametrize("locations", [
    None, {}, {"other": DATASTORE_URI}, {"drift_baseline": JOB_URI},
    {"drift_baseline": "https://storage/baseline?sig=secret"},
    {"drift_baseline": DATASTORE_URI + "?sig=secret"},
    {"drift_baseline": "azureml://datastores/store/paths/"},
])
def test_rejects_unresolved_or_non_datastore_output(locations):
    client = _client()
    client.jobs._get_named_output_uri.return_value = locations
    job = _job()
    with pytest.raises(ValueError, match="datastore URI"):
        submit_pipeline._resolve_drift_baseline_input(client, job)
    assert job.inputs["drift_baseline_in"].path == JOB_URI


def test_missing_sdk_resolver_fails_closed():
    client = _client()
    del client.jobs._get_named_output_uri
    with pytest.raises(RuntimeError, match="cannot resolve"):
        submit_pipeline._resolve_drift_baseline_input(client, _job())


def test_service_errors_propagate_without_guessing_a_path():
    client = _client()
    client.jobs._get_named_output_uri.side_effect = RuntimeError("service unavailable")
    job = _job()
    with pytest.raises(RuntimeError, match="service unavailable"):
        submit_pipeline._resolve_drift_baseline_input(client, job)
    assert job.inputs["drift_baseline_in"].path == JOB_URI


def test_canonical_entrypoint_resolves_after_guards_and_before_submission():
    source = inspect.getsource(submit_pipeline.main)
    binding = source.index("_resolve_drift_baseline_input(ml_client, job)")
    assert source.index("if args.dry_run:") < source.index("get_ml_client(") < binding
    assert source.index("_check_active_jobs(") < binding
    assert binding < source.index("ml_client.jobs.create_or_update(job)")
    assert source.count("_resolve_drift_baseline_input(ml_client, job)") == 1
