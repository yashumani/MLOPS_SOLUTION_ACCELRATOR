"""Offline regression tests for race-free model registration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from sklearn.dummy import DummyClassifier

import src.steps.s12_model_registration as s12
from src.orchestration.contracts import ExecutionManifest
from src.utils.model_bundle import ModelBundle, save_model_bundle
from utils.model_bundle import ModelBundle as RuntimeModelBundle


class _RegistryClient:
    def __init__(self, search_results=None):
        self.search_results = list(search_results or [])
        self.search_calls = []
        self.tag_versions = []
        self.tags = {}
        self.transition_versions = []

    def get_latest_versions(self, *_args, **_kwargs):
        raise AssertionError("global latest-version lookup must never be used")

    def search_model_versions(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_results

    def set_model_version_tag(self, *, version, key, value, **_kwargs):
        self.tag_versions.append(str(version))
        self.tags[key] = value

    def transition_model_version_stage(self, *, version, **_kwargs):
        self.transition_versions.append(str(version))


class _MetricsLogger:
    def log_metric(self, *_args, **_kwargs):
        pass

    def log_param(self, *_args, **_kwargs):
        pass

    def end_run(self):
        pass


def _registry(client):
    registry = s12.ModelRegistry.__new__(s12.ModelRegistry)
    registry.config_name = "config_classification_test.yml"
    registry.cfg = {}
    registry.dataset_name = "test_dataset"
    registry.model_name_override = None
    registry.client = client
    return registry


def _register(monkeypatch, tmp_path, client, log_result):
    execution_manifest = ExecutionManifest(
        config_hash="config-hash",
        task_type="classification",
        dataset={"name": "test", "version": "1"},
        split_policy={"strategy": "random"},
        engines=("pycaret",),
        recipe_paths=("classification/test.yml",),
        recipe_ids=("recipe-1",),
        candidate_ids=("candidate-1",),
        budgets={"round1_max_variants": 1},
        code_sha="code-sha",
        environment_hashes={"training": "environment-hash"},
        recipe_catalog_hash="catalog-hash",
    )
    estimator = DummyClassifier(strategy="most_frequent").fit(
        [[0.0], [1.0]],
        [0, 1],
    )
    save_model_bundle(
        ModelBundle(
            estimator=estimator,
            task_type="classification",
            candidate_id="candidate-1",
            input_schema={
                "columns": [{"name": "feature", "dtype": "float64"}],
                "column_order": ["feature"],
            },
            lineage={
                "execution_id": execution_manifest.execution_id,
                "config_hash": execution_manifest.config_hash,
                "code_sha": execution_manifest.code_sha,
                "parent_run_id": "parent-1",
                "candidate_run_id": "candidate-run-1",
            },
        ),
        tmp_path,
    )
    active_run = SimpleNamespace(info=SimpleNamespace(run_id="owned-run-id"))

    def _log_model(*_args, **_kwargs):
        if isinstance(log_result, Exception):
            raise log_result
        return log_result

    monkeypatch.setattr(s12.mlflow, "active_run", lambda: active_run)
    monkeypatch.setattr(s12, "_log_exact_model_bundle", _log_model)

    return _registry(client).register_champion_model(
        {
            "task_type": "classification",
            "algorithm": "RandomForestClassifier",
            "metrics": {"accuracy": 0.91},
            "quality_decision": {"decision": "pass"},
            "execution_manifest": execution_manifest.to_dict(),
            "lineage": {
                "execution_id": execution_manifest.execution_id,
                "config_hash": execution_manifest.config_hash,
                "code_sha": execution_manifest.code_sha,
            },
        },
        tmp_path,
        execution_manifest,
    )


def test_exact_bundle_logging_infers_signature_and_uses_raw_example(
    monkeypatch,
) -> None:
    raw = pd.DataFrame({"feature": [0.0, 1.0]})
    estimator = DummyClassifier(strategy="most_frequent").fit(
        raw,
        ["stay", "churn"],
    )
    bundle = ModelBundle(
        estimator=estimator,
        task_type="classification",
        candidate_id="candidate-signature",
        input_schema={
            "columns": [{"name": "feature", "dtype": "float64"}],
            "column_order": ["feature"],
        },
        input_example=[{"feature": 0.0}, {"feature": 1.0}],
    )
    captured = {}

    def fake_log_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(model_uri="models:/signed/1")

    monkeypatch.setattr(s12, "_MLFLOW_SKLEARN_LOG_MODEL", fake_log_model)

    result = s12._log_exact_model_bundle(bundle, "signed")

    assert result.model_uri == "models:/signed/1"
    assert captured["sk_model"] is bundle
    assert captured["registered_model_name"] == "signed"
    assert captured["signature"] is not None
    assert captured["serialization_format"] == "cloudpickle"
    assert len(captured["code_paths"]) == 1
    code_path = Path(captured["code_paths"][0])
    assert code_path.name == "utils"
    assert (code_path / "model_bundle.py").is_file()
    assert captured["input_example"].equals(
        pd.DataFrame([{"feature": 0.0}, {"feature": 1.0}])
    )
    assert set(bundle.predict(captured["input_example"])).issubset(
        {"stay", "churn"}
    )


def test_registration_uses_version_returned_by_log_model(monkeypatch, tmp_path) -> None:
    client = _RegistryClient(
        search_results=[
            SimpleNamespace(
                name="test_dataset_classification_mlops",
                run_id="other-run-id",
                version="999",
            )
        ]
    )

    result = _register(
        monkeypatch,
        tmp_path,
        client,
        SimpleNamespace(registered_model_version=7),
    )

    assert result["version"] == "7"
    assert client.search_calls == []
    assert set(client.tag_versions) == {"7"}
    assert client.transition_versions == []
    assert result["stage"] == "None"
    assert result["lifecycle_stage"] == "Unassigned"
    assert result["promotion_mode"] == "manual"
    assert result["promotion_performed"] is False
    assert result["model_uri"].endswith("/7")
    assert result["execution_id"]
    assert client.tags["execution_id"] == result["execution_id"]
    assert client.tags["code_sha"] == "code-sha"


def test_packaged_model_bundle_loads_without_repository_on_pythonpath(
    tmp_path,
) -> None:
    raw = pd.DataFrame({"feature": [0.0, 1.0]})
    estimator = DummyClassifier(strategy="most_frequent").fit(
        raw,
        ["stay", "churn"],
    )
    bundle = RuntimeModelBundle(
        estimator=estimator,
        task_type="classification",
        candidate_id="isolated-load",
        input_schema={
            "columns": [{"name": "feature", "dtype": "float64"}],
            "column_order": ["feature"],
        },
        input_example=[{"feature": 0.0}],
    )
    model_path = tmp_path / "model"
    s12.mlflow.sklearn.save_model(
        sk_model=bundle,
        path=str(model_path),
        serialization_format="cloudpickle",
        code_paths=s12._model_bundle_code_paths(),
        pip_requirements=[],
    )

    clean_cwd = tmp_path / "clean"
    clean_cwd.mkdir()
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI"):
        environment.pop(name, None)
    command = (
        "import json, mlflow.pyfunc, pandas as pd; "
        f"model = mlflow.pyfunc.load_model({str(model_path)!r}); "
        "result = model.predict(pd.DataFrame([{'feature': 0.0}])); "
        "print(json.dumps(result.tolist()))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=clean_cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip()) == ["churn"]


def test_registration_requires_version_returned_by_log_model(
    monkeypatch,
    tmp_path,
) -> None:
    with pytest.raises(RuntimeError, match="did not return the exact"):
        _register(
            monkeypatch,
            tmp_path,
            _RegistryClient(),
            SimpleNamespace(registered_model_version=None),
        )


def test_registration_fails_closed_when_run_version_is_ambiguous() -> None:
    model_name = "test_dataset_classification_mlops"
    registry = _registry(
        _RegistryClient(
            search_results=[
                SimpleNamespace(
                    name=model_name,
                    run_id="owned-run-id",
                    version="11",
                ),
                SimpleNamespace(
                    name=model_name,
                    run_id="owned-run-id",
                    version="12",
                ),
            ]
        )
    )

    with pytest.raises(RuntimeError, match="did not return the exact"):
        registry._resolve_registered_model_version(
            log_model_result=SimpleNamespace(registered_model_version=None),
            model_name=model_name,
            run_id="owned-run-id",
        )


def test_registration_never_falls_back_to_direct_sdk_folder_registration(
    monkeypatch,
    tmp_path,
) -> None:
    client = _RegistryClient()
    monkeypatch.setattr(
        s12.ModelRegistry,
        "_register_with_azureml_sdk",
        lambda *_args, **_kwargs: pytest.fail(
            "Direct SDK folder registration is forbidden"
        ),
    )

    with pytest.raises(RuntimeError, match="artifact repository"):
        _register(
            monkeypatch,
            tmp_path,
            client,
            RuntimeError(
                "Could not find a registered artifact repository for: "
                "azureml://experiments/test/runs/owned-run-id/artifacts"
            ),
        )
    assert client.search_calls == []
    assert client.tag_versions == []
    assert client.transition_versions == []


def test_azureml_job_context_logs_exact_bundle_with_mlflow(
    monkeypatch,
    tmp_path,
) -> None:
    for name, value in (
        ("AZUREML_ARM_SUBSCRIPTION", "subscription"),
        ("AZUREML_ARM_RESOURCEGROUP", "resource-group"),
        ("AZUREML_ARM_WORKSPACE_NAME", "workspace"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("AZUREML_RUN_ID", "owned-run-id")

    monkeypatch.setattr(
        s12.ModelRegistry,
        "_register_with_azureml_sdk",
        lambda *_args, **_kwargs: pytest.fail(
            "AML jobs must not register mounted folders"
        ),
    )

    result = _register(
        monkeypatch,
        tmp_path,
        _RegistryClient(),
        SimpleNamespace(registered_model_version="17"),
    )

    assert result["version"] == "17"
    assert result["registration_backend"] == "mlflow"


def test_azureml_job_constructs_workspace_mlflow_client(
    monkeypatch,
    tmp_path,
) -> None:
    for name, value in (
        ("AZUREML_ARM_SUBSCRIPTION", "subscription"),
        ("AZUREML_ARM_RESOURCEGROUP", "resource-group"),
        ("AZUREML_ARM_WORKSPACE_NAME", "workspace"),
        ("AZUREML_RUN_ID", "owned-run-id"),
    ):
        monkeypatch.setenv(name, value)

    registry_client = _RegistryClient()
    monkeypatch.setattr(s12, "MlflowClient", lambda: registry_client)
    monkeypatch.setattr(
        s12,
        "create_metrics_logger",
        lambda **_kwargs: pytest.fail(
            "AML jobs must not construct the shared metrics logger"
        ),
    )

    registry = s12.ModelRegistry(
        "config_classification_test.yml",
        cfg={"dataset": {"name": "test_dataset"}},
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    metrics_logger = s12._create_registration_metrics_logger()

    assert registry.client is registry_client
    assert isinstance(metrics_logger, s12._NoOpRegistrationMetricsLogger)


def test_partial_azureml_context_fails_before_model_serialization(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("AZUREML_ARM_SUBSCRIPTION", "subscription")
    monkeypatch.delenv("AZUREML_ARM_RESOURCEGROUP", raising=False)
    monkeypatch.delenv("AZUREML_ARM_WORKSPACE_NAME", raising=False)

    with pytest.raises(
        RuntimeError,
        match="Incomplete Azure ML workspace context",
    ):
        s12._get_azureml_workspace_context()


def test_azureml_tracking_uri_is_not_rewritten(monkeypatch) -> None:
    tracking_uri = "azureml://eastus2.api.azureml.ms/mlflow/v1.0/workspace"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr(s12.mlflow, "autolog", lambda **_kwargs: None)
    monkeypatch.setattr(
        s12.mlflow,
        "set_tracking_uri",
        lambda *_args, **_kwargs: pytest.fail(
            "S12 must preserve AML tracking identity"
        ),
    )

    s12._safe_disable_autolog()

    assert s12.os.environ["MLFLOW_TRACKING_URI"] == tracking_uri


def test_sdk_registration_uses_exact_service_assigned_version(
    monkeypatch,
    tmp_path,
) -> None:
    captured = {}

    class _Models:
        def list(self, *, name):
            captured["listed_model_name"] = name
            return []

        def create_or_update(self, model):
            captured["model"] = model
            return SimpleNamespace(
                version="17",
                tags=model.tags,
                properties=model.properties,
                path=model.path,
            )

    remote_model_uri = (
        "azureml://subscriptions/subscription/resourcegroups/resource-group/"
        "workspaces/workspace/datastores/models/paths/champion"
    )
    fake_client = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda run_id: SimpleNamespace(
                inputs={
                    "champion_model": SimpleNamespace(
                        path=remote_model_uri,
                        type="uri_folder",
                    )
                }
            )
        ),
        models=_Models(),
    )
    monkeypatch.setattr(
        s12,
        "_create_azureml_sdk_client",
        lambda **kwargs: captured.setdefault("workspace", kwargs) and fake_client,
    )
    monkeypatch.setattr(
        s12,
        "_create_azureml_model_asset",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setenv("AZUREML_ARM_SUBSCRIPTION", "subscription")
    monkeypatch.setenv("AZUREML_ARM_RESOURCEGROUP", "resource-group")
    monkeypatch.setenv("AZUREML_ARM_WORKSPACE_NAME", "workspace")

    registry = _registry(_RegistryClient())
    manifest = {
        "task": "classification",
        "selection": {"key": "phasec", "score": 0.7},
        "phasec_metrics": {"balanced_accuracy": 0.7},
    }
    metadata = s12._resolve_registration_metadata(
        manifest,
        SimpleNamespace(
            named_steps={"estimator": type("XGBClassifier", (), {})()}
        ),
    )
    version = registry._register_with_azureml_sdk(
        model_name="test_dataset_classification_mlops",
        model_path=tmp_path,
        manifest=manifest,
        metadata=metadata,
        run_id="owned-run-id",
    )

    assert version == "17"
    assert not hasattr(captured["model"], "version")
    assert captured["model"].path == remote_model_uri
    assert captured["model"].tags["source_run_id"] == "owned-run-id"
    assert captured["model"].tags["algorithm"] == "XGBClassifier"
    assert captured["model"].tags["lifecycle_stage"] == "Unassigned"
    assert captured["workspace"] == {
        "subscription_id": "subscription",
        "resource_group": "resource-group",
        "workspace_name": "workspace",
    }
    assert captured["listed_model_name"] == (
        "test_dataset_classification_mlops"
    )


def test_sdk_registration_requires_job_workspace_context(
    monkeypatch,
    tmp_path,
) -> None:
    for name in (
        "AZUREML_ARM_SUBSCRIPTION",
        "AZUREML_ARM_RESOURCEGROUP",
        "AZUREML_ARM_WORKSPACE_NAME",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="AZUREML_ARM_SUBSCRIPTION"):
        _registry(_RegistryClient())._register_with_azureml_sdk(
            model_name="test-model",
            model_path=tmp_path,
            manifest={},
            metadata=s12._resolve_registration_metadata({}),
            run_id="owned-run-id",
        )


@pytest.mark.parametrize("returned_version", [None, "run-other", "0", "-1"])
def test_sdk_registration_rejects_non_positive_returned_version(
    monkeypatch,
    tmp_path,
    returned_version,
) -> None:
    fake_client = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda _run_id: SimpleNamespace(
                inputs={
                    "champion_model": {
                        "uri": (
                            "azureml://datastores/models/paths/champion"
                        ),
                        "jobInputType": "uri_folder",
                    }
                }
            )
        ),
        models=SimpleNamespace(
            list=lambda **_kwargs: [],
            create_or_update=lambda _model: SimpleNamespace(
                version=returned_version,
                tags={"source_run_id": "owned-run-id"},
                properties={
                    "source_run_id": "owned-run-id",
                    "source_model_uri_sha256": s12._model_uri_fingerprint(
                        "azureml://datastores/models/paths/champion"
                    ),
                },
                path="azureml://datastores/models/paths/champion",
            )
        )
    )
    monkeypatch.setattr(
        s12,
        "_create_azureml_sdk_client",
        lambda **_kwargs: fake_client,
    )
    monkeypatch.setattr(
        s12,
        "_create_azureml_model_asset",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setenv("AZUREML_ARM_SUBSCRIPTION", "subscription")
    monkeypatch.setenv("AZUREML_ARM_RESOURCEGROUP", "resource-group")
    monkeypatch.setenv("AZUREML_ARM_WORKSPACE_NAME", "workspace")

    with pytest.raises(RuntimeError, match="positive integer model version"):
        _registry(_RegistryClient())._register_with_azureml_sdk(
            model_name="test-model",
            model_path=tmp_path,
            manifest={},
            metadata=s12._resolve_registration_metadata({}),
            run_id="owned-run-id",
        )


def test_sdk_registration_reuses_single_exact_run_bound_model(
    monkeypatch,
    tmp_path,
) -> None:
    existing = SimpleNamespace(
        version="23",
        tags={"source_run_id": "owned-run-id"},
        properties={"source_run_id": "owned-run-id"},
    )
    full_existing = SimpleNamespace(
        version="23",
        tags={"source_run_id": "owned-run-id"},
        properties={
            "source_run_id": "owned-run-id",
            "source_model_uri_sha256": s12._model_uri_fingerprint(
                "azureml://datastores/models/paths/champion"
            ),
        },
        path="azureml://datastores/models/paths/champion",
    )
    fake_client = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda _run_id: SimpleNamespace(
                inputs={
                    "champion_model": SimpleNamespace(
                        path="azureml://datastores/models/paths/champion",
                        type="uri_folder",
                    )
                }
            )
        ),
        models=SimpleNamespace(
            list=lambda **_kwargs: [existing],
            get=lambda **_kwargs: full_existing,
            create_or_update=lambda _model: pytest.fail(
                "An idempotent retry must reuse the run-bound model"
            ),
        ),
    )
    monkeypatch.setattr(
        s12,
        "_create_azureml_sdk_client",
        lambda **_kwargs: fake_client,
    )
    monkeypatch.setenv("AZUREML_ARM_SUBSCRIPTION", "subscription")
    monkeypatch.setenv("AZUREML_ARM_RESOURCEGROUP", "resource-group")
    monkeypatch.setenv("AZUREML_ARM_WORKSPACE_NAME", "workspace")

    version = _registry(_RegistryClient())._register_with_azureml_sdk(
        model_name="test-model",
        model_path=tmp_path,
        manifest={},
        metadata=s12._resolve_registration_metadata({}),
        run_id="owned-run-id",
    )

    assert version == "23"


def test_model_uri_validation_accepts_arm_resource_segment_casing() -> None:
    source_uri = (
        "azureml://subscriptions/subscription/resourcegroups/resource-group/"
        "workspaces/workspace/datastores/models/paths/Champion/model.pkl"
    )
    returned_uri = source_uri.replace(
        "/resourcegroups/",
        "/resourceGroups/",
    )
    model = SimpleNamespace(
        version="23",
        tags={"source_run_id": "owned-run-id"},
        properties={
            "source_run_id": "owned-run-id",
            "source_model_uri_sha256": s12._model_uri_fingerprint(
                source_uri
            ),
        },
        path=returned_uri,
    )

    assert s12._validate_run_bound_azureml_model(
        model,
        run_id="owned-run-id",
        model_uri=source_uri,
    ) == "23"
    assert s12._normalize_azureml_uri(source_uri).endswith(
        "/paths/Champion/model.pkl"
    )


def test_sdk_registration_rejects_ambiguous_run_bound_models(
    monkeypatch,
    tmp_path,
) -> None:
    duplicate_models = [
        SimpleNamespace(
            version=version,
            tags={"source_run_id": "owned-run-id"},
            properties={"source_run_id": "owned-run-id"},
        )
        for version in ("23", "24")
    ]
    fake_client = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda _run_id: SimpleNamespace(
                inputs={
                    "champion_model": SimpleNamespace(
                        path="azureml://datastores/models/paths/champion",
                        type="uri_folder",
                    )
                }
            )
        ),
        models=SimpleNamespace(
            list=lambda **_kwargs: duplicate_models,
            get=lambda **_kwargs: pytest.fail(
                "Ambiguous versions must fail before fetching an asset"
            ),
            create_or_update=lambda _model: pytest.fail(
                "Ambiguous run ownership must fail before registration"
            ),
        ),
    )
    monkeypatch.setattr(
        s12,
        "_create_azureml_sdk_client",
        lambda **_kwargs: fake_client,
    )
    monkeypatch.setenv("AZUREML_ARM_SUBSCRIPTION", "subscription")
    monkeypatch.setenv("AZUREML_ARM_RESOURCEGROUP", "resource-group")
    monkeypatch.setenv("AZUREML_ARM_WORKSPACE_NAME", "workspace")

    with pytest.raises(RuntimeError, match="2 versions map to source run"):
        _registry(_RegistryClient())._register_with_azureml_sdk(
            model_name="test-model",
            model_path=tmp_path,
            manifest={},
            metadata=s12._resolve_registration_metadata({}),
            run_id="owned-run-id",
        )


@pytest.mark.parametrize(
    "full_model",
    [
        SimpleNamespace(
            version="23",
            tags={"source_run_id": "owned-run-id"},
            properties={
                "source_run_id": "other-run-id",
                "source_model_uri_sha256": s12._model_uri_fingerprint(
                    "azureml://datastores/models/paths/champion"
                ),
            },
            path="azureml://datastores/models/paths/champion",
        ),
        SimpleNamespace(
            version="23",
            tags={"source_run_id": "owned-run-id"},
            properties={
                "source_run_id": "owned-run-id",
                "source_model_uri_sha256": s12._model_uri_fingerprint(
                    "azureml://datastores/models/paths/stale"
                ),
            },
            path="azureml://datastores/models/paths/stale",
        ),
    ],
)
def test_sdk_registration_rejects_mistagged_or_stale_reuse(
    monkeypatch,
    tmp_path,
    full_model,
) -> None:
    candidate = SimpleNamespace(
        version="23",
        tags={"source_run_id": "owned-run-id"},
        properties={},
    )
    fake_client = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda _run_id: SimpleNamespace(
                inputs={
                    "champion_model": SimpleNamespace(
                        path="azureml://datastores/models/paths/champion",
                        type="uri_folder",
                    )
                }
            )
        ),
        models=SimpleNamespace(
            list=lambda **_kwargs: [candidate],
            get=lambda **_kwargs: full_model,
            create_or_update=lambda _model: pytest.fail(
                "A candidate reuse must be validated before registration"
            ),
        ),
    )
    monkeypatch.setattr(
        s12,
        "_create_azureml_sdk_client",
        lambda **_kwargs: fake_client,
    )
    monkeypatch.setenv("AZUREML_ARM_SUBSCRIPTION", "subscription")
    monkeypatch.setenv("AZUREML_ARM_RESOURCEGROUP", "resource-group")
    monkeypatch.setenv("AZUREML_ARM_WORKSPACE_NAME", "workspace")

    with pytest.raises(RuntimeError, match="does not match"):
        _registry(_RegistryClient())._register_with_azureml_sdk(
            model_name="test-model",
            model_path=tmp_path,
            manifest={},
            metadata=s12._resolve_registration_metadata({}),
            run_id="owned-run-id",
        )


@pytest.mark.parametrize(
    "job_input",
    [
        None,
        SimpleNamespace(path="/mnt/azureml/champion", type="uri_folder"),
        SimpleNamespace(
            path="azureml://datastores/models/paths/champion",
            type="uri_file",
        ),
    ],
)
def test_sdk_registration_requires_remote_uri_folder_job_input(
    monkeypatch,
    tmp_path,
    job_input,
) -> None:
    inputs = {} if job_input is None else {"champion_model": job_input}
    fake_client = SimpleNamespace(
        jobs=SimpleNamespace(
            get=lambda _run_id: SimpleNamespace(inputs=inputs)
        ),
        models=SimpleNamespace(
            list=lambda **_kwargs: [],
            create_or_update=lambda _model: pytest.fail(
                "Invalid job inputs must fail before model registration"
            )
        ),
    )
    monkeypatch.setattr(
        s12,
        "_create_azureml_sdk_client",
        lambda **_kwargs: fake_client,
    )
    monkeypatch.setenv("AZUREML_ARM_SUBSCRIPTION", "subscription")
    monkeypatch.setenv("AZUREML_ARM_RESOURCEGROUP", "resource-group")
    monkeypatch.setenv("AZUREML_ARM_WORKSPACE_NAME", "workspace")

    with pytest.raises(RuntimeError, match="champion_model"):
        _registry(_RegistryClient())._register_with_azureml_sdk(
            model_name="test-model",
            model_path=tmp_path,
            manifest={},
            metadata=s12._resolve_registration_metadata({}),
            run_id="owned-run-id",
        )


def test_sdk_client_factory_prefers_obo_then_managed_identity(
    monkeypatch,
) -> None:
    captured = {}

    class _OBOCredential:
        def __init__(self):
            self.calls = []

        def get_token(self, *scopes, **kwargs):
            self.calls.append((scopes, kwargs))
            return "obo-token"

    obo_credential = _OBOCredential()
    default_credential = object()
    chained_credential = object()
    fake_client = object()

    monkeypatch.setenv("OBO_ENDPOINT", "https://identity.test")
    monkeypatch.setattr(
        "azure.ai.ml.identity.AzureMLOnBehalfOfCredential",
        lambda: obo_credential,
    )
    monkeypatch.setattr(
        "azure.identity.DefaultAzureCredential",
        lambda **kwargs: captured.setdefault("default_kwargs", kwargs)
        and default_credential,
    )
    monkeypatch.setattr(
        "azure.identity.ChainedTokenCredential",
        lambda *credentials: captured.setdefault("credentials", credentials)
        and chained_credential,
    )

    def _ml_client(credential, subscription_id, resource_group, workspace_name):
        captured["client_args"] = (
            credential,
            subscription_id,
            resource_group,
            workspace_name,
        )
        return fake_client

    monkeypatch.setattr("azure.ai.ml.MLClient", _ml_client)

    result = s12._create_azureml_sdk_client(
        subscription_id="subscription",
        resource_group="resource-group",
        workspace_name="workspace",
    )

    assert result is fake_client
    obo_adapter, chained_default = captured["credentials"]
    assert isinstance(obo_adapter, s12._AzureMLOBOCredentialAdapter)
    assert chained_default is default_credential
    assert obo_adapter.get_token(
        "https://management.azure.com/.default",
        claims="challenge",
    ) == "obo-token"
    assert obo_credential.calls == [
        (("https://management.azure.com/.default",), {})
    ]
    assert captured["default_kwargs"] == {
        "exclude_interactive_browser_credential": True
    }
    assert captured["client_args"] == (
        chained_credential,
        "subscription",
        "resource-group",
        "workspace",
    )


def test_sdk_client_factory_uses_default_credential_without_obo(
    monkeypatch,
) -> None:
    captured = {}
    default_credential = object()
    fake_client = object()
    monkeypatch.delenv("OBO_ENDPOINT", raising=False)
    monkeypatch.setattr(
        "azure.identity.DefaultAzureCredential",
        lambda **kwargs: captured.setdefault("default_kwargs", kwargs)
        and default_credential,
    )
    monkeypatch.setattr(
        "azure.identity.ChainedTokenCredential",
        lambda *_credentials: pytest.fail(
            "A single credential must not be wrapped in a chain"
        ),
    )

    def _ml_client(credential, subscription_id, resource_group, workspace_name):
        captured["client_args"] = (
            credential,
            subscription_id,
            resource_group,
            workspace_name,
        )
        return fake_client

    monkeypatch.setattr("azure.ai.ml.MLClient", _ml_client)

    result = s12._create_azureml_sdk_client(
        subscription_id="subscription",
        resource_group="resource-group",
        workspace_name="workspace",
    )

    assert result is fake_client
    assert captured["default_kwargs"] == {
        "exclude_interactive_browser_credential": True
    }
    assert captured["client_args"] == (
        default_credential,
        "subscription",
        "resource-group",
        "workspace",
    )


def test_real_azureml_model_entity_preserves_explicit_version(tmp_path) -> None:
    from azure.ai.ml.entities import Model

    asset = s12._create_azureml_model_asset(
        path=str(tmp_path),
        name="test-model",
        version="run-0123456789abcdef",
        description="test",
        tags={"source_run_id": "run-id"},
    )

    assert isinstance(asset, Model)
    assert asset.version == "run-0123456789abcdef"
    assert asset.tags["source_run_id"] == "run-id"


def test_s12_environment_uses_hashed_lock_and_build_smoke_checks() -> None:
    repo_root = Path(s12.__file__).resolve().parents[2]
    component = yaml.safe_load(
        (repo_root / "components" / "s12_model_registration.yml").read_text(
            encoding="utf-8"
        )
    )
    environment_dir = repo_root / "config" / "s12_registration_environment"
    dockerfile = (environment_dir / "Dockerfile").read_text(encoding="utf-8")
    lock_lines = (environment_dir / "requirements.lock").read_text(
        encoding="utf-8"
    ).splitlines()
    requirements = [
        line
        for line in lock_lines
        if line and line[0].isalnum()
    ]
    requirement_names = {
        line.split("==", 1)[0]: line.split("==", 1)[1].rstrip(" \\")
        for line in requirements
        if "==" in line
    }

    assert component["version"] == 15
    assert component["environment"] == "azureml:mlops-v3-registration:3-ee968ec2"
    assert requirements
    assert all("==" in requirement for requirement in requirements)
    requirement_blocks = []
    current_block = []
    for line in lock_lines:
        if line and line[0].isalnum():
            if current_block:
                requirement_blocks.append(current_block)
            current_block = [line]
        elif current_block:
            current_block.append(line)
    if current_block:
        requirement_blocks.append(current_block)

    assert len(requirements) >= 100
    assert all(
        any("--hash=sha256:" in line for line in block)
        for block in requirement_blocks
    )
    assert requirement_names["azure-ai-ml"] == "1.17.1"
    assert requirement_names["azureml-mlflow"] == "1.57.0.post1"
    assert requirement_names["catboost"] == "1.2.8"
    assert requirement_names["lightgbm"] == "4.6.0"
    assert requirement_names["marshmallow"] == "3.23.3"
    assert requirement_names["mlflow"] == "2.14.3"
    assert requirement_names["mlflow-skinny"] == "2.14.3"
    assert requirement_names["pandas"] == "2.1.4"
    assert requirement_names["joblib"] == "1.3.2"
    assert requirement_names["scikit-learn"] == "1.4.2"
    assert requirement_names["xgboost"] == "3.1.3"
    assert (
        "FROM python:3.10.14-bookworm@"
        "sha256:9c0e621579faf384d982986f2e0ba86bf09619076842cd0fbd2f24a3bf09f0bc"
        in dockerfile
    )
    assert ":latest" not in dockerfile
    assert "--require-hashes" in dockerfile
    assert "conda create" not in dockerfile
    assert "-m pip check" in dockerfile
    assert "from azure.ai.ml import MLClient" in dockerfile
    assert "import azureml.mlflow, catboost, lightgbm" in dockerfile
    assert "marshmallow.__version__ == '3.23.3'" in dockerfile


def test_phase_specific_metadata_is_consistent_with_concrete_estimator() -> None:
    manifest = {
        "task": "classification",
        "selection": {"key": "phasec", "score": 0.7},
        "phasec_metrics": {"balanced_accuracy": 0.7},
    }
    model = SimpleNamespace(
        named_steps={"estimator": type("XGBClassifier", (), {})()}
    )
    metadata = s12._resolve_registration_metadata(manifest, model)
    tags = _registry(_RegistryClient())._build_model_metadata_tags(
        manifest,
        metadata,
    )

    assert metadata["algorithm"] == "XGBClassifier"
    assert metadata["metrics"] == {"balanced_accuracy": 0.7}
    assert tags["algorithm"] == metadata["algorithm"]
    assert tags["metric_balanced_accuracy"] == "0.7"


def test_mlflow_metadata_failure_blocks_stage_promotion(
    monkeypatch,
    tmp_path,
) -> None:
    class _FailingTagClient(_RegistryClient):
        def set_model_version_tag(self, **_kwargs):
            raise PermissionError("tag write denied")

    client = _FailingTagClient()

    with pytest.raises(
        RuntimeError,
        match="Failed to persist required model metadata tags",
    ):
        _register(
            monkeypatch,
            tmp_path,
            client,
            SimpleNamespace(registered_model_version="7"),
        )

    assert client.transition_versions == []


def test_artifact_selection_requires_exact_model_file(tmp_path) -> None:
    registry = _registry(_RegistryClient())
    encoder = tmp_path / "label_encoder.pkl"
    encoder.write_bytes(b"encoder")
    model = tmp_path / "model.pkl"
    model.write_bytes(b"model")

    assert registry._find_model_artifact(tmp_path) is None
    bundle = tmp_path / "model_bundle.pkl"
    bundle.write_bytes(b"bundle")
    assert registry._find_model_artifact(tmp_path) == bundle


def test_artifact_selection_rejects_encoder_only_directory(tmp_path) -> None:
    registry = _registry(_RegistryClient())
    (tmp_path / "label_encoder.pkl").write_bytes(b"encoder")

    assert registry._find_model_artifact(tmp_path) is None


def test_main_fails_component_when_registration_is_unexpectedly_skipped(
    monkeypatch,
    tmp_path,
) -> None:
    raw_config = {
        "schema_version": "2.0",
        "experiment_name": "registration-failure-test",
        "preset": "production",
        "task_type": "classification",
        "dataset": {
            "name": "test",
            "version": "1",
            "target_column": "target",
        },
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    compiled = s12.compile_config(raw_config, source_name=config_path.name)
    execution_manifest = ExecutionManifest(
        config_hash=compiled["compiled_config_hash"],
        task_type="classification",
        dataset=compiled["dataset"],
        split_policy=compiled["split"],
        engines=("pycaret",),
        recipe_paths=("classification/test.yml",),
        recipe_ids=("recipe-1",),
        candidate_ids=("candidate-1",),
        budgets={"round1_max_variants": 1},
        code_sha="code-sha",
        environment_hashes={"training": "environment-hash"},
        recipe_catalog_hash="catalog-hash",
    )
    execution_path = tmp_path / "execution_manifest.json"
    execution_path.write_text(execution_manifest.to_json(), encoding="utf-8")
    manifest = tmp_path / "final_report.json"
    manifest.write_text(
        json.dumps(
            {
                "task": "classification",
                "champion_valid": True,
                "quality_gate_passed": True,
                "selection": {"key": "phasec", "score": 0.7},
                "lineage": {"execution_id": execution_manifest.execution_id},
            }
        ),
        encoding="utf-8",
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.pkl").write_bytes(b"model")
    output = tmp_path / "registry_info.json"

    class _FailingRegistry:
        def __init__(self, *_args, **_kwargs):
            pass

        def register_champion_model(self, **_kwargs):
            raise RuntimeError("registry unavailable")

    monkeypatch.setattr(s12, "ModelRegistry", _FailingRegistry)
    monkeypatch.setattr(
        s12,
        "create_metrics_logger",
        lambda **_kwargs: _MetricsLogger(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "s12_model_registration.py",
            "--champion_manifest",
            str(manifest),
            "--champion_model",
            str(model_dir),
            "--config_name",
            str(config_path),
            "--execution_manifest",
            str(execution_path),
            "--registry_info",
            str(output),
        ],
    )

    with pytest.raises(RuntimeError, match="Model registration failed"):
        s12.main()

    diagnostic = json.loads(output.read_text(encoding="utf-8"))
    assert diagnostic["registration_failed"] is True
    assert "registry unavailable" in diagnostic["failure_reason"]


@pytest.mark.parametrize(
    ("manifest_text", "create_model", "expected_reason"),
    [
        (None, True, "manifest_not_found"),
        ("{not-json", True, "invalid_manifest"),
        ("[]", True, "invalid_manifest_root"),
        (
            json.dumps({"selection": "phasec"}),
            True,
            "invalid_selection_metadata",
        ),
        (
            json.dumps({"algorithm": "RandomForestClassifier"}),
            True,
            "invalid_task_metadata",
        ),
        (
            json.dumps(
                {
                    "task": "classification",
                    "selection": {"key": "phasec", "score": None},
                }
            ),
            True,
            "null_score_no_algorithm",
        ),
        (
            json.dumps(
                {
                    "task": "classification",
                    "selection": {"key": "phasec", "score": 0.7},
                }
            ),
            False,
            "model_not_found",
        ),
    ],
)
def test_invalid_required_inputs_fail_component_with_diagnostic(
    monkeypatch,
    tmp_path,
    manifest_text,
    create_model,
    expected_reason,
) -> None:
    manifest = tmp_path / "final_report.json"
    if manifest_text is not None:
        manifest.write_text(manifest_text, encoding="utf-8")
    model_dir = tmp_path / "model"
    if create_model:
        model_dir.mkdir()
        (model_dir / "model.pkl").write_bytes(b"model")
    output = tmp_path / "registry_info.json"

    monkeypatch.setattr(
        s12,
        "create_metrics_logger",
        lambda **_kwargs: _MetricsLogger(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "s12_model_registration.py",
            "--champion_manifest",
            str(manifest),
            "--champion_model",
            str(model_dir),
            "--config_name",
            "missing.yml",
            "--registry_info",
            str(output),
        ],
    )

    with pytest.raises(RuntimeError, match="contract failed"):
        s12.main()

    diagnostic = json.loads(output.read_text(encoding="utf-8"))
    assert diagnostic["registration_failed"] is True
    assert expected_reason in diagnostic["failure_reason"]


def test_policy_rejection_remains_successful_skip(
    monkeypatch,
    tmp_path,
) -> None:
    manifest = tmp_path / "final_report.json"
    manifest.write_text(
        json.dumps(
            {
                "task": "classification",
                "champion_valid": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "registry_info.json"

    monkeypatch.setattr(
        s12,
        "create_metrics_logger",
        lambda **_kwargs: _MetricsLogger(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "s12_model_registration.py",
            "--champion_manifest",
            str(manifest),
            "--champion_model",
            str(tmp_path / "unused-model"),
            "--config_name",
            "missing.yml",
            "--registry_info",
            str(output),
        ],
    )

    s12.main()

    diagnostic = json.loads(output.read_text(encoding="utf-8"))
    assert diagnostic["registration_skipped"] is True
    assert diagnostic["skip_reason"] == "no_valid_champion"
