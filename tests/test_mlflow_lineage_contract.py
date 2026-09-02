from types import SimpleNamespace

import pytest
from sklearn.dummy import DummyClassifier

from src.steps import final_evaluation, phasec_optuna_hpo
from src.utils import azureml_metrics_logger
from src.utils.model_bundle import ModelBundle, load_model_bundle, save_model_bundle


def test_exact_execution_and_run_lineage_survives_bundle_roundtrip(tmp_path):
    lineage = {
        "execution_id": "execution-1",
        "parent_run_id": "parent-1",
        "candidate_run_id": "candidate-1",
        "final_evaluation_run_id": "final-1",
        "model_version": "17",
    }
    model = DummyClassifier(strategy="most_frequent").fit([[0], [1]], [0, 1])
    bundle = ModelBundle(
        estimator=model,
        task_type="classification",
        candidate_id="candidate-1",
        input_schema={
            "columns": [{"name": "feature", "dtype": "int64"}],
            "column_order": ["feature"],
        },
        lineage=lineage,
    )
    save_model_bundle(bundle, tmp_path)

    assert load_model_bundle(tmp_path).lineage == lineage


def test_workspace_tracking_uri_is_preserved_exactly(monkeypatch):
    tracking_uri = (
        "azureml://eastus2.api.azureml.ms/mlflow/v1.0/"
        "subscriptions/sub/resourceGroups/rg/providers/"
        "Microsoft.MachineLearningServices/workspaces/ws"
    )
    observed = []
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setattr(
        azureml_metrics_logger.mlflow,
        "set_tracking_uri",
        observed.append,
    )

    azureml_metrics_logger.normalize_mlflow_tracking_uri()

    assert observed == [tracking_uri]
    assert azureml_metrics_logger.os.environ["MLFLOW_TRACKING_URI"] == tracking_uri


def test_stage_metrics_use_only_exact_manifest_run_ids(monkeypatch):
    requested = []

    class ExactClient:
        def get_run(self, run_id):
            requested.append(run_id)
            return SimpleNamespace(
                data=SimpleNamespace(
                    tags={"execution_id": "execution-1"},
                    metrics={"balanced_accuracy": 0.8},
                    params={"candidate": run_id},
                )
            )

        def search_runs(self, *args, **kwargs):
            raise AssertionError("Recent-run scanning is forbidden")

    monkeypatch.setattr(
        final_evaluation.mlflow.tracking,
        "MlflowClient",
        lambda: ExactClient(),
    )
    result = final_evaluation.collect_all_stage_metrics(
        {"baseline": "child-1", "phaseb": "child-2"},
        "execution-1",
    )

    assert requested == ["child-1", "child-2"]
    assert result["aggregates"]["baseline"]["run_id"] == "child-1"


def test_stage_metrics_reject_mixed_live_execution(monkeypatch):
    class MixedClient:
        def get_run(self, run_id):
            return SimpleNamespace(
                data=SimpleNamespace(
                    tags={"execution_id": "other-execution"},
                    metrics={},
                    params={},
                )
            )

    monkeypatch.setattr(
        final_evaluation.mlflow.tracking,
        "MlflowClient",
        lambda: MixedClient(),
    )
    with pytest.raises(RuntimeError, match="belongs to execution"):
        final_evaluation.collect_all_stage_metrics(
            {"baseline": "child-1"},
            "execution-1",
        )


def test_phasec_creates_exact_tagged_candidate_child_run(monkeypatch):
    observed = {}

    class ExactClient:
        def get_run(self, run_id):
            assert run_id == "component-run"
            return SimpleNamespace(
                info=SimpleNamespace(experiment_id="experiment-1")
            )

        def create_run(self, **kwargs):
            observed.update(kwargs)
            return SimpleNamespace(info=SimpleNamespace(run_id="candidate-run"))

    monkeypatch.setattr(
        phasec_optuna_hpo,
        "MlflowClient",
        ExactClient,
    )
    monkeypatch.setattr(
        phasec_optuna_hpo.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="component-run")),
    )

    _, parent_run_id, candidate_run_id = (
        phasec_optuna_hpo.create_phasec_candidate_run(
            candidate_id="phasec:candidate-1",
            execution_id="execution-1",
        )
    )

    assert parent_run_id == "component-run"
    assert candidate_run_id == "candidate-run"
    assert observed["experiment_id"] == "experiment-1"
    assert observed["tags"]["execution_id"] == "execution-1"
    assert observed["tags"]["candidate_id"] == "phasec:candidate-1"
    assert observed["tags"]["mlflow.parentRunId"] == "component-run"
