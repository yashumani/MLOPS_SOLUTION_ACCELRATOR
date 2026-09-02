from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge

from src.orchestration.contracts import SplitManifest, canonical_hash
from src.steps.aggregate_baseline import select_champion
from src.steps.stage5_pycaret_train import (
    CLUSTERING_PYCARET_SELECTION_SAMPLE_ROWS,
    train_clustering_baseline,
)
from src.utils.common_evaluator import EvaluationSpec, evaluate_candidate
from src.utils.model_bundle import (
    BUNDLE_FILE_NAME,
    BUNDLE_MANIFEST_NAME,
    load_model_bundle,
)
from src.utils.phasea_model_bundle import (
    PhaseABundleError,
    build_phasea_evaluation_pipeline,
    fit_save_phasea_bundle,
    load_baseline_recipe,
    load_phasea_split_manifest,
    phasea_candidate_id,
    validate_phasea_bundle_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phasea_clustering_selection_is_sampled_then_refit(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class _Model:
        def fit(self, data):
            observed["full_refit_rows"] = len(data)
            return self

        def predict(self, data):
            observed["prediction_rows"] = len(data)
            return np.arange(len(data)) % 3

    module = types.ModuleType("pycaret.clustering")

    def _setup(*, data, **_kwargs):
        observed["selection_rows"] = len(data)

    module.setup = _setup
    module.create_model = lambda *_args, **_kwargs: _Model()
    module.pull = lambda: pd.DataFrame({"Silhouette": [0.42]})
    monkeypatch.setitem(sys.modules, "pycaret.clustering", module)

    def _silhouette(*_args, **kwargs):
        observed["silhouette_sample_size"] = kwargs["sample_size"]
        return 0.42

    monkeypatch.setattr(
        sklearn.metrics,
        "silhouette_score",
        _silhouette,
    )
    monkeypatch.setattr(
        sklearn.metrics,
        "davies_bouldin_score",
        lambda *_args, **_kwargs: 0.73,
    )

    total_rows = CLUSTERING_PYCARET_SELECTION_SAMPLE_ROWS + 25
    frame = pd.DataFrame(
        {
            "feature_a": np.arange(total_rows, dtype=float),
            "feature_b": np.arange(total_rows, dtype=float) % 7,
        }
    )

    model, _leaderboard, metrics = train_clustering_baseline(
        frame,
        random_seed=17,
    )

    assert model is not None
    assert observed["selection_rows"] == CLUSTERING_PYCARET_SELECTION_SAMPLE_ROWS
    assert observed["full_refit_rows"] == total_rows
    assert observed["prediction_rows"] == total_rows
    assert observed["silhouette_sample_size"] == 10_000
    assert metrics["pycaret_selection_rows"] == CLUSTERING_PYCARET_SELECTION_SAMPLE_ROWS
    assert metrics["full_refit_rows"] == total_rows


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [1.0, 1.2, np.nan, 1.4, 8.0, 8.2, 8.4, 8.6, 2.0, 2.2],
            "visits": [1, 2, 1, 3, 8, 9, 8, 10, 2, 3],
            "region": ["east", "east", "west", "east", "north", "north", "west", "north", "west", "east"],
        }
    )


def _target(task_type: str) -> pd.Series | None:
    if task_type == "classification":
        return pd.Series([0, 0, 0, 0, 1, 1, 1, 1, 0, 0], name="target")
    if task_type == "regression":
        return pd.Series([1.0, 1.4, 1.1, 1.8, 8.1, 8.5, 8.4, 9.0, 2.2, 2.5], name="target")
    return None


def _estimator(task_type: str):
    if task_type == "classification":
        return LogisticRegression(max_iter=200, random_state=17)
    if task_type == "regression":
        return Ridge(alpha=1.0)
    return KMeans(n_clusters=2, random_state=17, n_init=10)


def _split(task_type: str, row_count: int, seed: int = 17) -> SplitManifest:
    return SplitManifest(
        task_type=task_type,
        strategy="stratified" if task_type == "classification" else "random",
        random_seed=seed,
        train_count=row_count,
        validation_count=0,
        test_count=2,
        train_ids_hash=canonical_hash([f"train-{index}" for index in range(row_count)]),
        validation_ids_hash=canonical_hash([]),
        test_ids_hash=canonical_hash(["test-0", "test-1"]),
        data_version="phasea-test@1:test.csv",
    )


def _evidence(
    *,
    candidate_id: str,
    engine: str,
    task_type: str,
    status: str = "success",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "engine": engine,
        "task_type": task_type,
        "status": status,
        "primary_metric": {
            "classification": "balanced_accuracy",
            "regression": "r2",
            "clustering": "silhouette_score",
        }[task_type],
        "selection_score": 0.75 if status == "success" else None,
        "split_fingerprint": "shared-phasea-folds",
        "completed_folds": 1 if task_type == "clustering" else 3,
        "total_folds": 1 if task_type == "clustering" else 3,
        "seed": 17,
        "timeout_seconds": 60.0,
        "censored": status != "success",
    }


@pytest.mark.parametrize(
    ("task_type", "engine"),
    [
        ("classification", "pycaret"),
        ("regression", "flaml"),
        ("clustering", "pycaret"),
    ],
)
def test_phasea_bundle_round_trips_raw_mixed_type_input(
    tmp_path: Path,
    task_type: str,
    engine: str,
) -> None:
    raw = _raw_frame()
    target = _target(task_type)
    estimator = _estimator(task_type)
    recipe = load_baseline_recipe(ROOT, task_type)
    candidate_id = phasea_candidate_id(engine, estimator, recipe)
    evidence = _evidence(
        candidate_id=candidate_id,
        engine=engine,
        task_type=task_type,
    )

    artifact = fit_save_phasea_bundle(
        estimator,
        raw,
        target,
        task_type=task_type,
        engine=engine,
        candidate_id=candidate_id,
        recipe=recipe,
        evidence=evidence,
        split_manifest=_split(task_type, len(raw)),
        output_dir=tmp_path,
        random_seed=17,
        execution_id="execution-1",
        mlflow_parent_run_id="parent-1",
        mlflow_child_run_id="child-1",
    )

    bundle = load_model_bundle(tmp_path)
    predictions = bundle.predict(raw.head(3))
    assert len(predictions) == 3
    assert artifact.smoke_test["status"] == "passed"
    assert artifact.manifest["bundle_id"] == bundle.bundle_id
    assert bundle.preprocessing is not None
    assert bundle.lineage["split_id"] == _split(task_type, len(raw)).split_id


def test_phasea_common_evaluation_pipeline_starts_with_unfitted_preprocessing() -> None:
    raw = _raw_frame()
    target = _target("classification")
    recipe = load_baseline_recipe(ROOT, "classification")
    pipeline = build_phasea_evaluation_pipeline(
        LogisticRegression(max_iter=200, random_state=17),
        recipe=recipe,
        task_type="classification",
        random_seed=17,
    )

    assert not hasattr(pipeline.named_steps["preprocessing"], "input_columns_")
    fitted = clone(pipeline).fit(raw.iloc[:8], target.iloc[:8])
    assert fitted.named_steps["preprocessing"].input_columns_ == list(raw.columns)
    assert not hasattr(pipeline.named_steps["preprocessing"], "input_columns_")


def test_phasea_raw_pipeline_runs_inside_common_evaluator_process() -> None:
    raw = _raw_frame()
    target = _target("classification")
    recipe = load_baseline_recipe(ROOT, "classification")
    estimator = LogisticRegression(max_iter=200, random_state=17)
    candidate_id = phasea_candidate_id("pycaret", estimator, recipe)
    pipeline = build_phasea_evaluation_pipeline(
        estimator,
        recipe=recipe,
        task_type="classification",
        random_seed=17,
    )

    evidence = evaluate_candidate(
        pipeline,
        raw,
        target,
        candidate_id=candidate_id,
        engine="pycaret",
        spec=EvaluationSpec(
            task_type="classification",
            seed=17,
            folds=3,
            timeout_seconds=45,
            execution_id="execution-1",
        ),
    )

    assert evidence.status == "success"
    assert evidence.completed_folds == 3
    assert evidence.split_fingerprint


def test_failed_common_evidence_never_writes_bundle(tmp_path: Path) -> None:
    raw = _raw_frame()
    target = _target("classification")
    estimator = _estimator("classification")
    recipe = load_baseline_recipe(ROOT, "classification")
    candidate_id = phasea_candidate_id("pycaret", estimator, recipe)

    with pytest.raises(PhaseABundleError, match="not successful"):
        fit_save_phasea_bundle(
            estimator,
            raw,
            target,
            task_type="classification",
            engine="pycaret",
            candidate_id=candidate_id,
            recipe=recipe,
            evidence=_evidence(
                candidate_id=candidate_id,
                engine="pycaret",
                task_type="classification",
                status="timeout",
            ),
            split_manifest=_split("classification", len(raw)),
            output_dir=tmp_path,
            random_seed=17,
            execution_id="execution-1",
            mlflow_parent_run_id="parent-1",
            mlflow_child_run_id="child-1",
        )

    assert not (tmp_path / BUNDLE_FILE_NAME).exists()
    assert not (tmp_path / BUNDLE_MANIFEST_NAME).exists()


def test_failed_raw_smoke_removes_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.utils.phasea_model_bundle as phasea

    raw = _raw_frame()
    target = _target("regression")
    estimator = _estimator("regression")
    recipe = load_baseline_recipe(ROOT, "regression")
    candidate_id = phasea_candidate_id("flaml", estimator, recipe)

    def fail_smoke(*_args, **_kwargs):
        raise PhaseABundleError("smoke failed")

    monkeypatch.setattr(phasea, "smoke_test_saved_bundle", fail_smoke)
    with pytest.raises(PhaseABundleError, match="smoke failed"):
        phasea.fit_save_phasea_bundle(
            estimator,
            raw,
            target,
            task_type="regression",
            engine="flaml",
            candidate_id=candidate_id,
            recipe=recipe,
            evidence=_evidence(
                candidate_id=candidate_id,
                engine="flaml",
                task_type="regression",
            ),
            split_manifest=_split("regression", len(raw)),
            output_dir=tmp_path,
            random_seed=17,
            execution_id="execution-1",
            mlflow_parent_run_id="parent-1",
            mlflow_child_run_id="child-1",
        )

    assert not (tmp_path / BUNDLE_FILE_NAME).exists()
    assert not (tmp_path / BUNDLE_MANIFEST_NAME).exists()


def test_split_manifest_must_match_raw_training_count(tmp_path: Path) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text(_split("regression", 10).to_json(), encoding="utf-8")

    with pytest.raises(PhaseABundleError, match="row count"):
        load_phasea_split_manifest(
            split_path,
            task_type="regression",
            train_count=9,
            random_seed=17,
        )


def test_aggregate_validation_binds_bundle_candidate_and_split_identity(
    tmp_path: Path,
) -> None:
    raw = _raw_frame()
    target = _target("classification")
    estimator = _estimator("classification")
    recipe = load_baseline_recipe(ROOT, "classification")
    candidate_id = phasea_candidate_id("pycaret", estimator, recipe)
    evidence = _evidence(
        candidate_id=candidate_id,
        engine="pycaret",
        task_type="classification",
    )
    split = _split("classification", len(raw))
    artifact = fit_save_phasea_bundle(
        estimator,
        raw,
        target,
        task_type="classification",
        engine="pycaret",
        candidate_id=candidate_id,
        recipe=recipe,
        evidence=evidence,
        split_manifest=split,
        output_dir=tmp_path,
        random_seed=17,
        execution_id="execution-1",
        mlflow_parent_run_id="parent-1",
        mlflow_child_run_id="child-1",
    )
    manifest = {
        "schema_version": 2,
        "engine": "pycaret",
        "task_type": "classification",
        "candidate_id": candidate_id,
        "split_id": split.split_id,
        "status": "success",
        "raw_input_bundle_eligible": True,
        "evaluation": evidence,
        "model_bundle": dict(artifact.manifest),
    }

    bundle = validate_phasea_bundle_artifact(tmp_path, manifest)
    assert bundle.candidate_id == candidate_id
    selected = select_champion(
        manifest,
        None,
        task="classification",
        source_paths={"pycaret": tmp_path},
    )
    assert selected["candidate_id"] == candidate_id
    assert selected["bundle_id"] == bundle.bundle_id

    manifest["candidate_id"] = "different-candidate"
    with pytest.raises(PhaseABundleError, match="Evaluation candidate ID"):
        validate_phasea_bundle_artifact(tmp_path, manifest)
    rejected = select_champion(
        manifest,
        None,
        task="classification",
        source_paths={"pycaret": tmp_path},
    )
    assert rejected["source"] is None


def test_phasea_components_and_both_graphs_bind_stage2_contracts() -> None:
    pycaret_component = (ROOT / "components" / "stage5_pycaret_train.yml").read_text(
        encoding="utf-8"
    )
    flaml_component = (ROOT / "components" / "stage5_flaml_train.yml").read_text(
        encoding="utf-8"
    )
    pipeline_source = (ROOT / "pipelines" / "pipeline_builder.py").read_text(
        encoding="utf-8"
    )

    for component in (pycaret_component, flaml_component):
        assert "split_manifest:" in component
        assert "--split_manifest ${{inputs.split_manifest}}" in component
        assert "execution_manifest:" in component
        assert "--execution_manifest ${{inputs.execution_manifest}}" in component
    assert "version: 9" in pycaret_component
    assert "version: 10" in flaml_component
    assert pipeline_source.count("dataset_in=s2.outputs.raw_train_out") >= 4
    assert pipeline_source.count("split_manifest=s2.outputs.split_manifest_out") >= 4
    assert pipeline_source.count("execution_manifest=execution_manifest") >= 4


def test_flaml_clustering_remains_explicitly_skipped() -> None:
    source = (ROOT / "src" / "steps" / "stage5_flaml_train.py").read_text(
        encoding="utf-8"
    )

    assert 'if task_type == "clustering":' in source
    assert 'manifest["status"] = "skipped_unsupported"' in source
    assert 'manifest["eligibility_reason"] = "clustering_is_pycaret_only"' in source
