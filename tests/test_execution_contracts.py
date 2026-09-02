from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from orchestration.config_compiler import compile_config
from orchestration.contracts import (
    CandidateRecord,
    ContractValidationError,
    ExecutionManifest,
    QualityDecision,
    SplitManifest,
    canonical_hash,
    dataset_version_identity,
)
from steps.s06_phaseb_variant_runner import (
    bind_candidate_records_to_runtime_split,
    require_training_environment_hash,
    validate_execution_manifest_for_run,
)
from utils.recipe_catalog import normalize_recipe, semantic_recipe_hash


def _candidate(task_type: str = "classification", engine: str = "pycaret"):
    return CandidateRecord(
        task_type=task_type,
        recipe_id="recipe-1",
        recipe_hash="a" * 64,
        engine=engine,
        algorithm="engine_search",
        parameters={"scaling": {"method": "standard"}},
        split_id="b" * 64,
        data_version="sample@1",
        code_sha="c" * 40,
        environment_hash="d" * 64,
    )


def test_candidate_identity_is_stable_and_deeply_immutable() -> None:
    first = _candidate()
    second = CandidateRecord.from_dict(first.to_dict())

    assert first.candidate_id == second.candidate_id
    with pytest.raises(FrozenInstanceError):
        first.engine = "flaml"
    with pytest.raises(TypeError):
        first.parameters["new"] = "value"


def test_phaseb_requires_training_environment_identity() -> None:
    manifest = ExecutionManifest(
        config_hash="a" * 64,
        task_type="classification",
        dataset={"name": "sample", "version": "1"},
        split_policy={"strategy": "random"},
        engines=("pycaret",),
        recipe_paths=("classification/variant.yml",),
        recipe_ids=("recipe-1",),
        candidate_ids=("candidate-1",),
        budgets={"round1_max_variants": 1},
        code_sha="c" * 40,
        environment_hashes={},
        recipe_catalog_hash="e" * 64,
    )

    with pytest.raises(ValueError, match="environment_hashes.training"):
        require_training_environment_hash(manifest)


def test_split_manifest_detects_identity_tampering() -> None:
    split = SplitManifest(
        task_type="classification",
        strategy="stratified",
        random_seed=42,
        train_count=70,
        validation_count=10,
        test_count=20,
        train_ids_hash="train",
        validation_ids_hash="validation",
        test_ids_hash="test",
        data_version="sample@1",
    )
    payload = split.to_dict()
    payload["test_count"] = 21

    with pytest.raises(ContractValidationError, match="split_id"):
        SplitManifest.from_dict(payload)


def test_s10_component_and_pipeline_bind_stage2_split_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    component = yaml.safe_load(
        (root / "components" / "final_evaluation.yml").read_text(
            encoding="utf-8"
        )
    )
    command = component["command"]
    builder_source = (root / "pipelines" / "pipeline_builder.py").read_text(
        encoding="utf-8"
    )

    assert component["inputs"]["split_manifest_in"]["type"] == "uri_file"
    assert (
        "--split_manifest_in ${{inputs.split_manifest_in}}" in command
    )
    assert (
        builder_source.count(
            "split_manifest_in=s2.outputs.split_manifest_out"
        )
        == 2
    )


def test_s06_candidate_identity_binds_actual_runtime_split_manifest() -> None:
    submitted = _candidate()
    runtime_split = SplitManifest(
        task_type="classification",
        strategy="stratified",
        random_seed=42,
        train_count=70,
        validation_count=0,
        test_count=20,
        train_ids_hash="actual-train",
        validation_ids_hash=canonical_hash([]),
        test_ids_hash="actual-test",
        data_version="sample@1",
    )

    (runtime_candidate,) = bind_candidate_records_to_runtime_split(
        (submitted,),
        runtime_split,
    )

    assert runtime_candidate.split_id == runtime_split.split_id
    assert runtime_candidate.candidate_id != submitted.candidate_id
    assert runtime_candidate.to_dict()["parameters"] == submitted.to_dict()[
        "parameters"
    ]


def test_execution_manifest_binds_all_candidate_ids() -> None:
    candidate = _candidate()
    manifest = ExecutionManifest(
        config_hash="config",
        task_type="classification",
        dataset={"name": "sample", "version": "1"},
        split_policy={"locked_test": True},
        engines=("pycaret",),
        recipe_paths=("classification/variant.yml",),
        recipe_ids=("recipe-1",),
        candidate_ids=(candidate.candidate_id,),
        budgets={"round1_max_variants": 1, "round2_max_variants": 1},
        code_sha="c" * 40,
        environment_hashes={"training": "d" * 64},
        recipe_catalog_hash="e" * 64,
    )

    assert ExecutionManifest.from_json(manifest.to_json()).execution_id == (
        manifest.execution_id
    )
    assert manifest.execution_id == canonical_hash(manifest._identity_dict())


def test_clustering_rejects_flaml_candidate() -> None:
    with pytest.raises(ContractValidationError, match="PyCaret"):
        _candidate(task_type="clustering", engine="flaml")


def test_warning_can_register_exact_bundle_but_cannot_receive_alias() -> None:
    decision = QualityDecision(
        decision="warn",
        candidate_id="candidate",
        evaluated_bundle_hash="bundle",
        metric_name="balanced_accuracy",
        metric_value=0.48,
        threshold=0.50,
        registration_allowed=True,
        promotion_aliases=(),
        registration_tags={"warning": "quality"},
    )
    assert decision.registration_allowed is True
    assert decision.to_dict()["promotion_aliases"] == []
    assert QualityDecision.from_json(decision.to_json()).decision_hash == (
        decision.decision_hash
    )
    tampered = decision.to_dict()
    tampered["metric_value"] = 0.99
    with pytest.raises(ContractValidationError, match="decision_hash"):
        QualityDecision.from_dict(tampered)

    with pytest.raises(ContractValidationError, match="aliases"):
        QualityDecision(
            decision="warn",
            candidate_id="candidate",
            evaluated_bundle_hash="bundle",
            metric_name="balanced_accuracy",
            metric_value=0.48,
            threshold=0.50,
            registration_allowed=True,
            promotion_aliases=("champion",),
        )


def test_s06_validates_exact_config_recipe_engine_and_budget_binding(
    tmp_path,
) -> None:
    recipe = {
        "recipe_name": "manifest-recipe",
        "task_type": "classification",
        "stage3_preprocessing": {
            "imputation": {"method": "mean"},
            "encoding": {"categorical_method": "onehot"},
            "scaling": {"method": "none"},
            "imbalance_handling": {"method": "none"},
        },
        "stage4_feature_engineering": {
            "feature_selection": {"method": "correlation", "threshold": 0.85}
        },
        "variant_metadata": {"variant_id": "manifest-recipe"},
    }
    recipe_path = tmp_path / "variant.yml"
    recipe_path.write_text(
        yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8"
    )
    config = compile_config(
        {
            "schema_version": "2.0",
            "experiment_name": "manifest-test",
            "task_type": "classification",
            "preset": "production",
            "dataset": {
                "name": "sample",
                "version": "1",
                "blob_path": "datasets/sample.csv",
                "content_sha256": "f" * 64,
                "target_column": "target",
            },
            "phases": {
                "phase_b": {
                    "engines": ["pycaret"],
                    "max_variants": 1,
                    "time_budget_per_variant": 60,
                    "phase_timeout_seconds": 600,
                    "planner": {
                        "round1_max_variants": 1,
                        "round2_max_variants": 1,
                    },
                }
            },
        },
        source_name="config.yml",
    )
    recipe_id = semantic_recipe_hash(recipe, task_type="classification")
    code_sha = "c" * 40
    environment_hash = "d" * 64
    candidate = CandidateRecord(
        task_type="classification",
        recipe_id=recipe_id,
        recipe_hash=recipe_id,
        engine="pycaret",
        algorithm="engine_search",
        parameters=normalize_recipe(recipe, task_type="classification"),
        split_id=canonical_hash(config["split"]),
        data_version=dataset_version_identity(config["dataset"]),
        code_sha=code_sha,
        environment_hash=environment_hash,
    )
    budgets = {
        "round1_max_variants": 1,
        "round2_max_variants": 1,
        "proxy_prune_threshold": 0.5,
        "candidate_engine_timeout_seconds": 60,
        "phase_b_timeout_seconds": 600,
        "hpo_trials": 50,
        "hpo_timeout_seconds": 3600,
    }
    manifest = ExecutionManifest(
        config_hash=config["compiled_config_hash"],
        task_type="classification",
        dataset=config["dataset"],
        split_policy=config["split"],
        engines=("pycaret",),
        recipe_paths=("classification/variant.yml",),
        recipe_ids=(recipe_id,),
        candidate_ids=(candidate.candidate_id,),
        budgets=budgets,
        code_sha=code_sha,
        environment_hashes={"training": environment_hash},
        recipe_catalog_hash="e" * 64,
    )
    payload = {
        **manifest.to_dict(),
        "candidate_records": [candidate.to_dict()],
    }

    validated, expected_records = validate_execution_manifest_for_run(
        payload,
        config=config,
        requested_variant_paths=["classification/variant.yml"],
        resolved_variant_paths=[str(recipe_path)],
        engines=["pycaret"],
        round1_max_variants=1,
        round2_max_variants=1,
        proxy_prune_threshold=0.5,
        candidate_engine_timeout_seconds=60,
        phase_b_timeout_seconds=600,
    )
    assert validated.execution_id == manifest.execution_id
    assert [record.to_dict() for record in expected_records] == [
        candidate.to_dict()
    ]

    with pytest.raises(ValueError, match="budgets"):
        validate_execution_manifest_for_run(
            payload,
            config=config,
            requested_variant_paths=["classification/variant.yml"],
            resolved_variant_paths=[str(recipe_path)],
            engines=["pycaret"],
            round1_max_variants=1,
            round2_max_variants=2,
            proxy_prune_threshold=0.5,
            candidate_engine_timeout_seconds=60,
            phase_b_timeout_seconds=600,
        )
