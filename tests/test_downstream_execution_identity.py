from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from orchestration.config_compiler import compile_config
from orchestration.contracts import ContractValidationError, ExecutionManifest
from orchestration.execution_identity import (
    validate_execution_manifest_binding,
)
from steps.s12_model_registration import (
    validate_registration_execution_binding,
)


def _compiled_config() -> dict:
    return compile_config(
        {
            "schema_version": "2.0",
            "experiment_name": "execution-contract-test",
            "preset": "production",
            "task_type": "classification",
            "dataset": {
                "name": "sample",
                "version": "1",
                "target_column": "target",
            },
        }
    )


def _manifest(config: dict) -> ExecutionManifest:
    return ExecutionManifest(
        config_hash=config["compiled_config_hash"],
        task_type=config["task_type"],
        dataset=config["dataset"],
        split_policy=config["split"],
        engines=("pycaret",),
        recipe_paths=("classification/baseline_recipe.yml",),
        recipe_ids=("recipe-id",),
        candidate_ids=("candidate-id",),
        budgets={"round1_max_variants": 1},
        code_sha="code-sha",
        environment_hashes={"training": "environment-hash"},
        recipe_catalog_hash="catalog-hash",
    )


def test_downstream_manifest_binds_exact_compiled_config(tmp_path: Path) -> None:
    config = _compiled_config()
    manifest = _manifest(config)
    path = tmp_path / "execution_manifest.json"
    enriched = manifest.to_dict()
    enriched["realized_candidate_records"] = [{"candidate_id": "realized"}]
    enriched["runtime_candidate_ids"] = ["runtime-candidate"]
    enriched["runtime_candidate_records"] = [{"candidate_id": "runtime-candidate"}]
    enriched["runtime_split_id"] = "runtime-split"
    enriched["split_manifest"] = {"split_id": "runtime-split"}
    path.write_text(json.dumps(enriched), encoding="utf-8")

    assert validate_execution_manifest_binding(path, config) == manifest

    changed = dict(config)
    changed["compiled_config_hash"] = "different-config"
    with pytest.raises(RuntimeError, match="config_hash"):
        validate_execution_manifest_binding(path, changed)


def test_registration_requires_report_and_bundle_from_exact_execution() -> None:
    manifest = _manifest(_compiled_config())
    lineage = {
        "execution_id": manifest.execution_id,
        "config_hash": manifest.config_hash,
        "code_sha": manifest.code_sha,
    }
    report = {
        "execution_manifest": manifest.to_dict(),
        "lineage": dict(lineage),
    }
    bundle = SimpleNamespace(lineage=dict(lineage))

    validate_registration_execution_binding(report, bundle, manifest)

    bundle.lineage["code_sha"] = "other-source"
    with pytest.raises(ContractValidationError, match="code_sha"):
        validate_registration_execution_binding(report, bundle, manifest)


def test_execution_manifest_is_wired_through_all_downstream_components() -> None:
    root = Path(__file__).resolve().parents[1]
    phasec = yaml.safe_load(
        (root / "components/phasec_optuna_hpo.yml").read_text(encoding="utf-8")
    )
    assert phasec["version"] == 14
    final = yaml.safe_load(
        (root / "components/final_evaluation.yml").read_text(encoding="utf-8")
    )
    registration = yaml.safe_load(
        (root / "components/s12_model_registration.yml").read_text(
            encoding="utf-8"
        )
    )
    builder = (root / "pipelines/pipeline_builder.py").read_text(
        encoding="utf-8"
    )

    assert phasec["inputs"]["execution_manifest"]["type"] == "uri_file"
    assert final["version"] == 16
    assert final["inputs"]["execution_manifest_in"]["type"] == "uri_file"
    assert registration["version"] == 13
    assert registration["inputs"]["execution_manifest"]["type"] == "uri_file"
    assert builder.count(
        "execution_manifest=s06.outputs.execution_manifest_out"
    ) == 4
    assert builder.count(
        "execution_manifest_in=s06.outputs.execution_manifest_out"
    ) == 2
    assert "timeseries_train" not in builder
