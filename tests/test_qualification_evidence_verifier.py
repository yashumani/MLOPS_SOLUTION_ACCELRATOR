from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_qualification_evidence.py"
SCENARIO = "classification-healthcare-heart-disease"
CONTENT_SHA = "d" * 64
SCHEMA_SHA = "e" * 64
EXECUTION_ID = "a" * 64
CONFIG_HASH = "b" * 64
CODE_SHA = "c" * 64
SPLIT_ID = "1" * 64
SOURCE_BUNDLE_ID = "2" * 64
BUNDLE_ID = "3" * 64
DECISION_HASH = "4" * 64
GIT_COMMIT = "5" * 40


def _load_module():
    spec = importlib.util.spec_from_file_location("qualification_evidence_verifier", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    catalog = tmp_path / "catalog.yml"
    catalog.write_text(
        "\n".join(
            [
                "schema_version: '1.0'",
                "profile_run_id: test-profile",
                "configs:",
                f"- scenario_id: {SCENARIO}",
                "  task_type: classification",
                "  industry: healthcare",
                "  config_path: configs/test.yml",
                f"  dataset_content_sha256: {CONTENT_SHA}",
                f"  dataset_schema_sha256: {SCHEMA_SHA}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    split = {
        "schema_version": "2.0",
        "task_type": "classification",
        "locked_test": True,
        "split_id": SPLIT_ID,
        "data_version": "asset@1:path:hash",
        "strategy": "stratified",
        "random_seed": 42,
        "train_count": 80,
        "train_ids_hash": "6" * 64,
        "validation_count": 0,
        "validation_ids_hash": "7" * 64,
        "test_count": 20,
        "test_ids_hash": "8" * 64,
        "group_column": None,
        "time_column": None,
    }
    execution = {
        "schema_version": "2.0",
        "task_type": "classification",
        "execution_id": EXECUTION_ID,
        "config_hash": CONFIG_HASH,
        "code_sha": CODE_SHA,
        "dataset": {"content_sha256": CONTENT_SHA},
        "runtime_split_id": SPLIT_ID,
        "split_manifest": split,
    }
    selection = {
        "key": "phaseb",
        "score": 0.8,
        "source": "cross_validation_or_validation",
        "locked_test_used_for_selection": False,
        "candidate_id": "candidate-1",
        "source_bundle_id": SOURCE_BUNDLE_ID,
    }
    final_test = {
        "evaluated_once": True,
        "candidate_phase": "phaseb",
        "candidate_id": "candidate-1",
        "source_bundle_id": SOURCE_BUNDLE_ID,
        "metrics": {"balanced_accuracy": 0.81},
        "row_count": 20,
    }
    lineage = {
        "execution_id": EXECUTION_ID,
        "config_hash": CONFIG_HASH,
        "code_sha": CODE_SHA,
        "split_id": SPLIT_ID,
        "source_candidate_id": "candidate-1",
        "source_bundle_id": SOURCE_BUNDLE_ID,
        "parent_run_id": "parent-run",
        "final_evaluation_run_id": "final-run",
        "candidate_lineage": {
            "execution_id": EXECUTION_ID,
            "parent_run_id": "parent-run",
            "candidate_run_id": "candidate-run",
        },
    }
    quality = {
        "schema_version": "2.0",
        "decision": "pass",
        "candidate_id": "candidate-1",
        "evaluated_bundle_hash": BUNDLE_ID,
        "metric_name": "balanced_accuracy",
        "metric_value": 0.81,
        "threshold": 0.5,
        "registration_allowed": True,
        "decision_hash": DECISION_HASH,
    }
    bundle = {
        "bundle_schema_version": 5,
        "task_type": "classification",
        "candidate_id": "candidate-1",
        "input_schema": {"column_order": ["feature"]},
        "recipe": {"steps": []},
        "selection_metrics": {
            "selection_score": 0.8,
            "source": "cross_validation_or_validation",
        },
        "final_test_metrics": final_test["metrics"],
        "environment": {"python": "3.10"},
        "lineage": lineage,
        "dependencies": ["scikit-learn"],
        "signature": {"inputs": ["feature"], "outputs": ["prediction"]},
        "input_example": [{"feature": 1.0}],
        "model_state_sha256": "9" * 64,
        "bundle_id": BUNDLE_ID,
        "artifact_file": "model_bundle.pkl",
        "artifact_sha256": "f" * 64,
    }
    final = {
        "schema_version": 2,
        "task": "classification",
        "test_samples": 20,
        "champion_valid": True,
        "quality_gate_passed": True,
        "quality_decision": quality,
        "quality_threshold": 0.5,
        "holdout_source": "stage2_split_manifest_bound_component_input",
        "execution_manifest": {
            "schema_version": "2.0",
            "task_type": "classification",
            "execution_id": EXECUTION_ID,
            "config_hash": CONFIG_HASH,
            "code_sha": CODE_SHA,
        },
        "split_manifest": {
            "split_id": SPLIT_ID,
            "data_version": split["data_version"],
            "test_count": 20,
            "test_ids_hash": split["test_ids_hash"],
        },
        "selection": selection,
        "final_test": final_test,
        "lineage": lineage,
        "validation": {"valid": True, "errors": []},
        "model_bundle": bundle,
        "output_validation": {
            "valid": True,
            "files": [
                {"name": "model_bundle.pkl", "size": 10},
                {"name": "model_bundle_manifest.json", "size": 10},
            ],
            "errors": [],
        },
    }
    model_name = f"mlops-v3-qualification-{SCENARIO}"
    registry = {
        "model_name": model_name,
        "version": "1",
        "model_uri": f"models:/{model_name}/1",
        "stage": "None",
        "lifecycle_stage": "Unassigned",
        "quality_decision": "pass",
        "promotion_allowed": True,
        "promotion_mode": "manual",
        "promotion_performed": False,
        "requested_promotion_aliases": [],
        "task_type": "classification",
        "registration_run_id": "registration-run",
        "execution_id": EXECUTION_ID,
        "config_hash": CONFIG_HASH,
        "code_sha": CODE_SHA,
        "dataset_content_sha256": CONTENT_SHA,
        "registration_backend": "mlflow",
    }
    outputs = {
        "execution_manifest": execution,
        "split_manifest": split,
        "quality_decision": {
            "schema_version": "2.0",
            "decision": "block",
            "registration_allowed": False,
            "threshold": None,
            "registration_tags": {
                "quality_stage": "selection_only",
                "locked_test_evaluated": "false",
            },
        },
        "final_report": final,
        "registry_info": registry,
        "drift_report": {
            "task_type": "classification",
            "identity": {
                "execution_id": EXECUTION_ID,
                "config_hash": CONFIG_HASH,
                "model_name": model_name,
            },
        },
        "retrain_decision": {
            "decision_id": "decision-1",
            "task_type": "classification",
            "identity": {
                "execution_id": EXECUTION_ID,
                "config_hash": CONFIG_HASH,
                "source_sha": CODE_SHA,
                "model_name": model_name,
            },
            "revision_validation": {
                "status": "verified",
                "required_contract": "final_report.execution_manifest",
                "contract_error": None,
                "required_contract_present": True,
                "missing_fields": [],
                "conflicts": {},
            },
            "source_revision": {
                "schema_version": "1.0",
                "execution_id": EXECUTION_ID,
                "config_hash": CONFIG_HASH,
                "source_sha": CODE_SHA,
            },
            "decision": {"outcome": "refresh_baseline", "should_submit": False},
        },
        "decision_ledger_record": {
            "decision_id": "decision-1",
            "task_type": "classification",
            "promotion_mode": "manual",
            "promotion_status": "manual_pending",
            "outcome": "refresh_baseline",
            "metadata": {
                "source_revision": {
                    "schema_version": "1.0",
                    "execution_id": EXECUTION_ID,
                    "config_hash": CONFIG_HASH,
                    "source_sha": CODE_SHA,
                },
                "revision_validation": {
                    "status": "verified",
                    "required_contract": "final_report.execution_manifest",
                    "contract_error": None,
                    "required_contract_present": True,
                    "missing_fields": [],
                    "conflicts": {},
                },
                "identity": {
                    "execution_id": EXECUTION_ID,
                    "config_hash": CONFIG_HASH,
                    "source_sha": CODE_SHA,
                    "model_name": model_name,
                }
            },
        },
    }
    pipeline = tmp_path / "pipeline"
    for name, value in outputs.items():
        _write_json(pipeline / "named-outputs" / name / name, value)

    smoke_submission = {
        "schema_version": "1.0",
        "job_name": "smoke-job",
        "status": "Completed",
        "parent_job": "parent-job",
        "scenario_id": SCENARIO,
        "model_uri": registry["model_uri"],
        "execution_id": EXECUTION_ID,
        "code_sha": CODE_SHA,
        "source_git_commit": GIT_COMMIT,
        "environment": "azureml:test:1",
        "compute": "test-cluster",
        "output_datastore": "mlops_blob",
        "evidence_uri": "azureml://datastores/mlops_blob/paths/evidence",
    }
    smoke_evidence = {
        "schema_version": "1.0",
        "status": "passed",
        "azureml_run_id": "smoke-job",
        "model_name": model_name,
        "model_version": "1",
        "model_uri": registry["model_uri"],
        "execution_id": EXECUTION_ID,
        "code_sha": CODE_SHA,
        "dataset_content_sha256": CONTENT_SHA,
        "registration_run_id": "registration-run",
        "lineage_tags_verified": True,
        "input_type": "DataFrame",
        "input_rows": 2,
        "prediction_rows": 2,
        "aliases": [],
        "current_stage": "None",
        "signature": {"inputs": "feature", "outputs": "prediction"},
    }
    _write_json(tmp_path / "smoke-submission.json", smoke_submission)
    _write_json(tmp_path / "smoke-evidence" / "registered_model_inference_smoke.json", smoke_evidence)
    _write_json(
        tmp_path / "monitor-summary.json",
        {
            "schema_version": "1.0",
            "state": "passed",
            "expected_count": 1,
            "accepted_submission_count": 1,
            "submission_failures": [],
            "jobs": [
                {
                    "label": SCENARIO,
                    "job_id": "parent-job",
                    "status": "Completed",
                    "query_error": None,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "data-assets.json",
        {
            "schema_version": "1.0",
            "all_passed": True,
            "errors": [],
            "scenario_count": 1,
            "assets": [
                {
                    "scenario_ids": [SCENARIO],
                    "content_sha256": CONTENT_SHA,
                    "schema_sha256": SCHEMA_SHA,
                    "checks": {
                        "version": True,
                        "datastore_path": True,
                        "scope": True,
                        "hashes": True,
                        "scenario_tags": True,
                    },
                }
            ],
        },
    )
    manifest = tmp_path / "evidence-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "1.0",
            "monitor_summaries": ["monitor-summary.json"],
            "data_asset_audit": "data-assets.json",
            "scenarios": [
                {
                    "scenario_id": SCENARIO,
                    "parent_job": "parent-job",
                    "pipeline_evidence_dir": "pipeline",
                    "registered_model_smoke_submission": "smoke-submission.json",
                    "registered_model_smoke_evidence": "smoke-evidence",
                }
            ],
        },
    )
    return manifest, catalog, pipeline


def _codes(report: dict) -> set[str]:
    codes = {item["code"] for item in report["global_issues"]}
    for scenario in report["scenarios"]:
        codes.update(item["code"] for item in scenario["issues"])
    return codes


def test_accepts_terminal_single_scenario_artifact_contract(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, _ = _write_fixture(tmp_path)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "passed"
    assert report["accepted_scenario_count"] == 1
    assert report["release_matrix_accepted"] is False
    assert report["runtime_source_sha256_values"] == [CODE_SHA]


@pytest.mark.parametrize("schema_version", [3, 4, 5])
def test_accepts_supported_model_bundle_schemas(
    tmp_path: Path, schema_version: int,
) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    final_path = pipeline / "named-outputs" / "final_report" / "final_report"
    final = _read_json(final_path)
    final["model_bundle"]["bundle_schema_version"] = schema_version
    _write_json(final_path, final)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "passed"
    assert report["accepted_scenario_count"] == 1


def test_accepts_current_model_bundle_default_schema(tmp_path: Path) -> None:
    from utils.model_bundle import ModelBundle

    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    final_path = pipeline / "named-outputs" / "final_report" / "final_report"
    final = _read_json(final_path)
    final["model_bundle"]["bundle_schema_version"] = ModelBundle.bundle_schema_version
    _write_json(final_path, final)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "passed"


@pytest.mark.parametrize("schema_version", [None, True, False, "5", 5.0, 0, 2, 6, {}, []])
def test_rejects_unsupported_or_malformed_model_bundle_schemas(
    tmp_path: Path, schema_version: object,
) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    final_path = pipeline / "named-outputs" / "final_report" / "final_report"
    final = _read_json(final_path)
    final["model_bundle"]["bundle_schema_version"] = schema_version
    _write_json(final_path, final)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert report["accepted_scenario_count"] == 0
    assert "bundle_schema" in _codes(report)


def test_rejects_missing_model_bundle_schema(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    final_path = pipeline / "named-outputs" / "final_report" / "final_report"
    final = _read_json(final_path)
    del final["model_bundle"]["bundle_schema_version"]
    _write_json(final_path, final)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert "bundle_schema" in _codes(report)


def test_rejects_locked_test_used_for_selection(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    final_path = pipeline / "named-outputs" / "final_report" / "final_report"
    final = _read_json(final_path)
    final["selection"]["locked_test_used_for_selection"] = True
    _write_json(final_path, final)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert "selection_holdout_isolation" in _codes(report)


def test_rejects_stage6_registration_decision(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    quality_path = pipeline / "named-outputs" / "quality_decision" / "quality_decision"
    quality = _read_json(quality_path)
    quality["decision"] = "pass"
    quality["registration_allowed"] = True
    _write_json(quality_path, quality)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert {"s06_selection_only", "s06_registration"}.issubset(_codes(report))


def test_rejects_registry_to_smoke_identity_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, _ = _write_fixture(tmp_path)
    smoke_path = tmp_path / "smoke-evidence" / "registered_model_inference_smoke.json"
    smoke = _read_json(smoke_path)
    smoke["model_version"] = "2"
    _write_json(smoke_path, smoke)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert "smoke_model_version" in _codes(report)


def test_rejects_missing_named_output(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    (pipeline / "named-outputs" / "drift_report" / "drift_report").unlink()

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert "named_output" in _codes(report)


def test_complete_matrix_mode_requires_full_catalog_and_release_identity(
    tmp_path: Path,
) -> None:
    module = _load_module()
    manifest, catalog, _ = _write_fixture(tmp_path)

    report = module.verify_qualification_evidence(
        manifest,
        catalog_path=catalog,
        require_complete_matrix=True,
    )

    assert report["state"] == "failed"
    assert report["release_matrix_accepted"] is False
    assert {"release_candidate_required", "matrix_task_count"}.issubset(_codes(report))


def test_release_candidate_identity_is_enforced_for_partial_wave(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, _ = _write_fixture(tmp_path)
    payload = _read_json(manifest)
    payload["release_candidate"] = {
        "git_commit": "0" * 40,
        "runtime_source_sha256": "1" * 64,
    }
    _write_json(manifest, payload)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert {"release_runtime_mismatch", "release_git_mismatch"}.issubset(_codes(report))


def test_rejects_duplicate_scenario_and_parent(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, _ = _write_fixture(tmp_path)
    payload = _read_json(manifest)
    payload["scenarios"].append(dict(payload["scenarios"][0]))
    _write_json(manifest, payload)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert {"duplicate_scenario", "duplicate_parent_job"}.issubset(_codes(report))


def test_rejects_failed_parent_even_when_summary_claims_passed(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, _ = _write_fixture(tmp_path)
    path = tmp_path / "monitor-summary.json"
    payload = _read_json(path)
    payload["jobs"][0]["status"] = "Failed"
    _write_json(path, payload)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert "monitor_job_status" in _codes(report)


def test_rejects_data_asset_schema_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, _ = _write_fixture(tmp_path)
    path = tmp_path / "data-assets.json"
    payload = _read_json(path)
    payload["assets"][0]["schema_sha256"] = "0" * 64
    _write_json(path, payload)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert "asset_schema_hash" in _codes(report)


@pytest.mark.parametrize("field,value", [("champion_valid", 1), ("quality_decision", "pass"), ("output_validation", {"valid": True, "files": None})])
def test_malformed_final_evidence_fails_without_crashing(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    path = pipeline / "named-outputs" / "final_report" / "final_report"
    payload = _read_json(path)
    payload[field] = value
    _write_json(path, payload)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"


def test_warn_only_quality_can_register_without_promotion(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    final_path = pipeline / "named-outputs" / "final_report" / "final_report"
    final = _read_json(final_path)
    final["quality_decision"]["decision"] = "warn"
    final["quality_gate_passed"] = False
    _write_json(final_path, final)
    registry_path = pipeline / "named-outputs" / "registry_info" / "registry_info"
    registry = _read_json(registry_path)
    registry["quality_decision"] = "warn"
    _write_json(registry_path, registry)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "passed"


def test_rejects_promotion_in_qualification_evidence(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    path = pipeline / "named-outputs" / "registry_info" / "registry_info"
    registry = _read_json(path)
    registry["promotion_performed"] = True
    _write_json(path, registry)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert "registry_promotion" in _codes(report)


@pytest.mark.parametrize("output_name", ["quality_decision", "drift_report", "retrain_decision", "decision_ledger_record"])
def test_rejects_empty_required_artifact(tmp_path: Path, output_name: str) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    _write_json(pipeline / "named-outputs" / output_name / output_name, {})

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert "named_output" in _codes(report)


def test_quality_score_must_come_from_locked_test(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    path = pipeline / "named-outputs" / "final_report" / "final_report"
    final = _read_json(path)
    final["quality_decision"]["metric_value"] = final["selection"]["score"]
    _write_json(path, final)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert "quality_metric_binding" in _codes(report)


def test_rejects_ledger_from_another_source_revision(tmp_path: Path) -> None:
    module = _load_module()
    manifest, catalog, pipeline = _write_fixture(tmp_path)
    path = pipeline / "named-outputs" / "decision_ledger_record" / "decision_ledger_record"
    ledger = _read_json(path)
    ledger["metadata"]["source_revision"]["source_sha"] = "0" * 64
    _write_json(path, ledger)

    report = module.verify_qualification_evidence(manifest, catalog_path=catalog)

    assert report["state"] == "failed"
    assert "ledger_source_revision" in _codes(report)


def _replace_strings(value: object, replacements: dict[str, str]) -> object:
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            str(_replace_strings(key, replacements)): _replace_strings(item, replacements)
            for key, item in value.items()
        }
    return value


def test_complete_matrix_accepts_only_all_fifteen_exact_candidate_scenarios(
    tmp_path: Path,
) -> None:
    module = _load_module()
    configs = []
    entries = []
    monitors = []
    assets = []
    for task in module.TASK_TYPES:
        for industry in ("healthcare", "finance", "retail", "energy", "education"):
            scenario_id = f"{task}-{industry}-qualification"
            case_dir = tmp_path / scenario_id
            case_dir.mkdir()
            case_manifest, _, _ = _write_fixture(case_dir)
            replacements = {
                SCENARIO: scenario_id,
                "classification": task,
                EXECUTION_ID: hashlib.sha256(scenario_id.encode()).hexdigest(),
                "parent-job": f"parent-{scenario_id}",
                "smoke-job": f"smoke-{scenario_id}",
                "balanced_accuracy": module.QUALITY_METRICS[task],
            }
            for path in case_dir.rglob("*"):
                if path.is_file() and path.suffix != ".yml":
                    value = _replace_strings(_read_json(path), replacements)
                    assert isinstance(value, dict)
                    _write_json(path, value)
            configs.append(
                {
                    "scenario_id": scenario_id,
                    "task_type": task,
                    "industry": industry,
                    "dataset_content_sha256": CONTENT_SHA,
                    "dataset_schema_sha256": SCHEMA_SHA,
                }
            )
            entry = _read_json(case_manifest)["scenarios"][0]
            for field in ("pipeline_evidence_dir", "registered_model_smoke_submission", "registered_model_smoke_evidence"):
                entry[field] = str(case_dir / entry[field])
            entries.append(entry)
            monitors.append(str(case_dir / "monitor-summary.json"))
            assets.extend(_read_json(case_dir / "data-assets.json")["assets"])
    catalog = tmp_path / "complete-catalog.yml"
    _write_json(catalog, {"schema_version": "1.0", "configs": configs})
    _write_json(
        tmp_path / "complete-assets.json",
        {
            "schema_version": "1.0",
            "scenario_count": 15,
            "all_passed": True,
            "errors": [],
            "assets": assets,
        },
    )
    manifest = tmp_path / "complete-manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "1.0",
            "monitor_summaries": monitors,
            "data_asset_audit": "complete-assets.json",
            "release_candidate": {
                "git_commit": GIT_COMMIT,
                "runtime_source_sha256": CODE_SHA,
            },
            "scenarios": entries,
        },
    )

    report = module.verify_qualification_evidence(
        manifest,
        catalog_path=catalog,
        require_complete_matrix=True,
    )

    assert report["state"] == "passed", report
    assert report["accepted_scenario_count"] == 15
    assert report["release_matrix_accepted"] is True
    assert report["matrix"]["task_counts"] == dict.fromkeys(module.TASK_TYPES, 5)
    assert report["matrix"]["industry_counts"] == dict.fromkeys(module.TASK_TYPES, 5)
