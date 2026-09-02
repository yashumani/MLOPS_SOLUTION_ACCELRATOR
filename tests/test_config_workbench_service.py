from __future__ import annotations

from api.services.config_service import preview_config, validate_content


def _valid_config() -> dict:
    return {
        "experiment_name": "classification_test_v3",
        "preset": "production",
        "task_type": "classification",
        "dataset": {
            "name": "telecom_churn",
            "target_column": "churn",
            "blob_path": "telecom_churn.csv",
            "datastore_name": "mlops_blob",
        },
        "azureml": {"compute_target": "mlopsv2computecluster"},
        "phases": {
            "phase_a_baseline": {"engines": ["pycaret", "flaml"]},
            "phase_b": {"max_variants": 12, "engines": ["pycaret"]},
            "phase_c_hpo": {"optimizer": "optuna", "n_trials": 25, "timeout_seconds": 1800},
        },
        "recipes": [{"file": "recipes/baseline_recipe.yml"}],
    }


def test_validate_content_reports_valid_config() -> None:
    result = validate_content(_valid_config())

    assert result.valid is True
    assert result.errors == []


def test_validate_content_reports_cross_field_error() -> None:
    config = _valid_config()
    config["dataset"].pop("target_column")

    result = validate_content(config)

    assert result.valid is False
    assert any("target_column" in issue.message for issue in result.errors)


def test_preview_config_builds_s01_s09_plan() -> None:
    preview = preview_config(_valid_config(), config_name="config_test_azureml")

    assert preview.valid is True
    assert preview.config_name == "config_test_azureml"
    assert preview.task_type == "classification"
    assert preview.dataset_name == "telecom_churn"
    assert preview.dataset_uri_preview == "azureml://datastores/mlops_blob/paths/telecom_churn.csv"
    assert preview.baseline_engines == ["pycaret", "flaml"]
    assert preview.phase_b_engines == ["pycaret"]
    assert preview.phase_b_variant_budget == 12
    assert preview.phase_c_trials == 25
    assert [stage.stage_id for stage in preview.stage_plan] == [
        "S01",
        "S02",
        "S03",
        "S04",
        "S05a",
        "S05b",
        "S05t",
        "S05z",
        "S06",
        "S08",
        "S09",
    ]