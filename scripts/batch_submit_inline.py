#!/usr/bin/env python3
"""Submit the standard batch through ``pipelines/submit_pipeline.py``."""

from __future__ import annotations

from _canonical_batch_submit import run_config_batch


CONFIGS = (
    "configs/config_classification_telecom_churn_azureml.yml",
    "configs/config_classification_telco_churn_azureml.yml",
    "configs/config_classification_credit_default_azureml.yml",
    "configs/config_classification_titanic_azureml.yml",
    "configs/config_classification_cardiac_arrest_azureml.yml",
    "configs/config_regression_college_azureml.yml",
    "configs/config_regression_insurance_azureml.yml",
    "configs/config_regression_house_sales_azureml.yml",
    "configs/config_regression_length_of_stay_azureml.yml",
    "configs/config_regression_medical_charges_azureml.yml",
)


if __name__ == "__main__":
    raise SystemExit(
        run_config_batch(
            CONFIGS,
            label="Standard MLOps batch",
            tags={"batch_submit": "canonical"},
        )
    )
