#!/usr/bin/env python3
"""Resubmit the historical failed set through the canonical submitter."""

from __future__ import annotations

from _canonical_batch_submit import run_config_batch


CONFIGS = (
    "configs/config_regression_college_azureml.yml",
    "configs/config_regression_insurance_azureml.yml",
    "configs/config_regression_house_sales_azureml.yml",
    "configs/config_regression_medical_charges_azureml.yml",
    "configs/config_regression_length_of_stay_azureml.yml",
    "configs/config_classification_telco_churn_azureml.yml",
)


if __name__ == "__main__":
    raise SystemExit(
        run_config_batch(
            CONFIGS,
            label="Historical failed-job replay",
            tags={"resubmit": "canonical_current_revision"},
        )
    )
