#!/usr/bin/env bash
# DRY-RUN submission plan for all 16 azureml configs on prod-hardening-20260425.
# Echoes the commands ONLY — does NOT call ml_client.jobs.create_or_update.
#
# Usage:
#   bash scripts/submit_all_dryrun.sh           # dry-run (default)
#   FIRE=1 bash scripts/submit_all_dryrun.sh    # actually submit (NOT RUN HERE)
set -euo pipefail

SUB="93044a08-5661-4f1b-b424-5eafe066a9d1"
RG="mvpv1"
WS="mlops-accelerator"
COMPUTE="mlopsv2computecluster"

CONFIGS=(
  configs/config_classification_cardiac_arrest_azureml.yml
  configs/config_classification_credit_default_azureml.yml
  configs/config_classification_telco_churn_azureml.yml
  configs/config_classification_telecom_churn_azureml.yml
  configs/config_classification_titanic_azureml.yml
  configs/config_clustering_atp1d_azureml.yml
  configs/config_clustering_churn_uplift_azureml.yml
  configs/config_clustering_credit_default_azureml.yml
  configs/config_clustering_kidney_disease_azureml.yml
  configs/config_clustering_online_retail_azureml.yml
  configs/config_clustering_online_retail_ii_azureml.yml
  configs/config_regression_college_azureml.yml
  configs/config_regression_house_sales_azureml.yml
  configs/config_regression_insurance_azureml.yml
  configs/config_regression_length_of_stay_azureml.yml
  configs/config_regression_medical_charges_azureml.yml
)

FIRE="${FIRE:-0}"
echo "DRY-RUN=$([ "$FIRE" = "1" ] && echo NO || echo YES)  | configs=${#CONFIGS[@]}"
echo

for cfg in "${CONFIGS[@]}"; do
  cmd="python pipelines/submit_pipeline.py \
    --config $cfg \
    --subscription_id $SUB --resource_group $RG --workspace_name $WS \
    --compute $COMPUTE"
  if [ "$FIRE" = "1" ]; then
    echo ">>> SUBMIT: $cfg"
    eval "$cmd"
  else
    echo "[dry-run] $cmd"
  fi
done
