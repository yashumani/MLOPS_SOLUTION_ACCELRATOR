#!/usr/bin/env bash
# Submit all 15 V3 pipeline jobs (5 classification + 5 regression + 5 clustering).
# Drift detection (s13) is auto-wired in pipeline_builder.py.
# Sequential submission (no --wait); --force skips duplicate-submission guards.
set -u

cd "$(dirname "$0")/.."

SUB="93044a08-5661-4f1b-b424-5eafe066a9d1"
RG="mvpv1"
WS="mlops-accelerator"
COMPUTE="mlopsv2computecluster"

LOG_DIR="logs/submit_all_15_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/SUMMARY.txt"
: > "$SUMMARY"

CONFIGS=(
  # Classification (5)
  configs/config_classification_cardiac_arrest_azureml.yml
  configs/config_classification_credit_default_azureml.yml
  configs/config_classification_telco_churn_azureml.yml
  configs/config_classification_telecom_churn_azureml.yml
  configs/config_classification_titanic_azureml.yml
  # Regression (5)
  configs/config_regression_college_azureml.yml
  configs/config_regression_house_sales_azureml.yml
  configs/config_regression_insurance_azureml.yml
  configs/config_regression_length_of_stay_azureml.yml
  configs/config_regression_medical_charges_azureml.yml
  # Clustering (5)
  configs/config_clustering_atp1d_azureml.yml
  configs/config_clustering_churn_uplift_azureml.yml
  configs/config_clustering_credit_default_azureml.yml
  configs/config_clustering_kidney_disease_azureml.yml
  configs/config_clustering_online_retail_azureml.yml
)

i=0
for cfg in "${CONFIGS[@]}"; do
  i=$((i+1))
  name=$(basename "$cfg" .yml)
  log="$LOG_DIR/${i}_${name}.log"
  echo "=========================================================================="
  echo "[$i/15] $(date +%H:%M:%S) submitting $name"
  echo "=========================================================================="
  python pipelines/submit_pipeline.py \
      --config "$cfg" \
      --subscription_id "$SUB" \
      --resource_group "$RG" \
      --workspace_name "$WS" \
      --compute "$COMPUTE" \
      --force \
      > "$log" 2>&1
  rc=$?
  job_name=$(grep -oE "Submitted job: [a-zA-Z0-9_-]+" "$log" | head -1 | awk '{print $3}')
  studio=$(grep -oE "https://ml.azure.com/runs/[a-zA-Z0-9_-]+[^[:space:]]*" "$log" | head -1)
  if [ -z "$job_name" ]; then
    job_name=$(grep -oE "azureml://[^[:space:]]+/jobs/[a-zA-Z0-9_-]+" "$log" | head -1)
  fi
  status=$([ $rc -eq 0 ] && echo OK || echo FAIL)
  printf "%-2s %-6s %-55s %s\n" "$i" "$status" "$name" "${job_name:-<no-job-id>}" | tee -a "$SUMMARY"
  if [ -n "$studio" ]; then echo "    $studio" | tee -a "$SUMMARY"; fi
done

echo
echo "=========================================================================="
echo "Logs: $LOG_DIR"
echo "Summary:"
cat "$SUMMARY"
