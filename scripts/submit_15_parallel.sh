#!/usr/bin/env bash
# Submit all 15 Azure ML pipeline jobs in parallel on prod-hardening-20260425.
# Generated: 2026-04-30
# Azure context discovered from MLFLOW_TRACKING_URI + ~/.azure/clouds.config.
set -uo pipefail  # NOTE: -e omitted intentionally so one failure doesn't kill the loop

ROOT="/home/azureuser/cloudfiles/code/Users/yashu.savyminds/mlops-solution-accelerator-v3"
SUB="93044a08-5661-4f1b-b424-5eafe066a9d1"
RG="mvpv1"
WS="mlops-accelerator"
COMPUTE="mlopsv2computecluster"
LOG_DIR="${ROOT}/logs/parallel_submit_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

CONFIGS=(
  config_classification_cardiac_arrest_azureml.yml
  config_classification_credit_default_azureml.yml
  config_classification_telco_churn_azureml.yml
  config_classification_telecom_churn_azureml.yml
  config_classification_titanic_azureml.yml
  config_clustering_atp1d_azureml.yml
  config_clustering_churn_uplift_azureml.yml
  config_clustering_credit_default_azureml.yml
  config_clustering_kidney_disease_azureml.yml
  config_clustering_online_retail_azureml.yml
  config_clustering_online_retail_ii_azureml.yml
  config_regression_college_azureml.yml
  config_regression_house_sales_azureml.yml
  config_regression_insurance_azureml.yml
  config_regression_length_of_stay_azureml.yml
)

cd "$ROOT"
echo "Submitting ${#CONFIGS[@]} jobs in parallel. Logs: $LOG_DIR"
PIDS=()
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yml)
  log="$LOG_DIR/${name}.log"
  echo "  ➜ $name"
  # --force bypasses the lock file & active-job check (required for parallel fan-out).
  # No --wait → submit-and-detach; --stop_compute is a no-op without --wait so omit.
  nohup python pipelines/submit_pipeline.py \
    --config "configs/$cfg" \
    --subscription_id "$SUB" \
    --resource_group "$RG" \
    --workspace_name "$WS" \
    --compute "$COMPUTE" \
    --force \
    > "$log" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "Spawned ${#PIDS[@]} background submitters. PIDs: ${PIDS[*]}"
echo "Waiting for all submitters to finish enqueueing (not for jobs to complete)..."
FAIL=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAIL=$((FAIL+1)); fi
done

echo ""
echo "================================================================"
echo "Submission summary: $((${#PIDS[@]} - FAIL))/${#PIDS[@]} succeeded, $FAIL failed."
echo "Log dir: $LOG_DIR"
echo "Job URLs:"
grep -h "Studio URL" "$LOG_DIR"/*.log 2>/dev/null | sort -u
echo "================================================================"
exit $FAIL
