#!/bin/bash
# ==============================================================================
# PIPELINE MONITORING SCRIPT
# ==============================================================================
# Monitors Azure ML pipeline execution for critical fixes validation
# Usage: ./monitor_pipeline.sh <job_name>
# Example: ./monitor_pipeline.sh sharp_celery_lkdg3h7tqz

set -e

JOB_NAME=${1:-""}
SUBSCRIPTION_ID="93044a08-5661-4f1b-b424-5eafe066a9d1"
RESOURCE_GROUP="mlops-accelerator-rg"
WORKSPACE="mlops-accelerator"

if [ -z "$JOB_NAME" ]; then
    echo "❌ ERROR: Job name required"
    echo "Usage: $0 <job_name>"
    exit 1
fi

echo "="*80
echo "🔍 PIPELINE MONITORING - CRITICAL FIXES VALIDATION"
echo "="*80
echo "📋 Job: $JOB_NAME"
echo "📋 Workspace: $WORKSPACE"
echo ""

# ==============================================================================
# STAGE 3: SMOTE Validation
# ==============================================================================
echo "📊 STAGE 3: SMOTE IMPLEMENTATION"
echo "-"*80

echo "⏳ Checking for SMOTE logs..."
az ml job stream --name "$JOB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE" \
    2>/dev/null | grep -A 10 "IMBALANCE HANDLING" || echo "⚠️  No SMOTE logs yet"

echo ""
echo "✅ Expected Output:"
echo "   ⚖️  IMBALANCE HANDLING: method=smote"
echo "   📊 Original distribution: Class 0: 194,741 (79.95%)"
echo "   🔄 Applying SMOTE..."
echo "   ✅ SMOTE completed successfully!"
echo "   📊 New imbalance ratio: 1.000"
echo ""

# ==============================================================================
# STAGE 7: FLAML Artifact Export
# ==============================================================================
echo "📊 STAGE 7: FLAML ARTIFACT EXPORT"
echo "-"*80

echo "⏳ Checking for FLAML iteration export..."
az ml job stream --name "$JOB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE" \
    2>/dev/null | grep -E "Extracting FLAML|Processed:|Recipe iterations" || echo "⚠️  No FLAML logs yet"

echo ""
echo "✅ Expected Output:"
echo "   📊 Extracting FLAML iteration history..."
echo "   ✅ Processed: 205/207 trials"
echo "   ✅ Recipe iterations: outputs/phaseb_recipe_flaml_iterations.csv"
echo ""

# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================
echo "📊 PERFORMANCE METRICS VALIDATION"
echo "-"*80

echo "⏳ Checking for Recall > 0%..."
az ml job stream --name "$JOB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE" \
    2>/dev/null | grep -i "recall" | tail -5 || echo "⚠️  No recall metrics yet"

echo ""
echo "✅ Expected:"
echo "   - Recall > 0.60 (CRITICAL)"
echo "   - F1-Score > 0.65"
echo "   - AUC-ROC > 0.80"
echo ""

# ==============================================================================
# ARTIFACT DOWNLOAD
# ==============================================================================
echo "📦 ARTIFACT DOWNLOAD INSTRUCTIONS"
echo "-"*80
echo ""
echo "To download artifacts after completion:"
echo ""
echo "  az ml job download --name $JOB_NAME \\"
echo "    --resource-group $RESOURCE_GROUP \\"
echo "    --workspace-name $WORKSPACE \\"
echo "    --output-directory ./artifacts/$JOB_NAME"
echo ""
echo "Then validate:"
echo "  - outputs/phaseb_*_flaml_iterations.csv (should have 200+ rows)"
echo "  - outputs/phaseb_*_estimator_summary.json"
echo "  - outputs/champion/champion_model.pkl"
echo ""

# ==============================================================================
# MLFLOW METRICS
# ==============================================================================
echo "📊 MLFLOW METRICS QUERY"
echo "-"*80
echo ""
echo "View metrics in MLflow:"
echo ""
echo "  import mlflow"
echo "  mlflow.set_tracking_uri('https://...')"
echo "  runs = mlflow.search_runs(experiment_names=['telecom_churn_classification_v3'])"
echo "  print(runs[['metrics.test_recall', 'metrics.test_f1', 'metrics.test_auc_roc']])"
echo ""

# ==============================================================================
# SUCCESS CRITERIA
# ==============================================================================
echo "✅ SUCCESS CRITERIA"
echo "-"*80
echo ""
echo "Pipeline is successful if:"
echo "  ✅ SMOTE logs show: 'New imbalance ratio: 1.000'"
echo "  ✅ FLAML exports: 'Recipe iterations: ...csv'"
echo "  ✅ Recall > 0.60 (MUST be > 0%)"
echo "  ✅ F1-Score > 0.65"
echo "  ✅ Recipe 0005 outperforms Recipe 0001"
echo ""

echo "="*80
echo "🔍 Monitoring complete - check output above for issues"
echo "="*80
