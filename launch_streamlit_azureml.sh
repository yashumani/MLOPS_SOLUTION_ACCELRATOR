#!/usr/bin/env bash
# ============================================================
# launch_streamlit_azureml.sh
#
# Launches the Streamlit dashboard on the Azure ML Compute
# Instance Application Proxy so it is accessible via a public
# Azure-managed URL — NO port forwarding required.
#
# Public URL pattern:
#   https://{compute-name}-{port}.{region}.instances.azureml.ms/
#
# Usage:
#   bash launch_streamlit_azureml.sh
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_DIR="${REPO_ROOT}/ui"

# ── Source .env before Azure CLI detection ───────────────────
if [[ -f "${REPO_ROOT}/.env" ]]; then
  echo "🔑 Loading .env …"
  set -a; source "${REPO_ROOT}/.env"; set +a
fi

PORT="${STREAMLIT_PORT:-8501}"
API_PORT="${API_PORT:-8000}"
AZURE_RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-}"
AZURE_WORKSPACE_NAME="${AZURE_WORKSPACE_NAME:-}"
API_BASE_URL="${API_BASE_URL:-http://localhost:${API_PORT}}"
export API_BASE_URL

# ── Detect compute name and region from Azure CLI ────────────
echo "⏳ Detecting Azure ML workspace region…"
if [[ -n "${AZURE_RESOURCE_GROUP}" && -n "${AZURE_WORKSPACE_NAME}" ]]; then
  REGION=$(az ml workspace show \
    --resource-group "${AZURE_RESOURCE_GROUP}" \
    --name "${AZURE_WORKSPACE_NAME}" \
    --query "location" -o tsv 2>/dev/null || echo "unknown")
else
  REGION="unknown"
fi

COMPUTE_NAME=$(hostname -s)
if [[ "${REGION}" != "unknown" ]]; then
  PUBLIC_URL="https://${COMPUTE_NAME}-${PORT}.${REGION}.instances.azureml.ms/"
  export UI_BASE_URL="${UI_BASE_URL:-${PUBLIC_URL%/}}"
else
  PUBLIC_URL="https://${COMPUTE_NAME}-${PORT}.<region>.instances.azureml.ms/"
fi

echo ""
echo "============================================================"
echo "  Compute : ${COMPUTE_NAME}"
echo "  Region  : ${REGION}"
echo "  Port    : ${PORT}"
echo "  API     : ${API_BASE_URL}"
echo ""
echo "  🌐 Public URL (once Streamlit starts):"
echo "     ${PUBLIC_URL}"
echo ""
echo "  ℹ️  The :${API_PORT} URL is the FastAPI backend."
echo "     The Streamlit dashboard is exposed on :${PORT}."
echo ""
echo "  ℹ️  You can also open this from Azure ML Studio:"
echo "     Compute → ${COMPUTE_NAME} → Applications tab"
echo "============================================================"
echo ""

# ── Install / update Streamlit if needed ─────────────────────
echo "📦 Checking Streamlit dependencies…"
pip install --quiet --upgrade -r "${UI_DIR}/requirements.txt"

# ── Kill any existing Streamlit on this port ─────────────────
OLD_PID=$(lsof -ti tcp:${PORT} 2>/dev/null || true)
if [[ -n "${OLD_PID}" ]]; then
  echo "🔄 Stopping existing process on port ${PORT} (PID ${OLD_PID})…"
  kill "${OLD_PID}" 2>/dev/null || true
  sleep 2
fi

# ── Launch Streamlit ─────────────────────────────────────────
cd "${REPO_ROOT}"
echo "🚀 Starting Streamlit…"

nohup python -m streamlit run ui/app.py \
  --server.port "${PORT}" \
  --server.address "0.0.0.0" \
  --server.baseUrlPath "" \
  --server.enableCORS false \
  --server.enableXsrfProtection true \
  --browser.gatherUsageStats false \
  > /tmp/streamlit_azureml.log 2>&1 &

STREAMLIT_PID=$!
echo "✅ Streamlit started (PID ${STREAMLIT_PID})"
echo "${STREAMLIT_PID}" > /tmp/streamlit_azureml.pid

# ── Wait for it to be ready ───────────────────────────────────
echo "⏳ Waiting for Streamlit to be ready on port ${PORT}…"
for i in $(seq 1 30); do
  if curl -sf "http://localhost:${PORT}/healthz" > /dev/null 2>&1 || \
     curl -sf "http://localhost:${PORT}/" > /dev/null 2>&1; then
    echo ""
    echo "============================================================"
    echo "  ✅ Streamlit is UP!"
    echo ""
    echo "  🌐 Open in browser:"
    echo "     ${PUBLIC_URL}"
    echo ""
    echo "  Logs : tail -f /tmp/streamlit_azureml.log"
    echo "  Stop : kill \$(cat /tmp/streamlit_azureml.pid)"
    echo "============================================================"
    exit 0
  fi
  sleep 2
  echo -n "."
done

echo ""
echo "⚠️  Streamlit may still be starting. Check logs:"
echo "    tail -f /tmp/streamlit_azureml.log"
echo ""
echo "🌐 Try the URL:"
echo "   ${PUBLIC_URL}"
