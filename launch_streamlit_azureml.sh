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
PORT=8501

# ── Detect compute name and region from Azure CLI ────────────
echo "⏳ Detecting Azure ML workspace region…"
REGION=$(az ml workspace show \
  --resource-group mvpv1 \
  --name mlops-accelerator \
  --query "location" -o tsv 2>/dev/null || echo "unknown")

COMPUTE_NAME=$(hostname -s)

echo ""
echo "============================================================"
echo "  Compute : ${COMPUTE_NAME}"
echo "  Region  : ${REGION}"
echo "  Port    : ${PORT}"
echo ""
echo "  🌐 Public URL (once Streamlit starts):"
echo "     https://${COMPUTE_NAME}-${PORT}.${REGION}.instances.azureml.ms/"
echo ""
echo "  ℹ️  You can also open this from Azure ML Studio:"
echo "     Compute → ${COMPUTE_NAME} → Applications tab"
echo "============================================================"
echo ""

# ── Install / update Streamlit if needed ─────────────────────
echo "📦 Checking Streamlit installation…"
pip install --quiet --upgrade "streamlit>=1.35,<2" requests python-dotenv

# ── Source .env if present ───────────────────────────────────
if [[ -f "${REPO_ROOT}/.env" ]]; then
  echo "🔑 Loading .env …"
  set -a; source "${REPO_ROOT}/.env"; set +a
fi

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
  --server.enableXsrfProtection false \
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
    echo "     https://${COMPUTE_NAME}-${PORT}.${REGION}.instances.azureml.ms/"
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
echo "   https://${COMPUTE_NAME}-${PORT}.${REGION}.instances.azureml.ms/"
