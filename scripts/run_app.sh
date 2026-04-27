#!/usr/bin/env bash
# Launch both FastAPI (backend) and Streamlit (frontend) together.
# Usage: bash scripts/run_app.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# Load .env if present
if [ -f .env ]; then
    set -a; source .env; set +a
fi

API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-127.0.0.1}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
STREAMLIT_HOST="${STREAMLIT_HOST:-127.0.0.1}"

echo "🚀 Starting MLOps V3 Dashboard"
echo "   API:       http://${API_HOST}:${API_PORT}"
echo "   Dashboard: http://${STREAMLIT_HOST}:${STREAMLIT_PORT}"
echo ""

# Start FastAPI in background
uvicorn api.main:app --host "$API_HOST" --port "$API_PORT" &
API_PID=$!

# Start Streamlit in background
streamlit run ui/app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address "$STREAMLIT_HOST" \
    --server.headless true \
    --browser.gatherUsageStats false &
UI_PID=$!

# Trap to clean up both processes on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$API_PID" 2>/dev/null || true
    kill "$UI_PID" 2>/dev/null || true
    wait
    echo "Done."
}
trap cleanup EXIT INT TERM

# Wait for either to exit
wait -n "$API_PID" "$UI_PID"
