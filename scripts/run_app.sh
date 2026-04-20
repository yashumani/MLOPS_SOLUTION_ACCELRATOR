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
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

echo "🚀 Starting MLOps V3 Dashboard"
echo "   API:       http://0.0.0.0:${API_PORT}"
echo "   Dashboard: http://0.0.0.0:${STREAMLIT_PORT}"
echo ""

# Start FastAPI in background
uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!

# Start Streamlit in background
streamlit run ui/app.py \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
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
