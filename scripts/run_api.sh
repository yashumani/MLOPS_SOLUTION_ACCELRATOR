#!/usr/bin/env bash
# Start the MLOps V3 Pipeline Management API server.
# Usage: bash scripts/run_api.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Install API deps if needed
pip install -q -r api_requirements.txt 2>/dev/null || true

echo "Starting MLOps V3 Pipeline Management API …"
exec python -m api.main
