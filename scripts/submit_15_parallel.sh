#!/usr/bin/env bash
# Parallel force-bypass submission was removed. Delegate to the governed runner.
set -euo pipefail

cd "$(dirname "$0")/.."
echo "Parallel submission is disabled; using the governed qualification runner."
exec python scripts/batch_submit_all.py "$@"
