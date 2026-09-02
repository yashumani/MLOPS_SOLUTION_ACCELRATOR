#!/usr/bin/env bash
# Governed compatibility entrypoint. Defaults to a read-only plan.
set -euo pipefail

cd "$(dirname "$0")/.."
exec python scripts/batch_submit_all.py "$@"
