#!/usr/bin/env bash
# Read-only compatibility entrypoint for the governed qualification plan.
set -euo pipefail

cd "$(dirname "$0")/.."
for argument in "$@"; do
  if [[ "$argument" == "--execute" ]]; then
    echo "submit_all_dryrun.sh is read-only; use batch_submit_all.py to execute." >&2
    exit 2
  fi
done
exec python scripts/batch_submit_all.py "$@"
