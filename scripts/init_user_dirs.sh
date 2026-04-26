#!/usr/bin/env bash
# Initialize per-user directories used by the v3 pipeline.
# Run once per user/host before first --force submission.
set -euo pipefail

MLOPS_DIR="${HOME}/.mlops"
mkdir -p "${MLOPS_DIR}"
chmod 700 "${MLOPS_DIR}"
echo "✅ Created ${MLOPS_DIR} (mode 700)"

if [[ ! -f "${MLOPS_DIR}/.gitkeep" ]]; then
    touch "${MLOPS_DIR}/.gitkeep"
fi

echo "✅ User directories initialized"
