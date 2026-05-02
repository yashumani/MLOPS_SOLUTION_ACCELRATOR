#!/usr/bin/env bash
# Generate a CycloneDX SBOM for the V3 dependency tree.
#
# Output: sbom/sbom-cyclonedx.json
#
# Requires: pip install cyclonedx-bom

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/sbom"
OUT_FILE="${OUT_DIR}/sbom-cyclonedx.json"

mkdir -p "${OUT_DIR}"

if ! command -v cyclonedx-py >/dev/null 2>&1; then
    echo "❌ cyclonedx-py not installed. Run: pip install cyclonedx-bom" >&2
    exit 1
fi

REQ_FILE="${REPO_ROOT}/requirements.lock"
if [[ ! -f "${REQ_FILE}" ]]; then
    echo "⚠️  requirements.lock missing — falling back to requirements.txt" >&2
    REQ_FILE="${REPO_ROOT}/requirements.txt"
fi

echo "📦 Generating SBOM from ${REQ_FILE} → ${OUT_FILE}"
cyclonedx-py requirements \
    --input-file "${REQ_FILE}" \
    --output-format JSON \
    --output-file "${OUT_FILE}"

echo "✅ SBOM written to ${OUT_FILE}"
