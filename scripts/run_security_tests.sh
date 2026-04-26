#!/usr/bin/env bash
# Run the v3 pipeline security regression suite.
# Exits 0 only if ALL tests pass and coverage >= 80% on security-critical paths.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "🔧 Installing test dependencies..."
pip install --quiet 'pytest>=7.4' 'pytest-mock>=3.12' 'pytest-cov>=4.1' 'evidently>=0.4' || {
    echo "❌ Failed to install test dependencies"
    exit 1
}

echo ""
echo "🛡️  Running security regression tests..."
pytest tests/test_security/ \
    -v \
    --tb=short \
    --cov=pipelines \
    --cov=src/steps \
    --cov-report=term-missing \
    --cov-fail-under=80 \
    -m security
RC=$?

echo ""
if [[ $RC -ne 0 ]]; then
    echo "❌ SECURITY TESTS FAILED — NOT PRODUCTION READY"
    exit $RC
fi

echo "✅ ALL SECURITY TESTS PASSED — PRODUCTION GATE CLEARED"
exit 0
