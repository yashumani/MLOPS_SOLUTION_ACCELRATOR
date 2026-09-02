#!/usr/bin/env python3
"""Capture the exact Azure ML runtime package set for environment locking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--environment", required=True)
    args = parser.parse_args()

    freeze = _run([sys.executable, "-m", "pip", "freeze", "--all"])
    normalized_freeze = "\n".join(
        sorted(line.strip() for line in freeze.splitlines() if line.strip())
    ) + "\n"
    try:
        conda_packages = json.loads(_run(["conda", "list", "--json"]))
        conda_error = None
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        conda_packages = []
        conda_error = f"{type(exc).__name__}: {exc}"

    evidence = {
        "schema_version": "1.0",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "azureml_environment": args.environment,
        "azureml_run_id": os.environ.get("AZUREML_RUN_ID", "unknown"),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pip_freeze_sha256": hashlib.sha256(
            normalized_freeze.encode("utf-8")
        ).hexdigest(),
        "pip_package_count": len(normalized_freeze.splitlines()),
        "conda_package_count": len(conda_packages),
        "conda_capture_error": conda_error,
        "conda_packages": conda_packages,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "requirements.freeze.txt").write_text(
        normalized_freeze,
        encoding="utf-8",
    )
    (output_dir / "environment_lock.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
