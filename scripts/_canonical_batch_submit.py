"""Shared thin wrapper around the canonical pipeline submitter."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from _azure_ctx import MissingAzureContextError, load_azure_context


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "pipelines" / "submit_pipeline.py"


def run_config_batch(
    configs: Iterable[str],
    *,
    label: str,
    tags: dict[str, str],
) -> int:
    """Submit configs through the one production-owned submission entrypoint."""

    try:
        context = load_azure_context()
    except MissingAzureContextError as exc:
        print(f"Azure context error: {exc}", file=sys.stderr)
        return 2

    config_paths = [ROOT / path for path in configs]
    print(f"{label}: {len(config_paths)} canonical submissions")
    results: list[tuple[str, str, bool]] = []

    with tempfile.TemporaryDirectory(prefix="mlops-canonical-batch-") as tmp:
        result_root = Path(tmp)
        for index, config_path in enumerate(config_paths, start=1):
            if not config_path.is_file():
                print(f"[{index}/{len(config_paths)}] missing: {config_path}")
                results.append((config_path.name, "missing config", False))
                continue

            result_path = result_root / f"{index}.json"
            command = [
                sys.executable,
                str(SUBMITTER),
                "--config",
                str(config_path),
                "--subscription_id",
                context.subscription_id,
                "--resource_group",
                context.resource_group,
                "--workspace_name",
                context.workspace_name,
                "--compute",
                context.compute,
                "--use_phase1",
                "--force_rerun",
                "--tags_json",
                json.dumps(tags, sort_keys=True),
                "--result_json",
                str(result_path),
            ]
            print(f"[{index}/{len(config_paths)}] {config_path.name}")
            completed = subprocess.run(command, cwd=ROOT, check=False)
            payload = {}
            if result_path.is_file():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            job_name = str(payload.get("job_name") or f"exit={completed.returncode}")
            success = completed.returncode == 0 and bool(payload.get("job_name"))
            results.append((config_path.name, job_name, success))

    print("\nCanonical submission summary")
    for config_name, detail, success in results:
        print(f"  {'OK' if success else 'FAIL'} {config_name}: {detail}")
    return 0 if results and all(item[2] for item in results) else 1
