#!/usr/bin/env python3
"""Write one bounded artifact used to verify workspace datastore recovery."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema_version": "1.0",
        "status": "workspace_datastore_write_succeeded",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": os.environ.get("AZUREML_RUN_ID", "unknown"),
    }
    (output_dir / "workspace_datastore_probe.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
