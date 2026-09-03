#!/usr/bin/env python3
"""Load one registered MLflow model and score its saved raw input example."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
from typing import Any

import mlflow
from mlflow.models import Model
from mlflow.tracking import MlflowClient
import numpy as np


MODEL_URI_PATTERN = re.compile(r"^models:/([A-Za-z0-9_.-]+)/([1-9][0-9]*)$")
PROTECTED_ALIASES = {"champion", "production"}


def _parse_model_uri(model_uri: str) -> tuple[str, str]:
    match = MODEL_URI_PATTERN.fullmatch(str(model_uri).strip())
    if match is None:
        raise ValueError("model_uri must use models:/<name>/<positive-version>")
    return match.group(1), match.group(2)


def _row_count(value: Any) -> int:
    try:
        return int(len(value))
    except TypeError:
        return 1


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _validate_model_version(
    model_version: Any,
    *,
    expected_execution_id: str,
    expected_code_sha: str,
    expected_dataset_sha: str,
) -> dict[str, str]:
    tags = {
        str(key): str(value)
        for key, value in dict(getattr(model_version, "tags", None) or {}).items()
    }
    expected_tags = {
        "execution_id": expected_execution_id,
        "code_sha": expected_code_sha,
        "dataset_content_sha256": expected_dataset_sha,
        "promotion_mode": "manual",
        "promotion_performed": "false",
        "lifecycle_stage": "Unassigned",
    }
    mismatches = {
        key: {"expected": expected, "actual": tags.get(key)}
        for key, expected in expected_tags.items()
        if tags.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Registered model lineage/promotion tags do not match: "
            + json.dumps(mismatches, sort_keys=True)
        )

    current_stage = str(getattr(model_version, "current_stage", "None") or "None")
    if current_stage.lower() != "none":
        raise RuntimeError(
            f"Registered model was promoted before approval: stage={current_stage!r}"
        )
    aliases = [
        str(alias)
        for alias in (getattr(model_version, "aliases", None) or [])
    ]
    protected = sorted(PROTECTED_ALIASES.intersection(alias.lower() for alias in aliases))
    if protected:
        raise RuntimeError(
            "Registered model has protected aliases before approval: "
            + ", ".join(protected)
        )
    return tags


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--expected-execution-id", required=True)
    parser.add_argument("--expected-code-sha", required=True)
    parser.add_argument("--expected-dataset-sha", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    model_name, model_version_number = _parse_model_uri(args.model_uri)
    registry_client = MlflowClient()
    model_version = registry_client.get_model_version(
        model_name,
        model_version_number,
    )
    tags = _validate_model_version(
        model_version,
        expected_execution_id=args.expected_execution_id,
        expected_code_sha=args.expected_code_sha,
        expected_dataset_sha=args.expected_dataset_sha,
    )

    local_model_path = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=args.model_uri)
    ).resolve()
    metadata = Model.load(str(local_model_path))
    if metadata.signature is None:
        raise RuntimeError("Registered model has no MLflow signature")
    input_example = metadata.load_input_example(str(local_model_path))
    if input_example is None or _row_count(input_example) < 1:
        raise RuntimeError("Registered model has no usable input example")

    model = mlflow.pyfunc.load_model(str(local_model_path))
    predictions = model.predict(input_example)
    prediction_count = _row_count(predictions)
    if prediction_count != _row_count(input_example):
        raise RuntimeError(
            "Prediction row count does not match the saved input example: "
            f"{prediction_count} != {_row_count(input_example)}"
        )

    prediction_array = np.asarray(predictions)
    evidence = {
        "schema_version": "1.0",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "azureml_run_id": os.environ.get("AZUREML_RUN_ID", "unknown"),
        "model_uri": args.model_uri,
        "model_name": model_name,
        "model_version": model_version_number,
        "registration_run_id": str(getattr(model_version, "run_id", "") or ""),
        "execution_id": args.expected_execution_id,
        "code_sha": args.expected_code_sha,
        "dataset_content_sha256": args.expected_dataset_sha,
        "current_stage": str(getattr(model_version, "current_stage", "None")),
        "aliases": list(getattr(model_version, "aliases", None) or []),
        "lineage_tags_verified": True,
        "signature": metadata.signature.to_dict(),
        "input_type": type(input_example).__name__,
        "input_rows": _row_count(input_example),
        "prediction_type": type(predictions).__name__,
        "prediction_rows": prediction_count,
        "prediction_preview": prediction_array.reshape(-1)[:5].tolist(),
        "python_version": platform.python_version(),
        "mlflow_version": mlflow.__version__,
        "validated_tag_keys": sorted(tags),
        "status": "passed",
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "registered_model_inference_smoke.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    print(json.dumps(evidence, indent=2, sort_keys=True, default=_json_value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
