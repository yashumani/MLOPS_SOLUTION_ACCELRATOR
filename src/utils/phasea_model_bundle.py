"""Phase A raw-input evaluation and ModelBundle helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.base import clone
import yaml

try:
    from orchestration.contracts import SplitManifest, canonical_hash
    from utils.common_evaluator import (
        build_fold_local_pipeline,
        build_training_resampler,
    )
    from utils.fitted_variant_preprocessor import FittedVariantPreprocessor
    from utils.model_bundle import (
        BUNDLE_FILE_NAME,
        BUNDLE_MANIFEST_NAME,
        ModelBundle,
        capture_input_schema,
        load_model_bundle,
        save_model_bundle,
    )
    from utils.recipe_catalog import normalize_recipe
except ImportError:  # pragma: no cover - package-style local imports
    from src.orchestration.contracts import SplitManifest, canonical_hash
    from src.utils.common_evaluator import (
        build_fold_local_pipeline,
        build_training_resampler,
    )
    from src.utils.fitted_variant_preprocessor import FittedVariantPreprocessor
    from src.utils.model_bundle import (
        BUNDLE_FILE_NAME,
        BUNDLE_MANIFEST_NAME,
        ModelBundle,
        capture_input_schema,
        load_model_bundle,
        save_model_bundle,
    )
    from src.utils.recipe_catalog import normalize_recipe


SUPPORTED_TASKS = frozenset({"classification", "regression", "clustering"})
SUPPORTED_PHASEA_SPLIT_STRATEGIES = frozenset({"random", "stratified"})


class PhaseABundleError(RuntimeError):
    """Raised when Phase A cannot prove a selectable raw-input bundle."""


@dataclass(frozen=True)
class PhaseABundleArtifact:
    manifest: Mapping[str, Any]
    smoke_test: Mapping[str, Any]


def load_baseline_recipe(project_root: str | Path, task_type: str) -> dict[str, Any]:
    """Load and normalize the task-specific baseline recipe."""
    task = str(task_type).lower()
    if task not in SUPPORTED_TASKS:
        raise PhaseABundleError(f"Unsupported Phase A task type: {task!r}")
    path = Path(project_root) / "configs" / "recipes" / task / "baseline_recipe.yml"
    if not path.is_file():
        raise PhaseABundleError(f"Baseline recipe does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise PhaseABundleError("Baseline recipe must be a YAML object")
    normalized = normalize_recipe(payload, task_type=task)
    if normalized.get("task_type") != task:
        raise PhaseABundleError("Baseline recipe task type does not match runtime task")
    return normalized


def load_phasea_split_manifest(
    path: str | Path,
    *,
    task_type: str,
    train_count: int,
    random_seed: int,
) -> SplitManifest:
    """Load and bind Phase A training to the exact Stage 2 split contract."""
    source = Path(path)
    if source.is_dir():
        matches = sorted(source.glob("*.json"))
        if len(matches) != 1:
            raise PhaseABundleError(
                f"Expected one SplitManifest JSON in {source}, found {len(matches)}"
            )
        source = matches[0]
    try:
        split_manifest = SplitManifest.from_json(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise PhaseABundleError(f"Invalid Stage 2 SplitManifest: {error}") from error
    if split_manifest.task_type != str(task_type):
        raise PhaseABundleError(
            "SplitManifest task type does not match Phase A runtime task"
        )
    if split_manifest.train_count != int(train_count):
        raise PhaseABundleError(
            "Raw training row count does not match SplitManifest train_count"
        )
    if split_manifest.random_seed != int(random_seed):
        raise PhaseABundleError(
            "Configured random seed does not match SplitManifest random_seed"
        )
    if split_manifest.strategy not in SUPPORTED_PHASEA_SPLIT_STRATEGIES:
        raise PhaseABundleError(
            "Phase A common evaluation does not support split strategy "
            f"{split_manifest.strategy!r}"
        )
    return split_manifest


def split_raw_training_frame(
    frame: pd.DataFrame,
    *,
    task_type: str,
    target_column: str | None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separate raw features and target without mutating the input frame."""
    task = str(task_type)
    if task not in SUPPORTED_TASKS:
        raise PhaseABundleError(f"Unsupported Phase A task type: {task!r}")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise PhaseABundleError("Phase A requires a non-empty raw training frame")
    if task == "clustering":
        if target_column and target_column in frame.columns:
            return frame.drop(columns=[target_column]).copy(), None
        return frame.copy(), None
    if not target_column or target_column not in frame.columns:
        raise PhaseABundleError(
            f"Target column {target_column!r} is missing for {task}"
        )
    return frame.drop(columns=[target_column]).copy(), frame[target_column].copy()


def fit_discovery_preprocessor(
    raw_features: pd.DataFrame,
    target: pd.Series | None,
    *,
    recipe: Mapping[str, Any],
    random_seed: int,
) -> tuple[pd.DataFrame, FittedVariantPreprocessor]:
    """Create the train-only discovery frame; it is never selection evidence."""
    preprocessor = FittedVariantPreprocessor(recipe, random_seed=random_seed)
    transformed = preprocessor.fit_transform(raw_features, target)
    if transformed.empty:
        raise PhaseABundleError("Baseline recipe produced no discovery features")
    return transformed, preprocessor


def build_phasea_evaluation_pipeline(
    estimator: Any,
    *,
    recipe: Mapping[str, Any],
    task_type: str,
    random_seed: int,
) -> Any:
    """Compose an estimator with fresh preprocessing fitted in every CV fold."""
    try:
        exact_estimator = clone(estimator)
    except Exception as error:
        raise PhaseABundleError(
            "Discovered estimator is not cloneable for common evaluation"
        ) from error
    return build_fold_local_pipeline(
        FittedVariantPreprocessor(recipe, random_seed=random_seed),
        exact_estimator,
        recipe=recipe,
        task_type=task_type,
        random_seed=random_seed,
    )


def phasea_candidate_id(
    engine: str,
    estimator: Any,
    recipe: Mapping[str, Any],
) -> str:
    """Bind candidate identity to engine, estimator class/params, and recipe."""
    estimator_type = f"{type(estimator).__module__}.{type(estimator).__qualname__}"
    try:
        parameters = estimator.get_params(deep=False)
    except Exception:
        parameters = {"repr": repr(estimator)}
    payload = json.dumps(
        {
            "engine": str(engine),
            "estimator_type": estimator_type,
            "parameters": parameters,
            "recipe_hash": canonical_hash(recipe),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"baseline:{engine}:{type(estimator).__name__}:{digest}"


def _dependency_versions(engine: str) -> tuple[str, ...]:
    packages = ["pandas", "numpy", "scikit-learn", str(engine)]
    resolved = []
    for package in packages:
        try:
            resolved.append(f"{package}=={importlib_metadata.version(package)}")
        except importlib_metadata.PackageNotFoundError:
            continue
    return tuple(resolved)


def _evidence_dict(evidence: Any) -> dict[str, Any]:
    payload = evidence.to_dict() if hasattr(evidence, "to_dict") else evidence
    if not isinstance(payload, Mapping):
        raise PhaseABundleError("Common-evaluator evidence must be an object")
    return dict(payload)


def _assert_selectable_evidence(
    evidence: Mapping[str, Any],
    *,
    candidate_id: str,
    engine: str,
    task_type: str,
) -> None:
    if evidence.get("status") != "success":
        raise PhaseABundleError(
            "Common-evaluator evidence is not successful: "
            f"{evidence.get('status')!r}"
        )
    if evidence.get("selection_score") is None:
        raise PhaseABundleError("Common-evaluator selection score is missing")
    if evidence.get("candidate_id") != candidate_id:
        raise PhaseABundleError("Evidence candidate ID does not match estimator")
    if evidence.get("engine") != engine:
        raise PhaseABundleError("Evidence engine does not match estimator engine")
    if evidence.get("task_type") != task_type:
        raise PhaseABundleError("Evidence task type does not match bundle task")
    if not str(evidence.get("split_fingerprint") or "").strip():
        raise PhaseABundleError("Common-evaluator split fingerprint is missing")


def _predictions_match(expected: Any, actual: Any) -> bool:
    expected_array = np.asarray(expected)
    actual_array = np.asarray(actual)
    if expected_array.shape != actual_array.shape:
        return False
    if np.issubdtype(expected_array.dtype, np.number) and np.issubdtype(
        actual_array.dtype, np.number
    ):
        return bool(np.allclose(expected_array, actual_array, equal_nan=True))
    return bool(np.array_equal(expected_array, actual_array))


def smoke_test_saved_bundle(
    output_dir: str | Path,
    raw_features: pd.DataFrame,
) -> dict[str, Any]:
    """Reload a persisted bundle and prove deterministic raw-row inference."""
    if raw_features.empty:
        raise PhaseABundleError("Raw-input bundle smoke test requires sample rows")
    sample = raw_features.head(min(5, len(raw_features))).copy()
    try:
        first = load_model_bundle(output_dir)
        expected = first.predict(sample)
        second = load_model_bundle(output_dir)
        actual = second.predict(sample)
    except Exception as error:
        raise PhaseABundleError(f"Raw-input bundle smoke inference failed: {error}") from error
    if len(np.asarray(actual)) != len(sample):
        raise PhaseABundleError("Raw-input bundle smoke prediction count is invalid")
    if not _predictions_match(expected, actual):
        raise PhaseABundleError("Reloaded bundle predictions are not deterministic")
    return {
        "status": "passed",
        "raw_rows": len(sample),
        "bundle_id": second.bundle_id,
    }


def fit_save_phasea_bundle(
    estimator: Any,
    raw_features: pd.DataFrame,
    target: pd.Series | None,
    *,
    task_type: str,
    engine: str,
    candidate_id: str,
    recipe: Mapping[str, Any],
    evidence: Any,
    split_manifest: SplitManifest,
    output_dir: str | Path,
    random_seed: int,
    execution_id: str | None,
    mlflow_parent_run_id: str | None,
    mlflow_child_run_id: str | None,
    threshold: float | None = None,
) -> PhaseABundleArtifact:
    """Refit and persist the exact discovered estimator as a raw-input bundle."""
    evidence_payload = _evidence_dict(evidence)
    _assert_selectable_evidence(
        evidence_payload,
        candidate_id=candidate_id,
        engine=engine,
        task_type=task_type,
    )
    if split_manifest.task_type != task_type:
        raise PhaseABundleError("SplitManifest task type does not match bundle task")
    try:
        final_preprocessor = FittedVariantPreprocessor(
            recipe,
            random_seed=random_seed,
        )
        transformed = final_preprocessor.fit_transform(raw_features, target)
        fit_target = target
        sampler = build_training_resampler(recipe, random_seed)
        if sampler is not None:
            if task_type != "classification" or target is None:
                raise PhaseABundleError(
                    "Baseline imbalance resampling requires classification target"
                )
            transformed, fit_target = sampler.fit_resample(transformed, target)
        final_estimator = clone(estimator)
        if task_type == "clustering":
            final_estimator.fit(transformed)
        else:
            final_estimator.fit(transformed, fit_target)
    except Exception as error:
        raise PhaseABundleError(
            f"Exact estimator refit with baseline preprocessing failed: {error}"
        ) from error

    labels: tuple[Any, ...] = ()
    if task_type == "classification" and target is not None:
        labels = tuple(sorted(pd.unique(target).tolist(), key=lambda value: str(value)))
    environment = {
        "engine": engine,
        "python": platform.python_version(),
        "random_seed": int(random_seed),
    }
    lineage = {
        "phase": "baseline",
        "execution_id": execution_id,
        "mlflow_parent_run_id": mlflow_parent_run_id,
        "mlflow_child_run_id": mlflow_child_run_id,
        "split_id": split_manifest.split_id,
        "data_version": split_manifest.data_version,
        "split_manifest": split_manifest.to_dict(),
        "evaluation_split_fingerprint": evidence_payload["split_fingerprint"],
    }
    try:
        bundle = ModelBundle(
            estimator=final_estimator,
            preprocessing=final_preprocessor,
            task_type=task_type,
            candidate_id=candidate_id,
            input_schema=capture_input_schema(raw_features),
            recipe=recipe,
            selection_metrics=evidence_payload,
            final_test_metrics={},
            environment=environment,
            lineage=lineage,
            dependencies=_dependency_versions(engine),
            threshold=threshold,
            labels=labels,
            signature={
                "inputs": capture_input_schema(raw_features),
                "output": {"name": "prediction"},
            },
            input_example=raw_features.head(min(3, len(raw_features))).to_dict(
                orient="records"
            ),
        )
    except Exception as error:
        raise PhaseABundleError(f"ModelBundle construction failed: {error}") from error

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle_files = (
        destination / BUNDLE_FILE_NAME,
        destination / BUNDLE_MANIFEST_NAME,
    )
    try:
        manifest = save_model_bundle(bundle, destination)
        smoke = smoke_test_saved_bundle(destination, raw_features)
    except Exception as error:
        for path in bundle_files:
            if path.is_file():
                path.unlink()
        if isinstance(error, PhaseABundleError):
            raise
        raise PhaseABundleError(
            f"ModelBundle persistence or smoke validation failed: {error}"
        ) from error
    if manifest.get("bundle_id") != smoke.get("bundle_id"):
        for path in bundle_files:
            if path.is_file():
                path.unlink()
        raise PhaseABundleError("Bundle smoke identity does not match saved manifest")
    return PhaseABundleArtifact(manifest=manifest, smoke_test=smoke)


def validate_phasea_bundle_artifact(
    path: str | Path,
    candidate_manifest: Mapping[str, Any],
) -> ModelBundle:
    """Validate on-disk identity before a baseline candidate is selectable."""
    if candidate_manifest.get("raw_input_bundle_eligible") is not True:
        raise PhaseABundleError("Candidate does not declare raw-input eligibility")
    declared_bundle = candidate_manifest.get("model_bundle")
    if not isinstance(declared_bundle, Mapping):
        raise PhaseABundleError("Candidate ModelBundle manifest is missing")
    candidate_id = str(candidate_manifest.get("candidate_id") or "")
    evaluation = candidate_manifest.get("evaluation")
    if not candidate_id or not isinstance(evaluation, Mapping):
        raise PhaseABundleError("Candidate identity or evaluation evidence is missing")
    if evaluation.get("candidate_id") != candidate_id:
        raise PhaseABundleError("Evaluation candidate ID does not match manifest")
    try:
        bundle = load_model_bundle(path)
    except Exception as error:
        raise PhaseABundleError(f"ModelBundle artifact validation failed: {error}") from error
    if bundle.candidate_id != candidate_id:
        raise PhaseABundleError("On-disk bundle candidate ID does not match manifest")
    if declared_bundle.get("bundle_id") != bundle.bundle_id:
        raise PhaseABundleError("On-disk bundle ID does not match candidate manifest")
    if bundle.task_type != candidate_manifest.get("task_type"):
        raise PhaseABundleError("On-disk bundle task type does not match manifest")
    if bundle.selection_metrics.get("candidate_id") != candidate_id:
        raise PhaseABundleError("Bundled selection evidence has a different candidate ID")
    if bundle.selection_metrics.get("split_fingerprint") != evaluation.get(
        "split_fingerprint"
    ):
        raise PhaseABundleError("Bundled evaluation split fingerprint does not match")
    if bundle.lineage.get("split_id") != candidate_manifest.get("split_id"):
        raise PhaseABundleError("Bundled SplitManifest identity does not match manifest")
    return bundle
