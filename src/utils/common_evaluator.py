"""Adapter-neutral candidate evaluation for the MLOps V3 pipeline.

The evaluator owns the selection evidence contract.  Engines may discover a
candidate differently, but PyCaret and FLAML winners are compared only after
this module refits them on the same deterministic folds with the same metrics.
The locked test partition is intentionally not accepted by this API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import multiprocessing
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold


SUPPORTED_TASKS = frozenset({"classification", "regression", "clustering"})
SELECTABLE_STATUSES = frozenset({"success"})
PROCESS_TERMINATION_GRACE_SECONDS = 1.0


@dataclass(frozen=True)
class EvaluationSpec:
    task_type: str
    seed: int = 42
    folds: int = 5
    timeout_seconds: float = 600.0
    primary_metric: str | None = None
    execution_id: str | None = None

    def __post_init__(self) -> None:
        if self.task_type not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
        if self.folds < 2 and self.task_type != "clustering":
            raise ValueError("Supervised evaluation requires at least two folds")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass
class CandidateEvidence:
    candidate_id: str
    engine: str
    task_type: str
    status: str
    primary_metric: str
    selection_score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    completed_folds: int = 0
    total_folds: int = 0
    elapsed_seconds: float = 0.0
    seed: int = 42
    split_fingerprint: str | None = None
    failure_reason: str | None = None
    timeout_seconds: float | None = None
    censored: bool = False
    execution_id: str | None = None
    mlflow_parent_run_id: str | None = None
    mlflow_child_run_id: str | None = None
    schema_version: int = 2

    @property
    def selectable(self) -> bool:
        return (
            self.status in SELECTABLE_STATUSES
            and not self.censored
            and self.selection_score is not None
            and math.isfinite(float(self.selection_score))
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["selectable"] = self.selectable
        return result


def primary_metric_for_task(task_type: str) -> str:
    return {
        "classification": "balanced_accuracy",
        "regression": "r2",
        "clustering": "silhouette_score",
    }[task_type]


def build_training_resampler(
    recipe: Mapping[str, Any],
    random_seed: int,
) -> Any | None:
    """Build the recipe's deterministic fold-local training sampler."""

    stage3 = recipe.get("stage3_preprocessing") or {}
    imbalance = stage3.get("imbalance_handling") or {}
    method = str(imbalance.get("method", "none")).lower()
    if method in {"none", "", "null"}:
        return None
    if method == "smote":
        from imblearn.over_sampling import SMOTE

        return SMOTE(random_state=random_seed)
    if method == "adasyn":
        from imblearn.over_sampling import ADASYN

        return ADASYN(random_state=random_seed)
    if method == "smoteenn":
        from imblearn.combine import SMOTEENN

        return SMOTEENN(random_state=random_seed)
    if method == "smotetomek":
        from imblearn.combine import SMOTETomek

        return SMOTETomek(random_state=random_seed)
    raise ValueError(f"Unsupported imbalance method: {method!r}")


def build_fold_local_pipeline(
    preprocessor: Any,
    estimator: Any,
    *,
    recipe: Mapping[str, Any],
    task_type: str,
    random_seed: int,
) -> Any:
    """Bind preprocessing, optional resampling, and estimation per CV fold."""

    sampler = build_training_resampler(recipe, random_seed)
    steps = [("preprocessing", preprocessor)]
    if sampler is not None:
        if task_type != "classification":
            raise ValueError(
                "Imbalance resampling is valid only for classification"
            )
        from imblearn.pipeline import Pipeline as ImbalancedPipeline

        steps.append(("resampler", sampler))
        pipeline_type = ImbalancedPipeline
    else:
        from sklearn.pipeline import Pipeline

        pipeline_type = Pipeline
    steps.append(("estimator", estimator))
    return pipeline_type(steps)


def _take(values: Any, indices: Sequence[int]) -> Any:
    if hasattr(values, "iloc"):
        return values.iloc[list(indices)]
    return np.asarray(values)[list(indices)]


def _split_fingerprint(splits: Iterable[tuple[np.ndarray, np.ndarray]]) -> str:
    payload = [
        {
            "train": [int(value) for value in train_index],
            "validation": [int(value) for value in validation_index],
        }
        for train_index, validation_index in splits
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def deterministic_cv_splits(
    X: Any,
    y: Any,
    *,
    task_type: str,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the one canonical fold assignment used by every engine."""
    if task_type == "clustering":
        raise ValueError("Clustering does not use supervised CV splits")
    if len(X) != len(y):
        raise ValueError("X and y row counts differ")
    if task_type == "classification":
        _, counts = np.unique(np.asarray(y), return_counts=True)
        if len(counts) < 2:
            raise ValueError("Classification evaluation requires at least two classes")
        effective_folds = min(int(folds), int(counts.min()))
        if effective_folds < 2:
            raise ValueError("Each class must contain at least two rows for CV")
        splitter = StratifiedKFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=int(seed),
        )
        return list(splitter.split(np.zeros(len(y)), y))
    effective_folds = min(int(folds), int(len(X)))
    if effective_folds < 2:
        raise ValueError("Regression evaluation requires at least two rows")
    splitter = KFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=int(seed),
    )
    return list(splitter.split(np.zeros(len(y))))


def _classification_metrics(model: Any, X: Any, y: Any) -> dict[str, float]:
    prediction = model.predict(X)
    result = {
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(accuracy_score(y, prediction)),
        "f1_weighted": float(
            f1_score(y, prediction, average="weighted", zero_division=0)
        ),
    }
    if hasattr(model, "predict_proba"):
        try:
            probability = np.asarray(model.predict_proba(X))
            classes = np.unique(np.asarray(y))
            if probability.ndim == 2 and probability.shape[1] == 2:
                result["roc_auc"] = float(roc_auc_score(y, probability[:, 1]))
            elif probability.ndim == 2 and probability.shape[1] == len(classes):
                result["roc_auc"] = float(
                    roc_auc_score(
                        y,
                        probability,
                        multi_class="ovr",
                        average="weighted",
                    )
                )
        except (TypeError, ValueError):
            # Probability evidence is optional; the common primary metric is not.
            pass
    return result


def _regression_metrics(model: Any, X: Any, y: Any) -> dict[str, float]:
    prediction = model.predict(X)
    mse = float(mean_squared_error(y, prediction))
    return {
        "r2": float(r2_score(y, prediction)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y, prediction)),
    }


def _aggregate_fold_metrics(
    fold_metrics: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    names = sorted({name for row in fold_metrics for name in row})
    result: dict[str, float] = {}
    for name in names:
        values = np.asarray(
            [row[name] for row in fold_metrics if name in row],
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        if finite.size:
            result[name] = float(finite.mean())
            result[f"{name}_std"] = float(finite.std(ddof=0))
    return result


def _evaluate_candidate_impl(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    candidate_id: str,
    engine: str,
    spec: EvaluationSpec,
    mlflow_parent_run_id: str | None = None,
    mlflow_child_run_id: str | None = None,
) -> CandidateEvidence:
    """Evaluate one candidate inside the isolated worker process.

    Learned transforms remain fold-local because the full estimator (normally a
    sklearn ``Pipeline``) is cloned and fitted independently for every fold.
    """
    started_at = time.monotonic()
    metric_name = spec.primary_metric or primary_metric_for_task(spec.task_type)
    base = dict(
        candidate_id=str(candidate_id),
        engine=str(engine),
        task_type=spec.task_type,
        primary_metric=metric_name,
        total_folds=1 if spec.task_type == "clustering" else int(spec.folds),
        seed=int(spec.seed),
        timeout_seconds=float(spec.timeout_seconds),
        execution_id=spec.execution_id,
        mlflow_parent_run_id=mlflow_parent_run_id,
        mlflow_child_run_id=mlflow_child_run_id,
    )
    try:
        if spec.task_type == "clustering":
            if engine.lower() != "pycaret":
                return CandidateEvidence(
                    **base,
                    status="skipped_unsupported",
                    selection_score=None,
                    failure_reason="clustering_is_pycaret_only",
                    elapsed_seconds=time.monotonic() - started_at,
                )
            fitted = clone(estimator)
            labels = (
                fitted.fit_predict(X)
                if hasattr(fitted, "fit_predict")
                else fitted.fit(X).predict(X)
            )
            # Score the same learned representation that produced the labels.
            # Raw categorical inputs cannot be cast to float and must not be
            # compared in a feature space different from the fitted pipeline.
            feature_space = X
            if hasattr(fitted, "steps") and len(fitted.steps) > 1:
                feature_space = fitted[:-1].transform(X)
            labels = np.asarray(labels)
            clustered_mask = labels != -1
            clustered_fraction = float(clustered_mask.mean())
            if clustered_mask.any() and not clustered_mask.all():
                feature_space = feature_space[clustered_mask]
                labels = labels[clustered_mask]
            if len(np.unique(labels)) < 2:
                raise ValueError(
                    "Clustering evaluation requires at least two non-noise clusters"
                )
            sample_size = min(len(X), 10_000)
            score = float(
                silhouette_score(
                    feature_space,
                    labels,
                    sample_size=min(sample_size, len(labels)),
                    random_state=spec.seed,
                )
            )
            score *= clustered_fraction
            fingerprint = _split_fingerprint(
                [
                    (
                        np.arange(len(X), dtype=int),
                        np.asarray([], dtype=int),
                    )
                ]
            )
            return CandidateEvidence(
                **base,
                status="success",
                selection_score=score,
                metrics={
                    "silhouette_score": score,
                    "clustered_fraction": clustered_fraction,
                },
                fold_metrics=[
                    {
                        "silhouette_score": score,
                        "clustered_fraction": clustered_fraction,
                    }
                ],
                completed_folds=1,
                elapsed_seconds=time.monotonic() - started_at,
                split_fingerprint=fingerprint,
            )

        splits = deterministic_cv_splits(
            X,
            y,
            task_type=spec.task_type,
            folds=spec.folds,
            seed=spec.seed,
        )
        base["total_folds"] = len(splits)
        fingerprint = _split_fingerprint(splits)
        evidence_rows: list[dict[str, float]] = []
        for train_index, validation_index in splits:
            fold_model = clone(estimator)
            fold_model.fit(_take(X, train_index), _take(y, train_index))
            if spec.task_type == "classification":
                row = _classification_metrics(
                    fold_model,
                    _take(X, validation_index),
                    _take(y, validation_index),
                )
            else:
                row = _regression_metrics(
                    fold_model,
                    _take(X, validation_index),
                    _take(y, validation_index),
                )
            evidence_rows.append(row)

        metrics = _aggregate_fold_metrics(evidence_rows)
        score = metrics.get(metric_name)
        if score is None or not math.isfinite(float(score)):
            raise ValueError(f"Primary metric {metric_name!r} is missing or non-finite")
        return CandidateEvidence(
            **base,
            status="success",
            selection_score=float(score),
            metrics=metrics,
            fold_metrics=evidence_rows,
            completed_folds=len(evidence_rows),
            elapsed_seconds=time.monotonic() - started_at,
            split_fingerprint=fingerprint,
        )
    except Exception as error:
        return CandidateEvidence(
            **base,
            status="failure",
            selection_score=None,
            completed_folds=0,
            elapsed_seconds=time.monotonic() - started_at,
            failure_reason=f"{type(error).__name__}: {error}",
        )


def _evaluation_worker(
    connection: Any,
    estimator: Any,
    X: Any,
    y: Any,
    candidate_id: str,
    engine: str,
    spec: EvaluationSpec,
    mlflow_parent_run_id: str | None,
    mlflow_child_run_id: str | None,
) -> None:
    """Child-process entrypoint; kept module-level for Windows spawn."""
    try:
        evidence = _evaluate_candidate_impl(
            estimator,
            X,
            y,
            candidate_id=candidate_id,
            engine=engine,
            spec=spec,
            mlflow_parent_run_id=mlflow_parent_run_id,
            mlflow_child_run_id=mlflow_child_run_id,
        )
        connection.send(("ok", evidence.to_dict()))
    except BaseException as error:
        connection.send(
            (
                "error",
                {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )
        )
    finally:
        connection.close()


def evaluate_candidate(
    estimator: Any,
    X: Any,
    y: Any,
    *,
    candidate_id: str,
    engine: str,
    spec: EvaluationSpec,
    mlflow_parent_run_id: str | None = None,
    mlflow_child_run_id: str | None = None,
) -> CandidateEvidence:
    """Evaluate a candidate under a hard process-enforced wall-clock budget.

    The estimator and data run in an isolated spawned process.  On timeout the
    process is terminated (and killed if termination does not complete), so a
    blocked engine fit cannot overrun the candidate-engine budget.
    """
    started_at = time.monotonic()
    base = {
        "candidate_id": str(candidate_id),
        "engine": str(engine),
        "task_type": spec.task_type,
        "primary_metric": (
            spec.primary_metric or primary_metric_for_task(spec.task_type)
        ),
        "selection_score": None,
        "completed_folds": 0,
        "total_folds": 1 if spec.task_type == "clustering" else int(spec.folds),
        "seed": int(spec.seed),
        "timeout_seconds": float(spec.timeout_seconds),
        "execution_id": spec.execution_id,
        "mlflow_parent_run_id": mlflow_parent_run_id,
        "mlflow_child_run_id": mlflow_child_run_id,
    }
    if multiprocessing.current_process().daemon:
        return CandidateEvidence(
            **base,
            status="failure",
            elapsed_seconds=time.monotonic() - started_at,
            failure_reason="process_isolation_unavailable_in_daemon",
        )

    context = multiprocessing.get_context("spawn")
    receiving_connection, sending_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_evaluation_worker,
        args=(
            sending_connection,
            estimator,
            X,
            y,
            str(candidate_id),
            str(engine),
            spec,
            mlflow_parent_run_id,
            mlflow_child_run_id,
        ),
        name=f"candidate-evaluator-{candidate_id}",
    )
    deadline = started_at + float(spec.timeout_seconds)
    process.start()
    sending_connection.close()
    process.join(timeout=max(0.0, deadline - time.monotonic()))

    deadline_exceeded = time.monotonic() > deadline
    if process.is_alive() or deadline_exceeded:
        if process.is_alive():
            process.terminate()
            process.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        elapsed = time.monotonic() - started_at
        receiving_connection.close()
        return CandidateEvidence(
            **base,
            status="timeout",
            elapsed_seconds=elapsed,
            failure_reason="candidate_engine_timeout",
            censored=True,
        )

    elapsed = time.monotonic() - started_at

    try:
        if receiving_connection.poll():
            status, payload = receiving_connection.recv()
        else:
            status, payload = (
                "error",
                {
                    "type": "WorkerExit",
                    "message": f"exit_code={process.exitcode}",
                },
            )
    finally:
        receiving_connection.close()

    if status == "ok":
        payload["elapsed_seconds"] = elapsed
        payload.pop("selectable", None)
        return CandidateEvidence(**payload)
    return CandidateEvidence(
        **base,
        status="failure",
        elapsed_seconds=elapsed,
        failure_reason=f"{payload['type']}: {payload['message']}",
    )


def select_best_evidence(
    evidence: Sequence[CandidateEvidence | Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Select only complete, comparable evidence; censored results never win."""
    selectable: list[Mapping[str, Any]] = []
    for item in evidence:
        row = item.to_dict() if isinstance(item, CandidateEvidence) else dict(item)
        if row.get("status") != "success" or bool(row.get("censored")):
            continue
        score = row.get("selection_score")
        if score is None or not math.isfinite(float(score)):
            continue
        selectable.append(row)
    if not selectable:
        return None
    fingerprints = {row.get("split_fingerprint") for row in selectable}
    if len(fingerprints - {None}) > 1:
        raise ValueError("Candidate evidence uses different CV split fingerprints")
    return max(selectable, key=lambda row: float(row["selection_score"]))
