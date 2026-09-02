"""Immutable, JSON-serializable contracts for pipeline-correctness lineage.

The contract objects deliberately avoid dependencies on Azure ML, MLflow, or
individual training engines.  Components can therefore exchange the same
artifacts locally and in Azure ML without reconstructing identity from mutable
configuration files or run-name searches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional


SCHEMA_VERSION = "2.0"
SUPPORTED_TASKS = ("classification", "regression", "clustering")
SUPPORTED_ENGINES = ("pycaret", "flaml")
PROTECTED_PROMOTION_ALIASES = frozenset({"champion", "production"})


class ContractValidationError(ValueError):
    """Raised when a serialized pipeline contract is incomplete or invalid."""


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_deep_freeze(item) for item in value))
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _deep_thaw(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Return stable JSON used for every contract and identity hash."""

    return json.dumps(
        _deep_thaw(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_hash(value: Any) -> str:
    """Return the full SHA-256 digest for a canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def dataset_version_identity(dataset: Mapping[str, Any]) -> str:
    """Bind dataset name, version, path, and content digest into one identity."""

    if not isinstance(dataset, Mapping):
        raise ContractValidationError("dataset must be a mapping")
    name = _require_non_empty("dataset.name", dataset.get("name"))
    version = _require_non_empty("dataset.version", dataset.get("version"))
    blob_path = str(dataset.get("blob_path") or "")
    content_sha256 = str(
        dataset.get("content_sha256") or "content-unverified"
    )
    return f"{name}@{version}:{blob_path}:{content_sha256}"


def _require_non_empty(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ContractValidationError(f"{name} must be a non-empty string")
    return text


def _require_schema_version(value: str) -> None:
    if str(value) != SCHEMA_VERSION:
        raise ContractValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {value!r}"
        )


def _validate_engine(task_type: str, engine: str) -> None:
    if task_type not in SUPPORTED_TASKS:
        raise ContractValidationError(f"Unsupported task_type: {task_type!r}")
    if engine not in SUPPORTED_ENGINES:
        raise ContractValidationError(f"Unsupported engine: {engine!r}")
    if task_type == "clustering" and engine != "pycaret":
        raise ContractValidationError("Clustering candidates must use PyCaret")


@dataclass(frozen=True)
class SplitManifest:
    """Immutable identity for the selection and locked-test split policy."""

    task_type: str
    strategy: str
    random_seed: int
    train_count: int
    validation_count: int
    test_count: int
    train_ids_hash: str
    validation_ids_hash: str
    test_ids_hash: str
    data_version: str
    locked_test: bool = True
    group_column: Optional[str] = None
    time_column: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    split_id: str = ""

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if self.task_type not in SUPPORTED_TASKS:
            raise ContractValidationError(
                f"Unsupported task_type: {self.task_type!r}"
            )
        _require_non_empty("strategy", self.strategy)
        _require_non_empty("data_version", self.data_version)
        if self.random_seed < 0:
            raise ContractValidationError("random_seed must be non-negative")
        for name in ("train_count", "validation_count", "test_count"):
            if int(getattr(self, name)) < 0:
                raise ContractValidationError(f"{name} must be non-negative")
        for name in ("train_ids_hash", "validation_ids_hash", "test_ids_hash"):
            _require_non_empty(name, getattr(self, name))
        if not self.locked_test:
            raise ContractValidationError("The final test partition must be locked")

        identity = self._identity_dict()
        expected = canonical_hash(identity)
        if self.split_id and self.split_id != expected:
            raise ContractValidationError("split_id does not match split identity")
        object.__setattr__(self, "split_id", expected)

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "strategy": self.strategy,
            "random_seed": self.random_seed,
            "train_count": self.train_count,
            "validation_count": self.validation_count,
            "test_count": self.test_count,
            "train_ids_hash": self.train_ids_hash,
            "validation_ids_hash": self.validation_ids_hash,
            "test_ids_hash": self.test_ids_hash,
            "data_version": self.data_version,
            "locked_test": self.locked_test,
            "group_column": self.group_column,
            "time_column": self.time_column,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_dict(), "split_id": self.split_id}

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SplitManifest":
        return cls(**dict(value))

    @classmethod
    def from_json(cls, value: str) -> "SplitManifest":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ContractValidationError("SplitManifest JSON must be an object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class CandidateRecord:
    """One exact recipe/engine/algorithm candidate and its evidence state."""

    task_type: str
    recipe_id: str
    recipe_hash: str
    engine: str
    algorithm: str
    parameters: Mapping[str, Any]
    split_id: str
    data_version: str
    code_sha: str
    environment_hash: str
    status: str = "planned"
    metrics: Mapping[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    timed_out: bool = False
    mlflow_run_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    candidate_id: str = ""

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        _validate_engine(self.task_type, self.engine)
        for name in (
            "recipe_id",
            "recipe_hash",
            "algorithm",
            "split_id",
            "data_version",
            "code_sha",
            "environment_hash",
            "status",
        ):
            _require_non_empty(name, getattr(self, name))
        if self.failure_reason and self.status not in {
            "failed",
            "timed_out",
            "quarantined",
            "rejected",
        }:
            raise ContractValidationError(
                "failure_reason requires a failed/timed_out/quarantined/rejected status"
            )
        object.__setattr__(self, "parameters", _deep_freeze(self.parameters))
        object.__setattr__(self, "metrics", _deep_freeze(self.metrics))

        expected = canonical_hash(self._identity_dict())
        if self.candidate_id and self.candidate_id != expected:
            raise ContractValidationError(
                "candidate_id does not match immutable candidate identity"
            )
        object.__setattr__(self, "candidate_id", expected)

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "recipe_id": self.recipe_id,
            "recipe_hash": self.recipe_hash,
            "engine": self.engine,
            "algorithm": self.algorithm,
            "parameters": _deep_thaw(self.parameters),
            "split_id": self.split_id,
            "data_version": self.data_version,
            "code_sha": self.code_sha,
            "environment_hash": self.environment_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._identity_dict(),
            "candidate_id": self.candidate_id,
            "status": self.status,
            "metrics": _deep_thaw(self.metrics),
            "failure_reason": self.failure_reason,
            "timed_out": self.timed_out,
            "mlflow_run_id": self.mlflow_run_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateRecord":
        return cls(**dict(value))


@dataclass(frozen=True)
class QualityDecision:
    """Single downstream policy decision for registration and promotion."""

    decision: str
    candidate_id: str
    evaluated_bundle_hash: str
    metric_name: str
    metric_value: Optional[float]
    threshold: Optional[float]
    registration_allowed: bool
    promotion_aliases: tuple[str, ...] = ()
    registration_tags: Mapping[str, str] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    decision_hash: str = ""

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if self.decision not in {"pass", "warn", "block"}:
            raise ContractValidationError(
                "decision must be one of: pass, warn, block"
            )
        for name in ("candidate_id", "evaluated_bundle_hash", "metric_name"):
            _require_non_empty(name, getattr(self, name))
        aliases = tuple(str(alias).strip() for alias in self.promotion_aliases)
        object.__setattr__(self, "promotion_aliases", aliases)
        object.__setattr__(
            self,
            "registration_tags",
            _deep_freeze(
                {
                    **dict(self.registration_tags),
                    "quality_decision": self.decision,
                    "evaluated_bundle_hash": self.evaluated_bundle_hash,
                }
            ),
        )
        object.__setattr__(self, "reasons", tuple(self.reasons))

        protected = {alias.lower() for alias in aliases}.intersection(
            PROTECTED_PROMOTION_ALIASES
        )
        if self.decision == "warn":
            if not self.registration_allowed:
                raise ContractValidationError(
                    "A warn decision must permit exact-bundle registration"
                )
            if protected:
                raise ContractValidationError(
                    "Warn decisions cannot receive champion/production aliases"
                )
        if self.decision == "block":
            if self.registration_allowed or aliases:
                raise ContractValidationError(
                    "Block decisions cannot register or receive aliases"
                )
        expected = canonical_hash(self._identity_dict())
        if self.decision_hash and self.decision_hash != expected:
            raise ContractValidationError(
                "decision_hash does not match immutable quality decision"
            )
        object.__setattr__(self, "decision_hash", expected)

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision": self.decision,
            "candidate_id": self.candidate_id,
            "evaluated_bundle_hash": self.evaluated_bundle_hash,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "registration_allowed": self.registration_allowed,
            "promotion_aliases": list(self.promotion_aliases),
            "registration_tags": _deep_thaw(self.registration_tags),
            "reasons": list(self.reasons),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_dict(), "decision_hash": self.decision_hash}

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QualityDecision":
        payload = dict(value)
        payload["promotion_aliases"] = tuple(payload.get("promotion_aliases") or ())
        payload["reasons"] = tuple(payload.get("reasons") or ())
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> "QualityDecision":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ContractValidationError("QualityDecision JSON must be an object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class ExecutionManifest:
    """Frozen submission contract consumed by every pipeline stage."""

    config_hash: str
    task_type: str
    dataset: Mapping[str, Any]
    split_policy: Mapping[str, Any]
    engines: tuple[str, ...]
    recipe_paths: tuple[str, ...]
    recipe_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    budgets: Mapping[str, Any]
    code_sha: str
    environment_hashes: Mapping[str, str]
    recipe_catalog_hash: str
    parent_mlflow_run_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION
    execution_id: str = ""

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        if self.task_type not in SUPPORTED_TASKS:
            raise ContractValidationError(
                f"Unsupported task_type: {self.task_type!r}"
            )
        for name in ("config_hash", "code_sha", "recipe_catalog_hash"):
            _require_non_empty(name, getattr(self, name))
        engines = tuple(self.engines)
        paths = tuple(str(path).replace("\\", "/") for path in self.recipe_paths)
        recipe_ids = tuple(self.recipe_ids)
        candidate_ids = tuple(self.candidate_ids)
        if not paths or len(paths) != len(recipe_ids):
            raise ContractValidationError(
                "recipe_paths and recipe_ids must be non-empty and aligned"
            )
        if len(set(paths)) != len(paths) or len(set(recipe_ids)) != len(recipe_ids):
            raise ContractValidationError("Recipe paths and IDs must be unique")
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ContractValidationError("candidate_ids must be non-empty and unique")
        for engine in engines:
            _validate_engine(self.task_type, engine)
        object.__setattr__(self, "engines", engines)
        object.__setattr__(self, "recipe_paths", paths)
        object.__setattr__(self, "recipe_ids", recipe_ids)
        object.__setattr__(self, "candidate_ids", candidate_ids)
        object.__setattr__(self, "dataset", _deep_freeze(self.dataset))
        object.__setattr__(self, "split_policy", _deep_freeze(self.split_policy))
        object.__setattr__(self, "budgets", _deep_freeze(self.budgets))
        object.__setattr__(
            self, "environment_hashes", _deep_freeze(self.environment_hashes)
        )

        expected = canonical_hash(self._identity_dict())
        if self.execution_id and self.execution_id != expected:
            raise ContractValidationError(
                "execution_id does not match immutable execution identity"
            )
        object.__setattr__(self, "execution_id", expected)

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "task_type": self.task_type,
            "dataset": _deep_thaw(self.dataset),
            "split_policy": _deep_thaw(self.split_policy),
            "engines": list(self.engines),
            "recipe_paths": list(self.recipe_paths),
            "recipe_ids": list(self.recipe_ids),
            "candidate_ids": list(self.candidate_ids),
            "budgets": _deep_thaw(self.budgets),
            "code_sha": self.code_sha,
            "environment_hashes": _deep_thaw(self.environment_hashes),
            "recipe_catalog_hash": self.recipe_catalog_hash,
            "parent_mlflow_run_id": self.parent_mlflow_run_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_dict(), "execution_id": self.execution_id}

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionManifest":
        payload = dict(value)
        # Candidate records may be co-located for tolerant handoff to the
        # evaluator lane but are not part of the manifest dataclass itself.
        payload.pop("candidate_records", None)
        for name in ("engines", "recipe_paths", "recipe_ids", "candidate_ids"):
            payload[name] = tuple(payload.get(name) or ())
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> "ExecutionManifest":
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ContractValidationError("ExecutionManifest JSON must be an object")
        return cls.from_dict(payload)


def candidate_ids(records: Iterable[CandidateRecord]) -> tuple[str, ...]:
    """Return a stable tuple of unique candidate IDs."""

    values = tuple(record.candidate_id for record in records)
    if len(values) != len(set(values)):
        raise ContractValidationError("Candidate records contain duplicate IDs")
    return values
