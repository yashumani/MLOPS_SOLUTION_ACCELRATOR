"""Immutable raw-input model bundle contract for final evaluation and S12."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import FunctionType
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


BUNDLE_FILE_NAME = "model_bundle.pkl"
BUNDLE_MANIFEST_NAME = "model_bundle_manifest.json"


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _update_state_hash(
    digest: Any,
    value: Any,
    seen: dict[int, tuple[int, Any]],
    schema_version: int = 5,
) -> None:
    """Canonicalize fitted object state independent of pickle memory layout."""
    if value is None or isinstance(value, (bool, int, str)):
        digest.update(
            _canonical_json(
                [type(value).__name__, value]
            )
        )
        return
    if isinstance(value, float):
        digest.update(f"float:{value.hex()}".encode("ascii"))
        return
    if isinstance(value, np.generic):
        _update_state_hash(digest, value.item(), seen, schema_version)
        return
    if isinstance(value, bytes):
        digest.update(b"bytes:")
        digest.update(value)
        return
    if isinstance(value, bytearray):
        digest.update(b"bytearray:")
        digest.update(bytes(value))
        return
    if isinstance(value, Path):
        digest.update(f"path:{value.as_posix()}".encode("utf-8"))
        return
    if isinstance(value, FunctionType):
        digest.update(
            f"function:{value.__module__}.{value.__qualname__}".encode(
                "utf-8"
            )
        )
        return
    if schema_version >= 5 and isinstance(value, type):
        # Engine wrappers retain class references as constructor metadata.
        # Fitted instances are still traversed separately, including all state.
        digest.update(
            f"class:{value.__module__}.{value.__qualname__}".encode("utf-8")
        )
        return

    identity = id(value)
    if identity in seen:
        digest.update(f"ref:{seen[identity][0]}".encode("ascii"))
        return
    # Schema 4 tracks only ancestors: pickle may copy a shared array without
    # changing its logical values. Schema 3 retains its original alias rules.
    if schema_version >= 4:
        seen = dict(seen)
    # Retain a strong reference for the lifetime of this traversal. Several
    # estimators return temporary nested containers from ``__getstate__``;
    # without the object reference CPython may recycle an id after a child is
    # collected and the hash would emit a false alias for an unrelated value.
    seen[identity] = (len(seen), value)

    if isinstance(value, np.ndarray):
        digest.update(
            _canonical_json(
                ["ndarray", value.dtype.str, list(value.shape)]
            )
        )
        if value.dtype.names:
            # Structured dtypes (notably sklearn Tree nodes) may contain
            # uninitialized alignment padding. Raw ``tobytes`` therefore
            # changes across a lossless pickle roundtrip even when every
            # logical field is identical. Hash named fields independently.
            digest.update(_canonical_json(list(value.dtype.names)))
            for field_name in value.dtype.names:
                _update_state_hash(digest, value[field_name], seen, schema_version)
        elif value.dtype.hasobject:
            _update_state_hash(digest, value.tolist(), seen, schema_version)
        else:
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        return
    if isinstance(value, pd.DataFrame):
        digest.update(b"dataframe:")
        _update_state_hash(digest, list(value.columns), seen, schema_version)
        _update_state_hash(
            digest,
            [str(dtype) for dtype in value.dtypes],
            seen,
            schema_version,
        )
        digest.update(
            pd.util.hash_pandas_object(
                value,
                index=True,
                categorize=True,
            ).values.tobytes()
        )
        return
    if isinstance(value, (pd.Series, pd.Index)):
        digest.update(type(value).__name__.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(
                value,
                index=True,
                categorize=True,
            ).values.tobytes()
        )
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping:")
        for key in sorted(
            value,
            key=lambda item: (
                type(item).__module__,
                type(item).__qualname__,
                repr(item),
            ),
        ):
            _update_state_hash(digest, key, seen, schema_version)
            _update_state_hash(digest, value[key], seen, schema_version)
        return
    if isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _update_state_hash(digest, item, seen, schema_version)
        return
    if isinstance(value, (set, frozenset)):
        child_hashes = []
        for item in value:
            child = hashlib.sha256()
            _update_state_hash(child, item, {}, schema_version)
            child_hashes.append(child.digest())
        digest.update(type(value).__name__.encode("ascii"))
        for child_hash in sorted(child_hashes):
            digest.update(child_hash)
        return
    if (
        hasattr(value, "data")
        and hasattr(value, "indices")
        and hasattr(value, "indptr")
        and hasattr(value, "shape")
    ):
        digest.update(
            f"sparse:{type(value).__module__}.{type(value).__qualname__}".encode(
                "utf-8"
            )
        )
        for item in (
            value.data,
            value.indices,
            value.indptr,
            tuple(value.shape),
        ):
            _update_state_hash(digest, item, seen, schema_version)
        return
    if hasattr(value, "save_raw") and callable(value.save_raw):
        try:
            raw = value.save_raw()
        except TypeError:
            raw = value.save_raw(raw_format="ubj")
        digest.update(
            f"raw:{type(value).__module__}.{type(value).__qualname__}:".encode(
                "utf-8"
            )
        )
        digest.update(bytes(raw))
        return

    state = None
    if hasattr(value, "__getstate__"):
        state = value.__getstate__()
    elif hasattr(value, "__dict__"):
        state = vars(value)
    if state is None:
        if schema_version >= 4 and type(value).__module__ == "sklearn._loss._loss":
            # Cython loss objects expose their parameters through pickle's
            # reduction protocol, not __getstate__. Do not skip their state.
            digest.update(b"sklearn-cython-loss:")
            digest.update(pickle.dumps(value, protocol=4))
            return
        raise TypeError(
            "Unsupported fitted state value: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )
    if (
        schema_version >= 4
        and any(
            base.__module__ == "catboost.core" and base.__name__ == "CatBoost"
            for base in type(value).__mro__
        )
        and isinstance(state, Mapping)
        and "__model" in state
    ):
        # CatBoost's native binary can reorder metadata after a lossless load.
        # Hash its full structured export plus all Python wrapper state instead.
        with TemporaryDirectory(prefix="mlops-catboost-state-") as directory:
            path = Path(directory) / "model.json"
            value.save_model(str(path), format="json")
            state = dict(state)
            state["__model"] = json.loads(path.read_text(encoding="utf-8"))
    digest.update(
        f"object:{type(value).__module__}.{type(value).__qualname__}:".encode(
            "utf-8"
        )
    )
    _update_state_hash(digest, state, seen, schema_version)


def _model_state_sha256_once(
    estimator: Any,
    preprocessing: Any | None,
    schema_version: int = 5,
) -> str:
    digest = hashlib.sha256()
    _update_state_hash(digest, (estimator, preprocessing), {}, schema_version)
    return digest.hexdigest()


def _model_state_sha256(
    estimator: Any,
    preprocessing: Any | None,
    target_decoder: Any | None = None,
    *,
    schema_version: int = 5,
) -> str:
    """Hash logical fitted state after any library lazy-state materialization."""
    if schema_version not in {3, 4, 5}:
        raise ValueError(f"Unsupported ModelBundle schema version: {schema_version}")
    try:
        previous = _model_state_sha256_once(
            estimator,
            (preprocessing, target_decoder),
            schema_version,
        )
        # Some composed sklearn estimators lazily materialize nested state over
        # several successive ``__getstate__`` calls.  Bound the convergence
        # loop while allowing the entire fitted graph to settle.
        for _ in range(9):
            current = _model_state_sha256_once(
                estimator,
                (preprocessing, target_decoder),
                schema_version,
            )
            if current == previous:
                return current
            previous = current
    except Exception as error:
        raise TypeError(
            "ModelBundle estimator and preprocessing state must be hashable"
        ) from error
    raise ValueError("Fitted model state did not stabilize while hashing")


def capture_input_schema(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": [
            {"name": str(column), "dtype": str(frame[column].dtype)}
            for column in frame.columns
        ],
        "column_order": [str(column) for column in frame.columns],
    }


@dataclass(frozen=True)
class ModelBundle:
    estimator: Any
    task_type: str
    candidate_id: str
    preprocessing: Any | None = None
    target_decoder: Any | None = None
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    recipe: Mapping[str, Any] = field(default_factory=dict)
    selection_metrics: Mapping[str, Any] = field(default_factory=dict)
    final_test_metrics: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    lineage: Mapping[str, Any] = field(default_factory=dict)
    dependencies: Sequence[str] = field(default_factory=tuple)
    threshold: float | None = None
    labels: Sequence[Any] = field(default_factory=tuple)
    signature: Mapping[str, Any] = field(default_factory=dict)
    input_example: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None
    bundle_schema_version: int = 5
    _model_state_sha256: str = field(init=False, repr=False)
    _metadata_sha256: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.task_type not in {"classification", "regression", "clustering"}:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
        if not str(self.candidate_id).strip():
            raise ValueError("candidate_id is required")
        object.__setattr__(self, "input_schema", _json_value(self.input_schema))
        object.__setattr__(self, "recipe", _json_value(self.recipe))
        object.__setattr__(
            self,
            "selection_metrics",
            _json_value(self.selection_metrics),
        )
        object.__setattr__(
            self,
            "final_test_metrics",
            _json_value(self.final_test_metrics),
        )
        object.__setattr__(self, "environment", _json_value(self.environment))
        object.__setattr__(self, "lineage", _json_value(self.lineage))
        object.__setattr__(self, "dependencies", tuple(map(str, self.dependencies)))
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "signature", _json_value(self.signature))
        object.__setattr__(
            self,
            "input_example",
            _json_value(self.input_example)
            if self.input_example is not None
            else None,
        )
        object.__setattr__(
            self,
            "_model_state_sha256",
            _model_state_sha256(
                self.estimator,
                self.preprocessing,
                self.target_decoder,
                schema_version=self.bundle_schema_version,
            ),
        )
        object.__setattr__(
            self,
            "_metadata_sha256",
            _sha256_bytes(_canonical_json(self.metadata())),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "bundle_schema_version": self.bundle_schema_version,
            "task_type": self.task_type,
            "candidate_id": self.candidate_id,
            "input_schema": deepcopy(dict(self.input_schema)),
            "recipe": deepcopy(dict(self.recipe)),
            "selection_metrics": deepcopy(dict(self.selection_metrics)),
            "final_test_metrics": deepcopy(dict(self.final_test_metrics)),
            "environment": deepcopy(dict(self.environment)),
            "lineage": deepcopy(dict(self.lineage)),
            "dependencies": list(self.dependencies),
            "threshold": self.threshold,
            "labels": list(self.labels),
            "signature": deepcopy(dict(self.signature)),
            "input_example": deepcopy(self.input_example),
            "model_state_sha256": self._model_state_sha256,
            "preprocessing_type": (
                type(self.preprocessing).__name__
                if self.preprocessing is not None
                else None
            ),
            "target_decoder_type": (
                type(self.target_decoder).__name__
                if self.target_decoder is not None
                else None
            ),
            "estimator_type": type(self.estimator).__name__,
        }

    @property
    def bundle_id(self) -> str:
        self.assert_integrity()
        return self._metadata_sha256

    def assert_integrity(self) -> None:
        state_actual = _model_state_sha256(
            self.estimator,
            self.preprocessing,
            self.target_decoder,
            schema_version=self.bundle_schema_version,
        )
        if state_actual != self._model_state_sha256:
            raise ValueError("ModelBundle fitted model state changed after construction")
        actual = _sha256_bytes(_canonical_json(self.metadata()))
        if actual != self._metadata_sha256:
            raise ValueError("ModelBundle metadata changed after construction")

    def _validated_frame(self, raw_input: Any) -> Any:
        if not isinstance(raw_input, pd.DataFrame):
            return raw_input
        expected = list(self.input_schema.get("column_order") or [])
        if expected:
            missing = [column for column in expected if column not in raw_input]
            if missing:
                raise ValueError(f"Raw input is missing columns: {missing}")
            return raw_input.loc[:, expected]
        return raw_input

    def _transform(self, raw_input: Any) -> Any:
        frame = self._validated_frame(raw_input)
        if self.preprocessing is None:
            return frame
        return self.preprocessing.transform(frame)

    def predict(self, raw_input: Any) -> Any:
        self.assert_integrity()
        predictions = self.estimator.predict(self._transform(raw_input))
        if self.target_decoder is not None:
            return self.target_decoder.inverse_transform(
                np.asarray(predictions, dtype=int)
            )
        return predictions

    def transform_features(self, raw_input: Any) -> Any:
        """Apply the exact fitted preprocessing contract to raw features."""

        self.assert_integrity()
        return self._transform(raw_input)

    def predict_proba(self, raw_input: Any) -> Any:
        self.assert_integrity()
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("Bundled estimator does not implement predict_proba")
        return self.estimator.predict_proba(self._transform(raw_input))


def save_model_bundle(bundle: ModelBundle, output_dir: str | Path) -> dict[str, Any]:
    bundle.assert_integrity()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle_path = destination / BUNDLE_FILE_NAME
    joblib.dump(bundle, bundle_path)
    artifact_sha256 = _sha256_bytes(bundle_path.read_bytes())
    manifest = bundle.metadata()
    manifest.update(
        {
            "bundle_id": bundle.bundle_id,
            "artifact_file": BUNDLE_FILE_NAME,
            "artifact_sha256": artifact_sha256,
        }
    )
    (destination / BUNDLE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return manifest


def load_model_bundle(path: str | Path) -> ModelBundle:
    source = Path(path)
    if source.is_dir():
        source = source / BUNDLE_FILE_NAME
    bundle = joblib.load(source)
    compatible_bundle_type = (
        type(bundle).__name__ == "ModelBundle"
        and type(bundle).__module__.endswith("utils.model_bundle")
        and hasattr(bundle, "assert_integrity")
        and hasattr(bundle, "metadata")
    )
    if not isinstance(bundle, ModelBundle) and not compatible_bundle_type:
        raise TypeError(f"Expected ModelBundle, found {type(bundle).__name__}")
    bundle.assert_integrity()
    manifest_path = source.parent / BUNDLE_MANIFEST_NAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_sha256") != _sha256_bytes(source.read_bytes()):
            raise ValueError("ModelBundle artifact hash does not match manifest")
        if manifest.get("bundle_id") != bundle.bundle_id:
            raise ValueError("ModelBundle identity does not match manifest")
    return bundle


def find_model_bundle(path: str | Path) -> Path | None:
    source = Path(path)
    direct = source / BUNDLE_FILE_NAME if source.is_dir() else source
    if direct.name == BUNDLE_FILE_NAME and direct.is_file():
        return direct
    matches = sorted(source.rglob(BUNDLE_FILE_NAME)) if source.exists() else []
    return matches[0] if len(matches) == 1 else None
