"""Deterministic recipe-catalog compilation before Azure compute is consumed."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

try:
    from orchestration.contracts import canonical_hash, canonical_json
    from utils.variant_schema import validate_variant_yaml
except ImportError:  # pragma: no cover - package-style local imports
    from src.orchestration.contracts import canonical_hash, canonical_json
    from src.utils.variant_schema import validate_variant_yaml


_NON_SEMANTIC_KEYS = {
    "description",
    "expected_runtime_seconds",
    "performance_tier",
    "recommended_for",
    "research_citations",
    "research_note",
    "recipe_name",
    "v1_metadata",
    "variant_metadata",
    "version",
}


def _semantic_mapping(
    value: Any,
    *,
    drop_keys: set[str] = _NON_SEMANTIC_KEYS,
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_mapping(item, drop_keys=drop_keys)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in drop_keys and item is not None
        }
    if isinstance(value, list):
        return [_semantic_mapping(item, drop_keys=drop_keys) for item in value]
    return value


def normalize_recipe(
    raw_recipe: Mapping[str, Any],
    *,
    task_type: Optional[str] = None,
) -> dict[str, Any]:
    """Normalize every behavior-affecting recipe field for semantic identity."""

    raw = dict(raw_recipe)
    effective_task = str(task_type or raw.get("task_type") or "")
    stage3 = dict(raw.get("stage3_preprocessing") or {})
    stage4 = dict(raw.get("stage4_feature_engineering") or {})

    def operation(
        source: Any,
        *,
        method_key: str = "method",
        default_method: str = "none",
    ) -> dict[str, Any]:
        normalized = _semantic_mapping(dict(source or {}))
        normalized[method_key] = str(
            normalized.get(method_key, default_method)
        ).lower()
        return {
            key: normalized[key]
            for key in sorted(normalized)
        }

    normalized = {
        "task_type": effective_task,
        "stage3_preprocessing": {
            "imputation": operation(
                stage3.get("imputation"), default_method="mean"
            ),
            "encoding": operation(
                stage3.get("encoding"),
                method_key="categorical_method",
                default_method="onehot",
            ),
            "scaling": operation(stage3.get("scaling")),
            "imbalance_handling": operation(
                stage3.get("imbalance_handling")
            ),
            "outlier_handling": operation(stage3.get("outlier_handling")),
        },
        "stage4_feature_engineering": {
            "feature_selection": operation(
                stage4.get("feature_selection")
            ),
            "dimensionality_reduction": operation(
                stage4.get("dimensionality_reduction")
            ),
        },
    }
    return _semantic_mapping(normalized, drop_keys=set())


def semantic_recipe_hash(
    raw_recipe: Mapping[str, Any],
    *,
    task_type: Optional[str] = None,
) -> str:
    return canonical_hash(normalize_recipe(raw_recipe, task_type=task_type))


@dataclass(frozen=True)
class RecipeCatalogEntry:
    path: str
    source_paths: tuple[str, ...]
    task_type: str
    recipe_id: str
    semantic_hash: str
    normalized_recipe: Mapping[str, Any]
    estimated_runtime_sec: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source_paths": list(self.source_paths),
            "task_type": self.task_type,
            "recipe_id": self.recipe_id,
            "semantic_hash": self.semantic_hash,
            "normalized_recipe": json.loads(canonical_json(self.normalized_recipe)),
            "estimated_runtime_sec": self.estimated_runtime_sec,
        }


@dataclass(frozen=True)
class QuarantinedRecipe:
    path: str
    recipe_id: str
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "recipe_id": self.recipe_id,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class DuplicateRecipe:
    semantic_hash: str
    canonical_path: str
    duplicate_paths: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_hash": self.semantic_hash,
            "canonical_path": self.canonical_path,
            "duplicate_paths": list(self.duplicate_paths),
        }


@dataclass(frozen=True)
class RecipeCatalog:
    task_type: str
    entries: tuple[RecipeCatalogEntry, ...]
    quarantined: tuple[QuarantinedRecipe, ...]
    duplicates: tuple[DuplicateRecipe, ...]
    checked_count: int
    catalog_hash: str

    @property
    def valid_count(self) -> int:
        return self.checked_count - len(self.quarantined)

    @property
    def unique_count(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0",
            "task_type": self.task_type,
            "checked_count": self.checked_count,
            "valid_count": self.valid_count,
            "unique_count": self.unique_count,
            "quarantined_count": len(self.quarantined),
            "duplicate_group_count": len(self.duplicates),
            "catalog_hash": self.catalog_hash,
            "entries": [entry.to_dict() for entry in self.entries],
            "quarantined": [entry.to_dict() for entry in self.quarantined],
            "duplicates": [entry.to_dict() for entry in self.duplicates],
        }


def _path_priority(path: str) -> tuple[int, str]:
    normalized = path.replace("\\", "/")
    if "/variant_search/" in f"/{normalized}":
        return (0, normalized)
    if "/v1_generated/" not in f"/{normalized}":
        return (1, normalized)
    return (2, normalized)


def _runtime(raw: Mapping[str, Any]) -> int:
    return int(
        (raw.get("variant_metadata") or {}).get("estimated_runtime_sec")
        or (raw.get("v1_metadata") or {}).get("max_runtime_seconds")
        or raw.get("expected_runtime_seconds")
        or 0
    )


def compile_recipe_catalog(
    recipes_base_dir: str | Path,
    task_type: str,
) -> RecipeCatalog:
    """Validate, normalize, deduplicate, and logically quarantine a task catalog."""

    if task_type not in {"classification", "regression", "clustering"}:
        raise ValueError(f"Unsupported task_type: {task_type!r}")
    root = Path(recipes_base_dir).resolve()
    task_root = root / task_type
    if not task_root.is_dir():
        raise FileNotFoundError(f"Recipe task directory not found: {task_root}")

    files = sorted(
        {
            *task_root.rglob("*.yml"),
            *task_root.rglob("*.yaml"),
        },
        key=lambda path: path.as_posix(),
    )
    quarantined: list[QuarantinedRecipe] = []
    groups: dict[str, list[tuple[str, Mapping[str, Any], int]]] = {}

    for path in files:
        relative = path.relative_to(root).as_posix()
        report = validate_variant_yaml(str(path), task_type=task_type)
        if not report.get("valid"):
            quarantined.append(
                QuarantinedRecipe(
                    path=relative,
                    recipe_id=str(report.get("variant_id") or path.stem),
                    errors=tuple(
                        str(error)
                        for error in report.get("errors")
                        or ("unknown validation error",)
                    ),
                )
            )
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        normalized = normalize_recipe(raw, task_type=task_type)
        semantic_hash = canonical_hash(normalized)
        groups.setdefault(semantic_hash, []).append(
            (relative, normalized, _runtime(raw))
        )

    entries: list[RecipeCatalogEntry] = []
    duplicates: list[DuplicateRecipe] = []
    for semantic_hash, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda value: _path_priority(value[0]))
        canonical_path, normalized, runtime = ordered[0]
        all_paths = tuple(sorted(value[0] for value in values))
        entries.append(
            RecipeCatalogEntry(
                path=canonical_path,
                source_paths=all_paths,
                task_type=task_type,
                recipe_id=semantic_hash,
                semantic_hash=semantic_hash,
                normalized_recipe=normalized,
                estimated_runtime_sec=min(value[2] for value in values),
            )
        )
        if len(all_paths) > 1:
            duplicates.append(
                DuplicateRecipe(
                    semantic_hash=semantic_hash,
                    canonical_path=canonical_path,
                    duplicate_paths=tuple(
                        path for path in all_paths if path != canonical_path
                    ),
                )
            )

    entries.sort(key=lambda entry: (entry.semantic_hash, entry.path))
    quarantined.sort(key=lambda entry: entry.path)
    duplicates.sort(key=lambda entry: entry.semantic_hash)
    identity = {
        "schema_version": "2.0",
        "task_type": task_type,
        "entries": [
            {
                "semantic_hash": entry.semantic_hash,
                "source_paths": list(entry.source_paths),
            }
            for entry in entries
        ],
        "quarantined": [entry.to_dict() for entry in quarantined],
    }
    return RecipeCatalog(
        task_type=task_type,
        entries=tuple(entries),
        quarantined=tuple(quarantined),
        duplicates=tuple(duplicates),
        checked_count=len(files),
        catalog_hash=canonical_hash(identity),
    )


def _path_in_library(path: str, library: str, tier: str) -> bool:
    normalized = f"/{path.replace(chr(92), '/')}"
    if library == "variant_search":
        return "/variant_search/" in normalized
    if library == "v1_generated":
        return f"/v1_generated/{tier}/" in normalized
    if library == "manual":
        return "/variant_search/" not in normalized and "/v1_generated/" not in normalized
    if library == "all":
        return True
    raise ValueError(f"Unknown recipe library: {library!r}")


def select_catalog_entries(
    catalog: RecipeCatalog,
    *,
    library: str,
    tier: str = "progressive",
    max_variants: Optional[int],
    runtime_budget_sec: Optional[int] = None,
) -> tuple[RecipeCatalogEntry, ...]:
    """Return eligible semantic entries, optionally after a deterministic cap.

    Canonical submission passes ``max_variants=None`` so every eligible recipe
    reaches data-aware profiling.  The <=40 Round 1 cap is applied only after
    relevance/diversity scoring inside S06.
    """

    if max_variants is not None and (
        max_variants < 1 or max_variants > 40
    ):
        raise ValueError("max_variants must be between 1 and 40")
    selected: list[RecipeCatalogEntry] = []
    for entry in catalog.entries:
        eligible_paths = tuple(
            path
            for path in entry.source_paths
            if _path_in_library(path, library, tier)
        )
        if not eligible_paths:
            continue
        if (
            runtime_budget_sec is not None
            and entry.estimated_runtime_sec > runtime_budget_sec
        ):
            continue
        selected.append(
            replace(
                entry,
                path=sorted(eligible_paths)[0],
            )
        )

    # Semantic hash order is deterministic and independent of filename ordering.
    selected.sort(key=lambda entry: (entry.semantic_hash, entry.path))
    bounded = tuple(
        selected if max_variants is None else selected[:max_variants]
    )
    if not bounded:
        raise ValueError(
            f"No valid unique recipes for task={catalog.task_type!r}, "
            f"library={library!r}, tier={tier!r}"
        )
    return bounded


def catalog_evidence(
    catalog: RecipeCatalog,
    selected: Iterable[RecipeCatalogEntry],
) -> dict[str, Any]:
    selected_tuple = tuple(selected)
    return {
        **catalog.to_dict(),
        "selected_count": len(selected_tuple),
        "selected": [entry.to_dict() for entry in selected_tuple],
    }
