"""
Variant Selector - Load and filter variants from library

Provides variant selection logic that can be called from pipeline submission
or from within Phase B step.

Author: MLOps Solution Accelerator V3  
Date: 2026-01-26
"""

from typing import List, Optional
from pathlib import Path
import random
import yaml

from .variant_schema import VariantConfig, load_variant, validate_variant_for_task


def _nested_get(data: dict, *keys: str, default=None):
    """Safely read a nested YAML value."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _static_variant_score(variant_file: Path, task_type: str) -> float:
    """Score a variant deterministically from recipe metadata and dimensions."""
    with open(variant_file, "r") as file_handle:
        variant_yaml = yaml.safe_load(file_handle) or {}

    metadata = variant_yaml.get("variant_metadata", {}) or {}
    score = 0.0

    leakage_risk = str(metadata.get("leakage_risk", "medium")).lower()
    score += {"none": 30.0, "low": 20.0, "medium": 0.0, "high": -30.0}.get(leakage_risk, 0.0)

    runtime = metadata.get("estimated_runtime_sec")
    try:
        score -= min(float(runtime or 0.0), 600.0) / 60.0
    except (TypeError, ValueError):
        pass

    imputation = str(_nested_get(variant_yaml, "stage3_preprocessing", "imputation", "method", default="none")).lower()
    score += {"knn": 10.0, "iterative": 10.0, "median": 8.0, "mean": 6.0, "mode": 5.0}.get(imputation, 0.0)

    encoding = str(_nested_get(variant_yaml, "stage3_preprocessing", "encoding", "categorical_method", default="none")).lower()
    score += {"target": 9.0, "onehot": 7.0, "ordinal": 4.0, "label": 3.0}.get(encoding, 0.0)

    scaling = str(_nested_get(variant_yaml, "stage3_preprocessing", "scaling", "method", default="none")).lower()
    score += {"robust": 9.0, "quantile": 8.0, "standard": 7.0, "minmax": 5.0}.get(scaling, 0.0)

    feature_selection = str(_nested_get(variant_yaml, "stage4_feature_engineering", "feature_selection", "method", default="none")).lower()
    score += {"boruta": 10.0, "mutual_info": 9.0, "correlation": 6.0, "variance": 4.0}.get(feature_selection, 0.0)

    imbalance = str(_nested_get(variant_yaml, "stage3_preprocessing", "imbalance_handling", "method", default="none")).lower()
    if task_type == "classification":
        score += {"smote": 8.0, "adasyn": 7.0, "smoteenn": 6.0, "class_weight": 5.0, "none": 1.0}.get(imbalance, 0.0)
    elif imbalance not in ("none", ""):
        score -= 25.0

    return score


def select_variants(
    task_type: str,
    library_dir: str,
    max_variants: int = 10,
    selection_strategy: str = "alphabetical",
    runtime_budget_sec: Optional[int] = None,
    seed: int = 42
) -> List[str]:
    """Select variants from library based on strategy.
    
    Args:
        task_type: "classification", "regression", or "clustering"
        library_dir: Path to variant library directory (e.g., configs/recipes/classification/variant_search)
        max_variants: Maximum number of variants to select
        selection_strategy: "alphabetical", "random_seeded", or "scored" (requires profiling)
        runtime_budget_sec: Optional runtime filter (exclude variants exceeding this)
        seed: Random seed for reproducibility
        
    Returns:
        List of variant file paths (relative to repo root)
        
    Raises:
        FileNotFoundError: If library_dir doesn't exist
        ValueError: If no variants match criteria
    """
    library_path = Path(library_dir)
    if not library_path.exists():
        raise FileNotFoundError(f"Variant library not found: {library_dir}")
    
    # Find all variant YAML files (support both variant_*.yml and recipe_*.yml naming)
    # Search both top-level and subdirectories for maximum compatibility
    variant_files = sorted(library_path.glob("variant_*.yml"))
    if not variant_files:
        # Fallback: look for recipe_*.yml files (v1_generated structure)
        variant_files = sorted(library_path.glob("recipe_*.yml"))
    if not variant_files:
        # Fallback: recursive search in subdirectories
        variant_files = sorted(library_path.rglob("variant_*.yml"))
    if not variant_files:
        variant_files = sorted(library_path.rglob("recipe_*.yml"))
    
    if not variant_files:
        raise ValueError(f"No variant files found in {library_dir}")
    
    # Filter by runtime budget if specified
    if runtime_budget_sec is not None:
        filtered = []
        for vfile in variant_files:
            try:
                with open(vfile, 'r') as f:
                    variant_yaml = yaml.safe_load(f)
                runtime = variant_yaml.get("variant_metadata", {}).get("estimated_runtime_sec", 0)
                if runtime <= runtime_budget_sec:
                    filtered.append(vfile)
            except Exception:
                continue
        variant_files = filtered
    
    if not variant_files:
        raise ValueError(f"No variants match runtime budget of {runtime_budget_sec}s")
    
    # Apply selection strategy
    if selection_strategy == "alphabetical":
        selected = variant_files[:max_variants]
    
    elif selection_strategy == "random_seeded":
        random.seed(seed)
        selected = random.sample(variant_files, min(max_variants, len(variant_files)))

    elif selection_strategy == "scored":
        scored_files = []
        for variant_file in variant_files:
            try:
                scored_files.append((_static_variant_score(variant_file, task_type), variant_file))
            except Exception:
                scored_files.append((float("-inf"), variant_file))
        scored_files.sort(key=lambda item: (-item[0], str(item[1])))
        selected = [variant_file for _, variant_file in scored_files[:max_variants]]
    
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")
    
    # Return paths as strings relative to repo root
    # Assuming library_dir is already relative to repo root
    return [str(vfile) for vfile in selected]


def load_and_validate_variants(
    variant_paths: List[str],
    task_type: str,
    strict: bool = True
) -> List[VariantConfig]:
    """Load and validate a list of variant configurations.
    
    Args:
        variant_paths: List of variant file paths
        task_type: Expected task type
        strict: If True, raise on validation errors; if False, skip invalid variants
        
    Returns:
        List of validated VariantConfig instances
    """
    valid_variants = []
    
    for path in variant_paths:
        try:
            variant = load_variant(path)
            validate_variant_for_task(variant, task_type)
            valid_variants.append(variant)
        except Exception as e:
            if strict:
                raise ValueError(f"Variant validation failed for {path}: {e}")
            else:
                print(f"⚠️ Skipping invalid variant {path}: {e}")
                continue
    
    return valid_variants
