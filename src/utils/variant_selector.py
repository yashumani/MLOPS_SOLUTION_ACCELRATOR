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
