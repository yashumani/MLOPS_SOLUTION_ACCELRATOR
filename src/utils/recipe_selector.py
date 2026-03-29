"""
Recipe Selection Utility for V3 Pipeline

Dynamically selects recipes based on:
- Task type (classification, regression, clustering)
- Quality tier (lightning_fast, quick_exploration, balanced_performance, high_performance, state-of-the-art)
- Recipe library (v1_generated [DEPRECATED], variant_search [PRODUCTION], or manual)
- Runtime budget (optional filter)

Usage:
    from src.utils.recipe_selector import select_recipes_for_tier
    
    # 🚀 Production: Variant Search System
    variants = select_recipes_for_tier(
        task_type="classification",
        tier="progressive",  # ignored for variant_search
        count=50,
        library="variant_search",
        max_runtime_sec=180
    )
    # Returns: ["classification/variant_search/variant_18bc115b4c80.yml", ...]
    
    # 🔧 Legacy: V1 Generated Recipes (DEPRECATED)
    recipes = select_recipes_for_tier(
        task_type="classification",
        tier="balanced_performance",
        count=2,
        library="v1_generated"
    )
    # Returns: ["classification/v1_generated/balanced_performance/recipe_clas_balanced_performance_0001.yml", ...]
"""

import yaml
from pathlib import Path
from typing import List, Optional, Dict
import random


class RecipeSelector:
    """Select recipes based on tier, task type, and runtime constraints."""
    
    # Tier aliases (for config flexibility)
    TIER_ALIASES = {
        "lightning_fast": ["lightning_fast", "fast", "quick"],
        "quick_exploration": ["quick_exploration", "quick", "exploration"],
        "balanced_performance": ["balanced_performance", "balanced", "default"],
        "high_performance": ["high_performance", "high", "performance"],
        "state-of-the-art": ["state-of-the-art", "state_of_the_art", "sota"],
    }
    
    def __init__(self, recipes_base_dir: Path):
        """
        Initialize recipe selector.
        
        Args:
            recipes_base_dir: Base directory containing recipes (e.g., configs/recipes)
        """
        self.recipes_base_dir = Path(recipes_base_dir)
    
    def _normalize_tier(self, tier: str) -> str:
        """Normalize tier name to canonical form."""
        tier_lower = tier.lower().replace("-", "_")
        
        for canonical, aliases in self.TIER_ALIASES.items():
            if tier_lower in [a.replace("-", "_") for a in aliases]:
                return canonical
        
        # Default to balanced if unknown
        print(f"⚠️ Unknown tier '{tier}', defaulting to 'balanced_performance'")
        return "balanced_performance"
    
    def _get_recipes_in_tier(
        self,
        task_type: str,
        tier: str,
        library: str = "v1_generated"
    ) -> List[Path]:
        """
        Get all recipe files in a specific tier directory.
        
        Args:
            task_type: classification, regression, or clustering
            tier: Tier name (normalized)
            library: "v1_generated", "variant_search", or "manual"
            
        Returns:
            List of recipe file paths
        """
        # Build directory path
        if library == "v1_generated":
            tier_dir = self.recipes_base_dir / task_type / "v1_generated" / tier
        elif library == "variant_search":
            # 🚀 NEW: Variant search library (flat structure, no tiers)
            tier_dir = self.recipes_base_dir / task_type / "variant_search"
        else:
            # Manual recipes are flat in configs/recipes/{task_type}/
            tier_dir = self.recipes_base_dir / task_type
        
        if not tier_dir.exists():
            print(f"⚠️ Tier directory not found: {tier_dir}")
            return []
        
        # Find all YAML files
        if library == "v1_generated":
            recipe_files = sorted(tier_dir.glob("recipe_*.yml"))
        elif library == "variant_search":
            # Variant search uses variant_*.yml naming
            recipe_files = sorted(tier_dir.glob("variant_*.yml"))
        else:
            # Manual recipes have specific names
            recipe_files = sorted(tier_dir.glob("recipe_*.yml"))
        
        return recipe_files
    
    def _filter_by_runtime(
        self,
        recipe_paths: List[Path],
        max_runtime_sec: Optional[int] = None
    ) -> List[Path]:
        """
        Filter recipes by max runtime from v1_metadata or variant_metadata.
        
        Args:
            recipe_paths: List of recipe file paths
            max_runtime_sec: Maximum runtime threshold (None = no filter)
            
        Returns:
            Filtered list of recipe paths
        """
        if max_runtime_sec is None:
            return recipe_paths
        
        filtered = []
        for recipe_path in recipe_paths:
            try:
                with open(recipe_path) as f:
                    recipe = yaml.safe_load(f)
                
                # Check v1_metadata (old system) or variant_metadata (new system)
                runtime = recipe.get("v1_metadata", {}).get("max_runtime_seconds", 0)
                if runtime == 0:
                    # Try variant_metadata
                    runtime = recipe.get("variant_metadata", {}).get("estimated_runtime_sec", 0)
                
                if runtime <= max_runtime_sec:
                    filtered.append(recipe_path)
            except Exception as e:
                print(f"⚠️ Could not read recipe {recipe_path.name}: {e}")
                continue
        
        return filtered
    
    def select_recipes(
        self,
        task_type: str,
        tier: str = "balanced_performance",
        count: int = 2,
        library: str = "v1_generated",
        max_runtime_sec: Optional[int] = None,
        random_selection: bool = False
    ) -> List[str]:
        """
        Select recipes for Phase B based on tier and constraints.
        
        Args:
            task_type: classification, regression, or clustering
            tier: Quality tier (ignored for variant_search library)
            count: Number of recipes to select (default 2 for current Phase B structure)
            library: "v1_generated", "variant_search", or "manual"
            max_runtime_sec: Optional maximum runtime filter
            random_selection: If True, randomly sample; if False, select top-K by name
            
        Returns:
            List of recipe paths relative to configs/recipes/ (e.g., "classification/v1_generated/balanced_performance/recipe_0001.yml")
        """
        # Normalize tier (ignored for variant_search)
        tier_normalized = self._normalize_tier(tier) if library == "v1_generated" else tier
        
        # Get recipes in tier
        recipe_paths = self._get_recipes_in_tier(task_type, tier_normalized, library)
        
        if not recipe_paths:
            if library == "variant_search":
                print(f"❌ No variants found for {task_type} in variant_search library")
                print(f"   Run: python configs/generate_variant_library.py --task_type {task_type} --max_variants {count}")
                print(f"   Falling back to v1_generated...")
                # Fallback to v1_generated
                return self.select_recipes(task_type, tier, count, "v1_generated", max_runtime_sec, random_selection)
            else:
                print(f"❌ No recipes found for {task_type}/{tier_normalized} in library '{library}'")
                print(f"   Falling back to manual recipes...")
                # Fallback to manual recipes
                return self._get_manual_fallback(task_type, count)
        
        # Filter by runtime if specified
        if max_runtime_sec:
            recipe_paths = self._filter_by_runtime(recipe_paths, max_runtime_sec)
            print(f"🔍 Filtered to {len(recipe_paths)} recipes under {max_runtime_sec}s runtime")
        
        # Select count recipes
        if random_selection:
            selected = random.sample(recipe_paths, min(count, len(recipe_paths)))
        else:
            selected = recipe_paths[:count]  # Top-K by alphabetical order
        
        # Convert to relative paths from configs/recipes/
        relative_paths = []
        for recipe_path in selected:
            rel_path = recipe_path.relative_to(self.recipes_base_dir)
            relative_paths.append(str(rel_path).replace("\\", "/"))  # Normalize path separators
        
        if library == "variant_search":
            print(f"🚀 Selected {len(relative_paths)} Pipeline Variants:")
        else:
            print(f"✅ Selected {len(relative_paths)} recipes from {tier_normalized} tier:")
        for rp in relative_paths:
            print(f"   - {rp}")
        
        return relative_paths
    
    def _get_manual_fallback(self, task_type: str, count: int) -> List[str]:
        """Fallback to manual recipes if v1_generated not found."""
        manual_recipes = {
            "classification": [
                "classification/recipe_smote_target_standard.yml",
                "classification/recipe_knn_onehot_minmax.yml",
            ],
            "regression": [
                "regression/recipe_outlier_iqr_standard.yml",
                "regression/recipe_winsorize_robust.yml",
            ],
            "clustering": [
                "clustering/recipe_knn_onehot_minmax.yml",
                "clustering/recipe_baseline.yml",
            ],
        }
        
        fallback = manual_recipes.get(task_type, [])[:count]
        print(f"⚠️ Using manual fallback recipes: {fallback}")
        return fallback


def select_recipes_for_tier(
    task_type: str,
    tier: str = "balanced_performance",
    count: int = 2,
    library: str = "v1_generated",
    max_runtime_sec: Optional[int] = None,
    recipes_base_dir: Optional[Path] = None
) -> List[str]:
    """
    Convenience function for recipe selection.
    
    Args:
        task_type: classification, regression, or clustering
        tier: Quality tier (ignored for variant_search)
        count: Number of recipes to select
        library: "v1_generated", "variant_search", or "manual"
        max_runtime_sec: Optional runtime filter
        recipes_base_dir: Base directory (defaults to configs/recipes)
        
    Returns:
        List of recipe paths relative to configs/recipes/
    """
    if recipes_base_dir is None:
        # Auto-detect recipes directory
        recipes_base_dir = Path(__file__).resolve().parents[2] / "configs" / "recipes"
    
    selector = RecipeSelector(recipes_base_dir)
    return selector.select_recipes(
        task_type=task_type,
        tier=tier,
        count=count,
        library=library,
        max_runtime_sec=max_runtime_sec
    )


if __name__ == "__main__":
    """Test recipe selection."""
    import sys
    
    task_type = sys.argv[1] if len(sys.argv) > 1 else "classification"
    tier = sys.argv[2] if len(sys.argv) > 2 else "balanced_performance"
    
    print(f"\n🎯 Testing Recipe Selector")
    print(f"   Task type: {task_type}")
    print(f"   Tier: {tier}\n")
    
    # Test v1_generated recipes
    recipes = select_recipes_for_tier(
        task_type=task_type,
        tier=tier,
        count=2,
        library="v1_generated"
    )
    
    print(f"\n✅ Selected recipes:")
    for i, recipe in enumerate(recipes, 1):
        print(f"   {i}. {recipe}")
