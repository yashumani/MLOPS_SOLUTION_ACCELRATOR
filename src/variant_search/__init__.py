"""
Pipeline Variant Search - Production-Grade Configuration Space Explorer

Terminology:
- Pipeline Template = fixed stage architecture (stages 0-11)
- Pipeline Variant = one configuration across Stages 2-4 (prep/preprocess/feature engineering)
- Search Space = allowed options + constraints
- Winning Variant (Locked) = config frozen for production retraining
"""

from .variant_search_engine import (
    VariantSearchEngine,
    VariantSearchSpace,
    PipelineVariant,
    SearchMode,
    LeakageRisk,
)

__all__ = [
    "VariantSearchEngine",
    "VariantSearchSpace",
    "PipelineVariant",
    "SearchMode",
    "LeakageRisk",
]
