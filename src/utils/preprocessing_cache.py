"""
Preprocessing Cache - In-Memory Cache for Variant Preprocessing

Avoids redundant preprocessing transformations within a single pipeline run
by caching preprocessed DataFrames keyed by (preprocessing_config_hash, dataset_fingerprint).

SCOPE: v1 = in-memory only (no disk persistence)

Author: MLOps Solution Accelerator V3
Date: 2026-02-08
"""

import hashlib
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class CacheStats:
    """Statistics for cache usage."""
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PreprocessingCache:
    """
    In-memory cache for preprocessed DataFrames.
    
    Key = hash(preprocessing_config + dataset_fingerprint)
    Value = preprocessed DataFrame
    
    Safety:
    - Never cache when imbalance_handling is smote/adasyn (target-dependent)
    - Cache key includes dataset fingerprint to prevent cross-dataset contamination
    """
    
    def __init__(self, max_entries: int = 50, enabled: bool = True):
        """
        Initialize cache.
        
        Args:
            max_entries: Maximum cached DataFrames (LRU eviction)
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.max_entries = max_entries
        self._cache: Dict[str, pd.DataFrame] = {}
        self._access_order: list = []  # For LRU
        self.stats = CacheStats()
    
    def compute_key(
        self,
        preprocessing_config: Dict[str, Any],
        dataset_fingerprint: str
    ) -> str:
        """
        Compute deterministic cache key.
        
        DESIGN DECISION (R1 audit 2026-02): Key includes config hash AND data
        fingerprint.  Within one s06 run all variants share the same input
        dataset, so data_fingerprint is constant — cache differentiates purely
        on preprocessing config.  SMOTE/ADASYN are excluded via is_cacheable()
        and get() returns copies, so cross-variant contamination is impossible.
        
        Args:
            preprocessing_config: Dict with imputation, encoding, scaling
            dataset_fingerprint: Hash of dataset (from compute_data_fingerprint)
        
        Returns:
            16-char hex hash
        """
        # Extract cacheable preprocessing (exclude imbalance_handling)
        cacheable = {
            "imputation": preprocessing_config.get("imputation", "none"),
            "encoding": preprocessing_config.get("encoding", "onehot"),
            "scaling": preprocessing_config.get("scaling", "none"),
            # NOTE: feature_selection also excluded as it may be expensive
        }
        config_str = json.dumps(cacheable, sort_keys=True)
        combined = f"{config_str}::{dataset_fingerprint}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def is_cacheable(self, preprocessing_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if preprocessing config is safe to cache.
        
        Args:
            preprocessing_config: Full preprocessing config dict
        
        Returns:
            (is_cacheable, reason)
        """
        imbalance = preprocessing_config.get("imbalance_handling", {})
        imbalance_method = imbalance.get("method", "none") if isinstance(imbalance, dict) else imbalance
        
        if imbalance_method in ["smote", "adasyn"]:
            return False, f"imbalance_handling={imbalance_method} requires target-aware resampling"
        
        return True, "ok"
    
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached DataFrame.
        
        Args:
            key: Cache key from compute_key()
        
        Returns:
            DataFrame copy if hit, None if miss
        """
        if not self.enabled:
            return None
        
        if key in self._cache:
            self.stats.hits += 1
            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            # Return COPY to prevent mutation
            return self._cache[key].copy()
        
        self.stats.misses += 1
        return None
    
    def put(self, key: str, df: pd.DataFrame) -> bool:
        """
        Store preprocessed DataFrame in cache.
        
        Args:
            key: Cache key
            df: DataFrame to cache (will be copied)
        
        Returns:
            True if stored, False if disabled/error
        """
        if not self.enabled:
            return False
        
        # LRU eviction if at capacity
        while len(self._cache) >= self.max_entries and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
                self.stats.evictions += 1
        
        # Store copy
        self._cache[key] = df.copy()
        self._access_order.append(key)
        self.stats.stores += 1
        return True
    
    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics as dict."""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "stores": self.stats.stores,
            "evictions": self.stats.evictions,
            "hit_rate": round(self.stats.hit_rate, 4),
            "current_size": len(self._cache),
            "max_size": self.max_entries,
            "enabled": self.enabled
        }
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache


def create_preprocessing_cache(
    enabled: bool = True,
    max_entries: int = 50
) -> PreprocessingCache:
    """Factory function to create preprocessing cache."""
    return PreprocessingCache(max_entries=max_entries, enabled=enabled)
