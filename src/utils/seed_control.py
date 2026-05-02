"""
Global seed control for deterministic ML pipeline runs.

Sets seeds for: Python ``random``, NumPy, ``PYTHONHASHSEED``, and (when
available) PyTorch / TensorFlow. Safe to import even when those libraries
are missing — failures are silently skipped.

Usage:
    from utils.seed_control import set_global_seed
    set_global_seed(cfg.get("random_seed", 42))
"""

from __future__ import annotations

import os
import random
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> int:
    """Seed all known RNGs for reproducibility.

    Args:
        seed: Integer seed. Defaults to 42.

    Returns:
        The seed that was applied.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # pragma: no cover — optional dependency
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    try:  # pragma: no cover — optional dependency
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except Exception:
        pass

    logger.info(f"🎲 Global seed set to {seed}")
    return seed


def resolve_seed(cfg: Optional[dict] = None, default: int = 42) -> int:
    """Resolve the seed from a config dict (``cfg['random_seed']``) with fallback."""
    if cfg is None:
        return int(default)
    try:
        return int(cfg.get("random_seed", default))
    except Exception:
        return int(default)
