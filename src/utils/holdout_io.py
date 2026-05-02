"""Holdout-aware dataset I/O helpers.

Agent 1 (holdout leakage fix): stage4 emits ``train.csv`` and ``holdout.csv``
as siblings of the combined dataset_out. Trainers prefer the sibling
``train.csv`` so they can never see holdout rows. final_evaluation prefers
the sibling ``holdout.csv`` for honest evaluation.

These helpers are backward-compatible: if the sibling files are absent
(e.g. when re-running an older job artifact), they transparently fall back
to the original combined CSV path. Component YAMLs and pipeline_builder.py
are NOT modified — sibling files travel with the same Azure ML output mount.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _read_csv(path: Path, delimiter: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=delimiter)


def read_train_dataset(dataset_in: str, delimiter: str = ",",
                       prefer_sibling: bool = True) -> Tuple[pd.DataFrame, str]:
    """Read the training dataset.

    Returns ``(df, source)`` where ``source`` is one of:
      - ``"train_sibling"`` if ``train.csv`` was found next to ``dataset_in``
      - ``"combined"`` if we fell back to the combined dataset path
    """
    p = Path(dataset_in)
    if prefer_sibling:
        sibling = p.parent / "train.csv"
        if sibling.exists() and sibling.stat().st_size > 0:
            try:
                df = _read_csv(sibling, delimiter)
                logger.info("✅ Loaded sibling train.csv (%d rows) — holdout-leak-safe", len(df))
                return df, "train_sibling"
            except Exception as e:
                logger.warning("Sibling train.csv read failed (%s); falling back to combined", e)
    df = _read_csv(p, delimiter)
    logger.info("ℹ️ Loaded combined dataset (%d rows) — no train.csv sibling found", len(df))
    return df, "combined"


def read_holdout_dataset(dataset_in: str, delimiter: str = ",") -> Tuple[Optional[pd.DataFrame], str]:
    """Read the honest holdout dataset.

    Returns ``(df_or_None, source)`` where ``source`` is one of:
      - ``"holdout_sibling"`` if ``holdout.csv`` was found
      - ``"none"`` if no holdout sibling exists (caller must handle fallback)
    """
    p = Path(dataset_in)
    sibling = p.parent / "holdout.csv"
    if sibling.exists() and sibling.stat().st_size > 0:
        try:
            df = _read_csv(sibling, delimiter)
            logger.info("✅ Loaded sibling holdout.csv (%d rows) — honest evaluation", len(df))
            return df, "holdout_sibling"
        except Exception as e:
            logger.warning("Sibling holdout.csv read failed (%s)", e)
    return None, "none"


def read_holdout_manifest(dataset_in: str) -> Optional[dict]:
    """Read holdout split manifest if present."""
    p = Path(dataset_in).parent / "holdout_manifest.json"
    if p.exists():
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None
