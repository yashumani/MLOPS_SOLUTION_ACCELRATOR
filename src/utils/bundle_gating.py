"""
Bundle Gating — Data-driven variant bundle selection.

Computes statistical signals from the dataset and uses them to
enable / disable named "bundles" (groups of variant YAMLs).

Workflow::

    signals  = compute_data_signals(df, target_col, task_type)
    catalog  = load_bundle_catalog("configs/variant_bundles/classification")
    enabled  = select_enabled_bundles(signals, catalog)
    paths    = resolve_variant_paths(enabled, repo_root)
    write_gating_artifacts(signals, enabled, "outputs/signals")

Design:
  - Pure Python (no Azure / MLflow dependency)
  - Each bundle YAML declares ``gating_rules`` evaluated against signals
  - Bundles with ``default_enabled: true`` are always included
  - Threshold values are specified *in the bundle YAML* (auditable)
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GatingRule:
    """A single threshold test for a bundle."""
    signal: str
    operator: str  # "<", ">", "<=", ">=", "==", "!="
    threshold: float
    reason: str = ""


@dataclass
class BundleConfig:
    """A named group of variant YAML paths with gating rules."""
    bundle_name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    default_enabled: bool = False
    gating_rules: List[GatingRule] = field(default_factory=list)
    variant_paths: List[str] = field(default_factory=list)
    variant_glob: str = ""
    max_variants: int = 50

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BundleConfig":
        rules = []
        for r in d.get("gating_rules", []):
            rules.append(GatingRule(
                signal=r["signal"],
                operator=r["operator"],
                threshold=float(r["threshold"]),
                reason=r.get("reason", ""),
            ))
        return cls(
            bundle_name=d["bundle_name"],
            description=d.get("description", ""),
            tags=d.get("tags", []),
            default_enabled=d.get("default_enabled", False),
            gating_rules=rules,
            variant_paths=d.get("variant_paths", []),
            variant_glob=d.get("variant_glob", ""),
            max_variants=d.get("max_variants", 50),
        )


@dataclass
class BundleDecision:
    """Record of why a bundle was enabled or disabled."""
    bundle_name: str
    enabled: bool
    reasons: List[str] = field(default_factory=list)
    matched_rules: int = 0
    total_rules: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Signal computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_data_signals(
    df: pd.DataFrame,
    target_col: Optional[str],
    task_type: str,
) -> Dict[str, Any]:
    """Compute statistical signals from the dataset.

    Returns a flat dict of signal_name → value.  All values are float
    or int so that they serialise cleanly to JSON.
    """
    signals: Dict[str, Any] = {
        "task_type": task_type,
        "n_rows": len(df),
        "n_features": df.shape[1] - (1 if target_col and target_col in df.columns else 0),
    }

    # ── Sample size bucket ────────────────────────────────────────────
    n = len(df)
    if n < 500:
        signals["sample_bucket"] = "tiny"
    elif n < 5_000:
        signals["sample_bucket"] = "small"
    elif n < 50_000:
        signals["sample_bucket"] = "medium"
    else:
        signals["sample_bucket"] = "large"

    # ── Missing-value rate ────────────────────────────────────────────
    signals["missing_rate"] = round(float(df.isnull().mean().mean()), 4)
    signals["missing_cols"] = int((df.isnull().sum() > 0).sum())

    # ── Cardinality & sparsity ────────────────────────────────────────
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if target_col and target_col in cat_cols:
        cat_cols.remove(target_col)
    if cat_cols:
        cardinalities = [int(df[c].nunique()) for c in cat_cols]
        signals["max_cardinality"] = max(cardinalities)
        signals["mean_cardinality"] = round(float(np.mean(cardinalities)), 1)
        signals["high_cardinality_cols"] = int(sum(1 for c in cardinalities if c > 50))
    else:
        signals["max_cardinality"] = 0
        signals["mean_cardinality"] = 0.0
        signals["high_cardinality_cols"] = 0

    # Sparsity (fraction of zero values in numeric cols)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)
    if num_cols:
        zero_frac = float((df[num_cols] == 0).mean().mean())
        signals["sparsity"] = round(zero_frac, 4)
    else:
        signals["sparsity"] = 0.0

    # ── Skewness / kurtosis / outlier rate ────────────────────────────
    if num_cols:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skew_vals = df[num_cols].skew().dropna()
            kurt_vals = df[num_cols].kurtosis().dropna()
        signals["mean_abs_skewness"] = round(float(skew_vals.abs().mean()), 3) if len(skew_vals) else 0.0
        signals["max_abs_skewness"] = round(float(skew_vals.abs().max()), 3) if len(skew_vals) else 0.0
        signals["mean_kurtosis"] = round(float(kurt_vals.mean()), 3) if len(kurt_vals) else 0.0

        # IQR-based outlier rate
        outlier_counts = 0
        total_vals = 0
        for c in num_cols:
            col = df[c].dropna()
            if len(col) < 10:
                continue
            q1, q3 = col.quantile(0.25), col.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_counts += int(((col < q1 - 1.5 * iqr) | (col > q3 + 1.5 * iqr)).sum())
                total_vals += len(col)
        signals["outlier_rate"] = round(outlier_counts / max(total_vals, 1), 4)
    else:
        signals["mean_abs_skewness"] = 0.0
        signals["max_abs_skewness"] = 0.0
        signals["mean_kurtosis"] = 0.0
        signals["outlier_rate"] = 0.0

    # ── Collinearity / VIF summary ────────────────────────────────────
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        signals["max_correlation"] = round(float(corr.max().max()), 3)
        signals["high_corr_pairs"] = int((corr > 0.85).sum().sum() // 2)
    else:
        signals["max_correlation"] = 0.0
        signals["high_corr_pairs"] = 0

    # ── Classification-specific ───────────────────────────────────────
    if task_type == "classification" and target_col and target_col in df.columns:
        counts = df[target_col].value_counts()
        minority = int(counts.min())
        majority = int(counts.max())
        signals["imbalance_ratio"] = round(minority / max(majority, 1), 4)
        signals["minority_fraction"] = round(minority / max(len(df), 1), 4)
        signals["n_classes"] = int(counts.shape[0])
    else:
        signals["imbalance_ratio"] = 1.0
        signals["minority_fraction"] = 0.5
        signals["n_classes"] = 0

    return signals


# ──────────────────────────────────────────────────────────────────────────────
# Bundle loading
# ──────────────────────────────────────────────────────────────────────────────

def load_bundle_catalog(bundle_dir: str) -> List[BundleConfig]:
    """Load all ``*.yml`` bundle files from a directory."""
    bd = Path(bundle_dir)
    if not bd.is_dir():
        logger.warning("Bundle directory not found: %s", bundle_dir)
        return []
    bundles: List[BundleConfig] = []
    for p in sorted(bd.glob("*.yml")):
        try:
            with open(p, "r") as f:
                d = yaml.safe_load(f)
            if d and "bundle_name" in d:
                bundles.append(BundleConfig.from_dict(d))
        except Exception as exc:
            logger.warning("Skipping bundle %s: %s", p, exc)
    logger.info("Loaded %d bundles from %s", len(bundles), bundle_dir)
    return bundles


# ──────────────────────────────────────────────────────────────────────────────
# Gating logic
# ──────────────────────────────────────────────────────────────────────────────

_OPS = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _evaluate_rule(signal_value: Any, rule: GatingRule) -> bool:
    """Return True if the rule condition is met."""
    try:
        val = float(signal_value)
    except (TypeError, ValueError):
        return False
    op_fn = _OPS.get(rule.operator)
    if op_fn is None:
        logger.warning("Unknown operator: %s", rule.operator)
        return False
    return op_fn(val, rule.threshold)


def select_enabled_bundles(
    signals: Dict[str, Any],
    catalog: List[BundleConfig],
    default_bundles: Optional[List[str]] = None,
) -> Tuple[List[BundleConfig], List[BundleDecision]]:
    """Evaluate gating rules against signals and return enabled bundles.

    A bundle is enabled if:
      - ``default_enabled`` is True, **or**
      - its name is in *default_bundles*, **or**
      - ALL of its ``gating_rules`` are satisfied.

    Returns (enabled_bundles, decisions).
    """
    if default_bundles is None:
        default_bundles = []

    enabled: List[BundleConfig] = []
    decisions: List[BundleDecision] = []

    for bundle in catalog:
        is_default = bundle.default_enabled or bundle.bundle_name in default_bundles
        matched = 0
        reasons: List[str] = []

        for rule in bundle.gating_rules:
            sig_val = signals.get(rule.signal)
            if sig_val is None:
                reasons.append(f"Signal '{rule.signal}' not found")
                continue
            if _evaluate_rule(sig_val, rule):
                matched += 1
                reasons.append(
                    f"✅ {rule.signal}={sig_val} {rule.operator} {rule.threshold}"
                    + (f" ({rule.reason})" if rule.reason else "")
                )
            else:
                reasons.append(
                    f"❌ {rule.signal}={sig_val} NOT {rule.operator} {rule.threshold}"
                )

        total = len(bundle.gating_rules)
        rules_pass = (total == 0 and is_default) or (total > 0 and matched == total)
        final_enabled = is_default or rules_pass

        decisions.append(BundleDecision(
            bundle_name=bundle.bundle_name,
            enabled=final_enabled,
            reasons=reasons if reasons else (["default_enabled=True"] if is_default else ["no rules, not default"]),
            matched_rules=matched,
            total_rules=total,
        ))

        if final_enabled:
            enabled.append(bundle)

    return enabled, decisions


# ──────────────────────────────────────────────────────────────────────────────
# Variant path resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_variant_paths(
    bundles: List[BundleConfig],
    repo_root: str,
) -> List[str]:
    """Collect and deduplicate variant YAML paths from enabled bundles.

    Supports both explicit ``variant_paths`` and ``variant_glob`` patterns.
    Returns relative paths (from repo root).
    """
    root = Path(repo_root)
    seen: set = set()
    resolved: List[str] = []

    for bundle in bundles:
        count = 0
        # Explicit paths
        for vp in bundle.variant_paths:
            full = root / vp
            if full.is_file() and str(vp) not in seen:
                seen.add(str(vp))
                resolved.append(str(vp))
                count += 1
                if count >= bundle.max_variants:
                    break

        # Glob pattern
        if bundle.variant_glob and count < bundle.max_variants:
            for gp in sorted((root / "").parent.glob(str(root / bundle.variant_glob))):
                rel = str(gp.relative_to(root))
                if rel not in seen:
                    seen.add(rel)
                    resolved.append(rel)
                    count += 1
                    if count >= bundle.max_variants:
                        break

    logger.info("Resolved %d unique variant paths from %d bundles", len(resolved), len(bundles))
    return resolved


# ──────────────────────────────────────────────────────────────────────────────
# Artifact output
# ──────────────────────────────────────────────────────────────────────────────

def write_gating_artifacts(
    signals: Dict[str, Any],
    decisions: List[BundleDecision],
    output_dir: str,
) -> None:
    """Write ``stage_signals.json`` and ``bundle_decisions.json``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sig_path = out / "stage_signals.json"
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2, ensure_ascii=False)

    dec_list = []
    for d in decisions:
        dec_list.append({
            "bundle_name": d.bundle_name,
            "enabled": d.enabled,
            "matched_rules": d.matched_rules,
            "total_rules": d.total_rules,
            "reasons": d.reasons,
        })

    dec_path = out / "bundle_decisions.json"
    with open(dec_path, "w", encoding="utf-8") as f:
        json.dump(dec_list, f, indent=2, ensure_ascii=False)

    print(f"📡 Signals → {sig_path}")
    print(f"📋 Bundle decisions → {dec_path}")
    enabled_count = sum(1 for d in decisions if d.enabled)
    disabled_count = len(decisions) - enabled_count
    print(f"   ✅ {enabled_count} enabled, ❌ {disabled_count} disabled")
