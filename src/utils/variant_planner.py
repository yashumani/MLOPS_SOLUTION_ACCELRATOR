"""
Variant Planner - Adaptive Search Space Generation & Progressive Refinement

Provides data-driven variant scoring, diverse sampling, and shortlisting logic
for the V3-Proposed Phase B intelligent variant runner.

DESIGN PRINCIPLES:
- Explainability: Every decision traceable with human-readable reasoning
- Deterministic: Same config + dataset fingerprint = same shortlist
- Simple heuristics: No bandits/RL, just interpretable rules

Author: MLOps Solution Accelerator V3
Date: 2026-02-08
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime


@dataclass
class EdaPriors:
    """Data-driven priors extracted from Stage 1 EDA."""
    missing_rate: float = 0.0
    imbalance_ratio: float = 1.0  # 1.0 = balanced
    high_cardinality_cols: List[str] = field(default_factory=list)
    outlier_prevalence: float = 0.0
    skewness_issues: int = 0  # Count of columns with |skew| > 1
    n_rows: int = 0
    n_features: int = 0
    
    @classmethod
    def from_eda_report(cls, eda_report: Dict[str, Any]) -> 'EdaPriors':
        """Parse EDA report JSON into priors."""
        return cls(
            missing_rate=eda_report.get("missing_rate", 0.0),
            imbalance_ratio=eda_report.get("imbalance_ratio", 1.0),
            high_cardinality_cols=eda_report.get("high_cardinality_cols", []),
            outlier_prevalence=eda_report.get("outlier_prevalence", 0.0),
            skewness_issues=eda_report.get("skewness_issues", 0),
            n_rows=eda_report.get("n_rows", 0),
            n_features=eda_report.get("n_features", 0)
        )


@dataclass
class VariantScore:
    """Scored variant with reasoning."""
    variant_id: str
    variant_path: str
    relevance_score: float  # 0-100
    reasoning: List[str]
    preprocessing_hash: str
    imputation: str
    encoding: str
    scaling: str
    imbalance: str
    feature_selection: str


@dataclass
class VariantPlan:
    """Complete variant plan with shortlist and budget allocation."""
    planner_version: str = "1.0"
    dataset_fingerprint: str = ""
    eda_priors: Dict[str, Any] = field(default_factory=dict)
    round0_summary: Dict[str, Any] = field(default_factory=dict)
    round1_summary: Dict[str, Any] = field(default_factory=dict)
    shortlist: List[Dict[str, Any]] = field(default_factory=list)
    budget_allocation: Dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def compute_preprocessing_hash(variant_config: Dict[str, Any]) -> str:
    """Compute deterministic hash of preprocessing config."""
    # Extract just the preprocessing dimensions
    config_str = json.dumps({
        "imputation": variant_config.get("stage3_preprocessing", {}).get("imputation", {}).get("method", "none"),
        "encoding": variant_config.get("stage3_preprocessing", {}).get("encoding", {}).get("categorical_method", "none"),
        "scaling": variant_config.get("stage3_preprocessing", {}).get("scaling", {}).get("method", "none"),
        "imbalance": variant_config.get("stage3_preprocessing", {}).get("imbalance_handling", {}).get("method", "none"),
        "feature_selection": variant_config.get("stage4_feature_engineering", {}).get("feature_selection", {}).get("method", "none"),
    }, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


def score_variant_relevance(
    variant_config: Dict[str, Any],
    eda_priors: EdaPriors,
    task_type: str = "classification"
) -> Tuple[float, List[str]]:
    """
    Score a variant's relevance based on EDA priors.
    
    DESIGN DECISION (R5 audit 2026-02):
    - Round-1 is STATISTICAL_SCORING only (no model training in the planner).
    - Base score intentionally starts at 50 to give every variant a chance
      (vs recommender's base=0 which is pure alignment-driven).
    - Output clamped to [0, 100] at return via min/max.
    - If all variants are pruned post-proxy, build_variant_plan() falls back
      to the top-scored variant to prevent empty shortlists.
    
    Returns:
        (score 0-100, list of reasoning strings)
    """
    score = 50.0  # Base score
    reasoning = []
    
    # Extract preprocessing methods
    stage3 = variant_config.get("stage3_preprocessing", {})
    stage4 = variant_config.get("stage4_feature_engineering", {})
    
    imputation = stage3.get("imputation", {}).get("method", "none")
    encoding = stage3.get("encoding", {}).get("categorical_method", "onehot")
    scaling = stage3.get("scaling", {}).get("method", "none")
    imbalance = stage3.get("imbalance_handling", {}).get("method", "none")
    feat_sel = stage4.get("feature_selection", {}).get("method", "none")
    
    # === IMPUTATION SCORING (25 points max) ===
    if eda_priors.missing_rate > 0.15:
        # High missing: advanced/robust methods score highest
        if imputation in ["knn", "iterative"]:
            score += 25
            reasoning.append(f"missing_rate={eda_priors.missing_rate:.2f} → advanced imputation ({imputation}) +25")
        elif imputation in ["trimmed_mean", "winsorized_mean"]:
            score += 22
            reasoning.append(f"missing_rate={eda_priors.missing_rate:.2f} → robust statistical ({imputation}) +22")
        elif imputation in ["interpolate_linear", "forward_fill", "backward_fill"]:
            score += 18
            reasoning.append(f"missing_rate={eda_priors.missing_rate:.2f} → contextual imputation ({imputation}) +18")
        elif imputation in ["numeric_mean_cat_mode", "numeric_median_cat_mode"]:
            score += 17
            reasoning.append(f"missing_rate={eda_priors.missing_rate:.2f} → composite imputation ({imputation}) +17")
        elif imputation == "median":
            score += 15
            reasoning.append(f"missing_rate={eda_priors.missing_rate:.2f} → median imputation +15")
        elif imputation == "random_sample":
            score += 14
            reasoning.append(f"missing_rate={eda_priors.missing_rate:.2f} → distribution-preserving ({imputation}) +14")
        elif imputation in ["mean", "mode"]:
            score += 10
            reasoning.append(f"missing_rate={eda_priors.missing_rate:.2f} → basic imputation ({imputation}) +10")
        elif imputation in ["constant", "zero_fill"]:
            score += 6
            reasoning.append(f"missing_rate={eda_priors.missing_rate:.2f} → simple fill ({imputation}) +6")
        elif imputation == "drop":
            score += 3
            reasoning.append(f"missing_rate={eda_priors.missing_rate:.2f} → drop rows (risky at high missing) +3")
    elif eda_priors.missing_rate > 0.05:
        # Moderate missing: simple methods are adequate, composites shine
        if imputation in ["numeric_mean_cat_mode", "numeric_median_cat_mode"]:
            score += 22
            reasoning.append(f"moderate missing_rate={eda_priors.missing_rate:.2f} → composite imputation ({imputation}) +22")
        elif imputation in ["mean", "median", "mode"]:
            score += 20
            reasoning.append(f"moderate missing_rate={eda_priors.missing_rate:.2f} → simple imputation ({imputation}) +20")
        elif imputation in ["knn", "iterative"]:
            score += 18
            reasoning.append(f"moderate missing_rate={eda_priors.missing_rate:.2f} → advanced (overkill but safe) ({imputation}) +18")
        elif imputation in ["trimmed_mean", "winsorized_mean", "random_sample"]:
            score += 17
            reasoning.append(f"moderate missing_rate={eda_priors.missing_rate:.2f} → robust statistical ({imputation}) +17")
        elif imputation in ["forward_fill", "backward_fill", "interpolate_linear"]:
            score += 15
            reasoning.append(f"moderate missing_rate={eda_priors.missing_rate:.2f} → contextual ({imputation}) +15")
        elif imputation in ["constant", "zero_fill"]:
            score += 10
            reasoning.append(f"moderate missing_rate={eda_priors.missing_rate:.2f} → simple fill ({imputation}) +10")
        elif imputation == "drop":
            score += 12
            reasoning.append(f"moderate missing_rate={eda_priors.missing_rate:.2f} → drop rows (acceptable) +12")
    else:
        score += 15  # Low missing rate, any imputation is fine
        reasoning.append(f"low missing_rate={eda_priors.missing_rate:.2f} → imputation neutral +15")
    
    # === ENCODING SCORING (20 points max) ===
    has_high_card = len(eda_priors.high_cardinality_cols) > 0
    if has_high_card:
        if encoding in ["target", "label"]:
            score += 20
            reasoning.append(f"high_cardinality_cols={len(eda_priors.high_cardinality_cols)} → {encoding} encoding +20")
        elif encoding == "onehot":
            score -= 10
            reasoning.append(f"high_cardinality + onehot = feature explosion risk -10")
    else:
        if encoding == "onehot":
            score += 15
            reasoning.append(f"no high_cardinality → onehot safe +15")
        else:
            score += 10
            reasoning.append(f"encoding={encoding} +10")
    
    # === SCALING SCORING (15 points max) ===
    if eda_priors.outlier_prevalence > 0.05 or eda_priors.skewness_issues > 3:
        if scaling == "robust":
            score += 15
            reasoning.append(f"outliers/skew detected → robust scaling +15")
        elif scaling == "standard":
            score += 5
            reasoning.append(f"outliers/skew detected but standard scaling +5")
    else:
        score += 10  # Neutral
        reasoning.append(f"scaling={scaling} +10")
    
    # === IMBALANCE SCORING (25 points max) ===
    if task_type == "classification" and eda_priors.imbalance_ratio < 0.3:
        if imbalance == "smote":
            score += 25
            reasoning.append(f"imbalance_ratio={eda_priors.imbalance_ratio:.2f} → SMOTE critical +25")
        elif imbalance == "adasyn":
            score += 22
            reasoning.append(f"imbalance_ratio={eda_priors.imbalance_ratio:.2f} → ADASYN +22")
        elif imbalance == "none":
            score -= 5
            reasoning.append(f"imbalance_ratio={eda_priors.imbalance_ratio:.2f} but no handling -5")
    elif task_type != "classification" and imbalance == "smote":
        score -= 15
        reasoning.append(f"SMOTE invalid for {task_type} -15")
    else:
        score += 10  # Neutral
        reasoning.append(f"imbalance_handling={imbalance} +10")
    
    # === FEATURE SELECTION SCORING (15 points max) ===
    if eda_priors.n_features > 50:
        if feat_sel in ["correlation", "variance", "rfe", "mutual_info"]:
            score += 15
            reasoning.append(f"n_features={eda_priors.n_features} → feature selection ({feat_sel}) +15")
        else:
            score += 5
            reasoning.append(f"n_features={eda_priors.n_features} but no feature selection +5")
    else:
        score += 10
        reasoning.append(f"feature_selection={feat_sel} +10")
    
    # === LEAKAGE RISK PENALTY ===
    leakage_risk = variant_config.get("variant_metadata", {}).get("leakage_risk", "none")
    if leakage_risk == "high":
        score *= 0.5
        reasoning.append(f"LEAKAGE RISK HIGH → score ×0.5")
    elif leakage_risk == "critical":
        score *= 0.2
        reasoning.append(f"LEAKAGE RISK CRITICAL → score ×0.2")
    
    return round(min(100, max(0, score)), 2), reasoning


def compute_hamming_distance(v1: Dict[str, str], v2: Dict[str, str]) -> int:
    """Compute Hamming distance between two preprocessing configs."""
    distance = 0
    for key in ["imputation", "encoding", "scaling", "imbalance", "feature_selection"]:
        if v1.get(key) != v2.get(key):
            distance += 1
    return distance


def diverse_sample(
    scored_variants: List[VariantScore],
    max_count: int,
    min_hamming_distance: int = 2
) -> List[VariantScore]:
    """
    Select diverse subset using greedy furthest-first sampling.
    
    Ensures selected variants differ by at least min_hamming_distance
    in preprocessing dimensions.
    """
    if len(scored_variants) <= max_count:
        return scored_variants
    
    # Sort by score descending
    sorted_variants = sorted(scored_variants, key=lambda v: v.relevance_score, reverse=True)
    
    selected = [sorted_variants[0]]  # Start with highest scored
    
    for candidate in sorted_variants[1:]:
        if len(selected) >= max_count:
            break
        
        # Check diversity against all selected
        candidate_config = {
            "imputation": candidate.imputation,
            "encoding": candidate.encoding,
            "scaling": candidate.scaling,
            "imbalance": candidate.imbalance,
            "feature_selection": candidate.feature_selection
        }
        
        is_diverse = True
        for sel in selected:
            sel_config = {
                "imputation": sel.imputation,
                "encoding": sel.encoding,
                "scaling": sel.scaling,
                "imbalance": sel.imbalance,
                "feature_selection": sel.feature_selection
            }
            if compute_hamming_distance(candidate_config, sel_config) < min_hamming_distance:
                is_diverse = False
                break
        
        if is_diverse:
            selected.append(candidate)
    
    # If we couldn't fill quota with diversity, add remaining by score
    if len(selected) < max_count:
        for v in sorted_variants:
            if v not in selected:
                selected.append(v)
                if len(selected) >= max_count:
                    break
    
    return selected


def ensure_coverage(
    selected: List[VariantScore],
    all_scored: List[VariantScore],
    dimension: str,
    min_per_method: int = 1
) -> List[VariantScore]:
    """Ensure minimum coverage of a preprocessing dimension."""
    # Count methods in current selection
    method_counts = {}
    for v in selected:
        method = getattr(v, dimension)
        method_counts[method] = method_counts.get(method, 0) + 1
    
    # Get all methods available
    all_methods = set(getattr(v, dimension) for v in all_scored)
    
    # Find underrepresented methods
    for method in all_methods:
        if method_counts.get(method, 0) < min_per_method:
            # Find best candidate with this method not in selected
            candidates = [v for v in all_scored if getattr(v, dimension) == method and v not in selected]
            if candidates:
                best = max(candidates, key=lambda v: v.relevance_score)
                selected.append(best)
    
    return selected


def apply_proxy_pruning(
    variants: List[VariantScore],
    proxy_results: Dict[str, float],
    threshold: float,
    task_type: str = "classification"
) -> Tuple[List[VariantScore], List[str]]:
    """
    Prune variants below proxy threshold.
    
    Returns:
        (remaining_variants, pruned_variant_ids)
    """
    remaining = []
    pruned = []
    
    for v in variants:
        proxy_score = proxy_results.get(v.variant_id, 0.0)
        
        # Adjust threshold for regression (R2 can be negative)
        effective_threshold = threshold if task_type == "classification" else -0.5
        
        if proxy_score >= effective_threshold:
            remaining.append(v)
        else:
            pruned.append(v.variant_id)
    
    return remaining, pruned


def build_variant_plan(
    variant_configs: List[Dict[str, Any]],
    variant_paths: List[str],
    eda_priors: EdaPriors,
    task_type: str,
    planner_config: Dict[str, Any],
    round0_results: Optional[Dict[str, Dict]] = None,
    round1_results: Optional[Dict[str, float]] = None
) -> VariantPlan:
    """
    Build complete variant plan with shortlist and reasoning.
    
    Args:
        variant_configs: List of variant config dicts
        variant_paths: Corresponding file paths
        eda_priors: Data-driven priors from Stage 1
        task_type: classification/regression/clustering
        planner_config: Planner configuration from YAML
        round0_results: Optional dict {variant_id: {status, reason, ...}}
        round1_results: Optional dict {variant_id: proxy_score}
    
    Returns:
        VariantPlan with shortlist and explainability
    """
    plan = VariantPlan()
    plan.eda_priors = asdict(eda_priors)
    
    # Extract config
    round1_max = planner_config.get("round1_max_variants", 40)
    round2_max = planner_config.get("round2_max_variants", 10)
    proxy_threshold = planner_config.get("proxy_prune_threshold", 0.50)
    min_hamming = planner_config.get("diversity_min_hamming_distance", 2)
    
    # Score all variants
    scored_variants = []
    for cfg, path in zip(variant_configs, variant_paths):
        variant_id = cfg.get("recipe_name") or cfg.get("variant_metadata", {}).get("variant_id", "unknown")
        score, reasoning = score_variant_relevance(cfg, eda_priors, task_type)
        
        stage3 = cfg.get("stage3_preprocessing", {})
        stage4 = cfg.get("stage4_feature_engineering", {})
        
        scored_variants.append(VariantScore(
            variant_id=variant_id,
            variant_path=path,
            relevance_score=score,
            reasoning=reasoning,
            preprocessing_hash=compute_preprocessing_hash(cfg),
            imputation=stage3.get("imputation", {}).get("method", "none"),
            encoding=stage3.get("encoding", {}).get("categorical_method", "onehot"),
            scaling=stage3.get("scaling", {}).get("method", "none"),
            imbalance=stage3.get("imbalance_handling", {}).get("method", "none"),
            feature_selection=stage4.get("feature_selection", {}).get("method", "none")
        ))
    
    # Filter by Round 0 results if available
    if round0_results:
        passed_ids = {vid for vid, res in round0_results.items() if res.get("status") == "pass"}
        scored_variants = [v for v in scored_variants if v.variant_id in passed_ids]
        plan.round0_summary = {
            "total_variants": len(variant_configs),
            "passed": len(passed_ids),
            "failed": len(variant_configs) - len(passed_ids)
        }
    
    # Diverse sample for Round 1
    round1_candidates = diverse_sample(scored_variants, round1_max, min_hamming)
    
    # Ensure encoding coverage
    if planner_config.get("diversity_coverage_enabled", True):
        round1_candidates = ensure_coverage(round1_candidates, scored_variants, "encoding", min_per_method=1)
    
    # Apply proxy pruning if Round 1 results available
    if round1_results:
        round1_candidates, pruned = apply_proxy_pruning(
            round1_candidates, round1_results, proxy_threshold, task_type
        )
        plan.round1_summary = {
            "sampled": len(scored_variants),
            "trained": len(round1_results),
            "pruned_below_threshold": len(pruned),
            "shortlisted": min(round2_max, len(round1_candidates))
        }
    
    # Final shortlist (top-K by score)
    final_shortlist = sorted(round1_candidates, key=lambda v: v.relevance_score, reverse=True)[:round2_max]
    
    # GUARD (R5 audit 2026-02): If all variants are pruned, fall back to the
    # top-scored unpruned variant so s06 never receives an empty shortlist.
    if not final_shortlist and scored_variants:
        top_fallback = sorted(scored_variants, key=lambda v: v.relevance_score, reverse=True)[0]
        final_shortlist = [top_fallback]
        print(f"⚠️ VariantPlanner: All variants pruned — falling back to top-scored variant {top_fallback.variant_id} (score={top_fallback.relevance_score})")
    
    # Build shortlist with reasoning
    for rank, v in enumerate(final_shortlist, 1):
        plan.shortlist.append({
            "variant_id": v.variant_id,
            "variant_path": v.variant_path,
            "relevance_score": v.relevance_score,
            "ranking": rank,
            "reasoning": v.reasoning,
            "preprocessing_hash": v.preprocessing_hash,
            "config": {
                "imputation": v.imputation,
                "encoding": v.encoding,
                "scaling": v.scaling,
                "imbalance": v.imbalance,
                "feature_selection": v.feature_selection
            }
        })
    
    # Budget allocation
    round0_budget = planner_config.get("round0_budget_per_variant_sec", 10) * len(variant_configs)
    round1_budget = planner_config.get("round1_budget_per_variant_sec", 60) * round1_max
    round2_budget = planner_config.get("round2_budget_per_variant_sec", 300) * round2_max * 2  # 2 engines
    
    plan.budget_allocation = {
        "round0_budget_sec": round0_budget,
        "round1_budget_sec": round1_budget,
        "round2_budget_sec": round2_budget,
        "total_budget_sec": round0_budget + round1_budget + round2_budget
    }
    
    return plan


def load_eda_priors_from_file(eda_report_path: str) -> EdaPriors:
    """Load EDA priors from Stage 1 output file."""
    try:
        with open(eda_report_path, 'r') as f:
            eda_data = json.load(f)
        return EdaPriors.from_eda_report(eda_data)
    except Exception as e:
        print(f"⚠️ Could not load EDA priors: {e}")
        return EdaPriors()  # Return defaults


def get_default_planner_config() -> Dict[str, Any]:
    """Return default planner configuration."""
    return {
        "enabled": True,
        "round0_enabled": True,
        "round0_budget_per_variant_sec": 10,
        "round0_feature_explosion_limit": 500,
        "round1_max_variants": 40,
        "round1_budget_per_variant_sec": 60,
        "round1_sample_size": 5000,
        "proxy_prune_threshold": 0.50,
        "proxy_prune_threshold_regression": -0.5,
        "round2_max_variants": 10,
        "round2_budget_per_variant_sec": 300,
        "diversity_min_hamming_distance": 2,
        "diversity_coverage_enabled": True,
        "cache_enabled": True,
        "cache_scope": "in_memory",
        "total_budget_sec": 7200
    }
