"""
Variant Recommender - Intelligent variant selection based on dataset profiling

Scores and selects the most relevant preprocessing variants for a given dataset,
reducing search space from 180 to 10-20 highly relevant configurations.

Author: MLOps Solution Accelerator V3
Date: 2026-01-26
"""

from typing import List, Tuple
from pathlib import Path
import numpy as np

from .dataset_profiler import DatasetProfile
from .variant_schema import VariantConfig, load_variant


class VariantRecommender:
    """Intelligently select variants based on dataset characteristics."""
    
    def __init__(self, profile: DatasetProfile, variant_library: List[VariantConfig]):
        """
        Args:
            profile: Dataset profile with recommendations
            variant_library: List of all available variant configurations
        """
        self.profile = profile
        self.all_variants = variant_library
        self.recommendations = profile.recommend_preprocessing_strategies()
    
    def score_variant_relevance(self, variant: VariantConfig) -> Tuple[float, List[str]]:
        """Score how relevant a variant is for this dataset.
        
        Args:
            variant: Variant configuration to score
            
        Returns:
            Tuple of (score, reasoning) where:
            - score: 0-100 relevance score
            - reasoning: List of scoring explanations
        """
        score = 0.0
        reasoning = []
        
        # Get priority weights from dataset profile
        priorities = self.recommendations["priority_scores"]
        
        # === IMPUTATION SCORING (up to 25 points) ===
        imputation_weight = priorities.get("imputation", 0.5)
        if variant.stage3_preprocessing.imputation.method in self.recommendations["imputation"]:
            imputation_score = 25.0 * imputation_weight
            score += imputation_score
            reasoning.append(f"+{imputation_score:.1f} imputation matches recommendation")
        else:
            reasoning.append(f"+0.0 imputation not recommended for this dataset")
        
        # === ENCODING SCORING (up to 20 points) ===
        encoding_weight = priorities.get("encoding", 0.5)
        if variant.stage3_preprocessing.encoding.categorical_method in self.recommendations["encoding"]:
            encoding_score = 20.0 * encoding_weight
            score += encoding_score
            reasoning.append(f"+{encoding_score:.1f} encoding matches recommendation")
        else:
            reasoning.append(f"+0.0 encoding not recommended")
        
        # === SCALING SCORING (up to 15 points) ===
        scaling_weight = priorities.get("scaling", 0.5)
        if variant.stage3_preprocessing.scaling.method in self.recommendations["scaling"]:
            scaling_score = 15.0 * scaling_weight
            score += scaling_score
            reasoning.append(f"+{scaling_score:.1f} scaling matches recommendation")
        else:
            reasoning.append(f"+0.0 scaling not recommended")
        
        # === IMBALANCE HANDLING SCORING (up to 25 points) ===
        imbalance_weight = priorities.get("imbalance_handling", 0.5)
        variant_imbalance = variant.stage3_preprocessing.imbalance_handling
        if variant_imbalance:
            imbalance_method = variant_imbalance.method
            if imbalance_method in self.recommendations["imbalance_handling"]:
                imbalance_score = 25.0 * imbalance_weight
                score += imbalance_score
                reasoning.append(f"+{imbalance_score:.1f} imbalance handling matches recommendation")
            else:
                # Penalize using SMOTE on balanced data
                if imbalance_method != "none" and self.profile.imbalance_ratio > 0.5:
                    score -= 10.0
                    reasoning.append(f"-10.0 SMOTE not needed for balanced dataset")
        
        # === FEATURE SELECTION SCORING (up to 15 points) ===
        fsel_weight = priorities.get("feature_selection", 0.5)
        if variant.stage4_feature_engineering.feature_selection.method in self.recommendations["feature_selection"]:
            fsel_score = 15.0 * fsel_weight
            score += fsel_score
            reasoning.append(f"+{fsel_score:.1f} feature selection matches recommendation")
        else:
            reasoning.append(f"+0.0 feature selection not recommended")
        
        # === LEAKAGE RISK PENALTY ===
        if variant.leakage_risk == "high":
            score *= 0.5
            reasoning.append(f"×0.5 high leakage risk penalty")
        elif variant.leakage_risk == "critical":
            score *= 0.2
            reasoning.append(f"×0.2 critical leakage risk penalty")
        
        # === BONUS: Perfect alignment ===
        if score >= 80.0:
            score += 10.0
            reasoning.append(f"+10.0 bonus for near-perfect alignment")
        
        # Clamp to [0, 100] — bonus can push above 100 (R6 audit 2026-02)
        score = min(100.0, max(0.0, score))
        
        return score, reasoning
    
    def select_top_variants(
        self,
        max_variants: int = 20,
        min_score_threshold: float = 30.0,
        diversity_boost: bool = True
    ) -> List[Tuple[VariantConfig, float, List[str]]]:
        """Select top N most relevant variants.
        
        Args:
            max_variants: Maximum number of variants to return
            min_score_threshold: Minimum relevance score to consider
            diversity_boost: If True, boost scores for diverse preprocessing combinations
            
        Returns:
            List of (variant, score, reasoning) tuples, sorted by score descending
        """
        # Score all variants
        scored = []
        for variant in self.all_variants:
            score, reasoning = self.score_variant_relevance(variant)
            if score >= min_score_threshold:
                scored.append((variant, score, reasoning))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity boost if requested
        if diversity_boost and len(scored) > max_variants:
            scored = self._apply_diversity_boost(scored, max_variants)
        
        return scored[:max_variants]
    
    def _apply_diversity_boost(
        self,
        scored: List[Tuple[VariantConfig, float, List[str]]],
        target_count: int
    ) -> List[Tuple[VariantConfig, float, List[str]]]:
        """Boost scores to ensure diverse preprocessing strategies.
        
        Prevents selecting 10 variants that differ only in scaling method.
        Ensures coverage across imputation, encoding, and feature selection.
        """
        selected = []
        seen_combinations = set()
        
        for variant, score, reasoning in scored:
            # Create diversity key (ignore scaling for diversity check)
            diversity_key = (
                variant.stage3_preprocessing.imputation.method,
                variant.stage3_preprocessing.encoding.categorical_method,
                variant.stage4_feature_engineering.feature_selection.method,
                variant.stage3_preprocessing.imbalance_handling.method if variant.stage3_preprocessing.imbalance_handling else "none"
            )
            
            # Boost score if this adds diversity
            if diversity_key not in seen_combinations:
                score += 5.0
                score = min(100.0, score)  # Re-clamp after diversity boost
                reasoning.append("+5.0 diversity bonus")
                seen_combinations.add(diversity_key)
            
            selected.append((variant, score, reasoning))
        
        # Re-sort after diversity boost
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected
    
    def generate_selection_report(
        self,
        selected_variants: List[Tuple[VariantConfig, float, List[str]]]
    ) -> str:
        """Generate human-readable variant selection report."""
        report = f"""
=== VARIANT SELECTION REPORT ===
Dataset Profile: {self.profile.n_rows} rows × {self.profile.n_features} features
Task Type: {self.profile.target_type}
Selected: {len(selected_variants)} variants from {len(self.all_variants)} available

TOP RECOMMENDATIONS:
"""
        for i, (variant, score, reasoning) in enumerate(selected_variants[:10], 1):
            report += f"\n{i}. Variant {variant.variant_id} (Score: {score:.1f}/100)\n"
            report += f"   Config: {variant.stage3_preprocessing.imputation.method}+"
            report += f"{variant.stage3_preprocessing.encoding.categorical_method}+"
            report += f"{variant.stage3_preprocessing.scaling.method}+"
            imb_method = variant.stage3_preprocessing.imbalance_handling.method if variant.stage3_preprocessing.imbalance_handling else "none"
            report += f"{imb_method}+"
            report += f"{variant.stage4_feature_engineering.feature_selection.method}\n"
            report += f"   Leakage Risk: {variant.leakage_risk}\n"
            # Show top 3 reasoning items
            for reason in reasoning[:3]:
                report += f"   {reason}\n"
        
        if len(selected_variants) > 10:
            report += f"\n... and {len(selected_variants) - 10} more variants\n"
        
        # Coverage analysis
        report += f"\nCOVERAGE ANALYSIS:\n"
        imputation_methods = set(v.stage3_preprocessing.imputation.method for v, _, _ in selected_variants)
        encoding_methods = set(v.stage3_preprocessing.encoding.categorical_method for v, _, _ in selected_variants)
        scaling_methods = set(v.stage3_preprocessing.scaling.method for v, _, _ in selected_variants)
        
        report += f"  Imputation methods: {', '.join(imputation_methods)}\n"
        report += f"  Encoding methods: {', '.join(encoding_methods)}\n"
        report += f"  Scaling methods: {', '.join(scaling_methods)}\n"
        
        return report
