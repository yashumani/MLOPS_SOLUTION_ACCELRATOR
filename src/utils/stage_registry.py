"""
Stage Registry — Canonical mapping of Azure ML DSL job keys to stage names.

Azure ML DSL pipeline keys (s1, s5a, s07z, etc.) are defined in pipeline_builder.py
and MUST NOT be changed. This module provides a single source of truth for:
  - Mapping DSL keys to human-readable canonical names
  - Canonical ordering for reports and banners
  - Banner generation for consistent log headers

Usage:
    from utils.stage_registry import STAGE_REGISTRY, stage_banner

    # Print banner at start of stage
    print(stage_banner("s05z"))

    # Look up canonical info
    info = STAGE_REGISTRY["s05z"]
    print(info["canonical_name"])  # "baseline_aggregate"
"""

from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Canonical Stage Registry
# ---------------------------------------------------------------------------
# Keys MUST match Azure ML DSL pipeline_builder.py job names exactly.
# These are the identifiers visible in Azure ML Studio Jobs tab.
# ---------------------------------------------------------------------------

STAGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- Data Preparation (Stages 1-4) ---
    "s1": {
        "canonical_name": "ingestion",
        "canonical_id": "S01",
        "order": 1,
        "phase": "data_preparation",
        "script": "src/steps/stage1_ingestion.py",
        "description": "Load dataset from datastore, validate schema, run EDA",
    },
    "s2": {
        "canonical_name": "preparation",
        "canonical_id": "S02",
        "order": 2,
        "phase": "data_preparation",
        "script": "src/steps/stage2_preparation.py",
        "description": "Statistical tests, imputation, high-cardinality cleanup",
    },
    "s3": {
        "canonical_name": "preprocessing",
        "canonical_id": "S03",
        "order": 3,
        "phase": "data_preparation",
        "script": "src/steps/stage3_preprocessing.py",
        "description": "Encoding, scaling, VIF analysis, SMOTE (classification only)",
    },
    "s4": {
        "canonical_name": "feature_engineering",
        "canonical_id": "S04",
        "order": 4,
        "phase": "data_preparation",
        "script": "src/steps/stage4_feature_engineering.py",
        "description": "Feature selection (boruta/mutual_info/variance), PCA, imbalance detection",
    },

    # --- Phase A: Baseline (Stages 5a, 5b, 5z) ---
    "s5a": {
        "canonical_name": "baseline_pycaret",
        "canonical_id": "S05a",
        "order": 5,
        "phase": "phase_a_baseline",
        "script": "src/steps/stage5_pycaret_train.py",
        "description": "PyCaret baseline training — compare all model families",
    },
    "s5b": {
        "canonical_name": "baseline_flaml",
        "canonical_id": "S05b",
        "order": 6,
        "phase": "phase_a_baseline",
        "script": "src/steps/stage5_flaml_train.py",
        "description": "FLAML baseline training (skipped for clustering)",
    },
    "s5z": {
        "canonical_name": "baseline_aggregate",
        "canonical_id": "S05z",
        "order": 7,
        "phase": "phase_a_baseline",
        "script": "src/steps/aggregate_baseline.py",
        "description": "Select best baseline model from PyCaret vs FLAML",
    },

    # --- Phase B: Recipe/Variant Tournament (legacy: s6a/s6b/s7a/s7b/s07z; v2: s06) ---
    "s6a": {
        "canonical_name": "phaseb_recipe1_pycaret",
        "canonical_id": "S06a",
        "order": 8,
        "phase": "phase_b_recipes",
        "script": "src/steps/phaseb_pycaret_recipe.py",
        "description": "Phase B Recipe 1 — PyCaret training (legacy pipeline)",
    },
    "s6b": {
        "canonical_name": "phaseb_recipe1_flaml",
        "canonical_id": "S06b",
        "order": 9,
        "phase": "phase_b_recipes",
        "script": "src/steps/phaseb_flaml_recipe.py",
        "description": "Phase B Recipe 1 — FLAML training (legacy pipeline)",
    },
    "s7a": {
        "canonical_name": "phaseb_recipe2_pycaret",
        "canonical_id": "S07a",
        "order": 10,
        "phase": "phase_b_recipes",
        "script": "src/steps/phaseb_pycaret_recipe.py",
        "description": "Phase B Recipe 2 — PyCaret training (legacy pipeline)",
    },
    "s7b": {
        "canonical_name": "phaseb_recipe2_flaml",
        "canonical_id": "S07b",
        "order": 11,
        "phase": "phase_b_recipes",
        "script": "src/steps/phaseb_flaml_recipe.py",
        "description": "Phase B Recipe 2 — FLAML training (legacy pipeline)",
    },
    "s07z": {
        "canonical_name": "phaseb_aggregate",
        "canonical_id": "S07z",
        "order": 12,
        "phase": "phase_b_recipes",
        "script": "src/steps/aggregate_phaseb.py",
        "description": "Select best Phase B model across recipes × engines (legacy pipeline)",
    },
    "s06": {
        "canonical_name": "phaseb_variant_runner",
        "canonical_id": "S06",
        "order": 8,
        "phase": "phase_b_variants",
        "script": "src/steps/s06_phaseb_variant_runner.py",
        "description": "Intelligent variant runner — bounded tournament with N variants (v2 pipeline)",
    },

    # --- Phase C: HPO (Stages s08, s09) ---
    "s08": {
        "canonical_name": "phasec_optuna_hpo",
        "canonical_id": "S08",
        "order": 13,
        "phase": "phase_c_hpo",
        "script": "src/steps/phasec_optuna_hpo.py",
        "description": "Optuna hyperparameter optimization on best algorithm family",
    },
    "s09": {
        "canonical_name": "phasec_aggregate",
        "canonical_id": "S09",
        "order": 14,
        "phase": "phase_c_hpo",
        "script": "src/steps/aggregate_phasec.py",
        "description": "Passthrough: copy HPO champion model as Phase C output",
    },

    # --- Final Evaluation (Stage s10) ---
    "s10": {
        "canonical_name": "final_evaluation",
        "canonical_id": "S10",
        "order": 15,
        "phase": "final",
        "script": "src/steps/final_evaluation.py",
        "description": "Compare baseline vs Phase B vs Phase C — select overall champion",
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def stage_banner(dsl_key: str, extra: str = "") -> str:
    """Generate a consistent stage banner for log output.

    Example:
        ════════════════════════════════════════════════════════════════
        [S05z] BASELINE_AGGREGATE — aggregate_baseline.py
        Select best baseline model from PyCaret vs FLAML
        ════════════════════════════════════════════════════════════════
    """
    info = STAGE_REGISTRY.get(dsl_key)
    if not info:
        return f"{'=' * 70}\n[{dsl_key}] UNKNOWN STAGE\n{'=' * 70}"

    cid = info["canonical_id"]
    name = info["canonical_name"].upper()
    script = info["script"].split("/")[-1]
    desc = info["description"]

    lines = [
        "=" * 70,
        f"[{cid}] {name} — {script}",
        desc,
    ]
    if extra:
        lines.append(extra)
    lines.append("=" * 70)
    return "\n".join(lines)


def get_canonical_name(dsl_key: str) -> str:
    """Return canonical name for an Azure ML DSL job key."""
    info = STAGE_REGISTRY.get(dsl_key)
    return info["canonical_name"] if info else dsl_key


def get_canonical_id(dsl_key: str) -> str:
    """Return canonical ID (e.g., S05z) for an Azure ML DSL job key."""
    info = STAGE_REGISTRY.get(dsl_key)
    return info["canonical_id"] if info else dsl_key


def get_signal_filename(dsl_key: str) -> str:
    """Return the expected stage_signal JSON filename for a DSL key.

    Convention: {canonical_name}_stage_signal.json
    """
    info = STAGE_REGISTRY.get(dsl_key)
    if info:
        return f"{info['canonical_name']}_stage_signal.json"
    return f"{dsl_key}_stage_signal.json"


def list_stages_in_order() -> list:
    """Return all stage entries sorted by canonical order."""
    return sorted(STAGE_REGISTRY.items(), key=lambda kv: kv[1]["order"])


def get_phase_stages(phase: str) -> list:
    """Return all stage entries for a given phase."""
    return [
        (k, v) for k, v in STAGE_REGISTRY.items() if v["phase"] == phase
    ]
