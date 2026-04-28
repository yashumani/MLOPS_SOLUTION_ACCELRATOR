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
    "s5t": {
        "canonical_name": "baseline_timeseries",
        "canonical_id": "S05t",
        "order": 8,
        "phase": "phase_a_baseline",
        "script": "src/steps/stage5_timeseries_train.py",
        "description": "Time-series baseline training (skipped for non-forecasting tasks)",
    },

    # --- Phase B: Variant Runner (active: s06) ---
    # NOTE: Legacy keys (s6a/s6b/s7a/s7b/s07z) are not part of the active pipeline.
    # They are retained in LEGACY_STAGE_REGISTRY (below) for back-compat with
    # historical job artifacts only.
    "s06": {
        "canonical_name": "phaseb_variant_runner",
        "canonical_id": "S06",
        "order": 9,
        "phase": "phase_b_variants",
        "script": "src/steps/s06_phaseb_variant_runner.py",
        "description": "Intelligent variant runner — bounded tournament with N variants",
    },

    # --- Phase C: HPO (Stages s08, s09) ---
    "s08": {
        "canonical_name": "phasec_optuna_hpo",
        "canonical_id": "S08",
        "order": 10,
        "phase": "phase_c_hpo",
        "script": "src/steps/phasec_optuna_hpo.py",
        "description": "Optuna hyperparameter optimization on best algorithm family",
    },
    "s09": {
        "canonical_name": "phasec_aggregate",
        "canonical_id": "S09",
        "order": 11,
        "phase": "phase_c_hpo",
        "script": "src/steps/aggregate_phasec.py",
        "description": "Passthrough: copy HPO champion model as Phase C output",
    },

    # --- Final Evaluation (Stage s10) ---
    "s10": {
        "canonical_name": "final_evaluation",
        "canonical_id": "S10",
        "order": 12,
        "phase": "final",
        "script": "src/steps/final_evaluation.py",
        "description": "Compare baseline vs Phase B vs Phase C — select overall champion",
    },
    "s12": {
        "canonical_name": "model_registration",
        "canonical_id": "S12",
        "order": 13,
        "phase": "registration",
        "script": "src/steps/s12_model_registration.py",
        "description": "Register the selected champion model when quality gates pass",
    },
    "s13": {
        "canonical_name": "drift_monitor",
        "canonical_id": "S13",
        "order": 14,
        "phase": "monitoring",
        "script": "src/steps/s13_drift_monitor.py",
        "description": "Generate drift report and baseline artifacts for monitoring",
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


# ---------------------------------------------------------------------------
# Legacy stage entries (NOT in active pipeline)
# ---------------------------------------------------------------------------
# Retained only so historical job artifacts referencing s6a/s6b/s7a/s7b/s07z
# can still resolve to a human-readable label. Do NOT use for new code.
# ---------------------------------------------------------------------------

LEGACY_STAGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "s6a": {
        "canonical_name": "phaseb_recipe1_pycaret",
        "canonical_id": "S06a",
        "phase": "phase_b_recipes_legacy",
        "script": "src/steps/phaseb_pycaret_recipe.py",
        "description": "Phase B Recipe 1 — PyCaret training (legacy)",
    },
    "s6b": {
        "canonical_name": "phaseb_recipe1_flaml",
        "canonical_id": "S06b",
        "phase": "phase_b_recipes_legacy",
        "script": "src/steps/phaseb_flaml_recipe.py",
        "description": "Phase B Recipe 1 — FLAML training (legacy)",
    },
    "s7a": {
        "canonical_name": "phaseb_recipe2_pycaret",
        "canonical_id": "S07a",
        "phase": "phase_b_recipes_legacy",
        "script": "src/steps/phaseb_pycaret_recipe.py",
        "description": "Phase B Recipe 2 — PyCaret training (legacy)",
    },
    "s7b": {
        "canonical_name": "phaseb_recipe2_flaml",
        "canonical_id": "S07b",
        "phase": "phase_b_recipes_legacy",
        "script": "src/steps/phaseb_flaml_recipe.py",
        "description": "Phase B Recipe 2 — FLAML training (legacy)",
    },
    "s07z": {
        "canonical_name": "phaseb_aggregate",
        "canonical_id": "S07z",
        "phase": "phase_b_recipes_legacy",
        "script": "src/steps/aggregate_phaseb.py",
        "description": "Phase B aggregation across recipes × engines (legacy)",
    },
}


def lookup_stage(dsl_key: str) -> Optional[Dict[str, Any]]:
    """Resolve a DSL key against active first, then legacy registries."""
    return STAGE_REGISTRY.get(dsl_key) or LEGACY_STAGE_REGISTRY.get(dsl_key)
