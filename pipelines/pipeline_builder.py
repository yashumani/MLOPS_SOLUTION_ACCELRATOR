import logging
import yaml
from azure.ai.ml import dsl, Input, Output
from azure.ai.ml.entities import PipelineJob, UserIdentityConfiguration
from azure.ai.ml import load_component
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve component paths relative to repo root
ROOT = Path(__file__).resolve().parents[1]

# [PB1/PB2] Component manifest tracks every successfully loaded component for
# diagnostics. Populated by _load_component_safe at module import time.
_COMPONENT_MANIFEST: dict[str, str] = {}


def _load_component_safe(name: str, source: str):
    """Load an Azure ML component YAML with a clear error message on failure.

    Wraps load_component() so that import-time failures (missing YAML, schema
    errors, etc.) surface a RuntimeError naming the component and source path
    instead of an opaque azure-ai-ml traceback.
    """
    try:
        comp = load_component(source=source)
        _COMPONENT_MANIFEST[name] = source
        logger.info(f"[component] loaded {name} from {source}")
        return comp
    except Exception as e:
        raise RuntimeError(
            f"Failed to load component '{name}' from {source}: {e}"
        ) from e


def _phase_b_safety_net_review_required(config_name: str) -> bool:
    """Return the Phase B safety-net review flag for submit-time diagnostics."""
    try:
        path = ROOT / "configs" / str(config_name)
        if not path.exists():
            path = ROOT / "configs" / f"{config_name}.yml"
        if not path.exists():
            return True
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        phase_b = (cfg.get("phases") or {}).get("phase_b") or (cfg.get("phases") or {}).get("phase_b_recipes") or {}
        if isinstance(phase_b, dict):
            return bool(phase_b.get("safety_net_review_required", True))
    except Exception as exc:  # noqa: BLE001 - diagnostics only; never block graph construction
        logger.warning("could not inspect Phase B safety-net review flag: %s", exc)
    return True


# Load components fresh from YAML files
# Note: Version changes ensure Azure ML reloads from disk
ingestion = _load_component_safe("ingestion", str(ROOT / "components/stage1_ingestion.yml"))
preparation = _load_component_safe("preparation", str(ROOT / "components/stage2_preparation.yml"))
preprocessing = _load_component_safe("preprocessing", str(ROOT / "components/stage3_preprocessing.yml"))
feature_eng = _load_component_safe("feature_eng", str(ROOT / "components/stage4_feature_engineering.yml"))
pycaret_train = _load_component_safe("pycaret_train", str(ROOT / "components/stage5_pycaret_train.yml"))
flaml_train = _load_component_safe("flaml_train", str(ROOT / "components/stage5_flaml_train.yml"))
agg_baseline = _load_component_safe("agg_baseline", str(ROOT / "components/aggregate_baseline.yml"))
# Dead per-recipe components removed (P3-1): phaseb_pycaret, phaseb_flaml, agg_phaseb
# Replaced by s06_phaseb_variant_runner (single-step batch processor)
variant_runner = _load_component_safe("variant_runner", str(ROOT / "components/s06_phaseb_variant_runner.yml"))
phasec_hpo = _load_component_safe("phasec_hpo", str(ROOT / "components/phasec_optuna_hpo.yml"))
agg_phasec = _load_component_safe("agg_phasec", str(ROOT / "components/aggregate_phasec.yml"))
final_eval = _load_component_safe("final_eval", str(ROOT / "components/final_evaluation.yml"))
model_reg = _load_component_safe("model_reg", str(ROOT / "components/s12_model_registration.yml"))
# Drift detection (s13) — additive layer on top of v3-production
drift_monitor = _load_component_safe("drift_monitor", str(ROOT / "components/s13_drift_monitor.yml"))
# Auto-retrain decision gate (s14) — decision artifact only; controller owns submissions
retrain_decision = _load_component_safe("retrain_decision", str(ROOT / "components/s14_retrain_decision.yml"))

# [PB2] One-time summary log of all components loaded at import time.
logger.info(
    f"[component-manifest] loaded {len(_COMPONENT_MANIFEST)} components: "
    f"{list(_COMPONENT_MANIFEST.keys())}"
)

@dsl.pipeline(compute=None)
def full_pipeline(
    config_name: str,
    dataset_folder: Input(type="uri_folder"),
    execution_manifest: Input(type="uri_file"),
    candidate_catalog: Input(type="uri_file"),
    variants_list: str = "",
    engine_list: str = "pycaret,flaml",
    time_budget_per_variant: int = 300,
    phaseb_time_budget_sec: int = 10800,
    drift_baseline_in: Input(type="uri_folder", optional=True) = None,
    drift_baseline_uri: str = "",
):
    """V3 production pipeline — runs ALL selected variants via the variant runner.
    
    ARCHITECTURE (Feb 2026):
    - Stages 1-4: Ingestion → Preparation → Preprocessing → Feature Engineering
    - Stage 5 (Baseline): PyCaret + FLAML (parallel)
    - Stage 6 (Phase B): Single variant runner step that processes N variants
      with nested MLflow runs per variant×engine. No more 2-recipe limit.
    - Stage 8 (Phase C): Optuna HPO
    - Stage 10 (Final): Champion selection across all phases
    
    Args:
        config_name: Config YAML filename (from uploaded code/configs directory)
        dataset_folder: Datastore folder URI containing dataset
        candidate_catalog: Immutable complete recipe/candidate catalog artifact
        variants_list: Bounded compatibility input for older direct callers
        engine_list: Comma-separated engines (e.g. "pycaret,flaml")
        time_budget_per_variant: Max seconds per variant training
    """
    s1 = ingestion(config_name=config_name, dataset_in=dataset_folder)
    s2 = preparation(config_name=config_name, dataset_in=s1.outputs.dataset_out,
                     eda_report=s1.outputs.eda_report)
    s2.outputs.train_out = Output(type="uri_file")
    s2.outputs.raw_train_out = Output(type="uri_file")
    s2.outputs.raw_holdout_out = Output(type="uri_file")
    s2.outputs.split_manifest_out = Output(type="uri_file")
    s3 = preprocessing(config_name=config_name, dataset_in=s2.outputs.dataset_out, prep_report=s2.outputs.prep_report, recipe_name="recipe_baseline.yml")
    s4 = feature_eng(config_name=config_name, dataset_in=s3.outputs.dataset_out, recipe_name="recipe_baseline.yml")
    
    # Baseline training - explicitly wire outputs to force Azure ML recognition
    s5a = pycaret_train(
        config_name=config_name,
        dataset_in=s2.outputs.raw_train_out,
        split_manifest=s2.outputs.split_manifest_out,
    )
    s5a.outputs.metrics_json = Output(type="uri_file")
    s5a.outputs.manifest_json = Output(type="uri_file")
    s5a.outputs.best_model = Output(type="uri_folder")
    
    s5b = flaml_train(
        config_name=config_name,
        dataset_in=s2.outputs.raw_train_out,
        split_manifest=s2.outputs.split_manifest_out,
    )
    s5b.outputs.metrics_json = Output(type="uri_file")
    s5b.outputs.manifest_json = Output(type="uri_file")
    s5b.outputs.best_model = Output(type="uri_folder")
    
    s5z = agg_baseline(
        config_name=config_name,
        pycaret_manifest=s5a.outputs.manifest_json,
        pycaret_model=s5a.outputs.best_model,
        flaml_manifest=s5b.outputs.manifest_json,
        flaml_model=s5b.outputs.best_model,
    )
    
    # Phase B — ALL variants processed in a single step with nested MLflow runs.
    # The variant_runner handles N variants × M engines internally.
    # Each variant×engine gets its own nested MLflow run for full traceability.
    # C2 FIX: Wire to s2 (prepared data) so variant runner applies its OWN preprocessing recipes.
    # Using s4 (already preprocessed) would double-preprocess and defeat variant-specific recipes.
    if _phase_b_safety_net_review_required(config_name):
        logger.warning(
            "Phase B safety-net champions require operator review before registration."
        )
    s06_kwargs = dict(
        config_name=config_name,
        engine_list=engine_list,
        dataset_in=s2.outputs.raw_train_out,
        split_manifest=s2.outputs.split_manifest_out,
        time_budget_per_variant=time_budget_per_variant,
        phaseb_time_budget_sec=phaseb_time_budget_sec,
    )
    s06_kwargs["execution_manifest"] = execution_manifest
    s06_kwargs["candidate_catalog"] = candidate_catalog
    s06 = variant_runner(**s06_kwargs)
    s06.outputs.leaderboard_csv = Output(type="uri_file")
    s06.outputs.all_results_json = Output(type="uri_file")
    s06.outputs.champion_manifest = Output(type="uri_file")
    s06.outputs.champion_model = Output(type="uri_folder")
    s06.outputs.execution_manifest_out = Output(type="uri_file")
    s06.outputs.split_manifest_out = Output(type="uri_file")
    s06.outputs.quality_decision_out = Output(type="uri_file")
    
    # Phase C - Optuna HPO (s08) and aggregate (s09)
    # 🔥 FIX (A1): Wire Phase B champion manifest so Phase C tunes the correct algorithm
    # Phase C fits the complete champion recipe on the same raw/prepared
    # training-only boundary as Phase B.
    s08 = phasec_hpo(
        config_name=config_name,
        dataset_in=s2.outputs.raw_train_out,
        execution_manifest=s06.outputs.execution_manifest_out,
        phaseb_manifest=s06.outputs.champion_manifest,
    )
    s08.outputs.hpo_metrics_json = Output(type="uri_file")
    s08.outputs.optimized_model = Output(type="uri_folder")
    
    s09 = agg_phasec(
        config_name=config_name,
        hpo_metrics_json=s08.outputs.hpo_metrics_json,
        optimized_model=s08.outputs.optimized_model,
    )
    
    # Final evaluation (s10) - select champion among baseline, phase B, phase C
    s10 = final_eval(
        config_name=config_name,
        dataset_in=s2.outputs.raw_train_out,
        holdout_in=s2.outputs.raw_holdout_out,
        split_manifest_in=s2.outputs.split_manifest_out,
        execution_manifest_in=s06.outputs.execution_manifest_out,
        baseline_champion=s5z.outputs.champion_model,
        phaseb_champion=s06.outputs.champion_model,
        phasec_champion=s09.outputs.optimized_champion_model,
    )
    
    # Model Registration (s12) - register champion to MLflow Model Registry
    s12 = model_reg(
        config_name=config_name,
        champion_manifest=s10.outputs.final_report,
        champion_model=s10.outputs.final_champion_model,
        execution_manifest=s06.outputs.execution_manifest_out,
    )
    # Model registration uses the submitting user's delegated identity.
    s12.identity = UserIdentityConfiguration()
    
    # s13 — Drift monitoring & cadence assessment (additive layer)
    # s13_kwargs: optional wiring for s13 (model registration); empty dict if upstream did not produce expected outputs
    s13_kwargs = dict(
        config_name=config_name,
        # Drift reference evidence must exclude the locked final holdout.
        dataset_in=s4.outputs.train_out,
        final_report=s10.outputs.final_report,
        registry_info=s12.outputs.registry_info,
    )
    if drift_baseline_in is not None:
        s13_kwargs["baseline_in"] = drift_baseline_in
        s13_kwargs["baseline_uri"] = drift_baseline_uri
    s13 = drift_monitor(**s13_kwargs)

    # s14 — Auto-retrain decision gate (artifact-only; no nested submissions)
    s14 = retrain_decision(
        config_name=config_name,
        drift_report=s13.outputs.drift_report,
        candidate_baseline=s13.outputs.drift_baseline,
        final_report=s10.outputs.final_report,
        registry_info=s12.outputs.registry_info,
    )
    s14.outputs.retrain_decision = Output(type="uri_file")
    s14.outputs.decision_ledger_record = Output(type="uri_file")
    
    return {
        "eda_report": s1.outputs.eda_report,
        "prep_report": s2.outputs.prep_report,
        "prep3_report": s3.outputs.prep3_report,
        "fe_report": s4.outputs.fe_report,
        "dataset_processed": s4.outputs.dataset_out,
        "dataset_train": s4.outputs.train_out,
        "dataset_holdout": s2.outputs.raw_holdout_out,
        "baseline_pycaret_metrics": s5a.outputs.metrics_json,
        "baseline_flaml_metrics": s5b.outputs.metrics_json,
        "baseline_aggregate_report": s5z.outputs.aggregate_report,
        "baseline_champion_model": s5z.outputs.champion_model,
        "phaseb_leaderboard": s06.outputs.leaderboard_csv,
        "phaseb_all_results": s06.outputs.all_results_json,
        "phaseb_champion_manifest": s06.outputs.champion_manifest,
        "phaseb_champion_model": s06.outputs.champion_model,
        "execution_manifest": s06.outputs.execution_manifest_out,
        "split_manifest": s06.outputs.split_manifest_out,
        "quality_decision": s06.outputs.quality_decision_out,
        "phasec_aggregate_report": s09.outputs.aggregate_report,
        "phasec_champion_model": s09.outputs.optimized_champion_model,
        "final_report": s10.outputs.final_report,
        "final_champion_model": s10.outputs.final_champion_model,
        "registry_info": s12.outputs.registry_info,
        "drift_report": s13.outputs.drift_report,
        "drift_baseline": s13.outputs.drift_baseline,
        "retrain_decision": s14.outputs.retrain_decision,
        "decision_ledger_record": s14.outputs.decision_ledger_record,
    }


@dsl.pipeline(compute=None)
def full_pipeline_v2(
    config_name: str,
    dataset_folder: Input(type="uri_folder"),
    execution_manifest: Input(type="uri_file"),
    candidate_catalog: Input(type="uri_file"),
    variants_list: str = "",
    engine_list: str = "pycaret,flaml",
    time_budget_per_variant: int = 300,
    phaseb_time_budget_sec: int = 10800,
    # V3-Proposed Planner parameters
    planner_enabled: bool = False,
    round1_max_variants: int = 40,
    round2_max_variants: int = 8,
    # Validated upstream by K2 (config_schema.py); pipeline_builder receives sanitized values
    proxy_prune_threshold: float = 0.50,
    cache_enabled: bool = True,
    drift_baseline_in: Input(type="uri_folder", optional=True) = None,
    drift_baseline_uri: str = "",
):
    """V3 pipeline with intelligent variant recommendation (Phase 1).
    
    NEW ARCHITECTURE:
    - Dataset profiling → Variant scoring → Batch execution in single Phase B step
    - Replaces hardcoded 2-variant architecture with data-driven N-variant selection
    - Signal artifacts (Round0/Round1/Elimination) enabled by default for all runs
    
    V3-PROPOSED PLANNER MODE (when planner_enabled=True):
    - Adaptive search: EDA-driven scoring + diverse sampling
    - Progressive refinement: Round 0 → Round 1 proxy → Round 2 full
    - Preprocessing cache: Avoids redundant transformations
    
    Args:
        config_name: Config YAML filename (from uploaded code/configs directory)
        dataset_folder: Datastore folder URI containing dataset
        candidate_catalog: Immutable complete recipe/candidate catalog artifact
        variants_list: Bounded compatibility input for older direct callers
        engine_list: Comma-separated engines (e.g., "pycaret,flaml")
        time_budget_per_variant: Max time per variant training (seconds)
        planner_enabled: Enable V3-Proposed adaptive planner mode
        round1_max_variants: Max variants for Round 1 proxy training
        round2_max_variants: Max variants for Round 2 full training
        proxy_prune_threshold: Proxy metric threshold for pruning
        cache_enabled: Enable preprocessing cache
    """
    s1 = ingestion(config_name=config_name, dataset_in=dataset_folder)
    s2 = preparation(config_name=config_name, dataset_in=s1.outputs.dataset_out,
                     eda_report=s1.outputs.eda_report)
    s2.outputs.train_out = Output(type="uri_file")
    s2.outputs.raw_train_out = Output(type="uri_file")
    s2.outputs.raw_holdout_out = Output(type="uri_file")
    s2.outputs.split_manifest_out = Output(type="uri_file")
    
    # Use first recipe from variants for stage3/stage4 preprocessing baseline
    # NOTE: Stage 3/4 still use single recipe; variant-specific preprocessing happens in Phase B
    s3 = preprocessing(config_name=config_name, dataset_in=s2.outputs.dataset_out, prep_report=s2.outputs.prep_report, recipe_name="recipe_baseline.yml")
    s4 = feature_eng(config_name=config_name, dataset_in=s3.outputs.dataset_out, recipe_name="recipe_baseline.yml")
    
    # Baseline training
    s5a = pycaret_train(
        config_name=config_name,
        dataset_in=s2.outputs.raw_train_out,
        split_manifest=s2.outputs.split_manifest_out,
    )
    s5a.outputs.metrics_json = Output(type="uri_file")
    s5a.outputs.manifest_json = Output(type="uri_file")
    s5a.outputs.best_model = Output(type="uri_folder")
    
    s5b = flaml_train(
        config_name=config_name,
        dataset_in=s2.outputs.raw_train_out,
        split_manifest=s2.outputs.split_manifest_out,
    )
    s5b.outputs.metrics_json = Output(type="uri_file")
    s5b.outputs.manifest_json = Output(type="uri_file")
    s5b.outputs.best_model = Output(type="uri_folder")
    
    s5z = agg_baseline(
        config_name=config_name,
        pycaret_manifest=s5a.outputs.manifest_json,
        pycaret_model=s5a.outputs.best_model,
        flaml_manifest=s5b.outputs.manifest_json,
        flaml_model=s5b.outputs.best_model,
    )
    
    # Phase B - NEW: Single-step batch variant runner
    # Processes N variants (e.g., 20) with nested MLflow runs
    # Signal artifacts (Round0/Round1/Elimination) enabled by default
    # V3-Proposed: Planner mode enables adaptive search + preprocessing cache
    # C2 FIX: Wire to s2 (prepared data) so variant runner applies its OWN preprocessing recipes.
    # Using s4 (already preprocessed) would double-preprocess and defeat variant-specific recipes.
    s06_kwargs = dict(
        config_name=config_name,
        engine_list=engine_list,
        dataset_in=s2.outputs.raw_train_out,
        split_manifest=s2.outputs.split_manifest_out,
        time_budget_per_variant=time_budget_per_variant,
        phaseb_time_budget_sec=phaseb_time_budget_sec,
        # V3-Proposed Planner parameters
        planner_enabled=planner_enabled,
        round1_max_variants=round1_max_variants,
        round2_max_variants=round2_max_variants,
        proxy_prune_threshold=proxy_prune_threshold,
        cache_enabled=cache_enabled
    )
    s06_kwargs["execution_manifest"] = execution_manifest
    s06_kwargs["candidate_catalog"] = candidate_catalog
    s06 = variant_runner(**s06_kwargs)
    # Force output type declaration
    s06.outputs.leaderboard_csv = Output(type="uri_file")
    s06.outputs.all_results_json = Output(type="uri_file")
    s06.outputs.champion_manifest = Output(type="uri_file")
    s06.outputs.champion_model = Output(type="uri_folder")
    s06.outputs.execution_manifest_out = Output(type="uri_file")
    s06.outputs.split_manifest_out = Output(type="uri_file")
    s06.outputs.quality_decision_out = Output(type="uri_file")
    
    # Phase C - Optuna HPO (s08) and aggregate (s09)
    # 🔥 FIX (A1): Wire Phase B champion manifest so Phase C tunes the correct algorithm
    # Phase C fits the complete champion recipe on the raw/prepared train-only data.
    s08 = phasec_hpo(
        config_name=config_name,
        dataset_in=s2.outputs.raw_train_out,
        execution_manifest=s06.outputs.execution_manifest_out,
        phaseb_manifest=s06.outputs.champion_manifest,
    )
    s08.outputs.hpo_metrics_json = Output(type="uri_file")
    s08.outputs.optimized_model = Output(type="uri_folder")
    
    s09 = agg_phasec(
        config_name=config_name,
        hpo_metrics_json=s08.outputs.hpo_metrics_json,
        optimized_model=s08.outputs.optimized_model,
    )
    
    # Final evaluation (s10) - select champion among baseline, phase B, phase C
    s10 = final_eval(
        config_name=config_name,
        dataset_in=s2.outputs.raw_train_out,
        holdout_in=s2.outputs.raw_holdout_out,
        split_manifest_in=s2.outputs.split_manifest_out,
        execution_manifest_in=s06.outputs.execution_manifest_out,
        baseline_champion=s5z.outputs.champion_model,
        phaseb_champion=s06.outputs.champion_model,  # From variant runner
        phasec_champion=s09.outputs.optimized_champion_model,
    )
    
    # Model Registration (s12) - register champion to MLflow Model Registry
    s12 = model_reg(
        config_name=config_name,
        champion_manifest=s10.outputs.final_report,
        champion_model=s10.outputs.final_champion_model,
        execution_manifest=s06.outputs.execution_manifest_out,
    )
    # Model registration uses the submitting user's delegated identity.
    s12.identity = UserIdentityConfiguration()
    
    # s13 — Drift monitoring & cadence assessment (additive layer)
    # s13_kwargs: optional wiring for s13 (model registration); empty dict if upstream did not produce expected outputs
    s13_kwargs = dict(
        config_name=config_name,
        # Drift reference evidence must exclude the locked final holdout.
        dataset_in=s4.outputs.train_out,
        final_report=s10.outputs.final_report,
        registry_info=s12.outputs.registry_info,
    )
    if drift_baseline_in is not None:
        s13_kwargs["baseline_in"] = drift_baseline_in
        s13_kwargs["baseline_uri"] = drift_baseline_uri
    s13 = drift_monitor(**s13_kwargs)

    # s14 — Auto-retrain decision gate (artifact-only; no nested submissions)
    s14 = retrain_decision(
        config_name=config_name,
        drift_report=s13.outputs.drift_report,
        candidate_baseline=s13.outputs.drift_baseline,
        final_report=s10.outputs.final_report,
        registry_info=s12.outputs.registry_info,
    )
    s14.outputs.retrain_decision = Output(type="uri_file")
    s14.outputs.decision_ledger_record = Output(type="uri_file")
    
    return {
        "eda_report": s1.outputs.eda_report,
        "prep_report": s2.outputs.prep_report,
        "prep3_report": s3.outputs.prep3_report,
        "fe_report": s4.outputs.fe_report,
        "dataset_processed": s4.outputs.dataset_out,
        "dataset_train": s4.outputs.train_out,
        "dataset_holdout": s2.outputs.raw_holdout_out,
        "baseline_pycaret_metrics": s5a.outputs.metrics_json,
        "baseline_flaml_metrics": s5b.outputs.metrics_json,
        "baseline_aggregate_report": s5z.outputs.aggregate_report,
        "baseline_champion_model": s5z.outputs.champion_model,
        "phaseb_leaderboard": s06.outputs.leaderboard_csv,
        "phaseb_all_results": s06.outputs.all_results_json,
        "phaseb_champion_manifest": s06.outputs.champion_manifest,
        "phaseb_champion_model": s06.outputs.champion_model,
        "execution_manifest": s06.outputs.execution_manifest_out,
        "split_manifest": s06.outputs.split_manifest_out,
        "quality_decision": s06.outputs.quality_decision_out,
        "phasec_aggregate_report": s09.outputs.aggregate_report,
        "phasec_champion_model": s09.outputs.optimized_champion_model,
        "final_report": s10.outputs.final_report,
        "final_champion_model": s10.outputs.final_champion_model,
        "registry_info": s12.outputs.registry_info,
        "drift_report": s13.outputs.drift_report,
        "drift_baseline": s13.outputs.drift_baseline,
        "retrain_decision": s14.outputs.retrain_decision,
        "decision_ledger_record": s14.outputs.decision_ledger_record,
    }
