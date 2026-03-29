from azure.ai.ml import dsl, Input, Output
from azure.ai.ml.entities import PipelineJob
from azure.ai.ml import load_component
from pathlib import Path

# Resolve component paths relative to repo root
ROOT = Path(__file__).resolve().parents[1]

# Load components fresh from YAML files
# Note: Version changes ensure Azure ML reloads from disk
ingestion = load_component(source=str(ROOT / "components/stage1_ingestion.yml"))
preparation = load_component(source=str(ROOT / "components/stage2_preparation.yml"))
preprocessing = load_component(source=str(ROOT / "components/stage3_preprocessing.yml"))
feature_eng = load_component(source=str(ROOT / "components/stage4_feature_engineering.yml"))
pycaret_train = load_component(source=str(ROOT / "components/stage5_pycaret_train.yml"))
flaml_train = load_component(source=str(ROOT / "components/stage5_flaml_train.yml"))
agg_baseline = load_component(source=str(ROOT / "components/aggregate_baseline.yml"))
# Dead per-recipe components removed (P3-1): phaseb_pycaret, phaseb_flaml, agg_phaseb
# Replaced by s06_phaseb_variant_runner (single-step batch processor)
variant_runner = load_component(source=str(ROOT / "components/s06_phaseb_variant_runner.yml"))
phasec_hpo = load_component(source=str(ROOT / "components/phasec_optuna_hpo.yml"))
agg_phasec = load_component(source=str(ROOT / "components/aggregate_phasec.yml"))
final_eval = load_component(source=str(ROOT / "components/final_evaluation.yml"))
model_reg = load_component(source=str(ROOT / "components/s12_model_registration.yml"))
timeseries_train = load_component(source=str(ROOT / "components/stage5_timeseries_train.yml"))

@dsl.pipeline(compute=None)
def full_pipeline(
    config_name: str,
    dataset_folder: Input(type="uri_folder"),
    variants_list: str = "",
    engine_list: str = "pycaret,flaml",
    time_budget_per_variant: int = 300,
):
    """V3 production pipeline — runs ALL selected variants via the variant runner.
    
    ARCHITECTURE (Feb 2026):
    - Stages 1-4: Ingestion → Preparation → Preprocessing → Feature Engineering
    - Stage 5 (Baseline): PyCaret + FLAML + TimeSeries (parallel)
    - Stage 6 (Phase B): Single variant runner step that processes N variants
      with nested MLflow runs per variant×engine. No more 2-recipe limit.
    - Stage 8 (Phase C): Optuna HPO
    - Stage 10 (Final): Champion selection across all phases
    
    Args:
        config_name: Config YAML filename (from uploaded code/configs directory)
        dataset_folder: Datastore folder URI containing dataset
        variants_list: Comma-separated list of ALL recipe/variant paths to run
        engine_list: Comma-separated engines (e.g. "pycaret,flaml")
        time_budget_per_variant: Max seconds per variant training
    """
    s1 = ingestion(config_name=config_name, dataset_in=dataset_folder)
    s2 = preparation(config_name=config_name, dataset_in=s1.outputs.dataset_out,
                     eda_report=s1.outputs.eda_report)
    s3 = preprocessing(config_name=config_name, dataset_in=s2.outputs.dataset_out, prep_report=s2.outputs.prep_report, recipe_name="recipe_baseline.yml")
    s4 = feature_eng(config_name=config_name, dataset_in=s3.outputs.dataset_out, recipe_name="recipe_baseline.yml")
    
    # Baseline training - explicitly wire outputs to force Azure ML recognition
    s5a = pycaret_train(config_name=config_name, dataset_in=s4.outputs.dataset_out)
    s5a.outputs.metrics_json = Output(type="uri_file")
    s5a.outputs.manifest_json = Output(type="uri_file")
    s5a.outputs.best_model = Output(type="uri_folder")
    
    s5b = flaml_train(config_name=config_name, dataset_in=s4.outputs.dataset_out)
    s5b.outputs.metrics_json = Output(type="uri_file")
    s5b.outputs.manifest_json = Output(type="uri_file")
    s5b.outputs.best_model = Output(type="uri_folder")
    
    # Time-series forecasting (skips internally if not time-series)
    s5t = timeseries_train(config_name=config_name, dataset_in=s4.outputs.dataset_out)
    s5t.outputs.metrics_json = Output(type="uri_file")
    s5t.outputs.manifest_json = Output(type="uri_file")
    s5t.outputs.best_model = Output(type="uri_folder")
    
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
    s06 = variant_runner(
        config_name=config_name,
        variants_list=variants_list,
        engine_list=engine_list,
        dataset_in=s2.outputs.dataset_out,
        time_budget_per_variant=time_budget_per_variant,
    )
    s06.outputs.leaderboard_csv = Output(type="uri_file")
    s06.outputs.all_results_json = Output(type="uri_file")
    s06.outputs.champion_manifest = Output(type="uri_file")
    s06.outputs.champion_model = Output(type="uri_folder")
    
    # Phase C - Optuna HPO (s08) and aggregate (s09)
    # 🔥 FIX (A1): Wire Phase B champion manifest so Phase C tunes the correct algorithm
    # ARCHITECTURAL DECISION (2026-03-29): Phase C receives s4 (baseline-preprocessed)
    # data, NOT s2 or Phase B variant-preprocessed data. This is INTENTIONAL because:
    #   (a) s06 runs N variants × M engines — there is no single "Phase B dataset"
    #   (b) phasec_optuna_hpo.py replicates the champion recipe's encoding + scaling
    #       transforms onto s4 data (see phasec L390-440), bridging the gap
    #   (c) Blueprint Row 11 specifies "Engineered dataset from s04"
    #   (d) For tree-based champions, the preprocessing difference is negligible
    s08 = phasec_hpo(
        config_name=config_name,
        dataset_in=s4.outputs.dataset_out,
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
        dataset_in=s4.outputs.dataset_out,
        baseline_champion=s5z.outputs.champion_model,
        phaseb_champion=s06.outputs.champion_model,
        phasec_champion=s09.outputs.optimized_champion_model,
    )
    
    # Model Registration (s12) - register champion to MLflow Model Registry
    s12 = model_reg(
        config_name=config_name,
        champion_manifest=s10.outputs.final_report,
        champion_model=s10.outputs.final_champion_model,
    )
    
    return {
        "eda_report": s1.outputs.eda_report,
        "prep_report": s2.outputs.prep_report,
        "prep3_report": s3.outputs.prep3_report,
        "fe_report": s4.outputs.fe_report,
        "dataset_processed": s4.outputs.dataset_out,
        "baseline_pycaret_metrics": s5a.outputs.metrics_json,
        "baseline_flaml_metrics": s5b.outputs.metrics_json,
        "baseline_aggregate_report": s5z.outputs.aggregate_report,
        "baseline_champion_model": s5z.outputs.champion_model,
        "phaseb_leaderboard": s06.outputs.leaderboard_csv,
        "phaseb_all_results": s06.outputs.all_results_json,
        "phaseb_champion_manifest": s06.outputs.champion_manifest,
        "phaseb_champion_model": s06.outputs.champion_model,
        "phasec_aggregate_report": s09.outputs.aggregate_report,
        "phasec_champion_model": s09.outputs.optimized_champion_model,
        "final_report": s10.outputs.final_report,
        "final_champion_model": s10.outputs.final_champion_model,
        "registry_info": s12.outputs.registry_info,
    }


@dsl.pipeline(compute=None)
def full_pipeline_v2(
    config_name: str,
    dataset_folder: Input(type="uri_folder"),
    variants_list: str,
    engine_list: str = "pycaret,flaml",
    time_budget_per_variant: int = 300,
    # V3-Proposed Planner parameters
    planner_enabled: bool = False,
    round1_max_variants: int = 40,
    round2_max_variants: int = 10,
    proxy_prune_threshold: float = 0.50,
    cache_enabled: bool = True
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
        variants_list: Comma-separated list of variant paths (pre-scored by submission script)
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
    
    # Use first recipe from variants for stage3/stage4 preprocessing baseline
    # NOTE: Stage 3/4 still use single recipe; variant-specific preprocessing happens in Phase B
    s3 = preprocessing(config_name=config_name, dataset_in=s2.outputs.dataset_out, prep_report=s2.outputs.prep_report, recipe_name="recipe_baseline.yml")
    s4 = feature_eng(config_name=config_name, dataset_in=s3.outputs.dataset_out, recipe_name="recipe_baseline.yml")
    
    # Baseline training
    s5a = pycaret_train(config_name=config_name, dataset_in=s4.outputs.dataset_out)
    s5a.outputs.metrics_json = Output(type="uri_file")
    s5a.outputs.manifest_json = Output(type="uri_file")
    s5a.outputs.best_model = Output(type="uri_folder")
    
    s5b = flaml_train(config_name=config_name, dataset_in=s4.outputs.dataset_out)
    s5b.outputs.metrics_json = Output(type="uri_file")
    s5b.outputs.manifest_json = Output(type="uri_file")
    s5b.outputs.best_model = Output(type="uri_folder")
    
    # Time-series forecasting (skips internally if not time-series)
    s5t = timeseries_train(config_name=config_name, dataset_in=s4.outputs.dataset_out)
    s5t.outputs.metrics_json = Output(type="uri_file")
    s5t.outputs.manifest_json = Output(type="uri_file")
    s5t.outputs.best_model = Output(type="uri_folder")
    
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
    s06 = variant_runner(
        config_name=config_name,
        variants_list=variants_list,  # Pass as string, not file
        engine_list=engine_list,
        dataset_in=s2.outputs.dataset_out,
        time_budget_per_variant=time_budget_per_variant,
        # V3-Proposed Planner parameters
        planner_enabled=planner_enabled,
        round1_max_variants=round1_max_variants,
        round2_max_variants=round2_max_variants,
        proxy_prune_threshold=proxy_prune_threshold,
        cache_enabled=cache_enabled
    )
    # Force output type declaration
    s06.outputs.leaderboard_csv = Output(type="uri_file")
    s06.outputs.all_results_json = Output(type="uri_file")
    s06.outputs.champion_manifest = Output(type="uri_file")
    s06.outputs.champion_model = Output(type="uri_folder")
    
    # Phase C - Optuna HPO (s08) and aggregate (s09)
    # 🔥 FIX (A1): Wire Phase B champion manifest so Phase C tunes the correct algorithm
    # ARCHITECTURAL DECISION (2026-03-29): Phase C receives s4 data — see full_pipeline comment
    s08 = phasec_hpo(
        config_name=config_name,
        dataset_in=s4.outputs.dataset_out,
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
        dataset_in=s4.outputs.dataset_out,
        baseline_champion=s5z.outputs.champion_model,
        phaseb_champion=s06.outputs.champion_model,  # From variant runner
        phasec_champion=s09.outputs.optimized_champion_model,
    )
    
    # Model Registration (s12) - register champion to MLflow Model Registry
    s12 = model_reg(
        config_name=config_name,
        champion_manifest=s10.outputs.final_report,
        champion_model=s10.outputs.final_champion_model,
    )
    
    return {
        "eda_report": s1.outputs.eda_report,
        "prep_report": s2.outputs.prep_report,
        "prep3_report": s3.outputs.prep3_report,
        "fe_report": s4.outputs.fe_report,
        "dataset_processed": s4.outputs.dataset_out,
        "baseline_pycaret_metrics": s5a.outputs.metrics_json,
        "baseline_flaml_metrics": s5b.outputs.metrics_json,
        "baseline_aggregate_report": s5z.outputs.aggregate_report,
        "baseline_champion_model": s5z.outputs.champion_model,
        "phaseb_leaderboard": s06.outputs.leaderboard_csv,
        "phaseb_all_results": s06.outputs.all_results_json,
        "phaseb_champion_manifest": s06.outputs.champion_manifest,
        "phaseb_champion_model": s06.outputs.champion_model,
        "phasec_aggregate_report": s09.outputs.aggregate_report,
        "phasec_champion_model": s09.outputs.optimized_champion_model,
        "final_report": s10.outputs.final_report,
        "final_champion_model": s10.outputs.final_champion_model,
        "registry_info": s12.outputs.registry_info,
    }
