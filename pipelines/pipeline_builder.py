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
phaseb_pycaret = load_component(source=str(ROOT / "components/phaseb_pycaret_recipe.yml"))
phaseb_flaml = load_component(source=str(ROOT / "components/phaseb_flaml_recipe.yml"))
agg_phaseb = load_component(source=str(ROOT / "components/aggregate_phaseb.yml"))
phasec_hpo = load_component(source=str(ROOT / "components/phasec_optuna_hpo.yml"))
agg_phasec = load_component(source=str(ROOT / "components/aggregate_phasec.yml"))
final_eval = load_component(source=str(ROOT / "components/final_evaluation.yml"))
drift_monitor = load_component(source=str(ROOT / "components/s13_drift_monitor.yml"))

@dsl.pipeline(compute=None)
def full_pipeline(config_name: str, dataset_folder: Input(type="uri_folder"),
                  drift_baseline_in: Input(type="uri_folder", optional=True) = None):
    """V3 pipeline with read-only datastore access.
    
    Args:
        config_name: Config YAML filename (from uploaded code/configs directory)
        dataset_folder: Datastore folder URI containing dataset
        drift_baseline_in: Previous run's drift baseline folder for comparison (optional)
    """
    s1 = ingestion(config_name=config_name, dataset_in=dataset_folder)
    s2 = preparation(config_name=config_name, dataset_in=s1.outputs.dataset_out)
    s3 = preprocessing(config_name=config_name, dataset_in=s2.outputs.dataset_out)
    s4 = feature_eng(config_name=config_name, dataset_in=s3.outputs.dataset_out)
    
    # Baseline training - explicitly wire outputs to force Azure ML recognition
    s5a = pycaret_train(config_name=config_name, dataset_in=s4.outputs.dataset_out)
    # Force output type declaration
    s5a.outputs.metrics_json = Output(type="uri_file")
    s5a.outputs.manifest_json = Output(type="uri_file")
    s5a.outputs.best_model = Output(type="uri_folder")
    
    s5b = flaml_train(config_name=config_name, dataset_in=s4.outputs.dataset_out)
    # Force output type declaration
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
    
    # Phase B - recipes from uploaded code directory (read-only, local paths)
    r1_name = "recipe_smote_target_standard.yml"
    r2_name = "recipe_knn_onehot_minmax.yml"
    
    s6a = phaseb_pycaret(config_name=config_name, dataset_in=s4.outputs.dataset_out, recipe_name=r1_name)
    # Force output type declaration
    s6a.outputs.metrics_json = Output(type="uri_file")
    s6a.outputs.manifest_json = Output(type="uri_file")
    s6a.outputs.best_model = Output(type="uri_folder")
    
    s6b = phaseb_flaml(config_name=config_name, dataset_in=s4.outputs.dataset_out, recipe_name=r1_name)
    # Force output type declaration
    s6b.outputs.metrics_json = Output(type="uri_file")
    s6b.outputs.manifest_json = Output(type="uri_file")
    s6b.outputs.best_model = Output(type="uri_folder")
    
    s7a = phaseb_pycaret(config_name=config_name, dataset_in=s4.outputs.dataset_out, recipe_name=r2_name)
    # Force output type declaration
    s7a.outputs.metrics_json = Output(type="uri_file")
    s7a.outputs.manifest_json = Output(type="uri_file")
    s7a.outputs.best_model = Output(type="uri_folder")
    
    s7b = phaseb_flaml(config_name=config_name, dataset_in=s4.outputs.dataset_out, recipe_name=r2_name)
    # Force output type declaration
    s7b.outputs.metrics_json = Output(type="uri_file")
    s7b.outputs.manifest_json = Output(type="uri_file")
    s7b.outputs.best_model = Output(type="uri_folder")
    
    s08z = agg_phaseb(
        config_name=config_name,
        r1_pycaret_manifest=s6a.outputs.manifest_json,
        r1_pycaret_model=s6a.outputs.best_model,
        r1_flaml_manifest=s6b.outputs.manifest_json,
        r1_flaml_model=s6b.outputs.best_model,
        r2_pycaret_manifest=s7a.outputs.manifest_json,
        r2_pycaret_model=s7a.outputs.best_model,
        r2_flaml_manifest=s7b.outputs.manifest_json,
        r2_flaml_model=s7b.outputs.best_model,
    )
    
    # Phase C - Optuna HPO and aggregate
    s10 = phasec_hpo(config_name=config_name, dataset_in=s4.outputs.dataset_out)
    # Force output type declaration
    s10.outputs.hpo_metrics_json = Output(type="uri_file")
    s10.outputs.optimized_model = Output(type="uri_folder")
    
    s10z = agg_phasec(
        config_name=config_name,
        hpo_metrics_json=s10.outputs.hpo_metrics_json,
        optimized_model=s10.outputs.optimized_model,
    )
    
    # Final evaluation - select champion among baseline, phase B, phase C
    s11 = final_eval(
        config_name=config_name,
        dataset_in=s4.outputs.dataset_out,
        baseline_champion=s5z.outputs.champion_model,
        phaseb_champion=s08z.outputs.champion_model,
        phasec_champion=s10z.outputs.optimized_champion_model,
    )
    
    # s13 - Drift monitoring & cadence assessment
    s13_kwargs = dict(
        config_name=config_name,
        dataset_in=s4.outputs.dataset_out,
        final_report=s11.outputs.final_report,
    )
    if drift_baseline_in is not None:
        s13_kwargs["baseline_in"] = drift_baseline_in
    s13 = drift_monitor(**s13_kwargs)
    
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
        "phaseb_aggregate_report": s08z.outputs.aggregate_report,
        "phaseb_champion_model": s08z.outputs.champion_model,
        "phasec_aggregate_report": s10z.outputs.aggregate_report,
        "phasec_champion_model": s10z.outputs.optimized_champion_model,
        "final_report": s11.outputs.final_report,
        "final_champion_model": s11.outputs.final_champion_model,
        "drift_report": s13.outputs.drift_report,
        "drift_baseline": s13.outputs.drift_baseline,
    }
