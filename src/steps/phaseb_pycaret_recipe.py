import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
import mlflow

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.azureml_metrics_logger import create_metrics_logger
import joblib

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def get_primary_metric(task_type: str) -> str:
    """Return the primary metric column name for task type."""
    if task_type == "classification":
        return "Accuracy"
    elif task_type == "regression":
        return "R2"
    elif task_type == "clustering":
        return "Silhouette"
    else:
        return "Accuracy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--recipe_name", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--model_out", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(__file__).resolve().parents[2]
    recipe_path = root / "configs" / "recipes" / args.recipe_name
    if not recipe_path.is_file():
        raise FileNotFoundError(
            f"Recipe not found at {recipe_path}. Recipes must come from uploaded code (configs/recipes) and not workspaceblobstore."
        )
    with open(recipe_path, "r") as f:
        recipe = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")  # 🔥 CRITICAL FIX
    
    # Validate recipe compatibility with task_type
    recipe_task = recipe.get("task_type")
    if recipe_task and recipe_task != task_type:
        print(f"⚠️ WARNING: Recipe task_type '{recipe_task}' does not match config task_type '{task_type}'")
        print(f"⚠️ Proceeding anyway, but results may be suboptimal")
    
    # Check for classification-specific techniques in non-classification tasks
    if task_type != "classification":
        imbalance_method = recipe.get("stage3_preprocessing", {}).get("imbalance_handling", {}).get("method")
        if imbalance_method and imbalance_method.lower() in ["smote", "adasyn", "smoteenn", "smotetomek"]:
            print(f"⚠️ WARNING: Recipe contains SMOTE/resampling ('{imbalance_method}') which is classification-only")
            print(f"⚠️ SKIPPING imbalance handling for {task_type} task")
            # Override recipe to skip SMOTE
            recipe.setdefault("stage3_preprocessing", {}).setdefault("imbalance_handling", {})["method"] = "none"

    df = pd.read_csv(args.dataset_in, sep=delimiter)  # 🔥 FIXED
    
    # Validate target column (required for classification/regression, optional for clustering)
    if task_type != "clustering":
        if not target_col:
            raise ValueError(f"Target column required for {task_type} task but not specified in config")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing in dataset for PyCaret recipe training")

    # Disable autologging (do NOT change tracking URI — breaks MLflow hierarchy)
    try:
        mlflow.sklearn.autolog(disable=True)
    except Exception:
        pass
    # Set local model registry to avoid unsupported azureml:// registry errors
    os.makedirs("/tmp/mlflow-registry", exist_ok=True)
    mlflow.set_registry_uri("file:///tmp/mlflow-registry")

    metrics = {}
    manifest = {"engine": "pycaret", "recipe": recipe.get("recipe_name"), "models": []}

    # Import fast model list for bounded training
    try:
        from utils.model_universe import get_fast_model_list
        _FAST_AVAILABLE = True
    except ImportError:
        _FAST_AVAILABLE = False

    try:
        if task_type == "classification":
            from pycaret.classification import setup, compare_models, pull, save_model

            _fast = get_fast_model_list("classification", "pycaret") if _FAST_AVAILABLE else None
            print(f"\n🚀 Phase B PyCaret Classification (recipe: {args.recipe_name})")
            print(f"   Dataset : {df.shape[0]:,} rows × {df.shape[1]} cols")
            print(f"   Models  : {len(_fast) if _fast else 'ALL (no fast list)'} fast models")
            print(f"   CV folds: 3 | turbo: True | n_select: 5")

            setup(data=df, target=target_col, session_id=42, verbose=False,
                  log_experiment=False, fold=3)
            best = compare_models(
                include=_fast or None,
                n_select=5,
                turbo=True,
                sort="AUC",
            )
            leaderboard = pull()

        elif task_type == "regression":
            from pycaret.regression import setup, compare_models, pull, save_model

            _fast = get_fast_model_list("regression", "pycaret") if _FAST_AVAILABLE else None
            print(f"\n🚀 Phase B PyCaret Regression (recipe: {args.recipe_name})")
            print(f"   Dataset : {df.shape[0]:,} rows × {df.shape[1]} cols")
            print(f"   Models  : {len(_fast) if _fast else 'ALL (no fast list)'} fast models")
            print(f"   CV folds: 3 | turbo: True | n_select: 5")

            setup(data=df, target=target_col, session_id=42, verbose=False,
                  log_experiment=False, fold=3)
            best = compare_models(
                include=_fast or None,
                n_select=5,
                turbo=True,
                sort="R2",
            )
            leaderboard = pull()
        elif task_type == "clustering":
            from pycaret.clustering import setup, create_model, pull, save_model
            from sklearn.metrics import silhouette_score, davies_bouldin_score
            
            print(f"ℹ️ PyCaret Clustering (Phase B Recipe): training models with recipe '{args.recipe_name}'...")
            # Cast numeric columns to float64 to prevent dtype mismatch errors
            _numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[_numeric_cols] = df[_numeric_cols].astype(np.float64)
            print(f"   Cast {len(_numeric_cols)} numeric cols to float64")
            
            setup(data=df, session_id=42, verbose=False, log_experiment=False)
            best = create_model("kmeans")
            leaderboard = pull()
            
            # Compute clustering metrics
            predictions = best.predict(df)
            silhouette = silhouette_score(df.select_dtypes(include=[np.number]).astype(np.float64), predictions)
            davies_bouldin = davies_bouldin_score(df.select_dtypes(include=[np.number]).astype(np.float64), predictions)
            
            metrics["silhouette_score"] = silhouette
            metrics["davies_bouldin_score"] = davies_bouldin
            manifest["silhouette_score"] = silhouette
            manifest["davies_bouldin_score"] = davies_bouldin
            
            print(f"✅ Clustering (Phase B): silhouette_score={silhouette:.4f}, davies_bouldin={davies_bouldin:.4f}")
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        metrics["best_model"] = str(best)
        metrics["leaderboard"] = leaderboard.to_dict()
        manifest["best_model_name"] = str(best)
        manifest["leaderboard_columns"] = leaderboard.columns.tolist()
        manifest["rows"] = int(leaderboard.shape[0])
        # Save model to folder
        model_dir = Path(args.model_out).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model"
        save_model(best, str(model_path))
        
        # 🔍 CRITICAL: Validate model files were created
        print(f"\n🔍 PHASE B PYCARET MODEL SAVE VALIDATION:")
        print(f"  📂 Output directory: {model_dir}")
        print(f"  ✅ Directory exists: {model_dir.exists()}")
        print(f"  📄 Files in output directory:")
        file_count = 0
        total_size = 0
        for item in sorted(model_dir.rglob("*")):
            if item.is_file():
                size = item.stat().st_size
                rel_path = item.relative_to(model_dir)
                print(f"     📦 {rel_path} ({size:,} bytes)")
                file_count += 1
                total_size += size
        print(f"  📊 Total: {file_count} files, {total_size:,} bytes")
        if file_count == 0:
            print(f"  ❌ WARNING: No files found in output directory!")
        
        # 📊 CREATE OUTPUTS FOLDER FOR RECIPE-SPECIFIC TRACKING
        import shutil
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📊 RECIPE-SPECIFIC LOGGING TO outputs/ FOLDER:")
        
        recipe_name = recipe.get('recipe_name', 'unknown').replace('/', '_').replace(' ', '_')
        
        # 1. Export recipe-specific leaderboard
        leaderboard_path = outputs_dir / f"phaseb_{recipe_name}_pycaret_leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        print(f"  ✅ Recipe leaderboard: {leaderboard_path} ({len(leaderboard)} models)")
        
        # 2. Copy model to outputs
        if (model_dir / 'model.pkl').exists():
            shutil.copy2(model_dir / 'model.pkl', outputs_dir / f'phaseb_{recipe_name}_best_model.pkl')
            print(f"  ✅ Best model copied: phaseb_{recipe_name}_best_model.pkl")
        
        # 3. Recipe summary
        recipe_summary = {
            "recipe": recipe_name,
            "engine": "pycaret",
            "models_compared": int(leaderboard.shape[0]),
            "best_model": str(best),
            "best_metric_value": float(leaderboard.iloc[0][get_primary_metric(task_type)]) if task_type != "clustering" else None
        }
        summary_path = outputs_dir / f"phaseb_{recipe_name}_pycaret_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(recipe_summary, f, indent=2)
        print(f"  ✅ Recipe summary: {summary_path}")

        # ── Deep Model-Level Breakdown (shared utility) ─────────────────
        try:
            from utils.model_universe import build_pycaret_breakdown, write_model_breakdown
            _prim_metric = get_primary_metric(task_type)
            breakdown_df = build_pycaret_breakdown(
                leaderboard,
                step_name=f"phaseb_{recipe_name}",
                phase="phase_b",
                variant=recipe_name,
                task_type=task_type,
                metric_col=_prim_metric,
            )
            bd_path = write_model_breakdown(
                breakdown_df, str(outputs_dir),
                f"model_breakdown_phaseb_{recipe_name}_pycaret.csv",
            )
            print(f"  📊 Model breakdown: {len(breakdown_df)} models → {Path(bd_path).name}")
        except Exception as _bd_err:
            print(f"  ⚠️  Model breakdown write failed (non-fatal): {_bd_err}")
        
        # 🔥 ENTERPRISE-LEVEL RECIPE LOGGING
        print(f"\n📋 LOGGING RECIPE DETAILS TO MLFLOW:")
        try:
            # Recipe identification
            logger.log_param("recipe_name", recipe.get("recipe_name", args.recipe_name))
            logger.log_param("recipe_version", recipe.get("version", "1.0"))
            logger.log_param("recipe_description", recipe.get("description", "No description"))
            logger.log_param("recipe_file", args.recipe_name)
            
            # Stage 3 preprocessing techniques from recipe
            stage3 = recipe.get("stage3_preprocessing", {})
            if stage3:
                # Imputation
                imputation = stage3.get("imputation", {})
                logger.log_param("recipe_imputation_method", imputation.get("method", "none"))
                if imputation.get("n_neighbors"):
                    logger.log_param("recipe_imputation_n_neighbors", int(imputation["n_neighbors"]))
                
                # Imbalance handling
                imbalance = stage3.get("imbalance_handling", {})
                logger.log_param("recipe_imbalance_method", imbalance.get("method", "none"))
                if imbalance.get("sampling_strategy"):
                    logger.log_param("recipe_sampling_strategy", str(imbalance["sampling_strategy"]))
                
                # Encoding
                encoding = stage3.get("encoding", {})
                logger.log_param("recipe_encoding_method", encoding.get("categorical_method", "onehot"))
                logger.log_param("recipe_handle_unknown", encoding.get("handle_unknown", "error"))
                
                # Scaling
                scaling = stage3.get("scaling", {})
                logger.log_param("recipe_scaling_method", scaling.get("method", "standard"))
            
            # Stage 4 feature engineering from recipe
            stage4 = recipe.get("stage4_feature_engineering", {})
            if stage4:
                feature_sel = stage4.get("feature_selection", {})
                logger.log_param("recipe_feature_selection", feature_sel.get("method", "none"))
                if feature_sel.get("k_features"):
                    logger.log_param("recipe_k_features", int(feature_sel["k_features"]))
            
            # Log full recipe as artifact
            logger.log_dict(recipe, f"recipe_{recipe.get('recipe_name', 'unknown')}.json")
            
            print(f"   ✅ Recipe: {recipe.get('recipe_name')}")
            print(f"   ✅ Imputation: {stage3.get('imputation', {}).get('method', 'none')}")
            print(f"   ✅ Imbalance: {stage3.get('imbalance_handling', {}).get('method', 'none')}")
            print(f"   ✅ Encoding: {stage3.get('encoding', {}).get('categorical_method', 'onehot')}")
            print(f"   ✅ Scaling: {stage3.get('scaling', {}).get('method', 'standard')}")
            print(f"   ✅ Feature Selection: {stage4.get('feature_selection', {}).get('method', 'none')}")
            
        except Exception as recipe_log_err:
            print(f"⚠️  Recipe logging failed (non-fatal): {recipe_log_err}")
            
    except Exception as e:
        metrics["error"] = str(e)
        manifest["error"] = str(e)
        # Mark model dir with error sentinel
        model_dir = Path(args.model_out).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / ".error").write_text(str(e))

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f)
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f)

    # Start MLflow run
    # Create dual logger for MLflow and Azure ML
    logger = create_metrics_logger(
        run_name="s06a_phaseb_pycaret",
        tags={"pipeline": "v3_mlops", "phase": "phaseb", "step": "s06a"}
    )

    # Log metrics to MLflow for Azure ML Studio tracking (all wrapped to prevent step failure)
    try:
        logger.log_param("engine", "pycaret")
        logger.log_param("recipe", recipe.get("recipe_name", "unknown"))
        logger.log_param("task_type", task_type)
        logger.log_param("target_column", target_col)
        # Truncate best_model to 500 char limit for MLflow params
        best_model_str = str(metrics.get("best_model", "unknown"))[:500]
        logger.log_param("best_model", best_model_str)
        logger.log_metric("dataset_rows", int(df.shape[0]))
        logger.log_metric("dataset_cols", int(df.shape[1]))
        logger.log_dict(metrics, "pycaret_recipe_metrics.json")
        logger.log_dict(manifest, "pycaret_recipe_manifest.json")
        if "error" in metrics:
            logger.log_param("pycaret_recipe_error", metrics["error"])
    except Exception as mlflow_err:
        print(f"⚠️  MLflow logging failed (non-fatal): {mlflow_err}")

    # End logging
    logger.end_run()


if __name__ == "__main__":
    main()
