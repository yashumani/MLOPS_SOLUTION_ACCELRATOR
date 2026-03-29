import argparse
import json
import time as _time_mod
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, make_scorer
import mlflow

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.azureml_metrics_logger import create_metrics_logger
from utils.stage_signals import StageSignal, write_stage_signal
from utils.candidate_ledger import (
    make_row, normalize_metrics, write_stage_table,
)

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))


def _safe_disable_autolog():
    """Disable MLflow autologging and fix tracking URI for Azure ML compatibility."""
    try:
        mlflow.sklearn.autolog(disable=True)
    except Exception:
        pass
    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass
    # Fix: Convert azureml:// to https:// to avoid model registry errors
    import os as _os
    _mlflow_uri = _os.getenv("MLFLOW_TRACKING_URI", "")
    if _mlflow_uri.startswith("azureml://"):
        mlflow.set_tracking_uri(_mlflow_uri.replace("azureml://", "https://"))
    # Set local model registry as fallback
    _os.makedirs("/tmp/mlflow-registry", exist_ok=True)
    mlflow.set_registry_uri("file:///tmp/mlflow-registry")


def main():
    _t0 = _time_mod.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--study_out", required=True)
    parser.add_argument("--model_out", required=True)
    parser.add_argument("--phaseb_manifest", required=False, default=None,
                        help="Path to Phase B champion manifest JSON (optional)")
    args = parser.parse_args()

    print("=" * 80)
    print("STEP S08: PHASE C — OPTUNA HPO")
    print("=" * 80)

    # Disable autologging (do NOT change tracking URI — breaks MLflow hierarchy)
    _safe_disable_autolog()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")  # 🔥 CRITICAL FIX
    n_trials = cfg.get("phases", {}).get("phase_c_hpo", {}).get("n_trials", 50)
    random_seed = cfg.get("random_seed", 42)
    test_size = cfg.get("stages", {}).get("stage4_feature_engineering", {}).get("train_test_splits", [0.8])[0]
    test_size = 1 - float(test_size)

    # 🔍 DEBUG: Check dataset file before loading
    dataset_path = Path(args.dataset_in)
    print(f"\n🔍 PHASE C HPO - DATASET INSPECTION:")
    print(f"  📂 Dataset path: {dataset_path}")
    print(f"  ✅ File exists: {dataset_path.exists()}")
    if dataset_path.exists():
        file_size = dataset_path.stat().st_size
        print(f"  📊 File size: {file_size:,} bytes")
        if file_size == 0:
            raise ValueError(f"❌ Dataset file is EMPTY (0 bytes): {dataset_path}")
        
        # Peek at first few lines
        with open(dataset_path, 'r') as f:
            first_lines = [f.readline() for _ in range(3)]
        print(f"  📄 First 3 lines:")
        for i, line in enumerate(first_lines, 1):
            print(f"     {i}: {line[:100]}{'...' if len(line) > 100 else ''}")
    
    df = pd.read_csv(args.dataset_in, sep=delimiter)  # 🔥 FIXED
    
    print(f"\n📊 Dataset loaded:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''}")
    print(f"  Target column: '{target_col}'")
    print(f"  Target in columns: {target_col in df.columns}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    
    # 📊 CREATE OUTPUTS FOLDER FOR OPTUNA TRACKING
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"📊 Optuna HPO outputs will be saved to: {outputs_dir.resolve()}")
    
    import optuna
    
    # Handle clustering separately (no target column validation needed)
    if task_type == "clustering":
        print("ℹ️ Phase C HPO: Clustering task detected; using sklearn clustering + Optuna")
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score
        import gc
        
        # For clustering, use all data (no train/test split, no target)
        # Extract only numeric columns for sklearn clustering compatibility
        X_data = df.select_dtypes(include=['number']).copy()
        
        # Validate that numeric data exists for clustering
        if X_data.shape[1] == 0:
            raise ValueError("No numeric columns found in dataset for clustering. Clustering requires numeric features.")
        
        # OOM guard: silhouette_score computes O(n²) pairwise distances.
        # Cap the sample used for scoring to stay within memory on Standard_D4s_v3 (16 GB).
        _SILHOUETTE_SAMPLE_CAP = 10_000
        _sil_sample = min(_SILHOUETTE_SAMPLE_CAP, len(X_data))
        print(f"📊 Clustering HPO data: {len(X_data)} rows × {X_data.shape[1]} cols, "
              f"silhouette sample_size={_sil_sample}")

        # Down-cast to float32 to halve memory footprint
        X_data = X_data.astype(np.float32)
        gc.collect()

        # Ensure valid upper bound for KMeans n_clusters
        _max_k = max(2, min(10, len(X_data) // 5))

        def objective(trial: optuna.Trial):
            algo = trial.suggest_categorical("algorithm", ["kmeans", "dbscan"])
            if algo == "kmeans":
                n_clusters = trial.suggest_int("n_clusters", 2, _max_k)
                model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
                model.fit(X_data)
                unique_labels = set(model.labels_)
                if len(unique_labels) < 2:
                    return -1.0  # Degenerate: all samples in one cluster
                score = silhouette_score(
                    X_data, model.labels_,
                    sample_size=_sil_sample, random_state=random_seed
                )
                return score
            elif algo == "dbscan":
                eps = trial.suggest_float("eps", 0.1, 1.0)
                min_samples = trial.suggest_int("min_samples", 2, 10)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_data)
                # Filter out noise points (label == -1) before scoring
                mask = labels != -1
                n_clustered = mask.sum()
                unique_labels = set(labels[mask])
                if len(unique_labels) > 1 and n_clustered >= 2:
                    score = silhouette_score(
                        X_data[mask], labels[mask],
                        sample_size=min(_sil_sample, n_clustered),
                        random_state=random_seed,
                    )
                    # Penalise proportionally to noise fraction
                    noise_frac = 1.0 - (n_clustered / len(labels))
                    return score * (1.0 - noise_frac)
                else:
                    return -1.0  # All noise or single cluster
            gc.collect()
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, catch=(ValueError,))
        
        best_params = study.best_params
        best_value = study.best_value
        best_algo = best_params.get("algorithm", "unknown")
        
        # Train final model with best params
        if best_algo == "kmeans":
            final_model = KMeans(n_clusters=best_params.get("n_clusters", 3), random_state=random_seed, n_init=10)
        else:
            final_model = DBSCAN(eps=best_params.get("eps", 0.5), min_samples=best_params.get("min_samples", 5))
        final_model.fit(X_data)
        
        print(f"✅ Clustering HPO: Selected {best_algo} | silhouette_score={best_value:.4f} | params={best_params}")
        
        # Save model and study
        try:
            import joblib
            model_dir = Path(args.model_out).resolve()
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.pkl"
            joblib.dump(final_model, model_path)
            study_dir = Path(args.study_out).resolve()
            study_dir.mkdir(parents=True, exist_ok=True)
            study_path = study_dir / "study.pkl"
            joblib.dump(study, study_path)
            
            # 🔍 CRITICAL: Validate model files were created
            print(f"\n🔍 PHASE C HPO MODEL SAVE VALIDATION:")
            print(f"  📂 Model output directory: {model_dir}")
            print(f"  ✅ Model directory exists: {model_dir.exists()}")
            print(f"  📄 Model files:")
            model_file_count = 0
            model_total_size = 0
            for item in sorted(model_dir.rglob("*")):
                if item.is_file():
                    size = item.stat().st_size
                    rel_path = item.relative_to(model_dir)
                    print(f"     📦 {rel_path} ({size:,} bytes)")
                    model_file_count += 1
                    model_total_size += size
            print(f"  📊 Model total: {model_file_count} files, {model_total_size:,} bytes")
            print(f"  📂 Study output directory: {study_dir}")
            print(f"  ✅ Study directory exists: {study_dir.exists()}")
            print(f"  📄 Study files:")
            study_file_count = 0
            for item in sorted(study_dir.rglob("*")):
                if item.is_file():
                    size = item.stat().st_size
                    rel_path = item.relative_to(study_dir)
                    print(f"     📦 {rel_path} ({size:,} bytes)")
                    study_file_count += 1
            print(f"  📊 Study total: {study_file_count} files")
            if model_file_count == 0:
                print(f"  ❌ WARNING: No model files found in output directory!")
            if study_file_count == 0:
                print(f"  ❌ WARNING: No study files found in output directory!")
        except Exception as e:
            print(f"❌ Model/study save failed: {e}")
            model_dir = Path(args.model_out).resolve()
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / ".error").write_text(str(e))
            study_dir = Path(args.study_out).resolve()
            study_dir.mkdir(parents=True, exist_ok=True)
            (study_dir / ".error").write_text(str(e))
        
        metrics = {"algorithm": best_algo, "best_params": best_params, "best_score": best_value}
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f)
        
        # Create logger for clustering MLflow logging (before early return)
        logger = create_metrics_logger(
            run_name="s08_phasec_hpo",
            tags={"pipeline": "v3_mlops", "phase": "phasec", "step": "s08"}
        )
        
        # Log metrics to MLflow
        try:
            logger.log_param("optimizer", "optuna")
            logger.log_param("task_type", task_type)
            logger.log_param("algorithm", best_algo)
            logger.log_param("best_params", str(best_params)[:500])
            logger.log_param("n_trials", n_trials)
            logger.log_metric("best_score", float(best_value))
            logger.log_metric("dataset_rows", int(df.shape[0]))
            logger.log_metric("dataset_cols", int(df.shape[1]))
            logger.log_dict(metrics, "optuna_clustering_hpo_metrics.json")
        except Exception as mlflow_err:
            print(f"⚠️  MLflow logging failed (non-fatal): {mlflow_err}")
        
        logger.end_run()
        return  # Exit early for clustering
    
    # Classification/Regression: validate target column and prepare train/test split
    if not target_col:
        raise ValueError(f"Target column required for {task_type} task but not specified in config")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing in dataset for HPO")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 🔥 VALIDATION: Check for empty feature set
    if X.shape[1] == 0:
        raise ValueError(
            f"❌ Dataset has NO features after dropping target column '{target_col}'. "
            f"Original columns: {list(df.columns)}. Cannot perform HPO with zero features."
        )
    
    print(f"✅ Feature set: {X.shape[1]} columns, {X.shape[0]} rows")
    print(f"   Feature names: {list(X.columns)[:10]}{'...' if len(X.columns) > 10 else ''}")
    
    # 🔥 FIX: Encode target labels for XGBoost classification
    label_encoder = None
    if task_type == "classification":
        from sklearn.preprocessing import LabelEncoder
        
        # Check if target is string/object type
        if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            print(f"✅ Encoded target labels: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
            
            # Split with encoded labels
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_seed, stratify=y_encoded
            )
        else:
            # Already numeric
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed, stratify=y
            )
    else:
        # Regression - no encoding needed
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed
        )
    
    # Classification/Regression: use XGBoost + Optuna
    # 🔥 FIX (A1): Load Phase B champion metadata from wired pipeline input
    champion_algorithm = "xgboost"  # Default fallback
    champion_engine = "pycaret"

    # Determine manifest path: prefer explicit pipeline input, then legacy path
    manifest_path = None
    if getattr(args, "phaseb_manifest", None) and Path(args.phaseb_manifest).exists():
        manifest_path = Path(args.phaseb_manifest)
        print(f"✅ Phase B manifest received via pipeline input: {manifest_path}")
    else:
        legacy_path = Path("outputs") / "phaseb_champion_manifest.json"
        if legacy_path.exists():
            manifest_path = legacy_path
            print(f"⚠️  Phase B manifest found at legacy path: {legacy_path}")
        else:
            print(f"⚠️  No Phase B manifest found — defaulting to {champion_algorithm}")

    if manifest_path is not None:
        with open(manifest_path, 'r') as f:
            champion_metadata = json.load(f)
        champion_algorithm_raw = champion_metadata.get("algorithm", "xgboost").lower()
        champion_engine = champion_metadata.get("engine", "pycaret")
        
        print(f"\n🏆 PHASE C: Tuning Phase B Champion:")
        print(f"  Algorithm: {champion_metadata.get('algorithm')}")
        print(f"  Recipe: {champion_metadata.get('variant_path', champion_metadata.get('recipe', 'unknown'))}")
        print(f"  Engine: {champion_engine}")
        print(f"  Score: {champion_metadata.get('primary_metric_value', champion_metadata.get('champion_score'))}")
        
        # 🔥 FIX (A2): Log Phase B preprocessing context for transparency
        # Phase C receives Stage 4 (baseline-preprocessed) data, NOT Phase B's
        # variant-preprocessed data. Phase B applied recipe-specific transforms
        # (encoding, scaling, imputation) internally. For tree-based models this
        # difference is negligible; for linear models it matters more.
        phaseb_recipe = champion_metadata.get("variant_path", champion_metadata.get("recipe", "unknown"))
        phaseb_variant = champion_metadata.get("variant_id", champion_metadata.get("variant", "unknown"))
        preproc_cfg = champion_metadata.get("preprocessing_config", {})
        recipe_encoding = preproc_cfg.get("encoding", "none")
        recipe_scaling = preproc_cfg.get("scaling", "none")
        print(f"\n📋 PHASE C DATA CONTEXT (A2 transparency):")
        print(f"  ℹ️  Phase B champion was trained on variant: {phaseb_variant}")
        print(f"  ℹ️  Phase B recipe: {phaseb_recipe}")
        print(f"  ℹ️  Recipe encoding: {recipe_encoding}, scaling: {recipe_scaling}")
        print(f"  ✅  Phase C will replicate recipe transforms on Stage 4 data")
        
        # Map PyCaret/FLAML model names to sklearn/xgb/lgb/cat equivalents
        # Common mappings from Phase B results
        if "xgb" in champion_algorithm_raw or "xgboost" in champion_algorithm_raw or "Extreme Gradient Boosting" in champion_algorithm_raw:
            champion_algorithm = "xgboost"
        elif "lgb" in champion_algorithm_raw or "lightgbm" in champion_algorithm_raw or "Light Gradient Boosting" in champion_algorithm_raw:
            champion_algorithm = "lightgbm"
        elif "cat" in champion_algorithm_raw or "catboost" in champion_algorithm_raw or "CatBoost" in champion_algorithm_raw:
            champion_algorithm = "catboost"
        elif "rf" in champion_algorithm_raw or "randomforest" in champion_algorithm_raw or "Random Forest" in champion_algorithm_raw:
            champion_algorithm = "randomforest"
        elif "logistic" in champion_algorithm_raw or "lr" == champion_algorithm_raw:
            champion_algorithm = "logisticregression"
        elif "ridge" in champion_algorithm_raw:
            champion_algorithm = "ridge"
        else:
            print(f"  ⚠️  Unknown algorithm '{champion_algorithm_raw}', defaulting to XGBoost")
            champion_algorithm = "xgboost"
        
        # 🔥 FIX (Item 10): Recipe-aware preprocessing for Phase C
        # Replicate Phase B's recipe transforms (encoding + scaling) so the HPO
        # model trains on data consistent with what won Phase B.
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols and recipe_encoding not in ("none", "unknown"):
            if recipe_encoding == "label":
                for col in cat_cols:
                    X_train[col] = X_train[col].astype("category").cat.codes
                    X_test[col] = X_test[col].astype("category").cat.codes
                print(f"  ✅ Label-encoded {len(cat_cols)} categorical columns")
            elif recipe_encoding == "onehot":
                X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
                X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
                X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
                print(f"  ✅ One-hot encoded {len(cat_cols)} columns → {len(X_train.columns)} features")
            elif recipe_encoding == "target":
                try:
                    from category_encoders import TargetEncoder
                    te = TargetEncoder(cols=cat_cols)
                    X_train = te.fit_transform(X_train, y_train)
                    X_test = te.transform(X_test)
                    print(f"  ✅ Target-encoded {len(cat_cols)} categorical columns")
                except ImportError:
                    for col in cat_cols:
                        X_train[col] = X_train[col].astype("category").cat.codes
                        X_test[col] = X_test[col].astype("category").cat.codes
                    print(f"  ⚠️  category_encoders unavailable, fell back to label encoding")

        numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols and recipe_scaling not in ("none", "unknown"):
            if recipe_scaling == "standard":
                from sklearn.preprocessing import StandardScaler
                _scaler = StandardScaler()
            elif recipe_scaling == "robust":
                from sklearn.preprocessing import RobustScaler
                _scaler = RobustScaler()
            elif recipe_scaling == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                _scaler = MinMaxScaler()
            elif recipe_scaling == "yeo_johnson":
                from sklearn.preprocessing import PowerTransformer
                _scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            elif recipe_scaling == "quantile":
                from sklearn.preprocessing import QuantileTransformer
                _scaler = QuantileTransformer(output_distribution='normal', random_state=random_seed)
            else:
                _scaler = None
                print(f"  ⚠️  Unknown scaling '{recipe_scaling}', skipping")
            if _scaler is not None:
                X_train[numeric_cols] = _scaler.fit_transform(X_train[numeric_cols])
                X_test[numeric_cols] = _scaler.transform(X_test[numeric_cols])
                print(f"  ✅ Applied {recipe_scaling} scaling to {len(numeric_cols)} columns")
    else:
        print(f"\n⚠️  Phase B champion manifest not found")
        print(f"  Defaulting to XGBoost for HPO")
    
    print(f"\n🎯 Phase C HPO Target: {champion_algorithm}")
    
    # 🔥 NEW: Configure cross-validation for robust hyperparameter evaluation
    use_cv = cfg.get("phases", {}).get("phase_c_hpo", {}).get("use_cross_validation", True)
    cv_folds = cfg.get("phases", {}).get("phase_c_hpo", {}).get("cv_folds", 5)
    
    if use_cv:
        print(f"✅ Using {cv_folds}-fold cross-validation for hyperparameter evaluation")
    else:
        print(f"⚠️  Using single train/test split (no cross-validation)")
    
    # Import the champion algorithm
    try:
        if champion_algorithm == "xgboost":
            import xgboost as xgb
        elif champion_algorithm == "lightgbm":
            import lightgbm as lgb
        elif champion_algorithm == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor
        elif champion_algorithm == "randomforest":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        elif champion_algorithm == "logisticregression":
            from sklearn.linear_model import LogisticRegression, Ridge
        elif champion_algorithm == "ridge":
            from sklearn.linear_model import Ridge
        else:
            # Fallback to XGBoost
            import xgboost as xgb
            champion_algorithm = "xgboost"
    except ImportError as import_err:
        print(f"⚠️  Failed to import {champion_algorithm}: {import_err}")
        print(f"  Falling back to XGBoost")
        import xgboost as xgb
        champion_algorithm = "xgboost"

    def objective(trial: optuna.Trial):
        # Validate data before each trial
        if X_train.shape[1] == 0:
            raise ValueError(f"X_train has zero columns - cannot train model")
        
        # 🔥 NEW: Dynamic hyperparameter search space based on champion algorithm
        if champion_algorithm == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": random_seed,
                "n_jobs": -1,
            }
            if task_type == "classification":
                model = xgb.XGBClassifier(**params, objective="binary:logistic")
            else:
                model = xgb.XGBRegressor(**params, objective="reg:squarederror")
        
        elif champion_algorithm == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": random_seed,
                "n_jobs": -1,
                "verbose": -1
            }
            if task_type == "classification":
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)
        
        elif champion_algorithm == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 50, 300),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "random_seed": random_seed,
                "verbose": False
            }
            if task_type == "classification":
                model = CatBoostClassifier(**params)
            else:
                model = CatBoostRegressor(**params)
        
        elif champion_algorithm == "randomforest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                "random_state": random_seed,
                "n_jobs": -1
            }
            if task_type == "classification":
                model = RandomForestClassifier(**params)
            else:
                model = RandomForestRegressor(**params)
        
        elif champion_algorithm in ["logisticregression", "ridge"]:
            # For linear models, tune regularization only
            params = {
                "C": trial.suggest_float("C", 0.01, 100.0, log=True),
                "random_state": random_seed,
                "max_iter": 1000
            }
            if task_type == "classification":
                model = LogisticRegression(**params)
            else:
                model = Ridge(alpha=1.0/params["C"], random_state=random_seed)
        
        else:
            # Fallback to XGBoost params
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "random_state": random_seed
            }
            if task_type == "classification":
                model = xgb.XGBClassifier(**params)
            else:
                model = xgb.XGBRegressor(**params)
        
        # 🔥 NEW: Use cross-validation for more robust evaluation
        if use_cv and len(X_train) >= 50:  # Only use CV if enough data
            # Cross-validation on training set
            # Use balanced_accuracy for classification to handle class imbalance
            if task_type == "classification":
                scorer = make_scorer(balanced_accuracy_score)
            else:
                scorer = make_scorer(r2_score)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scorer, n_jobs=-1)
            score = cv_scores.mean()  # Use mean CV score
            
            # Log CV std dev for trial analysis
            trial.set_user_attr("cv_std", cv_scores.std())
            trial.set_user_attr("cv_scores", cv_scores.tolist())
        else:
            # Fallback to single train/test split
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            # Use balanced_accuracy for classification to handle class imbalance
            if task_type == "classification":
                score = balanced_accuracy_score(y_test, preds)
            else:
                score = r2_score(y_test, preds)
        
        return score

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,  # No pruning for first 10 trials
            n_warmup_steps=5,     # Start pruning after 5 CV folds if using CV
            interval_steps=1       # Check every step
        )
    )
    print(f"\n🔬 Starting Optuna study with MedianPruner (early stopping enabled)")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value
    
    # 📊 EXPORT ALL OPTUNA TRIALS TO CSV
    print(f"\n📊 EXPORTING OPTUNA TRIAL LOGS:")
    trials_df = study.trials_dataframe()
    trials_path = outputs_dir / "phasec_optuna_trials.csv"
    trials_df.to_csv(trials_path, index=False)
    print(f"  ✅ All trials: {trials_path} ({len(trials_df)} trials, {trials_path.stat().st_size:,} bytes)")
    
    # Save best hyperparameters
    best_params_path = outputs_dir / "phasec_optuna_best_params.json"
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"  ✅ Best params: {best_params_path} ({best_params_path.stat().st_size:,} bytes)")
    
    # Generate Optuna visualization plots
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        
        # Optimization history (convergence)
        fig1 = plot_optimization_history(study)
        history_path = outputs_dir / "phasec_optuna_optimization_history.html"
        fig1.write_html(str(history_path))
        print(f"  ✅ Optimization history: {history_path} ({history_path.stat().st_size:,} bytes)")
        
        # Parameter importance
        fig2 = plot_param_importances(study)
        importance_path = outputs_dir / "phasec_optuna_param_importance.html"
        fig2.write_html(str(importance_path))
        print(f"  ✅ Parameter importance: {importance_path} ({importance_path.stat().st_size:,} bytes)")
    except Exception as plot_err:
        print(f"  ⚠️  Could not generate Optuna plots: {plot_err}")

    # Train final model with best params
    print(f"\n🏆 Training final model with best hyperparameters:")
    print(f"  Algorithm: {champion_algorithm}")
    print(f"  Best params: {best_params}")
    print(f"  Best score: {best_value:.4f}")
    
    # 🔥 NEW: Train final model using champion algorithm (not hardcoded XGBoost)
    if champion_algorithm == "xgboost":
        import xgboost as xgb
        if task_type == "classification":
            final_model = xgb.XGBClassifier(**best_params, objective="binary:logistic")
        else:
            final_model = xgb.XGBRegressor(**best_params, objective="reg:squarederror")
    
    elif champion_algorithm == "lightgbm":
        import lightgbm as lgb
        if task_type == "classification":
            final_model = lgb.LGBMClassifier(**best_params)
        else:
            final_model = lgb.LGBMRegressor(**best_params)
    
    elif champion_algorithm == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        if task_type == "classification":
            final_model = CatBoostClassifier(**best_params)
        else:
            final_model = CatBoostRegressor(**best_params)
    
    elif champion_algorithm == "randomforest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if task_type == "classification":
            final_model = RandomForestClassifier(**best_params)
        else:
            final_model = RandomForestRegressor(**best_params)
    
    elif champion_algorithm in ["logisticregression", "ridge"]:
        from sklearn.linear_model import LogisticRegression, Ridge
        if task_type == "classification":
            final_model = LogisticRegression(**best_params)
        else:
            # Ridge uses alpha instead of C
            final_model = Ridge(alpha=1.0/best_params["C"], random_state=random_seed)
    
    else:
        # Fallback to XGBoost
        import xgboost as xgb
        if task_type == "classification":
            final_model = xgb.XGBClassifier(**best_params)
        else:
            final_model = xgb.XGBRegressor(**best_params)
    
    final_model.fit(X_train, y_train)

    # Save model, study, and label encoder
    try:
        import joblib
        model_dir = Path(args.model_out)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        joblib.dump(final_model, model_path)
        
        # Save label encoder for classification (needed for prediction decoding)
        if label_encoder is not None:
            encoder_path = model_dir / "label_encoder.pkl"
            joblib.dump(label_encoder, encoder_path)
            print(f"✅ Saved label encoder to {encoder_path}")
        
        study_dir = Path(args.study_out)
        study_dir.mkdir(parents=True, exist_ok=True)
        study_path = study_dir / "study.pkl"
        joblib.dump(study, study_path)
    except Exception as e:
        model_dir = Path(args.model_out)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / ".error").write_text(str(e))
        study_dir = Path(args.study_out)
        study_dir.mkdir(parents=True, exist_ok=True)
        (study_dir / ".error").write_text(str(e))

    metrics = {"algorithm": champion_algorithm, "best_params": best_params, "best_score": best_value}
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f)

    # Start MLflow run
    # Create dual logger for MLflow and Azure ML
    logger = create_metrics_logger(
        run_name="s08_phasec_hpo",
        tags={"pipeline": "v3_mlops", "phase": "phasec", "step": "s08"}
    )

    # Log metrics to MLflow for Azure ML Studio tracking (wrap in try-except to make non-fatal)
    try:
        logger.log_param("optimizer", "optuna")
        logger.log_param("algorithm", champion_algorithm)  # 🔥 NEW: Log which algorithm was tuned
        logger.log_param("task_type", task_type)
        logger.log_param("target_column", target_col)
        logger.log_param("best_params", str(best_params)[:500])
        logger.log_param("n_trials", n_trials)
        logger.log_metric("best_score", float(best_value))
        logger.log_metric("dataset_rows", int(df.shape[0]))
        logger.log_metric("dataset_cols", int(df.shape[1]))
        logger.log_dict(metrics, "optuna_hpo_metrics.json")
    except Exception as mlflow_err:
        print(f"⚠️  MLflow logging failed (non-fatal): {mlflow_err}")

    # End logging
    logger.end_run()

    # ── Emit stage signal ──────────────────────────────────────────────
    _elapsed = _time_mod.time() - _t0
    try:
        sig = StageSignal(
            stage_name="phasec_optuna_hpo",
            stage_id="S08",
            task_type=task_type,
            config_name=Path(args.config).name,
            candidate_count_in=n_trials,
            candidate_count_out=1,
            best_score=float(best_value),
            best_metric_name="best_score",
            compute_time_sec=round(_elapsed, 2),
            recommendation="proceed",
            recommendation_reason=f"HPO completed {n_trials} trials, best={best_value:.4f}",
            extra={"algorithm": champion_algorithm, "best_params": best_params},
        )
        write_stage_signal(sig, out_dir="outputs", filename="phasec_hpo_stage_signal.json")
    except Exception as _sig_err:
        print(f"⚠️  Stage signal write failed (non-fatal): {_sig_err}")

    # ── Candidate Ledger ──────────────────────────────────────────────────
    try:
        _ledger_rows = []
        _metric_name = "balanced_accuracy" if task_type == "classification" else "r2"
        for _trial in study.trials:
            _st = "ok" if _trial.state.name == "COMPLETE" else "failed"
            _val = _trial.value if _trial.value is not None else 0.0
            _norm = normalize_metrics(task_type, {_metric_name: _val})
            _row = make_row(
                stage="phase_c", step_name="s08", engine="optuna",
                candidate_id=f"trial_{_trial.number}",
                task_type=task_type,
                dataset_id=Path(args.config).name,
                status=_st,
                failure_reason="" if _st == "ok" else _trial.state.name,
                compute_time_sec=round((_trial.datetime_complete - _trial.datetime_start).total_seconds(), 2) if _trial.datetime_complete and _trial.datetime_start else 0.0,
                source_path="src/steps/phasec_optuna_hpo.py",
                recipe_name=champion_algorithm,
                candidate_rank=_trial.number + 1,
                is_stage_best=(_trial.number == study.best_trial.number),
                params_json=json.dumps(_trial.params, default=str),
                **_norm,
            )
            _ledger_rows.append(_row)
        write_stage_table(
            _ledger_rows,
            csv_path="outputs/s08_candidates.csv",
            parquet_path="outputs/s08_candidates.parquet",
        )
        print(f"📒 Candidate ledger: {len(_ledger_rows)} trial rows → s08_candidates.csv")
    except Exception as _ledger_err:
        print(f"⚠️  Candidate ledger write failed (non-fatal): {_ledger_err}")


if __name__ == "__main__":
    main()
