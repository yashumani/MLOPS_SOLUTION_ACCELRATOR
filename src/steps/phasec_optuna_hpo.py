import argparse
import hashlib
import json
import logging
import multiprocessing
import time as _time_mod
from pathlib import Path
import sys
import os
import tempfile
import traceback

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
import mlflow
from mlflow.tracking import MlflowClient

# Add src to path for imports (single canonical insertion at front of sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.azureml_metrics_logger import create_metrics_logger
from utils.stage_signals import StageSignal, write_stage_signal
from utils.candidate_ledger import (
    make_row, normalize_metrics, write_stage_table,
)
from utils.model_bundle import ModelBundle, capture_input_schema, save_model_bundle
from utils.fitted_variant_preprocessor import FittedVariantPreprocessor
from utils.common_evaluator import (
    EvaluationSpec,
    build_training_resampler,
    evaluate_candidate,
)
from orchestration.config_compiler import compile_config
from orchestration.execution_identity import validate_execution_manifest_binding

# Module-level logger for diagnostic/debug messages (does not shadow the
# per-run MetricsLogger created inside main()/cluster paths).
logger = logging.getLogger(__name__)


def _final_fit_worker(
    result_path: str,
    error_path: str,
    model: object,
    X: object,
    y: object,
    resampler: object | None,
) -> None:
    """Fit and serialize in an isolated process so the parent can kill hangs."""
    try:
        import joblib

        X_fit, y_fit = X, y
        if resampler is not None:
            X_fit, y_fit = resampler.fit_resample(X_fit, y_fit)
        if y_fit is None:
            model.fit(X_fit)
        else:
            model.fit(X_fit, y_fit)
        joblib.dump(model, result_path)
    except BaseException:
        failure = traceback.format_exc()
        # Report representation only, never feature values or target labels.
        try:
            from sklearn.utils.multiclass import type_of_target

            target_array = np.asarray(y)
            target_metadata = {
                "container": f"{type(y).__module__}.{type(y).__name__}",
                "dtype": str(getattr(y, "dtype", None)),
                "array_dtype": str(target_array.dtype),
                "array_kind": target_array.dtype.kind,
                "shape": list(target_array.shape),
                "target_type": type_of_target(y),
                "scalar_types": sorted({
                    f"{type(value).__module__}.{type(value).__name__}"
                    for value in target_array.reshape(-1)[:16]
                }),
            }
            failure += "\nFinal-fit target metadata: " + json.dumps(target_metadata)
        except Exception as diagnostic_error:
            failure += f"\nTarget metadata unavailable: {type(diagnostic_error).__name__}"
        Path(error_path).write_text(
            failure,
            encoding="utf-8",
        )


def fit_final_model_with_hard_timeout(
    model: object,
    X: object,
    y: object,
    *,
    resampler: object | None,
    timeout_seconds: float,
) -> object:
    """Return a fitted model or terminate the worker at the Phase C deadline."""
    if timeout_seconds <= 0:
        raise TimeoutError("Phase C final-fit budget is exhausted")
    if multiprocessing.current_process().daemon:
        raise RuntimeError(
            "Phase C final fit requires a non-daemon process boundary"
        )
    with tempfile.TemporaryDirectory(prefix="mlops-phasec-final-fit-") as temp:
        result_path = str(Path(temp) / "fitted_model.joblib")
        error_path = str(Path(temp) / "fit_error.txt")
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=_final_fit_worker,
            args=(
                result_path,
                error_path,
                model,
                X,
                y,
                resampler,
            ),
            name="phasec-final-fit",
        )
        process.start()
        process.join(timeout=float(timeout_seconds))
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=5.0)
            raise TimeoutError("Phase C final fit exceeded its hard deadline")
        if Path(error_path).is_file():
            raise RuntimeError(Path(error_path).read_text(encoding="utf-8"))
        if process.exitcode != 0 or not Path(result_path).is_file():
            raise RuntimeError(
                "Phase C final-fit worker exited without a fitted model"
            )
        import joblib

        return joblib.load(result_path)


def _safe_disable_autolog():
    """Disable MLflow autologging and fix tracking URI for Azure ML compatibility."""
    # Suppression is expected: Azure ML's azureml:// tracking URI is not fully
    # compatible with mlflow.sklearn.autolog / mlflow.autolog model registry
    # calls. We log at debug to retain forensic visibility without noise.
    try:
        mlflow.sklearn.autolog(disable=True)
    except Exception as e:
        logger.debug(f"MLflow sklearn autolog disable suppression (Azure ML tracking URI incompatibility): {e}")
    try:
        mlflow.autolog(disable=True)
    except Exception as e:
        logger.debug(f"MLflow autolog disable suppression (Azure ML tracking URI incompatibility): {e}")
    # Preserve the workspace-provided azureml:// tracking URI unchanged.


# T18: NumpyEncoder for safe JSON serialization of numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if not np.isfinite(v) else v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _phase_c_config(cfg: dict) -> dict:
    return (cfg.get("phases") or {}).get("phase_c_hpo") or {}


_ALGORITHM_ALIASES = {
    "xgboost": ("xgb", "xgboost", "extreme gradient boosting"),
    "lightgbm": ("lgb", "lightgbm", "light gradient boosting"),
    "catboost": ("catboost",),
    "randomforest": ("randomforest", "random forest", "rf"),
    "logisticregression": ("logisticregression", "logistic regression", "lr"),
    "ridge": ("ridge",),
    "kmeans": ("kmeans", "k-means"),
    "dbscan": ("dbscan",),
}


def normalize_phaseb_algorithm(value: str | None) -> str | None:
    """Resolve only families that Phase C can tune without substitution."""
    normalized = str(value or "").strip().lower()
    for family, aliases in _ALGORITHM_ALIASES.items():
        if normalized in aliases or any(alias in normalized for alias in aliases):
            return family
    return None


def phasec_candidate_id(
    phaseb_candidate_id: str,
    algorithm: str,
    best_params: dict,
) -> str:
    """Create a distinct immutable identity for the tuned candidate."""
    parameter_hash = hashlib.sha256(
        json.dumps(
            best_params,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return (
        f"phasec:{phaseb_candidate_id}:{algorithm}:tuned:{parameter_hash}"
    )


def seeded_optuna_sampler(random_seed: int):
    """Return the deterministic sampler shared by both Phase C branches."""
    import optuna as optuna_lib

    return optuna_lib.samplers.TPESampler(seed=int(random_seed))


def final_estimator_params(
    algorithm: str,
    best_params: dict,
    random_seed: int,
) -> dict:
    """Re-apply family seeds that are fixed outside Optuna's trial parameters."""
    params = dict(best_params)
    if algorithm in {"xgboost", "lightgbm", "randomforest", "logisticregression"}:
        params.setdefault("random_state", int(random_seed))
    elif algorithm == "catboost":
        params.setdefault("random_seed", int(random_seed))
    return params


def create_phasec_candidate_run(
    *,
    candidate_id: str,
    execution_id: str,
) -> tuple[MlflowClient, str | None, str]:
    """Create the exact MLflow child run referenced by Phase C artifacts."""
    if not str(execution_id).strip():
        raise ValueError("Phase C execution_id is required for MLflow lineage")
    client = MlflowClient()
    active = mlflow.active_run()
    parent_run_id = (
        active.info.run_id
        if active is not None
        else (os.getenv("AZUREML_RUN_ID") or "").strip() or None
    )
    if parent_run_id is not None:
        experiment_id = client.get_run(parent_run_id).info.experiment_id
    else:
        default_experiment = mlflow.get_experiment_by_name("Default")
        experiment_id = (
            default_experiment.experiment_id
            if default_experiment is not None
            else "0"
        )
    tags = {
        "execution_id": str(execution_id),
        "candidate_id": str(candidate_id),
        "pipeline_stage": "phase_c_hpo",
    }
    if parent_run_id is not None:
        tags["mlflow.parentRunId"] = parent_run_id
    child = client.create_run(
        experiment_id=experiment_id,
        run_name=f"phasec_candidate_{candidate_id[-24:]}",
        tags=tags,
    )
    return client, parent_run_id, child.info.run_id


def finish_phasec_candidate_run(
    client: MlflowClient,
    run_id: str,
    *,
    status: str,
) -> None:
    client.set_terminated(run_id, status=status)


def complete_phaseb_recipe(manifest: dict) -> dict | None:
    """Return only an explicit complete recipe mapping."""
    recipe = manifest.get("full_recipe")
    if recipe is None and isinstance(manifest.get("recipe"), dict):
        recipe = manifest["recipe"]
    if not isinstance(recipe, dict) or not recipe:
        return None
    stage3 = recipe.get("stage3_preprocessing")
    stage4 = recipe.get("stage4_feature_engineering")
    if not isinstance(stage3, dict) or not isinstance(stage4, dict):
        return None
    for field in (
        "imputation",
        "encoding",
        "scaling",
        "imbalance_handling",
    ):
        if not isinstance(stage3.get(field), dict) or not stage3[field]:
            return None
    feature_selection = stage4.get("feature_selection")
    if not isinstance(feature_selection, dict) or not feature_selection:
        return None
    if not stage3["imputation"].get("method"):
        return None
    if not (
        stage3["encoding"].get("categorical_method")
        or stage3["encoding"].get("method")
    ):
        return None
    if not stage3["scaling"].get("method"):
        return None
    if not stage3["imbalance_handling"].get("method"):
        return None
    if not feature_selection.get("method"):
        return None
    return json.loads(json.dumps(recipe, sort_keys=True, default=str))


def _write_skipped_unsupported(args, reason: str, phaseb: dict | None = None) -> None:
    payload = {
        "schema_version": 2,
        "status": "skipped_unsupported",
        "reason": str(reason),
        "preserve_phaseb": True,
        "phaseb_candidate_id": (phaseb or {}).get("candidate_id")
        or (phaseb or {}).get("variant_id"),
        "phaseb_algorithm": (phaseb or {}).get("algorithm"),
        "n_trials_completed": 0,
        "execution_id": getattr(args, "execution_id", None),
        "config_hash": getattr(args, "config_hash", None),
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for output in (Path(args.study_out), Path(args.model_out)):
        output.mkdir(parents=True, exist_ok=True)
        (output / ".skipped_unsupported").write_text(
            str(reason),
            encoding="utf-8",
        )


def _compute_hpo_cost(study, started_at: float, compute_rate_usd_per_hour: float) -> dict:
    trial_time_sec = 0.0
    for trial in getattr(study, "trials", []) or []:
        if getattr(trial, "datetime_start", None) and getattr(trial, "datetime_complete", None):
            trial_time_sec += max(0.0, (trial.datetime_complete - trial.datetime_start).total_seconds())
    elapsed_wall_clock_sec = max(0.0, _time_mod.time() - started_at)
    basis_sec = trial_time_sec if trial_time_sec > 0 else elapsed_wall_clock_sec
    trial_time_hours = basis_sec / 3600.0
    return {
        "compute_rate_usd_per_hour": float(compute_rate_usd_per_hour),
        "trial_time_hours": round(trial_time_hours, 6),
        "trial_time_sec": round(basis_sec, 2),
        "elapsed_wall_clock_sec": round(elapsed_wall_clock_sec, 2),
        "estimated_cost_usd": round(trial_time_hours * float(compute_rate_usd_per_hour), 4),
        "estimation_basis": "sum_trial_durations" if trial_time_sec > 0 else "wall_clock_elapsed",
    }


def _write_trial_score_plot(study, outputs_dir: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        trials = [trial for trial in getattr(study, "trials", []) or [] if trial.value is not None]
        if not trials:
            return None
        numbers = [trial.number for trial in trials]
        scores = [float(trial.value) for trial in trials]
        plt.figure(figsize=(10, 5))
        plt.plot(numbers, scores, marker="o", linewidth=1.5)
        plt.xlabel("Trial")
        plt.ylabel("Score")
        plt.title("Phase C HPO Trial Scores")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plot_path = outputs_dir / "phasec_trial_scores.png"
        plt.savefig(plot_path, dpi=120)
        plt.close()
        return str(plot_path)
    except Exception as exc:  # noqa: BLE001 - visualization must not fail HPO
        print(f"⚠️  Trial score PNG generation failed: {exc}")
        return None


def build_phasec_preprocessor(
    training_features: pd.DataFrame,
    encoding: str,
    scaling: str,
    random_seed: int,
):
    """Build the fitted-contract transformer used by Phase C and its model."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        MinMaxScaler,
        OneHotEncoder,
        OrdinalEncoder,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    encoding = str(encoding or "none").strip().lower()
    scaling = str(scaling or "none").strip().lower()
    categorical_columns = training_features.select_dtypes(
        include=["object", "category"],
    ).columns.tolist()

    transformers = []
    if categorical_columns:
        if encoding == "label":
            categorical_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
        elif encoding == "onehot":
            categorical_encoder = OneHotEncoder(
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
            )
        elif encoding == "target":
            from category_encoders import TargetEncoder

            categorical_encoder = TargetEncoder(
                cols=categorical_columns,
                handle_missing="value",
                handle_unknown="value",
                return_df=False,
            )
        elif encoding in {"none", "unknown"}:
            raise ValueError(
                "Phase C received categorical features but the champion recipe "
                "does not define an encoding contract"
            )
        else:
            raise ValueError(f"Unsupported Phase C encoding method: {encoding}")
        transformers.append(
            ("categorical", categorical_encoder, categorical_columns)
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough",
        sparse_threshold=0.0,
    )
    steps = [
        ("columns", column_transformer),
        ("imputer", SimpleImputer(strategy="median")),
    ]

    scaler = None
    if scaling == "standard":
        scaler = StandardScaler()
    elif scaling == "robust":
        scaler = RobustScaler()
    elif scaling == "minmax":
        scaler = MinMaxScaler()
    elif scaling == "yeo_johnson":
        scaler = PowerTransformer(method="yeo-johnson", standardize=True)
    elif scaling == "quantile":
        scaler = QuantileTransformer(
            n_quantiles=min(1000, max(1, len(training_features))),
            output_distribution="normal",
            random_state=random_seed,
        )
    elif scaling not in {"none", "unknown"}:
        raise ValueError(f"Unsupported Phase C scaling method: {scaling}")

    if scaler is not None:
        steps.append(("scaler", scaler))
    return Pipeline(steps)


def main():
    _t0 = _time_mod.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--execution_manifest", required=True)
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
        raw_cfg = yaml.safe_load(f) or {}
    cfg = compile_config(raw_cfg, source_name=Path(args.config).name)
    execution_manifest = validate_execution_manifest_binding(
        args.execution_manifest,
        cfg,
    )
    args.execution_id = execution_manifest.execution_id
    args.config_hash = execution_manifest.config_hash
    model_output = Path(args.model_out)
    model_output.mkdir(parents=True, exist_ok=True)
    (model_output / "execution_manifest.json").write_text(
        execution_manifest.to_json(indent=2),
        encoding="utf-8",
    )
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")  # 🔥 CRITICAL FIX
    phase_c_cfg = _phase_c_config(cfg)
    n_trials = min(50, max(1, int(phase_c_cfg.get("n_trials", 50))))
    compute_rate_usd_per_hour = float(phase_c_cfg.get("compute_rate_usd_per_hour", 1.50) or 0.0)
    # K9 fix: respect phase_c_hpo.timeout (seconds) -- previously ignored.
    hpo_timeout = phase_c_cfg.get("timeout_seconds", phase_c_cfg.get("timeout"))
    try:
        hpo_timeout = min(
            3600.0,
            float(hpo_timeout) if hpo_timeout is not None else 3600.0,
        )
    except (TypeError, ValueError):
        hpo_timeout = 3600.0
    random_seed = cfg.get("random_seed", 42)

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

    # 🔥 Agent 1: prefer sibling train.csv (holdout-leak-safe)
    _train_sibling = dataset_path.parent / "train.csv"
    if _train_sibling.exists() and _train_sibling.stat().st_size > 0:
        df = pd.read_csv(_train_sibling, sep=delimiter)
        print(f"  ✅ Switched to sibling train.csv ({len(df):,} rows) — holdout isolated")
    else:
        print(f"  ⚠️ No sibling train.csv — using combined dataset")
    
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

    if not getattr(args, "phaseb_manifest", None):
        _write_skipped_unsupported(
            args,
            "phaseb_manifest_required_for_same_family_hpo",
        )
        return
    manifest_path = Path(args.phaseb_manifest)
    if not manifest_path.is_file():
        _write_skipped_unsupported(args, "phaseb_manifest_not_found")
        return
    try:
        champion_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        _write_skipped_unsupported(args, f"invalid_phaseb_manifest: {error}")
        return
    champion_engine = str(champion_metadata.get("engine", "")).lower()
    champion_algorithm = normalize_phaseb_algorithm(
        champion_metadata.get("algorithm")
        or champion_metadata.get("best_model_name")
    )
    if task_type == "clustering" and champion_engine != "pycaret":
        _write_skipped_unsupported(
            args,
            "clustering_hpo_requires_pycaret_phaseb_candidate",
            champion_metadata,
        )
        return
    if champion_algorithm is None:
        _write_skipped_unsupported(
            args,
            "unsupported_phaseb_algorithm_family",
            champion_metadata,
        )
        return
    full_phaseb_recipe = complete_phaseb_recipe(champion_metadata)
    if full_phaseb_recipe is None:
        _write_skipped_unsupported(
            args,
            "complete_phaseb_recipe_required_for_deployable_bundle",
            champion_metadata,
        )
        return
    phaseb_candidate_id = str(
        champion_metadata.get("candidate_id")
        or champion_metadata.get("variant_id")
        or ""
    ).strip()
    if not phaseb_candidate_id:
        _write_skipped_unsupported(
            args,
            "phaseb_candidate_identity_required",
            champion_metadata,
        )
        return
    execution_id = str(champion_metadata.get("execution_id") or "").strip()
    if execution_id != execution_manifest.execution_id:
        raise RuntimeError(
            "Phase B champion execution_id does not match ExecutionManifest: "
            f"{execution_id!r} != {execution_manifest.execution_id!r}"
        )

    preproc_cfg = champion_metadata.get("preprocessing_config") or {}
    recipe_preprocessing = (
        full_phaseb_recipe.get("stage3_preprocessing")
        or full_phaseb_recipe.get("preprocessing")
        or {}
    )
    if not isinstance(recipe_preprocessing, dict):
        recipe_preprocessing = {}
    recipe_encoding_config = recipe_preprocessing.get("encoding") or {}
    recipe_scaling_config = recipe_preprocessing.get("scaling") or {}
    recipe_encoding = preproc_cfg.get(
        "encoding",
        recipe_encoding_config.get(
            "categorical_method",
            recipe_encoding_config.get(
                "method",
                full_phaseb_recipe.get("encoding", "none"),
            ),
        )
    )
    recipe_scaling = preproc_cfg.get(
        "scaling",
        recipe_scaling_config.get(
            "method",
            full_phaseb_recipe.get("scaling", "none"),
        )
    )
    try:
        training_resampler = build_training_resampler(
            full_phaseb_recipe,
            random_seed,
        )
    except (ImportError, ValueError) as error:
        _write_skipped_unsupported(
            args,
            f"phaseb_resampling_contract_unavailable: {error}",
            champion_metadata,
        )
        return
    
    import optuna
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sys.stdout.reconfigure(line_buffering=True)  # Force line-buffered stdout for Azure ML logs
    
    # Handle clustering separately (no target column validation needed)
    if task_type == "clustering":
        print("ℹ️ Phase C HPO: Clustering task detected; using sklearn clustering + Optuna")
        from sklearn.cluster import KMeans, DBSCAN
        import gc
        
        # Clustering uses training rows only. Persist the fitted raw-input
        # transformer instead of silently dropping categorical columns.
        raw_clustering_data = df.copy()
        if str(recipe_encoding).lower() == "target":
            _write_skipped_unsupported(
                args,
                "target_encoding_is_not_valid_for_clustering",
                champion_metadata,
            )
            return
        if training_resampler is not None:
            _write_skipped_unsupported(
                args,
                "imbalance_resampling_is_not_valid_for_clustering",
                champion_metadata,
            )
            return
        
        # OOM guard: DBSCAN computes O(n²) pairwise distances and KMeans on
        # very large datasets (>500K rows) across 50 trials blows 16 GB RAM.
        # Cap the dataset used for ALL HPO operations (fitting + scoring).
        _HPO_DATA_CAP = 15_000
        _full_rows = len(raw_clustering_data)
        if _full_rows > _HPO_DATA_CAP:
            raw_clustering_data = raw_clustering_data.sample(
                n=_HPO_DATA_CAP,
                random_state=random_seed,
            ).reset_index(drop=True)
            print(f"⚠️  Sampled dataset from {_full_rows:,} → {_HPO_DATA_CAP:,} rows for HPO (OOM guard)")

        phasec_preprocessor = FittedVariantPreprocessor(
            full_phaseb_recipe,
            random_seed=random_seed,
        )
        X_data = phasec_preprocessor.fit_transform(raw_clustering_data)
        _SILHOUETTE_SAMPLE_CAP = 5_000
        _sil_sample = min(_SILHOUETTE_SAMPLE_CAP, len(X_data))
        print(f"📊 Clustering HPO data: {len(X_data)} rows × {X_data.shape[1]} cols, "
              f"silhouette sample_size={_sil_sample}")

        # Down-cast to float32 to halve memory footprint
        X_data = X_data.astype(np.float32)
        gc.collect()

        # Ensure valid upper bound for KMeans n_clusters
        _max_k = max(2, min(10, len(X_data) // 5))
        clustering_hpo_deadline = _time_mod.monotonic() + float(hpo_timeout)

        def objective(trial: optuna.Trial):
            algo = champion_algorithm
            print(f"  🔄 Trial {trial.number}/{n_trials}: algo={algo}", end="", flush=True)
            if algo == "kmeans":
                n_clusters = trial.suggest_int("n_clusters", 2, _max_k)
                model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
            elif algo == "dbscan":
                eps = trial.suggest_float("eps", 0.1, 1.0)
                min_samples = trial.suggest_int("min_samples", 2, 10)
                model = DBSCAN(eps=eps, min_samples=min_samples)
            else:
                raise RuntimeError(
                    f"Unsupported same-family clustering HPO: {algo}"
                )
            trial_pipeline = Pipeline(
                [
                    (
                        "preprocessor",
                        FittedVariantPreprocessor(
                            full_phaseb_recipe,
                            random_seed=random_seed,
                        ),
                    ),
                    ("estimator", model),
                ]
            )
            remaining_hpo_seconds = (
                clustering_hpo_deadline - _time_mod.monotonic()
            )
            if remaining_hpo_seconds <= 0:
                raise optuna.TrialPruned("HPO wall-clock budget exhausted")
            evidence = evaluate_candidate(
                trial_pipeline,
                raw_clustering_data,
                None,
                candidate_id=phasec_candidate_id(
                    phaseb_candidate_id,
                    algo,
                    trial.params,
                ),
                engine="pycaret",
                spec=EvaluationSpec(
                    task_type="clustering",
                    seed=int(random_seed),
                    timeout_seconds=float(remaining_hpo_seconds),
                    execution_id=execution_id,
                ),
            )
            if evidence.status == "timeout":
                raise optuna.TrialPruned("HPO wall-clock budget exhausted")
            if not evidence.selectable:
                raise RuntimeError(
                    "Common clustering HPO evaluation failed: "
                    f"{evidence.failure_reason or evidence.status}"
                )
            score = float(evidence.selection_score)
            trial.set_user_attr(
                "clustered_fraction",
                evidence.metrics.get("clustered_fraction"),
            )
            trial.set_user_attr(
                "split_fingerprint",
                evidence.split_fingerprint,
            )
            trial.set_user_attr("total_folds", evidence.total_folds)
            print(f" → score={score:.4f}", flush=True)
            gc.collect()
            return score
        
        print(f"\n🔬 Starting Optuna clustering study ({n_trials} trials"
              + (f", timeout={hpo_timeout}s" if hpo_timeout else "") + "):", flush=True)
        study = optuna.create_study(
            direction="maximize",
            sampler=seeded_optuna_sampler(random_seed),
        )
        study.optimize(objective, n_trials=n_trials, timeout=hpo_timeout,
                       catch=(Exception,))
        print(f"✅ Optuna study complete. Best value: {study.best_value:.4f}", flush=True)
        
        best_params = study.best_params
        best_value = study.best_value
        best_algo = champion_algorithm
        best_split_fingerprint = study.best_trial.user_attrs.get(
            "split_fingerprint"
        )
        best_total_folds = study.best_trial.user_attrs.get("total_folds")
        
        # Train final model with best params
        if best_algo == "kmeans":
            final_model = KMeans(n_clusters=best_params.get("n_clusters", 3), random_state=random_seed, n_init=10)
        else:
            final_model = DBSCAN(eps=best_params.get("eps", 0.5), min_samples=best_params.get("min_samples", 5))
        final_model = fit_final_model_with_hard_timeout(
            final_model,
            X_data,
            None,
            resampler=None,
            timeout_seconds=(
                clustering_hpo_deadline - _time_mod.monotonic()
            ),
        )
        
        print(f"✅ Clustering HPO: Selected {best_algo} | silhouette_score={best_value:.4f} | params={best_params}")
        
        # Save model and study
        try:
            import joblib
            model_dir = Path(args.model_out).resolve()
            model_dir.mkdir(parents=True, exist_ok=True)
            tuned_candidate_id = phasec_candidate_id(
                phaseb_candidate_id,
                best_algo,
                best_params,
            )
            (
                lineage_client,
                parent_run_id,
                candidate_run_id,
            ) = create_phasec_candidate_run(
                candidate_id=tuned_candidate_id,
                execution_id=execution_id,
            )
            phasec_bundle = ModelBundle(
                estimator=final_model,
                preprocessing=phasec_preprocessor,
                task_type=task_type,
                candidate_id=tuned_candidate_id,
                input_schema=capture_input_schema(raw_clustering_data),
                recipe={
                    **full_phaseb_recipe,
                    "phasec_hpo": {
                        "algorithm_family": best_algo,
                        "best_params": best_params,
                    },
                },
                selection_metrics={
                    "primary_metric": "silhouette_score",
                    "selection_score": best_value,
                    "n_trials": len(study.trials),
                    "split_fingerprint": best_split_fingerprint,
                    "total_folds": best_total_folds,
                },
                environment={
                    "component": "phasec_optuna_hpo",
                    "environment_name": os.getenv("AZUREML_ENVIRONMENT_NAME"),
                    "environment_version": os.getenv(
                        "AZUREML_ENVIRONMENT_VERSION"
                    ),
                },
                lineage={
                    "execution_id": execution_id,
                    "parent_run_id": parent_run_id,
                    "candidate_run_id": candidate_run_id,
                    "phaseb_candidate_id": phaseb_candidate_id,
                },
                dependencies=("optuna", "scikit-learn", "pandas"),
                signature={
                    "inputs": list(
                        capture_input_schema(raw_clustering_data)[
                            "column_order"
                        ]
                    ),
                    "outputs": ["cluster"],
                },
                input_example=raw_clustering_data.head(1).to_dict(
                    orient="records"
                ),
            )
            try:
                bundle_manifest = save_model_bundle(phasec_bundle, model_dir)
                lineage_client.log_metric(
                    candidate_run_id,
                    "silhouette_score",
                    float(best_value),
                )
                for artifact_name in (
                    "model_bundle.pkl",
                    "model_bundle_manifest.json",
                ):
                    lineage_client.log_artifact(
                        candidate_run_id,
                        str(model_dir / artifact_name),
                        artifact_path="model_bundle",
                    )
                finish_phasec_candidate_run(
                    lineage_client,
                    candidate_run_id,
                    status="FINISHED",
                )
            except Exception:
                finish_phasec_candidate_run(
                    lineage_client,
                    candidate_run_id,
                    status="FAILED",
                )
                raise
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
        
        cost_summary = _compute_hpo_cost(study, _t0, compute_rate_usd_per_hour)
        trial_plot_path = _write_trial_score_plot(study, outputs_dir)
        metrics = {
            "schema_version": 2,
            "status": "success",
            "candidate_id": tuned_candidate_id,
            "algorithm": best_algo,
            "phaseb_algorithm": best_algo,
            "same_family": True,
            "best_params": best_params,
            "best_score": best_value,
            "selection_metric": "silhouette_score",
            "selection_evidence": "training_only_clustering",
            "split_fingerprint": best_split_fingerprint,
            "total_folds": best_total_folds,
            "model_bundle": bundle_manifest,
            "execution_id": phasec_bundle.lineage.get("execution_id"),
            "mlflow_parent_run_id": phasec_bundle.lineage.get("parent_run_id"),
            "mlflow_child_run_id": phasec_bundle.lineage.get(
                "candidate_run_id"
            ),
            **cost_summary,
        }
        if trial_plot_path:
            metrics["trial_score_plot"] = trial_plot_path
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, cls=NumpyEncoder)
        
        # Create logger for clustering MLflow logging (before early return)
        logger = create_metrics_logger(
            run_name="s08_phasec_hpo",
            tags={"pipeline": "v3_mlops", "phase": "phasec", "step": "s08"}
        )
        
        # Log metrics to MLflow (T13: clamp -inf/nan)
        try:
            logger.log_param("optimizer", "optuna")
            logger.log_param("task_type", task_type)
            logger.log_param("algorithm", best_algo)
            logger.log_param("best_params", str(best_params)[:500])
            logger.log_param("n_trials", n_trials)
            _bv = float(best_value) if np.isfinite(float(best_value)) else -999.0
            logger.log_metric("best_score", _bv)
            logger.log_metric("estimated_cost_usd", float(metrics.get("estimated_cost_usd", 0.0)))
            logger.log_metric("hpo_trial_time_hours", float(metrics.get("trial_time_hours", 0.0)))
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
            
            y_train = y_encoded
        else:
            y_train = y
    else:
        y_train = y
    
    # Same-family HPO uses only the explicit Phase B manifest loaded above.
    print(f"✅ Phase B manifest received via pipeline input: {manifest_path}")
    print(f"\n🏆 PHASE C: Tuning Phase B Champion:")
    print(f"  Algorithm family: {champion_algorithm}")
    print(f"  Engine: {champion_engine}")
    phaseb_recipe = champion_metadata.get(
        "variant_path",
        champion_metadata.get("recipe", "unknown"),
    )
    phaseb_variant = champion_metadata.get(
        "variant_id",
        champion_metadata.get("variant", "unknown"),
    )

    # Preserve the raw feature space for the persisted sklearn Pipeline. The
    # fitted transformer is reused for HPO and bundled with the final estimator,
    # so final evaluation and registered-model inference apply the same contract.
    # S02 already removed the one locked final-test partition. HPO selection
    # uses fold-local CV across every remaining training row; no second
    # pseudo-test split is created or evaluated.
    X_train_raw = X.copy()
    phasec_preprocessor = FittedVariantPreprocessor(
        full_phaseb_recipe,
        random_seed=random_seed,
    )
    X_train = phasec_preprocessor.fit_transform(X_train_raw, y_train)
    print(
        "  ✅ Fitted Phase C preprocessing on training rows only: "
        f"encoding={recipe_encoding}, scaling={recipe_scaling}, "
        f"features={X_train.shape[1]}"
    )
    
    print(f"\n🎯 Phase C HPO Target: {champion_algorithm}")
    
    # 🔥 NEW: Configure cross-validation for robust hyperparameter evaluation
    cv_folds = int(cfg["split"]["cv_folds"])
    print(
        f"✅ Using compiled {cv_folds}-fold cross-validation contract "
        "for hyperparameter evaluation"
    )
    hpo_deadline = _time_mod.monotonic() + float(hpo_timeout)
    
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
            _write_skipped_unsupported(
                args,
                f"unsupported_phaseb_algorithm_family: {champion_algorithm}",
                champion_metadata,
            )
            return
    except ImportError as import_err:
        _write_skipped_unsupported(
            args,
            f"phaseb_algorithm_dependency_unavailable: {import_err}",
            champion_metadata,
        )
        return

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
            raise optuna.TrialPruned(
                f"unsupported same-family HPO: {champion_algorithm}"
            )
        
        fold_steps = [
            (
                "preprocessor",
                FittedVariantPreprocessor(
                    full_phaseb_recipe,
                    random_seed=random_seed,
                ),
            ),
        ]
        if training_resampler is not None:
            from imblearn.pipeline import Pipeline as ImbalancedPipeline

            fold_steps.append(("resampler", clone(training_resampler)))
            pipeline_type = ImbalancedPipeline
        else:
            pipeline_type = Pipeline
        fold_steps.append(("estimator", model))
        fold_pipeline = pipeline_type(fold_steps)
        remaining_hpo_seconds = hpo_deadline - _time_mod.monotonic()
        if remaining_hpo_seconds <= 0:
            raise optuna.TrialPruned("HPO wall-clock budget exhausted")
        evidence = evaluate_candidate(
            fold_pipeline,
            X_train_raw,
            y_train,
            candidate_id=phasec_candidate_id(
                phaseb_candidate_id,
                champion_algorithm,
                params,
            ),
            engine=champion_engine,
            spec=EvaluationSpec(
                task_type=task_type,
                seed=int(random_seed),
                folds=int(cv_folds),
                timeout_seconds=float(remaining_hpo_seconds),
                execution_id=execution_id,
            ),
        )
        if evidence.status == "timeout":
            raise optuna.TrialPruned("HPO wall-clock budget exhausted")
        if not evidence.selectable:
            raise RuntimeError(
                "Common HPO evaluation failed: "
                f"{evidence.failure_reason or evidence.status}"
            )
        score = float(evidence.selection_score)
        metric_name = evidence.primary_metric
        trial.set_user_attr(
            "cv_std",
            evidence.metrics.get(f"{metric_name}_std"),
        )
        trial.set_user_attr("cv_scores", [
            row.get(metric_name) for row in evidence.fold_metrics
        ])
        trial.set_user_attr(
            "split_fingerprint",
            evidence.split_fingerprint,
        )
        trial.set_user_attr("total_folds", evidence.total_folds)
        
        print(f"  🔄 Trial {trial.number}/{n_trials}: {champion_algorithm} → score={score:.4f}", flush=True)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=seeded_optuna_sampler(random_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,  # No pruning for first 10 trials
            n_warmup_steps=5,     # Start pruning after 5 CV folds if using CV
            interval_steps=1       # Check every step
        )
    )
    print(f"\n🔬 Starting Optuna study ({n_trials} trials"
          + (f", timeout={hpo_timeout}s" if hpo_timeout else "")
          + ", MedianPruner enabled):", flush=True)
    _hpo_fallback = False
    try:
        # K8 fix: catch=(Exception,) so a single bad trial does not abort the
        # whole study. K9 fix: pass user-configured timeout.
        study.optimize(objective, n_trials=n_trials, timeout=hpo_timeout,
                       catch=(Exception,))
        if len(study.trials) == 0 or all(t.state.name != "COMPLETE" for t in study.trials):
            raise RuntimeError("Optuna study finished with zero successful trials")
        best_params = study.best_params
        best_value = study.best_value
        best_split_fingerprint = study.best_trial.user_attrs.get(
            "split_fingerprint"
        )
        best_total_folds = study.best_trial.user_attrs.get("total_folds")
    except Exception as _hpo_err:
        print(f"\n⚠️  HPO FAILED — no valid trials completed: {_hpo_err}")
        _write_skipped_unsupported(
            args,
            f"same_family_hpo_failed: {_hpo_err}",
            champion_metadata,
        )
        return
    
    # 📊 EXPORT ALL OPTUNA TRIALS TO CSV
    if not _hpo_fallback:
        print(f"\n📊 EXPORTING OPTUNA TRIAL LOGS:")
        trials_df = study.trials_dataframe()
        trials_path = outputs_dir / "phasec_optuna_trials.csv"
        trials_df.to_csv(trials_path, index=False)
        print(f"  ✅ All trials: {trials_path} ({len(trials_df)} trials, {trials_path.stat().st_size:,} bytes)")
        
        # Save best hyperparameters
        best_params_path = outputs_dir / "phasec_optuna_best_params.json"
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2, cls=NumpyEncoder)
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
        trial_plot_path = _write_trial_score_plot(study, outputs_dir)
        if trial_plot_path:
            print(f"  ✅ Trial score PNG: {trial_plot_path}")

    print(f"\n🏆 Training final model with best hyperparameters:")
    print(f"  Algorithm: {champion_algorithm}")
    print(f"  Best params: {best_params}")
    print(f"  Best score: {best_value:.4f}")
    seeded_best_params = final_estimator_params(
        champion_algorithm,
        best_params,
        random_seed,
    )
    
    # 🔥 NEW: Train final model using champion algorithm (not hardcoded XGBoost)
    if champion_algorithm == "xgboost":
        import xgboost as xgb
        if task_type == "classification":
            final_model = xgb.XGBClassifier(
                **seeded_best_params,
                objective="binary:logistic",
            )
        else:
            final_model = xgb.XGBRegressor(
                **seeded_best_params,
                objective="reg:squarederror",
            )
    
    elif champion_algorithm == "lightgbm":
        import lightgbm as lgb
        if task_type == "classification":
            final_model = lgb.LGBMClassifier(**seeded_best_params)
        else:
            final_model = lgb.LGBMRegressor(**seeded_best_params)
    
    elif champion_algorithm == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        if task_type == "classification":
            final_model = CatBoostClassifier(**seeded_best_params)
        else:
            final_model = CatBoostRegressor(**seeded_best_params)
    
    elif champion_algorithm == "randomforest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        if task_type == "classification":
            final_model = RandomForestClassifier(**seeded_best_params)
        else:
            final_model = RandomForestRegressor(**seeded_best_params)
    
    elif champion_algorithm in ["logisticregression", "ridge"]:
        from sklearn.linear_model import LogisticRegression, Ridge
        if task_type == "classification":
            final_model = LogisticRegression(**seeded_best_params)
        else:
            # Ridge uses alpha instead of C
            final_model = Ridge(alpha=1.0/best_params["C"], random_state=random_seed)
    
    else:
        _write_skipped_unsupported(
            args,
            f"unsupported_same_family_final_fit: {champion_algorithm}",
            champion_metadata,
        )
        return
    
    if not _hpo_fallback:
        final_model = fit_final_model_with_hard_timeout(
            final_model,
            X_train,
            y_train,
            resampler=(
                clone(training_resampler)
                if training_resampler is not None
                else None
            ),
            timeout_seconds=hpo_deadline - _time_mod.monotonic(),
        )

    # best_value is CV selection evidence. The locked test is evaluated only by
    # S10 after the champion is frozen.

    # Save the fitted preprocessing contract together with the estimator.
    try:
        import joblib

        model_dir = Path(args.model_out)
        model_dir.mkdir(parents=True, exist_ok=True)
        tuned_candidate_id = phasec_candidate_id(
            phaseb_candidate_id,
            champion_algorithm,
            best_params,
        )
        (
            lineage_client,
            parent_run_id,
            candidate_run_id,
        ) = create_phasec_candidate_run(
            candidate_id=tuned_candidate_id,
            execution_id=execution_id,
        )
        contract_input = X_train_raw.head(min(5, len(X_train_raw)))
        contract_predictions = final_model.predict(
            phasec_preprocessor.transform(contract_input)
        )
        if len(contract_predictions) != len(contract_input):
            raise RuntimeError(
                "Persisted Phase C model contract returned an invalid row count"
            )
        phasec_bundle = ModelBundle(
            estimator=final_model,
            preprocessing=phasec_preprocessor,
            target_decoder=label_encoder,
            task_type=task_type,
            candidate_id=tuned_candidate_id,
            input_schema=capture_input_schema(X_train_raw),
            recipe={
                **full_phaseb_recipe,
                "phasec_hpo": {
                    "algorithm_family": champion_algorithm,
                    "best_params": best_params,
                },
            },
            selection_metrics={
                "primary_metric": (
                    "balanced_accuracy"
                    if task_type == "classification"
                    else "r2"
                ),
                "selection_score": best_value,
                "n_trials": len(study.trials),
                "split_fingerprint": best_split_fingerprint,
                "total_folds": best_total_folds,
            },
            environment={
                "component": "phasec_optuna_hpo",
                "environment_name": os.getenv("AZUREML_ENVIRONMENT_NAME"),
                "environment_version": os.getenv("AZUREML_ENVIRONMENT_VERSION"),
            },
            lineage={
                "execution_id": execution_id,
                "parent_run_id": parent_run_id,
                "candidate_run_id": candidate_run_id,
                "phaseb_candidate_id": phaseb_candidate_id,
            },
            dependencies=("optuna", "scikit-learn", "pandas"),
            labels=(
                tuple(label_encoder.classes_)
                if label_encoder is not None
                else tuple()
            ),
            signature={
                "inputs": list(
                    capture_input_schema(X_train_raw)["column_order"]
                ),
                "outputs": ["prediction"],
            },
            input_example=X_train_raw.head(1).to_dict(orient="records"),
        )
        try:
            bundle_manifest = save_model_bundle(phasec_bundle, model_dir)
            lineage_client.log_metric(
                candidate_run_id,
                (
                    "balanced_accuracy"
                    if task_type == "classification"
                    else "r2"
                ),
                float(best_value),
            )
            for artifact_name in (
                "model_bundle.pkl",
                "model_bundle_manifest.json",
            ):
                lineage_client.log_artifact(
                    candidate_run_id,
                    str(model_dir / artifact_name),
                    artifact_path="model_bundle",
                )
            finish_phasec_candidate_run(
                lineage_client,
                candidate_run_id,
                status="FINISHED",
            )
        except Exception:
            finish_phasec_candidate_run(
                lineage_client,
                candidate_run_id,
                status="FAILED",
            )
            raise
        
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
        print(f"  ⚠️  Primary model save failed: {e}")
        _write_skipped_unsupported(
            args,
            f"same_family_model_bundle_failed: {e}",
            champion_metadata,
        )
        return

    cost_summary = _compute_hpo_cost(study, _t0, compute_rate_usd_per_hour)
    metrics = {
        "schema_version": 2,
        "status": "success",
        "candidate_id": tuned_candidate_id,
        "algorithm": champion_algorithm,
        "phaseb_algorithm": champion_algorithm,
        "same_family": True,
        "best_params": best_params,
        "best_score": best_value,
        "selection_metric": (
            "balanced_accuracy" if task_type == "classification" else "r2"
        ),
        "selection_evidence": "cross_validation",
        "split_fingerprint": best_split_fingerprint,
        "total_folds": best_total_folds,
        "n_trials_requested": n_trials,
        "n_trials_completed": len(
            [trial for trial in study.trials if trial.state.name == "COMPLETE"]
        ),
        "timeout_seconds": hpo_timeout,
        "model_bundle": bundle_manifest,
        "execution_id": phasec_bundle.lineage.get("execution_id"),
        "mlflow_parent_run_id": phasec_bundle.lineage.get("parent_run_id"),
        "mlflow_child_run_id": phasec_bundle.lineage.get("candidate_run_id"),
        **cost_summary,
    }
    if 'trial_plot_path' in locals() and trial_plot_path:
        metrics["trial_score_plot"] = trial_plot_path
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, cls=NumpyEncoder)

    # Start MLflow run
    # Create dual logger for MLflow and Azure ML
    logger = create_metrics_logger(
        run_name="s08_phasec_hpo",
        tags={
            "pipeline": "v3_mlops",
            "phase": "phasec",
            "step": "s08",
            "execution_id": execution_id,
        }
    )

    # Log metrics to MLflow for Azure ML Studio tracking (wrap in try-except to make non-fatal)
    try:
        logger.log_param("optimizer", "optuna")
        logger.log_param("algorithm", champion_algorithm)  # 🔥 NEW: Log which algorithm was tuned
        logger.log_param("task_type", task_type)
        logger.log_param("target_column", target_col)
        logger.log_param("best_params", str(best_params)[:500])
        logger.log_param("n_trials", n_trials)
        _bv = float(best_value) if np.isfinite(float(best_value)) else -999.0  # T13: clamp
        logger.log_metric("best_score", _bv)
        logger.log_metric("estimated_cost_usd", float(metrics.get("estimated_cost_usd", 0.0)))
        logger.log_metric("hpo_trial_time_hours", float(metrics.get("trial_time_hours", 0.0)))
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
