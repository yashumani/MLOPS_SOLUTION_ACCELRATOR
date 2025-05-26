"""
baseline_pipeline.py  – AutoML (RandomForest + XGBoost + FLAML)
----------------------------------------------------------------
• Cleans & one-hot encodes College.csv (optional Featuretools DFS)
• Hyper-parameter tuning via Optuna for RF & XGB (20 trials each)
• FLAML AutoML search (LightGBM, XGB, RF, ExtraTrees) – 10-min budget
• Logs every trial and best models to MLflow
• Generates and logs a leaderboard CSV of RMSE scores

Run:
  python src/baseline_pipeline.py
"""

import warnings, sys, optuna, joblib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn
import xgboost as xgb
from flaml import AutoML

# ───────── CONFIG ───────── #
DATA_PATH   = Path(__file__).resolve().parents[1] / "data/college.csv"
TARGET_COL  = "Grad.Rate"
EXPERIMENT  = "rf_xgb_flaml_regression"
MODEL_DIR   = Path("artifacts")
RND_STATE   = 42
TEST_SIZE   = 0.2
N_TRIALS    = 20        # Optuna trials for RF and XGB
TIME_BUDGET = 600       # seconds for FLAML
# ────────────────────────── #

# ---------- helpers ----------
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[ERROR] CSV not found: {path}")
    return pd.read_csv(path)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])
    return df

def encode_cats(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    return pd.get_dummies(df, columns=cat_cols.tolist(), drop_first=True) if len(cat_cols) else df

# OPTIONAL Featuretools DFS
def auto_feature_engineer(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Lightweight DFS (max_depth=2). Comment the call in main() to disable."""
    import featuretools as ft
    from sklearn.preprocessing import StandardScaler
    es = ft.EntitySet(id="college")
    es.add_dataframe("data", df.drop(columns=[target]).reset_index(drop=True),
                     make_index=True, index="row_id")
    fm, _ = ft.dfs(entityset=es, target_dataframe_name="data",
                   max_depth=2, verbose=False)
    num_cols = fm.select_dtypes(include=[np.number]).columns
    fm[num_cols] = StandardScaler().fit_transform(fm[num_cols])
    fm[target] = df[target].values
    return fm

def split(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RND_STATE)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ---------- FLAML ----------
def run_flaml(Xtr, ytr, Xte, yte):
    with mlflow.start_run(run_name="flaml_automl", nested=True):
        automl = AutoML()
        automl.fit(
            X_train=Xtr,
            y_train=ytr,
            time_budget=TIME_BUDGET,
            metric="rmse",
            task="regression",
            seed=RND_STATE,
            log_file_name="flaml.log",
        )
        score = rmse(yte, automl.predict(Xte))
        mlflow.log_metric("rmse", score)
        mlflow.sklearn.log_model(automl.model, "model")
        joblib.dump(automl.model,
                    MODEL_DIR / f"flaml_best_{datetime.now():%Y%m%d_%H%M%S}.joblib")
        print(f"FLAML best → {type(automl.model).__name__} | RMSE={score:.4f}")
        return automl.model, score

# ---------- main ----------
def main():
    warnings.filterwarnings("ignore")
    MODEL_DIR.mkdir(exist_ok=True)

    # ---- data prep ----
    df_raw = encode_cats(basic_clean(load_data(DATA_PATH)))
    # df_raw = auto_feature_engineer(df_raw, TARGET_COL)  # enable if desired
    Xtr, Xte, ytr, yte = split(df_raw)

    # MLflow setup
    mlflow.set_tracking_uri((Path.cwd() / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)

    # 1) Random Forest Optuna
    def obj_rf(trial):
        with mlflow.start_run(run_name=f"rf_trial_{trial.number}", nested=True):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
                "max_features": trial.suggest_categorical("max_features",
                                                          ["sqrt", "log2", None]),
                "random_state": RND_STATE,
                "n_jobs": -1,
            }
            model = RandomForestRegressor(**params).fit(Xtr, ytr)
            score = rmse(yte, model.predict(Xte))
            mlflow.log_params(params)
            mlflow.log_metric("rmse", score)
            return score

    rf_study = optuna.create_study(direction="minimize", study_name="rf")
    rf_study.optimize(obj_rf, n_trials=N_TRIALS, show_progress_bar=False)
    best_rf_params = rf_study.best_trial.params
    best_rf = RandomForestRegressor(**best_rf_params).fit(Xtr, ytr)
    best_rf_rmse = rmse(yte, best_rf.predict(Xte))

    with mlflow.start_run(run_name="best_model_rf", nested=True):
        mlflow.log_params(best_rf_params)
        mlflow.log_metric("rmse", best_rf_rmse)
        sig = infer_signature(Xte.iloc[:5], best_rf.predict(Xte.iloc[:5]))
        mlflow.sklearn.log_model(best_rf, "model", input_example=Xte.iloc[:5],
                                 signature=sig)
        joblib.dump(best_rf,
                    MODEL_DIR / f"rf_best_{datetime.now():%Y%m%d_%H%M%S}.joblib")

    # 2) XGBoost Optuna
    def obj_xgb(trial):
        with mlflow.start_run(run_name=f"xgb_trial_{trial.number}", nested=True):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3,
                                                     log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
                "random_state": RND_STATE,
            }
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
                **params,
            ).fit(Xtr, ytr)
            score = rmse(yte, model.predict(Xte))
            mlflow.log_params(params)
            mlflow.log_metric("rmse", score)
            return score

    xgb_study = optuna.create_study(direction="minimize", study_name="xgb")
    xgb_study.optimize(obj_xgb, n_trials=N_TRIALS, show_progress_bar=False)
    best_xgb_params = xgb_study.best_trial.params
    best_xgb = xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        **best_xgb_params
    ).fit(Xtr, ytr)
    best_xgb_rmse = rmse(yte, best_xgb.predict(Xte))

    with mlflow.start_run(run_name="best_model_xgb", nested=True):
        mlflow.log_params(best_xgb_params)
        mlflow.log_metric("rmse", best_xgb_rmse)
        sig = infer_signature(Xte.iloc[:5], best_xgb.predict(Xte.iloc[:5]))
        mlflow.sklearn.log_model(best_xgb, "model", input_example=Xte.iloc[:5],
                                 signature=sig)
        joblib.dump(best_xgb,
                    MODEL_DIR / f"xgb_best_{datetime.now():%Y%m%d_%H%M%S}.joblib")

    # 3) FLAML AutoML
    best_flaml_model, best_flaml_rmse = run_flaml(Xtr, ytr, Xte, yte)

    # ---- leaderboard ----
    leaderboard = pd.DataFrame([
        {"model": "RandomForest", "rmse": best_rf_rmse},
        {"model": "XGBoost",      "rmse": best_xgb_rmse},
        {"model": "FLAML",        "rmse": best_flaml_rmse},
    ]).sort_values(by="rmse")

    print("\n=== Leaderboard (lower RMSE is better) ===")
    print(leaderboard.to_string(index=False))

    lb_path = MODEL_DIR / "rmse_leaderboard.csv"
    leaderboard.to_csv(lb_path, index=False)
    mlflow.log_artifact(str(lb_path))       


if __name__ == "__main__":
    main()
