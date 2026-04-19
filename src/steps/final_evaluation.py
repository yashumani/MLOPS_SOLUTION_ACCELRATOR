import argparse
import json
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score
import mlflow

# Ensure src/ on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Setup MLflow registry URI BEFORE any mlflow.log_* calls
os.makedirs("/tmp/mlflow-registry", exist_ok=True)
mlflow.set_registry_uri("file:///tmp/mlflow-registry")


def load_model(path: str):
    try:
        import joblib
        from pathlib import Path
        path_obj = Path(path)
        
        # If path is a folder, look for model.pkl inside
        if path_obj.is_dir():
            model_file = path_obj / "model.pkl"
            if model_file.exists():
                return joblib.load(str(model_file))
        # If path is a file, load it directly
        elif path_obj.exists() and path_obj.suffix == ".pkl":
            return joblib.load(path)
        
        return None
    except Exception:
        return None


def eval_model(model, X_test, y_test, task: str):
    if model is None:
        return None
    try:
        preds = model.predict(X_test)
        if task == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall": float(recall_score(y_test, preds, zero_division=0)),
                "f1": float(f1_score(y_test, preds, zero_division=0)),
            }
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_test, prob))
            except Exception:
                pass
            return metrics
        else:
            return {"r2": float(r2_score(y_test, preds))}
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--baseline_model", required=True)
    parser.add_argument("--phaseb_model", required=True)
    parser.add_argument("--phasec_model", required=True)
    parser.add_argument("--report_out", required=True)
    parser.add_argument("--champion_out", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type") or cfg.get("dataset", {}).get("task_type") or "classification"
    target_col = cfg.get("dataset", {}).get("target_column")
    test_size = cfg.get("stages", {}).get("stage4_feature_engineering", {}).get("train_test_splits", [0.8])[0]
    test_size = 1 - float(test_size)

    df = pd.read_csv(args.dataset_in)

    # Clustering: evaluate using silhouette score (no target column, no train/test split)
    if task_type == "clustering":
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        numeric_df = df.select_dtypes(include=[np.number]).astype(np.float64)

        def eval_clustering_model(path):
            model = load_model(path)
            if model is None:
                return None
            try:
                preds = model.predict(df)
                sil = float(silhouette_score(numeric_df, preds))
                db = float(davies_bouldin_score(numeric_df, preds))
                return {"silhouette_score": round(sil, 4), "davies_bouldin_score": round(db, 4)}
            except Exception:
                return None

        mb = eval_clustering_model(args.baseline_model)
        pb = eval_clustering_model(args.phaseb_model)
        pc = eval_clustering_model(args.phasec_model)

        def primary_score(m):
            if m is None:
                return -np.inf
            return m.get("silhouette_score", -np.inf)

        candidates = {"baseline": (mb, args.baseline_model), "phaseb": (pb, args.phaseb_model), "phasec": (pc, args.phasec_model)}
        best_key = None
        best_val = -np.inf
        for k, (metrics, path) in candidates.items():
            val = primary_score(metrics)
            if val > best_val:
                best_key, best_val = k, val

        report = {
            "task": task_type,
            "baseline_metrics": mb,
            "phaseb_metrics": pb,
            "phasec_metrics": pc,
            "selection": {"key": best_key, "score": best_val},
        }
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump(report, f)

        chosen_path = candidates.get(best_key, (None, None))[1]
        if chosen_path:
            src = Path(chosen_path)
            out_dir = Path(args.champion_out)
            out_dir.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                model_file = src / "model.pkl"
                if model_file.exists():
                    import shutil
                    shutil.copy(str(model_file), str(out_dir / "model.pkl"))
            elif src.exists() and src.suffix == ".pkl":
                import shutil
                shutil.copy(str(src), str(out_dir / "model.pkl"))
        else:
            Path(args.champion_out).mkdir(parents=True, exist_ok=True)
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Sanitize column names to match what training steps used
    import re
    X.columns = [re.sub(r'[^\w]', '_', c) for c in X.columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task_type == "classification" else None)

    baseline = load_model(args.baseline_model)
    phaseb = load_model(args.phaseb_model)
    phasec = load_model(args.phasec_model)

    mb = eval_model(baseline, X_test, y_test, task_type)
    pb = eval_model(phaseb, X_test, y_test, task_type)
    pc = eval_model(phasec, X_test, y_test, task_type)

    def primary_score(m):
        if m is None:
            return -np.inf
        return m.get("accuracy", m.get("r2", -np.inf))

    candidates = {"baseline": (mb, args.baseline_model), "phaseb": (pb, args.phaseb_model), "phasec": (pc, args.phasec_model)}
    best_key = None
    best_val = -np.inf
    for k, (metrics, path) in candidates.items():
        val = primary_score(metrics)
        if val > best_val:
            best_key, best_val = k, val

    report = {
        "task": task_type,
        "baseline_metrics": mb,
        "phaseb_metrics": pb,
        "phasec_metrics": pc,
        "selection": {"key": best_key, "score": best_val},
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w") as f:
        json.dump(report, f)

    chosen_path = candidates.get(best_key, (None, None))[1]
    if chosen_path:
        src = Path(chosen_path)
        out_dir = Path(args.champion_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # If source is a folder, copy model.pkl from it
        if src.is_dir():
            model_file = src / "model.pkl"
            if model_file.exists():
                import shutil
                shutil.copy(str(model_file), str(out_dir / "model.pkl"))
        # If source is a file, copy it directly
        elif src.exists() and src.suffix == ".pkl":
            import shutil
            shutil.copy(str(src), str(out_dir / "model.pkl"))
    else:
        Path(args.champion_out).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
