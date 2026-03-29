"""
Stage 5t — Time-Series / Forecasting Training Step.

Only executes if Stage 1 detected the dataset as time-series
(``time_series_detection.json`` → ``is_time_series: true``).
If not time-series, the step writes a skip artefact and exits.

Uses statsmodels (ARIMA, SARIMA, Exponential Smoothing, Theta, etc.)
with a *temporal* train/test split (no random shuffle).

Outputs:
  - metrics_json   : per-model metrics
  - manifest_json  : summary manifest
  - best_model     : folder with champion .pkl
  - outputs/model_breakdown_s05t.csv (per-model metric rows)
"""

import argparse
import json
import time as _time_mod
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.azureml_metrics_logger import (
    create_metrics_logger, ensure_outputs_dir, safe_write_json,
)
from utils.candidate_ledger import (
    make_row, normalize_metrics, write_candidate_artifacts,
    write_stage_table,
)
from utils.model_universe import get_forecasting_models

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ───────────────────────────────────────────────────────────────────────
# Forecasting model wrappers (statsmodels)
# ───────────────────────────────────────────────────────────────────────

def _fit_arima(train_y, order=(1, 1, 1)):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train_y, order=order)
    return model.fit()


def _fit_sarima(train_y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(train_y, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False, maxiter=50)


def _fit_exponential_smoothing(train_y, seasonal_periods=12):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(
        train_y, trend="add", seasonal="add",
        seasonal_periods=min(seasonal_periods, len(train_y) // 3) or 2,
    )
    return model.fit(optimized=True)


def _fit_ses(train_y):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    model = SimpleExpSmoothing(train_y)
    return model.fit(optimized=True)


def _fit_theta(train_y):
    from statsmodels.tsa.forecasting.theta import ThetaModel
    model = ThetaModel(train_y)
    return model.fit()


def _fit_naive(train_y, seasonal_period=1):
    """Seasonal naive: repeats the last seasonal_period values."""
    class _NaiveResult:
        def __init__(self, data, sp):
            self._data = data
            self._sp = max(sp, 1)
        def forecast(self, steps):
            cycle = np.array(self._data[-self._sp:])
            reps = int(np.ceil(steps / len(cycle)))
            return pd.Series(np.tile(cycle, reps)[:steps])
    return _NaiveResult(train_y.values, seasonal_period)


MODEL_HANDLERS = {
    "arima": _fit_arima,
    "sarima": _fit_sarima,
    "exponential_smoothing": _fit_exponential_smoothing,
    "ses": _fit_ses,
    "theta": _fit_theta,
    "naive": _fit_naive,
}


def _evaluate_forecast(actual, predicted):
    """Compute MAE, RMSE, MAPE for forecast evaluation."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else None
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 2) if mape else None}


# ───────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────

def main():
    _t0 = _time_mod.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_in", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--manifest_out", required=True)
    parser.add_argument("--model_out", required=True)
    args = parser.parse_args()

    print("=" * 80)
    print("STEP S05t: TIME-SERIES FORECASTING")
    print("=" * 80)

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    task_type = cfg.get("task_type", "classification")
    target_col = cfg.get("dataset", {}).get("target_column")
    delimiter = cfg.get("dataset", {}).get("delimiter", ",")

    outputs_dir = ensure_outputs_dir()
    model_dir = Path(args.model_out).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Check time-series signal from Stage 1 outputs/ ────────────────
    ts_signal_path = Path("outputs") / "time_series_detection.json"
    # Also check parent directory patterns used by Azure ML inter-step
    if not ts_signal_path.exists():
        # Try component input directory pattern
        for candidate in [
            Path(args.dataset_in).parent.parent / "s1" / "time_series_detection.json",
            Path(args.dataset_in).parent / "time_series_detection.json",
        ]:
            if candidate.exists():
                ts_signal_path = candidate
                break

    ts_detection = {}
    if ts_signal_path.exists():
        with open(ts_signal_path) as f:
            ts_detection = json.load(f)

    is_ts = ts_detection.get("is_time_series", False)

    if not is_ts:
        print("🕐 Time-series NOT detected — skipping forecasting step")
        skip_info = {"status": "skipped", "reason": "Dataset not detected as time-series"}
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(skip_info, f)
        with open(args.manifest_out, "w") as f:
            json.dump({"engine": "statsmodels", **skip_info}, f)
        (model_dir / ".skipped").write_text("Not a time-series dataset")
        safe_write_json(outputs_dir / "stage5t_skipped.json", skip_info)
        print("✅ STEP S05t SKIPPED (non time-series)")
        return

    # ── Load dataset and prepare temporal series ──────────────────────
    df = pd.read_csv(args.dataset_in, sep=delimiter)
    print(f"📊 Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")

    time_col = ts_detection.get("time_column")
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).dropna(subset=[time_col])
        df = df.set_index(time_col)
        freq = ts_detection.get("frequency")
        if freq:
            try:
                df.index.freq = freq
            except Exception:
                pass

    if not target_col or target_col not in df.columns:
        print(f"⚠️  Target column '{target_col}' not found — skipping forecasting")
        skip_info = {"status": "skipped", "reason": f"Target '{target_col}' missing"}
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(skip_info, f)
        with open(args.manifest_out, "w") as f:
            json.dump({"engine": "statsmodels", **skip_info}, f)
        (model_dir / ".skipped").write_text(f"Target column missing")
        return

    y = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if len(y) < 30:
        print(f"⚠️  Insufficient data for forecasting ({len(y)} points) — skipping")
        skip_info = {"status": "skipped", "reason": f"Too few data points ({len(y)})"}
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump(skip_info, f)
        with open(args.manifest_out, "w") as f:
            json.dump({"engine": "statsmodels", **skip_info}, f)
        return

    # Temporal train/test split (80/20 — NO shuffle)
    split_idx = int(len(y) * 0.8)
    train_y = y.iloc[:split_idx]
    test_y = y.iloc[split_idx:]
    horizon = len(test_y)
    print(f"   Train: {len(train_y):,} | Test: {len(test_y):,} (horizon={horizon})")

    # ── Run each forecasting model ────────────────────────────────────
    model_list = get_forecasting_models()
    results = []
    best_model_name = None
    best_rmse = float("inf")
    best_fitted = None

    breakdown_rows = []  # for model_breakdown.csv

    for model_name in model_list:
        handler = MODEL_HANDLERS.get(model_name)
        if handler is None:
            print(f"   ⚠️  No handler for '{model_name}' — skipping")
            continue

        t_start = _time_mod.time()
        try:
            fitted = handler(train_y)
            forecast = fitted.forecast(horizon)
            forecast_vals = np.asarray(forecast, dtype=float)[:len(test_y)]
            eval_metrics = _evaluate_forecast(test_y.values, forecast_vals)
            elapsed = round(_time_mod.time() - t_start, 2)

            entry = {"model": model_name, "fit_time_sec": elapsed, **eval_metrics, "status": "ok"}
            results.append(entry)

            breakdown_rows.append({
                "model_name": model_name,
                "engine": "statsmodels",
                "variant": "time_series_baseline",
                "stage": "s05t",
                "metric_name": "rmse",
                "metric_value": eval_metrics["rmse"],
                "mae": eval_metrics["mae"],
                "mape": eval_metrics.get("mape"),
                "fit_time_sec": elapsed,
                "status": "ok",
            })

            print(f"   ✅ {model_name:25s} RMSE={eval_metrics['rmse']:.4f}  MAE={eval_metrics['mae']:.4f}  ({elapsed:.1f}s)")

            if eval_metrics["rmse"] < best_rmse:
                best_rmse = eval_metrics["rmse"]
                best_model_name = model_name
                best_fitted = fitted
        except Exception as e:
            elapsed = round(_time_mod.time() - t_start, 2)
            results.append({"model": model_name, "status": "failed", "error": str(e), "fit_time_sec": elapsed})
            breakdown_rows.append({
                "model_name": model_name,
                "engine": "statsmodels",
                "variant": "time_series_baseline",
                "stage": "s05t",
                "metric_name": "rmse",
                "metric_value": None,
                "mae": None,
                "mape": None,
                "fit_time_sec": elapsed,
                "status": "failed",
            })
            print(f"   ❌ {model_name:25s} FAILED ({e}) ({elapsed:.1f}s)")

    # ── Outputs ───────────────────────────────────────────────────────
    metrics = {
        "engine": "statsmodels",
        "models_attempted": len(model_list),
        "models_succeeded": sum(1 for r in results if r["status"] == "ok"),
        "best_model": best_model_name,
        "best_rmse": round(best_rmse, 4) if best_model_name else None,
        "per_model": results,
    }
    manifest = {
        "engine": "statsmodels",
        "best_model": best_model_name,
        "best_metric_value": round(best_rmse, 4) if best_model_name else None,
        "metric_name": "rmse",
    }

    # Save champion model
    if best_fitted is not None:
        try:
            import joblib
            model_path = model_dir / "model.pkl"
            joblib.dump(best_fitted, model_path)
            print(f"\n💾 Champion model saved: {best_model_name} → {model_path}")
        except Exception as save_err:
            print(f"⚠️  Model save failed: {save_err}")

    # Write pipeline outputs
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, default=str, indent=2)
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f, default=str, indent=2)

    # Write model breakdown CSV
    if breakdown_rows:
        bd_df = pd.DataFrame(breakdown_rows)
        bd_df.to_csv(outputs_dir / "model_breakdown_s05t.csv", index=False)
        print(f"📊 Model breakdown: {len(breakdown_rows)} rows → outputs/model_breakdown_s05t.csv")

    # Write summary
    safe_write_json(outputs_dir / "stage5t_forecasting_summary.json", {
        "stage": "5t_forecasting",
        "engine": "statsmodels",
        "models_attempted": metrics["models_attempted"],
        "models_succeeded": metrics["models_succeeded"],
        "best_model": best_model_name,
        "best_rmse": metrics.get("best_rmse"),
        "dataset_shape": list(df.shape),
    })

    # MLflow logging
    logger = create_metrics_logger(
        run_name="s05t_forecasting",
        tags={"pipeline": "v3_mlops", "phase": "baseline", "step": "s05t"},
    )
    try:
        logger.log_param("task_type", task_type)
        logger.log_param("best_model", str(best_model_name))
        logger.log_metric("models_attempted", len(model_list))
        logger.log_metric("models_succeeded", sum(1 for r in results if r["status"] == "ok"))
        if best_rmse < float("inf"):
            logger.log_metric("best_rmse", round(best_rmse, 4))
    except Exception as mlflow_err:
        print(f"⚠️  MLflow logging failed (non-fatal): {mlflow_err}")
    logger.end_run()

    # ── Candidate Ledger ──────────────────────────────────────────────
    try:
        _elapsed = _time_mod.time() - _t0
        _norm = normalize_metrics(task_type, metrics)
        _status = "ok" if best_model_name else "failed"
        row = make_row(
            stage="baseline", step_name="s05t", engine="statsmodels",
            candidate_id=f"ts_{best_model_name or 'none'}",
            task_type=task_type,
            dataset_id=Path(args.dataset_in).stem,
            status=_status,
            failure_reason="" if best_model_name else "All models failed",
            compute_time_sec=round(_elapsed, 2),
            source_path="src/steps/stage5_timeseries_train.py",
            recipe_name="time_series_baseline",
            is_stage_best=True,
            **_norm,
        )
        write_candidate_artifacts(
            "outputs", row,
            inputs_dict={"engine": "statsmodels", "task_type": task_type},
            metrics_dict=metrics,
        )
        write_stage_table(
            [row],
            csv_path="outputs/s05t_candidates.csv",
            parquet_path="outputs/s05t_candidates.parquet",
        )
    except Exception as _ledger_err:
        print(f"⚠️  Candidate ledger write failed (non-fatal): {_ledger_err}")

    print("\n" + "=" * 80)
    print("✅ STEP S05t COMPLETE — Time-Series Forecasting")
    print("=" * 80)


if __name__ == "__main__":
    main()
