#!/usr/bin/env python3
"""Phase 0 — Local end-to-end drift detection test.

Downloads the real telecom-churn dataset from Azure ML, splits it into
reference / current, injects synthetic drift at three levels (none, mild,
severe), runs all four drift checks, generates HTML reports, and prints
a signal summary.

Usage
-----
    cd mlops-solution-accelerator-v3
    python scripts/phase0_local_drift_test.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure src/ is on the import path ───────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_detection import (
    BaselineCapture,
    DriftChecker,
    DriftConfig,
    DriftResult,
    PipelineTrigger,
    ReportGenerator,
    generate_drifted_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase0")

# ── Constants ───────────────────────────────────────────────────
DATASET_URI = (
    "azureml://subscriptions/93044a08-5661-4f1b-b424-5eafe066a9d1"
    "/resourcegroups/mvpv1/workspaces/mlops-accelerator"
    "/datastores/mlops_blob/paths/telecom_churn.csv"
)
TARGET_COLUMN = "churn"
PREDICTION_COLUMN = "prediction"
# Use a manageable subset so local runs stay fast
SAMPLE_ROWS = 10_000
# Columns to drop before drift analysis (IDs, dates)
DROP_COLUMNS = ["customer_id", "date_of_registration"]


# ── Helpers ─────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    """Load the telecom-churn dataset.

    Tries the Azure ML datastore URI first (requires azureml-fsspec).
    Falls back to a local CSV if available.
    """
    local_cache = ROOT / "outputs" / "telecom_churn_cache.csv"

    if local_cache.exists():
        logger.info("Loading cached dataset from %s", local_cache)
        return pd.read_csv(local_cache)

    logger.info("Downloading dataset from Azure ML datastore …")
    try:
        df = pd.read_csv(DATASET_URI)
    except Exception as exc:
        logger.error("Failed to download from Azure ML: %s", exc)
        logger.info(
            "Tip: ensure azureml-fsspec is installed and you are logged in "
            "(az login or DefaultAzureCredential)."
        )
        raise

    # Cache locally so reruns are instant
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(local_cache, index=False)
    logger.info("Cached %d rows → %s", len(df), local_cache)
    return df


def add_synthetic_predictions(
    df: pd.DataFrame, accuracy: float = 0.85, seed: int = 42
) -> pd.DataFrame:
    """Add a fake ``prediction`` column that matches the target with
    ~``accuracy`` probability."""
    rng = np.random.default_rng(seed)
    preds = df[TARGET_COLUMN].copy().values
    n_flip = int(len(df) * (1 - accuracy))
    flip_idx = rng.choice(len(df), size=n_flip, replace=False)
    labels = df[TARGET_COLUMN].unique()
    for idx in flip_idx:
        current = preds[idx]
        preds[idx] = rng.choice([l for l in labels if l != current])
    df = df.copy()
    df[PREDICTION_COLUMN] = preds
    return df


def print_results_table(
    scenario: str, results: list[DriftResult], trigger_summary: dict
) -> None:
    """Pretty-print a drift-check summary for one scenario."""
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  SCENARIO: {scenario}")
    print(sep)
    for r in results:
        flag = "🔴 DRIFT" if r.drift_detected else "🟢 OK"
        cols = ", ".join(r.drifted_columns[:5]) if r.drifted_columns else "—"
        if len(r.drifted_columns) > 5:
            cols += f" (+{len(r.drifted_columns) - 5} more)"
        print(f"  {r.drift_type:<12}  {flag:<12}  score={r.drift_score:<10.6f}  cols={cols}")
        if r.details:
            for k, v in list(r.details.items())[:3]:
                if isinstance(v, dict):
                    continue  # skip full per-column dicts
                print(f"                  {k}: {v}")

    trig = trigger_summary
    action_str = "TRIGGER" if trig["should_trigger"] else "no trigger"
    print(f"\n  Pipeline decision: {action_str}")
    if trig["triggered_by"]:
        print(f"  Triggered by:      {', '.join(trig['triggered_by'])}")
    print(sep)


# ── Main ────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 72)
    print("  PHASE 0 — Local Drift Detection Test")
    print("=" * 72)

    # 1. Load data
    full_df = load_dataset()
    logger.info("Full dataset: %d rows × %d cols", *full_df.shape)

    # Drop ID / date columns that shouldn't be features
    for col in DROP_COLUMNS:
        if col in full_df.columns:
            full_df = full_df.drop(columns=[col])

    # 2. Sample and split: 60% reference, 40% current
    sample = full_df.sample(n=SAMPLE_ROWS, random_state=42).reset_index(drop=True)
    split = int(len(sample) * 0.6)
    reference_df = sample.iloc[:split].reset_index(drop=True)
    current_base = sample.iloc[split:].reset_index(drop=True)

    logger.info("Reference: %d rows | Current base: %d rows", len(reference_df), len(current_base))

    # 3. Add synthetic predictions to both sets
    reference_df = add_synthetic_predictions(reference_df, accuracy=0.85, seed=100)
    current_base = add_synthetic_predictions(current_base, accuracy=0.85, seed=200)

    # 4. Build config (override target_column)
    cfg = DriftConfig.from_yaml(str(ROOT / "configs" / "drift_config.yaml"))
    cfg.column_mapping.target_column = TARGET_COLUMN
    cfg.column_mapping.prediction_column = PREDICTION_COLUMN

    # Use a temp directory for artefacts
    out_root = ROOT / "outputs" / "phase0_drift_test"
    out_root.mkdir(parents=True, exist_ok=True)
    cfg.artifact_paths.baseline_dir = str(out_root / "baseline")
    cfg.artifact_paths.reports_dir = str(out_root / "reports")
    cfg.artifact_paths.logs_dir = str(out_root / "logs")

    # 5. Capture baseline
    logger.info("Capturing baseline …")
    bc = BaselineCapture(cfg)
    bc.capture(reference_df)
    baseline_dir = bc.save()
    logger.info("Baseline saved → %s", baseline_dir)

    # Reload from disk to exercise the load path
    bc = BaselineCapture.load(cfg)

    # 6. Run three scenarios
    report_gen = ReportGenerator(cfg)
    trigger = PipelineTrigger(cfg, dry_run=True)

    scenarios = {
        "NO DRIFT (raw current data)": current_base,
        "MILD DRIFT (synthetic)": generate_drifted_data(
            current_base,
            drift_level="mild",
            seed=300,
            target_column=TARGET_COLUMN,
            prediction_column=PREDICTION_COLUMN,
        ),
        "SEVERE DRIFT (synthetic)": generate_drifted_data(
            current_base,
            drift_level="severe",
            seed=400,
            target_column=TARGET_COLUMN,
            prediction_column=PREDICTION_COLUMN,
        ),
    }

    for name, cur_df in scenarios.items():
        logger.info("Running drift checks — %s …", name)
        checker = DriftChecker(cfg, bc)
        results = checker.run_all_checks(cur_df)

        # Generate HTML report
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        report_path = report_gen.generate(
            reference_df, cur_df, report_name=f"phase0_{safe_name}"
        )
        logger.info("Report → %s", report_path)

        # Evaluate trigger
        summary = trigger.evaluate(results)

        # Display
        print_results_table(name, results, summary)

    # 7. Save trigger log
    log_path = trigger.save_trigger_log()
    logger.info("Trigger log → %s", log_path)

    # 8. Final summary
    print("\n" + "=" * 72)
    print("  ARTIFACTS")
    print("=" * 72)
    for p in sorted(out_root.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"  {p.relative_to(ROOT)}  ({size_kb:.1f} KB)")
    print("=" * 72)
    print("  Phase 0 complete. Review the HTML reports for visual detail.")
    print("=" * 72)


if __name__ == "__main__":
    main()
