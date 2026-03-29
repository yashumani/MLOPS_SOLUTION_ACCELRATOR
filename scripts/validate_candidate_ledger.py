#!/usr/bin/env python3
"""
Validate Candidate Ledger — Post-run integrity checker.

Usage:
    python scripts/validate_candidate_ledger.py outputs/
    python scripts/validate_candidate_ledger.py /path/to/azure/ml/run/outputs/

Checks:
  1. all_candidates.csv exists and is non-empty
  2. Every expected column is present
  3. At least one row per expected stage
  4. Primary metric columns are numeric
  5. Champion flag is set exactly once in the final stage
  6. Stage-level CSVs (s05a, s05b, …, s10) exist
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Canonical columns (copied from candidate_ledger.py)
IDENTITY_COLS = [
    "dataset_id", "task_type", "preset", "pipeline_version",
    "stage", "step_name", "engine", "candidate_id", "run_id", "timestamp_utc",
]
INPUT_COLS = [
    "recipe_name", "recipe_hash", "params_json", "pipeline_dims_json",
]
OUTPUT_COLS = [
    "primary_metric_name", "primary_metric_value",
    "accuracy", "roc_auc", "f1", "precision", "recall", "logloss",
    "rmse", "mae", "r2", "mse",
    "silhouette", "davies_bouldin", "calinski_harabasz",
]
SIGNAL_COLS = [
    "candidate_rank", "delta_vs_baseline_best",
    "is_stage_best", "is_final_champion",
    "status", "failure_reason", "compute_time_sec",
]
PROVENANCE_COLS = [
    "source_path", "artifacts_json",
]
ALL_COLUMNS = IDENTITY_COLS + INPUT_COLS + OUTPUT_COLS + SIGNAL_COLS + PROVENANCE_COLS

EXPECTED_STAGES = {"baseline", "phase_b", "phase_c", "final"}

STAGE_CSV_PREFIXES = [
    "s05a", "s05b", "s05z",
    "s06", "s07z",
    "s08", "s09",
    "s10",
]


def validate(root: Path) -> int:
    """Run all checks. Returns number of failures."""
    errors = 0

    # --- 1. Check merged ledger exists ---
    merged = root / "all_candidates.csv"
    if not merged.exists():
        print(f"❌ FAIL  all_candidates.csv not found in {root}")
        errors += 1
        # Can still check stage CSVs
    else:
        df = pd.read_csv(merged)
        n = len(df)
        print(f"✅ all_candidates.csv found: {n} rows, {len(df.columns)} columns")

        # --- 2. Column check ---
        missing_cols = [c for c in ALL_COLUMNS if c not in df.columns]
        if missing_cols:
            print(f"❌ FAIL  Missing columns: {missing_cols}")
            errors += 1
        else:
            print(f"✅ All {len(ALL_COLUMNS)} expected columns present")

        # --- 3. Stage coverage ---
        stages_found = set(df["stage"].dropna().unique()) if "stage" in df.columns else set()
        missing_stages = EXPECTED_STAGES - stages_found
        if missing_stages:
            print(f"⚠️  WARN  Missing stages in merged ledger: {missing_stages}")
        else:
            print(f"✅ All expected stages present: {stages_found}")

        # --- 4. Numeric primary metric ---
        if "primary_metric_value" in df.columns:
            numeric_vals = pd.to_numeric(df["primary_metric_value"], errors="coerce")
            null_count = numeric_vals.isna().sum()
            if null_count == n:
                print(f"❌ FAIL  primary_metric_value is all null/non-numeric")
                errors += 1
            elif null_count > 0:
                print(f"⚠️  WARN  primary_metric_value has {null_count}/{n} null/non-numeric rows")
            else:
                print(f"✅ primary_metric_value is fully numeric ({n} values)")

        # --- 5. Champion flag ---
        if "is_final_champion" in df.columns:
            champ_rows = df[df["is_final_champion"] == True]  # noqa: E712
            if len(champ_rows) == 0:
                print(f"⚠️  WARN  No row has is_final_champion=True")
            elif len(champ_rows) == 1:
                print(f"✅ Exactly 1 champion flagged: {champ_rows.iloc[0].get('candidate_id', '?')}")
            else:
                print(f"⚠️  WARN  Multiple champion flags ({len(champ_rows)} rows)")

        # --- 6. Per-stage row counts ---
        print(f"\n{'─'*50}")
        print("Stage row counts:")
        for stage in sorted(stages_found):
            count = len(df[df["stage"] == stage])
            print(f"  {stage:20s}  {count:>4} rows")
        print(f"{'─'*50}")

    # --- 7. Stage-level CSVs ---
    print("\nStage-level CSV files:")
    for prefix in STAGE_CSV_PREFIXES:
        csv = root / f"{prefix}_candidates.csv"
        if csv.exists():
            rows = sum(1 for _ in open(csv)) - 1
            print(f"  ✅ {csv.name:30s}  {rows:>4} rows")
        else:
            print(f"  ⚠️  {prefix}_candidates.csv not found")

    # --- Summary files ---
    for fname in ["all_candidates_summary.json", "all_candidates_README.md", "all_candidates.parquet"]:
        fp = root / fname
        if fp.exists():
            print(f"  ✅ {fname:30s}  {fp.stat().st_size:>8,} bytes")
        else:
            print(f"  ⚠️  {fname} not found")

    # Final verdict
    print()
    if errors:
        print(f"❌ VALIDATION FAILED ({errors} error(s))")
    else:
        print(f"✅ VALIDATION PASSED")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate candidate ledger outputs")
    parser.add_argument("outputs_dir", help="Path to outputs/ folder from pipeline run")
    args = parser.parse_args()

    root = Path(args.outputs_dir)
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        sys.exit(1)

    sys.exit(validate(root))


if __name__ == "__main__":
    main()
