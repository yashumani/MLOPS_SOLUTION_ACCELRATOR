#!/usr/bin/env python
"""
validate_aim_tournament.py — Post-run validation for AIM-Tournament artifacts.

Usage::

    python scripts/validate_aim_tournament.py outputs/
    python scripts/validate_aim_tournament.py /path/to/job/outputs

Exits with code 0 on success, 1 on critical failures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Required metric columns per task type
REQUIRED_METRICS = {
    "classification": ["accuracy", "f1", "precision", "recall", "roc_auc"],
    "regression":     ["r2", "rmse", "mae"],
    "clustering":     ["silhouette", "davies_bouldin", "calinski_harabasz"],
}

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def _pass(msg: str) -> None:
    global CHECKS_PASSED
    CHECKS_PASSED += 1
    print(f"  ✅ {msg}")


def _fail(msg: str) -> None:
    global CHECKS_FAILED
    CHECKS_FAILED += 1
    print(f"  ❌ {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠️  {msg}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_aim_tournament.py <outputs_dir>")
        return 1

    out = Path(sys.argv[1])
    if not out.is_dir():
        print(f"ERROR: {out} is not a directory")
        return 1

    print("=" * 80)
    print("AIM-TOURNAMENT VALIDATION")
    print(f"Output directory: {out}")
    print("=" * 80)

    # ── 1. all_candidates.csv ─────────────────────────────────────────
    print("\n📋 1. Candidate Ledger")
    ledger_path = out / "all_candidates.csv"
    if not ledger_path.is_file():
        _fail("all_candidates.csv not found")
        # Can't continue without ledger
        _print_summary()
        return 1
    else:
        _pass("all_candidates.csv exists")

    import pandas as pd
    df = pd.read_csv(ledger_path)
    _pass(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Detect task type from data
    task_type = None
    if "task_type" in df.columns:
        task_type = df["task_type"].dropna().iloc[0] if len(df) > 0 else None
    if not task_type:
        # Infer from available metrics
        if "accuracy" in df.columns and df["accuracy"].notna().any():
            task_type = "classification"
        elif "r2" in df.columns and df["r2"].notna().any():
            task_type = "regression"
        elif "silhouette" in df.columns and df["silhouette"].notna().any():
            task_type = "clustering"
        else:
            _warn("Could not determine task_type — checking all metric types")
            task_type = "classification"  # default for validation
    _pass(f"Task type: {task_type}")

    # ── 2. Required metrics ───────────────────────────────────────────
    print(f"\n📊 2. Metrics ({task_type})")
    required = REQUIRED_METRICS.get(task_type, [])
    for metric in required:
        if metric in df.columns:
            non_null = df[metric].notna().sum()
            if non_null > 0:
                _pass(f"{metric}: {non_null}/{len(df)} non-null values")
            else:
                _fail(f"{metric}: column exists but all NULL")
        else:
            _fail(f"{metric}: column missing")

    # ── 3. Per-metric top-K tables ────────────────────────────────────
    print("\n📈 3. Per-metric Top-K Tables")
    topk_dir = out / "topk"
    if topk_dir.is_dir():
        topk_files = list(topk_dir.glob("top_*.csv"))
        if topk_files:
            _pass(f"{len(topk_files)} top-K files found")
            for f in sorted(topk_files):
                _pass(f"  {f.name}")
        else:
            _fail("topk/ directory exists but no top_*.csv files")
    else:
        _warn("topk/ directory not found (AIM-Tournament may not have run)")

    # ── 4. Pareto frontier ────────────────────────────────────────────
    print("\n🏆 4. Pareto Frontier")
    pareto_csv = out / "pareto_frontier.csv"
    pareto_json = out / "pareto_summary.json"

    if pareto_csv.is_file():
        pareto_df = pd.read_csv(pareto_csv)
        _pass(f"pareto_frontier.csv: {len(pareto_df)} Pareto-optimal candidates")
    else:
        _warn("pareto_frontier.csv not found")

    if pareto_json.is_file():
        with open(pareto_json) as f:
            ps = json.load(f)
        _pass(f"pareto_summary.json: {ps.get('pareto_size', '?')}/{ps.get('total_candidates', '?')} Pareto-optimal")
    else:
        _warn("pareto_summary.json not found")

    # ── 5. Ranked ledger ──────────────────────────────────────────────
    print("\n📊 5. Enriched / Ranked Ledger")
    ranked_path = out / "all_candidates_ranked.csv"
    if ranked_path.is_file():
        ranked_df = pd.read_csv(ranked_path)
        rank_cols = [c for c in ranked_df.columns if c.startswith("rank_")]
        _pass(f"all_candidates_ranked.csv: {len(ranked_df)} rows, {len(rank_cols)} rank columns")
        if "utility_score" in ranked_df.columns:
            _pass(f"utility_score present (max={ranked_df['utility_score'].max():.4f})")
        else:
            _warn("utility_score column not found")
        if "pareto_optimal" in ranked_df.columns:
            pareto_count = ranked_df["pareto_optimal"].sum()
            _pass(f"pareto_optimal column present ({int(pareto_count)} True)")
        else:
            _warn("pareto_optimal column not found")
    else:
        _warn("all_candidates_ranked.csv not found")

    # ── 6. Bundle gating signals ──────────────────────────────────────
    print("\n📡 6. Bundle Gating Signals")
    sig_dir = out / "signals"
    if sig_dir.is_dir():
        for fname in ("stage_signals.json", "bundle_decisions.json"):
            fp = sig_dir / fname
            if fp.is_file():
                _pass(f"{fname} exists")
                with open(fp) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    enabled = sum(1 for d in data if d.get("enabled"))
                    _pass(f"  {enabled}/{len(data)} bundles enabled")
                elif isinstance(data, dict):
                    _pass(f"  {len(data)} signals recorded")
            else:
                _warn(f"{fname} not found in signals/")
    else:
        _warn("signals/ directory not found (bundle gating may not have run)")

    # ── 7. Model Coverage ─────────────────────────────────────────────
    print("\n🔧 7. Model Coverage")
    coverage_path = out / "model_coverage.json"
    if coverage_path.is_file():
        with open(coverage_path) as f:
            cov = json.load(f)
        for engine, info in cov.get("engines", {}).items():
            _pass(f"{engine}: {info.get('available', '?')}/{info.get('total', '?')} available")
    else:
        _warn("model_coverage.json not found")

    # ── 8. Per-stage CSVs ─────────────────────────────────────────────
    print("\n📁 8. Per-stage Candidate CSVs")
    stage_csvs = sorted(out.glob("s*_candidates.csv"))
    if stage_csvs:
        _pass(f"{len(stage_csvs)} stage CSV files found")
        for sc in stage_csvs:
            stage_df = pd.read_csv(sc)
            _pass(f"  {sc.name}: {len(stage_df)} rows")
    else:
        _warn("No per-stage candidate CSVs found")

    # ── Summary ───────────────────────────────────────────────────────
    _print_summary()
    return 1 if CHECKS_FAILED > 0 else 0


def _print_summary() -> None:
    total = CHECKS_PASSED + CHECKS_FAILED
    print("\n" + "=" * 80)
    print(f"VALIDATION RESULT: {CHECKS_PASSED}/{total} checks passed, {CHECKS_FAILED} failed")
    if CHECKS_FAILED == 0:
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ SOME CHECKS FAILED")
    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
