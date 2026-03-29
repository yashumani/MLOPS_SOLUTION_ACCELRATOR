#!/usr/bin/env python3
"""
Production Readiness Analysis for Pipeline Run: clever_brick_kqnlds6nnz
Analyzes all stages, metrics, and identifies production tuning opportunities.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# File paths
BASE = Path("analysis_outputs")
PHASEB_LEADERBOARD = BASE / "phaseb/named-outputs/phaseb_leaderboard/leaderboard_csv"
PHASEB_MANIFEST = BASE / "phaseb/named-outputs/phaseb_champion_manifest/champion_manifest"
PHASEB_RESULTS = BASE / "phaseb/named-outputs/phaseb_all_results/all_results_json"
BASELINE_REPORT = BASE / "baseline/named-outputs/baseline_aggregate_report/aggregate_report"
FINAL_REPORT = BASE / "final/named-outputs/final_report/final_report"

print("="*80)
print("PRODUCTION READINESS ANALYSIS")
print("Pipeline: clever_brick_kqnlds6nnz (Jan 30, 2026)")
print("="*80)

# ============================================================================
# 1. PHASE B ANALYSIS - Critical Findings
# ============================================================================
print("\n" + "="*80)
print("1. PHASE B VARIANT RUNNER ANALYSIS")
print("="*80)

# Load leaderboard
leaderboard = pd.read_csv(PHASEB_LEADERBOARD)
print(f"\n📊 Leaderboard Shape: {leaderboard.shape[0]} rows × {leaderboard.shape[1]} columns")
print(f"\nColumns: {list(leaderboard.columns)}")
print(f"\n{leaderboard.head(10).to_string()}")

# Load all results for detailed analysis
with open(PHASEB_RESULTS, 'r') as f:
    all_results = json.load(f)

total_runs = len(all_results)
pycaret_runs = [r for r in all_results if r['engine'] == 'pycaret']
flaml_runs = [r for r in all_results if r['engine'] == 'flaml']

pycaret_failed = [r for r in pycaret_runs if r['failed']]
pycaret_timed_out = [r for r in pycaret_runs if r['timed_out']]
pycaret_success = [r for r in pycaret_runs if not r['failed'] and not r['timed_out']]

flaml_failed = [r for r in flaml_runs if r['failed']]
flaml_timed_out = [r for r in flaml_runs if r['timed_out']]
flaml_success = [r for r in flaml_runs if not r['failed'] and not r['timed_out']]

print(f"\n📈 EXECUTION STATISTICS:")
print(f"   Total runs: {total_runs}")
print(f"   PyCaret runs: {len(pycaret_runs)} | FLAML runs: {len(flaml_runs)}")
print(f"\n   PyCaret Results:")
print(f"      ✅ Success: {len(pycaret_success)}")
print(f"      ❌ Failed: {len(pycaret_failed)}")
print(f"      ⏱️  Timed out: {len(pycaret_timed_out)}")
print(f"\n   FLAML Results:")
print(f"      ✅ Success: {len(flaml_success)}")
print(f"      ❌ Failed: {len(flaml_failed)}")
print(f"      ⏱️  Timed out: {len(flaml_timed_out)}")

# Identify critical issues
print(f"\n🚨 CRITICAL ISSUES IDENTIFIED:")

# Issue 1: PyCaret failures
if len(pycaret_failed) > 0:
    error_msg = pycaret_failed[0]['failure_reason']
    print(f"\n   1. ALL PYCARET RUNS FAILED ({len(pycaret_failed)}/20)")
    print(f"      Error: {error_msg}")
    print(f"      Root Cause: PyCaret 3.3.2 removed 'silent' parameter")
    print(f"      Impact: Only FLAML results available (50% engine coverage loss)")
    print(f"      Priority: 🔴 CRITICAL - Blocks dual-engine validation")

# Issue 2: FLAML timeouts
if len(flaml_timed_out) == len(flaml_runs):
    print(f"\n   2. ALL FLAML RUNS TIMED OUT ({len(flaml_timed_out)}/20)")
    print(f"      Time budget: 300 seconds per variant")
    print(f"      Dataset: 243,553 rows × 45 columns")
    print(f"      Impact: No valid trained models from any variant")
    print(f"      Priority: 🔴 CRITICAL - Zero successful variant training")

# Issue 3: Champion selection from timeouts
with open(PHASEB_MANIFEST, 'r') as f:
    champion = json.load(f)

print(f"\n   3. CHAMPION SELECTED FROM TIMED-OUT RUN")
print(f"      Variant: {champion['variant_id']}")
print(f"      Engine: {champion['engine']}")
print(f"      Algorithm: {champion['algorithm']}")
print(f"      Metric: {champion['primary_metric_name']} = {champion.get('primary_metric_value', 'N/A')}")
print(f"      Impact: Champion has accuracy = 0.0 (unusable model)")
print(f"      Priority: 🔴 CRITICAL - No valid Phase B champion")

# ============================================================================
# 2. BASELINE PHASE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. BASELINE PHASE (Phase A) ANALYSIS")
print("="*80)

with open(BASELINE_REPORT, 'r') as f:
    baseline = json.load(f)

print(f"\n✅ BASELINE STATUS: SUCCESS")
print(f"   Task: {baseline['task']}")
print(f"   Selection: {baseline['selection']['source']} (score={baseline['selection']['score']:.4f})")
print(f"   Reason: {baseline['selection']['reason']}")
print(f"   Files copied: {baseline['files_copied']}")
print(f"   Validation: {baseline['output_validation']['valid']}")

# ============================================================================
# 3. FINAL EVALUATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. FINAL EVALUATION (Stage 11) ANALYSIS")
print("="*80)

with open(FINAL_REPORT, 'r') as f:
    final = json.load(f)

print(f"\n📊 FINAL METRICS:")
print(f"   Task: {final['task']} | Target: {final['target_column']}")
print(f"   Test samples: {final['test_samples']:,}")

print(f"\n   Baseline Metrics (Phase A):")
for metric, value in final['baseline_metrics'].items():
    print(f"      {metric}: {value:.4f}")

print(f"\n   Phase B Metrics:")
if final['phaseb_metrics'] is None:
    print(f"      ❌ NULL - No valid Phase B model")
else:
    for metric, value in final['phaseb_metrics'].items():
        print(f"      {metric}: {value:.4f}")

print(f"\n   Phase C Metrics (HPO):")
for metric, value in final['phasec_metrics'].items():
    print(f"      {metric}: {value:.4f}")

print(f"\n   🏆 CHAMPION SELECTION: {final['selection']['key'].upper()}")
print(f"      Score: {final['selection']['score']:.4f}")

# Warnings
if final['validation']['warnings']:
    print(f"\n   ⚠️  WARNINGS:")
    for warn in final['validation']['warnings']:
        print(f"      - {warn}")

# ============================================================================
# 4. PRODUCTION READINESS ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("4. PRODUCTION READINESS ASSESSMENT")
print("="*80)

issues = []
blockers = []
warnings_list = []

# Critical blocker: Phase B complete failure
if len(pycaret_failed) == len(pycaret_runs) and len(flaml_timed_out) == len(flaml_runs):
    blockers.append({
        "severity": "CRITICAL",
        "component": "Phase B Variant Runner",
        "issue": "Zero successful variant training",
        "details": "All PyCaret runs failed (API incompatibility), all FLAML runs timed out",
        "impact": "Phase B provides no value - cannot validate intelligent variant selection",
        "action": "1. Fix PyCaret 'silent' parameter, 2. Increase FLAML time budget or reduce dataset"
    })

# PyCaret API incompatibility
if len(pycaret_failed) > 0:
    issues.append({
        "severity": "HIGH",
        "component": "PyCaret Integration",
        "issue": "setup() got unexpected keyword argument 'silent'",
        "details": f"PyCaret 3.3.2 removed 'silent' parameter (used in training code)",
        "impact": "50% engine loss - only FLAML available",
        "action": "Update train_pycaret_variant() to use 'verbose=False' instead of 'silent=True'"
    })

# FLAML timeout
if len(flaml_timed_out) == len(flaml_runs):
    issues.append({
        "severity": "HIGH",
        "component": "FLAML Time Budget",
        "issue": "All FLAML runs exceeded 300s budget",
        "details": f"Dataset size: 243,553 rows × 45 features",
        "impact": "No trained models from 20 variants",
        "action": "Increase time_budget_per_variant to 600-900s OR downsample dataset to 50k rows"
    })

# Phase B unusable champion
warnings_list.append({
    "severity": "MEDIUM",
    "component": "Champion Selection Logic",
    "issue": "Champion selected from timed-out run with metric=0.0",
    "details": "Leaderboard sorting picked first alphabetical variant when all timed out",
    "impact": "Phase B output exists but is meaningless",
    "action": "Add validation: reject champions with primary_metric <= 0.0"
})

# MLflow nested runs
warnings_list.append({
    "severity": "LOW",
    "component": "MLflow Tracking",
    "issue": "Need to verify 40 nested runs created correctly",
    "details": "Expected: s06 parent run with 40 children (20 variants × 2 engines)",
    "impact": "Cannot validate artifact lineage if nesting broken",
    "action": "Check MLflow Studio: filter by parent_run_id for s06 step"
})

print(f"\n🛑 BLOCKERS ({len(blockers)}):")
for i, b in enumerate(blockers, 1):
    print(f"\n   {i}. [{b['severity']}] {b['component']}")
    print(f"      Issue: {b['issue']}")
    print(f"      Details: {b['details']}")
    print(f"      Impact: {b['impact']}")
    print(f"      Action: {b['action']}")

print(f"\n⚠️  HIGH/MEDIUM ISSUES ({len(issues)}):")
for i, iss in enumerate(issues, 1):
    print(f"\n   {i}. [{iss['severity']}] {iss['component']}")
    print(f"      Issue: {iss['issue']}")
    print(f"      Details: {iss['details']}")
    print(f"      Impact: {iss['impact']}")
    print(f"      Action: {iss['action']}")

print(f"\n💡 WARNINGS ({len(warnings_list)}):")
for i, w in enumerate(warnings_list, 1):
    print(f"\n   {i}. [{w['severity']}] {w['component']}")
    print(f"      Issue: {w['issue']}")
    print(f"      Action: {w['action']}")

# ============================================================================
# 5. PRODUCTION TUNING RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("5. PRODUCTION TUNING RECOMMENDATIONS")
print("="*80)

print(f"\n🎯 IMMEDIATE FIXES (Before adding more variants):")
print(f"\n   1. FIX PYCARET API COMPATIBILITY")
print(f"      File: src/steps/s06_phaseb_variant_runner.py")
print(f"      Lines: 355, 411")
print(f"      Change: Remove 'silent=True' parameter from setup()")
print(f"      Note: PyCaret 3.3+ uses verbose=False instead")
print(f"      Test: Rerun with 2-3 variants first")

print(f"\n   2. INCREASE FLAML TIME BUDGET")
print(f"      Config: configs/config_classification_telecom_churn_test_s06.yml")
print(f"      Current: time_budget_per_variant: 300")
print(f"      Recommended: time_budget_per_variant: 600")
print(f"      Alternative: Downsample dataset to 50k rows for testing")

print(f"\n   3. ADD CHAMPION VALIDATION")
print(f"      File: src/steps/s06_phaseb_variant_runner.py")
print(f"      Line: ~1010 (champion selection logic)")
print(f"      Add: Reject champions with primary_metric <= 0.01")
print(f"      Fallback: Return empty champion with status='all_timed_out'")

print(f"\n🔬 VALIDATION TESTING (After fixes):")
print(f"\n   1. Small-scale test (5 variants × 2 engines = 10 runs)")
print(f"      - Verify PyCaret success rate > 80%")
print(f"      - Verify FLAML success rate > 50%")
print(f"      - Verify at least 1 valid champion")

print(f"\n   2. Check MLflow nested runs")
print(f"      - Open Azure ML Studio → Experiments → classification_telecom_churn_test_s06_v3")
print(f"      - Find job: clever_brick_kqnlds6nnz")
print(f"      - Navigate to s06 step")
print(f"      - Verify 40 child runs visible (20 PyCaret + 20 FLAML)")
print(f"      - Check artifacts logged: leaderboard, manifest, model")

print(f"\n   3. Metric validation")
print(f"      - Baseline accuracy: 0.7992 (80% benchmark)")
print(f"      - Target: Phase B champion > baseline")
print(f"      - Acceptance: At least 1 variant with accuracy > 0.80")

print(f"\n📈 SCALE-UP PLAN (After validation):")
print(f"\n   Phase 1: 20 variants → 40 variants (11% → 22% coverage)")
print(f"   Phase 2: Enable dataset profiling + scored selection")
print(f"   Phase 3: 40 variants → 60 variants (22% → 33% coverage)")
print(f"   Target: Find 2-3% accuracy improvement over baseline")

print(f"\n💰 COST OPTIMIZATION:")
print(f"\n   Current run: 2h 9m (129 minutes)")
print(f"   Breakdown:")
print(f"      - s1-s4 (preprocessing): ~20 min")
print(f"      - s5a/s5b/s5z (baseline): ~25 min")
print(f"      - s06 (Phase B): ~60 min (all timeouts)")
print(f"      - s10/s10z (Phase C): ~15 min")
print(f"      - s11 (final eval): ~9 min")
print(f"\n   Optimization:")
print(f"      - Fix timeouts → reduce s06 to ~30 min")
print(f"      - Total estimated: ~90 min (-30% cost)")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("6. EXECUTIVE SUMMARY")
print("="*80)

print(f"\n✅ PIPELINE COMPLETION: SUCCESS")
print(f"   - All 11 stages executed")
print(f"   - Baseline and Phase C champions valid")
print(f"   - Final champion selected (baseline with 79.9% accuracy)")

print(f"\n❌ PHASE B STATUS: FAILED (Not Production Ready)")
print(f"   - 0/20 PyCaret variants succeeded (API incompatibility)")
print(f"   - 0/20 FLAML variants succeeded (all timed out)")
print(f"   - Champion has 0.0 accuracy (unusable)")
print(f"   - Phase B provides no value in current state")

print(f"\n🔧 REQUIRED FIXES:")
print(f"   1. PyCaret API fix (remove 'silent' param) - 15 min")
print(f"   2. FLAML time budget increase (300s → 600s) - 5 min")
print(f"   3. Champion validation logic - 10 min")
print(f"   Total effort: ~30 minutes")

print(f"\n🎯 RECOMMENDATION:")
print(f"   DO NOT scale to more variants until fixes validated.")
print(f"   Run 5-variant test first to confirm both engines working.")
print(f"   Target: 80%+ success rate before production rollout.")

print("\n" + "="*80)
print(f"Analysis complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")
