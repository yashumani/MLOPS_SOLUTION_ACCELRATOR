#!/usr/bin/env python3
"""Cross-job drift analysis — compare all Azure ML pipeline submissions.

Downloads ``dataset_processed`` and ``final_report`` from each completed
pipeline job, then runs Evidently-based drift detection comparing every
job against the earliest baseline within the same experiment group.

Outputs:
  outputs/drift_analysis/<experiment>/<job_name>/
    ├── drift_report.html          – Evidently full-feature drift report
    ├── drift_results.json         – per-check scores & drifted columns
    └── final_metrics.json         – model metrics from the job
  outputs/drift_analysis/consolidated_report.md  – summary across all jobs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.metrics import ColumnDriftMetric, DatasetDriftMetric
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

# Add project root to path for src.utils imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.drift_detector import (
    compute_feature_psi,
    compute_feature_volatility,
    compute_stability_score,
    determine_retraining_cadence,
)

# ──────────────────────────────────────────────────────────────────
# Configuration — 15 jobs spread across 3 experiment types
# ──────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _azure_ctx import load_azure_context, MissingAzureContextError  # noqa: E402

try:
    _ctx = load_azure_context()
except MissingAzureContextError as _exc:
    print(f"❌ {_exc}", file=sys.stderr)
    sys.exit(2)
AZURE_SUB = _ctx.subscription_id
AZURE_RG = _ctx.resource_group
AZURE_WS = _ctx.workspace_name

# 5 classification jobs spread across Jan–Mar 2026
CLASSIFICATION_JOBS = [
    ("sharp_milk_jcq7vlpb7w", "2026-01-25"),     # baseline
    ("mango_brick_r4807w402c", "2026-02-09"),
    ("coral_onion_d3cg9dzz7s", "2026-02-20"),
    ("wheat_feijoa_7458ytkdyq", "2026-02-26"),
    ("sincere_turnip_lb7t1dsgt0", "2026-03-07"),
]

# 5 regression jobs spread across Feb 2026
REGRESSION_JOBS = [
    ("goofy_queen_ml0g2ss4sh", "2026-02-09"),     # baseline
    ("careful_spider_kf2hnl0l6k", "2026-02-20"),
    ("olden_lime_ljvr8fc3h4", "2026-02-23"),
    ("brave_lamp_z0v94xckc9", "2026-02-26"),
    ("musing_melon_1swbglz9m1", "2026-02-25"),
]

# 5 clustering jobs spread across Feb 2026
CLUSTERING_JOBS = [
    ("olden_lunch_y4150mstjz", "2026-02-09"),     # baseline
    ("lemon_yam_b8q0m2qdkj", "2026-02-20"),
    ("busy_knot_l55v4y039p", "2026-02-23"),
    ("good_ocean_bnpbgd42bq", "2026-02-25"),
    ("keen_yak_2v8ptp3kg4", "2026-02-26"),
]

EXPERIMENT_GROUPS = {
    "classification_telecom_churn": {
        "jobs": CLASSIFICATION_JOBS,
        "target_column": "churn",
        "task_type": "classification",
        "primary_metric": "accuracy",
    },
    "regression_college": {
        "jobs": REGRESSION_JOBS,
        "target_column": None,  # determined from final_report
        "task_type": "regression",
        "primary_metric": "r2",
    },
    "clustering_online_retail": {
        "jobs": CLUSTERING_JOBS,
        "target_column": None,  # clustering may not have one
        "task_type": "clustering",
        "primary_metric": "silhouette_score",
    },
}

# Staging directory for large downloads (avoid filling workspace)
STAGING_DIR = Path("/tmp/drift_staging")
# Final output directory inside workspace
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "drift_analysis"

logger = logging.getLogger("cross_job_drift")


# ──────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────

@dataclass
class JobDriftResult:
    job_name: str
    experiment: str
    job_date: str
    is_baseline: bool = False
    feature_drift_detected: bool = False
    feature_drift_score: float = 0.0
    feature_drifted_columns: List[str] = field(default_factory=list)
    feature_drift_share: float = 0.0
    label_drift_detected: bool = False
    label_drift_score: float = 0.0
    concept_drift_detected: bool = False
    concept_drift_score: float = 0.0
    baseline_metric: float = 0.0
    current_metric: float = 0.0
    primary_metric_name: str = "accuracy"
    n_rows: int = 0
    n_columns: int = 0
    report_path: str = ""
    error: str = ""
    # Cadence / stability fields
    stability_score: int = 0
    stability_components: dict = field(default_factory=dict)
    cadence_name: str = ""
    cadence_days: int = 0
    cadence_rationale: str = ""


# ──────────────────────────────────────────────────────────────────
# Azure ML download helpers
# ──────────────────────────────────────────────────────────────────

def get_ml_client():
    """Lazy-load Azure ML client."""
    from azure.ai.ml import MLClient
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        ManagedIdentityCredential,
    )
    return MLClient(
        ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential()),
        AZURE_SUB, AZURE_RG, AZURE_WS,
    )


def download_job_output(ml_client, job_name: str, output_name: str,
                        staging: Path) -> Optional[Path]:
    """Download a single named output from a pipeline job.

    Returns the path to the downloaded file/directory, or None on error.
    """
    dest = staging / job_name / output_name
    if dest.exists():
        # Already downloaded in a previous run — reuse
        logger.info("  Reusing cached %s/%s", job_name, output_name)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        try:
            ml_client.jobs.download(
                job_name,
                download_path=str(dest),
                output_name=output_name,
            )
        except Exception as exc:
            logger.warning("  Failed to download %s/%s: %s",
                           job_name, output_name, str(exc)[:200])
            return None

    # Find the actual file inside the download tree
    for root, _dirs, files in os.walk(dest):
        for f in files:
            return Path(root) / f
    return None


def load_dataset(path: Path, max_rows: int = 50_000) -> pd.DataFrame:
    """Read CSV or Parquet, sampling if too large."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if len(df) > max_rows:
        logger.info("  Sampling %d → %d rows", len(df), max_rows)
        df = df.sample(n=max_rows, random_state=42)
    return df


# ──────────────────────────────────────────────────────────────────
# Drift checks
# ──────────────────────────────────────────────────────────────────

def run_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                      target_col: Optional[str],
                      output_dir: Path) -> Tuple[bool, float, float, List[str], str]:
    """Run Evidently DataDriftPreset and DatasetDriftMetric.

    Returns (detected, dataset_drift_share, dataset_drift_score,
             drifted_cols, report_path).
    """
    col_mapping = ColumnMapping(target=target_col)

    # Align columns (some jobs may have slightly different one-hot sets)
    shared_cols = sorted(set(reference_df.columns) & set(current_df.columns))
    ref = reference_df[shared_cols].copy()
    cur = current_df[shared_cols].copy()

    report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
    report.run(reference_data=ref, current_data=cur, column_mapping=col_mapping)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "drift_report.html"
    report.save_html(str(report_path))

    # Extract results
    result_dict = report.as_dict()
    metrics = result_dict.get("metrics", [])

    # DataDriftPreset results
    drifted_cols = []
    share = 0.0
    dataset_drift = False
    for m in metrics:
        mr = m.get("result", {})
        if "drift_by_columns" in mr:
            for col, info in mr["drift_by_columns"].items():
                if info.get("drift_detected", False):
                    drifted_cols.append(col)
            n_cols = mr.get("number_of_columns", 1)
            n_drifted = mr.get("number_of_drifted_columns", 0)
            share = n_drifted / max(n_cols, 1)
        if "dataset_drift" in mr:
            dataset_drift = mr["dataset_drift"]

    return dataset_drift, share, share, drifted_cols, str(report_path)


def run_label_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                    target_col: str) -> Tuple[bool, float]:
    """Check distribution drift on the target column using chi-squared."""
    if target_col not in reference_df.columns or target_col not in current_df.columns:
        return False, 0.0

    col_mapping = ColumnMapping(target=target_col)
    report = Report(metrics=[ColumnDriftMetric(column_name=target_col)])
    report.run(
        reference_data=reference_df[[target_col]],
        current_data=current_df[[target_col]],
        column_mapping=col_mapping,
    )
    result_dict = report.as_dict()
    for m in result_dict.get("metrics", []):
        mr = m.get("result", {})
        if "drift_detected" in mr:
            return mr["drift_detected"], mr.get("drift_score", 0.0)
    return False, 0.0


def run_concept_drift(baseline_metric: float, current_metric: float,
                      threshold: float = 0.05) -> Tuple[bool, float]:
    """Concept drift = significant drop in primary metric."""
    drop = baseline_metric - current_metric
    detected = drop > threshold
    return detected, drop


# ──────────────────────────────────────────────────────────────────
# Per-job analysis
# ──────────────────────────────────────────────────────────────────

def analyze_job(
    ml_client,
    job_name: str,
    job_date: str,
    experiment: str,
    baseline_df: Optional[pd.DataFrame],
    baseline_metric: float,
    group_config: dict,
    output_dir: Path,
) -> JobDriftResult:
    """Download outputs and run drift analysis for a single job."""

    result = JobDriftResult(
        job_name=job_name,
        experiment=experiment,
        job_date=job_date,
        primary_metric_name=group_config["primary_metric"],
    )

    # ── Download dataset_processed ──────────────────────────────
    logger.info("  Downloading dataset_processed…")
    data_path = download_job_output(ml_client, job_name, "dataset_processed", STAGING_DIR)
    if data_path is None:
        result.error = "Failed to download dataset_processed"
        logger.error("  %s", result.error)
        return result

    try:
        current_df = load_dataset(data_path)
    except Exception as exc:
        result.error = f"Failed to load dataset: {exc}"
        logger.error("  %s", result.error)
        return result

    result.n_rows = len(current_df)
    result.n_columns = current_df.shape[1]

    # ── Download final_report for metrics ───────────────────────
    logger.info("  Downloading final_report…")
    report_path = download_job_output(ml_client, job_name, "final_report", STAGING_DIR)
    metrics_json = {}
    if report_path:
        try:
            with open(report_path) as fh:
                metrics_json = json.load(fh)
        except Exception:
            logger.warning("  Could not parse final_report JSON")

    # Extract primary metric from whatever phase is best
    metric_val = 0.0
    primary = group_config["primary_metric"]
    for phase_key in ["phasec_metrics", "phaseb_metrics", "baseline_metrics"]:
        phase = metrics_json.get(phase_key) or {}
        if isinstance(phase, dict):
            if primary in phase:
                metric_val = phase[primary]
                if metric_val > 0:
                    break
            # Fallback: try selection.score for clustering
            elif not metric_val and phase:
                # Take first numeric value as proxy
                for v in phase.values():
                    if isinstance(v, (int, float)) and v > 0:
                        metric_val = v
                        break
    # Also check selection.score (often set for clustering)
    if not metric_val:
        sel = metrics_json.get("selection", {})
        if isinstance(sel, dict) and "score" in sel:
            metric_val = sel["score"]
    result.current_metric = metric_val

    # Also try to detect target column from final_report
    target_col = group_config.get("target_column")
    if not target_col and "target_column" in metrics_json:
        target_col = metrics_json["target_column"]

    # ── Is this the baseline? ───────────────────────────────────
    if baseline_df is None:
        result.is_baseline = True
        result.baseline_metric = metric_val
        result.current_metric = metric_val
        logger.info("  ✅ Baseline established (%d rows, metric=%.4f)", len(current_df), metric_val)

        # Save the raw data as baseline reference file
        output_dir.mkdir(parents=True, exist_ok=True)
        baseline_info = {
            "job_name": job_name, "date": job_date,
            "n_rows": len(current_df), "n_columns": current_df.shape[1],
            "primary_metric": metric_val, "target_column": target_col,
        }
        with open(output_dir / "drift_results.json", "w") as fh:
            json.dump({"is_baseline": True, "info": baseline_info}, fh, indent=2)
        result.report_path = str(output_dir / "drift_results.json")
        if metrics_json:
            with open(output_dir / "final_metrics.json", "w") as fh:
                json.dump(metrics_json, fh, indent=2)
        return result

    # ── Run drift checks ────────────────────────────────────────
    result.baseline_metric = baseline_metric

    # 1. Feature drift
    logger.info("  Running feature drift check…")
    try:
        (
            result.feature_drift_detected,
            result.feature_drift_share,
            result.feature_drift_score,
            result.feature_drifted_columns,
            html_path,
        ) = run_feature_drift(baseline_df, current_df, target_col, output_dir)
        result.report_path = html_path
        logger.info("    Feature drift: detected=%s, share=%.2f%%, drifted_cols=%d",
                     result.feature_drift_detected, result.feature_drift_share * 100,
                     len(result.feature_drifted_columns))
    except Exception as exc:
        logger.error("    Feature drift error: %s", exc)
        result.error += f"Feature drift error: {exc}; "

    # 2. Label drift
    if target_col and target_col in current_df.columns:
        logger.info("  Running label drift check…")
        try:
            result.label_drift_detected, result.label_drift_score = \
                run_label_drift(baseline_df, current_df, target_col)
            logger.info("    Label drift: detected=%s, score=%.4f",
                         result.label_drift_detected, result.label_drift_score)
        except Exception as exc:
            logger.error("    Label drift error: %s", exc)

    # 3. Concept drift (metric comparison)
    if baseline_metric > 0 and metric_val > 0:
        logger.info("  Running concept drift check…")
        result.concept_drift_detected, result.concept_drift_score = \
            run_concept_drift(baseline_metric, metric_val)
        logger.info("    Concept drift: detected=%s, drop=%.4f (baseline=%.4f, current=%.4f)",
                     result.concept_drift_detected, result.concept_drift_score,
                     baseline_metric, metric_val)

    # ── Save per-job results ────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "drift_results.json", "w") as fh:
        json.dump(asdict(result), fh, indent=2, default=str)
    if metrics_json:
        with open(output_dir / "final_metrics.json", "w") as fh:
            json.dump(metrics_json, fh, indent=2)

    return result


# ──────────────────────────────────────────────────────────────────
# Consolidated report
# ──────────────────────────────────────────────────────────────────

def generate_consolidated_report(all_results: List[JobDriftResult],
                                 output_path: Path) -> None:
    """Write a Markdown report summarising drift across all 15 jobs."""
    lines = [
        "# 🔍 Cross-Job Drift Analysis — Consolidated Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Total jobs analysed**: {len(all_results)}",
        "",
    ]

    # Summary stats
    n_feature = sum(1 for r in all_results if r.feature_drift_detected and not r.is_baseline)
    n_label = sum(1 for r in all_results if r.label_drift_detected and not r.is_baseline)
    n_concept = sum(1 for r in all_results if r.concept_drift_detected and not r.is_baseline)
    n_non_baseline = sum(1 for r in all_results if not r.is_baseline)
    n_errors = sum(1 for r in all_results if r.error)

    lines += [
        "## Executive Summary",
        "",
        f"| Drift Type | Jobs Analysed | Drift Detected | Detection Rate |",
        f"|------------|:------------:|:--------------:|:--------------:|",
        f"| Feature Drift  | {n_non_baseline} | {n_feature} | {n_feature / max(n_non_baseline, 1) * 100:.0f}% |",
        f"| Label Drift    | {n_non_baseline} | {n_label} | {n_label / max(n_non_baseline, 1) * 100:.0f}% |",
        f"| Concept Drift  | {n_non_baseline} | {n_concept} | {n_concept / max(n_non_baseline, 1) * 100:.0f}% |",
        "",
        f"**Errors**: {n_errors} job(s) had errors during analysis.",
        "",
    ]

    # Per-experiment group
    experiments = {}
    for r in all_results:
        experiments.setdefault(r.experiment, []).append(r)

    for exp_name, results in experiments.items():
        results_sorted = sorted(results, key=lambda r: r.job_date)
        lines += [
            f"---",
            f"## {exp_name.replace('_', ' ').title()}",
            "",
            f"| # | Job Name | Date | Feature Drift | Label Drift | Concept Drift | {results[0].primary_metric_name.title()} | Status |",
            f"|---|----------|------|:-------------:|:-----------:|:-------------:|:--------:|:------:|",
        ]
        for i, r in enumerate(results_sorted):
            if r.is_baseline:
                status = "🟢 BASELINE"
                feat = "—"
                label = "—"
                concept = "—"
            elif r.error:
                status = "🔴 ERROR"
                feat = "❌"
                label = "❌"
                concept = "❌"
            else:
                feat_icon = "🔴" if r.feature_drift_detected else "🟢"
                feat = f"{feat_icon} {r.feature_drift_share * 100:.1f}%"
                label_icon = "🔴" if r.label_drift_detected else "🟢"
                label = f"{label_icon} {r.label_drift_score:.4f}"
                concept_icon = "🔴" if r.concept_drift_detected else "🟢"
                concept = f"{concept_icon} Δ{r.concept_drift_score:+.4f}"
                any_drift = r.feature_drift_detected or r.label_drift_detected or r.concept_drift_detected
                status = "⚠️ DRIFT" if any_drift else "✅ STABLE"

            lines.append(
                f"| {i + 1} | `{r.job_name}` | {r.job_date} | {feat} | {label} | {concept} | {r.current_metric:.4f} | {status} |"
            )

        lines.append("")

        # Drifted columns detail
        drifted_jobs = [r for r in results_sorted if r.feature_drifted_columns and not r.is_baseline]
        if drifted_jobs:
            lines += [
                f"### Drifted Columns Detail",
                "",
            ]
            for r in drifted_jobs:
                lines.append(f"- **{r.job_name}** ({r.job_date}): {', '.join(r.feature_drifted_columns[:10])}"
                             + (f" … and {len(r.feature_drifted_columns) - 10} more" if len(r.feature_drifted_columns) > 10 else ""))
            lines.append("")

    # Retraining Cadence per experiment group
    lines += [
        "---",
        "## 🔄 Retraining Cadence Recommendations",
        "",
        "Stability scores are computed from the baseline dataset characteristics:",
        "PSI self-check (40%), dataset size (20%), feature complexity (20%),",
        "class balance (10%), and feature volatility (10%).",
        "",
    ]
    for exp_name, results in experiments.items():
        first = results[0] if results else None
        if first and first.cadence_name:
            # Cadence badge
            badge_map = {
                "quarterly": "🟢", "monthly": "🟡",
                "biweekly": "🟠", "weekly": "🔴",
            }
            badge = badge_map.get(first.cadence_name, "⚪")

            lines += [
                f"### {exp_name.replace('_', ' ').title()}",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Stability Score | **{first.stability_score}/100** |",
                f"| Cadence | {badge} **{first.cadence_name.upper()}** (every {first.cadence_days} days) |",
                "",
                f"> **Rationale**: {first.cadence_rationale}",
                "",
            ]

            # Component breakdown
            if first.stability_components:
                lines += [
                    "<details>",
                    "<summary>Score Breakdown</summary>",
                    "",
                    "| Component | Raw Value | Score | Weight |",
                    "|-----------|-----------|:-----:|:------:|",
                ]
                for comp_name, comp in first.stability_components.items():
                    raw_val = comp.get("raw", "—")
                    if isinstance(raw_val, float):
                        raw_val = f"{raw_val:.6f}"
                    elif raw_val is None:
                        raw_val = "N/A"
                    lines.append(
                        f"| {comp_name.replace('_', ' ').title()} | {raw_val} "
                        f"| {comp.get('score', 0):.1f} | {comp.get('weight', 0):.0%} |"
                    )
                lines += ["", "</details>", ""]
        else:
            lines += [
                f"### {exp_name.replace('_', ' ').title()}",
                "",
                "⚠️ Cadence not computed (baseline data unavailable).",
                "",
            ]

    # Metric evolution per group
    lines += [
        "---",
        "## Metric Evolution Over Time",
        "",
    ]
    for exp_name, results in experiments.items():
        results_sorted = sorted(results, key=lambda r: r.job_date)
        lines.append(f"### {exp_name.replace('_', ' ').title()}")
        lines.append("")
        lines.append("```")
        for r in results_sorted:
            bar_len = int(r.current_metric * 40) if r.current_metric else 0
            bar = "█" * bar_len + "░" * (40 - bar_len)
            tag = " ← BASELINE" if r.is_baseline else ""
            lines.append(f"  {r.job_date} |{bar}| {r.current_metric:.4f}{tag}")
        lines.append("```")
        lines.append("")

    # Recommendations
    lines += [
        "---",
        "## Recommendations",
        "",
    ]
    if n_feature + n_label + n_concept == 0:
        lines += [
            "✅ **No drift detected across any of the analysed jobs.**",
            "",
            "The pipeline produces consistent results when processing the same source data.",
            "This is the expected behaviour for deterministic ML pipelines on stable data.",
            "",
            "Consider scheduling periodic drift checks when source data is refreshed",
            "or when new data batches are ingested into the Azure ML datastore.",
        ]
    else:
        if n_feature > 0:
            lines.append(f"- ⚠️ **Feature drift** detected in {n_feature} job(s). "
                         "Investigate whether source data has changed.")
        if n_label > 0:
            lines.append(f"- ⚠️ **Label drift** detected in {n_label} job(s). "
                         "Check for target distribution shift.")
        if n_concept > 0:
            lines.append(f"- ⚠️ **Concept drift** detected in {n_concept} job(s). "
                         "Model performance is degrading — consider retraining.")

    lines += ["", f"---", f"*Report generated by MLOps V3 Drift Detection System*", ""]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info("Consolidated report → %s", output_path)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    # Suppress Azure SDK verbose HTTP request/response logging
    for noisy in ("azure", "urllib3", "msrest", "msal", "azure.identity",
                  "azure.core.pipeline"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s ┃ %(levelname)-7s ┃ %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("=" * 60)
    logger.info("Cross-Job Drift Analysis — 15 Azure ML Pipeline Jobs")
    logger.info("=" * 60)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    ml_client = get_ml_client()
    all_results: List[JobDriftResult] = []

    # Parse optional --group filter
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", help="Run only specific group(s), comma-separated")
    args = parser.parse_args()
    group_filter = set(args.group.split(",")) if args.group else None

    for group_name, group_config in EXPERIMENT_GROUPS.items():
        if group_filter and group_name not in group_filter:
            logger.info("Skipping group %s (not in filter)", group_name)
            continue

        logger.info("")
        logger.info("━" * 60)
        logger.info("Experiment: %s (%s)", group_name, group_config["task_type"])
        logger.info("━" * 60)

        baseline_df: Optional[pd.DataFrame] = None
        baseline_metric: float = 0.0
        jobs = group_config["jobs"]

        for idx, (job_name, job_date) in enumerate(jobs):
            logger.info("")
            logger.info("▸ [%d/%d] Job: %s (%s)", idx + 1, len(jobs), job_name, job_date)

            job_output_dir = OUTPUT_ROOT / group_name / job_name
            result = analyze_job(
                ml_client=ml_client,
                job_name=job_name,
                job_date=job_date,
                experiment=group_name,
                baseline_df=baseline_df,
                baseline_metric=baseline_metric,
                group_config=group_config,
                output_dir=job_output_dir,
            )
            all_results.append(result)

            # First job becomes the baseline for subsequent comparisons
            if idx == 0 and not result.error:
                data_path = download_job_output(ml_client, job_name, "dataset_processed", STAGING_DIR)
                if data_path:
                    baseline_df = load_dataset(data_path)
                    baseline_metric = result.current_metric
                    logger.info("  📌 Baseline set: %d rows, metric=%.4f",
                                len(baseline_df), baseline_metric)

        # ── Compute stability score & cadence for this experiment group ──
        group_results = [r for r in all_results if r.experiment == group_name]
        if baseline_df is not None and len(group_results) > 0:
            logger.info("")
            logger.info("  Computing stability score & retraining cadence…")
            try:
                target_col = group_config.get("target_column")
                task_type = group_config.get("task_type", "")

                # Feature-only DataFrame (exclude target for PSI self-check)
                feature_df = baseline_df.drop(columns=[target_col], errors="ignore") if target_col else baseline_df

                # PSI self-check: split baseline 50/50 and compute PSI
                n_half = len(feature_df) // 2
                if n_half > 50:
                    ref_half = feature_df.iloc[:n_half]
                    test_half = feature_df.iloc[n_half:]
                    psi_scores = compute_feature_psi(ref_half, test_half)
                else:
                    psi_scores = {c: 0.0 for c in feature_df.columns}

                # Feature volatility
                feat_vol = compute_feature_volatility(feature_df)

                # Imbalance ratio (classification only)
                imbalance_ratio = None
                if target_col and target_col in baseline_df.columns and task_type == "classification":
                    vc = baseline_df[target_col].value_counts()
                    if len(vc) >= 2:
                        imbalance_ratio = float(vc.min() / vc.max())

                stability, components = compute_stability_score(
                    psi_scores=psi_scores,
                    n_rows=len(baseline_df),
                    n_features=feature_df.shape[1],
                    imbalance_ratio=imbalance_ratio,
                    feature_volatility=feat_vol,
                )
                cadence_name, cadence_days, cadence_rationale = determine_retraining_cadence(stability)

                logger.info("  📊 Stability Score: %d/100 → %s (every %d days)",
                            stability, cadence_name.upper(), cadence_days)

                # Apply cadence info to ALL results in this group
                for r in group_results:
                    r.stability_score = stability
                    r.stability_components = components
                    r.cadence_name = cadence_name
                    r.cadence_days = cadence_days
                    r.cadence_rationale = cadence_rationale

            except Exception as exc:
                logger.error("  Cadence computation failed: %s", exc)

    # ── Generate consolidated report ────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Generating consolidated report…")
    logger.info("=" * 60)

    consolidated_path = OUTPUT_ROOT / "consolidated_report.md"
    generate_consolidated_report(all_results, consolidated_path)

    # Also save raw JSON
    raw_json_path = OUTPUT_ROOT / "all_results.json"
    with open(raw_json_path, "w") as fh:
        json.dump([asdict(r) for r in all_results], fh, indent=2, default=str)
    logger.info("Raw results JSON → %s", raw_json_path)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE — %d jobs analysed", len(all_results))
    n_ok = sum(1 for r in all_results if not r.error)
    n_err = sum(1 for r in all_results if r.error)
    n_drift = sum(1 for r in all_results
                  if (r.feature_drift_detected or r.label_drift_detected or r.concept_drift_detected)
                  and not r.is_baseline)
    logger.info("  Successful: %d  |  Errors: %d  |  Drift found: %d", n_ok, n_err, n_drift)
    logger.info("  Consolidated report: %s", consolidated_path)
    logger.info("  Output directory:    %s", OUTPUT_ROOT)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
