# Phase B Safety-Net Review

Phase B can produce a safety-net champion when regular variant execution fails,
times out, or yields no usable metric. Safety-net champions keep the Azure ML
pipeline moving so downstream reports and diagnostics are still produced, but
they are not automatically registration-ready.

## Operator Rule

When `phases.phase_b.safety_net_review_required` is true, any champion manifest
with `review_status: review_required` or `registration_eligible: false` must be
reviewed before promotion or registration.

## Review Checklist

- Open the Phase B `champion_manifest.json`.
- Confirm `algorithm`, `status`, `review_status`, and `review_reason`.
- Inspect `leaderboard.csv`, `all_results.json`, `variant_validation_report.*`,
  and `elimination_report.json` for the reason regular variants did not win.
- Check `variant_anomaly_report.*` for non-numeric features, missing values,
  infinite values, and high-skew signals after variant preprocessing.
- Decide whether to adjust recipe selection, increase time budgets, or accept
  the safety-net result for a diagnostic-only downstream run.

## Promotion Guidance

Do not manually promote a safety-net champion unless a reviewer records why the
fallback behavior is acceptable for the specific dataset, task type, and metric.
Prefer re-running Phase B with corrected recipe inputs or a larger budget when
the safety-net result exists only because all candidate variants failed.