# Commit Ledger - Production Branch

Branch: `prod-hardening-20260425`  
Default branch: `main`  
Comparison: `origin/main..HEAD`  
Commit count ahead of `origin/main`: `46`  
Latest production pipeline commit: `fd63a2e3`

This ledger documents the production branch history that led to the current V3 state. It is intentionally commit-focused; operational status and gates are covered in `PRODUCTION_FREEZE_SUMMARY.md`.

## Commit Groups

| Group | Commits | Summary |
|---|---:|---|
| Drift monitoring foundation | 6 | Added s13 drift monitoring, baseline chaining, AML upload controls, and early failure fixes. |
| API and UI surface | 10 | Added FastAPI, Streamlit UI, async submit, config CRUD, focus cockpit, and cache improvements. |
| Production realignment | 3 | Realigned components and step scripts to the V3 production pipeline while preserving s13 drift. |
| Critical and high hardening batches | 13 | Fixed preprocessing, Phase B selection, MLflow URI handling, JSONL/EDA robustness, active-job guard, and balanced-accuracy selection. |
| Backend production hardening and docs | 11 | Added backend guards, scheduler, drift alerts, production handoff docs, and dry-run sweep tooling. |
| Latest critical freeze fix | 1 | Fixed clustering final evaluation and restored quality-gate defaults. |
| Runtime/state cleanup | 2 | Removed runtime artifact tracking and committed sprint support files. |

## Full Ledger

| # | Commit | Date | Subject |
|---:|---|---|---|
| 1 | `9281b11f` | 2026-04-15 | feat(drift): add s13 drift monitor pipeline step with Evidently comparison |
| 2 | `54a8ba5e` | 2026-04-15 | Add .amlignore to reduce upload from 247MB to 0.26MB |
| 3 | `81d2e579` | 2026-04-15 | fix: resolve baseline_job URI via History API for cross-pipeline drift comparison |
| 4 | `afb9af77` | 2026-04-16 | Add --force_rerun flag to disable component reuse |
| 5 | `c633fb47` | 2026-04-18 | fix: resolve 6 pipeline job failures |
| 6 | `e299ff69` | 2026-04-19 | feat(drift): production-ready s13 drift monitor with pipeline fixes |
| 7 | `c0226cda` | 2026-04-19 | feat(pipeline): NaN safety, MLflow URI fixes, drift detection, API, clustering configs |
| 8 | `9ff6fc54` | 2026-04-20 | feat: Streamlit UI dashboard + API extensions for MLOps V3 |
| 9 | `9e2de8dd` | 2026-04-20 | feat: complete FastAPI + Streamlit - studio URLs, baseline capture, enhanced health |
| 10 | `3223a71d` | 2026-04-23 | Realign components/ and src/steps/ to v3-production; preserve drift detection delta (s13) |
| 11 | `1b39c4d5` | 2026-04-23 | Reset pipeline_builder + submit_pipeline to v3-production; layer s13 drift_monitor on top |
| 12 | `a6149ca1` | 2026-04-23 | Resync src/ to v3-production; preserve drift_detector.py |
| 13 | `18c5ccc6` | 2026-04-23 | chore(ui): snapshot prior FAST-API-v1 UI work before refresh |
| 14 | `83383e6d` | 2026-04-23 | feat(ui): Phase 1 - cache APIClient + TTL-based data cache + prewarm |
| 15 | `ffc1d0ac` | 2026-04-23 | feat(ui): Phase 1.4 + Phase 2 - fragment-based logs + Focus cockpit |
| 16 | `5faee539` | 2026-04-23 | feat(ui): Phase 3 - Dashboard filter bar + paginated jobs table with Focus action |
| 17 | `8d2f2f2e` | 2026-04-23 | feat: Phase 4 - async submit + config CRUD with running-job guard |
| 18 | `71d3c189` | 2026-04-23 | refactor(ui): Phase 6 hygiene - Drift page uses cached_job_drift |
| 19 | `c36f1cf3` | 2026-04-23 | feat(ui): Phase 2.4 - sidebar shows focused job with Open/Clear actions |
| 20 | `1d41b409` | 2026-04-24 | fix(drift): bool to float64 cast prevents numpy subtract crash on one-hot bool columns |
| 21 | `ccc8c998` | 2026-04-24 | chore: commit docs, scripts, and Streamlit config from Apr 18-23 sprint |
| 22 | `53630d23` | 2026-04-24 | chore: update last_submitted_job tracking file |
| 23 | `55ab9966` | 2026-04-25 | fix(critical): Batch 1 - K7+K8+K9+K10+K11+M6 |
| 24 | `abf0f842` | 2026-04-25 | fix(critical): Batch 2 - K1+K2+K12 |
| 25 | `abb4eee1` | 2026-04-25 | fix(critical): Batch 3 - K3+K4 delete legacy phaseb files |
| 26 | `3575b747` | 2026-04-25 | fix(critical): Batch 4 - K5 disable PyCaret double-preprocessing in all setup() calls |
| 27 | `88cd65c9` | 2026-04-25 | fix(high): Batch 5a normalize MLflow tracking URI in shared logger |
| 28 | `dba855d4` | 2026-04-25 | fix(high): Batch 5b use balanced accuracy for baseline selection |
| 29 | `c37ecf32` | 2026-04-25 | fix(high): Batch 5c repair variant scoring and bundle globs |
| 30 | `d56ec5ee` | 2026-04-25 | fix(high): Batch 5d harden submission and boolean plumbing |
| 31 | `28dc19e5` | 2026-04-25 | fix(high): Batch 5e harden JSONL and EDA utilities |
| 32 | `f427d74a` | 2026-04-25 | fix(high): make active job guard SDK-compatible |
| 33 | `e8979f7c` | 2026-04-25 | fix(high): remove hardcoded Phase B recipe fallback |
| 34 | `2894f253` | 2026-04-25 | fix(high): use balanced accuracy for Phase B classification |
| 35 | `e3c56aab` | 2026-04-25 | fix(high): raise Phase B default budget for submissions |
| 36 | `0bc75076` | 2026-04-26 | Production hardening: submit_pipeline, pipeline_builder, silent-pass cleanup, security tests |
| 37 | `3cd9a875` | 2026-04-26 | chore: untrack pipelines/.last_submitted_job runtime artifact |
| 38 | `dc007e7d` | 2026-04-26 | docs(deliverables): V3 production readiness report (25/25 jobs Completed) |
| 39 | `0364e059` | 2026-04-27 | feat(backend): P1 hardening + P5 drift alerts & schedule |
| 40 | `653add46` | 2026-04-27 | docs(handoff): P6 production handoff for backend prod-hardening branch |
| 41 | `8e6d3d12` | 2026-04-27 | docs(handoff): correct alert log string + record smoke job verification |
| 42 | `85a0af3a` | 2026-04-27 | fix(p5): scheduler builds full pipeline args; sync alert env-var names in docs |
| 43 | `d9d02341` | 2026-04-27 | docs(freeze): backend review + code-freeze exit criteria for prod-hardening |
| 44 | `a6721e4c` | 2026-04-28 | stage_registry: prune dead legacy keys; flag stage0 component as not wired |
| 45 | `a6d6d56d` | 2026-04-28 | scripts: add dry-run job sweep for 16 azureml configs (NO FIRING) |
| 46 | `fd63a2e3` | 2026-05-01 | fix(critical): clustering eval crash + quality gate defaults |

## Latest Commit Detail

`fd63a2e3` is the current production fix commit. It resolved:

1. Clustering `champion_score=-inf` in final evaluation by using numeric float data and feature-name alignment before model prediction.
2. Over-aggressive quality blocking by restoring default thresholds to `classification=0.50`, `regression=0.0`, `clustering=0.0` and making `block_on_quality_fail=false` the default.
3. Holdout-safe readers and model-registration hardening committed alongside the final evaluation fix.

Documentation cleanup was performed after `fd63a2e3` and should be committed separately when the docs are approved.