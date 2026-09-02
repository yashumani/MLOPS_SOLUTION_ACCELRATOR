# Pipeline I/O Contracts

Current as of: 2026-08-02

This document describes the active V3 Azure ML pipeline input/output contracts. Component YAMLs remain the binding implementation contract; this page explains those contracts for operators and developers.

## Pipeline Inputs

| Input | Type | Required | Source | Used by |
|---|---|---:|---|---|
| `config_name` | string | Yes | `pipelines/submit_pipeline.py` passes config filename only | All stages |
| `dataset_folder` | `uri_folder` | Yes | Azure ML datastore URI from config | `s01` |
| `execution_manifest` | `uri_file` | Yes | Canonical submitter; immutable config/code/environment identity | `s06` and downstream identity checks |
| `candidate_catalog` | `uri_file` | Yes | Canonical recipe compiler/selector | `s06` |
| `variants_list` | string | No | Legacy direct-caller compatibility; canonical submission passes an empty value | `s06` compatibility path |
| `engine_list` | string | Yes | Config or default `pycaret,flaml` | `s06` |
| `time_budget_per_variant` | int | Yes | Config/submission default | `s06` |
| `drift_baseline_in` | optional `uri_folder` | No | Previous run `drift_baseline` URI | `s13` |

Planner-mode inputs exist on `full_pipeline_v2()`: `planner_enabled`, `round1_max_variants`, `round2_max_variants`, `proxy_prune_threshold`, and `cache_enabled`.

## Pipeline Outputs

| Output | Type | Producer | Purpose |
|---|---|---|---|
| `eda_report` | `uri_folder` | `s01` | Initial EDA and ingestion quality artifacts. |
| `prep_report` | `uri_folder` | `s02` | Preparation report. |
| `prep3_report` | `uri_folder` | `s03` | Preprocessing report. |
| `fe_report` | `uri_folder` | `s04` | Feature engineering report. |
| `dataset_processed` | `uri_file` | `s04` | Feature-engineered dataset artifact. |
| `dataset_train` | `uri_file` | `s04` | Feature-engineered training partition used for drift evidence. |
| `dataset_holdout` | `uri_file` | `s02` | Canonical raw locked-test partition; consumed only by `s10`. |
| `baseline_pycaret_metrics` | `uri_file` | `s05a` | PyCaret baseline metrics. |
| `baseline_flaml_metrics` | `uri_file` | `s05b` | FLAML baseline metrics. |
| `baseline_aggregate_report` | `uri_file` | `s05z` | Phase A aggregate report. |
| `baseline_champion_model` | `uri_folder` | `s05z` | Phase A champion model. |
| `phaseb_leaderboard` | `uri_file` | `s06` | Phase B variant leaderboard. |
| `phaseb_all_results` | `uri_file` | `s06` | Full Phase B result set. |
| `phaseb_champion_manifest` | `uri_file` | `s06` | Phase B champion manifest. |
| `phaseb_champion_model` | `uri_folder` | `s06` | Phase B champion model. |
| `phasec_aggregate_report` | `uri_file` | `s09` | Phase C aggregate report. |
| `phasec_champion_model` | `uri_folder` | `s09` | Phase C optimized champion model. |
| `execution_manifest` | `uri_file` | `s06` | Validated execution identity propagated downstream. |
| `split_manifest` | `uri_file` | `s06` | Validated Stage 2 split identity propagated downstream. |
| `final_report` | `uri_file` | `s10` | CV-based champion selection evidence, one locked-test audit, and quality gate. |
| `final_champion_model` | `uri_folder` | `s10` | Exact CV-selected raw-input model bundle after locked-test audit. |
| `registry_info` | `uri_file` | `s12` | Model registration metadata or skip status. |
| `drift_report` | `uri_file` | `s13` | Drift, stability, cadence, and alert evidence; no policy or submission side effects. |
| `drift_baseline` | `uri_folder` | `s13` | Baseline folder for future comparison drift. |
| `retrain_decision` | `uri_file` | `s14` | Operator-readable retrain decision. |
| `decision_ledger_record` | `uri_file` | `s14` | Ledger-shaped decision record for review/append. |

## Stage 2 Partition Contract

`s02` creates the canonical partition before any learned transformation:

| Output | Consumer | Purpose |
|---|---|---|
| `raw_train_out` | `s05a`, `s05b`, `s06`, `s08`, `s10` contract validation | Raw training/search rows only. |
| `raw_holdout_out` | `s10` only | Immutable locked-test rows with canonical row identity. |
| `split_manifest_out` | Phase A/B and `s10` | Split strategy, seed, counts, row-identity hashes, dataset version, and `locked_test=true`. |

Expected behavior:

- Learned transformations fit within training rows/folds only and are persisted with each candidate bundle.
- Phase A/B/C selection uses comparable training/CV evidence and never reads `raw_holdout_out`.
- `s10` validates `split_manifest_out`, freezes one champion, then applies that exact bundle once to `raw_holdout_out`.
- Legacy Stage 4 sibling holdouts are not authoritative; resubmit through the current Stage 2 contract.

## Drift Report Contract

Produced by `s13`.

Important top-level fields:

| Field | Meaning |
|---|---|
| `execution_id` | s13-generated drift execution ID. |
| `config_name` | Config filename. |
| `task_type` | Exactly classification, regression, or clustering. |
| `dataset_name` | Dataset name from config. |
| `feature_psi_scores` | Per-feature PSI map. This is the current field name; do not use old `feature_psi`. |
| `stability_assessment` | Stability score, component scores, recommended cadence, recommended days. |
| `comparison_drift` | Previous-baseline comparison state and Evidently/concept results. |
| `warnings` | Non-blocking warnings. |

Retraining policy fields belong to the separate `s14` `retrain_decision` artifact. Consumers must not infer a submission decision from `s13` evidence alone.

`comparison_drift.available=false` means no previous baseline comparison was available. That is expected on first-cycle runs.

`comparison_drift.available=true` means `baseline_in` loaded and comparison drift ran.

## Drift Baseline Folder Contract

Produced by `s13`.

| File | Purpose |
|---|---|
| `feature_baseline.json` | Dataset, task, feature statistics, champion metric and algorithm metadata. |
| `reference_distributions.json` | Histogram/category distributions for PSI-style reference. |
| `reference_data.csv` | Reference split used by Evidently in future comparison runs. |

Azure ML job metadata can omit `outputs.drift_baseline.path`. If so, `az ml job download --output-name drift_baseline` prints the underlying datastore URI in the download banner. Store only approved baseline URIs in the decision ledger.

## Retrain Decision Contract

Produced by `s14`.

`retrain_decision` includes:

| Field | Meaning |
|---|---|
| `stage` | `s14_retrain_decision`. |
| `stage_id` | `S14`. |
| `config_name`, `task_type`, `dataset_name` | Context for the decision. |
| `decision.outcome` | `observe_only`, `refresh_baseline`, `candidate_retrain`, `promote_candidate`, or `blocked`. |
| `decision.should_submit` | Whether a candidate retrain/evaluation should be submitted by the external controller. |
| `decision.eligible_for_promotion` | Whether the candidate could be promoted if policy allowed it. Auto-promotion remains disabled by default. |
| `decision.severity` | `none`, `moderate`, `severe`, or policy-defined severity. |
| `decision.reasons` | Human-readable reasons for the decision. |
| `decision.signals` | Machine-readable metrics and blockers. |
| `comparison` | Baseline comparison availability and metadata. |
| `source` | Drift execution ID, Azure ML run ID if available, trigger label. |

`decision_ledger_record` has the append-only ledger shape used by `auto_retrain_decision_ledger.py`. `s14` writes the record as an artifact; operators or the external controller decide whether and when to append it to the durable ledger.

## Decision Ledger Record Contract

The current ledger format is append-only JSONL. Each line is one decision record. A local JSONL path is not multi-replica durable storage; production operation requires one shared server-owned storage root or a transactional service.

Important fields:

| Field | Meaning |
|---|---|
| `decision_id` | Unique decision identifier. |
| `config_name`, `task_type`, `dataset_name` | Resolver filters. |
| `input_baseline_uri` | Prior approved baseline used for comparison. |
| `output_baseline_uri` | Candidate baseline produced by the current run. |
| `candidate_job_name` | Azure ML job that produced the candidate. |
| `outcome`, `severity`, `reasons`, `signals` | Policy evidence. |
| `promotion_status` | Manual status such as `manual_pending`, `approved`, `production`. |
| `approved_for_future_baseline` | Boolean resolver gate. Only true records are automatically reused as approved baselines. |

Do not rewrite old records. Append a new record for changed status or completion evidence.

## Evidence And Acceptance Boundaries

| Evidence | Accepted meaning |
|---|---|
| Local tests/static checks | Local contract and syntax evidence only. |
| Azure ML SDK dry-run | Config compilation, component loading, and graph-shape evidence only. |
| Exact-source Azure ML job plus downloaded outputs | Pipeline-runtime behavior for that exact source/config/data/environment identity. |
| Registered-model load and raw-input prediction | Registry artifact usability; separate from pipeline completion. |
| Deployed endpoint request/response | Deployed-inference behavior; separate from registration and pipeline proof. |
