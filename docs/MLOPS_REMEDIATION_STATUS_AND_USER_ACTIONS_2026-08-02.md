# MLOps Remediation Status and User Actions

Date: 2026-08-02  
Repository: `mlops-solution-accelerator-v3`  
Branch: `codex_ys/mlops-pipeline-correctness`  
Base commit: `eb739444f9d439b75232e0342f5d5c52ab86986c` plus the current intentional working tree  
Azure target: subscription `93044a08-5661-4f1b-b424-5eafe066a9d1`, resource group `mvpv1`, workspace `mlops-accelerator`, compute `mlopsv2computecluster`

## Executive Status

**Release verdict: blocked for Azure acceptance and production release.**

The core model-selection, preprocessing, holdout, candidate, registration, drift-policy, submission-governance, and execution-identity defects identified in the 2026-08-01 review are now corrected at the local source/contract level. Classification, regression, and clustering graphs build through the Azure ML SDK. This is not yet an Azure-runtime or production-readiness claim.

The current critical path is external:

1. Live ARM reports the subscription as `Warned`; the bounded canonical classification canary was rejected with `ReadOnlyDisabledSubscription` before Azure created a job.
2. Python Azure SDK and Azure CLI requests intermittently fail TLS negotiation with Windows error `10054`. PowerShell ARM calls can succeed, so this is intermittent transport/proxy/VPN/TLS interception behavior rather than missing workspace RBAC.
3. The working tree is intentionally large and uncommitted, and the branch has no upstream. No reviewed release SHA exists yet.

The final pre-canary ARM recheck also hit error `10054`; no newer `Enabled` evidence was accepted and no second write was attempted.

The product should remain single-operator/private beta until the active product, persistence, authentication, and Azure evidence gates below are closed.

## Evidence Matrix

| Proof plane | Current evidence | Status |
|---|---|---|
| Local unit/contract suite | Final stable integrated run: `446 passed, 6 warnings` in 73.36 seconds | Strong local proof, not Azure proof |
| Focused API/UI/replay checks | Replay suite `65 passed`; final API/baseline/async slice `48 passed`; React production build passed | Pass |
| Baseline ownership/completeness checks | Focused baseline checks `5 passed`; S13 evidence checks `5 passed` | Pass |
| Canonical Azure ML SDK graphs | Classification hash `79b7d84c0be0`; regression `ad4fec3e3785`; clustering `0c6c284dfce7` | Pass, no training |
| Recipe catalogs | Classification `219/219` valid; regression `29/29`; clustering `18/18`; zero quarantine | Pass locally |
| Azure schedules | All three daily schedules exist, are enabled, and report `Succeeded` through live ARM | Pass |
| Azure workspace/compute reads | Workspace and `mlopsv2computecluster` were readable; cluster was `Succeeded`, `Standard_D4s_v3`, min 0, max 8 | Pass with intermittent transport resets |
| Azure subscription write gate | Live ARM state `Warned`; canonical create rejected with `ReadOnlyDisabledSubscription` | Blocked |
| Exact-source Azure jobs | No accepted job ID; no compute node was allocated by the rejected canary | Missing |
| MLflow/registration/raw inference | Local contracts only | Missing live proof |
| Release revision/CI | 348 status entries: 274 tracked, 74 untracked, including 188 intentional recipe deletions; no upstream | Blocked |

## Blocker Status by Impact

### Azure Acceptance and Connectivity

| ID | Status | Impact if not fixed | Owner and action |
|---|---|---|---|
| AX-01 subscription write state | **ACTIVE - external P0** | Azure cannot create exact-source jobs, so MLflow, mounted data, registration, inference, and compute shutdown cannot be accepted. | Subscription/billing owner: reactivate the subscription until live ARM says `Enabled`. |
| NET-01 intermittent TLS reset | **ACTIVE - workstation/platform** | SDK reads, schedule reconciliation, submission, and monitoring can fail nondeterministically with error `10054`. | Network owner: remove or configure VPN/proxy/TLS inspection and allow Azure management, Azure ML API, and Blob endpoints. |

### Wrong-Winner, Leakage, and Model Integrity

| IDs | Status | Impact if not fixed | Current disposition |
|---|---|---|---|
| MS-01, MS-03, MS-04, MS-05, MS-06, MS-08, MS-09 | **RESOLVED LOCALLY** | These previously allowed leaky pruning, excluded baselines, weak comparisons, cross-candidate relabeling, or misleading timeout evidence. | Split/fold-local preprocessing, raw-input Phase A bundles, minimum candidate coverage, SplitManifest/bundle identity validation, explicit task semantics, feasibility-first selection, and censored timeout evidence are covered by tests. Azure proof remains pending. |
| MS-02 unsupported group/time/preassigned splits | **RESOLVED BY REJECTION** | Accepting them without end-to-end CV semantics would cause leakage or runtime failure. | Compiler fails closed. Implement as a separately scoped product expansion if required later. |
| MS-07 recipe catalogs | **RESOLVED LOCALLY** | Quarantined production recipes would narrow the search while overstating optimization coverage. | Invalid recipes were retired/repaired; current compile has zero quarantined recipes. |

### Configuration, Identity, and Replay

| ID | Status | Impact if not fixed | Owner and action |
|---|---|---|---|
| CE-01 second engine definition | **ACTIVE - product decision** | Acceptance criteria remain ambiguous between FLAML and Azure AutoML, affecting adapters, UI wording, tests, and clustering support. | Product owner: approve either `PyCaret + FLAML on Azure ML` (matches current code and is recommended for this release) or fund `PyCaret + Azure AutoML` as a new engine implementation. |
| CE-02 accepted-but-ignored controls | **RESOLVED LOCALLY** | Runtime could silently differ from configuration and budget. | Retired relevance/diversity/imputation controls now fail schema-v2 compilation, legacy values migrate out explicitly, and every task honors its configured proxy threshold. |
| CE-03 immutable data identity | **PARTIAL - user onboarding** | Replacing data at the same path can make a run unreproducible. | Developer fix is fail-closed for production submission. Data owner must compute and add `dataset.content_sha256` to every production config. |
| CE-04 environment identity | **RESOLVED LOCALLY** | Manifest could claim dependencies different from graph execution. | Manifest now records the resolved per-component environment mapping/hashes. Azure proof pending. |
| CE-05 downstream execution identity | **RESOLVED LOCALLY** | HPO, final evaluation, or registration could accept cross-execution artifacts. | S08/S10/S12 require and validate the frozen execution manifest. |
| CE-06 resubmit/controller replay | **RESOLVED LOCALLY** | A mutable YAML/source tree could be run as if it were the previously approved revision. | S14/controller/resubmit require execution, config, source, and decision identities. Exact replay fails on changed bytes; a changed run must be an explicit new revision with a reason. Azure proof pending. |
| CE-07 legacy direct submitters | **RESOLVED LOCALLY** | Operators could use broken or unaudited submission paths. | Legacy scripts are canonical wrappers. |
| CE-08 force audit | **RESOLVED LOCALLY** | Duplicate expensive jobs could bypass guards with no durable reason/audit. | Force reason and fail-closed audit reservation are required. |

### Drift, Controller, API, and UI

| ID | Status | Impact if not fixed | Owner and action |
|---|---|---|---|
| DC-01, DC-02, DC-03 | **RESOLVED LOCALLY** | UI requests failed, drift evidence contradicted policy, and thresholds differed from configuration. | Both UIs send `decision_path`; API joins identity-matched S13/S14 artifacts; S14 consumes the validated drift policy. |
| DC-04 controller ledger durability | **PARTIAL - deployment** | Multiple replicas or ephemeral disks can lose reservations, duplicate jobs, or corrupt audit history. | Local locking, atomic reservation, fsync, and crash status records are implemented. Deploy one controller replica on a shared durable volume now; move to Blob leases, Table, or SQL before multi-replica production. |
| DC-05 baseline ownership/completeness | **RESOLVED LOCALLY** | Cross-dataset, stale, or incomplete baselines can create false or missed retraining. | Every API baseline entrypoint now requires the completed producing job, exact output URI, task/dataset/config identity, immutable job tags, and downloaded metadata/reference data. S13 treats missing reference data as unavailable. Azure proof pending. |
| DC-06 async request durability/polling | **RESOLVED LOCALLY; DEPLOYMENT PENDING** | API restart previously lost request state and encouraged duplicate retries. | Tickets are atomically persisted, jobs carry a request ID, stale requests reconcile against Azure without blind resubmission, and React polls to a terminal state. Production must place `MLOPS_SUBMISSION_REQUEST_ROOT` on durable shared storage. |
| DC-07 schedule truth | **RESOLVED** | Planned schedules could be shown as live when absent. | API now reports live `enabled`, `disabled`, `missing`, or `unverified`. Live ARM confirms all three planned schedules are enabled and `Succeeded`. |
| DC-08 actor/tenant authorization | **ACTIVE - release boundary** | A shared API key cannot provide actor roles, tenant isolation, or defensible approvals. | Security/platform owner: keep private single-operator access, then add Entra ID/OIDC, app roles, workspace authorization, and actor-bound audit before broader release. |

### Release and Validation

| ID | Status | Impact if not fixed | Owner and action |
|---|---|---|---|
| RV-01 release provenance | **ACTIVE** | No commit can reproduce the reviewed state; CI, rollback, and Azure evidence cannot bind to a SHA. | Repository owner must authorize a bounded inventory, commit, push, PR, and CI run. Do not use `git add -A` blindly in this large intentional worktree. |
| RV-02 live coverage | **PARTIAL** | A green local suite can hide adapter, mounted-data, MLflow, registry, or inference defects. | Run the three bounded Azure canaries and registered raw-input smoke tests after AX-01/NET-01. |
| RV-03 documentation truth | **RESOLVED FOR ACTIVE DOCS** | Engineers can reintroduce holdout leakage, forecasting, or MLflow URI defects by following stale guidance. | Sixteen active documents now align to the three-task, Stage 2 split, CV selection, locked-test-once, and S13/S14/controller contract. The obsolete Phase 1 document is prominently marked historical. |
| RV-04 unsupported forecasting graph | **RESOLVED** | Every run previously paid for an irrelevant skip node and exposed unsupported behavior. | Forecasting/timeseries was removed from the active three-task graph. |

## User Actions

### 1. Restore Azure Subscription Writes

This is a subscription lifecycle/billing issue, not a workspace RBAC issue. Do not add random workspace roles to address `ReadOnlyDisabledSubscription`.

1. Open Azure Portal, then `Subscriptions` -> `Azure subscription 1`.
2. Review the overview warning plus Billing, invoices, payment method, and account status.
3. Settle the balance or update the payment method and reactivate the subscription.
4. If no self-service reactivation is offered, open an Azure **Billing + Subscription Management** support request. Include subscription ID `93044a08-5661-4f1b-b424-5eafe066a9d1`, live state `Warned`, and error code `ReadOnlyDisabledSubscription`.

Verify with a live ARM read, not cached `az account show`:

```powershell
az rest --method get `
  --url "https://management.azure.com/subscriptions/93044a08-5661-4f1b-b424-5eafe066a9d1?api-version=2022-12-01" `
  --query "{state:state,displayName:displayName}" -o json
```

Required result: `state` is `Enabled`.

### 2. Stabilize Azure TLS Connectivity

1. Temporarily disconnect VPN and retry.
2. Inspect `HTTPS_PROXY`, `HTTP_PROXY`, corporate TLS inspection, antivirus HTTPS scanning, and firewall logs.
3. Allow outbound TCP 443 without certificate rewriting for `management.azure.com`, `*.api.azureml.ms`, `*.notebooks.azure.net`, and `*.blob.core.windows.net`.
4. If a proxy is mandatory, install its trusted root correctly for both Windows and the Python environment used by Azure CLI/SDK; do not disable TLS verification.
5. Confirm both `az rest` and the Azure ML Python SDK can list schedules repeatedly without `WinError 10054`.

### 3. Add Production Dataset Digests

Run this on the Azure notebook/compute against the exact CSV used by the config, not against a transformed local copy:

```powershell
$py = "C:\Users\yashu\.codex\scratch\mlops-pipeline-correctness-venv\Scripts\python.exe"
& $py scripts/compute_dataset_fingerprint.py "<exact-dataset.csv>"
```

Add the returned 64-character digest to the matching config:

```yaml
dataset:
  content_sha256: "<digest>"
```

Stage 1 recomputes and verifies this digest. A mismatch must block the run.

### 4. Select the Product Contract

Record one decision in requirements and UI terminology:

- Recommended current release: **PyCaret + FLAML on Azure ML**.
- Alternative: **PyCaret + Azure AutoML**, which requires a new adapter, identity contract, metrics parity, catalog capability matrix, cost controls, and three-task acceptance work.

### 5. Configure Durable Controller State

For the immediate private-beta deployment, run exactly one controller/API writer and place the ledger root on one shared durable mount:

```text
MLOPS_AUTO_RETRAIN_LEDGER_ROOT=<shared-durable-mounted-directory>
MLOPS_AUTO_RETRAIN_LEDGER=auto_retrain_decisions.jsonl
MLOPS_SUBMISSION_REQUEST_ROOT=<shared-durable-mounted-directory>/submission-requests
```

Do not use a container-local `outputs` directory for restart-sensitive operation. Multi-replica production requires a store with leases/transactions rather than relying only on a filesystem sidecar lock.

### 6. Run Bounded Azure Acceptance

Only after live ARM reports `Enabled` and repeated SDK reads are stable, run classification first:

```powershell
$py = "C:\Users\yashu\.codex\scratch\mlops-pipeline-correctness-venv\Scripts\python.exe"
& $py pipelines/submit_pipeline.py `
  --config configs/config_canary_classification_cardiac_arrest_workspace_azureml.yml `
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 `
  --resource_group mvpv1 `
  --workspace_name mlops-accelerator `
  --compute mlopsv2computecluster `
  --use_phase1 --wait --stop_compute
```

Proceed sequentially to regression and clustering only after classification completes. For each canary, retain:

- job ID and exact source/config/data/environment identities;
- S06/S08/S10/S12 identity evidence;
- S13 evidence and S14 decision identity;
- MLflow parent/child runs and artifacts;
- exact registered model version and warning/no-promotion behavior;
- raw-input prediction from the registered bundle;
- compute scale-down to min nodes.

## Project Plan and Exit Gates

| Order | Work item | Owner | Exit evidence |
|---|---|---|---|
| 1 | Restore subscription and network health | User/billing/network | ARM `Enabled`; repeated SDK reads pass |
| 2 | Immutable replay, async state, baseline validation, and docs integration | Development | Complete locally: `446 passed`; Azure proof pending |
| 3 | Add dataset digests and verified baseline jobs | Data/ML owner | Config digests committed; approvals identity-verified |
| 4 | Create a reviewable source revision | Repo owner + development | Commit, upstream branch, PR, CI SHA |
| 5 | Run three bounded canaries | Development on Azure cluster | Three successful job IDs and artifacts |
| 6 | Verify MLflow, registration, and raw inference | ML/release owner | Exact registered versions and inference smoke proof |
| 7 | Close production controls | Platform/security | Transactional multi-replica state plus Entra/OIDC/RBAC |

Do not label the solution production-ready until every exit gate has current evidence tied to the same committed source revision.
