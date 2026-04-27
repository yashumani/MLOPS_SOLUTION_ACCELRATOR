# Clustering Pipeline Failure Forensics — Last 7 Days

**Workspace:** `<AZURE_WORKSPACE_NAME>` (RG `<AZURE_RESOURCE_GROUP>`, sub `93044a08-…`)
**Window analyzed:** 2026-04-16 → 2026-04-23
**Branch / commit observed in failures:** `FAST-API-v1` @ `8d00d68`
**Author of report:** automated forensic investigation
**Remediation status (overall):** **NOT FIXED** — both root causes still present in `main` of `FAST-API-v1`.

---

## 1. Executive Summary

Six clustering pipeline jobs failed in the last seven days. They split into **two distinct root causes**:

| # | Bug | Affected jobs | Failed step | Root cause | Status |
|---|-----|---------------|-------------|------------|--------|
| **B1** | s08z empty-output regression | 5 | `s11` (final_evaluation) — fails before script starts | `mode: upload` added to `aggregate_phaseb.yml` `champion_model` output in commit `e299ff69` (2026-04-19). For clustering, aggregate_phaseb selects no champion (model_map keys are classification/regression-only) and writes an empty folder. `upload` mode skips empty folders → blob path never materializes → downstream s11 cannot mount `INPUT_phaseb_champion`. | **NOT FIXED** |
| **B2** | Non-UTF-8 dataset ingestion | 1 | `s1` (stage1_ingestion) | `pd.read_csv(dataset_uri)` is called with no `encoding=` argument; the dataset contains a 0xA3 (£) byte. | **NOT FIXED** |

The `final_evaluation.py` script (s11) **never executed** in any of the 5 B1 failures — `logs/azureml/stdoutlogs.txt` is empty in all 5 runs. Failure is in the Azure ML data-capability layer during input mount.

---

## 2. Inventory of Failed Jobs

| # | Dataset | Parent job (display name) | Parent ID | Failed step | Failed child ID | Bug | Submitted (UTC) |
|---|---------|---------------------------|-----------|-------------|-----------------|-----|-----------------|
| 1 | credit_default | `modest_stone_424zc9x1sb` | (parent) | s11 | `0422221d-1ee5-45c0-87b8-06f286b87b98` | B1 | 2026-04-22 06:17 |
| 2 | atp1d | `gray_bottle_qhysp2cp57` | (parent) | s11 | `4b7be91d-0344-4002-aed5-8a311c5f9c26` | B1 | 2026-04-22 ~06:30 |
| 3 | kidney_disease | `funny_dream_bwd5xw8fn2` | (parent) | s11 | `31685629-2810-4176-adf5-d9a50a56bbca` | B1 | 2026-04-22 ~06:30 |
| 4 | online_retail | `ashy_toe_g5q7sdqfld` | (parent) | s11 | `0094c750-95bf-4c31-a9bb-dc8ba9d72a04` | B1 | 2026-04-22 ~06:35 |
| 5 | churn_uplift | `affable_bone_rr37q01lb3` | (parent) | s11 | `16da2f35-71ec-47ad-a543-f9510129d9a0` | B1 | 2026-04-22 ~06:45 |
| 6 | (non-UTF8 source) | `willing_roti_p93qfd40b6` | (parent) | s1 | `b3839743-8eb5-4199-855a-61c1a6a49248` | B2 | 2026-04-18 16:12 |

For B1 jobs all 14 upstream child steps (`s1`, `s2`, `s3`, `s4`, `s5a`, `s5b`, `s5z`, `s6a`, `s6b`, `s7a`, `s7b`, `s08z`, `s10`, `s10z`) report **Completed**; only `s11` is `Failed`.

---

## 3. Bug B1 — Phase B Aggregate Empty-Output (5/6 failures)

### 3.1 Symptom (identical for all 5 runs)

`logs/azureml/stderrlogs.txt`:

```
[2026-04-22 06:17:47Z] Job failed, job RunId is 0422221d-…
Error: ScriptExecution.StreamAccess.NotFound
Native Error: error in streaming from input data sources
StreamError(NotFound) => stream not found
Error Message: The requested stream was not found. Please make sure the request uri is correct.
```

`logs/azureml/stdoutlogs.txt`: 0 bytes (script never started).

### 3.2 Decisive evidence — `system_logs/data_capability/data-capability.log` (credit_default)

```
INFO  06:17:46  create dir for /mnt/azureml/.../wd/INPUT_phaseb_champion
WARN  06:17:47  Failed to mount URI azureml://subscriptions/93044a08-.../resourcegroups/<AZURE_RESOURCE_GROUP>/
                workspaces/<AZURE_WORKSPACE_NAME>/datastores/mlops_blob/paths/azureml/
                ddc39112-d368-452b-88c2-c087c7063dd0/champion_model/
                due to exception of type ExecutionError
ERROR 06:17:47  ##[error] [CapabilitySession][start] Failed to start data session.
```

The id `ddc39112-d368-452b-88c2-c087c7063dd0` is the **`s08z` child** of the same parent (verified via `ml.jobs.list(parent_job_name=…)`).

For the other 4 parents the failed-mount URI follows the same pattern (s08z child id of that parent):

| Dataset | s08z child id | s11 child id |
|---------|---------------|--------------|
| credit_default | `ddc39112-d368-452b-88c2-c087c7063dd0` | `0422221d-…` |
| atp1d | `3c4200f0-721a-43da-84c3-5884173d01bd` | `4b7be91d-…` |
| kidney_disease | `75423597-cec2-491c-b6cc-0ba3a420d92b` | `31685629-…` |
| online_retail | `ccaca70a-9df1-4c7c-9b00-d8cbefbdd795` | `0094c750-…` |
| churn_uplift | `4f729e77-dc62-4597-922e-f0800cce9543` | `16da2f35-…` |

### 3.3 What s08z actually did (stdout, identical for both working and failing runs)

```
⚠️  r1_pycaret PyCaret: no leaderboard/metrics found
⚠️  r1_flaml   FLAML missing best_metric
⚠️  r2_pycaret PyCaret: no leaderboard/metrics found
⚠️  r2_flaml   FLAML missing best_metric
🏆 Phase B Aggregate: Selected None | Score: None | Reason: N/A
⚠️  Champion path missing or doesn't exist: None
```

[`src/steps/aggregate_phaseb.py`](../src/steps/aggregate_phaseb.py#L120-L160) hard-codes a 4-key model map (`r1_pycaret`, `r1_flaml`, `r2_pycaret`, `r2_flaml`) that targets classification/regression recipes; for clustering pipelines none of those manifests contain a usable score, `best_key` is `None`, and the script falls into the fallback branch:

```python
else:
    # Create empty output folder
    Path(args.champion_out).mkdir(parents=True, exist_ok=True)
    print(f"  ⚠️  Champion path missing or doesn't exist: {champion_path}")
```

This is **identical behavior** between the working 04-19 run (`plucky_night_gvjr5dtwfs` / online_retail / s08z `dd650d03-…`) and the failing 04-22 runs.

### 3.4 The actual regression

The behavior change is in the **component output mode**, not the step script.

`git show e299ff69 -- components/aggregate_phaseb.yml` (2026-04-19 16:56 UTC):

```diff
 outputs:
   aggregate_report:
     type: uri_file
+    mode: upload
   champion_model:
     type: uri_folder
+    mode: upload
```

Semantics:

* **Before (default `mount`):** the `champion_model` folder is mounted as a writable blob-backed folder; even if the script writes nothing, the URI is registered as an asset and downstream `mount` succeeds (Azure ML data plane handles the empty-folder case).
* **After (`mode: upload`):** Azure ML uploads the local folder *contents* on completion. **An empty folder uploads zero blobs**, and Azure Blob Storage has no concept of empty folders — so the URI `azureml://…/{run_id}/champion_model/` simply does not exist. Downstream `s11` `phaseb_champion` `mount` then fails with `ScriptExecution.StreamAccess.NotFound`.

Confirmed by comparing the s11 `data-capability.log` of the working 04-19 run (`d7f65273-…`, mounts succeed cleanly) and the failing 04-22 runs (mount fails on `INPUT_phaseb_champion`).

This affects **only** clustering because classification/regression pipelines find a champion in the 4-key map, copy a real model into the output folder, and `upload` mode then has real bytes to upload.

### 3.5 Why `final_evaluation.py` did not save us

`src/steps/final_evaluation.py` (read-only per V3 guardrails) handles missing models gracefully — `eval_clustering_model` swallows all exceptions and returns `None`. **But the script never runs**, because the failure is in the Azure ML data-capability layer (input-mount preflight) that executes *before* the user process starts. No defensive code inside the script can intercept it.

### 3.6 Remediation status: **NOT FIXED**

* No reverts of `e299ff69` are present.
* `mode: upload` is still set on `champion_model` in `components/aggregate_phaseb.yml` (HEAD).
* `aggregate_phaseb.py` still produces an empty folder for clustering.
* No subsequent commit added a sentinel-file write or task-aware model_map.

### 3.7 Recommended fixes (ranked, all leave `final_evaluation.py` untouched)

1. **Minimal — revert mode on `champion_model` to `mount`** (or simply delete the `mode: upload` line so it falls back to default). Restores 04-18 behavior. ~1-line change in `components/aggregate_phaseb.yml`. Low risk.
2. **Make the empty-folder upload deterministic** — in `aggregate_phaseb.py`, when no champion is selected, write a small sentinel file (e.g. `champion_out/_NO_CHAMPION.json` containing the report). Forces blob materialization regardless of upload mode; also self-documents downstream. ~5 lines, no component change.
3. **Task-aware model_map** — extend `aggregate_phaseb.py` to recognize clustering recipe outputs so a real Phase B clustering model is selected and copied. Deepest fix; requires understanding the clustering recipe runner outputs.

Recommendation: ship **Fix #1 + Fix #2 together** — Fix #1 stops the bleeding, Fix #2 hardens against future re-introduction of `upload` mode.

---

## 4. Bug B2 — Non-UTF-8 Dataset Ingestion (1/6 failures)

### 4.1 Symptom

`willing_roti_p93qfd40b6` / s1 child `b3839743-8eb5-4199-855a-61c1a6a49248`:

```
File "src/steps/stage1_ingestion.py", line 63, in main
    df = pd.read_csv(dataset_uri)
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3 in position 79780: invalid start byte
```

### 4.2 Root cause

`stage1_ingestion.py` line 63 calls `pd.read_csv(dataset_uri)` with no encoding, no fallback, and no error reporting. Byte `0xA3` is `£` in latin-1/cp1252 — common in retail/financial datasets exported from Excel.

### 4.3 Remediation status: **NOT FIXED**

Inspection of `src/steps/stage1_ingestion.py` HEAD shows no encoding-fallback logic.

### 4.4 Recommended fix

In `stage1_ingestion.py`, replace the bare `read_csv` with a fallback chain: `utf-8` → `utf-8-sig` → `cp1252` → `latin-1`, and log which encoding was used. ~10 lines.

---

## 5. How to verify the fixes (Azure-only, per V3 guardrails)

1. Apply Fix #1 + Fix #2 to a feature branch.
2. Submit one of the failing clustering configs via `pipelines/submit_pipeline.py`:
   ```bash
   python pipelines/submit_pipeline.py \
     --config configs/config_clustering_credit_default_azureml.yml \
     --subscription_id <AZURE_SUBSCRIPTION_ID> \
     --resource_group <AZURE_RESOURCE_GROUP> --workspace_name <AZURE_WORKSPACE_NAME> \
     --compute <AZURE_COMPUTE> --wait
   ```
3. Confirm `s11` reports **Completed** (not Failed). Inspect its stdout — clustering branch should run and emit silhouette/davies_bouldin metrics (or `None`s if no models, but `Completed`).
4. Repeat for a non-UTF8 dataset to validate B2.

---

## 6. Appendix — How this evidence was collected

* `azure.ai.ml.MLClient` for job listing and child-job resolution.
* `mlflow.MlflowClient.download_artifacts` (workspace MLflow URI) for log retrieval — this works under `DefaultAzureCredential` even though `ml.jobs.download()` failed with SAS signature errors.
* All log paths cited (`logs/azureml/stderrlogs.txt`, `logs/azureml/stdoutlogs.txt`, `system_logs/data_capability/data-capability.log`, `user_logs/std_log.txt`) were downloaded from Azure ML directly; no reproduction or speculation.
