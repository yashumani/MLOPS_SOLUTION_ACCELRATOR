# 1 · Submit Pipeline

**File:** [`ui/pages/1_Submit_Pipeline.py`](../pages/1_Submit_Pipeline.py)
**Icon:** 🚀
**Single sentence:** Launch one new Azure ML pipeline job from a config YAML.

## Who is this for?

ML engineers and data scientists who want to kick off a training run
without dropping into a terminal or the Azure ML SDK.

## What you see

1. **Configuration selector** — dropdown of every YAML in
   `configs/`, populated from `GET /api/v1/configs`.
2. **Summary card** — task type, target column, compute target, phase
   count, Phase B variant cap, Phase C HPO trial budget, dataset path.
3. **YAML preview expander** — full config rendered as YAML so you can
   double-check before submit. Editing happens on the **Configs** page.
4. **Advanced Options expander**:
   - *Compute target (override)* — leave blank to use the config default.
   - *Disable component cache* — force every step to re-execute (no cache reuse).
   - *Baseline job for drift comparison* — when set, enables the optional
     `s13_drift_monitor` step that produces the PSI report shown on the
     **Drift Monitor** page.
   - *Custom tag key/value* — applied as Azure ML job tags for attribution.
5. **Submit button** — calls `POST /api/v1/pipelines/submit/async`,
   returns a `request_id` immediately.
6. **Async polling panel** — polls `GET /api/v1/pipelines/submit/status/{request_id}`
   every 1 s for up to 60 s. On success, surfaces:
   - "Open Focus" → jumps to the Focus page with the new job pre-selected.
   - "Open in Azure ML Studio" → external link.

## Backend endpoints called

| Endpoint | Purpose |
|----------|---------|
| `GET /api/v1/configs` | List configs |
| `GET /api/v1/configs/{name}` | Load config detail for the preview card |
| `POST /api/v1/pipelines/submit/async` | Async submit |
| `GET /api/v1/pipelines/submit/status/{request_id}` | Poll for submission outcome |

## Common workflows

- **"Run telecom churn baseline"** → pick `config_classification_telecom_churn_azureml`
  → preview YAML → Submit → click *Open Focus*.
- **"Run with a clean cache"** → expand Advanced → tick *Disable component cache*.
- **"Detect drift vs last week's job"** → expand Advanced → paste the previous
  job name into *Baseline job for drift comparison*.

## What it does NOT do

- It does not edit configs (use **Configs** page).
- It does not validate the YAML client-side; the API does that on submit
  and returns errors that surface in the polling panel.
- It does not support batch submission (one job at a time).
