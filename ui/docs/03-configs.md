# 3 · Configs

**File:** [`ui/pages/3_Configs.py`](../pages/3_Configs.py)
**Icon:** ⚙️
**Single sentence:** Browse, edit, duplicate, and delete the YAML files that fully describe a pipeline run.

## Who is this for?

Data scientists tuning recipes ("what if I add target encoding?") and ML
engineers maintaining the catalog of canonical configs.

## What is a "pipeline config"?

A single YAML file under `configs/` that defines:

| Section | Contents |
|---------|----------|
| `dataset` | Datastore path, target column |
| `azureml` | Subscription, resource group, workspace, compute, env names |
| `stage1`…`stage5` | Validation, preparation, preprocessing, feature engineering, baseline settings |
| `phases.phase_a_baseline` | Baseline engines (PyCaret + FLAML) |
| `phases.phase_b_recipes` | List of recommended variants × engines |
| `phases.phase_c_hpo` | Optuna search (n_trials, timeout, search space) |
| `final_evaluation` | Metrics + plots to generate at the end |

The API validates against `src/orchestration/config_schema.py` on every
save, so invalid YAML is rejected before reaching Azure ML.

## What you see

1. **"What is a pipeline config?" expander** — built-in primer for new users.
2. **Search / filter bar** — text search by name · task type · source
   (Built-in / User copy / Custom).
3. **Summary cards** (one per matching config):
   - Source badge (🔷 Built-in / 🟦 Built-in local / 🟧 User copy / ⚪ Custom)
   - KPIs: task · target · compute · phase count
  - Chips: Phase B variant cap · Phase C HPO trial budget · dataset path
4. **Detail section** (after picking from the dropdown):
   - Formatted viewer + JSON download
   - Three tabs:
     - **✏️ Edit** — YAML text area + Save (validates server-side)
     - **📑 Duplicate** — type a new name, get a pre-filled YAML editor, Create copy
     - **🗑️ Delete** — type the name to confirm, then delete (refused if a
       non-terminal job is using the config)
5. **"Create new config" expander** — start from a minimal template.

## Backend endpoints called

| Endpoint | Purpose |
|----------|---------|
| `GET /api/v1/configs` | List configs (5 min cache) |
| `GET /api/v1/configs/{name}` | Detail (5 min cache) |
| `POST /api/v1/configs/{name}` | Create (returns 409 if name exists) |
| `PUT /api/v1/configs/{name}` | Update (returns 409 if a job is running) |
| `DELETE /api/v1/configs/{name}` | Delete (refused if a job is running) |

## Common workflows

- **"Find every classification config"** → set Task type filter to
  `classification`.
- **"Create a churn variant with target encoding"** → pick the
  base churn config → Duplicate tab → rename to
  `config_classification_churn_target_enc_azureml` → edit YAML → Create copy.
- **"Bump Phase C trial budget"** → Edit tab → change `phase_c_hpo.n_trials`
  → Save.

## What it does NOT do

- It does not run the config — use **Submit Pipeline** for that.
- It does not perform a visual diff before save (planned).
- It does not autocomplete YAML keys against the schema (planned).
