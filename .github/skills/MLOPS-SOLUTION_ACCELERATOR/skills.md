# MLOps Solution Accelerator V3 — Skills

## Skill 1: Submit Pipeline Job
**Trigger:** User asks to submit, run, or test a pipeline
**Action:**
```bash
cd /home/azureuser/cloudfiles/code/Users/yashu.savyminds/mlops-solution-accelerator-v3
python pipelines/submit_pipeline.py \
  --config configs/<task_config>.yml \
  --subscription_id 93044a08-5661-4f1b-b424-5eafe066a9d1 \
  --resource_group mvpv1 \
  --workspace_name mlops-accelerator \
  --compute mlopsv2computecluster \
  --wait
```
**Notes:** Add `--stop_compute` to stop the compute cluster after job completes.

## Skill 2: Check Pipeline Results
**Trigger:** User asks about pipeline outputs, results, or metrics
**Action:**
1. Navigate to `job_outputs_<run>/` directory
2. Check `_exports/unified_rankings.csv` for champion model
3. Check `_exports/phaseb_leaderboard.csv` for Phase B results
4. Check `s5z/named-outputs/` for Phase A baseline aggregate
5. Check `s10/named-outputs/` for Phase C aggregate

## Skill 3: Add Variant Recipes
**Trigger:** User asks to add new recipes, variants, or preprocessing strategies
**Action:**
1. Navigate to `configs/recipes/<task_type>/variant_search/`
2. Create YAML files with structure:
```yaml
recipe_name: variant_<hash>
version: "1.0"
description: "Pipeline Variant: <imp>+<enc>+<scl>+<fs>+<extra>"
task_type: <classification|regression|clustering>
stage3_preprocessing:
  imputation: {method: <method>}
  encoding: {method: <method>}
  scaling: {method: <method>}
stage4_feature_engineering:
  feature_selection: {method: <method>}
variant_metadata:
  source: variant_search
  search_dimensions: [imputation, encoding, scaling, feature_selection]
```
3. Use `hashlib.sha256` for deterministic variant IDs

## Skill 4: Debug Failed Steps
**Trigger:** User reports a pipeline step failure
**Action:**
1. Check Azure ML Studio for the failed step's logs
2. Look at `70_driver_log.txt` for the error traceback
3. Common issues:
   - Missing columns → check `target_column` in config
   - Import errors → check environment definition
   - MLflow errors → ensure HTTPS URI conversion
   - Timeout → increase `flaml_min_budget` or step timeout

## Skill 5: Analyze Model Performance
**Trigger:** User asks why a model/phase performs better or worse
**Action:**
1. Compare metrics across phases in `unified_rankings.csv`
2. Check if `preprocess=False` is set in PyCaret setup (avoid double-preprocessing)
3. Verify FLAML models are individually tracked (not just ensemble)
4. Check variant configurations for meaningful preprocessing differences
5. Look for metric consistency across task types

## Skill 6: Modify Pipeline Configuration
**Trigger:** User asks to change pipeline settings
**Action:**
1. Edit `configs/<task_config>.yml` — NEVER hardcode values in code
2. Key settings: `task_type`, `target_column`, `phases`, `azure_ml`
3. For Phase B: adjust `max_variants`, `selection_strategy`, `min_relevance_score`
4. For Phase C: adjust `n_trials`, `timeout`, search space
5. Validate config matches schema in `src/orchestration/config_schema.py`

## Skill 7: Review Code Changes
**Trigger:** User asks to review or validate code modifications
**Action:**
1. Run `git diff` to see changes
2. Verify task-type isolation (classification fixes don't break regression)
3. Check CLI arg contracts are preserved
4. Verify I/O contracts match component YAMLs
5. Ensure no hardcoded paths, credentials, or task-specific assumptions

## Skill 8: Stop Compute After Job
**Trigger:** User wants to stop compute cluster after job completion
**Action:**
```bash
python pipelines/submit_pipeline.py \
  --config configs/<config>.yml \
  --stop_compute \
  --wait
```
This wraps the job in try/finally to ensure compute is stopped even if the job fails.
