# Runbook: Monitor Azure ML Jobs

Current as of: 2026-08-02

## Check Parent Job

```bash
az ml job show \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query "{name:name,status:status,display_name:display_name,experiment_name:experiment_name}" \
  -o json
```

## Check Child Steps

```bash
az ml job list \
  --parent-job-name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --query "[].{step:display_name,status:status,name:name}" \
  -o table
```

Expected current graph includes:

```text
s01, s02, s03, s04, s05a, s05b, s05z, s06, s08, s09, s10, s12, s13, s14
```

`s00`, legacy forecasting stage `s05t`, `s07`, and `s11` are not expected in current Azure ML Studio runs.

## Download Key Outputs

```bash
az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/mlops_outputs \
  --output-name final_report

az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/mlops_outputs \
  --output-name drift_report

az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/mlops_outputs \
  --output-name drift_baseline
```

After a live `s14` run:

```bash
az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/mlops_outputs \
  --output-name retrain_decision

az ml job download \
  --name <job_name> \
  --resource-group mvpv1 \
  --workspace-name mlops-accelerator \
  --download-path /tmp/mlops_outputs \
  --output-name decision_ledger_record
```

## Common Status Patterns

| Pattern | Meaning | Action |
|---|---|---|
| All expected stages completed | Pipeline succeeded. | Download final/drift/retrain artifacts. |
| `s13` completed but comparison unavailable | No prior valid baseline was supplied. | Use `--drift_baseline_in` for second-cycle proof. |
| `s14` absent | Job was submitted before s14 graph change. | Submit a fresh job from current branch. |
| Parent failed early | Config, component, environment, or input issue. | Inspect failed child step and logs in Studio. |

## Notes

`az ml job stream` is not reliable for pipeline child jobs; it can return a Studio URL instead of log content. Use child step status plus Azure ML Studio logs, then download terminal outputs.

A completed pipeline is Azure runtime evidence for that exact job only. Registered-model loading and deployed-endpoint inference require separate checks.
