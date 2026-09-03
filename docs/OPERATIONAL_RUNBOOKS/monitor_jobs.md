# Runbook: Monitor Azure ML Jobs

Current as of: 2026-09-03

## Monitor A Qualification Wave

Use the canonical JSON written by `scripts/batch_submit_all.py`. The monitor
reloads an in-progress manifest, queries the same submitted job handles, and
writes a structured status snapshot after every poll.

```powershell
python scripts\monitor_batch.py `
  --submissions '<wave-submission-evidence.json>' `
  --sub '93044a08-5661-4f1b-b424-5eafe066a9d1' `
  --rg 'mvpv1' `
  --ws 'mlops-accelerator' `
  --interval 5 `
  --max-hours 8 `
  --output-dir '<wave-monitor-evidence>'
```

Monitor outputs are:

- `monitor-summary.json`: current machine-readable state and every job status.
- `monitor_status.log`: append-only observations.
- `BATCH_DONE.txt`: written only when all expected jobs are terminal.
- `FAILURES.txt`: written when submission or terminal job failure is observed.
- `BATCH_TIMEOUT.txt`: written when terminal evidence is not reached in time.

Exit code `0` means every expected parent is `Completed`. Exit code `1` means a
submission or terminal job failed, `2` means the manifest/context is invalid,
`3` means the bounded monitor timed out, and `4` means a `--once` poll was not
terminal. A timeout or query error is not permission to resubmit; resume
monitoring the same job handles.

Parent completion is only the first acceptance gate. Download and validate the
execution, split, final evaluation, quality, registration, drift, and S14
artifacts, then run registered-model inference against the exact numeric model
version before accepting the scenario. Use
`OPERATIONAL_RUNBOOKS/qualification_evidence.md` to bind those outputs into a
fail-closed scenario or full-matrix acceptance report.

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
