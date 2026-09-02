import { useMutation } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { ConfigSelect } from "../components/ConfigSelect";
import { JobSelect } from "../components/JobSelect";
import { JsonDetails } from "../components/JsonDetails";
import { PageHeader } from "../components/PageHeader";
import { StateBlock } from "../components/StateBlock";
import { StatusBadge } from "../components/StatusBadge";
import { useAutoRetrainDecisionsQuery, useAutoRetrainSchedulesQuery } from "../hooks/usePipelineQueries";
import { useApi } from "../services/ApiContext";
import { toUserMessage } from "../services/apiClient";
import { formatValue } from "../utils/formatters";

type BaselineApprovalSource = "uri" | "job";

function normalizeConfigName(configName: string): string {
  return configName.replace(/\.ya?ml$/i, "");
}

function recordMatchesConfig(record: Record<string, unknown>, configName: string): boolean {
  const expected = normalizeConfigName(configName).toLowerCase();
  const recordConfig = typeof record.config_name === "string" ? normalizeConfigName(record.config_name).toLowerCase() : "";
  if (recordConfig && recordConfig === expected) return true;
  return JSON.stringify(record).toLowerCase().includes(expected);
}

function recordLooksLikeApprovedBaseline(record: Record<string, unknown>): boolean {
  const text = JSON.stringify(record).toLowerCase();
  return text.includes("baseline") && text.includes("approved") && (text.includes("azureml://") || text.includes("baseline_uri"));
}

export function AutoRetrain() {
  const { api } = useApi();
  const schedules = useAutoRetrainSchedulesQuery();
  const decisions = useAutoRetrainDecisionsQuery();
  const [configName, setConfigName] = useState("");
  const [jobName, setJobName] = useState("");
  const [baselineUri, setBaselineUri] = useState("");
  const [approvalSource, setApprovalSource] = useState<BaselineApprovalSource>("uri");
  const [scheduleName, setScheduleName] = useState("");
  const [decisionPath, setDecisionPath] = useState("");
  const [forceSubmit, setForceSubmit] = useState(false);
  const [reason, setReason] = useState("Operator approved drift baseline for future auto-retrain.");

  const captureBaseline = useMutation({
    mutationFn: () => api.captureBaseline(jobName),
    onSuccess: (result) => {
      if (result.baseline_path) setBaselineUri(result.baseline_path);
    }
  });

  const approve = useMutation({
    mutationFn: () => api.approveAutoRetrainBaseline({
      config_name: configName,
      baseline_job_name: jobName || null,
      output_baseline_uri: baselineUri.trim(),
      schedule_name: scheduleName || null,
      reason,
    })
  });

  const ledgerRecords = useMemo(
    () => [...(schedules.data?.latest_records ?? []), ...(decisions.data?.records ?? [])],
    [decisions.data, schedules.data]
  );
  const approvedBaselineAvailable = Boolean(
    configName
    && (
      ledgerRecords.some((record) => recordMatchesConfig(record, configName) && recordLooksLikeApprovedBaseline(record))
      || approve.data?.baseline_uri
    )
  );

  const plan = useMutation({
    mutationFn: () => api.autoRetrainPlan({
      config_name: configName,
      decision_path: decisionPath.trim(),
      trigger: "react_ui",
      schedule_name: scheduleName || null,
      force_submit: forceSubmit,
      force_reason: forceSubmit ? reason : null,
    })
  });

  const planDisabled = !configName
    || !decisionPath.trim()
    || !approvedBaselineAvailable
    || plan.isPending;
  const approveDisabled = !configName
    || approve.isPending
    || !jobName
    || !baselineUri.trim();

  return (
    <main className="page">
      <PageHeader
        eyebrow="Automation"
        title="Auto Retrain"
        description="Review schedule intent and recent retrain decisions. Baseline approval and controller planning are guarded operator actions."
      />

      <section className="grid two">
        <section className="panel stack">
          <div className="panel-header">
            <div>
              <h2>Schedules</h2>
              <p>{schedules.data?.total ?? 0} planned schedules.</p>
            </div>
            <button className="button secondary" onClick={() => schedules.refetch()} type="button">Refresh</button>
          </div>
          {schedules.isLoading ? <StateBlock title="Loading schedules" kind="loading" /> : null}
          {schedules.isError ? <StateBlock title="Could not load schedules" kind="error" message={toUserMessage(schedules.error)} /> : null}
          {schedules.data?.azure_error ? (
            <StateBlock
              title="Azure schedule state is unverified"
              message={schedules.data.azure_error}
            />
          ) : null}
          <div className="table-wrap">
            <table>
              <thead><tr><th>Schedule</th><th>Task</th><th>Dataset</th><th>Cadence</th><th>Mode</th><th>Enabled</th></tr></thead>
              <tbody>
                {(schedules.data?.schedules ?? []).map((schedule) => (
                  <tr key={schedule.schedule_name}>
                    <td>{schedule.schedule_name}</td>
                    <td>{schedule.task_type}</td>
                    <td>{schedule.dataset_name}</td>
                    <td>{schedule.cadence} ({schedule.cadence_days}d)</td>
                    <td>{schedule.decision_mode}</td>
                    <td><StatusBadge status={schedule.live_state} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel stack">
          <div className="panel-header">
            <div>
              <h2>Recent decisions</h2>
              <p>{decisions.data?.ledger_path ?? schedules.data?.ledger_path ?? "Decision ledger"}</p>
            </div>
            <button className="button secondary" onClick={() => decisions.refetch()} type="button">Refresh</button>
          </div>
          {decisions.isLoading ? <StateBlock title="Loading decisions" kind="loading" /> : null}
          {decisions.isError ? <StateBlock title="Could not load decisions" kind="error" message={toUserMessage(decisions.error)} /> : null}
          <div className="table-wrap">
            <table>
              <thead><tr><th>Time</th><th>Config</th><th>Decision</th><th>Job</th><th>Baseline</th></tr></thead>
              <tbody>
                {(decisions.data?.records ?? []).slice(0, 50).map((record, index) => (
                  <tr key={index}>
                    <td>{formatValue(record.created_at ?? record.timestamp)}</td>
                    <td>{formatValue(record.config_name)}</td>
                    <td>{formatValue(record.decision ?? record.status)}</td>
                    <td>{formatValue(record.job_name ?? record.baseline_job_name)}</td>
                    <td>{formatValue(record.baseline_uri ?? record.output_baseline_uri)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </section>

      <section className="panel stack">
        <div className="panel-header">
          <div>
            <h2>Advanced baseline actions</h2>
            <p>Use these only when you are intentionally preparing auto-retrain baselines for a config.</p>
          </div>
        </div>
        <ConfigSelect value={configName} onChange={setConfigName} />
        <label className="field">
          <span>Schedule name</span>
          <input value={scheduleName} onChange={(event) => setScheduleName(event.target.value)} placeholder="Optional schedule" />
        </label>
        <label className="field">
          <span>S14 decision path</span>
          <input
            required
            value={decisionPath}
            onChange={(event) => setDecisionPath(event.target.value)}
            placeholder="retrain_decision.json"
          />
        </label>

        <JobSelect value={jobName} onChange={setJobName} label="Producing Azure ML job" />

        <div className="segmented-control" role="group" aria-label="Baseline approval source">
          <button className={approvalSource === "uri" ? "active" : ""} onClick={() => setApprovalSource("uri")} type="button">Baseline URI</button>
          <button className={approvalSource === "job" ? "active" : ""} onClick={() => setApprovalSource("job")} type="button">Discover from job</button>
        </div>
        {approvalSource === "uri" ? (
          <label className="field">
            <span>Drift baseline URI</span>
            <input value={baselineUri} onChange={(event) => setBaselineUri(event.target.value)} placeholder="azureml://.../drift_baseline" />
          </label>
        ) : (
          <div className="stack">
            <div className="button-row">
              <button className="button secondary" disabled={!jobName || captureBaseline.isPending} onClick={() => captureBaseline.mutate()} type="button">
                {captureBaseline.isPending ? "Finding baseline..." : "Find baseline URI"}
              </button>
            </div>
            {captureBaseline.isError ? <StateBlock title="Baseline URI not found" kind="error" message={toUserMessage(captureBaseline.error)} /> : null}
            {captureBaseline.data && captureBaseline.data.output_present && !captureBaseline.data.baseline_path ? (
              <StateBlock title="Baseline output found, URI unavailable" message="Azure ML confirmed the drift_baseline output exists, but the API could not expose a reusable URI. Paste the drift_baseline AzureML URI manually to approve it." />
            ) : null}
            {captureBaseline.data && !captureBaseline.data.output_present ? (
              <StateBlock title="No drift baseline output found" message="Choose a completed job with a drift_baseline output, or paste the AzureML baseline URI manually." />
            ) : null}
          </div>
        )}
        {baselineUri.trim() ? (
          <StateBlock title="Baseline URI ready" message={baselineUri.trim()} />
        ) : null}
        <label className="field">
          <span>Operator reason</span>
          <textarea value={reason} onChange={(event) => setReason(event.target.value)} rows={3} />
        </label>
        <div className="toggle-row">
          <label><input type="checkbox" checked={forceSubmit} onChange={(event) => setForceSubmit(event.target.checked)} /> Force controller submit in dry-run plan</label>
        </div>
        {!approvedBaselineAvailable ? (
          <StateBlock
            title="Controller plan requires an approved baseline"
            message="Approve a drift baseline, refresh decisions, and provide the relative S14 decision artifact path before building a plan."
          />
        ) : null}
        <div className="button-row">
          <button className="button primary" disabled={approveDisabled} onClick={() => approve.mutate()} type="button">Approve baseline</button>
          <button className="button secondary" disabled={planDisabled} onClick={() => plan.mutate()} type="button">Build dry-run plan</button>
        </div>
        {plan.isError ? <StateBlock title="Plan unavailable" kind="error" message={toUserMessage(plan.error)} /> : null}
        {approve.isError && baselineUri.trim() ? <StateBlock title="Baseline approval unavailable" kind="error" message={toUserMessage(approve.error)} /> : null}
        {approve.data ? <JsonDetails title="Baseline approval" value={approve.data} /> : null}
        {plan.data ? <JsonDetails title="Controller plan" value={plan.data} /> : null}
        {schedules.data ? <JsonDetails title="Latest schedule ledger records" value={schedules.data.latest_records} /> : null}
      </section>
    </main>
  );
}
