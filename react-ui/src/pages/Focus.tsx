import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { ExternalLink, RefreshCcw, Search } from "lucide-react";
import { JobPicker } from "../components/JobPicker";
import { MetricTable } from "../components/MetricTable";
import { OutputPresenter } from "../components/OutputPresenter";
import { StateBlock } from "../components/StateBlock";
import { StatusBadge } from "../components/StatusBadge";
import {
  useDriftQuery,
  useJobQuery,
  useMetricsQuery,
  useOutputContentQuery,
  useOutputsQuery,
  useSummaryQuery
} from "../hooks/usePipelineQueries";
import { toUserMessage } from "../services/apiClient";
import { formatValue } from "../utils/formatters";

const tabs = ["overview", "leaderboard", "outputs", "drift", "logs"] as const;
type FocusTab = (typeof tabs)[number];

export function Focus() {
  const { jobName } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [pickerOpen, setPickerOpen] = useState(!jobName);
  const selectedTab = (searchParams.get("tab") as FocusTab | null) ?? "overview";
  const activeTab = tabs.includes(selectedTab) ? selectedTab : "overview";

  const job = useJobQuery(jobName);

  useEffect(() => {
    if (!jobName) setPickerOpen(true);
  }, [jobName]);

  function setTab(tab: FocusTab) {
    setSearchParams({ tab });
  }

  function clearJob() {
    navigate("/focus");
    setPickerOpen(true);
  }

  if (!jobName) {
    return (
      <main className="page">
        <section className="empty-focus">
          <Search size={28} />
          <h1>Select a job</h1>
          <p>Open a completed or running Azure ML job to review its timeline, leaderboard, outputs, drift, and logs.</p>
          <button className="button primary" onClick={() => setPickerOpen(true)} type="button">Choose job</button>
        </section>
        <JobPicker open={pickerOpen} onClose={() => setPickerOpen(false)} />
      </main>
    );
  }

  return (
    <main className="page">
      <section className="focus-header">
        <div>
          <p className="eyebrow">Focus cockpit</p>
          <h1>{job.data?.display_name ?? jobName}</h1>
          <p>{job.data?.experiment_name ?? "Azure ML pipeline job"}</p>
        </div>
        <div className="header-actions">
          <StatusBadge status={job.data?.status ?? (job.isLoading ? "Loading" : "Unknown")} />
          {job.data?.studio_url ? <a className="button secondary" href={job.data.studio_url} target="_blank" rel="noreferrer"><ExternalLink size={16} /> Studio</a> : null}
          <button className="button secondary" onClick={() => job.refetch()} type="button"><RefreshCcw size={16} /> Refresh</button>
          <button className="button primary" onClick={clearJob} type="button">Change Job</button>
        </div>
      </section>

      {job.isError ? <StateBlock title="Could not load job" kind="error" message={toUserMessage(job.error)} actionLabel="Retry" onAction={() => job.refetch()} /> : null}

      <nav className="tabs" aria-label="Focus sections">
        {tabs.map((tab) => (
          <button key={tab} className={activeTab === tab ? "active" : ""} onClick={() => setTab(tab)} type="button">
            {tab[0].toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </nav>

      {activeTab === "overview" ? <OverviewTab jobName={jobName} /> : null}
      {activeTab === "leaderboard" ? <LeaderboardTab jobName={jobName} /> : null}
      {activeTab === "outputs" ? <OutputsTab jobName={jobName} /> : null}
      {activeTab === "drift" ? <DriftTab jobName={jobName} /> : null}
      {activeTab === "logs" ? <LogsTab jobName={jobName} /> : null}

      <JobPicker open={pickerOpen} onClose={() => setPickerOpen(false)} />
    </main>
  );
}

function OverviewTab({ jobName }: { jobName: string }) {
  const job = useJobQuery(jobName);
  if (job.isLoading) return <StateBlock title="Loading job timeline" kind="loading" />;
  if (!job.data) return null;
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h2>Stage timeline</h2>
          <p>All child stages reported by Azure ML.</p>
        </div>
      </div>
      <div className="timeline">
        {job.data.steps.map((step) => (
          <div className="timeline-row" key={step.name}>
            <div>
              <strong>{step.display_name ?? step.stage_key ?? step.name}</strong>
              <small>{step.name}</small>
            </div>
            <StatusBadge status={step.status} />
            <span>{step.start_time ?? "-"}</span>
            <span>{step.end_time ?? "-"}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function LeaderboardTab({ jobName }: { jobName: string }) {
  const metrics = useMetricsQuery(jobName);
  const summary = useSummaryQuery(jobName);
  if (metrics.isLoading) return <StateBlock title="Loading leaderboard" kind="loading" />;
  return (
    <MetricTable
      metrics={metrics.data}
      summary={summary.data}
      isError={metrics.isError}
      errorMessage={metrics.isError ? toUserMessage(metrics.error) : undefined}
      onRetry={() => metrics.refetch()}
    />
  );
}

function OutputsTab({ jobName }: { jobName: string }) {
  const outputs = useOutputsQuery(jobName);
  const [selectedOutput, setSelectedOutput] = useState<string | undefined>();
  const firstOutput = outputs.data?.outputs[0]?.name;
  const activeOutput = selectedOutput ?? firstOutput;
  const content = useOutputContentQuery(jobName, activeOutput);

  useEffect(() => {
    if (!selectedOutput && firstOutput) setSelectedOutput(firstOutput);
  }, [firstOutput, selectedOutput]);

  if (outputs.isLoading) return <StateBlock title="Loading outputs" kind="loading" />;
  if (outputs.isError) return <StateBlock title="Outputs unavailable" kind="error" message={toUserMessage(outputs.error)} actionLabel="Retry" onAction={() => outputs.refetch()} />;
  if (!outputs.data || outputs.data.outputs.length === 0) return <StateBlock title="No outputs found" message="This job did not expose named outputs through the API." />;

  return (
    <section className="outputs-layout">
      <aside className="output-list">
        {outputs.data.outputs.map((output) => (
          <button key={output.name} className={activeOutput === output.name ? "active" : ""} onClick={() => setSelectedOutput(output.name)} type="button">
            <strong>{output.name}</strong>
            <small>{output.type ?? "artifact"}</small>
          </button>
        ))}
      </aside>
      <div className="output-content">
        {content.isLoading ? <StateBlock title="Loading output preview" kind="loading" /> : null}
        {content.isError ? <StateBlock title="Preview unavailable" kind="error" message={toUserMessage(content.error)} actionLabel="Retry" onAction={() => content.refetch()} /> : null}
        {content.data ? <OutputPresenter content={content.data} /> : null}
      </div>
    </section>
  );
}

function DriftTab({ jobName }: { jobName: string }) {
  const drift = useDriftQuery(jobName);
  const severeCount = useMemo(() => drift.data?.features.filter((feature) => feature.severity === "severe").length ?? 0, [drift.data]);
  if (drift.isLoading) return <StateBlock title="Loading drift analysis" kind="loading" />;
  if (drift.isError) return <StateBlock title="Drift data unavailable" kind="error" message={toUserMessage(drift.error)} actionLabel="Retry" onAction={() => drift.refetch()} />;
  if (!drift.data) return null;
  return (
    <section className="panel">
      <div className="metric-grid">
        <div className="metric-card"><small>Stability score</small><strong>{formatValue(drift.data.stability_score)}</strong></div>
        <div className="metric-card"><small>Overall drift</small><strong>{drift.data.overall_drift_detected ? "Detected" : "Not detected"}</strong></div>
        <div className="metric-card"><small>Comparison</small><strong>{drift.data.comparison_available ? "Available" : "Unavailable"}</strong></div>
        <div className="metric-card"><small>Severe features</small><strong>{severeCount}</strong></div>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Feature</th><th>PSI</th><th>Severity</th><th>Drift</th></tr></thead>
          <tbody>
            {drift.data.features.slice(0, 50).map((feature) => (
              <tr key={feature.feature}>
                <td>{feature.feature}</td>
                <td>{formatValue(feature.psi)}</td>
                <td><StatusBadge status={feature.severity} /></td>
                <td>{feature.drift_detected ? "Yes" : "No"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function LogsTab({ jobName }: { jobName: string }) {
  const job = useJobQuery(jobName);
  const events = job.data?.steps.flatMap((step) => [
    step.start_time ? { time: step.start_time, stage: step.display_name ?? step.stage_key ?? step.name, message: `${step.status} started` } : null,
    step.end_time ? { time: step.end_time, stage: step.display_name ?? step.stage_key ?? step.name, message: `${step.status} ended` } : null
  ].filter(Boolean) as Array<{ time: string; stage: string; message: string }>).sort((left, right) => left.time.localeCompare(right.time)) ?? [];

  if (job.isLoading) return <StateBlock title="Loading execution trace" kind="loading" />;
  if (job.isError) return <StateBlock title="Execution trace unavailable" kind="error" message={toUserMessage(job.error)} actionLabel="Retry" onAction={() => job.refetch()} />;
  if (!job.data) return null;

  return (
    <section className="stack">
      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Execution trace</h2>
            <p>Job and child-step activity from the Azure ML status contract.</p>
          </div>
          <button className="button secondary" onClick={() => job.refetch()} type="button"><RefreshCcw size={16} /> Refresh</button>
        </div>
        <div className="metric-grid compact">
          <div className="metric-card"><small>Status</small><strong>{job.data.status}</strong></div>
          <div className="metric-card"><small>Experiment</small><strong>{job.data.experiment_name ?? "-"}</strong></div>
          <div className="metric-card"><small>Start</small><strong>{job.data.start_time ?? "-"}</strong></div>
          <div className="metric-card"><small>End</small><strong>{job.data.end_time ?? "-"}</strong></div>
        </div>
        <div className="timeline">
          {job.data.steps.map((step) => (
            <div className="timeline-row" key={step.name}>
              <div>
                <strong>{step.display_name ?? step.stage_key ?? step.name}</strong>
                <small>{step.name}</small>
              </div>
              <StatusBadge status={step.status} />
              <span>{step.start_time ?? "-"}</span>
              <span>{step.end_time ?? "-"}</span>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <div className="panel-header"><div><h2>Timeline events</h2><p>{events.length} reconstructed events.</p></div></div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Time</th><th>Stage</th><th>Event</th></tr></thead>
            <tbody>
              {events.map((event, index) => <tr key={`${event.time}-${index}`}><td>{event.time}</td><td>{event.stage}</td><td>{event.message}</td></tr>)}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}