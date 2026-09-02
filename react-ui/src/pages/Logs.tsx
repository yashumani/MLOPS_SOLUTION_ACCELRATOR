import { useState } from "react";
import { JobSelect } from "../components/JobSelect";
import { PageHeader } from "../components/PageHeader";
import { StateBlock } from "../components/StateBlock";
import { StatusBadge } from "../components/StatusBadge";
import { useJobQuery } from "../hooks/usePipelineQueries";
import { toUserMessage } from "../services/apiClient";

export function Logs() {
  const [jobName, setJobName] = useState("");
  const job = useJobQuery(jobName);
  const events = job.data?.steps.flatMap((step) => [
    step.start_time ? { time: step.start_time, stage: step.display_name ?? step.name, message: `${step.status} started` } : null,
    step.end_time ? { time: step.end_time, stage: step.display_name ?? step.name, message: `${step.status} ended` } : null
  ].filter(Boolean) as Array<{ time: string; stage: string; message: string }>).sort((left, right) => left.time.localeCompare(right.time)) ?? [];

  return (
    <main className="page">
      <PageHeader eyebrow="Execution trace" title="Logs" description="Follow job status, child step state, and reconstructed timeline events for a selected Azure ML job." actions={<button className="button secondary" onClick={() => job.refetch()} type="button">Refresh</button>} />
      <section className="panel"><JobSelect value={jobName} onChange={setJobName} /></section>
      {job.isLoading ? <StateBlock title="Loading job trace" kind="loading" /> : null}
      {job.isError ? <StateBlock title="Could not load job trace" kind="error" message={toUserMessage(job.error)} /> : null}
      {job.data ? <section className="panel stack"><div className="metric-grid compact"><div className="metric-card"><small>Status</small><strong>{job.data.status}</strong></div><div className="metric-card"><small>Experiment</small><strong>{job.data.experiment_name ?? "-"}</strong></div><div className="metric-card"><small>Start</small><strong>{job.data.start_time ?? "-"}</strong></div><div className="metric-card"><small>End</small><strong>{job.data.end_time ?? "-"}</strong></div></div><div className="timeline">{job.data.steps.map((step) => <div className="timeline-row" key={step.name}><div><strong>{step.display_name ?? step.stage_key ?? step.name}</strong><small>{step.name}</small></div><StatusBadge status={step.status} /><span>{step.start_time ?? "-"}</span><span>{step.end_time ?? "-"}</span></div>)}</div></section> : null}
      <section className="panel"><div className="panel-header"><div><h2>Timeline events</h2><p>{events.length} reconstructed events.</p></div></div><div className="table-wrap"><table><thead><tr><th>Time</th><th>Stage</th><th>Event</th></tr></thead><tbody>{events.map((event, index) => <tr key={`${event.time}-${index}`}><td>{event.time}</td><td>{event.stage}</td><td>{event.message}</td></tr>)}</tbody></table></div></section>
    </main>
  );
}