import { useState } from "react";
import { JobSelect } from "../components/JobSelect";
import { JsonDetails } from "../components/JsonDetails";
import { MetricTable } from "../components/MetricTable";
import { PageHeader } from "../components/PageHeader";
import { StateBlock } from "../components/StateBlock";
import { useLocalOutputsQuery, useMetricsQuery, useOutputsQuery, useSummaryQuery } from "../hooks/usePipelineQueries";
import { toUserMessage } from "../services/apiClient";
import { formatValue } from "../utils/formatters";

export function Reports() {
  const [jobName, setJobName] = useState("");
  const summary = useSummaryQuery(jobName);
  const metrics = useMetricsQuery(jobName);
  const outputs = useOutputsQuery(jobName);
  const localOutputs = useLocalOutputsQuery();

  return (
    <main className="page">
      <PageHeader eyebrow="Reports" title="Reports" description="Review combined pipeline summaries, model metrics, named outputs, and repo-local report artifacts." />
      <section className="panel"><JobSelect value={jobName} onChange={setJobName} /></section>
      <section className="grid two">
        <section className="panel stack">
          <div className="panel-header"><div><h2>Pipeline summary</h2><p>Combined aggregate reports for the selected job.</p></div><button className="button secondary" onClick={() => summary.refetch()} type="button">Refresh</button></div>
          {summary.isLoading ? <StateBlock title="Loading summary" kind="loading" /> : null}
          {summary.isError ? <StateBlock title="Summary unavailable" kind="error" message={toUserMessage(summary.error)} /> : null}
          {summary.data ? <><div className="metric-grid compact"><div className="metric-card"><small>Status</small><strong>{summary.data.status ?? "-"}</strong></div><div className="metric-card"><small>Task type</small><strong>{summary.data.task_type ?? "-"}</strong></div><div className="metric-card"><small>Champion phase</small><strong>{summary.data.champion_phase ?? "-"}</strong></div><div className="metric-card"><small>Champion score</small><strong>{formatValue(summary.data.champion_score)}</strong></div></div><JsonDetails title="Aggregate reports" value={{ baseline_aggregate: summary.data.baseline_aggregate, phaseb_aggregate: summary.data.phaseb_aggregate, phasec_aggregate: summary.data.phasec_aggregate, final_report: summary.data.final_report }} /></> : null}
        </section>
        <section className="panel stack">
          <div className="panel-header"><div><h2>Named outputs</h2><p>Artifacts exposed by the selected Azure ML job.</p></div><button className="button secondary" onClick={() => outputs.refetch()} type="button">Refresh</button></div>
          {outputs.isLoading ? <StateBlock title="Loading outputs" kind="loading" /> : null}
          {outputs.isError ? <StateBlock title="Outputs unavailable" kind="error" message={toUserMessage(outputs.error)} /> : null}
          <div className="table-wrap"><table><thead><tr><th>Output</th><th>Type</th></tr></thead><tbody>{(outputs.data?.outputs ?? []).map((output) => <tr key={output.name}><td>{output.name}</td><td>{output.type ?? "artifact"}</td></tr>)}</tbody></table></div>
        </section>
      </section>
      <section className="panel stack">
        <div className="panel-header"><div><h2>Live leaderboard</h2><p>Metrics endpoint with summary fallback for problematic artifacts.</p></div></div>
        {metrics.isLoading ? <StateBlock title="Loading metrics" kind="loading" /> : <MetricTable metrics={metrics.data} summary={summary.data} isError={metrics.isError} errorMessage={metrics.isError ? toUserMessage(metrics.error) : undefined} onRetry={() => metrics.refetch()} />}
      </section>
      <section className="panel">
        <div className="panel-header"><div><h2>Local report files</h2><p>{localOutputs.data?.root ?? "outputs"}</p></div><button className="button secondary" onClick={() => localOutputs.refetch()} type="button">Refresh</button></div>
        {localOutputs.isLoading ? <StateBlock title="Loading local outputs" kind="loading" /> : null}
        {localOutputs.isError ? <StateBlock title="Local outputs unavailable" kind="error" message={toUserMessage(localOutputs.error)} /> : null}
        <div className="table-wrap"><table><thead><tr><th>Path</th><th>Kind</th><th>Size</th><th>Modified</th></tr></thead><tbody>{(localOutputs.data?.files ?? []).filter((file) => !file.is_dir).slice(0, 80).map((file) => <tr key={file.relative_path}><td>{file.relative_path}</td><td>{file.kind ?? "file"}</td><td>{formatValue(file.size_bytes)}</td><td>{file.modified_time ?? "-"}</td></tr>)}</tbody></table></div>
      </section>
    </main>
  );
}