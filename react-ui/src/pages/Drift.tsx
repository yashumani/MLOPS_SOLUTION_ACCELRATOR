import { useMemo, useState } from "react";
import { JobSelect } from "../components/JobSelect";
import { JsonDetails } from "../components/JsonDetails";
import { PageHeader } from "../components/PageHeader";
import { StateBlock } from "../components/StateBlock";
import { StatusBadge } from "../components/StatusBadge";
import { useDriftQuery } from "../hooks/usePipelineQueries";
import { toUserMessage } from "../services/apiClient";
import { formatValue } from "../utils/formatters";

export function Drift() {
  const [jobName, setJobName] = useState("");
  const drift = useDriftQuery(jobName);
  const severeCount = useMemo(() => drift.data?.features.filter((feature) => feature.severity === "severe").length ?? 0, [drift.data]);
  const moderateCount = useMemo(() => drift.data?.features.filter((feature) => feature.severity === "moderate").length ?? 0, [drift.data]);

  return (
    <main className="page">
      <PageHeader eyebrow="Drift operations" title="Drift Monitor" description="Inspect drift summaries, feature PSI scores, retrain recommendations, and baseline metadata for a selected job." />
      <section className="panel"><JobSelect value={jobName} onChange={setJobName} /></section>
      {drift.isLoading ? <StateBlock title="Loading drift report" kind="loading" /> : null}
      {drift.isError ? <StateBlock title="Drift report unavailable" kind="error" message={toUserMessage(drift.error)} actionLabel="Retry" onAction={() => drift.refetch()} /> : null}
      {drift.data ? (
        <section className="stack">
          <section className="panel">
            <div className="metric-grid">
              <div className="metric-card"><small>Stability score</small><strong>{formatValue(drift.data.stability_score)}</strong></div>
              <div className="metric-card"><small>Overall drift</small><strong>{drift.data.overall_drift_detected ? "Detected" : "Not detected"}</strong></div>
              <div className="metric-card"><small>Recommended cadence</small><strong>{drift.data.recommended_cadence ?? "-"}</strong></div>
              <div className="metric-card"><small>Moderate / severe</small><strong>{moderateCount} / {severeCount}</strong></div>
            </div>
            <div className="metric-grid compact">
              <div className="metric-card"><small>Dataset</small><strong>{drift.data.dataset_name ?? "-"}</strong></div>
              <div className="metric-card"><small>Task type</small><strong>{drift.data.task_type ?? "-"}</strong></div>
              <div className="metric-card"><small>Comparison</small><strong>{drift.data.comparison_available ? "Available" : "Unavailable"}</strong></div>
              <div className="metric-card"><small>Baseline</small><strong>{drift.data.baseline_status ?? "-"}</strong></div>
            </div>
            {drift.data.cadence_rationale ? <p>{drift.data.cadence_rationale}</p> : null}
          </section>
          {drift.data.warnings.length > 0 ? <StateBlock title="Drift warnings" kind="error" message={drift.data.warnings.join(" ")} /> : null}
          <section className="panel"><div className="table-wrap"><table><thead><tr><th>Feature</th><th>PSI</th><th>Severity</th><th>Drift</th></tr></thead><tbody>{drift.data.features.map((feature) => <tr key={feature.feature}><td>{feature.feature}</td><td>{formatValue(feature.psi)}</td><td><StatusBadge status={feature.severity} /></td><td>{feature.drift_detected ? "Yes" : "No"}</td></tr>)}</tbody></table></div></section>
          <section className="grid two"><JsonDetails title="Baseline metadata" value={drift.data.baseline_metadata ?? {}} /><JsonDetails title="Auto-retrain decision" value={drift.data.auto_retrain_decision ?? {}} /></section>
        </section>
      ) : null}
    </main>
  );
}