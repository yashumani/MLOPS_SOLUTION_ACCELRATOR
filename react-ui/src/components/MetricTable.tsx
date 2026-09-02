import type { MetricsResponse, PipelineSummaryResponse } from "../types/api";
import { formatValue, metricHint, metricLabel } from "../utils/formatters";
import { StateBlock } from "./StateBlock";

interface MetricTableProps {
  metrics?: MetricsResponse;
  summary?: PipelineSummaryResponse;
  isError: boolean;
  errorMessage?: string;
  onRetry: () => void;
}

export function MetricTable({ metrics, summary, isError, errorMessage, onRetry }: MetricTableProps) {
  if (isError) {
    return (
      <div className="stack">
        <StateBlock
          title="No metrics yet"
          message={errorMessage ?? "Metrics could not be loaded from the API. Aggregate reports may still be usable below."}
          kind="error"
          actionLabel="Retry metrics"
          onAction={onRetry}
        />
        {summary ? <SummaryFallback summary={summary} /> : null}
      </div>
    );
  }

  if (!metrics || metrics.models.length === 0) {
    return <StateBlock title="No leaderboard rows" message="The metrics endpoint responded, but no model rows were returned." />;
  }

  const metricNames = Array.from(new Set(metrics.models.flatMap((model) => Object.keys(model.metrics)))).slice(0, 10);

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Phase</th>
            <th>Model</th>
            <th>Engine</th>
            {metricNames.map((metric) => (
              <th key={metric} title={metricHint(metric)}>{metricLabel(metric)}</th>
            ))}
            <th>Champion</th>
          </tr>
        </thead>
        <tbody>
          {metrics.models.map((model, index) => (
            <tr key={`${model.phase}-${model.model_name}-${index}`} className={model.is_champion ? "champion-row" : ""}>
              <td>{model.phase ?? "-"}</td>
              <td>{model.model_name}</td>
              <td>{model.engine ?? "-"}</td>
              {metricNames.map((metric) => <td key={metric}>{formatValue(model.metrics[metric])}</td>)}
              <td>{model.is_champion ? "Yes" : "No"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SummaryFallback({ summary }: { summary: PipelineSummaryResponse }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <div>
          <h3>Summary fallback</h3>
          <p>Available aggregate report fields are shown while metrics recover.</p>
        </div>
      </div>
      <div className="metric-grid">
        <div className="metric-card">
          <small>Task type</small>
          <strong>{summary.task_type ?? "-"}</strong>
        </div>
        <div className="metric-card">
          <small>Champion phase</small>
          <strong>{summary.champion_phase ?? "-"}</strong>
        </div>
        <div className="metric-card">
          <small>Champion score</small>
          <strong>{formatValue(summary.champion_score)}</strong>
        </div>
        <div className="metric-card">
          <small>Available outputs</small>
          <strong>{summary.available_outputs.length}</strong>
        </div>
      </div>
    </section>
  );
}