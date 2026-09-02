import type { OutputContentResponse } from "../types/api";
import { summarizeJsonArtifact } from "../utils/artifacts";
import { formatValue } from "../utils/formatters";

interface OutputPresenterProps {
  content: OutputContentResponse;
}

export function OutputPresenter({ content }: OutputPresenterProps) {
  const summary = content.json_content !== undefined && content.json_content !== null
    ? summarizeJsonArtifact(content.output_name, content.json_content)
    : null;

  return (
    <div className="stack">
      <section className="panel subtle-panel">
        <div className="panel-header">
          <div>
            <h3>{summary?.title ?? content.output_name}</h3>
            <p>{summary?.description ?? content.primary_file ?? "Preview available for this output."}</p>
          </div>
          {content.truncated ? <span className="status-badge status-warn">Truncated</span> : null}
        </div>

        {summary && summary.fields.length > 0 ? (
          <div className="metric-grid">
            {summary.fields.map((field) => (
              <div className="metric-card" key={field.label}>
                <small>{field.label}</small>
                <strong>{field.value}</strong>
              </div>
            ))}
          </div>
        ) : null}

        {content.csv_preview && content.csv_preview.length > 0 ? <DataTable rows={content.csv_preview} /> : null}
        {summary && summary.tableRows.length > 0 ? <DataTable rows={summary.tableRows} /> : null}
        {content.text_preview ? <pre className="text-preview">{content.text_preview}</pre> : null}
      </section>

      {content.files.length > 0 ? (
        <section className="panel">
          <h3>Files</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Name</th><th>Kind</th><th>Size</th><th>Path</th></tr>
              </thead>
              <tbody>
                {content.files.map((file) => (
                  <tr key={file.relative_path}>
                    <td>{file.name}</td>
                    <td>{file.kind}</td>
                    <td>{formatValue(file.size_bytes)}</td>
                    <td>{file.relative_path}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {content.json_content !== undefined && content.json_content !== null ? (
        <details className="advanced-json">
          <summary>Advanced: raw JSON</summary>
          <pre>{JSON.stringify(content.json_content, null, 2)}</pre>
        </details>
      ) : null}
    </div>
  );
}

function DataTable({ rows }: { rows: Array<Record<string, unknown>> }) {
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).slice(0, 12);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.slice(0, 25).map((row, index) => (
            <tr key={index}>
              {columns.map((column) => <td key={column}>{formatValue(row[column])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}