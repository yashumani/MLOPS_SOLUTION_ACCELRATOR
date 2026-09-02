import { getRuntimeConfig } from "../services/runtimeConfig";
import { useApi } from "../services/ApiContext";

export function Settings() {
  const config = getRuntimeConfig();
  const { setApiKey } = useApi();
  return (
    <main className="page">
      <section className="page-heading">
        <div>
          <p className="eyebrow">Runtime</p>
          <h1>Settings</h1>
          <p>Inspect the browser runtime configuration for this UI session.</p>
        </div>
      </section>
      <section className="panel">
        <div className="metric-grid">
          <div className="metric-card"><small>API base URL</small><strong>{config.apiBaseUrl}</strong></div>
          <div className="metric-card"><small>UI base URL</small><strong>{config.uiBaseUrl}</strong></div>
          <div className="metric-card"><small>Environment</small><strong>{config.environment}</strong></div>
          <div className="metric-card"><small>API key mode</small><strong>{config.apiKey ? "Runtime config" : "In-memory prompt"}</strong></div>
        </div>
        <button className="button secondary" onClick={() => setApiKey("")} type="button">Clear API key for this session</button>
      </section>
    </main>
  );
}