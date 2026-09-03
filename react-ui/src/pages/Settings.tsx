import { getRuntimeConfig } from "../services/runtimeConfig";
import { useApi } from "../services/ApiContext";
import { LogOut } from "lucide-react";

export function Settings() {
  const config = getRuntimeConfig();
  const { identity, signOut } = useApi();
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
          <div className="metric-card"><small>Sign-in</small><strong>{identity.mode === "entra" ? "Microsoft Entra" : "API key"}</strong></div>
          <div className="metric-card"><small>Role</small><strong>{identity.roles.join(", ")}</strong></div>
        </div>
        <button className="button secondary" onClick={signOut} type="button"><LogOut size={17} />Sign out</button>
      </section>
    </main>
  );
}
