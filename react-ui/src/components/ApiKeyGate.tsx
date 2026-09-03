import { FormEvent, ReactNode, useMemo, useState } from "react";
import { KeyRound } from "lucide-react";
import { MLOpsApiClient } from "../services/apiClient";
import { ApiContext } from "../services/ApiContext";

interface ApiKeyGateProps {
  children: ReactNode;
}

export function ApiKeyGate({ children }: ApiKeyGateProps) {
  const [apiKey, setApiKey] = useState("");
  const [draft, setDraft] = useState("");
  const api = useMemo(() => new MLOpsApiClient(apiKey), [apiKey]);

  function submitApiKey(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setApiKey(draft.trim());
  }

  if (!apiKey) {
    return (
      <main className="auth-shell">
        <form className="auth-panel" onSubmit={submitApiKey}>
          <div className="auth-icon"><KeyRound size={22} /></div>
          <h1>MLOps V3 Operations</h1>
          <p>Enter the API key for this session. The key is kept in memory and is not stored in browser local storage.</p>
          <label htmlFor="api-key">API key</label>
          <input
            id="api-key"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Paste API key"
            type="password"
            autoComplete="off"
          />
          <button className="button primary" disabled={!draft.trim()} type="submit">
            Start session
          </button>
        </form>
      </main>
    );
  }

  return <ApiContext.Provider value={{ api, apiKey, setApiKey }}>{children}</ApiContext.Provider>;
}
