import { FormEvent, ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { KeyRound, LogIn } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { apiUrl, MLOpsApiClient, toUserMessage, type UserIdentity } from "../services/apiClient";
import { ApiContext } from "../services/ApiContext";
import { createEntraSession, type EntraConfig } from "../services/entraSession";

type AuthConfig = EntraConfig | { mode: "api_key" };
type EntraSession = Awaited<ReturnType<typeof createEntraSession>>;

export function AuthenticationGate({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const [config, setConfig] = useState<AuthConfig | null>(null);
  const [entra, setEntra] = useState<EntraSession | null>(null);
  const [session, setSession] = useState<{ api: MLOpsApiClient; identity: UserIdentity; key: string } | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const generation = useRef(0);

  useEffect(() => {
    let cancelled = false;
    async function initialize() {
      const response = await fetch(apiUrl("/api/v1/auth/config"), { cache: "no-store" });
      if (!response.ok) throw new Error("Sign-in configuration is unavailable.");
      const data: AuthConfig = await response.json();
      if (data.mode !== "api_key" && data.mode !== "entra") throw new Error("Unsupported sign-in configuration.");
      const auth = data.mode === "entra" ? await createEntraSession(data) : null;
      if (!cancelled) { setConfig(data); setEntra(auth); }
    }
    initialize().catch((failure) => { if (!cancelled) setError(toUserMessage(failure)); });
    return () => { cancelled = true; };
  }, []);

  const signOut = useCallback(() => {
    generation.current += 1;
    setSession(null);
    setDraft("");
    queryClient.cancelQueries();
    queryClient.clear();
    void entra?.clear();
  }, [entra, queryClient]);

  const refreshIdentity = useCallback(async () => {
    if (!session) return;
    const current = generation.current;
    try {
      const identity = await session.api.identity();
      if (identity.mode !== session.identity.mode || identity.object_id !== session.identity.object_id || identity.tenant_id !== session.identity.tenant_id) throw new Error("Account changed; sign in again.");
      if (current === generation.current) setSession((active) => active ? { ...active, identity } : null);
    } catch (failure) {
      if (current === generation.current) { signOut(); setError(toUserMessage(failure)); }
    }
  }, [session, signOut]);

  useEffect(() => {
    if (!session) return;
    const timer = window.setInterval(() => { void refreshIdentity(); }, 60_000);
    const refresh = () => { void refreshIdentity(); };
    window.addEventListener("focus", refresh);
    return () => { window.clearInterval(timer); window.removeEventListener("focus", refresh); };
  }, [session, refreshIdentity]);

  async function signIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!config || busy) return;
    const current = generation.current;
    setBusy(true);
    setError("");
    try {
      if (config.mode === "entra") await entra!.login();
      const api = new MLOpsApiClient(config.mode === "entra" ? entra!.token : draft.trim());
      const identity = await api.identity();
      if (identity.mode !== config.mode || !Array.isArray(identity.roles)) throw new Error("Invalid workspace identity response.");
      if (current === generation.current) {
        queryClient.clear();
        setSession({ api, identity, key: config.mode === "api_key" ? draft.trim() : "" });
        setDraft("");
      }
    } catch (failure) {
      await entra?.clear();
      setError(toUserMessage(failure));
    } finally { setBusy(false); }
  }

  if (!session) return (
    <main className="auth-shell">
      <form className="auth-panel" onSubmit={signIn}>
        <div className="auth-icon"><KeyRound size={22} /></div>
        <h1>MLOps V3 Operations</h1>
        {error && <p role="alert">{error}</p>}
        {!config && !error && <p role="status">Loading sign-in...</p>}
        {config?.mode === "api_key" && <>
          <label htmlFor="api-key">API key</label>
          <input id="api-key" value={draft} onChange={(event) => setDraft(event.target.value)} type="password" autoComplete="off" />
        </>}
        <button className="button primary" disabled={!config || busy || (config.mode === "api_key" && !draft.trim())} type="submit">
          <LogIn size={17} />{busy ? "Signing in..." : config?.mode === "entra" ? "Sign in with Microsoft" : "Sign in"}
        </button>
        {!config && error && <button type="button" className="button secondary" onClick={() => window.location.reload()}>Retry</button>}
      </form>
    </main>
  );
  return <ApiContext.Provider value={{ api: session.api, apiKey: session.key, identity: session.identity, setApiKey: signOut, signOut, refreshIdentity }}>{children}</ApiContext.Provider>;
}
