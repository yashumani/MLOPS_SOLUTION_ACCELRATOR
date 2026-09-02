import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Save, ShieldCheck } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { JsonDetails } from "../components/JsonDetails";
import { PageHeader } from "../components/PageHeader";
import { StateBlock } from "../components/StateBlock";
import { useConfigQuery, useConfigsQuery } from "../hooks/usePipelineQueries";
import { useApi } from "../services/ApiContext";
import { toUserMessage } from "../services/apiClient";
import type { ConfigPreviewResponse, ConfigValidationResponse } from "../types/api";

function parseDraft(draft: string): Record<string, unknown> {
  const parsed = JSON.parse(draft) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Config draft must be a JSON object.");
  return parsed as Record<string, unknown>;
}

export function Configs() {
  const { api } = useApi();
  const queryClient = useQueryClient();
  const configs = useConfigsQuery();
  const [selectedName, setSelectedName] = useState("");
  const [draft, setDraft] = useState("");
  const [copyName, setCopyName] = useState("");
  const [draftError, setDraftError] = useState<string | null>(null);
  const detail = useConfigQuery(selectedName);

  useEffect(() => {
    if (!selectedName && configs.data?.configs[0]?.config_name) setSelectedName(configs.data.configs[0].config_name);
  }, [configs.data, selectedName]);

  useEffect(() => {
    if (detail.data?.content) {
      setDraft(JSON.stringify(detail.data.content, null, 2));
      setDraftError(null);
      setCopyName(`${detail.data.config_name}_copy`);
    }
  }, [detail.data]);

  const validation = useMutation<ConfigValidationResponse, unknown, Record<string, unknown>>({
    mutationFn: (content) => api.validateConfig(content)
  });
  const preview = useMutation<ConfigPreviewResponse, unknown, Record<string, unknown>>({
    mutationFn: (content) => api.previewConfig(content, selectedName)
  });
  const saveCopy = useMutation({
    mutationFn: (content: Record<string, unknown>) => api.createConfig(copyName.trim(), content),
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: ["configs"] });
      setSelectedName(created.config_name);
    }
  });

  const selectedSummary = useMemo(() => configs.data?.configs.find((config) => config.config_name === selectedName), [configs.data, selectedName]);

  function runMutation(action: "validate" | "preview" | "copy") {
    setDraftError(null);
    try {
      const content = parseDraft(draft);
      if (action === "validate") validation.mutate(content);
      if (action === "preview") preview.mutate(content);
      if (action === "copy") saveCopy.mutate(content);
    } catch (error) {
      setDraftError(error instanceof Error ? error.message : "Config JSON could not be parsed.");
    }
  }

  return (
    <main className="page">
      <PageHeader eyebrow="Config workbench" title="Configs" description="Browse committed pipeline configs, validate draft content, preview stage plans, and save guarded copies." />
      <section className="split-layout wide-left">
        <section className="panel">
          <div className="panel-header"><div><h2>Available configs</h2><p>{configs.data?.total ?? 0} configs returned by the API.</p></div></div>
          {configs.isLoading ? <StateBlock title="Loading configs" kind="loading" /> : null}
          {configs.isError ? <StateBlock title="Could not load configs" kind="error" message={toUserMessage(configs.error)} /> : null}
          <div className="table-wrap">
            <table>
              <thead><tr><th>Config</th><th>Task</th><th>Dataset</th><th>Target</th></tr></thead>
              <tbody>
                {(configs.data?.configs ?? []).map((config) => (
                  <tr key={config.config_name} className={config.config_name === selectedName ? "champion-row" : ""} onClick={() => setSelectedName(config.config_name)}>
                    <td><button className="link-button" type="button">{config.config_name}</button></td>
                    <td>{config.task_type ?? "-"}</td>
                    <td>{config.dataset_name ?? "-"}</td>
                    <td>{config.target_column ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="panel stack">
          <div className="panel-header"><div><h2>{selectedName || "Select a config"}</h2><p>{selectedSummary?.dataset_name ?? "Config content and workbench tools."}</p></div></div>
          {detail.isLoading ? <StateBlock title="Loading config detail" kind="loading" /> : null}
          {detail.isError ? <StateBlock title="Could not load config" kind="error" message={toUserMessage(detail.error)} /> : null}
          <textarea className="code-editor" value={draft} onChange={(event) => setDraft(event.target.value)} rows={18} spellCheck={false} />
          <div className="form-grid">
            <label className="field"><span>Save-as-copy name</span><input value={copyName} onChange={(event) => setCopyName(event.target.value)} /></label>
          </div>
          <div className="button-row">
            <button className="button secondary" onClick={() => runMutation("validate")} type="button"><ShieldCheck size={16} /> Validate</button>
            <button className="button secondary" onClick={() => runMutation("preview")} type="button">Preview stages</button>
            <button className="button primary" onClick={() => runMutation("copy")} disabled={!copyName.trim()} type="button"><Save size={16} /> Save copy</button>
          </div>
          {draftError ? <StateBlock title="Draft parse failed" kind="error" message={draftError} /> : null}
          {validation.isError ? <StateBlock title="Validation failed" kind="error" message={toUserMessage(validation.error)} /> : null}
          {saveCopy.isError ? <StateBlock title="Save failed" kind="error" message={toUserMessage(saveCopy.error)} /> : null}
          {saveCopy.data ? <StateBlock title="Config copy saved" message={`Created ${saveCopy.data.config_name}.`} /> : null}
          {validation.data ? <ValidationPanel validation={validation.data} /> : null}
          {preview.data ? <PreviewPanel preview={preview.data} /> : null}
        </section>
      </section>
    </main>
  );
}

function ValidationPanel({ validation }: { validation: ConfigValidationResponse }) {
  const issues = [...validation.errors, ...validation.warnings];
  return (
    <section className="panel subtle-panel">
      <h3>{validation.valid ? "Validation passed" : "Validation issues"}</h3>
      {issues.length === 0 ? <p>No errors or warnings were returned.</p> : null}
      <div className="issue-list">
        {issues.map((issue, index) => <div key={`${issue.path}-${index}`}><strong>{issue.level}</strong><span>{issue.path}</span><p>{issue.message}</p></div>)}
      </div>
    </section>
  );
}

function PreviewPanel({ preview }: { preview: ConfigPreviewResponse }) {
  return (
    <section className="panel subtle-panel stack">
      <div className="metric-grid compact">
        <div className="metric-card"><small>Experiment</small><strong>{preview.experiment_name ?? "-"}</strong></div>
        <div className="metric-card"><small>Compute</small><strong>{preview.compute_target ?? "-"}</strong></div>
        <div className="metric-card"><small>Phase B budget</small><strong>{preview.phase_b_variant_budget ?? "-"}</strong></div>
        <div className="metric-card"><small>Phase C trials</small><strong>{preview.phase_c_trials ?? "-"}</strong></div>
      </div>
      <div className="table-wrap"><table><thead><tr><th>Stage</th><th>Label</th><th>Enabled</th><th>Summary</th></tr></thead><tbody>{preview.stage_plan.map((stage) => <tr key={stage.stage_id}><td>{stage.stage_id}</td><td>{stage.label}</td><td>{stage.enabled ? "Yes" : "No"}</td><td>{stage.summary ?? stage.warnings.join(", ")}</td></tr>)}</tbody></table></div>
      <JsonDetails title="Preview details" value={preview} />
    </section>
  );
}