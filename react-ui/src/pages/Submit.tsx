import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ExternalLink } from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ConfigSelect } from "../components/ConfigSelect";
import { JsonDetails } from "../components/JsonDetails";
import { PageHeader } from "../components/PageHeader";
import { StateBlock } from "../components/StateBlock";
import { useConfigsQuery } from "../hooks/usePipelineQueries";
import { useApi } from "../services/ApiContext";
import { toUserMessage } from "../services/apiClient";
import type { SubmitRequest, SubmitResponse, SubmitStatusResponse } from "../types/api";

type SubmitResult = SubmitResponse | SubmitStatusResponse;

function normalizeConfigName(configName: string): string {
  return configName.replace(/\.ya?ml$/i, "");
}

function parseTags(text: string): Record<string, string> {
  const tags: Record<string, string> = {};
  for (const rawPart of text.split(/[\n,]/)) {
    const part = rawPart.trim();
    if (!part) continue;
    const [key, ...rest] = part.split("=");
    if (!key || rest.length === 0) throw new Error(`Tag '${part}' must use key=value format.`);
    tags[key.trim()] = rest.join("=").trim();
  }
  return tags;
}

export function Submit() {
  const { api } = useApi();
  const queryClient = useQueryClient();
  const configs = useConfigsQuery();
  const [configName, setConfigName] = useState("");
  const [compute, setCompute] = useState("");
  const [baselineJob, setBaselineJob] = useState("");
  const [forceRerun, setForceRerun] = useState(false);
  const [useAsync, setUseAsync] = useState(true);
  const [tagText, setTagText] = useState("client=react-ui");
  const [formError, setFormError] = useState<string | null>(null);
  const [asyncRequestId, setAsyncRequestId] = useState<string | null>(null);

  const selectedConfig = useMemo(
    () => configs.data?.configs.find((config) => config.config_name === configName),
    [configName, configs.data]
  );

  const submitMutation = useMutation<SubmitResult, unknown, SubmitRequest>({
    mutationFn: (request: SubmitRequest) => useAsync ? api.submitPipelineAsync(request) : api.submitPipeline(request),
    onSuccess: (result) => {
      const requestId = "request_id" in result && typeof result.request_id === "string"
        ? result.request_id
        : null;
      setAsyncRequestId(requestId);
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["experiments"] });
    }
  });

  const submitStatus = useQuery({
    queryKey: ["submit-status", asyncRequestId],
    queryFn: () => api.submitStatus(asyncRequestId as string),
    enabled: Boolean(asyncRequestId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "submitted" || status === "failed" || status === "reconciliation_required"
        ? false
        : 1500;
    }
  });

  useEffect(() => {
    if (submitStatus.data?.status !== "submitted") return;
    queryClient.invalidateQueries({ queryKey: ["jobs"] });
    queryClient.invalidateQueries({ queryKey: ["experiments"] });
  }, [queryClient, submitStatus.data?.status]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setFormError(null);
    setAsyncRequestId(null);
    try {
      const request: SubmitRequest = {
        config_name: normalizeConfigName(configName),
        compute: compute.trim() || null,
        force_rerun: forceRerun,
        baseline_job: baselineJob.trim() || null,
        tags: parseTags(tagText)
      };
      submitMutation.mutate(request);
    } catch (error) {
      setFormError(error instanceof Error ? error.message : "Submit form is invalid.");
    }
  }

  const result = submitStatus.data ?? submitMutation.data;
  const resultJobName = typeof result?.job_name === "string" ? result.job_name : undefined;
  const requestId = result && "request_id" in result && typeof result.request_id === "string" ? result.request_id : undefined;
  const studioUrl = result && "studio_url" in result && typeof result.studio_url === "string" ? result.studio_url : undefined;

  return (
    <main className="page">
      <PageHeader
        eyebrow="Pipeline submission"
        title="Submit Pipeline"
        description="Choose a committed config, set optional runtime overrides, and submit through the FastAPI canonical entrypoint."
      />

      <section className="split-layout">
        <form className="panel form-panel" onSubmit={submit}>
          <ConfigSelect value={configName} onChange={setConfigName} />
          <div className="metric-grid compact">
            <div className="metric-card"><small>Task type</small><strong>{selectedConfig?.task_type ?? "-"}</strong></div>
            <div className="metric-card"><small>Dataset</small><strong>{selectedConfig?.dataset_name ?? "-"}</strong></div>
            <div className="metric-card"><small>Target</small><strong>{selectedConfig?.target_column ?? "-"}</strong></div>
            <div className="metric-card"><small>Mode</small><strong>{useAsync ? "Async" : "Direct"}</strong></div>
          </div>
          <div className="form-grid">
            <label className="field"><span>Compute override</span><input value={compute} onChange={(event) => setCompute(event.target.value)} placeholder="Use config default" /></label>
            <label className="field"><span>Baseline job for drift</span><input value={baselineJob} onChange={(event) => setBaselineJob(event.target.value)} placeholder="Optional previous job" /></label>
          </div>
          <label className="field"><span>Tags</span><textarea value={tagText} onChange={(event) => setTagText(event.target.value)} rows={4} placeholder="key=value, one per line" /></label>
          <div className="toggle-row">
            <label><input type="checkbox" checked={forceRerun} onChange={(event) => setForceRerun(event.target.checked)} /> Force component rerun</label>
            <label><input type="checkbox" checked={useAsync} onChange={(event) => setUseAsync(event.target.checked)} /> Return immediately with async request</label>
          </div>
          {formError ? <StateBlock title="Submit form needs attention" kind="error" message={formError} /> : null}
          {submitMutation.isError ? <StateBlock title="Submission failed" kind="error" message={toUserMessage(submitMutation.error)} /> : null}
          {submitStatus.isError ? <StateBlock title="Submission status unavailable" kind="error" message={toUserMessage(submitStatus.error)} /> : null}
          <button className="button primary" disabled={!configName || submitMutation.isPending} type="submit">
            {submitMutation.isPending ? "Submitting..." : "Submit job"}
          </button>
        </form>

        <section className="panel stack">
          <div className="panel-header"><div><h2>Submission result</h2><p>Successful submissions link directly into Focus and Azure ML Studio.</p></div></div>
          {!result ? <StateBlock title="Ready to submit" message="No job has been submitted from this page in the current session." /> : null}
          {result ? (
            <div className="stack">
              <div className="result-banner"><strong>{String(result.status ?? "submitted")}</strong><span>{resultJobName ?? requestId ?? "request queued"}</span></div>
              <div className="button-row">
                {resultJobName ? <Link className="button secondary" to={`/focus/${resultJobName}`}>Open Focus</Link> : null}
                {studioUrl ? <a className="button secondary" href={studioUrl} target="_blank" rel="noreferrer"><ExternalLink size={16} /> Azure ML Studio</a> : null}
              </div>
              <JsonDetails title="Submission payload" value={result} />
            </div>
          ) : null}
        </section>
      </section>
    </main>
  );
}
