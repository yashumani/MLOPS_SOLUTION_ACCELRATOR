import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { JobSelect } from "../components/JobSelect";
import { JsonDetails } from "../components/JsonDetails";
import { PageHeader } from "../components/PageHeader";
import { StateBlock } from "../components/StateBlock";
import { useApi } from "../services/ApiContext";
import { toUserMessage } from "../services/apiClient";
import { formatValue } from "../utils/formatters";

export function Notifications() {
  const { api } = useApi();
  const [jobName, setJobName] = useState("");
  const notification = useMutation({ mutationFn: (dryRun: boolean) => api.sendNotification(jobName, dryRun) });

  return (
    <main className="page">
      <PageHeader eyebrow="Notification reports" title="Notifications" description="Generate Markdown, JSON, and CSV report packages for a job, or send the configured SMTP email with attachments." />
      <section className="grid two">
        <section className="panel stack">
          <JobSelect value={jobName} onChange={setJobName} />
          <div className="button-row">
            <button className="button secondary" disabled={!jobName || notification.isPending} onClick={() => notification.mutate(true)} type="button">Generate report only</button>
            <button className="button primary" disabled={!jobName || notification.isPending} onClick={() => notification.mutate(false)} type="button">Send configured email</button>
          </div>
          <StateBlock title="Recipient is configured by the backend" message="The browser does not collect SMTP secrets. Email recipient, sender, and SMTP settings are read by FastAPI from environment configuration." />
          {notification.isError ? <StateBlock title="Notification failed" kind="error" message={toUserMessage(notification.error)} /> : null}
        </section>
        <section className="panel stack">
          <div className="panel-header"><div><h2>Last notification result</h2><p>Generated files and delivery status.</p></div></div>
          {!notification.data ? <StateBlock title="No notification generated" message="Choose a job and generate a dry-run package or send the configured email." /> : null}
          {notification.data ? <><div className="metric-grid compact"><div className="metric-card"><small>Status</small><strong>{notification.data.status}</strong></div><div className="metric-card"><small>Sent</small><strong>{notification.data.sent ? "Yes" : "No"}</strong></div><div className="metric-card"><small>Recipient</small><strong>{notification.data.recipient}</strong></div><div className="metric-card"><small>Report dir</small><strong>{notification.data.report_dir}</strong></div></div><div className="table-wrap"><table><thead><tr><th>Artifact</th><th>Mime</th><th>Size</th><th>Included</th></tr></thead><tbody>{notification.data.artifacts.map((artifact) => <tr key={artifact.path}><td>{artifact.name}</td><td>{artifact.mime_type}</td><td>{formatValue(artifact.size_bytes)}</td><td>{artifact.included_in_email ? "Yes" : "No"}</td></tr>)}</tbody></table></div><JsonDetails title="Notification payload" value={notification.data} /></> : null}
        </section>
      </section>
    </main>
  );
}