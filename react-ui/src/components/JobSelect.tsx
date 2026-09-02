import { useEffect } from "react";
import { useJobsQuery } from "../hooks/usePipelineQueries";
import { StatusBadge } from "./StatusBadge";
import { StateBlock } from "./StateBlock";

interface JobSelectProps {
  value?: string;
  onChange: (jobName: string) => void;
  label?: string;
}

export function JobSelect({ value, onChange, label = "Job" }: JobSelectProps) {
  const jobs = useJobsQuery();
  const firstJob = jobs.data?.jobs[0]?.job_name;

  useEffect(() => {
    if (!value && firstJob) onChange(firstJob);
  }, [firstJob, onChange, value]);

  if (jobs.isLoading) return <StateBlock title="Loading jobs" kind="loading" />;
  if (jobs.isError) return <StateBlock title="Could not load jobs" kind="error" message="Check API connectivity and retry." />;

  return (
    <label className="field">
      <span>{label}</span>
      <select value={value ?? ""} onChange={(event) => onChange(event.target.value)}>
        {(jobs.data?.jobs ?? []).map((job) => (
          <option value={job.job_name} key={job.job_name}>
            {job.display_name ?? job.job_name} - {job.status}
          </option>
        ))}
      </select>
      {value ? <SelectedJobStatus jobName={value} /> : null}
    </label>
  );
}

function SelectedJobStatus({ jobName }: { jobName: string }) {
  const jobs = useJobsQuery();
  const job = jobs.data?.jobs.find((item) => item.job_name === jobName);
  if (!job) return null;
  return <StatusBadge status={job.status} />;
}