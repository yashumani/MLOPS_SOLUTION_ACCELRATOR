import { Search, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useExperimentsQuery, useJobsQuery } from "../hooks/usePipelineQueries";
import type { JobSummary } from "../types/api";
import { StatusBadge } from "./StatusBadge";
import { StateBlock } from "./StateBlock";

interface JobPickerProps {
  open: boolean;
  onClose: () => void;
}

export function JobPicker({ open, onClose }: JobPickerProps) {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const experiments = useExperimentsQuery();
  const jobsFallback = useJobsQuery();

  const jobs = useMemo(() => {
    const experimentJobs = experiments.data?.experiments.flatMap((experiment) =>
      experiment.jobs.map((job) => ({ ...job, experiment_name: job.experiment_name ?? experiment.experiment_name }))
    );
    const allJobs = experimentJobs && experimentJobs.length > 0 ? experimentJobs : jobsFallback.data?.jobs ?? [];
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) return allJobs.slice(0, 75);
    return allJobs
      .filter((job) => [job.job_name, job.display_name, job.experiment_name, job.status].some((value) => value?.toLowerCase().includes(normalizedQuery)))
      .slice(0, 75);
  }, [experiments.data, jobsFallback.data, query]);

  function choose(job: JobSummary) {
    navigate(`/focus/${job.job_name}`);
    onClose();
  }

  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation">
      <section className="modal" role="dialog" aria-modal="true" aria-labelledby="job-picker-title">
        <header className="modal-header">
          <div>
            <h2 id="job-picker-title">Change Job</h2>
            <p>Search recent Azure ML jobs and open a Focus cockpit.</p>
          </div>
          <button className="icon-button" onClick={onClose} type="button" aria-label="Close job picker">
            <X size={18} />
          </button>
        </header>
        <div className="search-box">
          <Search size={17} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search job, experiment, status" autoFocus />
        </div>
        {experiments.isLoading && jobsFallback.isLoading ? <StateBlock title="Loading jobs" kind="loading" /> : null}
        {experiments.isError && jobsFallback.isError ? <StateBlock title="Could not load jobs" kind="error" message="Check API connectivity and retry." /> : null}
        <div className="job-list">
          {jobs.map((job) => (
            <button key={job.job_name} className="job-row" onClick={() => choose(job)} type="button">
              <div>
                <strong>{job.display_name ?? job.job_name}</strong>
                <small>{job.job_name}</small>
                {job.experiment_name ? <small>{job.experiment_name}</small> : null}
              </div>
              <StatusBadge status={job.status} />
            </button>
          ))}
          {!experiments.isLoading && jobs.length === 0 ? <StateBlock title="No matching jobs" message="Try a different search term." /> : null}
        </div>
      </section>
    </div>
  );
}