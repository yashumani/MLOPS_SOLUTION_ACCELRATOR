import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { useJobsQuery } from "../hooks/usePipelineQueries";
import { StatusBadge } from "../components/StatusBadge";
import { StateBlock } from "../components/StateBlock";

const validatedJobs = [
  { label: "Classification", job: "orange_card_3g8kwtx9gm" },
  { label: "Regression", job: "cool_rhubarb_52tdhqw5bv" },
  { label: "Clustering", job: "crimson_insect_b0gkxz4nfx" }
];

export function Home() {
  const jobs = useJobsQuery();
  return (
    <main className="page">
      <section className="page-heading">
        <div>
          <p className="eyebrow">Operations dashboard</p>
          <h1>MLOps V3 command center</h1>
          <p>Monitor Azure ML pipeline runs, inspect client-readable results, and move between jobs without Streamlit session state.</p>
        </div>
      </section>

      <section className="grid three">
        {validatedJobs.map((item) => (
          <Link className="tile" to={`/focus/${item.job}`} key={item.job}>
            <small>{item.label}</small>
            <strong>{item.job}</strong>
            <span><ArrowRight size={15} /> Open Focus</span>
          </Link>
        ))}
      </section>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h2>Recent jobs</h2>
            <p>Latest jobs returned by the FastAPI service.</p>
          </div>
          <Link className="button secondary" to="/focus">Open job picker</Link>
        </div>
        {jobs.isLoading ? <StateBlock title="Loading jobs" kind="loading" /> : null}
        {jobs.isError ? <StateBlock title="Could not load recent jobs" kind="error" message="Check the API key and backend service." /> : null}
        {jobs.data ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Job</th><th>Experiment</th><th>Status</th><th>Start</th><th></th></tr>
              </thead>
              <tbody>
                {jobs.data.jobs.slice(0, 12).map((job) => (
                  <tr key={job.job_name}>
                    <td>{job.display_name ?? job.job_name}<br /><small>{job.job_name}</small></td>
                    <td>{job.experiment_name ?? "-"}</td>
                    <td><StatusBadge status={job.status} /></td>
                    <td>{job.start_time ?? "-"}</td>
                    <td><Link to={`/focus/${job.job_name}`}>Focus</Link></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>
    </main>
  );
}