#!/usr/bin/env python3
"""
Autonomous batch monitor for the 15-job production run.
Polls Azure ML every --interval minutes, logs status, and writes sentinel files.

Usage:
  cd /home/azureuser/cloudfiles/code/Users/yashu.savyminds/mlops-solution-accelerator-v3
  nohup /anaconda/envs/azureml_py38/bin/python scripts/monitor_batch.py \
    --tsv outputs/batch_15_prod_20260425/submissions.tsv \
    --interval 15 --expected 15 \
    > outputs/batch_15_prod_20260425/monitor_nohup.log 2>&1 &

Sentinel files written to same dir as --tsv:
  monitor_status.log  — running poll log (append-only)
  BATCH_DONE.txt      — written when all jobs reach terminal state
  FAILURES.txt        — written if any jobs Failed/Canceled
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TERMINAL = {"Completed", "Failed", "CanceledOrFailed", "Canceled", "NotResponding"}


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_tsv(path):
    """Read submissions.tsv and return list of {wave, config, job_id} dicts."""
    jobs = []
    try:
        with open(path) as f:
            lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[2].strip():
                jobs.append({
                    "wave":   parts[0].strip(),
                    "config": parts[1].strip() if len(parts) > 1 else "",
                    "job_id": parts[2].strip(),
                })
    except Exception:
        pass
    return jobs


def get_ml_client(sub, rg, ws):
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    return MLClient(DefaultAzureCredential(), sub, rg, ws)


def poll(ml, jobs, log_path, done_path, fail_path, expected):
    """Poll each job, log results, write sentinel files if done. Returns True when finished."""
    results = []
    for j in jobs:
        try:
            info = ml.jobs.get(j["job_id"])
            results.append({
                "wave":   j["wave"],
                "job_id": j["job_id"],
                "status": info.status,
            })
        except Exception as e:
            results.append({
                "wave":   j["wave"],
                "job_id": j["job_id"],
                "status": f"ERROR:{e}",
            })

    # Build log block
    lines = [
        "",
        "=" * 72,
        f"  POLL  {utcnow()}  ({len(jobs)}/{expected} submitted)",
        "=" * 72,
    ]
    counts = {}
    for r in results:
        lines.append(f"  {r['wave']:18s}  {r['job_id']:32s}  {r['status']}")
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines.append("  SUMMARY: " + "  ".join(f"{s}={n}" for s, n in sorted(counts.items())))

    block = "\n".join(lines) + "\n"
    with open(log_path, "a") as f:
        f.write(block)
    print(block, flush=True)

    # Evaluate terminal condition
    all_submitted = len(jobs) >= expected
    all_terminal  = all(r["status"] in TERMINAL for r in results)
    failures      = [r for r in results if r["status"] in ("Failed", "CanceledOrFailed", "Canceled")]

    if all_submitted and all_terminal:
        # Write BATCH_DONE.txt
        with open(done_path, "w") as f:
            f.write(f"BATCH_DONE {utcnow()}\n")
            f.write(f"Total:{len(results)}  Completed:{counts.get('Completed', 0)}  Failed:{len(failures)}\n")
            for r in results:
                f.write(f"  {r['wave']:18s}  {r['job_id']:32s}  {r['status']}\n")
        print(f"[DONE] All {expected} jobs terminal. Written: {done_path}", flush=True)

        # Write FAILURES.txt if needed
        if failures:
            with open(fail_path, "w") as f:
                for r in failures:
                    f.write(f"{r['wave']}\t{r['job_id']}\t{r['status']}\n")
            print(f"[WARN] {len(failures)} failed job(s) written to: {fail_path}", flush=True)
        return True

    return False


def main():
    ap = argparse.ArgumentParser(description="Autonomous Azure ML batch job monitor")
    ap.add_argument("--tsv",       default="outputs/batch_15_prod_20260425/submissions.tsv",
                    help="Path to submissions TSV file")
    ap.add_argument("--interval",  type=int,   default=15,
                    help="Poll interval in minutes (default: 15)")
    ap.add_argument("--expected",  type=int,   default=15,
                    help="Total number of jobs expected (default: 15)")
    ap.add_argument("--sub",       default="93044a08-5661-4f1b-b424-5eafe066a9d1")
    ap.add_argument("--rg",        default="mvpv1")
    ap.add_argument("--ws",        default="mlops-accelerator")
    ap.add_argument("--max_hours", type=float, default=6.0,
                    help="Hard timeout in hours (default: 6.0)")
    args = ap.parse_args()

    base      = Path(args.tsv).parent
    log_path  = base / "monitor_status.log"
    done_path = base / "BATCH_DONE.txt"
    fail_path = base / "FAILURES.txt"

    start_msg = (
        f"[MONITOR START] {utcnow()}  "
        f"interval={args.interval}m  expected={args.expected}  max_hours={args.max_hours}"
    )
    print(start_msg, flush=True)
    with open(log_path, "a") as f:
        f.write(start_msg + "\n")

    ml       = get_ml_client(args.sub, args.rg, args.ws)
    deadline = time.time() + args.max_hours * 3600

    while time.time() < deadline:
        jobs = read_tsv(args.tsv)
        if jobs:
            done = poll(ml, jobs, str(log_path), str(done_path), str(fail_path), args.expected)
            if done:
                sys.exit(0)
        else:
            msg = f"[{utcnow()}] TSV has no job IDs yet — waiting for submissions..."
            print(msg, flush=True)
            with open(log_path, "a") as f:
                f.write(msg + "\n")

        time.sleep(args.interval * 60)

    timeout_msg = f"[TIMEOUT] {utcnow()} — exceeded {args.max_hours}h limit, exiting"
    print(timeout_msg, flush=True)
    with open(log_path, "a") as f:
        f.write(timeout_msg + "\n")


if __name__ == "__main__":
    main()
