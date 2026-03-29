"""
JOB RESULTS EXTRACTION SCRIPT
==============================

Extracts comprehensive results from a completed Azure ML pipeline job for AI analysis.

Usage:
    python scripts/extract_job_results.py --job_name witty_lettuce_d46n1mxs4w

Output:
    - Console summary with key findings
    - JSON: job_results/<job_name>_summary.json
    - Markdown: job_results/<job_name>_report.md
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class JobResultsExtractor:
    """Extract and analyze Azure ML pipeline job results."""
    
    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        self.ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name,
        )
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
    
    def extract_job_metadata(self, job_name: str) -> Dict[str, Any]:
        """Extract job-level metadata."""
        print(f"📋 Extracting job metadata for: {job_name}")
        
        try:
            job = self.ml_client.jobs.get(job_name)
            
            metadata = {
                "job_name": job.name,
                "display_name": job.display_name if hasattr(job, 'display_name') else None,
                "experiment_name": job.experiment_name if hasattr(job, 'experiment_name') else None,
                "status": job.status,
                "tags": job.tags if hasattr(job, 'tags') else {},
                "compute": job.compute if hasattr(job, 'compute') else None,
            }
            
            print(f"   ✅ Status: {metadata['status']}")
            print(f"   ✅ Experiment: {metadata['experiment_name']}")
            
            return metadata
            
        except Exception as e:
            print(f"   ❌ Error extracting metadata: {e}")
            return {"error": str(e)}
    
    def extract_child_jobs(self, job_name: str) -> List[Dict[str, Any]]:
        """Extract all child job steps and their status."""
        print(f"\n📊 Extracting child job steps...")
        
        try:
            # List all child runs
            children = list(self.ml_client.jobs.list(parent_job_name=job_name))
            
            child_data = []
            for child in children:
                step_info = {
                    "step_name": child.display_name if hasattr(child, 'display_name') else child.name,
                    "job_name": child.name,
                    "status": child.status,
                }
                
                child_data.append(step_info)
                
                status_icon = "✅" if step_info["status"] == "Completed" else "❌" if step_info["status"] == "Failed" else "⏳"
                print(f"   {status_icon} {step_info['step_name']}: {step_info['status']}")
            
            print(f"   ✅ Found {len(child_data)} child steps")
            return child_data
            
        except Exception as e:
            print(f"   ❌ Error extracting child jobs: {e}")
            return []
    
    def check_smote_logs(self, job_name: str) -> Dict[str, Any]:
        """Check if SMOTE was applied by examining Stage 3 child job."""
        print(f"\n⚖️  Checking for SMOTE application...")
        
        smote_data = {
            "smote_detected": False,
            "stage3_status": "unknown",
            "notes": []
        }
        
        try:
            # Find Stage 3 child job
            children = list(self.ml_client.jobs.list(parent_job_name=job_name))
            stage3_jobs = [c for c in children if 'stage3' in c.name.lower() or 'preprocessing' in (c.display_name or '').lower()]
            
            if stage3_jobs:
                stage3_job = stage3_jobs[0]
                smote_data["stage3_status"] = stage3_job.status
                print(f"   ✅ Found Stage 3 job: {stage3_job.name} ({stage3_job.status})")
                smote_data["notes"].append("Stage 3 preprocessing job found")
            else:
                print(f"   ⚠️  Stage 3 job not found in child jobs")
                smote_data["notes"].append("Stage 3 not found - pipeline may still be running")
        
        except Exception as e:
            print(f"   ⚠️  Error checking SMOTE: {e}")
            smote_data["notes"].append(f"Error: {str(e)}")
        
        return smote_data
    
    def check_flaml_artifacts(self, job_name: str) -> Dict[str, Any]:
        """Check if FLAML Phase B jobs exist."""
        print(f"\n🔥 Checking for FLAML Phase B jobs...")
        
        flaml_data = {
            "flaml_jobs_found": False,
            "job_count": 0,
            "statuses": []
        }
        
        try:
            # Find Phase B FLAML child jobs
            children = list(self.ml_client.jobs.list(parent_job_name=job_name))
            flaml_jobs = [c for c in children if 'flaml' in c.name.lower() or 's06b' in c.name.lower() or 's07b' in c.name.lower()]
            
            flaml_data["job_count"] = len(flaml_jobs)
            flaml_data["flaml_jobs_found"] = len(flaml_jobs) > 0
            
            if flaml_jobs:
                print(f"   ✅ Found {len(flaml_jobs)} FLAML job(s)")
                for job in flaml_jobs:
                    status_info = {
                        "job_name": job.name,
                        "display_name": job.display_name if hasattr(job, 'display_name') else None,
                        "status": job.status
                    }
                    flaml_data["statuses"].append(status_info)
                    status_icon = "✅" if job.status == "Completed" else "❌" if job.status == "Failed" else "⏳"
                    print(f"      {status_icon} {status_info['display_name']}: {job.status}")
            else:
                print(f"   ⚠️  No FLAML jobs found - may not have reached Phase B yet")
        
        except Exception as e:
            print(f"   ⚠️  Error checking FLAML: {e}")
        
        return flaml_data
    
    def generate_summary(self, job_name: str) -> Dict[str, Any]:
        """Generate comprehensive job summary."""
        print("="*80)
        print(f"🔍 JOB RESULTS EXTRACTION: {job_name}")
        print("="*80)
        
        summary = {
            "extraction_timestamp": datetime.now().isoformat(),
            "job_name": job_name,
            "metadata": self.extract_job_metadata(job_name),
            "child_jobs": self.extract_child_jobs(job_name),
            "smote_check": self.check_smote_logs(job_name),
            "flaml_check": self.check_flaml_artifacts(job_name),
        }
        
        # Generate verdict
        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        
        status = summary["metadata"].get("status", "unknown")
        print(f"   Job Status: {status}")
        print(f"   Child Steps: {len(summary['child_jobs'])} found")
        print(f"   Stage 3 Status: {summary['smote_check']['stage3_status']}")
        print(f"   FLAML Jobs: {summary['flaml_check']['job_count']} found")
        
        print("\n" + "="*80)
        if status == "Completed":
            print("✅ JOB COMPLETED - Ready for detailed artifact analysis")
        elif status == "Running":
            print("⏳ JOB STILL RUNNING - Partial results available")
        elif status == "Failed":
            print("❌ JOB FAILED - Check child job logs for errors")
        else:
            print(f"⚠️  JOB STATUS: {status}")
        print("="*80)
        
        return summary
    
    def save_summary(self, summary: Dict[str, Any], output_dir: Path):
        """Save summary to JSON and Markdown."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        job_name = summary["job_name"]
        
        # Save JSON
        json_path = output_dir / f"{job_name}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n💾 Saved JSON summary: {json_path}")
        
        # Generate Markdown report
        md_path = output_dir / f"{job_name}_report.md"
        with open(md_path, 'w') as f:
            f.write(f"# Job Results Report: {job_name}\n\n")
            f.write(f"**Generated**: {summary['extraction_timestamp']}\n\n")
            
            f.write("## Job Metadata\n\n")
            f.write(f"- **Status**: {summary['metadata']['status']}\n")
            f.write(f"- **Experiment**: {summary['metadata'].get('experiment_name', 'unknown')}\n")
            f.write(f"- **Display Name**: {summary['metadata'].get('display_name', 'unknown')}\n")
            f.write("\n")
            
            f.write("## Child Jobs\n\n")
            for child in summary['child_jobs']:
                status_icon = "✅" if child['status'] == "Completed" else "❌" if child['status'] == "Failed" else "⏳"
                f.write(f"- {status_icon} **{child['step_name']}**: {child['status']}\n")
            f.write("\n")
            
            f.write("## SMOTE Check\n\n")
            f.write(f"- Stage 3 Status: {summary['smote_check']['stage3_status']}\n")
            if summary['smote_check']['notes']:
                f.write("\nNotes:\n")
                for note in summary['smote_check']['notes']:
                    f.write(f"- {note}\n")
            f.write("\n")
            
            f.write("## FLAML Check\n\n")
            f.write(f"- FLAML Jobs Found: {summary['flaml_check']['job_count']}\n")
            if summary['flaml_check']['statuses']:
                f.write("\nJob Statuses:\n")
                for status in summary['flaml_check']['statuses']:
                    status_icon = "✅" if status['status'] == "Completed" else "❌" if status['status'] == "Failed" else "⏳"
                    f.write(f"- {status_icon} {status['display_name']}: {status['status']}\n")
            f.write("\n")
            
            f.write("## Next Steps\n\n")
            if summary['metadata']['status'] == "Completed":
                f.write("- Download job outputs for detailed analysis\n")
                f.write("- Check MLflow metrics for performance results\n")
                f.write("- Review SMOTE logs in Stage 3 outputs\n")
                f.write("- Validate FLAML artifacts in Phase B outputs\n")
            elif summary['metadata']['status'] == "Running":
                f.write("- Wait for job completion\n")
                f.write("- Monitor progress in Azure ML Studio\n")
            else:
                f.write("- Review failed child job logs\n")
                f.write("- Check Azure ML Studio for error details\n")
        
        print(f"💾 Saved Markdown report: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract results from Azure ML pipeline job")
    parser.add_argument("--job_name", required=True, help="Job name to extract results from")
    parser.add_argument("--subscription_id", required=False, help="Azure subscription ID")
    parser.add_argument("--resource_group", required=False, help="Azure resource group")
    parser.add_argument("--workspace_name", required=False, help="Azure ML workspace name")
    parser.add_argument("--output_dir", default="job_results", help="Output directory for results")
    args = parser.parse_args()
    
    # Load workspace config from default location if not provided
    if not all([args.subscription_id, args.resource_group, args.workspace_name]):
        config_path = ROOT / "configs" / "config_classification_telecom_churn_azureml.yml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
                azure_cfg = cfg.get("azureml", {})
                args.subscription_id = args.subscription_id or azure_cfg.get("subscription_id")
                args.resource_group = args.resource_group or azure_cfg.get("resource_group")
                args.workspace_name = args.workspace_name or azure_cfg.get("workspace_name")
                print(f"✅ Loaded workspace config from {config_path.name}")
    
    if not all([args.subscription_id, args.resource_group, args.workspace_name]):
        print("❌ Error: Azure workspace credentials required")
        print("   Provide via CLI flags or ensure config_classification_telecom_churn_azureml.yml exists")
        sys.exit(1)
    
    # Extract results
    extractor = JobResultsExtractor(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name
    )
    
    summary = extractor.generate_summary(args.job_name)
    
    # Save results
    output_dir = Path(args.output_dir)
    extractor.save_summary(summary, output_dir)
    
    print(f"\n✅ Results extraction complete!")
    print(f"📂 Results saved to: {output_dir.resolve()}")
    print(f"\n💡 To view results:")
    print(f"   cat {output_dir}/{args.job_name}_report.md")


if __name__ == "__main__":
    main()
