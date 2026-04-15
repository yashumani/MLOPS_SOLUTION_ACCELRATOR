import argparse
import os
import uuid
from datetime import datetime
from pathlib import Path
import importlib
import sys

from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import PipelineJob, Environment
from azure.identity import DefaultAzureCredential
import yaml

# Force reload of pipeline_builder module to pick up latest component YAML changes
# This ensures component version increments are respected
if 'pipeline_builder' in sys.modules:
    import pipeline_builder
    importlib.reload(pipeline_builder)
    from pipeline_builder import full_pipeline
else:
    from pipeline_builder import full_pipeline


def _azure_from_local_config(config_path: str):
    """Load azureml connection defaults from a local YAML config if available."""
    p = Path(config_path)
    if not p.exists():
        return None, None, None
    try:
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
        azure_cfg = cfg.get("azureml") or cfg.get("azure_ml") or {}
        return (
            azure_cfg.get("subscription_id"),
            azure_cfg.get("resource_group"),
            azure_cfg.get("workspace_name"),
        )
    except Exception:
        return None, None, None


def derive_experiment_name(config_path: str) -> str:
    """Derive generic reusable experiment name from config filename.
    
    Example: config_classification_telecom_churn_azureml.yml → classification_telecom_churn_v3
    """
    config_stem = Path(config_path).stem
    normalized = config_stem.replace("config_", "").replace("_azureml", "").replace("_local", "")
    return f"{normalized}_v3"


def derive_display_name(experiment_name: str) -> str:
    """Generate unique display name for this job submission.
    
    Format: {experiment_name}_{timestamp}_{random_id}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{experiment_name}_{timestamp}_{unique_id}"


def main():
    parser = argparse.ArgumentParser(description="Submit V3 pipeline with proper experiment/display naming")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--subscription_id", required=False, help="Azure subscription ID")
    parser.add_argument("--resource_group", required=False, help="Azure resource group")
    parser.add_argument("--workspace_name", required=False, help="Azure ML workspace name")
    parser.add_argument("--compute", required=False, default="mlopsv2computecluster", help="Compute target")
    parser.add_argument(
        "--experiment_name",
        required=False,
        help="Reusable experiment name (auto-derived from config if not provided)",
    )
    parser.add_argument(
        "--display_name",
        required=False,
        help="Unique job display name (auto-generated with timestamp if not provided)",
    )
    parser.add_argument("--wait", action="store_true", help="Wait for job to complete")
    parser.add_argument(
        "--baseline_job",
        required=False,
        default=None,
        help="Previous pipeline job name whose drift_baseline output to use for comparison drift",
    )
    args = parser.parse_args()

    # If CLI context missing, try to read from local config YAML (when path is local)
    config_path = args.config
    if (not args.subscription_id or not args.resource_group or not args.workspace_name) and Path(args.config).exists():
        sub, rg, ws = _azure_from_local_config(args.config)
        args.subscription_id = args.subscription_id or sub
        args.resource_group = args.resource_group or rg
        args.workspace_name = args.workspace_name or ws
    
    # FIXED: Use config filename only (from uploaded code directory)
    # Avoid workspaceblobstore upload by passing filename as string parameter
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_name = Path(config_path).name
    print(f"✅ Using config filename: {config_name} (from uploaded code/configs directory)")

    # Derive experiment name (reusable, generic)
    if not args.experiment_name:
        args.experiment_name = derive_experiment_name(config_path)
    
    # Derive display name (unique per submission, with timestamp)
    if not args.display_name:
        args.display_name = derive_display_name(args.experiment_name)
    
    print("\n" + "="*80)
    print("NAMING CONFIGURATION")
    print("="*80)
    print(f"📊 Experiment name (reusable):  {args.experiment_name}")
    print(f"🎯 Display name (unique):       {args.display_name}")
    print("="*80 + "\n")
    datastore_name = "mlops_blob"
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            datastore_name = cfg.get('dataset', {}).get('datastore_name', 'mlops_blob')
    except Exception:
        pass
    
    # Dataset folder URI (Azure ML will mount it)
    dataset_folder_uri = (
        f"azureml://subscriptions/{args.subscription_id}"
        f"/resourcegroups/{args.resource_group}"
        f"/workspaces/{args.workspace_name}"
        f"/datastores/{datastore_name}/paths/"
    )
    print(f"Using datastore: {datastore_name}")
    print(f"Dataset folder URI: {dataset_folder_uri}")

    # Build pipeline job (config filename passed as string, no upload needed)
    pipeline_kwargs = dict(
        config_name=config_name,
        dataset_folder=Input(path=dataset_folder_uri, type="uri_folder"),
    )

    # If baseline_job provided, resolve drift_baseline output as a data asset URI
    if args.baseline_job:
        print(f"🔗 Resolving baseline from job: {args.baseline_job} ...")
        try:
            import requests
            # Use Azure ML History API to get the registered data asset URI
            cred = DefaultAzureCredential()
            token = cred.get_token("https://ml.azure.com/.default").token
            # First, determine the workspace region
            ws_info_url = (
                f"https://management.azure.com/subscriptions/{args.subscription_id}"
                f"/resourceGroups/{args.resource_group}"
                f"/providers/Microsoft.MachineLearningServices/workspaces/{args.workspace_name}"
                f"?api-version=2023-04-01-preview"
            )
            mgmt_token = cred.get_token("https://management.azure.com/.default").token
            ws_resp = requests.get(ws_info_url, headers={"Authorization": f"Bearer {mgmt_token}"})
            ws_region = ws_resp.json().get("location", "eastus2")
            # Query the pipeline run details for the drift_baseline output asset ID
            history_url = (
                f"https://{ws_region}.api.azureml.ms/history/v1.0"
                f"/subscriptions/{args.subscription_id}"
                f"/resourceGroups/{args.resource_group}"
                f"/providers/Microsoft.MachineLearningServices"
                f"/workspaces/{args.workspace_name}"
                f"/runs/{args.baseline_job}/details"
            )
            resp = requests.get(history_url, headers={"Authorization": f"Bearer {token}"})
            run_details = resp.json()
            outputs = run_details.get("outputs", {})
            baseline_output = outputs.get("drift_baseline", {})
            baseline_asset_id = baseline_output.get("assetId")
            if baseline_asset_id:
                pipeline_kwargs["drift_baseline_in"] = Input(
                    path=baseline_asset_id, type="uri_folder"
                )
                print(f"  ✅ Resolved drift_baseline asset: {baseline_asset_id}")
            else:
                print(f"  ⚠️ No drift_baseline output found in job {args.baseline_job}")
                print(f"     Available outputs: {list(outputs.keys())}")
        except Exception as e:
            print(f"  ⚠️ Failed to resolve baseline: {e}")
            print(f"     Pipeline will run without comparison drift")

    job = full_pipeline(**pipeline_kwargs)
    job.settings.default_compute = args.compute
    job.experiment_name = args.experiment_name
    job.display_name = args.display_name

    # Add job-level tags for dataset/task/preset and pipeline version
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        dataset = (cfg.get('dataset') or {}).get('name') or 'unknown'
        task = cfg.get('task_type') or 'unknown'
        preset = cfg.get('preset') or 'unknown'
        job.tags = {
            'dataset': dataset,
            'task': task,
            'preset': preset,
            'pipeline_version': 'v3',
            'environment': 'mlops-v3-unified:4',
        }
    except Exception:
        job.tags = {'pipeline_version': 'v3', 'environment': 'mlops-v3-unified:4'}

    # If Azure ML context provided, submit; else print YAML
    if args.subscription_id and args.resource_group and args.workspace_name:
        ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id=args.subscription_id,
            resource_group_name=args.resource_group,
            workspace_name=args.workspace_name,
        )
        # Note: Using pre-registered environment mlops-v3-unified:4
        print("Note: Using pre-registered environment mlops-v3-unified:4 (validated library versions) for all pipeline steps\n")
        
        submitted = ml_client.jobs.create_or_update(job)
        print(f"✅ Submitted job: {submitted.name}")
        print(f"🌐 Web View: https://ml.azure.com/runs/{submitted.name}?wsid=/subscriptions/{args.subscription_id}/resourcegroups/{args.resource_group}/workspaces/{args.workspace_name}")
        
        if args.wait:
            print("\n⏳ Waiting for pipeline to complete...")
            ml_client.jobs.stream(submitted.name)
    else:
        # Local dry-run: emit job yaml for inspection
        print(job)


if __name__ == "__main__":
    main()
