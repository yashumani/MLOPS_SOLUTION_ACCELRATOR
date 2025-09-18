"""
Example script demonstrating Azure ML SDK V2 data ingestion usage.

This script shows how to:
1. Load data from Azure ML data assets
2. Register local data as Azure ML data assets
3. Create Azure ML jobs with data inputs
4. Use the new DataIngestionManager class
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_ingestion import DataIngestionManager
from config_loader import load_config

# Azure ML SDK V2 imports
try:
    from azure.ai.ml import MLClient, command, Input
    from azure.ai.ml.constants import AssetTypes, InputOutputModes
    from azure.identity import DefaultAzureCredential
    AZURE_ML_AVAILABLE = True
except ImportError:
    print("Azure ML SDK V2 not available. Please install: pip install azure-ai-ml")
    AZURE_ML_AVAILABLE = False


def example_basic_data_loading():
    """Example 1: Basic data loading with configuration."""
    print("🔄 Example 1: Basic Data Loading")
    print("-" * 50)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "production_config.yaml"
    config = load_config(str(config_path))
    
    # Initialize data ingestion manager
    manager = DataIngestionManager(config)
    
    # Load data (will try Azure ML first, fallback to local)
    dataset_path = config['dataset_path']
    df, dataset_name = manager.load_data(dataset_path)
    
    print(f"✅ Loaded dataset: {dataset_name}")
    print(f"📊 Shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print()


def example_azure_ml_data_asset():
    """Example 2: Loading from Azure ML data asset."""
    if not AZURE_ML_AVAILABLE:
        print("⚠️ Azure ML SDK V2 not available. Skipping this example.")
        return
    
    print("🔄 Example 2: Azure ML Data Asset Loading")
    print("-" * 50)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "production_config.yaml"
    config = load_config(str(config_path))
    
    # Initialize data ingestion manager
    manager = DataIngestionManager(config)
    
    try:
        # Example: Load a specific data asset by name
        # Replace 'your_dataset_name' with an actual data asset in your workspace
        df, dataset_name = manager.load_data(
            path="", 
            data_asset_name="cardiac_arrest_dataset",  # Example from your sample
            data_asset_version="1"
        )
        
        print(f"✅ Loaded Azure ML data asset: {dataset_name}")
        print(f"📊 Shape: {df.shape}")
        
    except Exception as e:
        print(f"❌ Failed to load Azure ML data asset: {e}")
        print("💡 Make sure you have a data asset registered in your workspace")
    print()


def example_register_data_asset():
    """Example 3: Register local data as Azure ML data asset."""
    if not AZURE_ML_AVAILABLE:
        print("⚠️ Azure ML SDK V2 not available. Skipping this example.")
        return
    
    print("🔄 Example 3: Register Data Asset")
    print("-" * 50)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "production_config.yaml"
    config = load_config(str(config_path))
    
    # Initialize data ingestion manager
    manager = DataIngestionManager(config)
    
    try:
        # Example: Register a local CSV file as a data asset
        local_data_path = Path(__file__).parent.parent / "data" / "finance_sample.csv"
        
        if local_data_path.exists():
            manager.register_data_asset(
                name="finance_sample_dataset",
                path=str(local_data_path),
                description="Sample finance dataset for MLOps accelerator",
                version="1"
            )
            print("✅ Data asset registered successfully!")
        else:
            print(f"❌ Local data file not found: {local_data_path}")
            
    except Exception as e:
        print(f"❌ Failed to register data asset: {e}")
    print()


def example_azure_ml_job():
    """Example 4: Create Azure ML job with data input."""
    if not AZURE_ML_AVAILABLE:
        print("⚠️ Azure ML SDK V2 not available. Skipping this example.")
        return
    
    print("🔄 Example 4: Azure ML Job Creation")
    print("-" * 50)
    
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "production_config.yaml"
        config = load_config(str(config_path))
        
        # Initialize ML client
        azure_config = config['azure_ml']
        ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        
        # Get data asset (replace with your actual data asset name)
        data_asset = ml_client.data.get("cardiac_arrest_dataset", version="1")
        
        # Create a simple job to list the data
        job = command(
            command='ls "${{inputs.data}}" && head -5 "${{inputs.data}}"',
            inputs={
                "data": Input(
                    path=data_asset.id,
                    type=AssetTypes.URI_FILE,
                    mode=InputOutputModes.RO_MOUNT
                )
            },
            environment="azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
            compute=azure_config.get('compute_target', 'cpu-cluster')
        )
        
        # Submit the job
        returned_job = ml_client.jobs.create_or_update(job)
        print(f"✅ Job submitted: {returned_job.name}")
        print(f"🔗 Job URL: {returned_job.studio_url}")
        
    except Exception as e:
        print(f"❌ Failed to create Azure ML job: {e}")
        print("💡 Make sure your Azure ML workspace is properly configured")
    print()


def example_list_data_assets():
    """Example 5: List all data assets in workspace."""
    if not AZURE_ML_AVAILABLE:
        print("⚠️ Azure ML SDK V2 not available. Skipping this example.")
        return
    
    print("🔄 Example 5: List Data Assets")
    print("-" * 50)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "production_config.yaml"
    config = load_config(str(config_path))
    
    # Initialize data ingestion manager
    manager = DataIngestionManager(config)
    
    try:
        assets = manager.list_data_assets()
        
        if assets:
            print(f"📋 Found {len(assets)} data assets:")
            for asset in assets[:10]:  # Show first 10
                print(f"  • {asset['name']} (v{asset['version']}): {asset['description']}")
        else:
            print("📭 No data assets found in workspace")
            
    except Exception as e:
        print(f"❌ Failed to list data assets: {e}")
    print()


if __name__ == "__main__":
    print("🚀 Azure ML SDK V2 Data Ingestion Examples")
    print("=" * 60)
    print()
    
    # Run all examples
    example_basic_data_loading()
    example_azure_ml_data_asset()
    example_register_data_asset()
    example_azure_ml_job()
    example_list_data_assets()
    
    print("✅ All examples completed!")
    print("\n💡 Tips:")
    print("- Make sure you're authenticated with Azure CLI: 'az login'")
    print("- Ensure your Azure ML workspace is properly configured")
    print("- Replace example data asset names with your actual assets")