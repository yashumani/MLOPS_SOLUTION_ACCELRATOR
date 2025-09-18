
"""
Data ingestion module using Azure ML SDK V2.

Responsible for loading datasets from local files or Azure ML data assets.
Supports both local CSV ingestion and Azure ML SDK V2 data asset integration.
"""

import pandas as pd
import os
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import Data
    from azure.ai.ml.constants import AssetTypes
    from azure.identity import DefaultAzureCredential, AzureCliCredential, ChainedTokenCredential
    AZURE_ML_AVAILABLE = True
except ImportError:
    print("Azure ML SDK V2 not available. Only local data ingestion will work.")
    AZURE_ML_AVAILABLE = False


class DataIngestionManager:
    """Manages data ingestion from various sources using Azure ML SDK V2."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data ingestion manager.
        
        Args:
            config: Configuration dictionary containing Azure ML and data settings.
        """
        self.config = config
        self.ml_client = None
        
        if AZURE_ML_AVAILABLE and self._should_use_azure_ml():
            self._initialize_ml_client()
    
    def _should_use_azure_ml(self) -> bool:
        """Check if Azure ML should be used based on configuration."""
        azure_config = self.config.get('azure_ml', {})
        return (
            azure_config.get('subscription_id') and
            azure_config.get('resource_group') and
            azure_config.get('workspace_name')
        )
    
    def _initialize_ml_client(self):
        """Initialize Azure ML client with proper authentication."""
        try:
            azure_config = self.config.get('azure_ml', {})
            
            # Try multiple authentication methods
            credential = ChainedTokenCredential(
                AzureCliCredential(),
                DefaultAzureCredential()
            )
            
            self.ml_client = MLClient(
                credential=credential,
                subscription_id=azure_config['subscription_id'],
                resource_group_name=azure_config['resource_group'],
                workspace_name=azure_config['workspace_name']
            )
            
            print(f"✅ Connected to Azure ML workspace: {azure_config['workspace_name']}")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize Azure ML client: {e}")
            print("Falling back to local data ingestion only.")
            self.ml_client = None
    
    def load_data(self, path: str, data_asset_name: Optional[str] = None, 
                  data_asset_version: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """Load dataset from local path or Azure ML data asset.
        
        Args:
            path: Local path to CSV file or Azure ML data asset path.
            data_asset_name: Name of Azure ML data asset (optional).
            data_asset_version: Version of Azure ML data asset (optional, defaults to latest).
            
        Returns:
            Tuple[pd.DataFrame, str]: Loaded DataFrame and dataset name.
        """
        try:
            # If data asset name is provided, try to load from Azure ML
            if data_asset_name and self.ml_client:
                return self._load_from_azure_ml_asset(data_asset_name, data_asset_version)
            
            # Check if path looks like an Azure ML asset reference
            elif path.startswith('azureml:') and self.ml_client:
                return self._load_from_azure_ml_path(path)
            
            # Otherwise, load from local path
            else:
                return self._load_from_local_path(path)
                
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")
    
    def _load_from_azure_ml_asset(self, asset_name: str, version: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """Load data from Azure ML data asset.
        
        Args:
            asset_name: Name of the data asset.
            version: Version of the data asset (latest if None).
            
        Returns:
            Tuple[pd.DataFrame, str]: Loaded DataFrame and dataset name.
        """
        try:
            print(f"🔄 Loading data asset '{asset_name}' from Azure ML...")
            
            # Get the data asset
            data_asset = self.ml_client.data.get(name=asset_name, version=version)
            print(f"✅ Found data asset: {data_asset.name} (version: {data_asset.version})")
            
            # Download the data asset to a temporary location
            download_path = self.ml_client.data.download(
                name=asset_name, 
                version=data_asset.version, 
                download_path="./temp_data"
            )
            
            # Find CSV files in the downloaded path
            csv_files = list(Path(download_path).rglob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the data asset")
            
            # Load the first CSV file found
            csv_path = csv_files[0]
            df = pd.read_csv(csv_path)
            
            print(f"✅ Loaded {len(df)} rows from Azure ML data asset")
            return df, f"{asset_name}_v{data_asset.version}"
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Azure ML data asset '{asset_name}': {e}")
    
    def _load_from_azure_ml_path(self, path: str) -> Tuple[pd.DataFrame, str]:
        """Load data from Azure ML path reference.
        
        Args:
            path: Azure ML path (e.g., 'azureml:dataset_name:1').
            
        Returns:
            Tuple[pd.DataFrame, str]: Loaded DataFrame and dataset name.
        """
        try:
            # Parse Azure ML path: azureml:name:version
            parts = path.split(':')
            if len(parts) >= 2:
                asset_name = parts[1]
                version = parts[2] if len(parts) > 2 else None
                return self._load_from_azure_ml_asset(asset_name, version)
            else:
                raise ValueError(f"Invalid Azure ML path format: {path}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load from Azure ML path '{path}': {e}")
    
    def _load_from_local_path(self, path: str) -> Tuple[pd.DataFrame, str]:
        """Load data from local CSV file.
        
        Args:
            path: Local path to CSV file.
            
        Returns:
            Tuple[pd.DataFrame, str]: Loaded DataFrame and dataset name.
        """
        try:
            print(f"🔄 Loading data from local path: {path}")
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            
            df = pd.read_csv(path)
            dataset_name = os.path.basename(path).replace('.csv', '')
            
            print(f"✅ Loaded {len(df)} rows from local file")
            return df, dataset_name
            
        except Exception as e:
            raise RuntimeError(f"Failed to load local file '{path}': {e}")
    
    def register_data_asset(self, name: str, path: str, description: str = "", 
                          version: Optional[str] = None) -> None:
        """Register a local dataset as an Azure ML data asset.
        
        Args:
            name: Name for the data asset.
            path: Local path to the data file.
            description: Description of the dataset.
            version: Version string (auto-generated if None).
        """
        if not self.ml_client:
            raise RuntimeError("Azure ML client not available")
        
        try:
            print(f"🔄 Registering data asset '{name}' in Azure ML...")
            
            data_asset = Data(
                name=name,
                path=path,
                type=AssetTypes.URI_FILE,
                description=description,
                version=version
            )
            
            registered_asset = self.ml_client.data.create_or_update(data_asset)
            print(f"✅ Data asset registered: {registered_asset.name} (version: {registered_asset.version})")
            
        except Exception as e:
            raise RuntimeError(f"Failed to register data asset: {e}")
    
    def list_data_assets(self) -> list:
        """List all data assets in the Azure ML workspace.
        
        Returns:
            List of data asset names and versions.
        """
        if not self.ml_client:
            raise RuntimeError("Azure ML client not available")
        
        try:
            assets = []
            for asset in self.ml_client.data.list():
                assets.append({
                    'name': asset.name,
                    'version': asset.version,
                    'description': asset.description or "No description"
                })
            return assets
            
        except Exception as e:
            raise RuntimeError(f"Failed to list data assets: {e}")


# Backward compatibility functions
def load_data(path: str, config: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, str]:
    """Load dataset from a given path (backward compatible function).
    
    Args:
        path: Path to the CSV file or Azure ML data asset.
        config: Configuration dictionary (optional).
        
    Returns:
        Tuple[pd.DataFrame, str]: Loaded DataFrame and dataset name.
    """
    if config is None:
        config = {}
    
    manager = DataIngestionManager(config)
    return manager.load_data(path)
