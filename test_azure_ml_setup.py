"""
Quick test script to verify Azure ML SDK V2 setup and connectivity.

This script checks:
1. Azure ML SDK V2 installation
2. Authentication to Azure
3. Workspace connectivity
4. Basic data asset operations
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_azure_ml_sdk_installation():
    """Test if Azure ML SDK V2 is properly installed."""
    print("🔄 Testing Azure ML SDK V2 installation...")
    
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Data
        from azure.ai.ml.constants import AssetTypes
        from azure.identity import DefaultAzureCredential
        print("✅ Azure ML SDK V2 packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Azure ML SDK V2 import failed: {e}")
        print("💡 Install with: pip install azure-ai-ml azure-identity")
        return False


def test_azure_authentication():
    """Test Azure authentication."""
    print("🔄 Testing Azure authentication...")
    
    try:
        from azure.identity import DefaultAzureCredential
        credential = DefaultAzureCredential()
        
        # Try to get token
        token = credential.get_token("https://management.azure.com/.default")
        print("✅ Azure authentication successful")
        return True
    except Exception as e:
        print(f"❌ Azure authentication failed: {e}")
        print("💡 Try: az login")
        return False


def test_workspace_connectivity():
    """Test connection to Azure ML workspace."""
    print("🔄 Testing workspace connectivity...")
    
    try:
        from config_loader import load_config
        from data_ingestion import DataIngestionManager
        
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "production_config.yaml"
        config = load_config(str(config_path))
        
        # Initialize data manager (this will test workspace connection)
        manager = DataIngestionManager(config)
        
        if manager.ml_client:
            print("✅ Azure ML workspace connection successful")
            
            # Get workspace info
            workspace = manager.ml_client.workspaces.get(config['azure_ml']['workspace_name'])
            print(f"📍 Connected to workspace: {workspace.name} in {workspace.location}")
            return True
        else:
            print("❌ Failed to connect to Azure ML workspace")
            return False
            
    except Exception as e:
        print(f"❌ Workspace connectivity test failed: {e}")
        return False


def test_data_asset_operations():
    """Test basic data asset operations."""
    print("🔄 Testing data asset operations...")
    
    try:
        from config_loader import load_config
        from data_ingestion import DataIngestionManager
        
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "production_config.yaml"
        config = load_config(str(config_path))
        
        # Initialize data manager
        manager = DataIngestionManager(config)
        
        if manager.ml_client:
            # Try to list data assets
            assets = manager.list_data_assets()
            print(f"✅ Successfully listed {len(assets)} data assets")
            
            if assets:
                print("📋 Sample data assets:")
                for asset in assets[:3]:  # Show first 3
                    print(f"  • {asset['name']} (v{asset['version']})")
            else:
                print("📭 No data assets found (this is normal for new workspaces)")
            
            return True
        else:
            print("❌ No ML client available for data asset operations")
            return False
            
    except Exception as e:
        print(f"❌ Data asset operations test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Azure ML SDK V2 Setup Verification")
    print("=" * 50)
    print()
    
    # Run tests
    tests = [
        ("SDK Installation", test_azure_ml_sdk_installation),
        ("Azure Authentication", test_azure_authentication),
        ("Workspace Connectivity", test_workspace_connectivity),
        ("Data Asset Operations", test_data_asset_operations),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("📊 Test Summary")
    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print()
    print(f"📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Azure ML SDK V2 setup is ready.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("\n💡 Common solutions:")
        print("1. Install Azure ML SDK V2: pip install azure-ai-ml azure-identity")
        print("2. Login to Azure: az login")
        print("3. Check your workspace configuration in production_config.yaml")


if __name__ == "__main__":
    main()