# Deploying Streamlit UI to Azure ML Studio

## Prerequisites
- Azure ML workspace (<AZURE_WORKSPACE_NAME>)
- Resource group (<AZURE_RESOURCE_GROUP>)
- Azure CLI and Azure ML extension installed
- Permissions to create endpoints

## Steps

1. **Navigate to the UI directory:**
   ```sh
   cd ui
   ```
2. **Login to Azure:**
   ```sh
   az login
   az account set --subscription <your-subscription-id>
   ```
3. **Register the environment (optional):**
   ```sh
   az ml environment create --file requirements.txt --resource-group <AZURE_RESOURCE_GROUP> --workspace-name <AZURE_WORKSPACE_NAME>
   ```
4. **Create the endpoint:**
   ```sh
   az ml online-endpoint create --name streamlit-ui-endpoint --file azureml-streamlit-deployment.yml --resource-group <AZURE_RESOURCE_GROUP> --workspace-name <AZURE_WORKSPACE_NAME>
   ```
5. **Deploy the app:**
   ```sh
   az ml online-deployment create --name streamlit-ui --endpoint-name streamlit-ui-endpoint --file azureml-streamlit-deployment.yml --resource-group <AZURE_RESOURCE_GROUP> --workspace-name <AZURE_WORKSPACE_NAME> --all-traffic
   ```
6. **Get the endpoint URL:**
   ```sh
   az ml online-endpoint show --name streamlit-ui-endpoint --resource-group <AZURE_RESOURCE_GROUP> --workspace-name <AZURE_WORKSPACE_NAME> --query "scoring_uri"
   ```

## Notes
- The entrypoint is `app.py` in the `ui/` directory.
- Adjust `requirements.txt` as needed for extra dependencies.
- For custom domains or authentication, configure in Azure ML Studio portal.
