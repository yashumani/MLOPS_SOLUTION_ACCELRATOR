# Running the Streamlit UI on Azure ML

Streamlit is hosted through the Azure ML Compute Instance application proxy, not through an Azure ML managed online endpoint. Online endpoints are for inference-style HTTP scoring services and do not work well for Streamlit's WebSocket UI.

## Public URLs

Use the port-specific Azure ML app-proxy URLs:

| Service | Port | Purpose |
|---|---:|---|
| Streamlit dashboard | `8501` | Frontend UI |
| FastAPI backend | `8000` | API, OpenAPI docs, health checks |

For the current compute instance, the expected frontend URL is:

```text
https://mlopspipelinev2-8501.eastus2.instances.azureml.ms/
```

The `:8000` URL is expected to return API metadata JSON unless the browser is redirected to the dashboard.

## Launch

From the repository root:

```bash
bash launch_streamlit_azureml.sh
```

Required environment variables, usually set in `.env`:

```bash
API_KEY=<shared-api-key>
API_BASE_URL=http://localhost:8000
AZURE_RESOURCE_GROUP=mvpv1
AZURE_WORKSPACE_NAME=mlops-accelerator
STREAMLIT_PORT=8501
```

The script installs `ui/requirements.txt`, starts `ui/app.py`, and prints the public `8501` dashboard URL.

## API Docs

The FastAPI docs remain on the backend service:

```text
https://mlopspipelinev2-8000.eastus2.instances.azureml.ms/docs
```
