# MLOps V3 Pipeline Management API

FastAPI wrapper around the Azure ML V3 pipeline. Submit, monitor, cancel, and
inspect pipeline jobs without touching the Azure portal.

## Quick Start

```bash
# 1. Copy and fill environment variables
cp .env.example .env

# 2. Install dependencies
pip install -r api/requirements.txt

# 3. Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check (no auth) |
| `GET` | `/api/v1/configs` | List available pipeline configs |
| `GET` | `/api/v1/configs/{name}` | Get a specific config |
| `POST` | `/api/v1/pipelines/submit` | Submit a new pipeline job |
| `GET` | `/api/v1/pipelines/jobs` | List pipeline jobs |
| `GET` | `/api/v1/pipelines/jobs/{name}` | Job status + child steps |
| `POST` | `/api/v1/pipelines/jobs/{name}/cancel` | Cancel a running job |
| `GET` | `/api/v1/pipelines/jobs/{name}/outputs` | List job outputs |
| `GET` | `/api/v1/pipelines/jobs/{name}/outputs/{out}/download` | Download output |
| `GET` | `/api/v1/pipelines/jobs/{name}/metrics` | MLflow metrics |
| `GET` | `/api/v1/pipelines/jobs/{name}/drift` | Drift analysis |
| `POST` | `/api/v1/pipelines/baseline/capture` | Capture drift baseline |
| `POST` | `/api/v1/pipelines/resubmit` | Resubmit a job |

## Authentication

All `/api/v1/pipelines/*` and `/api/v1/configs/*` endpoints require an
`X-API-Key` header matching the `API_KEY` environment variable.

## Environment Variables

See [`.env.example`](../.env.example) for the full list.
