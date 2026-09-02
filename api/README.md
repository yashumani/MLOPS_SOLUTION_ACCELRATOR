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
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API metadata and links (no auth) |
| `GET` | `/healthz` | Lightweight liveness probe (no auth) |
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
| `POST` | `/api/v1/pipelines/jobs/{name}/notifications/email` | Generate Markdown/JSON/CSV report files and send by SMTP |
| `POST` | `/api/v1/pipelines/baseline/capture` | Capture drift baseline |
| `POST` | `/api/v1/pipelines/resubmit` | Resubmit a job |
| `GET` | `/api/v1/configs/schema` | Return config JSON schema for guided forms |
| `POST` | `/api/v1/configs/validate` | Validate a config draft without saving |
| `POST` | `/api/v1/configs/preview` | Preview S01-S09 plan and key execution settings |

## Authentication

All `/api/v1/pipelines/*` and `/api/v1/configs/*` endpoints require an
`X-API-Key` header matching the `API_KEY` environment variable.

## Environment Variables

See [`.env.example`](../.env.example) for the full list.

### SMTP Notification Reports

`POST /api/v1/pipelines/jobs/{name}/notifications/email` creates a report
folder under `outputs/notifications/` and writes four attachments:

- Markdown operator brief
- JSON machine-readable payload
- Drift feature PSI CSV
- Pipeline step status CSV

SMTP is configured only through environment variables so the server can be
changed without code edits. The default profile uses Gmail SMTP over STARTTLS;
set `NOTIFICATION_SMTP_PASSWORD` to a Gmail app password in `.env`.

```bash
NOTIFICATION_RECIPIENT_EMAIL=mlops-oncall@example.com
NOTIFICATION_SENDER_EMAIL=mlops-notifications@example.com
NOTIFICATION_SMTP_HOST=smtp.gmail.com
NOTIFICATION_SMTP_PORT=587
NOTIFICATION_SMTP_USERNAME=mlops-notifications@example.com
NOTIFICATION_SMTP_PASSWORD=<gmail-app-password>
NOTIFICATION_SMTP_STARTTLS=true
NOTIFICATION_SMTP_SSL=false
```

To generate files without sending email, post `{"dry_run": true}`.
