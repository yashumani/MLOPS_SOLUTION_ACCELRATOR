# MLOps V3 Pipeline Management API

FastAPI wrapper around the Azure ML V3 pipeline. Submit, monitor, cancel, and
inspect pipeline jobs without touching the Azure portal.

## Quick Start

Run these commands only on the approved existing server or Azure compute
instance. This project's release work prohibits local application execution.
Confirm the host, identity, durable state and HTTPS configuration before
starting a shared service; these commands do not provision or approve a host.

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

Protected pipeline and configuration routes use the selected deployment profile:

- `development` and `private_single_operator` require `X-API-Key` matching
  `API_KEY`. The private profile requires at least 32 characters, explicit HTTPS
  origins, reload/config mutation disabled and absolute durable state/report
  paths. Neither profile authorizes a public shared deployment.
- `multi_user` requires a single-tenant Entra delegated bearer token, an allowed
  client/application scope and a server-managed user record. Startup requires
  the configured bootstrap allowlist to contain exactly one admin, initially
  Yashu's confirmed tenant/object identity. API keys are not a bearer fallback.
  Admins manage users through `/api/v1/users`; roles are `viewer`, `operator`
  and `admin`. Mutating operations require operator/admin access and user
  management requires admin access. Request-audit writes fail closed when
  durable state is unavailable.

The multi-user configuration requires the existing Entra application bindings,
explicit HTTPS origins/redirect and an absolute local-disk
`MLOPS_OPERATIONAL_STATE_DB` SQLite path. This is a single-host transactional
state implementation, not a distributed state service. Do not put SQLite WAL
on a network filesystem or infer host/identity approval from credentials.

The React UI does not read a server key from Vite/public runtime configuration.
The full website remains optional for release. Implemented authentication and
green tests do not replace live hosted authorized/unauthorized request evidence.

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
