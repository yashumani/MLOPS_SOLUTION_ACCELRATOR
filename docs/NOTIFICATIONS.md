# Pipeline Notification Reports

The FastAPI backend can generate a job notification package for any Azure ML
pipeline job and send it by SMTP to the single configured recipient.

## What Gets Sent

Each notification creates a timestamped folder under `outputs/notifications/`
and attaches:

| File | Purpose |
|---|---|
| `*_notification.md` | Human-readable operator brief with job status, drift signals, retrain decision, top features, and Studio link. |
| `*_notification.json` | Machine-readable payload for automation or audit capture. |
| `*_drift_features.csv` | Per-feature PSI values and severity. Empty except for headers when no drift report exists yet. |
| `*_steps.csv` | Pipeline stage status and timestamps. |

The email subject is built as:

```text
MLOps V3 | <experiment> | <YYYY-MM-DD HH:MM UTC> | <status> | <task>/<dataset>
```

## SMTP Configuration

Set these environment variables in the API runtime. No SMTP credentials are
stored in code. The defaults are configured for Gmail SMTP using STARTTLS.
Use a Gmail app password for `NOTIFICATION_SMTP_PASSWORD`; do not use your
normal Google account password.

```bash
NOTIFICATION_RECIPIENT_EMAIL=mlops-oncall@example.com
NOTIFICATION_SENDER_EMAIL=mlops-notifications@example.com
NOTIFICATION_SMTP_HOST=smtp.gmail.com
NOTIFICATION_SMTP_PORT=587
NOTIFICATION_SMTP_USERNAME=mlops-notifications@example.com
NOTIFICATION_SMTP_PASSWORD=<gmail-app-password>
NOTIFICATION_SMTP_STARTTLS=true
NOTIFICATION_SMTP_SSL=false
NOTIFICATION_REPORT_DIR=outputs/notifications
NOTIFICATION_MAX_ATTACHMENT_BYTES=5000000
```

`NOTIFICATION_RECIPIENT_EMAIL` is intentionally singular. The API request does
not accept an arbitrary recipient, so UI users cannot redirect operational
reports to other addresses.

## API Usage

Generate report files without sending email:

```bash
curl -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}' \
  "$API_BASE_URL/api/v1/pipelines/jobs/<job-name>/notifications/email"
```

Send the email using configured SMTP:

```bash
curl -X POST \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false}' \
  "$API_BASE_URL/api/v1/pipelines/jobs/<job-name>/notifications/email"
```

If SMTP is not configured, the endpoint still generates the report files and
returns `status: not_configured` with the missing settings.
