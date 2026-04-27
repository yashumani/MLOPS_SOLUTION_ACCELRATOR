"""Drift alerting: Microsoft Teams webhook + Azure Communication Services email.

Both channels are no-ops when their environment variables are unset, so this
module is safe to import from steps that may run without alerting configured.

Environment variables:
  TEAMS_WEBHOOK_URL                  Incoming-webhook URL for the target channel.
  ACS_CONNECTION_STRING              Azure Communication Services connection string.
  ACS_SENDER_ADDRESS                 Verified sender address (e.g. DoNotReply@<domain>).
  DRIFT_ALERT_RECIPIENTS             Comma-separated list of recipient email addresses.

Design notes:
  - Never raises on send failure; logs and returns False so callers cannot break
    the pipeline because of a transient alerting outage.
  - Uses urllib (stdlib) for Teams to avoid adding a runtime dependency.
  - ACS email uses azure-communication-email if installed; otherwise skipped.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_TEAMS_TIMEOUT_SEC = 10


def _post_teams_webhook(webhook_url: str, payload: dict) -> bool:
    """POST a MessageCard to a Teams incoming webhook. Returns True on 2xx."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_TEAMS_TIMEOUT_SEC) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                logger.warning("Teams webhook returned HTTP %s", resp.status)
            return ok
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        logger.warning("Teams webhook failed: %s", exc)
        return False


def _build_teams_card(title: str, summary: str, facts: dict[str, Any]) -> dict:
    """Construct a minimal Teams MessageCard payload."""
    return {
        "@type": "MessageCard",
        "@context": "https://schema.org/extensions",
        "summary": title,
        "themeColor": "D32F2F",
        "title": title,
        "sections": [
            {
                "activityTitle": summary,
                "facts": [{"name": str(k), "value": str(v)} for k, v in facts.items()],
                "markdown": True,
            }
        ],
    }


def send_teams_alert(title: str, summary: str, facts: dict[str, Any] | None = None) -> bool:
    """Send a drift alert to Microsoft Teams. No-op when TEAMS_WEBHOOK_URL is unset."""
    webhook = os.environ.get("TEAMS_WEBHOOK_URL", "").strip()
    if not webhook:
        logger.info("TEAMS_WEBHOOK_URL not set; skipping Teams alert")
        return False
    payload = _build_teams_card(title, summary, facts or {})
    return _post_teams_webhook(webhook, payload)


def _parse_recipients(raw: str) -> list[str]:
    return [addr.strip() for addr in raw.split(",") if addr.strip()]


def send_acs_email(
    subject: str,
    plain_text: str,
    html: str | None = None,
    recipients: Iterable[str] | None = None,
) -> bool:
    """Send a drift alert email via Azure Communication Services.

    No-op when ACS_CONNECTION_STRING, ACS_SENDER_ADDRESS, or recipient list
    are missing. Returns True on accepted send.
    """
    conn_str = os.environ.get("ACS_CONNECTION_STRING", "").strip()
    sender = os.environ.get("ACS_SENDER_ADDRESS", "").strip()
    if recipients is None:
        recipients = _parse_recipients(os.environ.get("DRIFT_ALERT_RECIPIENTS", ""))
    else:
        recipients = list(recipients)

    if not conn_str or not sender or not recipients:
        logger.info(
            "ACS email skipped (conn_str=%s sender=%s recipients=%d)",
            bool(conn_str), bool(sender), len(recipients),
        )
        return False

    try:
        from azure.communication.email import EmailClient  # type: ignore
    except ImportError:
        logger.warning("azure-communication-email not installed; skipping ACS email")
        return False

    try:
        client = EmailClient.from_connection_string(conn_str)
        message = {
            "senderAddress": sender,
            "recipients": {"to": [{"address": addr} for addr in recipients]},
            "content": {
                "subject": subject,
                "plainText": plain_text,
                "html": html or f"<pre>{plain_text}</pre>",
            },
        }
        poller = client.begin_send(message)
        result = poller.result()
        status = getattr(result, "status", None) or result.get("status", "Unknown")
        logger.info("ACS email send status: %s", status)
        return str(status).lower() in {"succeeded", "running"}
    except Exception as exc:  # noqa: BLE001 — alerting must never break pipeline
        logger.warning("ACS email send failed: %s", exc)
        return False


def emit_drift_alert(
    *,
    config_name: str,
    job_name: str | None,
    self_check_status: str,
    overall_psi: float,
    drifted_features: list[str],
    cadence: str,
    studio_url: str | None = None,
    extra_facts: dict[str, Any] | None = None,
) -> dict[str, bool]:
    """Emit drift alert to all configured channels.

    Returns a dict of channel→sent. Channels with no env config are silently
    skipped and reported as False. Never raises.
    """
    drifted_summary = (
        f"{len(drifted_features)} feature(s): {', '.join(drifted_features[:5])}"
        + ("…" if len(drifted_features) > 5 else "")
        if drifted_features
        else "none"
    )
    facts: dict[str, Any] = {
        "Config": config_name,
        "Job": job_name or "n/a",
        "Self-check status": self_check_status,
        "Overall PSI": f"{overall_psi:.4f}",
        "Drifted features": drifted_summary,
        "Recommended cadence": cadence,
    }
    if studio_url:
        facts["Studio"] = studio_url
    if extra_facts:
        facts.update(extra_facts)

    title = f"⚠️ Drift detected — {config_name}"
    summary = f"PSI {overall_psi:.4f} ({self_check_status}); cadence={cadence}"
    plain = "\n".join(f"{k}: {v}" for k, v in facts.items())

    results = {
        "teams": send_teams_alert(title, summary, facts),
        "email": send_acs_email(subject=title, plain_text=plain),
    }
    logger.info("Drift alert dispatch: %s", results)
    return results
