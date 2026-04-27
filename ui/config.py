"""UI configuration loaded from environment variables.

Environment variables
---------------------
ENV               : "dev" (default) or "prod".
API_BASE_URL      : FastAPI base URL (default http://localhost:8000).
API_KEY           : X-API-Key value. Required for all UI starts.
STREAMLIT_PORT    : Port the Streamlit server listens on (default 8501).
UI_REFRESH_INTERVAL : Default autorefresh interval in seconds (default 30).
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "dev").lower()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

_raw_key = os.getenv("API_KEY")

if _raw_key:
    API_KEY = _raw_key
else:
    raise RuntimeError(
        "API_KEY environment variable is required before starting the UI. "
        "Set API_KEY in your environment or .env file."
    )

STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
REFRESH_INTERVAL = int(os.getenv("UI_REFRESH_INTERVAL", "30"))
