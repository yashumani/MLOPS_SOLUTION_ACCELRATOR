"""UI configuration loaded from environment variables.

Environment variables
---------------------
ENV               : "dev" (default) or "prod". In "prod", missing API_KEY raises.
API_BASE_URL      : FastAPI base URL (default http://localhost:8000).
API_KEY           : X-API-Key value. Required in prod; dev uses a fallback.
STREAMLIT_PORT    : Port the Streamlit server listens on (default 8501).
UI_REFRESH_INTERVAL : Default autorefresh interval in seconds (default 30).
"""

from __future__ import annotations

import os
import warnings

from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "dev").lower()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Dev-only fallback. In prod we require an explicit API_KEY env var.
_DEV_FALLBACK_KEY = "svm-mlops-dev-key-x9k2p8r4t7"
_raw_key = os.getenv("API_KEY")

if _raw_key:
    API_KEY = _raw_key
elif ENV == "prod":
    raise RuntimeError(
        "API_KEY environment variable is required when ENV=prod. "
        "Set API_KEY in your environment before starting the UI."
    )
else:
    warnings.warn(
        "API_KEY not set — using the dev fallback key. "
        "Set API_KEY in your environment for any non-dev deployment.",
        stacklevel=2,
    )
    API_KEY = _DEV_FALLBACK_KEY

STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
REFRESH_INTERVAL = int(os.getenv("UI_REFRESH_INTERVAL", "30"))
