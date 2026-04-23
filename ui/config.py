"""UI configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Hardcoded default key for development — override via API_KEY env var in prod
_DEFAULT_API_KEY = "svm-mlops-dev-key-x9k2p8r4t7"
API_KEY = os.getenv("API_KEY", _DEFAULT_API_KEY)

STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
REFRESH_INTERVAL = int(os.getenv("UI_REFRESH_INTERVAL", "30"))
