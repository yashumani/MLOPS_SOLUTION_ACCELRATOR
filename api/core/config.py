"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    api_key: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_reload: bool = False
    cors_allow_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    # UI
    ui_base_url: str = ""
    ui_port: int = 8501

    # Azure ML
    azure_subscription_id: str = ""
    azure_resource_group: str = ""
    azure_workspace_name: str = ""
    compute_target: str = ""

    # Experiments cache (warmed at startup, refreshed in background)
    experiment_cache_enabled: bool = True
    experiment_cache_preload_count: int = 20
    experiment_cache_ttl_seconds: int = 120

    # Job notification reports (SMTP-backed, secrets supplied through env)
    notification_recipient_email: str = ""
    notification_sender_email: str = ""
    notification_smtp_host: str = "smtp.gmail.com"
    notification_smtp_port: int = 587
    notification_smtp_username: str = ""
    notification_smtp_password: str = ""
    notification_smtp_starttls: bool = True
    notification_smtp_ssl: bool = False
    notification_smtp_timeout_seconds: int = 20
    notification_report_dir: str = "outputs/notifications"
    notification_max_attachment_bytes: int = 5_000_000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def cors_origins(self) -> list[str]:
        """Return explicit CORS origins from comma-separated configuration."""
        origins = [
            origin.strip()
            for origin in self.cors_allow_origins.split(",")
            if origin.strip()
        ]
        if self.ui_base_url.strip():
            origins.append(self.ui_base_url.strip().rstrip("/"))
        return list(dict.fromkeys(origins))


settings = Settings()
