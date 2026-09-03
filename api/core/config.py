"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    api_key: str = ""
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_reload: bool = False
    cors_allow_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    api_deployment_profile: str = "development"
    api_config_mutation_enabled: bool = True

    # Server-owned operational state. Production profiles must set absolute
    # paths backed by one durable mount.
    mlops_state_dir: str = ""
    mlops_submission_request_root: str = ""
    mlops_auto_retrain_ledger_root: str = ""

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

    def validate_runtime_security(self) -> None:
        """Fail closed when a deployment profile exceeds implemented controls."""
        profile = self.api_deployment_profile.strip().lower()
        allowed_profiles = {"development", "private_single_operator", "multi_user"}
        if profile not in allowed_profiles:
            raise RuntimeError(
                "API_DEPLOYMENT_PROFILE must be one of: "
                + ", ".join(sorted(allowed_profiles))
            )
        if profile == "development":
            return
        if profile == "multi_user":
            raise RuntimeError(
                "The multi_user API profile is not implemented: Entra/OIDC, "
                "actor authorization, and transactional shared state are required"
            )

        errors: list[str] = []
        if len(self.api_key) < 32:
            errors.append("API_KEY must contain at least 32 characters")
        if self.api_reload:
            errors.append("API_RELOAD must be false")
        if self.api_config_mutation_enabled:
            errors.append("API_CONFIG_MUTATION_ENABLED must be false")

        origins = self.cors_origins()
        if not origins:
            errors.append("CORS_ALLOW_ORIGINS must contain the deployed UI origin")
        elif any(origin == "*" or not origin.startswith("https://") for origin in origins):
            errors.append("every production CORS/UI origin must use explicit https://")

        durable_paths = {
            "MLOPS_STATE_DIR": self.mlops_state_dir,
            "MLOPS_SUBMISSION_REQUEST_ROOT": self.mlops_submission_request_root,
            "MLOPS_AUTO_RETRAIN_LEDGER_ROOT": self.mlops_auto_retrain_ledger_root,
            "NOTIFICATION_REPORT_DIR": self.notification_report_dir,
        }
        for name, raw_path in durable_paths.items():
            if not raw_path.strip() or not Path(raw_path).expanduser().is_absolute():
                errors.append(f"{name} must be an absolute durable path")

        if errors:
            raise RuntimeError(
                "Unsafe private_single_operator API configuration: " + "; ".join(errors)
            )


settings = Settings()
