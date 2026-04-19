"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    api_key: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Azure ML
    azure_subscription_id: str = ""
    azure_resource_group: str = ""
    azure_workspace_name: str = ""
    compute_target: str = "mlopsv2computecluster"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
