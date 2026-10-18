from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_VERSION = "0.1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIETDOCAI_", env_file=".env", extra="ignore")

    env: str = "dev"
    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5173"  # comma-separated
    storage_dir: str = "storage"
    max_upload_mb: int = 10

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
