from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_VERSION = "0.5.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIETDOCAI_", env_file=".env", extra="ignore")

    env: str = "dev"
    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5173"  # comma-separated

    # Storage: uploaded files live on disk; PostgreSQL only holds their key.
    database_url: str = "postgresql+psycopg://vietdocai:vietdocai@localhost:5432/vietdocai"
    storage_dir: str = "storage"
    max_upload_mb: int = 10
    max_image_side: int = 10_000  # pixels; larger images are rejected
    uploads_per_minute: int = 30  # per client address; 0 disables the limit

    # Inference: a frozen pipeline from configs/pipeline.json.
    pipeline: str = "default"
    device: str = "gpu:0"
    warmup: bool = False  # load the models at startup instead of on the first request

    # Optional shared secret (X-API-Key header) for deployments reachable from the internet.
    api_key: str | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def storage_path(self) -> Path:
        return Path(self.storage_dir)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
