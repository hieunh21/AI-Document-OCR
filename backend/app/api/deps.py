"""FastAPI dependencies: settings, database session, storage, pipeline, auth, rate limit."""

from __future__ import annotations

import secrets
import threading
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, Header, Request

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.db import get_db  # noqa: F401  (re-exported for routes)
from app.services.processing import PipelineRunner, get_runner
from app.services.storage import Storage


@lru_cache
def _storage(root: str) -> Storage:
    return Storage(Path(root))


def get_storage(settings: Settings = Depends(get_settings)) -> Storage:
    return _storage(settings.storage_dir)


def get_pipeline_runner() -> PipelineRunner:
    return get_runner()


def require_api_key(
    settings: Settings = Depends(get_settings), x_api_key: str | None = Header(default=None)
) -> None:
    """No-op unless ``VIETDOCAI_API_KEY`` is set."""
    if settings.api_key and not (x_api_key and secrets.compare_digest(x_api_key, settings.api_key)):
        raise AppError(401, "unauthorized", "Missing or wrong X-API-Key header.")


class RateLimiter:
    """Sliding one-minute window per client address (in memory, per process)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= limit:
                retry = int(60 - (now - hits[0])) + 1
                raise AppError(429, "rate_limited", f"Too many uploads; try again in {retry} s.")
            hits.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


upload_limiter = RateLimiter()


def limit_uploads(request: Request, settings: Settings = Depends(get_settings)) -> None:
    client = request.client.host if request.client else "unknown"
    upload_limiter.check(client, settings.uploads_per_minute)
