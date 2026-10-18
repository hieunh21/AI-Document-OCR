"""FastAPI entry point.

Run from the repository root (so both ``ai`` and ``app`` are importable)::

    python -m uvicorn app.main:app --app-dir backend --reload
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health
from app.core.config import APP_VERSION, get_settings

logger = logging.getLogger("vietdocai.api")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="VietDocAI API", version=APP_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%s latency_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    app.include_router(health.router)
    app.include_router(documents.router, prefix=settings.api_prefix)
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
