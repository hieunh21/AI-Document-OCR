"""FastAPI entry point.

Run from the repository root (so ``ai``, ``configs/`` and ``models/`` resolve)::

    alembic -c backend/alembic.ini upgrade head          # create / migrate the tables
    python -m uvicorn app.main:app --app-dir backend --reload
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import documents, health
from app.core.config import APP_VERSION, get_settings
from app.core.errors import AppError
from app.db import get_sessionmaker
from app.services.documents import recover_interrupted
from app.services.processing import get_runner

logger = logging.getLogger("vietdocai.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        with get_sessionmaker()() as db:
            n = recover_interrupted(db)
        if n:
            logger.warning("%d document(s) left in 'processing' were marked failed", n)
    except Exception:  # the API still starts; /health reports the database state
        logger.exception("database not reachable at startup")
    if settings.warmup:
        get_runner().warmup()
    yield


def _error(
    request: Request, status: int, code: str, message, headers=None, **extra
) -> JSONResponse:
    """Every error has the same shape: ``detail`` (message), ``code``, ``request_id``."""
    body = {"detail": message, "code": code, **extra}
    body["request_id"] = getattr(request.state, "rid", None)
    return JSONResponse(body, status_code=status, headers=headers)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="VietDocAI API", version=APP_VERSION, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Content-Disposition"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.rid = request_id
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

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError):
        return _error(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = "not_found" if exc.status_code == 404 else f"http_{exc.status_code}"
        return _error(request, exc.status_code, code, exc.detail, getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        message = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in errors)
        return _error(request, 422, "invalid_request", message, errors=errors)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error request_id=%s", getattr(request.state, "rid", None))
        return _error(request, 500, "internal_error", "Internal server error.")

    app.include_router(health.router)
    app.include_router(documents.router, prefix=settings.api_prefix)
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
