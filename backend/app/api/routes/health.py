from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_pipeline_runner
from app.core.config import APP_VERSION, Settings, get_settings
from app.services.processing import PipelineRunner

router = APIRouter(tags=["health"])


@router.get("/health")
def health(
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
    runner: PipelineRunner = Depends(get_pipeline_runner),
) -> dict:
    """``status`` is "ok" only when the database answers; the pipeline loads lazily."""
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        database = "unavailable"
    return {
        "status": "ok" if database == "ok" else "degraded",
        "version": APP_VERSION,
        "env": settings.env,
        "database": database,
        "limits": {"max_upload_mb": settings.max_upload_mb},
        "pipeline": {
            "name": runner.info.name,
            "version": runner.info.version,
            "loaded": runner.loaded,
        },
    }
