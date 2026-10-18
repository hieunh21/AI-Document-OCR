from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.config import APP_VERSION, Settings, get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, str]:
    return {"status": "ok", "version": APP_VERSION, "env": settings.env}
