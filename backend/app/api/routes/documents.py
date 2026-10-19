from __future__ import annotations

import json
from datetime import date
from typing import Literal

from fastapi import APIRouter, Body, Depends, File, Query, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ai.schemas import DocumentStatus
from app.api.deps import get_db, get_pipeline_runner, get_storage, limit_uploads, require_api_key
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.schemas.documents import (
    ApproveRequest,
    DocumentDetail,
    DocumentList,
    DocumentStats,
    FieldsUpdate,
)
from app.services import documents as svc
from app.services.processing import PipelineRunner
from app.services.storage import Storage

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_api_key)])

ExportFormat = Literal["json", "csv"]
MAX_EXPORT_ROWS = 10_000


def _detail(doc, settings: Settings) -> DocumentDetail:
    return svc.to_detail(doc, f"{settings.api_prefix}/documents/{doc.id}/image")


class _Filters:
    def __init__(
        self,
        status: DocumentStatus | None = None,
        seller: str | None = Query(default=None, max_length=200, description="contains"),
        created_from: date | None = None,
        created_to: date | None = None,
        receipt_from: date | None = Query(default=None, description="receipt date >="),
        receipt_to: date | None = Query(default=None, description="receipt date <="),
    ) -> None:
        self.values = {
            "status": status,
            "seller": seller,
            "created_from": created_from,
            "created_to": created_to,
            "receipt_from": receipt_from,
            "receipt_to": receipt_to,
        }


@router.get("", response_model=DocumentList)
def list_documents(
    filters: _Filters = Depends(),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> DocumentList:
    items, total = svc.list_documents(db, page, page_size, **filters.values)
    return DocumentList(
        items=[svc.to_summary(d) for d in items], total=total, page=page, page_size=page_size
    )


@router.get("/export")
def export_documents(
    filters: _Filters = Depends(),
    format: ExportFormat = "csv",
    db: Session = Depends(get_db),
) -> Response:
    """Every document matching the filters, e.g. ``?status=approved&format=csv``."""
    q = svc.query_documents(db, **filters.values).limit(MAX_EXPORT_ROWS)
    records = [svc.export_record(d) for d in db.scalars(q)]
    return _export_response(records, format, "documents")


@router.get("/stats", response_model=DocumentStats)
def document_stats(db: Session = Depends(get_db)) -> DocumentStats:
    return DocumentStats(**svc.stats(db))


@router.post("", response_model=DocumentDetail, status_code=201)
def upload_document(
    response: Response,
    file: UploadFile = File(..., description="JPG, PNG or WebP receipt photo"),
    process: bool = Query(default=False, description="also run the pipeline now"),
    _: None = Depends(limit_uploads),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    runner: PipelineRunner = Depends(get_pipeline_runner),
    settings: Settings = Depends(get_settings),
) -> DocumentDetail:
    """Store the image (201). The same bytes uploaded again return the existing
    document with 200 instead of a duplicate."""
    data = file.file.read(settings.max_upload_bytes + 1)
    doc, created = svc.create_document(db, storage, data, file.filename, settings)
    if not created:
        response.status_code = 200
    elif process:
        svc.process_document(db, storage, runner, doc)
    return _detail(doc, settings)


@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(
    doc_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_settings)
) -> DocumentDetail:
    return _detail(svc.get_document(db, doc_id), settings)


@router.get("/{doc_id}/image")
def get_image(
    doc_id: str, db: Session = Depends(get_db), storage: Storage = Depends(get_storage)
) -> Response:
    doc = svc.get_document(db, doc_id)
    path = storage.path(doc.storage_key)
    if not path.exists():
        raise AppError(404, "file_missing", "The image file is missing from storage.")
    headers = {"Cache-Control": "private, max-age=3600"}
    upright = svc.oriented_image(path)
    if upright is not None:
        return Response(upright[0], media_type=upright[1], headers=headers)
    return FileResponse(path, media_type=doc.content_type, headers=headers)


@router.post("/{doc_id}/process", response_model=DocumentDetail)
def process_document(
    doc_id: str,
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    runner: PipelineRunner = Depends(get_pipeline_runner),
    settings: Settings = Depends(get_settings),
) -> DocumentDetail:
    """Run the pipeline (synchronously, about 1 s on GPU once the models are loaded)."""
    doc = svc.process_document(db, storage, runner, svc.get_document(db, doc_id))
    return _detail(doc, settings)


@router.patch("/{doc_id}/fields", response_model=DocumentDetail)
def update_fields(
    doc_id: str,
    update: FieldsUpdate,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DocumentDetail:
    doc = svc.update_fields(db, svc.get_document(db, doc_id), update)
    return _detail(doc, settings)


@router.post("/{doc_id}/approve", response_model=DocumentDetail)
def approve_document(
    doc_id: str,
    body: ApproveRequest | None = Body(default=None),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> DocumentDetail:
    reviewer = body.reviewer if body else None
    doc = svc.approve_document(db, svc.get_document(db, doc_id), reviewer)
    return _detail(doc, settings)


@router.get("/{doc_id}/export")
def export_document(
    doc_id: str, format: ExportFormat = "json", db: Session = Depends(get_db)
) -> Response:
    record = svc.export_record(svc.get_document(db, doc_id))
    return _export_response(record if format == "json" else [record], format, doc_id)


def _export_response(data, format: ExportFormat, name: str) -> Response:
    if format == "csv":
        rows = data if isinstance(data, list) else [data]
        return Response(
            svc.to_csv(rows),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
        )
    return Response(
        json.dumps(data, ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{name}.json"'},
    )
