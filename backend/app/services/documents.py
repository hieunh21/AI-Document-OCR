"""Document workflow: upload -> process -> review (correct fields) -> approve -> export.

Status flow (``ai.schemas.DocumentStatus``)::

    uploaded -> processing -> ready | needs_review | failed
    ready | needs_review --(corrections)--> ready | needs_review
    ready | needs_review --(approve)--> approved          (locked: no edit, no re-process)
    failed | ready | needs_review --(process again)--> processing -> ...

Re-processing keeps every earlier extraction and correction as history; the current
values restart from the new AI result.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import uuid
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ai.kie.validate import validate_extraction
from ai.normalization.datetime_vi import parse_timestamp
from ai.normalization.money import parse_vnd
from ai.preprocessing.image import ImageError, load_image
from ai.schemas import (
    FIELD_NAMES,
    DocumentStatus,
    ExtractionWarning,
    OCRResult,
    ReceiptExtraction,
    ReceiptFields,
)
from app.core.config import Settings
from app.core.errors import AppError
from app.db import utcnow
from app.models import Document, DocumentEvent, Extraction, FieldCorrection
from app.schemas.documents import DocumentDetail, DocumentSummary, FieldsUpdate
from app.services.processing import PipelineRunner
from app.services.storage import Storage

logger = logging.getLogger("vietdocai.documents")

FORMATS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}
COLUMN = {  # field name -> Document attribute
    "seller": "seller",
    "seller_address": "seller_address",
    "timestamp": "receipt_timestamp",
    "total_cost": "total_cost",
}
EXPORT_COLUMNS = [
    "id",
    "original_name",
    "status",
    "seller",
    "seller_address",
    "timestamp",
    "total_cost",
    "created_at",
    "approved_at",
    "approved_by",
    "pipeline_version",
]


# ----------------------------------------------------------------------------- upload


def inspect_upload(data: bytes, settings: Settings) -> tuple[str, str, int, int]:
    """Content type, extension and EXIF-corrected size, or an ``AppError``."""
    if not data:
        raise AppError(400, "empty_file", "The uploaded file is empty.")
    if len(data) > settings.max_upload_bytes:
        raise AppError(
            413, "file_too_large", f"The file is larger than {settings.max_upload_mb} MB."
        )
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise AppError(415, "unsupported_file", "The file is not a JPG, PNG or WebP image.") from e
    if fmt not in FORMATS:
        raise AppError(415, "unsupported_file", f"{fmt} images are not supported; use JPG or PNG.")
    if max(w, h) > settings.max_image_side:
        raise AppError(
            422, "image_too_large", f"Image is {w}x{h}px; the limit is {settings.max_image_side}px."
        )
    try:
        image = load_image(data)  # decodes fully: catches truncated files
    except ImageError as e:
        raise AppError(422, e.code, str(e)) from e
    content_type, ext = FORMATS[fmt]
    return content_type, ext, *image.size


def create_document(
    db: Session, storage: Storage, data: bytes, filename: str | None, settings: Settings
) -> tuple[Document, bool]:
    """Store an upload. The same bytes uploaded again return the existing document."""
    content_type, ext, width, height = inspect_upload(data, settings)
    sha = hashlib.sha256(data).hexdigest()
    existing = db.scalar(
        select(Document).where(Document.sha256 == sha).order_by(Document.created_at).limit(1)
    )
    if existing is not None:
        return existing, False

    doc_id = uuid.uuid4().hex
    key = storage.save(doc_id, ext, data)
    name = Path(filename or f"upload{ext}").name[:255] or f"upload{ext}"
    doc = Document(
        id=doc_id,
        storage_key=key,
        original_name=name,
        content_type=content_type,
        size_bytes=len(data),
        sha256=sha,
        image_width=width,
        image_height=height,
        status=DocumentStatus.UPLOADED.value,
    )
    doc.events.append(DocumentEvent(action="uploaded", detail={"size_bytes": len(data)}))
    db.add(doc)
    try:
        db.commit()
    except Exception:
        db.rollback()
        storage.delete(key)
        raise
    return doc, True


# ----------------------------------------------------------------------------- process


def _fail(db: Session, doc: Document, code: str, message: str) -> None:
    doc.status = DocumentStatus.FAILED.value
    doc.error = message
    doc.events.append(DocumentEvent(action="failed", detail={"code": code, "message": message}))
    db.commit()


def process_document(
    db: Session, storage: Storage, runner: PipelineRunner, doc: Document
) -> Document:
    status = DocumentStatus(doc.status)
    if status is DocumentStatus.APPROVED:
        raise AppError(409, "approved_locked", "An approved document cannot be processed again.")
    if status is DocumentStatus.PROCESSING:
        raise AppError(409, "already_processing", "The document is being processed.")
    doc.status = DocumentStatus.PROCESSING.value
    doc.error = None
    db.commit()

    path = storage.path(doc.storage_key)
    if not path.exists():
        _fail(db, doc, "file_missing", "The image file is missing from storage.")
        raise AppError(500, "file_missing", "The image file is missing from storage.")
    try:
        image = load_image(path)
        result, model_version = runner.run(image)
    except ImageError as e:
        _fail(db, doc, e.code, str(e))
        raise AppError(422, e.code, str(e)) from e
    except FileNotFoundError as e:
        logger.exception("document %s: file or model missing", doc.id)
        _fail(db, doc, "file_missing", f"A required file is missing: {e}")
        raise AppError(500, "file_missing", f"A required file is missing: {e}") from e
    except Exception as e:  # inference must never leave the document stuck in "processing"
        logger.exception("document %s: inference failed", doc.id)
        _fail(db, doc, "inference_failed", f"Inference failed: {type(e).__name__}")
        raise AppError(500, "inference_failed", "Inference failed; see the server log.") from e

    ex = result.extraction
    usage = result.llm_usage
    extraction = Extraction(
        pipeline_name=runner.info.name,
        pipeline_version=runner.info.version,
        model_version=model_version[:255],
        raw_json={
            "ocr": result.ocr.model_dump(mode="json"),
            "quality": result.quality.model_dump(mode="json") if result.quality else None,
            "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
            "llm_usage": asdict(usage) if is_dataclass(usage) else None,
        },
        normalized_json=ex.model_dump(mode="json"),
        processing_time_ms=round(result.timings_ms["total"], 1),
    )
    doc.extractions.append(extraction)
    for f, col in COLUMN.items():
        setattr(doc, col, getattr(ex.fields, f))
    doc.status = ex.suggested_status().value
    doc.events.append(
        DocumentEvent(
            action="processed",
            detail={
                "pipeline_version": runner.info.version,
                "model_version": model_version,
                "status": doc.status,
                "processing_time_ms": extraction.processing_time_ms,
            },
        )
    )
    db.commit()
    return doc


# ----------------------------------------------------------------------------- review


def current_fields(doc: Document) -> ReceiptFields:
    return ReceiptFields(**{f: getattr(doc, col) for f, col in COLUMN.items()})


def field_sources(doc: Document) -> dict[str, str]:
    latest = doc.latest_extraction
    if latest is None:
        return {}
    human = {c.field_name for c in doc.corrections if c.extraction_id == latest.id}
    return {f: "human" if f in human else "ai" for f in FIELD_NAMES}


def current_warnings(doc: Document) -> list[ExtractionWarning]:
    """AI warnings on fields nobody corrected, plus checks on the current values.

    A field a reviewer set to ``null`` is confirmed absent, so it is not "missing".
    """
    latest = doc.latest_extraction
    if latest is None:
        return []
    ai = ReceiptExtraction.model_validate(latest.normalized_json)
    sources = field_sources(doc)
    keep = [w for w in ai.warnings if w.field is None or sources.get(w.field) == "ai"]
    ex = ReceiptExtraction(
        model_version=ai.model_version, fields=current_fields(doc), warnings=keep
    )
    have = {(w.code, w.field) for w in keep}
    for f in FIELD_NAMES:
        if (
            getattr(ex.fields, f) is None
            and sources[f] == "ai"
            and ("missing_field", f) not in have
        ):
            ex.warnings.append(
                ExtractionWarning(code="missing_field", field=f, message=f"No {f} found.")
            )
    return validate_extraction(ex).warnings


def _review_status(doc: Document) -> str:
    ready = not current_warnings(doc)
    return (DocumentStatus.READY if ready else DocumentStatus.NEEDS_REVIEW).value


def _normalize(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field == "timestamp":
        try:
            return ReceiptFields(timestamp=str(value).strip()).timestamp
        except ValidationError:
            parsed = parse_timestamp(str(value))
            if parsed is None or parsed.iso is None:
                raise AppError(
                    422,
                    "invalid_timestamp",
                    f"Cannot read a date from {value!r}; use YYYY-MM-DD[THH:MM[:SS]].",
                ) from None
            return parsed.iso
    if field == "total_cost":
        amount = value if isinstance(value, int) else parse_vnd(str(value))
        if amount is None or amount < 0:
            raise AppError(422, "invalid_total_cost", f"Cannot read an amount from {value!r}.")
        return amount
    return getattr(ReceiptFields(**{field: value}), field)  # collapses whitespace, "" -> None


def update_fields(db: Session, doc: Document, update: FieldsUpdate) -> Document:
    if doc.status == DocumentStatus.APPROVED:
        raise AppError(409, "approved_locked", "An approved document cannot be edited.")
    latest = doc.latest_extraction
    if latest is None or doc.status == DocumentStatus.PROCESSING:
        raise AppError(409, "not_processed", "Process the document before editing its fields.")
    requested = [f for f in FIELD_NAMES if f in update.model_fields_set]
    if not requested:
        raise AppError(422, "no_fields", "The request does not change any field.")

    changed = []
    for f in requested:
        new = _normalize(f, getattr(update, f))
        old = getattr(doc, COLUMN[f])
        if new == old:
            continue
        doc.corrections.append(
            FieldCorrection(
                extraction_id=latest.id,
                field_name=f,
                old_value=old,
                new_value=new,
                reviewer=update.reviewer,
            )
        )
        setattr(doc, COLUMN[f], new)
        changed.append(f)
    if changed:
        db.flush()  # corrections must be visible to field_sources()
        doc.status = _review_status(doc)
        doc.events.append(
            DocumentEvent(action="corrected", actor=update.reviewer, detail={"fields": changed})
        )
        db.commit()
    return doc


def approve_document(db: Session, doc: Document, reviewer: str | None) -> Document:
    status = DocumentStatus(doc.status)
    if status is DocumentStatus.APPROVED:
        raise AppError(409, "already_approved", "The document is already approved.")
    if status not in (DocumentStatus.READY, DocumentStatus.NEEDS_REVIEW):
        raise AppError(409, "not_processed", "Only a processed document can be approved.")
    warnings = current_warnings(doc)
    doc.status = DocumentStatus.APPROVED.value
    doc.approved_at = utcnow()
    doc.approved_by = reviewer
    doc.events.append(
        DocumentEvent(
            action="approved",
            actor=reviewer,
            detail={
                "fields": current_fields(doc).model_dump(),
                "open_warnings": [w.code for w in warnings],
            },
        )
    )
    db.commit()
    return doc


# ----------------------------------------------------------------------------- read


def get_document(db: Session, doc_id: str) -> Document:
    doc = db.get(Document, doc_id)
    if doc is None:
        raise AppError(404, "not_found", f"Document {doc_id!r} not found")
    return doc


def _summary_values(doc: Document) -> dict[str, Any]:
    return {
        "id": doc.id,
        "original_name": doc.original_name,
        "status": doc.status,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "seller": doc.seller,
        "seller_address": doc.seller_address,
        "timestamp": doc.receipt_timestamp,
        "total_cost": doc.total_cost,
    }


def to_summary(doc: Document) -> DocumentSummary:
    return DocumentSummary(**_summary_values(doc))


def to_detail(doc: Document, image_url: str) -> DocumentDetail:
    latest = doc.latest_extraction
    raw = latest.raw_json if latest else {}
    return DocumentDetail(
        **_summary_values(doc),
        image_url=image_url,
        content_type=doc.content_type,
        size_bytes=doc.size_bytes,
        sha256=doc.sha256,
        image_width=doc.image_width,
        image_height=doc.image_height,
        error=doc.error,
        approved_at=doc.approved_at,
        approved_by=doc.approved_by,
        fields=current_fields(doc),
        field_sources=field_sources(doc),
        warnings=[w.model_dump() for w in current_warnings(doc)],
        extraction_id=latest.id if latest else None,
        extraction=ReceiptExtraction.model_validate(latest.normalized_json) if latest else None,
        ocr=OCRResult.model_validate(raw["ocr"]) if raw.get("ocr") else None,
        quality=raw.get("quality"),
        pipeline_version=latest.pipeline_version if latest else None,
        model_version=latest.model_version if latest else None,
        processing_time_ms=latest.processing_time_ms if latest else None,
        timings_ms=raw.get("timings_ms"),
        corrections=doc.corrections,
        events=doc.events,
    )


def _day_start(d: date) -> datetime:
    return datetime.combine(d, time.min, tzinfo=UTC)


def query_documents(
    db: Session,
    *,
    status: DocumentStatus | None = None,
    seller: str | None = None,
    created_from: date | None = None,
    created_to: date | None = None,
    receipt_from: date | None = None,
    receipt_to: date | None = None,
):
    """SELECT for the filtered list, newest first (no paging)."""
    q = select(Document)
    if status is not None:
        q = q.where(Document.status == status.value)
    if seller:
        q = q.where(Document.seller.ilike(f"%{seller.strip()}%"))
    if created_from:
        q = q.where(Document.created_at >= _day_start(created_from))
    if created_to:
        q = q.where(Document.created_at < _day_start(date.fromordinal(created_to.toordinal() + 1)))
    if receipt_from:  # ISO strings sort like dates
        q = q.where(Document.receipt_timestamp >= receipt_from.isoformat())
    if receipt_to:
        q = q.where(Document.receipt_timestamp < f"{receipt_to.isoformat()}~")  # includes times
    return q.order_by(Document.created_at.desc(), Document.id)


def list_documents(db: Session, page: int, page_size: int, **filters) -> tuple[list[Document], int]:
    q = query_documents(db, **filters)
    total = db.scalar(select(func.count()).select_from(q.order_by(None).subquery())) or 0
    items = db.scalars(q.offset((page - 1) * page_size).limit(page_size)).all()
    return list(items), total


def stats(db: Session) -> dict[str, Any]:
    """Counts per status and the total of approved receipts (dashboard)."""
    by_status = dict.fromkeys((st.value for st in DocumentStatus), 0)
    for status, n in db.execute(select(Document.status, func.count()).group_by(Document.status)):
        by_status[status] = n
    approved_total = db.scalar(
        select(func.coalesce(func.sum(Document.total_cost), 0)).where(
            Document.status == DocumentStatus.APPROVED.value
        )
    )
    return {
        "total": sum(by_status.values()),
        "by_status": by_status,
        "approved_total_vnd": int(approved_total or 0),
    }


def oriented_image(path: Path) -> tuple[bytes, str] | None:
    """The image turned upright when its EXIF orientation is not "normal", else ``None``.

    OCR boxes are in EXIF-corrected coordinates; serving the upright pixels keeps the
    overlay aligned in every browser (SVG ``<image>`` does not always honour EXIF).
    """
    with Image.open(path) as im:
        if im.getexif().get(0x0112, 1) == 1:
            return None
        upright = ImageOps.exif_transpose(im)
        fmt = "PNG" if im.format == "PNG" else "JPEG"
        buf = io.BytesIO()
        (upright if fmt == "PNG" else upright.convert("RGB")).save(buf, format=fmt, quality=92)
    return buf.getvalue(), f"image/{fmt.lower()}"


# ----------------------------------------------------------------------------- export


def export_record(doc: Document) -> dict[str, Any]:
    latest = doc.latest_extraction
    fields = current_fields(doc).model_dump()
    return {
        "id": doc.id,
        "original_name": doc.original_name,
        "status": doc.status,
        **fields,
        "created_at": doc.created_at.isoformat(),
        "approved_at": doc.approved_at.isoformat() if doc.approved_at else None,
        "approved_by": doc.approved_by,
        "pipeline_version": latest.pipeline_version if latest else None,
        "field_sources": field_sources(doc),
        "evidence": (latest.normalized_json.get("evidence") if latest else None) or {},
        "corrections": [
            {
                "field": c.field_name,
                "old_value": c.old_value,
                "new_value": c.new_value,
                "reviewer": c.reviewer,
                "created_at": c.created_at.isoformat(),
            }
            for c in doc.corrections
        ],
    }


def to_csv(records: list[dict[str, Any]]) -> str:
    """UTF-8 CSV with a BOM so Excel shows Vietnamese correctly."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return "﻿" + buf.getvalue()


def recover_interrupted(db: Session) -> int:
    """Documents left in "processing" by a crash or restart are marked failed."""
    docs = db.scalars(
        select(Document).where(Document.status == DocumentStatus.PROCESSING.value)
    ).all()
    n = 0
    for doc in docs:
        _fail(db, doc, "interrupted", "Processing was interrupted by a server restart; retry.")
        n += 1
    return n
