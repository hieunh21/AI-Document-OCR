"""In-memory documents produced by the mock pipeline (Phase 0).

Replaced by the PostgreSQL repository in Phase 5; the API layer only depends
on ``list_documents`` / ``get_document``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache

from PIL import Image

from ai.mock import MockKIE, MockOCR
from ai.pipeline import ReceiptPipeline
from ai.schemas import DocumentStatus
from app.schemas.documents import DocumentDetail, DocumentSummary


@lru_cache
def _documents() -> dict[str, DocumentDetail]:
    pipeline = ReceiptPipeline(MockOCR(), MockKIE())
    result = pipeline.run(Image.new("RGB", (768, 1024), "white"))
    ex = result.extraction
    base = datetime(2026, 10, 8, 9, 30, tzinfo=UTC)

    def doc(i: int, name: str, status: DocumentStatus, with_result: bool = True, error=None):
        return DocumentDetail(
            id=f"mock-{i:04d}",
            original_name=name,
            status=status,
            created_at=base + timedelta(minutes=7 * i),
            seller=ex.fields.seller if with_result else None,
            timestamp=ex.fields.timestamp if with_result else None,
            total_cost=ex.fields.total_cost if with_result else None,
            image_width=result.ocr.image_width,
            image_height=result.ocr.image_height,
            ocr=result.ocr if with_result else None,
            extraction=ex if with_result else None,
            error=error,
            processing_time_ms=round(result.timings_ms["total"], 1) if with_result else None,
        )

    docs = [
        doc(1, "receipt_sample_1.jpg", ex.suggested_status()),
        doc(2, "receipt_sample_2.jpg", DocumentStatus.APPROVED),
        doc(
            3,
            "blurry_photo.jpg",
            DocumentStatus.FAILED,
            with_result=False,
            error="No text detected in image (mock).",
        ),
    ]
    return {d.id: d for d in docs}


def list_documents(status: DocumentStatus | None = None) -> list[DocumentSummary]:
    docs = sorted(_documents().values(), key=lambda d: d.created_at, reverse=True)
    return [
        DocumentSummary(**d.model_dump(include=set(DocumentSummary.model_fields)))
        for d in docs
        if status is None or d.status == status
    ]


def get_document(doc_id: str) -> DocumentDetail | None:
    return _documents().get(doc_id)
