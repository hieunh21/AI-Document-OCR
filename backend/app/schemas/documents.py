from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from ai.schemas import DocumentStatus, OCRResult, ReceiptExtraction


class DocumentSummary(BaseModel):
    id: str
    original_name: str
    status: DocumentStatus
    created_at: datetime
    seller: str | None = None
    timestamp: str | None = None
    total_cost: int | None = None


class DocumentDetail(DocumentSummary):
    image_width: int | None = None
    image_height: int | None = None
    ocr: OCRResult | None = None
    extraction: ReceiptExtraction | None = None
    error: str | None = None
    processing_time_ms: float | None = None


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    total: int
