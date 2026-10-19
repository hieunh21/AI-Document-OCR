from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai.schemas import DocumentStatus, FieldName, OCRResult, ReceiptExtraction, ReceiptFields

FieldSource = Literal["ai", "human"]


def _aware(v: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; every stored time is UTC."""
    return v.replace(tzinfo=UTC) if v is not None and v.tzinfo is None else v


class DocumentSummary(BaseModel):
    id: str
    original_name: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime | None = None
    seller: str | None = None
    seller_address: str | None = None
    timestamp: str | None = None
    total_cost: int | None = None

    aware_times = field_validator("created_at", "updated_at")(_aware)


class FieldCorrectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    field_name: FieldName
    old_value: Any = None
    new_value: Any = None
    reviewer: str | None = None
    extraction_id: int | None = None
    created_at: datetime

    aware_times = field_validator("created_at")(_aware)


class DocumentEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    action: str
    actor: str | None = None
    detail: dict[str, Any] | None = None
    created_at: datetime

    aware_times = field_validator("created_at")(_aware)


class DocumentDetail(DocumentSummary):
    image_url: str
    content_type: str
    size_bytes: int
    sha256: str
    image_width: int | None = None
    image_height: int | None = None
    error: str | None = None
    approved_at: datetime | None = None
    approved_by: str | None = None

    # Current values = latest AI extraction with later human corrections applied.
    fields: ReceiptFields = Field(default_factory=ReceiptFields)
    field_sources: dict[FieldName, FieldSource] = Field(default_factory=dict)
    # Warnings re-computed on the current values (missing / implausible fields).
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    # Latest AI run, unchanged by corrections.
    extraction_id: int | None = None
    extraction: ReceiptExtraction | None = None
    ocr: OCRResult | None = None
    quality: dict[str, Any] | None = None
    pipeline_version: str | None = None
    model_version: str | None = None
    processing_time_ms: float | None = None
    timings_ms: dict[str, float] | None = None

    corrections: list[FieldCorrectionOut] = Field(default_factory=list)
    events: list[DocumentEventOut] = Field(default_factory=list)

    aware_approved = field_validator("approved_at")(_aware)


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    total: int
    page: int = 1
    page_size: int = 20


class FieldsUpdate(BaseModel):
    """Corrections from a reviewer. Only the fields present in the body are changed;
    ``null`` means the value is not on the receipt."""

    model_config = ConfigDict(extra="forbid")

    seller: str | None = None
    seller_address: str | None = None
    timestamp: str | None = Field(
        default=None, description="ISO YYYY-MM-DD[THH:MM[:SS]], or a date as printed"
    )
    total_cost: int | str | None = Field(default=None, description="VND, e.g. 510760 or '510.760'")
    reviewer: str | None = Field(default=None, max_length=128)


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str | None = Field(default=None, max_length=128)


class DocumentStats(BaseModel):
    total: int
    by_status: dict[DocumentStatus, int]
    approved_total_vnd: int
