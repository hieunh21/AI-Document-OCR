"""Shared data contracts for the AI core (OCR -> KIE -> validation).

Every component (OCR provider, KIE provider, backend, evaluation) speaks these
types, so a model can be swapped without touching the rest of the system.

Coordinate convention: every ``bbox`` is ``[x_min, y_min, x_max, y_max]`` in
pixels of the *original* uploaded image (after EXIF orientation is applied),
never of a resized/rotated working copy.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

FIELD_NAMES: tuple[str, ...] = ("seller", "seller_address", "timestamp", "total_cost")
FieldName = Literal["seller", "seller_address", "timestamp", "total_cost"]


def _check_bbox(b: list[float]) -> list[float]:
    if b[2] < b[0] or b[3] < b[1]:
        raise ValueError(f"bbox must be [x_min, y_min, x_max, y_max], got {b}")
    return b


BBox = Annotated[list[float], Field(min_length=4, max_length=4), AfterValidator(_check_bbox)]
"""[x_min, y_min, x_max, y_max] in original-image pixels."""

Point = Annotated[list[float], Field(min_length=2, max_length=2)]

_ISO_TS = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?$")


class DocumentStatus(StrEnum):
    """Lifecycle of an uploaded document."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    NEEDS_REVIEW = "needs_review"  # AI finished, but warnings / missing fields
    READY = "ready"  # AI finished without warnings; still not human-approved
    APPROVED = "approved"  # a human reviewed and approved the values
    FAILED = "failed"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def union_bbox(boxes: list[list[float]]) -> list[float] | None:
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


# --------------------------------------------------------------------------- OCR


class OCRSpan(_Model):
    """One detected text region (usually a line)."""

    id: str
    text: str
    bbox: BBox
    polygon: list[Point] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class OCRResult(_Model):
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    spans: list[OCRSpan] = Field(default_factory=list)
    model_version: str | None = None
    latency_ms: float | None = None


# --------------------------------------------------------------------------- KIE


class Evidence(_Model):
    """A region on the document that supports a field value.

    Must point at a real OCR span (``source_id``) or a real annotated region —
    never at coordinates invented by a language model.
    """

    page: int = Field(default=1, ge=1)
    bbox: BBox
    text: str | None = None
    source_id: str | None = None


class FieldEvidence(_Model):
    span_ids: list[str] = Field(default_factory=list)
    bbox: BBox | None = None  # union of all regions, for a single highlight
    regions: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_union(self) -> FieldEvidence:
        if self.bbox is None and self.regions:
            self.bbox = union_bbox([r.bbox for r in self.regions])
        if not self.span_ids:
            self.span_ids = [r.source_id for r in self.regions if r.source_id]
        return self


class ReceiptFields(_Model):
    """Normalized values. ``None`` = not present / not readable — never a guess."""

    seller: str | None = None
    seller_address: str | None = None
    timestamp: str | None = Field(
        default=None,
        description="ISO 8601 local time without timezone: YYYY-MM-DD or YYYY-MM-DDTHH:MM[:SS]",
        examples=["2020-08-11T08:06", "2020-08-14"],
    )
    total_cost: int | None = Field(default=None, ge=0, description="Amount in VND (integer dong)")

    @field_validator("seller", "seller_address")
    @classmethod
    def _blank_to_none(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = " ".join(v.split())
        return v or None

    @field_validator("timestamp")
    @classmethod
    def _iso(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not _ISO_TS.match(v):
            raise ValueError(f"timestamp must be ISO YYYY-MM-DD[THH:MM[:SS]], got {v!r}")
        (datetime if "T" in v else date).fromisoformat(v)  # raises on impossible dates
        return v


class RawFields(_Model):
    """Text as read from the document, before normalization."""

    seller: str | None = None
    seller_address: str | None = None
    timestamp: str | None = None
    total_cost: str | None = None


class ExtractionWarning(_Model):
    code: str  # machine readable, e.g. "missing_field", "low_confidence"
    field: FieldName | None = None
    message: str


class ReceiptExtraction(_Model):
    """Output of a KIE provider for one receipt image."""

    document_type: Literal["receipt"] = "receipt"
    model_version: str
    fields: ReceiptFields = Field(default_factory=ReceiptFields)
    raw_fields: RawFields = Field(default_factory=RawFields)
    evidence: dict[FieldName, FieldEvidence] = Field(default_factory=dict)
    warnings: list[ExtractionWarning] = Field(default_factory=list)

    def missing_fields(self) -> list[str]:
        return [f for f in FIELD_NAMES if getattr(self.fields, f) is None]

    def suggested_status(self) -> DocumentStatus:
        """READY only when every field is present and there is no warning."""
        if self.warnings or self.missing_fields():
            return DocumentStatus.NEEDS_REVIEW
        return DocumentStatus.READY
