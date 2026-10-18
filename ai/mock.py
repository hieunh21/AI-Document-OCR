"""Deterministic fake providers for UI/API development and tests.

The receipt content below is invented; it is not taken from MC-OCR.
"""

from __future__ import annotations

from PIL import Image

from ai.kie.base import KIEProvider
from ai.ocr.base import OCRProvider
from ai.schemas import (
    Evidence,
    ExtractionWarning,
    FieldEvidence,
    OCRResult,
    OCRSpan,
    RawFields,
    ReceiptExtraction,
    ReceiptFields,
)

MOCK_SPANS: list[OCRSpan] = [
    OCRSpan(id="span_00", text="CỬA HÀNG TIỆN LỢI MẪU", bbox=[180, 60, 590, 100], confidence=0.97),
    OCRSpan(
        id="span_01",
        text="12 Nguyễn Trãi, Thanh Xuân, Hà Nội",
        bbox=[150, 105, 620, 135],
        confidence=0.93,
    ),
    OCRSpan(id="span_02", text="HÓA ĐƠN BÁN HÀNG", bbox=[230, 170, 540, 210], confidence=0.98),
    OCRSpan(id="span_03", text="Ngày: 08/10/2026 09:26", bbox=[60, 240, 420, 270], confidence=0.95),
    OCRSpan(id="span_04", text="TỔNG THANH TOÁN:", bbox=[60, 800, 330, 840], confidence=0.96),
    OCRSpan(id="span_05", text="150.000", bbox=[520, 800, 690, 840], confidence=0.91),
]


class MockOCR(OCRProvider):
    name = "mock-ocr"
    version = "1"

    def predict(self, image: Image.Image) -> OCRResult:
        w, h = image.size if image is not None else (768, 1024)
        return OCRResult(
            image_width=w, image_height=h, spans=MOCK_SPANS, model_version=self.version
        )


def _ev(*spans: OCRSpan) -> FieldEvidence:
    return FieldEvidence(
        regions=[Evidence(bbox=s.bbox, text=s.text, source_id=s.id) for s in spans]
    )


class MockKIE(KIEProvider):
    name = "mock-kie"
    version = "1"

    def extract(self, image: Image.Image | None, ocr_spans: list[OCRSpan]) -> ReceiptExtraction:
        s = {sp.id: sp for sp in MOCK_SPANS}
        return ReceiptExtraction(
            model_version=f"{self.name}-{self.version}",
            fields=ReceiptFields(
                seller="CỬA HÀNG TIỆN LỢI MẪU",
                seller_address="12 Nguyễn Trãi, Thanh Xuân, Hà Nội",
                timestamp="2026-10-08T09:26",
                total_cost=150000,
            ),
            raw_fields=RawFields(
                seller=s["span_00"].text,
                seller_address=s["span_01"].text,
                timestamp=s["span_03"].text,
                total_cost=s["span_05"].text,
            ),
            evidence={
                "seller": _ev(s["span_00"]),
                "seller_address": _ev(s["span_01"]),
                "timestamp": _ev(s["span_03"]),
                "total_cost": _ev(s["span_04"], s["span_05"]),
            },
            warnings=[
                ExtractionWarning(
                    code="low_confidence",
                    field="total_cost",
                    message="Mock warning: OCR confidence of the amount is 0.91.",
                )
            ],
        )
