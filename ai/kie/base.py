from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from ai.schemas import Evidence, FieldEvidence, OCRResult, OCRSpan, ReceiptExtraction


class KIEProvider(ABC):
    """Key information extraction for the four MC-OCR fields.

    Evidence returned in the extraction must reference ids of ``ocr_spans``;
    a provider must not invent coordinates.

    Providers reason about layout (lines, left/right, above/below), so they must
    receive spans of an *upright* page. Call ``extract_from_ocr`` rather than
    ``extract`` when the OCR result may come from a rotated photo.
    """

    name: str = "kie"
    version: str = "0"

    @abstractmethod
    def extract(self, image: Image.Image | None, ocr_spans: list[OCRSpan]) -> ReceiptExtraction:
        raise NotImplementedError


def _rotate_box(b: list[float], w: float, h: float, rotation: int) -> list[float]:
    """Original-image box -> box in the page rotated CCW by ``rotation`` degrees."""
    x0, y0, x1, y1 = b
    if rotation == 90:  # (x, y) -> (y, W - x)
        return [y0, w - x1, y1, w - x0]
    if rotation == 180:  # (x, y) -> (W - x, H - y)
        return [w - x1, h - y1, w - x0, h - y0]
    if rotation == 270:  # (x, y) -> (H - y, x)
        return [h - y1, x0, h - y0, x1]
    return list(b)


def upright_spans(ocr: OCRResult) -> list[OCRSpan]:
    """Copies of the spans with boxes in the upright page frame (ids unchanged)."""
    if not ocr.rotation:
        return list(ocr.spans)
    return [
        s.model_copy(
            update={
                "bbox": _rotate_box(s.bbox, ocr.image_width, ocr.image_height, ocr.rotation),
                "polygon": None,
            }
        )
        for s in ocr.spans
    ]


def restore_evidence(ex: ReceiptExtraction, ocr: OCRResult) -> ReceiptExtraction:
    """Point evidence back at the original-image boxes of the cited spans."""
    if not ocr.rotation:
        return ex
    by_id = {s.id: s for s in ocr.spans}
    for field_name, ev in list(ex.evidence.items()):
        regions = [
            Evidence(page=r.page, bbox=by_id[r.source_id].bbox, text=r.text, source_id=r.source_id)
            for r in ev.regions
            if r.source_id in by_id
        ]
        ex.evidence[field_name] = FieldEvidence(regions=regions)
    return ex


def extract_from_ocr(kie: KIEProvider, ocr: OCRResult, image: Image.Image | None = None):
    """Run ``kie`` on an OCR result of any orientation.

    Returns ``(extraction, usage)``; ``usage`` is the LLM token usage when the
    provider reports it, else ``None``.
    """
    spans = upright_spans(ocr)
    if hasattr(kie, "extract_with_usage"):
        ex, usage = kie.extract_with_usage(spans)
    else:
        ex, usage = kie.extract(image, spans), None
    return restore_evidence(ex, ocr), usage
