"""End-to-end inference: image -> OCR spans -> 4 fields + evidence.

The same ``ReceiptPipeline`` object is used by the online app (backend) and by
offline evaluation, so reported metrics always belong to the served model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from PIL import Image, ImageOps

from ai.kie.base import KIEProvider
from ai.ocr.base import OCRProvider
from ai.schemas import OCRResult, ReceiptExtraction


@dataclass
class PipelineResult:
    ocr: OCRResult
    extraction: ReceiptExtraction
    timings_ms: dict[str, float] = field(default_factory=dict)


class ReceiptPipeline:
    def __init__(self, ocr: OCRProvider, kie: KIEProvider) -> None:
        self.ocr = ocr
        self.kie = kie

    @property
    def version(self) -> str:
        return f"{self.ocr.name}-{self.ocr.version}+{self.kie.name}-{self.kie.version}"

    def run(self, image: Image.Image) -> PipelineResult:
        t0 = time.perf_counter()
        # Apply EXIF orientation so "original image" coordinates match what a browser shows.
        image = ImageOps.exif_transpose(image).convert("RGB")
        t1 = time.perf_counter()
        ocr = self.ocr.predict(image)
        t2 = time.perf_counter()
        extraction = self.kie.extract(image, ocr.spans)
        t3 = time.perf_counter()
        return PipelineResult(
            ocr=ocr,
            extraction=extraction,
            timings_ms={
                "preprocess": (t1 - t0) * 1000,
                "ocr": (t2 - t1) * 1000,
                "kie": (t3 - t2) * 1000,
                "total": (t3 - t0) * 1000,
            },
        )
