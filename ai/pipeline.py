"""End-to-end inference: image -> OCR spans -> 4 fields + evidence.

The same ``ReceiptPipeline`` object is used by the online app (backend) and by
offline evaluation, so reported metrics always belong to the served model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ai.kie.base import KIEProvider, extract_from_ocr
from ai.ocr.base import OCRProvider
from ai.quality.assess import QualityAssessor, QualityReport
from ai.schemas import ExtractionWarning, OCRResult, ReceiptExtraction


@dataclass
class PipelineResult:
    ocr: OCRResult
    extraction: ReceiptExtraction
    quality: QualityReport | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    llm_usage: Any = None  # ai.kie.llm.LLMUsage when the KIE provider is an LLM


class ReceiptPipeline:
    def __init__(
        self, ocr: OCRProvider, kie: KIEProvider, quality: QualityAssessor | None = None
    ) -> None:
        self.ocr = ocr
        self.kie = kie
        self.quality = quality

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
        extraction, usage = extract_from_ocr(self.kie, ocr, image)
        t3 = time.perf_counter()
        quality = self.quality.assess(image, ocr) if self.quality else None
        t4 = time.perf_counter()
        if quality:  # advisory only: surfaces as warnings, never blocks
            known = {w.code for w in extraction.warnings}
            messages = {
                "low_image_quality": f"Image quality looks poor (score {quality.score}); "
                "consider retaking the photo.",
                "no_text_detected": "No text was detected in the image.",
            }
            for code in quality.warnings:
                if code not in known:
                    extraction.warnings.append(
                        ExtractionWarning(code=code, message=messages.get(code, code))
                    )
        return PipelineResult(
            ocr=ocr,
            extraction=extraction,
            quality=quality,
            llm_usage=usage,
            timings_ms={
                "preprocess": (t1 - t0) * 1000,
                "ocr": (t2 - t1) * 1000,
                "kie": (t3 - t2) * 1000,
                "quality": (t4 - t3) * 1000,
                "total": (t4 - t0) * 1000,
            },
        )


PIPELINE_CONFIG = Path("configs/pipeline.json")


def pipeline_config(name: str = "default", path: Path = PIPELINE_CONFIG) -> dict[str, str]:
    """OCR preset and KIE name of a frozen pipeline (chosen in Phase 4)."""
    config = json.loads(path.read_text(encoding="utf-8"))
    return config["pipelines"][name] | {"version": config["version"]}


def build_pipeline(name: str = "default", device: str = "gpu:0") -> ReceiptPipeline:
    from ai.kie.factory import build_kie
    from ai.ocr.factory import build_ocr

    cfg = pipeline_config(name)
    return ReceiptPipeline(build_ocr(cfg["ocr"], device), build_kie(cfg["kie"]), QualityAssessor())
