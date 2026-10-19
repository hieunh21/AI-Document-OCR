"""The frozen receipt pipeline (``configs/pipeline.json``), loaded once per process.

Models load lazily on the first request (several seconds) unless
``VIETDOCAI_WARMUP=true``. A lock serialises inference: one GPU, one image at a time.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

from PIL import Image

from ai.pipeline import PipelineResult, ReceiptPipeline, build_pipeline, pipeline_config
from app.core.config import get_settings

logger = logging.getLogger("vietdocai.pipeline")


@dataclass(frozen=True)
class PipelineInfo:
    name: str
    version: str  # configs/pipeline.json "version", e.g. "pipeline-v3"


class PipelineRunner:
    def __init__(self, info: PipelineInfo, factory) -> None:
        self.info = info
        self._factory = factory
        self._pipeline: ReceiptPipeline | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def _get(self) -> ReceiptPipeline:
        if self._pipeline is None:
            t0 = time.perf_counter()
            self._pipeline = self._factory()
            logger.info(
                "pipeline %s (%s) loaded in %.1f s",
                self.info.name,
                self.info.version,
                time.perf_counter() - t0,
            )
        return self._pipeline

    def warmup(self) -> None:
        with self._lock:
            pipeline = self._get()
            pipeline.run(Image.new("RGB", (480, 640), "white"))

    def run(self, image: Image.Image) -> tuple[PipelineResult, str]:
        """Pipeline result and the model version string (OCR + KIE)."""
        with self._lock:
            pipeline = self._get()
            return pipeline.run(image), pipeline.version


@lru_cache
def get_runner() -> PipelineRunner:
    settings = get_settings()
    cfg = pipeline_config(settings.pipeline)
    info = PipelineInfo(name=settings.pipeline, version=cfg["version"])
    return PipelineRunner(info, lambda: build_pipeline(settings.pipeline, settings.device))
