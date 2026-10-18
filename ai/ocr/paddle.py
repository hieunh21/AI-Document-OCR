"""PaddleOCR (PP-OCR) implementation of :class:`ai.ocr.base.OCRProvider`.

Presets name the det/rec model pairs we benchmark (see ``PRESETS``). The model
is loaded lazily on first ``predict`` so importing this module is cheap.

Windows note: Paddle 3.3 CPU crashes with oneDNN enabled
("ConvertPirAttribute2RuntimeAttribute not support"), so ``enable_mkldnn``
defaults to False.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

import ai.ocr._runtime  # noqa: F401  (load torch before paddle)
from ai.layout import reading_order
from ai.ocr.base import OCRProvider
from ai.preprocessing.image import prepare_for_ocr
from ai.schemas import OCRResult, OCRSpan


@dataclass(frozen=True)
class PaddlePreset:
    det: str
    rec: str
    max_side: int | None = None  # downscale before OCR (None = keep size)
    det_limit_side_len: int | None = None
    det_limit_type: str | None = None  # "min" | "max"
    extra: dict = field(default_factory=dict)


PRESETS: dict[str, PaddlePreset] = {
    "v6_medium": PaddlePreset("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
    "v6_small": PaddlePreset("PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
    "v6_tiny": PaddlePreset("PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec"),
    "v5_mobile_latin": PaddlePreset("PP-OCRv5_mobile_det", "latin_PP-OCRv5_mobile_rec"),
    "v5_server_latin": PaddlePreset("PP-OCRv5_server_det", "latin_PP-OCRv5_mobile_rec"),
}


class PaddleOCRProvider(OCRProvider):
    name = "paddleocr"

    def __init__(
        self,
        preset: str | PaddlePreset = "v6_medium",
        device: str = "cpu",
        enable_mkldnn: bool = False,
        rec_score_thresh: float = 0.0,
        auto_orient: bool = False,
    ) -> None:
        self.preset_name = preset if isinstance(preset, str) else "custom"
        self.preset = PRESETS[preset] if isinstance(preset, str) else preset
        self.device = device
        self.enable_mkldnn = enable_mkldnn
        self.rec_score_thresh = rec_score_thresh
        self.auto_orient = auto_orient
        self._ocr = None
        self._orient = None

    @property
    def version(self) -> str:  # type: ignore[override]
        import paddleocr

        orient = "+orient" if self.auto_orient else ""
        return f"{paddleocr.__version__}:{self.preset.det}+{self.preset.rec}{orient}"

    def _engine(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            kwargs = dict(
                text_detection_model_name=self.preset.det,
                text_recognition_model_name=self.preset.rec,
                device=self.device,
                enable_mkldnn=self.enable_mkldnn,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                text_rec_score_thresh=self.rec_score_thresh,
            )
            if self.preset.det_limit_side_len:
                kwargs["text_det_limit_side_len"] = self.preset.det_limit_side_len
            if self.preset.det_limit_type:
                kwargs["text_det_limit_type"] = self.preset.det_limit_type
            kwargs.update(self.preset.extra)
            self._ocr = PaddleOCR(**kwargs)
        return self._ocr

    def warmup(self) -> None:
        self.predict(Image.new("RGB", (480, 640), "white"))

    def _rotation(self, image: Image.Image, rotation: int | None) -> int:
        if rotation is not None or not self.auto_orient:
            return rotation or 0
        if self._orient is None:
            from ai.preprocessing.orientation import OrientationClassifier

            self._orient = OrientationClassifier(
                device=self.device, enable_mkldnn=self.enable_mkldnn
            )
        return self._orient.predict(image)[0]

    def predict(self, image: Image.Image, rotation: int | None = None) -> OCRResult:
        t0 = time.perf_counter()
        rotation = self._rotation(image, rotation)
        work, tf = prepare_for_ocr(image, self.preset.max_side, rotation)
        bgr = np.ascontiguousarray(np.asarray(work.convert("RGB"))[:, :, ::-1])
        res = self._engine().predict(bgr)[0]

        spans = []
        for text, score, poly in zip(
            res["rec_texts"], res["rec_scores"], res["rec_polys"], strict=True
        ):
            text = (text or "").strip()
            if not text:
                continue
            pts = tf.points_to_original(np.asarray(poly))
            spans.append(
                OCRSpan(
                    id="",
                    text=text,
                    bbox=tf.bbox_to_original(np.asarray(poly)),
                    polygon=[[float(x), float(y)] for x, y in pts],
                    confidence=float(min(max(score, 0.0), 1.0)),
                )
            )
        spans = reading_order(spans, lambda s: s.bbox)
        for i, s in enumerate(spans):
            s.id = f"span_{i:03d}"
        return OCRResult(
            image_width=tf.orig_w,
            image_height=tf.orig_h,
            spans=spans,
            model_version=self.version,
            latency_ms=(time.perf_counter() - t0) * 1000,
            rotation=rotation,
        )
