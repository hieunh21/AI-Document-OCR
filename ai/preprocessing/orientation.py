"""Whole-page orientation (0/90/180/270) with PaddleX ``PP-LCNet_x1_0_doc_ori``.

The predicted label is the counter-clockwise rotation that makes the page
upright — the same convention as ``prepare_for_ocr(rotation=...)``.

``min_score=0.8``: on the dev split, every truly rotated receipt scored ~0.92,
while the one false rotation (an upright receipt on a busy patterned table
cloth, predicted 180°) scored 0.64.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import ai.ocr._runtime  # noqa: F401  (load torch before paddle)

MODEL_NAME = "PP-LCNet_x1_0_doc_ori"


class OrientationClassifier:
    def __init__(
        self, device: str = "cpu", min_score: float = 0.8, enable_mkldnn: bool = False
    ) -> None:
        from paddleocr import DocImgOrientationClassification

        self.min_score = min_score
        self._m = DocImgOrientationClassification(
            model_name=MODEL_NAME, device=device, enable_mkldnn=enable_mkldnn
        )

    def predict(self, image: Image.Image) -> tuple[int, float]:
        """Return ``(ccw_rotation, score)``; 0 when the model is not confident."""
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        res = self._m.predict(bgr)[0]
        angle, score = int(res["label_names"][0]), float(res["scores"][0])
        return (angle if score >= self.min_score else 0), score
