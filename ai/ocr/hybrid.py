"""Two-stage OCR: PaddleOCR text *detection* + a pluggable line *recognizer*.

PP-OCR's multilingual recognizers emit non-Vietnamese diacritics ("Tőng",
"TÖNG", "Tāy"). A recognizer trained on Vietnamese (VietOCR) can be plugged in
behind the same detector, so the comparison isolates recognition quality.

Detected quadrilaterals are rectified (perspective warp of the minimum-area
rectangle, as PP-OCR does) and padded vertically before recognition; boxes
returned are in original-image coordinates.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image

import ai.ocr._runtime  # noqa: F401  (load torch before paddle)
from ai.layout import reading_order
from ai.ocr.base import OCRProvider
from ai.preprocessing.image import prepare_for_ocr
from ai.schemas import OCRResult, OCRSpan


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Minimum-area rectangle of ``pts`` as TL, TR, BR, BL."""
    box = cv2.boxPoints(cv2.minAreaRect(np.asarray(pts, dtype=np.float32)))
    box = box[np.argsort(box[:, 0])]
    left, right = box[:2], box[2:]
    tl, bl = left[np.argsort(left[:, 1])]
    tr, br = right[np.argsort(right[:, 1])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def crop_line(rgb: np.ndarray, pts: np.ndarray, pad_ratio: float = 0.15) -> np.ndarray:
    """Rectified crop of a text line, padded by ``pad_ratio`` of its height."""
    q = order_quad(pts)
    w = int(round(max(np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[3] - q[2]))))
    h = int(round(max(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))))
    w, h = max(w, 1), max(h, 1)
    pad = int(round(h * pad_ratio))
    dst = np.float32([[pad, pad], [w + pad, pad], [w + pad, h + pad], [pad, h + pad]])
    M = cv2.getPerspectiveTransform(q, dst)
    crop = cv2.warpPerspective(
        rgb, M, (w + 2 * pad, h + 2 * pad), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )
    if crop.shape[0] / crop.shape[1] >= 1.5:  # vertical text line
        crop = np.rot90(crop)
    return crop


class LineRecognizer(ABC):
    name: str
    version: str

    @abstractmethod
    def recognize(self, crops: list[np.ndarray]) -> list[tuple[str, float | None]]:
        """RGB crops -> (text, confidence)."""


class PaddleLineRecognizer(LineRecognizer):
    def __init__(
        self,
        model_name: str = "PP-OCRv6_medium_rec",
        device: str = "cpu",
        enable_mkldnn: bool = False,
    ) -> None:
        from paddleocr import TextRecognition

        self.name, self.version = "paddle", model_name
        self._m = TextRecognition(model_name=model_name, device=device, enable_mkldnn=enable_mkldnn)

    def recognize(self, crops):
        out = []
        for res in self._m.predict([np.ascontiguousarray(c[:, :, ::-1]) for c in crops]):
            out.append((res["rec_text"], float(res["rec_score"])))
        return out


class VietOCRRecognizer(LineRecognizer):
    """VietOCR (pbcquoc/vietocr) pretrained on Vietnamese text lines."""

    def __init__(
        self,
        arch: str = "vgg_transformer",
        device: str = "cpu",
        batch_size: int = 32,
        beamsearch: bool = False,
    ) -> None:
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        cfg = Cfg.load_config_from_name(arch)
        cfg["device"] = "cuda:0" if device.startswith("gpu") else device
        cfg["cnn"]["pretrained"] = False
        cfg["predictor"]["beamsearch"] = beamsearch
        self.name, self.version = "vietocr", arch
        self.batch_size = batch_size
        self._p = Predictor(cfg)

    def recognize(self, crops):
        """Width-sorted batches padded to a common width.

        VietOCR's own ``predict_batch`` buckets by *exact* width, so a receipt with
        40 lines runs ~30 separate greedy decodes (≈10 s on an RTX 3050). Lines of
        similar width (within ``MAX_PAD_RATIO``) are padded on the right by edge
        replication and decoded together instead.
        """
        import torch
        import torch.nn.functional as F
        from vietocr.tool.translate import process_input, translate

        ds = self._p.config["dataset"]
        tensors = [
            process_input(
                Image.fromarray(np.ascontiguousarray(c)),
                ds["image_height"],
                ds["image_min_width"],
                ds["image_max_width"],
            )
            for c in crops
        ]
        order = sorted(range(len(tensors)), key=lambda i: tensors[i].shape[-1])
        out: list[tuple[str, float | None]] = [("", None)] * len(tensors)
        start = 0
        while start < len(order):
            group = [order[start]]
            min_w = tensors[order[start]].shape[-1]
            while (
                start + len(group) < len(order)
                and len(group) < self.batch_size
                and tensors[order[start + len(group)]].shape[-1] <= min_w * self.MAX_PAD_RATIO
            ):
                group.append(order[start + len(group)])
            max_w = max(tensors[i].shape[-1] for i in group)
            batch = torch.cat(
                [
                    F.pad(tensors[i], (0, max_w - tensors[i].shape[-1], 0, 0), mode="replicate")
                    for i in group
                ]
            ).to(self._p.device)
            s, prob = translate(batch, self._p.model)
            texts = self._p.vocab.batch_decode(s.tolist())
            for i, t, p in zip(group, texts, prob.tolist(), strict=True):
                out[i] = (t, float(p))
            start += len(group)
        return out

    MAX_PAD_RATIO = 1.5


class HybridOCRProvider(OCRProvider):
    name = "hybrid"

    def __init__(
        self,
        det_model: str,
        recognizer: LineRecognizer,
        device: str = "cpu",
        max_side: int | None = None,
        enable_mkldnn: bool = False,
        auto_orient: bool = False,
    ) -> None:
        from paddleocr import TextDetection

        self.det_model = det_model
        self.recognizer = recognizer
        self.max_side = max_side
        self.orient = None
        if auto_orient:
            from ai.preprocessing.orientation import OrientationClassifier

            self.orient = OrientationClassifier(device=device, enable_mkldnn=enable_mkldnn)
        self._det = TextDetection(model_name=det_model, device=device, enable_mkldnn=enable_mkldnn)

    @property
    def version(self) -> str:  # type: ignore[override]
        orient = "+orient" if self.orient else ""
        return f"{self.det_model}+{self.recognizer.name}:{self.recognizer.version}{orient}"

    def warmup(self) -> None:
        self.predict(Image.new("RGB", (480, 640), "white"))

    def predict(self, image: Image.Image, rotation: int | None = None) -> OCRResult:
        t0 = time.perf_counter()
        if rotation is None:
            rotation = self.orient.predict(image)[0] if self.orient else 0
        work, tf = prepare_for_ocr(image, self.max_side, rotation)
        rgb = np.asarray(work.convert("RGB"))
        det = self._det.predict(np.ascontiguousarray(rgb[:, :, ::-1]))[0]
        polys = [np.asarray(p, dtype=np.float32) for p in det["dt_polys"]]
        crops = [crop_line(rgb, p) for p in polys]
        recs = self.recognizer.recognize(crops) if crops else []
        spans = []
        for poly, (text, conf) in zip(polys, recs, strict=True):
            text = (text or "").strip()
            if not text:
                continue
            pts = tf.points_to_original(poly)
            spans.append(
                OCRSpan(
                    id="",
                    text=text,
                    bbox=tf.bbox_to_original(poly),
                    polygon=[[float(x), float(y)] for x, y in pts],
                    confidence=None if conf is None else float(min(max(conf, 0.0), 1.0)),
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
            rotation=rotation,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
