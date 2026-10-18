"""Receipt image quality: predicts the official MC-OCR IQA score (0–1).

The MC-OCR score is the share of text lines annotators judged *clear*, so the
predictor combines cheap image statistics (sharpness, exposure) with signals
from the OCR pass that runs anyway (recognition confidence, text size).

A linear model (ridge regression on standardized features) is fitted on the
internal *train* split by ``python -m evaluation.iqa_eval fit`` and stored in
``configs/quality_model.json``. The result is advisory: it raises a warning
and never blocks processing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel

from ai.schemas import OCRResult

MODEL_PATH = Path("configs/quality_model.json")
WORK_SIDE = 1024  # image statistics are computed at a fixed scale

IMAGE_FEATURES = [
    "log_lap_var",
    "log_tenengrad",
    "brightness",
    "contrast",
    "overexposed",
    "underexposed",
    "log_megapixels",
]
OCR_FEATURES = [
    "ocr_mean_conf",
    "ocr_p10_conf",
    "ocr_low_conf_frac",
    "ocr_log_spans",
    "ocr_text_area_frac",
    "ocr_median_height_frac",
    "ocr_conf_lt95_frac",
    "text_sharpness_med",
    "text_sharpness_p10",
    "text_contrast_med",
]


def image_features(image: Image.Image) -> dict[str, float]:
    w, h = image.size
    gray = np.asarray(image.convert("L"))
    s = WORK_SIDE / max(w, h)
    if s < 1:
        gray = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g = gray.astype(np.float64)
    return {
        "log_lap_var": math.log1p(float(lap.var())),
        "log_tenengrad": math.log1p(float((gx**2 + gy**2).mean())),
        "brightness": float(g.mean() / 255),
        "contrast": float(g.std() / 255),
        "overexposed": float((gray >= 245).mean()),
        "underexposed": float((gray <= 15).mean()),
        "log_megapixels": math.log(w * h / 1e6),
    }


def text_region_features(image: Image.Image, ocr: OCRResult) -> dict[str, float]:
    """Sharpness/contrast measured inside detected text boxes only.

    The official score is about whether *text lines* are readable; global
    statistics are dominated by the table/background around the receipt.
    Each box is resized to a fixed height so values are comparable.
    """
    gray = np.asarray(image.convert("L"))
    sharp, contrast = [], []
    for s in ocr.spans:
        x0, y0, x1, y1 = (int(round(v)) for v in s.bbox)
        crop = gray[max(y0, 0) : y1, max(x0, 0) : x1]
        if crop.shape[0] < 6 or crop.shape[1] < 6:
            continue
        scale = 32 / crop.shape[0]
        crop = cv2.resize(
            crop, (max(8, round(crop.shape[1] * scale)), 32), interpolation=cv2.INTER_AREA
        )
        sharp.append(math.log1p(float(cv2.Laplacian(crop, cv2.CV_64F).var())))
        contrast.append(float(crop.std() / 255))
    if not sharp:
        return {"text_sharpness_med": 0.0, "text_sharpness_p10": 0.0, "text_contrast_med": 0.0}
    return {
        "text_sharpness_med": float(np.median(sharp)),
        "text_sharpness_p10": float(np.percentile(sharp, 10)),
        "text_contrast_med": float(np.median(contrast)),
    }


def ocr_features(ocr: OCRResult, image: Image.Image | None = None) -> dict[str, float]:
    conf = np.array([s.confidence for s in ocr.spans if s.confidence is not None], dtype=float)
    heights = np.array([s.bbox[3] - s.bbox[1] for s in ocr.spans], dtype=float)
    area = sum((s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1]) for s in ocr.spans)
    img_area = ocr.image_width * ocr.image_height
    return {
        "ocr_mean_conf": float(conf.mean()) if conf.size else 0.0,
        "ocr_p10_conf": float(np.percentile(conf, 10)) if conf.size else 0.0,
        "ocr_low_conf_frac": float((conf < 0.8).mean()) if conf.size else 1.0,
        "ocr_log_spans": math.log1p(len(ocr.spans)),
        "ocr_text_area_frac": float(area / img_area) if img_area else 0.0,
        "ocr_median_height_frac": float(np.median(heights) / ocr.image_height)
        if heights.size
        else 0.0,
        "ocr_conf_lt95_frac": float((conf < 0.95).mean()) if conf.size else 1.0,
    } | (text_region_features(image, ocr) if image is not None else {})


class LinearQualityModel(BaseModel):
    features: list[str]
    mean: list[float]
    std: list[float]
    coef: list[float]
    intercept: float
    poor_threshold: float  # predicted score below this -> "poor"
    trained_on: str = ""
    version: str = "iqa-ridge-v1"

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        features: list[str],
        alpha: float = 1.0,
        poor_threshold: float = 0.5,
        trained_on: str = "",
    ) -> LinearQualityModel:
        mean, std = X.mean(0), X.std(0)
        std[std == 0] = 1.0
        Z = (X - mean) / std
        A = Z.T @ Z + alpha * np.eye(Z.shape[1])
        coef = np.linalg.solve(A, Z.T @ (y - y.mean()))
        return cls(
            features=features,
            mean=mean.tolist(),
            std=std.tolist(),
            coef=coef.tolist(),
            intercept=float(y.mean()),
            poor_threshold=poor_threshold,
            trained_on=trained_on,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(X) - np.array(self.mean)) / np.array(self.std)
        return np.clip(Z @ np.array(self.coef) + self.intercept, 0.0, 1.0)


class QualityReport(BaseModel):
    score: float | None  # predicted MC-OCR IQA score, None if no model is available
    level: str  # "good" | "poor" | "unknown"
    model_version: str | None = None
    features: dict[str, float]
    warnings: list[str] = []


class QualityAssessor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model = (
            LinearQualityModel.model_validate(json.loads(model_path.read_text(encoding="utf-8")))
            if model_path.exists()
            else None
        )

    def assess(self, image: Image.Image, ocr: OCRResult | None = None) -> QualityReport:
        feats = image_features(image)
        if ocr is not None:
            feats |= ocr_features(ocr, image)
        warnings = []
        if ocr is not None and not ocr.spans:
            warnings.append("no_text_detected")
        if self.model is None or any(f not in feats for f in self.model.features):
            return QualityReport(score=None, level="unknown", features=feats, warnings=warnings)
        x = np.array([[feats[f] for f in self.model.features]])
        score = float(self.model.predict(x)[0])
        level = "poor" if score < self.model.poor_threshold else "good"
        if level == "poor":
            warnings.append("low_image_quality")
        return QualityReport(
            score=round(score, 3),
            level=level,
            model_version=self.model.version,
            features=feats,
            warnings=warnings,
        )
