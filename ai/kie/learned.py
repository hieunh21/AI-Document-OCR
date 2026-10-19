"""KIE with a learned span classifier (LightGBM) instead of hand-written line rules.

Every OCR span gets a probability for seller / address / timestamp / total / other
(``ai.kie.line_features``); fields are then assembled from the spans:

* **seller**, **address**: spans above the class threshold, in reading order, joined;
  then the same normalisation as the rules (gazetteer spelling, official place names);
* **timestamp**: the spans above the threshold are parsed together (date and time are
  often separate boxes); if that fails, the most likely span alone;
* **total_cost**: the amount of the most likely span that holds one; a keyword-only
  span ("Tổng cộng:") takes the amount printed next to it.

A field whose best probability is below ``confident`` gets a ``low_confidence`` warning,
so the document goes to review. The model and its thresholds come from
``python -m evaluation.train_line_classifier`` (``configs/kie_line_model.*``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ai.kie.base import KIEProvider
from ai.kie.gazetteer import Gazetteer
from ai.kie.line_features import CLASSES, FEATURE_NAMES, order_spans, span_features
from ai.kie.rule_based import RuleBasedKIE, _evidence
from ai.kie.validate import validate_extraction
from ai.layout import reading_order
from ai.normalization.address import AdminUnits
from ai.normalization.datetime_vi import parse_timestamp
from ai.normalization.text import normalize_text
from ai.schemas import (
    FIELD_NAMES,
    ExtractionWarning,
    OCRSpan,
    RawFields,
    ReceiptExtraction,
    ReceiptFields,
)

MODEL_PATH = Path("configs/kie_line_model.txt")
META_PATH = Path("configs/kie_line_model.json")


@dataclass(frozen=True)
class Thresholds:
    seller: float = 0.5
    seller_address: float = 0.5
    timestamp: float = 0.5
    total_cost: float = 0.5
    confident: float = 0.8  # below: low_confidence warning

    def of(self, field: str) -> float:
        return getattr(self, field)


def _time_nearby(spans: list[OCRSpan], p_time: np.ndarray, used: list[OCRSpan]) -> int | None:
    """Index of a time-only span close to the date spans (the most likely one)."""
    taken = {s.id for s in used}
    y0 = min(s.bbox[1] for s in used)
    y1 = max(s.bbox[3] for s in used)
    h = max(s.bbox[3] - s.bbox[1] for s in used)
    best = None
    for j, s in enumerate(spans):
        if s.id in taken or s.bbox[1] > y1 + 2.5 * h or s.bbox[3] < y0 - 2.5 * h:
            continue
        t = parse_timestamp(s.text)
        if t and t.time and t.date is None and (best is None or p_time[j] > p_time[best]):
            best = j
    return best


def assemble(
    spans: list[OCRSpan],
    proba: np.ndarray,
    th: Thresholds,
    gazetteer: Gazetteer | None,
    admin: AdminUnits | None,
    model_version: str,
) -> ReceiptExtraction:
    """Fields from span probabilities (rows of ``proba`` follow ``spans``)."""
    fields: dict = {}
    raw: dict = {}
    evidence: dict = {}
    conf: dict[str, float] = {}

    def picked(field: str) -> list[int]:
        c = CLASSES.index(field)
        return [i for i in range(len(spans)) if proba[i, c] >= th.of(field)]

    def best(field: str, idx: list[int]) -> float:
        c = CLASSES.index(field)
        return float(max(proba[i, c] for i in idx)) if idx else 0.0

    # seller
    idx = picked("seller")
    if idx:
        used = reading_order([spans[i] for i in idx], lambda s: s.bbox)
        text = normalize_text(" ".join(s.text for s in used))
        canon = gazetteer.canonical_seller(text) if gazetteer else None
        fields["seller"], raw["seller"] = canon or text, text
        evidence["seller"], conf["seller"] = _evidence(used), best("seller", idx)

    # address
    idx = picked("seller_address")
    if idx:
        used = reading_order([spans[i] for i in idx], lambda s: s.bbox)
        text = normalize_text(" ".join(s.text for s in used))
        canon = gazetteer.match_address(text, fields.get("seller")) if gazetteer else None
        fields["seller_address"] = canon or (admin.correct(text) if admin else text)
        raw["seller_address"] = text
        evidence["seller_address"] = _evidence(used)
        conf["seller_address"] = best("seller_address", idx)

    # timestamp: parse the picked spans together, else the single most likely span
    c = CLASSES.index("timestamp")
    idx = picked("timestamp")
    order = sorted(range(len(spans)), key=lambda i: -proba[i, c])
    attempts = [idx] if idx else []
    attempts += [[i] for i in order[:3]]
    for group in attempts:
        used = reading_order([spans[i] for i in group], lambda s: s.bbox)
        text = " ".join(s.text for s in used)
        parsed = parse_timestamp(text)
        if not (parsed and parsed.iso):
            continue
        if parsed.time is None:  # "Ngày: 21/05/2020" and "16:44:18" in separate boxes
            j = _time_nearby(spans, proba[:, c], used)
            if j is not None:
                used = reading_order([*used, spans[j]], lambda s: s.bbox)
                text = " ".join(s.text for s in used)
                parsed = parse_timestamp(text) or parsed
        fields["timestamp"], raw["timestamp"] = parsed.iso, normalize_text(text)
        evidence["timestamp"] = _evidence(used)
        conf["timestamp"] = float(max(proba[i, c] for i in group))
        break

    # total: the most likely span with an amount; a keyword span takes its neighbour's
    c = CLASSES.index("total_cost")
    for i in sorted(range(len(spans)), key=lambda i: -proba[i, c])[:4]:
        if proba[i, c] < th.total_cost:
            break
        amounts = RuleBasedKIE._amounts(spans[i].text)
        if amounts:
            fields["total_cost"], used = amounts[-1], [spans[i]]
        else:
            near = RuleBasedKIE()._amount_near(spans[i], spans)
            if not near:
                continue
            fields["total_cost"], used = near[0], [spans[i], near[1]]
        raw["total_cost"] = normalize_text(" ".join(s.text for s in used))
        evidence["total_cost"], conf["total_cost"] = _evidence(used), float(proba[i, c])
        break

    warnings = []
    for f in FIELD_NAMES:
        if f not in fields:
            warnings.append(
                ExtractionWarning(code="missing_field", field=f, message=f"No {f} found.")
            )
        elif conf[f] < th.confident:
            warnings.append(
                ExtractionWarning(
                    code="low_confidence",
                    field=f,
                    message=f"The model is unsure about this value ({conf[f]:.0%}).",
                )
            )
    return validate_extraction(
        ReceiptExtraction(
            model_version=model_version,
            fields=ReceiptFields(**fields),
            raw_fields=RawFields(**raw),
            evidence=evidence,
            warnings=warnings,
        )
    )


class LineClassifierKIE(KIEProvider):
    name = "lines"

    def __init__(
        self,
        gazetteer: Gazetteer | None = None,
        admin: AdminUnits | None = None,
        model_path: Path = MODEL_PATH,
        meta_path: Path = META_PATH,
    ) -> None:
        import lightgbm as lgb

        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} missing: run python -m evaluation.train_line_classifier"
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta["features"] != FEATURE_NAMES:
            raise ValueError(f"{model_path} was trained on other features; retrain it")
        self.booster = lgb.Booster(model_file=str(model_path))
        self.thresholds = Thresholds(**self.meta["thresholds"])
        self.gazetteer = gazetteer
        self.admin = admin

    @property
    def version(self) -> str:  # type: ignore[override]
        return (
            self.meta["version"]
            + ("+gaz" if self.gazetteer else "")
            + ("+adm" if self.admin else "")
        )

    def predict_proba(self, ocr_spans: list[OCRSpan]) -> tuple[list[OCRSpan], np.ndarray]:
        spans = order_spans(ocr_spans)
        if not spans:
            return spans, np.zeros((0, len(CLASSES)))
        x = span_features(spans, self.gazetteer, self.admin)
        return spans, self.booster.predict(x)

    def extract(self, image: Image.Image | None, ocr_spans: list[OCRSpan]) -> ReceiptExtraction:
        spans, proba = self.predict_proba(ocr_spans)
        return assemble(
            spans,
            proba,
            self.thresholds,
            self.gazetteer,
            self.admin,
            f"{self.name}-{self.version}",
        )
