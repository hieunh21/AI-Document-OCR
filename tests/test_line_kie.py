"""Learned span classifier KIE: labels, features, field assembly, saved model."""

from dataclasses import dataclass

import numpy as np
import pytest

from ai.kie.learned import META_PATH, MODEL_PATH, Thresholds, assemble
from ai.kie.line_features import CLASSES, FEATURE_NAMES, order_spans, span_features, span_labels
from ai.mock import MOCK_SPANS
from ai.schemas import OCRSpan


@dataclass
class _Region:
    field: str
    bbox: list[float]


def _span(i: int, text: str, bbox) -> OCRSpan:
    return OCRSpan(id=f"s{i}", text=text, bbox=bbox)


def _proba(spans, cls_by_id: dict[str, tuple[str, float]]) -> np.ndarray:
    p = np.zeros((len(spans), len(CLASSES)))
    p[:, 0] = 1.0
    for i, s in enumerate(spans):
        if s.id in cls_by_id:
            name, v = cls_by_id[s.id]
            p[i] = (1 - v) / (len(CLASSES) - 1)
            p[i, CLASSES.index(name)] = v
    return p


def test_span_labels_use_the_smaller_box():
    spans = [
        _span(0, "Số GD: 000AB2212008003990 Ngày: 16/08/2020-17:36", [0, 100, 600, 130]),
        _span(1, "Xin cảm ơn", [100, 300, 300, 330]),
    ]
    # The annotation covers only the date at the right end of the long OCR span.
    regions = [_Region("timestamp", [400, 100, 600, 130])]
    assert span_labels(spans, regions).tolist() == [CLASSES.index("timestamp"), 0]


def test_features_have_one_row_per_span():
    spans = order_spans(MOCK_SPANS)
    x = span_features(spans, None, None)
    assert x.shape == (len(spans), len(FEATURE_NAMES))
    assert np.isfinite(x).all()
    assert span_features([], None, None).shape == (0, len(FEATURE_NAMES))
    date_col = FEATURE_NAMES.index("date")
    assert x[[s.id for s in spans].index("span_03"), date_col] == 1.0


def test_assemble_builds_fields_from_span_classes():
    spans = order_spans(
        [
            _span(0, "SIÊU THỊ MINI AN KHANG", [100, 10, 500, 50]),
            _span(1, "112 Lê Lợi, P. Bến Thành, Q.1", [80, 60, 520, 85]),
            _span(2, "Ngày: 05/10/2026", [40, 120, 300, 145]),
            _span(3, "19:22", [400, 121, 480, 145]),
            _span(4, "TỔNG CỘNG:", [40, 400, 220, 430]),
            _span(5, "307.000", [420, 400, 560, 430]),
            _span(6, "Tiền khách đưa: 350.000", [40, 440, 560, 470]),
        ]
    )
    proba = _proba(
        spans,
        {
            "s0": ("seller", 0.9),
            "s1": ("seller_address", 0.9),
            "s2": ("timestamp", 0.9),  # the time box is not picked: it is attached
            "s4": ("total_cost", 0.9),  # keyword box: the amount is read next to it
            "s6": ("total_cost", 0.3),
        },
    )
    ex = assemble(spans, proba, Thresholds(confident=0.8), None, None, "test")
    f = ex.fields
    assert (f.seller, f.seller_address) == (
        "SIÊU THỊ MINI AN KHANG",
        "112 Lê Lợi, P. Bến Thành, Q.1",
    )
    assert f.timestamp == "2026-10-05T19:22"
    assert f.total_cost == 307000
    assert ex.evidence["timestamp"].span_ids == ["s2", "s3"]
    assert ex.evidence["total_cost"].span_ids == ["s4", "s5"]
    assert ex.warnings == []


def test_assemble_flags_unsure_and_missing_fields():
    spans = order_spans(
        [_span(0, "CỬA HÀNG ABC", [0, 0, 300, 40]), _span(1, "xyz", [0, 50, 100, 80])]
    )
    proba = _proba(spans, {"s0": ("seller", 0.6)})
    ex = assemble(spans, proba, Thresholds(seller=0.5, confident=0.8), None, None, "test")
    codes = {(w.code, w.field) for w in ex.warnings}
    assert ("low_confidence", "seller") in codes
    assert {("missing_field", f) for f in ("seller_address", "timestamp", "total_cost")} <= codes


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="no trained model")
def test_saved_model_runs_on_a_receipt():
    from ai.kie.learned import LineClassifierKIE

    kie = LineClassifierKIE()
    assert kie.meta["features"] == FEATURE_NAMES and META_PATH.exists()
    ex = kie.extract(None, MOCK_SPANS)
    ids = {s.id for s in MOCK_SPANS}
    for ev in ex.evidence.values():
        assert set(ev.span_ids) <= ids
    assert ex.fields.total_cost == 150000 and ex.fields.timestamp == "2026-10-08T09:26"
