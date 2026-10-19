"""Features of every OCR span of a receipt, for the learned line classifier.

The same function runs at training time (``evaluation/train_line_classifier.py``) and
at inference (``ai.kie.learned.LineClassifierKIE``), so the model always sees features
computed the same way. Spans must be in the *upright* page frame (``upright_spans``).

Feature groups (one row per span, in reading order):

* geometry: position on the page, size relative to the median text height, place in its
  visual line, distance from the top and bottom;
* text: length, share of digits / letters / capitals;
* patterns reused from the rule extractor: dates and times, total keywords and their
  negatives, titles, phone / tax lines, address hints, amounts (and whether the amount is
  the largest on the receipt);
* knowledge: similarity to a known seller or address (gazetteer) and official place names;
* context: a few of the above for the previous and next span and for the same line.

Nothing here reads labels; the gazetteer passed in decides what counts as "known".
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process

from ai.kie.gazetteer import Gazetteer
from ai.kie.rule_based import (
    _ADDRESS_HINT,
    _DATE_KW,
    _HEADER_WORDS,
    _META,
    _NOT_TX_DATE,
    _PRICE,
    _STORE_CODE,
    _TITLE,
    _TOTAL_NEG,
    RuleBasedKIE,
    _total_priority,
)
from ai.layout import group_lines, reading_order
from ai.normalization.address import AdminUnits
from ai.normalization.datetime_vi import parse_timestamp
from ai.normalization.text import fold_noacc
from ai.schemas import OCRSpan

CLASSES = ("other", "seller", "seller_address", "timestamp", "total_cost")
FIELD_CLASS = {name: i for i, name in enumerate(CLASSES)}

_STORE_WORD = re.compile(
    r"\b(sieu thi|cua hang|minimart|mini mart|mart|shop|store|cafe|ca phe|coffee|tea|tra sua|"
    r"nha thuoc|quay thuoc|duoc|tap hoa|bach hoa|thuc pham|nha hang|quan|bakery|banh|"
    r"vinmart|vinmart\+|co op|coopmart|circle k|family ?mart|ministop|gs25|7 eleven|"
    r"cong ty|cty|tnhh|chi nhanh|cn)\b"
)
_LONG_NUMBER = re.compile(r"\d{6,}")


@dataclass
class _Base:
    """Per-span values that neighbours also read."""

    date: float
    time_only: float
    total_prio: float
    address_hint: float
    admin: float
    store_word: float
    amount: float
    rel_h: float


BASE_NAMES = [f.name for f in _Base.__dataclass_fields__.values()]  # type: ignore[attr-defined]

FEATURE_NAMES = [
    # geometry
    "y0", "y1", "yc", "x0", "x1", "xc", "rel_h", "rel_w", "center_offset", "line_idx",
    "line_pos", "lines_below", "n_in_line", "pos_in_line", "is_first_in_line",
    "is_last_in_line", "in_header",
    # text
    "n_chars", "n_words", "digit_ratio", "alpha_ratio", "upper_ratio", "has_colon",
    "starts_digit",
    # patterns
    "date", "time_only", "date_kw", "not_tx_date", "total_prio", "total_neg", "title",
    "meta", "address_hint", "store_code", "store_word", "header_word", "long_number",
    "price", "amount", "n_amounts", "amount_is_max", "amount_rank",
    # knowledge
    "gaz_seller", "gaz_address", "admin",
    # context
    "line_total_prio", "line_has_amount", "line_has_date", "left_total_prio",
    "line_admin", "line_address_hint",
    *[f"prev_{n}" for n in BASE_NAMES],
    *[f"next_{n}" for n in BASE_NAMES],
    "prev_line_address_hint", "next_line_address_hint", "prev_line_admin", "next_line_admin",
]  # fmt: skip


def order_spans(spans: list[OCRSpan]) -> list[OCRSpan]:
    """Non-empty spans in reading order: the row order of ``span_features``."""
    return reading_order([s for s in spans if s.text.strip()], lambda s: s.bbox)


def _header_ids(lines: list[list[OCRSpan]]) -> set[str]:
    """Spans of the header block as the rule extractor sees it."""
    from ai.kie.rule_based import Line

    return {s.id for line in RuleBasedKIE._header([Line(ls) for ls in lines]) for s in line.spans}


def span_features(
    spans: list[OCRSpan], gazetteer: Gazetteer | None, admin: AdminUnits | None
) -> np.ndarray:
    """Feature matrix ``(len(spans), len(FEATURE_NAMES))`` for spans already in reading
    order (``order_spans``)."""
    n = len(spans)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    lines = group_lines(spans, lambda s: s.bbox)
    line_of = {s.id: (li, pos) for li, line in enumerate(lines) for pos, s in enumerate(line)}
    x_min = min(s.bbox[0] for s in spans)
    x_max = max(s.bbox[2] for s in spans)
    y_min = min(s.bbox[1] for s in spans)
    y_max = max(s.bbox[3] for s in spans)
    W, H = max(x_max - x_min, 1.0), max(y_max - y_min, 1.0)
    med_h = statistics.median(max(s.bbox[3] - s.bbox[1], 1.0) for s in spans)
    header = _header_ids(lines)

    keys = [fold_noacc(s.text) for s in spans]
    amounts = [RuleBasedKIE._amounts(s.text) for s in spans]
    all_amounts = sorted({a for am in amounts for a in am}, reverse=True)
    seller_keys = [e["key"] for e in gazetteer.sellers] if gazetteer else []
    address_keys = [e["key"] for e in gazetteer.addresses] if gazetteer else []

    base: list[_Base] = []
    rows = []
    for i, (s, key) in enumerate(zip(spans, keys, strict=True)):
        text = s.text
        parsed = parse_timestamp(text)
        letters = [c for c in text if c.isalpha()]
        digits = sum(c.isdigit() for c in text)
        prio = _total_priority(key)
        admin_hits = len(admin.find(text)) if admin else 0
        b = _Base(
            date=float(bool(parsed and parsed.date)),
            time_only=float(bool(parsed and parsed.time and not parsed.date)),
            total_prio=9.0 if prio is None else prio,
            address_hint=float(len(_ADDRESS_HINT.findall(key))),
            admin=float(admin_hits),
            store_word=float(bool(_STORE_WORD.search(key))),
            amount=float(bool(amounts[i])),
            rel_h=(s.bbox[3] - s.bbox[1]) / med_h,
        )
        base.append(b)
        li, pos = line_of[s.id]
        gaz_seller = (
            process.extractOne(key, seller_keys, scorer=fuzz.ratio)[1]
            if seller_keys and len(key) >= 3
            else 0.0
        )
        gaz_address = (
            process.extractOne(key, address_keys, scorer=fuzz.partial_ratio)[1]
            if address_keys and len(key) >= 6
            else 0.0
        )
        top_amount = max(amounts[i]) if amounts[i] else 0
        rows.append(
            [
                (s.bbox[1] - y_min) / H,
                (s.bbox[3] - y_min) / H,
                ((s.bbox[1] + s.bbox[3]) / 2 - y_min) / H,
                (s.bbox[0] - x_min) / W,
                (s.bbox[2] - x_min) / W,
                ((s.bbox[0] + s.bbox[2]) / 2 - x_min) / W,
                b.rel_h,
                (s.bbox[2] - s.bbox[0]) / W,
                abs(((s.bbox[0] + s.bbox[2]) / 2 - x_min) / W - 0.5),
                min(li, 60),
                li / max(len(lines) - 1, 1),
                min(len(lines) - 1 - li, 60),
                len(lines[li]),
                pos,
                float(pos == 0),
                float(pos == len(lines[li]) - 1),
                float(s.id in header),
                len(text),
                len(text.split()),
                digits / max(len(text), 1),
                len(letters) / max(len(text), 1),
                sum(c.isupper() for c in letters) / max(len(letters), 1),
                float(":" in text),
                float(text.strip()[:1].isdigit()),
                b.date,
                b.time_only,
                float(bool(_DATE_KW.search(key))),
                float(bool(_NOT_TX_DATE.search(key))),
                b.total_prio,
                float(bool(_TOTAL_NEG.search(key))),
                float(bool(_TITLE.search(key))),
                float(bool(_META.search(key))),
                b.address_hint,
                float(bool(_STORE_CODE.search(key))),
                b.store_word,
                float(any(fuzz.partial_ratio(key, w) >= 90 for w in _HEADER_WORDS)),
                float(bool(_LONG_NUMBER.search(key))),
                float(bool(_PRICE.search(text))),
                b.amount,
                len(amounts[i]),
                float(bool(top_amount) and top_amount == all_amounts[0]),
                float(all_amounts.index(top_amount) if top_amount else 11),
                gaz_seller,
                gaz_address,
                b.admin,
            ]
        )

    for i, s in enumerate(spans):
        li, pos = line_of[s.id]
        same = [spans.index(o) for o in lines[li]]
        left = [j for j in same if spans[j].bbox[0] < s.bbox[0]]
        prev_b = base[i - 1] if i > 0 else None
        next_b = base[i + 1] if i + 1 < n else None

        def line_stat(line_idx: int, attr: str) -> float:
            if not 0 <= line_idx < len(lines):
                return -1.0
            return max(getattr(base[spans.index(o)], attr) for o in lines[line_idx])

        rows[i] += [
            min(base[j].total_prio for j in same),
            max(base[j].amount for j in same),
            max(base[j].date for j in same),
            min((base[j].total_prio for j in left), default=9.0),
            max(base[j].admin for j in same),
            max(base[j].address_hint for j in same),
            *[getattr(prev_b, k) if prev_b else -1.0 for k in BASE_NAMES],
            *[getattr(next_b, k) if next_b else -1.0 for k in BASE_NAMES],
            line_stat(li - 1, "address_hint"),
            line_stat(li + 1, "address_hint"),
            line_stat(li - 1, "admin"),
            line_stat(li + 1, "admin"),
        ]
    out = np.asarray(rows, dtype=np.float32)
    assert out.shape[1] == len(FEATURE_NAMES), (out.shape, len(FEATURE_NAMES))
    return np.nan_to_num(out, nan=-1.0, posinf=1e6, neginf=-1e6)


def span_labels(spans: list[OCRSpan], regions, min_overlap: float = 0.5) -> np.ndarray:
    """Class of each span from the annotated regions (``evaluation.dataset_adapter.Region``).

    Overlap is measured against the *smaller* of the two boxes: OCR often reads a whole
    printed line ("Số GD: 000AB… Ngày: 13/08/2020-19:34") as one span while the
    annotation covers only the date, and a long annotated address can span several OCR
    boxes. The span takes the field with the largest overlap (at least ``min_overlap``),
    else "other". Boxes must be in the same frame.
    """

    def area(b) -> float:
        return max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)

    out = np.zeros(len(spans), dtype=np.int64)
    for i, s in enumerate(spans):
        best, best_share = 0, min_overlap
        for r in regions:
            ix = min(s.bbox[2], r.bbox[2]) - max(s.bbox[0], r.bbox[0])
            iy = min(s.bbox[3], r.bbox[3]) - max(s.bbox[1], r.bbox[1])
            share = max(ix, 0) * max(iy, 0) / min(area(s.bbox), area(r.bbox))
            if share >= best_share:
                best, best_share = FIELD_CLASS[r.field], share
        out[i] = best
    return out
