"""Rule-based key information extraction (Phase 3A baseline).

Works on OCR spans only (no image), field by field:

* **timestamp** — every visual line is parsed with the Vietnamese date parser;
  lines with a date keyword ("ngày", "thời gian", "giờ") and a time win. A time on
  the next line is attached to a date-only line ("Ngày: …" / "Giờ: …").
* **total_cost** — keyword lines in priority order ("… phải thanh toán" >
  "tổng tiền" > "tổng cộng" > "cộng tiền hàng"), negatives excluded ("tiền khách
  đưa", "tiền thừa", "chiết khấu", "VAT", item counts). The amount is taken from
  the keyword span itself, then spans to its right on the same line, then the
  line below.
* **header block** — spans above the first title/date/table-header line.
  **seller**: a gazetteer match (optional) or the tallest non-address line;
  **seller_address**: address-like lines of the header, in reading order.

All matching runs on accent-free folded text because OCR diacritics are
unreliable. Every value carries evidence pointing at the OCR spans it came from.
Nothing is guessed: a field without support is ``None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from PIL import Image
from rapidfuzz import fuzz

from ai.kie.base import KIEProvider
from ai.kie.gazetteer import Gazetteer
from ai.kie.validate import validate_extraction
from ai.layout import group_lines, reading_order
from ai.normalization.datetime_vi import parse_timestamp
from ai.normalization.money import find_amounts, to_vnd
from ai.normalization.text import fold_noacc, normalize_text
from ai.schemas import (
    Evidence,
    ExtractionWarning,
    FieldEvidence,
    OCRSpan,
    RawFields,
    ReceiptExtraction,
    ReceiptFields,
)

MIN_TOTAL_VND = 1_000  # smallest total in train is 3,000; smaller numbers are counts/codes
MAX_TOTAL_VND = 1_000_000_000


@dataclass
class Line:
    spans: list[OCRSpan]
    key: str = field(init=False)  # accent-free folded text
    text: str = field(init=False)

    def __post_init__(self) -> None:
        self.text = " ".join(s.text for s in self.spans)
        self.key = fold_noacc(self.text)

    @property
    def y0(self) -> float:
        return min(s.bbox[1] for s in self.spans)

    @property
    def y1(self) -> float:
        return max(s.bbox[3] for s in self.spans)

    @property
    def height(self) -> float:
        return max(s.bbox[3] - s.bbox[1] for s in self.spans)


def _evidence(spans: list[OCRSpan]) -> FieldEvidence:
    return FieldEvidence(
        regions=[Evidence(bbox=s.bbox, text=s.text, source_id=s.id) for s in spans]
    )


# --------------------------------------------------------------------------- patterns
# All patterns run on fold_noacc() text: lower-case, no accents, punctuation -> space.

_DATE_KW = re.compile(r"\b(ngay|thoi gian|gio|date|time|tg)\b")
_NOT_TX_DATE = re.compile(
    r"\b(han su dung|hsd|het han|exp|ngay sinh|in lai|reprint|hieu luc|den ngay|voucher)\b"
)

# (priority, pattern): lower number wins
_TOTAL_KW: list[tuple[int, re.Pattern]] = [
    (0, re.compile(r"phai t\w* toan|phai thanh toan|tien phai tra|khach (can|phai) tra")),
    (1, re.compile(r"\btong (so )?(tien )?thanh toan\b|\bcan thanh toan\b|\bso tien thanh toan")),
    (2, re.compile(r"^(tien )?thanh toan\b|\btotal\b|\bgrand total\b")),
    (3, re.compile(r"\btong tien\b")),
    (4, re.compile(r"\btong cong\b|\btong cng\b")),
    (5, re.compile(r"\bcong tien hang\b|^cong\b")),
    (6, re.compile(r"^tong\b|\bthanh tien\b|\btai quay\b")),
]
_TOTAL_NEG = re.compile(
    r"khach dua|khach tra\b|tien thua|tra lai|thoi lai|chiet khau|giam gia|da giam|"
    r"\bvat\b|\bthue\b|so luong|\bsl\b|tong so\s*(:|\d|mat hang|san pham)|tich luy|\bdiem\b|"
    r"truoc thue|tien mat|the tin dung|\bck\b|\bthua\b"
)
_CURRENCY_PREFIX = re.compile(r"(?i)\b[vu]n[dđ]\.?\s*")  # OCR reads "VND" as "UND"
_CURRENCY_SUFFIX = re.compile(r"(?i)(?<=\d)\s*(vn[dđ]|[dđ₫])\b")
_MIXED_TOKEN = re.compile(r"\b(?=[\w.]*[^\W\d_])(?=[\w.]*\d)[\w.]+\b")  # "XXXX8722", "A4"
_LONG_DIGITS = re.compile(r"(?<![\d.,])\d{9,}(?![\d.,])")  # phones, ids
_NEGATIVE = re.compile(r"-\s*\d[\d.,]*")
_TAX_ID = re.compile(r"(?i)\b(mst|ma so thue|tax)\W*[\d\s.-]+")

# Fuzzy fallback for keywords mangled by OCR ("Tőngcng", "Cng tièn hàng").
# ratio >= 85 keeps "thanh tien" (column header, 80) away from "thanh toan".
_TOTAL_FUZZY: list[tuple[float, str]] = [
    (0.5, "tong tien phai thanh toan"),
    (0.5, "tong tien phai t toan"),
    (0.5, "khach phai tra"),
    (1.5, "tong thanh toan"),
    (1.5, "tong so tien thanh toan"),
    (1.5, "tong so thanh toan"),
    (3.5, "tong tien"),
    (4.5, "tong cong"),
    (5.5, "cong tien hang"),
    (5.5, "tong tien hang"),
]
_FUZZY_MIN = 85.0


def _total_priority(key: str) -> float | None:
    prio = next((p for p, rx in _TOTAL_KW if rx.search(key)), None)
    if prio is not None:
        return float(prio)
    head = re.sub(r"[\d\s]+$", "", key).strip()  # drop a trailing amount
    if not 4 <= len(head) <= 30:
        return None
    best = max(((fuzz.ratio(head, phrase), p) for p, phrase in _TOTAL_FUZZY), default=(0, None))
    return best[1] if best[0] >= _FUZZY_MIN else None


_TITLE = re.compile(
    r"\b(hoa d\w*n|phieu|bien lai|ban hang|tinh tien|thanh toan|bill|receipt|invoice|"
    r"ten hang|mat hang|don gia|sl|so hd|so ct|so gd|quay|thu ngan|nhan vien|khach hang)\b"
)
_META = re.compile(
    r"\b(dt|d t|tel|dien thoai|sdt|hotline|fax|mst|ma so thue|tax|website|web|email|www|"
    r"http|com|vn)\b|@|^\W*$"
)
_ADDRESS_HINT = re.compile(
    r"\b(so|duong|d|pho|ngo|ngach|hem|kiet|phuong|p|quan|q|huyen|h|xa|thon|to|khu|ap|"
    r"tp|thanh pho|tinh|t|thi xa|tx|dc|dia chi|cho|kdt|toa|tang|chung cu|lo|"
    r"ha noi|hn|hcm|ho chi minh|quang ninh|qn|cam pha|gia lam|vn)\b|\d+[a-z]?\s*[/,-]"
)
_PRICE = re.compile(r"(?<![\d.,])\d{1,3}[.,]\d{3}(?![\d])")  # "149,000": item lines begin
_HEADER_WORDS = ("don gia", "thanh tien", "hoa don", "phieu", "so luong", "mat hang", "page")
_STORE_CODE = re.compile(r"^(vm|vin|vmt)\b")  # "VM+ QNH …" store-name lines: part of the address


# --------------------------------------------------------------------------- extractor


class RuleBasedKIE(KIEProvider):
    name = "rules"

    def __init__(self, gazetteer: Gazetteer | None = None, seller_fallback: bool = True) -> None:
        self.gazetteer = gazetteer
        self.seller_fallback = seller_fallback

    @property
    def version(self) -> str:  # type: ignore[override]
        return "v1" + ("+gaz" if self.gazetteer else "") + ("" if self.seller_fallback else "-nofb")

    def extract(self, image: Image.Image | None, ocr_spans: list[OCRSpan]) -> ReceiptExtraction:
        spans = [s for s in ocr_spans if s.text.strip()]
        lines = [Line(ls) for ls in group_lines(spans, lambda s: s.bbox)]
        fields: dict = {}
        raw: dict = {}
        evidence: dict = {}
        warnings: list[ExtractionWarning] = []

        for name, fn in (
            ("timestamp", self._timestamp),
            ("total_cost", self._total),
        ):
            value, raw_text, used = fn(lines)
            if value is not None:
                fields[name], raw[name], evidence[name] = value, raw_text, _evidence(used)

        header = self._header(lines)
        seller, seller_raw, seller_spans = self._seller(header)
        if seller:
            fields["seller"], raw["seller"] = seller, seller_raw
            evidence["seller"] = _evidence(seller_spans)
        address, address_raw, address_spans = self._address(header, seller_spans, seller)
        if address:
            fields["seller_address"], raw["seller_address"] = address, address_raw
            evidence["seller_address"] = _evidence(address_spans)

        for f in ("seller", "seller_address", "timestamp", "total_cost"):
            if f not in fields:
                warnings.append(
                    ExtractionWarning(
                        code="missing_field", field=f, message=f"No {f} found on the receipt."
                    )
                )
        return validate_extraction(
            ReceiptExtraction(
                model_version=f"{self.name}-{self.version}",
                fields=ReceiptFields(**fields),
                raw_fields=RawFields(**raw),
                evidence=evidence,
                warnings=warnings,
            )
        )

    # ------------------------------------------------------------------ timestamp
    def _timestamp(self, lines: list[Line]):
        best = None  # (score, -index, iso, raw, spans)
        for i, line in enumerate(lines):
            parsed = parse_timestamp(line.text)
            if not parsed or parsed.date is None:
                continue
            used = [s for s in line.spans if re.search(r"\d", s.text)]
            text = line.text
            if parsed.time is None and i + 1 < len(lines):  # "Ngày: …" then "Giờ: …"
                nxt = lines[i + 1]
                t = parse_timestamp(nxt.text)
                if t and t.time and t.date is None:
                    parsed.time, parsed.has_seconds = t.time, t.has_seconds
                    used += [s for s in nxt.spans if re.search(r"\d", s.text)]
                    text = f"{text} {nxt.text}"
            score = 2 * bool(_DATE_KW.search(line.key)) + (parsed.time is not None)
            score -= 3 * bool(_NOT_TX_DATE.search(line.key))
            cand = (score, -i, parsed.iso, text, used)
            if best is None or cand[:2] > best[:2]:
                best = cand
        if best is None:
            return None, None, []
        return best[2], normalize_text(best[3]), best[4]

    # ------------------------------------------------------------------ total
    @staticmethod
    def _amounts(text: str) -> list[int]:
        """Plausible VND amounts in ``text``.

        Currency marks are removed first ("VND75,500", "UND461,288" — OCR reads V as U —
        "27.000đ"); tokens mixing letters and digits (card numbers "XXXX8722", codes
        "VIETTEL20", "A4") and long separator-free digit runs (phones, ids) are dropped.
        """
        t = _TAX_ID.sub(" ", text)
        t = _CURRENCY_PREFIX.sub(" ", t)
        t = _CURRENCY_SUFFIX.sub(" ", t)
        t = _MIXED_TOKEN.sub(" ", t)
        t = _LONG_DIGITS.sub(" ", t)
        t = _NEGATIVE.sub(" ", t)  # refunds / change are not totals
        return [v for v in map(to_vnd, find_amounts(t)) if MIN_TOTAL_VND <= v <= MAX_TOTAL_VND]

    @staticmethod
    def _cy(s: OCRSpan) -> float:
        return (s.bbox[1] + s.bbox[3]) / 2

    def _amount_near(self, kw: OCRSpan, spans: list[OCRSpan]):
        """Amount right of ``kw`` on the same visual line (closest centre), else just below."""
        h = kw.bbox[3] - kw.bbox[1]
        kx_mid = (kw.bbox[0] + kw.bbox[2]) / 2
        same, below = [], []
        for s in spans:
            if s is kw or s.bbox[0] < kx_mid:
                continue
            vals = self._amounts(s.text)
            if not vals:
                continue
            dy = self._cy(s) - self._cy(kw)
            tol = 0.6 * max(h, s.bbox[3] - s.bbox[1])
            if abs(dy) <= tol:
                same.append((abs(dy), s.bbox[0], vals[-1], s))
            elif 0 < dy <= 1.8 * h:
                below.append((dy, -s.bbox[2], vals[-1], s))
        pick = min(same) if same else (min(below) if below else None)
        return (pick[2], pick[3]) if pick else None

    def _total(self, lines: list[Line]):
        spans = [s for line in lines for s in line.spans]
        cands = []  # (priority, -y, value, raw, spans)
        for span in spans:
            key = fold_noacc(span.text)
            prio = _total_priority(key)
            if prio is None or (prio > 1 and _TOTAL_NEG.search(key)):
                continue
            found = self._amounts(span.text)  # "Tổng cộng: 74,000" in one span
            if found:
                cands.append((prio, -span.bbox[1], found[-1], span.text, [span]))
                continue
            near = self._amount_near(span, spans)
            if near:
                value, s = near
                cands.append((prio, -span.bbox[1], value, f"{span.text} {s.text}", [span, s]))
        if not cands:
            return None, None, []
        best = min(cands, key=lambda c: (c[0], c[1]))  # priority, then bottom-most
        return best[2], normalize_text(best[3]), best[4]

    # ------------------------------------------------------------------ header
    @staticmethod
    def _header(lines: list[Line]) -> list[Line]:
        """Lines above the first title / date / table-header / price line (max top 50 %)."""
        if not lines:
            return []
        page_h = max(line.y1 for line in lines)
        out = []
        for line in lines:
            if line.y0 > 0.5 * page_h:
                break
            if out and (
                _TITLE.search(line.key) or parse_timestamp(line.text) or _PRICE.search(line.text)
            ):
                break
            if not out and _TITLE.search(line.key) and not _ADDRESS_HINT.search(line.key):
                continue  # a title above the store name ("HÓA ĐƠN" on top): skip it
            out.append(line)
        return out

    @staticmethod
    def _is_meta(key: str) -> bool:
        digits = sum(c.isdigit() for c in key)
        letters = sum(c.isalpha() for c in key)
        return bool(_META.search(key)) or (digits >= 7 and digits > letters)

    def _seller(self, header: list[Line]):
        spans = [s for line in header for s in line.spans]
        if self.gazetteer:
            hit = self.gazetteer.match_seller(spans)
            if hit:
                name, used = hit
                return name, normalize_text(" ".join(s.text for s in used)), used
        if not self.seller_fallback or not header:
            return None, None, []
        page_top = header[0].y0
        cands = []
        for idx, line in enumerate(header[:6]):
            for s in line.spans:
                key = fold_noacc(s.text)
                letters = sum(c.isalpha() for c in key)
                if letters < 4 or self._is_meta(key) or _TITLE.search(key):
                    continue
                if any(fuzz.partial_ratio(key, w) >= 90 for w in _HEADER_WORDS):
                    continue  # OCR-mangled column headers ("dn giá")
                if s.bbox[1] - page_top > 6 * line.height:
                    continue  # store names sit at the very top of the receipt
                addressy = bool(_ADDRESS_HINT.search(key)) or bool(_STORE_CODE.search(key))
                # tall text near the top, not address-like
                score = (s.bbox[3] - s.bbox[1]) * (0.5 if addressy else 1.0) - 2 * idx
                cands.append((score, s))
        if not cands:
            return None, None, []
        span = max(cands, key=lambda c: c[0])[1]
        text = normalize_text(span.text)
        return text, text, [span]

    def _address(self, header: list[Line], seller_spans: list[OCRSpan], seller: str | None):
        taken = {s.id for s in seller_spans}
        used = []
        for line in header:
            for s in line.spans:
                if s.id in taken:
                    continue
                key = fold_noacc(s.text)
                if (
                    not key
                    or self._is_meta(key)
                    or (_TITLE.search(key) and not _ADDRESS_HINT.search(key))
                ):
                    continue
                if _ADDRESS_HINT.search(key) or _STORE_CODE.search(key):
                    used.append(s)
        if not used:
            return None, None, []
        used = reading_order(used, lambda s: s.bbox)
        raw = normalize_text(" ".join(s.text for s in used))
        value = raw
        if self.gazetteer:
            canon = self.gazetteer.match_address(raw, seller)
            if canon:
                value = canon
        return value, raw, used
