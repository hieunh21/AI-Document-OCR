"""Parse VND amounts as printed on Vietnamese receipts.

Handled: ``150.000``, ``150,000``, ``150 000``, ``150000``, ``150.000đ``,
``VND 150,000``, ``880,400.00`` (decimal part), ``0.357.075`` (leading zero),
``47 . 500`` (spaces around the separator).
Dates (``14/08/2020``, ``14-08-2020``) and clock times (``08:06``) are masked
first, so they never produce amounts.
Not handled on purpose: words ("một trăm nghìn"), ``150k``.
"""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

_TOKEN = re.compile(r"\d[\d.,]*")
_SPACED_SEP = re.compile(r"(?<=\d) ?([.,]) ?(?=\d{3}(?!\d))")
_DATE = re.compile(r"(?<!\d)\d{1,4}[/\-]\d{1,2}[/\-]\d{2,4}(?!\d)")
_CLOCK = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)")


def _clean(text: str) -> str:
    text = _DATE.sub(" ", text)
    text = _CLOCK.sub(" ", text)
    return _SPACED_SEP.sub(lambda m: m.group(1), text)


def _merge_space_groups(text: str) -> list[str]:
    """Find numeric tokens, merging thousand groups split by single spaces.

    ``"35 000"`` -> ``["35000"]`` but ``"10.000 10.000"`` -> ``["10.000", "10.000"]``.
    """
    tokens: list[tuple[int, int, str]] = [
        (m.start(), m.end(), m.group().rstrip(".,")) for m in _TOKEN.finditer(text)
    ]
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        start, end, tok = tokens[i]
        while (
            i + 1 < len(tokens)
            and text[end : tokens[i + 1][0]] == " "
            and re.fullmatch(r"\d{3}", tokens[i + 1][2])
            and re.fullmatch(r"\d{1,3}", tok.split(".")[-1].split(",")[-1])
            and not re.search(r"[.,]\d{1,2}$", tok)  # tok already has a decimal part
        ):
            i += 1
            tok, end = tok + tokens[i][2], tokens[i][1]
        merged.append(tok)
        i += 1
    return merged


def parse_number_token(tok: str) -> Decimal | None:
    tok = tok.strip(".,")
    if not tok:
        return None
    seps = [(i, c) for i, c in enumerate(tok) if c in ".,"]
    if not seps:
        return Decimal(tok)
    last_i = seps[-1][0]
    tail = tok[last_i + 1 :]
    decimal_part = ""
    if len(tail) in (1, 2):
        # Thousand groups always have 3 digits, so a 1-2 digit tail is decimals:
        # "880,400.00", "43.500,00", "12.5" and even "43.500.00" (same separator).
        decimal_part = tail
        tok = tok[:last_i]
    integer = re.sub(r"[.,]", "", tok)
    if not integer.isdigit():
        return None
    return Decimal(f"{integer}.{decimal_part}" if decimal_part else integer)


def find_amounts(text: str | None) -> list[Decimal]:
    """All numeric amounts in ``text`` (in order of appearance)."""
    if not text:
        return []
    out = []
    for tok in _merge_space_groups(_clean(text)):
        value = parse_number_token(tok)
        if value is not None:
            out.append(value)
    return out


def to_vnd(value: Decimal) -> int:
    """VND has no sub-unit in practice: round half-up to whole dong."""
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def parse_vnd(text: str | None) -> int | None:
    """Single amount from a text that is expected to hold one value.

    Returns ``None`` when there is no number or when the text holds several
    *different* numbers (ambiguous). Repeated identical numbers are accepted.
    """
    values = {to_vnd(v) for v in find_amounts(text)}
    return values.pop() if len(values) == 1 else None
