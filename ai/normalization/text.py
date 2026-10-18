from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(s: str | None) -> str | None:
    """Unicode NFC + collapse whitespace. Keeps case, accents and punctuation.

    Returns ``None`` for ``None`` or whitespace-only input.
    """
    if s is None:
        return None
    s = unicodedata.normalize("NFC", s)
    s = _WS.sub(" ", s).strip()
    return s or None


def fold_for_match(s: str | None) -> str:
    """Lenient form used for *matching* strings in evaluation (not for display).

    NFC, lower-case, punctuation -> space, collapsed whitespace. Vietnamese
    diacritics are kept on purpose: dropping them would merge different words.
    """
    s = normalize_text(s) or ""
    s = _NON_WORD.sub(" ", s.casefold())
    return _WS.sub(" ", s).strip()


def strip_accents(s: str) -> str:
    """Remove Vietnamese diacritics (``đ`` -> ``d``); keeps case."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def fold_noacc(s: str | None) -> str:
    """``fold_for_match`` without diacritics: robust key for keyword matching on OCR text."""
    return strip_accents(fold_for_match(s))
