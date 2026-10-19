"""Vietnamese administrative names in free text: detection and accent repair.

OCR often keeps the letters of a place name but loses its diacritics ("Câm Phå",
"Gò Vãp"). The official directory (``configs/vn_admin_units.json``, built by
``scripts/build_admin_units.py``) gives the correctly accented names. A window of
2–5 words whose accent-free form equals a directory name is rewritten with that
name's accents, keeping the original letter case and separators.

Only accents change, never letters: the accent-free text is identical before and
after. Two safeguards:

* **hierarchy** — when the text names a province or district, a name is only
  corrected to a unit inside it. Village names (thôn) are not in the directory and
  often equal a ward elsewhere ("Thôn Phú Thụy, … Gia Lâm" must not become the
  "Phú Thủy" ward of Quảng Bình);
* **ambiguity** — among the remaining spellings the one closest to what OCR read
  wins; a tie is left untouched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from ai.normalization.text import strip_accents

UNITS_PATH = Path("configs/vn_admin_units.json")
_WORD = re.compile(r"\w+")
MIN_WORDS, MAX_WORDS = 2, 5


def _key(words: list[str]) -> str:
    return " ".join(strip_accents(w).casefold() for w in words)


@dataclass(frozen=True)
class AdminName:
    name: str
    level: int  # 1 province, 2 district, 3 ward
    districts: frozenset[str]  # accent-free keys of its district's names (wards)
    provinces: frozenset[str]  # accent-free keys of its province's names

    def inside(self, mentioned: set[str]) -> bool:
        return bool(self.provinces & mentioned or self.districts & mentioned)


def _match_case(target: str, model: str) -> str:
    if model.isupper() and len(model) > 1:
        return target.upper()
    if model[:1].isupper():
        return target[:1].upper() + target[1:].lower()
    return target.lower()


class AdminUnits:
    def __init__(self, names: list[AdminName]) -> None:
        self.index: dict[str, list[AdminName]] = {}
        for n in names:
            words = _WORD.findall(n.name)
            if MIN_WORDS <= len(words) <= MAX_WORDS:
                self.index.setdefault(_key(words), []).append(n)

    @classmethod
    @lru_cache(maxsize=2)
    def load(cls, path: Path = UNITS_PATH) -> AdminUnits | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))

        def keys(names: list[str]) -> frozenset[str]:
            return frozenset(_key(_WORD.findall(x)) for x in names)

        return cls([AdminName(n, lvl, keys(d), keys(p)) for n, lvl, d, p in data["units"]])

    def find(self, text: str) -> list[tuple[int, int, list[AdminName]]]:
        """Non-overlapping ``(first_word, end_word, candidates)``, longest windows first."""
        words = _WORD.findall(text)
        taken = [False] * len(words)
        hits = []
        for n in range(min(MAX_WORDS, len(words)), MIN_WORDS - 1, -1):
            for i in range(len(words) - n + 1):
                if any(taken[i : i + n]):
                    continue
                cands = self.index.get(_key(words[i : i + n]))
                if cands:
                    hits.append((i, i + n, cands))
                    taken[i : i + n] = [True] * n
        return sorted(hits)

    def contains_name(self, text: str) -> bool:
        return bool(self.find(text))

    def correct(self, text: str) -> str:
        words = _WORD.findall(text)
        hits = self.find(text)
        replace: dict[int, str] = {}  # word index -> corrected word
        for i, j, cands in hits:
            # provinces / districts named elsewhere in the text
            mentioned = {
                _key(words[a:b])
                for a, b, cs in hits
                if (a, b) != (i, j) and any(c.level <= 2 for c in cs)
            }
            if mentioned:
                cands = [c for c in cands if c.level == 1 or c.inside(mentioned)]
            seen = " ".join(words[i:j])
            spellings = sorted({c.name for c in cands})
            if not spellings or any(seen.casefold() == s.casefold() for s in spellings):
                continue  # nothing consistent, or already correct

            dist = {n: Levenshtein.distance(seen.casefold(), n.casefold()) for n in spellings}
            ranked = sorted(spellings, key=dist.__getitem__)
            if len(ranked) > 1 and dist[ranked[0]] == dist[ranked[1]]:
                continue  # ambiguous: leave as read
            for k, w in enumerate(_WORD.findall(ranked[0])):
                replace[i + k] = _match_case(w, words[i + k])
        counter = iter(range(len(words)))

        def sub(m: re.Match) -> str:
            return replace.get(next(counter), m.group())

        return _WORD.sub(sub, text)
