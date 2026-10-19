"""Known sellers and store addresses ("vendor master data").

Built from the *train* split only (``python -m evaluation.build_gazetteer``) and
stored in ``configs/kie_gazetteer.json``. Matching is fuzzy on accent-free folded
text, so OCR accent errors ("PHÓ MÒ") still map to the canonical name ("PHỐ MỎ").
A real deployment would load this from the customer's supplier list instead.
"""

from __future__ import annotations

import json
from pathlib import Path

from rapidfuzz import fuzz, process

from ai.normalization.text import fold_noacc
from ai.schemas import OCRSpan

GAZETTEER_PATH = Path("configs/kie_gazetteer.json")


class Gazetteer:
    def __init__(
        self,
        sellers: list[dict],
        addresses: list[dict],
        seller_min: float = 85.0,
        address_min: float = 85.0,
    ) -> None:
        self.sellers = sellers
        self.addresses = addresses
        self.seller_min = seller_min
        self.address_min = address_min
        self._seller_keys = [s["key"] for s in sellers]

    @classmethod
    def from_dict(cls, data: dict, **kw) -> Gazetteer:
        return cls(data["sellers"], data["addresses"], **kw)

    @classmethod
    def load(cls, path: Path = GAZETTEER_PATH, **kw) -> Gazetteer | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["sellers"], data["addresses"], **kw)

    def match_seller(self, spans: list[OCRSpan]) -> tuple[str, list[OCRSpan]] | None:
        """Best seller among single spans and pairs of consecutive spans."""
        cands: list[tuple[str, list[OCRSpan]]] = [(fold_noacc(s.text), [s]) for s in spans]
        cands += [
            (fold_noacc(f"{a.text} {b.text}"), [a, b])
            for a, b in zip(spans, spans[1:], strict=False)
        ]
        best = None
        for key, used in cands:
            if len(key) < 3:
                continue
            hit = process.extractOne(key, self._seller_keys, scorer=fuzz.ratio)
            if hit is None:
                continue
            _, score, idx = hit
            need = 95.0 if len(key) < 6 else self.seller_min
            # prefer the longest matching name: "SCTC CÔ THỎ … CẨM PHẢ" over a stray "PHẢ"
            entry_len = len(self._seller_keys[idx])
            if score >= need and (best is None or (entry_len, score) > best[:2]):
                best = (entry_len, score, idx, used)
        if best is None:
            return None
        return self.sellers[best[2]]["name"], best[3]

    def canonical_seller(self, text: str) -> str | None:
        """Canonical name for a seller string read by another extractor (e.g. an LLM)."""
        key = fold_noacc(text)
        if len(key) < 3:
            return None
        hit = process.extractOne(key, self._seller_keys, scorer=fuzz.ratio)
        need = 95.0 if len(key) < 6 else self.seller_min
        return self.sellers[hit[2]]["name"] if hit and hit[1] >= need else None

    def match_address(self, raw: str, seller: str | None) -> str | None:
        key = fold_noacc(raw)
        if not key:
            return None
        pool = self.addresses
        if seller:
            skey = fold_noacc(seller)
            own = [a for a in pool if a["seller_key"] == skey]
            pool = own or pool
        hit = process.extractOne(key, [a["key"] for a in pool], scorer=fuzz.ratio)
        if hit is None or hit[1] < self.address_min:
            return None
        return pool[hit[2]]["address"]
