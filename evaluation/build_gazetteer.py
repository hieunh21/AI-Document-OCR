"""Build configs/kie_gazetteer.json from the *train* split ground truth only.

    python -m evaluation.build_gazetteer [--min-seller-count 2]

Sellers seen at least ``--min-seller-count`` times (after accent-free folding)
become entries; the canonical spelling is the most frequent annotated variant.
Store addresses are kept per seller (any count): they are the store master data.
Dev/test labels are never read.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from ai.kie.gazetteer import GAZETTEER_PATH
from ai.normalization.text import fold_noacc
from evaluation.dataset_adapter import load_split


def build(min_seller_count: int = 2) -> dict:
    records = load_split("train")
    seller_variants: dict[str, Counter] = defaultdict(Counter)
    address_variants: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in records:
        f = r.gt.fields
        if f.seller:
            seller_variants[fold_noacc(f.seller)][f.seller] += 1
        if f.seller and f.seller_address:
            address_variants[(fold_noacc(f.seller), fold_noacc(f.seller_address))][
                f.seller_address
            ] += 1
    sellers = [
        {"key": k, "name": v.most_common(1)[0][0], "count": sum(v.values())}
        for k, v in seller_variants.items()
        # < 5 chars are annotation fragments ("PHẢ", "CHẾ"), not store names
        if sum(v.values()) >= min_seller_count and len(k) >= 5
    ]
    kept = {s["key"] for s in sellers}
    addresses = [
        {"seller_key": sk, "key": ak, "address": v.most_common(1)[0][0], "count": sum(v.values())}
        for (sk, ak), v in address_variants.items()
        if sk in kept
    ]
    sellers.sort(key=lambda s: -s["count"])
    addresses.sort(key=lambda a: (a["seller_key"], -a["count"]))
    return {
        "built_from": f"train split ({len(records)} images), ground truth only",
        "min_seller_count": min_seller_count,
        "sellers": sellers,
        "addresses": addresses,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seller-count", type=int, default=2)
    args = ap.parse_args()
    data = build(args.min_seller_count)
    GAZETTEER_PATH.parent.mkdir(parents=True, exist_ok=True)
    GAZETTEER_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"{GAZETTEER_PATH}: {len(data['sellers'])} sellers, {len(data['addresses'])} addresses")


if __name__ == "__main__":
    main()
