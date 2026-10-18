"""Create reproducible internal train/dev/test splits of the labelled MC-OCR data.

Only ``mcocr_train_df.csv`` (1,155 images) has labels, so the internal
held-out test comes from it. The public ``val_images`` have no labels and are
not part of any split.

Leakage control: images are grouped (exact copy, near-duplicate photo, or the
same transaction re-photographed — see :mod:`evaluation.dedup`) and a group
never spans two splits. Groups are stratified by seller so each split keeps the
same mix of large chains vs. long-tail shops.

Note: this measures performance on receipts from *known* stores (the realistic
setting for the app). Generalisation to *unseen* stores is not what the test
split measures; ``stratum`` in the manifests allows that analysis later.

CLI::

    python -m evaluation.splits                # writes data/splits/*.json
    python -m evaluation.splits --check        # recompute and compare, write nothing
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ai.normalization.text import fold_for_match
from evaluation.dataset_adapter import (
    DEFAULT_ROOT,
    TRAIN_CSV,
    TRAIN_IMAGES,
    McocrRecord,
    list_unlabelled_val,
    load_mcocr,
)
from evaluation.dedup import DEFAULT_MAX_HAMMING, dhash, duplicate_groups, sha256_file

SPLITS_DIR = Path("data/splits")
DEFAULT_SEED = 2026
DEFAULT_RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}
MIN_STRATUM_SIZE = 15  # sellers with fewer images go to the "other" stratum
SMOKE_SIZE = 15
SPLITTER_VERSION = "splits-v1"


def transaction_key(r: McocrRecord) -> tuple | None:
    """Same seller + same minute + same total = same receipt (needs all three)."""
    f = r.gt.fields
    if not (f.seller and f.timestamp and "T" in f.timestamp and f.total_cost):
        return None
    return (fold_for_match(f.seller), f.timestamp[:16], f.total_cost)


def seller_strata(records: list[McocrRecord], min_size: int) -> dict[str, str]:
    key = {r.img_id: fold_for_match(r.gt.fields.seller) or "" for r in records}
    counts = Counter(key.values())
    return {
        i: ("no_seller" if not k else k if counts[k] >= min_size else "other")
        for i, k in key.items()
    }


def anonymize_strata(strata: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """Seller names are label content: committed manifests use ``seller_NN`` ids
    (by size, largest first). Returns ``(name -> id, id -> name)``."""
    counts = Counter(strata.values())
    named = sorted(
        (n for n in counts if n not in ("other", "no_seller")), key=lambda n: (-counts[n], n)
    )
    to_id = {n: f"seller_{k:02d}" for k, n in enumerate(named, 1)} | {
        "other": "other",
        "no_seller": "no_seller",
    }
    return to_id, {v: k for k, v in to_id.items()}


def assign_groups(
    groups: dict[str, list[str]],
    stratum_of_group: dict[str, str],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Greedy, deterministic: per stratum, place the largest groups first into the
    split that is furthest below its target size."""
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for gid in sorted(groups):
        by_stratum[stratum_of_group[gid]].append(gid)

    split_of_group: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        gids = by_stratum[stratum]
        random.Random(f"{seed}:{stratum}").shuffle(gids)
        gids.sort(key=lambda g: -len(groups[g]))  # stable: ties keep shuffled order
        total = sum(len(groups[g]) for g in gids)
        filled = dict.fromkeys(ratios, 0)
        for gid in gids:
            name = max(ratios, key=lambda s: (ratios[s] * total - filled[s], ratios[s]))
            split_of_group[gid] = name
            filled[name] += len(groups[gid])
    return split_of_group


def pick_smoke(dev: list[McocrRecord], strata: dict[str, str], seed: int, n: int) -> list[str]:
    """Small, diverse subset of dev for quick local runs: one per stratum, the
    lowest-quality images, then random fill."""
    rng = random.Random(f"{seed}:smoke")
    chosen: list[str] = []
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for r in sorted(dev, key=lambda r: r.img_id):
        by_stratum[strata[r.img_id]].append(r.img_id)
    for s in sorted(by_stratum):
        chosen.append(rng.choice(by_stratum[s]))
    for r in sorted(dev, key=lambda r: r.quality)[:3]:
        if r.img_id not in chosen:
            chosen.append(r.img_id)
    rest = sorted(r.img_id for r in dev if r.img_id not in chosen)
    rng.shuffle(rest)
    chosen += rest[: max(0, n - len(chosen))]
    return chosen[:n]


def build(root: Path, seed: int, ratios: dict[str, float], max_hamming: int) -> dict[str, dict]:
    records = load_mcocr(root)
    by_id = {r.img_id: r for r in records}
    val_paths = list_unlabelled_val(root)

    paths = {r.img_id: r.image_path for r in records} | {p.name: p for p in val_paths}
    sha = {i: sha256_file(p) for i, p in paths.items()}
    dh = {i: dhash(p) for i, p in paths.items()}
    tx = {r.img_id: transaction_key(r) for r in records}

    # Group over train+val so cross-set duplicates are visible, then split train only.
    group_of, link_reasons = duplicate_groups(sha, dh, tx, max_hamming)
    members: dict[str, list[str]] = defaultdict(list)
    for i in sorted(by_id):
        members[group_of[i]].append(i)
    val_ids = {p.name for p in val_paths}
    val_linked = (
        sorted(i for i in by_id if any(group_of.get(v) == group_of[i] for v in val_ids))
        if val_ids
        else []
    )

    strata = seller_strata(records, MIN_STRATUM_SIZE)
    stratum_of_group = {
        gid: sorted(Counter(strata[i] for i in ids).most_common(), key=lambda x: (-x[1], x[0]))[0][
            0
        ]
        for gid, ids in members.items()
    }
    split_of_group = assign_groups(dict(members), stratum_of_group, ratios, seed)
    stratum_id, _ = anonymize_strata(strata)

    common = {
        "version": SPLITTER_VERSION,
        "seed": seed,
        "ratios": ratios,
        "source_csv": TRAIN_CSV,
        "source_csv_sha256": sha256_file(root / TRAIN_CSV),
        "dedup": {
            "dhash_size": 32,
            "max_hamming": max_hamming,
            "signals": ["sha256", "dhash", "transaction"],
        },
        "paths_relative_to": "VIETDOCAI_MCOCR_DIR (default data/raw/mcocr)",
    }

    def item(i: str) -> dict:
        return {
            "img_id": i,
            "path": f"{TRAIN_IMAGES}/{i}",
            "sha256": sha[i],
            "group_id": group_of[i],
            "stratum": stratum_id[strata[i]],
        }

    manifests: dict[str, dict] = {}
    for name in ratios:
        ids = sorted(i for i in by_id if split_of_group[group_of[i]] == name)
        manifests[name] = {
            "name": name,
            **common,
            "num_images": len(ids),
            "items": [item(i) for i in ids],
        }

    dev_records = [by_id[it["img_id"]] for it in manifests["dev"]["items"]]
    smoke = pick_smoke(dev_records, strata, seed, SMOKE_SIZE)
    manifests["smoke"] = {
        "name": "smoke",
        **common,
        "description": (
            f"{SMOKE_SIZE} dev images for quick local runs (subset of dev, never of test)"
        ),
        "num_images": len(smoke),
        "items": [item(i) for i in smoke],
    }

    multi = {g: ids for g, ids in members.items() if len(ids) > 1}
    manifests["summary"] = {
        **common,
        "num_labelled": len(records),
        "num_unlabelled_val": len(val_paths),
        "groups": {
            "num_groups": len(members),
            "num_multi_image_groups": len(multi),
            "images_in_multi_image_groups": sum(len(v) for v in multi.values()),
            "largest_group": max((len(v) for v in members.values()), default=0),
            "linking_pairs_by_signal": link_reasons,
        },
        "train_images_duplicated_in_val": val_linked,
        "split_sizes": {n: manifests[n]["num_images"] for n in ratios},
        "strata": {
            s: {n: sum(1 for it in manifests[n]["items"] if it["stratum"] == s) for n in ratios}
            for s in sorted({stratum_id[v] for v in strata.values()})
        },
    }
    return manifests


def _write(manifests: dict[str, dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, m in manifests.items():
        (out / f"{name}.json").write_text(
            json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=SPLITS_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--max-hamming", type=int, default=DEFAULT_MAX_HAMMING)
    ap.add_argument("--check", action="store_true", help="verify existing manifests are reproduced")
    args = ap.parse_args()

    manifests = build(args.root, args.seed, DEFAULT_RATIOS, args.max_hamming)
    if args.check:
        bad = []
        for name, m in manifests.items():
            path = args.out / f"{name}.json"
            if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != m:
                bad.append(name)
        print("reproducible" if not bad else f"MISMATCH in: {bad}")
        return 1 if bad else 0

    _write(manifests, args.out)
    s = manifests["summary"]
    print(
        json.dumps(
            {k: s[k] for k in ("split_sizes", "groups", "strata")}, ensure_ascii=False, indent=1
        )
    )
    print(f"train images duplicated in val: {len(s['train_images_duplicated_in_val'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
