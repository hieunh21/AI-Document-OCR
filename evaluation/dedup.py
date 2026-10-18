"""Exact and near-duplicate detection used to keep copies in one split.

Three independent signals are unioned into groups:

1. identical file bytes (SHA-256);
2. perceptual difference hash (dHash 32x32 = 1024 bit) within ``max_hamming``
   bits — the same photo re-encoded, resized or screenshotted;
3. the same transaction in the labels: identical seller + timestamp (with time)
   + total — the same receipt photographed twice from a different angle.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Hashable, Iterable
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Calibrated on MC-OCR (see docs/dataset_eda.md): re-encodes/screenshots of one
# photo are <= 110 bits apart, different receipts of one store >= 223. A 64-bit
# dHash is useless here: different receipts collide at 1-2 bits.
DHASH_SIZE = 32
DEFAULT_MAX_HAMMING = 150
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path, size: int = DHASH_SIZE) -> np.ndarray:
    """Difference hash as packed bits (``size*size/8`` bytes)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.LANCZOS)
        px = np.asarray(im, dtype=np.int16)
    return np.packbits((px[:, 1:] > px[:, :-1]).flatten())


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(_POPCOUNT[np.bitwise_xor(a, b)].sum())


class UnionFind:
    def __init__(self, items: Iterable[Hashable]) -> None:
        self.parent = {i: i for i in items}

    def find(self, x: Hashable) -> Hashable:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: Hashable, b: Hashable) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # deterministic root

    def groups(self) -> dict[Hashable, list[Hashable]]:
        out: dict[Hashable, list[Hashable]] = defaultdict(list)
        for i in self.parent:
            out[self.find(i)].append(i)
        return dict(out)


def near_duplicate_pairs(
    hashes: dict[str, np.ndarray], max_hamming: int
) -> list[tuple[str, str, int]]:
    ids = sorted(hashes)
    arr = np.stack([hashes[i] for i in ids])
    pairs = []
    for k, a in enumerate(ids[:-1]):
        d = _POPCOUNT[np.bitwise_xor(arr[k + 1 :], arr[k])].sum(axis=1)
        for j in np.nonzero(d <= max_hamming)[0]:
            pairs.append((a, ids[k + 1 + j], int(d[j])))
    return pairs


def duplicate_groups(
    sha: dict[str, str],
    dh: dict[str, np.ndarray],
    transaction_keys: dict[str, tuple | None],
    max_hamming: int = DEFAULT_MAX_HAMMING,
) -> tuple[dict[str, str], dict[str, int]]:
    """Return ``(img_id -> group_id, reason -> number of linking pairs)``.

    ``group_id`` is the smallest img_id of the group, so it is stable across runs.
    """
    uf = UnionFind(sha)
    reasons = {"sha256": 0, "dhash": 0, "transaction": 0}

    by_sha: dict[str, list[str]] = defaultdict(list)
    for i, h in sha.items():
        by_sha[h].append(i)
    for ids in by_sha.values():
        for other in ids[1:]:
            uf.union(ids[0], other)
            reasons["sha256"] += 1

    for a, b, _ in near_duplicate_pairs(dh, max_hamming):
        if sha[a] != sha[b]:
            reasons["dhash"] += 1
        uf.union(a, b)

    by_tx: dict[tuple, list[str]] = defaultdict(list)
    for i, key in transaction_keys.items():
        if key is not None:
            by_tx[key].append(i)
    for ids in by_tx.values():
        for other in ids[1:]:
            if uf.find(ids[0]) != uf.find(other):
                reasons["transaction"] += 1
            uf.union(ids[0], other)

    group_of = {}
    for members in uf.groups().values():
        gid = min(members)
        for m in members:
            group_of[m] = gid
    return group_of, reasons
