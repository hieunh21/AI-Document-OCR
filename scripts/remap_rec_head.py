"""Carry the stock PP-OCRv6 rec output layers over to a new character dictionary.

    python remap_rec_head.py --pretrained PP-OCRv6_medium_rec_pretrained.pdparams \
        --stock-dict PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt \
        --new-dict dict_vi.txt --out PP-OCRv6_medium_rec_vi_init.pdparams

With a new dictionary, PaddleOCR skips the dictionary-sized layers when loading the
pretrained weights and initialises them randomly. Round 1 of the fine-tune did that, and
the new head never matched the stock one on digits. It repeated or dropped characters:
"119.000" became "1119.000". This script builds those layers from the stock ones
instead:

* a character present in both dictionaries keeps its stock weights;
* a Vietnamese letter missing from the stock dictionary starts from its base letter
  without the tone mark ("ẩ" from "â", "ạ" from "a");
* anything else starts from the mean of the stock characters.

Index layout (PaddleOCR ``CTCLabelEncode`` / ``NRTRLabelEncode`` / ``Transformer``):

* CTC: ``[blank] + dict + [space]``;
* NRTR: ``[blank, <unk>, <s>, </s>] + dict + [space]`` plus one extra output.

Only numpy is needed: ``.pdparams`` is a pickled dict of arrays.
"""

from __future__ import annotations

import argparse
import pickle
import unicodedata
from pathlib import Path

import numpy as np

CTC_W, CTC_B = "head.ctc_head.fc.weight", "head.ctc_head.fc.bias"  # (C, N), (N,)
NRTR_EMB = "head.gtc_head.embedding.embedding.weight"  # (N, d)
NRTR_PRJ = "head.gtc_head.tgt_word_prj.weight"  # (d, N)
TONE_MARKS = frozenset("̣̀́̃̉")  # grave acute tilde hook dot


def read_dict(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.open(encoding="utf-8")]


def source_char(c: str, stock: dict[str, int]) -> str | None:
    """Stock character whose weights ``c`` starts from, or None."""
    nfd = unicodedata.normalize("NFD", c)
    no_tone = unicodedata.normalize("NFC", "".join(x for x in nfd if x not in TONE_MARKS))
    bare = "".join(x for x in nfd if unicodedata.category(x) != "Mn")
    return next((x for x in (c, no_tone, bare) if x in stock), None)


def remap(sd: dict, stock_dict: list[str], new_dict: list[str]) -> dict:
    stock = {c: i for i, c in enumerate(stock_dict)}
    n_ctc = len(stock_dict) + 2
    if sd[CTC_W].shape[1] != n_ctc:
        raise SystemExit(f"{CTC_W} has {sd[CTC_W].shape[1]} outputs, stock dict implies {n_ctc}")

    # Row of each new output in the stock layout (None -> mean of stock characters).
    src = [source_char(c, stock) for c in new_dict]
    ctc_rows = [0] + [None if s is None else 1 + stock[s] for s in src] + [n_ctc - 1]
    nrtr_rows = [0, 1, 2, 3] + [None if s is None else 4 + stock[s] for s in src]
    nrtr_rows += [4 + len(stock_dict), 5 + len(stock_dict)]  # space, extra output

    def take(a: np.ndarray, rows: list[int | None], axis: int, first: int) -> np.ndarray:
        mean = np.take(a, range(first, first + len(stock_dict)), axis=axis).mean(axis=axis)
        parts = [mean if r is None else np.take(a, r, axis=axis) for r in rows]
        return np.stack(parts, axis=axis).astype(a.dtype)

    out = dict(sd)
    out[CTC_W] = take(sd[CTC_W], ctc_rows, 1, 1)
    out[CTC_B] = take(sd[CTC_B], ctc_rows, 0, 1)
    out[NRTR_EMB] = take(sd[NRTR_EMB], nrtr_rows, 0, 4)
    out[NRTR_PRJ] = take(sd[NRTR_PRJ], nrtr_rows, 1, 4)

    same = sum(s == c for s, c in zip(src, new_dict, strict=True))
    based = [f"{c}<{s}" for s, c in zip(src, new_dict, strict=True) if s is not None and s != c]
    mean = [c for s, c in zip(src, new_dict, strict=True) if s is None]
    print(
        f"{len(new_dict)} chars: {same} copied, {len(based)} from a base letter, {len(mean)} mean"
    )
    print("  base:", " ".join(based))
    if mean:
        print("  mean:", " ".join(mean))
    for k in (CTC_W, CTC_B, NRTR_EMB, NRTR_PRJ):
        print(f"  {k}: {sd[k].shape} -> {out[k].shape}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", type=Path, required=True)
    ap.add_argument("--stock-dict", type=Path, required=True)
    ap.add_argument("--new-dict", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with args.pretrained.open("rb") as fh:
        sd = pickle.load(fh)
    out = remap(sd, read_dict(args.stock_dict), read_dict(args.new_dict))
    with args.out.open("wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
