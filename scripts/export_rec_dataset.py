"""Export text-line crops of MC-OCR for fine-tuning a recognizer (PaddleOCR format).

    python scripts/export_rec_dataset.py [--ocr-run "v6_medium+orient@gpu0"] [--zip]

For every annotated region of the **train** and **dev** splits (never test), the
detector box of the cached OCR run that matches it (IoU > 0.7) is cropped exactly as
at inference time: the photo is turned upright with the orientation the OCR used, then
the line is rectified with ``ai.ocr.hybrid.crop_line``. The label is the annotated text.

Noisy pairs are dropped: a label whose accent-free form is far from what the current
OCR read on that box (CER > ``MAX_NOACC_CER``) is usually a misaligned box or a wrong
label, and a recognizer must not learn it.

Output (``outputs/rec_dataset/``, git-ignored)::

    images/train/*.jpg, images/dev/*.jpg
    train_list.txt, val_list.txt     "images/train/<name>.jpg\\t<label>" (PaddleOCR / VietOCR)
    dict_vi.txt                      one character per line (no space: use_space_char)
    corpus_train.txt                 train label texts, for synthetic lines
    admin_names.txt                  official place names (configs/vn_admin_units.json)
    synth_rec_lines.py               generator run on Colab (copy of scripts/)
    remap_rec_head.py                builds the new output layers from the stock ones (copy)
    manifest.json                    counts, sources, filters
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ai.ocr.hybrid import crop_line  # noqa: E402
from ai.preprocessing.image import load_image  # noqa: E402
from evaluation.dataset_adapter import GT_VERSION, load_split  # noqa: E402
from evaluation.metrics import levenshtein  # noqa: E402
from evaluation.ocr_eval import load_results  # noqa: E402

OUT = Path("outputs/rec_dataset")
MIN_IOU = 0.7
MAX_NOACC_CER = 0.5
MAX_LEN = 48  # matches max_text_length of the training config
VI_LETTERS = (
    "aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệfghiìíỉĩịjklmnoòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvwxyỳýỷỹỵz"
)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union > 0 else 0.0


def rotate_points(pts: np.ndarray, w: float, h: float, rotation: int) -> np.ndarray:
    """Original-image points -> points in the page rotated CCW (PIL ``rotate``)."""
    x, y = pts[:, 0], pts[:, 1]
    if rotation == 90:
        return np.stack([y, w - x], 1)
    if rotation == 180:
        return np.stack([w - x, h - y], 1)
    if rotation == 270:
        return np.stack([h - y, x], 1)
    return pts


def noacc(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return " ".join(s.casefold().split())


def export_split(split: str, ocr_run: str, stats: Counter) -> list[tuple[str, str]]:
    ocr = load_results(split, ocr_run)
    out_dir = OUT / "images" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in load_split(split):
        res = ocr.get(rec.img_id)
        if res is None:
            stats[f"{split}_no_ocr"] += 1
            continue
        pairs = []
        for k, region in enumerate(rec.regions):
            text = unicodedata.normalize("NFC", " ".join(region.text.split()))
            if not text:
                continue
            stats[f"{split}_regions"] += 1
            best = max(res.spans, key=lambda s: iou(s.bbox, region.bbox), default=None)
            if best is None or iou(best.bbox, region.bbox) <= MIN_IOU or not best.polygon:
                stats[f"{split}_no_matching_box"] += 1
                continue
            if len(text) > MAX_LEN:
                stats[f"{split}_too_long"] += 1
                continue
            a, b = noacc(text), noacc(best.text)
            if a and levenshtein(a, b) / len(a) > MAX_NOACC_CER:
                stats[f"{split}_label_disagrees"] += 1
                continue
            pairs.append((k, text, np.asarray(best.polygon, dtype=np.float32)))
        if not pairs:
            continue
        image = load_image(rec.image_path).convert("RGB")
        w, h = image.size
        if res.rotation:
            image = image.rotate(res.rotation, expand=True)
        rgb = np.asarray(image)
        stem = rec.img_id.removesuffix(".jpg")
        for k, text, poly in pairs:
            crop = crop_line(rgb, rotate_points(poly, w, h, res.rotation))
            name = f"{stem}_{k:02d}.jpg"
            Image.fromarray(np.ascontiguousarray(crop)).save(out_dir / name, quality=95)
            rows.append((f"images/{split}/{name}", text))
            stats[f"{split}_kept"] += 1
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-run", default="v6_medium+orient@gpu0")
    ap.add_argument("--zip", action="store_true", help="also write outputs/rec_dataset.zip")
    args = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    stats: Counter = Counter()
    train = export_split("train", args.ocr_run, stats)
    dev = export_split("dev", args.ocr_run, stats)

    def write(name: str, rows: list[tuple[str, str]]) -> None:
        with open(OUT / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(f"{p}\t{t}\n" for p, t in rows)

    write("train_list.txt", train)
    write("val_list.txt", dev)

    # Dictionary: printable ASCII + the full Vietnamese alphabet + anything in train labels.
    chars = {chr(c) for c in range(33, 127)}
    chars |= set(VI_LETTERS) | set(VI_LETTERS.upper()) | set("₫–—…“”‘’°")
    chars |= {c for _, t in train for c in t}
    chars.discard(" ")
    (OUT / "dict_vi.txt").write_text(
        "\n".join(sorted(chars, key=ord)) + "\n", encoding="utf-8", newline="\n"
    )
    dev_unknown = Counter(c for _, t in dev for c in t if c != " " and c not in chars)

    # Text sources for synthetic lines: train labels and official place names only.
    corpus = sorted({t for _, t in train})
    (OUT / "corpus_train.txt").write_text("\n".join(corpus) + "\n", encoding="utf-8")
    units = json.loads(Path("configs/vn_admin_units.json").read_text(encoding="utf-8"))["units"]
    (OUT / "admin_names.txt").write_text(
        "\n".join(sorted({u[0] for u in units})) + "\n", encoding="utf-8"
    )
    for script in ("synth_rec_lines.py", "remap_rec_head.py"):
        shutil.copy(Path(__file__).with_name(script), OUT / script)

    manifest = {
        "ocr_run": args.ocr_run,
        "ground_truth": GT_VERSION,
        "splits": {"train": "train_list.txt", "dev (validation)": "val_list.txt"},
        "test_split": "not exported",
        "filters": {"min_iou": MIN_IOU, "max_noacc_cer": MAX_NOACC_CER, "max_len": MAX_LEN},
        "dict_size": len(chars),
        "dev_chars_not_in_dict": dict(dev_unknown),
        "stats": dict(sorted(stats.items())),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.zip:
        path = shutil.make_archive(str(OUT), "zip", root_dir=OUT.parent, base_dir=OUT.name)
        print(f"{path}: {Path(path).stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
