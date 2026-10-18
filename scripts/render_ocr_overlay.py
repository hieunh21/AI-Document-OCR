"""Draw OCR spans (green) and annotated KIE regions (red) on the original image.

Used to check visually that OCR boxes are in original-image coordinates.

    python scripts/render_ocr_overlay.py --run v6_medium@gpu0 --split dev --n 4
Output: outputs/overlays/<run>/<img_id>.jpg (git-ignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.preprocessing.image import load_image  # noqa: E402
from evaluation.dataset_adapter import load_split  # noqa: E402
from evaluation.ocr_eval import load_results  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="smoke")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--id", action="append")
    args = ap.parse_args()

    results = load_results(args.split, args.run)
    records = [r for r in load_split(args.split) if r.img_id in results]
    if args.id:
        records = [r for r in records if r.img_id in set(args.id)]
    out = Path("outputs/overlays") / args.run
    out.mkdir(parents=True, exist_ok=True)
    for rec in records[: args.n]:
        im = load_image(rec.image_path)
        d = ImageDraw.Draw(im)
        for reg in rec.regions:
            d.rectangle(reg.bbox, outline=(220, 0, 0), width=4)
        for s in results[rec.img_id].spans:
            if s.polygon:
                d.polygon([tuple(p) for p in s.polygon], outline=(0, 170, 0), width=2)
            else:
                d.rectangle(s.bbox, outline=(0, 170, 0), width=2)
        path = out / rec.img_id
        im.save(path, quality=88)
        print(path)


if __name__ == "__main__":
    main()
