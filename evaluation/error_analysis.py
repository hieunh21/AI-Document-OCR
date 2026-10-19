"""Attribute every KIE field error to a pipeline stage (dev only).

For each wrong field (``evaluation.metrics.score``), the annotated GT regions, the
cached OCR spans and the prediction's evidence decide the *first* stage that failed:

=====================  ==========================================================
``label_absent``       GT is null (no region annotated) but the model found a value
``ocr_not_detected``   no OCR span overlaps the GT regions
``ocr_misread``        OCR text of the GT regions does not contain the GT value
                       (text: folded substring CER > 10 %; timestamp: parsed value
                       differs; total: amount not among the OCR amounts)
``kie_missed``         OCR read it, but the model returned null
``kie_wrong_region``   OCR read it, but the evidence does not overlap the GT regions
``kie_partial``        evidence overlaps the GT regions but the text differs
                       (extra/missing words or lines) — text fields
``kie_value``          evidence overlaps the GT regions but the normalized value
                       differs (parsing, precision, digit choice) — timestamp/total
=====================  ==========================================================

Each error also carries image conditions (official quality < 0.5, rotated photo,
store seen in train) for cross-tabulation, and ``--render`` draws crops (GT green,
prediction red) for manual review under ``outputs/error_analysis/crops``.

CLI::

    python -m evaluation.error_analysis --split dev --model rules llm [--render 40]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from ai.normalization.datetime_vi import parse_timestamp
from ai.normalization.money import find_amounts, to_vnd
from ai.normalization.text import fold_noacc
from ai.schemas import FIELD_NAMES, ReceiptExtraction
from evaluation.dataset_adapter import McocrRecord, load_split
from evaluation.metrics import TEXT_FIELDS, score, text_key, ts_match
from evaluation.ocr_eval import load_results, overlap_ratio, spans_for_region
from evaluation.ocr_eval import substring_edit_distance as sub_ed
from evaluation.run_benchmark import PIPELINES, ocr_run_name, train_sellers

OUT = Path("outputs/error_analysis")
REPORTS = Path("evaluation/reports")
CATEGORIES = (
    "label_absent",
    "ocr_not_detected",
    "ocr_misread",
    "kie_missed",
    "kie_wrong_region",
    "kie_partial",
    "kie_value",
)
MAX_TEXT_CER = 0.10


def ocr_reads_gt(field: str, gt, ocr_text: str) -> bool:
    if field in TEXT_FIELDS:
        g = text_key(gt)
        return sub_ed(g, text_key(ocr_text)) <= MAX_TEXT_CER * len(g)
    if field == "timestamp":
        parsed = parse_timestamp(ocr_text)
        return bool(parsed and parsed.iso) and ts_match(parsed.iso, gt)
    return gt in {to_vnd(a) for a in find_amounts(ocr_text)}


def categorize(rec: McocrRecord, ocr, pred: ReceiptExtraction, field: str) -> str:
    gt, p = getattr(rec.gt.fields, field), getattr(pred.fields, field)
    if gt is None:
        return "label_absent"
    regions = rec.regions_of(field)
    hit = []
    for region in regions:
        hit += [s for s in spans_for_region(region, ocr.spans) if s not in hit]
    if not hit:
        return "ocr_not_detected"
    if not ocr_reads_gt(field, gt, " ".join(s.text for s in hit)):
        return "ocr_misread"
    if p is None:
        return "kie_missed"
    ev = pred.evidence.get(field)
    boxes = [r.bbox for r in ev.regions] if ev else []
    if not any(overlap_ratio(b, g.bbox) >= 0.5 for b in boxes for g in regions):
        return "kie_wrong_region"
    return "kie_partial" if field in TEXT_FIELDS else "kie_value"


@lru_cache(maxsize=8)
def load_preds(split: str, kie: str, ocr_run: str) -> dict[str, ReceiptExtraction]:
    path = Path("outputs/kie") / f"{kie}@{ocr_run}".replace(":", "_") / f"{split}.jsonl"
    preds = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        img_id = d.pop("img_id")  # pop before validating: the schema forbids extra keys
        preds[img_id] = ReceiptExtraction.model_validate(d)
    return preds


def analyse(split: str, name: str, device: str = "gpu:0") -> list[dict]:
    p = PIPELINES[name]
    ocr_run = ocr_run_name(p.ocr, device)
    records = load_split(split)
    by_id = {r.img_id: r for r in records}
    ocr = load_results(split, ocr_run)
    preds = load_preds(split, p.kie, ocr_run)
    rows = []
    for e in score(records, preds)["errors"]:
        rec = by_id[e["img_id"]]
        seller = rec.gt.fields.seller
        rows.append(
            e
            | {
                "pipeline": name,
                "category": categorize(rec, ocr[rec.img_id], preds[rec.img_id], e["field"]),
                "poor_quality": rec.quality < 0.5,
                "rotated": bool(ocr[rec.img_id].rotation),
                "store_seen": None if not seller else fold_noacc(seller) in train_sellers(),
                "quality": round(rec.quality, 3),
            }
        )
    return rows


def render(rows: list[dict], split: str, n: int, seed: int = 2026) -> Path:
    """Crops around GT (green) and predicted evidence (red) for a stratified sample."""
    from PIL import ImageDraw

    from ai.preprocessing.image import load_image

    records = {r.img_id: r for r in load_split(split)}
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    sample: list[dict] = []
    while len(sample) < n and any(by_cat.values()):
        for cat in CATEGORIES:
            if by_cat.get(cat) and len(sample) < n:
                sample.append(by_cat[cat].pop(rng.randrange(len(by_cat[cat]))))
    out = OUT / "crops"
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for k, r in enumerate(sample):
        rec = records[r["img_id"]]
        img = load_image(rec.image_path).convert("RGB")
        preds = load_preds(split, PIPELINES[r["pipeline"]].kie, r["ocr_run"])
        ev = preds[r["img_id"]].evidence.get(r["field"])
        gt_boxes = [g.bbox for g in rec.regions_of(r["field"])]
        pr_boxes = [e.bbox for e in ev.regions] if ev else []
        draw = ImageDraw.Draw(img)
        for b in gt_boxes:
            draw.rectangle(b, outline=(0, 200, 0), width=4)
        for b in pr_boxes:
            draw.rectangle(b, outline=(230, 0, 0), width=3)
        boxes = gt_boxes + pr_boxes
        if boxes:
            x0 = max(0, min(b[0] for b in boxes) - 80)
            y0 = max(0, min(b[1] for b in boxes) - 80)
            x1 = min(img.width, max(b[2] for b in boxes) + 80)
            y1 = min(img.height, max(b[3] for b in boxes) + 80)
            img = img.crop((x0, y0, x1, y1))
        img.thumbnail((1000, 1000))
        name = f"{k:02d}_{r['category']}_{r['field']}.jpg"
        img.save(out / name, quality=85)
        index.append(r | {"crop": name})
    with open(out / "index.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=1, default=str)
    return out


def write_report(split: str, all_rows: dict[str, list[dict]]) -> Path:
    lines = [
        f"# Error breakdown — split `{split}`\n",
        "> Generated by `python -m evaluation.error_analysis`. Categories: module docstring "
        "of `evaluation/error_analysis.py` (first stage that failed). Interpretation and "
        "examples: `docs/error_analysis.md`.\n",
    ]
    for name, rows in all_rows.items():
        c = Counter((r["field"], r["category"]) for r in rows)
        lines += [
            f"## `{name}` — {len(rows)} wrong field values",
            "",
            "| Category | " + " | ".join(FIELD_NAMES) + " | Total | Share |",
            "|---|" + "---:|" * (len(FIELD_NAMES) + 2),
        ]
        for cat in CATEGORIES:
            tot = sum(c[(f, cat)] for f in FIELD_NAMES)
            if not tot:
                continue
            lines.append(
                f"| {cat} | "
                + " | ".join(str(c[(f, cat)]) for f in FIELD_NAMES)
                + f" | {tot} | {tot / len(rows):.0%} |"
            )
        stage = Counter(
            "label"
            if r["category"] == "label_absent"
            else "ocr"
            if r["category"].startswith("ocr")
            else "kie"
            for r in rows
        )
        lines += [
            "",
            "By stage: " + ", ".join(f"{k} {v} ({v / len(rows):.0%})" for k, v in stage.items()),
            "",
            "Errors by image condition (share of wrong values whose image has the condition):",
            "",
            "| Condition | Errors with it | Of which OCR-stage |",
            "|---|---:|---:|",
        ]
        for cond, label in (
            ("poor_quality", "official quality < 0.5"),
            ("rotated", "rotated photo"),
        ):
            rs = [r for r in rows if r[cond]]
            ocr_n = sum(r["category"].startswith("ocr") for r in rs)
            lines.append(f"| {label} | {len(rs)} ({len(rs) / len(rows):.0%}) | {ocr_n} |")
        rs = [r for r in rows if r["store_seen"] is False]
        lines += [f"| store not seen in train | {len(rs)} ({len(rs) / len(rows):.0%}) | – |", ""]
    path = REPORTS / f"error_breakdown_{split}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["smoke", "dev"])  # never test
    ap.add_argument("--model", nargs="+", default=["rules", "llm"], choices=list(PIPELINES))
    ap.add_argument("--device", default="gpu:0")
    ap.add_argument("--render", type=int, default=0, help="crops for manual review")
    args = ap.parse_args()
    all_rows = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.model:
        rows = analyse(args.split, name, args.device)
        for r in rows:
            r["ocr_run"] = ocr_run_name(PIPELINES[name].ocr, args.device)
        all_rows[name] = rows
        with open(OUT / f"{name}_{args.split}.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(write_report(args.split, all_rows))
    if args.render:
        print(render([r for rows in all_rows.values() for r in rows], args.split, args.render))
    return 0


if __name__ == "__main__":
    sys.exit(main())
