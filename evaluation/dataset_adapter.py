"""MC-OCR 2021 -> internal schema.

Reads ``mcocr_train_df.csv`` (the only labelled split we have) and turns each
row into an :class:`McocrRecord` whose ground truth is a
:class:`ai.schemas.ReceiptExtraction` — the exact type a KIE model returns —
so predictions and ground truth are compared with the same code.

Source format (one row per image)::

    img_id, anno_polygons, anno_texts, anno_labels, anno_num, anno_image_quality

* ``anno_polygons``: Python-literal list of COCO-like dicts
  ``{category_id, segmentation, area, bbox=[x, y, w, h], width, height}``
* ``anno_texts`` / ``anno_labels``: ``|||``-joined, aligned with the polygons
* category ids (no ``label_dict.json`` shipped; inferred and verified against
  labels): 15 SELLER, 16 ADDRESS, 17 TIMESTAMP, 18 TOTAL_COST

Rules (see ``evaluation/data_contract.md``):

* ``category_id`` is the source of truth for the field; a label string that
  disagrees (e.g. the typo ``TOTAL_TOTAL_COST``) is recorded as an issue.
* Multi-region fields are joined with a single space in *reading order*
  (top-to-bottom, left-to-right per line), not CSV order.
* A field with no region is ``None`` (absent/unreadable) — never inferred.
* ``total_cost`` = the amount found in the TOTAL_COST regions (keyword regions
  such as "Tổng tiền:" carry no number). Numbers below ``MIN_TOTAL_VND`` are
  item counts ("TẠI QUẦY 03"), not amounts. Several different amounts -> the
  most frequent one, flagged; a tie -> ``None``, flagged.
* ``timestamp`` needs a date; a time alone stays ``None`` (flagged).
* Regions whose text looks like another field (a date inside TOTAL_COST, an
  amount keyword inside TIMESTAMP) are kept as annotated but flagged
  ``suspect_*`` so evaluation can report a clean subset.

CLI::

    python -m evaluation.dataset_adapter --show 3
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ai.layout import polygon_bbox, reading_order, xywh_to_xyxy
from ai.normalization.datetime_vi import parse_timestamp
from ai.normalization.money import find_amounts, to_vnd
from ai.normalization.text import fold_for_match, normalize_text
from ai.schemas import (
    FIELD_NAMES,
    Evidence,
    FieldEvidence,
    RawFields,
    ReceiptExtraction,
    ReceiptFields,
)

DEFAULT_ROOT = Path(os.environ.get("VIETDOCAI_MCOCR_DIR", "data/raw/mcocr"))
TRAIN_CSV = "mcocr_train_df.csv"
TRAIN_IMAGES = "train_images"
VAL_IMAGES = "val_images"  # public validation images, no labels available
GT_VERSION = "mcocr2021-gt-v2"

CATEGORY_TO_FIELD: dict[int, str] = {
    15: "seller",
    16: "seller_address",
    17: "timestamp",
    18: "total_cost",
}
LABEL_TO_CATEGORY: dict[str, int] = {"SELLER": 15, "ADDRESS": 16, "TIMESTAMP": 17, "TOTAL_COST": 18}
SEP = "|||"
# A field whose annotation exists but cannot give a trustworthy normalized value
# is *unscorable* for normalized metrics (it is not "absent": the receipt does
# have the field). Absent fields (no region) stay scorable with target None.
UNSCORABLE_FLAGS: dict[str, tuple[str, ...]] = {
    "timestamp": (
        "timestamp_unparsed",
        "timestamp_time_only",
        "suspect_timestamp_contains_total",
    ),
    "total_cost": (
        "total_cost_no_amount",
        "total_cost_ambiguous",
        "suspect_total_cost_contains_date",
    ),
}
MIN_TOTAL_VND = 100  # smallest VND denomination in circulation is 100/200 dong
_TOTAL_KEYWORDS = ("tổng", "thanh toán", "phải trả", "tiền")
_DATE_LIKE = re.compile(r"\d{1,2}\s?[/\-.]\s?\d{1,2}\s?[/\-.]\s?\d{2,4}")


@dataclass
class Region:
    """One annotated text region, in original-image pixels."""

    index: int  # position in the CSV row (0-based)
    category_id: int
    field: str
    label: str  # label string exactly as in the CSV
    text: str
    bbox: list[float]  # xyxy
    polygons: list[list[list[float]]]  # one or more rings of [x, y] points

    @property
    def source_id(self) -> str:
        return f"gt_{self.index:02d}"


@dataclass
class McocrRecord:
    img_id: str
    image_path: Path
    width: int
    height: int
    quality: float  # official IQA score in [0, 1]
    regions: list[Region]
    gt: ReceiptExtraction
    issues: list[str] = field(default_factory=list)

    def unscorable_fields(self) -> set[str]:
        """Fields to skip in normalized-value metrics for this image."""
        if "no_annotations" in self.issues:
            return set(FIELD_NAMES)  # an empty row is more likely unlabelled than empty
        return {f for f, flags in UNSCORABLE_FLAGS.items() if any(i in flags for i in self.issues)}

    def regions_of(self, field_name: str) -> list[Region]:
        return [r for r in self.regions if r.field == field_name]

    def to_dict(self) -> dict:
        return {
            "img_id": self.img_id,
            "image_path": self.image_path.as_posix(),
            "width": self.width,
            "height": self.height,
            "quality": self.quality,
            "issues": self.issues,
            "unscorable_fields": sorted(self.unscorable_fields()),
            "regions": [
                {
                    "index": r.index,
                    "field": r.field,
                    "label": r.label,
                    "text": r.text,
                    "bbox": r.bbox,
                }
                for r in self.regions
            ],
            "gt": self.gt.model_dump(mode="json"),
        }


# --------------------------------------------------------------------------- parsing


def _split(cell: str) -> list[str]:
    return cell.split(SEP) if cell else []


def _rings(segmentation: list) -> list[list[list[float]]]:
    rings = []
    for flat in segmentation:
        pts = [[float(flat[i]), float(flat[i + 1])] for i in range(0, len(flat) - 1, 2)]
        if len(pts) >= 3:
            rings.append(pts)
    return rings


def parse_regions(row: dict[str, str]) -> tuple[list[Region], tuple[int, int] | None, list[str]]:
    issues: list[str] = []
    polys = ast.literal_eval(row["anno_polygons"]) if row["anno_polygons"] else []
    texts, labels = _split(row["anno_texts"]), _split(row["anno_labels"])
    n_declared = int(row["anno_num"] or 0)
    if not (len(polys) == len(texts) == len(labels) == n_declared):
        issues.append(
            f"length_mismatch:polys={len(polys)},texts={len(texts)},"
            f"labels={len(labels)},anno_num={n_declared}"
        )
    size = (int(polys[0]["width"]), int(polys[0]["height"])) if polys else None

    regions = []
    for i, (poly, text, label) in enumerate(zip(polys, texts, labels, strict=False)):
        cat = int(poly["category_id"])
        if cat not in CATEGORY_TO_FIELD:
            issues.append(f"unknown_category:{cat}")
            continue
        label = label.strip()
        if LABEL_TO_CATEGORY.get(label) != cat:
            issues.append(f"label_category_mismatch:{label}->{cat}")
        rings = _rings(poly.get("segmentation") or [])
        bbox = (
            xywh_to_xyxy(poly["bbox"])
            if poly.get("bbox")
            else polygon_bbox([p for r in rings for p in r])
        )
        regions.append(
            Region(
                index=i,
                category_id=cat,
                field=CATEGORY_TO_FIELD[cat],
                label=label,
                text=normalize_text(text) or "",
                bbox=bbox,
                polygons=rings,
            )
        )
    return regions, size, issues


def _join(regions: list[Region]) -> str | None:
    return normalize_text(" ".join(r.text for r in regions))


def _evidence(regions: list[Region]) -> FieldEvidence:
    return FieldEvidence(
        regions=[Evidence(bbox=r.bbox, text=r.text, source_id=r.source_id) for r in regions]
    )


def _total_cost(regions: list[Region], issues: list[str]) -> tuple[int | None, str | None]:
    with_amount = [
        (r, [v for v in map(to_vnd, find_amounts(r.text)) if v >= MIN_TOTAL_VND]) for r in regions
    ]
    with_amount = [(r, a) for r, a in with_amount if a]
    if not with_amount:
        issues.append("total_cost_no_amount")
        return None, None
    raw = _join([r for r, _ in with_amount])
    counts = Counter(v for _, amounts in with_amount for v in set(amounts))
    if len(counts) == 1:
        return next(iter(counts)), raw
    (best, n1), (_, n2) = counts.most_common(2)
    if n1 == n2:
        issues.append("total_cost_ambiguous")
        return None, raw
    issues.append("total_cost_multiple_values")
    return best, raw


def build_ground_truth(regions: list[Region]) -> tuple[ReceiptExtraction, list[str]]:
    issues: list[str] = []
    by_field = {
        f: reading_order([r for r in regions if r.field == f], lambda r: r.bbox)
        for f in FIELD_NAMES
    }
    seller = _join(by_field["seller"])
    address = _join(by_field["seller_address"])
    ts_raw = _join(by_field["timestamp"])

    ts_iso = None
    if ts_raw:
        parsed = parse_timestamp(ts_raw)
        ts_iso = parsed.iso if parsed else None
        if ts_iso is None:
            has_time = parsed is not None and parsed.time is not None
            issues.append("timestamp_time_only" if has_time else "timestamp_unparsed")
        else:
            if parsed.time is None:
                issues.append("timestamp_no_time")
            issues.extend(f"timestamp_{f}" for f in parsed.flags)

    total, total_raw = (None, None)
    if by_field["total_cost"]:
        total, total_raw = _total_cost(by_field["total_cost"], issues)

    for f in FIELD_NAMES:
        if not by_field[f]:
            issues.append(f"missing_{f}")
    if any(_DATE_LIKE.search(r.text) for r in by_field["total_cost"]):
        issues.append("suspect_total_cost_contains_date")
    if ts_raw and not _DATE_LIKE.search(ts_raw):
        folded = fold_for_match(ts_raw)
        if any(k in folded for k in _TOTAL_KEYWORDS):
            issues.append("suspect_timestamp_contains_total")

    gt = ReceiptExtraction(
        model_version=GT_VERSION,
        fields=ReceiptFields(
            seller=seller, seller_address=address, timestamp=ts_iso, total_cost=total
        ),
        raw_fields=RawFields(
            seller=seller, seller_address=address, timestamp=ts_raw, total_cost=total_raw
        ),
        evidence={f: _evidence(rs) for f, rs in by_field.items() if rs},
    )
    return gt, issues


# --------------------------------------------------------------------------- loading


def iter_rows(root: Path = DEFAULT_ROOT, csv_name: str = TRAIN_CSV) -> Iterator[dict[str, str]]:
    with open(root / csv_name, encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def record_from_row(row: dict[str, str], root: Path = DEFAULT_ROOT) -> McocrRecord:
    regions, size, issues = parse_regions(row)
    image_path = root / TRAIN_IMAGES / row["img_id"]
    if size is None:  # no annotation at all: read the size from the image header
        from PIL import Image

        with Image.open(image_path) as im:
            size = im.size
        issues.append("no_annotations")
    gt, gt_issues = build_ground_truth(regions)
    return McocrRecord(
        img_id=row["img_id"],
        image_path=image_path,
        width=size[0],
        height=size[1],
        quality=float(row["anno_image_quality"]),
        regions=regions,
        gt=gt,
        issues=issues + gt_issues,
    )


def load_mcocr(root: Path = DEFAULT_ROOT, ids: set[str] | None = None) -> list[McocrRecord]:
    """All labelled records (or only ``ids``), in CSV order."""
    return [
        record_from_row(row, root) for row in iter_rows(root) if ids is None or row["img_id"] in ids
    ]


def load_split(name: str, root: Path = DEFAULT_ROOT, splits_dir: Path = Path("data/splits")):
    """Records of a split manifest (``train``/``dev``/``test``/``smoke``), in manifest order."""
    manifest = json.loads((splits_dir / f"{name}.json").read_text(encoding="utf-8"))
    order = [it["img_id"] for it in manifest["items"]]
    by_id = {r.img_id: r for r in load_mcocr(root, set(order))}
    return [by_id[i] for i in order]


def list_unlabelled_val(root: Path = DEFAULT_ROOT) -> list[Path]:
    d = root / VAL_IMAGES
    return sorted(d.glob("*.jpg")) if d.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect the MC-OCR adapter output")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--show", type=int, default=3, help="print N records as JSON")
    ap.add_argument("--id", action="append", help="print this img_id (repeatable)")
    args = ap.parse_args()

    records = load_mcocr(args.root, set(args.id) if args.id else None)
    issues = Counter(i.split(":")[0] for r in records for i in r.issues)
    print(f"records: {len(records)}  root: {args.root}")
    print("issues:", dict(issues.most_common()))
    for r in records[: args.show] if not args.id else records:
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
