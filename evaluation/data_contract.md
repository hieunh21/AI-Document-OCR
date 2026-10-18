# Data contract — MC-OCR 2021 for VietDocAI

Status: **frozen for Phase 2–4** (version `mcocr2021-gt-v2`, splits `splits-v1`).
Changing anything below requires bumping the version, regenerating
`data/splits/` and `docs/dataset_eda.md`, and saying so in the next report.

## 1. Source

| Item | Value |
|---|---|
| Dataset | MC-OCR 2021 (RIVF 2021 challenge), public Kaggle copy — see `docs/data_usage.md` |
| Labelled file | `mcocr_train_df.csv` (1,155 rows, one per image) + `train_images/` |
| Unlabelled | `val_images/` (391 images) — **never** used for metrics |
| Local location | `$VIETDOCAI_MCOCR_DIR` (default `data/raw/mcocr/`), git-ignored |
| CSV SHA-256 | recorded in every manifest (`source_csv_sha256`) |

CSV columns: `img_id, anno_polygons, anno_texts, anno_labels, anno_num, anno_image_quality`.
`anno_polygons` is a Python-literal list of `{category_id, segmentation, area, bbox=[x,y,w,h], width, height}`;
texts and labels are `|||`-joined and aligned with the polygons.

`label_dict.json` was not shipped with our copy. Category ids were inferred and
verified against the label strings on all 6,585 regions (one typo, see §3):

| category_id | label | internal field |
|---:|---|---|
| 15 | `SELLER` | `seller` |
| 16 | `ADDRESS` | `seller_address` |
| 17 | `TIMESTAMP` | `timestamp` |
| 18 | `TOTAL_COST` | `total_cost` |

## 2. Ground truth construction (`evaluation/dataset_adapter.py`)

The adapter returns, per image, an `ai.schemas.ReceiptExtraction` — the same type
every KIE model returns — plus the annotated regions and data-quality `issues`.

1. **Field = category id.** The label string is only checked against it.
2. **Coordinates** are converted from `[x, y, w, h]` to `[x_min, y_min, x_max, y_max]`
   in original-image pixels (annotation width/height equal the real image size for all 1,155 images).
3. **Text** is Unicode NFC with collapsed whitespace (`ai.normalization.text.normalize_text`).
   Case, accents and punctuation are kept.
4. **Multi-region fields** are joined with one space in *reading order*: regions are
   grouped into visual lines (vertical overlap ≥ 50 % of the smaller height), lines
   top-to-bottom, regions left-to-right. CSV order is *not* reading order: for TOTAL_COST alone it
   differs from top-to-bottom order in 472 images (keyword and amount share a line).
5. **`seller`, `seller_address`**: the joined text (no further normalization in GT).
6. **`timestamp`**: ISO 8601 local time — `YYYY-MM-DDTHH:MM[:SS]`, or `YYYY-MM-DD` when the
   receipt shows no time (`ai.normalization.datetime_vi.parse_timestamp`):
   - day-first only; separators `/ - . : ,`; `ngày … tháng … năm …`; ISO; `15 Aug 20`;
   - `CH`/`PM` → +12 h, `SA`/`AM`;
   - seconds kept only if printed;
   - several times (check-in/check-out) → the **first** one, flagged;
   - impossible or garbled values are **not repaired** → `null`, flagged.
7. **`total_cost`**: integer VND (`ai.normalization.money`):
   - amounts are read only from TOTAL_COST regions; dates/clock times inside them are ignored;
   - thousands separators `.`, `,`, space; a final 1–2 digit group is decimals → rounded half-up;
   - numbers `< 100` are item counts, not amounts (`TẠI QUẦY 03`);
   - several distinct amounts → the most frequent (flagged); a tie → `null` (flagged).
8. **Absent field** (no region) → `null`. This is a valid target: a model must output `null`.

## 3. Scorable vs. unscorable fields

A field whose annotation **exists** but cannot be turned into a trustworthy value is
*unscorable* for that image (`McocrRecord.unscorable_fields()`): it counts neither
as right nor as wrong in normalized-value metrics. It is **not** the same as absent.

| Field | Unscorable when the image is flagged |
|---|---|
| `timestamp` | `timestamp_unparsed`, `timestamp_time_only`, `suspect_timestamp_contains_total` |
| `total_cost` | `total_cost_no_amount`, `total_cost_ambiguous`, `suspect_total_cost_contains_date` |
| all four | `no_annotations` (an empty row is more likely unlabelled than empty) |

Counts per field are in `docs/dataset_eda.md` §5.

Known label noise that is **not** detectable automatically (found by the manual check,
`docs/annotation_manual_check.md`): wrong digits in amounts, skipped address lines,
typos in names. It caps achievable exact match; it is reported, not corrected.

## 4. Splits (`evaluation/splits.py`)

- Internal **train / dev / test = 70 / 15 / 15 %** of the 1,155 labelled images
  → 808 / 174 / 173. Seed `2026`.
- **Grouping** (a group is never split): identical bytes (SHA-256), near-duplicate photo
  (1024-bit dHash ≤ 150 bits), or same transaction (folded seller + timestamp to the
  minute + total). 37 groups with ≥ 2 images.
- **Stratified by seller**: sellers with ≥ 15 images form their own stratum, the rest
  `other`, missing seller `no_seller`.
- `smoke.json`: 15 images **from dev** for quick local runs.
- Manifests store `img_id`, relative `path`, `sha256`, `group_id`, `stratum` (anonymous
  `seller_NN` id — seller names are label content) — no image
  content or labels. `python -m evaluation.splits --check` verifies reproducibility.

What the test split measures: receipts from stores **seen** in training (the app's
realistic setting). It does not measure generalisation to unseen store templates;
per-stratum reporting is required to expose long-tail behaviour.

### Usage rules

| Split | Allowed use |
|---|---|
| train | Writing rules/prompts, few-shot examples, any training, error browsing |
| dev | Model/prompt/threshold selection, error analysis, ablations |
| smoke | Quick runs while coding (subset of dev) |
| test | **One** final run per chosen pipeline, after it is frozen. Never for tuning, never for looking at examples |

## 5. Metrics (definitions for Phase 4)

All metrics are reported on **dev** during development, **test** once at the end, both
on *all* scorable images and per stratum. Every report states split, number of images,
model/config version, and this contract's version.

Per field `f`, per image where `f` is scorable:

- **Exact match (EM)** of normalized values:
  - `seller`, `seller_address`: `fold_for_match(pred) == fold_for_match(gt)` (NFC, case-fold,
    punctuation → space, whitespace collapsed, diacritics **kept**);
  - `timestamp`: equal ISO strings compared at the precision of the GT (date-only GT → date
    part; GT without seconds → `HH:MM`); a date-only match is also reported separately;
  - `total_cost`: integer equality;
  - `null` vs `null` is a match; value vs `null` (either direction) is a miss.
- **Field precision / recall / F1** over non-null values: TP = pred non-null and EM;
  FP = pred non-null and not EM; FN = GT non-null and (pred null or not EM).
- **CER** (character error rate, Levenshtein / len(GT)) on `fold_for_match` text for
  `seller` and `seller_address` — KIE text CER, not full-page OCR CER (MC-OCR has no full-page transcription).
- **IQA**: MAE / RMSE against `anno_image_quality` if a quality model is evaluated.
- **Latency**: `latency_ms` per stage (preprocess / OCR / KIE / total), P50 and P95, with hardware stated.

Our numbers are **not** comparable with the RIVF 2021 leaderboard (different split and protocol).

## Changelog

| Version | Date | Change | Effect |
|---|---|---|---|
| `mcocr2021-gt-v1` | 2026-10-18 | Initial contract (Phase 1) | — |
| `mcocr2021-gt-v2` | 2026-10-18 | Date parser rejected any date directly after `:` (`"Ngày:16/08/2020"`); now only a preceding `<digit>:` (clock time) is excluded. Found in Phase 2 when OCR text dropped the space. | 5 more normalized timestamps (1040 → 1045); splits unchanged (`--check` reproducible) |
