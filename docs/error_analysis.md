# Error analysis (Phase 4, dev split)

**Scope.** Errors of the pipelines on the **dev** split (174 images). No test image was
looked at. Numbers: `evaluation/reports/model_comparison.md` (accuracy, subsets) and
`evaluation/reports/error_breakdown_dev.md` (automatic attribution). Image ids are given by
their last five characters.

## 1. Method

1. **Automatic attribution** (`python -m evaluation.error_analysis`). Every wrong field
   value is assigned to the *first* stage that failed, using the annotated regions, the
   cached OCR spans and the prediction's evidence boxes. The stages are: label absent,
   OCR did not detect the region, OCR misread it, KIE returned null, KIE cited the wrong
   region, KIE cut the text wrongly, or KIE produced a wrong value from the right region.
2. **Manual review.** 29 wrong field values on 26 images were drawn at random within each
   category and mixed across pipelines (`--render 40`, duplicates removed). For each one
   the crop (green = GT, red = prediction) was compared with the printed receipt.
   Because the sample is stratified by category, the shares below describe the
   categories. They are **not** unbiased estimates for all errors.

## 2. Where the errors come from (automatic)

| Pipeline | Wrong values | Label absent | OCR stage | KIE stage |
|---|---:|---:|---:|---:|
| `rules` (PP-OCRv6 + rules + gazetteer) | 174 | 20 (11 %) | 116 (67 %) | 38 (22 %) |
| `llm` (PP-OCRv6 + Gemini + gazetteer) | 149 | 26 (17 %) | 97 (65 %) | 26 (17 %) |
| `llm-vietocr` (VietOCR + Gemini + gazetteer) | 158 | 25 (16 %) | 67 (42 %) | 66 (42 %) |

- With PP-OCRv6, two thirds of the wrong values sit on text that OCR already got wrong:
  more than 10 % folded CER against the label, a wrong date, or a missing amount. Almost
  all of these are **addresses** (70 of 113 for rules), which are long and printed small.
- VietOCR reads Vietnamese text much better (region CER 5.2 % vs 11.0 %), and the OCR-stage
  errors fall from 97 to 67. **Overall EM does not rise** (76.6 % vs 78.0 %, within noise),
  because the errors move to the KIE stage. Accurate OCR text exposes the line-boundary
  and spelling disagreements described in §3 below. VietOCR also reads timestamps slightly
  worse (timestamp EM 84.4 % vs 88.6 %).
- **Image quality** matters most. Overall EM is about 55 % on images whose official score
  is below 0.5, against 83–89 % at 0.7 or above (all pipelines). Low-quality images are 13 %
  of dev but carry 22–25 % of the errors, mostly at the OCR stage.

## 3. Manual review (29 values)

| Verdict | Count | Meaning |
|---|---:|---|
| Label missing or wrong, model matches the print | 10 | not a model error |
| Field definition ambiguous | 3 | both answers defensible |
| OCR error | 5 | |
| KIE error | 11 | |

### 3.1 Label omissions and typos (10)

- **Label absent, but the value is printed.** This holds for all 5 reviewed cases. The
  model is right in every one, yet EM counts it as wrong:
  - a total `100.000` printed under "TỔNG TIỀN PHẢI T.TOÁN" (`uavtd`);
  - a `VinCommerce` logo (`aqeha`);
  - the timestamp `10/08/2020 (09:49 CH)` (`bjvsn`; the prediction `21:49` is correct);
  - The Coffee House name and address (`cmdbw`, 2 values).

  Most of the `label_absent` category, 11–17 % of all errors, is therefore probably
  correct output.
- **Typos in labels.**
  - `THE COFFEE HQUSE` for "HOUSE" (`zlljn`);
  - `PHÓ MỎ` for "PHỐ MỎ" (`ajgqu`);
  - `10604062` for "1060-1062" (`iwlls`).
- **Accents that are not printed.** The receipt prints `BIG C AN PHU`, while the label
  reads `AN PHÚ` (`izcur`).

### 3.2 Ambiguous field definitions (3)

- **Seller: logo or store line?** The label takes the logo `co.opmart`, and the model the
  printed store line `Co.opMart HAU GIANG` (`bcbgu`).
- **Total: subtotal or amount to pay?**
  - `miwkj`: the label is "Thành tiền 80 000" (before a 35 000 discount), while the
    model gives "Thanh Toán 45 000", the amount actually paid.
  - `avtwc`: a VinID receipt shows both "Tổng thanh toán 26.010.250" and
    "Tổng tiền 26.032.250", and the label picks the second one.
- The app's review screen exists for exactly these cases.

### 3.3 OCR errors (5)

- **Faint or serif lines are dropped by the PaddleOCR pipeline.** Two clearly printed
  address lines (`nzgjm`, `fdlxd`) are missing from the OCR output, although the same
  PP-OCRv6 detector run standalone finds them. The cause is the pipeline's
  `text_det_box_thresh` of 0.6. Lowering it to 0.5 recovers every annotated region
  (detected 99.6 % → 100 %, folded CER 11.0 % → 10.5 %). KIE does not improve: rules go
  74.3 % → 74.1 % and Gemini 78.0 % → 77.7 % (31 re-run calls). The recovered addresses
  belong to stores outside the gazetteer, and the extra low-confidence boxes add noise.
  **Not adopted.** The `v6_medium_bt05` / `bt04` presets are kept for reproducibility.
- **Blurred digits.** `554 000` is read as `534 000` (`tzbvy`, quality 0.41).
- **Letter confusion.** `VINMART` is read as `VINNART` (`yryue`).
- **Lost dot-below accents.** `cọc` is read as `coc` (`ecfzg`).

### 3.4 KIE errors (11)

**Rules:**

- **Amount printed above the keyword line** (`wrzex`, `iurdz`). The receipt prints
  "Tổng tiền" with its amount half a line higher, and the geometric search picks the
  nearer amount on the next line ("Tiền khách trả 200,000"). It is the same template in
  both cases, so a fix would be worth trying.
- **Time printed in another column.** "Ngày:12/08/2020 … Thời gian:10:43" (`sqwxo`): only
  the date is kept.
- **Addresses without cue words** for stores not in the gazetteer: "22 Quang Trung, P10,
  Gò Vấp" (`gobum`, `plnfr`), "Chợ Sủi Phú Thị Gia Lâm" (`lpghh`), and "403 Phan Huy Ích…"
  (`ivclx`). The rules return null. This is safe for review, but it is a miss.

**Gemini:**

- **The store name is included in the address.** "104 TRẦN PHÚ - CẨM PHẢ" is joined to the
  real address (`yibwb`).
- **Address lines are reordered**, so the result is not in reading order (`bokrq`).
- **The garbled logo text is chosen** instead of the store line (`wgimk`).

## 4. Known vs unseen stores

The split keeps stores together with their seller stratum, so most dev/test receipts come
from stores seen in train. The **13 dev receipts from stores never seen in train** show
the real long-tail behaviour:

| Field EM, unseen stores (dev, n = 13) | rules | Gemini + gaz |
|---|---:|---:|
| Seller | 0 % | 7.7 % |
| Address | 15.4 % | 15.4 % |
| Timestamp | 66.7 % | 75.0 % |
| Total | 80.0 % | 80.0 % |

Seller and address extraction for an unknown store is close to unsolved by both methods.
The rules have no fallback (it was removed because it lowered precision on known stores),
and Gemini rarely reproduces the label's exact spelling. Totals and timestamps carry over
much better. The test split confirms the pattern with n = 20 (`final_test.md`).

## 5. Consequences

1. **Exact match understates accuracy.** In the reviewed sample about a third of the
   "errors" are label problems. Every reported number is a lower bound, and dev/test
   differences of 1–2 points are within label noise. Paired bootstrap intervals are
   reported for that reason.
2. **The review UI is essential, not decoration.** Ambiguous definitions (logo vs store
   line, subtotal vs amount paid) need a human decision. The app shows the evidence box
   so the reviewer can decide quickly.
3. **Possible improvements, in order of expected gain:**
   - raise a warning for low image quality (already done) and suggest retaking the photo;
   - add a rule for an amount printed above its keyword;
   - add address cue words (`P10`, `Q.`, "Chợ …") with a review flag for unknown stores;
   - add Gemini prompt rules: keep reading order, never include the store name in the
     address.

   None of these was tuned on test.

## 6. Update for pipeline-v3 (fine-tuned Vietnamese recognizer, dev)

Report: `evaluation/reports/error_breakdown_dev.md`. It now also covers `rules-adm` (v2),
`rules-vi3` and `llm-vi3`.

| Rules pipeline | Wrong values | label | OCR | KIE |
|---|---:|---:|---:|---:|
| v2 (`rules-adm`, stock OCR) | 165 | 21 (13 %) | 111 (67 %) | 33 (20 %) |
| v3 (`rules-vi3`) | 161 | 22 (14 %) | 47 (29 %) | 92 (57 %) |

The new recognizer removes most OCR-stage errors (OCR misreads: 108 → 44). The total
barely moves, because the errors move to the KIE stage. The OCR text now matches the
label, so a wrong value is blamed on extraction.

Most of the new KIE-stage errors are `kie_partial`: the right region was chosen, but the
text differs. That category grew from 15 to 65. Of those 65:
- 38 differ from the label by at most 2 characters;
- 9 differ by 3–5 characters;
- 18 differ by more than 5 characters.

In a sample of these near misses, the label is often the wrong side:
- "690 Trần Phú" for a printed "590";
- "Chợ sửi" for "Chợ Sủi";
- "Vin+QNH" or "NHÀ QNH" for "VM+QNH";
- "12 Thanh Niên" for "112".

Real misses remain as well:
- labels typed without accents ("Duong", "dich vu") against an accented reading;
- one "dich" read as "đich".

Gemini on v3 OCR shows the same pattern: 36 of its 67 `kie_partial` values are within 2
characters of the label.

Consequences:
- Exact match against these labels now understates v3 more than it did v2. A cleaned dev
  label set would measure the gap between the two.
- The extraction errors worth fixing are elsewhere:
  - `kie_missed`: 23 values, mostly seller and address of stores without a gazetteer
    entry;
  - `kie_partial` with large differences: 18 values (wrong lines joined or dropped).

  A learned line classifier targets exactly these.
