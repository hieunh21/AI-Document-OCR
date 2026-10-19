# Learned span classifier for KIE (pipeline-v4) — split `dev`

- **Change:** the hand-written line rules are replaced by a LightGBM classifier. It gives
  each OCR span a probability for seller / address / timestamp / total / other
  (`ai/kie/line_features.py`, `ai/kie/learned.py`).
- **Unchanged:** value parsing and normalisation (dates, amounts, gazetteer spelling,
  official place names) work as in the rules.
- **Training:** `python -m evaluation.train_line_classifier`, on the **train** split
  only.
  - Data: 808 images × 2 OCR runs (fine-tuned and stock recognizer), 67,151 spans.
  - Model: 99 boosting rounds.
  - Thresholds and the confidence level were chosen on out-of-fold train predictions.
- **Measurement:** dev was used once, to compare against the rules. The test split was
  not used.
- **Statistics:** paired bootstrap with 2,000 resamples, seed 2026, metrics
  `kie-metrics-v2`.

## Out-of-fold, train (5 folds; gazetteer built from the other folds for both extractors)

| OCR | Extractor | seller | address | timestamp | total | overall |
|---|---|---:|---:|---:|---:|---:|
| fine-tuned (vi3) | rules | 73.2 % | 48.8 % | 87.1 % | 89.1 % | 74.2 % |
| fine-tuned (vi3) | **classifier** | **78.5 %** | **56.3 %** | 87.3 % | 89.5 % | **77.6 %** |
| stock | rules | 73.6 % | 48.3 % | 86.6 % | 86.9 % | 73.5 % |
| stock | **classifier** | **78.0 %** | **55.2 %** | 86.6 % | 88.6 % | **76.8 %** |

## Dev

| OCR | Extractor | seller | address | timestamp | total | overall | all 4 fields |
|---|---|---:|---:|---:|---:|---:|---:|
| vi3 | rules (v3 default) | 75.3 % | 52.3 % | 87.4 % | 91.3 % | 76.2 % | 39 % |
| vi3 | **classifier (v4 default)** | **78.2 %** | **60.9 %** | 87.4 % | 90.7 % | **79.0 %** | 42 % |
| stock | rules | 74.7 % | 51.1 % | 88.0 % | 90.1 % | 75.6 % | 37 % |
| stock | classifier | 76.4 % | 59.8 % | 88.0 % | 89.4 % | 78.1 % | 41 % |
| vi3 | Gemini 2.5 Flash + gazetteer | 74.7 % | 57.5 % | 87.4 % | 94.4 % | 78.1 % | 41 % |

| Paired difference in overall EM | 95 % interval |
|---|---|
| classifier − rules (vi3 OCR) | **+2.8 [+1.0, +4.7]** |
| classifier − rules (stock OCR) | +2.5 [+0.9, +4.1] |
| Gemini − classifier (vi3 OCR) | −0.9 [−2.7, +0.9] |

| KIE latency, 174 dev OCR results, laptop CPU | load | P50 | P95 |
|---|---:|---:|---:|
| rules | 0.2 s | 2 ms | 4 ms |
| classifier | 0.9 s | 29 ms | 84 ms |

## Notes

- **Main gain: the address.** It rises from 52.3 % to 60.9 %. The classifier decides line
  by line from position, size, keywords, place names and its neighbours, where the rules
  only kept lines that matched a list of address hints.
- **The free pipeline now matches the paid one.** On dev the classifier is at least as
  accurate as Gemini over the same OCR. Gemini still reads the total better (94.4 % vs
  90.7 %).
- **"Unseen" stores.** The 13 dev receipts whose seller label is not in the train list are
  mostly label typos ("THỨC COFEE", "Saiigono Co.op", "VinCommercce"); on those the
  prediction is usually the correct spelling.

  Of the 5 genuinely new stores:
  - the rules found none;
  - the classifier found 2 ("BIG C AN PHÚ", with an extra company line, and "Milano
    Coffee Vietnam", partial), picked a wrong line once, and left 2 empty.

  Seller exact match on this group is 0/13 for the rules and the classifier, 1/13 for
  Gemini.
- **The class probability does not make "ready" reliable.** On out-of-fold train
  predictions, raising the confidence level from 0.5 to 0.95 raises the share of receipts
  sent to review from 20 % to 76 %, while the share of "ready" receipts that are fully correct stays at 44–47 %.
  The probability says *where* a field is, not whether its text is right; the remaining
  errors are text-level (OCR and label typos). The confidence level is therefore the
  lowest one (0.5). A trustworthy "ready" needs a value-level confidence, which is not
  built yet.
- **Most important features (gain):**
  - similarity to a known address and to a known seller;
  - a total keyword on the same line;
  - a date pattern;
  - vertical position;
  - a date on the same line;
  - membership of the header block.
