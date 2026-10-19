# Fine-tuned Vietnamese recognizer (round 3) — split `dev`

- **Recognizer:** PP-OCRv6 medium rec fine-tuned on the MC-OCR train lines with a
  237-character Vietnamese dictionary (`docs/finetune_rec.md`), preset `v6_medium_vi3`.
- **Unchanged:** same detector, orientation and KIE as pipeline-v2.
- **Data:** dev split, 174 images. Dev crops also selected the training checkpoint, so these
  are dev numbers.
- **Statistics:** paired bootstrap with 2,000 resamples, seed 2026, metrics `kie-metrics-v2`.

## OCR (annotated regions, `evaluation/reports/ocr_dev.md`)

| OCR run | CER | CER folded | CER no-accent | Line exact | Amount read | Date-time read |
|---|---:|---:|---:|---:|---:|---:|
| `v6_medium+orient` | 13.0 % | 11.0 % | 4.4 % | 28.0 % | 95.5 % | 88.8 % |
| `v6_medium_vi3+orient` | **4.9 %** | **3.2 %** | **2.4 %** | **63.5 %** | 95.5 % | 86.9 % |

Line crops only (784 dev lines, lower-cased):

| Recognizer | CER | CER no-accent | Digit CER | Line exact |
|---|---:|---:|---:|---:|
| stock | 12.94 % | 6.04 % | 2.24 % | 29.1 % |
| vi3 | **3.81 %** | **3.31 %** | **1.90 %** | **68.6 %** |

## KIE exact match

| Pipeline | OCR | KIE | seller | address | timestamp | total | overall |
|---|---|---|---:|---:|---:|---:|---:|
| v2 default | stock | `rules+gaz+adm-nofb` | 74.7 % | 51.1 % | 88.0 % | 90.1 % | 75.6 % |
| **v3 default** | vi3 | `rules+gaz+adm-nofb` | 75.3 % | 52.3 % | 87.4 % | 91.3 % | **76.2 %** |
| v2 llm, first run | stock | `llm:gemini+gaz` | 75.3 % | 56.3 % | 88.6 % | 93.2 % | 78.0 % |
| v2 llm, re-sent on current backend | stock | `llm:gemini:rerun2+gaz` | 75.3 % | 55.2 % | 89.8 % | 91.3 % | 77.5 % |
| **v3 llm** | vi3 | `llm:gemini+gaz` | 74.7 % | 57.5 % | 87.4 % | 94.4 % | **78.1 %** |

Paired differences in overall EM (95 % interval):

| Comparison | Difference |
|---|---|
| rules: vi3 − stock OCR | +0.6 [−0.4, +1.6] |
| Gemini: vi3 − stock OCR (same proxy backend, `rerun2`) | +0.6 [−1.0, +2.1] |
| Gemini: vi3 − stock OCR (first run) | +0.1 [−1.3, +1.6] |
| vi3: Gemini − rules | +1.9 [−0.3, +4.4] |

- **The recognizer cuts OCR error by about 3.4×** (folded CER 11.0 → 3.2 %). It reads
  digits as well as the stock model.
- **The KIE gain is small and not significant** for both extractors. The gazetteer and the
  place-name directory already restore accents for the rules. Gemini restores them itself.
- **Adopted as `pipeline-v3`** (`configs/pipeline.json`) because:
  - OCR quality improves clearly;
  - KIE does not get worse;
  - text outside the four fields now carries correct Vietnamese: evidence, raw OCR, unseen
    stores.
- **The test split was not used.**
- **Cost:** the v3 Gemini row cost 174 new requests through the proxy. The two stock-OCR
  Gemini rows were read from the response cache.
