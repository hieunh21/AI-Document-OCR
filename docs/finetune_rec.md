# Fine-tuning the PP-OCRv6 recognizer for Vietnamese

## Why

The stock `PP-OCRv6_medium_rec` dictionary has 18,708 characters, but **none of the 90
Vietnamese letters carrying a tone mark**: ả, ạ, ầ, ẩ, ộ, ờ, ị, ủ, ỵ and the rest of
Unicode block U+1EA0–U+1EF9. Those letters occur 6,572 times in the MC-OCR labels. The
model cannot output them, so it substitutes look-alikes ("Câm Phå" for "Cẩm Phả"). This
explains most of its error: folded CER is 11.0 %, but only 4.4 % once accents are
ignored. `latin_PP-OCRv5_mobile_rec` lacks the same letters.

The fine-tune does three things.
- It **replaces the dictionary with a 237-character Vietnamese one** (`dict_vi.txt`).
- It builds the new output layers from the stock ones (`scripts/remap_rec_head.py`).
- It fine-tunes the whole model on the MC-OCR train lines.

The current recipe is round 3. Rounds 1 and 2 failed because of a training-width trap,
described below.

## Steps

1. **Export the line crops (local machine):**

   ```bash
   python scripts/export_rec_dataset.py --zip      # -> outputs/rec_dataset.zip (~17 MB)
   ```

   | Split | Annotated regions | Kept | Dropped (no matching detector box / label disagrees / > 48 chars) |
   |---|---:|---:|---|
   | train | 4,623 | 3,711 | 805 / 72 / 35 |
   | dev (validation only) | 975 | 784 | 175 / 7 / 9 |

   - Each crop is cut from the detector box exactly as at inference: the photo is turned
     upright first, then the line is rectified.
   - The test split is not exported.
   - Every character in the dev labels is in the dictionary.

2. **Train.** Round 3 was trained on the local RTX 3050 (4 GB) in about 25 minutes,
   evaluation included. `notebooks/finetune_ppocrv6_rec_vi.ipynb` runs the same
   configuration on Colab: upload `outputs/rec_dataset.zip` to `MyDrive/vietdocai/` and run
   the cells in order.

   The configuration is the official `configs/rec/PP-OCRv6/PP-OCRv6_medium_rec.yml`
   (PaddleOCR commit `dab3fe3`), with these changes:

   | Setting | Value | Why |
   |---|---|---|
   | dictionary | `dict_vi.txt`, 237 characters, `use_space_char` | the tone-marked letters |
   | initial weights | stock pretrained weights; output layers from `remap_rec_head.py` | every layer starts trained |
   | training width | **320 px**, as in the stock config | see the width trap below |
   | `max_text_length` | 36 (stock: 25) | the maximum that fits in 40 time steps; drops 2.6 % of train lines |
   | data | real train lines only, used once | synthetic lines did not help (round 3, below) |
   | schedule | 15 epochs, batch 16, Adam, cosine LR 1e-4, warm-up 0.3 epoch | |
   | checkpoint | best line accuracy on the dev crops | |

3. **Install the model.** Copy the exported inference model into
   `models/ppocrv6_medium_rec_vi3/` (git-ignored). The folder must contain `inference.yml`.
   Preset: `v6_medium_vi3`.
4. **Evaluate on dev:**

   ```bash
   python -m evaluation.ocr_eval run --split dev --preset "v6_medium_vi3+orient" --device gpu:0
   python -m evaluation.kie_eval --split dev --ocr-run "v6_medium+orient@gpu0" "v6_medium_vi3+orient@gpu0" --kie rules+gaz+adm-nofb
   ```

   Train lines also feed the gazetteer and the rules, so dev is the only fair check.

## The width trap (cause of rounds 1–2 failing)

In training mode, the PPLCNetV4 backbone pools its features to a **fixed 40 time steps**:

```python
if self.training:
    x = F.adaptive_avg_pool2d(x, [1, 40])
else:
    x = F.avg_pool2d(x, [3, 2])  # time steps = image width / 8
```

40 steps match the stock 320 px training width. Rounds 1 and 2 used 640 px images to make
room for long labels. Training then squeezed 80 steps of image into 40, while inference
kept 80.
- The model learnt characters half as wide as they appear at inference, so it **repeated
  characters**: "119.000" → "1119.000", "ANAN" → "ANANAN".
- Labels longer than about 38 characters cannot fit 40 steps under CTC. The CTC loss
  started near 50 although the starting model was the stock one.

There are two working fixes. Both were tested for 1 epoch on the real lines and scored on
the dev crops:

| Training (1 epoch, real lines) | CER | CER no-accent | Digit CER | Line exact |
|---|---:|---:|---:|---:|
| start (remapped stock model) | 12.92 % | 6.06 % | 2.28 % | 29.1 % |
| 640 px, as rounds 1–2 | 36.20 % | 35.17 % | 32.17 % | 4.1 % |
| 640 px, BatchNorm frozen | 17.34 % | 15.18 % | 8.99 % | 38.5 % |
| 640 px, training pool set to width / 2 (patched) | 5.30 % | 3.81 % | 2.24 % | 60.7 % |
| **320 px, `max_text_length` 36** | **5.14 %** | **3.68 %** | **1.90 %** | **60.6 %** |

Round 3 uses 320 px. It needs no code patch and trains six times faster. At inference,
long lines keep their aspect ratio and get more time steps, as with the stock model.

Earlier explanations were wrong:
- round 1 blamed the random output layers;
- round 2 suspected BatchNorm statistics and synthetic data.

Freezing BatchNorm only damped the damage. A run on 11 % real and 89 % synthetic lines (the
round-2 mix) was less bad than real lines alone at 640 px: 22.2 % against 32.2 % digit CER.

## Results on dev

Dev line crops. These crops also selected the checkpoints. Digit CER counts only the
digits of each line.

| Recognizer | CER | CER no-accent | Digit CER | Line exact |
|---|---:|---:|---:|---:|
| PP-OCRv6 medium (stock) | 12.94 % | 6.04 % | 2.24 % | 29.1 % |
| round 1: random output layers, 640 px, real ×3 + 30k synthetic | 6.04 % | 5.57 % | 4.05 % | 54.5 % |
| round 2: remapped output layers, 640 px, real ×1 + 30k synthetic | 7.93 % | 7.44 % | 7.43 % | 48.5 % |
| **round 3: remapped output layers, 320 px, real only** | **3.81 %** | **3.31 %** | **1.90 %** | **68.6 %** |

Round 3 also tried real ×3 plus 30k synthetic lines. That run was stopped during epoch 2
because the machine ran low on memory. After epoch 1, its dev line accuracy (PaddleOCR
metric) was 73.8 %, against 77.7 % for the best real-only checkpoint.

Full dev pipeline (`evaluation/reports/ocr_dev.md`, same detector and orientation):

| OCR run | CER folded | CER no-accent | Line exact | Amount read | Date-time read |
|---|---:|---:|---:|---:|---:|
| `v6_medium+orient` | 11.0 % | 4.4 % | 28.0 % | 95.5 % | 88.8 % |
| `v6_medium_vi+orient` (round 1) | 5.5 % | 4.9 % | 51.5 % | 86.5 % | 63.1 % |
| `v6_medium_vi3+orient` (round 3) | **3.2 %** | **2.4 %** | **63.5 %** | 95.5 % | 86.9 % |

KIE on dev, `rules+gaz+adm-nofb`:

| OCR run | seller | address | timestamp | total | overall EM |
|---|---:|---:|---:|---:|---:|
| `v6_medium+orient` | 74.7 % | 51.1 % | 88.0 % | 90.1 % | 75.6 % |
| `v6_medium_vi+orient` (round 1) | 70.7 % | 54.0 % | 64.1 % | 83.2 % | 67.8 % |
| `v6_medium_vi3+orient` (round 3) | 75.3 % | 52.3 % | 87.4 % | 91.3 % | **76.2 %** |

- **OCR.** Round 3 cuts folded CER from 11.0 % to 3.2 % and reads digits slightly better
  than stock.
- **KIE.** EM gains +0.6 points, with a paired bootstrap 95 % CI of [−0.4, +1.6] (2,000
  resamples, seed 2026). This is not significant.

The small KIE gain was expected. In an offline test on round 1, the fine-tuned reading
replaced the stock one only where the two agree once accents, case and spaces are ignored.
This changed 1,221 spans and left KIE EM at exactly 75.6 %. The gazetteer and the
administrative directory already restore accents on the seller and address values.

What round 3 adds is correct Vietnamese text everywhere else: unseen stores, evidence text
and raw OCR output.

**Adopted as `pipeline-v3`** (`configs/pipeline.json`) for both the rules and the Gemini
pipeline. With Gemini on the same proxy backend, overall EM goes from 77.5 % to 78.1 %
(+0.6 [−1.0, +2.1]). See `evaluation/reports/ocr_vi3_dev.md`.

## Caveats

- **Validation set.** Dev crops select the checkpoint, and KIE on dev then measures the
  gain. Both use dev, which slightly favours the chosen checkpoint. Report it as a dev
  result, as for every other Phase 4 choice.
- **Small real data.** 3.7k real lines is little. Rare letters such as "ỵ" (32
  occurrences in all labels) are seen only a few times.
- **Label noise.** Labels contain typos ("1060-1082" for a printed "1060-1062"). The
  filter only removes gross mismatches.
