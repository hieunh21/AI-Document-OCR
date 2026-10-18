# Manual annotation check (Phase 1)

**Goal:** verify by eye that the adapter's ground truth matches what is printed on the
receipt, and estimate how much label noise to expect.

**Method (2026-10-18):** 12 images drawn from the internal **train** split only (no
dev/test images were looked at), spread over seller strata (2 VinCommerce, 1 each of
the other named strata, 3 `other`, 1 `no_seller`). Each receipt was read by eye and
compared with the normalized ground truth on the 4 fields (48 values).
Per-image values are in `data/private/manual_annotation_checks.json` (git-ignored)
and are asserted by `tests/test_dataset_adapter.py::test_manual_annotation_checks`.

## Result

| Outcome | Values |
|---|---:|
| Ground truth equals the printed value | 41 / 48 |
| Ground truth differs from the printed value | 7 / 48 |
| Adapter bugs found | 1 (fixed) |

Disagreements (all are annotation issues, kept as-is in the ground truth):

| Image (suffix) | Field | Kind of issue |
|---|---|---|
| `…cuqln` | seller_address | digit typo in house number |
| `…quwvb` | seller_address | one printed address line not annotated |
| `…mmpgk` | timestamp | annotated time lost its colon → GT is date-only although a time is printed |
| `…jizcq` | total_cost | extra digit in the annotated amount (×10 error) |
| `…dkfnl` | seller | wrong diacritic in the store name |
| `…ncpxq` | total_cost | extra zero in the annotated amount (×10 error) |
| `…ncpxq` | timestamp | annotation garbled → correctly *unscorable* |

Adapter bug found and fixed: an amount written `43.500.00` (same separator for
thousands and decimals; printed `43.500,00`) was normalized to 4,350,000. Rule now: a
final group of 1–2 digits is always decimals. Regression tests added.

## Take-aways

- On this small sample ~15 % of field values disagree with the print. Exact-match
  scores are therefore capped below 100 % by the labels themselves; error analysis in
  Phase 4 must check the image before blaming the model.
- Amount errors by a factor of 10 exist in the labels. Report `total_cost` EM, and
  inspect large disagreements manually.
- Address errors are mostly partial (typos, missing lines): CER is more informative
  than EM for `seller_address`.
- The sample is small (12 images): treat 15 % as an order of magnitude, not an estimate
  with a tight interval.
