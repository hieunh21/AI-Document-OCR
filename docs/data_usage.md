# Data handling

## MC-OCR 2021

- Source: public Kaggle copy of the RIVF 2021 MC-OCR dataset
  (`mcocr_train_df.csv`, `train_images/`, `val_images/`).
- Local location: `data/raw/mcocr/` (or `$VIETDOCAI_MCOCR_DIR`).
- Cite the MC-OCR 2021 challenge and paper in the README and reports.

### Repository rules

1. Raw images and CSV files are **not committed** (size, and anyone reproducing the
   benchmark downloads the same public copy). Enforced by `.gitignore`
   (`data/raw/`, `data/private/`, `*.jpg`, `*.png`, `mcocr_*.csv`).
2. Committed: code, split manifests (image ids, relative paths, SHA-256, group and
   stratum ids), aggregate statistics (`docs/dataset_eda.md`), metric reports.
3. Per-image values from the manual annotation check live in
   `data/private/manual_annotation_checks.json`; only a summary is in `docs/`.

## User uploads (the app)

- Uploaded receipts are user data: stored outside any public folder, never used as
  ground truth or for training without explicit consent (Phase 8 option D).
- A public deployment requires authentication so users cannot read others' documents.
