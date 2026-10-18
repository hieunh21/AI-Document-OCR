from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.dataset_adapter import DEFAULT_ROOT, TRAIN_CSV

HAS_DATASET = (DEFAULT_ROOT / TRAIN_CSV).exists()
requires_dataset = pytest.mark.skipif(
    not HAS_DATASET, reason=f"MC-OCR data not found under {DEFAULT_ROOT}"
)
PRIVATE_CHECKS = Path("data/private/manual_annotation_checks.json")


@pytest.fixture(scope="session")
def mcocr_records():
    from evaluation.dataset_adapter import load_mcocr

    return {r.img_id: r for r in load_mcocr()}
