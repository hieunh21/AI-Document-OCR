"""Build an OCR provider from a preset name (used by the app and the benchmarks)."""

from __future__ import annotations

from ai.ocr.base import OCRProvider

# Phase 2 default (evaluation/reports/ocr_dev.md): fastest GPU option with the best
# date/amount reading. VietOCR transformer reads Vietnamese text far better
# (folded CER 5.2 % vs 11.0 %) at ~3 s/image; the final choice is made in Phase 4
# on KIE metrics, not on OCR CER alone.
DEFAULT_PRESET = "v6_medium+orient"
ACCURATE_PRESET = "v6med_det+vietocr_transformer+orient"
CPU_PRESET = "v6_tiny+orient"

HYBRID_PRESETS: dict[str, tuple[str, str, str]] = {
    # name: (paddle detector, recognizer kind, recognizer model)
    "v6med_det+v6med_rec": ("PP-OCRv6_medium_det", "paddle", "PP-OCRv6_medium_rec"),
    "v6med_det+vietocr_transformer": ("PP-OCRv6_medium_det", "vietocr", "vgg_transformer"),
    "v6med_det+vietocr_seq2seq": ("PP-OCRv6_medium_det", "vietocr", "vgg_seq2seq"),
    "v6small_det+vietocr_seq2seq": ("PP-OCRv6_small_det", "vietocr", "vgg_seq2seq"),
}


def available_presets() -> list[str]:
    from ai.ocr.paddle import PRESETS

    return sorted(PRESETS) + sorted(HYBRID_PRESETS)


def build_ocr(preset: str, device: str = "cpu") -> OCRProvider:
    """``preset`` may end with ``+orient`` to classify page orientation first."""
    full_name = preset
    auto_orient = preset.endswith("+orient")
    preset = preset.removesuffix("+orient")
    if preset in HYBRID_PRESETS:
        from ai.ocr.hybrid import HybridOCRProvider, PaddleLineRecognizer, VietOCRRecognizer

        det, kind, model = HYBRID_PRESETS[preset]
        rec = (
            VietOCRRecognizer(model, device=device)
            if kind == "vietocr"
            else PaddleLineRecognizer(model, device=device)
        )
        provider = HybridOCRProvider(det, rec, device=device, auto_orient=auto_orient)
    else:
        from ai.ocr.paddle import PRESETS, PaddleOCRProvider

        if preset not in PRESETS:
            raise ValueError(f"unknown OCR preset {preset!r}; choose from {available_presets()}")
        provider = PaddleOCRProvider(preset, device=device, auto_orient=auto_orient)
    provider.preset_name = full_name  # type: ignore[attr-defined]
    return provider
