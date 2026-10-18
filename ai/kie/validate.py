"""Plausibility checks shared by every KIE provider (Phase 3C).

They add warnings; they never change or drop a value, so a reviewer sees what was
read and why it looks suspicious.
"""

from __future__ import annotations

from datetime import date

from ai.schemas import ExtractionWarning, ReceiptExtraction

MIN_YEAR = 2000
MIN_TOTAL_VND = 1_000
MAX_TOTAL_VND = 1_000_000_000


def validate_extraction(ex: ReceiptExtraction, today: date | None = None) -> ReceiptExtraction:
    today = today or date.today()
    f = ex.fields
    have = {(w.code, w.field) for w in ex.warnings}

    def warn(code: str, field: str, message: str) -> None:
        if (code, field) not in have:
            ex.warnings.append(ExtractionWarning(code=code, field=field, message=message))

    if f.timestamp:
        d = date.fromisoformat(f.timestamp[:10])
        if d.year < MIN_YEAR or d > today:
            warn(
                "timestamp_out_of_range",
                "timestamp",
                f"Receipt date {d.isoformat()} is before {MIN_YEAR} or in the future.",
            )
    if f.total_cost is not None and not MIN_TOTAL_VND <= f.total_cost <= MAX_TOTAL_VND:
        warn(
            "total_cost_out_of_range",
            "total_cost",
            f"Total {f.total_cost:,} VND is outside {MIN_TOTAL_VND:,}–{MAX_TOTAL_VND:,}.",
        )
    if f.total_cost is not None and f.total_cost % 100 and f.total_cost < 10_000:
        warn(
            "total_cost_unusual",
            "total_cost",
            f"Total {f.total_cost:,} VND is small and not a multiple of 100; check the digits.",
        )
    return ex
