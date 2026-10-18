"""Parse dates/times printed on Vietnamese receipts.

Day-first only (Vietnamese convention). Examples handled::

    "Ngày bán: 15/08/2020 11:41"            -> 2020-08-15T11:41
    "Thời gian: 16:40:27 - 14/08/2020"      -> 2020-08-14T16:40:27
    "Ngày: 10/08/2020 (08:57 CH - 08:57 CH)" -> 2020-08-10T20:57   (CH = chiều = PM)
    "Thời gian : 01.10.2020 08.58"          -> 2020-10-01T08:58
    "NGÀY KINH DOANH: 14/08/2020"           -> 2020-08-14
    "Ngày:"                                 -> None

Never guesses: an impossible date/time is dropped, not repaired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, time

_MONTHS_EN = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}
_Y4 = r"((?:19|20)\d{2})"
# Patterns with an unambiguous 4-digit year (or month name), tried together.
# Each entry: (regex, function(match) -> (year, month, day)).
_DATE_PATTERNS_STRONG = [
    # ISO 2020-08-18
    (
        re.compile(rf"(?<!\d){_Y4}-(\d{{1,2}})-(\d{{1,2}})(?!\d)"),
        lambda m: (int(m[1]), int(m[2]), int(m[3])),
    ),
    # "Ngày 14 tháng 08 năm 2020"
    (
        re.compile(rf"ngày\s*(\d{{1,2}})\s*tháng\s*(\d{{1,2}})\s*năm\s*{_Y4}", re.IGNORECASE),
        lambda m: (int(m[3]), int(m[2]), int(m[1])),
    ),
    # 14/08/2020, 14-08-2020, 14.08.2020; ':' and ',' also seen as separators (13:06:2020).
    # OCR may glue the time to the year: "11/06/202021:07:25", "01.10.202009.14".
    (
        re.compile(
            rf"(?<!\d)(?<!\d:)(\d{{1,2}})\s?[/\-.:,]\s?(\d{{1,2}})\s?[/\-.:,]\s?{_Y4}"
            r"(?:(?!\d)|(?=\d{1,2}[:.]\d{2}(?!\d)))"
        ),
        lambda m: (int(m[3]), int(m[2]), int(m[1])),
    ),
    # "15 Aug 20", "15-Aug-2020"
    (
        re.compile(
            r"(?<!\d)(\d{1,2})[\s\-]+(" + "|".join(_MONTHS_EN) + r")[a-z]*\.?[\s\-,]+"
            r"((?:19|20)?\d{2})(?!\d)",
            re.IGNORECASE,
        ),
        lambda m: (
            int(m[3]) + (2000 if len(m[3]) == 2 else 0),
            _MONTHS_EN[m[2].lower()],
            int(m[1]),
        ),
    ),
]
# 2-digit year: stricter, used only when no strong pattern matched.
_DATE2 = re.compile(r"(?<!\d)(?<!\d:)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})(?![\d:])")
_TIME = re.compile(
    # a sentence dot after the minutes ("09:03. 13/07/2020") is fine; "09.03.2020" is not
    r"(?<![\d.,/])(\d{1,2})\s?[:hH.]\s?(\d{2})(?:\s?[:.]\s?(\d{2}))?(?![\d,/])(?![.]\d)"
    r"(?:\s?(SA|CH|AM|PM)\b)?",
    flags=re.IGNORECASE,
)
_PM = {"CH", "PM"}
_AM = {"SA", "AM"}


@dataclass
class ParsedTimestamp:
    date: date | None = None
    time: time | None = None
    has_seconds: bool = False
    flags: list[str] = field(default_factory=list)  # e.g. "multiple_dates", "multiple_times"

    @property
    def iso(self) -> str | None:
        """ISO string, or ``None`` when no date was found (time alone is not a timestamp)."""
        if self.date is None:
            return None
        if self.time is None:
            return self.date.isoformat()
        fmt = "%H:%M:%S" if self.has_seconds else "%H:%M"
        return f"{self.date.isoformat()}T{self.time.strftime(fmt)}"


def _find_dates(text: str) -> tuple[list[date], list[tuple[int, int]]]:
    found: list[tuple[int, date, tuple[int, int]]] = []
    taken: list[tuple[int, int]] = []
    tiers = [_DATE_PATTERNS_STRONG, [(_DATE2, lambda m: (2000 + int(m[3]), int(m[2]), int(m[1])))]]
    for tier in tiers:
        for rx, ymd in tier:
            for m in rx.finditer(text):
                a, b = m.span()
                if any(a < tb and ta < b for ta, tb in taken):  # overlaps an earlier match
                    continue
                try:
                    found.append((a, date(*ymd(m)), (a, b)))
                except ValueError:
                    continue
                taken.append((a, b))
        if found:
            break
    found.sort(key=lambda x: x[0])
    return [d for _, d, _ in found], [sp for _, _, sp in found]


def _find_times(text: str) -> list[tuple[time, bool]]:
    out = []
    for m in _TIME.finditer(text):
        h, mi = int(m.group(1)), int(m.group(2))
        s = int(m.group(3)) if m.group(3) else 0
        mer = (m.group(4) or "").upper()
        if mer in _PM and h < 12:
            h += 12
        elif mer in _AM and h == 12:
            h = 0
        if h < 24 and mi < 60 and s < 60:
            out.append((time(h, mi, s), m.group(3) is not None))
    return out


def parse_timestamp(text: str | None) -> ParsedTimestamp | None:
    if not text:
        return None
    dates, spans = _find_dates(text)
    # Blank out date substrings so "14.08.2020" is not re-read as a time.
    masked = text
    for a, b in spans:
        masked = masked[:a] + " " * (b - a) + masked[b:]
    times = _find_times(masked)
    if not dates and not times:
        return None

    result = ParsedTimestamp()
    if dates:
        result.date = dates[0]
        if len(set(dates)) > 1:
            result.flags.append("multiple_dates")
    if times:
        result.time, result.has_seconds = times[0]
        if len({t for t, _ in times}) > 1:
            result.flags.append("multiple_times")
    return result
