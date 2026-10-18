"""Small geometry helpers shared by the dataset adapter and KIE."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def xywh_to_xyxy(b: Sequence[float]) -> list[float]:
    x, y, w, h = b
    return [float(x), float(y), float(x + w), float(y + h)]


def polygon_bbox(points: Sequence[Sequence[float]]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def reading_order(items: Sequence[T], bbox: Callable[[T], Sequence[float]]) -> list[T]:
    """Sort boxes top-to-bottom, then left-to-right within a visual line.

    Two boxes are on the same line when their vertical overlap is at least half
    of the smaller box height — receipts often print "Tổng tiền:" and the amount
    on one line with slightly different y.
    """
    by_top = sorted(items, key=lambda it: (bbox(it)[1] + bbox(it)[3]) / 2)
    lines: list[list[T]] = []
    for it in by_top:
        y0, y1 = bbox(it)[1], bbox(it)[3]
        if lines:
            last = lines[-1]
            ly0 = min(bbox(o)[1] for o in last)
            ly1 = max(bbox(o)[3] for o in last)
            overlap = min(y1, ly1) - max(y0, ly0)
            min_h = min(y1 - y0, min(bbox(o)[3] - bbox(o)[1] for o in last))
            if min_h > 0 and overlap >= 0.5 * min_h:
                last.append(it)
                continue
        lines.append([it])
    return [it for line in lines for it in sorted(line, key=lambda o: bbox(o)[0])]
