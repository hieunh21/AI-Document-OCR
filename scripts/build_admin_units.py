"""Build ``configs/vn_admin_units.json`` from the official administrative directory.

Source: the General Statistics Office directory (danhmuchanhchinh.gso.gov.vn), as
published in JSON by https://github.com/dvhcvn/data (``tree.json``: current and former
names of provinces, districts and wards). The receipts are from 2020, so the
pre-2025 three-level units are what they print; former names are kept too.

    python scripts/build_admin_units.py            # downloads into data/external/ if missing

Output: one entry per distinct (name, level, parent) *without* the type prefix
("Phường Cẩm Tây" -> "Cẩm Tây"): level (1 province, 2 district, 3 ward), the names of
its district (wards only) and of its province — every name a unit has had, so 2020
spellings and post-2025 names both work as context.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/dvhcvn/data/master/tree.json"
RAW = Path("data/external/dvhcvn/tree.json")
OUT = Path("configs/vn_admin_units.json")
_PREFIX = re.compile(
    r"^(Thành phố Trung ương|Thành phố|Tỉnh|Quận|Huyện|Thị xã|Phường|Xã|Thị trấn)\s+"
)


def strip_type(name: str) -> str:
    return _PREFIX.sub("", name.strip())


def walk(node: dict, level: int, parents: list[list[str]], out: set) -> None:
    for item in node.values():
        names = [n for n in (strip_type(x) for x in item[0]) if n]
        province = parents[0] if parents else names
        district = parents[1] if level == 3 and len(parents) > 1 else []
        for n in names:
            if not n.isdigit():
                out.add((n, level, tuple(district), tuple(province)))
        if len(item) > 1 and isinstance(item[1], dict):
            walk(item[1], level + 1, [*parents, names], out)


def main() -> int:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, RAW)
    raw = RAW.read_bytes()
    units: set = set()
    walk(json.loads(raw), 1, [], units)
    entries = sorted([n, lvl, list(d), list(p)] for n, lvl, d, p in units)
    OUT.write_text(
        json.dumps(
            {
                "source": "GSO danhmuchanhchinh.gso.gov.vn via github.com/dvhcvn/data tree.json",
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "fields": ["name", "level", "district_names", "province_names"],
                "units": entries,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    by_level = {lvl: sum(e[1] == lvl for e in entries) for lvl in (1, 2, 3)}
    print(f"{OUT}: {len(entries)} names {by_level}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
