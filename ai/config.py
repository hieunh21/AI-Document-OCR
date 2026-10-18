"""Tiny .env loader for the AI core (the backend uses pydantic-settings instead).

Values already present in the environment win; the file is never required.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ("'", '"') and value.count(value[0]) >= 2:
            value = value[1 : value.index(value[0], 1)]  # quoted: keep "#" inside
        else:
            value = "" if value.startswith("#") else value
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()  # inline comment
        if value:
            os.environ.setdefault(key.strip(), value)
