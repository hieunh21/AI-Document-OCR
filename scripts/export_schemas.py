"""Write JSON Schemas of the public data contracts to docs/schemas/.

python scripts/export_schemas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.schemas import OCRResult, ReceiptExtraction  # noqa: E402

OUT = Path("docs/schemas")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, model in {
        "receipt_extraction": ReceiptExtraction,
        "ocr_result": OCRResult,
    }.items():
        path = OUT / f"{name}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
