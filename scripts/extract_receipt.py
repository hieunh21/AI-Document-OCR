"""Run the full pipeline on one image and print the extraction as JSON.

    python scripts/extract_receipt.py path/to/receipt.jpg \
        [--ocr "v6_medium+orient"] [--kie rules+gaz-nofb] [--device gpu:0]

KIE names: see ai/kie/factory.py (e.g. rules+gaz-nofb, llm:gemini+gaz).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.kie.factory import build_kie  # noqa: E402
from ai.ocr.factory import build_ocr  # noqa: E402
from ai.pipeline import ReceiptPipeline, pipeline_config  # noqa: E402
from ai.preprocessing.image import load_image  # noqa: E402
from ai.quality.assess import QualityAssessor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--pipeline", default="default")
    ap.add_argument("--ocr")
    ap.add_argument("--kie")
    ap.add_argument("--device", default="gpu:0")
    args = ap.parse_args()
    cfg = pipeline_config(args.pipeline)
    args.ocr, args.kie = args.ocr or cfg["ocr"], args.kie or cfg["kie"]

    pipeline = ReceiptPipeline(
        build_ocr(args.ocr, args.device), build_kie(args.kie), QualityAssessor()
    )
    result = pipeline.run(load_image(args.image))
    out = {
        "extraction": result.extraction.model_dump(mode="json"),
        "quality": result.quality.model_dump(exclude={"features"}) if result.quality else None,
        "ocr": {
            "spans": len(result.ocr.spans),
            "rotation": result.ocr.rotation,
            "model": result.ocr.model_version,
        },
        "timings_ms": {k: round(v) for k, v in result.timings_ms.items()},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
