"""Phase 0 environment check: does PaddleOCR run on this machine?

Usage:
    python scripts/check_paddleocr.py path/to/receipt.jpg [--device cpu|gpu:0]

Prints library versions, the selected device, the number of detected lines,
the first few lines and the wall-clock latency. It does not save anything.
The first run downloads model weights (~100 MB) into the user cache.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lang", default="vi")
    ap.add_argument("--show", type=int, default=15, help="number of lines to print")
    ap.add_argument(
        "--enable-mkldnn",
        action="store_true",
        help="oneDNN on CPU; off by default because Paddle 3.3 on Windows fails with "
        "'ConvertPirAttribute2RuntimeAttribute not support' when it is on",
    )
    args = ap.parse_args()

    import paddle
    import paddleocr
    from paddleocr import PaddleOCR

    print(f"python      : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print(
        f"paddle      : {paddle.__version__}  cuda_compiled={paddle.device.is_compiled_with_cuda()}"
    )
    print(f"paddleocr   : {paddleocr.__version__}")
    print(f"device      : {args.device}  lang={args.lang}")

    t0 = time.perf_counter()
    ocr = PaddleOCR(
        lang=args.lang,
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=args.enable_mkldnn,
    )
    t_load = time.perf_counter() - t0

    ocr.predict(args.image)  # warm-up (first call includes graph/kernel init)
    t1 = time.perf_counter()
    result = ocr.predict(args.image)
    t_pred = time.perf_counter() - t1

    res = result[0]
    texts, scores, boxes = res["rec_texts"], res["rec_scores"], res["rec_boxes"]
    print(f"load_s      : {t_load:.1f}")
    print(f"predict_ms  : {t_pred * 1000:.0f} (2nd call, warm)")
    print(f"lines       : {len(texts)}")
    for text, score, box in list(zip(texts, scores, boxes, strict=True))[: args.show]:
        print(f"  {score:.2f} {list(map(int, box))} {text}")
    return 0 if texts else 1


if __name__ == "__main__":
    raise SystemExit(main())
