"""Draw a few demo receipts for the web app (invented stores, items and prices).

    python scripts/make_demo_receipts.py        # -> frontend/public/demo/*.jpg

They let visitors try the app without a receipt of their own. The content is made up;
nothing comes from MC-OCR.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path("frontend/public/demo")
FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]
BOLD_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/tahomabd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]

RECEIPTS = {
    "demo_coffee.jpg": {
        "header": [
            "CÀ PHÊ PHỐ CỔ",
            "28 Hàng Bạc, P. Hàng Bạc, Q. Hoàn Kiếm, Hà Nội",
            "ĐT: 024 3826 1234",
        ],
        "title": "HÓA ĐƠN THANH TOÁN",
        "meta": ["Số HĐ: 000318", "Ngày: 12/09/2026 08:47", "Thu ngân: Lan"],
        "items": [("Cà phê sữa đá", 2, 35000), ("Bạc xỉu", 1, 39000), ("Bánh mì trứng", 1, 25000)],
        "footer": ["Cảm ơn quý khách - Hẹn gặp lại!"],
    },
    "demo_minimart.jpg": {
        "header": ["SIÊU THỊ MINI AN KHANG", "112 Lê Lợi, P. Bến Thành, Q.1, TP. Hồ Chí Minh"],
        "title": "PHIẾU TÍNH TIỀN",
        "meta": ["Quầy: 02    NV: Tuấn", "Thời gian: 05/10/2026 19:22:15"],
        "items": [
            ("Sữa tươi không đường 1L", 2, 32500),
            ("Mì ly hải sản", 4, 9000),
            ("Nước suối 500ml", 6, 5000),
            ("Bánh quy bơ", 1, 47000),
            ("Dầu gội 650g", 1, 129000),
        ],
        "footer": ["Giữ hóa đơn để đổi trả trong 7 ngày", "Xin cảm ơn!"],
    },
    "demo_tilted.jpg": {
        "header": ["NHÀ THUỐC TÂM ĐỨC", "45 Trần Phú, P. Cẩm Tây, TP. Cẩm Phả, Quảng Ninh"],
        "title": "HÓA ĐƠN BÁN LẺ",
        "meta": ["Ngày bán: 20/08/2026 15:05"],
        "items": [("Vitamin C 500mg", 2, 45000), ("Khẩu trang y tế (hộp)", 1, 38000)],
        "footer": ["Chúc quý khách mau khỏe!"],
        "tilt": 7,
    },
}


def font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    sys.exit("No font with Vietnamese glyphs found; install DejaVu Sans or Arial.")


def money(v: int) -> str:
    return f"{v:,}".replace(",", ".")


def draw_receipt(spec: dict, rng: random.Random) -> Image.Image:
    w = 640
    regular, small = font(FONT_CANDIDATES, 22), font(FONT_CANDIDATES, 19)
    bold, big = font(BOLD_CANDIDATES, 26), font(BOLD_CANDIDATES, 30)
    lines: list[tuple[str, ImageFont.FreeTypeFont, str]] = []  # text, font, align
    lines.append((spec["header"][0], big, "center"))
    lines += [(t, small, "center") for t in spec["header"][1:]]
    lines += [("", small, "left"), (spec["title"], bold, "center"), ("", small, "left")]
    lines += [(t, regular, "left") for t in spec["meta"]]
    lines.append(("-" * 60, small, "left"))
    total = 0
    for name, qty, price in spec["items"]:
        amount = qty * price
        total += amount
        lines.append((name, regular, "left"))
        lines.append((f"   {qty} x {money(price)}\t{money(amount)}", regular, "row"))
    lines.append(("-" * 60, small, "left"))
    lines.append((f"TỔNG CỘNG:\t{money(total)}", bold, "row"))
    lines.append((f"Tiền khách đưa:\t{money(-(-total // 50000) * 50000)}", regular, "row"))
    lines.append(("", small, "left"))
    lines += [(t, small, "center") for t in spec["footer"]]

    h = 60 + sum(int(f.size * 1.55) for _, f, _ in lines) + 40
    paper = Image.new("RGB", (w, h), (250, 249, 245))
    d = ImageDraw.Draw(paper)
    y = 40
    for text, f, align in lines:
        ink = (40, 40, 40)
        if align == "center":
            tw = d.textlength(text, font=f)
            d.text(((w - tw) / 2, y), text, font=f, fill=ink)
        elif align == "row":
            left, right = text.split("\t")
            d.text((36, y), left, font=f, fill=ink)
            d.text((w - 36 - d.textlength(right, font=f), y), right, font=f, fill=ink)
        else:
            d.text((36, y), text, font=f, fill=ink)
        y += int(f.size * 1.55)
    paper = paper.filter(ImageFilter.GaussianBlur(0.6))

    # A photo of the receipt on a table.
    bg = Image.new("RGB", (w + 220, h + 200), (118, 101, 86))
    noise = Image.effect_noise(bg.size, 18).convert("RGB")
    bg = Image.blend(bg, noise, 0.12)
    tilt = spec.get("tilt", rng.uniform(-1.5, 1.5))
    rotated = paper.rotate(tilt, expand=True, resample=Image.BICUBIC, fillcolor=(118, 101, 86))
    bg.paste(rotated, ((bg.width - rotated.width) // 2, (bg.height - rotated.height) // 2))
    return bg


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    for name, spec in RECEIPTS.items():
        draw_receipt(spec, rng).save(OUT / name, quality=88)
        print(OUT / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
