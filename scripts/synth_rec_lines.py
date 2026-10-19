"""Render synthetic receipt text lines for recognizer training (runs on Colab).

    python synth_rec_lines.py --out synth --n 30000 \
        --corpus corpus_train.txt --admin admin_names.txt --fonts /usr/share/fonts

Text comes only from the **train** labels and official place names, never from dev
or test, so validating on dev stays fair. Five kinds of lines:

* train label lines re-rendered in other fonts;
* addresses built from place names ("12 Lê Lợi, P. Cẩm Tây, TP. Cẩm Phả");
* timestamps in the formats seen on receipts;
* amounts with keywords ("Tổng tiền: 1,234,000đ");
* random word sequences that favour rare accented letters.

Rendering mimics thermal receipts: grey or dark ink on off-white paper, blur, faded
strokes, noise, JPEG artefacts, slight rotation. Only fonts covering every
Vietnamese letter are used.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

VI_LETTERS = (
    "aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệfghiìíỉĩịjklmnoòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvwxyỳýỷỹỵz"
)
MAX_LEN = 48
WARD = ["P.", "P. ", "Phường ", "phường ", "F.", "X.", "Xã "]
DISTRICT = ["Q.", "Q. ", "Quận ", "H.", "Huyện ", "TX.", "Thị xã "]
PROVINCE = ["TP.", "TP. ", "Tp.", "Tp ", "T.", "Tỉnh ", "Thành phố "]
TOTAL_KW = [
    "Tổng tiền:",
    "Tổng cộng:",
    "TỔNG TIỀN",
    "Tổng thanh toán:",
    "Tổng tiền phải T.Toán",
    "Thanh toán:",
    "Cộng tiền hàng:",
    "Khách phải trả",
    "Tiền khách đưa",
    "Tiền thừa",
]
_ITALIC = re.compile(r"(?i)italic|oblique|^\w+(bi|i|z)$")  # Windows: ariali, arialbi
TS_KW = ["", "Ngày: ", "Ngày bán: ", "Thời gian: ", "Thời gian:", "Ngày ", "Giờ: ", "TG: "]


def fonts_with_vietnamese(roots: list[str]) -> list[Path]:
    from fontTools.ttLib import TTFont

    ok = []
    for root in roots:
        for p in sorted(Path(root).rglob("*")):
            if p.suffix.lower() not in (".ttf", ".otf") or _ITALIC.search(p.stem):
                continue
            try:
                cmap = TTFont(p, fontNumber=0, lazy=True).getBestCmap() or {}
            except Exception:  # noqa: BLE001 - unreadable font file
                continue
            if all(ord(c) in cmap for c in VI_LETTERS + VI_LETTERS.upper()):
                ok.append(p)
    return ok


class TextSource:
    def __init__(self, corpus: list[str], admin: list[str], rng: random.Random) -> None:
        self.corpus, self.admin, self.rng = corpus, admin, rng
        self.words = sorted(
            {w for line in corpus for w in line.split()} | set(" ".join(admin).split())
        )
        rare = set("ẩẫậẳẵặểễệỉịỏọổỗộởỡợủụửữựỳỷỹỵẻẽẹằắ")
        self.rare_words = [w for w in self.words if set(w.lower()) & rare]

    def case(self, s: str) -> str:
        r = self.rng.random()
        return s.upper() if r < 0.3 else s

    def address(self) -> str:
        r = self.rng
        parts = [f"{r.randint(1, 999)}{r.choice(['', '', '/' + str(r.randint(1, 99)), 'A', 'B'])}"]
        parts.append(r.choice(self.admin))
        if r.random() < 0.8:
            parts.append(r.choice(WARD) + r.choice(self.admin))
        if r.random() < 0.7:
            parts.append(r.choice(DISTRICT) + r.choice(self.admin))
        if r.random() < 0.6:
            parts.append(r.choice(PROVINCE) + r.choice(self.admin))
        sep = r.choice([", ", " - ", ",", " "])
        return self.case(sep.join(parts))

    def timestamp(self) -> str:
        r = self.rng
        d, m, y = r.randint(1, 28), r.randint(1, 12), r.choice([2019, 2020, 2021, 20])
        H, M, S = r.randint(0, 23), r.randint(0, 59), r.randint(0, 59)
        ds = r.choice(["/", "-", "."])
        date = f"{d:02d}{ds}{m:02d}{ds}{y}"
        clock = r.choice([f"{H:02d}:{M:02d}", f"{H:02d}:{M:02d}:{S:02d}", f"{H:02d}.{M:02d}"])
        fmt = r.choice(["{d} {t}", "{t} {d}", "{d}", "{d}-{t}", "{t} - {d}", "{d} ({t} CH)"])
        return r.choice(TS_KW) + fmt.format(d=date, t=clock)

    def amount(self) -> str:
        r = self.rng
        v = r.choice(
            [r.randint(1, 999) * 1000, r.randint(1, 9999) * 100, r.randint(10, 99999) * 1000]
        )
        sep = r.choice([",", ".", " "])
        s = f"{v:,}".replace(",", sep)
        s += r.choice(["", "", "đ", " đ", " VND", ".00", ",00"])
        if r.random() < 0.3:
            s = "VND " + s
        return (r.choice(TOTAL_KW) + " " + s) if r.random() < 0.7 else s

    def words_line(self) -> str:
        r = self.rng
        pool = self.rare_words if self.rare_words and r.random() < 0.6 else self.words
        return self.case(" ".join(r.choice(pool) for _ in range(r.randint(2, 6))))

    def sample(self) -> str:
        r = self.rng
        kind = r.choices(["label", "address", "timestamp", "amount", "words"], [3, 3, 1.5, 1.5, 2])[
            0
        ]
        text = (
            r.choice(self.corpus)
            if kind == "label"
            else getattr(
                self,
                {
                    "address": "address",
                    "timestamp": "timestamp",
                    "amount": "amount",
                    "words": "words_line",
                }[kind],
            )()
        )
        text = " ".join(text.split())
        return text[:MAX_LEN].rstrip()


def render(text: str, font_path: Path, rng: random.Random) -> Image.Image | None:
    """One line image, or ``None`` if the text did not come out legibly.

    Degradations are kept mild on purpose: diacritics are a few pixels tall and must
    stay visible, or the model learns to invent accents that are not in the image.
    """
    size = rng.randint(28, 44)
    font = ImageFont.truetype(str(font_path), size)
    left, top, right, bottom = font.getbbox(text)
    pad_x, pad_y = rng.randint(4, 16), rng.randint(4, 12)
    w, h = right - left + 2 * pad_x, bottom - top + 2 * pad_y
    paper = rng.randint(205, 255)
    ink = rng.randint(0, min(100, paper - 110))  # contrast >= 110 grey levels
    img = Image.new("L", (max(w, 8), max(h, 8)), paper)
    ImageDraw.Draw(img).text((pad_x - left, pad_y - top), text, font=font, fill=ink)
    if rng.random() < 0.15:  # smudged / bold print
        img = img.filter(ImageFilter.MinFilter(3))
    if rng.random() < 0.4:
        img = img.rotate(
            rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=True, fillcolor=paper
        )
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    if rng.random() < 0.4:  # low-resolution phone photo
        f = rng.uniform(0.6, 0.85)
        small = img.resize(
            (max(8, int(img.width * f)), max(8, int(img.height * f))), Image.BILINEAR
        )
        img = small.resize(img.size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    if (arr < (paper + ink) / 2).mean() < 0.01:  # nothing visible was drawn
        return None
    arr += np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2, 8), arr.shape)
    gradient = np.linspace(0, rng.uniform(-25, 25), arr.shape[1])[None, :]  # uneven light
    arr = np.clip(arr + gradient, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("synth"))
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--corpus", type=Path, default=Path("corpus_train.txt"))
    ap.add_argument("--admin", type=Path, default=Path("admin_names.txt"))
    ap.add_argument("--fonts", nargs="+", default=["/usr/share/fonts"])
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fonts = fonts_with_vietnamese(args.fonts)
    if not fonts:
        raise SystemExit("no font covers Vietnamese: apt-get install fonts-noto-core fonts-dejavu")
    print(
        f"{len(fonts)} fonts:",
        ", ".join(p.stem for p in fonts[:12]),
        "…" if len(fonts) > 12 else "",
    )
    src = TextSource(
        [x for x in args.corpus.read_text(encoding="utf-8").splitlines() if x.strip()],
        [x for x in args.admin.read_text(encoding="utf-8").splitlines() if x.strip()],
        rng,
    )
    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out / "synth_list.txt", "w", encoding="utf-8", newline="\n") as fh:
        for i in range(args.n):
            text = src.sample()
            if not text:
                continue
            img = render(text, rng.choice(fonts), rng)
            if img is None:
                continue
            name = f"images/s{i:06d}.jpg"
            img.save(args.out / name, quality=rng.randint(70, 95))
            fh.write(f"{args.out.name}/{name}\t{text}\n")
            if (i + 1) % 5000 == 0:
                print(f"{i + 1}/{args.n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
