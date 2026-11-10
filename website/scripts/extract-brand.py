#!/usr/bin/env python3
"""Extract the logo, mark and founder photo from the slide deck PDF.

Usage:
    python3 scripts/extract-brand.py path/to/deck.pdf [--portrait]

Requires `pdfimages` (poppler-utils) and Pillow. Writes:
    src/assets/brand/logo.png   (full logo with alpha, cropped; Astro makes the web sizes)
    src/assets/brand/mark.png   (logo mark with alpha, cropped)
    src/assets/brand/founder.png   (only with --portrait)
    public/favicon.ico, public/favicon.svg (kept), public/apple-touch-icon.png

The image indices below follow the brief: img-000 + img-001 (mask) is the
full logo, img-021 + img-022 (mask) is the mark, img-005 is the portrait.
Pass --list to print every extracted image with its size and check them.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "src" / "assets" / "brand"
PUBLIC = ROOT / "public"

PAIRS = {
    "logo": ("img-000.png", "img-001.png"),
    "mark": ("img-021.png", "img-022.png"),
}
PORTRAIT = "img-005.png"


def with_alpha(image: Path, mask: Path | None) -> Image.Image:
    im = Image.open(image).convert("RGB")
    if mask and mask.exists():
        m = Image.open(mask).convert("L")
        if m.size != im.size:
            m = m.resize(im.size)
        im.putalpha(m)
    else:
        im = im.convert("RGBA")
    # Crop to the solid shapes (alpha above ~43%) plus a small margin, so the soft
    # glow does not pad the box and shrink the logo at header sizes.
    alpha = im.getchannel("A")
    solid = alpha.point(lambda a: 255 if a > 110 else 0)
    bbox = solid.getbbox() or alpha.getbbox()
    if not bbox:
        return im
    pad_x = int((bbox[2] - bbox[0]) * 0.04)
    pad_y = int((bbox[3] - bbox[1]) * 0.08)
    box = (max(0, bbox[0] - pad_x), max(0, bbox[1] - pad_y), min(im.width, bbox[2] + pad_x), min(im.height, bbox[3] + pad_y))
    return im.crop(box)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pdf = Path(sys.argv[1])
    list_only = "--list" in sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdfimages", "-png", "-f", "1", "-l", "6", str(pdf), f"{tmp}/img"], check=True)
        files = sorted(Path(tmp).glob("img-*.png"))
        if list_only:
            for f in files:
                print(f.name, Image.open(f).size)
            return
        BRAND.mkdir(parents=True, exist_ok=True)
        for name, (img, mask) in PAIRS.items():
            out = with_alpha(Path(tmp) / img, Path(tmp) / mask)
            out.save(BRAND / f"{name}.png", optimize=True)
            print(f"{name}: {out.size} -> {BRAND / (name + '.png')}")
        if "--portrait" in sys.argv:
            portrait = Image.open(Path(tmp) / PORTRAIT).convert("RGB")
            portrait.save(BRAND / "founder.png", optimize=True)
            print(f"founder: {portrait.size} -> {BRAND / 'founder.png'}")

        mark = Image.open(BRAND / "mark.png")
        square = max(mark.size)
        canvas = Image.new("RGBA", (square, square), (0, 0, 0, 0))
        canvas.paste(mark, ((square - mark.width) // 2, (square - mark.height) // 2))
        touch = Image.new("RGBA", (180, 180), (255, 255, 255, 255))
        inner = canvas.resize((140, 140), Image.LANCZOS)
        touch.alpha_composite(inner, (20, 20))
        touch.convert("RGB").save(PUBLIC / "apple-touch-icon.png", optimize=True)
        canvas.resize((48, 48), Image.LANCZOS).save(PUBLIC / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
        print("favicon.ico and apple-touch-icon.png written")


if __name__ == "__main__":
    main()
