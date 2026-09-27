"""Contact sheet of texture PNGs (dirty retail dumps or clean previews), labelled by frame id.

    python -m games.hm64.sheet <png dir> <id substring,...> <out.png> [--scale 2] [--width 1600] [--bg 80]
"""
import argparse
import os

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("match")
    ap.add_argument("out")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--bg", type=int, default=80)
    ap.add_argument("--max", type=int, default=400)
    a = ap.parse_args()
    keys = a.match.split(",")
    files = sorted(f for f in os.listdir(a.dir) if f.endswith(".png") and any(k in f for k in keys))[:a.max]
    ims = []
    for f in files:
        im = Image.open(os.path.join(a.dir, f)).convert("RGBA")
        im = im.resize((im.width * a.scale, im.height * a.scale), Image.NEAREST)
        ims.append((f[:-4].split("__")[-2][-14:] + "/" + f[:-4].split("__")[-1], im))
    x = y = rowh = 0
    pos = []
    for lab, im in ims:
        w = max(im.width, 60)
        if x + w > a.width:
            x, y, rowh = 0, y + rowh + 14, 0
        pos.append((x, y))
        x += w + 4
        rowh = max(rowh, im.height)
    sheet = Image.new("RGBA", (a.width, y + rowh + 14), (a.bg, a.bg, a.bg, 255))
    d = ImageDraw.Draw(sheet)
    for (lab, im), (px, py) in zip(ims, pos):
        d.text((px, py), lab, fill=(255, 255, 0, 255))
        sheet.alpha_composite(im, (px, py + 12))
    sheet.convert("RGB").save(a.out)
    print(f"{len(ims)} images -> {a.out} ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
