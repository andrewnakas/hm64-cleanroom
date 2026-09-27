"""CLEAN ROOM: the dialogue font (700 cells of 14x14 CI2, 3 palettes of 16 colours).

Latin letters, digits and symbols are drawn with our own stroke font, each fitted into the kept ink
box of its cell (so the game's width table still spaces them). The Japanese kana/kanji cells are not
used by the US text and stay empty. Coverage maps to the game's index roles: 0 clear, 1-2 edge
shades, 3 ink. Palettes are our own ramps in the three roles (brown ink, plum ink, light ink).
"""
import math
import os
import struct
import sys

import numpy as np

from cleanroom.gfx import strokefont

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = 14
FIRST = 0x0B          # charmap code of cell 0

# design-grid extras the shared stroke font lacks (4 x 6 grid, y down, baseline y = 6)
EXTRA = {
    "~": [[(0, 3.4), (1, 2.6), (3, 3.4), (4, 2.6)]],
    ";": [[(2, 2), (2, 2.3)], [(2, 5), (2, 6), (1.3, 7)]],
    "…": [[(0.3, 6), (0.3, 6.1)], [(2, 6), (2, 6.1)], [(3.7, 6), (3.7, 6.1)]],
    "・": [[(2, 3), (2, 3.2)]],
    "“": [[(1, 0), (1.5, 1.4)], [(2.6, 0), (3.1, 1.4)]],
    "”": [[(1.4, 0), (0.9, 1.4)], [(3, 0), (2.5, 1.4)]],
    "♥": [[(2, 5.6), (0.2, 3), (0.2, 1.4), (1, 0.8), (2, 1.6), (3, 0.8), (3.8, 1.4), (3.8, 3), (2, 5.6)],
          [(2, 4.4), (1, 3), (1, 2), (2, 2.6), (3, 2), (3, 3), (2, 4.4)]],
    "♡": [[(2, 5.6), (0.2, 3), (0.2, 1.4), (1, 0.8), (2, 1.6), (3, 0.8), (3.8, 1.4), (3.8, 3), (2, 5.6)]],
    "★": [[(2, 0), (2.6, 2), (4, 2.2), (2.9, 3.4), (3.3, 5.6), (2, 4.4), (0.7, 5.6), (1.1, 3.4), (0, 2.2),
           (1.4, 2), (2, 0)], [(2, 1.5), (2, 4)], [(1.3, 3), (2.7, 3)]],
    "☆": [[(2, 0), (2.6, 2), (4, 2.2), (2.9, 3.4), (3.3, 5.6), (2, 4.4), (0.7, 5.6), (1.1, 3.4), (0, 2.2),
           (1.4, 2), (2, 0)]],
    "♪": [[(1, 5), (0.3, 5.5), (0.6, 6.2), (1.4, 5.8), (1.4, 0.5), (3.6, 1.6)]],
    "×": [[(0.5, 1.5), (3.5, 4.5)], [(3.5, 1.5), (0.5, 4.5)]],
    "○": [[(2, 0.8), (3.4, 1.6), (3.8, 3.2), (3.2, 4.8), (2, 5.4), (0.8, 4.8), (0.2, 3.2), (0.6, 1.6), (2, 0.8)]],
    "¥": [[(0, 0), (2, 3), (4, 0)], [(2, 3), (2, 6)], [(0.8, 3.6), (3.2, 3.6)], [(0.8, 4.8), (3.2, 4.8)]],
    "『": [[(3, 0), (1, 0), (1, 4)], [(3, 1), (2, 1), (2, 4)]],
    "』": [[(1, 6), (3, 6), (3, 2)], [(1, 5), (2, 5), (2, 2)]],
    "「": [[(3, 0), (1, 0), (1, 4)]],
    "」": [[(1, 6), (3, 6), (3, 2)]],
    "ー": [[(0, 3), (4, 3)]],
    "－": [[(0.5, 3), (3.5, 3)]],
    "〜": [[(0, 3.4), (1, 2.6), (3, 3.4), (4, 2.6)]],
    "·": [[(2, 3), (2, 3.2)]],
    "→": [[(0, 3), (4, 3)], [(2.5, 1.5), (4, 3), (2.5, 4.5)]],
    "←": [[(0, 3), (4, 3)], [(1.5, 1.5), (0, 3), (1.5, 4.5)]],
    "↑": [[(2, 0), (2, 6)], [(0.5, 1.5), (2, 0), (3.5, 1.5)]],
    "↓": [[(2, 0), (2, 6)], [(0.5, 4.5), (2, 6), (3.5, 4.5)]],
}
EXTRA.update({
    "’": [[(2.2, 0), (2.2, 0.9), (1.4, 2)]],
    "—": [[(0, 3), (4, 3)]],
    "‥": [[(1, 6), (1, 6.1)], [(3, 6), (3, 6.1)]],
    "※": [[(0.6, 1.2), (3.4, 4.8)], [(3.4, 1.2), (0.6, 4.8)], [(2, 0.3), (2, 0.4)], [(2, 5.6), (2, 5.7)],
               [(0, 3), (0, 3.1)], [(4, 3), (4, 3.1)]],
    "℃": [[(0.5, 0.3), (1.1, 0.3), (1.1, 0.9), (0.5, 0.9), (0.5, 0.3)],
               [(4, 1.5), (3.3, 0.8), (2.4, 0.8), (1.7, 1.6), (1.7, 5), (2.4, 5.8), (3.3, 5.8), (4, 5.1)]],
    "Ა": [[(2, 0), (0.6, 3), (0.6, 4.6), (1.3, 5.6), (2.7, 5.6), (3.4, 4.6), (3.4, 3), (2, 0)]],   # drop
    "∴": [[(0.6, 1.6), (0.6, 1.8)], [(2, 0.8), (2, 1.0)], [(3.4, 1.6), (3.4, 1.8)],               # paw print
               [(1.2, 4.4), (2, 3.4), (2.8, 4.4), (2.8, 5.2), (1.2, 5.2), (1.2, 4.4)], [(2, 4.4), (2, 4.6)]],
    "↖": [[(4, 6), (0, 0)], [(0, 2.5), (0, 0), (2, 0)]],
    "↘": [[(0, 0), (4, 6)], [(4, 3.5), (4, 6), (2, 6)]],
    "↗": [[(0, 6), (4, 0)], [(2, 0), (4, 0), (4, 2.5)]],
    "↙": [[(4, 0), (0, 6)], [(0, 3.5), (0, 6), (2, 6)]],
})
CIRCLE = [(2 + 2.1 * math.cos(a / 16 * 2 * math.pi), 3 + 2.9 * math.sin(a / 16 * 2 * math.pi)) for a in range(17)]
for _k, _d in enumerate("123456789"):
    EXTRA[chr(0x2460 + _k)] = [CIRCLE] + [[(1.1 + x * 0.45, 1.65 + y * 0.45) for x, y in line]
                                           for line in strokefont.G[_d]]
DESC = {"g": 1.6, "j": 1.6, "p": 1.6, "q": 1.6, "y": 1.6}   # design units below the baseline
LOWER_DESCENDER = {  # lower-case forms with real descenders (design grid, x-height top y = 2)
    "g": [[(4, 2), (4, 7), (3, 7.6), (1, 7.6)], [(4, 3), (3, 2), (1, 2), (0, 3), (0, 5), (1, 6), (3, 6), (4, 5)]],
    "j": [[(3, 2), (3, 7), (2, 7.6), (0.5, 7.6)], [(3, 0.4), (3, 0.6)]],
    "p": [[(0, 2), (0, 7.6)], [(0, 3), (1, 2), (3, 2), (4, 3), (4, 5), (3, 6), (1, 6), (0, 5)]],
    "q": [[(4, 2), (4, 7.6)], [(4, 3), (3, 2), (1, 2), (0, 3), (0, 5), (1, 6), (3, 6), (4, 5)]],
    "y": [[(0, 2), (0, 5), (1, 6), (4, 6)], [(4, 2), (4, 7), (3, 7.6), (1, 7.6)]],
}


def charmap(pristine=None):
    pristine = pristine or os.environ.get("HM64_PRISTINE", "D:/n64work/hm64/pristine")
    sys.path.insert(0, os.path.join(pristine, "tools"))
    from libhm64.text.charmap import CHAR_MAP
    return CHAR_MAP


def design(c):
    if c in LOWER_DESCENDER:
        return LOWER_DESCENDER[c]
    if c in EXTRA:
        return EXTRA[c]
    g = strokefont.G.get(c)
    if g is None and c.isascii() and c.isalpha():
        g = strokefont.glyph(c)
    return g


def glyph(c, box, th=0.95):
    """Coverage (14x14) of c fitted into the kept ink box [x0, y0, x1, y1]."""
    lines = design(c)
    m = np.zeros((CELL, CELL), np.float32)
    if not lines or box is None:
        return m
    x0, y0, x1, y1 = box
    xs = [x for l in lines for x, _ in l]
    ys = [y for l in lines for _, y in l]
    dx0, dx1, dy0, dy1 = min(xs), max(xs), min(ys), max(ys)
    # inner edge of the stroke sits th inside the kept box
    bx0, bx1, by0, by1 = x0 + 0.5 + th * 0.6, x1 + 0.5 - th * 0.6, y0 + 0.5 + th * 0.6, y1 + 0.5 - th * 0.6
    sx = (bx1 - bx0) / (dx1 - dx0) if dx1 > dx0 else 0
    sy = (by1 - by0) / (dy1 - dy0) if dy1 > dy0 else 0
    ox = bx0 - dx0 * sx if dx1 > dx0 else (x0 + x1 + 1) / 2
    oy = by0 - dy0 * sy if dy1 > dy0 else (y0 + y1 + 1) / 2
    return strokefont._stroke(m, lines, ox, oy, sx, sy, th)


def to_index(cov):
    idx = np.zeros(cov.shape, np.uint8)
    idx[cov > 0.15] = 1
    idx[cov > 0.4] = 2
    idx[cov > 0.68] = 3
    return idx


def pack_cell(idx):
    b = bytearray(64)
    for r in range(CELL):
        row = list(idx[r]) + [0, 0]
        for k, byte_pos in enumerate((3, 2, 1, 0)):
            p = row[4 * k:4 * k + 4]
            b[4 * r + byte_pos] = (p[0] << 6) | (p[1] << 4) | (p[2] << 2) | p[3]
    return bytes(b)


def rgba5551(r, g, b, a=1):
    return ((r * 31 // 255) << 11) | ((g * 31 // 255) << 6) | ((b * 31 // 255) << 1) | a


def ramp(ink, paper):
    out = [0]
    for t in (0.35, 0.65, 1.0):
        c = [int(round(p + (i - p) * t)) for i, p in zip(ink, paper)]
        out.append(rgba5551(*c))
    return out + [0] * 12


PALETTES = [ramp((44, 34, 24), (236, 214, 176)),     # brown ink on paper
            ramp((60, 36, 52), (236, 206, 214)),     # plum ink on pink paper
            ramp((196, 196, 188), (40, 40, 40))]     # light ink on dark boxes


def cells(spec_font):
    cm = charmap()
    out = []
    for i in range(spec_font["cells"]):
        c = cm.get(i + FIRST)
        th = 0.55 if c and 0x2460 <= ord(c[0]) <= 0x2468 else 0.95     # circled digits: thin strokes
        out.append(to_index(glyph(c, spec_font["boxes"][i], th)) if c else np.zeros((CELL, CELL), np.uint8))
    return out


def build(spec_font, put):
    cs = cells(spec_font)
    put(spec_font["off"], b"".join(pack_cell(c) for c in cs))
    for p, words in zip(spec_font["palettes"], PALETTES):
        put(p["off"], struct.pack(">16H", *words))


def sheet(spec_font, path, chars=None):
    """Preview: the Latin part of the font, 4x scale."""
    from cleanroom.gfx import png
    cm = charmap()
    cs = cells(spec_font)
    pick = [i for i in range(len(cs)) if cm.get(i + FIRST, "") and cm[i + FIRST].isascii()] + \
        [i for i in range(len(cs)) if cm.get(i + FIRST, "") in EXTRA]
    cols = 24
    rows = (len(pick) + cols - 1) // cols
    img = np.full((rows * 16, cols * 16, 4), (236, 214, 176, 255), np.uint8)
    pal = [(236, 214, 176), (170, 150, 120), (110, 90, 70), (44, 34, 24)]
    for k, i in enumerate(pick):
        y, x = (k // cols) * 16 + 1, (k % cols) * 16 + 1
        for v in (1, 2, 3):
            img[y:y + 14, x:x + 14, :3][cs[i] == v] = pal[v]
    img = img.repeat(4, 0).repeat(4, 1)
    png.write(path, img)
