"""DIRTY ROOM: read the retail ROM and write the texture spec (facts only).

    python -m games.hm64.extract_spec <rom.z64> <pristine decomp> <spec dir> [--png <dirty png dir>]

Slots come from the decomp's own address tables (tools/libhm64): entity/overlay/effect/map-object
sprites, map tile + core-object textures, the dialogue font. Per texture we keep only
format, size, byte range, which palette it uses, a colour grid (4x4, 16x16 when >= 128 px) and the
2-bit alpha outline. Palettes: byte range and colour count only (colours are regenerated).
--png writes decoded retail images for dirty-room contact sheets (never committed or published).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def rgba5551(b):
    v = np.frombuffer(b[:len(b) // 2 * 2], ">u2").astype(np.uint32)
    out = np.empty((len(v), 4), np.uint8)
    out[:, 0] = ((v >> 11) & 31) * 255 // 31
    out[:, 1] = ((v >> 6) & 31) * 255 // 31
    out[:, 2] = ((v >> 1) & 31) * 255 // 31
    out[:, 3] = (v & 1) * 255
    return out


def pal_mean(b):
    """Mean colour of a palette's opaque entries (a coarse fact, like a 1x1 grid)."""
    c = rgba5551(b)
    op = c[c[:, 3] > 0]
    return [int(v) for v in (op if len(op) else c)[:, :3].mean(0)]


def indices(data, w, h, fmt):
    b = np.frombuffer(data, np.uint8)
    if fmt == "ci4":
        n = (w * h + 1) // 2
        b = b[:n]
        idx = np.empty(len(b) * 2, np.uint8)
        idx[0::2], idx[1::2] = b >> 4, b & 15
    else:
        idx = b[:w * h].copy()
    if len(idx) < w * h:
        idx = np.concatenate([idx, np.zeros(w * h - len(idx), np.uint8)])
    return idx[:w * h].reshape(h, w)


def grid(rgba, n):
    """Mean colour per cell over opaque texels (so the kept colour is the sprite's, not the background's)."""
    h, w = rgba.shape[:2]
    out = []
    for gy in range(n):
        for gx in range(n):
            y0, y1 = gy * h // n, max(gy * h // n + 1, (gy + 1) * h // n)
            x0, x1 = gx * w // n, max(gx * w // n + 1, (gx + 1) * w // n)
            c = rgba[y0:y1, x0:x1].reshape(-1, 4).astype(np.float32)
            op = c[c[:, 3] >= 128]
            rgb = (op if len(op) else c)[:, :3].mean(0)
            out.append([int(round(v)) for v in rgb] + [int(round(c[:, 3].mean()))])
    return out


def alpha2(a):
    a = (a.astype(np.uint8) >> 6).ravel()
    a = np.concatenate([a, np.zeros((-len(a)) % 4, np.uint8)])
    return ((a[0::4] << 6) | (a[1::4] << 4) | (a[2::4] << 2) | a[3::4]).astype(np.uint8).tobytes().hex()


class Spec:
    def __init__(self, rom, pngdir):
        self.rom = rom
        self.tex, self.pal = [], []
        self.pngdir = pngdir
        self.pal_at = {}
        self.orphans = []

    def palette(self, pid, off, end, fmt):
        """n = the whole slot (up to 256 entries); k = entries the frames can index (16 for CI4)."""
        n = min(max(0, (end - off) // 2), 256)
        k = min(n, 16 if fmt == "ci4" else 256)
        if off in self.pal_at:
            p = self.pal[self.pal_at[off]]
            p["k"] = min(p["k"], k)
            return self.pal_at[off]
        self.pal.append(dict(id=pid, off=off, n=n, k=k))
        self.pal_at[off] = len(self.pal) - 1
        return len(self.pal) - 1

    def texture(self, tid, hdr, end, pal_index, group):
        rom = self.rom
        flags = rom[hdr + 3]
        w = int.from_bytes(rom[hdr + 4:hdr + 6], "little", signed=True)
        h = int.from_bytes(rom[hdr + 6:hdr + 8], "little", signed=True)
        fmt = "ci4" if flags == 0x10 else "ci8"
        off = hdr + 8
        if w <= 0 or h <= 0 or end <= off:
            return
        need = (w * h + 1) // 2 if fmt == "ci4" else w * h
        n = end - off if end - off < 0x40000 else need      # the whole slot, padding included
        d = dict(id=tid, group=group, off=off, n=n, w=w, h=h, fmt=fmt, pal=pal_index)
        if pal_index is not None:
            p = self.pal[pal_index]
            cols = rgba5551(rom[p["off"]:p["off"] + 2 * p["k"]])
            idx = indices(rom[off:off + n], w, h, fmt)
            idx = np.minimum(idx, len(cols) - 1)
            img = cols[idx]
            gn = 16 if max(w, h) >= 128 else 4
            d["grid"] = grid(img, gn)
            if (img[..., 3] < 250).any():
                d["alpha2"] = alpha2(img[..., 3])
            if self.pngdir:
                from cleanroom.gfx import png
                png.write(os.path.join(self.pngdir, tid.replace("/", "__") + ".png"), img)
        else:
            idx = indices(rom[off:off + n], w, h, fmt)
            scale = 17 if fmt == "ci4" else 1
            img = np.repeat((idx * scale).astype(np.uint8)[..., None], 4, 2)
            img[..., 3] = 255
            d["grid"] = grid(img, 16 if max(w, h) >= 128 else 4)
        self.tex.append(d)


def sprites(S, lib):
    from libhm64.sprites import addresses as A
    rom = S.rom
    u32 = lambda o: int.from_bytes(rom[o:o + 4], "big")
    stats = dict(assets=0, simple=0)
    for info in A.get_all_sprites():
        base = info.addr_base
        try:
            ss = A.get_spritesheet_offsets(info)
        except Exception as e:  # noqa
            print("  skip", info.label, e)
            continue
        cnt = len(ss) - 1
        while cnt > 0 and ss[cnt] == 0:
            cnt -= 1
        ao = A.get_asset_offsets(info.addr_index)
        group = f"sprite/{info.subdir}/{info.label}"
        stats["assets"] += 1
        pal_offsets, s2p = [], []
        first = u32(base + ao[1]) if ao[1] else 0
        if ao[1] and first and first <= 0x1000 and first % 4 == 0:
            for i in range(first // 4):
                o = u32(base + ao[1] + i * 4)
                if o > 0x100000:
                    break
                pal_offsets.append(o)
            s2p_size = ao[4] - ao[3]
            if s2p_size > 4 and s2p_size - 4 < cnt:
                cnt = s2p_size - 4
            s2p = list(A.get_sprite_to_palette_map(base, ao, cnt))
        else:
            stats["simple"] += 1
        pal_map = {}
        for pi in range(len(pal_offsets) - 1):
            o, nx = pal_offsets[pi], pal_offsets[pi + 1]
            if nx == 0 or o == nx:
                continue
            st, en = base + ao[1] + o + 4, base + ao[1] + nx - 4
            if en <= st or en - st > 512:
                continue
            pal_map[pi] = (st, en)
        for si in range(cnt, len(ss) - 1):
            # frames past the sprite-to-palette table: unused by the game, but their pixels are in the
            # ROM, so they are regenerated too (with the last frame's palette)
            if 0 < ss[si] < ss[si + 1] and 8 < ss[si + 1] - ss[si] < 0x20000:
                S.orphans.append(dict(id=f"{group}/{si:03d}", hdr=base + ss[si], end=base + ss[si + 1]))
        for si in range(cnt):
            o, nx = ss[si], ss[si + 1]
            if o == nx or nx == 0 and si + 1 < len(ss):
                if o == nx:
                    continue
            hdr, end = base + o, base + nx
            pi = s2p[si] if si < len(s2p) else None
            fmt = "ci4" if rom[hdr + 3] == 0x10 else "ci8"
            pidx = None
            if pi is not None and pi in pal_map:
                pidx = S.palette(f"{group}/pal{pi}", pal_map[pi][0], pal_map[pi][1], fmt)
            S.texture(f"{group}/{si:03d}", hdr, end, pidx, group)
        # palettes no frame references statically (the game switches to them at runtime): record them too,
        # with the nearest referenced palette of the same size and both mean colours (coarse facts)
        ref = {pi: S.pal_at[st] for pi, (st, en) in pal_map.items() if st in S.pal_at}
        for pi, (st, en) in sorted(pal_map.items()):
            if st in S.pal_at:
                continue
            n = min((en - st) // 2, 256)
            sib = [(abs(pj - pi), idx) for pj, idx in ref.items() if S.pal[idx]["n"] == n]
            if not sib:
                sib = [(abs(pj - pi), idx) for pj, idx in ref.items()]
            if not sib:
                continue
            sidx = min(sib)[1]
            S.pal.append(dict(id=f"{group}/pal{pi}", off=st, n=n, sibling=sidx,
                              mean=pal_mean(rom[st:st + 2 * n]),
                              sibling_mean=pal_mean(rom[S.pal[sidx]["off"]:S.pal[sidx]["off"] + 2 * S.pal[sidx]["n"]])))
            S.pal_at[st] = len(S.pal) - 1
    return stats


def maps(S, lib):
    from libhm64.maps import addresses as M
    rows = M.get_all_map_addresses()
    n = 0
    for ri, row in enumerate(rows):
        if M.is_placeholder_map(ri, rows):
            continue
        name, mb = row[1], int(row[0], 16)
        ao = M.get_asset_offsets_array(row)
        for kind, ti, pi in (("tile", 5, 6), ("obj", 7, 8)):
            if ao[ti] == 0:
                continue
            try:
                to = M.get_texture_offsets_array(mb, ao[ti])
                po = M.get_palette_offsets_array(mb, ao[pi])
            except Exception as e:  # noqa
                print("  skip", name, kind, e)
                continue
            group = f"map/{name}/{kind}"
            for i in range(len(to) - 1):
                ts, te = mb + ao[ti] + to[i], mb + ao[ti] + to[i + 1]
                if i + 1 >= len(po):
                    break
                ps, pe = mb + ao[pi] + po[i] + 4, mb + ao[pi] + po[i + 1]
                fmt = "ci4" if S.rom[ts + 3] == 0x10 else "ci8"
                pidx = S.palette(f"{group}/pal{i}", int(ps), int(pe), fmt)
                S.texture(f"{group}/{i:03d}", int(ts), int(te), pidx, group)
                n += 1
    return n


def font_glyph(rom, off, i):
    b = rom[off + i * 64: off + i * 64 + 64]
    px = []
    for r in range(14):
        row = []
        for k in (3, 2, 1, 0):
            v = b[4 * r + k]
            row += [v >> 6, (v >> 4) & 3, (v >> 2) & 3, v & 3]
        px.append(row[:14])
    return np.array(px)


def font_boxes(rom, font):
    """Per glyph: ink bounding box [x0, y0, x1, y1] (texels with coverage index >= 2), or None."""
    out = []
    for i in range(font["cells"]):
        g = font_glyph(rom, font["off"], i)
        ys, xs = np.nonzero(g >= 2)
        out.append(None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return out


DIRS = ["S", "SW", "W", "NW", "N", "NE", "E", "SE"]
MIRROR = {"S": "S", "SW": "SE", "W": "E", "NW": "NE", "N": "N", "NE": "NW", "E": "W", "SE": "SW"}


def facing(rom, spec_dir):
    """Per entity sprite frame: the direction it faces (from the game's direction-grouped animation
    scripts, flips undone) and whether it is a body (first bitmap of a frame) or an overlaid part."""
    import csv
    from libhm64.sprites import addresses as A
    from libhm64.animations.metadata import read_animation_from_offsets
    from libhm64.data import ANIMATION_SCRIPTS_ADDRESSES_CSV
    out = {}
    for row in csv.reader(open(ANIMATION_SCRIPTS_ADDRESSES_CSV)):
        if len(row) < 3:
            continue
        st, en, label = int(row[0], 16), int(row[1], 16), row[2].strip()
        info = A.get_sprite_by_label(label)
        if info is None:
            continue
        ao = A.get_asset_offsets(info.addr_index)
        try:
            anim_offsets = A.get_animation_offsets(info.addr_base, ao)
        except Exception:  # noqa
            continue
        u16 = np.frombuffer(rom[st:en], ">u2")
        g = out.setdefault(f"sprite/{info.subdir}/{info.label}", {})
        for i, v in enumerate(u16.tolist()):
            if v == 0:
                continue
            meta, flip = v & 0x1FFF, bool(v & 0x8000)
            d = DIRS[i % 8]
            d = MIRROR[d] if flip else d
            try:
                an = read_animation_from_offsets(info.addr_base + ao[2], anim_offsets, meta)
            except Exception:  # noqa
                continue
            for fr in an.get("frames", []):
                for k, b in enumerate(fr["sprites"]):
                    e = g.setdefault(str(b["spritesheet_index"]), {"dir": {}, "body": 0, "part": 0})
                    e["dir"][d] = e["dir"].get(d, 0) + (0 if flip else 2) + 1
                    e["body" if k == 0 else "part"] += 1
    for g in out.values():
        for k, e in g.items():
            g[k] = {"dir": max(e["dir"], key=e["dir"].get), "role": "body" if e["body"] >= e["part"] else "part"}
    json.dump(out, open(os.path.join(spec_dir, "facing.json"), "w"), separators=(",", ":"))
    n = sum(len(g) for g in out.values())
    print(f"facing: {len(out)} sprite sets, {n} frames with a direction")


PTR, DATA, DATA_END =0xE93080, 0xE99910, 0xFD8510


def audio(rom, spec_dir):
    """Sample bank facts: per instrument the byte ranges, length, loop, book shape, a coarse spectral
    outline (descriptor) and the median pitch. No sample data."""
    from libhm64.audio.wavetable import container, adpcm
    from cleanroom.audio import descriptor
    from cleanroom.audio.pitch import median_f0
    pb, db = rom[PTR:DATA], rom[DATA:DATA_END]
    bank = container.parse(pb, db)
    u32 = lambda o: int.from_bytes(pb[o:o + 4], "big")
    wl, count = u32(0x2C), u32(0x20)
    out = []
    for i in range(count):
        ip = u32(wl + i * 4)
        if ip == 0:
            continue
        base, ln, loop_off, book_off = u32(ip), u32(ip + 4), u32(ip + 0xC), u32(ip + 0x10)
        ins = bank.instruments[i]
        cb = ins.sound.codebook
        pcm = np.asarray(adpcm.decode(ins.sound.adpcm_bytes, cb), np.float64)
        rate = 22050
        d = dict(i=i, data=DATA + base, n=ln, nframes=len(pcm), rate=rate, book=PTR + book_off,
                 order=cb.order, npred=cb.npredictors, basenote=ins.basenote, detune=ins.detune,
                 desc=descriptor.describe(pcm, rate))
        if loop_off:
            lp = ins.sound.loop
            d["loop"] = dict(off=PTR + loop_off, start=lp.start, end=lp.end, count=lp.count)
        f0 = median_f0((pcm / 32768).astype(np.float32), rate) if len(pcm) > 512 else None
        if f0:
            d["f0"] = round(f0, 1)
        out.append(d)
    json.dump(dict(ptr=[PTR, DATA], data=[DATA, DATA_END], samples=out),
              open(os.path.join(spec_dir, "samples.json"), "w"), separators=(",", ":"))
    books = {}
    for d in out:
        books[(d["order"], d["npred"])] = books.get((d["order"], d["npred"]), 0) + 1
    print(f"samples: {len(out)} of {count} slots, {sum(d['n'] for d in out) / 1e6:.2f} MB ADPCM, "
          f"{sum('loop' in d for d in out)} looped; books (order,npred): {books}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rom")
    ap.add_argument("pristine")
    ap.add_argument("spec")
    ap.add_argument("--png", default="")
    ap.add_argument("--only", default="", help="audio: only the sample bank")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.pristine, "tools"))
    import libhm64  # noqa
    from libhm64.common import rom as R
    R.set_rom_path(a.rom)
    rom = bytes(R.get_rom())
    os.makedirs(a.spec, exist_ok=True)
    if a.png:
        os.makedirs(a.png, exist_ok=True)
    if a.only == "audio":
        return audio(rom, a.spec)
    if a.only == "facing":
        return facing(rom, a.spec)
    S = Spec(rom, a.png)
    st = sprites(S, libhm64)
    nm = maps(S, libhm64)
    # dialogue font: 700 cells, 16x16 CI2 (drawn 14x14), 3 palettes of 16 colours
    font = dict(off=0xE08870, end=0xE13770, cells=700, cell_bytes=64,
                palettes=[dict(off=o + 4, n=16) for o in (0xE13770, 0xE137A0, 0xE137D0)])
    font["boxes"] = font_boxes(rom, font)
    out = dict(textures=S.tex, palettes=S.pal, font=font, orphans=S.orphans)
    json.dump(out, open(os.path.join(a.spec, "textures.json"), "w"), separators=(",", ":"))
    audio(rom, a.spec)
    facing(rom, a.spec)
    tb = sum(t["n"] for t in S.tex)
    pb = sum(2 * p["n"] for p in S.pal)
    print(f"sprites: {st['assets']} assets ({st['simple']} without palettes); map textures: {nm}")
    print(f"textures: {len(S.tex)} ({tb / 1e6:.2f} MB), palettes: {len(S.pal)} ({pb / 1e3:.1f} KB); "
          f"no palette: {sum(t['pal'] is None for t in S.tex)}")
    fm = {}
    for t in S.tex:
        fm[t["fmt"]] = fm.get(t["fmt"], 0) + 1
    print(f"runtime-only palettes: {sum('sibling' in p for p in S.pal)}")
    print(f"orphan frames (past the palette map): {len(S.orphans)}, {sum(o['end'] - o['hdr'] for o in S.orphans) / 1e3:.1f} KB")
    print("formats:", fm, "; largest:", sorted({(t["w"], t["h"]) for t in S.tex}, key=lambda s: -s[0] * s[1])[:6])


if __name__ == "__main__":
    main()
