"""Clean ROM = kept-facts base image + clean patch.

    python -m games.hm64.pack base <retail.z64> <spec dir> <base.z64>     (DIRTY ROOM, once)
    python -m games.hm64.pack rom <base.z64> <patch> <out.z64>             (clean)

`base` zeroes every range the spec regenerates (textures, palettes, font, samples, books, loop
states); what is left is the decomp's matching build output: code, data, text and cutscene/dialogue
bytecode, note sequences, map geometry, sprite headers/animation metadata, IPL3 and RSP microcode.
"""
import json
import os
import struct
import sys


def ranges(spec_dir):
    t = json.load(open(os.path.join(spec_dir, "textures.json")))
    for x in t["textures"]:
        yield x["off"], x["n"]
    for p in t["palettes"]:
        yield p["off"], 2 * p["n"]
    f = t["font"]
    yield f["off"], f["end"] - f["off"]
    for p in f["palettes"]:
        yield p["off"], 32
    s = json.load(open(os.path.join(spec_dir, "samples.json")))
    for x in s["samples"]:
        yield x["data"], x["n"]
        yield x["book"], 8 + 2 * x["order"] * x["npred"] * 8
        if "loop" in x:
            yield x["loop"]["off"] + 12, 32


def read_patch(path):
    b = open(path, "rb").read()
    pos = 0
    while pos < len(b):
        off, n = struct.unpack(">II", b[pos:pos + 8])
        yield off, b[pos + 8:pos + 8 + n]
        pos += 8 + n


def main():
    cmd = sys.argv[1]
    if cmd == "base":
        rom = bytearray(open(sys.argv[2], "rb").read())
        tot = 0
        for off, n in ranges(sys.argv[3]):
            rom[off:off + n] = bytes(n)
            tot += n
        open(sys.argv[4], "wb").write(rom)
        print(f"base: zeroed {tot / 1e6:.2f} MB of regenerated ranges -> {sys.argv[4]}")
    elif cmd == "rom":
        rom = bytearray(open(sys.argv[2], "rb").read())
        n = 0
        for off, b in read_patch(sys.argv[3]):
            rom[off:off + len(b)] = b
            n += len(b)
        open(sys.argv[4], "wb").write(rom)
        print(f"rom: wrote {n / 1e6:.2f} MB of clean assets -> {sys.argv[4]}")


if __name__ == "__main__":
    main()
