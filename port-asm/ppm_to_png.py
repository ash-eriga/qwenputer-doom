#!/usr/bin/env python3
"""Tiny PPM(P6)->PNG converter, nearest-neighbor upscale, no external deps."""
import struct
import sys
import zlib
from pathlib import Path


def read_ppm(path):
    b = Path(path).read_bytes()
    parts = []
    i = 0
    while len(parts) < 4:
        while b[i:i + 1].isspace():
            i += 1
        if b[i:i + 1] == b'#':
            while b[i:i + 1] != b'\n':
                i += 1
            continue
        j = i
        while not b[j:j + 1].isspace():
            j += 1
        parts.append(b[i:j])
        i = j
    i += 1
    w, h, maxv = map(int, parts[1:4])
    if parts[0] != b'P6' or maxv != 255:
        raise ValueError('expected binary P6 PPM with maxval 255')
    return w, h, b[i:i + w * h * 3]


def write_png(path, w, h, rgb, scale=1):
    W, H = w * scale, h * scale
    rows = []
    for y in range(h):
        row = bytearray()
        src = rgb[y * w * 3:(y + 1) * w * 3]
        for x in range(w):
            row += src[x * 3:x * 3 + 3] * scale
        for _ in range(scale):
            rows.append(b'\0' + bytes(row))

    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data +
                struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))

    raw = b''.join(rows)
    png = (b'\x89PNG\r\n\x1a\n' +
           chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 8, 2, 0, 0, 0)) +
           chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
    Path(path).write_bytes(png)


if __name__ == '__main__':
    inp, out = sys.argv[1], sys.argv[2]
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    w, h, rgb = read_ppm(inp)
    write_png(out, w, h, rgb, scale)
