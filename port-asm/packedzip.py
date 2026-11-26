#!/usr/bin/env python3
"""Compress/decompress emitted packed.pt artifacts and print deterministic checksums.

The emitter's torch.save zip stores dense bf16 tensors uncompressed. The assembly-small circuit is
extremely sparse/structured, so even zlib level 1 currently reduces a 3.89 GiB artifact to about
32 MiB. This utility uses gzip format for streaming compatibility with standard tools.
"""
import argparse
import gzip
import hashlib
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['compress', 'decompress'])
    ap.add_argument('src')
    ap.add_argument('dst')
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if a.mode == 'compress':
        with open(src, 'rb') as fi, gzip.GzipFile(filename='', mode='wb', fileobj=open(dst, 'wb'), compresslevel=1, mtime=0) as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
    else:
        with gzip.open(src, 'rb') as fi, open(dst, 'wb') as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
    print(f'{dst} bytes={dst.stat().st_size} sha256={digest(dst)}')


if __name__ == '__main__':
    main()
