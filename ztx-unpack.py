#!/usr/bin/env python

import argparse
import json
from pathlib import Path
import zlib

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("output_template", type=str)
parser.add_argument("--report", type=Path)
args = parser.parse_args()

# load palette
pal_bytes = args.input.with_suffix(".pal").read_bytes()
pal = np.frombuffer(pal_bytes, dtype="byte").reshape((-1, 3))

compressed = args.input.read_bytes()[4:]
unpacked = zlib.decompress(compressed)

num_textures = unpacked[0]

for i in range(num_textures):
    W = 128
    H = 128
    offset = 1 + W * H * i
    indexes = np.frombuffer(unpacked[offset:offset + W * H], dtype="uint8").reshape((H, W))
    indexes = np.flip(indexes, axis=0)
    rgb = np.zeros_like(pal, shape=(indexes.shape[0], indexes.shape[1], 3))
    np.take(pal, indexes, out=rgb, axis=0)
    img = Image.frombytes("RGB", (indexes.shape[1], indexes.shape[0]), rgb)

    #misc_images[str(rel_path.with_name(rel_path.stem + f"_{i}.png"))] = img
    output_path = Path(args.output_template % i)
    img.save(output_path)

if args.report is not None:
    args.report.write_text(json.dumps(dict(num_textures=num_textures)))
