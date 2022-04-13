#!/usr/bin/env python

import argparse
import csv
import json
from pathlib import Path
import struct


parser = argparse.ArgumentParser()
parser.add_argument("models_idx", type=Path)
parser.add_argument("models_huge", type=Path)
parser.add_argument("models_txt", type=Path)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()


models = []

# parse models.txt
with open(args.models_txt, "rt") as f:
    reader = csv.reader(f, delimiter=" ")
    for id, unk1, unk2, unk3, filename in reader:
        id = int(id)
        assert id == len(models)

        models.append(dict(index=id, name=filename))


with open(args.models_idx, "rb") as f:
    count, = struct.unpack("<I", f.read(4))
    assert count == len(models)

    for i in range(count):
        models[i]["file_offset"], models[i]["file_length"] = struct.unpack("<II", f.read(8))

    assert not f.read(1)


args.report.write_text(json.dumps(models, indent=2))
