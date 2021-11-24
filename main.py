import argparse
import logging
import os
from pathlib import Path
import warnings

from output import generate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("gamedir", type=Path)
parser.add_argument("output_dir", type=Path)
args = parser.parse_args()

all_content = []

unrecognized_paths = []

for root, subdirs, files in os.walk(args.gamedir):
    for filename in files:
        relative_path = (Path(root) / filename).relative_to(args.gamedir)
        all_content.append(relative_path)
        # print(relative_path)

all_content.sort()

for path in all_content:
    if path.suffix == ".s":
        warnings.warn("Script files (.s) not handled")
        continue

    unrecognized_paths.append(path)

logger.info("Initial processing finished, generating outputs.")

generate(args.output_dir, unrecognized_paths=unrecognized_paths)
