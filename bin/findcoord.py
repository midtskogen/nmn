#!/usr/bin/env python3

# Find ra and dec given star name

# Example:
# awk '{print("echo "$1" $(~/bin/findcoord.py \""$2"\" 2> /dev/null) "$2" "$3" "$4" "$5)}' stars.id.orig | bash > stars.id

import sys
from stars import cat
import argparse
import difflib

from pathlib import Path

# Ensure local project modules are importable even when this script is executed via symlink
_SCRIPT_PATH = Path(__file__).resolve()
_PROJECT_DIR = None
for _cand in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
    if (_cand / 'bin').is_dir() and (_cand / 'server').is_dir():
        _PROJECT_DIR = _cand
        break
if _PROJECT_DIR is not None:
    _BIN_DIR = _PROJECT_DIR / 'bin'
    _SRC_DIR = _PROJECT_DIR / 'src'
    for _p in (_BIN_DIR, _SRC_DIR, _PROJECT_DIR):
        if _p.exists():
            _ps = str(_p)
            if _ps not in sys.path:
                sys.path.insert(0, _ps)

parser = argparse.ArgumentParser(description='Find ra and dec given star name.')

parser.add_argument(action='store', dest='star', help="star name")
parser.add_argument('-m', '--magnitude', action='store_const', dest='mag', const=sum, default=0, help="print magnitude instead")

args = parser.parse_args()

match = difflib.get_close_matches(args.star, [x[5] for x in cat])[0]

for (ra2, p_ra2, dec2, p_dec2, mag, name) in cat:
    if match == name:
        if args.mag:
            print(mag)
        else:
            print(ra2, dec2)
        break
