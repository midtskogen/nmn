#!/usr/bin/env python3

# Find a star name given ra and dec or find ra and dev given a star name

import sys
from stars import cat
import argparse

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

parser = argparse.ArgumentParser(description='Find star given ra and dec or find name given ra and dec.')

if len(sys.argv) > 2:
    parser.add_argument(action='store', dest='ra', help="ra coordinate")
    parser.add_argument(action='store', dest='dec', help="dec coordinate")
    parser.add_argument('-m', '--magnitude', action='store_const', dest='mag', const=sum, default=0, help="print magnitude instead")

    args = parser.parse_args()
    bestdist = 9999
    bestname = '?'
    bestmag = 9999

    for (ra2, p_ra2, dec2, p_dec2, mag, name) in cat:
        dist = (float(ra2)*360/24-float(args.ra)*360/24)*(float(ra2)*360/24-float(args.ra)*360/24)+(float(dec2)-float(args.dec))*(float(dec2)-float(args.dec))
        if dist < 0.2*0.2 and dist < bestdist:
            bestdist = dist
            bestname = name
            bestmag = mag

    if args.mag:
        print(bestmag)
    else:
        print(bestname)

else:
    parser.add_argument(action='store', dest='name', help="star name")
    args = parser.parse_args()
    Heap = []
    for (ra2, p_ra2, dec2, p_dec2, mag, name) in cat:
        if args.name.lower() == name.lower():
            print(str(ra2) + ' ' + str(dec2))
        



