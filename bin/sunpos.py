#!/usr/bin/env python3

# Usage: sunpos.py
# Returns current sun position

import ephem
from datetime import datetime, UTC
import math
import os
import configparser

import sys
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

pos = ephem.Observer()

config = configparser.ConfigParser()
config.read(['/etc/meteor.cfg', os.path.expanduser('~/meteor.cfg')])

pos.lat = config.get('astronomy', 'latitude')
pos.lon = config.get('astronomy', 'longitude')
pos.elevation = float(config.get('astronomy', 'elevation'))
pos.temp = float(config.get('astronomy', 'temperature'))
pos.pressure = float(config.get('astronomy', 'pressure'))
pos.date = datetime.now(UTC)

body = ephem.Sun()
body.compute(pos)

print(math.degrees(float(repr(body.az))), math.degrees(float(repr(body.alt))))
