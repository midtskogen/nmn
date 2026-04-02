#!/usr/bin/env python3

# Usage map.py stars.txt stars.id

import ephem
import sys
from datetime import datetime, UTC
import math
import os
import configparser
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

parser = argparse.ArgumentParser(description='List the brightest visible stars.')
config = configparser.ConfigParser()
config.read(['/etc/meteor.cfg', os.path.expanduser('~/meteor.cfg')])

parser.add_argument('-x', '--longitude', dest='longitude', help='observer longitude', type=float)
parser.add_argument('-y', '--latitude', dest='latitude', help='observer latitude', type=float)
parser.add_argument('-e', '--elevation', dest='elevation', help='observer elevation (m)', type=float)
parser.add_argument('-t', '--temperature', dest='temperature', help='observer temperature (C)', type=float)
parser.add_argument('-p', '--pressure', dest='pressure', help='observer air pressure (hPa)', type=float)
parser.add_argument(action='store', dest='stars', help='stars.txt file')
parser.add_argument(action='store', dest='id', help='stars.id file')

args = parser.parse_args()


def altaz(timestamp, ra, dec):
    pos = ephem.Observer()
    pos.lat = config.get('astronomy', 'latitude')
    pos.lon = config.get('astronomy', 'longitude')
    pos.elevation = float(config.get('astronomy', 'elevation'))
    pos.temp = float(config.get('astronomy', 'temperature'))
    pos.pressure = float(config.get('astronomy', 'pressure'))
    pos.date = datetime.fromtimestamp(timestamp, UTC).strftime('%Y-%m-%d %H:%M:%S')

    if args.longitude:
        pos.lon = str(args.longitude)
    if args.latitude:
        pos.lat = str(args.latitude)
    if args.elevation:
        pos.elevation = args.elevation
    if args.temperature:
        pos.temp = args.temperature
    if args.pressure:
        pos.pressure = args.pressure
    star = ephem.FixedBody()
    star._ra, star._dec, star._epoch = ra, dec, ephem.J2000
    star.compute(pos)
    
    return math.degrees(float(repr(star.az))), math.degrees(float(repr(star.alt)))

with open(args.stars) as f:
    cols = len(f.readline().split())
rows = 1 + sum(1 for line in open(args.stars))

with open(args.id) as id:
    idlines = id.readlines()
    for col in range(1, cols):
        if col > len(idlines):
            continue
        with open(args.stars) as track:
            for row in range(1, rows):
                t = track.readline().split()
                pos = t[col]
                s = idlines[col-1].split()
                try:
                    float(pos.split(',')[0])
                except ValueError:
                    continue
                if len(s) >= 4:
                    az, alt = altaz(float(t[0]), s[1], s[2])
                    print(pos.replace(',', ' '), az, alt)
