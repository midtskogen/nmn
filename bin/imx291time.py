#!/usr/bin/env python3

from dvrip import DVRIPCam

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

ips = [ "192.168.76.71", "192.168.76.72", "192.168.76.73", "192.168.76.74", "192.168.76.75", "192.168.76.76", "192.168.76.77" ]

for ip in ips:
    try:
        cam = DVRIPCam(ip, user='admin', password='')
        cam.login()
        cam.set_time()
    except:
        pass
