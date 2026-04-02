#!/usr/bin/env python3

# Usage: sunpos.py
# Returns current sun position

import ephem
from datetime import datetime, UTC
import math
import os
import configparser

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
