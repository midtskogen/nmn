#!/usr/bin/env python3

# Usage: altaz.py [-x longitude] [-y latitude] <Unix time> <RA> <Dec>
# Compute local azimuth and altitude of an object at RA/Dec (2000 epoch)
# example: altaz.py -x 10.649639 -y 59.97056 $(date +%s -u -d "2014-09-30 03:03:33") 4.5986944 16.5085277
#          altaz.py -x 10.649639 -y 59.97056 $(date +%s -u -d "2014-09-30 03:03:33") 4:35:55.237 16:30:33.39
#  Compute position of Aldebaran (alt 46.490 az 174.612) at 2014-09-30 03:03:33UT

import ephem
from datetime import datetime, UTC
import math
import configparser
import argparse
import os

parser = argparse.ArgumentParser(description='Compute local altitude and azimuth of an object at RA/Dec (2000 epoch).')
parser.add_argument('-x', '--longitude', dest='longitude', help='observer longitude')
parser.add_argument('-y', '--latitude', dest='latitude', help='observer latitude')
parser.add_argument(action='store', dest='timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00UTC)')
parser.add_argument(action='store', dest='ra', help='RA')
parser.add_argument(action='store', dest='dec', help='dec')
args = parser.parse_args()

#if (len(sys.argv) != 4):
#    print()
#    sys.exit(0)

pos = ephem.Observer()

config = configparser.ConfigParser()
config.read(['/etc/meteor.cfg', os.path.expanduser('~/meteor.cfg')])
pos.lat = args.latitude if args.latitude else config.get('astronomy', 'latitude')
pos.lon = args.longitude if args.longitude else config.get('astronomy', 'longitude')
pos.elevation = float(config.get('astronomy', 'elevation'))
pos.temp = float(config.get('astronomy', 'temperature'))
pos.pressure = float(config.get('astronomy', 'pressure'))
pos.date = datetime.fromtimestamp(float(args.timestamp), UTC).strftime('%Y-%m-%d %H:%M:%S')

star = ephem.FixedBody()
star._ra, star._dec, star._epoch = args.ra, args.dec, ephem.J2000
star.compute(pos)

print(math.degrees(float(repr(star.az))), math.degrees(float(repr(star.alt))))
