#!/usr/bin/env python3

import math
import numpy
import configparser
import sys
import time

def tocartesian(az, alt):
    az *= math.pi / 180
    alt *= math.pi / 180
    return [math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt)]

def fromcartesian(p):
    return (math.fmod(math.atan2(p[1], p[0]) * 180/math.pi + 360, 360),
            math.asin(p[2]) * 180/math.pi)

obs = configparser.ConfigParser()
config = configparser.ConfigParser()
obs.read([sys.argv[1]])
config.read('/etc/meteor.cfg')

coordinates = obs.get('trail', 'coordinates').split()
timestamps = obs.get('trail', 'timestamps').split()
az1, alt1 = obs.get('summary', 'startpos').split()
az2, alt2 = obs.get('summary', 'endpos').split()


# TODO: move into report.py

# Calculate points on path from az1,alt1 to az2,alt2 closest to points in "coordinates"
frame = 0
coordinates2 = []
for c in coordinates:
    az3, alt3 = c.split(',')
    p1 = tocartesian(float(az1), float(alt1))
    p2 = tocartesian(float(az2), float(alt2))
    p3 = tocartesian(float(az3), float(alt3))
    t = numpy.cross(numpy.cross(p1, p2), numpy.cross(p3, numpy.cross(p1, p2)))
    coordinates2.append(fromcartesian(t / numpy.linalg.norm(t)))

for (t, (az, alt)) in zip(timestamps, coordinates2):
    print(str(frame) + ' ' + str(round(float(t) - float(timestamps[0]), 2)) + ' ' + str(round(alt, 2)) + ' ' + str(round(az, 2)) + ' 1.0 ' + config.get('station', 'code') + time.strftime(' %Y-%m-%d %H:%M:%S', time.gmtime(float(t))) + '.' + str(round(float(t)-math.floor(float(t)), 2))[2:] + ' UTC')
    frame += 1

