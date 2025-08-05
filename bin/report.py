#!/usr/bin/env python3

# Usage report.py <event.txt>

import configparser
import sys
import os
from dateutil import parser
import calendar
import datetime
import subprocess
import math
import socket
import sys
import time
import ephem
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
from math import factorial

if len(sys.argv) != 2:
    print('Usage: ' + sys.argv[0] + ' <event.txt>')
    exit(0)


# Calculate arc
def calcarc(az1, alt1, az2, alt2):
    x1 = math.radians(az1)
    x2 = math.radians(az2)
    y1 = math.radians(alt1)
    y2 = math.radians(alt2)
    a = math.sin((y2-y1)/2) * math.sin((y2-y1)/2) + math.cos(y1) * math.cos(y2) * math.sin((x2-x1)/2) * math.sin((x2-x1)/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return math.degrees(c)

config = configparser.ConfigParser()
config.read([sys.argv[1]])

try:
    flash = bool(int(config.get('video', 'flash')))
except:
    flash = False

# Discard objects near the horizon moving upwards
try:
    t = config.get('trail', 'coordinates').split()
    startalt, endalt = float(t[0].split(',')[1]), float(t[-1].split(',')[1])

    if endalt > startalt and endalt < 20 and not flash:
        exit(0)
except:
    pass

# Short path?
timestamps = config.get('trail', 'timestamps').split()
if float(timestamps[-1]) - float(timestamps[0]) < 0.65 or int(config.get('trail', 'frames')) < 5:

    count = 3600
    command = ['lynx', '-dump', 'http://norskmeteornettverk.no/meteor/event.php?time=' + re.sub('.*events/', '', os.path.dirname(sys.argv[1]))]

    # Check server for matching event
    while count > 0 and time.time() < float(timestamps[0]) + 3600:
        result = int(subprocess.Popen(command, stdout=subprocess.PIPE).communicate()[0].strip())
        if result:
            break
        count -= 300
        time.sleep(300)

    if count <= 0:
        exit(0)

duration = float(config.get('trail', 'duration'))
positions = config.get('trail', 'positions').split()
arc = float(config.get('trail', 'arc'))

config2 = configparser.ConfigParser()
config2.read(['/etc/meteor.cfg', os.path.expanduser('~/meteor.cfg')])
name = config2.get('station', 'name')

videostart = config.get('video', 'start').rsplit(' ', 1)[0]
start = float(calendar.timegm(parser.parse(videostart).utctimetuple())) + parser.parse(videostart).microsecond/1000000.0
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.argv[1]))))
dir2 = os.path.dirname(sys.argv[1])

name += '-' + (datetime.datetime.strftime(parser.parse(videostart), '%Y%m%d%H%M%S'))

# Compute sun altitude
pos = ephem.Observer()
pos.lat = config2.get('astronomy', 'latitude')
pos.lon = config2.get('astronomy', 'longitude')
pos.elevation = float(config2.get('astronomy', 'elevation'))
pos.temp = float(config2.get('astronomy', 'temperature'))
pos.pressure = float(config2.get('astronomy', 'pressure'))
pos.date = parser.parse(videostart)
body = ephem.Sun()
body.compute(pos)
sunalt = math.degrees(float(repr(body.alt)))
sunaz = math.degrees(float(repr(body.az)))

midpoint = config.get('trail', 'midpoint').split(',')

# Too close to the sun?
if sunalt > 1 and calcarc(sunaz, sunalt, float(midpoint[0]), float(midpoint[1])) < 20:
    exit(0)
 
# Wait for other processes to finish to reduce load.  Use a socket to lock.
# This works even if we get a SIGKILL, but will probably only work on Linux.
while True:
    global lock_socket
    lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        lock_socket.bind('\0' + sys.argv[0])
        break
    except socket.error:
        time.sleep(3)

pos=positions[0].split(',')
xpos1=float(pos[0])
ypos1=float(pos[1])
pos=positions[-1].split(',')
xpos2=float(pos[0])
ypos2=float(pos[1])

command = [os.path.expanduser('~/bin/makevideos.sh'), dir, str(int(round(start))), str(int(float(math.ceil(duration)))), name, str(xpos1) + ',' + str(ypos1), str(xpos2) + ',' + str(ypos2), "0" if sunalt > -10 else "1", videostart + '\n' + config2.get('station', 'name').title() + ', ' + config2.get('astronomy', 'latitude') + 'N ' + config2.get('astronomy', 'longitude') + 'E ' + str(int(float(config2.get('astronomy', 'elevation')))) + ' m.a.s.l.']

print(' '.join(command))

with open("/tmp/cmd.txt", "a") as myfile:
    myfile.write(' '.join(command))

proc = subprocess.Popen(command, cwd=(dir2 if dir2 else '.'), stdout=subprocess.PIPE)
proc.wait()
output = proc.stdout.read().split()

if proc.returncode != 0:
    exit(proc.returncode)

duration *= calcarc(float(output[-5]), float(output[-4]), float(output[-2]), float(output[-1])) / arc

# Write metrack file
dat_file = open(dir2 + '/' + name + '.txt', "w")
dat_file.write(' '.join([config2.get('astronomy', 'longitude'), config2.get('astronomy', 'latitude'), str(output[-5].decode('utf-8')), str(output[-2].decode('utf-8')), str(output[-4].decode('utf-8')), str(output[-1].decode('utf-8')), '1', str(round(duration, 2)), '400', '128', '255', '255', config2.get('station', 'code'), str(round(start, 2)), config2.get('astronomy', 'elevation')]) + '\n')
dat_file.close()

if not config.has_section('summary'):
    config.add_section('summary')

config.set('summary', 'latitude', config2.get('astronomy', 'latitude'));
config.set('summary', 'longitude', config2.get('astronomy', 'longitude'));
config.set('summary', 'elevation', config2.get('astronomy', 'elevation'));
config.set('summary', 'timestamp', config.get('video', 'start'));
config.set('summary', 'startpos', ' '.join([str(output[-5].decode('utf-8')), str(output[-4].decode('utf-8'))]))
config.set('summary', 'endpos', ' '.join([str(output[-2].decode('utf-8')), str(output[-1].decode('utf-8'))]))
config.set('summary', 'duration', str(round(duration, 2)))
config.set('summary', 'sunalt', str(round(sunalt, 1)))
config.set('summary', 'recalibrated', "0" if sunalt > -10 else "1")

with open(sys.argv[1], 'w') as configfile:
    config.write(configfile)

frame = 0
timestamps = config.get('trail', 'timestamps').split()
coordinates = config.get('trail', 'coordinates').split()
brightness = config.get('trail', 'brightness').split()
fbrightness = config.get('trail', 'frame_brightness').split()
size = config.get('trail', 'size').split()

with open(dir2 + '/centroid.txt', 'w') as centroid:
    for (t, c) in zip(timestamps, coordinates):
        az, alt = c.split(',')
        centroid.write(str(frame) + ' ' + str(round(float(t) - float(timestamps[0]), 2)) + ' ' + str(alt) + ' ' + str(az) + ' 1.0 ' + config2.get('station', 'code') + time.strftime(' %Y-%m-%d %H:%M:%S', time.gmtime(float(t))) + '.' + str(round(float(t)-math.floor(float(t)), 2))[2:] + ' UTC\n')
        frame += 1

with open(dir2 + '/light.txt', 'w') as light:
    for (t, b, s, f) in zip(timestamps, brightness, size, fbrightness):
        light.write(str(round(float(t) - float(timestamps[0]), 2)) + ' ' + str(b) + ' ' + str(s) + ' ' + str(f) + '\n')

plt.plot([float(x) - float(timestamps[0]) for x in timestamps], list(map(float, brightness)))
plt.xlabel('Tid [s]')
plt.ylabel('Lysstyrke')
plt.savefig(dir2 + '/brightness.svg')
plt.savefig(dir2 + '/brightness.jpg')

with open('/etc/default/ssh_tunnel') as f:
    for line in f:
        l, r = line.split('=')
        if l == 'PORT':
            port = r.rstrip()

if int(port) == 0:
    command = [ 'lftp', '-e' ,'mirror -R ' + dir2 + ' upload/meteor/' + config2.get('station', 'name') + '/' + dir2, 'norskmeteornettverk.no' ]
    proc = subprocess.Popen(command)
    proc.communicate()

command = ['curl', '-s', '-o', '/dev/null', 'http://norskmeteornettverk.no/ssh/report.php?station=' + config2.get('station', 'name') + '&port=' + port + '&dir=' + (dir2 if dir2 else '.') ]
proc = subprocess.Popen(command)
