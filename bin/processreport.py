#!/usr/bin/env python3

# Usage processreport.py <event.txt>

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

if len(sys.argv) != 2:
    print('Usage: ' + sys.argv[0] + ' <event.txt>')
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

config = configparser.ConfigParser()
config.read([sys.argv[1]])

duration = float(config.get('trail', 'duration'))
positions = config.get('trail', 'positions').split()
framebrightness = max(config.get('trail', 'frame_brightness').split())
ptofile = config.get('config', 'ptofile')
arc = float(config.get('trail', 'arc'))

config2 = configparser.ConfigParser()
config2.read(['/etc/meteor.cfg', os.path.expanduser('~/meteor.cfg')])
name = config2.get('station', 'name')

videostart = config.get('video', 'start').rsplit(' ', 1)[0]
start = float(calendar.timegm(parser.parse(videostart).utctimetuple())) + parser.parse(videostart).microsecond/1000000.0
dir = os.path.dirname(ptofile)
dir2 = os.path.dirname(sys.argv[1])

name += '-' + (datetime.datetime.strftime(parser.parse(videostart), '%Y%m%d%H%M%S'))

command = [os.path.expanduser('~/bin/makevideos.sh'), dir, str(int(round(start))-4), str(int(float(duration))+8), name, ','.join(positions[0].split(',')[:2]), ','.join(positions[-1].split(',')[:2]), "0" if float(framebrightness) > 25 else "1"]

print(' '.join(command))

proc = subprocess.Popen(command, cwd=(dir2 if dir2 else '.'), stdout=subprocess.PIPE)
output = proc.stdout.read().split()

# Calculate arc
def calcarc(az1, alt1, az2, alt2):
    x1 = math.radians(az1)
    x2 = math.radians(az2)
    y1 = math.radians(alt1)
    y2 = math.radians(alt2)
    a = math.sin((y2-y1)/2) * math.sin((y2-y1)/2) + math.cos(y1) * math.cos(y2) * math.sin((x2-x1)/2) * math.sin((x2-x1)/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return math.degrees(c)


duration *= calcarc(float(output[-5]), float(output[-4]), float(output[-2]), float(output[-1])) / arc

# Write metrack file
dat_file = open(dir2 + '/' + name + '.txt', "w")
dat_file.write(' '.join([config2.get('astronomy', 'longitude'), config2.get('astronomy', 'latitude'), output[-5], output[-2], output[-4], output[-1], '1', str(round(duration, 2)), '400', '128', '255', '255', config2.get('station', 'code'), str(round(start, 2)), config2.get('astronomy', 'elevation')]) + '\n')
dat_file.close()

if not config.has_section('summary'):
    config.add_section('summary')

config.set('summary', 'latitude', config2.get('astronomy', 'latitude'));
config.set('summary', 'longitude', config2.get('astronomy', 'longitude'));
config.set('summary', 'elevation', config2.get('astronomy', 'elevation'));
config.set('summary', 'timestamp', config.get('video', 'start'));
config.set('summary', 'startpos', ' '.join([output[-5], output[-4]]))
config.set('summary', 'endpos', ' '.join([output[-2], output[-1]]))
config.set('summary', 'duration', str(round(duration, 2)))
with open(sys.argv[1], 'w') as configfile:
    config.write(configfile)
