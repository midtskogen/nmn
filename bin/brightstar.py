#!/usr/bin/env python3

# List the brightest visible stars at a given Unix timestamp
# Usage: brightstar.py <Unix timestamp> <pto file>

import ephem
import math
import hsi
import argparse
import configparser
import os
from datetime import datetime, UTC
from stars import cat

def brightstar(pano, image, pos, faintest, brightest, objects, extend = 0):
    def test_body(body, name, faintest, brightest):
        body.compute(pos)
        az = math.degrees(float(repr(body.az)))
        alt = math.degrees(float(repr(body.alt)))
        #alt2 = alt + 0.016 / math.tan(math.pi/180*(alt + (7.31/(alt + 4.4))))
        alt2 = alt + 0.006 / math.tan(math.pi/180*(alt + (7.31/(alt + 4.4))))
        mag = body.mag
        scale = int(pano.getOptions().getWidth() / pano.getOptions().getHFOV())
        if alt > -0.5 and body.mag <= faintest and body.mag >= brightest:
            trafo.transformImgCoord(dst, hsi.FDiff2D(az*scale, (90-alt2)*scale));
            if -extend < dst.x < width + extend and -extend < dst.y < height + extend:
                return (dst.x, dst.y, az, alt, name, mag)
        return None

    img = pano.getImage(image)
    width = img.getWidth()
    height = img.getHeight()
    dst = hsi.FDiff2D();
    trafo = hsi.Transform()
    trafo.createTransform(img, pano.getOptions())
    count = 0
    res = []
    for (body, name) in [ (ephem.Sun(), "Sol"), (ephem.Moon(), "Moon") , (ephem.Mercury(), "Mercury"), (ephem.Venus(), "Venus"), (ephem.Mars(), "Mars"), (ephem.Jupiter(), "Jupiter"), (ephem.Saturn(), "Saturn") ]:
        r = test_body(body, name, faintest, brightest)
        if r != None:
            count += 1
            res.append(r)
        if count == objects:
            break

    if count < objects:
        for (ra, pmra, dec, pmdec, mag, name) in cat:
            if (mag <= faintest and mag >= brightest):
                body = ephem.FixedBody()
                body._ra, body._pmra, body._dec, body._pmdec, body._epoch = str(ra), pmra, str(dec), pmdec, ephem.J2000
                body.mag = mag
                r = test_body(body, name, faintest, -30)
                if r != None:
                    count += 1
                    res.append(r)
                if count == objects:
                    break
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List the brightest visible stars.')

    parser.add_argument('-n', '--number', dest='objects', help='maximum number of objects (default: 500)', default=500, type=int)
    parser.add_argument('-f', '--faintest', dest='faintest', help='faintest objects to include (default: 5)', default=5, type=float)
    parser.add_argument('-b', '--brightest', dest='brightest', help='brightest objects to include (default: -30)', default=-30, type=float)
    parser.add_argument('-x', '--longitude', dest='longitude', help='observer longitude', type=float)
    parser.add_argument('-y', '--latitude', dest='latitude', help='observer latitude', type=float)
    parser.add_argument('-a', '--altitude', dest='elevation', help='observer altitude (m)', type=float)
    parser.add_argument('-t', '--temperature', dest='temperature', help='observer temperature (C)', type=float)
    parser.add_argument('-p', '--pressure', dest='pressure', help='observer air pressure (hPa)', type=float)
    parser.add_argument('-i', '--image', dest='image', help='which image in the .pto file to use (default: 0)', default=0, type=int)
    parser.add_argument('-e', '--extend', dest='extend', help='include stars within this number of pixels outside frame (default: 0)', default=0, type=int)
    parser.add_argument('-c', '--config', dest='config', help='meteor config file (default: /etc/meteor.cfg)', default='/etc/meteor.cfg', type=str)

    parser.add_argument(action='store', dest='timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00UTC)')
    parser.add_argument(action='store', dest='ptofile', help='Hugin .pto file')

    args = parser.parse_args()
    pos = ephem.Observer()
    config = configparser.ConfigParser()
    config.read([args.config, os.path.expanduser('~/meteor.cfg')])

    pos.lat = config.get('astronomy', 'latitude')
    pos.lon = config.get('astronomy', 'longitude')
    pos.elevation = float(config.get('astronomy', 'elevation'))
    pos.temp = float(config.get('astronomy', 'temperature'))
    pos.pressure = float(config.get('astronomy', 'pressure'))

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

    pos.date = datetime.fromtimestamp(float(args.timestamp), UTC).strftime('%Y-%m-%d %H:%M:%S')

    pano = hsi.Panorama()
    try:
        pano.ReadPTOFile(args.ptofile)
    except:
        pano.readData(hsi.ifstream(args.ptofile))

    for l in brightstar(pano, args.image, pos, args.faintest, args.brightest, args.objects, args.extend):
        print(*l)
