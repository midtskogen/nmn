#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Create a .pto file from AMS calibration
# Usage: amscalib2lens.py <AMS calibration json file> <pto file>
#        amscalib2lens.py /mnt/ams2/cal/freecal/2022_05_06_00_32_20_000_011193/2022_05_06_00_32_20_000_011193-stacked-calparams.json lens.pto

import hsi
import ephem
from datetime import datetime, UTC
import math
import argparse
import configparser
import os
import json
import subprocess
from stars import cat

parser = argparse.ArgumentParser(description='Convert AMS calibration into a Hugin/panotools pto file.')

parser.add_argument(action='store', dest='amscalib', help='AMS calibration json file')
parser.add_argument(action='store', dest='ptofile', help='Hugin .pto file')
parser.add_argument('-W', '--width', dest='width', help='image width (default: 1920)', type=int, default=1920)
parser.add_argument('-H', '--height', dest='height', help='image height (default: 1080)', type=int, default=1080)
parser.add_argument('-d', '--match_dist', dest='match_dist', help='maximum allowed match distance (default: 0.2)', type=float, default=0.2)
parser.add_argument('-x', '--longitude', dest='longitude', help='observer longitude', type=float)
parser.add_argument('-y', '--latitude', dest='latitude', help='observer latitude', type=float)
parser.add_argument('-e', '--elevation', dest='elevation', help='observer elevation (m)', type=float)
parser.add_argument('-t', '--temperature', dest='temperature', help='observer temperature (C)', type=float)
parser.add_argument('-p', '--pressure', dest='pressure', help='observer air pressure (hPa)', type=float)
parser.add_argument('-c', '--config', dest='config', help='meteor config file (default: /etc/meteor.cfg)', default='/etc/meteor.cfg', type=str)
parser.add_argument('-T', '--timestamp', dest='timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00UTC)')

args = parser.parse_args()

if __name__ == '__main__':
    if not args.timestamp:
        tmp = os.path.basename(args.amscalib).split('_')
        args.timestamp = datetime.strptime(tmp[0] + '-' + tmp[1] + '-' + tmp[2] + ' ' + tmp[3] + ':' + tmp[4] + ':' + tmp[5], "%Y-%m-%d %H:%M:%S").timestamp()

    with open(args.amscalib) as f:
        data = json.load(f)

        if 'imagew' in data:
            args.width = data['imagew']
                          
        if 'imageh' in data:
            args.height = data['imageh']
                          
        config = configparser.ConfigParser()
        config.read([args.config, os.path.expanduser('~/meteor.cfg')])

        pos = ephem.Observer()
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

        if 'device_lat' in data:
            pos.lat = data['device_lat']
        if 'device_lon' in data:
            pos.lon = data['device_lon']
        if 'device_alt' in data:
            pos.elevation = int(data['device_alt'])
        
        pos.date = datetime.fromtimestamp(float(args.timestamp), UTC).strftime('%Y-%m-%d %H:%M:%S')

        if 'pixel_scale' in data:
            pixel_scale = float(data['pixel_scale'])
        else:
            pixel_scale = float(data['pixscale'])

        output = open(args.ptofile, 'w')
        print('''# hugin project file
#hugin_ptoversion 2
p f2 w36000 h18000 v360  E0 R0 n"TIFF_m c:LZW"
m g1 i0 m2 p0.00784314

# image lines
#-hugin  cropFactor=1
i w''' + str(args.width) + ' h' + str(args.height) + ' f3 v' + str(args.width * pixel_scale / 3600) + ' Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p' + str(data['center_el']) + ' y' + str(data['center_az'] - 180) + ''' TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5
i w36000 h18000 f4 v360 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p0 y0 TrX0 TrY0 TrZ0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5 n"dummy.jpg"


# specify variables that should be optimized
v v0
v r0
v p0
v y0
v a0
v b0
v c0
v d0
v e0
v

''' + ('# ' + str(pos.date)) + '''
# control points''', file=output)

        for star in data['cat_image_stars']:
            dcname,mag,ra,dec,img_ra,img_dec,match_dist,new_x,new_y,img_az,img_el,new_cat_x,new_cat_y,six,siy,cat_dist,bp = star
            ra = ra * 24 / 360

            if match_dist > args.match_dist:
                continue

            # Use AMS ra/dec to find the star in the NMN catalogue
            min = 99999
            for (ra2, pmra, dec2, pmdec, mag2, name) in cat:
                if abs(mag - mag2) > 0.2:  # Quick test whether same star
                    continue
                body1 = ephem.FixedBody()
                body1._ra, body1._pmra, body1._dec, body1._pmdec, body1._epoch = str(ra), pmra, str(dec), pmdec, ephem.J2000
                body1.mag = mag
                body1.compute(pos)
                body2 = ephem.FixedBody()
                body2._ra, body2._pmra, body2._dec, body2._pmdec, body2._epoch = str(ra2), pmra, str(dec2), pmdec, ephem.J2000
                body2.mag = mag
                body2.compute(pos)
                separation = float(repr(ephem.separation(body1, body2)))
                if (separation < min):
                    min = separation
                    best = name
                    bestbody = body2
            if min < 0.0001:
                bestbody.compute(pos)
                az = math.degrees(float(repr(bestbody.az)))
                alt = math.degrees(float(repr(bestbody.alt)))
                if alt > 1:
                    print('c n0 N1 x' + str(six) + ' y' + str(siy) + ' X' + str(az*100) + ' Y' + str((90-alt)*100) + ' t0  # ' + best, file=output)

        output.close()
        proc = subprocess.Popen(['autooptimiser', '-n', args.ptofile, '-o', args.ptofile])
        proc.wait()
