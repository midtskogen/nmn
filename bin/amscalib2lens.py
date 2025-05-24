#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Create a .pto file from AMS calibration
# Usage: amscalib2lens.py <AMS calibration json file> <pto file>
#        amscalib2lens.py /mnt/ams2/cal/freecal/2022_05_06_00_32_20_000_011193/2022_05_06_00_32_20_000_011193-stacked-calparams.json lens.pto

import ephem
from datetime import datetime, UTC
import math
import argparse
import os
import json
import subprocess

parser = argparse.ArgumentParser(description='Convert AMS calibration into a Hugin/panotools pto file.')

parser.add_argument(action='store', dest='amscalib', help='AMS calibration json file')
parser.add_argument(action='store', dest='ptofile', help='Hugin .pto file')
parser.add_argument('-W', '--width', dest='width', help='image width if not found in calibration file (default: 1920)', type=int, default=1920)
parser.add_argument('-H', '--height', dest='height', help='image height if not found in calibration file (default: 1080)', type=int, default=1080)
parser.add_argument('-d', '--match_dist', dest='match_dist', help='maximum allowed match distance (default: 0.2)', type=float, default=0.2)
parser.add_argument('-c', '--config', dest='config', help='ams config file (default: /home/ams/amscams/conf/as6.json)', default='/home/ams/amscams/conf/as6.json', type=str)
parser.add_argument('-T', '--timestamp', dest='timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00UTC), default: guess from filename')

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
                          
        pos = ephem.Observer()

        with open(args.config) as c:
            conf = json.load(c)
            if 'site' in conf:
                if 'device_lat' in conf['site']:
                    pos.lat = conf['site']['device_lat']
                if 'device_lng' in conf['site']:
                    pos.lon = conf['site']['device_lng']
                if 'device_alt' in conf['site']:
                    pos.elevation = int(conf['site']['device_alt'])

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

            body = ephem.FixedBody()
            body._ra, body._dec, body._epoch = str(ra), str(dec), ephem.J2000
            body.compute(pos)
            az = math.degrees(float(repr(body.az)))
            alt = math.degrees(float(repr(body.alt)))
            if alt > 1:
                print('c n0 N1 x' + str(six) + ' y' + str(siy) + ' X' + str(az*100) + ' Y' + str((90-alt)*100) + ' t0  # ' + str(dcname), file=output)
                
        output.close()
        proc = subprocess.Popen(['autooptimiser', '-n', args.ptofile, '-o', args.ptofile])
        proc.wait()
