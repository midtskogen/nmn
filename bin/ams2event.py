#!/usr/bin/env python3

# Convert AMS json meteor detection to NMN event
# Example: ams2event.py /mnt/ams2/meteors/2022_08_07/2022_08_07_23_46_01_000_011193-trim-1313-reduced.json event.txt

import json
import argparse
import math
import hsi
import os
import os.path
import subprocess
from datetime import datetime, UTC

parser = argparse.ArgumentParser(description='Convert AMS json meteor detection to NMN event.')

parser.add_argument('-m', '--manual', help='manually verified event', action='store_true')
parser.add_argument('-p', '--path', dest='path', help='top level path to NMN videos (default: "/meteor")', default="/meteor")
parser.add_argument('-e', '--execute', dest='exefile', help='program to pass event file to (default: "/home/meteor/bin/report.py")', default="/home/meteor/bin/report.py")
parser.add_argument(action='store', dest='infile', help='input .json file')
parser.add_argument(action='store', dest='outfile', help='output event file (default "event.txt")', nargs='?', default="event.txt")
args = parser.parse_args()


def midpoint(az1, alt1, az2, alt2):
    x1 = math.radians(az1)
    x2 = math.radians(az2)
    y1 = math.radians(alt1)
    y2 = math.radians(alt2)
    Bx = math.cos(y2) * math.cos(x2-x1)
    By = math.cos(y2) * math.sin(x2-x1)
    y3 = math.atan2(math.sin(y1) + math.sin(y2),
                    math.sqrt( (math.cos(y1)+Bx)*(math.cos(y1)+Bx) + By*By ) )
    x3 = x1 + math.atan2(By, math.cos(y1) + Bx)
    a = math.sin((y2-y1)/2) * math.sin((y2-y1)/2) + math.cos(y1) * math.cos(y2) * math.sin((x2-x1)/2) * math.sin((x2-x1)/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return math.degrees(x3), math.degrees(y3), math.degrees(c)

if __name__ == '__main__':
    with open(args.infile) as f:
        data = json.load(f)
        device = data['device_name']
        cam = device[-1]
        start_ts = datetime.strptime(data['meteor_frame_data'][0][0], '%Y-%m-%d %H:%M:%S.%f')
        end_ts = datetime.strptime(data['meteor_frame_data'][-1][0], '%Y-%m-%d %H:%M:%S.%f')
        duration = (end_ts - start_ts).total_seconds()
        #full = args.path + "/cam" + cam + "/" + str(start_ts.year) + f'{start_ts.month:02d}' + f'{start_ts.day:02d}' + "/" + f'{start_ts.hour:02d}' + "/full_" + f'{start_ts.minute:02d}' + ".mp4"
        #mini = full.replace('full', 'mini')

        eventpath = args.path + '/cam' + cam + '/amsevents'
        eventdir = eventpath + '/' + str(start_ts.year) + f'{start_ts.month:02d}' + f'{start_ts.day:02d}' + "/" + f'{start_ts.hour:02d}' + f'{start_ts.minute:02d}' + f'{start_ts.second:02d}'
        eventfile = eventdir + '/' + args.outfile
        ptofile = args.path + '/cam' + cam + '/lens.pto'
        have_pto = os.path.exists(ptofile)

        if have_pto:
            pano = hsi.Panorama()
            try:
                pano.ReadPTOFile(ptofile)
            except:
                pano.readData(hsi.ifstream(ptofile))
            img = pano.getImage(0)
            dst = hsi.FDiff2D();
            inv = hsi.Transform()
            inv.createInvTransform(img, pano.getOptions())

        os.makedirs(eventdir, exist_ok=True)
        output = open(eventfile, 'w')

        # Indices: dt=0, frn=1, x=2, y=3, w=4, h=5, oint=6, ra=7, dec=8, az=9, el=10

        print('[trail]', file=output)
        print('frames = ' + str(len(data['meteor_frame_data'])), file=output)
        print('duration = ' + str(duration), file=output)
        print('positions = ', end = '', file=output)
        for f in data['meteor_frame_data']:
            print(str(f[2]) + ',' + str(f[3]), end = ' ', file=output)
        print(file=output)

        print('timestamps = ', end = '', file=output)
        for f in data['meteor_frame_data']:
            print(str(datetime.strptime(f[0], '%Y-%m-%d %H:%M:%S.%f').timestamp()), end = ' ', file=output)
        print(file=output)
    
        az = []
        alt = []
        if have_pto:
            scale = int(pano.getOptions().getWidth() / pano.getOptions().getHFOV())
            print('coordinates = ', end = '', file=output)
            for f in data['meteor_frame_data']:
                inv.transformImgCoord(dst, hsi.FDiff2D(float(f[2]), float(f[3])))
                az.append(dst.x / scale)
                alt.append(90 - (dst.y / scale))
                print(f'{az[-1]:.2f}' + ',' + f'{alt[-1]:.2f}', end = ' ', file=output)
            midaz, midalt, arc = midpoint(az[0], alt[0], az[-1], alt[-1])
            print(file=output)
            print('ams_coords = ', end = '', file=output)
        else:
            midaz, midalt, arc = midpoint(data['meteor_frame_data'][0][9], data['meteor_frame_data'][0][10], data['meteor_frame_data'][-1][9], data['meteor_frame_data'][-1][10])
            print('coordinates = ', end = '', file=output)
        for f in data['meteor_frame_data']:
            print(f'{f[9]:.2f}' + ',' + f'{f[10]:.2f}', end = ' ', file=output)
        print(file=output)
    
        print('midpoint = ' + f'{midaz:.2f}' + ',' + f'{midalt:.2f}', file=output)
        print('arc = ' + f'{arc:.2f}', file=output)
    
        print('size = ', end = '', file=output)
        for f in data['meteor_frame_data']:
            print(str(f[4] * f[5]), end = ' ', file=output)
        print(file=output)

        print('brightness = ', end = '', file=output)
        for f in data['meteor_frame_data']:
            print(str((f[4] * f[5]) ** 2), end = ' ', file=output)
        print(file=output)

        print('frame_brightness = ', end = '', file=output)
        for f in data['meteor_frame_data']:
            print('-1', end = ' ', file=output)
        print(file=output)

        print('speed = ' + str(arc / duration), file=output)

        if args.manual:
            print('manual = 1', file=output)

        print('ams = 1', file=output)
        print(file=output)
        print('[video]', file=output)
        print('start = ' + str(start_ts) + ' (' + str(start_ts.timestamp()) + ')', file=output)
        print('end = ' + str(end_ts) + ' (' + str(end_ts.timestamp()) + ')', file=output)
        print('wallclock = ' + str(datetime.now(UTC)), file=output)
        print('width = ' + str(data['cal_params']['imagew']), file=output)
        print('height = ' + str(data['cal_params']['imageh']), file=output)
    
        print(file=output)
        print('[config]', file=output)
        if have_pto:
            print('ptofile = ' + ptofile, file=output)
            print('ptoscale = ' + str(scale), file=output)
            print('ptowidth = ' + str(img.getWidth()), file=output)
            print('ptoheiht = ' + str(img.getHeight()), file=output)
            print('execute = ' + args.exefile, file=output)
        print('eventdir = ' + eventpath, file=output)

        print(file=output)
        print('[summary]', file=output)
        print('timestamp = ' + str(start_ts) + ' (' + str(start_ts.timestamp()) + ')', file=output)
        print('startpos = ' + f'{az[0]:.2f}' + ' ' + f'{alt[0]:.2f}', file=output)
        print('endpos = ' + f'{az[-1]:.2f}' + ' ' + f'{alt[-1]:.2f}', file=output)
        print('duration = ' + str(duration), file=output)
        output.close()
    
        if alt[0] > -2 and alt[-1] > -2 and len(data['meteor_frame_data']) > 15:
            proc = subprocess.Popen([args.exefile, eventfile])
#            proc.wait()
