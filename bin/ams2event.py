#!/usr/bin/env python3

# Convert AMS json meteor detection to NMN event
# Example: ams2event.py /mnt/ams2/meteors/2022_08_07/2022_08_07_23_46_01_000_011193-trim-1313-reduced.json event.txt

import json
import argparse
import math
import os
import os.path
import subprocess
import sys
from datetime import datetime, UTC

# Import the pto_mapper module for coordinate transformations
import pto_mapper

parser = argparse.ArgumentParser(description='Convert AMS json meteor detection to NMN event.')

parser.add_argument('-m', '--manual', help='manually verified event', action='store_true')
parser.add_argument('-p', '--path', dest='path', help='top level path to NMN videos (default: "/meteor")', default="/meteor")
parser.add_argument('-e', '--execute', dest='exefile', help='program to pass event file to (default: "/home/meteor/bin/report.py")', default="/home/meteor/bin/report.py")
parser.add_argument(action='store', dest='infile', help='input .json file')
parser.add_argument(action='store', dest='outfile', help='output event file (default "event.txt")', nargs='?', default="event.txt")
args = parser.parse_args()


def midpoint(az1, alt1, az2, alt2):
    """Calculates the midpoint and distance between two celestial coordinates."""
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

        eventpath = args.path + '/cam' + cam + '/amsevents'
        eventdir = eventpath + '/' + str(start_ts.year) + f'{start_ts.month:02d}' + f'{start_ts.day:02d}' + "/" + f'{start_ts.hour:02d}' + f'{start_ts.minute:02d}' + f'{start_ts.second:02d}'
        eventfile = eventdir + '/' + args.outfile
        ptofile = args.path + '/cam' + cam + '/lens.pto'
        have_pto = os.path.exists(ptofile)

        # To be populated with coordinate data
        az = []
        alt = []
        calculated_coords_str = []
        img_w, img_h = None, None

        os.makedirs(eventdir, exist_ok=True)
        
        # --- NEW: Use pto_mapper for coordinate conversion ---
        if have_pto:
            try:
                # Parse the .pto file
                pto_data = pto_mapper.parse_pto_file(ptofile)
                global_options, images = pto_data
                pano_w = global_options.get('w')
                pano_h = global_options.get('h')
                # The script assumes a single-image PTO file
                img_w = images[0].get('w')
                img_h = images[0].get('h')
                
                # We need an equirectangular panorama destination (f=2) for az/alt conversion
                is_equirectangular = global_options.get('f', 2) == 2

                if not all([pano_w, pano_h, img_w, img_h, is_equirectangular]):
                    print(f"Warning: PTO file '{ptofile}' is missing required parameters (p w, p h, p f=2, i w, i h).", file=sys.stderr)
                    have_pto = False
                else:
                    # Iterate through meteor frames and map coordinates
                    for frame in data['meteor_frame_data']:
                        # Indices: dt=0, frn=1, x=2, y=3
                        img_x, img_y = float(frame[2]), float(frame[3])
                        
                        # Map from source image (x,y) to panorama (x,y)
                        pano_coords = pto_mapper.map_image_to_pano(pto_data, 0, img_x, img_y)

                        if pano_coords:
                            pano_x, pano_y = pano_coords
                            # Convert equirectangular panorama pixels to Azimuth/Altitude
                            azimuth = math.degrees((pano_x / pano_w - 0.5) * 2 * math.pi) - 180
                            altitude = math.degrees(-(pano_y / pano_h - 0.5) * math.pi)
                            
                            if azimuth < 0: azimuth += 360 # Normalize Azimuth to 0-360
                            
                            az.append(azimuth)
                            alt.append(altitude)
                            calculated_coords_str.append(f'{azimuth:.2f},{altitude:.2f}')
                        else:
                            # Handle cases where a point can't be mapped
                            calculated_coords_str.append('nan,nan')
            except Exception as e:
                print(f"Warning: Could not process PTO file '{ptofile}'. Error: {e}", file=sys.stderr)
                have_pto = False

        # --- Write event data to file ---
        with open(eventfile, 'w') as output:
            print('[trail]', file=output)
            print('frames = ' + str(len(data['meteor_frame_data'])), file=output)
            print('duration = ' + str(duration), file=output)
            
            # Write pixel positions
            print('positions = ', end = '', file=output)
            for f in data['meteor_frame_data']:
                print(str(f[2]) + ',' + str(f[3]), end = ' ', file=output)
            print(file=output)

            # Write timestamps
            print('timestamps = ', end = '', file=output)
            for f in data['meteor_frame_data']:
                print(str(datetime.strptime(f[0], '%Y-%m-%d %H:%M:%S.%f').timestamp()), end = ' ', file=output)
            print(file=output)
        
            # Write celestial coordinates
            if have_pto and calculated_coords_str:
                # If PTO conversion was successful, write the new coordinates
                print('coordinates = ' + ' '.join(calculated_coords_str), file=output)
                # Also write the original AMS coordinates for reference
                print('ams_coords = ', end='', file=output)
            else:
                # Otherwise, use the coordinates from the JSON file
                print('coordinates = ', end='', file=output)
                # Populate az/alt from JSON for subsequent calculations
                for f in data['meteor_frame_data']:
                    az.append(f[9])
                    alt.append(f[10])

            # This loop writes the original AMS coordinates
            for f in data['meteor_frame_data']:
                print(f'{f[9]:.2f},{f[10]:.2f}', end=' ', file=output)
            print(file=output)
            
            # --- Continue with calculations and file writing ---
            if not az or not alt:
                print("Error: No coordinate data available to calculate midpoint.", file=sys.stderr)
                sys.exit(1)

            midaz, midalt, arc = midpoint(az[0], alt[0], az[-1], alt[-1])
        
            print('midpoint = ' + f'{midaz:.2f},{midalt:.2f}', file=output)
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

            print('speed = ' + str(arc / duration if duration > 0 else 0), file=output)

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
            if have_pto and img_w is not None:
                print('ptofile = ' + ptofile, file=output)
                print('ptowidth = ' + str(img_w), file=output)
                print('ptoheight = ' + str(img_h), file=output) # Corrected typo from 'ptoheiht'
                print('execute = ' + args.exefile, file=output)
            print('eventdir = ' + eventpath, file=output)

            print(file=output)
            print('[summary]', file=output)
            print('timestamp = ' + str(start_ts) + ' (' + str(start_ts.timestamp()) + ')', file=output)
            print('startpos = ' + f'{az[0]:.2f}' + ' ' + f'{alt[0]:.2f}', file=output)
            print('endpos = ' + f'{az[-1]:.2f}' + ' ' + f'{alt[-1]:.2f}', file=output)
            print('duration = ' + str(duration), file=output)
    
        # Execute external reporting script for significant events
        if alt and alt[0] > -2 and alt[-1] > -2 and len(data['meteor_frame_data']) > 5:
            proc = subprocess.Popen([args.exefile, eventfile])
            # proc.wait() # Uncomment if the script needs to wait for completion
