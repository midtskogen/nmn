#!/usr/bin/python3

import hsi
import argparse
import math
from datetime import datetime

parser = argparse.ArgumentParser(description='Make an event file from a lens file and centroid file.')
parser.add_argument('-i', '--image', dest='image', help='which image in the .pto file to use (default: 0)', default=0, type=int)
parser.add_argument(action='store', dest='ptofile', help='input .pto file')
parser.add_argument(action='store', dest='centroid', help='input centroid file')
args = parser.parse_args()

pano = hsi.Panorama()
try:
    pano.ReadPTOFile(args.ptofile)
except:
    pano.readData(hsi.ifstream(args.ptofile))
img = pano.getImage(args.image)
width = img.getSize().width()
height = img.getSize().height()
scale = int(pano.getOptions().getWidth() / pano.getOptions().getHFOV())
dst = hsi.FDiff2D();
trafo = hsi.Transform()
trafo.createTransform(img, pano.getOptions())

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

with open(args.centroid, 'r') as file: 

    timestamps = []
    timestamps2 = []
    coordinates = []
    positions = []
    lines = 0

    for line in file: 
        lines = lines + 1
        l = line.split()
        trafo.transformImgCoord(dst, hsi.FDiff2D(scale * float(l[3]), scale * (90 - float(l[2]))))

        positions.append((dst.x, dst.y))
        coordinates.append((l[3], l[2]))
        ts = datetime.strptime(l[6] + " " + l[7], "%Y-%m-%d %H:%M:%S.%f")
        timestamps.append(str((ts - datetime(1970, 1, 1)).total_seconds()))
        timestamps2.append(l[6] + " " + l[7] + " " + l[8])

print("[trail]")
print("frames = " + str(lines))
print("duration = " + str(float(timestamps[-1]) - float(timestamps[0])))
print("positions = ", end = "")
for i in positions:
    print(str(i[0]) + "," + str(i[1]) + " ", end = "")
print()
print("coordinates = ", end = "")
for i in coordinates:
    print(i[0] + "," + i[1] + " ", end = "")
print()
print("timestamps = ", end = "")
for i in timestamps:
    print(i + " ", end = "")
print()
midaz, midalt, arc = midpoint(float(coordinates[0][0]), float(coordinates[0][1]),
                              float(coordinates[-1][0]), float(coordinates[-1][1]))
print("midpoint = " + str(midaz) + "," + str(midalt))
print("arc = " + str(arc))
print("brightness = ", end = "")
for i in coordinates:
    print("0 ", end = "")
print()
print("frame_brightness = ", end = "")
for i in coordinates:
    print("0 ", end = "")
print()
print("size = ", end = "")
for i in coordinates:
    print("0 ", end = "")
print()

print()
print("[video]")
print("start = " + timestamps2[0] + " (" + timestamps[0] + ")")
print("end = " + timestamps2[-1] + " (" + timestamps[-1] + ")")
print("width = " + str(width))
print("height = " + str(height))
