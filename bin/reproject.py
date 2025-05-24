#!/usr/bin/env python3

import math
import hsi
import argparse

def pos(s):
    try:
        az, alt = map(float, s.split(','))
        return az, alt
    except:
        raise argparse.ArgumentTypeError("Positions must be az,alt")

parser = argparse.ArgumentParser(description='Change projection and centre of a .pto file.')

parser.add_argument('-s', '--size', dest='size', help='destination canvas width and height', default=None, type=int)
parser.add_argument('--width', dest='width', help='destination canvas width', default=2560, type=int)
parser.add_argument('--height', dest='height', help='destination canvas height', default=2560, type=int)
parser.add_argument('-f', '--fov', dest='fov', help='horizontal field of view (default: auto)', default=0.0, type=float)
parser.add_argument('-o', '--output', action='store', dest='outfile', help='destination .pto file (fisheye to rectilinear)')
parser.add_argument('-g', '--grid', action='store', dest='gridfile', help='grid .pto file (rectilinear to equirectangular)')
parser.add_argument(action='store', dest='infile', help="source .pto file")
parser.add_argument(action='store', dest='pos', help="position", type=pos)
parser.add_argument('-e', '--endpos', action='store', dest='endpos', help="end position", type=pos)

args = parser.parse_args()

if args.endpos == None:
    args.endpos = args.pos

if args.outfile == None:
    tmp = args.infile.split('.')
    args.outfile = '.'.join(tmp[:-1]) + '_reproj.' + tmp[-1]

if args.size != None:
    args.width = args.height = args.size


# Calculate midpoint and arc
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



startaz, startalt = args.pos
endaz, endalt = args.endpos
midaz, midalt, arc = midpoint(startaz, startalt, endaz, endalt)

pano = hsi.Panorama()

hfov = max(args.fov, arc*1.4*max(1, float(args.width) / args.height))

if args.gridfile:
    try:
        pano.readPTOFile(args.infile)
    except:
        pano.readData(hsi.ifstream(args.infile))
    img = hsi.SrcPanoImage()
    img.setFilename("dummy.jpg")
    img.setProjection(0)
#    img.setCropFactor(1.0);
    img.setSize(hsi.Size2D(args.width, args.height))
    img.setHFOV(hfov)
    img.setYaw(midaz-180)
    img.setPitch(midalt)
    while pano.getNrOfImages():
        pano.removeImage(0)
    pano.addImage(img)
    try:
        pano.writePTOFile(args.gridfile)
    except:
        pano.writeData(hsi.ofstream(args.gridfile))
try:
    pano.readPTOFile(args.infile)
except:
    pano.readData(hsi.ifstream(args.infile))
while pano.getNrOfImages() > 1:
    pano.removeImage(pano.getNrOfImages()-1)
opt = pano.getOptions()
opt.setProjection(0) # Rectilinear
opt.setHFOV(hfov)
opt.setWidth(args.width);
opt.setHeight(args.height);
hsi.RotatePanorama(pano, 180 - midaz, 0, 0).run()
hsi.RotatePanorama(pano, 0, -midalt, 0).run()


try:
    pano.writePTOFile(args.outfile)
except:
    pano.writeData(hsi.ofstream(args.outfile))
