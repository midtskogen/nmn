#!/usr/bin/python3

# Makes a file grid.png from a .pto file
# Result usage example: composite -blend 40 meteor-20141010.jpg grid.png x.jpg

import ephem
import subprocess
import math
import hsi
import argparse
import wand.image
import wand.drawing
import wand.color
import configparser
import os
import re
from datetime import datetime, UTC
from brightstar import brightstar

#from wand.image import Image
#from wand.drawing import Drawing
#from wand.color import Color

parser = argparse.ArgumentParser(description='Create a grid file from a .pto file.')

parser.add_argument('-p', '--picture', dest='image', help='which picture in the .pto file to use (default: 0)', default=0, type=int)
parser.add_argument('-W', '--width', dest='width', help='width of output (if different from input)', default=-1, type=int)
parser.add_argument('-H', '--height', dest='height', help='height of output (if different from input)', default=-1, type=int)
parser.add_argument('-x', '--xscale', dest='xscale', help='X scaling factor (default: 1)', default=1, type=float)
parser.add_argument('-y', '--yscale', dest='yscale', help='Y scaling factor (default: 1)', default=1, type=float)
parser.add_argument('-s', '--spacing', dest='label_spacing', help='spacing between labels in degrees (default: 10)', default=10, type=int)
parser.add_argument('-n', '--number', dest='objects', help='maximum number of objects (default: 500)', default=500, type=int)
parser.add_argument('-f', '--faintest', dest='faintest', help='faintest objects to include (default: 5)', default=5, type=float)
parser.add_argument('-b', '--brightest', dest='brightest', help='brightest objects to include (default: -30)', default=-30, type=float)
parser.add_argument('-X', '--longitude', dest='longitude', help='observer longitude', type=float)
parser.add_argument('-Y', '--latitude', dest='latitude', help='observer latitude', type=float)
parser.add_argument('-e', '--elevation', dest='elevation', help='observer elevation (m)', type=float)
parser.add_argument('-t', '--temperature', dest='temperature', help='observer temperature (C)', type=float)
parser.add_argument('-P', '--pressure', dest='pressure', help='observer air pressure (hPa)', type=float)
parser.add_argument('-c', '--config', dest='config', help='meteor config file', type=str)
parser.add_argument('-d', '--date', dest='timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00UTC)', type=int)
parser.add_argument("--radec", help="use RA/DEC instead of az/alt", action="store_true")
parser.add_argument("--verbose", help="more verbose labels", action="store_true")

parser.add_argument(action='store', dest='infile', help='input .pto file')
parser.add_argument(action='store', dest='outfile', help='output grid file (default "grid.png")', default="grid.png", nargs='?')
args = parser.parse_args()

pos = ephem.Observer()
if args.config:
    config = configparser.ConfigParser()
    config.read([args.config])
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
if args.timestamp:
    pos.date = datetime.fromtimestamp(float(args.timestamp), UTC).strftime('%Y-%m-%d %H:%M:%S')

pano = hsi.Panorama()
try:
    pano.ReadPTOFile(args.infile)
except:
    pano.readData(hsi.ifstream(args.infile))
img0 = pano.getImage(args.image)
width = img0.getWidth()
height = img0.getHeight()
scale = int(pano.getOptions().getWidth() / pano.getOptions().getHFOV())

inv = hsi.Transform()
trafo = hsi.Transform()
inv.createInvTransform(img0, pano.getOptions())

# Rotate from az/alt to RA/DEC?
if pos.date and pos.lat and pos.long and args.timestamp and args.radec:
    # To find pitch compute the altitude of the celestial pole
    star = ephem.FixedBody()
    star._ra, star._dec, star._epoch = str(0), str(90), ephem.J2000
    star.compute(pos)
    pitchrot = math.degrees(float(repr(star.alt)))-90

    # To find yaw take some point (like the frame centre) and find its RA in the current
    # coordinate system and compute how much off it is from the RA it should have
    center_az = 180 + img0.getYaw()
    center_alt = img0.getPitch()
    dst = hsi.FDiff2D();
    trafo.createTransform(img0, pano.getOptions())
    trafo.transformImgCoord(dst, hsi.FDiff2D(center_az*scale, (90-center_alt)*scale))
    hsi.RotatePanorama(pano, 0, pitchrot, 0).run()
    inv.createInvTransform(img0, pano.getOptions())
    inv.transformImgCoord(dst, hsi.FDiff2D(dst.x, dst.y))
    ra, _ = pos.radec_of(str(center_az), str(center_alt))
    yawrot = (-dst.x / scale - math.degrees(float(repr(ra)))) % 360
    hsi.RotatePanorama(pano, yawrot, 0, 0).run()
    inv.createInvTransform(img0, pano.getOptions())
    
trafo.createTransform(img0, pano.getOptions())

dst = hsi.FDiff2D();
centre = hsi.FDiff2D();
top = width/2

if args.width > 0:
    args.xscale = float(args.width) / width
if args.height > 0:
    args.yscale = float(args.height) / height

# Find diagonal FOV
inv.transformImgCoord(dst, hsi.FDiff2D(0, 0))
tlx = dst.x / scale
tly = dst.y / scale
inv.transformImgCoord(dst, hsi.FDiff2D(width, height))
brx = dst.x / scale
bry = dst.y / scale
x1 = math.radians(tlx)
x2 = math.radians(brx)
y1 = math.radians(tly)
y2 = math.radians(bry)
a = math.sin((y2-y1)/2) * math.sin((y2-y1)/2) + math.cos(y1) * math.cos(y2) * math.sin((x2-x1)/2) * math.sin((x2-x1)/2)
fov = math.degrees(2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

# Find az and alt range
minaz = minaz2 = 360
maxaz = maxaz2 = -360
minalt = 90
maxalt = -90

inv.transformImgCoord(centre, hsi.FDiff2D(width/2, height/2))

for i in range(0, width+1, 4):
    inv.transformImgCoord(dst, hsi.FDiff2D(i, 0))
    minaz = min(minaz, dst.x / scale)
    if dst.x / scale > 180:
        minaz2 = min(minaz2, dst.x / scale - 360)
    minalt = min(minalt, dst.y / scale)
    maxaz = max(maxaz, dst.x / scale)
    if dst.x / scale < 180:
        maxaz2 = max(maxaz2, dst.x / scale)
    maxalt = max(maxalt, dst.y / scale)
    inv.transformImgCoord(dst, hsi.FDiff2D(i, height))
    minaz = min(minaz, dst.x / scale)
    if dst.x / scale > 180:
        minaz2 = min(minaz2, dst.x / scale - 360)
    minalt = min(minalt, dst.y / scale)
    maxaz = max(maxaz, dst.x / scale)
    if dst.x / scale < 180:
        maxaz2 = max(maxaz2, dst.x / scale)
    maxalt = max(maxalt, dst.y / scale)

for i in range(0, height+1, 4):
    inv.transformImgCoord(dst, hsi.FDiff2D(0, i))
    minaz = min(minaz, dst.x / scale)
    if dst.x / scale > 180:
        minaz2 = min(minaz2, dst.x / scale - 360)
    minalt = min(minalt, dst.y / scale)
    maxaz = max(maxaz, dst.x / scale)
    if dst.x / scale < 180:
        maxaz2 = max(maxaz2, dst.x / scale)
    maxalt = max(maxalt, dst.y / scale)
    inv.transformImgCoord(dst, hsi.FDiff2D(width, i))
    minaz = min(minaz, dst.x / scale)
    if dst.x / scale > 180:
        minaz2 = min(minaz2, dst.x / scale - 360)
    minalt = min(minalt, dst.y / scale)
    maxaz = max(maxaz, dst.x / scale)
    if dst.x / scale < 180:
        maxaz2 = max(maxaz2, dst.x / scale)
    maxalt = max(maxalt, dst.y / scale)

# Handle az wrap
if maxaz - minaz > 180:
    minaz, maxaz = minaz2 + 360, maxaz2 + 360

azsep = 15 if args.radec else 10
minaz = max(int(minaz/azsep)*azsep, 0)
maxaz = int(maxaz/azsep)*azsep+azsep
maxalt = min(int(maxalt/10)*10+10, 180 if args.radec else 90)

if maxaz < minaz:
    maxaz += 360

minalt = max(int(minalt/10)*10, 0)

# Zenith in the frame?
trafo.transformImgCoord(dst, hsi.FDiff2D(0, 0))
if -10 < dst.x < width + 10 and -10 < dst.y < height + 10:
    minalt = 0
    minaz = 0
    maxaz = 360

minalt = int(minalt)
minalt3 = max(minalt, 10)

def refract(alt):
    return 0 if args.radec else 0.016/(math.tan(3.141592/180*(alt + (7.31/(alt + 4.4)))))

def scalealt(alt, scale):
    return alt*scale if args.radec else (90-((90-alt)+refract(90-alt)))*scale

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


def drawline(outermin, outermax, outerstep, innermin, innermax, innerstep, d, linelist):
    for outer in frange(outermin, outermax+0.01, outerstep):
        line = []
        for inner in frange(innermin, innermax+0.01, innerstep):
            x = (inner if d else outer)*scale
            y = scalealt(outer if d else inner, scale)
            trafo.transformImgCoord(dst, hsi.FDiff2D(x, y))
            #print(x / scale, y / scale, " -> ", dst.x, dst.y)
            dst.x = min(width+1000, max(-1000, dst.x))
            dst.y = min(height+1000, max(-1000, dst.y))
            line.append((dst.x * args.xscale, dst.y * args.yscale))
        linelist.append(line)

tiny_list = []
small_list = []
big_list = []

for alt in range(minalt + 10, maxalt+10, 10):
    if (maxaz - minaz >= 180):
        x = centre.x/scale
    else:
        x = math.cos(math.radians(maxaz)) + math.cos(math.radians(minaz));
        y = math.sin(math.radians(maxaz)) + math.sin(math.radians(minaz));
        x = math.degrees(math.atan2(y, x))
    trafo.transformImgCoord(dst, hsi.FDiff2D(x*scale, alt*scale))
    tmp1 = dst.x
    tmp2 = dst.y
    trafo.transformImgCoord(dst, hsi.FDiff2D((x+1)*scale, alt*scale))
    dist = math.sqrt((dst.x - tmp1)*(dst.x - tmp1)+(dst.y - tmp2)*(dst.y - tmp2))
    base = 16
    if dist > base:
        spacing = 1
        dspacing = 1.25
    elif dist > base/2:
        spacing = 2
        dspacing = 2.5
    elif dist > base/5:
        spacing = 5
        dspacing = 5
    else:
        spacing = 10
        dspacing = 7.5

    if spacing < 10:
        drawline(minaz, maxaz, dspacing if args.radec else spacing, alt - 10, alt, 2, False, small_list)
        if dist <= base:
            if args.radec:
                drawline(minaz+dspacing/2.0, maxaz+dspacing/2.0, dspacing, alt - 10, alt, 2, False, tiny_list)
            else:
                drawline(minaz+spacing/2.0, maxaz+spacing/2.0, spacing, alt - 10, alt, 2, False, tiny_list)
            drawline(alt-9.5, alt+0.5, 1, minaz, maxaz, 1, True, tiny_list)

    if spacing < 10:
        if dist > base*5:
            drawline(minaz, maxaz, 0.25 if args.radec else 0.1, alt - 10, alt, 2, False, tiny_list)
            drawline(alt - 10, alt, 0.1, minaz, maxaz, 1, True, tiny_list)
        elif dist > base*2.5:
            drawline(minaz, maxaz, 0.625 if args.radec else 0.2, alt - 10, alt, 2, False, tiny_list)
            drawline(alt - 10, alt, 0.2, minaz, maxaz, 1, True, tiny_list)
        elif dist > base:
            drawline(minaz, maxaz, 0.625 if args.radec else 0.5, alt - 10, alt, 2, False, tiny_list)
            drawline(alt - 10, alt, 0.5, minaz, maxaz, 1, True, tiny_list)

drawline(minalt, maxalt, 1, minaz, maxaz, 1, True, small_list)
if args.radec:
    drawline(minaz, maxaz-1, 15, minalt3, maxalt, 1, False, big_list)
else:
    drawline(minaz, maxaz-1, 10, minalt3, maxalt, 1, False, big_list)

drawline(minalt, maxalt, 10, minaz, maxaz, 1, True, big_list)
if abs(minalt) <= 10:
    drawline(minaz, maxaz-1, 15 if args.radec else 20, 0, 10, 1, False, big_list)

minalt = max(minalt, 10)

textlist = []
label_spacing_lat = args.label_spacing
label_spacing_lon = int(args.label_spacing * 1.5) if args.radec else args.label_spacing
for az in range(minaz, maxaz, label_spacing_lon):
    for alt in range(maxalt-label_spacing_lat, minalt3-1, -label_spacing_lat):
        if alt == 10 and az % (30 if args.radec else 20) == 0:
            continue
        alt2 = (alt-refract(90-alt))*scale
        alt3 = 90-alt
        az2 = az*scale
        trafo.transformImgCoord(dst, hsi.FDiff2D(az2, alt2))
        x = dst.x
        y = dst.y
        trafo.transformImgCoord(dst, hsi.FDiff2D(az2, alt2+scale))
        angle = -math.atan2(dst.x - x, dst.y - y)*180/math.pi

        trafo.transformImgCoord(dst, hsi.FDiff2D(az2, alt2))
        x2 = dst.x
        y2 = dst.y
        trafo.transformImgCoord(dst, hsi.FDiff2D(az2+scale, alt2))
        angle2 = -math.atan2(dst.x - x2, dst.y - y2)*180/math.pi

        if -100 < x < width+100 and -100 < y < height+100:
            if args.radec:
                lontext = str(int((360 - az) / 15) % 24) + ':00:00'
            else:
                lontext = str(int(az % 360))
            textlist.append((' ' + lontext, angle+90, x+3*math.cos(angle/180*math.pi), y+3*math.sin(angle/180*math.pi), 'left'))
            textlist.append((str(int(alt3)) + ' ', angle2+90, x+4*math.cos(angle2/180*math.pi), y+4*math.sin(angle2/180*math.pi), 'right'))


image = wand.image.Image(width=int(round(width * args.xscale)), height=int(round(height * args.yscale)), background=wand.color.Color('transparent'))

# Draw text at an angle
def writetext(draw, text, x, y, angle, alignment):
    if x < 0 or y < 0:
        return
    x = x * args.xscale
    y = y * args.yscale
    offset = 0
    draw.text_alignment = alignment
    draw.translate(x+offset, y-offset)
    draw.rotate(angle)
    draw.text(0, 0, text)
    draw.rotate(-angle)
    draw.translate(-x-offset, -y+offset)

with wand.drawing.Drawing() as draw:
    draw.fill_color = wand.color.Color('none')
    draw.stroke_color = wand.color.Color('grey50')
    draw.stroke_opacity = 0.1
    for line in tiny_list:
        draw.polyline(line)
    draw.stroke_opacity = 0.5
    for line in small_list:
        draw.polyline(line)
    draw.stroke_opacity = 0.8
    draw.stroke_color = wand.color.Color('yellow')
    for line in big_list:
        draw.polyline(line)

    draw.font = 'helvetica'
    draw.fill_color = wand.color.Color('yellow')
    draw.stroke_color = wand.color.Color('transparent')
    draw.stroke_opacity = 0
    draw.font_size = 14
    for text, angle, x, y, alignment in textlist:
        writetext(draw, text, x, y, angle, alignment)

    # Annotate if we have a position and timestamp
    if pos.date and pos.lat and pos.long and args.timestamp:
        # Rotate back to az/alt if we have a RA/DEC coordinate system
        if args.radec:
            hsi.RotatePanorama(pano, -yawrot, 0, 0).run()
            hsi.RotatePanorama(pano, 0, -pitchrot, 0).run()

        stars = brightstar(pano, args.image, pos, args.faintest, args.brightest, args.objects*100)
        draw.stroke_color = wand.color.Color('white')
        draw.stroke_width = 1
        draw.fill_color = wand.color.Color('transparent')
        for s in stars:
            x, y = s[0] * args.xscale, s[1] * args.yscale
            draw.circle((x, y), (x + 5, y + 5))
        draw.translate(0, 0)
        draw.rotate(0)
        draw.text_alignment = 'left'
        draw.stroke_opacity = 1
        draw.stroke_color = wand.color.Color('transparent')
        for s in stars:
            draw.fill_color = wand.color.Color('white')
            if args.radec:
                ra, dec = pos.radec_of(str(s[2]), str(s[3]))
                if args.verbose:
                    draw.text(int(s[0] * args.xscale + 10), int(s[1] * args.yscale + 6),
                              s[4] + " " + str(round(s[5], 1)) + " [" + re.sub(r'\..*', '', str(ra)) + ", " + str(round(math.degrees(float(repr(dec))), 2)) + "]")
                else:
                    draw.text(int(s[0] * args.xscale + 10), int(s[1] * args.yscale + 6), s[4])
            else:
                draw.text(int(s[0] * args.xscale + 10), int(s[1] * args.yscale + 6),
                          s[4] + " " + str(round(s[5], 1)) + " [" + str(round(s[2], 2)) + ", " + str(round(s[3], 2)) + "]")
    
    draw(image)

    image.format = 'png'
    image.save(filename=args.outfile)
