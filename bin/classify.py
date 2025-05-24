#!/usr/bin/env python3

# Usage processreport.py <event dir>

import configparser
import sys
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
import glob
import hsi
import math

# Width of final track in pixels
width = 128

# Read start and end positions from summary
config = configparser.ConfigParser()
config.read([sys.argv[1] + '/event.txt'])
startpos = config.get('summary', 'startpos').split()
endpos = config.get('summary', 'endpos').split()

# Get projection
pano = hsi.Panorama()
try:
    try:
        pano.ReadPTOFile(sys.argv[1] + '/gnomonic_corr_grid.pto')
    except:
        pano.readData(hsi.ifstream(sys.argv[1] + '/gnomonic_corr_grid.pto'))
except:
    try:
        pano.ReadPTOFile(sys.argv[1] + '/gnomonic_grid.pto')
    except:
        pano.readData(hsi.ifstream(sys.argv[1] + '/gnomonic_grid.pto'))
    
# Translate az/alt of start and end positions into pixel coordinates
dst = hsi.FDiff2D();
trafo = hsi.Transform()
trafo.createTransform(pano.getImage(0), pano.getOptions())
scale = int(pano.getOptions().getWidth() / pano.getOptions().getHFOV())
trafo.transformImgCoord(dst, hsi.FDiff2D(float(startpos[0]) * scale, (90 - float(startpos[1])) * scale))
start = [dst.x, dst.y]
trafo.transformImgCoord(dst, hsi.FDiff2D(float(endpos[0]) * scale, (90 - float(endpos[1])) * scale))
end = [dst.x, dst.y]
pic = Image(filename=glob.glob(sys.argv[1] + '/*-gnomonic.jpg')[0])

# Background removal
pic2 = pic.clone()
pic2.resize(filter='box', blur=512)
pic2.modulate(brightness=150,saturation=0)
with Drawing() as draw:
    draw.composite(operator='minus', left=0, top=0, width=pic.width, height=pic.height, image=pic)
    draw(pic2)

# Rotate.  The fireball should be in the centre
pic2.rotate(-math.atan2(end[1] - start[1], end[0] - start[0])*180/math.pi, background=Color('black'))

# Draw a line in a blank frame for the track we want and use that as a mask
track = Image(width=pic2.width, height=pic2.height, background=Color('black'))
with Drawing() as draw:
    len = math.sqrt((start[0] - end[0]) * (start[0] - end[0]) + (start[1] - end[1]) * (start[1] - end[1])) + width
    draw.stroke_width = width
    draw.stroke_color = Color('white')
    draw.line([(pic2.width - len) / 2, pic2.height / 2], [(pic2.width + len) / 2, pic2.height / 2])
    draw(track)
    draw.composite(operator='multiply', left=0, top=0, width=pic2.width, height=pic2.height, image=pic2)
    draw(track)

# Trim, normalise and save image.
track.trim()
track.normalize()
track.format = 'jpeg'
track.save(filename=sys.argv[1] + '/fireball.jpg')
