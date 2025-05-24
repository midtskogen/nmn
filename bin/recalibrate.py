#!/usr/bin/python3

# Improve calibration of a pto file using stars in the image.
# Usage: recalibrate.py <Unix timestamp> <input pto file> <image file> <output pto file>
import sys
import ephem
from datetime import datetime, UTC
import math
import hsi
import argparse
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
import os
import tempfile
import subprocess
from astropy.io import fits
import configparser
from stars import cat

parser = argparse.ArgumentParser(description='Improve calibration of a pto file using stars in the image. This tool works in these steps: 1) Use the timestamp, input .pto file and image to look for stars nearby their expected positions. 2) Get the coordinates of the actual positions of the stars. 3) Reoptimise the pto file and output it into a new file.')

parser.add_argument('-n', '--number', dest='objects', help='maximum number of objects (default: 500)', default=500, type=int)
parser.add_argument('-f', '--faintest', dest='faintest', help='faintest objects to include (default: 3)', default=3, type=float)
parser.add_argument('-a', '--altitude', dest='sunalt', help="don't recalibrate if the sun is this many degrees below the horizon or higher (default: 8)", default=-5, type=float)
parser.add_argument('-b', '--brightest', dest='brightest', help='brightest objects to include (default: -5)', default=-5, type=float)
parser.add_argument('-x', '--longitude', dest='longitude', help='observer longitude', type=float)
parser.add_argument('-y', '--latitude', dest='latitude', help='observer latitude', type=float)
parser.add_argument('-e', '--elevation', dest='elevation', help='observer elevation (m)', type=float)
parser.add_argument('-t', '--temperature', dest='temperature', help='observer temperature (C)', type=float)
parser.add_argument('-p', '--pressure', dest='pressure', help='observer air pressure (hPa)', type=float)
parser.add_argument('-i', '--image', dest='image', help='which image in the .pto file to use (default: 0)', default=0, type=int)
parser.add_argument('-r', '--radius', dest='radius', help='search radius around stars (degrees, default: 0.8)', default=0.8, type=float)
parser.add_argument('-g', '--gaussian', dest='blur', help='amount of blurring (percentage of radius default: 50)', default=50, type=float)
parser.add_argument('-q', '--sigma', dest='sigma', help='noise level (default: 20)', default=20.0, type=float)
parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='be more verbose')
parser.add_argument('-l', '--lens_optimise', action='store_true', dest='lensopt', help='optimise lens, not only orientation')

parser.add_argument(action='store', dest='timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00UTC)')
parser.add_argument(action='store', dest='infile', help='Hugin .pto file (input)')
parser.add_argument(action='store', dest='picture', help='Image file')
parser.add_argument(action='store', dest='outfile', help='Hugin .pto file (output)')


def recalibrate(timestamp, infile, picture, outfile, pos, image, radius = 1.0, lensopt = False, faintest = 3, brightest = -5, objects = 500, blur = 50, verbose = False, sigma = 20):
    pano = hsi.Panorama()
    try:
        pano.ReadPTOFile(infile)
    except:
        pano.readData(hsi.ifstream(infile))
    img = pano.getImage(image)
    width = img.getWidth()
    height = img.getHeight()
    scale = int(pano.getOptions().getWidth() / pano.getOptions().getHFOV())
    dst = hsi.FDiff2D();
    trafo = hsi.Transform()
    trafo.createTransform(img, pano.getOptions())

    radius = int(radius*width/img.getHFOV())
    blur = radius * blur / 100

    starlist = []

    def test_body(body, name, faintest, brightest):
        body.compute(pos)
        az = math.degrees(float(repr(body.az)))
        alt = math.degrees(float(repr(body.alt)))
        alt2 = alt + 0.01666 / math.tan(math.pi/180*(alt + (7.31/(alt + 4.4))))
        if alt > -0.5 and body.mag <= faintest and body.mag >= brightest:
            trafo.transformImgCoord(dst, hsi.FDiff2D(az*scale, (90-alt2)*scale));
            if dst.x > 0 and dst.y > 0 and dst.x < width and dst.y < height:
                starlist.append((dst.x, dst.y, 0, 0))
                return 1
        return 0

    count = 0
    for (body, name) in [ (ephem.Sun(), "Sol"), (ephem.Moon(), "Moon") , (ephem.Mercury(), "Mercury"), (ephem.Venus(), "Venus"), (ephem.Mars(), "Mars"), (ephem.Jupiter(), "Jupiter"), (ephem.Saturn(), "Saturn") ]:
        count += test_body(body, name, faintest, brightest)
        if count == objects:
            break

    if count < objects:
        for (ra, pmra, dec, pmdec, mag, name) in cat:
            if (mag <= faintest and mag >= brightest):
                body = ephem.FixedBody()
                body._ra, body._pmra, body._dec, body._pmdec, body._epoch = str(ra), pmra, str(dec), pmdec, ephem.J2000
                count += test_body(body, name, 5, -30)

                if count == objects:
                    break

    stars = Image(width=width, height=height, background=Color('black'))
    if isinstance(picture, str):
        pic = Image(filename=picture)
    else:
        pic = picture
    axy = []

    with Drawing() as draw:
        draw.fill_color = Color('white')
        masked = stars.clone()
        draw.stroke_color = Color('white')
        for (x, y, _, _) in starlist:
            draw.circle((x, y), (x+radius, y))

        draw(stars)
        stars.gaussian_blur(blur, blur)
        pic.gaussian_blur(blur/32, blur/32)
        masked = stars.clone()
        with Drawing() as draw:
            draw.composite(operator='bumpmap', left=0, top=0,
                           width=pic.width, height=pic.height, image=pic)
            draw(masked)



        temp = tempfile.NamedTemporaryFile(prefix='recalibrate_', suffix='.png', dir='/tmp', delete=False)
        masked.format = 'png'
        masked.save(temp)
        temp.close()

        out = sys.stdout if verbose else open(os.devnull, 'wb')

        try:
            #print(['solve-field', '--overwrite', '--sigma', str(sigma), '--just-augment', temp.name])
            subprocess.Popen(['solve-field', '--sigma', str(sigma), '--just-augment', temp.name], stdout=out, stderr=out).wait()
            axyfile = os.path.splitext(temp.name)[0] + '.axy'
            axy = fits.open(axyfile)
            os.unlink(axyfile)
        except Exception as e:
            print(str(e))
            os.unlink(temp.name)
            exit(1)

        #os.unlink(temp.name)

    def dist(x1, y1, x2, y2):
        return math.sqrt((x2-x1)**2+(y2-y1)**2)

    remap = []

    for (x1, y1, _, _) in starlist:
        mindist = 9999999
        x, y = -1, -1
        for (x2, y2, _, _) in axy[1].data:
            x2 -= 1
            y2 -= 1
            d = dist(x1, y1, x2, y2)
            if d < mindist and d <= radius:
                mindist = d
                x, y = x2, y2
        if mindist <= radius:
            remap.append((x1, y1, x, y))

    if verbose:
        print('Stars expected: ' + str(len(starlist)))
        print('Stars found: ' + str(len(axy[1].data)))
        print('Stars remapped: ' + str(len(remap)))
        for (x1, y1, x2, y2) in remap:
            print('{0:8.3f},{1:8.3f} -> {2:8.3f},{3:8.3f}'.format(x1, y1, x2, y2))

    pano.addImage(img)
    pano.addImage(img)
    while pano.getNrOfImages() > 2:
        pano.removeImage(0)

    for (x1, y1, x2, y2) in remap:
        pano.addCtrlPoint(hsi.ControlPoint(0, float(x2), float(y2), 1, float(x1), float(y1)))

    if lensopt:
        #pano.setOptimizeVector([('r', 'p', 'y', 'v', 'a', 'b', 'c', 'd', 'e'), ()])
        pano.setOptimizeVector([('r', 'p', 'y', 'v', 'b', 'c', 'd', 'e'), ()])
    else:
        pano.setOptimizeVector([('r', 'p', 'y'), ()])


    try:
        pano.WritePTOFile(outfile)
    except:
        pano.writeData(hsi.ofstream(outfile))
    try:
        subprocess.Popen(['cpclean', '-n', '1', '-o', outfile, outfile], stdout=out, stderr=out).wait()
    except:
        print('Failed to run cpclean')

    try:
        subprocess.Popen(['autooptimiser', '-n', '-o', outfile, outfile], stdout=out, stderr=out).wait()
    except:
        print('Failed to run autooptimiser')


if __name__ == '__main__':
    args = parser.parse_args()
    pos = ephem.Observer()
    config = configparser.ConfigParser()
    config.read(['/etc/meteor.cfg', os.path.expanduser('~/meteor.cfg')])
    pos.lat = config.get('astronomy', 'latitude')
    pos.lon = config.get('astronomy', 'longitude')
    pos.elevation = float(config.get('astronomy', 'elevation'))
    pos.temp = float(config.get('astronomy', 'temperature'))
    pos.pressure = float(config.get('astronomy', 'pressure'))
    if args.longitude:
        pos.lon = args.longitude
    if args.latitude:
        pos.lat = args.latitude
    if args.elevation:
        pos.elevation = args.elevation
    if args.temperature:
        pos.temp = args.temperature
    if args.pressure:
        pos.pressure = args.pressure

    pos.date = datetime.fromtimestamp(float(args.timestamp), UTC).strftime('%Y-%m-%d %H:%M:%S')
    recalibrate(args.timestamp, args.infile, args.picture, args.outfile, pos, args.image, args.radius, args.lensopt, args.faintest, args.brightest, args.objects, args.blur, args.verbose, args.sigma)
