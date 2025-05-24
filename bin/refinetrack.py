#!/usr/bin/env python3

from PIL import Image
import argparse
import math
import numpy as np
from math import factorial

def coords(s):
    try:
        x, y = map(float, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x,y")


parser = argparse.ArgumentParser(description='Find the accurate start and end positions of a meteor track in rectilinear projection given approximate start and end positions.')

parser.add_argument('-r', '--radius', dest='radius', help='search radius (default: 100)', default=100, type=int)
parser.add_argument(action='store', dest='img', help='Image (rectilinear projection)')
parser.add_argument(action='store', dest='start', help='Approximate start position', type=coords)
parser.add_argument(action='store', dest='end', help='Approximate end position', type=coords)
args = parser.parse_args()

im = Image.open(args.img)
img = im.load()

x1, y1 = args.start
x2, y2 = args.end

def frange(x, y, step):
    while x < y:
        yield x
        x += step

def dist(start, end):
    return math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)

def line(center, angle, length, steps):
    start = (center[0] - math.cos(angle) * length / 2,
             center[1] - math.sin(angle) * length / 2)
    line = []
    for i in frange(0, length, float(length)/steps):
        line.append((start[0] + math.cos(angle) * i, start[1] + math.sin(angle) * i))
    return line

def brightness(img, start, end, steps):
    l = []
    b = 0
    length = dist(start, end)
    angle = math.atan2(end[1]-start[1], end[0]-start[0])
    for i in frange(0, length, length / steps):
        x, y = (start[0] + math.cos(angle) * i, start[1] + math.sin(angle) * i)
        if steps < length:
            try:
                l.append(sum(img[x, y][0:2]))
            except:
                pass
        else:
            try:
                xfrac = x - int(x)
                yfrac = y - int(y)
                p00 = sum(img[x, y][0:2])
                p10 = sum(img[x + 1, y][0:2])
                p01 = sum(img[x, y + 1][0:2])
                p11 = sum(img[x + 1, y + 1][0:2])
                p0 = p00 * (1-xfrac) + p10 * xfrac
                p1 = p01 * (1-xfrac) + p11 * xfrac
            
                l.append(p0 * (1-yfrac) + p1 * yfrac)
            except:
                pass
    return l

def track(img, start, end, steps):
    l = []
    b = 0
    length = dist(start, end)
    angle = math.atan2(end[1]-start[1], end[0]-start[0])
    (lastx, lasty) = start
    for i in frange(0, length, length / steps):
        x, y = (start[0] + math.cos(angle) * i, start[1] + math.sin(angle) * i)
        try:
            xfrac = x - int(x)
            yfrac = y - int(y)
            p00 = sum(img[x, y][0:2])
            p10 = sum(img[x + 1, y][0:2])
            p01 = sum(img[x, y + 1][0:2])
            p11 = sum(img[x + 1, y + 1][0:2])
            p0 = p00 * (1-xfrac) + p10 * xfrac
            p1 = p01 * (1-xfrac) + p11 * xfrac
            
            avg, cnt = 0.0, 0
            for j in line((x, y), angle + math.pi/2, 30, 30):
                cnt += 1
                avg += sum(img[j[0], j[1]][0:2])
            avg /= cnt
            b = p0 * (1-yfrac) + p1 * yfrac <= avg
            b2 = p0 * (1-yfrac) + p1 * yfrac > avg + 24
            l.append((b, lastx, lasty, x, y, b2))
            if b:
                lastx, lasty = x, y
        except:
            pass
    return l

angle = math.atan2(y2-y1, x2-x1)
length = dist(args.start, args.end)

# Fast approximation
startline = line(args.start, angle-math.pi/2, args.radius*2, args.radius)
endline = line(args.end, angle-math.pi/2, args.radius*2, args.radius)

best = 0
besta, bestb = (0, 0), (0, 0)
bestl = []
for a in startline:
    for b in endline:
        l = brightness(img, a, b, length/5)
        c = sum(l)
        if c > best:
            besta, bestb, best, bestl = a, b, c, l

# Look for periodic brightness fluctuations (airplanes)
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

def smooth(x,window_len=6,window='flat'):
        if x.ndim != 1:
                raise ValueError("smooth only accepts 1 dimension arrays.")
        if x.size < window_len:
                raise ValueError("Input vector needs to be bigger than window size.")
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:  
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

# Refine
angle = math.atan2(bestb[1]-besta[1], bestb[0]-besta[0])
besta = (besta[0] + math.cos(angle + math.pi) * length*10, besta[1] + math.sin(angle + math.pi) * length*10)
bestb = (bestb[0] + math.cos(angle) * length*10, bestb[1] + math.sin(angle) * length*10)

first = None
last = None
for i in track(img, besta, bestb, dist(besta, bestb)*10):
    if i[5]:
        last = (i[3],i[4])
        if not first:
            first = (i[3],i[4])

if first:
    besta = first
if last:
    bestb = last
args.radius = max(80, args.radius)

startline = line(besta, angle-math.pi/2, args.radius/40, args.radius/8)
endline = line(bestb, angle-math.pi/2, args.radius/40, args.radius/8)

best = 0
besta, bestb = (0, 0), (0, 0)
for a in startline:
    for b in endline:
        c = sum(brightness(img, a, b, length*2))
        if c >= best:
            besta, bestb = a, b
            best = c


t = track(img, besta, bestb, dist(besta, bestb)*10)

# Find longest run where the track line is brigher than the average of the perpendicular line
cnt = 0
run = 0
l = []
for i in t:
    if i[0] == True:
        l.append((run, i[1], i[2], i[3], i[4]))
        run = 0
    else:
        run += 1
    cnt += 1

if run:
    l.append((run, i[1], i[2], i[3], i[4]))
    

if l:
    r = max(l, key=lambda x:x[0])
    # Use besta and bestb if we got a very short line
    if length > dist((r[1], r[2]), (r[3], r[4]))*1.5:
        print('{0:.2f},{1:.2f} {2:.2f},{3:.2f}'.format(besta[0], besta[1], bestb[0], bestb[1]))
    else:
        print('{0:.2f},{1:.2f} {2:.2f},{3:.2f}'.format(r[1], r[2], r[3], r[4]))
else:
    print('{0:.2f},{1:.2f} {2:.2f},{3:.2f}'.format(args.start[0], args.start[1], args.end[0], args.end[1]))

# Look for periodic brightness fluctuations (airplanes)
try:
    a = autocorr(smooth(np.array(bestl)))
    b = smooth(a, 10, "hamming")
    c = smooth(a - b, a.size/8, "hamming")
    c = c - sum(c)/c.size
    n = c.size
    if (((c[:-1] * c[1:]) < 0).sum() > 8 and sum(abs(c))/c.size > 1000):
        exit(-1)
except:
    pass
