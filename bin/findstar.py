#!/usr/bin/env python3

# Find a star name given ra and dec or find ra and dev given a star name

import sys
from stars import cat
import argparse

parser = argparse.ArgumentParser(description='Find star given ra and dec or find name given ra and dec.')

if len(sys.argv) > 2:
    parser.add_argument(action='store', dest='ra', help="ra coordinate")
    parser.add_argument(action='store', dest='dec', help="dec coordinate")
    parser.add_argument('-m', '--magnitude', action='store_const', dest='mag', const=sum, default=0, help="print magnitude instead")

    args = parser.parse_args()
    bestdist = 9999
    bestname = '?'
    bestmag = 9999

    for (ra2, p_ra2, dec2, p_dec2, mag, name) in cat:
        dist = (float(ra2)*360/24-float(args.ra)*360/24)*(float(ra2)*360/24-float(args.ra)*360/24)+(float(dec2)-float(args.dec))*(float(dec2)-float(args.dec))
        if dist < 0.2*0.2 and dist < bestdist:
            bestdist = dist
            bestname = name
            bestmag = mag

    if args.mag:
        print(bestmag)
    else:
        print(bestname)

else:
    parser.add_argument(action='store', dest='name', help="star name")
    args = parser.parse_args()
    Heap = []
    for (ra2, p_ra2, dec2, p_dec2, mag, name) in cat:
        if args.name.lower() == name.lower():
            print(str(ra2) + ' ' + str(dec2))
        



