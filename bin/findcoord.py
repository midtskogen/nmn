#!/usr/bin/env python3

# Find ra and dec given star name

# Example:
# awk '{print("echo "$1" $(~/bin/findcoord.py \""$2"\" 2> /dev/null) "$2" "$3" "$4" "$5)}' stars.id.orig | bash > stars.id

import sys
from stars import cat
import argparse
import difflib

parser = argparse.ArgumentParser(description='Find ra and dec given star name.')

parser.add_argument(action='store', dest='star', help="star name")
parser.add_argument('-m', '--magnitude', action='store_const', dest='mag', const=sum, default=0, help="print magnitude instead")

args = parser.parse_args()

match = difflib.get_close_matches(args.star, [x[5] for x in cat])[0]

for (ra2, p_ra2, dec2, p_dec2, mag, name) in cat:
    if match == name:
        if args.mag:
            print(mag)
        else:
            print(ra2, dec2)
        break
