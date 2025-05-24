#!/usr/bin/env python3

import sys
import re
from dateutil.parser import parse

f = open(sys.argv[1])
l = f.readline().split(' ')
framerate = float(l[2]) / float(l[5])
date = parse(l[8] + ' ' + l[9] + ' ' + l[10])
l = f.readline().split(' ')

path = []
for i in l:
    t = re.split(r',|\(|\)', i)[:3]
    if len(t) > 1:
        path.append(tuple(t))

print(path[0])
print(path[-1])

print(framerate)
print(date)
