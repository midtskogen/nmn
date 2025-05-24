#!/usr/bin/env python3

from PIL import Image
import sys

if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + " <filename>")
    exit(1)

img = Image.open(sys.argv[1]).convert('YCbCr')
sum = 0
for y in range(0, img.height, 8):
    for x in range(0, img.width, 8):
        (_, u, v) = img.getpixel((x, y))
        sum = sum + max(0, 64-min(u, v))

print(sum)
