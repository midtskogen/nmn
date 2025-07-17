#!/bin/bash

echo 'p f2 w4480 h2240 v360 E0 R0 S0,4480,0,1280 n"TIFF_m c:LZW"' > equirect.pto
echo 'm g1 i0 f0 m2 p0.00784314' >> equirect.pto

echo 'p f3 w5120 h5120 v360  k0 E0 R0 S1248,3872,1248,3872 n"TIFF_m c:LZW"' > fisheye.pto
echo 'm g1 i0 f0 m2 p0.00784314' >> fisheye.pto

for i in /meteor/cam?/lens.pto; do
    str=$(dirname $i)
    j="${str: -1}"
    echo $(grep -m1 "^i w" $i | sed 's/n".*//')" n\"full$j.jpg\"" >> equirect.pto
    echo $(grep -m1 "^i w" $i | sed 's/n".*//')" n\"full$j.jpg\"" >> fisheye.pto
done

pano_modify --rotate 0,-90,0 fisheye.pto -o fisheye.pto
