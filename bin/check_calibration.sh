#!/bin/bash

dir=$(dirname $1)/../..

read width height <<< $(ffmpeg -nostdin -i $1 2>&1 |egrep -o '[0-9][0-9]+x[0-9]+' | sed 's/x/ /')

let timestamp=$(~/bin/timestamp $1)+5

ffmpeg -nostdin -loglevel quiet -vsync 0 -i $1 -t 10 -pix_fmt yuv420p -f rawvideo - | ~/bin/stack $width $height > /tmp/image-$$.jpg

convert -size $(identify /tmp/image-$$.jpg | awk '{print $3}') xc:none /tmp/empty-$$.png
echo convert -pointsize 12 $(~/bin/brightstar.py -f 2.5 $(~/bin/timestamp /tmp/image-$$.jpg) $dir/lens.pto 2> /dev/null | sed 's/[(),]//g;s/'\''//g' | awk '{x=$1; y=$2; az=$3; alt=$4; $1=$2=$3=$4=""; sub(/ */, ""); printf("-stroke white -fill none -draw \"circle %f,%f %f,%f\" -stroke none -fill white -annotate +%f+%f \"%s [%.2f %.2f]\"\n", x, y, x+7, y, x+11, y-4, $0, az, alt)}') /tmp/empty-$$.png /tmp/labels-$$.png | bash
composite -blend 70 /tmp/image-$$.jpg /tmp/labels-$$.png cal.jpg
rm -f /tmp/labels-$$.png /tmp/empty-$$.png /tmp/image-$$.jpg

echo Wrote cal.jpg
