#!/bin/bash
# Usage: recalibrate_lens.sh [-l] <video.mp4>
# Output: calibration_old.jpg calibration_new.jpg lens_new.pto

# Look at the stars in the video and ../../lens.pto and try to improve.

if [ "$1" == "-l" ]; then
    recopt="-l"
    file=$2
else
    file=$1
fi

size=$(ffmpeg -i $file 2>&1 | grep Video: | sed 's/.*\([0-9]\{4\}x[0-9]\{4\}\).*/\1/')
width=$(echo $size | sed 's/x.*//')
height=$(echo $size | sed 's/.*x//')
ts=$(~/bin/timestamp $file)
let ts=ts+10

echo "Resolution $width"x"$height"
echo "Timestamp: "$(date -u -d "1970-01-01 $ts seconds")" ($ts)"

lens="$(realpath $(dirname $file)/../..)/lens.pto"
grid="$(realpath $(dirname $file)/../..)/grid.png"
date=$(date +%Y%m%d -u -d "1970-01-01 UTC $ts seconds")
lens2="$(realpath $(dirname $file)/../..)/lens-$date.pto"
grid2="$(realpath $(dirname $file)/../..)/grid-$date.png"

echo -n "Recalibrating... "

ffmpeg -loglevel quiet -i $file -v warning -t 20 -vsync 0 -nostdin -pix_fmt yuv420p -f rawvideo - 2> /dev/null | ~/bin/avgstack $width $height > /tmp/recal_$$.jpg

convert -size $size xc:none /tmp/empty-$$.png

echo convert -pointsize 12 $(~/bin/brightstar.py -f 3.5 $ts $lens 2> /dev/null | sed 's/[(),]//g;s/'\''//g' | awk '{x=$1; y=$2; az=$3; alt=$4; $1=$2=$3=$4=""; sub(/ */, ""); printf("-stroke white -fill none -draw \"circle %f,%f %f,%f\" -stroke none -fill white -annotate +%f+%f \"%s [%.2f %.2f]\"\n", x, y, x+7, y, x+11, y-4, $0, az, alt)}') /tmp/empty-$$.png /tmp/labels-$$.png | bash

composite -blend 80 /tmp/recal_$$.jpg /tmp/labels-$$.png calibration_old.jpg

~/bin/recalibrate.py $recopt -r 1 $ts $lens /tmp/recal_$$.jpg lens_new.pto

echo convert -pointsize 12 $(~/bin/brightstar.py -f 4 $ts lens_new.pto 2> /dev/null | sed 's/[(),]//g;s/'\''//g' | awk '{x=$1; y=$2; az=$3; alt=$4; $1=$2=$3=$4=""; sub(/ */, ""); printf("-stroke white -fill none -draw \"circle %f,%f %f,%f\" -stroke none -fill white -annotate +%f+%f \"%s [%.2f %.2f]\"\n", x, y, x+7, y, x+11, y-4, $0, az, alt)}') /tmp/empty-$$.png /tmp/labels-$$.png | bash

composite -blend 80 /tmp/recal_$$.jpg /tmp/labels-$$.png calibration_new.jpg

rm -f /tmp/empty-$$.png /tmp/recal-$$.png /tmp/labels-$$.png

echo done.

echo Showing new calibaration...  Please check.
echo

eog calibration_new.jpg 2> /dev/null &

read -p "Update $lens and $grid? > " update

sed "s/i w$width h$height.*/$(grep -m1 "i w$width h$height" lens_new.pto | sed "s/ n.*//")/" < $lens > lens_new.pto

if [[ "${update:0:1}"  == "y" ]]; then
    echo -n "Making new grid (this may take several minutes)... "
    ~/bin/drawgrid.py lens_new.pto
    cp lens_new.pto $lens2; rm $lens; ln -s $(basename $lens2) $lens; cp grid.png $grid2; rm $grid; ln -s $(basename $grid2) $grid
    echo done.
fi
