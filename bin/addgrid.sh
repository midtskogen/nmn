#!/bin/bash

# addgrid.sh <input mp4> <grid file> <pto file> <output mp4> [start] [length]

if [ -n "$5" ]; then start=$5; else start=0; fi
if [ -n "$6" ]; then len=$6; else len=60; fi

jpg=$(dirname $4)/$(basename $4 .mp4).jpg

ts=$(~/bin/timestamp $1 | tail -n1)
let ts=ts+$start
let ts2=ts+$len/2
let ts3=ts+$len-1

echo convert -pointsize 12 $(~/bin/brightstar.py $ts2 $3 | awk '{x=$1; y=$2; az=$3; alt=$4; $1=$2=$3=$4=""; sub(/ */, ""); printf("-stroke white -fill none -draw \"circle %f,%f %f,%f\" -stroke none -fill white -annotate +%f+%f \"%s [%.2f %.2f]\"\n", x, y, x+7, y, x+11, y-4, $0, az, alt)}') $2 /tmp/grid-labels.png | bash

#count=0;
#for i in $(seq $ts 1 $ts3); do
#    c=$(echo $count | awk '{printf("%03d", $count)}')
#    echo convert -pointsize 12 $(~/bin/brightstar.py $i $3 | awk '{x=$1; y=$2; az=$3; alt=$4; $1=$2=$3=$4=""; sub(/ */, ""); printf("-stroke yellow -fill none -draw \"circle %f,%f %f,%f\" -stroke none -fill yellow -annotate +%f+%f \"%s [%.2f %.2f]\"\n", x, y, x+7, y, x+11, y-4, $0, az, alt)}') $2 /tmp/grid-labels-$c.png | bash
#    let count=count+1
#done

ffmpeg -loglevel quiet -loop 1 -r 0.001 -i /tmp/grid-labels.png -i $1 -filter_complex "[0:0]setsar=1,format=rgb24[a];[1:0]setsar=1,lut="y=val+8",hqdn3d=10,format=rgb24[b];[b][a]blend=all_mode='addition':all_opacity=0.12,format=yuva420p10le" -ss $start -t $len -tune zerolatency -refs 1 -qp 18 -threads 1 -shortest -y $4
ffmpeg -loglevel quiet -vsync 0 -i $1 -ss $start -t $len -pix_fmt yuv420p -f rawvideo - | ~/bin/stack 2560 1920 | composite -blend 80 - /tmp/grid-labels.png $jpg
