#!/bin/bash

if [ "$#" -ne "2" ]; 
    then echo "Usage: $0 <input video> <jpg/avif file>"
    exit
fi

read width height <<< $(ffmpeg -nostdin -i $1 2>&1 |egrep -o '[0-9][0-9]+x[0-9]+' | sed 's/x/ /')
ffmpeg -loglevel quiet -i $1 -t 0.2 -vsync 0 -timeout 3 -f rawvideo -pix_fmt yuv420p pipe:1 | /home/meteor/bin/stack $width $height $(dirname $2)/tmp_$(basename $2)
mv $(dirname $2)/tmp_$(basename $2) $2
