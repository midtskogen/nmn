#!/bin/bash

for i in /meteor/cam[0-9]; do
    cd $i/$(date -u +%Y%m%d -d yesterday)
    mencoder -mf fps=30:type=jpg 'mf://[0-2][0-9][0-5][0-9][0-5][0-9].jpg' -vf eq2=1.67,pp=hb/vb/dr/tn:0:256:512 -of rawvideo -ovc x264 -x264encopts crf=26 -o timelapse.264 2> /dev/null > /dev/null && (rm -f timelapse.mp4; MP4Box timelapse.mp4 -add timelapse.264 -fps 30 2> /dev/null; rm -f timelapse.264 [0-2][0-9][0-5][0-9][0-5][0-9].jpg)
done
