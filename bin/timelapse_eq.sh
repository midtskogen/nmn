#!/bin/bash
DIR=/meteor/cam8/$(date -u +%Y%m%d -d yesterday)
find "$DIR" -name 'mini_*.jpg' | sort | awk '{print "file \047" $0 "\047"}' > /tmp/frames_eq.txt
ffmpeg -f concat -safe 0 -r 30 -i /tmp/frames_eq.txt -crf 22 -c:v libx264 -pix_fmt yuv420p -y "$DIR/timelapse_new.mp4" \
  && mv "$DIR/timelapse_new.mp4" "$DIR/timelapse.mp4"
rm -f /tmp/frames_eq.txt
