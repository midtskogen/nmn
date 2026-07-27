#!/bin/bash
DIR=/meteor/cam8/$(date -u +%Y%m%d -d yesterday)
find "$DIR" -name 'mini_*.jpg' | sort | awk '{print "file \047" $0 "\047"}' > /tmp/frames_eq.txt
ffmpeg -f concat -safe 0 -r 30 -i /tmp/frames_eq.txt -crf 22 -c:v libx264 -preset veryfast -pix_fmt yuv420p -y "$DIR/timelapse_new.mp4" \
  && mv "$DIR/timelapse_new.mp4" "$DIR/timelapse.mp4"
rm -f /tmp/frames_eq.txt

find "$DIR" -name 'full_*.jpg' | sort | awk '{print "file \047" $0 "\047"}' > /tmp/frames_eq_hires.txt
if [ -s /tmp/frames_eq_hires.txt ]; then
    ffmpeg -f concat -safe 0 -r 30 -i /tmp/frames_eq_hires.txt -crf 25 -c:v libx264 -preset veryfast -pix_fmt yuv420p -y "$DIR/timelapse_hires_new.mp4" \
      && mv "$DIR/timelapse_hires_new.mp4" "$DIR/timelapse_hires.mp4"
fi
rm -f /tmp/frames_eq_hires.txt
