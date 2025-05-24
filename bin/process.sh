#!/bin/bash

mkdir -p $1/events

while [ $(grep -c $1 /proc/mounts) -ne $(grep -v ^# /etc/fstab | grep -c $1) ]; do sleep 5; mount $1 2> /dev/null; done

while true; do
    /home/meteor/bin/cammon.py -x$1/max.txt -l$1/cammon.log -s$1/last.txt -b120 $1 | /home/meteor/bin/metdetect -C $1/metdetect.conf -m$1/mask.jpg -d$1/events -o$1/lens.pto -x$1/max.txt -l$1/metdetect.log -D$1 - > /dev/null
done
