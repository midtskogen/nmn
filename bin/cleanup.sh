#!/bin/bash

find /meteor/cam*/amsevents -depth -type d -mtime +180 -exec rm -r {} \; 2> /dev/null

~/bin/cleancam.py

for x in $(cat /etc/cameras.id); do
    touch /meteor/cam$x/metdetect.log /meteor/cam$x/cammon.log
    sed -i -e :a -e '$q;N;100001,$D;ba' /meteor/cam$x/metdetect.log
    sed -i -e :a -e '$q;N;100001,$D;ba' /meteor/cam$x/cammon.log
    find /meteor/cam$x/ -empty -delete
done
