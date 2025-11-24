#!/bin/bash

find /meteor/cam*/*.3gp /meteor/cam*/cifs* /tmp/tmp* -mtime +1 -exec rm -f {} \; 2> /dev/null
find /meteor/cam*/amsevents -depth -type d -mtime +180 -exec rm -r {} \; 2> /dev/null

~/bin/cleancam.py

for x in $(cat /etc/cameras.id); do
    find /meteor/cam$x -empty -delete
    sed -i -e :a -e '$q;N;100001,$D;ba' /meteor/cam$x/metdetect.log
    sed -i -e :a -e '$q;N;100001,$D;ba' /meteor/cam$x/cammon.log
done
