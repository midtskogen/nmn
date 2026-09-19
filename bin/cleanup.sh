#!/bin/bash

# Pin the cleanup root: resolve /meteor (it may legitimately be a symlink
# to mounted storage) and require it to be a real directory before we run
# any recursive find/delete under it.
meteor_root=$(realpath -e -- /meteor 2>/dev/null) || exit 1
[ -d "$meteor_root" ] || exit 1

find "$meteor_root"/cam*/amsevents -depth -type d -mtime +180 -exec rm -r {} \; 2> /dev/null

~/bin/cleancam.py

for x in $(cat /etc/cameras.id); do
    # Only accept plain safe camera ids — the id list feeds find paths below.
    [[ "$x" =~ ^[0-9]+$ ]] || continue
    camdir=$meteor_root/cam$x
    [ -d "$camdir" ] || continue
    touch "$camdir/metdetect.log" "$camdir/cammon.log"
    sed -i -e :a -e '$q;N;100001,$D;ba' "$camdir/metdetect.log"
    sed -i -e :a -e '$q;N;100001,$D;ba' "$camdir/cammon.log"
    find "$camdir/" -empty -delete
done
