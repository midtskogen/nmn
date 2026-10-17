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

# Emergency pressure valve for stations running on a small filesystem
# (e.g. data disk lost, captures landing on rootfs): when the filesystem
# holding /mnt/ams2/HD is >=95% full, delete the oldest raw HD clips
# until it drops below 90%.  No-op on normal stations where doDay.py's
# day-based retention keeps usage well under this.  Hardlinked copies in
# /meteor keep the data even when the HD link is removed.
hd_dir=/mnt/ams2/HD
if [ -d "$hd_dir" ]; then
    for _ in $(seq 1 30); do
        usage=$(df --output=pcent "$hd_dir" 2>/dev/null | tail -1 | tr -dc '0-9')
        [ "${usage:-0}" -ge 95 ] || break
        mapfile -t oldest < <(find "$hd_dir" -maxdepth 1 -name '*.mp4' -type f \
            -printf '%T@ %p\n' 2>/dev/null | sort -n | head -200 | cut -d' ' -f2-)
        [ ${#oldest[@]} -eq 0 ] && break
        printf '%s\n' "${oldest[@]}" | xargs -r rm -f --
        echo "cleanup.sh: ${usage}% full, deleted ${#oldest[@]} oldest HD clips"
        usage=$(df --output=pcent "$hd_dir" 2>/dev/null | tail -1 | tr -dc '0-9')
        [ "${usage:-0}" -lt 90 ] && break
    done
fi
