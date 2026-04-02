#!/bin/bash

for i in /remote/cam[1-7]; do
    src_date=$(date -u +%Y%m%d -d yesterday)
    src_dir="$i/$src_date"

    # Replace "/remote/" with "/meteor/timelapse/"
    out_dir="${i/\/remote\//\/meteor\/timelapse\/}/$src_date"
    mkdir -p "$out_dir"

    # Generate comma-separated list of image files
    file_list=$(find "$src_dir" -regex '.*[0-2][0-9]/full_[0-5][0-9]\.jpg' | sort | tr '\n' ',' | sed 's/,$//')

    # Only proceed if file_list is non-empty
    if [ -n "$file_list" ]; then
        mencoder -mf fps=30:type=jpg mf://$file_list \
            -vf eq2=1.67,pp=hb/vb/dr/tn:0:256:512 \
            -of rawvideo -ovc x264 -x264encopts crf=26 -o "$out_dir/timelapse.264"

        MP4Box "$out_dir/timelapse.mp4" -add "$out_dir/timelapse.264" -fps 30 2> /dev/null
        rm -f "$out_dir/timelapse.264"
    fi
done
