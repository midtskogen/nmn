#!/bin/bash

export LIBVA_DRIVER_NAME=iHD

for i in /remote/cam[1-7]; do
    src_date=$(date -u +%Y%m%d -d yesterday)
    src_dir="$i/$src_date"
    out_dir="${i/\/remote\//\/meteor\/timelapse\/}/$src_date"
    mkdir -p "$out_dir"

    tmp_dir=$(mktemp -d)
    list_file="$tmp_dir/inputs.txt"

    # Build the input list for ffmpeg's concat demuxer
    find "$src_dir" -regex '.*[0-2][0-9]/full_[0-5][0-9]\.mp4' | sort | while read -r file; do
        echo "file '$file'" >> "$list_file"
    done

    # Run ffmpeg with full GPU pipeline: VAAPI decode + I-frame filtering + VAAPI encode
    ffmpeg -hide_banner -loglevel warning -stats -y \
	   -hwaccel vaapi -hwaccel_device /dev/dri/renderD128 -hwaccel_output_format vaapi \
	   -skip_frame nokey \
	   -f concat -safe 0 -i "$list_file" \
	   -an -c:v h264_vaapi -qp 30 -profile:v high \
	   -g 250 -bf 3 -refs 4 \
	   -fps_mode passthrough \
	   "$out_dir/timelapse.264"

    MP4Box -add "$out_dir/timelapse.264#video:fps=60" -new "$out_dir/timelapse.mp4"

    rm -rf "$tmp_dir" "$out_dir/timelapse.264"
done
