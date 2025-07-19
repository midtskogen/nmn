#!/bin/bash

# --- Default values ---
DATE_ARG="yesterday"
DIRS_ARG=()

# --- Argument Parsing ---
# Parse command-line options
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --date)
      DATE_ARG="$2"
      shift # past argument
      shift # past value
      ;;
    --dirs)
      shift # past --dirs argument
      # Consume all subsequent arguments until a new option is found
      while [[ $# -gt 0 ]] && ! [[ "$1" =~ ^-- ]]; do
        DIRS_ARG+=("$1")
        shift # past value
      done
      ;;
    *) # unknown option
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# --- Set default directories if --dirs was not used ---
if [ ${#DIRS_ARG[@]} -eq 0 ]; then
    # Default to all cam directories in /meteor/ and /remote/
    DIRS_ARG=(/meteor/cam* /remote/cam*)
fi

export LIBVA_DRIVER_NAME=iHD

# --- Main Processing Loop ---
src_date=$(date -u +%Y%m%d -d "$DATE_ARG")

for base_dir in "${DIRS_ARG[@]}"; do
    # Set a trap for this loop iteration that will clean up temp files on exit
    # This command will execute on EXIT, INT (Ctrl+C), or TERM signals.
    trap 'rm -rf "$tmp_dir" "$out_dir/timelapse.264"; trap - EXIT INT TERM' EXIT INT TERM

    src_dir="$base_dir/$src_date"

    # Ignore non-existent source directories without warnings
    if [ ! -d "$src_dir" ]; then
        continue
    fi

    # Derive a robust output directory path
    cam_name=$(basename "$base_dir")
    out_dir="/meteor/timelapse/$cam_name/$src_date"
    mkdir -p "$out_dir"

    tmp_dir=$(mktemp -d)
    list_file="$tmp_dir/inputs.txt"

    # Build the input list for ffmpeg's concat demuxer
    find "$src_dir" -regex '.*[0-2][0-9]/full_[0-5][0-9]\.mp4' | sort | while read -r file; do
        echo "file '$file'" >> "$list_file"
    done

    # Continue to next directory if no files were found
    if [ ! -s "$list_file" ]; then
        echo "No video files found in $src_dir. Skipping."
        rm -rf "$tmp_dir" # Still clean up the empty temp dir
        continue
    fi

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

    echo "✅ Successfully created timelapse for $src_dir"
done
