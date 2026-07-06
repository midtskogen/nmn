#!/bin/bash
# stitch_latest.sh — Cron job to stitch the latest full_MM.jpg from all 7 cameras
# into /meteor/equirect.jpg and /meteor/fisheye.jpg (atomic replacement).
#
# Typical crontab entry (every minute):
#   * * * * * /home/steinar/norskmeteornettverk.no/nmn/bin/stitch_latest.sh

set -euo pipefail

PIDFILE=/tmp/stitch_latest.pid
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE" 2>/dev/null || true)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        exit 0
    fi
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

exec >>/tmp/stitch_latest.log 2>&1
echo "--- $(date -u '+%Y-%m-%d %H:%M:%S') ---"

STITCHER=/home/meteor/nmn/bin/stitcher.py
OUTDIR=/meteor
NCAMS=7
CAMS=$(seq 1 $NCAMS)

# Try recent minutes (current-2 down to current-5) to find the latest
# minute where all 7 cameras have a fully-written full_MM.jpg.
NOW_EPOCH=$(date -u +%s)

find_latest() {
    for OFFSET in 2 3 4 5; do
        TARGET_EPOCH=$((NOW_EPOCH - OFFSET * 60))
        YYYYMMDD=$(date -u -d @$TARGET_EPOCH +%Y%m%d)
        HH=$(date -u -d @$TARGET_EPOCH +%H)
        MM=$(date -u -d @$TARGET_EPOCH +%M)

        ALL_EXIST=true
        FILES=()
        for CAM in $CAMS; do
            F="/meteor/cam${CAM}/${YYYYMMDD}/${HH}/full_${MM}.jpg"
            if [ ! -f "$F" ]; then
                ALL_EXIST=false
                break
            fi
            # Check no process has the file open (still being written)
            if fuser "$F" >/dev/null 2>&1; then
                ALL_EXIST=false
                break
            fi
            FILES+=("$F")
        done

        if [ "$ALL_EXIST" = true ]; then
            echo "${YYYYMMDD:0:4}-${YYYYMMDD:4:2}-${YYYYMMDD:6:2}_${HH}:${MM}:00 ${FILES[@]}"
            return 0
        fi
    done
    return 1
}

FOUND=$(find_latest) || { echo "No complete set of images found"; exit 1; }
read -ra PARTS <<< "$FOUND"
FOUND_TS="${PARTS[0]/_/ }"
INPUT_FILES=("${PARTS[@]:1}")
echo "Using inputs from $FOUND_TS: ${INPUT_FILES[*]}"

# Stitch equirect
TMP_EQ=$(mktemp "${OUTDIR}/equirect.XXXXXX.jpg")
if "$STITCHER" --equirect --quiet --devignette -0.20 --input-datetime "$FOUND_TS" "${INPUT_FILES[@]}" "$TMP_EQ"; then
    mv -f "$TMP_EQ" "${OUTDIR}/equirect.jpg"
    touch -d "$FOUND_TS" "${OUTDIR}/equirect.jpg"
else
    rm -f "$TMP_EQ"
    echo "Equirect stitch failed" >&2
    exit 1
fi

# Stitch fisheye
TMP_FE=$(mktemp "${OUTDIR}/fisheye.XXXXXX.jpg")
if "$STITCHER" --fisheye --quiet --devignette -0.20 --input-datetime "$FOUND_TS" "${INPUT_FILES[@]}" "$TMP_FE"; then
    mv -f "$TMP_FE" "${OUTDIR}/fisheye.jpg"
    touch -d "$FOUND_TS" "${OUTDIR}/fisheye.jpg"
else
    rm -f "$TMP_FE"
    echo "Fisheye stitch failed" >&2
    exit 1
fi

# Archive copies in cam8 (equirect) and cam9 (fisheye)
# FOUND_TS is "YYYY-MM-DD HH:MM:SS"
ARCH_YYYYMMDD="${FOUND_TS:0:4}${FOUND_TS:5:2}${FOUND_TS:8:2}"
ARCH_HH=${FOUND_TS:11:2}
ARCH_MM=${FOUND_TS:14:2}

mkdir -p "${OUTDIR}/cam8/${ARCH_YYYYMMDD}/${ARCH_HH}"
mkdir -p "${OUTDIR}/cam9/${ARCH_YYYYMMDD}/${ARCH_HH}"
cp "${OUTDIR}/equirect.jpg" "${OUTDIR}/cam8/${ARCH_YYYYMMDD}/${ARCH_HH}/full_${ARCH_MM}.jpg"
cp "${OUTDIR}/fisheye.jpg" "${OUTDIR}/cam9/${ARCH_YYYYMMDD}/${ARCH_HH}/full_${ARCH_MM}.jpg"
touch -d "$FOUND_TS" "${OUTDIR}/cam8/${ARCH_YYYYMMDD}/${ARCH_HH}/full_${ARCH_MM}.jpg"
touch -d "$FOUND_TS" "${OUTDIR}/cam9/${ARCH_YYYYMMDD}/${ARCH_HH}/full_${ARCH_MM}.jpg"
