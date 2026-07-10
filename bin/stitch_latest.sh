#!/bin/bash
# stitch_latest.sh — Cron job to stitch the latest full_MM.jpg from all 7 cameras
# into /meteor/equirect.jpg and /meteor/fisheye.jpg (atomic replacement).
#
# Usage:
#   stitch_latest.sh [--sd] [--ssh HOST] [-v|--verbose]
#
# Options:
#   --sd         Use mini_MM.jpg instead of full_MM.jpg
#   --ssh HOST   Fetch inputs from and upload results to HOST via SSH
#   -v, --verbose Print extra progress information and omit --quiet from stitcher.py

set -euo pipefail

# Parse options
PREFIX=full
OUT_SUFFIX=_hd
SSH_HOST=""
EQ_SIZE_ARGS=()
FE_SIZE_ARGS=()
VERBOSE=false
while [ $# -gt 0 ]; do
    case "$1" in
        --sd)
            PREFIX=mini; OUT_SUFFIX=""
            EQ_SIZE_ARGS=(--output-width 1280 --output-height 848)
            FE_SIZE_ARGS=(--output-width 2048 --output-height 2048)
            shift ;;
        --ssh)   SSH_HOST="$2"; shift 2 ;;
        -v|--verbose) VERBOSE=true; shift ;;
        *)       echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

vlog() { if [ "$VERBOSE" = true ]; then echo "$@"; fi }

if [ "$VERBOSE" = true ]; then
    STITCHER_QUIET=()
else
    STITCHER_QUIET=(--quiet)
fi

PIDFILE=/tmp/stitch_latest${OUT_SUFFIX}.pid
if [ -f "$PIDFILE" ]; then
    OLD_PID=$(cat "$PIDFILE" 2>/dev/null || true)
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        exit 0
    fi
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

exec >>/tmp/stitch_latest${OUT_SUFFIX}.log 2>&1
echo "--- $(date -u '+%Y-%m-%d %H:%M:%S') ---"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
STITCHER="${SCRIPT_DIR}/stitcher.py"
OUTDIR=/meteor
NCAMS=7

# --- SSH ControlMaster setup ---
SSH_OPTS=()
CTRL_SOCK=""
if [ -n "$SSH_HOST" ]; then
    CTRL_SOCK=$(mktemp -u /tmp/stitch_ssh_XXXXXX)
    ssh -fNM -S "$CTRL_SOCK" -o ConnectTimeout=10 "$SSH_HOST"
    SSH_OPTS=(-o "ControlPath=$CTRL_SOCK")
    cleanup_ssh() { ssh -S "$CTRL_SOCK" -O exit "$SSH_HOST" 2>/dev/null || true; rm -f "$PIDFILE"; }
    trap cleanup_ssh EXIT
    echo "SSH ControlMaster to $SSH_HOST established"
fi

rcmd() {
    # Run a command locally or remotely
    if [ -n "$SSH_HOST" ]; then
        ssh "${SSH_OPTS[@]}" "$SSH_HOST" "$@"
    else
        eval "$@"
    fi
}

# --- Find the latest complete set of images ---
# One remote/local command checks all offsets, all cameras, and fuser.
# Output: YYYY-MM-DD_HH:MM:SS file1 file2 ... file7
FIND_SCRIPT=$(cat <<'FINDEOF'
NOW_EPOCH=$(date -u +%s)
for OFFSET in 2 3 4 5; do
    TARGET_EPOCH=$((NOW_EPOCH - OFFSET * 60))
    YYYYMMDD=$(date -u -d @$TARGET_EPOCH +%Y%m%d)
    HH=$(date -u -d @$TARGET_EPOCH +%H)
    MM=$(date -u -d @$TARGET_EPOCH +%M)
    ALL=true
    FILES=""
    for CAM in $(seq 1 __NCAMS__); do
        F="/meteor/cam${CAM}/${YYYYMMDD}/${HH}/XPREFX_${MM}.jpg"
        if [ ! -f "$F" ] || fuser "$F" >/dev/null 2>&1; then
            ALL=false
            break
        fi
        FILES="$FILES $F"
    done
    if [ "$ALL" = true ]; then
        echo "${YYYYMMDD:0:4}-${YYYYMMDD:4:2}-${YYYYMMDD:6:2}_${HH}:${MM}:00${FILES}"
        exit 0
    fi
done
exit 1
FINDEOF
)
FIND_SCRIPT=${FIND_SCRIPT//__NCAMS__/$NCAMS}
FIND_SCRIPT=${FIND_SCRIPT//XPREFX/$PREFIX}

vlog "Searching for the latest complete set of ${PREFIX}_MM.jpg images..."
FOUND=$(rcmd "$FIND_SCRIPT") || { echo "No complete set of images found"; exit 1; }
read -ra PARTS <<< "$FOUND"
FOUND_TS="${PARTS[0]/_/ }"
REMOTE_FILES=("${PARTS[@]:1}")
echo "Using inputs from $FOUND_TS: ${REMOTE_FILES[*]}"
vlog "Found ${#REMOTE_FILES[@]} input files"

# --- Fetch input files ---
if [ -n "$SSH_HOST" ]; then
    vlog "Fetching inputs from $SSH_HOST via SSH"
    TMPDIR=$(mktemp -d /tmp/stitch_inputs_XXXXXX)
    trap 'rm -rf "$TMPDIR" "$LOCAL_OUTDIR"; cleanup_ssh' EXIT
    # Mirror remote directory structure so stitcher can find lens.pto files.
    # Collect unique cam directories and their lens.pto files.
    INPUT_FILES=()
    LENS_SRCS=()
    for F in "${REMOTE_FILES[@]}"; do
        LOCAL="${TMPDIR}${F}"
        mkdir -p "$(dirname "$LOCAL")"
        INPUT_FILES+=("$LOCAL")
        # lens.pto is two dirs up from the image (e.g. /meteor/cam1/lens.pto)
        CAM_DIR=$(dirname "$(dirname "$(dirname "$F")")")
        LENS_SRCS+=("${CAM_DIR}/lens.pto")
    done
    # Deduplicate lens.pto paths
    UNIQUE_LENS=($(printf '%s\n' "${LENS_SRCS[@]}" | sort -u))
    # Download lens.pto files
    vlog "Downloading ${#UNIQUE_LENS[@]} lens.pto calibration files"
    for LP in "${UNIQUE_LENS[@]}"; do
        LOCAL_LP="${TMPDIR}${LP}"
        mkdir -p "$(dirname "$LOCAL_LP")"
        scp -o "ControlPath=$CTRL_SOCK" "$SSH_HOST:$LP" "$LOCAL_LP"
    done
    # Download image files
    vlog "Downloading ${#REMOTE_FILES[@]} image files"
    for i in "${!REMOTE_FILES[@]}"; do
        scp -o "ControlPath=$CTRL_SOCK" "$SSH_HOST:${REMOTE_FILES[$i]}" "${INPUT_FILES[$i]}"
    done
    vlog "Inputs ready in $TMPDIR"
else
    vlog "Using local input files"
    INPUT_FILES=("${REMOTE_FILES[@]}")
fi

# --- Stitch equirect ---
LOCAL_OUTDIR="${OUTDIR}"
[ -n "$SSH_HOST" ] && LOCAL_OUTDIR=$(mktemp -d /tmp/stitch_out_XXXXXX)

TMP_EQ=$(mktemp "${LOCAL_OUTDIR}/equirect.XXXXXX.jpg")
vlog "Stitching equirect: $TMP_EQ"
if "$STITCHER" --equirect "${STITCHER_QUIET[@]}" --devignette -0.20 "${EQ_SIZE_ARGS[@]}" --input-datetime "$FOUND_TS" "${INPUT_FILES[@]}" "$TMP_EQ"; then
    mv -f "$TMP_EQ" "${LOCAL_OUTDIR}/equirect${OUT_SUFFIX}.jpg"
else
    rm -f "$TMP_EQ"
    echo "Equirect stitch failed" >&2
    exit 1
fi

# --- Stitch fisheye ---
TMP_FE=$(mktemp "${LOCAL_OUTDIR}/fisheye.XXXXXX.jpg")
vlog "Stitching fisheye: $TMP_FE"
if "$STITCHER" --fisheye "${STITCHER_QUIET[@]}" --devignette -0.20 "${FE_SIZE_ARGS[@]}" --input-datetime "$FOUND_TS" "${INPUT_FILES[@]}" "$TMP_FE"; then
    mv -f "$TMP_FE" "${LOCAL_OUTDIR}/fisheye${OUT_SUFFIX}.jpg"
else
    rm -f "$TMP_FE"
    echo "Fisheye stitch failed" >&2
    exit 1
fi

# --- Archive paths ---
ARCH_YYYYMMDD="${FOUND_TS:0:4}${FOUND_TS:5:2}${FOUND_TS:8:2}"
ARCH_HH=${FOUND_TS:11:2}
ARCH_MM=${FOUND_TS:14:2}
ARCH8="cam8/${ARCH_YYYYMMDD}/${ARCH_HH}/${PREFIX}_${ARCH_MM}.jpg"
ARCH9="cam9/${ARCH_YYYYMMDD}/${ARCH_HH}/${PREFIX}_${ARCH_MM}.jpg"
vlog "Archiving results to ${ARCH8} and ${ARCH9}"

if [ -n "$SSH_HOST" ]; then
    # Upload equirect.jpg and fisheye.jpg as .tmp, then atomically move them
    # and create archive copies — all in one SSH round-trip.
    vlog "Uploading results to $SSH_HOST"
    scp -o "ControlPath=$CTRL_SOCK" "${LOCAL_OUTDIR}/equirect${OUT_SUFFIX}.jpg" "$SSH_HOST:${OUTDIR}/equirect${OUT_SUFFIX}.jpg.tmp"
    scp -o "ControlPath=$CTRL_SOCK" "${LOCAL_OUTDIR}/fisheye${OUT_SUFFIX}.jpg"  "$SSH_HOST:${OUTDIR}/fisheye${OUT_SUFFIX}.jpg.tmp"

    ssh "${SSH_OPTS[@]}" "$SSH_HOST" bash -s <<REMEOF
        set -euo pipefail
        mv -f "${OUTDIR}/equirect${OUT_SUFFIX}.jpg.tmp" "${OUTDIR}/equirect${OUT_SUFFIX}.jpg"
        mv -f "${OUTDIR}/fisheye${OUT_SUFFIX}.jpg.tmp" "${OUTDIR}/fisheye${OUT_SUFFIX}.jpg"
        touch -d "$FOUND_TS" "${OUTDIR}/equirect${OUT_SUFFIX}.jpg" "${OUTDIR}/fisheye${OUT_SUFFIX}.jpg"
        mkdir -p "${OUTDIR}/cam8/${ARCH_YYYYMMDD}/${ARCH_HH}" "${OUTDIR}/cam9/${ARCH_YYYYMMDD}/${ARCH_HH}"
        cp "${OUTDIR}/equirect${OUT_SUFFIX}.jpg" "${OUTDIR}/${ARCH8}"
        cp "${OUTDIR}/fisheye${OUT_SUFFIX}.jpg" "${OUTDIR}/${ARCH9}"
        touch -d "$FOUND_TS" "${OUTDIR}/${ARCH8}" "${OUTDIR}/${ARCH9}"
REMEOF
    rm -rf "$LOCAL_OUTDIR"
else
    touch -d "$FOUND_TS" "${OUTDIR}/equirect${OUT_SUFFIX}.jpg" "${OUTDIR}/fisheye${OUT_SUFFIX}.jpg"
    mkdir -p "${OUTDIR}/cam8/${ARCH_YYYYMMDD}/${ARCH_HH}" "${OUTDIR}/cam9/${ARCH_YYYYMMDD}/${ARCH_HH}"
    cp "${OUTDIR}/equirect${OUT_SUFFIX}.jpg" "${OUTDIR}/${ARCH8}"
    cp "${OUTDIR}/fisheye${OUT_SUFFIX}.jpg" "${OUTDIR}/${ARCH9}"
    touch -d "$FOUND_TS" "${OUTDIR}/${ARCH8}" "${OUTDIR}/${ARCH9}"
fi
