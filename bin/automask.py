#!/usr/bin/env python3
"""
Automate the full sky-mask pipeline: find the latest equirect timelapse
video, build the equirectangular sky mask, reverse-project it into each
camera's native view, and install the results where amscams expects them.

This is a thin orchestrator around make_equirect_mask.py and
make_camera_masks.py — see those scripts for the underlying algorithms.

Pipeline:
  1. Find the latest cam8/{YYYYMMDD}/timelapse_hires.mp4 (or timelapse.mp4
     with --sd), matching the archive layout written by stitch_latest.sh /
     timelapse_eq.sh.
  2. make_equirect_mask.build_sky_mask() on that video -> equirect mask
     (white=non-sky, i.e. --invert convention, since that is what
     make_camera_masks / scan_stack.py need).
  3. make_camera_masks.build_camera_masks() on that equirect mask -> one
     native-resolution mask per camera, installed into
     /mnt/ams2/meteor_archive/{ams_id}/CAL/MASKS/{cams_id}_mask.png
     (existing files are backed up first), unless --no-install is given.

Usage:
    automask.py [--cam-dir /meteor] [--ncams 7] [--sd]
                [--timelapse-video PATH | --date YYYYMMDD] [--lookback N]
                [--output-dir DIR] [--as6-json /home/ams/amscams/conf/as6.json]
                [--no-install] [--fill-gaps] [--max-frames N]
"""
import argparse
import glob
import os
import re
import sys
import tempfile

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_equirect_mask
import make_camera_masks


def _video_frame_count(path):
    """Cheap frame-count probe (reads the container header only)."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return 0
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def find_latest_timelapse(cam_dir, sd, archive_cam="cam8", lookback=10):
    """Find the most complete equirect timelapse video written by
    timelapse_eq.sh, under {cam_dir}/{archive_cam}/{YYYYMMDD}/{fname}.

    Looks at the most recent `lookback` dates that have the file, and
    picks the one with the most frames (a partial/interrupted archive day
    would otherwise be picked just for being newest).
    """
    archive_dir = os.path.join(cam_dir, archive_cam)
    fname = "timelapse.mp4" if sd else "timelapse_hires.mp4"
    if not os.path.isdir(archive_dir):
        raise FileNotFoundError(f"Archive directory not found: {archive_dir}")

    date_dirs = sorted(
        (d for d in os.listdir(archive_dir) if re.match(r'^\d{8}$', d)),
        reverse=True)
    if not date_dirs:
        raise FileNotFoundError(f"No date directories found under {archive_dir}")

    candidates = []
    for d in date_dirs:
        candidate = os.path.join(archive_dir, d, fname)
        if os.path.isfile(candidate):
            candidates.append(candidate)
        if len(candidates) >= lookback:
            break

    if not candidates:
        raise FileNotFoundError(
            f"No {fname} found in any date directory under {archive_dir} "
            f"(checked {len(date_dirs)} dates)")

    counts = [(c, _video_frame_count(c)) for c in candidates]
    best_path, best_count = max(counts, key=lambda x: x[1])

    for path, n in counts:
        marker = " <- selected" if path == best_path else ""
        print(f"  {path}: {n} frames{marker}", file=sys.stderr)

    if best_count <= 0:
        raise FileNotFoundError(
            f"All {len(candidates)} candidate timelapse videos under "
            f"{archive_dir} failed to open or had 0 frames.")

    return best_path


def main():
    parser = argparse.ArgumentParser(
        description="Find the latest timelapse, build the sky mask, and "
                    "install per-camera masks for amscams.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cam-dir", default="/meteor",
                        help="Base directory containing cam1..camN and the "
                             "cam8 equirect archive")
    parser.add_argument("--ncams", type=int, default=7,
                        help="Number of cameras")
    parser.add_argument("--sd", action="store_true",
                        help="Use SD timelapse/inputs (timelapse.mp4, "
                             "mini_*.jpg, 1280x848 canvas) instead of HD")
    parser.add_argument("--timelapse-video", default=None,
                        help="Override: use this timelapse video instead of "
                             "auto-discovering the latest one")
    parser.add_argument("--date", default=None, metavar="YYYYMMDD",
                        help="Use the timelapse from this specific archive "
                             "date (cam8/{YYYYMMDD}/timelapse[_hires].mp4) "
                             "instead of auto-discovering the latest one")
    parser.add_argument("--lookback", type=int, default=10,
                        help="Number of most recent timelapse videos to "
                             "consider; the one with the most frames is "
                             "used (avoids picking a newer but incomplete "
                             "archive day)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write cam{N}_mask.png and the "
                             "intermediate equirect mask (default: a "
                             "temporary directory, cleaned up unless "
                             "--keep-output is given)")
    parser.add_argument("--keep-output", action="store_true",
                        help="Do not delete --output-dir when it is an "
                             "auto-created temporary directory")
    parser.add_argument("--as6-json", default="/home/ams/amscams/conf/as6.json",
                        help="Path to as6.json, used to resolve ams_id and "
                             "each camera's cams_id for installation")
    parser.add_argument("--no-install", action="store_true",
                        help="Skip installing into the amscams CAL/MASKS "
                             "directory; only write cam{N}_mask.png files "
                             "to --output-dir")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to analyse when building the "
                             "equirect mask (0 = all)")
    parser.add_argument("--close-kernel", type=int, default=15,
                        help="Morphological close kernel size")
    parser.add_argument("--open-kernel", type=int, default=3,
                        help="Morphological open kernel size")
    parser.add_argument("--bottom-band", type=int, default=45,
                        help="Height of bottom band to mask out")
    parser.add_argument("--mean-weight", type=float, default=0.7,
                        help="Weight of mean brightness in the sky score")
    parser.add_argument("--var-weight", type=float, default=0.3,
                        help="Weight of temporal variance in the sky score")
    parser.add_argument("--fill-gaps", action="store_true", default=True,
                        help="Fill small sky-through-canopy gaps (default: "
                             "on, matches recommended hires settings)")
    parser.add_argument("--no-fill-gaps", action="store_false", dest="fill_gaps",
                        help="Disable gap filling")
    parser.add_argument("--gap-max-area", type=int, default=8000,
                        help="Max area in pixels of a gap to fill")
    args = parser.parse_args()

    # --- Step 1: find the timelapse video -----------------------------
    if args.timelapse_video and args.date:
        print("Error: --timelapse-video and --date cannot be used together.",
              file=sys.stderr)
        sys.exit(1)

    if args.timelapse_video:
        video_path = args.timelapse_video
        if not os.path.isfile(video_path):
            print(f"Error: --timelapse-video not found: {video_path}", file=sys.stderr)
            sys.exit(1)
    elif args.date:
        if not re.match(r'^\d{8}$', args.date):
            print(f"Error: --date must be in YYYYMMDD format, got: {args.date}",
                  file=sys.stderr)
            sys.exit(1)
        fname = "timelapse.mp4" if args.sd else "timelapse_hires.mp4"
        video_path = os.path.join(args.cam_dir, "cam8", args.date, fname)
        if not os.path.isfile(video_path):
            print(f"Error: no timelapse found for date {args.date}: {video_path}",
                  file=sys.stderr)
            sys.exit(1)
    else:
        print("Searching for the latest equirect timelapse video...")
        try:
            video_path = find_latest_timelapse(args.cam_dir, args.sd,
                                               lookback=args.lookback)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"Using timelapse video: {video_path}")

    # --- Output directory (temporary unless overridden) ---------------
    cleanup_output_dir = False
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = tempfile.mkdtemp(prefix="automask_")
        # Keep the temp dir around if the user asked to, or if masks weren't
        # installed anywhere else (it would otherwise be the only copy).
        cleanup_output_dir = not args.keep_output and not args.no_install
    print(f"Output directory: {output_dir}")

    equirect_mask_path = os.path.join(output_dir, "equirect_mask.png")

    try:
        # --- Step 2: build the equirect mask ---------------------------
        print("\n--- Building equirect sky mask ---")
        make_equirect_mask.build_sky_mask(
            video_path, equirect_mask_path,
            max_frames=args.max_frames,
            close_kernel=args.close_kernel,
            open_kernel=args.open_kernel,
            bottom_band=args.bottom_band,
            mean_weight=args.mean_weight,
            var_weight=args.var_weight,
            invert_output=True,  # white=non-sky, required by make_camera_masks/scan_stack.py
            fill_gaps_enabled=args.fill_gaps,
            gap_max_area=args.gap_max_area,
        )

        # --- Step 3: build (and install) per-camera masks --------------
        print("\n--- Building per-camera masks ---")
        install_ams_id, install_cams_id_map = None, None
        if not args.no_install:
            try:
                install_ams_id, install_cams_id_map = \
                    make_camera_masks.load_as6_camera_mapping(args.as6_json)
                print(f"Install target: ams_id={install_ams_id}, "
                      f"cams_id map={install_cams_id_map}")
            except (OSError, ValueError) as e:
                print(f"Error: could not resolve install target from "
                      f"{args.as6_json}: {e}", file=sys.stderr)
                sys.exit(1)

        prefix = "mini" if args.sd else "full"
        output_width, output_height = (1280, 848) if args.sd else (None, None)

        make_camera_masks.build_camera_masks(
            equirect_mask_path, output_dir, args.cam_dir, prefix,
            output_width=output_width, output_height=output_height,
            mask_white_is_sky=False, ncams=args.ncams,
            install_ams_id=install_ams_id,
            install_cams_id_map=install_cams_id_map,
        )

        if args.no_install:
            print(f"\n--install skipped (--no-install); per-camera masks "
                  f"are in {output_dir}")
        else:
            print("\nDone: per-camera masks built and installed.")
    finally:
        if cleanup_output_dir:
            import shutil
            try:
                shutil.rmtree(output_dir)
            except OSError:
                pass


if __name__ == "__main__":
    main()
