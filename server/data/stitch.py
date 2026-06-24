#!/usr/bin/env python3
"""
stitch.py — Thin wrapper around stitcher.py for the Norsk Meteornettverk web interface.

This module no longer implements the stitching pipeline itself. It only:
  1. Fetches lens.pto calibration files for the requested cameras via SCP.
  2. Orders the downloaded per-camera images by camera number.
  3. Calls stitcher.stitch() to produce fisheye and/or equirect outputs.
  4. Returns the same result dict that controller.py expects.

The heavy lifting (reprojection, blending, gap filling, exposure correction,
timestamp erasure and fisheye masking) is all done by nmn/bin/stitcher.py.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import shutil

# ---------------------------------------------------------------------------
# Logging — write to the same activity.log as controller.py
# ---------------------------------------------------------------------------
BASE_DIR = os.environ.get('NMN_DATA_DIR', os.path.dirname(os.path.abspath(__file__)))
_LOG_FILE = os.path.join(BASE_DIR, 'logs', 'activity.log')
os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(_LOG_FILE),
        logging.StreamHandler(sys.stderr),
    ]
)

def _find_stitcher_path():
    """Locate the nmn/bin directory using relative paths from the project
    data directory or the script location. This works regardless of the
    absolute NFS mount point on the server."""
    candidates = []

    # 1. Relative to the canonical data directory (project/server/data -> ../../bin).
    #    BASE_DIR may be a symlink, so resolve it first.
    base_dir = os.environ.get('NMN_DATA_DIR', os.path.dirname(os.path.abspath(__file__)))
    base_dir_real = os.path.realpath(base_dir)
    candidates.append(os.path.abspath(os.path.join(base_dir_real, '..', '..', 'bin')))

    # 2. Relative to this script: server/data -> ../../bin
    this_script = os.path.abspath(__file__)
    script_dir = os.path.dirname(this_script)
    candidates.append(os.path.abspath(os.path.join(script_dir, '..', '..', 'bin')))

    # 3. Relative to the current working directory (often the project root).
    candidates.append(os.path.abspath('bin'))

    # 4. Walk up from the current working directory looking for bin/stitcher.py.
    root = os.getcwd()
    while True:
        candidates.append(os.path.abspath(os.path.join(root, 'bin')))
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent

    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, 'stitcher.py')):
            return candidate
    return None


# Import stitcher from the project bin directory.
_stitcher_path = _find_stitcher_path()
if _stitcher_path:
    if _stitcher_path not in sys.path:
        sys.path.insert(0, _stitcher_path)
    try:
        import stitcher
        logging.info(f"stitch: imported stitcher from {_stitcher_path}")
    except ImportError as e:
        stitcher = None
        logging.error(f"stitch: failed to import stitcher: {e}")
else:
    stitcher = None
    logging.error("stitch: could not locate stitcher.py relative to script or cwd")

# ---------------------------------------------------------------------------
# Fullres/hires panorama output dimensions
# ---------------------------------------------------------------------------
FISHEYE_HIRES_W  = 5120
FISHEYE_HIRES_H  = 5120
FISHEYE_LOWRES_W = 2048
FISHEYE_LOWRES_H = 2048

EQUIRECT_HIRES_W  = 5120
EQUIRECT_HIRES_H  = 3392
EQUIRECT_LOWRES_W = 1280
EQUIRECT_LOWRES_H = 848

# Remote path template for lens calibration files
REMOTE_LENS_PTO = "/meteor/cam{cam}/lens.pto"


def _stitch_projection_worker(projection, input_files, output_path, stitcher_path, kwargs):
    """Run a single stitcher projection in a child process.

    The worker re-imports stitcher in the child so it works regardless of
    whether the process was forked or spawned.
    """
    if stitcher_path not in sys.path:
        sys.path.insert(0, stitcher_path)
    import stitcher
    stitcher.stitch(input_files, output_path, projection=projection, **kwargs)
    return projection, output_path


def scp_lens_batch(station_id: str, cams: list, workdir: str) -> dict:
    """
    Fetch lens.pto for all cameras in a single SSH session using a tar pipe.
    Returns {cam_num: local_path} for each successfully fetched file.
    """
    remote_files = " ".join(REMOTE_LENS_PTO.format(cam=c) for c in cams)
    remote_cmd = f"tar -c -h --ignore-failed-read -f - {remote_files} 2>/dev/null"
    logging.info(f"stitch scp_lens_batch: fetching lens.pto for cams {cams} from {station_id}")
    tar_subdir = os.path.join(workdir, "lens_tar")
    os.makedirs(tar_subdir, exist_ok=True)
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=60", "-o", "BatchMode=yes",
             station_id, remote_cmd],
            capture_output=True, timeout=120
        )
        if result.returncode not in (0, 1) or not result.stdout:
            stderr = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ''
            logging.warning(f"stitch scp_lens_batch: ssh/tar failed rc={result.returncode} stderr={stderr!r}")
            return _scp_lens_fallback(station_id, cams, workdir)
        extract = subprocess.run(
            ["tar", "-x", "-f", "-", "-C", tar_subdir],
            input=result.stdout, capture_output=True, timeout=60
        )
        if extract.returncode != 0:
            stderr = extract.stderr.decode('utf-8', errors='ignore') if extract.stderr else ''
            logging.warning(f"stitch scp_lens_batch: tar extract failed: {stderr!r}")
            return _scp_lens_fallback(station_id, cams, workdir)
    except (subprocess.TimeoutExpired, OSError) as e:
        logging.warning(f"stitch scp_lens_batch: exception: {e}")
        return _scp_lens_fallback(station_id, cams, workdir)

    fetched = {}
    for cam in cams:
        src = os.path.join(tar_subdir, "meteor", f"cam{cam}", "lens.pto")
        dest = os.path.join(workdir, f"lens_cam{cam}.pto")
        if os.path.exists(src) and os.path.getsize(src) > 0:
            shutil.move(src, dest)
            fetched[cam] = dest
            logging.info(f"stitch scp_lens_batch: cam {cam} OK ({os.path.getsize(dest)} bytes)")
        else:
            logging.warning(f"stitch scp_lens_batch: lens.pto missing for cam {cam}")
    return fetched


def _scp_lens_fallback(station_id: str, cams: list, workdir: str) -> dict:
    """Fallback: fetch lens.pto files one at a time via scp."""
    fetched = {}
    for cam in cams:
        dest = os.path.join(workdir, f"lens_cam{cam}.pto")
        tmp = dest + ".part"
        remote = f"{station_id}:{REMOTE_LENS_PTO.format(cam=cam)}"
        try:
            subprocess.run(
                ["scp", "-B", "-o", "ConnectTimeout=60", remote, tmp],
                check=True, timeout=90, capture_output=True
            )
            shutil.move(tmp, dest)
            fetched[cam] = dest
            logging.info(f"stitch scp_lens_fallback: cam {cam} OK")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
            logging.warning(f"stitch scp_lens_fallback: cam {cam} failed: {stderr!r}")
        except (subprocess.TimeoutExpired, OSError) as e:
            logging.warning(f"stitch scp_lens_fallback: cam {cam} exception: {e}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return fetched


def stitch(
    station_id: str,
    station_code: str,
    image_paths: dict,   # {cam_num: absolute_path}
    output_dir: str,
    base_name: str,
    hires: bool,
    long_integration: bool,
    do_fisheye: bool,
    do_equirect: bool,
) -> dict:
    """
    Main entry point. Returns dict with keys 'fisheye' and/or 'equirect',
    each containing {'path': ..., 'name': ...} on success, or None on failure.
    """
    results = {}

    if not do_fisheye and not do_equirect:
        logging.info("stitch: nothing requested, returning empty")
        return results

    if stitcher is None:
        logging.error("stitch: stitcher module is not available, cannot stitch")
        return results

    logging.info(f"stitch: START station_id={station_id}, base_name={base_name}, hires={hires}, do_fisheye={do_fisheye}, do_equirect={do_equirect}")
    logging.info(f"stitch: image_paths={image_paths}")

    res_suffix = ("hires" if hires else "lowres") + ("_long" if long_integration else "")
    fw = FISHEYE_HIRES_W  if hires else FISHEYE_LOWRES_W
    fh = FISHEYE_HIRES_H  if hires else FISHEYE_LOWRES_H
    ew = EQUIRECT_HIRES_W if hires else EQUIRECT_LOWRES_W
    eh = EQUIRECT_HIRES_H if hires else EQUIRECT_LOWRES_H

    # Return cached output files immediately if they already exist.
    eq_name = f"{base_name}_{res_suffix}_equirect.jpg"
    fy_name = f"{base_name}_{res_suffix}_fisheye.jpg"
    if do_equirect and os.path.exists(os.path.join(output_dir, eq_name)):
        results['equirect'] = {'path': os.path.join(output_dir, eq_name), 'name': eq_name}
        logging.info(f"stitch: equirect cached -> {eq_name}")
        do_equirect = False
    if do_fisheye and os.path.exists(os.path.join(output_dir, fy_name)):
        results['fisheye'] = {'path': os.path.join(output_dir, fy_name), 'name': fy_name}
        logging.info(f"stitch: fisheye cached -> {fy_name}")
        do_fisheye = False
    if not do_fisheye and not do_equirect:
        return results

    logging.info(f"stitch: fisheye size={fw}x{fh}, equirect size={ew}x{eh}")

    workdir = tempfile.mkdtemp(prefix="stitch_")
    logging.info(f"stitch: workdir={workdir}")
    try:
        # --- 1. Fetch lens.pto files (all in one SSH session) ---
        cam_lens = scp_lens_batch(station_id, sorted(image_paths.keys()), workdir)
        logging.info(f"stitch: fetched lens files for cams {sorted(cam_lens.keys())}")
        if not cam_lens:
            logging.error("stitch: no lens.pto files fetched; cannot stitch")
            return results

        # Order images by camera number, matching the lens file mapping.
        ordered_cams = [
            cam for cam in sorted(image_paths.keys())
            if cam in cam_lens and os.path.exists(image_paths[cam])
        ]
        if not ordered_cams:
            logging.error("stitch: no images available for stitching after lens matching")
            return results

        input_files = [image_paths[cam] for cam in ordered_cams]
        lens_files = {cam: cam_lens[cam] for cam in ordered_cams}
        logging.info(f"stitch: ordered_cams={ordered_cams}")

        # --- 2. Equirectangular and Fisheye in parallel ---
        tasks = {}
        if do_equirect:
            final_name = f"{base_name}_{res_suffix}_equirect.jpg"
            final_path = os.path.join(output_dir, final_name)
            logging.info(f"stitch: equirect output will be {final_path}")
            tasks['equirect'] = (
                final_path,
                {
                    'lens_files': lens_files,
                    'output_width': ew, 'output_height': eh,
                    'crop_to_content': True,
                    'quiet': True,
                },
            )
        if do_fisheye:
            final_name = f"{base_name}_{res_suffix}_fisheye.jpg"
            final_path = os.path.join(output_dir, final_name)
            logging.info(f"stitch: fisheye output will be {final_path}")
            tasks['fisheye'] = (
                final_path,
                {
                    'lens_files': lens_files,
                    'output_width': fw, 'output_height': fh,
                    'fisheye_mask': True,
                    'crop_to_content': False,
                    'quiet': True,
                },
            )

        if tasks:
            max_workers = min(len(tasks), 2)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _stitch_projection_worker,
                        projection, input_files, output_path,
                        _stitcher_path, kwargs,
                    ): projection
                    for projection, (output_path, kwargs) in tasks.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    projection = futures[future]
                    try:
                        future.result()
                        final_path = tasks[projection][0]
                        final_name = os.path.basename(final_path)
                        if os.path.exists(final_path):
                            results[projection] = {'path': final_path, 'name': final_name}
                            logging.info(f"stitch: {projection} SUCCESS -> {final_path}")
                        else:
                            logging.error(f"stitch: {projection} failed: output file not created")
                    except Exception as e:
                        logging.exception(f"stitch: {projection} failed: {e}")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Stitch camera images into fisheye/equirect panoramas")
    parser.add_argument('--station-id',    required=True)
    parser.add_argument('--station-code',  required=True)
    parser.add_argument('--image-paths',   nargs='+', required=True,
                        help="Absolute paths to per-camera images")
    parser.add_argument('--cameras',       nargs='+', type=int, required=True,
                        help="Camera numbers corresponding to --image-paths")
    parser.add_argument('--output-dir',    required=True)
    parser.add_argument('--base-name',     required=True)
    parser.add_argument('--hires',         action='store_true')
    parser.add_argument('--long',          action='store_true', help='Long-integration suffix for filenames')
    parser.add_argument('--fisheye',       action='store_true')
    parser.add_argument('--equirect',      action='store_true')
    args = parser.parse_args()

    if len(args.image_paths) != len(args.cameras):
        print(json.dumps({"error": "image-paths and cameras must have the same length"}))
        sys.exit(1)

    image_paths = dict(zip(args.cameras, args.image_paths))

    results = stitch(
        station_id=args.station_id,
        station_code=args.station_code,
        image_paths=image_paths,
        output_dir=args.output_dir,
        base_name=args.base_name,
        hires=args.hires,
        long_integration=args.long,
        do_fisheye=args.fisheye,
        do_equirect=args.equirect,
    )

    print(json.dumps(results))


if __name__ == '__main__':
    main()
