#!/usr/bin/env python3
"""
stitch.py — Panoramic image stitcher for Norsk Meteornettverk.

Given a list of per-camera JPEG images (already downloaded to DOWNLOAD_DIR)
and the station's lens.pto calibration files (fetched via SCP), produces one
or both of:
  • Fisheye  — 8192×8192 (hires) / 2048×2048 (lowres), circular crop applied
  • Equirectangular — 3380×2240 (hires) / 1280×848 (lowres)

The script is called from download_for_single_station() after the individual
per-camera images have been downloaded.

Usage (internal, called by controller.py):
    python3 stitch.py \
        --station-id ams173 \
        --station-code AMS173 \
        --image-paths /path/img_cam1.jpg /path/img_cam2.jpg ... \
        --cameras 1 2 3 4 5 6 7 \
        --output-dir /path/download/ \
        --base-name ams173_20250101_2200 \
        --hires \
        --fisheye \
        --equirect
"""

import argparse
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

# ---------------------------------------------------------------------------
# Constants matching stitch.sh
# ---------------------------------------------------------------------------
FISHEYE_HIRES_W  = 8192
FISHEYE_HIRES_H  = 8192
FISHEYE_LOWRES_W = 2048
FISHEYE_LOWRES_H = 2048

EQUIRECT_HIRES_W  = 3380
EQUIRECT_HIRES_H  = 2240
EQUIRECT_LOWRES_W = 1280
EQUIRECT_LOWRES_H = 848

# Remote path template for lens calibration files
REMOTE_LENS_PTO = "/meteor/cam{cam}/lens.pto"

# Absolute paths to tools (needed because www-data's PATH omits /usr/local/bin)
BIN_NONA        = "/usr/local/bin/nona"
BIN_MULTIBLEND  = "/usr/local/bin/multiblend"
BIN_PANO_MODIFY = "/usr/local/bin/pano_modify"
BIN_IDENTIFY    = "/usr/bin/identify"
BIN_CONVERT     = "/usr/bin/convert"


def scp_lens_batch(station_id: str, cams: list, workdir: str) -> dict:
    """
    Fetch lens.pto for all cameras in a single SSH session using a tar pipe.
    Returns {cam_num: local_path} for each successfully fetched file.
    """
    # -h: dereference symlinks so we get the file content, not the symlink itself
    # --ignore-failed-read: skip cams that have no lens.pto without aborting
    remote_files = " ".join(REMOTE_LENS_PTO.format(cam=c) for c in cams)
    remote_cmd = f"tar -c -h --ignore-failed-read -f - {remote_files} 2>/dev/null"
    logging.info(f"stitch scp_lens_batch: fetching lens.pto for cams {cams} from {station_id}")
    tar_subdir = os.path.join(workdir, "lens_tar")
    os.makedirs(tar_subdir, exist_ok=True)
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=60", "-o", "BatchMode=yes",
             station_id, remote_cmd],
            capture_output=True, timeout=90
        )
        if result.returncode not in (0, 1) or not result.stdout:
            logging.warning(f"stitch scp_lens_batch: ssh/tar failed rc={result.returncode} "
                            f"stderr={result.stderr.decode('utf-8', errors='ignore')!r}")
            return _scp_lens_fallback(station_id, cams, workdir)
        # Extract preserving full path structure (meteor/camN/lens.pto)
        extract = subprocess.run(
            ["tar", "-x", "-f", "-", "-C", tar_subdir],
            input=result.stdout, capture_output=True, timeout=30
        )
        if extract.returncode != 0:
            logging.warning(f"stitch scp_lens_batch: tar extract failed: "
                            f"{extract.stderr.decode('utf-8', errors='ignore')!r}")
            return _scp_lens_fallback(station_id, cams, workdir)
    except (subprocess.TimeoutExpired, OSError) as e:
        logging.warning(f"stitch scp_lens_batch: exception: {e}")
        return _scp_lens_fallback(station_id, cams, workdir)

    # Move extracted files to expected names: lens_tar/meteor/camN/lens.pto -> workdir/lens_camN.pto
    fetched = {}
    for cam in cams:
        # Remote path is /meteor/camN/lens.pto; tar strips leading / so it's meteor/camN/lens.pto
        src = os.path.join(tar_subdir, "meteor", f"cam{cam}", "lens.pto")
        dest = os.path.join(workdir, f"lens_cam{cam}.pto")
        if os.path.exists(src) and os.path.getsize(src) > 0:
            os.rename(src, dest)
            fetched[cam] = dest
            logging.info(f"stitch scp_lens_batch: cam {cam} OK ({os.path.getsize(dest)} bytes)")
        else:
            logging.warning(f"stitch scp_lens_batch: cam {cam} lens.pto not found on remote")
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
            os.rename(tmp, dest)
            fetched[cam] = dest
            logging.info(f"stitch scp_lens_fallback: cam {cam} OK")
        except subprocess.CalledProcessError as e:
            logging.warning(f"stitch scp_lens_fallback: cam {cam} failed: "
                            f"{e.stderr.decode('utf-8', errors='ignore')!r}")
        except (subprocess.TimeoutExpired, OSError) as e:
            logging.warning(f"stitch scp_lens_fallback: cam {cam} exception: {e}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return fetched


def build_pto_header(w: int, h: int, projection: str) -> str:
    """
    Return the two-line PTO header for nona/hugin.
    projection: 'fisheye' (f3) or 'equirect' (f2)
    """
    if projection == 'fisheye':
        # f3 = equi-solid-angle (fisheye); full canvas, apply_fisheye_mask does the circular crop
        p_line = f'p f3 w{w} h{h} v360 E0 R0 n"TIFF_m c:LZW"'
    else:
        # f2 = equirectangular; S crops to the useful strip used in stitch.sh
        p_line = f'p f2 w{w} h{h} v360 E0 R0 n"TIFF_m c:LZW"'
    m_line = 'm g1 i0 f0 m2 p0.00784314'
    return p_line + "\n" + m_line + "\n"


def get_image_dimensions(path: str):
    """Return (width, height) of an image using identify, or None on failure."""
    try:
        result = subprocess.run(
            [BIN_IDENTIFY, "-format", "%w %h", path],
            capture_output=True, text=True, timeout=30, check=True
        )
        w, h = result.stdout.strip().split()
        return int(w), int(h)
    except Exception as e:
        logging.warning(f"stitch get_image_dimensions: failed for {path}: {e}")
        return None


def build_pto(header: str, cam_lens_files: dict, image_paths: dict) -> str:
    """
    Build a complete PTO file string.
    cam_lens_files:  {cam_num: path_to_lens.pto}
    image_paths:     {cam_num: relative_path_for_pto}
    """
    lines = [header]
    for cam, lens_path in sorted(cam_lens_files.items()):
        if cam not in image_paths:
            continue
        try:
            with open(lens_path, 'r') as f:
                for line in f:
                    if line.startswith('i ') or line.startswith('i\t'):
                        stripped = line.rstrip()
                        # Remove existing n"..." token
                        stripped = re.sub(r'\s+n"[^"]*"', '', stripped)
                        img_rel = image_paths[cam]
                        lines.append(f'{stripped} n"{img_rel}"\n')
                        break
        except OSError as e:
            logging.warning(f"stitch: could not read lens.pto for cam {cam}: {e}")
    return "".join(lines)


def run_nona(pto_path: str, out_prefix: str, images: list, workdir: str) -> bool:
    """Run nona to reproject images."""
    cmd = [
        BIN_NONA,
        "-z", "DEFLATE",
        "-o", out_prefix,
        "-m", "TIFF_m",
        pto_path,
    ] + images
    logging.info(f"stitch run_nona: cmd={cmd} cwd={workdir}")
    try:
        result = subprocess.run(cmd, check=True, timeout=600, capture_output=True, cwd=workdir)
        if result.stderr:
            logging.info(f"stitch run_nona: stderr={result.stderr.decode('utf-8', errors='ignore')!r}")
        logging.info(f"stitch run_nona: SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
        logging.error(f"stitch run_nona: FAILED returncode={e.returncode} stderr={stderr!r}")
        return False
    except subprocess.TimeoutExpired:
        logging.error("stitch run_nona: TIMED OUT")
        return False
    except FileNotFoundError:
        logging.error("stitch run_nona: 'nona' not found in PATH")
        return False


def run_multiblend(out_tifs: list, out_jpg: str, quality: int, workdir: str) -> bool:
    """Run multiblend to merge reprojected TIFFs."""
    existing = [t for t in out_tifs if os.path.exists(os.path.join(workdir, t))]
    logging.info(f"stitch run_multiblend: existing tifs={existing}, out_jpg={out_jpg}")
    if not existing:
        logging.error("stitch run_multiblend: no nona output TIFFs found")
        return False
    cmd = [
        BIN_MULTIBLEND,
        "--primary-seam-generator=graph-cut",
        "-d", "8",
        f"--compression={quality}",
        "-o", out_jpg,
    ] + existing
    logging.info(f"stitch run_multiblend: cmd={cmd} cwd={workdir}")
    try:
        result = subprocess.run(cmd, check=True, timeout=600, capture_output=True, cwd=workdir)
        if result.stderr:
            logging.info(f"stitch run_multiblend: stderr={result.stderr.decode('utf-8', errors='ignore')!r}")
        logging.info("stitch run_multiblend: SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
        stdout = e.stdout.decode('utf-8', errors='ignore') if e.stdout else ''
        logging.error(f"stitch run_multiblend: FAILED returncode={e.returncode} stderr={stderr!r} stdout={stdout!r}")
        return False
    except subprocess.TimeoutExpired:
        logging.error("stitch run_multiblend: TIMED OUT")
        return False
    except FileNotFoundError:
        logging.error("stitch run_multiblend: 'multiblend' not found in PATH")
        return False


def apply_fisheye_mask(jpg_path: str, w: int, h: int, out_path: str) -> bool:
    """Sharpen and apply a circular black border mask to the fisheye image."""
    cx, cy = w // 2, h // 2
    r = min(cx, cy)  # full radius touching edges
    mask_path = jpg_path + "_mask.png"
    try:
        # Create a white circle on black background as mask
        subprocess.run([
            BIN_CONVERT,
            "-size", f"{w}x{h}", "xc:black",
            "-fill", "white",
            "-draw", f"circle {cx},{cy} {cx},{cy - r}",
            mask_path
        ], check=True, timeout=120, capture_output=True)

        # Sharpen source, then multiply with mask to black out corners
        subprocess.run([
            BIN_CONVERT,
            "-sharpen", "2x2",
            jpg_path, mask_path,
            "-compose", "Multiply", "-composite",
            "-quality", "90",
            out_path
        ], check=True, timeout=120, capture_output=True)
        logging.info(f"stitch: fisheye mask applied, out={out_path}")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = e.stderr.decode('utf-8', errors='ignore') if hasattr(e, 'stderr') and e.stderr else str(e)
        logging.error(f"stitch: fisheye mask failed: {stderr}")
        return False
    finally:
        if os.path.exists(mask_path):
            os.remove(mask_path)


def sharpen_jpg(in_path: str, out_path: str) -> bool:
    try:
        subprocess.run(
            [BIN_CONVERT, "-sharpen", "2x2", "-quality", "80", in_path, out_path],
            check=True, timeout=120, capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logging.error(f"stitch: convert sharpen failed: {e}")
        return False


def rotate_fisheye_pto(pto_path: str) -> bool:
    """Apply the -90° pitch rotation that stitch.sh does for the fisheye."""
    try:
        subprocess.run(
            [BIN_PANO_MODIFY, "--rotate", "0,-90,0", pto_path, "-o", pto_path],
            check=True, timeout=60, capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        logging.warning(f"stitch: pano_modify not available or failed: {e}")
        return False


def stitch(
    station_id: str,
    station_code: str,
    image_paths: dict,   # {cam_num: absolute_path}
    output_dir: str,
    base_name: str,
    hires: bool,
    do_fisheye: bool,
    do_equirect: bool,
) -> dict:
    """
    Main entry point.  Returns dict with keys 'fisheye' and/or 'equirect',
    each containing {'path': ..., 'name': ...} on success, or None on failure.
    """
    results = {}
    if not do_fisheye and not do_equirect:
        logging.info("stitch: nothing requested, returning empty")
        return results

    logging.info(f"stitch: START station_id={station_id}, base_name={base_name}, hires={hires}, do_fisheye={do_fisheye}, do_equirect={do_equirect}")
    logging.info(f"stitch: image_paths={image_paths}")

    res_suffix = "hires" if hires else "lowres"
    fw = FISHEYE_HIRES_W  if hires else FISHEYE_LOWRES_W
    fh = FISHEYE_HIRES_H  if hires else FISHEYE_LOWRES_H
    ew = EQUIRECT_HIRES_W if hires else EQUIRECT_LOWRES_W
    eh = EQUIRECT_HIRES_H if hires else EQUIRECT_LOWRES_H
    logging.info(f"stitch: fisheye size={fw}x{fh}, equirect size={ew}x{eh}")

    # Work in a temp dir so nona's out0000.tif etc. don't pollute download/
    workdir = tempfile.mkdtemp(prefix="stitch_")
    logging.info(f"stitch: workdir={workdir}")
    try:
        # --- 1. Fetch lens.pto files (all in one SSH session) ---
        cam_lens = scp_lens_batch(station_id, sorted(image_paths.keys()), workdir)

        logging.info(f"stitch: fetched lens files for cams {sorted(cam_lens.keys())}")
        if not cam_lens:
            logging.error("stitch: no lens.pto files fetched; cannot stitch")
            return results

        # Place images into workdir, scaling up to calibrated dimensions if needed.
        # Upscaling preserves the exact projection geometry from lens.pto.
        img_rel = {}
        for cam, abs_path in image_paths.items():
            if cam not in cam_lens:
                logging.warning(f"stitch: cam {cam} has image but no lens.pto, skipping")
                continue
            if not os.path.exists(abs_path):
                logging.warning(f"stitch: cam {cam} image not found on disk: {abs_path}")
                continue
            camdir = os.path.join(workdir, f"cam{cam}")
            os.makedirs(camdir, exist_ok=True)
            rel_name = os.path.basename(abs_path)
            dst = os.path.join(camdir, rel_name)
            if not os.path.exists(dst):
                # Read calibrated dimensions from lens.pto i-line
                cal_w, cal_h = None, None
                try:
                    with open(cam_lens[cam], 'r') as lf:
                        for line in lf:
                            if line.startswith('i ') or line.startswith('i\t'):
                                m_w = re.search(r'\bw(\d+)\b', line)
                                m_h = re.search(r'\bh(\d+)\b', line)
                                if m_w and m_h:
                                    cal_w, cal_h = int(m_w.group(1)), int(m_h.group(1))
                                break
                except OSError:
                    pass
                actual = get_image_dimensions(abs_path)
                if cal_w and cal_h and actual and (actual[0] != cal_w or actual[1] != cal_h):
                    logging.info(f"stitch: cam {cam} scaling {actual[0]}x{actual[1]} -> {cal_w}x{cal_h}")
                    subprocess.run(
                        [BIN_CONVERT, abs_path, "-resize", f"{cal_w}x{cal_h}!", dst],
                        check=True, timeout=60, capture_output=True
                    )
                else:
                    os.symlink(abs_path, dst)
                    logging.info(f"stitch: symlinked cam {cam}: {abs_path} -> {dst}")
            img_rel[cam] = os.path.join(f"cam{cam}", rel_name)
        logging.info(f"stitch: img_rel={img_rel}")

        ordered_imgs = [img_rel[c] for c in sorted(img_rel.keys())]
        # nona tif outputs: out0000.tif … out000N.tif
        out_tifs = [f"out{str(i).zfill(4)}.tif" for i in range(len(ordered_imgs))]
        logging.info(f"stitch: ordered_imgs={ordered_imgs}")
        if not ordered_imgs:
            logging.error("stitch: no images available for stitching after lens matching")
            return results

        # --- 2. Equirectangular ---
        if do_equirect:
            logging.info(f"stitch: building equirect.pto ({ew}x{eh})")
            eq_pto_path = os.path.join(workdir, "equirect.pto")
            header = build_pto_header(ew, eh, 'equirect')
            pto_content = build_pto(header, cam_lens, img_rel)
            logging.info(f"stitch: equirect.pto content:\n{pto_content}")
            with open(eq_pto_path, 'w') as f:
                f.write(pto_content)

            raw_jpg = os.path.join(workdir, "equirect_raw.jpg")
            final_name = f"{base_name}_{res_suffix}_equirect.jpg"
            final_path = os.path.join(output_dir, final_name)
            logging.info(f"stitch: equirect output will be {final_path}")

            if run_nona(eq_pto_path, os.path.join(workdir, "out"), ordered_imgs, workdir):
                existing_tifs = [t for t in out_tifs if os.path.exists(os.path.join(workdir, t))]
                logging.info(f"stitch: equirect nona produced tifs: {existing_tifs}")
                if run_multiblend(existing_tifs, raw_jpg, 80, workdir):
                    logging.info(f"stitch: equirect multiblend OK, raw_jpg exists={os.path.exists(raw_jpg)}")
                    if sharpen_jpg(raw_jpg, final_path):
                        results['equirect'] = {'path': final_path, 'name': final_name}
                        logging.info(f"stitch: equirect SUCCESS -> {final_path}")
                    else:
                        logging.error("stitch: equirect sharpen failed")
                else:
                    logging.error("stitch: equirect multiblend failed")
            else:
                logging.error("stitch: equirect nona failed")

            # Clean up tifs for next pass
            for t in out_tifs:
                tp = os.path.join(workdir, t)
                if os.path.exists(tp):
                    os.remove(tp)
            if os.path.exists(raw_jpg):
                os.remove(raw_jpg)

        # --- 3. Fisheye ---
        if do_fisheye:
            logging.info(f"stitch: building fisheye.pto ({fw}x{fh})")
            fy_pto_path = os.path.join(workdir, "fisheye.pto")
            header = build_pto_header(fw, fh, 'fisheye')
            pto_content = build_pto(header, cam_lens, img_rel)
            logging.info(f"stitch: fisheye.pto content:\n{pto_content}")
            with open(fy_pto_path, 'w') as f:
                f.write(pto_content)

            rotate_fisheye_pto(fy_pto_path)

            raw_jpg   = os.path.join(workdir, "fisheye_raw.jpg")
            final_name = f"{base_name}_{res_suffix}_fisheye.jpg"
            final_path = os.path.join(output_dir, final_name)
            logging.info(f"stitch: fisheye output will be {final_path}")

            if run_nona(fy_pto_path, os.path.join(workdir, "out"), ordered_imgs, workdir):
                existing_tifs = [t for t in out_tifs if os.path.exists(os.path.join(workdir, t))]
                logging.info(f"stitch: fisheye nona produced tifs: {existing_tifs}")
                if run_multiblend(existing_tifs, raw_jpg, 90, workdir):
                    logging.info(f"stitch: fisheye multiblend OK, raw_jpg exists={os.path.exists(raw_jpg)}")
                    actual_dims = get_image_dimensions(raw_jpg)
                    mask_w, mask_h = actual_dims if actual_dims else (fw, fh)
                    logging.info(f"stitch: fisheye raw_jpg actual dims={mask_w}x{mask_h}")
                    masked_path = raw_jpg + "_masked.jpg"
                    if apply_fisheye_mask(raw_jpg, mask_w, mask_h, masked_path):
                        shutil.move(masked_path, final_path)
                        results['fisheye'] = {'path': final_path, 'name': final_name}
                        logging.info(f"stitch: fisheye SUCCESS -> {final_path}")
                    else:
                        logging.warning("stitch: fisheye mask failed, falling back to sharpened un-masked")
                        # fall back: at least save the un-masked version
                        if sharpen_jpg(raw_jpg, final_path):
                            results['fisheye'] = {'path': final_path, 'name': final_name}
                            logging.info(f"stitch: fisheye fallback SUCCESS -> {final_path}")
                else:
                    logging.error("stitch: fisheye multiblend failed")
            else:
                logging.error("stitch: fisheye nona failed")

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
        do_fisheye=args.fisheye,
        do_equirect=args.equirect,
    )

    print(json.dumps(results))


if __name__ == '__main__':
    main()
