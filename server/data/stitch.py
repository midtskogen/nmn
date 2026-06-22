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
import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

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

if Image is None:
    logging.warning("PIL not available, some image operations may be limited")

# Import multiblend from our worktree instead of using subprocess
# stitch.py is at nmn/server/data/stitch.py, multiblend.py is at nmn/bin/multiblend.py
# Use realpath to resolve symlinks and get the actual path
stitch_realpath = os.path.realpath(os.path.abspath(__file__))
multiblend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(stitch_realpath))), 'bin')
sys.path.insert(0, multiblend_path)
logging.info(f"stitch: attempting to import multiblend from {multiblend_path}")
logging.info(f"stitch: __file__={__file__}, realpath={stitch_realpath}")
try:
    import multiblend
    logging.info(f"stitch: multiblend module imported successfully, version={getattr(multiblend, '__version__', 'unknown')}")
except ImportError as e:
    multiblend = None
    logging.warning(f"stitch: multiblend module not available, falling back to subprocess: {e}")

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

# Cameras that carry a timestamp overlay in the lower-left corner.
# The overlay is made transparent (alpha=0) before nona sees the image so it
# is never blended into the panorama.
TIMESTAMP_CAMS = frozenset({6, 7})
# (x1, y1, x2, y2) — exclusive end coords, matched by image height threshold.
_TIMESTAMP_BOX_HD = (0, 1040, 305, 1080)   # height >= 900
_TIMESTAMP_BOX_SD = (0,  430, 155, 448)    # height < 900

# Absolute paths to tools (needed because www-data's PATH omits /usr/local/bin)
BIN_NONA        = "/usr/local/bin/nona"
BIN_MULTIBLEND  = "/usr/local/bin/multiblend"
BIN_PANO_MODIFY = "/usr/local/bin/pano_modify"
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
    """Return the two-line PTO header for nona/hugin.
    projection: 'fisheye' (f3) or 'equirect' (f2)
    """
    f = 3 if projection == 'fisheye' else 2
    return (f'p f{f} w{w} h{h} v360 E0 R0 n"TIFF_m c:LZW"\n'
            f'm g1 i0 f0 m2 p0.00784314\n')


def get_image_dimensions(path: str):
    """Return (width, height) of an image via PIL, or None on failure."""
    try:
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception as e:
        logging.warning(f"stitch get_image_dimensions: failed for {path}: {e}")
        return None


def _read_cal_dims(lens_path: str):
    """Return (cal_w, cal_h) from the i-line of a lens.pto, or (None, None)."""
    try:
        with open(lens_path, 'r') as f:
            for line in f:
                if line.startswith('i ') or line.startswith('i\t'):
                    m_w = re.search(r'\bw(\d+)\b', line)
                    m_h = re.search(r'\bh(\d+)\b', line)
                    if m_w and m_h:
                        return int(m_w.group(1)), int(m_h.group(1))
                    break
    except OSError:
        pass
    return None, None


def erase_timestamp(src_path: str, dst_tif: str) -> bool:
    """Write a copy of src_path as TIFF with the timestamp region set to alpha=0.

    The timestamp box in the lower-left corner is made fully transparent so
    nona ignores those pixels entirely (transparent TIFF input → no contribution
    to the blended panorama from that region).
    """
    try:
        img = Image.open(src_path)
        iw, ih = img.size
        x1, y1, x2, y2 = _TIMESTAMP_BOX_HD if ih >= 900 else _TIMESTAMP_BOX_SD
        arr = np.array(img.convert('RGBA'))
        arr[y1:y2, x1:x2, 3] = 0
        Image.fromarray(arr).save(dst_tif, compression='tiff_lzw')
        logging.info(f"stitch erase_timestamp: {os.path.basename(src_path)} "
                     f"box=({x1},{y1},{x2},{y2}) size={iw}x{ih} -> {dst_tif}")
        return True
    except Exception as e:
        logging.warning(f"stitch erase_timestamp: failed for {src_path}: {e}")
        return False


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
    """Run multiblend to merge reprojected TIFFs, filling gaps via alpha channel.

    Runs multiblend with TIFF output so we can read the alpha channel.  Any
    pixel with alpha==0 in the blended TIFF is a gap (no camera covers it);
    these are filled with the nearest covered pixel colour before saving as JPEG.
    """
    existing = [t for t in out_tifs if os.path.exists(os.path.join(workdir, t))]
    logging.info(f"stitch run_multiblend: existing tifs={existing}, out_jpg={out_jpg}")
    if not existing:
        logging.error("stitch run_multiblend: no nona output TIFFs found")
        return False

    # Output to TIFF so we can inspect the alpha channel for gap filling.
    out_tif = out_jpg + '_blend.tif'
    
    # Use direct function call if multiblend module is available
    if multiblend is not None:
        logging.info(f"stitch run_multiblend: using direct multiblend.go() call with exposure correction")
        try:
            # Convert to absolute paths for multiblend
            input_paths = [os.path.join(workdir, t) for t in existing]
            output_path = os.path.join(workdir, out_tif)
            logging.info(f"stitch run_multiblend: input_paths={input_paths}, output_path={output_path}")
            
            # Define a print function that logs to activity.log
            log_count = [0]
            def log_print(text):
                log_count[0] += 1
                if text.strip():
                    for line in text.strip().split('\n'):
                        if line:
                            logging.info(f"stitch run_multiblend: {line}")
            
            logging.info("stitch run_multiblend: calling multiblend.go()")
            multiblend.go(
                input_files=input_paths,
                output_file=output_path,
                workbpp_cmd=8,
                exposure_correct=True,
                saturation_correct=False,
                verbosity=2,
                print_func=log_print,
            )
            logging.info(f"stitch run_multiblend: multiblend.go() returned, log_print called {log_count[0]} times")
            logging.info("stitch run_multiblend: blend SUCCESS")
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            logging.error(f"stitch run_multiblend: FAILED {e}")
            return False
    else:
        # Fallback to subprocess call
        cmd = [
            BIN_MULTIBLEND,
            "--primary-seam-generator=graph-cut",
            "-d", "8",
            "-o", out_tif,
        ] + existing
        logging.info(f"stitch run_multiblend: cmd={cmd} cwd={workdir}")
        try:
            result = subprocess.run(cmd, check=True, timeout=600, capture_output=True, cwd=workdir)
            if result.stderr:
                logging.info(f"stitch run_multiblend: stderr={result.stderr.decode('utf-8', errors='ignore')!r}")
            logging.info("stitch run_multiblend: blend SUCCESS")
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

    # Convert TIFF to JPEG with gap filling via alpha channel.
    try:
        img = Image.open(out_tif)
        arr = np.array(img)
        logging.info(f"stitch run_multiblend: blend TIFF mode={img.mode} shape={arr.shape}")
        if arr.ndim == 3 and arr.shape[2] == 4:
            rgb = arr[:, :, :3].astype(np.float32)
            alpha = arr[:, :, 3]
            gap = alpha == 0
            n_gap = int(gap.sum())
            logging.info(f"stitch run_multiblend: gap pixels={n_gap} total={gap.size}")
            if n_gap > 0:
                # Fill gaps with normalized Gaussian convolution: each gap pixel
                # receives a weighted average of nearby covered pixels, producing a
                # smooth colour gradient.  Pixels far from any camera (e.g. outer
                # fisheye corners) have near-zero weight and stay black — no explicit
                # border exclusion needed.  The fisheye mask cleans up corners later.
                try:
                    from scipy.ndimage import gaussian_filter, distance_transform_edt
                    # Downsample before blurring: 64x fewer pixels → 64x faster.
                    # Smooth gradients are low-frequency so the quality loss is negligible.
                    S = 8
                    h, w = gap.shape
                    sw, sh = max(1, w // S), max(1, h // S)
                    sigma_s = max(3, min(w // 80, 80) // S)

                    coverage = (~gap).astype(np.float32)
                    masked_rgb = rgb * coverage[:, :, np.newaxis]  # gap pixels → 0

                    small_cov = np.array(
                        Image.fromarray((coverage * 255).astype(np.uint8)).resize(
                            (sw, sh), Image.BOX)).astype(np.float32) / 255.0
                    small_masked = np.array(
                        Image.fromarray(masked_rgb.clip(0, 255).astype(np.uint8)).resize(
                            (sw, sh), Image.BOX)).astype(np.float32)

                    blurred_w  = gaussian_filter(small_cov, sigma=sigma_s)
                    blurred_ch = gaussian_filter(small_masked, sigma=(sigma_s, sigma_s, 0))
                    w_b = blurred_w[:, :, np.newaxis]
                    filled_small = np.where(w_b > 1e-3, blurred_ch / np.maximum(w_b, 1e-3), 0.0)

                    fill_full = np.array(
                        Image.fromarray(filled_small.clip(0, 255).astype(np.uint8)).resize(
                            (w, h), Image.BILINEAR)).astype(np.float32)

                    # Feathered blend: compute each pixel's distance to the nearest
                    # gap pixel, then blend original → fill over feather_radius px.
                    feather_radius = 20
                    dist = distance_transform_edt(~gap)
                    blend_w = np.clip(dist / feather_radius, 0.0, 1.0)[:, :, np.newaxis]
                    filled = (rgb * blend_w + fill_full * (1 - blend_w)).clip(0, 255)
                    logging.info(f"stitch run_multiblend: gap fill done S={S} sigma_s={sigma_s} feather={feather_radius}px")
                except ImportError:
                    logging.info("stitch run_multiblend: scipy unavailable, using iterative fill")
                    interior_gap = gap.copy()
                    filled = rgb.copy()
                    remaining = interior_gap.copy()
                    for _ in range(600):
                        if not remaining.any():
                            break
                        changed = False
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            valid_nb = np.roll(~remaining, dy, axis=0)
                            valid_nb = np.roll(valid_nb, dx, axis=1)
                            can_fill = remaining & valid_nb
                            if can_fill.any():
                                src = np.roll(np.roll(filled, dy, axis=0), dx, axis=1)
                                filled[can_fill] = src[can_fill]
                                remaining[can_fill] = False
                                changed = True
                        if not changed:
                            break
            else:
                filled = rgb
        else:
            # No alpha — just use as-is (no gap fill possible)
            filled = arr[:, :, :3] if arr.ndim == 3 and arr.shape[2] >= 3 else arr
        Image.fromarray(filled.astype(np.uint8)).save(out_jpg, 'JPEG', quality=quality)
        logging.info(f"stitch run_multiblend: saved JPEG -> {out_jpg}")
        return True
    except Exception as e:
        logging.error(f"stitch run_multiblend: TIFF->JPEG conversion failed: {e}")
        # Fall back: try direct JPEG output if TIFF approach failed
        try:
            cmd2 = [
                BIN_MULTIBLEND,
                "--primary-seam-generator=graph-cut",
                "-d", "8",
                f"--compression={quality}",
                "-o", out_jpg,
            ] + existing
            subprocess.run(cmd2, check=True, timeout=600, capture_output=True, cwd=workdir)
            return True
        except Exception as e2:
            logging.error(f"stitch run_multiblend: fallback JPEG also failed: {e2}")
            return False
    finally:
        if os.path.exists(out_tif):
            os.remove(out_tif)


def apply_fisheye_mask(jpg_path: str, w: int, h: int, out_path: str) -> bool:
    """Sharpen and apply a circular black border mask to the fisheye image."""
    cx, cy = w // 2, h // 2
    r = min(cx, cy)
    try:
        # Single convert call: sharpen the source, generate the circular mask inline,
        # then multiply them together — no temp file needed.
        subprocess.run([
            BIN_CONVERT,
            jpg_path, "-sharpen", "2x2",
            "(", "-size", f"{w}x{h}", "xc:black",
                 "-fill", "white",
                 "-draw", f"circle {cx},{cy} {cx},{cy - r}",
            ")",
            "-compose", "Multiply", "-composite",
            "-quality", "90", out_path
        ], check=True, timeout=120, capture_output=True)
        logging.info(f"stitch: fisheye mask applied, out={out_path}")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = e.stderr.decode('utf-8', errors='ignore') if hasattr(e, 'stderr') and e.stderr else str(e)
        logging.error(f"stitch: fisheye mask failed: {stderr}")
        return False


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
        result = subprocess.run(
            [BIN_PANO_MODIFY, "--rotate", "0,-90,0", pto_path, "-o", pto_path],
            check=True, timeout=60, capture_output=True
        )
        # Log any stdout/stderr for debugging, but don't let it pollute JSON output
        if result.stdout:
            logging.debug(f"stitch: pano_modify stdout: {result.stdout.decode('utf-8', errors='ignore')[:200]}")
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
    long_integration: bool,
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
            stem = os.path.basename(abs_path).rsplit('.', 1)[0]
            rel_name = os.path.basename(abs_path)

            # For timestamp cameras erase overlay into a clean TIFF before any scaling.
            if cam in TIMESTAMP_CAMS:
                src = os.path.join(camdir, stem + '_clean.tif')
                if not erase_timestamp(abs_path, src):
                    src = abs_path
            else:
                src = abs_path

            # Scale to calibrated dimensions if needed, otherwise symlink.
            cal_w, cal_h = _read_cal_dims(cam_lens[cam])
            actual = get_image_dimensions(abs_path)
            if cal_w and cal_h and actual and (actual[0] != cal_w or actual[1] != cal_h):
                if cam in TIMESTAMP_CAMS:
                    out = os.path.join(camdir, stem + '_clean_scaled.tif')
                    img_rel[cam] = f"cam{cam}/{stem}_clean_scaled.tif"
                else:
                    out = os.path.join(camdir, rel_name)
                    img_rel[cam] = f"cam{cam}/{rel_name}"
                logging.info(f"stitch: cam {cam} scaling {actual[0]}x{actual[1]} -> {cal_w}x{cal_h}")
                subprocess.run([BIN_CONVERT, src, "-resize", f"{cal_w}x{cal_h}!", out],
                               check=True, timeout=60, capture_output=True)
            elif cam in TIMESTAMP_CAMS:
                img_rel[cam] = f"cam{cam}/{stem}_clean.tif"
            else:
                dst = os.path.join(camdir, rel_name)
                if not os.path.exists(dst):
                    os.symlink(abs_path, dst)
                logging.info(f"stitch: symlinked cam {cam}: {abs_path} -> {dst}")
                img_rel[cam] = f"cam{cam}/{rel_name}"

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

            logging.info("stitch: using nona+multiblend for equirect")
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

            logging.info("stitch: using nona+multiblend for fisheye")
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
    parser.add_argument('--long',          action='store_true', help='Long-integration stacked image')
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
