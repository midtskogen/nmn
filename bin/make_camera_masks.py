#!/usr/bin/env python3
"""
Map an equirectangular sky mask back to each of the individual camera views
that stitch_latest.sh combined to create it, producing one mask PNG per
camera suitable for scan_stack.py.  Masks are always written at 1920x1080
(the size scan_stack.py's load_mask_imgs resizes to); SD mini-derived
masks are upscaled from 800x448 with nearest-neighbour interpolation.

This reuses stitcher.generate_pto_from_lens_files() to build the exact same
combined PTO project that stitch_latest.sh implicitly builds when stitching
--equirect (same lens.pto files, same output canvas sizing/scaling logic),
then uses a vectorized camera-pixel -> panorama-pixel mapping
(pto_mapper.build_image_to_pano_map) to resample the equirect mask into each
camera's native resolution via cv2.remap().

Usage:
    make_camera_masks.py <equirect_mask.png> <output_dir> \
        [--cam-dir /meteor] [--ncams 7] [--sd] [--invert] \
        [--output-width W --output-height H]

The equirect mask must have been generated at the SAME output resolution
stitch_latest.sh used for the mode it corresponds to:
  - HD (full_*.jpg):  omit --output-width/--output-height (defaults apply)
  - SD (--sd, mini_*.jpg): pass --output-width 1280 --output-height 848
    to match the EQ_SIZE_ARGS used in stitch_latest.sh --sd mode.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pto_mapper
import stitcher

# All per-camera masks are written at the amscams-native size; scan_stack.py
# (load_mask_imgs) resizes masks to this before applying them.
MASK_OUT_W, MASK_OUT_H = 1920, 1080


def load_as6_camera_mapping(as6_json_path):
    """Read as6.json and return (ams_id, {cam_num: cams_id}).

    This is the same site/camera identification used by the real amscams
    pipeline (scan_stack.py's load_mask_imgs()) to find
    /mnt/ams2/meteor_archive/{ams_id}/CAL/MASKS/{cams_id}_mask.png.
    """
    with open(as6_json_path) as f:
        as6 = json.load(f)
    ams_id = as6.get('site', {}).get('ams_id')
    if not ams_id:
        raise ValueError(f"No 'site.ams_id' found in {as6_json_path}")

    cams_id_map = {}
    for cam_key, cam_info in as6.get('cameras', {}).items():
        cams_id = cam_info.get('cams_id')
        if not cams_id:
            continue
        m = None
        match = re.search(r'(\d+)$', cam_key)
        if match:
            m = int(match.group(1))
        if m is not None:
            cams_id_map[m] = cams_id
    return ams_id, cams_id_map


def find_camera_inputs(cam_dir, prefix, ncams=7):
    """Find one lens.pto and one sample <prefix>_*.jpg image per camera.

    Returns (lens_files, sample_images), both dicts keyed by camera number.
    """
    lens_files = {}
    sample_images = {}
    for cam_num in range(1, ncams + 1):
        cam_path = os.path.join(cam_dir, f"cam{cam_num}")
        lens_pto = os.path.join(cam_path, "lens.pto")
        if not os.path.exists(lens_pto):
            raise FileNotFoundError(f"lens.pto not found for cam{cam_num}: {lens_pto}")
        lens_files[cam_num] = lens_pto

        matches = sorted(glob.glob(os.path.join(cam_path, "**", f"{prefix}_*.jpg"),
                                   recursive=True))
        if not matches:
            raise FileNotFoundError(
                f"No {prefix}_*.jpg sample image found for cam{cam_num} under {cam_path}")
        sample_images[cam_num] = matches[-1]
    return lens_files, sample_images


def build_camera_masks(equirect_mask_path, output_dir, cam_dir, prefix,
                       output_width=None, output_height=None,
                       mask_white_is_sky=True, ncams=7,
                       install_ams_id=None, install_cams_id_map=None):
    """Build one native-resolution mask PNG per camera from an equirect mask."""
    pano_mask = cv2.imread(str(equirect_mask_path), cv2.IMREAD_GRAYSCALE)
    if pano_mask is None:
        raise RuntimeError(f"Could not read equirect mask: {equirect_mask_path}")

    # Work internally with white=sky so that out-of-bounds camera pixels can
    # be safely defaulted to "non-sky" (border_value=0) regardless of the
    # input mask's own convention. Convert back to the original convention
    # before saving each camera mask.
    if not mask_white_is_sky:
        pano_mask = cv2.bitwise_not(pano_mask)

    lens_files, sample_images = find_camera_inputs(cam_dir, prefix, ncams)
    cam_numbers = sorted(sample_images.keys())
    input_files = [sample_images[cam_num] for cam_num in cam_numbers]

    pto_path = stitcher.generate_pto_from_lens_files(
        input_files, 'equirect', lens_files=lens_files,
        w=output_width, h=output_height)
    if pto_path is None:
        raise RuntimeError("Failed to generate combined PTO from lens.pto files")

    try:
        pto_data = pto_mapper.parse_pto_file(pto_path)
    finally:
        try:
            os.unlink(pto_path)
        except OSError:
            pass

    global_options, images = pto_data
    orig_w, orig_h = global_options['w'], global_options['h']
    pano_hfov = float(global_options.get('v', 360))
    is_full_360 = int(global_options.get('f', 2)) == 2 and pano_hfov >= 359.99
    print(f"Panorama canvas: {orig_w}x{orig_h} (mask file is {pano_mask.shape[1]}x{pano_mask.shape[0]})")

    os.makedirs(output_dir, exist_ok=True)

    # An equirect panorama with a full 360-degree horizontal FOV wraps at its
    # left/right edges (pano_x=0 and pano_x=orig_w represent the same yaw).
    # cv2.remap has no per-axis wrap mode, and coordinates that round to
    # exactly orig_w (e.g. 4095.8 -> nearest index 4096) are treated as
    # out-of-bounds border pixels rather than wrapping to column 0. Pad the
    # mask horizontally with a copy of its wrapped-around edge columns so
    # every coordinate near the seam resolves to the correct pixel.
    wrap_pad = 4
    if is_full_360:
        pano_mask_for_remap = cv2.copyMakeBorder(
            pano_mask, 0, 0, wrap_pad, wrap_pad, cv2.BORDER_WRAP)
    else:
        pano_mask_for_remap = pano_mask

    out_paths = {}
    for idx, cam_num in enumerate(cam_numbers):
        coords = pto_mapper.build_image_to_pano_map(pto_data, idx)
        map_x = coords[:, :, 0]
        map_y = coords[:, :, 1]

        invalid = map_x <= -99998
        if is_full_360:
            # Bring into [0, orig_w) first, then shift into the padded
            # image's coordinate space.
            map_x = np.mod(map_x, orig_w) + wrap_pad
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        cam_mask = cv2.remap(pano_mask_for_remap, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cam_mask[invalid] = 0

        if not mask_white_is_sky:
            cam_mask = cv2.bitwise_not(cam_mask)

        # scan_stack.py's load_mask_imgs resizes every mask to 1920x1080
        # before use; store them at that size directly (nearest-neighbour
        # keeps the binary mask binary).  SD mini-derived masks (800x448)
        # share the same aspect ratio, so this upscale is distortion-free.
        sw, sh = int(images[idx]['w']), int(images[idx]['h'])
        if (sw, sh) != (MASK_OUT_W, MASK_OUT_H):
            cam_mask = cv2.resize(cam_mask, (MASK_OUT_W, MASK_OUT_H),
                                  interpolation=cv2.INTER_NEAREST)

        out_path = os.path.join(output_dir, f"cam{cam_num}_mask.png")
        cv2.imwrite(out_path, cam_mask)
        out_paths[cam_num] = out_path

        non_sky_pct = 100 * np.count_nonzero(cam_mask == 255) / cam_mask.size if not mask_white_is_sky \
            else 100 * np.count_nonzero(cam_mask == 0) / cam_mask.size
        print(f"  cam{cam_num}: {sw}x{sh} -> {MASK_OUT_W}x{MASK_OUT_H} -> {out_path} "
              f"(non-sky: {non_sky_pct:.1f}%)")

        if install_ams_id and install_cams_id_map:
            cams_id = install_cams_id_map.get(cam_num)
            if not cams_id:
                print(f"  WARNING: no cams_id for cam{cam_num} in as6.json, skipping install")
                continue
            # amscams (scan_stack.py) expects white=non-sky (the region to
            # subtract/exclude), regardless of the --invert flag used for
            # the plain output_dir copy above.
            install_mask = cam_mask if not mask_white_is_sky else cv2.bitwise_not(cam_mask)
            masks_dir = f"/mnt/ams2/meteor_archive/{install_ams_id}/CAL/MASKS"
            os.makedirs(masks_dir, exist_ok=True)
            install_path = os.path.join(masks_dir, f"{cams_id}_mask.png")
            if os.path.exists(install_path):
                backup_path = install_path + f".bak-{time.strftime('%Y%m%d-%H%M%S')}"
                shutil.copy2(install_path, backup_path)
                print(f"  Backed up existing {install_path} -> {backup_path}")
            cv2.imwrite(install_path, install_mask)
            print(f"  Installed -> {install_path}")

    return out_paths


def main():
    parser = argparse.ArgumentParser(
        description="Map an equirectangular sky mask to per-camera mask files.")
    parser.add_argument("equirect_mask", help="Path to the equirectangular mask PNG")
    parser.add_argument("output_dir", help="Directory to write cam{N}_mask.png files")
    parser.add_argument("--cam-dir", default="/meteor",
                        help="Base directory containing cam1..camN subfolders (default /meteor)")
    parser.add_argument("--ncams", type=int, default=7,
                        help="Number of cameras (default 7)")
    parser.add_argument("--sd", action="store_true",
                        help="Use mini_*.jpg (SD) instead of full_*.jpg (HD) sample images")
    parser.add_argument("--output-width", type=int, default=None,
                        help="Panorama canvas width used when the mask was generated "
                             "(must match stitch_latest.sh's mode: omit for HD, "
                             "1280 for --sd)")
    parser.add_argument("--output-height", type=int, default=None,
                        help="Panorama canvas height used when the mask was generated "
                             "(omit for HD, 848 for --sd)")
    parser.add_argument("--invert", action="store_true",
                        help="The input mask uses white=non-sky (scan_stack.py convention) "
                             "instead of the default white=sky")
    parser.add_argument("--install", action="store_true",
                        help="Also write each mask directly into the location the real "
                             "amscams pipeline loads from: "
                             "/mnt/ams2/meteor_archive/{ams_id}/CAL/MASKS/{cams_id}_mask.png "
                             "(always written white=non-sky, regardless of --invert). "
                             "Any pre-existing file is backed up first.")
    parser.add_argument("--as6-json", default="/home/ams/amscams/conf/as6.json",
                        help="Path to as6.json, used with --install to resolve ams_id and "
                             "each camera's cams_id (default /home/ams/amscams/conf/as6.json)")
    args = parser.parse_args()

    prefix = "mini" if args.sd else "full"

    install_ams_id, install_cams_id_map = None, None
    if args.install:
        install_ams_id, install_cams_id_map = load_as6_camera_mapping(args.as6_json)
        print(f"Install target: ams_id={install_ams_id}, "
              f"cams_id map={install_cams_id_map}")

    build_camera_masks(
        args.equirect_mask, args.output_dir, args.cam_dir, prefix,
        output_width=args.output_width, output_height=args.output_height,
        mask_white_is_sky=not args.invert, ncams=args.ncams,
        install_ams_id=install_ams_id, install_cams_id_map=install_cams_id_map,
    )


if __name__ == "__main__":
    main()
