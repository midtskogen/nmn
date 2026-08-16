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
        [--output-width W --output-height H] \
        [--fisheye [--fisheye-width W --fisheye-height H]]

With --fisheye, a cam9_mask.png is also assembled: each camera's own
native-resolution mask (computed en route to cam{N}_mask.png) is reprojected
into an f3/190deg-HFOV fisheye canvas using the same shared per-camera
geometry stitch_latest.sh's --fisheye stitch uses, combining overlapping
cameras conservatively (non-sky wins) instead of photometrically blending.

The equirect mask must have been generated at the SAME output resolution
stitch_latest.sh used for the mode it corresponds to:
  - HD (full_*.jpg):  omit --output-width/--output-height (defaults apply)
  - SD (--sd, mini_*.jpg): pass --output-width 1280 --output-height 848
    to match the EQ_SIZE_ARGS used in stitch_latest.sh --sd mode.
"""
import argparse
import copy
import glob
import json
import math
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


def _largest_arc_gap_center(vals, period):
    """Return the centre of the largest uncovered arc of the values' circle.

    Used to place the unwrap cut of a camera's pano-x map in the part of the
    panorama the camera does not see, so the coordinate field is continuous
    across its FOV.  Falls back to 0 (the natural seam) when degenerate.
    """
    if len(vals) == 0:
        return 0.0
    bins = 720
    covered = np.zeros(bins, dtype=bool)
    idx = np.floor(vals / period * bins).astype(int) % bins
    covered[idx] = True
    if covered.all():
        return 0.0
    # longest run of uncovered bins on the circle
    doubled = np.concatenate([covered, covered])
    best_len = best_start = 0
    run_start = None
    for i, c in enumerate(doubled):
        if not c and run_start is None:
            run_start = i
        elif c and run_start is not None:
            if i - run_start > best_len:
                best_len, best_start = i - run_start, run_start
            run_start = None
    if run_start is not None and 2 * bins - run_start > best_len:
        best_len, best_start = 2 * bins - run_start, run_start
    center_bin = (best_start + best_len / 2) % bins
    return (center_bin + 0.5) * period / bins


def _build_fisheye_mask(native_masks_white_sky, images, cam_numbers, fe_w, fe_h):
    """Assemble a fisheye-projection sky mask (white=sky) from each camera's
    native-resolution equirect-derived mask.

    This mirrors the geometry stitch_latest.sh's --fisheye stitch uses (same
    per-camera intrinsics/extrinsics from the shared PTO 'i' lines, f3/190
    degree HFOV panorama canvas), reprojecting each camera's own mask
    straight into the fisheye canvas rather than blending photos, since a
    binary sky/non-sky classification doesn't need photometric blending: a
    fisheye pixel is only "sky" if every camera that covers it agrees, and
    pixels no camera covers (e.g. the corners outside the fisheye circle)
    default to non-sky.
    """
    fe_sky = np.full((fe_h, fe_w), 255, dtype=np.uint8)
    fe_valid_any = np.zeros((fe_h, fe_w), dtype=bool)

    for idx, cam_num in enumerate(cam_numbers):
        img = images[idx]
        sw, sh = int(img['w']), int(img['h'])
        fov, src_proj_f = img.get('v'), int(img.get('f', 0))
        fov_rad = math.radians(fov)
        if src_proj_f == 0:
            src_focal = sw / (2 * math.tan(fov_rad / 2)) if fov_rad > 0 else 0
        elif src_proj_f == 3:
            src_focal = sw / fov_rad if fov_rad > 0 else 0
        else:
            continue

        src_norm_radius = min(sw, sh) / 2.
        y, p, r = img.get('y', 0), img.get('p', 0), -img.get('r', 0)
        a, b, c = img.get('a', 0), img.get('b', 0), img.get('c', 0)
        cx, cy = -img.get('d', 0), img.get('e', 0)
        R_pr_inv = pto_mapper.create_pr_rotation_matrix(p, r).T

        coords = np.empty((fe_h, fe_w, 2), dtype=np.float32)
        pto_mapper.calculate_source_coords(
            coords, fe_w, fe_h, fe_w, fe_h, 0, 0, 3, 190.0,
            sw, sh, R_pr_inv, y, src_focal, src_norm_radius,
            a, b, c, cx, cy, src_proj_f, 0.0, 1.0)

        # calculate_source_coords() only flags a coordinate invalid when the
        # projection itself fails (e.g. behind the camera); its lens model
        # is valid well beyond the physical sensor extent (it's normally
        # used with explicit padding for blend overlap), so it will happily
        # return e.g. x=-300 or x=1200 for an 800-wide sensor. Without also
        # bounding to the actual [0, sw) x [0, sh) sensor rectangle here,
        # those out-of-frame samples fall through cv2.remap's border
        # handling as "non-sky" and incorrectly veto real sky pixels that
        # this camera doesn't actually cover.
        raw_x, raw_y = coords[:, :, 0], coords[:, :, 1]
        valid = (raw_x > -99999.0) & (raw_x >= 0) & (raw_x < sw) & (raw_y >= 0) & (raw_y < sh)

        # cv2.remap's cubic interpolation blends in a small neighbourhood of
        # source pixels, so right at a camera's own frame edge it mixes in
        # the BORDER_CONSTANT (non-sky) fill used for pixels just outside
        # that camera's coverage. Composited into the fisheye canvas, that
        # shows up as a thin curved artefact tracing each camera's own image
        # boundary. Erode the valid footprint a few pixels inward (in
        # fisheye-canvas space, so it follows the actual projected/curved
        # boundary) so only samples safely inside each camera's frame -- away
        # from that edge-interpolation noise -- are trusted; overlapping
        # camera coverage fills in the rest.
        valid_u8 = valid.astype(np.uint8)
        erode_kernel = np.ones((5, 5), np.uint8)
        valid = cv2.erode(valid_u8, erode_kernel, iterations=1) > 0

        # Scale from the camera's native PTO resolution to the canonical
        # MASK_OUT_W x MASK_OUT_H size the per-camera masks are stored at.
        map_x = np.where(valid, raw_x * (MASK_OUT_W / sw), -1e6).astype(np.float32)
        map_y = np.where(valid, raw_y * (MASK_OUT_H / sh), -1e6).astype(np.float32)

        cam_contrib = cv2.remap(native_masks_white_sky[cam_num], map_x, map_y,
                                interpolation=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _, cam_contrib = cv2.threshold(cam_contrib, 127, 255, cv2.THRESH_BINARY)

        fe_sky = np.where(valid, np.minimum(fe_sky, cam_contrib), fe_sky)
        fe_valid_any |= valid

    fe_sky[~fe_valid_any] = 0

    # Final cleanup: remove any remaining thin sky-coloured slivers (residual
    # per-camera edge/interpolation noise the footprint erosion above didn't
    # fully catch) with a morphological opening. Only opening -- never
    # closing -- is used here: opening can only turn thin "sky" noise into
    # "non-sky", the same safe direction as everything else in this
    # pipeline, whereas closing would risk painting over genuinely thin
    # foreground (antennas, wires) with sky.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fe_sky = cv2.morphologyEx(fe_sky, cv2.MORPH_OPEN, open_kernel)

    return fe_sky


def build_camera_masks(equirect_mask_path, output_dir, cam_dir, prefix,
                       output_width=None, output_height=None,
                       mask_white_is_sky=True, ncams=7,
                       install_ams_id=None, install_cams_id_map=None,
                       build_fisheye=False, fisheye_width=None, fisheye_height=None):
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

    # Create the install directory (including .../CAL and .../CAL/MASKS) once,
    # up front, so a missing archive tree is created before any install.
    install_masks_dir = None
    if install_ams_id and install_cams_id_map:
        install_masks_dir = f"/mnt/ams2/meteor_archive/{install_ams_id}/CAL/MASKS"
        if not os.path.isdir(install_masks_dir):
            print(f"Creating install directory: {install_masks_dir}")
        try:
            os.makedirs(install_masks_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"Could not create install directory {install_masks_dir}: {e}")

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
    native_masks_white_sky = {}
    for idx, cam_num in enumerate(cam_numbers):
        coords = pto_mapper.build_image_to_pano_map(pto_data, idx)
        map_x = coords[:, :, 0]
        map_y = coords[:, :, 1]

        invalid = map_x <= -99998
        # Park invalid coordinates far out of bounds so that, after the
        # coordinate-map upsampling below, their blend halo still resolves
        # to the non-sky border value instead of sampling garbage.
        if is_full_360:
            map_x = np.mod(map_x, orig_w)
            # Unwrap the azimuth discontinuity BEFORE upsampling: a camera
            # whose FOV straddles the pano seam has map_x values jumping
            # between ~orig_w and ~0, and interpolating across that jump
            # would sample unrelated azimuths (a vertical artefact stripe).
            # Place the cut in the largest uncovered arc of this camera's
            # map so the field is continuous across the FOV.
            cut = _largest_arc_gap_center(map_x[~invalid], orig_w)
            map_x = np.where(map_x < cut, map_x + orig_w, map_x)
        map_x = np.where(invalid, -1e6, map_x)
        map_y = np.where(invalid, -1e6, map_y)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        # scan_stack.py's load_mask_imgs resizes every mask to 1920x1080
        # before use; store them at that size directly.  Rather than
        # nearest-neighbour-upscaling the binary mask (stair-stepped edges),
        # upsample the *coordinate maps* -- a smooth field, where bilinear
        # interpolation is sub-pixel accurate -- and then resample the
        # equirect mask with cubic interpolation.  The final threshold
        # re-binarises, leaving a smooth, anti-aliased boundary that follows
        # the projection geometry instead of either pixel grid.
        sw, sh = int(images[idx]['w']), int(images[idx]['h'])
        if (sw, sh) != (MASK_OUT_W, MASK_OUT_H):
            map_x = cv2.resize(map_x, (MASK_OUT_W, MASK_OUT_H),
                               interpolation=cv2.INTER_LINEAR)
            map_y = cv2.resize(map_y, (MASK_OUT_W, MASK_OUT_H),
                               interpolation=cv2.INTER_LINEAR)
            invalid = cv2.resize(invalid.astype(np.uint8), (MASK_OUT_W, MASK_OUT_H),
                                 interpolation=cv2.INTER_NEAREST) > 0

        if is_full_360:
            # Wrap AFTER upsampling, then shift into the padded image's
            # coordinate space.
            map_x = np.where(map_x <= -9e5, map_x,
                             np.mod(map_x, orig_w) + wrap_pad)

        cam_mask = cv2.remap(pano_mask_for_remap, map_x, map_y,
                             interpolation=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        _, cam_mask = cv2.threshold(cam_mask, 127, 255, cv2.THRESH_BINARY)
        cam_mask[invalid] = 0

        if build_fisheye:
            native_masks_white_sky[cam_num] = cam_mask.copy()

        if not mask_white_is_sky:
            cam_mask = cv2.bitwise_not(cam_mask)

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
            install_path = os.path.join(install_masks_dir, f"{cams_id}_mask.png")
            if os.path.exists(install_path):
                backup_path = install_path + f".bak-{time.strftime('%Y%m%d-%H%M%S')}"
                shutil.copy2(install_path, backup_path)
                print(f"  Backed up existing {install_path} -> {backup_path}")
            cv2.imwrite(install_path, install_mask)
            print(f"  Installed -> {install_path}")

            # Also expose the installed mask where the capture-side tooling
            # looks for it, e.g. /meteor/cam{N}/mask.png. Only create the
            # symlink if nothing is there yet (file or symlink, even a
            # broken one) so we never clobber a manually placed mask.
            link_path = os.path.join(cam_dir, f"cam{cam_num}", "mask.png")
            if not os.path.lexists(link_path):
                try:
                    os.symlink(install_path, link_path)
                    print(f"  Symlinked {link_path} -> {install_path}")
                except OSError as e:
                    print(f"  WARNING: could not symlink {link_path} -> "
                          f"{install_path}: {e}")

    if build_fisheye:
        fe_w = fisheye_width or 4096
        fe_h = fisheye_height or 4096
        # generate_pto_from_lens_files() applies this same (yaw=0, pitch=-90,
        # roll=0) global rotation to every camera pose when it builds a real
        # 'fisheye' projection PTO for stitch_latest.sh's --fisheye stitch,
        # so that the fisheye canvas centre lands on zenith instead of the
        # cameras' horizon-forward axis. `images` here came from the
        # *equirect* PTO and doesn't have that rotation, so apply it to a
        # copy before reprojecting into the fisheye canvas.
        fisheye_images = copy.deepcopy(images)
        fisheye_pto_data = (dict(global_options), fisheye_images)
        pto_mapper.rotate_panorama(fisheye_pto_data, yaw_deg=0, pitch_deg=-90, roll_deg=0)
        fe_mask = _build_fisheye_mask(native_masks_white_sky, fisheye_images, cam_numbers, fe_w, fe_h)
        if not mask_white_is_sky:
            fe_mask = cv2.bitwise_not(fe_mask)
        fe_out_path = os.path.join(output_dir, "cam9_mask.png")
        cv2.imwrite(fe_out_path, fe_mask)
        out_paths[9] = fe_out_path
        non_sky_pct = 100 * np.count_nonzero(fe_mask == 255) / fe_mask.size if not mask_white_is_sky \
            else 100 * np.count_nonzero(fe_mask == 0) / fe_mask.size
        print(f"  cam9 (fisheye): {fe_w}x{fe_h} -> {fe_out_path} (non-sky: {non_sky_pct:.1f}%)")

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
                             "Any pre-existing file is backed up first. Also symlinks "
                             "{cam-dir}/cam{N}/mask.png to the installed file, if that "
                             "path doesn't already exist.")
    parser.add_argument("--as6-json", default="/home/ams/amscams/conf/as6.json",
                        help="Path to as6.json, used with --install to resolve ams_id and "
                             "each camera's cams_id (default /home/ams/amscams/conf/as6.json)")
    parser.add_argument("--fisheye", action="store_true",
                        help="Also assemble a cam9_mask.png fisheye-projection mask "
                             "(f3, 190deg HFOV) in output_dir, by reprojecting each "
                             "camera's own native mask into the fisheye canvas the same "
                             "way stitch_latest.sh's --fisheye stitch does, the same "
                             "shared per-camera geometry.")
    parser.add_argument("--fisheye-width", type=int, default=None,
                        help="Fisheye canvas width (default 4096, or 2048 to match "
                             "stitch_latest.sh --sd)")
    parser.add_argument("--fisheye-height", type=int, default=None,
                        help="Fisheye canvas height (default 4096, or 2048 to match "
                             "stitch_latest.sh --sd)")
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
        build_fisheye=args.fisheye, fisheye_width=args.fisheye_width,
        fisheye_height=args.fisheye_height,
    )


if __name__ == "__main__":
    main()
