#!/usr/bin/env python3
"""
draw_camera_boundaries.py — Draw camera FOV boundary edges onto a stitched
panorama overlay (equirectangular or fisheye).

Each camera's sensor rectangle edge (or crop rectangle if S is set) is traced in
camera-pixel space and projected forward into panorama pixel space using
pto_mapper.map_image_to_pano().  The result is a transparent PNG with one
coloured polyline per camera, suitable for overlaying on grid_eq_hd.png etc.

Usage:
    draw_camera_boundaries.py --pano grid_eq_hd.pto \
        --lens lens_cam1.pto lens_cam2.pto ... \
        --output cam_bounds_eq.png [--width W] [--height H] [--samples N]

The panorama PTO sets the output canvas dimensions and projection.
Each lens PTO provides one camera's calibration (i-line).
"""

import argparse
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate project modules (pto_mapper, wand) regardless of working directory
# ---------------------------------------------------------------------------
_SCRIPT_PATH = Path(__file__).resolve()
_PROJECT_DIR = None
for _cand in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
    if (_cand / 'bin').is_dir() and (_cand / 'server').is_dir():
        _PROJECT_DIR = _cand
        break
if _PROJECT_DIR is not None:
    for _p in (_PROJECT_DIR / 'bin', _PROJECT_DIR / 'src', _PROJECT_DIR):
        _ps = str(_p)
        if _ps not in sys.path:
            sys.path.insert(0, _ps)

import numpy as np
import pto_mapper
import wand.image
import wand.drawing
import wand.color

# ---------------------------------------------------------------------------
# Colours assigned per camera (cycle if more than len(COLOURS) cameras)
# ---------------------------------------------------------------------------
COLOURS = [
    '#FF4444',  # red
    '#44FF44',  # green
    '#4488FF',  # blue
    '#FFaa00',  # orange
    '#FF44FF',  # magenta
    '#00FFFF',  # cyan
    '#FFFF44',  # yellow
    '#FF8888',  # light red
]

STROKE_COLOURS = [
    '#CC0000',  # dark red
    '#008800',  # dark green
    '#0044CC',  # dark blue
    '#CC6600',  # dark orange
    '#CC00CC',  # dark magenta
    '#008888',  # dark cyan
    '#888800',  # dark yellow/olive
    '#CC4444',  # dark light-red
]

STROKE_WIDTH = 5.0
STROKE_OPACITY = 1.0
FILL_OPACITY = 0.20


def _parse_lens_pto(path: str) -> dict:
    """Return the i-line parameter dict from a single-image lens PTO."""
    _, images = pto_mapper.parse_pto_file(path)
    if not images:
        raise ValueError(f"No i-line found in {path}")
    return images[0]


def _trace_camera_edge(img: dict, n_samples: int) -> list:
    """
    Return a list of (cam_x, cam_y) points tracing the camera's sensor rectangle
    edge in camera-pixel coordinates.

    The boundary is always the rectangular sensor edge (or the S crop box if
    present).  We never use the inscribed circle even for fisheye cameras,
    because the sensor is rectangular and that is the true camera boundary.
    n_samples points are distributed per side so curves project smoothly.
    """
    w = img.get('w', 1920)
    h = img.get('h', 1080)

    S = img.get('S')  # optional crop: (x0, y0, x1, y1) in image pixels
    if S is not None:
        x0, y0, x1, y1 = float(S[0]), float(S[1]), float(S[2]), float(S[3])
    else:
        x0, y0, x1, y1 = 0.0, 0.0, float(w), float(h)

    points = []
    # Walk around the rectangle: top edge left→right, right edge top→bottom,
    # bottom edge right→left, left edge bottom→top.
    for i in range(n_samples + 1):
        t = i / n_samples
        points.append((x0 + t * (x1 - x0), y0))        # top
    for i in range(n_samples + 1):
        t = i / n_samples
        points.append((x1, y0 + t * (y1 - y0)))         # right
    for i in range(n_samples + 1):
        t = i / n_samples
        points.append((x1 - t * (x1 - x0), y1))         # bottom
    for i in range(n_samples + 1):
        t = i / n_samples
        points.append((x0, y1 - t * (y1 - y0)))         # left

    return points


def _project_edge_to_pano(pto_data: tuple, img_index: int,
                          cam_points: list, out_w: int, out_h: int,
                          x_scale: float, y_scale: float,
                          crop_top: float = 0.0,
                          fisheye_radius: float = None):
    """
    Project a list of camera-pixel points through pto_mapper into panorama
    pixel coordinates.  Returns a list of (segment, clipped) pairs where
    segment is a list of (px, py) tuples and clipped is True if any point
    was dropped due to fisheye radius or out-of-bounds clipping (meaning the
    segment boundary is open and should not be filled).

    A new segment begins whenever a point fails to project (e.g. behind the
    camera) or jumps more than 10 % of the canvas width (seam wrap).

    crop_top: subtract this many pixels from every projected y (used when the
    stitched image is a vertically-cropped strip of the full equirect canvas).
    Points that fall outside [0, out_h] after cropping are discarded.

    fisheye_radius: if set, points further than this from the canvas centre are
    discarded (the stitcher clips the fisheye projection to this circle).
    """
    max_jump = out_w * 0.12   # threshold to detect wrap-around discontinuity
    fc_x = out_w / 2.0
    fc_y = out_h / 2.0

    segments = []     # list of (points_list, was_clipped)
    current = []
    current_clipped = False

    def _flush():
        nonlocal current, current_clipped
        if len(current) >= 2:
            segments.append((current, current_clipped))
        current = []
        current_clipped = False

    for (cx, cy) in cam_points:
        result = pto_mapper.map_image_to_pano(pto_data, img_index, cx, cy)
        if result is None:
            current_clipped = True
            _flush()
            continue
        px = result[0] * x_scale
        py = (result[1] - crop_top) * y_scale

        # Discard points outside the cropped canvas vertically
        if py < -out_h * 0.1 or py > out_h * 1.1:
            current_clipped = True
            _flush()
            continue

        # For fisheye: discard points outside the valid projection circle
        if fisheye_radius is not None:
            if math.hypot(px - fc_x, py - fc_y) > fisheye_radius:
                current_clipped = True
                _flush()
                continue

        if current:
            prev_px, prev_py = current[-1]
            if abs(px - prev_px) > max_jump or abs(py - prev_py) > max_jump * 0.5:
                _flush()

        current.append((px, py))

    _flush()
    return segments


def draw_camera_boundaries(pano_pto: str, lens_ptos: list, output: str,
                           out_w: int = None, out_h: int = None,
                           map_height: int = None,
                           crop_top: int = 0,
                           n_samples: int = 500,
                           label: bool = True) -> None:
    """
    Main routine. Parses PTOs, traces edges, projects, draws, saves PNG.

    Replicates what stitcher.py does to build the combined PTO:
    - Reads the panorama PTO for projection type (f) and HFOV (v).
    - For equirect: maps against orig_w=out_w, orig_h=map_height, which must
      match the stitcher's actual canvas height (e.g. 2160 for out_w=4096).
      crop_top is in these units and is subtracted from projected y.
    - For fisheye: applies rotate_panorama(pitch=-90) to all camera poses,
      exactly as stitcher.py does.
    """
    # --- Parse panorama PTO for projection params only ---
    pano_global, _ = pto_mapper.parse_pto_file(pano_pto)

    proj_f   = int(pano_global.get('f', 2))    # 2=equirect, 3=fisheye
    pano_v   = float(pano_global.get('v', 360))
    pano_w   = int(pano_global.get('w', 4096))
    pano_h   = int(pano_global.get('h', 2160))

    # out_w/out_h are the final rendered PNG size (matching the actual stitched image).
    canvas_w = out_w if out_w is not None else pano_w
    canvas_h = out_h if out_h is not None else pano_h

    # Build the mapping canvas that matches stitcher.py's calculate_source_coords:
    #   pano_x = x_dest + crop_offset_x  (crop_offset_x=0 for equirect)
    #   pano_y = y_dest + crop_offset_y  (crop_offset_y = crop_top)
    # Then angles are derived from (pano_x / orig_w, pano_y / orig_h).
    # orig_w = canvas_w (== stitch output width, crop_offset_x=0)
    # orig_h = map_height (must equal stitcher's canvas height, e.g. 2160 for 4096-wide)
    # So overlay pixel (px, py) = (pano_x, pano_y - crop_top) with 1:1 scale.
    if proj_f == 2:  # equirect
        map_w = canvas_w
        # map_h must match stitcher's orig_h exactly.  Caller provides this via
        # --map-height; fall back to pano_h from the PTO if not given.
        map_h = map_height if map_height is not None else pano_h
        x_scale = 1.0
        y_scale = 1.0
        fisheye_radius = None
    else:            # fisheye — output == full canvas, no crop
        map_w, map_h = canvas_w, canvas_h
        x_scale = 1.0
        y_scale = 1.0
        # The fisheye projection circle has radius = half the shorter canvas side.
        # Points outside it are clipped by the stitcher and should not be drawn.
        fisheye_radius = min(canvas_w, canvas_h) / 2.0

    # Build a clean global_options dict — no canvas roll (r=0), no scale (s=1).
    # This matches what stitcher.py produces via build_pto_header().
    mapping_global = {
        'f': proj_f,
        'v': pano_v,
        'w': map_w,
        'h': map_h,
        'r': 0.0,
        's': 1.0,
    }

    # --- Parse all camera lens PTOs ---
    camera_imgs = []
    for path in lens_ptos:
        try:
            img_params = _parse_lens_pto(path)
            camera_imgs.append((path, img_params))
        except Exception as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)

    if not camera_imgs:
        print("Error: no valid lens PTOs loaded.", file=sys.stderr)
        sys.exit(1)

    combined_images = [img for _, img in camera_imgs]
    pto_data = (mapping_global, combined_images)

    # For fisheye: apply the same pitch=-90 rotation that stitcher.py applies
    # via rotate_panorama(yaw=0, pitch=-90, roll=0).  This tilts all cameras so
    # that the zenith maps to the centre of the fisheye canvas.
    if proj_f == 3:
        pto_mapper.rotate_panorama(pto_data, yaw_deg=0, pitch_deg=-90, roll_deg=0)

    # --- Build fill layer via inverse mapping (panorama → camera) ---
    # Compute at reduced resolution then upscale — keeps Python loop fast.
    FILL_RES = max(canvas_w, canvas_h) // 3  # 3× upscale to canvas at end
    fill_scale = min(1.0, FILL_RES / max(canvas_w, canvas_h))
    fw = max(1, int(canvas_w * fill_scale))
    fh = max(1, int(canvas_h * fill_scale))

    fill_colour_u8 = []
    for c in [COLOURS[i % len(COLOURS)] for i in range(len(camera_imgs))]:
        fill_colour_u8.append((int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)))

    fill_alpha = int(round(FILL_OPACITY * 255))
    cw, ch_i = int(canvas_w), int(canvas_h)

    # camera_bounds: (x0,y0,x1,y1) per camera for fast inside-check
    cam_bounds = []
    for _, img_params in camera_imgs:
        S = img_params.get('S')
        if S is not None:
            cam_bounds.append((float(S[0]), float(S[1]), float(S[2]), float(S[3])))
        else:
            cam_bounds.append((0.0, 0.0, float(img_params.get('w', 1920)),
                               float(img_params.get('h', 1080))))

    # Fisheye mask at low res
    fr_fisheye = fisheye_radius * fill_scale if fisheye_radius is not None else None
    if fr_fisheye is not None:
        gy, gx = np.mgrid[0:fh, 0:fw]
        fe_mask = np.hypot(gx - fw / 2.0, gy - fh / 2.0) <= fr_fisheye
    else:
        fe_mask = np.ones((fh, fw), dtype=bool)

    # Low-res panorama canvas dimensions fed into calculate_source_coords:
    # The function maps (final_w, final_h) dest pixels at (orig_w, orig_h) canvas.
    # We work at fill_scale of the canvas, so orig dimensions stay the same
    # but final_w/h = fw/fh.  crop_offset accounts for equirect crop_top.
    orig_w_map = map_w   # mapping canvas width  (e.g. 4096 for equirect)
    orig_h_map = map_h   # mapping canvas height (e.g. 2160 for equirect hires)
    pano_hfov  = float(pano_v)
    pano_proj_f = proj_f
    # crop_offset: the fill canvas covers [0..canvas_w] × [crop_top..canvas_h+crop_top]
    # in mapping canvas coords.  Each fill pixel (px_f, py_f) maps to:
    #   pano_x = px_f / fill_scale,  pano_y = py_f / fill_scale + crop_top
    # calculate_source_coords uses crop_offset_x/y as the panorama-pixel offset
    # of the top-left of the dest grid within the full mapping canvas.
    crop_offset_x = 0.0
    crop_offset_y = float(crop_top)
    # pano_s scales dest → pano; we want dest_pixel * (1/fill_scale) = pano_x,
    # so pano_s = fill_scale (dest pixel is fill_scale of a full pano pixel).
    pano_s = fill_scale

    cam_masks_lr = []
    for cam_idx, (_, img_params) in enumerate(camera_imgs):
        img = combined_images[cam_idx]
        sw   = img.get('w');  sh  = img.get('h')
        fov  = img.get('v');  src_proj_f = int(img.get('f', 0))
        fov_rad = math.radians(fov)
        if src_proj_f == 0:
            src_focal = sw / (2 * math.tan(fov_rad / 2)) if fov_rad > 0 else 0
        elif src_proj_f == 3:
            src_focal = sw / fov_rad if fov_rad > 0 else 0
        else:
            src_focal = sw / (2 * math.tan(fov_rad / 2)) if fov_rad > 0 else 0
        src_norm_radius = min(sw, sh) / 2.0
        cam_y = img.get('y', 0)
        cam_p = img.get('p', 0)
        cam_r = -img.get('r', 0)
        a = img.get('a', 0); b = img.get('b', 0); c = img.get('c', 0)
        cx = -img.get('d', 0); cy = img.get('e', 0)
        R_pr_inv = pto_mapper.create_pr_rotation_matrix(cam_p, cam_r).T

        coords = np.empty((fh, fw, 2), dtype=np.float32)
        pto_mapper.calculate_source_coords(
            coords, fw, fh, orig_w_map, orig_h_map,
            crop_offset_x, crop_offset_y,
            pano_proj_f, pano_hfov,
            sw, sh, R_pr_inv, cam_y,
            src_focal, src_norm_radius,
            a, b, c, cx, cy, src_proj_f,
            0.0, pano_s
        )

        INVALID = -99999.0
        sx = coords[:, :, 0]
        sy = coords[:, :, 1]
        x0b, y0b, x1b, y1b = cam_bounds[cam_idx]
        valid = (
            fe_mask &
            (sx > INVALID + 1) &
            (sx >= x0b) & (sx <= x1b) &
            (sy >= y0b) & (sy <= y1b)
        )
        cam_masks_lr.append(valid)

    # Porter-Duff blend at LOW resolution (fw×fh), then upscale to canvas.
    # This avoids ~5s of masked boolean ops on the full 4096×4096 array.
    sa = fill_alpha / 255.0
    acc_rgb_lr = np.zeros((fh, fw, 3), dtype=np.float32)
    acc_a_lr   = np.zeros((fh, fw),    dtype=np.float32)
    for cam_idx, cam_mask_lr in enumerate(cam_masks_lr):
        if not cam_mask_lr.any():
            continue
        rc, gc, bc = fill_colour_u8[cam_idx]
        da = acc_a_lr[cam_mask_lr]
        out_a = sa + da * (1.0 - sa)
        out_a_safe = np.where(out_a > 0, out_a, 1.0)
        for ch, cv in enumerate((rc, gc, bc)):
            acc_rgb_lr[cam_mask_lr, ch] = (
                cv * sa + acc_rgb_lr[cam_mask_lr, ch] * da * (1.0 - sa)
            ) / out_a_safe
        acc_a_lr[cam_mask_lr] = out_a

    # Fill uncovered pixels (inside valid panorama area) with 50% black
    gap_mask = (acc_a_lr == 0) & fe_mask
    acc_rgb_lr[gap_mask] = 0.0
    acc_a_lr[gap_mask]   = 0.5

    # Build low-res RGBA and upscale with bilinear filter for smooth boundaries
    rgba_lr = np.empty((fh, fw, 4), dtype=np.uint8)
    rgba_lr[:, :, :3] = np.clip(acc_rgb_lr, 0, 255).astype(np.uint8)
    rgba_lr[:, :,  3] = np.clip(acc_a_lr * 255, 0, 255).astype(np.uint8)
    fill_img = wand.image.Image.from_array(rgba_lr, channel_map='RGBA')
    fill_img.resize(cw, ch_i, filter='mitchell')

    # Upscale individual masks nearest-neighbour for label centroid only
    ys = (np.arange(ch_i) * fill_scale).astype(int).clip(0, fh - 1)
    xs = (np.arange(cw)   * fill_scale).astype(int).clip(0, fw - 1)
    cam_masks = [m[np.ix_(ys, xs)] for m in cam_masks_lr]

    # --- Draw outline + labels on a separate transparent layer ---
    img_canvas = wand.image.Image(
        width=int(canvas_w),
        height=int(canvas_h),
        background=wand.color.Color('transparent')
    )

    # Collect label positions before drawing
    label_draws = []
    if label:
        font_size = max(8, int(32 * min(canvas_w, canvas_h) / 448))
        half_fs = font_size // 2
        for cam_idx in range(len(camera_imgs)):
            if cam_idx >= len(cam_masks) or not cam_masks[cam_idx].any():
                continue
            stroke_colour = STROKE_COLOURS[cam_idx % len(STROKE_COLOURS)]
            ys_m, xs_m = np.where(cam_masks[cam_idx])
            ly = int(ys_m.mean())
            rows_near = np.abs(ys_m - ly) <= max(1, font_size // 2)
            if rows_near.any():
                near_xs = np.sort(xs_m[rows_near])
                gaps   = np.where(np.diff(near_xs) > canvas_w // 4)[0]
                starts = np.concatenate(([0], gaps + 1))
                ends   = np.concatenate((gaps + 1, [len(near_xs)]))
                best   = int(np.argmax(ends - starts))
                lx = int(near_xs[starts[best]:ends[best]].mean())
            else:
                lx = int(xs_m.mean())
            lx = int(np.clip(lx, half_fs, canvas_w - half_fs))
            ly = int(np.clip(ly, half_fs, canvas_h - half_fs))
            label_draws.append((cam_idx, stroke_colour, lx, int(ly + font_size * 0.35)))

    # Pass 1: polylines only (clean stroke state, no fill)
    with wand.drawing.Drawing() as draw:
        draw.fill_color   = wand.color.Color('none')
        draw.fill_opacity = 0.0
        draw.stroke_width = STROKE_WIDTH

        for cam_idx, (path, img_params) in enumerate(camera_imgs):
            stroke_colour = STROKE_COLOURS[cam_idx % len(STROKE_COLOURS)]
            cam_points = _trace_camera_edge(img_params, n_samples)
            segments = _project_edge_to_pano(
                pto_data, cam_idx, cam_points,
                canvas_w, canvas_h, x_scale, y_scale,
                crop_top=crop_top,
                fisheye_radius=fisheye_radius
            )
            draw.stroke_color   = wand.color.Color(stroke_colour)
            draw.stroke_opacity = STROKE_OPACITY
            for seg, _clipped in segments:
                if len(seg) >= 2:
                    draw.polyline(seg)

        draw(img_canvas)

    # Pass 2: labels only (clean fill state, no stroke)
    if label_draws:
        with wand.drawing.Drawing() as draw:
            draw.stroke_color   = wand.color.Color('none')
            draw.stroke_opacity = 0.0
            draw.stroke_width   = 0
            draw.font           = 'helvetica'
            draw.font_size      = font_size
            draw.font_weight    = 700
            draw.text_alignment = 'center'
            draw.fill_opacity   = 0.5
            for cam_idx, stroke_colour, tx, ty in label_draws:
                draw.fill_color = wand.color.Color(stroke_colour)
                draw.text(tx, ty, str(cam_idx + 1))
            draw(img_canvas)

    # Composite: fill layer underneath, outlines on top
    fill_img.composite(img_canvas, left=0, top=0)
    fill_img.save(filename=output)
    print(f"Saved: {output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Draw camera FOV boundaries onto a stitched panorama overlay.'
    )
    parser.add_argument('--pano', required=True,
                        help='Panorama PTO file (sets output projection and canvas size)')
    parser.add_argument('--lens', nargs='+', required=True,
                        help='One or more lens.pto files (one per camera)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output PNG filename')
    parser.add_argument('--width', '-W', type=int, default=None,
                        help='Override output width (default: from pano PTO)')
    parser.add_argument('--height', '-H', type=int, default=None,
                        help='Override output height (default: from pano PTO)')
    parser.add_argument('--map-height', type=int, default=None,
                        help='For equirect: stitcher canvas height (orig_h), e.g. 2160 for 4096-wide output')
    parser.add_argument('--crop-top', type=int, default=0,
                        help='For equirect: row offset of the cropped strip within the stitcher canvas (in map-height units)')
    parser.add_argument('--samples', '-n', type=int, default=500,
                        help='Number of sample points per camera edge (default: 500)')
    parser.add_argument('--no-label', dest='label', action='store_false',
                        help='Suppress camera name labels')
    parser.set_defaults(label=True)
    args = parser.parse_args()

    draw_camera_boundaries(
        pano_pto=args.pano,
        lens_ptos=args.lens,
        output=args.output,
        out_w=args.width,
        out_h=args.height,
        map_height=args.map_height,
        crop_top=args.crop_top,
        n_samples=args.samples,
        label=args.label,
    )


if __name__ == '__main__':
    main()
