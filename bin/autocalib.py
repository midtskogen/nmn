#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a Hugin .pto lens file from a star-field image using a local tetra3 blind plate solver.

The script is intended for wide-field / all-sky cameras. The local solver's pattern
database is built from the project's stars.py catalogue and covers roughly 10-30 deg
fields of view, so the solver is run on a central crop whose FOV is inside that range.
The resulting centre, roll and FOV are then scaled back to the full image and written
as a single-image .pto file in the same style as nmn/bin/amscalib2lens.py and lens.pto.

Usage:
    autocalib.py <image> <output.pto> [options]

Example:
    autocalib.py ~/OSL_cam2.jpg ~/OSL_cam2.pto \
        -T 1756172400 -y 59.97056 -x 10.649639 -e 348
"""

import argparse
import configparser
import copy
import io
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numba import njit

from PIL import Image, ImageChops, ImageDraw, ImageFilter

import ephem

try:
    import pto_mapper
except ImportError:
    pto_mapper = None

try:
    from tetra3 import Tetra3, get_centroids_from_image
except ImportError as exc:
    Tetra3 = None
    get_centroids_from_image = None
    TETRA3_ERR = exc

# Initial lens-model guesses: median values from 84 lens.pto files across
# ams123, ams135, ams171-180 (cam1-7). Yaw/pitch/roll and centre RA/Dec are
# solved per-image; the rest are kept fixed as a starting point.
INITIAL_FOV = 83.46803934632945
INITIAL_A = -0.00415368419392758
INITIAL_B = -0.00658489763649841
INITIAL_C = -0.0182364818863498
INITIAL_D = 37.10688673257604
INITIAL_E = 21.55064329950385

# Acceptance gates for the blind solve. A marginal solve (e.g. clouds producing
# cloud-edge centroids instead of stars) should be reported as a failure rather
# than silently adopted as a calibration.
ACCEPT_MAX_PROB = 1e-3        # tetra3 false-positive probability (upper bound)
ACCEPT_MIN_MATCHES = 8        # stars matched in the central tetra3 solve
ACCEPT_MAX_RMSE_PX = 8.0      # final pixel RMSE of inlier control points
ACCEPT_MIN_INLIERS = 15       # control points surviving outlier rejection
ACCEPT_MIN_INLIER_FRAC = 0.5  # inliers / matched centroid candidates
ACCEPT_MIN_COVERAGE = 0.4     # fraction of a 4x4 image grid containing control points


def _coverage_fraction(control_points, width, height, grid=4, mask=None):
    """Fraction of usable grid cells containing at least one control point.

    'Usable' cells are those not dominated by the foreground mask (mask is
    white where foreground). A cloud bank covering part of the sky leaves
    matched stars clustered in one region; full-sky solutions spread across
    most usable cells."""
    if not control_points or not width or not height:
        return 0.0
    usable_cells = None
    if mask is not None:
        import numpy as _np
        m = _np.asarray(mask.resize((grid, grid), Image.Resampling.BOX))
        usable_cells = {(cx, cy) for cy in range(grid) for cx in range(grid)
                        if m[cy, cx] < 128}
    cells = set()
    for x, y, *_ in control_points:
        cells.add((min(grid - 1, int(x / width * grid)),
                   min(grid - 1, int(y / height * grid))))
    if usable_cells is not None:
        return len(cells & usable_cells) / max(1, len(usable_cells))
    return len(cells) / (grid * grid)


def evaluate_solve(result, rmse_px, control_points, matched_candidates,
                   width, height, max_prob=None, min_matches=None,
                   max_rmse_px=None, min_inliers=None, min_inlier_frac=None,
                   min_coverage=None, mask=None):
    """Score a blind solve against the acceptance gates.

    Returns a dict with the individual metrics, a human-readable 'summary',
    and a 'failures' list (empty = accept). Thresholds default to the
    ACCEPT_* module constants."""
    max_prob = ACCEPT_MAX_PROB if max_prob is None else max_prob
    min_matches = ACCEPT_MIN_MATCHES if min_matches is None else min_matches
    max_rmse_px = ACCEPT_MAX_RMSE_PX if max_rmse_px is None else max_rmse_px
    min_inliers = ACCEPT_MIN_INLIERS if min_inliers is None else min_inliers
    min_inlier_frac = ACCEPT_MIN_INLIER_FRAC if min_inlier_frac is None else min_inlier_frac
    min_coverage = ACCEPT_MIN_COVERAGE if min_coverage is None else min_coverage

    prob = float(result['Prob']) if result.get('Prob') is not None else None
    n_matches = result.get('Matches')
    inliers = len(control_points)
    inlier_frac = inliers / matched_candidates if matched_candidates else 0.0
    coverage = _coverage_fraction(control_points, width, height, mask=mask)

    failures = []
    if prob is not None and prob > max_prob:
        failures.append(f'tetra3 false-positive probability {prob:.3g} > {max_prob:g}')
    if n_matches is not None and n_matches < min_matches:
        failures.append(f'only {n_matches} matched stars (< {min_matches})')
    if rmse_px is None or rmse_px > max_rmse_px:
        failures.append(f'pixel RMSE {"infinite" if rmse_px is None else f"{rmse_px:.2f}px"} '
                        f'over limit {max_rmse_px}px')
    if inliers < min_inliers:
        failures.append(f'only {inliers} inlier control points (< {min_inliers})')
    if inlier_frac < min_inlier_frac:
        failures.append(f'inlier fraction {inlier_frac:.2f} < {min_inlier_frac}')
    if coverage < min_coverage:
        failures.append(f'control-point sky coverage {coverage:.2f} < {min_coverage}')

    return {
        'prob': prob, 'matches': n_matches, 'rmse_px': rmse_px,
        'inliers': inliers, 'matched': matched_candidates,
        'inlier_frac': inlier_frac, 'coverage': coverage,
        'failures': failures,
        'summary': (f'prob={prob if prob is not None else "?"}, matches={n_matches}, '
                    f'rmse={rmse_px if rmse_px is None else f"{rmse_px:.3f}px"}, '
                    f'inliers={inliers}/{matched_candidates} ({inlier_frac:.2f}), '
                    f'coverage={coverage:.2f}'),
    }


@njit(cache=True, fastmath=True)
def _project_stars_numba(azimuths, altitudes, params, width, height, projection=3):
    fov, yaw, pitch, roll, a, b, c, d, e = params
    fov_rad = math.radians(fov)
    if projection == 3:
        focal = width / fov_rad
    else:
        half = math.tan(fov_rad / 2.0)
        focal = width / (2.0 * half) if half > 1e-9 else width * 1e9
    norm_radius = min(width, height) / 2.0
    distortion_base = 1.0 - a - b - c
    p = math.radians(pitch)
    r = math.radians(-roll)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    output = np.empty((len(azimuths), 2), dtype=np.float64)
    for i in range(len(azimuths)):
        altitude = math.radians(altitudes[i])
        adjusted_yaw = math.radians(azimuths[i] - 180.0 - yaw)
        ca = math.cos(altitude)
        vx = ca * math.sin(adjusted_yaw)
        vy = math.sin(altitude)
        vz = -ca * math.cos(adjusted_yaw)
        x_rot = cr*vx + cp*sr*vy + sp*sr*vz
        y_rot = -sr*vx + cp*cr*vy + sp*cr*vz
        z_rot = -sp*vy + cp*vz
        if projection == 3:
            theta = math.atan2(math.hypot(x_rot, y_rot), -z_rot)
            phi = math.atan2(y_rot, x_rot)
            radius = focal * theta
            x_ideal = radius * math.cos(phi)
            y_ideal = radius * math.sin(phi)
        else:
            if z_rot >= -1e-6:
                x_ideal = 0.0
                y_ideal = 0.0
                radius = 0.0
            else:
                x_ideal = focal * x_rot / -z_rot
                y_ideal = focal * y_rot / -z_rot
                radius = math.hypot(x_ideal, y_ideal)
        rn = radius / norm_radius
        magnification = distortion_base + rn * (c + rn * (b + rn * a))
        output[i, 0] = x_ideal * magnification + d + width / 2.0
        output[i, 1] = -y_ideal * magnification + e + height / 2.0
    return output


# Ensure local project modules are importable even when this script is executed via symlink
_SCRIPT_PATH = Path(__file__).resolve()
_PROJECT_DIR = None
for _cand in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
    if (_cand / 'bin').is_dir() and (_cand / 'server').is_dir():
        _PROJECT_DIR = _cand
        break
if _PROJECT_DIR is not None:
    _BIN_DIR = _PROJECT_DIR / 'bin'
    _SRC_DIR = _PROJECT_DIR / 'src'
    for _p in (_BIN_DIR, _SRC_DIR, _PROJECT_DIR):
        if _p.exists():
            _ps = str(_p)
            if _ps not in sys.path:
                sys.path.insert(0, _ps)


def _load_config(args):
    """Load observer settings from a .cfg or .json config file, if available."""
    path = args.config
    if not path:
        path = '/etc/meteor.cfg' if os.path.isfile('/etc/meteor.cfg') else None
    if not path:
        path = '/home/ams/amscams/conf/as6.json' if os.path.isfile('/home/ams/amscams/conf/as6.json') else None
    if not path or not os.path.exists(path):
        return configparser.ConfigParser()

    config = configparser.ConfigParser()
    if path.endswith('.json'):
        try:
            site = json.load(open(path)).get('site', {})
            config.add_section('astronomy')
            for key, opt in (('device_lat', 'latitude'), ('device_lng', 'longitude'), ('device_alt', 'elevation')):
                if key in site:
                    config.set('astronomy', opt, str(site[key]))
        except (json.JSONDecodeError, OSError):
            pass
    else:
        config.read(path)
    return config


def _camera_mask_path(path):
    """Return the camera mask path for a /meteor/camN/YYYYMMDD/HH/ image path."""
    m = re.search(r'(/meteor/cam[^/]+)/\d{8}/\d{2}/', path)
    return f'{m.group(1)}/mask.png' if m else None


def _resolve_mask_path(image_path, mask_path=None):
    """Find a usable mask file: the inferred/explicit path first, then a
    cached copy under bin/data/masks/cam<N>.png for hosts where the camera
    mask symlink is dangling."""
    candidates = [mask_path or _camera_mask_path(image_path)]
    m = re.search(r'/meteor/(cam[^/]+)/', image_path)
    if m:
        candidates.append(str(_SCRIPT_PATH.parent / 'data' / 'masks'
                              / f'{m.group(1)}.png'))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return candidates[0]


def _auto_mask(image, win=31):
    """Derive a rough foreground mask from the image itself.

    Night-sky background is smooth and dark; foreground (trees, masts, lamps,
    cloud glow, out-of-frame bleed) is locally bright or high-contrast.
    Returns a boolean array — True where foreground. Only used when no camera
    mask file is available; it exists to keep terrain centroids out of the
    solver, not to be a precise sky mask."""
    import numpy as _np
    from scipy.ndimage import (uniform_filter, binary_closing, binary_dilation,
                               binary_fill_holes, label)
    a = _np.asarray(image, dtype=_np.float64)
    mean = uniform_filter(a, win)
    sq = uniform_filter(a * a, win)
    std = _np.sqrt(_np.maximum(sq - mean * mean, 0))
    med_mean = _np.median(mean)
    med_std = _np.median(std)
    bright = mean > med_mean + max(12.0, 3.0 * _np.median(_np.abs(mean - med_mean)) * 1.4826)
    textured = std > med_std + max(6.0, 3.0 * _np.median(_np.abs(std - med_std)) * 1.4826)
    fg = bright | textured
    fg = binary_closing(fg, iterations=4)
    fg = binary_dilation(fg, iterations=4)
    fg = binary_fill_holes(fg)
    # Keep only components touching the frame border or covering >=1% of the
    # image — interior bright blobs are often legitimate sky features.
    lab, n = label(fg)
    if n:
        border = _np.zeros(fg.shape, bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        sizes = _np.bincount(lab.ravel())
        big = sizes >= 0.01 * a.size
        big[0] = False
        touch = _np.unique(lab[border])
        keep_ids = set(touch[touch != 0]) | set(_np.nonzero(big)[0])
        fg = _np.isin(lab, list(keep_ids))
    return fg


def _apply_camera_mask(image, image_path, mask_path=None, verbose=False):
    """Remove foreground where the camera mask is white.

    Returns (masked_image, mask_image, source) where source is 'file', 'auto'
    or 'none'. The mask comes from the camera mask file when available;
    otherwise a rough mask is derived from the image itself so terrain still
    stays out of the solve (chicken-and-egg: the camera mask depends on the
    calibration we may not have)."""
    mask_path = _resolve_mask_path(image_path, mask_path)
    mask = None
    source = 'none'
    if mask_path and os.path.isfile(mask_path):
        mask = Image.open(mask_path).convert('L')
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        if verbose:
            print(f'Applying foreground mask (white pixels excluded): {mask_path}')
        source = 'file'
    else:
        if not mask_path:
            print('Warning: no camera mask path inferred from input filename.',
                  file=sys.stderr)
        elif os.path.lexists(mask_path):
            print(f'Warning: camera mask is a broken link: {mask_path} -> '
                  f'{os.readlink(mask_path)}.', file=sys.stderr)
        else:
            print(f'Warning: camera mask not found: {mask_path}.', file=sys.stderr)
        fg = _auto_mask(image)
        if fg.any():
            print(f'Using self-derived foreground mask ({100.0 * fg.mean():.0f}% '
                  'of frame masked).', file=sys.stderr)
            mask = Image.fromarray(np.where(fg, 255, 0).astype(np.uint8))
            source = 'auto'
        if mask is None:
            return image, None, source
    keep_mask = ImageChops.invert(mask)
    smooth_foreground = image.filter(ImageFilter.GaussianBlur(25))
    return Image.composite(image, smooth_foreground, keep_mask), mask, source


def _parse_timestamp_from_path(path):
    """Parse a meteor-style path like .../20260826/23/full_00.jpg into a UTC Unix timestamp."""
    m = re.search(r'(\d{4})(\d{2})(\d{2})/(\d{2})/full_(\d{2})', path)
    if not m:
        return None
    year, month, day, hour, minute = map(int, m.groups())
    try:
        dt = datetime(year, month, day, hour, minute, 0, tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _setup_observer(args, config):
    cfg = config if config.has_section('astronomy') else None
    lat = args.latitude if args.latitude is not None else (cfg.get('astronomy', 'latitude') if cfg else None)
    lon = args.longitude if args.longitude is not None else (cfg.get('astronomy', 'longitude') if cfg else None)
    if lat is None or lon is None:
        raise ValueError('Observer latitude/longitude are required (use -y/-x or a config file).')
    obs = ephem.Observer()
    obs.lat = str(lat)
    obs.lon = str(lon)
    obs.elevation = args.elevation if args.elevation is not None else cfg.getfloat('astronomy', 'elevation', fallback=0.0) if cfg else 0.0
    if cfg:
        obs.temp = cfg.getfloat('astronomy', 'temperature', fallback=10.0)
        obs.pressure = cfg.getfloat('astronomy', 'pressure', fallback=1010.0)

    if args.timestamp is not None:
        timestamp = args.timestamp
    else:
        timestamp = _parse_timestamp_from_path(args.image)
        if timestamp is None:
            timestamp = os.path.getmtime(args.image)
            if args.verbose:
                print(f'Warning: no timestamp given and path not recognised, using image mtime: {timestamp}')
        elif args.verbose:
            dt = datetime.fromtimestamp(timestamp, timezone.utc)
            print(f'Parsed timestamp from path: {dt}')
    obs.date = datetime.fromtimestamp(float(timestamp), timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    return obs, timestamp


def _equidistant_warp(image, f_px, axis_xy, out_size, out_fov_deg,
                      lens_xy=None, mask=None):
    """Reproject an off-axis region of an equidistant fisheye into a centred
    equidistant view.

    The lens model (equidistant about the lens centre, scale f px/rad) is
    known; only the pointing is unknown. The output image is a spherical
    rotation of the source, so it is itself a valid centred equidistant
    fisheye image that tetra3 can solve directly. Returns the warped image
    plus a callable mapping output (x, y) pixels back to source pixels."""
    import numpy as _np
    w, h = image.size
    if lens_xy is None:
        lens_xy = (w / 2.0 + INITIAL_D, h / 2.0 + INITIAL_E)
    src = _np.asarray(image, dtype=_np.float64)
    W = out_size
    f_out = W / math.radians(out_fov_deg)

    yy, xx = _np.mgrid[0:W, 0:W].astype(_np.float64) - (W - 1) / 2.0
    r = _np.hypot(xx, yy)
    theta = r / f_out                       # angular dist from virtual axis
    phi = _np.arctan2(yy, xx)

    # direction in the canonical frame: +z = forward, azimuth phi
    dx = _np.sin(theta) * _np.cos(phi)
    dy = _np.sin(theta) * _np.sin(phi)
    dz = _np.cos(theta)

    # axis direction in the source lens frame
    ax, ay = axis_xy[0] - lens_xy[0], axis_xy[1] - lens_xy[1]
    t0 = math.hypot(ax, ay) / f_px
    p0 = math.atan2(ay, ax)

    # Rodrigues rotation carrying +z onto the axis direction
    a = _np.array([math.sin(t0) * math.cos(p0), math.sin(t0) * math.sin(p0), math.cos(t0)])
    k = _np.cross([0, 0, 1], a)
    kn = _np.linalg.norm(k)
    if kn > 1e-12:
        k /= kn
        ct, st = math.cos(t0), math.sin(t0)
        vx = k[1]*dz - k[2]*dy; vy = k[2]*dx - k[0]*dz; vz = k[0]*dy - k[1]*dx
        kd = k[0]*dx + k[1]*dy + k[2]*dz
        d_x = dx*ct + vx*st + k[0]*kd*(1-ct)
        d_y = dy*ct + vy*st + k[1]*kd*(1-ct)
        d_z = dz*ct + vz*st + k[2]*kd*(1-ct)
    else:
        d_x, d_y, d_z = dx, dy, dz

    theta_s = _np.arccos(_np.clip(d_z, -1, 1))
    phi_s = _np.arctan2(d_y, d_x)
    r_s = f_px * theta_s
    sx = lens_xy[0] + r_s * _np.cos(phi_s)
    sy = lens_xy[1] + r_s * _np.sin(phi_s)

    from scipy.ndimage import map_coordinates
    warped = map_coordinates(src, [sy, sx], order=1, mode='constant', cval=0.0)
    # Black out pixels sampled from outside the frame or from masked
    # foreground, so the solver's centroid extractor only sees real sky.
    out = (sx < 0) | (sx >= w - 1) | (sy < 0) | (sy >= h - 1)
    if mask is not None:
        m = _np.asarray(mask.resize((w, h), Image.Resampling.NEAREST), dtype=_np.float64) > 128
        out |= map_coordinates(m.astype(_np.float64), [sy, sx], order=0,
                               mode='constant', cval=1.0) > 0.5
    warped[out] = 0.0

    def from_warp(xw, yw):
        """Map a warped-image pixel back to source-image coordinates."""
        rx, ry = xw - (W - 1) / 2.0, yw - (W - 1) / 2.0
        th = math.hypot(rx, ry) / f_out
        ph = math.atan2(ry, rx)
        ddx = math.sin(th) * math.cos(ph); ddy = math.sin(th) * math.sin(ph); ddz = math.cos(th)
        if kn > 1e-12:
            vx_ = k[1]*ddz - k[2]*ddy; vy_ = k[2]*ddx - k[0]*ddz; vz_ = k[0]*ddy - k[1]*ddx
            kd_ = k[0]*ddx + k[1]*ddy + k[2]*ddz
            ddx = ddx*ct + vx_*st + k[0]*kd_*(1-ct)
            ddy = ddy*ct + vy_*st + k[1]*kd_*(1-ct)
            ddz = ddz*ct + vz_*st + k[2]*kd_*(1-ct)
        th_s = math.acos(max(-1.0, min(1.0, ddz)))
        ph_s = math.atan2(ddy, ddx)
        rr = f_px * th_s
        return lens_xy[0] + rr * math.cos(ph_s), lens_xy[1] + rr * math.sin(ph_s)

    return Image.fromarray(warped.astype(_np.uint8)), from_warp


def _largest_sky_disk(mask, w, h):
    """Centre (x, y) and pixel radius of the largest circle of unmasked sky.

    Uses a distance transform over the foreground mask so the axis lands in
    the middle of the widest contiguous clear region rather than at the
    density peak near the sky's edge. Returns None when no mask is given."""
    if mask is None:
        return None
    import numpy as _np
    from scipy.ndimage import distance_transform_edt
    free = _np.asarray(mask.resize((w, h), Image.Resampling.NEAREST)) < 128
    dist = distance_transform_edt(free)
    iy, ix = _np.unravel_index(int(dist.argmax()), dist.shape)
    return float(ix), float(iy), float(dist[iy, ix])


def _centroid_density_centers(image, w, h, cols=8, rows=5, top_n=4, mask=None):
    """Return the centers of the grid cells with the most star centroids.

    Used to place solve crops where the sky is actually visible for cameras
    whose lower half is terrain (central crops hit trees/masts there). Cells
    dominated by foreground mask are skipped — they would otherwise win on
    spurious centroids from lights and mask edges."""
    import numpy as _np
    centroids = get_centroids_from_image(
        image, sigma=2.0, image_th=None, crop=None, downsample=None,
        filtsize=15, max_area=500, min_area=3, max_returned=2000)
    mask_frac = {}
    if mask is not None:
        m = _np.asarray(mask.resize((cols, rows), Image.Resampling.BOX), dtype=_np.float64)
        mask_frac = {(cx, cy): m[cy, cx] / 255.0 for cy in range(rows) for cx in range(cols)}
    counts = {}
    for pt in centroids:
        y, x = float(pt[0]), float(pt[1])
        cell = (min(cols - 1, int(x / w * cols)), min(rows - 1, int(y / h * rows)))
        counts[cell] = counts.get(cell, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    centers = []
    for (cx, cy), n in ranked:
        if n < 8:
            break
        if mask_frac.get((cx, cy), 0.0) > 0.5:
            continue
        centers.append(((cx + 0.5) * w / cols, (cy + 0.5) * h / rows))
        if len(centers) >= top_n:
            break
    return centers


def _solve_image(t3, image, verbose=False, mask=None):
    """Blind-solve the sky in `image`.

    Strategy:
    1. centred crops under equidistant and rectilinear models;
    2. if weak/failed, raw crops centred on star-density peaks (for cameras
       with terrain covering the image centre);
    3. if still weak, equidistant *reprojected* views centred on the density
       peaks — a spherical rotation of the known fisheye model, so the result
       is a valid centred fisheye patch regardless of where the sky is.

    Returns (result, seed_pairs, est_fov, est_proj): seed_pairs is a list of
    (x, y, ra_deg, dec_deg, cat_id) in full-image coordinates; est_fov is the
    full-frame FOV implied by the winning crop and est_proj its projection
    ('equidistant'/'rectilinear'), or (None, None, None, None) on failure."""
    w, h = image.size
    extract = {'sigma': 3, 'filtsize': 15, 'max_area': 500, 'min_area': 3, 'max_returned': 100}
    central = []
    for fov in (60, 50, 40, 30, 25, 20):
        cw = min(w, int(round(w * fov / INITIAL_FOV)))
        central.append((cw, cw, 'equidistant', float(fov), w / 2, h / 2))
    for size, fov in ((int(round(w * 0.3)), 25.0),
                      (640, 20.0), (768, 30.0), (480, 15.0)):
        central.append((size, size, 'rectilinear', fov, w / 2, h / 2))
    candidates = _try_solve_crops(t3, image, central, extract, verbose)
    if candidates and min(candidates, key=lambda x: x[0])[0] < 1e-12:
        best = min(candidates, key=lambda x: x[0])
        _report_solve(best, verbose)
        return best[1], _seed_pairs(best), *_estimate_full_fov(best, w)

    # Central crops failed or produced a weak solve: place raw crops on the
    # densest star regions (terrain/obstructions may cover the centre).
    centers = _centroid_density_centers(image, w, h, mask=mask)
    disk = _largest_sky_disk(mask, w, h)
    if disk and not any(math.hypot(disk[0] - c[0], disk[1] - c[1]) < w / 16
                        for c in centers):
        centers = [disk[:2]] + centers
    off_center = []
    if centers:
        for cx, cy in centers:
            for size, fov in ((min(w, int(round(w * 30 / INITIAL_FOV))), 30.0),
                              (min(w, int(round(w * 40 / INITIAL_FOV))), 40.0)):
                off_center.append((size, size, 'equidistant', fov, cx, cy))
        candidates += _try_solve_crops(t3, image, off_center, extract, verbose)
    if candidates and min(candidates, key=lambda x: x[0])[0] < 1e-12:
        best = min(candidates, key=lambda x: x[0])
        _report_solve(best, verbose)
        return best[1], _seed_pairs(best), *_estimate_full_fov(best, w)

    # Raw crops still failed: reproject to centred equidistant views pointed
    # at the sky — the largest unmasked disk first, then the density peaks —
    # and solve those (exact fisheye geometry).
    warp_axes = []
    disk = _largest_sky_disk(mask, w, h)
    if disk:
        warp_axes.append(disk[:2])
    warp_axes += [c for c in centers
                  if not any(math.hypot(c[0] - a[0], c[1] - a[1]) < w / 16
                             for a in warp_axes)]
    if warp_axes:
        f_px = w / math.radians(INITIAL_FOV)
        warp_candidates = []
        for cx, cy in warp_axes[:4]:
            for out_fov, out_size in ((50.0, 768), (70.0, 960)):
                warped, from_warp = _equidistant_warp(image, f_px, (cx, cy),
                                                      out_size, out_fov,
                                                      mask=mask)
                if verbose:
                    print(f'warped equidistant view {out_size}x{out_size} '
                          f'axis=({cx:.0f},{cy:.0f}), FOV {out_fov:.0f} deg')
                best = None
                for flip in (False, True):
                    test = warped.transpose(Image.FLIP_LEFT_RIGHT) if flip else warped
                    res = t3.solve_from_image(
                        test, fov_estimate=out_fov, fov_max_error=12.0,
                        projection='equidistant', distortion=None, return_matches=True,
                        pattern_checking_stars=12, match_radius=0.015, **extract)
                    if res and res.get('RA') is not None:
                        prob = float(res.get('Prob', 0.0))
                        if best is None or prob < best[0]:
                            if flip:
                                to_src = (lambda x, y, fw=from_warp, W=out_size:
                                          fw((W - 1.0) - x, y))
                            else:
                                to_src = from_warp
                            best = (prob, res, to_src, (out_size, out_size), 'warp',
                                    'equidistant')
                if best is not None:
                    warp_candidates.append(best)
        candidates += warp_candidates

    # Last resort for trailed or noisy fields (long-exposure DSLR stills):
    # stricter extraction plus merging of close duplicate centroids (trail
    # endpoints) before solving the cleaned point set.
    if not candidates or min(candidates, key=lambda x: x[0])[0] >= 1e-12:
        dedupe_extract = {'sigma': 5, 'filtsize': 25, 'max_area': 300,
                          'min_area': 4, 'max_returned': 400}
        rect_extra = [(min(w, int(round(w * f))), min(h, int(round(w * f))),
                       'rectilinear', v, w / 2, h / 2)
                      for f, v in ((0.5, 15.0), (0.7, 20.0), (0.9, 25.0))]
        candidates += _try_solve_crops(
            t3, image, central + off_center[:8] + rect_extra,
            dedupe_extract, verbose, dedupe_sep=16)
    if not candidates:
        return None, None, None, None
    best = min(candidates, key=lambda x: x[0])
    _report_solve(best, verbose)
    return best[1], _seed_pairs(best), *_estimate_full_fov(best, w)


def _estimate_full_fov(best, image_w):
    """Full-frame FOV and projection implied by the winning attempt. A raw
    crop of width cw covers res['FOV'] degrees, so the frame is about
    FOV * w / cw. Warped views already render the whole frame (fixed
    equidistant model)."""
    _prob, res, _to_src, size, tag = best[:5]
    if tag == 'warp' or not res.get('FOV'):
        return INITIAL_FOV, 'equidistant'
    return res['FOV'] * image_w / size[0], best[5]


def _seed_pairs(best):
    """Convert the winning solve's matched stars to (x, y, ra, dec, cat_id)
    in full-image coordinates via the candidate's back-map."""
    _prob, res, to_src, _size, _tag = best[:5]
    seeds = []
    cat_ids = res.get('matched_catID') or [None] * len(res['matched_centroids'])
    for (cy_c, cx_c), (sra, sdec, _mag), cat_id in zip(
            res['matched_centroids'], res['matched_stars'], cat_ids):
        x, y = to_src(float(cx_c), float(cy_c))
        seeds.append((x, y, sra, sdec,
                      int(cat_id) if cat_id is not None else None))
    return seeds


def _dedupe_centroids(centroids, min_sep):
    """Merge centroids closer than min_sep px into their midpoint.

    Long-exposure star trails get one blob per trail endpoint; the phantom
    duplicates corrupt quad patterns. Clustering close centroids restores a
    single point per trail. Brightness order (brightest first) is kept."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    a = np.asarray(centroids, dtype=float)
    if len(a) < 2:
        return a
    pairs = cKDTree(a).query_pairs(min_sep)
    n = len(a)
    ii = [p[0] for p in pairs] + [p[1] for p in pairs]
    jj = [p[1] for p in pairs] + [p[0] for p in pairs]
    graph = csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    groups = [np.flatnonzero(labels == k) for k in range(labels.max() + 1)]
    groups.sort(key=lambda ix: ix.min())
    return np.array([a[ix].mean(axis=0) for ix in groups])


def _try_solve_crops(t3, image, attempts, extract, verbose=False, dedupe_sep=None):
    """Solve each (crop_w, crop_h, projection, fov, cx, cy) attempt; return
    (prob, res, to_src, size, 'crop') candidates."""
    w, h = image.size
    candidates = []
    for crop_w, crop_h, projection, fov_estimate, cx, cy in attempts:
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        if projection == 'equidistant':
            fov_estimate = INITIAL_FOV * crop_w / w
        left = int(min(max(cx - crop_w / 2, 0), w - crop_w))
        top = int(min(max(cy - crop_h / 2, 0), h - crop_h))
        crop = image.crop((left, top, left + crop_w, top + crop_h))
        if verbose:
            tag = 'dedupe ' if dedupe_sep else ''
            print(f'{tag}{projection} crop: {crop_w}x{crop_h} at ({left},{top}), '
                  f'estimated FOV {fov_estimate:.1f} deg')
        best = None
        for flip in (False, True):
            test = crop.transpose(Image.FLIP_LEFT_RIGHT) if flip else crop
            if dedupe_sep:
                cents = _dedupe_centroids(
                    get_centroids_from_image(test, **extract), dedupe_sep)
                res = t3.solve_from_centroids(
                    cents, size=(crop_h, crop_w), fov_estimate=fov_estimate,
                    fov_max_error=12.0, projection=projection, distortion=None,
                    return_matches=True, pattern_checking_stars=12,
                    match_radius=0.03, solve_timeout=45000)
            else:
                res = t3.solve_from_image(
                    test, fov_estimate=fov_estimate, fov_max_error=12.0,
                    projection=projection, distortion=None, return_matches=True,
                    pattern_checking_stars=12, match_radius=0.015, **extract)
            if res and res.get('RA') is not None:
                # 'Prob' is the false-positive probability: smaller is better.
                prob = float(res.get('Prob', 0.0))
                if best is None or prob < best[0]:
                    if flip:
                        to_src = (lambda x, y, l=left, t=top, cw=crop_w:
                                  ((cw - 1.0) - x + l, y + t))
                    else:
                        to_src = (lambda x, y, l=left, t=top: (x + l, y + t))
                    best = (prob, res, to_src, (crop_w, crop_h), 'crop', projection)
        if best is not None:
            candidates.append(best)
    return candidates


def _report_solve(best, verbose):
    if verbose:
        p = best[0]
        if p > 0:
            one_in = 1.0 / p
            chance = f'1 in {one_in:.2e}' if one_in >= 1e6 else f'1 in {one_in:,.0f}'
            print(f'Selected {best[4]} attempt: {chance} false-positive chance ({p * 100:.4g}%)')
        else:
            print(f'Selected {best[4]} attempt (probability unknown)')


def _radec_to_azel(ra_deg, dec_deg, observer):
    """Convert J2000 RA/Dec to observed azimuth/altitude."""
    b = ephem.FixedBody()
    b._ra = math.radians(ra_deg)
    b._dec = math.radians(dec_deg)
    b._epoch = ephem.J2000
    b.compute(observer)
    return math.degrees(b.az), math.degrees(b.alt)


def _collect_control_points(t3, full_image, observer, initial_pto_data,
                            seed_pairs, tolerance=0.15, verbose=False):
    """Refine the camera pose on the tetra3 seed matches, then match stars
    across the whole field."""
    from scipy.spatial import cKDTree

    w, h = full_image.size

    # 1. Seed control points from the winning tetra3 solve.
    seed_points = {}
    for x, y, sra, sdec, cat_id in seed_pairs:
        az, alt = _radec_to_azel(sra, sdec, observer)
        if alt <= 0:
            continue
        seed_points[(round(float(x), 4), round(float(y), 4))] = (x, y, az, alt, cat_id)

    if not seed_points:
        return [], initial_pto_data, None

    # 2. Refine yaw/pitch/roll/FOV on the seed points, keeping lens distortion fixed.
    refined_pto_data, _, seed_rmse, ok = _optimise_pto(
        initial_pto_data, list(seed_points.values()), ('v', 'y', 'p', 'r'))
    if verbose:
        status = 'complete' if ok else 'did not converge'
        print(f'  Seed refinement {status}: {seed_rmse:.3f} px RMSE')

    # 3. Match every detected star to the tetra3 catalogue via the refined model.
    if verbose:
        print(f'  Refining on {len(seed_points)} central stars; '
              f'matching full field with tolerance {tolerance} deg...')

    centroids = get_centroids_from_image(
        full_image, sigma=1.0, image_th=None, crop=None, downsample=None,
        filtsize=15, max_area=500, min_area=3, max_returned=2000,
    )

    star_table = t3.star_table
    catalogue_ids = t3.star_catalog_IDs
    if star_table is None:
        print('Warning: tetra3 database star table not available.')
        return list(seed_points.values()), refined_pto_data, seed_rmse

    tree = cKDTree(star_table[:, 2:5].astype(np.float64))
    tol_rad = math.radians(tolerance)

    control_points = dict(seed_points)
    for pt in centroids:
        y, x = float(pt[0]), float(pt[1])
        mapped = pto_mapper.map_image_to_pano(refined_pto_data, 0, x, y)
        if mapped is None:
            continue
        az, alt = mapped[0] / 100.0, 90.0 - mapped[1] / 100.0
        if alt <= 0:
            continue
        ra, dec = observer.radec_of(math.radians(az), math.radians(alt))
        vec = np.array([math.cos(dec) * math.cos(ra), math.cos(dec) * math.sin(ra), math.sin(dec)])
        dist, idx = tree.query(vec, k=1)
        if 2.0 * math.asin(min(1.0, float(dist) / 2.0)) > tol_rad:
            continue
        az, alt = _radec_to_azel(ra, dec, observer)
        cat_id = int(catalogue_ids[idx]) if catalogue_ids is not None else None
        key = (round(float(x), 4), round(float(y), 4))
        if key not in control_points:
            control_points[key] = (x, y, az, alt, cat_id)

    if verbose:
        print(f'Collected {len(control_points)} unique control points '
              f'({len(seed_points)} from central crop).')
    return list(control_points.values()), refined_pto_data, seed_rmse


def _project_catalog(pto_data, observer, star_table, objects=500, image_idx=0):
    """Project catalogue stars onto the image using the current PTO model."""
    expected = []
    # star_table columns: ra, dec, vx, vy, vz, mag
    mags = star_table[:, 5]
    indices = np.argsort(mags)[:objects]
    for i in indices:
        ra, dec, *_ = star_table[i]
        az, alt = _radec_to_azel(math.degrees(ra), math.degrees(dec), observer)
        if alt <= 0:
            continue
        res = pto_mapper.map_pano_to_image(pto_data, az * 100, (90 - alt) * 100)
        if res and res[0] == image_idx:
            expected.append((res[1], res[2], az, alt))
    return expected


def _create_star_mask(width, height, positions, radius_px, blur_px=None):
    """Create a black mask with white circles around expected star positions."""
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for x, y, *_ in positions:
        draw.ellipse([x - radius_px, y - radius_px, x + radius_px, y + radius_px], fill=255)
    if blur_px:
        mask = mask.filter(ImageFilter.GaussianBlur(blur_px))
    return mask


def _extract_masked_centroids(masked_image, sigma=1.0):
    """Run centroid detection on the masked image and return (x, y) tuples."""
    pts = get_centroids_from_image(
        masked_image, sigma=sigma, image_th=None, crop=None, downsample=None,
        filtsize=15, max_area=500, min_area=3, max_returned=2000,
    )
    return [(float(pt[1]), float(pt[0])) for pt in pts]


def _match_to_expected(expected, found, radius_px):
    """Match found centroids to expected positions within a pixel radius."""
    from scipy.spatial import cKDTree
    if not found:
        return []
    tree = cKDTree(found)
    matches = []
    for x_exp, y_exp, az, alt in expected:
        dist, idx = tree.query((x_exp, y_exp), k=1)
        if dist <= radius_px:
            fx, fy = found[idx]
            matches.append((fx, fy, az, alt))
    return matches


def _refine_calibration(pto_data, full_image, observer, star_table, iterations=3,
                        radius_deg=1.0, objects=500, verbose=False):
    """
    Iteratively refine the calibration by masking the image to expected star
    positions, extracting the actual star centroids, and reoptimising all lens
    and orientation parameters. Return the iteration with the lowest RMSE,
    plus the RMSE and matched-candidate count of that iteration for use as
    solve-confidence statistics.
    """
    w, h = full_image.size
    current = pto_data
    best_data = pto_data
    best_control_points = []
    best_rmse = float('inf')
    best_match_count = 0
    best_iteration = None
    solution_history = []
    matches = []
    for i in range(iterations):
        expected = _project_catalog(current, observer, star_table, objects=objects)
        if len(expected) < 10:
            if verbose:
                print(f'  Refine iter {i + 1}: only {len(expected)} expected stars, stopping.')
            break
        fov = float(current[1][0]['v'])
        pixel_radius = radius_deg * w / fov
        mask = _create_star_mask(w, h, expected, pixel_radius, blur_px=pixel_radius * 0.5)
        masked = ImageChops.multiply(full_image, mask)
        found = _extract_masked_centroids(masked, sigma=1.0)
        matches = _match_to_expected(expected, found, pixel_radius)
        if len(matches) < 15:
            if verbose:
                print(f'  Refine iter {i + 1}: only {len(matches)} remapped stars, stopping.')
            break

        cps = [(x, y, az, alt, None) for x, y, az, alt in matches]
        current, control_points, rmse, ok = _optimise_pto(
            current, cps, ('v', 'y', 'p', 'r', 'a', 'b', 'c', 'd', 'e'))
        if verbose:
            print(f'  Refine iter {i + 1}: {len(control_points)}/{len(matches)} inliers, '
                  f'{rmse:.3f} px RMSE.')
        if ok and rmse < best_rmse:
            best_data = copy.deepcopy(current)
            best_control_points = list(control_points)
            best_rmse = rmse
            best_match_count = len(matches)
            best_iteration = i + 1
        if not ok:
            break
        image = current[1][0]
        solution = tuple(float(image[name]) for name in ('v', 'y', 'p', 'r', 'a', 'b', 'c', 'd', 'e'))
        repeated_iteration = next((
            iteration for iteration, previous in enumerate(solution_history, start=1)
            if np.allclose(solution, previous, rtol=1e-10, atol=1e-12)
        ), None)
        if repeated_iteration is not None:
            if verbose:
                cycle_length = i + 1 - repeated_iteration
                print(f'  Refine iter {i + 1}: repeats iter {repeated_iteration} '
                      f'(cycle length {cycle_length}), stopping.')
            break
        solution_history.append(solution)
    if verbose and best_iteration is not None:
        print(f'  Selected refine iter {best_iteration}: {best_rmse:.3f} px RMSE.')
    return best_data, best_control_points, best_rmse, best_match_count


def _build_pto(image_path, width, height, fov, yaw, pitch, roll,
               control_points=(), dummy_path=None, var_lines='', projection=3,
               a=INITIAL_A, b=INITIAL_B, c=INITIAL_C, d=INITIAL_D, e=INITIAL_E):
    """Build a Hugin .pto string. If dummy_path is given, add the dummy equirect image and variables."""
    img_line = (f'i w{width} h{height} f{projection} v{fov} y{yaw} p{pitch} r{roll} '
                f'a{a} b{b} c{c} d{d} e{e} g0 t0 n"{os.path.basename(image_path)}" '
                f'Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 Vm5')
    lines = [
        '# Hugin project file created by autocalib.py',
        'p f2 w36000 h18000 v360 E0 R0 n"TIFF_m c:LZW" k0',
    ]
    if dummy_path is not None:
        lines.append('m g1 i0 m2 p0.00784314')
    lines.append(img_line)
    if dummy_path is not None:
        lines.append(
            f'i w36000 h18000 f4 v360 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p0 y0 TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 '
            f'a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 Vm5 n"{dummy_path}"')
    if var_lines:
        lines.append('')
        lines.append(var_lines)
    if control_points:
        lines.append('')
        for x, y, az, alt, _ in control_points:
            lines.append(f'c n0 N1 x{x:.4f} y{y:.4f} X{az * 100:.4f} Y{(90 - alt) * 100:.4f} t0')
    return '\n'.join(lines) + '\n'


def _annotate_control_points(pto_text, control_points):
    """Append catalogue IDs as inline comments to Hugin control-point lines."""
    cp_map = {(round(float(x), 4), round(float(y), 4)): cat_id
              for x, y, _az, _alt, cat_id in control_points if cat_id is not None}

    def repl(m):
        x = round(float(m.group(1)), 4)
        y = round(float(m.group(2)), 4)
        cat_id = cp_map.get((x, y))
        return m.group(0) + (f' # cat_id={int(cat_id)}' if cat_id is not None else '')

    return re.sub(r'^c n0 N1 x([0-9.eE+-]+) y([0-9.eE+-]+).*', repl, pto_text, flags=re.MULTILINE)


def _build_optimisation_pto_from_data(pto_data, control_points, dummy_path,
                                     var_lines='v v0\nv y0\nv p0\nv r0\nv a0\nv b0\nv c0\nv d0\nv e0\nv\n'):
    """Build an optimisation PTO re-using the image parameters from parsed pto_data."""
    _, images = pto_data
    img = images[0]
    image_name = img['n'].strip('"')
    width = int(img['w'])
    height = int(img['h'])
    fov = float(img['v'])
    yaw = float(img['y'])
    pitch = float(img['p'])
    roll = float(img['r'])
    a = float(img.get('a', INITIAL_A))
    b = float(img.get('b', INITIAL_B))
    c = float(img.get('c', INITIAL_C))
    d = float(img.get('d', INITIAL_D))
    e = float(img.get('e', INITIAL_E))
    projection = int(img.get('f', 3))
    return _build_pto(
        image_name, width, height, fov, yaw, pitch, roll,
        control_points=control_points, dummy_path=dummy_path,
        a=a, b=b, c=c, d=d, e=e, var_lines=var_lines, projection=projection,
    )


def _optimise_pto(pto_data, control_points, parameters):
    """Robustly fit PTO image parameters directly to astrometric control points."""
    from scipy.optimize import least_squares

    result_data = copy.deepcopy(pto_data)
    image = result_data[1][0]
    projection = int(image.get('f', 3))
    parameter_order = ('v', 'y', 'p', 'r', 'a', 'b', 'c', 'd', 'e')
    base_params = np.array([float(image[name]) for name in parameter_order])
    parameter_indices = np.array([parameter_order.index(name) for name in parameters])
    x0 = base_params[parameter_indices]
    scales = {'v': 10, 'y': 10, 'p': 10, 'r': 10,
              'a': 0.01, 'b': 0.01, 'c': 0.01, 'd': 50, 'e': 50}
    limits = {'v': (5, 150), 'y': (-720, 720), 'p': (-90, 90), 'r': (-180, 180),
              'a': (-0.5, 0.5), 'b': (-0.5, 0.5), 'c': (-0.5, 0.5),
              'd': (-500, 500), 'e': (-500, 500)}

    def residual(values, points):
        params = base_params.copy()
        params[parameter_indices] = values
        observed = np.asarray([(point[0], point[1]) for point in points],
                              dtype=float).reshape(-1, 2)
        azimuths = np.asarray([point[2] for point in points])
        altitudes = np.asarray([point[3] for point in points])
        predicted = _project_stars_numba(
            azimuths, altitudes, params, float(image['w']), float(image['h']), projection)
        return (predicted - observed).ravel()

    def jacobian(values, points):
        steps = {'v': 1e-3, 'y': 1e-3, 'p': 1e-3, 'r': 1e-3,
                 'a': 1e-5, 'b': 1e-5, 'c': 1e-5, 'd': 1e-3, 'e': 1e-3}
        jac = np.empty((len(points) * 2, len(parameters)))
        for column, name in enumerate(parameters):
            step = steps[name]
            high, low = values.copy(), values.copy()
            high[column] += step
            low[column] -= step
            jac[:, column] = (residual(high, points) - residual(low, points)) / (2 * step)
        residual(values, points)
        return jac

    lower = np.array([limits[name][0] for name in parameters])
    upper = np.array([limits[name][1] for name in parameters])
    x0 = np.clip(x0, lower + 1e-6, upper - 1e-6)
    if len(control_points) <= len(parameters):
        return result_data, [], float('inf'), False
    fit = least_squares(residual, x0, jac=jacobian, args=(control_points,), bounds=(lower, upper),
                        x_scale=[scales[name] for name in parameters], loss='soft_l1',
                        f_scale=1.0, max_nfev=1000)
    errors = residual(fit.x, control_points).reshape(-1, 2)
    distances = np.linalg.norm(errors, axis=1)
    median = np.median(distances)
    mad = np.median(np.abs(distances - median))
    cutoff = max(2.0, median + 4 * max(mad, 0.1))
    inliers = [point for point, distance in zip(control_points, distances) if distance <= cutoff]
    if len(inliers) >= max(8, len(parameters)):
        fit = least_squares(residual, fit.x, jac=jacobian, args=(inliers,), bounds=(lower, upper),
                            x_scale=[scales[name] for name in parameters], loss='linear',
                            max_nfev=1000)
    for name, value in zip(parameters, fit.x):
        image[name] = float(value)
    final_errors = residual(fit.x, inliers).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(final_errors**2, axis=1))))
    return result_data, inliers, rmse, fit.success


def _build_and_evaluate(args, result, seed_pairs, t3, full_image, observer,
                        width, height, mask_img, est_fov=None, est_proj=None):
    """Build the PTO model from a solve, collect/refine control points and
    evaluate the confidence gate. Returns (refined_pto_data, control_points,
    stats)."""
    # RA/Dec at the IMAGE centre. tetra3 reports the centre of the solved
    # (possibly off-centre or reprojected) patch; fit a local affine map
    # px->RA/Dec from the matched stars and evaluate at the image centre.
    ra_deg = result['RA']
    dec_deg = result['Dec']
    if len(seed_pairs) >= 4:
        fx = np.asarray([s[0] for s in seed_pairs])
        fy = np.asarray([s[1] for s in seed_pairs])
        ras = np.unwrap(np.radians([s[2] for s in seed_pairs]))
        decs = np.radians([s[3] for s in seed_pairs])
        A = np.column_stack([fx, fy, np.ones_like(fx)])
        ra_coef, *_ = np.linalg.lstsq(A, ras, rcond=None)
        dec_coef, *_ = np.linalg.lstsq(A, decs, rcond=None)
        ra_deg = math.degrees(ra_coef @ [width / 2, height / 2, 1.0]) % 360.0
        dec_deg = math.degrees(dec_coef @ [width / 2, height / 2, 1.0])

    # Convert centre to azimuth/altitude and then to Hugin yaw/pitch.
    body = ephem.FixedBody()
    body._ra = math.radians(ra_deg)
    body._dec = math.radians(dec_deg)
    body._epoch = ephem.J2000
    body.compute(observer)
    centre_az = math.degrees(body.az)
    centre_alt = math.degrees(body.alt)
    hugin_yaw = (centre_az - 180.0) % 360.0
    hugin_pitch = centre_alt
    hugin_roll = 0.0

    # Seed the model from the winning solve: its implied full-frame FOV and
    # projection. Fisheye seeds get the fleet-median lens params; rectilinear
    # seeds start undistorted. A model check below switches projection if the
    # other model fits the control points far better.
    seed_fov = est_fov or INITIAL_FOV
    pto_projection = 0 if (est_proj == 'rectilinear' or
                         (est_fov or 999) < 60) else 3
    force_rect = getattr(args, 'rectilinear', False)
    if force_rect and seed_fov >= 60:
        # Wide field: collect under fisheye first (converges reliably), the
        # projection switch below re-collects under f0.
        pto_projection = 3
    seed_lens = ((INITIAL_A, INITIAL_B, INITIAL_C, INITIAL_D, INITIAL_E)
                 if pto_projection == 3 else (0.0, 0.0, 0.0, 0.0, 0.0))
    if args.verbose:
        print(f'tetra3 centre: RA={ra_deg:.4f} Dec={dec_deg:.4f}')
        print(f'Image centre Az/Alt: {centre_az:.4f} / {centre_alt:.4f}')
        print(f'Hugin yaw/pitch/roll: {hugin_yaw:.4f} / {hugin_pitch:.4f} / {hugin_roll:.4f}')
        print(f'Full image FOV: {seed_fov:.2f} deg, projection f{pto_projection}')

    tmp_pto = tempfile.NamedTemporaryFile(mode='w', suffix='.pto', delete=False)
    tmp_pto_path = tmp_pto.name
    with tmp_pto as f:
        f.write(_build_pto(args.image, width, height, seed_fov,
                           hugin_yaw, hugin_pitch, hugin_roll,
                           projection=pto_projection,
                           a=seed_lens[0], b=seed_lens[1], c=seed_lens[2],
                           d=seed_lens[3], e=seed_lens[4]))
    pto_data = pto_mapper.parse_pto_file(tmp_pto_path)
    os.unlink(tmp_pto_path)

    # Collect control points by matching every detected star to the tetra3
    # catalogue via the initial camera model.
    if args.verbose:
        print('Collecting tetra3 control points...')
    control_points, refined_pto_data, seed_rmse = _collect_control_points(
        t3, full_image, observer, pto_data,
        seed_pairs,
        tolerance=args.match_tolerance,
        verbose=args.verbose,
    )

    # Model check: fit the same control points under the other projection —
    # a genuinely rectilinear camera fits f0 far better than f3 and vice
    # versa. The alternate model must win by a wide margin to switch.
    if len(control_points) >= 10 and not (force_rect and pto_projection == 0):
        alt_proj = 0 if pto_projection == 3 else 3
        alt_lens = ((0.0, 0.0, 0.0, 0.0, 0.0) if alt_proj == 0
                    else (INITIAL_A, INITIAL_B, INITIAL_C, INITIAL_D, INITIAL_E))
        alt_text = _build_pto(args.image, width, height, seed_fov,
                              hugin_yaw, hugin_pitch, hugin_roll,
                              projection=alt_proj, a=alt_lens[0], b=alt_lens[1],
                              c=alt_lens[2], d=alt_lens[3], e=alt_lens[4])
        tmp0 = tempfile.NamedTemporaryFile(mode='w', suffix='.pto', delete=False)
        with tmp0 as f:
            f.write(alt_text)
        pto0 = pto_mapper.parse_pto_file(tmp0.name)
        os.unlink(tmp0.name)
        pto0, _, rmse_alt, ok0 = _optimise_pto(
            pto0, control_points, ('v', 'y', 'p', 'r', 'a', 'b', 'c', 'd', 'e'))
        switch = force_rect and alt_proj == 0
        if ok0 and (switch or rmse_alt < 0.5 * seed_rmse):
            if args.verbose:
                print(f'f{alt_proj} model {"forced" if switch else "fits better"} '
                      f'({rmse_alt:.2f}px vs f{pto_projection} {seed_rmse:.2f}px); '
                      're-collecting.')
            pto_projection = alt_proj
            control_points, refined_pto_data, seed_rmse = _collect_control_points(
                t3, full_image, observer, pto0,
                seed_pairs,
                tolerance=args.match_tolerance,
                verbose=args.verbose,
            )

    # Iteratively refine by masking the image to expected star positions and
    # reoptimising all lens/orientation parameters.
    rmse_px = seed_rmse
    matched_candidates = len(control_points)
    collected = control_points
    if args.refine_iterations > 0:
        if args.verbose:
            print('Refining calibration with masked-star optimisation...')
        refined_pto_data, refine_cps, refine_rmse, refine_matched = _refine_calibration(
            refined_pto_data, full_image, observer, t3.star_table,
            iterations=args.refine_iterations,
            radius_deg=args.refine_radius,
            verbose=args.verbose,
        )
        if refine_rmse != float('inf') and refine_cps:
            control_points = refine_cps
            rmse_px = refine_rmse
            matched_candidates = refine_matched
    if not control_points and collected:
        # Refinement found nothing; inlier-filter the collected set instead.
        refined_pto_data, control_points, rmse_px, _ok = _optimise_pto(
            refined_pto_data, collected,
            ('v', 'y', 'p', 'r', 'a', 'b', 'c', 'd', 'e'))

    # --- Solve-confidence gate ---
    stats = evaluate_solve(result, rmse_px, control_points, matched_candidates,
                           width, height, max_prob=args.accept_prob,
                           min_matches=args.accept_matches,
                           max_rmse_px=args.accept_rmse,
                           min_inliers=args.accept_inliers,
                           min_inlier_frac=args.accept_inlier_frac,
                           min_coverage=args.accept_coverage, mask=mask_img)
    return refined_pto_data, control_points, stats


def _run_all_cameras(args):
    """--all mode: calibrate cam1..cam7 from the 23:00 still of the given
    date (default: yesterday, UTC), then rotate the lens.pto/grid.png links."""
    import subprocess
    from datetime import timedelta

    meteor = os.environ.get('NMN_METEOR_ROOT', '/meteor')
    if args.image:
        dstr = args.image.replace('-', '')
        try:
            datetime.strptime(dstr, '%Y%m%d')
        except ValueError:
            print(f'Error: invalid date: {args.image} (want YYYYMMDD or YYYY-MM-DD)',
                  file=sys.stderr)
            sys.exit(2)
    else:
        dstr = (datetime.now(timezone.utc).date() - timedelta(days=1)
                ).strftime('%Y%m%d')
    timestamp = int(datetime.strptime(dstr + '2300', '%Y%m%d%H%M')
                    .replace(tzinfo=timezone.utc).timestamp())
    drawgrid = Path(__file__).resolve().parent / 'drawgrid.py'
    config = args.config or '/etc/meteor.cfg'

    print(f'--all: calibrating cam1-7 for {dstr} (still 23/full_00.jpg), '
          f'meteor root {meteor}', flush=True)
    results = {}
    for cam in range(1, 8):
        camdir = Path(meteor) / f'cam{cam}'
        img = camdir / dstr / '23' / 'full_00.jpg'
        lens_dated = camdir / f'lens-{dstr}.pto'
        grid_dated = camdir / f'grid-{dstr}.png'
        print(f'cam{cam}: solving {img} ...', flush=True)
        if not img.is_file():
            results[cam] = ('skipped', 'no 23:00 still image')
            print(f'cam{cam}: skipped - no still image', flush=True)
            continue
        tmp = tempfile.NamedTemporaryFile(suffix='.pto', dir=camdir,
                                          delete=False).name
        cmd = [sys.executable, str(Path(__file__).resolve()), str(img), tmp,
               '-T', str(timestamp)]
        if os.path.isfile(config):
            cmd += ['-c', config]
        if args.verbose:
            cmd.append('-v')
        if args.automask:
            cmd.append('--automask')
        if args.nomask:
            cmd.append('--nomask')
        if args.rectilinear:
            cmd.append('--rectilinear')
        if args.force:
            cmd.append('--force')
        for name in ('accept_prob', 'accept_matches', 'accept_rmse',
                     'accept_inliers', 'accept_inlier_frac', 'accept_coverage',
                     'match_tolerance', 'refine_iterations', 'refine_radius'):
            cmd += ['--' + name.replace('_', '-'), str(getattr(args, name))]
        t0 = datetime.now()
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = (r.stderr.strip().splitlines() or [''])[-1]
        conf = ''
        for line in r.stdout.splitlines():
            if line.startswith('Confidence:'):
                conf = line.split(':', 1)[1].strip()
        ok = r.returncode == 0 and os.path.isfile(tmp)
        if not ok:
            results[cam] = ('failed', tail)
            print(f'cam{cam}: FAILED ({tail})', flush=True)
            continue
        # Orientation shift vs the existing calibration, if any.
        shift_msg = ''
        link = camdir / 'lens.pto'
        if link.exists():
            try:
                old = pto_mapper.parse_pto_file(str(link))[1][0]
                new = pto_mapper.parse_pto_file(tmp)[1][0]

                def _canon(y, p):
                    while p > 90: p = 180 - p; y += 180
                    while p < -90: p = -180 - p; y += 180
                    return y % 360, p
                oy, op = _canon(float(old['y']), float(old['p']))
                ny, np_ = _canon(float(new['y']), float(new['p']))
                dy = min(abs(ny - oy), 360 - abs(ny - oy))
                pointing = math.hypot(dy, np_ - op)
                dr = min(abs(float(new['r']) - float(old['r'])),
                         360 - abs(float(new['r']) - float(old['r'])))
                dv = abs(float(new['v']) - float(old['v']))
                shift_msg = (f' shift: {pointing:.2f}deg pointing, '
                             f'{dr:.2f}deg roll, {dv:.2f}deg fov')
            except Exception:
                shift_msg = ' shift: n/a (old pto unreadable)'
        if args.dryrun:
            os.unlink(tmp)
            results[cam] = ('ok', f'{conf}{shift_msg} [dry-run: not installed]')
            print(f'cam{cam}: OK - {conf}{shift_msg} [dry-run]', flush=True)
            continue
        os.replace(tmp, lens_dated)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(lens_dated.name)
        # grid overlay
        gcmd = [sys.executable, str(drawgrid), str(lens_dated), str(grid_dated),
                '-d', str(timestamp), '-p', '0']
        if os.path.isfile(config):
            gcmd += ['-c', config]
        g = subprocess.run(gcmd, capture_output=True, text=True)
        if g.returncode == 0 and grid_dated.is_file():
            glink = camdir / 'grid.png'
            if glink.exists() or glink.is_symlink():
                glink.unlink()
            glink.symlink_to(grid_dated.name)
            grid_msg = ''
        else:
            grid_msg = f' (grid failed: {g.stderr.strip()[-120:]})'
        dt = (datetime.now() - t0).seconds
        results[cam] = ('ok', f'{conf}{shift_msg}{grid_msg}')
        print(f'cam{cam}: OK in {dt}s - {conf}{shift_msg}{grid_msg}', flush=True)

    print('\n=== Summary ===')
    n_ok = sum(1 for s, _ in results.values() if s == 'ok')
    for cam, (state, detail) in results.items():
        print(f'cam{cam}: {state:7} {detail}')
    print(f'{n_ok}/7 cameras calibrated for {dstr}')
    sys.exit(0 if n_ok == 7 else 1)


def main():
    parser = argparse.ArgumentParser(
        description='Create a Hugin .pto file from a star-field image using tetra3.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('image', nargs='?',
                        help='Input image (e.g. JPEG). With --all: optional '
                             'date YYYYMMDD or YYYY-MM-DD (default: yesterday).')
    parser.add_argument('ptofile', nargs='?', help='Output .pto file')
    parser.add_argument('overlay', nargs='?', default=None,
                        help='Optional output image: draw az/alt grid + star '
                             'annotations over the input image (via drawgrid.py).')
    parser.add_argument('--all', action='store_true',
                        help='Calibrate all cameras: solve /meteor/camN/<date>/23/'
                             'full_00.jpg for cam1-7, install lens-YYYYMMDD.pto + '
                             'grid-YYYYMMDD.png and re-point the lens.pto/grid.png '
                             'links. Takes a date argument instead of image/ptofile.')
    parser.add_argument('--dryrun', action='store_true',
                        help='With --all: solve but do not install the dated '
                             'lens/grid files or change the links.')
    parser.add_argument('-c', '--config', help='Meteor config file (default: /etc/meteor.cfg)')
    parser.add_argument('-y', '--latitude', type=float, help='Observer latitude')
    parser.add_argument('-x', '--longitude', type=float, help='Observer longitude')
    parser.add_argument('-e', '--elevation', type=float, help='Observer elevation (m)')
    parser.add_argument('-T', '--timestamp', type=float, help='Unix timestamp of the image')
    parser.add_argument('--match-tolerance', type=float, default=0.15,
                        help='Maximum angular distance (deg) for accepting a catalogue match '
                             'in full-field matching (default: 0.15).')
    parser.add_argument('--refine-iterations', type=int, default=3,
                        help='Number of masked-star refinement iterations after the initial '
                             'full-field match (default: 3). Set to 0 to skip refinement.')
    parser.add_argument('--refine-radius', type=float, default=1.0,
                        help='Search radius in degrees for the masked-star refinement '
                             '(default: 1.0).')
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument('--mask', metavar='FILE',
                            help='Use this foreground mask instead of /meteor/camN/mask.png.')
    mask_group.add_argument('--automask', action='store_true',
                            help='Ignore any mask file and derive a foreground mask '
                                 'from the image itself.')
    mask_group.add_argument('--nomask', action='store_true',
                            help='Do not load or apply a foreground mask.')
    parser.add_argument('--rectilinear', action='store_true',
                        help='Force a rectilinear (f0) lens model instead of '
                             'auto-detecting the projection.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    gate = parser.add_argument_group('confidence gate',
                                     'Reject the solve (exit 1, no .pto) unless all metrics pass.')
    gate.add_argument('--accept-prob', type=float, default=ACCEPT_MAX_PROB,
                      help=f'Max tetra3 false-positive probability (default: {ACCEPT_MAX_PROB:g}).')
    gate.add_argument('--accept-matches', type=int, default=ACCEPT_MIN_MATCHES,
                      help=f'Min stars matched in central solve (default: {ACCEPT_MIN_MATCHES}).')
    gate.add_argument('--accept-rmse', type=float, default=ACCEPT_MAX_RMSE_PX,
                      help=f'Max final pixel RMSE of inlier control points (default: {ACCEPT_MAX_RMSE_PX}).')
    gate.add_argument('--accept-inliers', type=int, default=ACCEPT_MIN_INLIERS,
                      help=f'Min inlier control points (default: {ACCEPT_MIN_INLIERS}).')
    gate.add_argument('--accept-inlier-frac', type=float, default=ACCEPT_MIN_INLIER_FRAC,
                      help=f'Min inlier/matched fraction (default: {ACCEPT_MIN_INLIER_FRAC}).')
    gate.add_argument('--accept-coverage', type=float, default=ACCEPT_MIN_COVERAGE,
                      help=f'Min sky coverage fraction on a 4x4 grid (default: {ACCEPT_MIN_COVERAGE}).')
    parser.add_argument('--force', action='store_true',
                        help='Write the .pto even when the solve fails the confidence gate.')
    args = parser.parse_args()

    if args.all:
        if args.ptofile or args.overlay:
            parser.error('--all takes an optional date argument, not image/ptofile')
        _run_all_cameras(args)
        return

    if not args.image or not args.ptofile:
        parser.error('image and ptofile are required (unless --all is used)')

    if Tetra3 is None:
        print(f'Error: could not import local tetra3 solver: {TETRA3_ERR}', file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.image):
        print(f'Error: image not found: {args.image}', file=sys.stderr)
        sys.exit(1)

    config = _load_config(args)
    observer, timestamp = _setup_observer(args, config)

    # Open image and convert to luminance for solving.
    raw_image = Image.open(args.image).convert('L')
    full_image = raw_image
    mask_img = None
    mask_src = 'none'
    if args.nomask:
        if args.verbose:
            print('Camera mask disabled by --nomask.')
    elif args.automask:
        fg = _auto_mask(raw_image)
        mask_img = Image.fromarray(np.where(fg, 255, 0).astype(np.uint8))
        full_image = Image.composite(
            raw_image, raw_image.filter(ImageFilter.GaussianBlur(25)),
            ImageChops.invert(mask_img))
        mask_src = 'auto'
        print(f'Using self-derived foreground mask ({100.0 * fg.mean():.0f}% '
              'of frame masked).', file=sys.stderr)
    else:
        full_image, mask_img, mask_src = _apply_camera_mask(
            raw_image, args.image, mask_path=args.mask, verbose=args.verbose)
    width, height = raw_image.size

    if args.verbose:
        print(f'Image: {args.image} ({width}x{height})')
        print(f'Observer: lat={observer.lat} lon={observer.lon} elev={observer.elevation}')
        print(f'Timestamp (UTC): {observer.date}')

    # Use the local tetra3 solver; its pattern database is built from stars.py.
    t3 = Tetra3()
    stats = None
    result = None
    for attempt in range(2):
        result, seed_pairs, est_fov, est_proj = _solve_image(
            t3, full_image, verbose=args.verbose, mask=mask_img)
        if result is not None:
            refined_pto_data, control_points, stats = _build_and_evaluate(
                args, result, seed_pairs, t3, full_image,
                observer, width, height, mask_img,
                est_fov=est_fov, est_proj=est_proj)
        if result is not None and stats is not None and not stats['failures']:
            break
        if attempt == 0 and mask_src == 'file':
            why = 'no solution' if result is None else 'low confidence'
            print(f'First pass: {why} with camera mask; retrying with '
                  'self-derived foreground mask (mask may be stale)...',
                  file=sys.stderr)
            fg = _auto_mask(raw_image)
            mask_img = Image.fromarray(np.where(fg, 255, 0).astype(np.uint8))
            full_image = Image.composite(
                raw_image, raw_image.filter(ImageFilter.GaussianBlur(25)),
                ImageChops.invert(mask_img))
            mask_src = 'auto'
        else:
            break
    if result is None:
        print('Error: tetra3 could not solve the image (or a central crop).', file=sys.stderr)
        sys.exit(1)

    print('Confidence: ' + stats['summary'])
    if stats['failures'] and not args.force:
        print('Error: solve rejected as low confidence (clouds?): '
              + '; '.join(stats['failures']), file=sys.stderr)
        sys.exit(1)

    dummy_path = 'dummy_equirect.jpg'

    # Write a Hugin project with the camera image and a dummy equirect image,
    # linked by the tetra3 control points. This is the same structure used by
    # amscalib2lens.py, and lets autooptimiser adjust the lens model.
    pto_text = _build_optimisation_pto_from_data(
        refined_pto_data, control_points, dummy_path,
    )

    with open(args.ptofile, 'w') as f:
        f.write(_annotate_control_points(pto_text, control_points))
    print(f'Wrote .pto with {len(control_points)} control points: {args.ptofile}')

    if args.overlay:
        _write_grid_overlay(args, timestamp, observer)


def _write_grid_overlay(args, timestamp, observer):
    """Draw the az/alt grid and star annotations over the input image using
    drawgrid.py and save the composite to args.overlay."""
    import subprocess
    drawgrid = Path(__file__).resolve().parent / 'drawgrid.py'
    grid_png = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    cmd = [sys.executable, str(drawgrid), args.ptofile, grid_png,
           '-d', str(int(timestamp)), '-p', '0',
           '-Y', str(math.degrees(float(observer.lat))),
           '-X', str(math.degrees(float(observer.lon))),
           '-e', str(float(observer.elevation))]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(grid_png):
        print(f'Warning: drawgrid.py failed ({r.stderr.strip()[-200:]}); '
              'overlay not written.', file=sys.stderr)
        return
    base = Image.open(args.image).convert('RGBA')
    grid = Image.open(grid_png)
    if grid.size != base.size:
        grid = grid.resize(base.size)
    alpha = grid.getchannel('A').point(lambda a: int(a * 0.55))
    grid.putalpha(alpha)
    out = Image.alpha_composite(base, grid).convert('RGB')
    out.save(args.overlay)
    os.unlink(grid_png)
    print(f'Wrote annotated overlay: {args.overlay}')


if __name__ == '__main__':
    main()
