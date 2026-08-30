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


def _apply_camera_mask(image, image_path, mask_path=None, verbose=False):
    """Remove foreground where the AMS camera mask is white."""
    mask_path = mask_path or _camera_mask_path(image_path)
    if not mask_path:
        if verbose:
            print('No camera mask path inferred from input filename.')
        return image
    if not os.path.isfile(mask_path):
        if verbose:
            print(f'Camera mask not found: {mask_path}')
        return image
    mask = Image.open(mask_path).convert('L')
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    if verbose:
        print(f'Applying foreground mask (white pixels excluded): {mask_path}')
    keep_mask = ImageChops.invert(mask)
    smooth_foreground = image.filter(ImageFilter.GaussianBlur(25))
    return Image.composite(image, smooth_foreground, keep_mask)


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


def _solve_image(t3, image, verbose=False):
    """Solve the central sky; try several fisheye and rectilinear crops and pick the best."""
    w, h = image.size
    extract = {'sigma': 3, 'filtsize': 15, 'max_area': 500, 'min_area': 3, 'max_returned': 100}
    attempts = []
    for fov in (60, 50, 40, 30, 25, 20):
        cw = min(w, int(round(w * fov / INITIAL_FOV)))
        attempts.append((cw, cw, 'equidistant', float(fov)))
    for size, fov in ((int(round(w * 0.3)), 25.0),
                      (640, 20.0), (768, 30.0), (480, 15.0)):
        attempts.append((size, size, 'rectilinear', fov))
    candidates = []
    for crop_w, crop_h, projection, fov_estimate in attempts:
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        if projection == 'equidistant':
            fov_estimate = INITIAL_FOV * crop_w / w
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        crop = image.crop((left, top, left + crop_w, top + crop_h))
        if verbose:
            print(f'Central {projection} crop: {crop_w}x{crop_h} at ({left},{top}), '
                  f'estimated FOV {fov_estimate:.1f} deg')
        best = None
        for flip in (False, True):
            test = crop.transpose(Image.FLIP_LEFT_RIGHT) if flip else crop
            res = t3.solve_from_image(
                test, fov_estimate=fov_estimate, fov_max_error=12.0,
                projection=projection, distortion=None, return_matches=True,
                pattern_checking_stars=12, match_radius=0.015, **extract)
            if res and res.get('RA') is not None:
                prob = float(res.get('Prob', 0.0))
                if best is None or prob > best[0]:
                    best = (prob, res, (left, top, crop_w, crop_h), flip, projection)
        if best is not None:
            candidates.append(best)
    if not candidates:
        return None, None, False, None
    best = max(candidates, key=lambda x: x[0])
    if verbose:
        p = best[0]
        if p > 0:
            one_in = 1.0 / p
            chance = f'1 in {one_in:.2e}' if one_in >= 1e6 else f'1 in {one_in:,.0f}'
            print(f'Selected {best[4]} crop: {chance} false-positive chance ({p * 100:.4g}%)')
        else:
            print(f'Selected {best[4]} crop (probability unknown)')
    return best[1], best[2], best[3], best[4]


def _radec_to_azel(ra_deg, dec_deg, observer):
    """Convert J2000 RA/Dec to observed azimuth/altitude."""
    b = ephem.FixedBody()
    b._ra = math.radians(ra_deg)
    b._dec = math.radians(dec_deg)
    b._epoch = ephem.J2000
    b.compute(observer)
    return math.degrees(b.az), math.degrees(b.alt)


def _collect_control_points(t3, full_image, observer, initial_pto_data,
                            central_result, central_crop_box, central_flipped,
                            tolerance=0.15, verbose=False):
    """Refine the camera pose on the central crop, then match stars across the whole field."""
    from scipy.spatial import cKDTree

    w, h = full_image.size
    cleft, ctop = central_crop_box[:2]

    # 1. Seed control points from the central tetra3 solve.
    seed_points = {}
    cat_ids = central_result.get('matched_catID') or [None] * len(central_result['matched_centroids'])
    for (cy_c, cx_c), (sra, sdec, _), cat_id in zip(
            central_result['matched_centroids'], central_result['matched_stars'], cat_ids):
        x = cx_c + cleft
        y = cy_c + ctop
        if central_flipped:
            x = (w - 1.0) - x
        az, alt = _radec_to_azel(sra, sdec, observer)
        if alt <= 0:
            continue
        seed_points[(round(float(x), 4), round(float(y), 4))] = (
            x, y, az, alt, int(cat_id) if cat_id is not None else None)

    if not seed_points:
        return [], initial_pto_data

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
        return list(seed_points.values()), refined_pto_data

    tree = cKDTree(star_table[:, 2:5].astype(np.float64))
    tol_rad = math.radians(tolerance)

    control_points = dict(seed_points)
    for pt in centroids:
        y, x = float(pt[0]), float(pt[1])
        mapped = pto_mapper.map_image_to_pano(refined_pto_data, 0, (w - 1.0 - x) if central_flipped else x, y)
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
    return list(control_points.values()), refined_pto_data


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
    and orientation parameters. Return the iteration with the lowest RMSE.
    """
    w, h = full_image.size
    current = pto_data
    best_data = pto_data
    best_control_points = []
    best_rmse = float('inf')
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
    return best_data, best_control_points


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
    limits = {'v': (30, 150), 'y': (-720, 720), 'p': (-90, 90), 'r': (-180, 180),
              'a': (-0.5, 0.5), 'b': (-0.5, 0.5), 'c': (-0.5, 0.5),
              'd': (-500, 500), 'e': (-500, 500)}

    def residual(values, points):
        params = base_params.copy()
        params[parameter_indices] = values
        observed = np.asarray([(point[0], point[1]) for point in points])
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


def main():
    parser = argparse.ArgumentParser(
        description='Create a Hugin .pto file from a star-field image using tetra3.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('image', help='Input image (e.g. JPEG)')
    parser.add_argument('ptofile', help='Output .pto file')
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
    mask_group.add_argument('--nomask', action='store_true',
                            help='Do not load or apply a foreground mask.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    if Tetra3 is None:
        print(f'Error: could not import local tetra3 solver: {TETRA3_ERR}', file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.image):
        print(f'Error: image not found: {args.image}', file=sys.stderr)
        sys.exit(1)

    config = _load_config(args)
    observer, timestamp = _setup_observer(args, config)

    # Open image and convert to luminance for solving.
    full_image = Image.open(args.image).convert('L')
    if args.nomask:
        if args.verbose:
            print('Camera mask disabled by --nomask.')
    else:
        full_image = _apply_camera_mask(
            full_image, args.image, mask_path=args.mask, verbose=args.verbose)
    width, height = full_image.size

    if args.verbose:
        print(f'Image: {args.image} ({width}x{height})')
        print(f'Observer: lat={observer.lat} lon={observer.lon} elev={observer.elevation}')
        print(f'Timestamp (UTC): {observer.date}')

    # Use the local tetra3 solver; its pattern database is built from stars.py.
    t3 = Tetra3()
    result, crop_box, flipped, projection = _solve_image(t3, full_image, verbose=args.verbose)
    if result is None:
        print('Error: tetra3 could not solve the image (or a central crop).', file=sys.stderr)
        sys.exit(1)

    ra_deg = result['RA']
    dec_deg = result['Dec']

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

    crop_w = crop_box[2]
    solved_fov = float(result['FOV'])
    if projection == 'equidistant':
        full_fov = solved_fov * width / crop_w
        pto_projection = 3
    else:
        full_fov = math.degrees(2 * math.atan(
            (width * math.tan(math.radians(solved_fov) / 2.0)) / crop_w))
        pto_projection = 0

    if args.verbose:
        print(f'tetra3 centre: RA={ra_deg:.4f} Dec={dec_deg:.4f}')
        print(f'Image centre Az/Alt: {centre_az:.4f} / {centre_alt:.4f}')
        print(f'Hugin yaw/pitch/roll: {hugin_yaw:.4f} / {hugin_pitch:.4f} / {hugin_roll:.4f}')
        print(f'Full image FOV: {full_fov:.2f} deg, projection f{pto_projection}')

    # Build the initial PTO using the solved yaw/pitch/roll and a fixed set of
    # lens parameters taken from a known-good calibration. This gives a good
    # enough model to project the whole image and match stars across the field.
    if pto_projection == 3:
        init_a, init_b, init_c, init_d, init_e = INITIAL_A, INITIAL_B, INITIAL_C, INITIAL_D, INITIAL_E
    else:
        init_a = init_b = init_c = init_d = init_e = 0.0
    tmp_pto_path = tempfile.mktemp(suffix='.pto')
    with open(tmp_pto_path, 'w') as f:
        f.write(_build_pto(args.image, width, height, full_fov,
                           hugin_yaw, hugin_pitch, hugin_roll,
                           projection=pto_projection,
                           a=init_a, b=init_b, c=init_c, d=init_d, e=init_e))
    pto_data = pto_mapper.parse_pto_file(tmp_pto_path)
    os.unlink(tmp_pto_path)

    # Collect control points by matching every detected star to the tetra3
    # catalogue via the initial camera model.
    if args.verbose:
        print('Collecting tetra3 control points...')
    control_points, refined_pto_data = _collect_control_points(
        t3, full_image, observer, pto_data,
        central_result=result,
        central_crop_box=crop_box,
        central_flipped=flipped,
        tolerance=args.match_tolerance,
        verbose=args.verbose,
    )

    # Iteratively refine by masking the image to expected star positions and
    # reoptimising all lens/orientation parameters.
    if args.refine_iterations > 0:
        if args.verbose:
            print('Refining calibration with masked-star optimisation...')
        refined_pto_data, control_points = _refine_calibration(
            refined_pto_data, full_image, observer, t3.star_table,
            iterations=args.refine_iterations,
            radius_deg=args.refine_radius,
            verbose=args.verbose,
        )

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


if __name__ == '__main__':
    main()
