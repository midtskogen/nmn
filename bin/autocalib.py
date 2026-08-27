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
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
    """Solve the central crop with tetra3, trying the original and a flipped version."""
    w, h = image.size
    crop_w = int(round(w * 0.3))
    left = (w - crop_w) // 2
    top = (h - crop_w) // 2
    crop = image.crop((left, top, left + crop_w, top + crop_w))
    if verbose:
        print(f'Central crop: {crop_w}x{crop_w} at ({left},{top})')

    extract = {'sigma': 3, 'filtsize': 15, 'max_area': 500, 'min_area': 3, 'max_returned': 80}
    for flip in (False, True):
        test = crop.transpose(Image.FLIP_LEFT_RIGHT) if flip else crop
        res = t3.solve_from_image(test, fov_estimate=25.0, fov_max_error=15.0,
                                  distortion=0, return_matches=True, **extract)
        if res and res.get('RA') is not None:
            if verbose and flip:
                print('Solved using a horizontally flipped crop.')
            return res, (left, top, crop_w, crop_w), flip
    return None, None, False


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
    tmpdir = tempfile.mkdtemp(prefix='autocalib_')
    seed_pto = os.path.join(tmpdir, 'seed.pto')
    dummy_path = os.path.join(tmpdir, 'dummy_equirect.jpg')
    Image.fromarray(np.zeros((180, 360, 3), dtype=np.uint8)).save(dummy_path)
    with open(seed_pto, 'w') as f:
        f.write(_build_optimisation_pto_from_data(
            initial_pto_data, list(seed_points.values()), dummy_path,
            var_lines='v v0\nv y0\nv p0\nv r0\nv\n'))
    ok, output = _run_autooptimiser(seed_pto, seed_pto)
    refined_pto_data = pto_mapper.parse_pto_file(seed_pto) if ok else initial_pto_data
    if not ok and verbose:
        print(f'  Seed refinement failed: {output}')
    shutil.rmtree(tmpdir, ignore_errors=True)

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
    and orientation parameters with Hugin.
    """
    w, h = full_image.size
    current = pto_data
    all_vars = 'v v0\nv y0\nv p0\nv r0\nv a0\nv b0\nv c0\nv d0\nv e0\nv\n'
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

        tmpdir = tempfile.mkdtemp(prefix='tetra3_refine_')
        refine_pto = os.path.join(tmpdir, 'refine.pto')
        dummy_path = os.path.join(tmpdir, 'dummy.jpg')
        Image.fromarray(np.zeros((180, 360, 3), dtype=np.uint8)).save(dummy_path)
        cps = [(x, y, az, alt, None) for x, y, az, alt in matches]
        control_points = cps
        with open(refine_pto, 'w') as f:
            f.write(_build_optimisation_pto_from_data(current, cps, dummy_path, var_lines=all_vars))
        try:
            subprocess.run(['cpclean', '-n', '1', '-o', refine_pto, refine_pto],
                           check=True, capture_output=True)
            subprocess.run(['autooptimiser', '-n', refine_pto, '-o', refine_pto],
                           check=True, capture_output=True)
            current = pto_mapper.parse_pto_file(refine_pto)
            if verbose:
                print(f'  Refine iter {i + 1}: remapped {len(matches)} stars, model updated.')
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f'  Refine iter {i + 1}: optimisation failed ({e}), stopping.')
            break
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return current, control_points


def _build_pto(image_path, width, height, fov, yaw, pitch, roll,
               control_points=(), dummy_path=None, var_lines='',
               a=INITIAL_A, b=INITIAL_B, c=INITIAL_C, d=INITIAL_D, e=INITIAL_E):
    """Build a Hugin .pto string. If dummy_path is given, add the dummy equirect image and variables."""
    img_line = (f'i w{width} h{height} f3 v{fov} y{yaw} p{pitch} r{roll} '
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
    return _build_pto(
        image_name, width, height, fov, yaw, pitch, roll,
        control_points=control_points, dummy_path=dummy_path,
        a=a, b=b, c=c, d=d, e=e, var_lines=var_lines,
    )


def _run_autooptimiser(input_pto, output_pto):
    """Run Hugin autooptimiser and return True on success."""
    try:
        proc = subprocess.run(
            ['autooptimiser', '-n', input_pto, '-o', output_pto],
            capture_output=True, text=True, timeout=120,
        )
        return proc.returncode == 0, proc.stdout + proc.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return False, str(exc)


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
    width, height = full_image.size

    if args.verbose:
        print(f'Image: {args.image} ({width}x{height})')
        print(f'Observer: lat={observer.lat} lon={observer.lon} elev={observer.elevation}')
        print(f'Timestamp (UTC): {observer.date}')

    # Use the local tetra3 solver; its pattern database is built from stars.py.
    t3 = Tetra3()
    result, crop_box, flipped = _solve_image(t3, full_image, verbose=args.verbose)
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

    if args.verbose:
        print(f'tetra3 centre: RA={ra_deg:.4f} Dec={dec_deg:.4f}')
        print(f'Image centre Az/Alt: {centre_az:.4f} / {centre_alt:.4f}')
        print(f'Hugin yaw/pitch/roll: {hugin_yaw:.4f} / {hugin_pitch:.4f} / {hugin_roll:.4f}')

    # Build the initial PTO using the solved yaw/pitch/roll and a fixed set of
    # lens parameters taken from a known-good calibration. This gives a good
    # enough model to project the whole image and match stars across the field.
    tmp_pto_path = tempfile.mktemp(suffix='.pto')
    with open(tmp_pto_path, 'w') as f:
        f.write(_build_pto(args.image, width, height, INITIAL_FOV,
                           hugin_yaw, hugin_pitch, hugin_roll))
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
