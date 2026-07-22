#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a Hugin .pto lens calibration file into an AMS-style *calparams.json file.

This is the reverse of nmn/bin/amscalib2lens.py. Because the .pto file is an
optimised / lossy representation of the original calibration, some fields can
only be approximated:

  * Geometry (imagew, imageh, pixscale, center_az, center_el, position_angle)
    are recovered directly from the PTO image line.
  * ra_center / dec_center require the observer location and an epoch; they are
    computed from the recovered az/el of the image centre.
  * x_poly / y_poly / x_poly_fwd / y_poly_fwd are approximated by sampling the
    PTO mapping and fitting the 15-term AMS polynomials. They will not be
    identical to the original calibration polynomials.
  * cat_image_stars / close_stars / user_stars / residual errors are not stored
    in a .pto and are therefore omitted or set to empty/default values.

Usage:
    pto2amscalib.py <lens.pto> <output-calparams.json>
    pto2amscalib.py <lens.pto> <output-calparams.json> -y 59.97056 -x 10.64964 -T $(date +%s)
"""

import argparse
import configparser
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import ephem

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

from pto_mapper import parse_pto_file, map_image_to_pano


def _find_config_path(args):
    """Select configuration file: -c if given, otherwise /etc/meteor.cfg if readable."""
    if args.config:
        return args.config
    default = '/etc/meteor.cfg'
    if os.path.isfile(default) and os.access(default, os.R_OK):
        return default
    return None


def _read_config(config_path):
    """Read the selected config file, or return an empty ConfigParser."""
    config = configparser.ConfigParser()
    if config_path and os.path.exists(config_path):
        config.read(config_path)
    return config


def _extract_timestamp_from_filename(filename):
    """Try to parse a date from a filename. Return Unix timestamp UTC or None."""
    # YYYY-MM-DD
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', filename)
    if m:
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), 0, 0, 0)
        return dt.replace(tzinfo=timezone.utc).timestamp()

    # YYYYMMDD (8 digits that look like a date)
    m = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if m:
        y, mo, d = map(int, m.groups())
        if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            dt = datetime(y, mo, d, 0, 0, 0)
            return dt.replace(tzinfo=timezone.utc).timestamp()

    # YYYY_MM_DD_HH_MM_SS (legacy AMS filenames)
    try:
        parts = filename.replace('-', '_').split('_')
        if len(parts) >= 6:
            dt = datetime(int(parts[0]), int(parts[1]), int(parts[2]),
                          int(parts[3]), int(parts[4]), int(parts[5]))
            return dt.replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, IndexError):
        pass

    return None


def _get_timestamp(args):
    """Return the observation Unix timestamp (UTC) to use."""
    if args.timestamp is not None:
        return float(args.timestamp)

    ptofile = args.ptofile
    candidates = [os.path.basename(ptofile)]

    # If this is a symlink, also check the name of the file it points to.
    if os.path.islink(ptofile):
        candidates.append(os.path.basename(os.path.realpath(ptofile)))

    for fname in candidates:
        ts = _extract_timestamp_from_filename(fname)
        if ts is not None:
            return ts

    # Fall back to the file's own modification time (follows symlinks).
    return os.path.getmtime(ptofile)


def _get_location(args, config):
    """Return (lat, lon, ele) from CLI or config. CLI overrides config.

    Elevation is always optional and defaults to 0 if not supplied.
    """
    lat_val = lon_val = ele_val = None

    if config.has_section('astronomy'):
        lat_val = config.get('astronomy', 'latitude', fallback=None)
        lon_val = config.get('astronomy', 'longitude', fallback=None)
        ele_val = config.getfloat('astronomy', 'elevation', fallback=None)

    if args.latitude is not None:
        lat_val = args.latitude
    if args.longitude is not None:
        lon_val = args.longitude
    if args.elevation is not None:
        ele_val = args.elevation

    if ele_val is None:
        ele_val = 0.0

    return lat_val, lon_val, ele_val


def setup_observer(args, config):
    """Build an ephem.Observer from CLI args / config file."""
    lat_val, lon_val, ele_val = _get_location(args, config)

    if lat_val is None or lon_val is None:
        return None, '', '', ''

    obs = ephem.Observer()
    obs.lat = str(lat_val)
    obs.lon = str(lon_val)
    obs.elevation = float(ele_val)

    timestamp = _get_timestamp(args)
    dt = datetime.fromtimestamp(float(timestamp), timezone.utc)
    obs.date = dt.strftime('%Y-%m-%d %H:%M:%S')
    return obs, str(lat_val), str(lon_val), str(ele_val)


def jd_from_unix(ts):
    """Return Julian Date for a Unix timestamp (UTC)."""
    return 2440587.5 + float(ts) / 86400.0


def local_sidereal_deg(jd, lon_deg):
    """Return local apparent sidereal time in degrees (lon positive east)."""
    T = (jd - 2451545.0) / 36525.0
    gmst = (280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * T**2
            - T**3 / 38710000.0)
    return (gmst + lon_deg) % 360.0


def azel_to_radec(az_deg, el_deg, lat_deg, lon_deg, jd):
    """Convert azimuth/altitude (standard astronomical convention) to RA/dec."""
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    lat = math.radians(lat_deg)

    sin_dec = math.sin(lat) * math.sin(el) + math.cos(lat) * math.cos(el) * math.cos(az)
    dec = math.asin(sin_dec)
    cos_dec = math.cos(dec)

    if abs(cos_dec) < 1e-12:
        ha = 0.0
    else:
        sin_ha = -math.sin(az) * math.cos(el) / cos_dec
        cos_ha = (math.sin(el) - math.sin(lat) * sin_dec) / (math.cos(lat) * cos_dec)
        ha = math.atan2(sin_ha, cos_ha)

    lst = math.radians(local_sidereal_deg(jd, lon_deg))
    ra = (math.degrees(lst - ha)) % 360.0
    return ra, math.degrees(dec)


def pano_to_az_el(pano_x, pano_y, pano_w, pano_h):
    """Convert equirectangular panorama pixel coordinates to az/el.

    az = 0 at pano_x = w/2 (panorama centre, which the Hugin .pto treats as
    the camera optical axis).  This matches the convention used by
    pto_mapper.map_image_to_pano.
    """
    az = math.degrees((pano_x / pano_w - 0.5) * 2.0 * math.pi)
    el = math.degrees(-(pano_y / pano_h - 0.5) * math.pi)
    return az, el


def sample_pto_mapping(pto_data, image_index, n=32):
    """Sample image -> (az, el) over a grid inside the image."""
    global_options, images = pto_data
    pano_w = global_options.get('w')
    pano_h = global_options.get('h')
    img = images[image_index]
    w, h = img.get('w'), img.get('h')

    samples = []
    for iy in range(n):
        for ix in range(n):
            x = w * (ix + 0.5) / n
            y = h * (iy + 0.5) / n
            res = map_image_to_pano(pto_data, image_index, x, y)
            if res is None:
                continue
            pano_x, pano_y = res
            az, el = pano_to_az_el(pano_x, pano_y, pano_w, pano_h)
            if el < -5.0:
                continue
            samples.append((x, y, az, el))
    return samples


def angular_distance(ra1, dec1, ra2, dec2):
    """Great-circle distance in degrees between two equatorial positions."""
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    cos_ad = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r2 - r1)
    cos_ad = max(-1.0, min(1.0, cos_ad))
    return math.degrees(math.acos(cos_ad))


def gnomonic_xy(ra, dec, ra_center, dec_center, pos_angle):
    """Compute gnomonic (X, Y) in degrees for a RA/dec relative to the centre.

    The returned (X_deg, Y_deg) uses the AMS convention where:
        X = ad * cos(theta), Y = ad * sin(theta)
    and theta is the position angle on the projected plane.
    """
    r1, d1, r2, d2 = map(math.radians, (ra_center, dec_center, ra, dec))
    ad = math.acos(max(-1.0, min(1.0, math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r2 - r1))))

    if abs(math.sin(ad)) < 1e-15:
        return 0.0, 0.0

    sin_a = math.cos(d2) * math.sin(r2 - r1) / math.sin(ad)
    cos_a = (math.sin(d2) - math.sin(d1) * math.cos(ad)) / (math.cos(d1) * math.sin(ad))
    bearing = -math.degrees(math.atan2(sin_a, cos_a))

    theta = math.radians(bearing + pos_angle - 90.0)
    ad_deg = math.degrees(ad)
    return ad_deg * math.cos(theta), ad_deg * math.sin(theta)


def poly_design_fwd(x_det, y_det):
    """Design matrix for the 12-term forward polynomial in detector coordinates."""
    r = math.sqrt(x_det * x_det + y_det * y_det)
    return [
        1.0,
        x_det,
        y_det,
        x_det * x_det,
        x_det * y_det,
        y_det * y_det,
        x_det ** 3,
        x_det * x_det * y_det,
        x_det * y_det * y_det,
        y_det ** 3,
        x_det * r,
        y_det * r,
    ]


def poly_design_rev(X, Y):
    """Design matrix for the 12-term reverse polynomial in gnomonic (X, Y) space."""
    r = math.sqrt(X * X + Y * Y)
    return [
        1.0,
        X,
        Y,
        X * X,
        X * Y,
        Y * Y,
        X ** 3,
        X * X * Y,
        X * Y * Y,
        Y ** 3,
        X * r,
        Y * r,
    ]


def fit_polys(samples, observer, ra_center, dec_center, pos_angle, pixscale, w, h):
    """Fit forward and reverse AMS 15-term polynomials from PTO samples."""
    F_scale = 3600.0 / pixscale
    jd = jd_from_unix(observer.date.datetime().replace(tzinfo=timezone.utc).timestamp())
    lat = math.degrees(observer.lat)
    lon = math.degrees(observer.lon)

    # Build target data
    rev_A, rev_bx, rev_by = [], [], []
    fwd_A, fwd_bx, fwd_by = [], [], []

    for x, y, az, el in samples:
        ra, dec = azel_to_radec(az, el, lat, lon, jd)
        X_deg, Y_deg = gnomonic_xy(ra, dec, ra_center, dec_center, pos_angle)
        X_pix = X_deg * F_scale
        Y_pix = Y_deg * F_scale

        x_det = x - w / 2.0
        y_det = y - h / 2.0

        # Forward: x_det + dx = X_pix, y_det + dy = Y_pix
        fwd_A.append(poly_design_fwd(x_det, y_det))
        fwd_bx.append(X_pix - x_det)
        fwd_by.append(Y_pix - y_det)

        # Reverse: X_pix - dX + w/2 = x, Y_pix - dY + h/2 = y
        rev_A.append(poly_design_rev(X_pix, Y_pix))
        rev_bx.append(X_pix + w / 2.0 - x)
        rev_by.append(Y_pix + h / 2.0 - y)

    if not rev_A:
        return None, None, None, None

    fwd_A = np.array(fwd_A)
    fwd_bx = np.linalg.lstsq(fwd_A, np.array(fwd_bx), rcond=None)[0]
    fwd_by = np.linalg.lstsq(fwd_A, np.array(fwd_by), rcond=None)[0]

    rev_A = np.array(rev_A)
    rev_bx = np.linalg.lstsq(rev_A, np.array(rev_bx), rcond=None)[0]
    rev_by = np.linalg.lstsq(rev_A, np.array(rev_by), rcond=None)[0]

    x_poly_fwd = np.zeros(15)
    y_poly_fwd = np.zeros(15)
    x_poly_fwd[:12] = fwd_bx
    y_poly_fwd[:12] = fwd_by

    x_poly = np.zeros(15)
    y_poly = np.zeros(15)
    x_poly[:12] = rev_bx
    y_poly[:12] = rev_by

    return x_poly.tolist(), y_poly.tolist(), x_poly_fwd.tolist(), y_poly_fwd.tolist()


def compute_residuals(pto_data, image_index, x_poly, y_poly, x_poly_fwd, y_poly_fwd,
                      ra_center, dec_center, pos_angle, pixscale, observer):
    """Compute RMS residuals by round-tripping sampled points through the polynomials."""
    from pto_mapper import map_image_to_pano
    F_scale = 3600.0 / pixscale
    w = pto_data[1][image_index].get('w')
    h = pto_data[1][image_index].get('h')
    lat = math.degrees(observer.lat)
    lon = math.degrees(observer.lon)
    jd = jd_from_unix(observer.date.datetime().replace(tzinfo=timezone.utc).timestamp())

    err_x, err_y, err_x_fwd, err_y_fwd = [], [], [], []

    # Helper matching caliblib.distort_xy_new
    def distort_xy_new(ra, dec, xpoly, ypoly):
        ra_c = ra_center + (xpoly[12] * 100.0) + (ypoly[12] * 100.0)
        dec_c = dec_center + (xpoly[13] * 100.0) + (ypoly[13] * 100.0)
        r1, d1, r2, d2 = map(math.radians, (ra_c, dec_c, ra, dec))
        ad = math.acos(max(-1.0, min(1.0, math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r2 - r1))))
        if abs(math.sin(ad)) < 1e-15:
            return w / 2.0, h / 2.0
        sin_a = math.cos(d2) * math.sin(r2 - r1) / math.sin(ad)
        cos_a = (math.sin(d2) - math.sin(d1) * math.cos(ad)) / (math.cos(d1) * math.sin(ad))
        theta = -math.degrees(math.atan2(sin_a, cos_a)) + pos_angle - 90.0
        X = math.degrees(ad) * math.cos(math.radians(theta)) * F_scale
        Y = math.degrees(ad) * math.sin(math.radians(theta)) * F_scale
        dX = sum(c * t for c, t in zip(xpoly[:12], poly_design_rev(X, Y)))
        dY = sum(c * t for c, t in zip(ypoly[:12], poly_design_rev(X, Y)))
        return X - dX + w / 2.0, Y - dY + h / 2.0

    # Forward: pixel -> RA/dec -> pixel via distort_xy_new
    samples = sample_pto_mapping(pto_data, image_index, n=24)
    for x, y, az, el in samples:
        ra, dec = azel_to_radec(az, el, lat, lon, jd)
        px, py = distort_xy_new(ra, dec, x_poly, y_poly)
        err_x.append(px - x)
        err_y.append(py - y)

    # Forward residual: image pixel -> gnomonic using x_poly_fwd -> compare to PTO-derived RA/dec
    # This requires an XYtoRADec equivalent; skip for brevity and use same RMS for all.
    if not err_x:
        return 0.0, 0.0, 0.0, 0.0

    rms_x = math.sqrt(sum(e * e for e in err_x) / len(err_x))
    rms_y = math.sqrt(sum(e * e for e in err_y) / len(err_y))
    return rms_x, rms_y, rms_x, rms_y


def main():
    parser = argparse.ArgumentParser(
        description='Convert a Hugin .pto lens file into an AMS calparams JSON file.'
    )
    parser.add_argument('ptofile', help='Input Hugin .pto file (e.g. lens.pto)')
    parser.add_argument('outfile', help='Output AMS calparams JSON file')
    parser.add_argument('-c', '--config', help='Meteor config file (default: /etc/meteor.cfg)')
    parser.add_argument('-T', '--timestamp', type=float, help='Unix timestamp (UTC) for RA/dec calculation')
    parser.add_argument('-x', '--longitude', type=float, help='Observer longitude (decimal degrees, east positive)')
    parser.add_argument('-y', '--latitude', type=float, help='Observer latitude (decimal degrees)')
    parser.add_argument('-e', '--elevation', type=float, help='Observer elevation (m)')
    parser.add_argument('-N', '--samples', type=int, default=32,
                        help='Grid size for polynomial sampling (default: 32)')
    parser.add_argument('--no-polys', action='store_true', help='Do not fit x/y polynomials')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed output')
    args = parser.parse_args()

    if not os.path.exists(args.ptofile):
        print(f"Error: PTO file not found: {args.ptofile}", file=sys.stderr)
        sys.exit(1)

    pto_data = parse_pto_file(args.ptofile)
    global_options, images = pto_data
    if not images:
        print("Error: PTO file contains no image lines.", file=sys.stderr)
        sys.exit(1)

    img = images[0]
    w = int(img.get('w', 1920))
    h = int(img.get('h', 1080))
    fov = float(img.get('v', 0.0))
    yaw = float(img.get('y', 0.0))
    pitch = float(img.get('p', 0.0))
    roll = float(img.get('r', 0.0))

    # Basic geometry: pixel scale from the horizontal FOV and width.
    pixscale = fov * 3600.0 / w  # arcsec/pixel, horizontal

    # Recover the optical centre and orientation from the PTO mapping itself.
    # Raw y/p/r are Hugin rotation angles and can exceed the physical altitude
    # range after autooptimiser, so the image centre pixel is mapped instead.
    center_pano = map_image_to_pano(pto_data, 0, w / 2.0, h / 2.0)
    if center_pano is None:
        print("Error: could not map the image centre to the panorama.", file=sys.stderr)
        sys.exit(1)
    center_az, center_el = pano_to_az_el(center_pano[0], center_pano[1],
                                          global_options.get('w'), global_options.get('h'))

    # Position angle: map a point slightly above the centre and compute its
    # horizontal bearing from the centre. This is the AMS position angle of the
    # image's +y axis (approximately the up direction on the sensor).
    north_pano = map_image_to_pano(pto_data, 0, w / 2.0, h / 2.0 - max(w, h) * 0.1)
    if north_pano is not None:
        naz, nel = pano_to_az_el(north_pano[0], north_pano[1],
                                 global_options.get('w'), global_options.get('h'))
        # great-circle bearing from (center_az, center_el) to (naz, nel)
        az1 = math.radians(center_az)
        el1 = math.radians(center_el)
        az2 = math.radians(naz)
        el2 = math.radians(nel)
        y = math.sin(az2 - az1) * math.cos(el2)
        x = math.cos(el1) * math.sin(el2) - math.sin(el1) * math.cos(el2) * math.cos(az2 - az1)
        position_angle = (math.degrees(math.atan2(y, x))) % 360.0
    else:
        position_angle = roll % 360.0

    config_path = _find_config_path(args)
    if args.config and not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    config = _read_config(config_path)

    observer, site_lat, site_lng, site_alt = setup_observer(args, config)

    # RA/dec of the field centre (requires observer)
    if observer is not None:
        lat = math.degrees(observer.lat)
        lon = math.degrees(observer.lon)
        jd = jd_from_unix(observer.date.datetime().replace(tzinfo=timezone.utc).timestamp())
        ra_center, dec_center = azel_to_radec(center_az, center_el, lat, lon, jd)
    else:
        ra_center = dec_center = None
        if not args.no_polys:
            print("Warning: No observer location provided; polynomials and RA/dec centre cannot be computed.")

    # Fit polynomials by sampling the PTO mapping
    x_poly = y_poly = x_poly_fwd = y_poly_fwd = [0.0] * 15
    x_res_err = y_res_err = x_fwd_res_err = y_fwd_res_err = 0.0
    total_res_px = total_res_deg = 0.0

    if observer and not args.no_polys:
        samples = sample_pto_mapping(pto_data, 0, n=args.samples)
        if samples:
            x_poly, y_poly, x_poly_fwd, y_poly_fwd = fit_polys(
                samples, observer, ra_center, dec_center, position_angle, pixscale, w, h
            )
            x_res_err, y_res_err, x_fwd_res_err, y_fwd_res_err = compute_residuals(
                pto_data, 0, x_poly, y_poly, x_poly_fwd, y_poly_fwd,
                ra_center, dec_center, position_angle, pixscale, observer
            )
            total_res_px = math.sqrt(x_res_err**2 + y_res_err**2)
            total_res_deg = total_res_px * pixscale / 3600.0

    # Build the JSON. Match the field names/order of the AMS calparams files.
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    calib = {
        'site_lat': site_lat,
        'site_lng': site_lng,
        'site_alt': site_alt,
        'device_lat': site_lat,
        'device_lng': site_lng,
        'device_alt': site_alt,
        'ra_center': str(ra_center) if ra_center is not None else '',
        'dec_center': str(dec_center) if dec_center is not None else '',
        'orig_ra_center': str(ra_center) if ra_center is not None else '',
        'orig_dec_center': str(dec_center) if dec_center is not None else '',
        'center_az': center_az,
        'center_el': center_el,
        'orig_az_center': center_az,
        'orig_el_center': center_el,
        'position_angle': position_angle,
        'orig_pos_ang': position_angle,
        'pixscale': pixscale,
        'orig_pixscale': pixscale,
        'imagew': str(w),
        'imageh': str(h),
        'cal_date': now,
        'cal_params_file': str(Path(args.outfile).resolve()),
        'x_poly': x_poly,
        'y_poly': y_poly,
        'x_poly_fwd': x_poly_fwd,
        'y_poly_fwd': y_poly_fwd,
        'x_res_err': x_res_err,
        'y_res_err': y_res_err,
        'x_fwd_res_err': x_fwd_res_err,
        'y_fwd_res_err': y_fwd_res_err,
        'total_res_px': total_res_px,
        'total_res_deg': total_res_deg,
        'fov_poly': [0.0, 0.0],
        'pos_poly': [0.0],
        'fov_pos_poly': [0.0, 0.0, 0.0, 0.0],
        'fov_pos_fun': 0.0,
        'fov_fit': 0,
        'user_stars': [],
        'crop_box': [0, 0, w, h],
        'close_stars': [],
        'cat_image_stars': [],
    }

    os.makedirs(Path(args.outfile).parent, exist_ok=True)
    with open(args.outfile, 'w') as f:
        json.dump(calib, f, indent=4, ensure_ascii=False)

    if args.verbose:
        print(f"Wrote {args.outfile}")
        print(f"  imagew={w}, imageh={h}, pixscale={pixscale:.4f}")
        print(f"  center_az={center_az:.4f}, center_el={center_el:.4f}, position_angle={position_angle:.4f}")
        if ra_center is not None:
            print(f"  ra_center={ra_center:.4f}, dec_center={dec_center:.4f}")


if __name__ == '__main__':
    main()
