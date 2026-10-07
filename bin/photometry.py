#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photometry.py — Rough photometric mass estimation for meteors.

Approach
--------
1. Per station camera we have a gnomonic image (`*-gnomonic.jpg`) and
   `gnomonic_corr_grid.pto` (image <-> az/alt). brightstar.py maps catalog
   stars to image pixels at the event time; we measure each star's peak
   pixel above background and fit a per-camera zero point.

2. The meteor's per-point trail brightness (event.txt 'trail/brightness',
   peak luma 0-255 measured in the same gnomonic image) is converted to
   apparent magnitude via that zero point. Because the meteor moves
   between frames, its flux is smeared along the trail, so peak-pixel
   photometry *underestimates* total flux -> the resulting mass is a
   conservative lower bound (order-of-magnitude).

3. The range to each trail point is computed geometrically by intersecting
   the camera line-of-sight (per-point az/alt from 'coordinates') with the
   fitted trajectory line — no timing alignment needed.

4. Luminous power L = 4 pi r^2 F0 * 10^(-0.4 m); photometric mass
   M_ph = integral L dt / (0.5 tau v^2) with speed-dependent luminous
   efficiency tau (meteor-photometry convention, ~0.7% at 20 km/s).

This is deliberately approximate (consumer cameras, JPEG, smearing,
unmodeled extinction) — used as an order-of-magnitude cross-check on the
deceleration-derived mass, not a precision measurement.

Usage: photometry.py <event_dir>
"""

import configparser
import json
import logging
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).resolve().parent
for p in (_SCRIPT, _SCRIPT.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

F_VEGA = 2.5e-6        # W/m2 for a mag-0 star (approx, broadband)
EXTINCTION_K = 0.28    # mag per airmass


def llh2ecef(lon, lat, h_m):
    """WGS-84 geodetic to ECEF (duplicated from darkflight to stay standalone)."""
    a, f = 6378137.0, 1.0 / 298.257223563
    e2 = f * (2 - f)
    sin_lat, cos_lat = math.sin(math.radians(lat)), math.cos(math.radians(lat))
    n = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    return np.array([(n + h_m) * cos_lat * math.cos(math.radians(lon)),
                     (n + h_m) * cos_lat * math.sin(math.radians(lon)),
                     (n * (1 - e2) + h_m) * sin_lat])


def altaz_to_vec_ecef(az_deg, alt_deg, lon, lat):
    """Line-of-sight unit vector in ECEF for az/alt at observer lon/lat."""
    az, alt = math.radians(az_deg), math.radians(alt_deg)
    e = -math.sin(az); n_ = math.cos(az) * math.cos(alt)
    u = math.sin(alt)
    # ENU: east = -sin(az) cos(alt), north = cos(az) cos(alt), up = sin(alt)
    en = -math.sin(az) * math.cos(alt)
    nn = math.cos(az) * math.cos(alt)
    uu = math.sin(alt)
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    # ENU -> ECEF rotation
    east = np.array([-math.sin(lon_r), math.cos(lon_r), 0.0])
    north = np.array([-math.sin(lat_r) * math.cos(lon_r),
                      -math.sin(lat_r) * math.sin(lon_r), math.cos(lat_r)])
    up = np.array([math.cos(lat_r) * math.cos(lon_r),
                   math.cos(lat_r) * math.sin(lon_r), math.sin(lat_r)])
    return en * east + nn * north + uu * up


def _load_event_txt(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def _star_zeropoint(image_path, pto_path, timestamp, lat, lon, elev,
                    bright_limit=3.5):
    """Zero point from catalog stars measured as peak-pixel flux.

    Returns (zp, n_stars, scatter_mag) where mag = zp - 2.5log10(peak-bkg).
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        out = subprocess.run(
            [sys.executable, str(_SCRIPT / 'brightstar.py'), str(timestamp),
             str(pto_path), '-f', str(bright_limit), '-n', '80',
             '-x', str(lon), '-y', str(lat), '-a', str(elev)],
            capture_output=True, text=True, timeout=60)
    except Exception as e:
        logging.debug(f'brightstar failed: {e}')
        return None
    if out.returncode != 0:
        return None

    img = np.asarray(Image.open(image_path).convert('L'), dtype=float)
    ih, iw = img.shape
    yy, xx = np.mgrid[-9:10, -9:10]
    ring = (np.hypot(yy, xx) > 5) & (np.hypot(yy, xx) <= 9)

    zps = []
    for line in out.stdout.splitlines():
        m = re.match(r"^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
                     r"\s+'?(\w+)'?\s+([-\d.]+)", line)
        if not m:
            continue
        x, y = int(round(float(m.group(1)))), int(round(float(m.group(2))))
        mag = float(m.group(6))
        if not (12 <= x < iw - 12 and 12 <= y < ih - 12):
            continue
        ap = img[y - 4:y + 5, x - 4:x + 5]
        ann = img[y - 9:y + 10, x - 9:x + 10]
        bkg = float(np.median(ann[ring]))
        peak = float(ap.max()) - bkg
        if peak <= 3:          # too faint / saturated-out
            continue
        # extinction-correct catalog mag to apparent at that altitude
        alt = float(m.group(4))
        m_corr = mag + extinction_mag(alt)
        zps.append(m_corr + 2.5 * math.log10(peak))
    if len(zps) < 3:
        return None
    zp, sc = float(np.median(zps)), float(np.std(zps))
    if sc > 1.5:                     # >~1.5 mag scatter = bad calibration
        return None
    return zp, len(zps), sc


def extinction_mag(alt_deg):
    """Magnitude penalty (positive) from airmass at altitude alt_deg."""
    if alt_deg >= 45:
        x = 1.0 / math.sin(math.radians(max(alt_deg, 1)))
    else:
        x = 1.0 / (math.cos(math.radians(90 - alt_deg)) +
                   0.50572 * (96.07995 - alt_deg) ** -1.6364)
    return EXTINCTION_K * max(x, 0)


def range_to_track(lon, lat, elev, az, alt, r_start, r_end):
    """Distance from observer to the closest approach of the track line
    to the line-of-sight ray, evaluated at the point of closest approach.

    Returns (range_km, t_frac along track 0..1) or None.
    """
    r_obs = llh2ecef(lon, lat, elev)
    d = altaz_to_vec_ecef(az, alt, lon, lat)
    t_dir = r_end - r_start
    tl = np.linalg.norm(t_dir)
    if tl == 0:
        return None
    t_dir /= tl
    w0 = r_obs - r_start
    a = np.dot(d, t_dir)
    denom = 1 - a * a
    if abs(denom) < 1e-9:
        return None
    b, c = np.dot(d, w0), np.dot(t_dir, w0)
    s_los = (a * c - b) / denom      # distance along LOS
    s_trk = (c - a * b) / denom      # distance along track
    if s_los <= 0:
        return None
    return s_los / 1000.0, s_trk / tl


def camera_photometry(cam_dir, r_start=None, r_end=None, timestamp=None):
    """Photometric estimate for one camera directory."""
    cam_dir = Path(cam_dir)
    etxt = cam_dir / 'event.txt'
    if not etxt.exists():
        return None
    cfg = _load_event_txt(etxt)
    try:
        brightness = [float(b) for b in
                      cfg.get('trail', 'brightness').split()]
        coords = [[float(v) for v in c.split(',')] for c in
                  cfg.get('trail', 'coordinates').split()]
        timestamps = [float(t) for t in
                      cfg.get('trail', 'timestamps').split()]
    except Exception:
        return None
    if not (brightness and coords and len(brightness) == len(coords)):
        return None

    gnom = list(cam_dir.glob('*-gnomonic.jpg'))
    pto = cam_dir / 'gnomonic_corr_grid.pto'
    if not gnom or not pto.exists():
        return None
    if timestamp is None:
        timestamp = float(timestamps[0]) if timestamps else None
    if timestamp is None:
        return None

    lat = float(cfg.get('summary', 'latitude', fallback='60.0'))
    lon = float(cfg.get('summary', 'longitude', fallback='10.0'))
    elev = float(cfg.get('summary', 'elevation', fallback='0'))

    zp_res = _star_zeropoint(gnom[0], pto, timestamp, lat, lon, elev)
    if zp_res is None:
        return None
    zp, n_stars, scatter = zp_res

    try:
        from PIL import Image
        img = np.asarray(Image.open(gnom[0]).convert('L'), dtype=float)
        bkg = float(np.median(img))
    except Exception:
        bkg = 20.0

    m_app, ranges, fracs = [], [], []
    for b, (az, alt) in zip(brightness, coords):
        peak = b - bkg
        m_app.append(zp - 2.5 * math.log10(max(peak, 1.0)))
        if r_start is not None and r_end is not None:
            rg = range_to_track(lon, lat, elev, az, alt, r_start, r_end)
            ranges.append(rg[0] if rg else None)
            fracs.append(rg[1] if rg else None)
        else:
            ranges.append(None)
            fracs.append(None)

    return {'zp': zp, 'n_stars': n_stars, 'scatter_mag': scatter,
            'bkg': bkg, 'm_app': m_app, 'brightness': brightness,
            'timestamps': timestamps, 'ranges_km': ranges,
            'track_frac': fracs, 'lat': lat, 'lon': lon, 'elev': elev}


def photometric_mass(m_app, ranges_km, times_s, speeds_ms):
    """Integrate light curve -> photometric mass [kg] (lower bound).

    Luminous efficiency tau ~ speed-dependent (Sansom et al. 2019):
    tau = 0.007 at ~20 km/s scaling roughly linearly below, capped.
    """
    def tau(v_ms):
        return min(0.20, max(0.001, 0.0007 * (v_ms / 1000.0)))
    pts = [(t, m, r, v) for t, m, r, v in
           zip(times_s, m_app, ranges_km, speeds_ms)
           if m is not None and r and r > 1.0]
    if len(pts) < 2:
        return None
    ts = np.array([p[0] for p in pts])
    ts = ts - ts[0]
    lum = np.array([4 * math.pi * (r * 1e3) ** 2 * F_VEGA * 10 ** (-0.4 * m)
                    for t, m, r, v in pts])          # W
    w = np.array([tau(v) * v ** 2 / 2.0 for t, m, r, v in pts])
    energy = np.trapz(lum, ts)                       # J radiated
    eff = np.trapz(w, ts)                            # ∫(tau v²/2)dt
    return energy / eff if eff > 0 else None


def event_photometry(event_dir, resdat=None, plot_data=None, timestamp=None):
    """Aggregate photometric estimates over all station cams.

    Returns {'per_cam': [...], 'm_phot_kg': median or None}.
    """
    event_dir = Path(event_dir)
    if timestamp is None:
        m = re.search(r'(\d{8})/(\d{6})', str(event_dir))
        if m:
            from datetime import datetime, timezone
            dt = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
            timestamp = dt.replace(tzinfo=timezone.utc).timestamp()

    r_start = r_end = None
    if resdat is not None:
        r_start = llh2ecef(float(resdat.long1[0]), float(resdat.lat1[0]),
                           float(resdat.height[0]) * 1000.0)
        r_end = llh2ecef(float(resdat.long1[1]), float(resdat.lat1[1]),
                         float(resdat.height[1]) * 1000.0)

    results = []
    for etxt in sorted(event_dir.glob('*/cam*/event.txt')):
        r = camera_photometry(etxt.parent, r_start, r_end, timestamp)
        if r:
            r['cam'] = str(etxt.parent.relative_to(event_dir))
            results.append(r)

    # speed along track from fbspd fit (fallback: nominal)
    def speed_at_frac(frac):
        if plot_data is not None and plot_data.get('final_params') is not None:
            try:
                from fbspd_merge import expfunc_1stder
                rt = np.asarray(plot_data['final_merged_data']['reltime'])
                t = rt[0] + frac * (rt[-1] - rt[0])
                return float(expfunc_1stder(t, *plot_data['final_params'])) * 1000.0
            except Exception:
                pass
        return 12000.0

    m_all = []
    for r in results:
        speeds = [speed_at_frac(f) if f is not None else 12000.0
                  for f in r['track_frac']]
        m_est = photometric_mass(r['m_app'], r['ranges_km'],
                                 r['timestamps'], speeds)
        if m_est:
            r['m_phot_kg'] = m_est
            m_all.append(m_est)
    return {'per_cam': results,
            'm_phot_kg': float(np.median(m_all)) if m_all else None,
            'n_cams': len(m_all)}


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Photometric mass estimation')
    ap.add_argument('event_dir')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    event_dir = Path(args.event_dir)
    from fbspd_merge import readres
    import pickle
    res_files = list(event_dir.glob('obs_*.res'))
    resdat = readres(str(res_files[0])) if res_files else None
    pkl = event_dir / '_fbspd_plot_data.pkl'
    plot_data = None
    if pkl.exists():
        with pkl.open('rb') as f:
            _, plot_data = pickle.load(f)
    r = event_photometry(event_dir, resdat, plot_data)
    for c in r['per_cam']:
        print(c['cam'], 'zp=', round(c['zp'], 2), 'stars=', c['n_stars'],
              'mags=', [round(m, 1) if m else None for m in c['m_app']],
              'ranges=', [round(x, 1) if x else None for x in c['ranges_km']],
              'm_phot=', c.get('m_phot_kg'))
    print('combined m_phot_kg:', r['m_phot_kg'])


if __name__ == '__main__':
    main()
