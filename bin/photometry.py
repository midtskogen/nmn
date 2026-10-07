#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photometry.py — Rough photometric mass estimation for meteors.

Approach
--------
1. For each station camera we have a lens PTO (image <-> az/alt mapping) and
   an event frame (fireball_orig.jpg). brightstar.py gives catalog stars'
   pixel coordinates and magnitudes at the event time; we measure each
   star's flux with a small aperture and fit a per-camera zero point
   (instrumental magnitude vs catalog magnitude).

2. The meteor's per-frame peak brightness (event.txt 'trail/brightness',
   pixel luma 0-255) is converted to apparent magnitude via that zero
   point, then to absolute magnitude using the range from the station to
   the trajectory point at that instant, plus a simple airmass extinction
   correction.

3. Luminous power L = 4 pi r^2 F_vega * 10^(-0.4 m); photometric mass
   M_ph = 2/(tau v^2) * integral L dt  with luminous efficiency tau from
   speed (Sansom et al. 2019 scaling).

This is deliberately approximate — uncalibrated consumer cameras, JPEG
compression, saturated pixels, non-linear response. It is used as an
order-of-magnitude cross-check on the deceleration-derived mass, not as a
precision measurement.

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

F_VEGA = 2.5e-6        # W/m2 for a mag-0 star (V band, approx)
EXTINCTION_K = 0.28    # mag per airmass, typical clear night


def _load_event_txt(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def _star_zeropoint(image_path, pto_path, timestamp, lat, lon, elev,
                    bright_limit=3.5):
    """Instrumental zero point: mag = -2.5 log10(flux) + zp.

    Returns (zp, n_stars, scatter_mag) or None.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        out = subprocess.run(
            [sys.executable, str(_SCRIPT / 'brightstar.py'), str(timestamp),
             str(pto_path), '-f', str(bright_limit), '-n', '60',
             '-x', str(lon), '-y', str(lat), '-a', str(elev)],
            capture_output=True, text=True, timeout=60)
    except Exception as e:
        logging.debug(f'brightstar failed: {e}')
        return None
    if out.returncode != 0:
        return None

    # brightstar prints: sx sy az alt 'name' mag  (source-image coords)
    stars = []
    for line in out.stdout.splitlines():
        m = re.match(r"^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+'?(\w+)'?\s+([-\d.]+)", line)
        if m:
            sx, sy, az, alt, name, mag = m.groups()
            stars.append((float(sx), float(sy), float(alt),
                          name, float(mag)))
    if not stars:
        return None

    img = np.asarray(Image.open(image_path).convert('L'), dtype=float)
    ih, iw = img.shape

    m_cat, m_inst = [], []
    for sx, sy, alt, name, mag in stars:
        x, y = int(round(sx)), int(round(sy))
        if not (10 <= x < iw - 10 and 10 <= y < ih - 10):
            continue
        ap = img[y - 4:y + 5, x - 4:x + 5]          # aperture r=4
        ann = img[y - 10:y + 11, x - 10:x + 11]      # annulus 7..10
        mask = np.ones(ann.shape, bool)
        mask[7 - 10 + 6:7 + 10 - 6, 7 - 10 + 6:7 + 10 - 6] = False
        # inner 7x7 excluded -> annulus pixels
        yy, xx = np.mgrid[-10:11, -10:11]
        ring = (np.hypot(yy, xx) > 6)
        bkg = np.median(ann[ring]) if ring.sum() else 0.0
        flux = float(np.sum(ap - bkg))
        if flux <= 0:
            continue
        m_cat.append(mag)
        m_inst.append(-2.5 * math.log10(flux))
    if len(m_cat) < 3:
        return None
    zps = [mi + mc for mi, mc in zip(m_inst, m_cat)]
    zp = float(np.median(zps))
    scatter = float(np.std(zps)) if len(zps) > 3 else 1.0
    return zp, len(zps), scatter


def meteor_magnitudes(brightness, zp, exposure_s=0.04):
    """Per-frame peak luma -> rough apparent magnitude.

    Treats peak-pixel brightness as a flux proxy proportional to total
    streak flux (moving meteor smears flux along the trail, so this is a
    lower bound on flux / upper bound on magnitude in faint cases).
    Aperture-equivalent flux ~ b (counts per exposure); stars are measured
    the same way so the zero point absorbs the constant.
    """
    out = []
    for b in brightness:
        if b <= 1:
            out.append(None)
            continue
        out.append(zp - 2.5 * math.log10(b))
    return out


def extinction_corr(alt_deg):
    """Airmass extinction correction (mag to add to observed m)."""
    if alt_deg <= 0:
        return 0.0
    z = math.radians(90 - alt_deg)
    x = 1.0 / (math.cos(z) + 0.50572 * (96.07995 - (90 - alt_deg)) ** -1.6364)
    return -EXTINCTION_K * x     # subtract extinction


def photometric_mass(m_app_list, ranges_km, times_s, speeds_ms,
                     tau_fn=None):
    """Integrate the light curve -> photometric mass [kg].

    m_app_list: apparent magnitudes per sample (None = skip)
    ranges_km: station-to-meteor distance per sample
    times_s: timestamps (s)
    speeds_ms: meteoroid speed per sample (m/s)
    tau_fn: luminous efficiency as f(v_kms); default ~0.7%.
    """
    if tau_fn is None:
        # Sansom et al. (2019): tau ~ 0.007 around 20 km/s, scales ~v
        def tau_fn(v_ms):
            return min(0.20, max(0.001, 0.0007 * (v_ms / 1000.0)))
    pts = [(t, m, r, v) for t, m, r, v in
           zip(times_s, m_app_list, ranges_km, speeds_ms) if m is not None]
    if len(pts) < 2:
        return None
    lum = []
    for t, m, r, v in pts:
        flux = F_VEGA * 10 ** (-0.4 * m)          # W/m2 at observer
        lum.append(4 * math.pi * (r * 1e3) ** 2 * flux)   # W
    energy = np.trapz(lum, [p[0] for p in pts])           # J radiated
    masses = []
    for t, m, r, v in pts:
        tau = tau_fn(v)
        masses.append(2.0 / (tau * v ** 2))
    # M = 2/(tau v^2) * ∫L dt, evaluated with mean weighting over curve
    w = np.array([tau_fn(v) * v ** 2 for _, _, _, v in pts])
    eff = np.trapz(w, [p[0] for p in pts]) / 2.0          # ∫(tau v²/2)dt
    if eff <= 0:
        return None
    return energy / eff


def camera_photometry(cam_dir, resdat=None, plot_data=None, timestamp=None):
    """Best-effort photometric estimate for one camera directory.

    Returns dict with per-frame magnitudes + photometric mass, or None.
    """
    cam_dir = Path(cam_dir)
    etxt = cam_dir / 'event.txt'
    if not etxt.exists():
        return None
    cfg = _load_event_txt(etxt)
    try:
        brightness = [float(b) for b in cfg.get('trail', 'brightness').split()]
        coords = cfg.get('trail', 'coordinates').split()
        timestamps = [float(t) for t in cfg.get('trail', 'timestamps').split()]
    except Exception:
        return None
    if not brightness or not coords or len(brightness) != len(coords):
        return None

    pto = None
    for cand in ('lens.pto', 'fireball.pto'):
        if (cam_dir / cand).exists():
            pto = cam_dir / cand
            break
    img = None
    for cand in ('fireball_orig.jpg', 'fireball.jpg'):
        if (cam_dir / cand).exists():
            img = cam_dir / cand
            break
    if pto is None or img is None or timestamp is None:
        return None

    lat = float(cfg.get('summary', 'latitude', fallback=60.0))
    lon = float(cfg.get('summary', 'longitude', fallback=10.0))
    elev = float(cfg.get('summary', 'elevation', fallback=0.0))

    zp_res = _star_zeropoint(img, pto, timestamp, lat, lon, elev)
    if zp_res is None:
        return None
    zp, n_stars, scatter = zp_res

    # per-frame apparent magnitude (peak-pixel proxy)
    m_app = meteor_magnitudes(brightness, zp)

    # trajectory geometry -> range to each centroid point
    ranges = None
    if resdat is not None and plot_data is not None:
        try:
            from fbspd_merge import lonlat2xyz, expfunc
            r_obs = llh2ecef_station(lon, lat, elev)
            merged = plot_data['final_merged_data']
            params = plot_data['final_params']
            reltime = np.asarray(merged['reltime'])
            pos_km = np.asarray(merged['pos'])
            r_start = lonlat2xyz(float(resdat.long1[0]), float(resdat.lat1[0]),
                                 float(resdat.height[0])) * 1000.0
            r_end = lonlat2xyz(float(resdat.long1[1]), float(resdat.lat1[1]),
                               float(resdat.height[1])) * 1000.0
            track = (r_end - r_start) / np.linalg.norm(r_end - r_start)
            p0 = float(expfunc(0.0, *params))
            ranges = []
            for t in timestamps:
                s = (float(expfunc(t - reltime[0], *params)) - p0) * 1000.0
                pt = r_start + track * s
                ranges.append(np.linalg.norm(pt - r_obs) / 1000.0)
        except Exception as e:
            logging.debug(f'range calc failed: {e}')
            ranges = None
    if ranges is None:
        # crude fallback: slant range ~ h/sin(alt)
        ranges = [None] * len(m_app)

    return {'zp': zp, 'n_stars': n_stars, 'scatter_mag': scatter,
            'm_app': m_app, 'brightness': brightness,
            'timestamps': timestamps, 'ranges_km': ranges}


def llh2ecef_station(lon, lat, elev_m):
    import darkflight
    return darkflight.llh2ecef(lon, lat, elev_m)


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
    results = []
    for etxt in sorted(event_dir.glob('*/cam*/event.txt')):
        r = camera_photometry(etxt.parent, resdat, plot_data, timestamp)
        if r:
            r['cam'] = str(etxt.parent.relative_to(event_dir))
            results.append(r)

    # combine light curves across cameras (use fitted speed if available)
    m_all = []
    for r in results:
        ts, mags = r['timestamps'], r['m_app']
        rngs = r['ranges_km']
        speeds = None
        if plot_data is not None:
            from fbspd_merge import expfunc_1stder
            reltime0 = float(np.min(plot_data['final_merged_data']['reltime']))
            speeds = [float(expfunc_1stder(t - reltime0,
                                         *plot_data['final_params'])) * 1000.0
                      for t in ts]
        else:
            speeds = [12000.0] * len(ts)
        if any(x is None for x in rngs):
            continue
        valid = [(t, m, rng, v) for t, m, rng, v in
                 zip(ts, mags, rngs, speeds) if m is not None]
        if len(valid) < 2:
            continue
        m_est = photometric_mass([v[1] for v in valid],
                                 [v[2] for v in valid],
                                 [v[0] for v in valid],
                                 [v[3] for v in valid])
        if m_est:
            r['m_phot_kg'] = m_est
            m_all.append(m_est)
    out = {'per_cam': results,
           'm_phot_kg': float(np.median(m_all)) if m_all else None,
           'n_cams': len(m_all)}
    return out


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
              'm_phot=', c.get('m_phot_kg'))
    print('combined m_phot_kg:', r['m_phot_kg'])


if __name__ == '__main__':
    main()
