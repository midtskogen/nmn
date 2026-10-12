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
   filtered peak luma measured in the same gnomonic image) is converted to
   apparent magnitude via that zero point. When a '*-orig.mp4' event video
   exists, an integrated aperture flux is measured in the raw frames at the
   trail positions instead (recovering cameras whose brightness field is
   zero-filled, e.g. 'manual = 1' tracks); its zero point is transferred
   from the gnomonic-image calibration via stars measurable in both
   domains. Saturation (peak >= 250) is flagged per point. Both measures
   only approximate flux -> the resulting mass is a rough lower bound
   (order-of-magnitude).

3. The range to each trail point is computed geometrically by intersecting
   the camera line-of-sight (per-point az/alt from 'coordinates') with the
   fitted trajectory line — no timing alignment needed. The observed
   magnitude is also corrected for atmospheric extinction at the point's
   altitude (0.28 mag/airmass); below ~10-15 deg that correction is large
   and the model unreliable, so low-elevation cameras are flagged.

4. Luminous power L = 4 pi r^2 F0 * 10^(-0.4 m); photometric mass
   M_ph = integral L dt / (0.5 tau v^2) with speed-dependent luminous
   efficiency tau (approximate adopted convention, ~1.4% at 20 km/s).

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
from scipy.integrate import trapezoid

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


def _catalog_stars(pto_path, timestamp, lat, lon, elev, bright_limit=3.5):
    """Catalog stars mapped into the image described by pto_path.

    Returns a list of (x, y, az_deg, alt_deg, mag) or [].
    """
    try:
        out = subprocess.run(
            [sys.executable, str(_SCRIPT / 'brightstar.py'), str(timestamp),
             str(pto_path), '-f', str(bright_limit), '-n', '300',
             '-x', str(lon), '-y', str(lat), '-a', str(elev)],
            capture_output=True, text=True, timeout=60)
    except Exception as e:
        logging.debug(f'brightstar failed: {e}')
        return []
    if out.returncode != 0:
        return []
    stars = []
    for line in out.stdout.splitlines():
        m = re.match(r"^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
                     r"\s+'?([^']+?)'?\s+([-\d.]+)", line)
        if not m:
            continue
        stars.append((float(m.group(1)), float(m.group(2)),
                      float(m.group(3)), float(m.group(4)),
                      float(m.group(6))))
    return stars


def _peak_flux(img, x, y):
    """Peak pixel minus local ring background (gnomonic-image convention)."""
    x, y = int(round(x)), int(round(y))
    ih, iw = img.shape
    if not (12 <= x < iw - 12 and 12 <= y < ih - 12):
        return None
    yy, xx = np.mgrid[-9:10, -9:10]
    ring = (np.hypot(yy, xx) > 5) & (np.hypot(yy, xx) <= 9)
    ap = img[y - 4:y + 5, x - 4:x + 5]
    ann = img[y - 9:y + 10, x - 9:x + 10]
    bkg = float(np.median(ann[ring]))
    return float(ap.max()) - bkg


def _aperture_flux(img, x, y, r_in=6.0, r_out0=11.0, r_out1=16.0):
    """Background-subtracted aperture flux at (x, y).

    Returns (net_flux, bkg, peak_luma) or None near the frame edge.
    """
    ih, iw = img.shape
    x0, x1 = int(x - r_out1), int(x + r_out1 + 1)
    y0, y1 = int(y - r_out1), int(y + r_out1 + 1)
    if x0 < 0 or y0 < 0 or x1 > iw or y1 > ih:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    ap = img[y0:y1, x0:x1]
    m_in = d <= r_in
    m_ann = (d >= r_out0) & (d <= r_out1)
    if m_ann.sum() < 8 or m_in.sum() == 0:
        return None
    bkg = float(np.median(ap[m_ann]))
    return float((ap[m_in] - bkg).sum()), bkg, float(ap[m_in].max())


def _star_zeropoint(image_path, pto_path, timestamp, lat, lon, elev,
                    bright_limit=3.5):
    """Zero point from catalog stars measured as peak-pixel flux.

    Returns (zp, n_stars, scatter_mag) where mag = zp - 2.5log10(peak-bkg).
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    stars = _catalog_stars(pto_path, timestamp, lat, lon, elev, bright_limit)
    if not stars:
        return None

    img = np.asarray(Image.open(image_path).convert('L'), dtype=float)
    zps = []
    for x, y, az, alt, mag in stars:
        peak = _peak_flux(img, x, y)
        if peak is None or peak <= 3:    # too faint / saturated-out
            continue
        # extinction-correct catalog mag to apparent at that altitude
        m_corr = mag + extinction_mag(alt)
        zps.append(m_corr + 2.5 * math.log10(peak))
    return _fit_zeropoint(zps)


def _fit_zeropoint(zps, clip=1.5):
    """Robust zero-point fit: sigma-clip outliers (misidentified lights,
    contaminated apertures), require >=3 stars and <=clip mag scatter."""
    if len(zps) < 3:
        return None
    med = float(np.median(zps))
    clipped = [z for z in zps if abs(z - med) <= clip]
    if len(clipped) < 3:
        return None
    zp, sc = float(np.median(clipped)), float(np.std(clipped))
    if sc > clip:
        return None
    return zp, len(clipped), sc


def extinction_mag(alt_deg):
    """Magnitude penalty (positive) from airmass at altitude alt_deg."""
    if alt_deg >= 45:
        x = 1.0 / math.sin(math.radians(max(alt_deg, 1)))
    else:
        x = 1.0 / (math.cos(math.radians(90 - alt_deg)) +
                   0.50572 * (96.07995 - alt_deg) ** -1.6364)
    return EXTINCTION_K * max(x, 0)


VIDEO_DET_THRESH = 30.0    # detection threshold above mean background (luma)
VIDEO_MATCH_PX = 40.0      # max distance for matching a detection to a point
VIDEO_SAT_LUMA = 250.0     # aperture peak at/above this = saturated


def _find_event_video(cam_dir):
    """Pick the full-frame event video (largest '*-orig.mp4'), or None."""
    cands = sorted(Path(cam_dir).glob('*-orig.mp4'),
                   key=lambda p: -p.stat().st_size)
    return cands[0] if cands else None


def _read_gray_frames(video_path):
    """Generator of float32 grayscale frames from a video file."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    finally:
        cap.release()


def _video_detections(frames_iter, xymin, xymax, thresh=VIDEO_DET_THRESH):
    """Detect the meteor blob per frame inside a search corridor.

    Two logical passes over the decoded frames: a running mean provides the
    background for blob detection; frames without detections accumulate a
    meteor-free mean image for stellar photometry.

    Returns (detections, mean_img) where detections is a list of
    (frame_idx, x, y, flux, peak, npix).
    """
    from scipy import ndimage
    x0, x1 = int(xymin[0]), int(xymax[0])
    y0, y1 = int(xymin[1]), int(xymax[1])
    acc = clean = None
    n_acc = n_clean = 0
    det = []
    # single streaming pass: keep a running background estimate for the
    # corridor, and a clean running mean that excludes meteor frames
    run_bg = None
    for i, g in enumerate(frames_iter):
        if acc is None:
            acc = np.zeros_like(g)
            clean = np.zeros_like(g)
        acc += g
        n_acc += 1
        if run_bg is None:
            run_bg = acc / n_acc
        gx0, gx1 = max(0, x0), min(g.shape[1], x1)
        gy0, gy1 = max(0, y0), min(g.shape[0], y1)
        sub = (g - run_bg)[gy0:gy1, gx0:gx1]
        m = sub > thresh
        found = False
        if m.any():
            lab, n = ndimage.label(m)
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            k = int(np.argmax(sizes)) + 1
            if sizes[k - 1] >= 5:
                ys, xs = np.nonzero(lab == k)
                cx, cy = gx0 + float(xs.mean()), gy0 + float(ys.mean())
                a = _aperture_flux(g, cx, cy)
                if a:
                    # excess = all positive excess counts in the corridor;
                    # saturated pixels only contribute their clipped value,
                    # so this is a hard lower bound on the incident light
                    excess = float(np.clip(sub, 0, None).sum())
                    det.append((i, cx, cy, a[0], a[2],
                                int(sizes[k - 1]), excess))
                    found = True
        if found:
            pass          # keep the clean background frozen while the
                          # meteor is in view so it cannot mask itself
        else:
            clean += g
            n_clean += 1
        run_bg = (clean / n_clean) if n_clean else acc / n_acc
    mean_img = (clean / n_clean) if n_clean else (acc / n_acc if n_acc else None)
    return det, mean_img


def _video_zeropoint(cam_dir, mean_img, stars, zp_gnom):
    """Zero point for per-frame video aperture flux.

    Transfers the gnomonic-image zero point via stars measurable in both
    domains, else fits a direct zero point from the video mean frame.
    Returns (zp, n_stars, source) or (None, 0, None).
    """
    pto_g = Path(cam_dir) / 'gnomonic.pto'
    gnom = sorted(Path(cam_dir).glob('*-gnomonic.jpg'))
    if mean_img is None or not stars or not pto_g.exists() or not gnom:
        return None, 0, None
    try:
        import pto_mapper
    except ImportError:
        return None, 0, None
    try:
        gpto = pto_mapper.parse_pto_file(str(pto_g))
    except Exception:
        return None, 0, None

    # star positions in the raw video frame: gnomonic px -> source image
    raw_pos = []
    for gx, gy, az, alt, mag in stars:
        try:
            r = pto_mapper.map_pano_to_image(gpto, gx, gy)
        except Exception:
            continue
        if r:
            raw_pos.append((float(r[1]), float(r[2]), alt, mag))
    if not raw_pos:
        return None, 0, None

    f_vid = []
    for rx, ry, alt, mag in raw_pos:
        a = _aperture_flux(mean_img, rx, ry)
        f_vid.append(a[0] if a and a[0] > 30 else None)

    # transfer: zp_v = zp_gnom + 2.5 log10(F_video / peak_gnom)
    if zp_gnom is not None:
        try:
            from PIL import Image
            g_img = np.asarray(Image.open(gnom[0]).convert('L'), dtype=float)
        except Exception:
            g_img = None
        if g_img is not None:
            ratios = []
            for (gx, gy, az, alt, mag), fv in zip(stars, f_vid):
                if fv is None:
                    continue
                pk = _peak_flux(g_img, gx, gy)
                if pk and pk > 3:
                    ratios.append(fv / pk)
            if len(ratios) >= 2:
                return (zp_gnom + 2.5 * math.log10(float(np.median(ratios))),
                        len(ratios), 'transfer')

    # direct fit in the video domain
    zps = [mag + extinction_mag(alt) + 2.5 * math.log10(fv)
           for fv, (rx, ry, alt, mag) in zip(f_vid, raw_pos)
           if fv is not None]
    fit = _fit_zeropoint(zps)
    if fit:
        return fit[0], fit[1], 'direct'
    return None, 0, None


def _video_photometry(cam_dir, positions, timestamps, lat, lon, elev,
                      zp_gnom, stars):
    """Aperture photometry of the trail in the raw event video.

    Recovers photometry for cameras whose event.txt brightness is absent or
    zero-filled (e.g. 'manual = 1' merged tracks). Returns a dict with
    per-point 'flux' (counts above local background, r=6 aperture), 'sat'
    flags and the transferred 'zp', or None when the video cannot be
    located, read, or registered to the trail positions.
    """
    try:
        import cv2  # noqa: F401
    except ImportError:
        return None
    if positions is None or len(positions) == 0:
        return None
    vid = _find_event_video(cam_dir)
    if vid is None:
        return None

    margin = VIDEO_MATCH_PX + 20.0
    xymin = positions.min(axis=0) - margin
    xymax = positions.max(axis=0) + margin
    det, mean_img = _video_detections(_read_gray_frames(vid), xymin, xymax)
    if not det or mean_img is None:
        return None

    # match each trail point to the nearest detection in any frame
    flux, peak, sat, matched = [], [], [], 0
    max_excess = max((d[6] for d in det), default=None)
    for x, y in positions:
        best = None
        for f, cx, cy, fl, pk, npix, ex in det:
            d = math.hypot(cx - x, cy - y)
            if d <= VIDEO_MATCH_PX and (best is None or d < best[0]):
                best = (d, fl, pk)
        if best is None:
            flux.append(None)
            peak.append(None)
            sat.append(False)
        else:
            matched += 1
            flux.append(best[1])
            peak.append(best[2])
            sat.append(bool(best[2] is not None and best[2] >= VIDEO_SAT_LUMA))
    if matched < max(2, len(positions) // 5):
        return None      # positions do not live in this video's frame

    zp, nzp, zp_src = _video_zeropoint(cam_dir, mean_img, stars, zp_gnom)
    res = {'flux': flux, 'peak': peak, 'sat': sat, 'n_matched': matched,
           'n_det': len(det), 'zp': zp, 'zp_stars': nzp,
           'zp_source': zp_src, 'video': vid.name,
           'peak_excess': max_excess}
    # hard lower bound on the brightest instant: the brightest frame's
    # positive excess over the corridor counts every incident photon,
    # with saturated pixels contributing only their clipped value
    if zp is not None and max_excess:
        res['m_app_peak_bound'] = zp - 2.5 * math.log10(max_excess)
    return res


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
    if s_los <= 0 and 0 <= s_trk <= tl:
        return None
    # The closest approach may fall outside the finite luminous segment;
    # clamp to the nearest endpoint so the range stays physically
    # meaningful instead of extrapolating along an infinite line.
    pt = r_start + t_dir * min(max(s_trk, 0.0), tl)
    return float(np.linalg.norm(pt - r_obs)) / 1000.0, s_trk / tl


def camera_photometry(cam_dir, r_start=None, r_end=None, timestamp=None,
                      use_video=True):
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
    try:
        positions = [[float(v) for v in p.split(',')] for p in
                     cfg.get('trail', 'positions').split()]
        positions = np.array(positions) if positions else None
    except Exception:
        positions = None

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
    zp, n_stars, scatter = zp_res if zp_res else (None, 0, None)

    try:
        from PIL import Image
        img = np.asarray(Image.open(gnom[0]).convert('L'), dtype=float)
        bkg = float(np.median(img))
    except Exception:
        bkg = 20.0

    m_app, ranges, fracs = [], [], []
    n_signal = 0
    for b, (az, alt) in zip(brightness, coords):
        peak = b - bkg
        if peak > 0 and zp is not None:
            # observed -> extinction-free apparent magnitude: the meteor's
            # light crosses a long atmospheric path at low altitudes
            m_app.append(zp - 2.5 * math.log10(peak) -
                         extinction_mag(alt))
            n_signal += 1
        else:
            # no detected signal -> no magnitude; a floor would fabricate
            # luminosity (and hence a phantom mass) from an empty trail
            m_app.append(None)
        if r_start is not None and r_end is not None:
            rg = range_to_track(lon, lat, elev, az, alt, r_start, r_end)
            ranges.append(rg[0] if rg else None)
            fracs.append(rg[1] if rg else None)
        else:
            ranges.append(None)
            fracs.append(None)

    warnings = []
    if zp is None:
        warnings.append('no star calibration')
    elif n_stars < 5:
        warnings.append('few calibration stars')
    if scatter is not None and scatter > 0.5:
        warnings.append('large zp scatter')
    n_sat = sum(1 for b in brightness if b >= VIDEO_SAT_LUMA)
    if n_signal == 0:
        warnings.append('no brightness signal')
    elif n_sat > len(brightness) // 3:
        warnings.append('saturated')

    video = None
    if use_video:
        stars = _catalog_stars(pto, timestamp, lat, lon, elev,
                               bright_limit=4.5)
        try:
            video = _video_photometry(cam_dir, positions, timestamps,
                                      lat, lon, elev, zp, stars)
        except Exception as e:
            logging.debug(f'video photometry failed for {cam_dir}: {e}')
    vid_m_app = None
    if video and video.get('zp') is not None:
        vid_m_app = [video['zp'] - 2.5 * math.log10(f) -
                     extinction_mag(coords[i][1]) if f and f > 0
                     else None for i, f in enumerate(video['flux'])
                     if i < len(coords)]
        video['m_app'] = vid_m_app
        if sum(1 for s in video['sat'] if s) > len(video['sat']) // 3:
            if 'saturated' not in warnings:
                warnings.append('saturated')

    alts = [c[1] for c in coords if c[1] is not None]
    if alts and float(np.median(alts)) < 15.0:
        warnings.append('low elevation: large, poorly modeled extinction')

    confidence = 'none' if (n_signal == 0 and vid_m_app is None) else \
        ('low' if (n_stars < 5 or (scatter or 0) > 0.5 or 'saturated'
                   in warnings) else 'medium')
    return {'zp': zp, 'n_stars': n_stars, 'scatter_mag': scatter,
            'bkg': bkg, 'm_app': m_app, 'brightness': brightness,
            'timestamps': timestamps, 'ranges_km': ranges,
            'track_frac': fracs, 'lat': lat, 'lon': lon, 'elev': elev,
            'n_signal': n_signal, 'n_sat': n_sat, 'video': video,
            'confidence': confidence, 'warnings': warnings}


def photometric_mass(m_app, ranges_km, times_s, speeds_ms):
    """Integrate 2*L/(tau*v**2) for a luminous mass-loss proxy [kg].

    Neglects deceleration luminosity; this is not surviving mass.
    The approximate efficiency convention gives tau=0.014 at 20 km/s.
    Camera calibration and efficiency dominate the uncertainty.
    """
    def tau(v_ms):
        return min(0.20, max(0.001, 0.0007 * (v_ms / 1000.0)))
    pts = [(t, m, r, v) for t, m, r, v in
           zip(times_s, m_app, ranges_km, speeds_ms)
           if all(x is not None and np.isfinite(x) for x in (t, m, r, v))
           and r > 0 and v > 0]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda p: p[0])
    # reject corrupt inputs: meteor light curves span seconds, not minutes
    if not 0 < pts[-1][0] - pts[0][0] <= 60:
        return None
    ts = np.array([p[0] for p in pts])
    ts = ts - ts[0]
    lum = np.array([1500.0 * (r / 100.0) ** 2 * 10 ** (-0.4 * m)
                    for t, m, r, v in pts])          # W
    w = np.array([tau(v) * v ** 2 / 2.0 for t, m, r, v in pts])
    energy = trapezoid(lum, ts)                       # J radiated
    eff = trapezoid(w, ts)                            # ∫(tau v²/2)dt
    if not np.isfinite(energy) or not np.isfinite(eff) or eff <= 0:
        return None
    mass = trapezoid(lum / w, ts)
    return float(mass) if np.isfinite(mass) and mass >= 0 else None


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
        # Prefer video aperture photometry (integrated flux); fall back to
        # the stacked-image trail brightness measure.
        vid = r.get('video') or {}
        vid_m = vid.get('m_app')
        n_vid = sum(1 for m in (vid_m or []) if m is not None)
        n_evt = sum(1 for m in r['m_app'] if m is not None)
        if n_vid >= 2:
            m_est = photometric_mass(vid_m, r['ranges_km'],
                                     r['timestamps'], speeds)
            r['mass_source'] = 'video'
        elif n_evt >= 2:
            m_est = photometric_mass(r['m_app'], r['ranges_km'],
                                     r['timestamps'], speeds)
            r['mass_source'] = 'event_txt'
        else:
            m_est = None
            r['mass_source'] = None
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
        vid = c.get('video') or {}
        print(c['cam'], 'zp=', c['zp'], 'stars=', c['n_stars'],
              'conf=', c.get('confidence'), 'warn=', c.get('warnings'),
              'mags=', [round(m, 1) if m is not None else None
                        for m in c['m_app']],
              'vid_mags=', [round(m, 1) if m is not None else None
                            for m in vid.get('m_app', [])],
              'vid_zp=', vid.get('zp'), f"({vid.get('zp_source')})",
              'peak_bound=', vid.get('m_app_peak_bound'),
              'ranges=', [round(x, 1) if x else None
                          for x in c['ranges_km']],
              'm_phot=', c.get('m_phot_kg'),
              'src=', c.get('mass_source'))
    print('combined m_phot_kg:', r['m_phot_kg'])


if __name__ == '__main__':
    main()
