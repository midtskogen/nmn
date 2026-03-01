#!/usr/bin/env python3

import argparse
import configparser
import datetime
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class InfraArrival:
    station: str
    code: str
    lat: float
    lon: float
    elev_m: float
    arrival_ts: float
    path: Path


def _load_track_segment_from_res(event_dir: Path) -> Optional[Tuple[float, float, float, float]]:
    """Return (lon1, lat1, lon2, lat2) for the fitted meteor trajectory if a .res exists."""
    try:
        res_files = sorted(event_dir.glob('obs_*.res'))
        if not res_files:
            res_files = sorted(event_dir.glob('*.res'))
        if not res_files:
            return None
        res_path = res_files[0]
        lons: List[float] = []
        lats: List[float] = []
        with res_path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    lons.append(float(parts[0]))
                    lats.append(float(parts[1]))
                except Exception:
                    continue
        if len(lons) < 2 or len(lats) < 2:
            return None
        return float(lons[0]), float(lats[0]), float(lons[1]), float(lats[1])
    except Exception:
        return None


def _load_per_station_wind_profiles(event_dir: Path, arrivals: List[InfraArrival]) -> List[Dict[str, object]]:
    winds: List[Dict[str, object]] = []
    for a in arrivals:
        try:
            code = str(a.code).strip()
            if not code:
                continue
            p = event_dir / f"wind_profile_{code}.csv"
            if not p.exists():
                continue
            prof = read_wind_profile_csv(p)
            if prof is None:
                continue
            winds.append({'lat': float(a.lat), 'lon': float(a.lon), 'code': code, 'path': str(p), 'profile': prof})
        except Exception:
            continue
    return winds


def _blend_profiles_at(
    wind_profiles: List[Dict[str, object]],
    lat: float,
    lon: float,
    h_m: float,
) -> Tuple[float, float, float]:
    """Blend (temp_k, u_east, v_north) from multiple profiles using inverse-distance weights."""
    if not wind_profiles:
        return 288.15, 0.0, 0.0
    # Approx km conversion at query latitude.
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(float(lat)))
    if km_per_deg_lon < 1e-6:
        km_per_deg_lon = 1e-6

    w_sum = 0.0
    temp_sum = 0.0
    u_sum = 0.0
    v_sum = 0.0
    for wp in wind_profiles:
        try:
            plat = float(wp.get('lat'))
            plon = float(wp.get('lon'))
            prof = wp.get('profile')
            if not isinstance(prof, dict):
                continue
            dx_km = (float(lon) - plon) * km_per_deg_lon
            dy_km = (float(lat) - plat) * km_per_deg_lat
            d2 = dx_km * dx_km + dy_km * dy_km
            # Avoid singularity; 10 km softening.
            w = 1.0 / (d2 + 100.0)
            t_k, u_e, v_n = _interp_profile(prof, float(h_m))
            temp_sum += w * float(t_k)
            u_sum += w * float(u_e)
            v_sum += w * float(v_n)
            w_sum += w
        except Exception:
            continue

    if w_sum <= 0.0:
        return 288.15, 0.0, 0.0
    return temp_sum / w_sum, u_sum / w_sum, v_sum / w_sum


def _huber_weights(resid_s: np.ndarray, k: float = 1.5) -> np.ndarray:
    try:
        r = np.asarray(resid_s, dtype=float)
        if r.size == 0:
            return np.ones_like(r)
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        sigma = float(mad / 0.6745) if mad > 0.0 else float(np.std(r))
        if not math.isfinite(sigma) or sigma <= 1e-6:
            return np.ones_like(r)
        c = float(k) * float(sigma)
        # Weights should depend on deviation from the robust center, not absolute timestamp offset.
        a = np.abs(r - med)
        w = np.ones_like(r)
        m = a > c
        w[m] = c / a[m]
        w = np.clip(w, 0.0, 1.0)
        return w
    except Exception:
        return np.ones_like(np.asarray(resid_s, dtype=float))


def _point_to_segment_distance_km(
    lon: float,
    lat: float,
    seg_lon1: float,
    seg_lat1: float,
    seg_lon2: float,
    seg_lat2: float,
) -> float:
    """Approx distance from (lon,lat) to segment in km using local equirectangular projection."""
    try:
        lat0 = float(0.5 * (seg_lat1 + seg_lat2))
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * math.cos(math.radians(lat0))
        if km_per_deg_lon < 1e-6:
            km_per_deg_lon = 1e-6

        x = (float(lon) - seg_lon1) * km_per_deg_lon
        y = (float(lat) - seg_lat1) * km_per_deg_lat
        x2 = (float(seg_lon2) - seg_lon1) * km_per_deg_lon
        y2 = (float(seg_lat2) - seg_lat1) * km_per_deg_lat

        vv = x2 * x2 + y2 * y2
        if vv < 1e-12:
            return float(math.hypot(x, y))
        t = (x * x2 + y * y2) / vv
        t = max(0.0, min(1.0, float(t)))
        px = t * x2
        py = t * y2
        return float(math.hypot(x - px, y - py))
    except Exception:
        return float('inf')


def _segment_below_horizon(p0: Tuple[float, float, float], p1: Tuple[float, float, float]) -> bool:
    try:
        r_earth = 6371000.0
        min_r = _segment_min_radius_m(p0, p1)
        return bool(min_r < (r_earth + 1.0))
    except Exception:
        return False


def _parse_time(s: str) -> float:
    # infra.txt appears to contain naive local/UTC timestamps; assume UTC.
    # Accept both with and without seconds.
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            continue
    # Try ISO format
    try:
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.timestamp()
    except Exception as e:
        raise ValueError(f"Unrecognized time format: {s}") from e


def read_infra_file(path: Path) -> Optional[InfraArrival]:
    cfg = configparser.ConfigParser()
    try:
        cfg.read(path)
        if not cfg.has_section('arrival') or not cfg.has_option('arrival', 'time'):
            return None

        station = cfg.get('station', 'name', fallback=path.parent.parent.name)
        code = cfg.get('station', 'code', fallback=station[:3].upper()).strip()

        # Note: some provided infra.txt have a bug where 'longitude' was written as 'latitude' a second time.
        lat = cfg.getfloat('station', 'latitude', fallback=np.nan)
        lon = cfg.getfloat('station', 'longitude', fallback=np.nan)

        if not math.isfinite(lon):
            # Heuristic: pick the second 'latitude' line as longitude if present.
            # configparser keeps the last value for duplicate keys, so we need to recover via raw text.
            try:
                lats = []
                for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
                    line = line.strip()
                    if line.lower().startswith('latitude') and '=' in line:
                        _, v = line.split('=', 1)
                        lats.append(v.strip())
                if len(lats) >= 2:
                    lat = float(lats[0])
                    lon = float(lats[1])
            except Exception:
                pass

        elev_m = cfg.getfloat('station', 'elevation', fallback=0.0)
        arrival_ts = _parse_time(cfg.get('arrival', 'time'))

        if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(arrival_ts)):
            return None

        return InfraArrival(
            station=station,
            code=code,
            lat=float(lat),
            lon=float(lon),
            elev_m=float(elev_m),
            arrival_ts=float(arrival_ts),
            path=path,
        )
    except Exception:
        return None


def _read_infra_file_diagnostic(path: Path) -> Tuple[Optional[InfraArrival], str]:
    cfg = configparser.ConfigParser()
    try:
        cfg.read(path)
        if not cfg.has_section('arrival') or not cfg.has_option('arrival', 'time'):
            return None, 'missing [arrival]/time'

        station = cfg.get('station', 'name', fallback=path.parent.parent.name)
        code = cfg.get('station', 'code', fallback=station[:3].upper()).strip()

        lat = cfg.getfloat('station', 'latitude', fallback=np.nan)
        lon = cfg.getfloat('station', 'longitude', fallback=np.nan)
        if not math.isfinite(lon):
            try:
                lats = []
                for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
                    line = line.strip()
                    if line.lower().startswith('latitude') and '=' in line:
                        _, v = line.split('=', 1)
                        lats.append(v.strip())
                if len(lats) >= 2:
                    lat = float(lats[0])
                    lon = float(lats[1])
            except Exception:
                pass

        elev_m = cfg.getfloat('station', 'elevation', fallback=0.0)

        try:
            arrival_ts = _parse_time(cfg.get('arrival', 'time'))
        except Exception as e:
            return None, f'bad arrival time: {e}'

        if not math.isfinite(lat):
            return None, 'bad latitude'
        if not math.isfinite(lon):
            return None, 'bad longitude'

        return InfraArrival(
            station=str(station),
            code=str(code),
            lat=float(lat),
            lon=float(lon),
            elev_m=float(elev_m),
            arrival_ts=float(arrival_ts),
            path=path,
        ), 'ok'
    except Exception as e:
        return None, f'parse exception: {e}'


def _collect_infra_diagnostics(event_dir: Path) -> Tuple[List[InfraArrival], List[Tuple[str, str]]]:
    arrivals: List[InfraArrival] = []
    diag: List[Tuple[str, str]] = []
    for p in sorted(event_dir.glob('*/cam*/infra.txt')):
        a, reason = _read_infra_file_diagnostic(p)
        diag.append((str(p), reason))
        if a is not None:
            arrivals.append(a)
    return arrivals, diag


def read_infra_arrivals(event_dir: Path) -> List[InfraArrival]:
    arrivals: List[InfraArrival] = []
    for p in sorted(event_dir.glob('*/cam*/infra.txt')):
        a = read_infra_file(p)
        if a is not None:
            arrivals.append(a)
    return arrivals


def _wind_to_uv(speed_ms: float, dir_deg: float) -> Tuple[float, float]:
    # WindDir_deg in the CSV is the direction wind is coming FROM (meteorological convention is typical).
    # So wind vector points TOWARDS (dir+180).
    to_dir = math.radians((dir_deg + 180.0) % 360.0)
    u_east = speed_ms * math.sin(to_dir)
    v_north = speed_ms * math.cos(to_dir)
    return u_east, v_north


def read_wind_profile_csv(path: Path) -> Optional[Dict[str, np.ndarray]]:
    try:
        lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
        rows = []
        for ln in lines:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = [p.strip() for p in ln.split(',')]
            if len(parts) < 5:
                continue
            rows.append(parts)
        if not rows:
            return None
        # Sort by altitude (np.interp requires ascending x).
        parsed = []
        for r in rows:
            try:
                parsed.append((float(r[0]), float(r[1]), float(r[3]), float(r[4])))
            except Exception:
                continue
        if not parsed:
            return None
        parsed.sort(key=lambda x: x[0])
        h_m = np.array([p[0] for p in parsed], dtype=float)
        temp_k = np.array([p[1] for p in parsed], dtype=float)
        ws = np.array([p[2] for p in parsed], dtype=float)
        wd = np.array([p[3] for p in parsed], dtype=float)

        # Basic sanity: enforce finite values.
        m = np.isfinite(h_m) & np.isfinite(temp_k) & np.isfinite(ws) & np.isfinite(wd)
        h_m = h_m[m]
        temp_k = temp_k[m]
        ws = ws[m]
        wd = wd[m]
        if h_m.size < 2:
            return None
        u = np.zeros_like(ws)
        v = np.zeros_like(ws)
        for i in range(len(ws)):
            u[i], v[i] = _wind_to_uv(float(ws[i]), float(wd[i]))
        h_min = None
        h_max = None
        try:
            if isinstance(h_m, np.ndarray) and h_m.size:
                h_min = float(h_m[0])
                h_max = float(h_m[-1])
        except Exception:
            h_min = None
            h_max = None
        return {
            'h_m': h_m,
            'temp_k': temp_k,
            'u': u,
            'v': v,
            'ws': ws,
            'wd': wd,
            'h_min': h_min,
            'h_max': h_max,
        }
    except Exception:
        return None


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _ecef_xyz_m(lon_deg: float, lat_deg: float, alt_m: float) -> Tuple[float, float, float]:
    """Simple spherical-Earth ECEF (meters)."""
    r = 6371000.0 + float(alt_m)
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    clat = math.cos(lat)
    x = r * clat * math.cos(lon)
    y = r * clat * math.sin(lon)
    z = r * math.sin(lat)
    return float(x), float(y), float(z)


def _segment_min_radius_m(p0: Tuple[float, float, float], p1: Tuple[float, float, float]) -> float:
    """Minimum distance to origin along the line segment p(t)=p0+t(p1-p0), t∈[0,1]."""
    try:
        x0, y0, z0 = (float(p0[0]), float(p0[1]), float(p0[2]))
        x1, y1, z1 = (float(p1[0]), float(p1[1]), float(p1[2]))
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        vv = dx * dx + dy * dy + dz * dz
        if vv <= 0.0:
            return float(math.sqrt(x0 * x0 + y0 * y0 + z0 * z0))
        t = - (x0 * dx + y0 * dy + z0 * dz) / vv
        t = max(0.0, min(1.0, float(t)))
        x = x0 + t * dx
        y = y0 + t * dy
        z = z0 + t * dz
        return float(math.sqrt(x * x + y * y + z * z))
    except Exception:
        return 0.0


def _bearing_unit_east_north(lon1: float, lat1: float, lon2: float, lat2: float) -> Tuple[float, float]:
    # Unit vector in local EN plane pointing from (lon1,lat1) to (lon2,lat2)
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    brng = math.atan2(y, x)  # from north, east-positive
    e = math.sin(brng)
    n = math.cos(brng)
    return e, n


def _speed_of_sound_ms(temp_k: float) -> float:
    # c = sqrt(gamma * R * T)
    # Use approx: c = 20.05 * sqrt(T)
    try:
        return float(20.05 * math.sqrt(max(1.0, float(temp_k))))
    except Exception:
        return 340.0


def _effective_c_ms(temp_k: float, u_e: float, v_n: float, e_hat: float, n_hat: float) -> float:
    c = _speed_of_sound_ms(temp_k)
    wind_proj = float(u_e) * float(e_hat) + float(v_n) * float(n_hat)
    c_eff = float(c + wind_proj)
    return max(180.0, min(500.0, c_eff))


def _interp_profile(wind: Optional[Dict[str, np.ndarray]], h_m: float) -> Tuple[float, float, float]:
    """Return (temp_k, u_east_ms, v_north_ms) at altitude h_m."""
    if wind is None:
        return 288.15, 0.0, 0.0
    try:
        h = wind['h_m']
        temp = wind.get('temp_k')
        if temp is None:
            temp = np.full_like(h, 288.15, dtype=float)
        u = wind.get('u')
        v = wind.get('v')
        if u is None:
            u = np.zeros_like(h)
        if v is None:
            v = np.zeros_like(h)

        hh = float(h_m)
        # Clamp to range so np.interp behaves.
        h_min = wind.get('h_min')
        h_max = wind.get('h_max')
        if h_min is None or h_max is None:
            # h is sorted by construction in read_wind_profile_csv
            try:
                h_min = float(h[0])
                h_max = float(h[-1])
            except Exception:
                h_min = float(np.min(h))
                h_max = float(np.max(h))
        hh = max(float(h_min), min(float(h_max), hh))
        temp_k = float(np.interp(hh, h, temp))
        u_e = float(np.interp(hh, h, u))
        v_n = float(np.interp(hh, h, v))
        return temp_k, u_e, v_n
    except Exception:
        return 288.15, 0.0, 0.0


def _travel_time_s(
    src_lon: float,
    src_lat: float,
    src_alt_m: float,
    st_lon: float,
    st_lat: float,
    st_alt_m: float,
    wind: object,
    n_steps: int = 80,
) -> Tuple[float, float, float, float]:
    """Compute travel time via altitude-integrated effective sound speed.

    Model: straight line in 3D from source to station.
    At each step, local speed is c(T(h)) + wind(h)·u_hat.
    Returns (t_s, path_m, ground_m, c_eff_mean_ms).
    """
    # Ground distance uses great-circle arc length for reporting and for the horizontal leg in ducting.
    ground_m = _haversine_m(src_lon, src_lat, st_lon, st_lat)

    # Geometry mode:
    # - arc: always use arc-based horizontal + dz (never goes through Earth)
    # - chord: always use ECEF chord (can go through Earth if altitudes are low and range is large)
    # - auto: use chord if it stays above Earth, otherwise fall back to arc
    geom = 'auto'

    dz = float(src_alt_m - st_alt_m)
    path_arc = float(math.hypot(ground_m, dz))

    sx, sy, sz = _ecef_xyz_m(src_lon, src_lat, src_alt_m)
    tx, ty, tz = _ecef_xyz_m(st_lon, st_lat, st_alt_m)
    path_chord = float(math.sqrt((sx - tx) ** 2 + (sy - ty) ** 2 + (sz - tz) ** 2))

    r_earth = 6371000.0
    min_r = _segment_min_radius_m((sx, sy, sz), (tx, ty, tz))
    chord_ok = bool(min_r >= (r_earth + 1.0))

    if geom == 'arc':
        path_m = path_arc
    elif geom == 'chord':
        path_m = path_chord
    else:
        # auto
        path_m = path_chord if chord_ok else path_arc
    if path_m < 1e-3:
        return 0.0, 0.0, ground_m, 340.0

    e_hat, n_hat = _bearing_unit_east_north(src_lon, src_lat, st_lon, st_lat)

    n_steps = int(max(10, n_steps))
    ds = path_m / n_steps
    t_s = 0.0
    c_eff_acc = 0.0
    for k in range(n_steps):
        f = (k + 0.5) / n_steps
        h_here = float(src_alt_m + (st_alt_m - src_alt_m) * f)
        if isinstance(wind, list):
            lat_here = float(src_lat + (st_lat - src_lat) * f)
            lon_here = float(src_lon + (st_lon - src_lon) * f)
            temp_k, u_e, v_n = _blend_profiles_at(wind, lat_here, lon_here, h_here)
        else:
            temp_k, u_e, v_n = _interp_profile(wind if isinstance(wind, dict) else None, h_here)
        c_eff = _effective_c_ms(temp_k, u_e, v_n, e_hat, n_hat)
        t_s += ds / c_eff
        c_eff_acc += c_eff

    c_eff_mean = c_eff_acc / n_steps
    return float(t_s), float(path_m), float(ground_m), float(c_eff_mean)


def _path_altitude_diagnostics(
    src_lon: float,
    src_lat: float,
    src_alt_m: float,
    st_lon: float,
    st_lat: float,
    st_alt_m: float,
) -> Dict[str, float]:
    try:
        h0 = float(src_alt_m)
        h1 = float(st_alt_m)
        h_min_lin = float(min(h0, h1))
        h_max_lin = float(max(h0, h1))
        h_mid_lin = float(0.5 * (h0 + h1))

        r_earth = 6371000.0
        sx, sy, sz = _ecef_xyz_m(src_lon, src_lat, src_alt_m)
        tx, ty, tz = _ecef_xyz_m(st_lon, st_lat, st_alt_m)
        dx = float(tx - sx)
        dy = float(ty - sy)
        dz = float(tz - sz)
        vv = float(dx * dx + dy * dy + dz * dz)
        if vv <= 0.0:
            vv = 1.0

        # Midpoint altitude along the chord
        xm = float(sx + 0.5 * dx)
        ym = float(sy + 0.5 * dy)
        zm = float(sz + 0.5 * dz)
        rm = float(math.sqrt(xm * xm + ym * ym + zm * zm))
        h_mid_chord = float(rm - r_earth)

        # Interior minimum altitude along the chord (exclude endpoints).
        # This is more informative than the full segment minimum when one endpoint is at ground.
        t_star = - (float(sx) * dx + float(sy) * dy + float(sz) * dz) / vv
        eps = 1e-3
        t_star = max(eps, min(1.0 - eps, float(t_star)))
        xi = float(sx + t_star * dx)
        yi = float(sy + t_star * dy)
        zi = float(sz + t_star * dz)
        ri = float(math.sqrt(xi * xi + yi * yi + zi * zi))
        h_min_chord_inner = float(ri - r_earth)

        # Full segment minimum (includes endpoints) for reference.
        min_r = _segment_min_radius_m((sx, sy, sz), (tx, ty, tz))
        h_min_chord = float(min_r - r_earth)

        return {
            'h_min_lin_m': h_min_lin,
            'h_mid_lin_m': h_mid_lin,
            'h_max_lin_m': h_max_lin,
            'h_min_chord_m': h_min_chord,
            'h_mid_chord_m': h_mid_chord,
            'h_min_chord_inner_m': h_min_chord_inner,
        }
    except Exception:
        return {
            'h_min_lin_m': float('nan'),
            'h_mid_lin_m': float('nan'),
            'h_max_lin_m': float('nan'),
            'h_min_chord_m': float('nan'),
            'h_mid_chord_m': float('nan'),
            'h_min_chord_inner_m': float('nan'),
        }


def _destination_latlon(lat_deg: float, lon_deg: float, bearing_deg: float, distance_km: float) -> Tuple[float, float]:
    # Sufficient for our ring visualization. Not used for precise geodesy.
    lat = float(lat_deg)
    lon = float(lon_deg)
    br = math.radians(float(bearing_deg))
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
    if km_per_deg_lon < 1e-3:
        km_per_deg_lon = 1e-3
    dlat = (distance_km * math.cos(br)) / km_per_deg_lat
    dlon = (distance_km * math.sin(br)) / km_per_deg_lon
    return lat + dlat, lon + dlon


def compute_isochrone_ring(
    station_lat: float,
    station_lon: float,
    station_alt_m: float,
    target_time_s: float,
    ring_alt_m: float,
    wind: Optional[Dict[str, np.ndarray]],
    c_scale: float = 1.0,
    n_az: int = 72,
    r_max_km: float = 1800.0,
) -> Optional[Dict[str, List[float]]]:
    if not math.isfinite(target_time_s) or target_time_s <= 0:
        return None

    lats: List[float] = []
    lons: List[float] = []
    for j in range(n_az):
        az = (360.0 * j) / n_az

        # Bracket a root for f(r)=travel_time(r)-target
        def f(r_km: float) -> float:
            cand_lat, cand_lon = _destination_latlon(station_lat, station_lon, az, r_km)
            tt, _, _, _ = _travel_time_s(cand_lon, cand_lat, ring_alt_m, station_lon, station_lat, station_alt_m, wind)
            try:
                tt = float(tt) / float(c_scale)
            except Exception:
                pass
            return float(tt) - float(target_time_s)

        f0 = f(0.0)
        f1 = f(r_max_km)
        if not (math.isfinite(f0) and math.isfinite(f1)):
            continue
        if f0 > 0:
            # Even at zero range we're slower than the required time -> no solution for this az.
            continue
        if f1 < 0:
            # Still too fast even at r_max -> extend would be needed.
            continue

        lo, hi = 0.0, r_max_km
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if not math.isfinite(fm):
                break
            if fm > 0:
                hi = mid
            else:
                lo = mid
        r_sol = 0.5 * (lo + hi)
        cand_lat, cand_lon = _destination_latlon(station_lat, station_lon, az, r_sol)
        lats.append(float(cand_lat))
        lons.append(float(cand_lon))

    if len(lats) < max(8, n_az // 4):
        return None
    # close ring
    lats.append(lats[0])
    lons.append(lons[0])
    return {'lats': lats, 'lons': lons}


def fit_infrasound_source(
    arrivals: List[InfraArrival],
    wind_profile: Optional[Dict[str, np.ndarray]],
    initial_guess: Tuple[float, float, float],
    search_km: float = 250.0,
    n_samples: int = 4000,
    seed: int = 0,
    max_residual_s: float = 10.0,
    track_segment: Optional[Tuple[float, float, float, float]] = None,
    track_prior_s_per_km: float = 0.02,
    t0_prior_unix: Optional[float] = None,
    t0_prior_s_per_s: float = 0.0,
    progress: bool = False,
) -> Optional[Dict]:
    if len(arrivals) < 3:
        return None

    rng = np.random.default_rng(seed)

    init_lat, init_lon, init_alt_m = initial_guess

    # Sample candidate source locations around initial guess.
    # For simplicity use lat/lon perturbations scaled by km.
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(init_lat))
    if km_per_deg_lon < 1e-3:
        km_per_deg_lon = 1e-3

    best = None

    # Use a cheap travel-time evaluation during sampling, then recompute the final best with more steps.
    sample_n_steps = 20
    final_n_steps = 80

    # Pre-extract station arrays for speed
    st_lats = np.array([a.lat for a in arrivals], dtype=float)
    st_lons = np.array([a.lon for a in arrivals], dtype=float)
    st_elev = np.array([a.elev_m for a in arrivals], dtype=float)
    t_arr = np.array([a.arrival_ts for a in arrivals], dtype=float)

    # Precompute station ECEF for horizon checks
    st_xyz = [
        _ecef_xyz_m(float(arrivals[i].lon), float(arrivals[i].lat), float(arrivals[i].elev_m))
        for i in range(len(arrivals))
    ]

    def _evaluate_candidate(args: Tuple[float, float, float]) -> Optional[Dict]:
        try:
            src_lat, src_lon, src_alt_m = args

            # Horizon/line-of-sight filter: drop stations where the chord intersects Earth.
            src_xyz = _ecef_xyz_m(float(src_lon), float(src_lat), float(src_alt_m))
            vis_idx = []
            for i in range(len(st_xyz)):
                if not _segment_below_horizon(src_xyz, st_xyz[i]):
                    vis_idx.append(i)
            if len(vis_idx) < 3:
                return None

            # Base travel times (no global scaling)
            t_travel_base = np.zeros(len(vis_idx), dtype=float)
            for j, i in enumerate(vis_idx):
                tt, _, _, _ = _travel_time_s(
                    float(src_lon), float(src_lat), float(src_alt_m),
                    float(st_lons[i]), float(st_lats[i]), float(st_elev[i]),
                    wind_profile,
                    n_steps=sample_n_steps,
                )

                if not math.isfinite(float(tt)) or float(tt) <= 0.0:
                    return None
                t_travel_base[j] = float(tt)

            # During sampling, c_scale is fixed at 1.0.
            t_travel = t_travel_base
            t_arr_v = t_arr[np.asarray(vis_idx, dtype=int)]
            t0 = float(np.mean(t_arr_v - t_travel))
            resid = (t0 + t_travel) - t_arr_v
            w = _huber_weights(resid)
            for _ in range(2):
                sw = float(np.sum(w))
                if not math.isfinite(sw) or sw <= 1e-9:
                    break
                t0 = float(np.sum(w * (t_arr_v - t_travel)) / sw)
                resid = (t0 + t_travel) - t_arr_v
                w = _huber_weights(resid)
            sw = float(np.sum(w))
            if sw <= 1e-9:
                sw = float(len(resid))
                w = np.ones_like(resid)
            rms = float(np.sqrt(np.sum(w * (resid ** 2)) / sw))
            rms_all = float(np.sqrt(float(np.mean(resid ** 2))))

            score = float(rms)
            if track_segment is not None and math.isfinite(track_prior_s_per_km) and track_prior_s_per_km > 0.0:
                d_track_km = _point_to_segment_distance_km(
                    float(src_lon), float(src_lat),
                    track_segment[0], track_segment[1], track_segment[2], track_segment[3],
                )
                if math.isfinite(d_track_km):
                    score = float(score + track_prior_s_per_km * d_track_km)

            # Optional timing prior penalty (used only in refinement stage).
            try:
                if t0_prior_unix is not None and math.isfinite(float(t0_prior_unix)) and math.isfinite(float(t0_prior_s_per_s)) and float(t0_prior_s_per_s) > 0.0:
                    score = float(score + float(t0_prior_s_per_s) * abs(float(t0) - float(t0_prior_unix)))
            except Exception:
                pass

            return {
                'lat': float(src_lat),
                'lon': float(src_lon),
                'elev_m': float(src_alt_m),
                't0_unix': float(t0),
                'rms_s': float(rms),
                'rms_all_s': float(rms_all),
                'score_s': float(score),
                'c_scale': 1.0,
                'residuals_s': np.asarray(resid, dtype=float).tolist(),
                'weights': np.asarray(w, dtype=float).tolist(),
                'visible_station_indices': [int(x) for x in vis_idx],
            }
        except Exception:
            return None

    # Deterministically generate candidates in the main process.
    candidates: List[Tuple[float, float, float]] = []
    for _ in range(int(n_samples)):
        r = search_km * math.sqrt(float(rng.random()))
        theta = 2.0 * math.pi * float(rng.random())
        dlat = (r * math.cos(theta)) / km_per_deg_lat
        dlon = (r * math.sin(theta)) / km_per_deg_lon
        src_lat = init_lat + dlat
        src_lon = init_lon + dlon
        src_alt_m = float(rng.uniform(0.0, 80000.0)) if init_alt_m <= 0 else float(np.clip(init_alt_m + rng.normal(0, 15000.0), 0.0, 80000.0))
        candidates.append((float(src_lat), float(src_lon), float(src_alt_m)))

    step = max(1, int(len(candidates) // 10))
    for j, cand in enumerate(candidates):
        if progress and (j == 0 or (j + 1) % step == 0 or (j + 1) == len(candidates)):
            try:
                pct = int(round(100.0 * (j + 1) / max(1, len(candidates))))
                best_rms = None if best is None else best.get('rms_s')
                print(f"infrasound_fit sampling progress {pct}% best_rms_s={best_rms}", flush=True)
            except Exception:
                pass
        res = _evaluate_candidate(cand)
        if res is None:
            continue
        if best is None or float(res.get('score_s', float('inf'))) < float(best.get('score_s', float('inf'))):
            best = res

    if best is None:
        return None

    # Recompute best with higher-resolution travel time evaluation.
    try:
        vis_idx_best = best.get('visible_station_indices')
        if not isinstance(vis_idx_best, list):
            vis_idx_best = list(range(len(st_lats)))
        if len(vis_idx_best) < 3:
            return None

        tt_best = np.zeros(len(vis_idx_best), dtype=float)
        for j, i in enumerate(vis_idx_best):
            tt, _, _, _ = _travel_time_s(
                float(best['lon']), float(best['lat']), float(best['elev_m']),
                float(st_lons[i]), float(st_lats[i]), float(st_elev[i]),
                wind_profile,
                n_steps=final_n_steps,
            )
            tt_best[j] = float(tt)

        t_arr_v = t_arr[np.asarray(vis_idx_best, dtype=int)]
        t0 = float(np.mean(t_arr_v - tt_best))
        resid = (t0 + tt_best) - t_arr_v
        w = _huber_weights(resid)
        for _ in range(2):
            sw = float(np.sum(w))
            if not math.isfinite(sw) or sw <= 1e-9:
                break
            t0 = float(np.sum(w * (t_arr_v - tt_best)) / sw)
            resid = (t0 + tt_best) - t_arr_v
            w = _huber_weights(resid)
        sw = float(np.sum(w))
        if sw <= 1e-9:
            sw = float(len(resid))
            w = np.ones_like(resid)
        rms = float(np.sqrt(np.sum(w * (resid ** 2)) / sw))
        rms_all = float(np.sqrt(float(np.mean(resid ** 2))))

        score = float(rms)
        if track_segment is not None and math.isfinite(track_prior_s_per_km) and track_prior_s_per_km > 0.0:
            d_track_km = _point_to_segment_distance_km(
                float(best['lon']), float(best['lat']),
                track_segment[0], track_segment[1], track_segment[2], track_segment[3],
            )
            if math.isfinite(d_track_km):
                score = float(rms + track_prior_s_per_km * d_track_km)

        best['t0_unix'] = float(t0)
        best['rms_s'] = float(rms)
        best['rms_all_s'] = float(rms_all)
        best['score_s'] = float(score)
        best['residuals_s'] = resid.tolist()
        best['weights'] = w.tolist()
    except Exception:
        pass

    # Add per-station predicted times
    preds = []
    horizon_dropped = []
    src_xyz_best = _ecef_xyz_m(float(best['lon']), float(best['lat']), float(best['elev_m']))
    vis_idx_best = best.get('visible_station_indices')
    if not isinstance(vis_idx_best, list):
        vis_idx_best = list(range(len(arrivals)))
    vis_map = {int(idx): int(j) for j, idx in enumerate([int(x) for x in vis_idx_best])}

    for i, a in enumerate(arrivals):
        below = False
        try:
            below = _segment_below_horizon(src_xyz_best, st_xyz[i])
        except Exception:
            below = False
        if below or (vis_map and (i not in vis_map)):
            horizon_dropped.append(str(a.code))
            preds.append({
                'station': a.station,
                'code': a.code,
                'lat': a.lat,
                'lon': a.lon,
                'elev_m': a.elev_m,
                'below_horizon': True,
                'arrival_unix': float(a.arrival_ts),
                'predicted_unix': float('nan'),
                'residual_s': float('nan'),
                'travel_time_s': float('nan'),
                'distance_km': float('nan'),
                'path_km': float('nan'),
                'c_eff_ms': float('nan'),
                'celerity_km_s': None,
                'weight': 0.0,
                'h_min_lin_m': float('nan'),
                'h_mid_lin_m': float('nan'),
                'h_max_lin_m': float('nan'),
                'h_min_chord_m': float('nan'),
                'h_mid_chord_m': float('nan'),
                'h_min_chord_inner_m': float('nan'),
            })
            continue

        t_travel, path_m, d_m, c_eff = _travel_time_s(
            best['lon'], best['lat'], best['elev_m'],
            a.lon, a.lat, a.elev_m,
            wind_profile,
            n_steps=final_n_steps,
        )
        diag = _path_altitude_diagnostics(
            float(best['lon']), float(best['lat']), float(best['elev_m']),
            float(a.lon), float(a.lat), float(a.elev_m),
        )
        celerity_km_s = None
        try:
            if t_travel and float(t_travel) > 1e-6:
                celerity_km_s = float((d_m / 1000.0) / float(t_travel))
        except Exception:
            celerity_km_s = None
        w_i = None
        try:
            j = int(vis_map.get(i))
            if j is not None:
                w_i = float((best.get('weights') or [])[j])
        except Exception:
            w_i = None
        preds.append({
            'station': a.station,
            'code': a.code,
            'lat': a.lat,
            'lon': a.lon,
            'elev_m': a.elev_m,
            'below_horizon': False,
            'arrival_unix': float(a.arrival_ts),
            'predicted_unix': float(best['t0_unix'] + float(t_travel)),
            'residual_s': float((best['t0_unix'] + float(t_travel)) - float(a.arrival_ts)),
            'travel_time_s': float(t_travel),
            'distance_km': d_m / 1000.0,
            'path_km': path_m / 1000.0,
            'c_eff_ms': c_eff,
            'celerity_km_s': celerity_km_s,
            'weight': w_i,
            'h_min_lin_m': float(diag.get('h_min_lin_m', float('nan'))),
            'h_mid_lin_m': float(diag.get('h_mid_lin_m', float('nan'))),
            'h_max_lin_m': float(diag.get('h_max_lin_m', float('nan'))),
            'h_min_chord_m': float(diag.get('h_min_chord_m', float('nan'))),
            'h_mid_chord_m': float(diag.get('h_mid_chord_m', float('nan'))),
            'h_min_chord_inner_m': float(diag.get('h_min_chord_inner_m', float('nan'))),
        })

    best['stations'] = preds
    try:
        best['horizon_dropped_codes'] = [str(x) for x in horizon_dropped]
    except Exception:
        pass

    # Wind-corrected station isochrones (rings) at the fitted source altitude.
    # These are useful for visualizing the timing constraint and will be non-circular when wind is present.
    try:
        rings = []
        for idx, a in enumerate(arrivals):
            try:
                if bool(preds[idx].get('below_horizon')):
                    continue
            except Exception:
                pass
            target_dt = float(a.arrival_ts - best['t0_unix'])
            ring = compute_isochrone_ring(
                station_lat=a.lat,
                station_lon=a.lon,
                station_alt_m=a.elev_m,
                target_time_s=target_dt,
                ring_alt_m=float(best['elev_m']),
                wind=wind_profile,
                c_scale=float(best.get('c_scale', 1.0) or 1.0),
            )
            if ring is None:
                continue
            rings.append({
                'code': a.code,
                'station': a.station,
                'station_lat': float(a.lat),
                'station_lon': float(a.lon),
                'station_elev_m': float(a.elev_m),
                'alt_km': float(best['elev_m']) / 1000.0,
                'lats': ring['lats'],
                'lons': ring['lons'],
            })
        if rings:
            best['isochrones'] = rings
    except Exception:
        pass
    return best


def fit_infrasound_source_with_outlier_rejection(
    arrivals: List[InfraArrival],
    wind_profile: Optional[Dict[str, np.ndarray]],
    initial_guess: Tuple[float, float, float],
    search_km: float = 250.0,
    n_samples: int = 4000,
    seed: int = 0,
    max_residual_s: float = 10.0,
    track_segment: Optional[Tuple[float, float, float, float]] = None,
    track_prior_s_per_km: float = 0.02,
    t0_prior_unix: Optional[float] = None,
    t0_prior_s_per_s: float = 0.0,
    progress: bool = False,
) -> Optional[Dict]:
    if len(arrivals) < 3:
        return None

    if progress:
        try:
            inl = ','.join([a.code for a in arrivals])
            print(f"infrasound_fit round 1/1 inliers={len(arrivals)} [{inl}] outliers=0 []", flush=True)
        except Exception:
            pass

    res = fit_infrasound_source(
        arrivals,
        wind_profile,
        initial_guess,
        search_km=search_km,
        n_samples=n_samples,
        seed=seed,
        max_residual_s=max_residual_s,
        track_segment=track_segment,
        track_prior_s_per_km=track_prior_s_per_km,
        t0_prior_unix=t0_prior_unix,
        t0_prior_s_per_s=t0_prior_s_per_s,
        progress=progress,
    )
    if res is None:
        return None

    # Preserve visual timing prior in the result for reporting/debugging.
    try:
        res['t0_prior_unix'] = t0_prior_unix
        res['t0_prior_s_per_s'] = float(t0_prior_s_per_s)
    except Exception:
        pass

    hard_outlier_s = float(max(60.0, 3.0 * float(max_residual_s)))
    inliers: List[str] = []
    outliers: List[str] = []
    abs_resid_all: List[float] = []
    abs_resid_inliers: List[float] = []
    for s in (res.get('stations') or []):
        try:
            if bool(s.get('below_horizon')):
                continue
            code = str(s.get('code', '')).strip()
            r = float(s.get('residual_s'))
            w = s.get('weight')
            wv = float(w) if w is not None else 1.0
            ar = abs(float(r))
            if math.isfinite(ar):
                abs_resid_all.append(float(ar))

            is_inlier = (ar <= hard_outlier_s) and (wv >= 0.2)
            if is_inlier:
                inliers.append(code)
                if math.isfinite(ar):
                    abs_resid_inliers.append(float(ar))
            else:
                outliers.append(code)
        except Exception:
            continue

    res['inliers'] = inliers
    res['outliers'] = outliers
    res['n_inliers'] = int(len(inliers))
    res['n_outliers'] = int(len(outliers))
    res['score_adj_s'] = float(res.get('score_s', res.get('rms_s', float('inf'))))
    res['max_residual_threshold_s'] = float(max_residual_s)
    res['hard_outlier_s'] = float(hard_outlier_s)
    res['max_abs_residual_all_s'] = float(max(abs_resid_all) if abs_resid_all else float('nan'))
    # Preserve backward-compatible field name used in reports: max abs residual among inliers.
    res['max_residual_s'] = float(max(abs_resid_inliers) if abs_resid_inliers else float('nan'))

    try:
        inlier_set = set(inliers)
        stations_all = []
        for s in (res.get('stations') or []):
            s2 = dict(s)
            try:
                s2['is_inlier'] = (str(s2.get('code', '')).strip() in inlier_set)
            except Exception:
                s2['is_inlier'] = True
            stations_all.append(s2)
        res['stations_all'] = stations_all
    except Exception:
        pass

    if progress:
        try:
            base_score = float(res.get('score_s', res.get('rms_s', float('inf'))))
            t0u = float(res.get('t0_unix', float('nan')))
            t0s = ''
            try:
                if math.isfinite(t0u):
                    t0s = datetime.datetime.fromtimestamp(t0u, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                t0s = ''
            t0_part = (f"t0=\"{t0s}\" " if t0s else '') + f"t0_unix={t0u:.3f} "

            t0p = res.get('t0_prior_unix')
            t0ps = ''
            try:
                if t0p is not None and math.isfinite(float(t0p)):
                    t0ps = datetime.datetime.fromtimestamp(float(t0p), tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                t0ps = ''
            if t0ps:
                dt_s = None
                try:
                    dt_s = float(t0u) - float(t0p)
                except Exception:
                    dt_s = None
                if dt_s is not None and math.isfinite(float(dt_s)):
                    t0_part = t0_part + f"t0_prior=\"{t0ps}\" t0_prior_unix={float(t0p):.3f} dt_s={float(dt_s):+.3f} "
                else:
                    t0_part = t0_part + f"t0_prior=\"{t0ps}\" t0_prior_unix={float(t0p):.3f} "
            msg1 = (
                "infrasound_fit round_result "
                f"lat={float(res.get('lat', float('nan'))):.6f} "
                f"lon={float(res.get('lon', float('nan'))):.6f} "
                f"elev_m={float(res.get('elev_m', float('nan'))):.1f} "
                + t0_part
                + f"rms_s={float(res.get('rms_s', float('nan'))):.3f} "
                + f"score_s={base_score:.3f} "
                + f"score_adj_s={float(res.get('score_adj_s', base_score)):.3f} "
                + f"n_inliers={int(res.get('n_inliers', len(arrivals)))} "
                + f"n_outliers={int(res.get('n_outliers', 0))}"
            )
            print(msg1, flush=True)

            msg2 = (
                "infrasound_fit selected "
                + t0_part
                + f"score_adj_s={float(res.get('score_adj_s', base_score)):.3f} "
                + f"n_total={int(len(arrivals))} "
                + f"n_used={int(res.get('n_inliers', len(arrivals)))}"
            )
            print(msg2, flush=True)
        except Exception:
            pass

    return res


def run_fit_for_event_dir(
    event_dir: Path,
    initial_guess: Optional[Tuple[float, float, float]] = None,
    progress: bool = False,
) -> Optional[Dict]:
    arrivals = read_infra_arrivals(event_dir)
    if len(arrivals) < 3:
        return None

    wind_csv = event_dir / 'wind_profile.csv'
    wind_profile = read_wind_profile_csv(wind_csv) if wind_csv.exists() else None
    per_station_winds = _load_per_station_wind_profiles(event_dir, arrivals)
    # Prefer per-station profiles if any exist; otherwise use the single event wind profile.
    wind_for_fit: object = per_station_winds if per_station_winds else wind_profile

    # Visual timing prior from event.txt (if present): t_prior = start + 0.8*(end-start)
    t0_prior_unix = None
    visual_start_unix = None
    visual_end_unix = None
    visual_event_txt = None
    visual_event_txt_candidates = 0
    try:
        def _parse_unix_in_parens(line: str) -> Optional[float]:
            if '(' not in line or ')' not in line:
                return None
            try:
                inner = line.split('(', 1)[1].split(')', 1)[0]
                return float(inner.strip())
            except Exception:
                return None

        event_txts = []
        try:
            event_txts = sorted(event_dir.rglob('event.txt'))
        except Exception:
            event_txts = sorted(event_dir.glob('*/cam*/event.txt'))
        visual_event_txt_candidates = int(len(event_txts))

        for ep in event_txts:
            try:
                txt = ep.read_text(encoding='utf-8', errors='ignore').splitlines()
                t_start = None
                t_end = None
                for ln in txt:
                    l = ln.strip().lower()
                    # Only use the [video] start/end lines (avoid [summary] timestamp etc.)
                    if l.startswith('start') and '=' in l:
                        v = ln.split('=', 1)[1].strip()
                        ts = _parse_unix_in_parens(v)
                        if ts is not None:
                            t_start = ts
                    elif l.startswith('end') and '=' in l:
                        v = ln.split('=', 1)[1].strip()
                        te = _parse_unix_in_parens(v)
                        if te is not None:
                            t_end = te
                if t_start is not None and t_end is not None and math.isfinite(float(t_start)) and math.isfinite(float(t_end)) and float(t_end) > float(t_start):
                    visual_event_txt = str(ep)
                    visual_start_unix = float(t_start)
                    visual_end_unix = float(t_end)
                    t0_prior_unix = float(t_start) + 0.8 * (float(t_end) - float(t_start))
                    break
            except Exception:
                continue
    except Exception:
        t0_prior_unix = None

    # Baseline fit ignores visual timing. Refinement stage may use it.
    t0_prior_s_per_s = 0.0

    track_segment = _load_track_segment_from_res(event_dir)

    if initial_guess is None:
        # Default guess: track midpoint if available, otherwise mean station position. Altitude 30 km.
        if track_segment is not None:
            lon0 = float(0.5 * (track_segment[0] + track_segment[2]))
            lat0 = float(0.5 * (track_segment[1] + track_segment[3]))
        else:
            lat0 = float(np.mean([a.lat for a in arrivals]))
            lon0 = float(np.mean([a.lon for a in arrivals]))
        initial_guess = (float(lat0), float(lon0), 30000.0)

    seed = 0
    # Try a sequence of increasingly relaxed configs so we still get a best-effort solution
    # in imperfect real-world data.
    configs = [
        {'search_km': 250.0, 'n_samples': 4000, 'max_residual_s': 10.0},
        {'search_km': 350.0, 'n_samples': 6000, 'max_residual_s': 20.0},
        {'search_km': 500.0, 'n_samples': 8000, 'max_residual_s': 35.0},
    ]

    result = None
    last_err = None
    for idx, cfg in enumerate(configs):
        try:
            cand = fit_infrasound_source_with_outlier_rejection(
                arrivals,
                wind_for_fit,
                initial_guess,
                search_km=float(cfg['search_km']),
                n_samples=int(cfg['n_samples']),
                seed=seed + (idx * 100),
                max_residual_s=float(cfg['max_residual_s']),
                track_segment=track_segment,
                track_prior_s_per_km=0.02,
                t0_prior_unix=t0_prior_unix,
                t0_prior_s_per_s=t0_prior_s_per_s,
                progress=progress,
            )
            if cand is not None:
                result = cand
                result['relaxed_fit'] = (idx != 0)
                break
        except Exception as e:
            last_err = str(e)
            continue

    if result is None:
        return None

    # Optional refinement stage: search locally around the baseline solution and
    # include a penalty for deviation from the visual timing prior.
    refined = None
    try:
        if t0_prior_unix is not None and math.isfinite(float(t0_prior_unix)):
            base_lat = float(result.get('lat'))
            base_lon = float(result.get('lon'))
            base_alt = float(result.get('elev_m'))
            base_rms = float(result.get('rms_s', float('inf')))
            base_rms_all = float(result.get('rms_all_s', float('inf')))
            base_max_all = float(result.get('max_abs_residual_all_s', float('nan')))
            base_p90_all = float(result.get('p90_abs_residual_all_s', float('nan')))
            base_n_out = int(result.get('n_outliers', 0) or 0)
            base_worst_code = result.get('worst_station_code')
            base_worst_abs = result.get('worst_station_abs_residual_s')
            base_t0 = float(result.get('t0_unix', float('nan')))
            base_dt = float(base_t0 - float(t0_prior_unix)) if math.isfinite(base_t0) else float('nan')
            base_n_out_all = None
            try:
                hard0 = float(result.get('hard_outlier_s', float('nan')))
                if math.isfinite(hard0):
                    st0_all = (result.get('stations') or [])
                    st0 = []
                    rr_list = []
                    for s in st0_all:
                        try:
                            if bool(s.get('below_horizon')):
                                continue
                            r0 = float(s.get('residual_s'))
                            if not math.isfinite(r0):
                                continue
                            st0.append(s)
                            rr_list.append(r0)
                        except Exception:
                            continue

                    rr0 = np.asarray(rr_list, dtype=float)
                    if rr0.size:
                        base_n_out_all = int(np.sum(np.abs(rr0) > float(hard0)))

                        # Fill distribution metrics if the base fit result didn't populate them.
                        if (not math.isfinite(base_p90_all)) or base_p90_all < 0:
                            base_p90_all = float(np.percentile(np.abs(rr0), 90))

                        if base_worst_code in (None, '') or (base_worst_abs is None) or (not math.isfinite(float(base_worst_abs))):
                            k_worst = int(np.argmax(np.abs(rr0)))
                            base_worst_abs = float(abs(rr0[k_worst]))
                            try:
                                base_worst_code = str(st0[k_worst].get('code'))
                            except Exception:
                                base_worst_code = None
            except Exception:
                base_n_out_all = None

            def _eval_on_all(sol: Dict) -> Dict:
                # Recompute per-station predictions/residuals using *all* arrivals so outliers can influence selection.
                try:
                    t0u = float(sol.get('t0_unix', float('nan')))
                    lon_s = float(sol.get('lon', float('nan')))
                    lat_s = float(sol.get('lat', float('nan')))
                    alt_s = float(sol.get('elev_m', float('nan')))
                    if not (math.isfinite(t0u) and math.isfinite(lon_s) and math.isfinite(lat_s) and math.isfinite(alt_s)):
                        return sol
                except Exception:
                    return sol

                final_n_steps_eval = 80
                stations_all = []
                resid_all = []
                src_xyz = _ecef_xyz_m(float(lon_s), float(lat_s), float(alt_s))
                horizon_dropped = []
                stations_used = []
                for a in arrivals:
                    try:
                        st_xyz0 = _ecef_xyz_m(float(a.lon), float(a.lat), float(a.elev_m))
                        below = _segment_below_horizon(src_xyz, st_xyz0)
                        if below:
                            horizon_dropped.append(str(a.code))
                            stations_all.append({
                                'station': a.station,
                                'code': a.code,
                                'lat': a.lat,
                                'lon': a.lon,
                                'elev_m': a.elev_m,
                                'below_horizon': True,
                                'arrival_unix': float(a.arrival_ts),
                                'predicted_unix': float('nan'),
                                'residual_s': float('nan'),
                                'travel_time_s': float('nan'),
                                'distance_km': float('nan'),
                                'path_km': float('nan'),
                                'c_eff_ms': float('nan'),
                            })
                            continue
                        t_travel, path_m, d_m, c_eff = _travel_time_s(
                            lon_s, lat_s, alt_s,
                            float(a.lon), float(a.lat), float(a.elev_m),
                            wind_for_fit,
                            n_steps=final_n_steps_eval,
                        )
                        pred = float(t0u + float(t_travel))
                        r = float(pred - float(a.arrival_ts))
                        resid_all.append(float(r))
                        st_rec = {
                            'station': a.station,
                            'code': a.code,
                            'lat': a.lat,
                            'lon': a.lon,
                            'elev_m': a.elev_m,
                            'below_horizon': False,
                            'arrival_unix': float(a.arrival_ts),
                            'predicted_unix': float(pred),
                            'residual_s': float(r),
                            'travel_time_s': float(t_travel),
                            'distance_km': float(d_m) / 1000.0,
                            'path_km': float(path_m) / 1000.0,
                            'c_eff_ms': float(c_eff),
                        }
                        stations_all.append(st_rec)
                        stations_used.append(st_rec)
                    except Exception:
                        continue

                try:
                    rr = np.asarray(resid_all, dtype=float)
                    if rr.size:
                        sol['rms_all_s'] = float(np.sqrt(float(np.mean(rr ** 2))))
                        sol['max_abs_residual_all_s'] = float(np.max(np.abs(rr)))
                        try:
                            sol['p90_abs_residual_all_s'] = float(np.percentile(np.abs(rr), 90.0))
                        except Exception:
                            pass
                        try:
                            idx = int(np.argmax(np.abs(rr)))
                            if 0 <= idx < len(stations_used):
                                sol['worst_station_code'] = str(stations_used[idx].get('code'))
                                sol['worst_station_abs_residual_s'] = float(abs(float(rr[idx])))
                        except Exception:
                            pass
                    sol['stations'] = stations_all
                    sol['n_stations'] = int(len(stations_all))
                    try:
                        sol['horizon_dropped_codes'] = [str(x) for x in horizon_dropped]
                    except Exception:
                        pass

                    # Derive an all-station outlier count using the same hard threshold rule.
                    hard = None
                    try:
                        hard = float(sol.get('hard_outlier_s', float('nan')))
                    except Exception:
                        hard = None
                    if hard is None or not math.isfinite(float(hard)):
                        try:
                            thr = float(sol.get('max_residual_threshold_s', 10.0) or 10.0)
                        except Exception:
                            thr = 10.0
                        hard = float(max(60.0, 3.0 * float(thr)))
                    sol['hard_outlier_s'] = float(hard)
                    try:
                        n_out_all = int(np.sum(np.abs(rr) > float(hard)))
                        sol['n_outliers_all'] = n_out_all
                        sol['n_inliers_all'] = int(rr.size) - int(n_out_all)
                    except Exception:
                        pass
                except Exception:
                    pass
                return sol

            # Determine baseline outliers and worst residual to generate station subsets.
            try:
                base_out_codes = [str(x) for x in (result.get('outliers') or [])]
            except Exception:
                base_out_codes = []
            try:
                base_in_codes = [str(x) for x in (result.get('inliers') or [])]
            except Exception:
                base_in_codes = []
            worst_code = None
            try:
                worst_r = -1.0
                for s in (result.get('stations') or []):
                    c = str(s.get('code', '')).strip()
                    r = abs(float(s.get('residual_s', 0.0)))
                    if math.isfinite(r) and r > worst_r:
                        worst_r = float(r)
                        worst_code = c
            except Exception:
                worst_code = None

            subsets: List[Tuple[str, List[InfraArrival]]] = []
            subsets.append(('all', arrivals))
            if base_in_codes:
                subsets.append(('inliers_only', [a for a in arrivals if a.code in set(base_in_codes)]))
            for oc in base_out_codes:
                subsets.append((f'drop_{oc}', [a for a in arrivals if a.code != oc]))
            if worst_code:
                subsets.append((f'drop_worst_{worst_code}', [a for a in arrivals if a.code != worst_code]))

            # Local search settings: smaller area, slightly more samples.
            refine_cfg = {
                'search_km': 180.0,
                'n_samples': 7000,
                'max_residual_s': float(result.get('max_residual_threshold_s', 10.0) or 10.0),
            }
            refine_weight = 0.25  # seconds of score per second timing deviation
            refined_candidates: List[Dict] = []
            for j, (variant, arr_sub) in enumerate(subsets):
                if len(arr_sub) < 3:
                    continue
                cand0 = fit_infrasound_source_with_outlier_rejection(
                    arr_sub,
                    wind_for_fit,
                    (base_lat, base_lon, base_alt),
                    search_km=float(refine_cfg['search_km']),
                    n_samples=int(refine_cfg['n_samples']),
                    seed=int(seed + 9000 + (j * 37)),
                    max_residual_s=float(refine_cfg['max_residual_s']),
                    track_segment=track_segment,
                    track_prior_s_per_km=0.02,
                    t0_prior_unix=t0_prior_unix,
                    t0_prior_s_per_s=float(refine_weight),
                    progress=progress,
                )
                if cand0 is None:
                    continue
                cand0['refine_variant'] = str(variant)
                try:
                    cand0['refine_used_codes'] = [a.code for a in arr_sub]
                except Exception:
                    pass
                cand0 = _eval_on_all(cand0)
                refined_candidates.append(cand0)
                if progress:
                    try:
                        t0u = float(cand0.get('t0_unix', float('nan')))
                        dt_s = float(t0u - float(t0_prior_unix)) if (t0_prior_unix is not None and math.isfinite(t0u) and math.isfinite(float(t0_prior_unix))) else float('nan')
                        elev_m = float(cand0.get('elev_m', float('nan')))
                        rms_s = float(cand0.get('rms_s', float('nan')))
                        rms_all_s = float(cand0.get('rms_all_s', float('nan')))
                        p90 = float(cand0.get('p90_abs_residual_all_s', float('nan')))
                        mx = float(cand0.get('max_abs_residual_all_s', float('nan')))
                        n_out = int(cand0.get('n_outliers', 0) or 0)
                        n_out_all = cand0.get('n_outliers_all')
                        try:
                            n_out_all = int(n_out_all) if n_out_all is not None else None
                        except Exception:
                            n_out_all = None
                        used = cand0.get('refine_used_codes')
                        hd = cand0.get('horizon_dropped_codes')
                        print(
                            "infrasound_refine cand "
                            + f"variant={cand0.get('refine_variant')} "
                            + f"used={used} "
                            + f"horizon_dropped={hd} "
                            + f"elev_m={elev_m:.1f} "
                            + (f"dt_s={dt_s:+.3f} " if math.isfinite(dt_s) else "")
                            + f"rms_s={rms_s:.3f} rms_all_s={rms_all_s:.3f} "
                            + (f"p90_abs_all_s={p90:.3f} " if math.isfinite(p90) else "")
                            + (f"max_abs_all_s={mx:.3f} " if math.isfinite(mx) else "")
                            + f"n_out={n_out} "
                            + (f"n_out_all={n_out_all}" if n_out_all is not None else "")
                        ,
                        flush=True)
                    except Exception:
                        pass

            # Pick best refined candidate (time delta first, then all-station RMS).
            min_ref_elev_m = 5000.0
            max_ref_elev_m = 80000.0

            best_ref = None
            best_key = None
            for cc in refined_candidates:
                try:
                    ref_alt_m = float(cc.get('elev_m', float('nan')))
                    if not math.isfinite(ref_alt_m):
                        continue
                    if not (min_ref_elev_m <= ref_alt_m <= max_ref_elev_m):
                        continue
                    ref_t0 = float(cc.get('t0_unix', float('nan')))
                    ref_dt = float(ref_t0 - float(t0_prior_unix)) if math.isfinite(ref_t0) else float('nan')
                    ref_rms_all = float(cc.get('rms_all_s', float('inf')))
                    if not (math.isfinite(ref_dt) and math.isfinite(ref_rms_all)):
                        continue
                    key = (abs(ref_dt), ref_rms_all)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_ref = cc
                except Exception:
                    continue

            refined = best_ref

            chosen = 'base'
            if refined is not None:
                ref_rms = float(refined.get('rms_s', float('inf')))
                ref_rms_all = float(refined.get('rms_all_s', float('inf')))
                ref_max_all = float(refined.get('max_abs_residual_all_s', float('nan')))
                ref_p90_all = float(refined.get('p90_abs_residual_all_s', float('nan')))
                ref_n_out = int(refined.get('n_outliers', 0) or 0)
                ref_n_out_all = refined.get('n_outliers_all')
                try:
                    ref_n_out_all = int(ref_n_out_all) if ref_n_out_all is not None else None
                except Exception:
                    ref_n_out_all = None
                ref_t0 = float(refined.get('t0_unix', float('nan')))
                ref_dt = float(ref_t0 - float(t0_prior_unix)) if math.isfinite(ref_t0) else float('nan')

                # Accept refinement only if timing improves meaningfully and RMS does not degrade too much.
                time_improved = (
                    math.isfinite(base_dt) and math.isfinite(ref_dt)
                    and abs(ref_dt) <= max(0.75 * abs(base_dt), abs(base_dt) - 10.0)
                )
                rms_ok = (math.isfinite(ref_rms) and math.isfinite(base_rms) and (ref_rms <= max(base_rms * 1.20, base_rms + 8.0)))
                # Also consider outliers/all-station residuals so outliers can influence acceptance.
                outliers_ok = True
                try:
                    # Prefer distribution-based guard: do not let typical station residuals blow up.
                    if math.isfinite(base_p90_all) and math.isfinite(ref_p90_all):
                        outliers_ok = outliers_ok and (ref_p90_all <= max(base_p90_all * 1.25, base_p90_all + 15.0))
                except Exception:
                    pass
                try:
                    # Prefer comparing outlier counts after evaluating against all stations.
                    if base_n_out_all is not None and ref_n_out_all is not None:
                        outliers_ok = outliers_ok and (int(ref_n_out_all) <= int(base_n_out_all) + 1)
                    else:
                        outliers_ok = outliers_ok and (ref_n_out <= base_n_out + 1)
                except Exception:
                    pass
                try:
                    if math.isfinite(base_rms_all) and math.isfinite(ref_rms_all):
                        outliers_ok = outliers_ok and (ref_rms_all <= max(base_rms_all * 1.25, base_rms_all + 10.0))
                except Exception:
                    pass

                if progress:
                    try:
                        print(
                            "infrasound_refine decision "
                            + f"time_improved={bool(time_improved)} rms_ok={bool(rms_ok)} outliers_ok={bool(outliers_ok)} "
                            + f"base_dt_s={base_dt:+.3f} ref_dt_s={ref_dt:+.3f} "
                            + f"base_rms_s={base_rms:.3f} ref_rms_s={ref_rms:.3f} "
                            + f"base_rms_all_s={base_rms_all:.3f} ref_rms_all_s={ref_rms_all:.3f} "
                            + f"base_p90_abs_all_s={base_p90_all:.3f} ref_p90_abs_all_s={ref_p90_all:.3f}"
                        ,
                        flush=True)
                    except Exception:
                        pass

                if time_improved and rms_ok and outliers_ok:
                    result = refined
                    chosen = 'refined'

            # Record diagnostics regardless of whether we switch solutions.
            try:
                result['refine_enabled'] = True
                result['refine_weight'] = float(refine_weight)
                result['base_solution'] = {
                    'lat': base_lat,
                    'lon': base_lon,
                    'elev_m': base_alt,
                    't0_unix': base_t0,
                    'rms_s': base_rms,
                    'rms_all_s': base_rms_all,
                    'max_abs_residual_all_s': base_max_all,
                    'p90_abs_residual_all_s': base_p90_all,
                    'n_outliers': base_n_out,
                    'n_outliers_all': base_n_out_all,
                    'worst_station_code': base_worst_code,
                    'worst_station_abs_residual_s': base_worst_abs,
                    't0_dt_s': base_dt,
                }
                if refined is not None:
                    ref_t0 = float(refined.get('t0_unix', float('nan')))
                    ref_dt = float(ref_t0 - float(t0_prior_unix)) if math.isfinite(ref_t0) else float('nan')
                    result['refined_solution'] = {
                        'lat': float(refined.get('lat', float('nan'))),
                        'lon': float(refined.get('lon', float('nan'))),
                        'elev_m': float(refined.get('elev_m', float('nan'))),
                        't0_unix': ref_t0,
                        'rms_s': float(refined.get('rms_s', float('nan'))),
                        'rms_all_s': float(refined.get('rms_all_s', float('nan'))),
                        'max_abs_residual_all_s': float(refined.get('max_abs_residual_all_s', float('nan'))),
                        'p90_abs_residual_all_s': float(refined.get('p90_abs_residual_all_s', float('nan'))),
                        'n_outliers': int(refined.get('n_outliers', 0) or 0),
                        'n_outliers_all': refined.get('n_outliers_all'),
                        'refine_variant': refined.get('refine_variant'),
                        'refine_used_codes': refined.get('refine_used_codes'),
                        'worst_station_code': refined.get('worst_station_code'),
                        'worst_station_abs_residual_s': refined.get('worst_station_abs_residual_s'),
                        't0_dt_s': ref_dt,
                    }
                result['solution_choice'] = chosen
            except Exception:
                pass
    except Exception:
        pass

    # Persist visual timing diagnostics in the final result so the report can explain what happened.
    try:
        result['visual_event_txt'] = visual_event_txt
        result['visual_start_unix'] = visual_start_unix
        result['visual_end_unix'] = visual_end_unix
        result['visual_event_txt_candidates'] = int(visual_event_txt_candidates)
        result['t0_prior_unix'] = t0_prior_unix
        result['t0_prior_s_per_s'] = float(t0_prior_s_per_s)
    except Exception:
        pass

    result['n_stations'] = len(arrivals)
    result['used_wind_profile'] = bool(per_station_winds) or (wind_profile is not None)
    try:
        if per_station_winds:
            result['wind_profiles'] = [str(w.get('path')) for w in per_station_winds if w.get('path')]
        elif wind_csv.exists():
            result['wind_profiles'] = [str(wind_csv)]
    except Exception:
        pass

    try:
        fp = result.get('fit_params') or {}
        fp.update({
            'seed': int(seed),
        })
        result['fit_params'] = fp
    except Exception:
        pass

    try:
        inl = ','.join(result.get('inliers') or [])
        outl = ','.join(result.get('outliers') or [])
        result['summary'] = (
            f"lat={result.get('lat'):.5f} lon={result.get('lon'):.5f} elev_m={result.get('elev_m'):.0f} "
            f"rms_s={result.get('rms_s'):.2f} inliers=[{inl}] outliers=[{outl}] wind={result.get('used_wind_profile')}"
        )
    except Exception:
        pass

    return result


def format_fit_report(event_dir: Path, result: Dict) -> str:
    lines: List[str] = []
    lines.append(f"event_dir={event_dir}")
    if result.get('summary'):
        lines.append(f"summary={result.get('summary')}")
    if result.get('used_wind_profile') is not None:
        lines.append(f"used_wind_profile={bool(result.get('used_wind_profile'))}")
    if result.get('fit_params'):
        fp = result.get('fit_params')
        lines.append(
            "fit_params="
            + ", ".join(f"{k}={v}" for k, v in fp.items())
        )
    try:
        hd = result.get('horizon_dropped_codes')
        if isinstance(hd, list) and hd:
            lines.append(f"horizon_dropped_codes={hd}")
    except Exception:
        pass

    # Always print c_scale so it's unambiguous whether scaling ran.
    try:
        cs = float(result.get('c_scale', 1.0) or 1.0)
        if not math.isfinite(cs) or cs <= 0.0:
            cs = 1.0
        lines.append(f"c_scale={cs:.5f}")
    except Exception:
        lines.append("c_scale=1.00000")

    # Visual timing debug (always print, even if missing)
    try:
        lines.append(f"visual_event_txt={result.get('visual_event_txt')}")
        lines.append(f"visual_start_unix={result.get('visual_start_unix')}")
        lines.append(f"visual_end_unix={result.get('visual_end_unix')}")
        lines.append(f"t0_prior_unix={result.get('t0_prior_unix')}")
    except Exception:
        pass

    # Refinement diagnostics (optional)
    try:
        if result.get('refine_enabled') is not None:
            lines.append(f"refine_enabled={bool(result.get('refine_enabled'))}")
        if result.get('solution_choice'):
            lines.append(f"solution_choice={result.get('solution_choice')}")
        if result.get('refine_weight') is not None:
            lines.append(f"refine_weight={result.get('refine_weight')}")

        b = result.get('base_solution')
        if isinstance(b, dict):
            lines.append(
                "base_solution="
                + f"lat={b.get('lat')} lon={b.get('lon')} elev_m={b.get('elev_m')} "
                + f"rms_s={b.get('rms_s')} rms_all_s={b.get('rms_all_s')} "
                + f"max_abs_residual_all_s={b.get('max_abs_residual_all_s')} n_outliers={b.get('n_outliers')} "
                + f"n_outliers_all={b.get('n_outliers_all')} "
                + f"p90_abs_residual_all_s={b.get('p90_abs_residual_all_s')} "
                + f"worst_station={b.get('worst_station_code')} worst_abs_residual_s={b.get('worst_station_abs_residual_s')} "
                + f"t0_dt_s={b.get('t0_dt_s')}"
            )
        r = result.get('refined_solution')
        if isinstance(r, dict):
            lines.append(
                "refined_solution="
                + f"lat={r.get('lat')} lon={r.get('lon')} elev_m={r.get('elev_m')} "
                + f"rms_s={r.get('rms_s')} rms_all_s={r.get('rms_all_s')} "
                + f"max_abs_residual_all_s={r.get('max_abs_residual_all_s')} n_outliers={r.get('n_outliers')} "
                + f"n_outliers_all={r.get('n_outliers_all')} "
                + f"p90_abs_residual_all_s={r.get('p90_abs_residual_all_s')} "
                + f"worst_station={r.get('worst_station_code')} worst_abs_residual_s={r.get('worst_station_abs_residual_s')} "
                + f"refine_variant={r.get('refine_variant')} refine_used_codes={r.get('refine_used_codes')} "
                + f"t0_dt_s={r.get('t0_dt_s')}"
            )
    except Exception:
        pass
    try:
        t0u = float(result.get('t0_unix', float('nan')))
        t0p = result.get('t0_prior_unix')
        t0s = ''
        t0ps = ''
        try:
            if math.isfinite(t0u):
                t0s = datetime.datetime.fromtimestamp(t0u, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            t0s = ''
        try:
            if t0p is not None and math.isfinite(float(t0p)):
                t0ps = datetime.datetime.fromtimestamp(float(t0p), tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            t0ps = ''
        if t0ps:
            lines.append(f"visual_t0_prior={t0ps} ({float(t0p):.3f})")
        if t0s:
            lines.append(f"fitted_t0={t0s} ({t0u:.3f})")
        try:
            if t0ps and t0s and t0p is not None and math.isfinite(float(t0p)) and math.isfinite(float(t0u)):
                lines.append(f"t0_dt_s={float(t0u) - float(t0p):+.3f}")
        except Exception:
            pass
    except Exception:
        pass
    lines.append(
        f"solution lat={result.get('lat'):.6f} lon={result.get('lon'):.6f} elev_m={result.get('elev_m'):.1f} t0_unix={result.get('t0_unix'):.3f}"
    )
    try:
        rms_s = float(result.get('rms_s', float('nan')))
    except Exception:
        rms_s = float('nan')
    try:
        max_inl = float(result.get('max_residual_s', float('nan')))
    except Exception:
        max_inl = float('nan')
    try:
        max_all = float(result.get('max_abs_residual_all_s', float('nan')))
    except Exception:
        max_all = float('nan')
    try:
        thr = float(result.get('max_residual_threshold_s', float('nan')))
    except Exception:
        thr = float('nan')
    try:
        hard = float(result.get('hard_outlier_s', float('nan')))
    except Exception:
        hard = float('nan')

    lines.append(
        f"rms_s={rms_s:.3f} "
        + f"max_abs_residual_inliers_s={max_inl:.3f} "
        + f"max_abs_residual_all_s={max_all:.3f} "
        + f"max_residual_threshold_s={thr:.3f} "
        + f"hard_outlier_s={hard:.3f}"
    )
    lines.append("")
    lines.append("per_station:")
    lines.append("code station arrival_unix predicted_unix residual_s travel_time_s distance_km celerity_km_s path_km c_eff_ms weight hmin_lin_km hmid_lin_km hmax_lin_km hmin_chord_km hmid_chord_km hmin_chord_in_km")
    for s in (result.get('stations') or []):
        try:
            cel = s.get('celerity_km_s')
            cel_str = f"{float(cel):.3f}" if cel is not None and math.isfinite(float(cel)) else "nan"
            hmin_lin = s.get('h_min_lin_m')
            hmid_lin = s.get('h_mid_lin_m')
            hmax_lin = s.get('h_max_lin_m')
            hmin_ch = s.get('h_min_chord_m')
            hmid_ch = s.get('h_mid_chord_m')
            hmin_ch_in = s.get('h_min_chord_inner_m')
            hmin_lin_km = f"{float(hmin_lin)/1000.0:.1f}" if hmin_lin is not None and math.isfinite(float(hmin_lin)) else "nan"
            hmid_lin_km = f"{float(hmid_lin)/1000.0:.1f}" if hmid_lin is not None and math.isfinite(float(hmid_lin)) else "nan"
            hmax_lin_km = f"{float(hmax_lin)/1000.0:.1f}" if hmax_lin is not None and math.isfinite(float(hmax_lin)) else "nan"
            hmin_ch_km = f"{float(hmin_ch)/1000.0:.1f}" if hmin_ch is not None and math.isfinite(float(hmin_ch)) else "nan"
            hmid_ch_km = f"{float(hmid_ch)/1000.0:.1f}" if hmid_ch is not None and math.isfinite(float(hmid_ch)) else "nan"
            hmin_ch_in_km = f"{float(hmin_ch_in)/1000.0:.1f}" if hmin_ch_in is not None and math.isfinite(float(hmin_ch_in)) else "nan"
            lines.append(
                f"{s.get('code','')} {s.get('station','')} {s.get('arrival_unix'):.3f} {s.get('predicted_unix'):.3f} "
                f"{s.get('residual_s'):+.3f} {float(s.get('travel_time_s') or 0.0):.2f} {s.get('distance_km'):.1f} {cel_str} {s.get('path_km'):.1f} {s.get('c_eff_ms'):.1f} {float(s.get('weight') or 1.0):.2f} "
                f"{hmin_lin_km} {hmid_lin_km} {hmax_lin_km} {hmin_ch_km} {hmid_ch_km} {hmin_ch_in_km}"
            )
        except Exception:
            continue
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description='Fit an infrasound source location from infra.txt arrival times.')
    ap.add_argument('event_dir', type=Path, help='Event directory, e.g. meteor/20260222/154850')
    ap.add_argument('--guess-lat', type=float, default=None)
    ap.add_argument('--guess-lon', type=float, default=None)
    ap.add_argument('--guess-alt-m', type=float, default=None)
    ap.add_argument('--output', type=Path, default=None, help='Output JSON file (default: <event_dir>/infra_fit.json)')
    ap.add_argument('--report', type=Path, default=None, help='Output report text file (default: <event_dir>/infra_fit_report.txt)')
    ap.add_argument('--verbose', action='store_true', help='Print fit summary and residual table to stdout')
    args = ap.parse_args()

    initial_guess = None
    if args.guess_lat is not None and args.guess_lon is not None:
        initial_guess = (args.guess_lat, args.guess_lon, float(args.guess_alt_m or 30000.0))

    out = args.output or (args.event_dir / 'infra_fit.json')
    report_path = args.report or (args.event_dir / 'infra_fit_report.txt')

    try:
        res = run_fit_for_event_dir(args.event_dir, initial_guess=initial_guess, progress=bool(args.verbose))
        if res is None:
            try:
                arrivals, diag = _collect_infra_diagnostics(args.event_dir)
                wind_csv = args.event_dir / 'wind_profile.csv'
                infra_paths = [p for p, _ in diag]
                reasons = [f"{p} :: {r}" for p, r in diag]
                report_path.write_text(
                    "\n".join([
                        f"event_dir={args.event_dir}",
                        "status=FAILED", 
                        f"reason=no_solution", 
                        f"n_arrivals={len(arrivals)}",
                        f"n_infra_files={len(infra_paths)}",
                        f"used_wind_profile={wind_csv.exists()}",
                        "", 
                        "infra_files:",
                        *infra_paths,
                        "",
                        "parse_diagnostics:",
                        *reasons,
                    ]) + "\n",
                    encoding='utf-8',
                )
            except Exception:
                pass
            return 1

        out.write_text(json.dumps(res, indent=2, sort_keys=True), encoding='utf-8')

        try:
            report_txt = format_fit_report(args.event_dir, res)
            report_path.write_text(report_txt, encoding='utf-8')
            if args.verbose:
                print(report_txt)
        except Exception:
            pass

        # Keep stdout stable for callers: print JSON path first.
        print(str(out))
        return 0
    except Exception:
        # Always emit a report on unexpected crashes
        try:
            import traceback
            tb = traceback.format_exc()
            report_path.write_text(
                "\n".join([
                    f"event_dir={args.event_dir}",
                    "status=CRASH", 
                    "traceback:",
                    tb,
                ]) + "\n",
                encoding='utf-8',
            )
        except Exception:
            pass
        raise


if __name__ == '__main__':
    raise SystemExit(main())
