#!/usr/bin/env python3
"""Finite-distance atmospheric-refraction correction for meteor observations."""

import math
from functools import lru_cache

EARTH_RADIUS_KM = 6371.0
REFRACTIVITY_SEA_LEVEL = 0.000276567
RAY_TOP_KM = 86.0

_US76_LAYERS = (
    (0.0, 288.15, -6.5e-3, 101325.0),
    (11000.0, 216.65, 0.0, 22632.1),
    (20000.0, 216.65, 1.0e-3, 5474.89),
    (32000.0, 228.65, 2.8e-3, 868.02),
    (47000.0, 270.65, 0.0, 110.91),
    (51000.0, 270.65, -2.8e-3, 66.94),
    (71000.0, 214.65, -2.0e-3, 3.9564),
)
_G0 = 9.80665
_M_AIR = 0.0289644
_R_UNIV = 8.31432

# Ray-traced negative-angle ratios at a 1 km site (Visuri et al., 2026).
_NEGATIVE_ANGLE_DZ_FACTOR = (
    (-0.9, 3318.0 / 1831.0), (-0.8, 3095.0 / 1831.0),
    (-0.7, 2894.0 / 1831.0), (-0.6, 2721.0 / 1831.0),
    (-0.5, 2530.0 / 1831.0), (-0.4, 2369.0 / 1831.0),
    (-0.3, 2210.0 / 1831.0), (-0.2, 2092.0 / 1831.0),
    (-0.1, 1951.0 / 1831.0), (0.0, 1.0),
)
# Same source: ray-traced total refraction below the local horizon.
_NEGATIVE_ANGLE_REFRACTION_DEG = (
    (-0.9, 0.7049), (-0.8, 0.6756), (-0.7, 0.6490),
    (-0.6, 0.6296), (-0.5, 0.5980), (-0.4, 0.5748),
    (-0.3, 0.5463), (-0.2, 0.5383), (-0.1, 0.5081),
    (0.0, 0.4867),
)


def _interp(x, table):
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def refraction_deg_apparent_minus_true(alt_deg, pressure_hpa=1010.0,
                                       temp_c=10.0):
    """Approximate refraction in degrees: apparent minus true altitude."""
    try:
        alt = float(alt_deg)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(alt) or alt < -1.0:
        return 0.0
    if alt < 0.0:
        return _interp(alt, _NEGATIVE_ANGLE_REFRACTION_DEG)
    alt_eff = max(0.1, min(89.9, alt))
    r_arcmin = 1.02 / math.tan(
        math.radians(alt_eff + 10.3 / (alt_eff + 5.11)))
    r_arcmin *= (pressure_hpa / 1010.0) * (283.0 / (273.0 + temp_c))
    return float(r_arcmin / 60.0)


def _us76_pressure_temperature(h_m):
    h_geo = h_m * 6356766.0 / (6356766.0 + h_m)
    h_geo = max(0.0, min(h_geo, 84852.0))
    for h_base, t_base, lapse, p_base in reversed(_US76_LAYERS):
        if h_geo >= h_base:
            if lapse == 0.0:
                pressure = p_base * math.exp(
                    -_G0 * _M_AIR * (h_geo - h_base) / (_R_UNIV * t_base))
                temperature = t_base
            else:
                temperature = t_base + lapse * (h_geo - h_base)
                pressure = p_base * (t_base / temperature) ** (
                    _G0 * _M_AIR / (_R_UNIV * lapse))
            return temperature, pressure
    t_base, p_base = _US76_LAYERS[0][1], _US76_LAYERS[0][3]
    return t_base, p_base


def _refractive_index(height_km):
    temperature, pressure = _us76_pressure_temperature(height_km * 1000.0)
    t0, p0 = _us76_pressure_temperature(0.0)
    return 1.0 + REFRACTIVITY_SEA_LEVEL * (pressure / p0) * (t0 / temperature)


def _raytraced_delta_z_m(apparent_alt_deg, site_height_m):
    site_height_km = max(0.0, float(site_height_m) / 1000.0)
    if site_height_km >= RAY_TOP_KM:
        return 0.0

    step_km = 0.001 if abs(apparent_alt_deg) < 1.0 else \
        0.002 if apparent_alt_deg < 3.0 else 0.01
    r0 = EARTH_RADIUS_KM + site_height_km
    theta0 = math.radians(90.0 - apparent_alt_deg)
    if theta0 <= 0.0 or theta0 >= math.pi:
        return 0.0
    ray_invariant = (_refractive_index(site_height_km) * r0 *
                     math.sin(theta0))

    phi = 0.0
    height_km = site_height_km
    while height_km < RAY_TOP_KM:
        next_height = min(RAY_TOP_KM, height_km + step_km)
        radius = EARTH_RADIUS_KM + 0.5 * (height_km + next_height)
        sin_theta = ray_invariant / (
            _refractive_index(0.5 * (height_km + next_height)) * radius)
        if not math.isfinite(sin_theta) or sin_theta >= 1.0:
            return 0.0
        theta = math.asin(sin_theta)
        phi += math.tan(theta) / radius * (next_height - height_km)
        height_km = next_height

    radius = EARTH_RADIUS_KM + RAY_TOP_KM
    sin_theta = ray_invariant / (_refractive_index(RAY_TOP_KM) * radius)
    if not math.isfinite(sin_theta) or abs(sin_theta) >= 1.0:
        return 0.0
    theta_top = math.asin(sin_theta)
    x_top = radius * math.sin(phi)
    z_top = radius * math.cos(phi)
    radial = (math.sin(phi), math.cos(phi))
    tangent = (math.cos(phi), -math.sin(phi))
    vx = (math.cos(theta_top) * radial[0] +
          math.sin(theta_top) * tangent[0])
    vz = (math.cos(theta_top) * radial[1] +
          math.sin(theta_top) * tangent[1])
    if abs(vx) < 1e-15:
        return 0.0
    z_observer = z_top - (x_top / vx) * vz
    return max(0.0, (z_observer - r0) * 1000.0)


@lru_cache(maxsize=4096)
def _delta_z_m_cached(altitude_key, site_height_key):
    altitude = altitude_key / 1000.0
    site_height = site_height_key / 1000.0
    if altitude < 0.0:
        base = _raytraced_delta_z_m(0.0, site_height)
        return base * _interp(altitude, _NEGATIVE_ANGLE_DZ_FACTOR)
    return _raytraced_delta_z_m(altitude, site_height)


def delta_z_m(apparent_alt_deg, site_height_m=0.0):
    """Virtual upward site displacement for finite-distance refraction.

    The returned value implements the delta-z construction: use the full
    refraction-corrected apparent altitude, while locating the observer at
    ``site_height_m + delta_z_m``. Values are keyed to millidegrees and
    millimetres so repeated centroid calls can share the ray trace.
    """
    try:
        altitude = float(apparent_alt_deg)
        site_height = float(site_height_m)
    except (TypeError, ValueError):
        return 0.0
    if (not math.isfinite(altitude) or not math.isfinite(site_height)
            or altitude < -1.0 or altitude >= 90.0 or site_height < 0.0):
        return 0.0
    return _delta_z_m_cached(
        int(round(altitude * 1000.0)), int(round(site_height * 1000.0)))
