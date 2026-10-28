#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
darkflight.py — Meteoroid dark-flight propagator.

Propagates a meteoroid from the end of its luminous trajectory to the ground,
in the rotating ECEF frame (gravity + atmospheric drag + ablation +
Earth-rotation fictitious forces). Inspired by the Desert Fireball Network
DFN_DarkFlight.py (Jansen-Sturgeon & Towner) and Vida's Supracenter
darkflight, adapted to NMN event data:

  * .res file          — straight-line track start/end (lon, lat, height)
  * fbspd_merge fit    — speed v(t) and deceleration a(t) along the track
  * wind_profile.csv   — Open-Meteo pressure-level wind/density profile
  * US76 standard atmosphere above/below the measured profile

Outputs impact predictions for a set of fragmentation scenarios, Monte-Carlo
uncertainty runs, JSON/KML/GeoJSON files and a fall-area map.

All quantities SI (m, kg, s) unless noted.

Can be used as a module:  darkflight.run_darkflight(event_dir, ...)
or as a script:           python3 darkflight.py <event_dir> [--mc N]
"""

import argparse
import functools
import json
import logging
import math
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d

# --- Constants (SI) -----------------------------------------------------------
MU_E = 3.986005e14            # Earth gravitational parameter [m3/s2]
OMEGA = 7.292115e-5           # Earth rotation rate [rad/s]
OMEGA_VEC = np.array([0.0, 0.0, OMEGA])
A_EQ = 6378137.0              # WGS-84 semi-major axis [m]
E2 = 6.69437999014e-3         # WGS-84 first eccentricity squared
try:
    from numba import njit as _njit
except ImportError:
    def _njit(*a, **k):
        def deco(f):
            return f
        return deco

R_AIR = 287.05                # specific gas constant dry air [J/(kg K)]

SHAPES = {'s': 1.21, 'c': 1.60, 'b': 2.7}   # sphere, cylinder, brick
DENSITY_GRID = [1500, 2500, 3500, 5000, 7000]  # meteoroid densities kg/m3


# --- Geodesy ------------------------------------------------------------------
def llh2ecef(lon_deg, lat_deg, h_m):
    """WGS-84 lon/lat(deg)/height(m) -> ECEF [m]."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    N = A_EQ / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.array([(N + h_m) * np.cos(lat) * np.cos(lon),
                     (N + h_m) * np.cos(lat) * np.sin(lon),
                     (N * (1 - E2) + h_m) * np.sin(lat)])


def ecef2llh(r):
    """ECEF [m] -> (lon_deg, lat_deg, h_m) iterative geodetic conversion."""
    p = np.hypot(r[0], r[1])
    if p < 1e-8:
        if abs(r[2]) < 1e-8:
            raise ValueError('Geodetic coordinates are undefined at Earth centre')
        return 0.0, math.copysign(90.0, r[2]), abs(r[2]) - A_EQ * math.sqrt(1 - E2)
    lon = np.degrees(np.arctan2(r[1], r[0]))
    lat = np.arctan2(r[2], p * (1 - E2))
    for _ in range(15):
        sin_lat = np.sin(lat)
        N = A_EQ / np.sqrt(1 - E2 * sin_lat ** 2)
        h = p / np.cos(lat) - N
        lat_new = np.arctan2(r[2], p * (1 - E2 * N / (N + h)))
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    return lon, np.degrees(lat), h


def _straight_track_llh(start_llh, end_llh, n=240):
    """Sample the fitted straight luminous segment in geodetic coordinates."""
    if not np.isscalar(n) or int(n) < 2:
        raise ValueError('Track sample count must be an integer >= 2')
    r0, r1 = llh2ecef(*start_llh), llh2ecef(*end_llh)
    fractions = np.linspace(0.0, 1.0, int(n))
    points = r0 + fractions[:, None] * (r1 - r0)
    lon, lat, h = _ecef2llh_arr(points[:, 0], points[:, 1], points[:, 2])
    return np.degrees(lon), np.degrees(lat), h


def _clip_track_max_h(lon, lat, h, max_h=40000.0):
    """Return the track below max_h, interpolating its 40 km entry point."""
    lon, lat, h = map(lambda v: np.asarray(v, dtype=float), (lon, lat, h))
    if not (len(lon) == len(lat) == len(h)):
        raise ValueError('Track coordinate arrays must have equal lengths')
    below = np.flatnonzero(h <= max_h)
    if not below.size:
        return np.array([]), np.array([]), np.array([])
    first = below[0]
    if first:
        denom = float(h[first - 1] - h[first])
        if abs(denom) > np.finfo(float).eps:
            f = float((h[first - 1] - max_h) / denom)
            lon = np.r_[lon[first - 1] + f * (lon[first] - lon[first - 1]), lon]
            lat = np.r_[lat[first - 1] + f * (lat[first] - lat[first - 1]), lat]
            h = np.r_[max_h, h]
    mask = h <= max_h
    return lon[mask], lat[mask], h[mask]


def enu2ecef_mat(lat_deg, lon_deg):
    """Rotation matrix mapping an ENU vector at (lat, lon) to ECEF."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-so, -sl * co, cl * co],
                     [co, -sl * so, cl * so],
                     [0.0, cl, sl]])


def earth_radius_geodetic(lat_deg):
    """WGS-84 ellipsoid surface radius at geodetic latitude [m]."""
    lat = np.radians(lat_deg)
    N = A_EQ / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.hypot(N * np.cos(lat), N * (1 - E2) * np.sin(lat))


# --- Standard atmosphere (US76, 0–84.9 km geopotential) -----------------------
_US76_LAYERS = [  # (h_base_m, T_base_K, lapse_K/m, P_base_Pa)
    (0.0,    288.15, -6.5e-3, 101325.0),
    (11000., 216.65,  0.0,    22632.1),
    (20000., 216.65,  1.0e-3, 5474.89),
    (32000., 228.65,  2.8e-3, 868.02),
    (47000., 270.65,  0.0,    110.91),
    (51000., 270.65, -2.8e-3, 66.94),
    (71000., 214.65, -2.0e-3, 3.9564),
]
_G0 = 9.80665
_M_AIR = 0.0289644
_R_UNIV = 8.31432

MAG_FLASH_T_EFF_S = 0.15          # terminal-flash duration (~a few frames);
                                  # the sustained pre-flare light is added
                                  # separately from the measured light curve
V_REF_SURV = 15000.0              # reference speed for the survival fraction


def _survival_fraction(v_ms):
    """Surviving/entry mass fraction range vs characteristic speed.

    Ablated energy per unit mass grows with v**2, so the surviving
    fraction scales as (v_ref/v)**2 around a ~15 km/s reference bolide
    (5-30%), clamped to a plausible range. Fast cometary fireballs keep
    almost nothing; slow asteroidal ones keep the most.
    """
    f = (V_REF_SURV / max(v_ms, 1.0)) ** 2
    return (min(0.30, max(0.001, 0.05 * f)),
            min(0.60, max(0.005, 0.30 * f)))


def us76(h_m):
    """US Standard Atmosphere 1976. Returns (T[K], P[Pa], rho[kg/m3])."""
    h_geo = h_m * 6356766.0 / (6356766.0 + h_m)  # geometric -> geopotential
    h_geo = max(0.0, min(h_geo, 84852.0))
    for hb, Tb, Lb, Pb in reversed(_US76_LAYERS):
        if h_geo >= hb:
            if Lb == 0.0:
                P = Pb * math.exp(-_G0 * _M_AIR * (h_geo - hb) / (_R_UNIV * Tb))
                T = Tb
            else:
                T = Tb + Lb * (h_geo - hb)
                P = Pb * (Tb / T) ** (_G0 * _M_AIR / (_R_UNIV * Lb))
            rho = P * _M_AIR / (_R_UNIV * T)
            return T, P, rho
    return _US76_LAYERS[0][1], _US76_LAYERS[0][3], \
        _US76_LAYERS[0][3] * _M_AIR / (_R_UNIV * _US76_LAYERS[0][1])


# --- Wind + atmosphere ---------------------------------------------------------
_MSIS_CACHE = {}


def _msis_profile(lat, lon, dt):
    """NRLMSISE-00 (via pymsis) mass density & temperature on a height grid
    for the event location/time. Returns (h_m, rho_kgm3, T_K) or None.
    Uses climatological solar indices — at 10-60 km the solar-cycle
    modulation of density is only a few percent."""
    key = (round(lat, 1), round(lon, 1), dt.strftime('%Y-%m-%dT%H'))
    if key in _MSIS_CACHE:
        return _MSIS_CACHE[key]
    prof = None
    try:
        import pymsis
        hs = np.linspace(0.0, 90.0, 241)  # km
        out = pymsis.calculate(dt, lon, lat, hs,
                               f107s=150.0, f107as=150.0,
                               aps=[[4.0] * 7])
        rho = out[0, 0, 0, :, pymsis.Variable.MASS_DENSITY]
        T = out[0, 0, 0, :, pymsis.Variable.TEMPERATURE]
        prof = (hs * 1000.0, np.asarray(rho, float), np.asarray(T, float))
    except Exception as e:
        logging.debug(f'NRLMSISE unavailable ({e}); using US76')
    _MSIS_CACHE[key] = prof
    return prof


class WindAtmosphere:
    """1-D vertical profile (Open-Meteo wind_profile.csv) over MSIS/US76.

    CSV columns: Height_m, Temp_K, Pressure_Pa, WindSpeed_ms, WindDir_deg
    (direction wind blows *from*, meteorological convention).
    Above the profile top: NRLMSISE-00 (if lon/lat/event time given, else
    US76) density, wind held at the top-layer value.
    Below the profile bottom: bottom-layer values.
    """

    def __init__(self, csv_path=None, lon=None, lat=None, event_dt=None):
        self.has_profile = False
        self._rho_scale = 1.0
        self._msis = None
        if lon is not None and lat is not None and event_dt is not None:
            self._msis = _msis_profile(lat, lon, event_dt)
        if csv_path and Path(csv_path).exists():
            rows = []
            with open(csv_path) as f:
                for row in csv.reader(f):
                    if not row or row[0].startswith('#'):
                        continue
                    rows.append([float(x) for x in row[:5]])
            if len(rows) >= 4:
                d = np.asarray(sorted(rows, key=lambda r: r[0]))
                if (d.ndim != 2 or d.shape[1] != 5 or not np.all(np.isfinite(d))
                        or np.any(np.diff(d[:, 0]) <= 0) or np.any(d[:, 1:3] <= 0)
                        or np.any(d[:, 3] < 0)):
                    raise ValueError('Wind profile requires unique heights, positive temperature/pressure and finite winds')
                self.h = d[:, 0]
                self.T = d[:, 1]
                self.P = d[:, 2]
                self.wspd = d[:, 3]
                self.wdir = d[:, 4]
                self.rho_p = self.P / (R_AIR * self.T)
                # wind-to (opposite of wind-from), ENU components
                rad = np.radians(self.wdir)
                self.we = -self.wspd * np.sin(rad)
                self.wn = -self.wspd * np.cos(rad)
                self.h_min, self.h_max = self.h[0], self.h[-1]
                # blend the background density to match the profile top
                _, rho_top = self._base_T_rho(self.h_max)
                if rho_top > 0:
                    self._rho_scale = self.rho_p[-1] / rho_top
                self.has_profile = True

    def _base_T_rho(self, h_m):
        """Background atmosphere (T, rho): NRLMSISE-00 if available else US76."""
        if self._msis is not None:
            hm, rho, T = self._msis
            i = np.searchsorted(hm, h_m)
            if i == 0:
                return T[0], rho[0]
            if i >= len(hm):
                return T[-1], rho[-1]
            f = (h_m - hm[i - 1]) / (hm[i] - hm[i - 1])
            return T[i - 1] + f * (T[i] - T[i - 1]), \
                np.exp(np.log(rho[i - 1]) + f * (np.log(rho[i]) - np.log(rho[i - 1])))
        T, _, rho = us76(h_m)
        return T, rho

    def at(self, h_m):
        """Returns (wind_enu[3], rho_a, T) at geometric height h_m."""
        if self.has_profile and h_m <= self.h_max:
            idx = np.searchsorted(self.h, h_m)
            if idx == 0:
                i0, i1, f = 0, 0, 0.0
            elif idx >= len(self.h):
                i0 = i1 = len(self.h) - 1
                f = 0.0
            else:
                i0, i1 = idx - 1, idx
                f = (h_m - self.h[i0]) / max(self.h[i1] - self.h[i0], 1e-9)
            we = self.we[i0] + f * (self.we[i1] - self.we[i0])
            wn = self.wn[i0] + f * (self.wn[i1] - self.wn[i0])
            T = self.T[i0] + f * (self.T[i1] - self.T[i0])
            rho = self.rho_p[i0] + f * (self.rho_p[i1] - self.rho_p[i0])
            return np.array([we, wn, 0.0]), rho, T
        # above profile (or no profile): MSIS/US76, wind = last layer or zero
        T, rho = self._base_T_rho(h_m)
        rho *= self._rho_scale
        if self.has_profile:
            return np.array([self.we[-1], self.wn[-1], 0.0]), rho, T
        return np.zeros(3), rho, T


def _packed_atmosphere(atm):
    """Return cached flat profile arrays used by the jitted dynamics."""
    packed = getattr(atm, '_packed_atmosphere', None)
    if packed is not None:
        return packed
    empty = np.zeros(1)
    if atm.has_profile:
        ph, pT, prho, pwe, pwn = [np.asarray(a, dtype=float) for a in
                                   (atm.h, atm.T, atm.rho_p, atm.we, atm.wn)]
    else:
        ph = pT = prho = pwe = pwn = empty
    if atm._msis is not None:
        mh, mrho, mT = [np.asarray(a, dtype=float) for a in atm._msis]
    else:
        mh = mrho = mT = empty
    packed = (bool(atm.has_profile), ph, pT, prho, pwe, pwn,
              atm._msis is not None, mh, mrho, mT, float(atm._rho_scale))
    try:
        atm._packed_atmosphere = packed
    except (AttributeError, TypeError):
        pass
    return packed


# --- Aerodynamics (ported from DFN atm_functions.py, MIT) ----------------------
def _viscosity(T):
    return 18.27e-6 * (291.15 + 120.0) / (T + 120.0) * (T / 291.15) ** 1.5


def _speed_of_sound(T):
    return 331.3 * math.sqrt(T / 273.15)


def _interp_shape(A, vals):
    return np.interp(A, [1.21, 1.6, 2.7], vals)


def cd_hypersonic(A):
    """Hypersonic drag coefficient by shape parameter A."""
    return _interp_shape(A, [0.92, 1.3, 2.0])


def _cd_subsonic(re, A):
    """Sub-critical drag — Haider & Levenspiel (1989), ellipsoid approx."""
    V = 1.0
    sa_eq = (36 * np.pi * V ** 2) ** (1. / 3)
    a_ax = np.sqrt(A * V ** (2. / 3) / np.pi)
    c_ax = (3 * V ** (1. / 3)) / (4 * A)
    thi = sa_eq / (4 * np.pi * ((a_ax ** 3.2 + 2 * (a_ax * c_ax) ** 1.6) / 3) ** (1. / 1.6))
    thi_perp = sa_eq / 4 / (np.pi * a_ax ** 2)
    a = np.exp(2.3288 - 6.4581 * thi + 2.4486 * thi ** 2)
    b = 0.0964 + 0.5565 * thi
    c = np.exp(4.905 - 13.8944 * thi + 18.4222 * thi ** 2 - 10.2599 * thi ** 3)
    d = np.exp(1.4681 + 12.2584 * thi - 20.7322 * thi ** 2 + 15.8855 * thi ** 3)
    return 24. / re * (1 + a * re ** b) + c / (1 + d / re)


def _cd_fm(vel):
    """Free-molecular drag — Khanukaeva (2005)."""
    vk = vel / 1000.0
    return 2.0 + np.sqrt(1.2) / (2.0 * vk) * (1.0 + vk ** 2 / 16.0 + 30.0)


def dragcoeff(vel, temp, rho_a, A):
    """Drag coefficient blending free-molecular, transition and continuum
    regimes (DFN implementation: Miller & Bailey 1979, Khanukaeva 2005)."""
    vel = max(vel, 1e-6)
    mu_a = _viscosity(temp)
    mach = vel / _speed_of_sound(temp)
    re = rho_a * vel * 0.1 / mu_a          # characteristic length 0.1 m
    kn = mach / re * np.sqrt(np.pi * 1.4 / 2.0)

    if kn > 10.0:
        return _cd_fm(vel)
    if kn > 0.01:
        cd_sub = _cd_subsonic(max(re, 1.0), A)
        return cd_sub + (_cd_fm(vel) - cd_sub) * np.exp(-0.001 * re ** 2)

    cd_sub = _cd_subsonic(max(re, 1.0), A)
    cd_hyp = cd_hypersonic(A)
    hw = _interp_shape(A, [0.5, 0.3, 0.1])
    M_c = _interp_shape(A, [1.5, 1.2, 1.1])
    logistic = lambda M: cd_sub + (cd_hyp - cd_sub) / (1 + np.exp(-(M - M_c) / hw))
    cd_crit = _interp_shape(A, [1.0, logistic(mach) / 0.92, logistic(mach) / 0.92])
    gumbel = lambda M: (cd_crit - logistic(M)) * np.exp(-(M - M_c) / hw
                        - np.exp(-(M - M_c) / hw)) / np.exp(-1)
    return logistic(mach) + gumbel(mach)


def ablation_coeff(rho_m, A):
    """Mass-loss coefficient c_ml [s2/m2] by meteoroid density class
    (Sansom 2019 / DFN; iron revised using the 2020 Ådalen fall)."""
    cd_hyp = cd_hypersonic(A)
    high_density_coeff = 0.07e-6 * cd_hyp
    if rho_m >= 7400:
        return 5.50e-8
    if rho_m > 6500:
        blend = (float(rho_m) - 6500.) / 900.
        return high_density_coeff + blend * (5.50e-8 - high_density_coeff)
    if rho_m > 5000:
        return high_density_coeff
    if rho_m > 2500:
        return 0.014e-6 * cd_hyp
    if rho_m > 1500:
        return 0.042e-6 * cd_hyp
    return 0.1e-6 * cd_hyp


def compressive_strength(rho_m):
    """Effective compressive strength [Pa] by density class — a
    fragmentation threshold on the dynamic pressure Cd*rho_a*v^2.
    Ranges from Popova et al. (2011) / Svestka (1994) reviews and the
    ~0.1-1 MPa values fitted for chondritic meteorite falls."""
    if rho_m > 5000:
        return 5e6        # iron
    if rho_m > 2500:
        return 1e6        # ordinary chondrite
    if rho_m > 1500:
        return 2e5        # carbonaceous
    return 5e4            # cometary/fragile


def fragmentation_spectrum(m_total, mass_index=2.0,
                           min_ratio=0.005, max_ratio=0.10,
                           grain_ratio=0.25, severity=1.0):
    """Heuristic disruption spectrum with severity = dyn_press/strength.

    Grain mass is untracked, not independently proven to be ablated.
    min_ratio/max_ratio set the relative power-law shape, not bounds
    on final masses: the spectrum is renormalised to conserve the
    remaining mass. Count ~severity**2 and grain share ~sqrt(severity)
    are sensitivity assumptions, not a calibrated fragmentation law.
    """
    if (not all(np.isfinite(v) for v in (m_total, mass_index, min_ratio, max_ratio, grain_ratio, severity))
            or m_total < 0 or not 0 < min_ratio < max_ratio <= 1
            or not 0 <= grain_ratio < 1 or severity < 1):
        raise ValueError('Invalid fragmentation parameters')
    if m_total == 0:
        return np.empty(0)
    n = int(5 * min(severity, math.sqrt(80.0)) ** 2)
    n = max(10, n)
    grain_ratio = float(np.clip(grain_ratio * np.sqrt(severity),
                                0.25, 0.75))
    m_frag = m_total * (1.0 - grain_ratio)
    lo, hi = min_ratio * m_frag, max_ratio * m_frag
    # inverse CDF for dN/dm ~ m^-a between lo and hi
    u = np.linspace(0.05, 0.95, n)
    a = mass_index
    inv = lambda x: ((1 - x) * lo ** (1 - a) + x * hi ** (1 - a)) ** (1 / (1 - a))
    ms = lo * (hi / lo) ** u if abs(a - 1.0) < 1e-12 else inv(u)
    # renormalise so the fragment masses sum to m_frag
    return ms * (m_frag / ms.sum())


# --- Numba-compiled dynamics --------------------------------------------------
# All scalar math — scipy/numpy calls per ODE step are the bottleneck
# (np.cross alone was ~40% of runtime). The atmosphere is passed as
# flat arrays so the jitted function never touches Python objects.

@_njit(cache=True)
def _ecef2llh_nb(rx, ry, rz):
    p = math.hypot(rx, ry)
    if p < 1e-8:
        if abs(rz) < 1e-8:
            raise ValueError('Geodetic coordinates are undefined at Earth centre')
        return 0.0, math.copysign(math.pi / 2, rz), abs(rz) - A_EQ * math.sqrt(1 - E2)
    lon = math.atan2(ry, rx)
    lat = math.atan2(rz, p * (1 - E2))
    for _ in range(15):
        sin_lat = math.sin(lat)
        N = A_EQ / math.sqrt(1 - E2 * sin_lat * sin_lat)
        h = p / math.cos(lat) - N
        lat_new = math.atan2(rz, p * (1 - E2 * N / (N + h)))
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new
    return lon, lat, h          # radians


@_njit(cache=True)
def _ecef2llh_arr(rx, ry, rz):
    """Vectorised ECEF->(lon, lat rad, h) for path output."""
    n = len(rx)
    lons = np.empty(n); lats = np.empty(n); hs = np.empty(n)
    for i in range(n):
        lons[i], lats[i], hs[i] = _ecef2llh_nb(rx[i], ry[i], rz[i])
    return lons, lats, hs


@_njit(cache=True)
def _us76_nb(h_m):
    h_geo = h_m * 6356766.0 / (6356766.0 + h_m)
    if h_geo < 0.0:
        h_geo = 0.0
    if h_geo > 84852.0:
        h_geo = 84852.0
    hb = (0.0, 11000., 20000., 32000., 47000., 51000., 71000.)
    Tb = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65)
    Lb = (-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3)
    Pb = (101325.0, 22632.1, 5474.89, 868.02, 110.91, 66.94, 3.9564)
    for i in range(6, -1, -1):
        if h_geo >= hb[i]:
            if Lb[i] == 0.0:
                P = Pb[i] * math.exp(-_G0 * _M_AIR * (h_geo - hb[i])
                                     / (_R_UNIV * Tb[i]))
                T = Tb[i]
            else:
                T = Tb[i] + Lb[i] * (h_geo - hb[i])
                P = Pb[i] * (Tb[i] / T) ** (_G0 * _M_AIR
                                            / (_R_UNIV * Lb[i]))
            return T, P * _M_AIR / (_R_UNIV * T)
    return 288.15, 101325.0 * _M_AIR / (_R_UNIV * 288.15)


@_njit(cache=True)
def _interp_nb(x, xs, ys):
    """np.interp equivalent for ascending xs."""
    if x <= xs[0]:
        return ys[0]
    n = len(xs)
    if x >= xs[n - 1]:
        return ys[n - 1]
    i = np.searchsorted(xs, x)
    f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + f * (ys[i] - ys[i - 1])


@_njit(cache=True)
def _atm_at_nb(h_m, has_prof, ph, pT, prho, pwe, pwn,
               has_msis, mh, mrho, mT, rho_scale):
    """(wind_e, wind_n, wind_u, rho, T). Wind profile below ph[-1],
    MSIS/US76 above."""
    if has_prof and h_m <= ph[len(ph) - 1]:
        we = _interp_nb(h_m, ph, pwe)
        wn = _interp_nb(h_m, ph, pwn)
        T = _interp_nb(h_m, ph, pT)
        rho = _interp_nb(h_m, ph, prho)
        return we, wn, 0.0, rho, T
    # background atmosphere
    if has_msis:
        T = _interp_nb(h_m, mh, mT)
        # log-linear density interp
        if h_m <= mh[0]:
            rho = mrho[0]
        elif h_m >= mh[len(mh) - 1]:
            rho = mrho[len(mrho) - 1]
        else:
            i = np.searchsorted(mh, h_m)
            f = (h_m - mh[i - 1]) / (mh[i] - mh[i - 1])
            rho = math.exp(math.log(mrho[i - 1])
                           + f * (math.log(mrho[i]) - math.log(mrho[i - 1])))
    else:
        T, rho = _us76_nb(h_m)
    rho *= rho_scale
    if has_prof:
        return pwe[len(pwe) - 1], pwn[len(pwn) - 1], 0.0, rho, T
    return 0.0, 0.0, 0.0, rho, T


@_njit(cache=True)
def _shape_interp_nb(A, v0, v1, v2):
    xs = (1.21, 1.6, 2.7)
    ys = (v0, v1, v2)
    if A <= xs[0]:
        return ys[0]
    if A >= xs[2]:
        return ys[2]
    if A < xs[1]:
        f = (A - xs[0]) / (xs[1] - xs[0])
        return ys[0] + f * (ys[1] - ys[0])
    f = (A - xs[1]) / (xs[2] - xs[1])
    return ys[1] + f * (ys[2] - ys[1])


@_njit(cache=True)
def _cd_subsonic_nb(re, A):
    V = 1.0
    sa_eq = (36 * np.pi * V * V) ** (1. / 3)
    a_ax = np.sqrt(A * V ** (2. / 3) / np.pi)
    c_ax = (3 * V ** (1. / 3)) / (4 * A)
    thi = sa_eq / (4 * np.pi * ((a_ax ** 3.2 + 2 * (a_ax * c_ax) ** 1.6)
                                / 3) ** (1. / 1.6))
    a = np.exp(2.3288 - 6.4581 * thi + 2.4486 * thi ** 2)
    b = 0.0964 + 0.5565 * thi
    c = np.exp(4.905 - 13.8944 * thi + 18.4222 * thi ** 2
               - 10.2599 * thi ** 3)
    d = np.exp(1.4681 + 12.2584 * thi - 20.7322 * thi ** 2
               + 15.8855 * thi ** 3)
    return 24. / re * (1 + a * re ** b) + c / (1 + d / re)


@_njit(cache=True)
def _cd_fm_nb(vel):
    vk = vel / 1000.0
    return 2.0 + np.sqrt(1.2) / (2.0 * vk) \
        * (1.0 + vk ** 2 / 16.0 + 30.0)


@_njit(cache=True)
def _dragcoeff_nb(vel, temp, rho_a, A, characteristic_length=0.1):
    if vel < 1e-6:
        vel = 1e-6
    mu_a = 18.27e-6 * (291.15 + 120.0) / (temp + 120.0) \
        * (temp / 291.15) ** 1.5
    sos = 331.3 * math.sqrt(temp / 273.15)
    mach = vel / sos
    re = rho_a * vel * characteristic_length / mu_a
    kn = mach / re * np.sqrt(np.pi * 1.4 / 2.0)
    if kn > 10.0:
        return _cd_fm_nb(vel)
    if kn > 0.01:
        cd_sub = _cd_subsonic_nb(max(re, 1.0), A)
        return cd_sub + (_cd_fm_nb(vel) - cd_sub) \
            * np.exp(-0.001 * re * re)
    cd_sub = _cd_subsonic_nb(max(re, 1.0), A)
    cd_hyp = _shape_interp_nb(A, 0.92, 1.3, 2.0)
    hw = _shape_interp_nb(A, 0.5, 0.3, 0.1)
    M_c = _shape_interp_nb(A, 1.5, 1.2, 1.1)
    # logistic at mach
    e1 = np.exp(-(mach - M_c) / hw)
    logi = cd_sub + (cd_hyp - cd_sub) / (1 + e1)
    cd_crit = _shape_interp_nb(A, 1.0, logi / 0.92, logi / 0.92)
    gum = (cd_crit - logi) * np.exp(-(mach - M_c) / hw
                                    - np.exp(-(mach - M_c) / hw)) / np.exp(-1)
    return logi + gum


@_njit(cache=True)
def _dynamics_nb(t, X, c_ml, A_sh, rho_m,
                 has_prof, ph, pT, prho, pwe, pwn,
                 has_msis, mh, mrho, mT, rho_scale, wind_offset):
    rx, ry, rz = X[0], X[1], X[2]
    vx, vy, vz = X[3], X[4], X[5]
    m = X[6]
    lon, lat, h = _ecef2llh_nb(rx, ry, rz)
    we, wn, wu, rho_a, T = _atm_at_nb(
        h if h > 0.0 else 0.0, has_prof, ph, pT, prho, pwe, pwn,
        has_msis, mh, mrho, mT, rho_scale)
    we += wind_offset[0]
    wn += wind_offset[1]
    wu += wind_offset[2]
    sl = math.sin(lat); cl = math.cos(lat)
    so = math.sin(lon); co = math.cos(lon)
    # enu2ecef: w_ecef = R @ wind_enu
    wx = -so * we - sl * co * wn + cl * co * wu
    wy = co * we - sl * so * wn + cl * so * wu
    wz = cl * wn + sl * wu
    vrx, vry, vrz = vx - wx, vy - wy, vz - wz
    vmag = math.sqrt(vrx * vrx + vry * vry + vrz * vrz)
    rnorm = math.sqrt(rx * rx + ry * ry + rz * rz)
    g = -MU_E / rnorm ** 3
    agx, agy, agz = g * rx, g * ry, g * rz
    # Coriolis -2*OMEGA x v ; centrifugal -OMEGA x (OMEGA x r)
    om = OMEGA
    acx = -2.0 * om * -vy + om * om * rx
    acy = -2.0 * om * vx + om * om * ry
    acz = 0.0
    m_eff = m if m > 1e-6 else 1e-6
    ax = ay = az = 0.0
    dm = 0.0
    if vmag > 1e-9:
        cd = _dragcoeff_nb(vmag, T, rho_a, A_sh)
        k = cd * A_sh * rho_a * vmag \
            / (2 * m_eff ** (1. / 3) * rho_m ** (2. / 3))
        ax, ay, az = -k * vrx, -k * vry, -k * vrz
        dm = -c_ml * A_sh * rho_a * vmag ** 3 * m_eff ** (2. / 3) \
            / (2 * rho_m ** (2. / 3))
    out = np.empty(7)
    out[0], out[1], out[2] = vx, vy, vz
    out[3] = agx + acx + ax
    out[4] = agy + acy + ay
    out[5] = agz + acz + az
    out[6] = dm
    return out


# --- Propagator ---------------------------------------------------------------
def propagate(r0_ecef, v0_ecef, m0, rho_m, A_shape, atm, h_ground,
              erode=True, record_dt=0.5, wind_offset_enu=None,
              ablation_scale=1.0):
    """Integrate a meteoroid to the ground.

    r0_ecef, v0_ecef: ECEF state [m], [m/s];  m0 [kg]; rho_m [kg/m3];
    A_shape: shape parameter (sphere 1.21 / cylinder 1.6 / brick 2.7);
    atm: WindAtmosphere; h_ground [m]; erode: enable ablation;
    ablation_scale: diagnostic multiplier on the material ablation coefficient.

    Returns dict with time series (lon, lat, h, v, m) + impact dict.
    """
    r0_ecef = np.asarray(r0_ecef, dtype=float)
    v0_ecef = np.asarray(v0_ecef, dtype=float)
    wind_offset = np.zeros(3) if wind_offset_enu is None else np.asarray(wind_offset_enu, dtype=float)
    if any(v.shape != (3,) or not np.all(np.isfinite(v))
           for v in (r0_ecef, v0_ecef, wind_offset)):
        raise ValueError('Position, velocity and wind offset must be finite 3-vectors')
    if not all(np.isfinite(v) and v > 0 for v in (m0, rho_m, A_shape, record_dt)):
        raise ValueError('Mass, density, shape and recording interval must be positive and finite')
    if not np.isfinite(ablation_scale) or ablation_scale < 0.0:
        raise ValueError('Ablation scale must be finite and nonnegative')
    if not np.isfinite(h_ground) or ecef2llh(r0_ecef)[2] <= h_ground:
        raise ValueError('Release position must be above finite ground height')
    c_ml = ablation_scale * ablation_coeff(rho_m, A_shape) if erode else 0.0

    # pack/cache the atmosphere once — the jitted dynamics sees only arrays
    (has_profile, ph, pT, prho, pwe, pwn,
     has_msis, mh, mrho, mT, rho_scale) = _packed_atmosphere(atm)
    dyn_args = (c_ml, A_shape, rho_m,
                has_profile, ph, pT, prho, pwe, pwn,
                has_msis, mh, mrho, mT, rho_scale, wind_offset)

    def dynamics(t, X):
        return _dynamics_nb(t, X, *dyn_args)

    def hit_ground(t, X):
        _, _, h = _ecef2llh_nb(X[0], X[1], X[2])
        return h - h_ground
    hit_ground.terminal = True
    hit_ground.direction = -1

    def dust(t, X):
        return X[6] - 1e-3
    dust.terminal = True
    dust.direction = -1

    X0 = np.hstack([r0_ecef, v0_ecef, m0])
    if m0 <= 1e-3:
        t_end, X_end, termination = 0.0, X0, 'dust'
        ts, Xs = np.array([0.0]), X0[:, None]
    else:
        sol = solve_ivp(dynamics, (0, 1800), X0, method='DOP853',
                        dense_output=True, events=[hit_ground, dust],
                        rtol=1e-7, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f'Dark-flight integration failed: {sol.message}')
        event = next((i for i, times in enumerate(sol.t_events) if times.size), None)
        t_end = float(sol.t_events[event][0] if event is not None else sol.t[-1])
        X_end = sol.y_events[event][0] if event is not None else sol.y[:, -1]
        termination = ('ground', 'dust')[event] if event is not None else 'timeout'
        n = max(int(t_end / record_dt) + 2, 2)
        ts = np.linspace(0, t_end, n)
        Xs = sol.sol(ts)
    lon_arr, lat_arr, h_arr = _ecef2llh_arr(Xs[0], Xs[1], Xs[2])
    lon_arr = np.degrees(lon_arr)
    lat_arr = np.degrees(lat_arr)
    v_arr = np.linalg.norm(Xs[3:6], axis=0)

    lon_i, lat_i, h_i = ecef2llh(X_end[:3])
    return {
        't': ts, 'lon': lon_arr, 'lat': lat_arr, 'h': h_arr,
        'v': v_arr, 'm': Xs[6],
        'impact': {'lon': lon_i, 'lat': lat_i, 'h': h_i,
                   'v': float(np.linalg.norm(X_end[3:6])),
                   'm': float(X_end[6]), 't': float(t_end),
                   'landed': termination == 'ground',
                   'termination': termination},
    }


@_njit(cache=True)
def _fragmentation_environment_nb(X, atmosphere):
    """Altitude, air-relative speed and dynamic pressure for event tests."""
    lon, lat, h = _ecef2llh_nb(X[0], X[1], X[2])
    we, wn, wu, rho_air, temperature = _atm_at_nb(max(h, 0.0), *atmosphere)
    if not (math.isfinite(rho_air) and rho_air > 0.0
            and math.isfinite(temperature) and temperature > 0.0
            and math.isfinite(we) and math.isfinite(wn) and math.isfinite(wu)):
        raise ValueError('Atmosphere must have positive finite density/temperature and finite winds')
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    ux = X[3] - (-so * we - sl * co * wn + cl * co * wu)
    uy = X[4] - (co * we - sl * so * wn + cl * so * wu)
    uz = X[5] - (cl * wn + sl * wu)
    speed = math.sqrt(ux * ux + uy * uy + uz * uz)
    return h, speed, rho_air * speed ** 2, temperature, rho_air, ux, uy, uz


@_njit(cache=True)
def _fragmentation_rates_nb(X, c_ml, eta, A_shape, rho_m, atmosphere):
    """Hot-flight ECEF rates, using actual grain mass down to 1e-10 kg.

    State is (r[3], v[3], mass, eroded_mass, dissipated_energy). Pressure is
    rho_air*v_air**2. The returned unscaled luminous power includes only
    ablation and aerodynamic kinetic-energy loss, not solid erosion.
    Reynolds/Knudsen numbers use the instantaneous volume-equivalent diameter.
    """
    (h, speed, pressure, temperature, rho_air,
     ux, uy, uz) = _fragmentation_environment_nb(X, atmosphere)
    mass = max(X[6], 1e-10)
    area = A_shape * (mass / rho_m) ** (2.0 / 3.0)
    diameter = 2.0 * (3.0 * mass / (4.0 * math.pi * rho_m)) ** (1.0 / 3.0)
    cd = _dragcoeff_nb(speed, temperature, rho_air, A_shape, diameter)
    drag_factor = cd * area * rho_air * speed / (2.0 * mass)
    energy_flux = area * rho_air * speed ** 3 / 2.0
    ablated_rate = c_ml * energy_flux
    eroded_rate = eta * energy_flux
    radius = math.sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)
    gravity = -MU_E / radius ** 3
    out = np.empty(9)
    out[0:3] = X[3:6]
    out[3] = gravity * X[0] + 2.0 * OMEGA * X[4] + OMEGA ** 2 * X[0] - drag_factor * ux
    out[4] = gravity * X[1] - 2.0 * OMEGA * X[3] + OMEGA ** 2 * X[1] - drag_factor * uy
    out[5] = gravity * X[2] - drag_factor * uz
    out[6] = -ablated_rate - eroded_rate
    out[7] = eroded_rate
    power = 0.5 * speed ** 2 * ablated_rate + mass * speed ** 2 * drag_factor
    out[8] = power
    valid = (math.isfinite(h) and math.isfinite(speed)
             and math.isfinite(pressure) and math.isfinite(power))
    for i in range(9):
        valid = valid and math.isfinite(out[i])
    return out, h, speed, pressure, power, valid


@_njit(cache=True)
def _fragmentation_derivative_nb(time, X, work, max_evaluations,
                                 c_ml, eta, A_shape, rho_m, atmosphere):
    """ODE-only wrapper counted against the same propagation work budget."""
    work[0] += 1
    if work[0] > max_evaluations:
        raise RuntimeError(
            'Fragmentation rate-evaluation work cap exceeded')
    result = _fragmentation_rates_nb(X, c_ml, eta, A_shape, rho_m, atmosphere)
    if not result[5]:
        raise RuntimeError(
            'Nonfinite fragmentation dynamics; check physical input ranges')
    return result[0]


def _fragmentation_grain_bins(grain_min, grain_max, mass_index, grain_bins):
    """Return representative masses and analytical mass fractions of log bins.

    For dN/dm proportional to m**(-mass_index), bin mass is the integral
    of m**(1-mass_index). Geometric bin centres represent independently
    propagated grains; packet_mass*fraction/centre is a count weight,
    possibly fractional, not a Monte Carlo probability or surviving fraction.
    """
    edges = np.linspace(math.log(grain_min), math.log(grain_max), grain_bins + 1)
    exponent = 2.0 - mass_index
    if abs(exponent) < 1e-12:
        log_integrals = np.log(np.diff(edges))
    elif exponent > 0:
        log_integrals = (exponent * edges[1:]
                         + np.log(-np.expm1(-exponent * np.diff(edges)))
                         - math.log(exponent))
    else:
        log_integrals = (exponent * edges[:-1]
                         + np.log(-np.expm1(exponent * np.diff(edges)))
                         - math.log(-exponent))
    if not np.all(np.isfinite(log_integrals)):
        raise ValueError('Grain spectrum exceeds floating-point range')
    fractions = np.exp(log_integrals - np.max(log_integrals))
    fractions /= math.fsum(float(f) for f in fractions)
    fractions[int(np.argmax(fractions))] += 1.0 - math.fsum(float(f) for f in fractions)
    return np.exp(0.5 * (edges[:-1] + edges[1:])), fractions


def propagate_fragmentation(r0_ecef, v0_ecef, m0, rho_m, A_shape, atm,
                            h_ground, model='single', strength_pa=1e6,
                            breakup_height_m=None, fragment_count=8,
                            strength_exponent=0.3, erosion_coeff=1e-7,
                            grain_min=1e-6, grain_max=1e-3, mass_index=2.0,
                            grain_bins=6, ablation_scale=1.0,
                            luminous_efficiency=0.03, max_step=0.05,
                            record_dt=0.02, max_generations=12,
                            erosion_release_dt=0.005, max_evaluations=4000000):
    """Forward reduced entry/fragmentation model, with all inputs in SI.

    ``atm`` is a WindAtmosphere-compatible packed profile. ECEF position and
    velocity are terrestrial rotating-frame vectors. Single bodies, equal
    discrete daughters, one-shot disruption into a power-law grain spectrum,
    and continuously eroding parents are supported. All daughters inherit
    the parent's state with zero separation velocity. An optional descending
    height crossing replaces the strength trigger ONLY for the first event;
    otherwise the trigger is rho_air*v_air**2 >= strength_pa (no Cd or 1/2).
    Discrete daughters have S_child = S_parent*N**strength_exponent.

    Erosion removes solid mass at eta*A_shape*(m/rho_m)**(2/3)*rho_air*v_air**3/2.
    Here eta has SI units s**2/m**2 in this precise convention; it is NOT a
    directly interchangeable published sigma. Cumulative integrated erosion
    is released in packets at most min(max_step, erosion_release_dt) apart, at interpolated
    midpoint states. Each log-mass bin is a weighted cohort, not random draws.
    Eroded mass is transferred to grains, not immediately vaporized/luminous.
    Ablation uses ablation_scale*ablation_coeff(rho_m, A_shape).

    Hot integration ends at air-relative 3000 m/s, ground, 1e-10 kg, or 1800 s.
    At the speed transition, >=1 g bodies use propagate(erode=False); smaller
    solids and numerical-floor remnants are unresolved, never called vapor.
    Luminosity is tau*(0.5*v_air**2*(-dm_abl/dt) + m*v_air*abs(a_drag)).
    The ODE integrates dissipated energy; each record_dt exposure bin sums
    energy from overlapping live segments, then divides by its duration.
    This resolves grain lifetimes shorter than an exposure without aliasing.
    Magnitude uses 1500 W for zero absolute magnitude and None for zero light.
    Dark flight contributes no modeled luminosity.

    Returns a JSON-ready dict with parameters, a closed mass_budget, weighted
    cohort fragments (including transfer outcomes and complete hot+dark paths),
    breakups, light_curve,
    largest_landed_kg (individual mass, not cohort total), peak magnitude,
    warnings, generation_cap_reached, assumptions and literature references.
    Work limits (max_evaluations rate evaluations, 20,000 cohorts, 1,000,000 light
    samples) raise RuntimeError rather than discard mass. Generation limits
    instead explicitly flag/warn and continue intact. These are reduced
    sensitivity models, not published-model reproductions or fitted results.
    """
    r0 = np.asarray(r0_ecef, dtype=float)
    v0 = np.asarray(v0_ecef, dtype=float)
    if any(x.shape != (3,) or not np.all(np.isfinite(x)) for x in (r0, v0)):
        raise ValueError('Position and velocity must be finite ECEF 3-vectors')
    if np.linalg.norm(v0) == 0 or not np.isfinite(np.linalg.norm(v0)):
        raise ValueError('Initial velocity must be finite and nonzero')
    if model not in ('single', 'discrete', 'disruption', 'erosion'):
        raise ValueError('Model must be single, discrete, disruption or erosion')
    positive = dict(m0=m0, rho_m=rho_m, A_shape=A_shape,
                    strength_pa=strength_pa, grain_min=grain_min,
                    grain_max=grain_max, mass_index=mass_index,
                    max_step=max_step, record_dt=record_dt,
                    erosion_release_dt=erosion_release_dt)
    nonnegative = dict(strength_exponent=strength_exponent,
                       erosion_coeff=erosion_coeff, ablation_scale=ablation_scale,
                       luminous_efficiency=luminous_efficiency)
    for name, value in positive.items():
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    for name, value in nonnegative.items():
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be nonnegative and finite')
    for name, value, minimum in (('fragment_count', fragment_count, 2),
                                 ('grain_bins', grain_bins, 1),
                                 ('max_generations', max_generations, 0),
                                 ('max_evaluations', max_evaluations, 1)):
        if (not np.isscalar(value) or not np.isfinite(value)
                or isinstance(value, (bool, np.bool_))
                or value != int(value) or value < minimum):
            raise ValueError(f'{name} must be an integer >= {minimum}')
    fragment_count, grain_bins, max_generations = map(int, (fragment_count, grain_bins, max_generations))
    if luminous_efficiency > 1 or grain_min >= grain_max:
        raise ValueError('Require luminous_efficiency <= 1 and grain_min < grain_max')
    h_start = float(ecef2llh(r0)[2])
    if not np.isfinite(h_ground) or h_start <= h_ground:
        raise ValueError('Release position must be above finite ground height')
    if breakup_height_m is not None and (not np.isfinite(breakup_height_m)
                                          or not h_ground < breakup_height_m <= h_start + 1e-3):
        raise ValueError('Breakup height must lie above ground and at/below release')
    if grain_bins > 20000:
        raise RuntimeError('Fragmentation cohort work cap exceeded (20,000)')
    if not all(hasattr(atm, name) for name in ('has_profile', '_msis', '_rho_scale')):
        raise ValueError('atm must be a WindAtmosphere-compatible packed profile')
    if not np.isfinite(atm._rho_scale) or atm._rho_scale <= 0:
        raise ValueError('Atmospheric density scale must be positive and finite')
    atmosphere = _packed_atmosphere(atm)
    (has_profile, ph, pT, prho, pwe, pwn,
     has_msis, mh, mrho, mT, rho_scale) = atmosphere
    for active, arrays, positive_indices in ((has_profile, (ph, pT, prho, pwe, pwn), (1, 2)),
                                              (has_msis, (mh, mrho, mT), (1, 2))):
        if active and (any(a.ndim != 1 or not a.size or not np.all(np.isfinite(a)) for a in arrays)
                       or any(a.size != arrays[0].size for a in arrays)
                       or np.any(np.diff(arrays[0]) <= 0)
                       or any(np.any(arrays[i] <= 0) for i in positive_indices)):
            raise ValueError('Atmosphere arrays must be finite, ordered and physically positive')
    c_ml = float(ablation_scale * ablation_coeff(rho_m, A_shape))
    if not np.isfinite(c_ml):
        raise ValueError('Scaled ablation coefficient must be finite')
    grain_masses, grain_fractions = _fragmentation_grain_bins(grain_min, grain_max, mass_index, grain_bins)
    fragments, breakups, warnings = [], [], []
    pending = []
    plot_paths = {}
    plot_dt = 0.1
    budget_terms = {key: [] for key in ('ablated_kg', 'landed_kg', 'unresolved_kg', 'timeout_kg')}
    light = {}
    work = np.zeros(1, dtype=np.int64)
    next_id = [0]
    generation_cap_reached = False
    largest_landed = 0.0
    hot_end = 0.0
    floor = 1e-10

    def environment(state):
        work[0] += 1
        if work[0] > max_evaluations:
            raise RuntimeError(f'Fragmentation rate-evaluation work cap exceeded ({max_evaluations:,})')
        result = _fragmentation_environment_nb(state, atmosphere)
        if not all(math.isfinite(x) for x in result):
            raise RuntimeError('Nonfinite fragmentation dynamics; check physical input ranges')
        return result[:3]

    def rates(state, eta=0.0):
        work[0] += 1
        if work[0] > max_evaluations:
            raise RuntimeError(f'Fragmentation rate-evaluation work cap exceeded ({max_evaluations:,})')
        result = _fragmentation_rates_nb(state, c_ml, eta, A_shape, rho_m, atmosphere)
        if not result[5] or not all(math.isfinite(x) for x in result[1:5]):
            raise RuntimeError('Nonfinite fragmentation dynamics; check physical input ranges')
        return result

    def enqueue(state, t, weight, strength, generation, kind, parent_id):
        if next_id[0] >= 20000:
            raise RuntimeError('Fragmentation cohort work cap exceeded (20,000)')
        if not math.isfinite(weight) or weight <= 0 or not np.all(np.isfinite(state)):
            raise RuntimeError('Fragmentation cohort exceeds floating-point range')
        ident = next_id[0]
        next_id[0] += 1
        lon, lat, height = ecef2llh(state[:3])
        plot_paths[ident] = (parent_id, [float(lon)], [float(lat)], [float(height)])
        pending.append((ident, parent_id, state.copy(), float(t), float(weight),
                        float(strength), generation, kind))

    def append_plot_path(ident, solution, start, end):
        n = max(2, int(math.ceil((end - start) / plot_dt)) + 1)
        sample = solution.sol(np.linspace(start, end, n))
        lon, lat, height = _ecef2llh_arr(sample[0], sample[1], sample[2])
        _, lons, lats, heights = plot_paths[ident]
        lons.extend(float(v) for v in np.degrees(lon[1:]))
        lats.extend(float(v) for v in np.degrees(lat[1:]))
        heights.extend(float(v) for v in height[1:])

    def joined_plot_path(ident, dark_path=None):
        segments, node = [], ident
        while node is not None:
            parent, lons, lats, heights = plot_paths[node]
            segments.append((lons, lats, heights))
            node = parent
        joined = {key: [] for key in ('lon', 'lat', 'h')}
        for lons, lats, heights in reversed(segments):
            start = 1 if joined['lon'] else 0
            joined['lon'] += lons[start:]
            joined['lat'] += lats[start:]
            joined['h'] += heights[start:]
        if dark_path is not None:
            for key, values in dark_path.items():
                joined[key] += list(values[1:] if joined[key] else values)
        return joined

    def release_grains(state, t, packet_mass, parent_weight, parent_id):
        if packet_mass <= 0:
            return
        packets = packet_mass * grain_fractions
        largest = int(np.argmax(packets))
        packets[largest] += packet_mass - math.fsum(float(p) for p in packets)
        for mass, packet in zip(grain_masses, packets):
            if packet > 0:
                child = state.copy()
                child[6], child[7] = mass, 0.0
                enqueue(child, t, parent_weight * float(packet / mass),
                        strength_pa, 0, 'grain', parent_id)

    initial = np.r_[r0, v0, float(m0), 0.0, 0.0]
    enqueue(initial, 0.0, 1.0, strength_pa, 0, 'parent', None)
    while pending:
        ident, parent_id, state, t, weight, strength, generation, kind = pending.pop()
        born, initial_mass = t, float(state[6])
        can_break = model != 'single' and kind != 'grain'
        eroding, cap_reached = False, False
        cohort_ablated, cohort_eroded = 0.0, 0.0
        outcome, reason, impact = None, None, None
        trajectory, dark_path = None, None
        plot_release_mass = initial_mass
        dark_release = None
        pending_event = None
        while outcome is None:
            _, height, speed, pressure, _, _ = rates(state)
            if height <= h_ground or pending_event == 0:
                outcome, reason = 'landed', 'hot_ground'
                break
            if state[6] <= floor or pending_event == 2:
                outcome, reason = 'unresolved', 'numerical_mass_floor'
                break
            if speed <= 3000.0 or pending_event == 1:
                if state[6] < 1e-3:
                    outcome, reason = 'unresolved', 'subgram_at_dark_transition'
                else:
                    dark_mass = float(state[6])
                    supplied_mass = np.nextafter(dark_mass, np.inf) if dark_mass == 1e-3 else dark_mass
                    dark_release = {'r_ecef': state[:3].tolist(),
                                    'v_ecef': state[3:6].tolist(),
                                    'mass_kg': dark_mass, 't_s': float(t)}
                    dark = propagate(state[:3], state[3:6], supplied_mass, rho_m,
                                     A_shape, atm, h_ground, erode=False, record_dt=10.0)
                    impact = {key: (bool(value) if isinstance(value, (bool, np.bool_))
                                    else str(value) if isinstance(value, str) else float(value))
                              for key, value in dark['impact'].items()}
                    impact['m'] = dark_mass
                    if impact['landed']:
                        dark_path = {key: dark[key].tolist() for key in ('lon', 'lat', 'h')}
                        plot_release_mass = dark_mass
                    impact['t'] += t
                    t = impact['t']
                    outcome = 'landed' if impact['landed'] else ('timeout' if impact['termination'] == 'timeout' else 'unresolved')
                    reason = 'dark_' + impact['termination']
                break
            if t >= 1800.0:
                outcome, reason = 'timeout', 'hot_timeout'
                break
            height_trigger = breakup_height_m is not None and generation == 0 and kind == 'parent'
            lon_rad, lat_rad, _ = _ecef2llh_nb(state[0], state[1], state[2])
            vertical_speed = (math.cos(lat_rad) * math.cos(lon_rad) * state[3]
                              + math.cos(lat_rad) * math.sin(lon_rad) * state[4]
                              + math.sin(lat_rad) * state[5])
            triggered = can_break and (pending_event == 3 or
                                       ((height <= breakup_height_m and vertical_speed < 0)
                                        if height_trigger else pressure >= strength))
            pending_event = None
            if triggered:
                capped = model == 'discrete' and generation >= max_generations
                breakups.append({'t': float(t), 'h': float(height), 'pressure': float(pressure),
                                 'mass': float(state[6] * weight), 'mass_per_particle_kg': float(state[6]),
                                 'multiplicity': weight, 'cohort_id': ident, 'generation': generation,
                                 'strength_pa': strength, 'trigger': 'height' if height_trigger else 'pressure',
                                 'kind': 'generation_cap' if capped else model,
                                 'generation_cap_reached': capped})
                if capped:
                    generation_cap_reached = cap_reached = True
                    message = f'Cohort {ident}: generation cap {max_generations} reached; continuing intact'
                    warnings.append(message)
                    logging.warning(message)
                    can_break = False
                    continue
                if model == 'discrete':
                    child = state.copy()
                    child[6] /= fragment_count
                    child[7] = 0.0
                    log_strength = math.log(strength) + strength_exponent * math.log(fragment_count)
                    if log_strength > math.log(np.finfo(float).max):
                        raise RuntimeError('Daughter strength exceeds floating-point range')
                    enqueue(child, t, weight * fragment_count, math.exp(log_strength),
                            generation + 1, 'daughter', ident)
                    outcome, reason = 'split', 'equal_daughters'
                    break
                if model == 'disruption':
                    release_grains(state, t, float(state[6]), weight, ident)
                    outcome, reason = 'disrupted', 'power_law_grains'
                    break
                can_break, eroding = False, True
                continue

            def ground_event(time, x):
                return _ecef2llh_nb(x[0], x[1], x[2])[2] - h_ground

            def speed_event(time, x):
                return environment(x)[1] - 3000.0

            def mass_event(time, x):
                return x[6] - floor

            def breakup_event(time, x):
                if height_trigger:
                    return _ecef2llh_nb(x[0], x[1], x[2])[2] - breakup_height_m
                return environment(x)[2] - strength

            for event in (ground_event, speed_event, mass_event):
                event.terminal, event.direction = True, -1
            breakup_event.terminal = True
            breakup_event.direction = -1 if height_trigger else 1
            events = [ground_event, speed_event, mass_event]
            if can_break:
                events.append(breakup_event)
            eta = float(erosion_coeff) if eroding else 0.0
            state[7:] = 0.0
            segment_start, mass_start = t, float(state[6])
            mass_atol = max(1e-18, min(1e-12, mass_start * 1e-10))
            derivative = functools.partial(
                _fragmentation_derivative_nb, work=work,
                max_evaluations=max_evaluations, c_ml=c_ml, eta=eta,
                A_shape=A_shape, rho_m=rho_m, atmosphere=atmosphere)
            try:
                solution = solve_ivp(derivative, (t, 1800.0), state,
                                     method='DOP853', dense_output=True, events=events,
                                     max_step=max_step, rtol=1e-8,
                                     atol=np.array([1e-4] * 3 + [1e-6] * 3 + [mass_atol] * 2 + [1e-8]))
            except RuntimeError as exc:
                if work[0] > max_evaluations:
                    raise RuntimeError(
                        f'Fragmentation rate-evaluation work cap exceeded ({max_evaluations:,})') from exc
                raise
            if not solution.success:
                raise RuntimeError('Hot-flight integration failed: ' + solution.message)
            t = float(solution.t[-1])
            state = solution.y[:, -1].copy()
            hot_end = max(hot_end, t)
            append_plot_path(ident, solution, segment_start, t)
            lost = mass_start - float(state[6])
            eroded = float(state[7])
            if lost < -mass_atol or eroded < -mass_atol or eroded > lost + 10 * mass_atol:
                raise RuntimeError('Nonphysical mass accounting during hot integration')
            eroded = min(max(eroded, 0.0), max(lost, 0.0))
            ablated = max(lost - eroded, 0.0)
            cohort_ablated += ablated * weight
            cohort_eroded += eroded * weight
            budget_terms['ablated_kg'].append(ablated * weight)
            if t / record_dt > 999999:
                raise RuntimeError('Fragmentation light-curve work cap exceeded (1,000,000 samples)')
            first_sample = int(math.floor(segment_start / record_dt))
            last_sample = int(math.ceil(t / record_dt))
            if first_sample < last_sample:
                left = max(segment_start, first_sample * record_dt)
                left_state = solution.sol(left)
                for sample in range(first_sample, last_sample):
                    right = min(t, (sample + 1) * record_dt)
                    right_state = solution.sol(right)
                    energy = float(right_state[8] - left_state[8])
                    if energy < -1e-6 * max(1., float(state[8])):
                        raise RuntimeError('Nonmonotonic integrated luminous energy')
                    power = luminous_efficiency * weight * max(energy, 0.) / record_dt
                    light[sample] = light.get(sample, 0.0) + power
                    if not math.isfinite(light[sample]):
                        raise RuntimeError('Summed luminosity exceeds floating-point range')
                    left_state = right_state
            if eroded > 0:
                intervals = max(1, int(math.ceil((t - segment_start) / min(max_step, erosion_release_dt))))
                if intervals * grain_bins + next_id[0] > 20000:
                    raise RuntimeError('Fragmentation erosion-cohort work cap exceeded (20,000)')
                times = np.linspace(segment_start, t, intervals + 1)
                cumulative = solution.sol(times)[7]
                packets = np.maximum(np.diff(cumulative), 0.0)
                packet_sum = math.fsum(float(p) for p in packets)
                if packet_sum <= 0:
                    raise RuntimeError('Integrated erosion has no resolvable release packets')
                packets *= eroded / packet_sum
                packets[int(np.argmax(packets))] += eroded - math.fsum(float(p) for p in packets)
                midpoint_times = 0.5 * (times[:-1] + times[1:])
                midpoint_states = solution.sol(midpoint_times)
                for index, (midpoint, packet) in enumerate(
                        zip(midpoint_times, packets)):
                    release_grains(midpoint_states[:, index], float(midpoint),
                                   float(packet), weight, ident)
            pending_event = next((i for i, event_times in enumerate(solution.t_events)
                                  if event_times.size), None)
            if pending_event is None:
                outcome, reason = 'timeout', 'hot_timeout'
        final_mass = max(float(state[6]), 0.0)
        total_mass = final_mass * weight
        if outcome in ('landed', 'unresolved', 'timeout'):
            budget_terms[outcome + '_kg'].append(total_mass)
        if outcome == 'landed':
            largest_landed = max(largest_landed, final_mass)
            if impact is None:
                lon, lat, height = ecef2llh(state[:3])
                impact = {'lon': float(lon), 'lat': float(lat), 'h': float(height),
                          'v': float(np.linalg.norm(state[3:6])), 'm': final_mass,
                          't': t, 'landed': True, 'termination': 'ground'}
            if impact['landed']:
                trajectory = joined_plot_path(ident, dark_path)
        fragments.append({'id': ident, 'parent_id': parent_id, 'kind': kind,
                          'weight': weight, 'multiplicity': weight, 'generation': generation,
                          'strength_pa': strength, 't_birth_s': born, 't_end_s': t,
                          'initial_mass_kg': initial_mass, 'final_mass_kg': final_mass,
                          'total_final_mass_kg': total_mass, 'ablated_kg': cohort_ablated,
                          'eroded_kg': cohort_eroded, 'outcome': outcome, 'termination': reason,
                          'generation_cap_reached': cap_reached, 'impact': impact,
                          'dark_release': dark_release,
                          'plot_release_mass_kg': plot_release_mass, 'trajectory': trajectory})
    budget = {'initial_kg': float(m0)}
    budget.update({key: math.fsum(values) for key, values in budget_terms.items()})
    budget['balance_error_kg'] = float(m0) - math.fsum(budget[key] for key in budget_terms)
    count = int(math.ceil(hot_end / record_dt)) + 1
    if count > 1000000:
        raise RuntimeError('Fragmentation light-curve work cap exceeded (1,000,000 samples)')
    times = [float(i * record_dt) for i in range(count)]
    powers = [float(light.get(i, 0.0)) for i in range(count)]
    magnitudes = [-2.5 * (math.log10(p) - math.log10(1500.0)) if p > 0 else None for p in powers]
    finite_magnitudes = [m for m in magnitudes if m is not None]
    parameters = {key: float(value) for key, value in {**positive, **nonnegative}.items()}
    parameters.update({'h_ground': float(h_ground), 'breakup_height_m': None if breakup_height_m is None else float(breakup_height_m),
                       'fragment_count': fragment_count, 'grain_bins': grain_bins,
                       'max_generations': max_generations, 'max_evaluations': int(max_evaluations),
                       'r0_ecef': r0.tolist(), 'v0_ecef': v0.tolist()})
    return {'model': model, 'parameters': parameters, 'mass_budget': budget,
            'rate_evaluations': int(work[0]),
            'largest_landed_kg': float(largest_landed),
            'fragments': sorted(fragments, key=lambda item: item['id']),
            'breakups': sorted(breakups, key=lambda item: item['t']),
            'light_curve': {'t_s': times, 'power_w': powers, 'absolute_magnitude': magnitudes,
                            'exposure_s': record_dt, 'time_convention': 'exposure_start',
                            'radiated_energy_j': math.fsum(powers) * record_dt},
            'peak_absolute_magnitude': min(finite_magnitudes) if finite_magnitudes else None,
            'generation_cap_reached': generation_cap_reached, 'warnings': warnings,
            'assumptions': [
                'Reduced sensitivity models, not exact literature reproductions, calibrations or fits; no assumed surviving fraction.',
                'Breakup pressure is rho_air*v_air^2, without Cd or 1/2; optional height replaces only the first trigger.',
                'Equal daughters have zero separation velocity and are compressed into one identical-state cohort with multiplicity.',
                'Power-law log bins use analytical mass fractions and geometric representative masses; count weights may be fractional.',
                'Erosion eta [s^2/m^2] uses dm/dt=-eta*A*(m/rho_m)^(2/3)*rho_air*v_air^3/2, not directly published sigma.',
                'Erosion packet mass is integrated cumulatively and released at midpoint states, at intervals <= min(max_step, erosion_release_dt); refine this timestep to check flash convergence.',
                'Ablation retains the existing density/shape coefficient times ablation_scale; erosion transfers solid mass to independently ablating grains.',
                'Hot drag uses instantaneous volume-equivalent diameter; legacy dark-flight drag retains its 0.1 m characteristic length. Grain and parent material densities are equal.',
                'Hot grains are followed to 1e-10 kg; remaining subgram solids are unresolved, not vapor. Hot timeout is 1800 s.',
                'Dark flight uses propagate(erode=False) for >=1 g; exactly 1 g is advanced by one floating-point ulp for its legacy cutoff, then restored.',
                'Luminosity is tau*(0.5*v_air^2*(-dm_abl/dt)+m*v_air*abs(a_drag)); no instantaneous erosion luminosity.',
                'Light is integrated over shared exposure bins, including partial segment lifetimes; zero magnitude is 1500 W and zero light has null magnitude.',
                'Generation cap continues intact with a warning; work caps raise exceptions rather than return truncated mass budgets.',
                'Fragment split/disrupted outcomes denote mass transfers; do not sum these intermediate masses as surviving mass.'],
            'references': [
                {'citation': 'Borovicka, Spurny & Koten (2007), A&A 473, 661', 'doi': '10.1051/0004-6361:20078131', 'use': 'Erosion and grain-release motivation; coefficient convention is explicitly defined here.'},
                {'citation': 'Borovicka, Spurny & Shrbeny (2020), AJ 160, 42', 'doi': '10.3847/1538-3881/ab9608', 'use': 'Discrete fragmentation, dust and erosion model taxonomy.'},
                {'citation': 'Wheeler, Register & Mathias (2017), Icarus 295, 149', 'doi': '10.1016/j.icarus.2017.02.011', 'use': 'Mass-dependent daughter strength motivation; not a full fragment-cloud model.'}]}


# --- Mass estimation -----------------------------------------------------------
def _decel_to_mass(a, v, rho_a, T, rho_m, A):
    """Invert the drag equation: a = cd*A*rhoa*v^2/(2*M^{1/3}*rho_m^{2/3})."""
    if a <= 0 or v <= 0:
        return np.inf
    cd = dragcoeff(v, T, rho_a, A)
    k = cd * A * rho_a * v ** 2 / (2.0 * a)
    return k ** 3 / rho_m ** 2


def estimate_mass(v_of_t, a_of_t, h_of_t, t_range, rho_grid=None, A=1.4,
                  atm=None, params=None, pcov=None, rng=None):
    """Least-squares fit of mass to the whole deceleration curve.

    Predicted decel for a body of mass M: a_pred(t) = cd(t)*A*rhoa(t)*v(t)^2
    / (2*M^{1/3}*rho_m^{2/3}). We solve for M per density; uncertainty from
    Monte-Carlo over the fit covariance (pcov) if supplied.

    Returns list of dicts: rho, A, m_fade_kg (median), lo/hi 68% interval,
    m_crit_kg (mass that exactly ablates to zero at the fade point).
    """
    if rho_grid is None:
        rho_grid = DENSITY_GRID
    if atm is None:
        atm = WindAtmosphere()
    if rng is None:
        rng = np.random.default_rng(0)

    ts = np.linspace(t_range[0], t_range[1], 25)
    vs = np.array([v_of_t(t) for t in ts])
    aa = np.array([a_of_t(t) for t in ts])
    hs = np.array([h_of_t(t) for t in ts])
    out = []
    for rho_m in rho_grid:
        # least-squares: a_pred = C(t) * M^{-1/3} -> solve M^{-1/3}
        states = [atm.at(max(h, 0.0)) for h in hs]
        C = np.array([dragcoeff(max(v, 1.0), state[1], state[2], A)
                      * A * state[1] * v ** 2
                      / (2.0 * rho_m ** (2. / 3))
                      for v, state in zip(vs, states)])
        m13inv = np.sum(C * aa) / np.sum(C * C) if np.sum(C * C) > 0 else 0.0
        m_med = (1.0 / m13inv) ** 3 if m13inv > 0 else np.inf

        # uncertainty: bootstrap over pcov
        m_samples = []
        if params is not None and pcov is not None:
            try:
                from fbspd_merge import expfunc_2ndder, expfunc_1stder
                covariance = np.asarray(pcov, dtype=float)
                if (covariance.shape != (4, 4) or not np.all(np.isfinite(covariance))
                        or not np.allclose(covariance, covariance.T)):
                    raise ValueError('Invalid fit covariance')
                draws = np.random.default_rng(rng.integers(1 << 30))
                for p in draws.multivariate_normal(params, covariance, size=200, check_valid='raise'):
                    a_s = -expfunc_2ndder(ts, *p) * 1000.0
                    v_s = expfunc_1stder(ts, *p) * 1000.0
                    if not (np.all(np.isfinite(a_s)) and np.all(np.isfinite(v_s))
                            and np.all(a_s > 0) and np.all(v_s > 0)):
                        m_samples.append(np.inf)
                        continue
                    C_s = np.array([dragcoeff(v, state[1], state[2], A)
                                    * A * state[1] * v ** 2
                                    / (2.0 * rho_m ** (2. / 3))
                                    for v, state in zip(v_s, states)])
                    m13i = np.sum(C_s * a_s) / np.sum(C_s * C_s)
                    m_samples.append((1.0 / m13i) ** 3 if m13i > 0 else np.inf)
            except Exception:
                pass
        if len(m_samples) > 10:
            # Clip at a hard meteoroid-plausibility ceiling but report the
            # range as unbounded (hi=inf) when the percentile saturates,
            # so the UI shows 'poorly constrained' instead of '10 t–10 t'.
            cl = np.clip(m_samples, 1e-6, 1e4)
            lo, hi = np.percentile(cl, [16, 84])
            if hi >= 1e4:
                hi = np.inf
        else:
            lo, hi = 0.0, np.inf

        # critical entry mass: ablation integral along track — a body of
        # this mass fully ablates by the fade point. Deliberately
        # ablation-only: the surviving fragment's own mass loss is
        # ablative; grain erosion applies to the parent body's weak
        # matrix (handled as a breakup-scenario grain fraction).
        B = ablation_coeff(rho_m, A) * A / (2 * rho_m ** (2. / 3))
        rhos = np.array([state[1] for state in states])
        integral = trapezoid(rhos * vs ** 3, ts)
        m_crit = (B / 3 * integral) ** 3

        out.append({'rho': rho_m, 'A': A, 'm_fade_kg': float(m_med),
                    'm_fade_lo': float(lo), 'm_fade_hi': float(hi),
                    'uncertainty_source': 'fit_covariance' if len(m_samples) > 10 else 'unavailable',
                    'm_crit_kg': float(m_crit)})
    return out


def entry_mass_estimate(m_fade, rho_m, A, v_of_t, h_of_t, t_range, atm=None):
    """Back-integrate ablation over the luminous track.

    dM^(1/3)/dτ = (B/3) rhoa v^3 backward in time, B = c_ml*A/(2 rho_m^(2/3))
    => M_entry^(1/3) = M_fade^(1/3) + (B/3) ∫ rhoa v^3 dt   (end -> start)

    Ablation only: this estimates the entry mass of the *surviving
    fragment*, not the original parent body — the parent's mass lost to
    fragmentation/grain erosion before the luminous end is not
    recoverable from this track segment.
    """
    if atm is None:
        atm = WindAtmosphere()
    B = ablation_coeff(rho_m, A) * A / (2 * rho_m ** (2. / 3))
    ts = np.linspace(t_range[0], t_range[1], 400)
    vs = np.array([v_of_t(t) for t in ts])
    hs = np.array([h_of_t(t) for t in ts])
    rhos = np.array([atm.at(max(h, 0.0))[1] for h in hs])
    integral = trapezoid(rhos * vs ** 3, ts)
    m13 = m_fade ** (1. / 3) + (B / 3) * integral
    return m13 ** 3


# --- Fragmentation scenarios ---------------------------------------------------
def build_scenarios(m_est, rho_m=3500.0, A=1.4, masses_grid=None,
                    dyn_press=None, fragment_masses=None):
    """Return list of scenario dicts:
    {'name', 'label', 'runs': [{'m':..,'rho':..,'A':..}], 'erode'}
    m_est: estimated surviving mass [kg], or None for hypothetical grid only.
    dyn_press: dynamic pressure at luminous fade-out [Pa]; when it
    exceeds the material's compressive strength, a strength-limited
    breakup scenario (Borovicka power-law + grain loss) is added.
    """
    if ((m_est is not None and (not np.isfinite(m_est) or m_est < 0))
            or not np.isfinite(rho_m) or rho_m <= 0):
        raise ValueError('Scenario mass must be non-negative or unknown, and density positive')
    if m_est == 0:
        return []
    if masses_grid is None:
        masses_grid = np.logspace(np.log10(5e-3), np.log10(5.0), 16)  # 5 g .. 5 kg
    if m_est is None:
        return [{'name': 'S1_hypothetical',
                 'label': 'Hypothetical fall line (5 g–5 kg at release)',
                 'runs': [{'m': m, 'rho': rho_m, 'A': A} for m in masses_grid],
                 'erode': True, 'hypothetical': True}]
    sc = []
    sc.append({'name': 'S0_intact', 'label': 'Single object (estimated mass)',
               'runs': [{'m': m_est, 'rho': rho_m, 'A': A}], 'erode': True})
    sc.append({'name': 'S1_fallline', 'label': 'Fall line (mass grid)',
               'runs': [{'m': m, 'rho': rho_m, 'A': A} for m in masses_grid],
               'erode': True, 'hypothetical': True})
    for fr in (0.9, 0.7, 0.5):
        sc.append({'name': f'S2_split_{int(fr*100)}', 'label':
                   f'Two-piece split {fr:.0%}/{1-fr:.0%}',
                   'runs': [{'m': m_est * fr, 'rho': rho_m, 'A': A},
                            {'m': m_est * (1 - fr), 'rho': rho_m, 'A': A}],
                   'erode': True})
    for n in (2, 4, 8, 16, 32):
        sc.append({'name': f'S3_equal_{n}', 'label': f'{n} equal fragments',
                   'runs': [{'m': m_est / n, 'rho': rho_m, 'A': A}
                            for _ in range(n)], 'erode': True})
    # power-law dN/dM ~ M^-2, fragments down to 5 g (inverse-CDF sampling)
    m_min = 0.005
    u = np.linspace(0.05, 0.95, 60)
    ms = 1.0 / (1.0 / m_min - u * (1.0 / m_min - 1.0 / m_est))
    ms *= m_est / ms.sum()
    sc.append({'name': 'S4_powerlaw', 'label': 'Power-law fragment swarm',
               'runs': [{'m': m, 'rho': rho_m, 'A': A} for m in ms],
               'erode': True})
    # strength-limited breakup (Borovicka 2007 / WMPL disruption):
    # dynamic pressure at fade-out exceeding the material's compressive
    # strength means the body was already fragmenting — 25% of the mass
    # is lost as grains, the rest breaks into a power-law spectrum
    strength = compressive_strength(rho_m)
    if fragment_masses is not None or (dyn_press is not None and dyn_press > strength):
        frags = (np.asarray(fragment_masses, dtype=float) if fragment_masses is not None
                 else fragmentation_spectrum(m_est, severity=dyn_press / strength))
        if frags.ndim != 1 or not np.all(np.isfinite(frags)) or np.any(frags < 0):
            raise ValueError('Invalid surviving fragment masses')
        sc.append({'name': 'S5_strength', 'label':
                   'Breakup at fade-out (strength-limited)',
                   'runs': [{'m': m, 'rho': rho_m, 'A': A} for m in frags],
                   'erode': True})
    # no-erosion variants of the interesting scenarios
    for name in ('S0_intact', 'S3_equal_4'):
        base = next(s for s in sc if s['name'] == name)
        sc.append({**base, 'name': name + '_noerosion',
                   'label': base['label'] + ' (no ablation)', 'erode': False})
    # nothing smaller than 5 gram is shown — drop runs below the cut
    # and scenarios left empty by it
    for s in sc:
        s['runs'] = [r for r in s['runs'] if r['m'] >= 5e-3]
    return [s for s in sc if s['runs']]


def _merge_label_masses(labels, tol=0.1):
    """labels: [(lon, lat, txt, col, mass)]. Drop entries whose mass is
    within ±tol of an already-kept one (keeps the first in track order,
    which follows scenario order — intact body first)."""
    kept, seen = [], []
    for lon, lat, txt, col, m in labels:
        if any(abs(m - s) <= tol * s for s in seen):
            continue
        seen.append(m)
        kept.append((lon, lat, txt, col, m))
    return kept


def _localized_number(value, spec, t):
    text = format(float(value), spec)
    dec = (t or {}).get('dec_sep', '.')
    return text.replace('.', dec) if dec != '.' else text


def sc_label(sc, t):
    """Localised scenario label; falls back to the built-in English label."""
    name = sc['name']
    import re as _re
    if sc.get('conditional') and sc.get('model'):
        model = t.get(f'df_frag_model_{sc["model"]}', sc['model'])
        mass = _localized_number(sc.get('initial_mass_kg', 0.), '.4g', t)
        label = t.get('df_frag_scenario',
                      '{model}, {mass} kg at first observation (modelled)').format(
                          model=model, mass=mass)
        density = sc.get('density_kg_m3')
        return (label + f' ({_localized_number(density, ".0f", t)} kg/m³)'
                if density is not None else label)
    if name.endswith('_noerosion'):
        base = sc_label({**sc, 'name': name[:-10]}, t)
        return base + ' ' + t.get('df_sc_noabl', '(no ablation)')
    key = {'S0_intact': 'df_sc_intact', 'S1_fallline': 'df_sc_fallline',
           'S1_hypothetical': 'df_sc_hypothetical',
           'S4_powerlaw': 'df_sc_powerlaw', 'S5_strength': 'df_sc_strength'}.get(name)
    if key:
        return t.get(key, sc['label'])
    m = _re.match(r'S2_split_(\d+)', name)
    if m:
        return t.get('df_sc_split', sc['label']).format(
            p=int(m.group(1)), q=100 - int(m.group(1)))
    m = _re.match(r'S3_equal_(\d+)', name)
    if m:
        return t.get('df_sc_equal', sc['label']).format(n=int(m.group(1)))
    return sc['label']


# --- Monte Carlo ---------------------------------------------------------------
def _idle_cpu_workers(task_count, max_workers=16, min_tasks=2):
    """Return a conservative worker count when more than half the CPUs are idle."""
    if task_count < max(2, int(min_tasks)):
        return 0
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    try:
        load = os.getloadavg()[0]
    except (AttributeError, OSError):
        return 0
    idle = int(cores - load)
    if idle <= cores // 2:
        return 0
    return min(int(task_count), idle, int(max_workers))


def _single_thread_worker():
    """Keep nested BLAS/Numba pools from oversubscribing worker processes."""
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    try:
        import numba
        numba.set_num_threads(1)
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except (ImportError, RuntimeError, ValueError):
        pass


def _parallel_map(worker, jobs, pool=None, max_workers=16,
                  min_tasks=2, allow_processes=True):
    """Map jobs through a supplied pool or an idle-load process pool."""
    jobs = list(jobs)
    if pool is not None:
        return list(pool.map(worker, jobs))
    workers = (_idle_cpu_workers(len(jobs), max_workers, min_tasks)
               if allow_processes else 0)
    if workers < 2:
        return [worker(job) for job in jobs]
    logging.info('Using %d idle CPU workers for %d simulation jobs',
                 workers, len(jobs))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_single_thread_worker) as active:
        chunksize = max(1, len(jobs) // (workers * 4))
        return list(active.map(worker, jobs, chunksize=chunksize))


def _prop_worker(job):
    (si, run, r0, v0, atm, h_ground, erode) = job
    return si, run, propagate(r0, v0, run['m'], run['rho'], run['A'],
                              atm, h_ground, erode=erode)


def _mc_worker(job):
    (r0, v0, m, rho, A, csv_path, h_ground, erode,
     lon, lat, event_dt, wind_offset) = job
    atm = WindAtmosphere(csv_path, lon=lon, lat=lat, event_dt=event_dt)
    return propagate(r0, v0, m, rho, A, atm, h_ground, erode=erode,
                     record_dt=1.0, wind_offset_enu=wind_offset)['impact']


def _mc_path_worker(job):
    (r0, v0, m, rho, A, csv_path, h_ground, erode,
     lon, lat, event_dt, wind_offset) = job
    atm = WindAtmosphere(csv_path, lon=lon, lat=lat, event_dt=event_dt)
    return propagate(r0, v0, m, rho, A, atm, h_ground, erode=erode,
                     record_dt=1.0, wind_offset_enu=wind_offset)


def monte_carlo(r0, v0_vec, m0, rho_m, A, csv_path, h_ground, n_runs=300,
                pos_err=100.0, vel_frac_err=0.05, dir_err_deg=0.1,
                mass_err=0.3, rho_err=500.0, shape_err=0.15,
                wind_err=2.0, seed=0, pool=None,
                lon=None, lat=None, event_dt=None,
                mass_samples=None, mass_bounds=None,
                return_paths=False):
    """Perturbed runs around nominal state. Returns list of impact dicts.

    mass_samples: optional array — draw each run's mass from this
    distribution (e.g. the breakup fragment spectrum) instead of
    m0*(1±mass_err), so a fragmented event produces a strewn field
    rather than a cloud around a single body. Alternatively mass_bounds
    gives a (min, max) end-of-luminous fragment mass range sampled
    stratified log-uniformly — every decade gets equal representation
    so the largest assumed fragments still appear in the cloud; it is
    a search-space prior, not a physical fragment number distribution.
    """
    r0, v0_vec = np.asarray(r0, dtype=float), np.asarray(v0_vec, dtype=float)
    if (any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in (r0, v0_vec))
            or not isinstance(n_runs, (int, np.integer)) or n_runs < 0
            or not all(np.isfinite(v) and v >= 0 for v in
                       (pos_err, vel_frac_err, dir_err_deg, mass_err, rho_err, shape_err, wind_err))
            or mass_err >= 1 or not all(np.isfinite(v) and v > 0 for v in (m0, rho_m, A))):
        raise ValueError('Invalid Monte Carlo state or uncertainty parameters')
    if mass_samples is not None:
        mass_samples = np.asarray(mass_samples, dtype=float)
        if mass_samples.ndim != 1 or not mass_samples.size or not np.all(np.isfinite(mass_samples)) or np.any(mass_samples <= 0):
            raise ValueError('Mass samples must be finite positive masses')
    if mass_bounds is not None:
        bounds = np.asarray(mass_bounds, dtype=float)
        if (mass_samples is not None or bounds.shape != (2,) or not np.all(np.isfinite(bounds))
                or not 0 < bounds[0] < bounds[1]):
            raise ValueError('Mass bounds must be finite, positive, increasing and exclusive of mass_samples')
        mass_min, mass_max = bounds
    rng = np.random.default_rng(seed)
    if mass_bounds is not None:
        # one draw per stratum over log-mass: guarantees the full
        # assumed range is covered even for modest n_runs
        bound_masses = mass_min * (mass_max / mass_min) ** (
            (np.arange(n_runs) + rng.random(n_runs)) / n_runs)
        rng.shuffle(bound_masses)
    else:
        bound_masses = None
    jobs = []
    speed = np.linalg.norm(v0_vec)
    direction = v0_vec / speed if speed > 0 else np.array([0., 0., 1.])
    for i in range(n_runs):
        r_p = r0 + rng.normal(0, pos_err, 3)
        v_p = v0_vec * max(0.0, rng.normal(1.0, vel_frac_err))
        # small angular perturbation of direction
        ortho = np.cross(direction, [1, 0, 0])
        if np.linalg.norm(ortho) < 1e-9:
            ortho = np.cross(direction, [0, 1, 0])
        ortho /= np.linalg.norm(ortho)
        d1 = rng.normal(0, np.radians(dir_err_deg)) * speed
        d2 = rng.normal(0, np.radians(dir_err_deg)) * speed
        ortho2 = np.cross(direction, ortho); ortho2 /= np.linalg.norm(ortho2)
        v_p = v_p + d1 * ortho + d2 * ortho2
        if bound_masses is not None:
            m_p = float(bound_masses[i])
        elif mass_samples is not None and len(mass_samples):
            m_p = float(rng.choice(mass_samples)) * rng.uniform(1 - mass_err, 1 + mass_err)
        else:
            m_p = m0 * rng.uniform(1 - mass_err, 1 + mass_err)
        rho_p = max(500.0, rng.normal(rho_m, rho_err))
        A_p = float(np.clip(rng.normal(A, shape_err), 0.9, 3.0))
        wind_offset = np.r_[rng.normal(0.0, wind_err, 2), 0.0]
        jobs.append((r_p, v_p, m_p, rho_p, A_p, csv_path, h_ground, True,
                     lon, lat, event_dt, wind_offset))
    worker = _mc_path_worker if return_paths else _mc_worker
    if pool is not None:
        chunksize = max(1, len(jobs) // 64)
        try:
            return pool.map(worker, jobs, chunksize=chunksize)
        except TypeError:
            return pool.map(worker, jobs)
    allow_processes = getattr(propagate, '__module__', None) == __name__
    return _parallel_map(worker, jobs, min_tasks=64,
                         allow_processes=allow_processes)


def _release_approach_path(fragment):
    """Path from the parent entry through this cohort's dark release."""
    trajectory = fragment.get('trajectory') or {}
    release = fragment.get('dark_release') or {}
    if not trajectory or not release:
        return {}
    try:
        lon = np.asarray(trajectory.get('lon', []), dtype=float)
        lat = np.asarray(trajectory.get('lat', []), dtype=float)
        height = np.asarray(trajectory.get('h', []), dtype=float)
        r_release = np.asarray(release.get('r_ecef'), dtype=float)
        if (not (len(lon) == len(lat) == len(height)) or not len(lon)
                or r_release.shape != (3,)):
            return {}
        points = np.array([llh2ecef(lo, la, hh)
                           for lo, la, hh in zip(lon, lat, height)])
        index = int(np.argmin(np.linalg.norm(points - r_release, axis=1)))
    except Exception:
        return {}
    return {'lon': list(lon[:index + 1]),
            'lat': list(lat[:index + 1]),
            'h': list(height[:index + 1])}


def _forward_experiment_worker(job):
    """Run one conditional forward experiment or magnitude inversion."""
    (index, mass, model, density, r_start, direction, speed, atm,
     h_ground, model_options, target_mag, duration_s) = job
    def propagate_one(run_mass):
        return propagate_fragmentation(
            r_start, direction * speed, float(run_mass), density, 1.4,
            atm, float(h_ground), model=model, **model_options)
    try:
        if target_mag is None:
            return index, propagate_one(mass), None, None
        experiment, inversion = _solve_fragmentation_mass_for_magnitude(
            model, propagate_one, target_mag, speed, duration_s)
        return index, experiment, inversion, None
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        return index, None, None, error


def _forward_mc_allocations(releases, n_runs):
    """Allocate total MC draws across landed cohorts by landed mass."""
    n_releases = len(releases)
    if not releases or n_runs <= 0:
        return np.zeros(n_releases, dtype=int)
    counts = np.zeros(n_releases, dtype=int)
    if n_runs <= n_releases:
        order = np.argsort([-r['total_final_mass_kg'] for r in releases])
        counts[order[:n_runs]] = 1
        return counts
    counts[:] = 1
    remaining = n_runs - n_releases
    masses = np.asarray([r['total_final_mass_kg'] for r in releases],
                        dtype=float)
    weights = masses / math.fsum(masses) if math.fsum(masses) > 0 else np.full(
        n_releases, 1.0 / n_releases)
    shares = weights * remaining
    counts += np.floor(shares).astype(int)
    left = n_runs - int(np.sum(counts))
    if left:
        order = np.argsort(-(shares - np.floor(shares)))
        counts[order[:left]] += 1
    return counts


def _forward_mc_impacts(experiments, wind_csv, h_ground, n_runs, seed,
                        event_dt=None, pool=None):
    """Conditional MC dark-flight samples from landed cohort release states.

    Fragmenting models use a legacy-style disruption spectrum scaled to the
    represented landed cohort mass. The intact ``single`` control instead
    perturbs its actual release mass; imposing a fragment spectrum there would
    make the control model appear unrealistically light. These are conditional
    sensitivity realizations, not stochastic fragmentation draws.
    """
    if not isinstance(n_runs, (int, np.integer)) or n_runs < 0:
        raise ValueError('Forward Monte Carlo run count must be a non-negative integer')
    all_impacts = []
    for experiment_index, experiment in enumerate(experiments):
        releases = [fragment for fragment in experiment['fragments']
                    if fragment.get('dark_release') is not None
                    and (fragment.get('impact') or {}).get('landed')]
        allocations = _forward_mc_allocations(releases, int(n_runs))
        parameters = experiment.get('parameters', {})
        use_fragment_spectrum = experiment['model'] != 'single'
        impacts = []
        attempted = 0
        for cohort_index, (fragment, count) in enumerate(
                zip(releases, allocations)):
            if count <= 0:
                continue
            release = fragment['dark_release']
            r0 = np.asarray(release['r_ecef'], dtype=float)
            v0 = np.asarray(release['v_ecef'], dtype=float)
            cohort_mass = float(fragment.get('total_final_mass_kg',
                                             release['mass_kg']))
            if (r0.shape != (3,) or v0.shape != (3,)
                    or not np.all(np.isfinite(r0))
                    or not np.all(np.isfinite(v0)) or not cohort_mass > 0):
                continue
            lon, lat, height = ecef2llh(r0)
            speed = float(np.linalg.norm(v0))
            mc_atm = WindAtmosphere(wind_csv, lon=lon, lat=lat,
                                    event_dt=event_dt)
            rho_air = float(mc_atm.at(max(height, 0.0))[1])
            strength = max(float(fragment.get(
                'strength_pa', parameters.get('strength_pa', 1e6))), 1e-12)
            severity = max(1.0, rho_air * speed ** 2 / strength)
            if use_fragment_spectrum:
                mass_samples = fragmentation_spectrum(
                    cohort_mass, mass_index=parameters.get('mass_index', 2.0),
                    severity=severity)
                mass_samples = mass_samples[mass_samples > 0]
            else:
                mass_samples = None
            if mass_samples is not None and not mass_samples.size:
                mass_samples = None
            approach = _release_approach_path(fragment)
            attempted += int(count)
            for result in monte_carlo(
                    r0, v0, cohort_mass, parameters.get('rho_m', 3500.0),
                    parameters.get('A_shape', 1.4), wind_csv, h_ground,
                    n_runs=int(count), seed=seed + 100003 * experiment_index +
                    1009 * cohort_index, pool=pool, lon=lon, lat=lat,
                    event_dt=event_dt,
                    mass_samples=mass_samples,
                    return_paths=True):
                impact = result['impact']
                if not impact.get('landed', True):
                    continue
                step = max(1, len(result['lon']) // 150)
                path_idx = np.unique(np.r_[
                    np.arange(0, len(result['lon']), step),
                    len(result['lon']) - 1])
                dark_path = {key: result[key][path_idx].tolist()
                             for key in ('lon', 'lat', 'h')}
                trajectory = ({key: approach[key] + dark_path[key]
                               for key in ('lon', 'lat', 'h')}
                              if approach else dark_path)
                impacts.append({**impact,
                                'trajectory': trajectory,
                                'model': experiment['model'],
                                'density_kg_m3': parameters.get('rho_m'),
                                'experiment_index': experiment_index,
                                'fragment_id': fragment['id'],
                                'multiplicity': fragment['multiplicity'],
                                'cohort_final_mass_kg':
                                    fragment['final_mass_kg'],
                                'release_parent_mass_kg': cohort_mass,
                                'spectrum_parent_mass_kg': (cohort_mass if use_fragment_spectrum
                                                            else None),
                                'spectrum_severity': (severity if use_fragment_spectrum
                                                      else None),
                                'conditional': True})
        experiment['monte_carlo'] = {
            'requested': int(n_runs), 'attempted': int(attempted),
            'n': len(impacts), 'n_release_cohorts': len(releases),
            'impacts': impacts, 'conditional': True,
            'allocation': 'landed-cohort mass, minimum one draw per cohort',
            'mass_sampling': ('legacy-style disruption spectrum per landed cohort'
                              if use_fragment_spectrum else
                              'mass perturbation around the intact release mass'),
            'method': ('modelled dark-flight realizations from each '
                       'deterministic landed cohort release state'),
            'caveat': ('MC outcomes are modelled what-if realizations, not '
                       'measured fragment masses or calibrated probabilities')}
        all_impacts.extend(impacts)
    return all_impacts


def _forward_mc_impacts_parallel(experiments, wind_csv, h_ground, n_runs,
                                 seed, event_dt=None, pool=None):
    """Use one temporary process pool for all forward MC cohort draws."""
    if not isinstance(n_runs, (int, np.integer)) or n_runs < 0:
        raise ValueError('Forward Monte Carlo run count must be a non-negative integer')
    if pool is not None:
        return _forward_mc_impacts(
            experiments, wind_csv, h_ground, n_runs, seed,
            event_dt=event_dt, pool=pool)
    allow_processes = getattr(propagate, '__module__', None) == __name__
    workers = (_idle_cpu_workers(
        int(n_runs) * max(1, len(experiments)), min_tasks=256)
        if allow_processes and int(n_runs) > 0 else 0)
    if workers < 2:
        return _forward_mc_impacts(
            experiments, wind_csv, h_ground, n_runs, seed,
            event_dt=event_dt)
    logging.info('Using %d idle CPU workers for forward Monte Carlo',
                 workers)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_single_thread_worker) as active:
        return _forward_mc_impacts(
            experiments, wind_csv, h_ground, n_runs, seed,
            event_dt=event_dt, pool=active)


# --- Ground elevation ----------------------------------------------------------
def ground_elevation(lat, lon):
    """Ground height [m]. Kartverket for Norway, else Open-Meteo elevation."""
    import requests
    if 57.5 <= lat <= 71.5 and 3.0 <= lon <= 32.0:
        try:
            r = requests.get('https://ws.geonorge.no/hoydedata/v1/punkt',
                             params={'nord': lat, 'ost': lon, 'koordsys': 4258},
                             timeout=10)
            if r.ok:
                h = r.json().get('hoyde')
                if h is not None:
                    return float(h)
        except Exception as e:
            logging.debug(f'Kartverket elevation failed: {e}')
    try:
        r = requests.get('https://api.open-meteo.com/v1/elevation',
                         params={'latitude': lat, 'longitude': lon},
                         timeout=10)
        if r.ok:
            el = r.json().get('elevation')
            if el:
                return max(0.0, float(el[0]))
    except Exception as e:
        logging.debug(f'Open-Meteo elevation failed: {e}')
    return 0.0


# --- Output writers ------------------------------------------------------------
def write_json(results, path):
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        if isinstance(o, np.generic):
            return conv(o.item())
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    Path(path).write_text(json.dumps(conv(results), indent=1, allow_nan=False),
                          encoding='utf-8')


def write_geojson(scenarios, path, mc_impacts=None):
    feats = []
    for sc in scenarios:
        for run in sc['results']:
            imp = run['impact']
            if not imp.get('landed', True):
                continue
            feats.append({
                'type': 'Feature',
                'geometry': {'type': 'Point',
                             'coordinates': [imp['lon'], imp['lat'], imp['h']]},
                'properties': {'scenario': sc['name'], 'mass_kg': imp['m'],
                               'model': sc.get('model'),
                               'experiment_index': sc.get('experiment_index'),
                               'density_kg_m3': sc.get('density_kg_m3'),
                               'release_mass_kg': run['m'],
                               'hypothetical': sc.get('hypothetical', False),
                               'conditional': sc.get('conditional', False),
                               'impact_speed_ms': imp['v']}})
    for imp in (mc_impacts or []):
        trajectory = imp.get('trajectory') or {}
        if len(trajectory.get('lon', ())) >= 2:
            feats.append({
                'type': 'Feature',
                'geometry': {'type': 'LineString',
                             'coordinates': [[lo, la, hh] for lo, la, hh in zip(
                                 trajectory['lon'], trajectory['lat'],
                                 trajectory['h'])]},
                'properties': {'scenario': imp.get('model', 'monte_carlo'),
                               'model': imp.get('model'), 'monte_carlo': True,
                               'density_kg_m3': imp.get('density_kg_m3'),
                               'experiment_index': imp.get('experiment_index'),
                               'feature_kind': 'path',
                               'fragment_id': imp.get('fragment_id'),
                               'multiplicity': imp.get('multiplicity'),
                               'conditional': imp.get('conditional', False)}})
        feats.append({
            'type': 'Feature',
            'geometry': {'type': 'Point',
                         'coordinates': [imp['lon'], imp['lat'], imp.get('h', 0.0)]},
            'properties': {'scenario': imp.get('model', 'monte_carlo'),
                           'model': imp.get('model'), 'monte_carlo': True,
                           'density_kg_m3': imp.get('density_kg_m3'),
                           'experiment_index': imp.get('experiment_index'),
                           'mass_kg': imp['m'],
                           'fragment_id': imp.get('fragment_id'),
                           'multiplicity': imp.get('multiplicity'),
                           'conditional': imp.get('conditional', False),
                           'impact_speed_ms': imp['v']}})
    Path(path).write_text(json.dumps(
        {'type': 'FeatureCollection', 'features': feats}), encoding='utf-8')


def write_kml(scenarios, end_llh, path, mc_impacts=None, mc_assumed=False):
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             f'<name>Dark flight</name>',
             '<Folder><name>Fall paths</name>']
    colours = ['ff1400ff', 'ff00a5ff', 'ff00ff00', 'ffaa00ff', 'ffffaa00']
    for si, sc in enumerate(scenarios):
        col = colours[si % len(colours)]
        if sc.get('conditional', False):
            description = ('<description>Modelled fragmentation-model trajectory; '
                           'not a fitted result or evidence of meteorite survival.</description>')
        elif sc.get('hypothetical', False):
            description = ('<description>Hypothetical mass-grid trajectory; '
                           'not evidence of meteorite survival.</description>')
        else:
            description = ''
        for run in sc['results']:
            path_pts = ' '.join(
                f'{lo:.6f},{la:.6f},{hh:.0f}'
                for lo, la, hh in zip(run['lon'], run['lat'], run['h']))
            parts.append(
                f'<Placemark><name>{sc["name"]} {run["m"]*1000:.1f}g</name>{description}'
                f'<Style><LineStyle><color>{col}</color><width>1.5</width></LineStyle>'
                f'</Style><LineString><altitudeMode>absolute</altitudeMode>'
                f'<coordinates>{path_pts}</coordinates></LineString></Placemark>')
            imp = run['impact']
            if not imp.get('landed', True):
                continue
            parts.append(
                f'<Placemark><name>{imp["m"]*1000:.1f} g</name>{description}'
                f'<Point><coordinates>{imp["lon"]:.6f},{imp["lat"]:.6f},'
                f'{imp["h"]:.0f}</coordinates></Point></Placemark>')
    parts.append('</Folder>')
    # Monte Carlo fall area — translucent red polygon + hatch lines
    ring = _mc_ring_ll(mc_impacts)
    if ring is not None:
        area_name = ('Modelled fall area (MC)' if mc_assumed
                     else 'Fall area (MC)')
        area_desc = ('<description>Monte Carlo samples from modelled release '
                     'states; not a measured surviving mass.</description>'
                     if mc_assumed else '')
        coords = ' '.join(f'{lo:.6f},{la:.6f},0'
                          for lo, la in ring)
        parts.append(f'<Folder><name>{area_name}</name>')
        parts.append(
            f'<Placemark><name>{area_name}</name>{area_desc}'
            '<Style><PolyStyle><color>4d0000ff</color></PolyStyle>'
            '<LineStyle><color>ff0000ff</color><width>2</width>'
            '</LineStyle></Style>'
            '<Polygon><altitudeMode>clampToGround</altitudeMode>'
            '<outerBoundaryIs><LinearRing><tessellate>1</tessellate><coordinates>'
            f'{coords}</coordinates></LinearRing></outerBoundaryIs>'
            '</Polygon></Placemark>')
        # individual MC trajectories and impact points — terrain-clamped points
        if mc_impacts:
            for i, impact in enumerate(mc_impacts):
                trajectory = impact.get('trajectory') or {}
                if len(trajectory.get('lon', ())) < 2:
                    continue
                coords = ' '.join(
                    f'{lo:.6f},{la:.6f},{hh:.0f}' for lo, la, hh in zip(
                        trajectory['lon'], trajectory['lat'],
                        trajectory['h']))
                parts.append(
                    f'<Placemark><name>MC path {i + 1}: '
                    f'{impact.get("model", "sample")}</name>{area_desc}'
                    '<Style><LineStyle><color>660000ff</color>'
                    '<width>1</width></LineStyle></Style>'
                    '<LineString><altitudeMode>absolute</altitudeMode>'
                    f'<coordinates>{coords}</coordinates></LineString></Placemark>')
            pt_geoms = ''.join(
                f'<Point><coordinates>{i["lon"]:.6f},{i["lat"]:.6f},0'
                '</coordinates></Point>' for i in mc_impacts)
            parts.append(
                f'<Placemark><name>MC impacts (n={len(mc_impacts)})</name>'
                '<Style><IconStyle><color>ff0000ff</color>'
                '<scale>0.4</scale></IconStyle>'
                '<LabelStyle><scale>0</scale></LabelStyle></Style>'
                f'<MultiGeometry>{pt_geoms}</MultiGeometry></Placemark>')
        # hatch lines at 45 deg clipped to the ring (KML has no
        # hatch fill — draw them as ground-clamped LineStrings)
        try:
            from shapely.geometry import Polygon, LineString, \
                MultiLineString
            poly = Polygon(list(zip(ring[:, 0], ring[:, 1])))
            x0, y0, x1, y1 = poly.bounds
            diag = math.hypot(x1 - x0, y1 - y0)
            spacing = max(diag / 15.0, 1e-4)
            segs = []
            t = -diag
            while t < diag:
                cxx = (x0 + x1) / 2 - t / math.sqrt(2)
                cyy = (y0 + y1) / 2 + t / math.sqrt(2)
                ex1 = (cxx - diag / math.sqrt(2),
                       cyy - diag / math.sqrt(2))
                ex2 = (cxx + diag / math.sqrt(2),
                       cyy + diag / math.sqrt(2))
                cl = poly.intersection(LineString([ex1, ex2]))
                geoms = ([cl] if isinstance(cl, LineString)
                         else list(cl.geoms) if isinstance(
                             cl, MultiLineString) else [])
                for ls in geoms:
                    if not ls.is_empty and len(ls.coords) >= 2:
                        segs.append(list(ls.coords))
                t += spacing
            if segs:
                parts.append(
                    '<Placemark><name>MC hatch</name>'
                    '<Style><LineStyle><color>ff0000ff</color>'
                    '<width>1</width></LineStyle></Style>'
                    '<MultiGeometry>')
                for seg in segs:
                    sc = ' '.join(f'{lo:.6f},{la:.6f},0'
                                  for lo, la in seg)
                    parts.append(
                        '<LineString><tessellate>1</tessellate>'
                        '<altitudeMode>clampToGround'
                        f'</altitudeMode><coordinates>{sc}'
                        '</coordinates></LineString>')
                parts.append('</MultiGeometry></Placemark>')
        except Exception:
            pass
        parts.append('</Folder>')
    parts.append('</Document></kml>')
    Path(path).write_text(''.join(parts), encoding='utf-8')


def _mc_outline(pts, pad_frac=0.15, pad_min=None):
    """Buffered outline around a point cloud -> closed ring (N+1,2).

    Convex hull buffered outward by pad_frac * max cloud radius
    (or pad_min if larger). Buffering rounds corners and always
    encloses every input point, even for degenerate/sliver hulls."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return None
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.all(np.isfinite(pts)):
        raise ValueError('MC outline requires finite 2D points')
    if not np.isfinite(pad_frac) or pad_frac < 0 or (pad_min is not None and (not np.isfinite(pad_min) or pad_min < 0)):
        raise ValueError('MC outline padding must be finite and non-negative')
    if len(pts) == 1:
        pts = np.vstack([pts, pts + 1e-9])  # make a tiny segment
    cx, cy = pts.mean(axis=0)
    rad = np.sqrt(((pts - [cx, cy]) ** 2).sum(axis=1)).max()
    pad = max(pad_frac * rad, pad_min or 0.0, 1e-9)
    try:
        from shapely.geometry import MultiPoint
        poly = MultiPoint(pts).convex_hull.buffer(pad, 64)
        return np.asarray(poly.exterior.coords)
    except Exception:
        # no shapely: fallback = convex hull of padded point circles
        from scipy.spatial import ConvexHull
        angles = np.linspace(0, 2 * math.pi, 128, endpoint=False)
        offsets = np.column_stack((np.cos(angles), np.sin(angles))) * pad / math.cos(math.pi / 128)
        expanded = (pts[:, None, :] + offsets).reshape(-1, 2)
        hull = ConvexHull(expanded)
        ring = expanded[hull.vertices]
        return np.vstack([ring, ring[0]])


def _points_in_polygon(pts, ring):
    """Vectorised even-odd ray-casting test. `ring` is a closed
    (N+1, 2) polygon (last vertex == first). Returns bool array."""
    px, py = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = ring[:-1, 0][None, :], ring[:-1, 1][None, :]
    x2, y2 = ring[1:, 0][None, :], ring[1:, 1][None, :]
    dy = y2 - y1
    slope = np.divide(x2 - x1, dy, out=np.zeros_like(dy), where=dy != 0)
    crosses = ((y1 > py) != (y2 > py)) & (px < x1 + (py - y1) * slope)
    return crosses.sum(axis=1) % 2 == 1


def _mc_ring_ll(mc_impacts, coverage=0.85, pad_min_km=0.15, n_pts=96):
    """Curved fall-area outline enclosing at least `coverage` of the
    MC impacts. Fragment mass sorting lays the cloud out along the
    (generally curved) fall line, so the outline is a variable-width
    band around a spine fitted through the inlier cloud — not an
    ellipse. The spine is the binned median cross-track position along
    the principal axis; half-widths come from local bin residuals
    (floored at pad_min_km), with semicircular end caps. Inliers are
    the central `coverage` fraction in Mahalanobis distance, so true
    outliers can neither extend nor reshape the band while the
    elongated ends are preserved. Returns a closed (N+1, 2) lon/lat
    ring or None."""
    if not mc_impacts:
        return None
    if not 0 < coverage <= 1:
        raise ValueError('MC ring coverage must be in (0, 1]')
    if not np.isfinite(pad_min_km) or pad_min_km < 0:
        raise ValueError('MC ring padding must be finite and non-negative')
    lats = np.array([i['lat'] for i in mc_impacts])
    lons = np.array([i['lon'] for i in mc_impacts])
    if not np.all(np.isfinite(lats)) or not np.all(np.isfinite(lons)):
        raise ValueError('MC outline requires finite impact coordinates')
    lat0 = math.radians(lats.mean())
    c = 111.32
    pts = np.vstack([lons * c * math.cos(lat0), lats * c]).T
    n = len(pts)
    centre = np.median(pts, axis=0)
    keep = max(3, int(math.ceil(coverage * n)))
    if n > 1:
        cov0 = np.cov((pts - centre).T)
        w0, V0 = np.linalg.eigh(np.atleast_2d(cov0))
        d_sel = np.sqrt(((pts - centre) @ V0) ** 2
                        @ (1.0 / np.maximum(w0, 1e-12)))
    else:
        d_sel = np.zeros(n)
    sel = np.argsort(d_sel)[:keep] if keep < n else np.arange(n)
    fit_pts = pts[sel]
    # principal axis of the inlier cloud = along-track direction
    if len(fit_pts) > 1:
        cov = np.cov((fit_pts - fit_pts.mean(axis=0)).T)
        eigvals, eigvecs = np.linalg.eigh(np.atleast_2d(cov))
        e1 = eigvecs[:, int(np.argmax(eigvals))]
    else:
        e1 = np.array([1.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.array([-e1[1], e1[0]])
    uv_all = (pts - centre) @ np.vstack([e1, e2]).T
    uv = uv_all[sel]
    u, v = uv[:, 0], uv[:, 1]
    # spine: median cross-track position in quantile bins of u
    n_bin = int(np.clip(len(fit_pts) // 8, 3, 12))
    edges = np.quantile(u, np.linspace(0., 1., n_bin + 1))
    edges[0] -= 1e-9
    su, sv, hw = [], [], []
    for b in range(n_bin):
        upper = u <= edges[b + 1] if b == n_bin - 1 else u < edges[b + 1]
        m = (u >= edges[b]) & upper
        if not m.any():
            continue
        mv = float(np.median(v[m]))
        su.append(float(np.median(u[m])))
        sv.append(mv)
        hw.append(max(float(np.abs(v[m] - mv).max()), pad_min_km))
    if len(su) < 2:
        # degenerate cloud: a pad_min circle
        t = np.linspace(0, 2 * math.pi, 33)
        ring = centre + pad_min_km * np.column_stack([np.cos(t),
                                                      np.sin(t)])
        return np.vstack([ring[:, 0] / (c * math.cos(lat0)),
                          ring[:, 1] / c]).T
    su, sv, hw = np.asarray(su), np.asarray(sv), np.asarray(hw)
    if len(su) >= 5:
        kernel = np.array([0.25, 0.5, 0.25])
        sv[1:-1] = np.convolve(sv, kernel, 'same')[1:-1]
        hw[1:-1] = np.convolve(hw, kernel, 'same')[1:-1]
    need = int(math.ceil(coverage * n))
    cap_pts = 24
    for _ in range(5):
        # chord-sag factor: keeps the discretised caps outside the
        # true semicircle so boundary points stay covered
        inflate = hw / math.cos(math.pi / cap_pts)
        uu = np.linspace(su[0], su[-1], max(len(su) * 4, n_pts // 2))
        vv = np.interp(uu, su, sv)
        ww = np.interp(uu, su, inflate)
        top = np.column_stack([uu, vv + ww])
        th = np.linspace(math.pi / 2, -math.pi / 2, cap_pts)
        cap_e = np.column_stack([su[-1] + ww[-1] * np.cos(th),
                                 vv[-1] + ww[-1] * np.sin(th)])
        bot = np.column_stack([uu[::-1], vv[::-1] - ww[::-1]])
        th = np.linspace(-math.pi / 2, -3 * math.pi / 2, cap_pts)
        cap_s = np.column_stack([su[0] + ww[0] * np.cos(th),
                                 vv[0] + ww[0] * np.sin(th)])
        ring_uv = np.vstack([top, cap_e, bot, cap_s])
        inside = _points_in_polygon(uv_all,
                                    np.vstack([ring_uv, ring_uv[:1]]))
        if inside.sum() >= need:
            break
        hw = hw * 1.35
    ring_xy = centre + ring_uv[:, 0:1] * e1 + ring_uv[:, 1:2] * e2
    return np.vstack([ring_xy[:, 0] / (c * math.cos(lat0)),
                      ring_xy[:, 1] / c]).T


def _disintegration_burst_points(sc, start_llh, end_llh, n_rays=48):
    """Deterministic dark-orange fan centered on the luminous-path endpoint."""
    if (start_llh is None
            or sc.get('mass_budget', {}).get('landed_kg', 0.0) > 0):
        return None
    start = np.asarray(start_llh, dtype=float)
    end = np.asarray(end_llh, dtype=float)
    r_start, r_end = llh2ecef(*start), llh2ecef(*end)
    flight_ecef = r_end - r_start
    flight_norm = np.linalg.norm(flight_ecef)
    if not flight_norm:
        return None
    flight_ecef /= flight_norm
    rotation = enu2ecef_mat(end[1], end[0])
    forward = rotation.T @ flight_ecef
    forward /= np.linalg.norm(forward)
    lateral = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    if np.linalg.norm(lateral) < 1e-8:
        lateral = np.array([1.0, 0.0, 0.0])
    else:
        lateral /= np.linalg.norm(lateral)
    transverse = np.cross(forward, lateral)
    transverse /= np.linalg.norm(transverse)
    rng = np.random.default_rng(20260423 + len(sc.get('fragments', [])))
    lateral_angles = (np.linspace(-0.10, 0.10, n_rays)
                      + rng.normal(0.0, 0.007, n_rays))
    transverse_angles = rng.normal(0.0, 0.018, n_rays)
    distances = np.linspace(4.0, 12.0, n_rays) * rng.uniform(0.85, 1.15, n_rays)
    directions_enu = (forward[None, :]
                      + np.tan(lateral_angles)[:, None] * lateral
                      + np.tan(transverse_angles)[:, None] * transverse)
    directions_enu /= np.linalg.norm(directions_enu, axis=1)[:, None]
    directions_ecef = directions_enu @ rotation.T
    ecef_points = r_end[None, :] + directions_ecef * distances[:, None] * 1000.0
    lon, lat, h = _ecef2llh_arr(ecef_points[:, 0], ecef_points[:, 1],
                                ecef_points[:, 2])
    endpoints = np.column_stack([np.degrees(lon), np.degrees(lat),
                                 np.maximum(0.0, h)])
    return np.repeat(end[None, :], n_rays, axis=0), endpoints


def write_map(scenarios, end_llh, mc_impacts, out_svg, title='',
              translations=None, mc_assumed=False, start_llh=None,
              observed_duration_s=None):
    """Fall-area map: Kartverket basemap, impact markers, MC ellipse."""
    translations = translations or {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lons_all, lats_all = [end_llh[0]], [end_llh[1]]
    luminous_track = None
    if start_llh is not None:
        luminous_track = _straight_track_llh(start_llh, end_llh)
        lons_all += list(luminous_track[0])
        lats_all += list(luminous_track[1])
    disintegration_clouds = []
    for sc in scenarios:
        burst = _disintegration_burst_points(sc, start_llh, end_llh)
        if burst is not None:
            origins, endpoints = burst
            disintegration_clouds.append((sc, origins, endpoints))
            lons_all += [end_llh[0]] + [point[0] for point in endpoints]
            lats_all += [end_llh[1]] + [point[1] for point in endpoints]
    for sc in scenarios:
        for run in sc['results']:
            lons_all += list(run['lon']) + [run['impact']['lon']]
            lats_all += list(run['lat']) + [run['impact']['lat']]
    for imp in (mc_impacts or []):
        lons_all.append(imp['lon']); lats_all.append(imp['lat'])
    pad = 0.15
    lon_min, lon_max = min(lons_all) - pad, max(lons_all) + pad
    lat_min, lat_max = min(lats_all) - pad, max(lats_all) + pad

    fig = plt.figure(figsize=(10, 9))
    kv = None
    try:
        from metrack import _fetch_kartverket_topo
        # Kartverket covers Norway only — require real content at the
        # luminous-path end point and every landed impact point.
        poi = [(end_llh[0], end_llh[1])]
        for sc in scenarios:
            for run in sc['results']:
                if run['impact'].get('landed', True):
                    poi.append((run['impact']['lon'], run['impact']['lat']))
        kv = _fetch_kartverket_topo([lon_min, lon_max], [lat_min, lat_max],
                                    poi_lonlat=poi)
    except Exception as e:
        logging.debug(f'Kartverket tiles unavailable for darkflight map: {e}')

    pc = None
    ax = None
    try:
        import cartopy.crs as ccrs
        pc = ccrs.PlateCarree()
        if kv:
            ax = fig.add_subplot(projection=ccrs.UTM(32))
            img, ext = kv
            ax.imshow(img, extent=ext, origin='upper',
                      transform=ccrs.UTM(32))
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pc)
        else:
            # Outside Kartverket coverage — fall back to OSM tiles
            ax = fig.add_subplot(projection=ccrs.UTM(32))
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=pc)
            lat_span = lat_max - lat_min
            # ~4 tiles across the extent
            zoom = int(math.ceil(math.log2(1440 / max(lat_span, 0.05))))
            zoom = max(8, min(zoom, 12))
            from cartopy.io.img_tiles import OSM
            ax.add_image(OSM(), zoom)
        ax.gridlines(draw_labels=True, alpha=0.3)
    except Exception as e:
        logging.debug(f'cartopy/OSM unavailable for darkflight map: {e}')
        pc = None
        if ax is not None:
            ax.remove()
            ax = None
    if ax is None:
        pc = None
        ax = fig.add_subplot()
        ax.set_xlim(lon_min, lon_max); ax.set_ylim(lat_min, lat_max)

    def trplot(*args, **kw):
        if pc:
            ax.plot(*args, transform=pc, **kw)
        else:
            ax.plot(*args, **kw)

    # unique runs: S3_equal_N scenarios have N identical trajectories —
    # one line + one marker is enough.  The S4 power-law swarm is ~60
    # near-identical sub-gram tracks — excluded from the plots entirely
    # (kept in JSON/KML for completeness).
    plot_sc = [s for s in scenarios
               if s.get('erode', True) and s['name'] != 'S4_powerlaw']
    labels2draw = []
    cmap = plt.cm.viridis
    for si, sc in enumerate(plot_sc):
        col = cmap(si / max(len(plot_sc) - 1, 1))
        uniq, seen_m = [], set()
        for run in sc['results']:
            if run['m'] not in seen_m:
                seen_m.add(run['m'])
                uniq.append(run)
        first_landed = next((r for r in uniq
                             if r['impact'].get('landed', True)), None)
        for run in uniq:
            if len(run.get('lon', ())) >= 2:
                trplot(run['lon'], run['lat'], color=col, lw=1.1,
                       alpha=0.55)
            if run['impact'].get('landed', True):
                trplot([run['impact']['lon']], [run['impact']['lat']],
                       'o', ms=max(3, min(10, np.log10(max(run['m'],1e-6) * 1e6) / 1.5)),
                       color=col)
                m_kg = run['impact']['m']
                m_txt = (f'{m_kg:.3g} kg' if m_kg >= 1
                         else f'{m_kg * 1000:.3g} g')
                dec = translations.get('dec_sep', '.')
                if dec != '.':
                    m_txt = m_txt.replace('.', dec)
                labels2draw.append((run['impact']['lon'],
                                    run['impact']['lat'], m_txt, col,
                                    run['impact']['m']))
    # merge labels within ±10% mass across all scenarios — a swarm of
    # near-identical fragments gets one label for the lot
    labels2draw = [(lon, lat, txt, col) for lon, lat, txt, col, m in
                   _merge_label_masses(labels2draw)]
    if luminous_track is not None and len(luminous_track[0]):
        trplot(luminous_track[0], luminous_track[1], '-', color='red',
               lw=1.8, alpha=0.85)
    trplot([end_llh[0]], [end_llh[1]], 'r*', ms=14)
    for sc, origins, endpoints in disintegration_clouds:
        for origin, endpoint in zip(origins, endpoints):
            midpoint = origin + 0.58 * (endpoint - origin)
            trplot([origin[0], midpoint[0]], [origin[1], midpoint[1]],
                   color='chocolate', lw=1.7, alpha=0.70)
            trplot([midpoint[0], endpoint[0]], [midpoint[1], endpoint[1]],
                   color='chocolate', lw=0.9, alpha=0.20)
        trplot(endpoints[:, 0], endpoints[:, 1], '.', ms=2,
               color='chocolate', alpha=0.20)
        trplot([end_llh[0]], [end_llh[1]], 'D', ms=7, color='chocolate')

    # collision-aware placement: measure real text bboxes with the
    # renderer, try candidate offsets in order, first fit wins
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed_boxes = []
    # reserve space around every impact marker — a label must not cover
    # its own or any other point
    from matplotlib.transforms import Bbox
    for lon, lat, txt, col in labels2draw:
        if pc:
            px, py = ax.transData.transform(
                ax.projection.transform_point(lon, lat, pc))
        else:
            px, py = ax.transData.transform((lon, lat))
        placed_boxes.append(Bbox.from_extents(px - 5, py - 5,
                                              px + 5, py + 5))
    # radial candidate offsets — try every 30° at increasing radii, so a
    # label lands in the nearest free direction (leader line included)
    candidates = []
    for r in (12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100):
        for ang in range(0, 360, 30):
            dx = r * math.cos(math.radians(ang))
            dy = r * math.sin(math.radians(ang))
            ha = ('center' if abs(dx) < r * 0.35
                  else ('left' if dx > 0 else 'right'))
            candidates.append((dx, dy, ha))
    labels2draw.sort(key=lambda L: L[0])   # stable order along fall line
    arrow_kw = dict(arrowstyle='-', lw=0.6, alpha=0.7,
                    shrinkA=0, shrinkB=0)
    for lon, lat, txt, col in labels2draw:
        for dx, dy, ha in candidates:
            # measure with a probe first — get_window_extent on an
            # annotate with arrowprops includes the arrow and would
            # always overlap its own marker reservation
            probe = ax.annotate(txt, xy=(lon, lat), xytext=(dx, dy),
                                textcoords='offset points', ha=ha,
                                fontsize=6.5, color='black',
                                transform=pc if pc else ax.transData)
            bb = probe.get_window_extent(renderer).expanded(1.05, 1.2)
            probe.remove()
            if any(bb.overlaps(b) for b in placed_boxes):
                continue
            ax.annotate(txt, xy=(lon, lat), xytext=(dx, dy),
                        textcoords='offset points', ha=ha,
                        fontsize=6.5, color='black',
                        arrowprops=dict(arrow_kw, color='black'),
                        transform=pc if pc else ax.transData)
            placed_boxes.append(bb)
            break
    if mc_impacts:
        kw = {'transform': pc} if pc else {}
        for impact in mc_impacts:
            trajectory = impact.get('trajectory') or {}
            if len(trajectory.get('lon', ())) >= 2:
                ax.plot(trajectory['lon'], trajectory['lat'], color='red',
                        linewidth=0.35, alpha=0.12, **kw)
        ring = _mc_ring_ll(mc_impacts)
        if ring is not None:
            # light fill + hatching so the terrain stays readable
            ax.fill(ring[:, 0], ring[:, 1], facecolor='red', alpha=0.08,
                    edgecolor='none', **kw)
            ax.fill(ring[:, 0], ring[:, 1], facecolor='none',
                    edgecolor='red', linewidth=0.0, hatch='///',
                    **kw)
            ax.plot(ring[:, 0], ring[:, 1], color='red',
                    linewidth=1.2, **kw)
        ax.scatter([i['lon'] for i in mc_impacts],
                   [i['lat'] for i in mc_impacts], s=2, c='red',
                   alpha=0.25, **kw)
    # scale bar: ~1/4 of map width, rounded to a nice value
    span_m = (lon_max - lon_min) * 111320 * math.cos(
        math.radians((lat_min + lat_max) / 2))
    candidates = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    bar_m = min(candidates, key=lambda c: abs(c - span_m / 4))
    if bar_m >= 1000:
        bar_txt = f'{bar_m / 1000:.4g} km'
        if translations.get('dec_sep', '.') != '.':
            bar_txt = bar_txt.replace('.', ',')
    else:
        bar_txt = f'{bar_m} m'
    # draw in axes fraction so the bar stays horizontal under any
    # projection (a constant-latitude line is slanted in UTM)
    bx = 0.05
    by = 0.07
    bw = bar_m / span_m
    ax.plot([bx, bx + bw], [by, by], lw=3, color='black',
            solid_capstyle='butt', transform=ax.transAxes,
            clip_on=False)
    for _bx in (bx, bx + bw):
        ax.plot([_bx, _bx], [by - 0.008, by + 0.008], lw=2,
                color='black', transform=ax.transAxes, clip_on=False)
    ax.text(bx + bw / 2, by + 0.012, bar_txt, ha='center', va='bottom',
            fontsize=8, color='black', transform=ax.transAxes)
    from matplotlib.lines import Line2D
    handles, labels = [], []
    if luminous_track is not None and len(luminous_track[0]):
        handles.append(Line2D([], [], color='red', lw=1.8))
        labels.append(translations.get('df_luminous_path',
                                       'Luminous trajectory'))
    handles.append(Line2D([], [], marker='*', ls='', color='red', ms=12))
    labels.append(translations.get('df_end_luminous', 'End of luminous path'))
    if mc_impacts:
        handles.append(Line2D([], [], marker='.', ls='', color='red',
                              ms=7))
        labels.append(translations.get(
            'df_mc_assumed' if mc_assumed else 'df_mc',
            'Monte Carlo (assumed masses)' if mc_assumed else 'Monte Carlo'))
    if disintegration_clouds:
        handles.append(Line2D([], [], color='chocolate', lw=1.6))
        labels.append(translations.get('df_frag_disintegration',
                                       'Disintegration (illustrative)'))
    ax.legend(handles, labels, fontsize=7, loc='upper center',
              bbox_to_anchor=(0.5, -0.04), ncol=2, framealpha=0.9)
    if not pc:
        ax.set_xlabel(translations.get('longitude', 'Longitude'))
        ax.set_ylabel(translations.get('latitude', 'Latitude'))
    if title:
        ax.set_title(title)
    plt.savefig(out_svg, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def write_map3d(scenarios, end_llh, mc_impacts, out_html,
                translations=None, mc_assumed=False, start_llh=None,
                wind_csv=None, observed_duration_s=None):
    """Interactive 3D fall-area map (plotly), similar to metrack map.html.

    Ground tile image is rendered as a textured z=0 surface; scenario
    trajectories are 3D lines ending at ground-impact markers; the
    luminous-path end point is starred; Monte-Carlo impacts are a
    ground-level scatter cloud.
    """
    translations = translations or {}
    try:
        import plotly.graph_objects as go
        import cartopy.crs as ccrs
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        import io, html as html_mod
        from metrack import (darken_blacks, _fetch_kartverket_topo,
                             _rotation_controls_html, _wind_overlay_html)
    except Exception as e:
        logging.debug(f'darkflight 3d map unavailable: {e}')
        return False

    lons_all, lats_all = [end_llh[0]], [end_llh[1]]
    luminous_track = None
    if start_llh is not None:
        luminous_track = _clip_track_max_h(*_straight_track_llh(start_llh, end_llh))
        if len(luminous_track[0]):
            lons_all.append(luminous_track[0][0])
            lats_all.append(luminous_track[1][0])
    for sc in scenarios:
        for run in sc['results']:
            track = _clip_track_max_h(run['lon'], run['lat'], run['h'])
            lons_all += list(track[0]) + [run['impact']['lon']]
            lats_all += list(track[1]) + [run['impact']['lat']]
    for imp in (mc_impacts or []):
        lons_all.append(imp['lon']); lats_all.append(imp['lat'])
    pad = 0.15
    lon_min, lon_max = min(lons_all) - pad, max(lons_all) + pad
    lat_min, lat_max = min(lats_all) - pad, max(lats_all) + pad
    centre_lat = (lat_min + lat_max) / 2.0
    min_half_deg_lat = 20.0 / 111.32
    min_half_deg_lon = 20.0 / max(
        111.32 * math.cos(math.radians(centre_lat)), 1e-6)
    lon_min = min(lon_min, end_llh[0] - min_half_deg_lon)
    lon_max = max(lon_max, end_llh[0] + min_half_deg_lon)
    lat_min = min(lat_min, end_llh[1] - min_half_deg_lat)
    lat_max = max(lat_max, end_llh[1] + min_half_deg_lat)

    proj = ccrs.Gnomonic(central_longitude=(lon_min + lon_max) / 2,
                         central_latitude=(lat_min + lat_max) / 2)
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw={'projection': proj})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                  crs=ccrs.PlateCarree())

    # Ground image: Kartverket (if POIs covered) else OSM
    poi = [(end_llh[0], end_llh[1])]
    for sc in scenarios:
        for run in sc['results']:
            imp = run['impact']
            if imp.get('landed', True):
                poi.append((imp['lon'], imp['lat']))
    kv = None
    try:
        kv = _fetch_kartverket_topo([lon_min, lon_max], [lat_min, lat_max],
                                    poi_lonlat=poi)
    except Exception:
        pass
    if kv is not None:
        img_kv, (kx0, kx1, ky0, ky1) = kv
        ax.imshow(img_kv, extent=[kx0, kx1, ky0, ky1],
                  transform=ccrs.UTM(32), origin='upper', zorder=0)
    else:
        lat_span = lat_max - lat_min
        zoom = int(math.ceil(math.log2(1440 / max(lat_span, 0.05))))
        zoom = max(6, min(zoom, 13))
        try:
            from cartopy.io.img_tiles import OSM
            ax.add_image(OSM(), zoom)
        except Exception:
            import cartopy.feature as cfeature
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.OCEAN)
    try:
        fig.canvas.draw()
    except Exception:
        pass
    x_min_m, x_max_m, y_min_m, y_max_m = ax.get_extent()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                pad_inches=0, transparent=False)
    plt.close(fig)
    buf.seek(0)

    def project_points(lons, lats):
        pts = proj.transform_points(ccrs.PlateCarree(),
                                    np.asarray(lons, float),
                                    np.asarray(lats, float))
        return pts[:, 0] / 1000.0, pts[:, 1] / 1000.0

    # 640px texture is plenty for a scene panel; 1024 -> ~2.5x smaller JSON
    img = Image.open(buf).convert('RGB')
    if max(img.size) > 640:
        img.thumbnail((640, 640), Image.Resampling.LANCZOS)
    img = darken_blacks(img, 112)
    quant = img.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
    pal = quant.getpalette()
    pal_rgb = [tuple(pal[i:i + 3]) for i in range(0, len(pal), 3)]
    lum = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in pal_rgb]
    order = sorted(range(len(pal_rgb)), key=lambda i: lum[i])
    sorted_pal = [pal_rgb[i] for i in order]
    old2new = {o: n for n, o in enumerate(order)}
    idx = np.array(quant)
    lut = np.zeros(256, dtype=np.uint8)
    for o, n in old2new.items():
        lut[o] = n
    remapped = np.flipud(lut[idx])
    cscale = [[i / max(1, len(sorted_pal) - 1),
               f'rgb({r},{g},{b})']
              for i, (r, g, b) in enumerate(sorted_pal)]
    hgt, wid = remapped.shape

    traces = [go.Surface(
        x=np.linspace(x_min_m / 1000.0, x_max_m / 1000.0, wid),
        y=np.linspace(y_min_m / 1000.0, y_max_m / 1000.0, hgt),
        z=np.zeros((hgt, wid)),
        surfacecolor=remapped, cmin=0, cmax=max(1, len(sorted_pal) - 1),
        colorscale=cscale, showscale=False, hoverinfo='none')]

    # scale bar on the ground plane (x = east, km)
    span_km = (x_max_m - x_min_m) / 1000.0
    _cands = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    bar_km = min(_cands, key=lambda c: abs(c - span_km / 4))
    if bar_km < 1:
        bar_txt = f'{bar_km * 1000:.4g} m'
    else:
        bar_txt = f'{bar_km:.4g} km'
    if translations.get('dec_sep', '.') != '.':
        bar_txt = bar_txt.replace('.', ',')
    bx0 = x_min_m / 1000.0 + 0.05 * span_km
    by0 = y_min_m / 1000.0 + 0.05 * (y_max_m - y_min_m) / 1000.0
    traces.append(go.Scatter3d(
        x=[bx0, bx0 + bar_km], y=[by0, by0], z=[0, 0],
        mode='lines', line=dict(color='black', width=4),
        showlegend=False, hoverinfo='none'))
    traces.append(go.Scatter3d(
        x=[bx0 + bar_km / 2], y=[by0], z=[0], mode='text',
        text=[bar_txt], textfont=dict(size=10, color='black'),
        showlegend=False, hoverinfo='none'))

    if luminous_track is not None and len(luminous_track[0]):
        lon_l, lat_l, h_l = luminous_track
        x_l, y_l = project_points(lon_l, lat_l)
        traces.append(go.Scatter3d(
            x=x_l, y=y_l, z=h_l / 1000.0, mode='lines',
            line=dict(color='red', width=5), opacity=0.9,
            name=translations.get('df_luminous_path', 'Luminous trajectory'),
            hoverinfo='name'))

    if start_llh is not None:
        disintegration_legend = True
        for sc in scenarios:
            burst = _disintegration_burst_points(sc, start_llh, end_llh)
            if burst is None:
                continue
            origins, endpoints = burst
            ox, oy = project_points(origins[:, 0], origins[:, 1])
            ex, ey = project_points(endpoints[:, 0], endpoints[:, 1])
            origin_xy = np.column_stack([ox, oy])
            endpoint_xy = np.column_stack([ex, ey])
            midpoint_xy = origin_xy + 0.58 * (endpoint_xy - origin_xy)
            oz = origins[:, 2] / 1000.0
            ez = endpoints[:, 2] / 1000.0
            mz = oz + 0.58 * (ez - oz)
            traces.append(go.Scatter3d(
                x=np.column_stack([origin_xy[:, 0], midpoint_xy[:, 0],
                                   np.full(len(origins), np.nan)]).ravel(),
                y=np.column_stack([origin_xy[:, 1], midpoint_xy[:, 1],
                                   np.full(len(origins), np.nan)]).ravel(),
                z=np.column_stack([oz, mz,
                                   np.full(len(origins), np.nan)]).ravel(),
                mode='lines', line=dict(color='rgba(210,105,30,0.75)',
                                        width=3),
                name=translations.get('df_frag_disintegration',
                                      'Disintegration (illustrative)'),
                legendgroup='disintegration',
                showlegend=disintegration_legend,
                hoverinfo='skip'))
            disintegration_legend = False
            traces.append(go.Scatter3d(
                x=np.column_stack([midpoint_xy[:, 0], endpoint_xy[:, 0],
                                   np.full(len(origins), np.nan)]).ravel(),
                y=np.column_stack([midpoint_xy[:, 1], endpoint_xy[:, 1],
                                   np.full(len(origins), np.nan)]).ravel(),
                z=np.column_stack([mz, ez,
                                   np.full(len(origins), np.nan)]).ravel(),
                mode='lines', line=dict(color='rgba(210,105,30,0.22)',
                                        width=1.5),
                legendgroup='disintegration',
                showlegend=False, hoverinfo='skip'))
            traces.append(go.Scatter3d(
                x=endpoint_xy[:, 0], y=endpoint_xy[:, 1], z=ez,
                mode='markers', marker=dict(size=2.0, color='chocolate',
                                            opacity=0.18),
                name=translations.get('df_frag_disintegration',
                                      'Disintegration (illustrative)'),
                legendgroup='disintegration',
                showlegend=False, hoverinfo='name'))
            ex0, ey0 = project_points([end_llh[0]], [end_llh[1]])
            traces.append(go.Scatter3d(
                x=[ex0[0]], y=[ey0[0]], z=[end_llh[2] / 1000.0],
                mode='markers', marker=dict(size=11, color='chocolate',
                                            symbol='diamond'),
                name=translations.get('df_frag_disintegration',
                                      'Disintegration (illustrative)'),
                legendgroup='disintegration',
                showlegend=False, hoverinfo='name'))

    plot_sc = [s for s in scenarios
               if s.get('erode', True) and s['name'] != 'S4_powerlaw']
    label_flip = [0]
    cmap = matplotlib.pyplot.get_cmap('viridis')
    for si, sc in enumerate(plot_sc):
        c = cmap(si / max(len(plot_sc) - 1, 1))
        col = f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
        uniq, seen_m = [], []
        for run in sc['results']:
            # merge masses within ±10% — no point labelling near-identical
            # impact points separately
            if any(abs(run['m'] - m) <= 0.1 * m for m in seen_m):
                continue
            seen_m.append(run['m'])
            uniq.append(run)
        for ri, run in enumerate(uniq):
            # decimate the polyline — full res has ~1800 pts/trace
            lon_d, lat_d, h_d = _clip_track_max_h(run['lon'], run['lat'], run['h'])
            if len(lon_d):
                step = max(1, len(lon_d) // 300)
                lon_d = lon_d[::step]; lat_d = lat_d[::step]; h_d = h_d[::step]
                x, y = project_points(lon_d, lat_d)
                traces.append(go.Scatter3d(
                    x=x, y=y, z=h_d / 1000.0,
                    mode='lines',
                    line=dict(color=col, width=3),
                    name=sc_label(sc, translations),
                    legendgroup=sc['name'],
                    showlegend=(ri == 0 and sc['name'] != 'S1_fallline'),
                    hoverinfo='name', opacity=0.85))
            imp = run['impact']
            if imp.get('landed', True):
                ix, iy = project_points([imp['lon']], [imp['lat']])
                m_kg = run['impact']['m']
                m_txt = (f'{m_kg:.3g} kg' if m_kg >= 1
                         else f'{m_kg * 1000:.3g} g')
                dec = translations.get('dec_sep', '.')
                if dec != '.':
                    m_txt = m_txt.replace('.', dec)
                traces.append(go.Scatter3d(
                    x=ix, y=iy, z=[0], mode='markers+text',
                    marker=dict(size=5, color=col, symbol='circle'),
                    text=[m_txt],
                    textposition=('top center' if label_flip[0] == 0
                                  else 'bottom center'),
                    textfont=dict(size=9, color='black'),
                    legendgroup=sc['name'], showlegend=False,
                    hoverinfo='none'))
                label_flip[0] ^= 1
    # end of luminous path
    ex, ey = project_points([end_llh[0]], [end_llh[1]])
    traces.append(go.Scatter3d(
        x=ex, y=ey, z=[end_llh[2] / 1000.0], mode='markers',
        marker=dict(size=9, color='red', symbol='diamond'),
        name=translations.get('df_end_luminous', 'End of luminous path')))
    # MC impacts — lifted slightly above the ground plane so they
    # don't z-fight with the textured surface
    if mc_impacts:
        for impact in mc_impacts:
            trajectory = impact.get('trajectory') or {}
            lon_mc, lat_mc, h_mc = _clip_track_max_h(
                trajectory.get('lon', []), trajectory.get('lat', []),
                trajectory.get('h', []))
            if len(lon_mc) < 2:
                continue
            mx_path, my_path = project_points(lon_mc, lat_mc)
            traces.append(go.Scatter3d(
                x=mx_path, y=my_path, z=h_mc / 1000.0,
                mode='lines', line=dict(color='red', width=1),
                opacity=0.12, showlegend=False, hoverinfo='skip'))
        mx, my = project_points([i['lon'] for i in mc_impacts],
                                [i['lat'] for i in mc_impacts])
        traces.append(go.Scatter3d(
            x=mx, y=my, z=[0.02] * len(mx), mode='markers',
            marker=dict(size=3, color='red', opacity=0.35),
            name=translations.get(
                'df_mc_assumed' if mc_assumed else 'df_mc',
                'Monte Carlo (assumed masses)' if mc_assumed
                else 'Monte Carlo'),
            hoverinfo='skip'))
        # shaded buffered-hull ring around the MC cloud — translucent
        # fill + opaque edge on the ground plane
        ring = _mc_ring_ll(mc_impacts)
        if ring is not None:
            sx, sy = project_points(ring[:, 0], ring[:, 1])
            vxs, vys = [], []
            tri_i, tri_j, tri_k = [], [], []
            try:
                # Delaunay-triangulate the (possibly concave) outline
                # and keep only triangles inside it — a centroid fan
                # would paint over the concave side of a curved band
                from shapely.geometry import Polygon as _Poly
                from shapely.ops import triangulate as _triangulate
                poly_xy = _Poly(list(zip(sx, sy)))
                vmap = {}
                for tri in _triangulate(poly_xy):
                    if not poly_xy.covers(tri):
                        continue
                    idx = []
                    for x_, y_ in list(tri.exterior.coords)[:3]:
                        key = (x_, y_)
                        if key not in vmap:
                            vmap[key] = len(vxs)
                            vxs.append(x_)
                            vys.append(y_)
                        idx.append(vmap[key])
                    tri_i.append(idx[0])
                    tri_j.append(idx[1])
                    tri_k.append(idx[2])
            except Exception:
                pass
            if not tri_i:
                cx, cy = np.mean(sx[:-1]), np.mean(sy[:-1])
                vxs = [cx] + list(sx[:-1])
                vys = [cy] + list(sy[:-1])
                for k in range(len(sx) - 1):
                    tri_i.append(0)
                    tri_j.append(k + 1)
                    tri_k.append(1 + (k + 1) % (len(sx) - 1))
            traces.append(go.Mesh3d(
                x=vxs, y=vys, z=[0.02] * len(vxs),
                i=tri_i, j=tri_j, k=tri_k,
                color='red', opacity=0.15,
                name=translations.get(
                    'df_fall_area_assumed' if mc_assumed else 'df_fall_area',
                    'Modelled fall area (MC)' if mc_assumed
                    else 'Fall area (MC)'),
                showscale=False, hoverinfo='skip'))
            traces.append(go.Scatter3d(
                x=list(sx), y=list(sy), z=[0.02] * len(sx),
                mode='lines', line=dict(color='red', width=3),
                name=translations.get('df_fall_area_edge', 'Fall area edge'),
                showlegend=False,
                hoverinfo='skip'))
            # hatch lines at 45°, clipped to the polygon
            try:
                from shapely.geometry import (Polygon, LineString,
                                              MultiLineString)
                poly = Polygon(list(zip(sx, sy)))
                x0, y0, x1, y1 = poly.bounds
                diag = math.hypot(x1 - x0, y1 - y0)
                spacing = max(diag / 15.0, 0.05)
                segs, hx_ll, hy_ll = [], [], []
                t = -diag
                while t < diag:
                    # line at 45 deg offset t along the normal (-1,1)/sqrt2
                    cxx, cyy = (x0 + x1) / 2 - t / math.sqrt(2), \
                               (y0 + y1) / 2 + t / math.sqrt(2)
                    ex1 = (cxx - diag / math.sqrt(2),
                           cyy - diag / math.sqrt(2))
                    ex2 = (cxx + diag / math.sqrt(2),
                           cyy + diag / math.sqrt(2))
                    cl = poly.intersection(LineString([ex1, ex2]))
                    if isinstance(cl, LineString):
                        cc = list(cl.coords)
                        segs += [(cc[i], cc[i + 1])
                                 for i in range(len(cc) - 1)]
                    elif isinstance(cl, MultiLineString):
                        for ls in cl.geoms:
                            cc = list(ls.coords)
                            segs += [(cc[i], cc[i + 1])
                                     for i in range(len(cc) - 1)]
                    t += spacing
                for a, b in segs:
                    hx_ll += [a[0], b[0], None]
                    hy_ll += [a[1], b[1], None]
                traces.append(go.Scatter3d(
                    x=hx_ll, y=hy_ll, z=[0.02] * len(hx_ll),
                    mode='lines',
                    line=dict(color='red', width=1.5),
                    name=translations.get('df_mc_hatch', 'MC hatch'),
                    showlegend=False,
                    hoverinfo='skip'))
            except Exception:
                pass

    scene_dx = (x_max_m - x_min_m) / 1000.0
    scene_dy = (y_max_m - y_min_m) / 1000.0
    half = max(scene_dx, scene_dy) / 2.0 or 1.0
    cx = (x_min_m + x_max_m) / 2000.0
    cy = (y_min_m + y_max_m) / 2000.0
    # default rotation centre: above the Monte Carlo impact cloud when
    # it exists, else above the end of the luminous path
    if mc_impacts:
        fx, fy = project_points(
            [float(np.mean([i['lon'] for i in mc_impacts]))],
            [float(np.mean([i['lat'] for i in mc_impacts]))])
        cx0, cy0 = fx[0], fy[0]
        z0 = 0.15          # hover above the ground plane
    else:
        cx0, cy0, z0 = ex[0], ey[0], 0.0
    center = dict(x=float((cx0 - cx) / half),
                  y=float((cy0 - cy) / half), z=z0)
    dist = 1.3
    elev = math.radians(35)
    eye = dict(x=center['x'] + dist, y=center['y'],
               z=center['z'] + dist * math.tan(elev))

    fig3 = go.Figure(data=traces, layout=go.Layout(
        title=translations.get('dark_flight', 'Dark flight'),
        title_x=0.5, title_y=0.95, showlegend=True,
        legend=dict(font=dict(size=10), x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.6)'),
        scene=dict(
            xaxis=dict(title=translations.get('plot_map_interactive_xaxis',
                                              'East/West Distance (km)'),
                       range=[x_min_m / 1000.0, x_max_m / 1000.0]),
            yaxis=dict(title=translations.get('plot_map_interactive_yaxis',
                                              'North/South Distance (km)'),
                       range=[y_min_m / 1000.0, y_max_m / 1000.0]),
            zaxis=dict(title=translations.get('plot_map_interactive_zaxis',
                                              'Height (km)'),
                       range=[0.0, 40.0]),
            aspectmode='data', dragmode='turntable',
            camera=dict(up=dict(x=0, y=0, z=1), center=center, eye=eye)),
        margin=dict(l=0, r=0, b=0, t=40)))
    out_html = Path(out_html)
    plotly_locale = translations.get('plotly_locale')
    plotly_config = ({'locale': plotly_locale}
                     if plotly_locale and plotly_locale != 'en' else {})
    fig3.write_html(str(out_html), include_plotlyjs='cdn',
                    config=plotly_config)
    html_txt = out_html.read_text(encoding='utf-8')
    html_txt = html_txt.replace(
        '<head>', '<head><style>html,body{margin:0;padding:0;'
        'overflow:hidden;height:100%}</style>', 1)
    # Rotation controls — same animation as metrack's map.html
    rot = _rotation_controls_html(
        center, dist, eye['z'] - center['z'],
        translations.get('plot_interactive_play'),
        translations.get('plot_interactive_pause'))
    if '</body>' in html_txt:
        html_txt = html_txt.replace('</body>', rot + '\n</body>', 1)
    else:
        html_txt += rot
    # Wind particles — same overlay as metrack's map.html. Box covers the
    # fall area (projected km coords).
    try:
        wind_csv = (Path(wind_csv) if wind_csv is not None
                    else out_html.parent / 'wind_profile.csv')
        box = [x_min_m / 1000.0, x_max_m / 1000.0,
               y_min_m / 1000.0, y_max_m / 1000.0]
        wjs = _wind_overlay_html(
            wind_csv, box, translations.get('plot_interactive_wind'))
        if wjs:
            if '</body>' in html_txt:
                html_txt = html_txt.replace('</body>', wjs + '\n</body>', 1)
            else:
                html_txt += wjs
    except Exception as e:
        logging.debug(f'darkflight wind overlay failed: {e}')
    out_html.write_text(html_txt, encoding='utf-8')
    return True


# --- Orchestration --------------------------------------------------------------
def load_language_files():
    lang_files = {'': {'dec_sep': '.'}}
    loc_dir = Path(__file__).resolve().parent.parent / 'server' / 'loc'
    default = {}
    default_path = loc_dir / 'nb.json'
    if default_path.exists():
        default = json.loads(default_path.read_text(encoding='utf-8'))
        lang_files[''] = {**default, 'dec_sep': ',', 'plotly_locale': 'no'}
    for lang in ('en', 'cs', 'de', 'fi', 'lv'):
        path = loc_dir / f'{lang}.json'
        if path.exists():
            translations = {**default, **json.loads(path.read_text(encoding='utf-8'))}
            translations['dec_sep'] = '.' if lang == 'en' else ','
            translations['plotly_locale'] = lang
            lang_files[f'{lang}_'] = translations
    return lang_files


def _clean_outputs(event_dir):
    """Remove generated dark-flight artifacts (maps, tables, exports)."""
    for pat in ('map_darkflight.*', '*_map_darkflight.*',
                'darkflight_map3d.html', '*_darkflight_map3d.html',
                'darkflight.geojson', 'darkflight.kml',
                'fragmentation.json', 'fragmentation_summary.csv',
                'fragmentation.geojson',
                'darkflight_table.html', '*_darkflight_table.html'):
        for p in event_dir.glob(pat):
            try:
                p.unlink()
            except Exception:
                pass


def _mag_to_photometric_mass(mag, v_ms, duration_s=1.0):
    """Mass-loss proxy [kg] for constant magnitude/speed over duration_s.

    Uses 1500 W for magnitude zero at 100 km and integrates
    2*L/(tau*v**2) over time, neglecting deceleration luminosity.
    Duration does not cancel; this is not the surviving mass.
    """
    if not all(np.isfinite(v) for v in (mag, v_ms, duration_s)) or v_ms <= 0 or duration_s <= 0:
        raise ValueError('Magnitude must be finite; speed and duration must be positive')
    L = 1500.0 * 10 ** (-0.4 * mag)
    tau = min(0.20, max(0.001, 0.0007 * (v_ms / 1000.0)))
    return 2 * L * duration_s / (tau * v_ms ** 2)


def run_darkflight(event_dir, resdat, fbspd_results=None, fbspd_plot_data=None,
                   wind_csv=None, mc_runs=300, seed=0, pool=None,
                   verbose=False, lang_files=None, peak_mag=None,
                   rho_ref=None):
    """Top-level: compute dark flight for an event directory.

    resdat: ResData (from fbspd_merge.readres) with track start/end.
    fbspd_plot_data: needs 'final_params' + merged reltime for end speed.
    Returns results dict (also written to darkflight.json).
    """
    event_dir = Path(event_dir)
    if rho_ref is not None and (not np.isfinite(rho_ref) or rho_ref <= 0):
        raise ValueError('Reference density must be positive and finite')
    if peak_mag is not None and (not np.isfinite(peak_mag) or peak_mag >= -1.0):
        hint = (f' — did you mean {-abs(peak_mag):.1f}?' if peak_mag > 0 else '')
        raise ValueError(f'Peak magnitude must be finite and brighter '
                         f'than -1 for a surviving-mass estimate '
                         f'(got {peak_mag}){hint}')
    if not isinstance(mc_runs, (int, np.integer)) or mc_runs < 0:
        raise ValueError('Monte Carlo run count must be a non-negative integer')
    end_lon = float(resdat.long1[1]); end_lat = float(resdat.lat1[1])
    end_h = float(resdat.height[1]) * 1000.0   # km -> m
    start_lon = float(resdat.long1[0]); start_lat = float(resdat.lat1[0])
    start_h = float(resdat.height[0]) * 1000.0
    if (not all(np.isfinite(v) for v in (start_lon, start_lat, start_h, end_lon, end_lat, end_h))
            or not all(-90 <= v <= 90 for v in (start_lat, end_lat))
            or not all(-180 <= v <= 180 for v in (start_lon, end_lon))):
        raise ValueError('Trajectory coordinates must be finite valid geodetic coordinates')

    # Track direction at end point (straight-line assumption)
    r_start = llh2ecef(start_lon, start_lat, start_h)
    r_end = llh2ecef(end_lon, end_lat, end_h)
    track_dir = r_end - r_start
    if np.linalg.norm(track_dir) < 1.0:
        raise ValueError('Trajectory endpoints must be distinct')
    track_dir /= np.linalg.norm(track_dir)

    # End speed & deceleration from fbspd fit
    have_fit = (fbspd_plot_data is not None
                and fbspd_plot_data.get('final_params') is not None)
    pcov = None
    release_truncated = False
    if have_fit:
        from fbspd_merge import expfunc, expfunc_1stder, expfunc_2ndder
        params = fbspd_plot_data['final_params']
        pcov = fbspd_plot_data.get('pcov')
        ts_obs = np.asarray(fbspd_plot_data['final_merged_data']['reltime'], dtype=float)
        hs_obs = np.asarray(fbspd_plot_data['final_merged_data']['height'], dtype=float)
        params = np.asarray(params, dtype=float)
        if (params.shape != (4,) or not np.all(np.isfinite(params))
                or ts_obs.ndim != 1 or hs_obs.shape != ts_obs.shape or len(ts_obs) < 2
                or not np.all(np.isfinite(ts_obs)) or not np.all(np.isfinite(hs_obs))
                or np.ptp(ts_obs) <= 0):
            raise ValueError('Invalid luminous-track fit samples')
        t0_obs = float(np.min(ts_obs)); t_last = float(np.max(ts_obs))
        n_obs = len(ts_obs)

        def v_of_t(t):
            return float(expfunc_1stder(t, *params)) * 1000.0   # km/s -> m/s

        def a_of_t(t):
            return -float(expfunc_2ndder(t, *params)) * 1000.0

        hfit = np.polyfit(ts_obs, hs_obs, 1) if len(ts_obs) > 2 else None

        def h_of_t(t):
            if hfit is not None:
                return float(np.polyval(hfit, t)) * 1000.0
            return end_h

        v_end = v_of_t(t_last)
        a_end = a_of_t(t_last)

        # Exponential speed fits extrapolate through zero for meteors
        # that decelerate out of the luminous regime before the last
        # centroid. The luminous track effectively ends where the fit
        # crosses ~3 km/s — truncate the analysis range there (otherwise
        # negative v corrupts the ablation integral and mass inversion)
        # and release the fragment at the luminous-fade speed.
        V_LUM_MIN = 3000.0
        if v_end < V_LUM_MIN:
            ts_scan = np.linspace(t0_obs, t_last, 2000)
            v_scan = np.array([v_of_t(t) for t in ts_scan])
            below = np.nonzero(v_scan < V_LUM_MIN)[0]
            if len(below) and below[0] > 0:
                from scipy.optimize import brentq
                t_fade = float(brentq(lambda t: v_of_t(t) - V_LUM_MIN,
                                     ts_scan[below[0] - 1], ts_scan[below[0]]))
                pos0, pos1 = expfunc(t0_obs, *params), expfunc(t_last, *params)
                if pos1 <= pos0:
                    raise ValueError('Non-increasing fitted trajectory distance')
                fraction = float(np.clip((expfunc(t_fade, *params) - pos0) / (pos1 - pos0), 0., 1.))
                r_end = r_start + fraction * (r_end - r_start)
                end_lon, end_lat, end_h = ecef2llh(r_end)
                release_truncated = True
                logging.info(f'darkflight: fit crosses luminous threshold '
                             f'at t={t_fade:.2f}s (of {t_last:.2f}s) — '
                             f'truncating analysis and releasing at 3 km/s')
                t_last = t_fade
                a_end = a_of_t(t_fade)
                v_end = V_LUM_MIN
            else:
                logging.warning(f'darkflight: fitted v_end={v_end/1000:.2f} km/s '
                                f'below luminous threshold — retaining fitted speed')
    else:
        v_end, a_end, t0_obs, t_last, n_obs = 3000.0, 1e4, 0.0, 1.0, 0
        logging.warning('darkflight: no fbspd fit, using nominal v_end/a_end')

    # --- Input sanity: reject physically impossible fits -------------------------
    fit_valid = (np.isfinite(v_end) and np.isfinite(a_end)
                 and 100.0 < v_end < 72000.0        # luminous regime ~3–72 km/s
                 and 0.0 <= end_h <= 120000.0
                 and 0.0 <= start_h <= 150000.0
                 and a_end >= 0.0
                 and t_last > t0_obs
                 and t_last - t0_obs < 60.0)
    if not fit_valid:
        issues_pre = [f'invalid end state (v={v_end/1000:.2f} km/s, '
                      f'h={end_h/1000:.1f} km, dur={t_last-t0_obs:.1f} s)']
        logging.warning(f'darkflight: {issues_pre[0]} — skipping')
        results = {
            'event_dir': str(event_dir),
            'end_state': {'lon': end_lon, 'lat': end_lat, 'h_m': end_h,
                          'v_end_ms': float(v_end), 'a_end_ms2': float(a_end)},
            'wind_used': False, 'h_ground_m': 0.0,
            'mass_estimates': [], 'entry_estimates': [],
            'm_est_ref_kg': None, 'mass_source': None, 'mass_status': 'unconstrained',
            'density_ref_kg_m3': float(rho_ref or 3500.0),
            'photometric': None, 'mag_anchor': None, 'phot_anchor': None,
            'magnitude_input': None, 'model_assumptions': [],
            'reliability': 'unreliable', 'issues': issues_pre,
            'n_obs': int(n_obs if have_fit else 0),
            'track_duration_s': float(t_last - t0_obs),
            'scenarios': [], 'mc': {'n': 0, 'impacts': [], 'attempted': 0,
                                   'requested': mc_runs, 'skipped_reason': 'invalid_state'},
        }
        write_json(results, event_dir / 'darkflight.json')
        _clean_outputs(event_dir)
        return results

    # event time from the directory name .../YYYYMMDD/HHMMSS
    try:
        from datetime import datetime, timezone
        event_dt = datetime.strptime(
            f'{event_dir.parent.name} {event_dir.name}',
            '%Y%m%d %H%M%S').replace(tzinfo=timezone.utc)
    except Exception:
        event_dt = None
    atm = WindAtmosphere(wind_csv, lon=end_lon, lat=end_lat,
                         event_dt=event_dt)

    # Surviving-mass estimates over the density grid (whole-track fit)
    mass_estimates = []
    if have_fit:
        rho_grid = DENSITY_GRID
        if rho_ref is not None and rho_ref not in rho_grid:
            rho_grid = sorted(rho_grid + [float(rho_ref)])
        mass_estimates = estimate_mass(v_of_t, a_of_t, h_of_t,
                                       (t0_obs, t_last), rho_grid=rho_grid,
                                       atm=atm, params=params, pcov=pcov)
    ref = next((e for e in mass_estimates if e['rho'] == (rho_ref or 3500)),
               None)

    # --- Reliability assessment -------------------------------------------------
    # A fast luminous endpoint alone does not establish complete ablation:
    # fragmentation, sensitivity, cloud or field-of-view loss may explain
    # the observed disappearance. Treat a deceleration-inferred mass far
    # above the ablation scale as a model-consistency warning, not proof
    # of disintegration or of a bad observation. The following thresholds
    # are screening heuristics rather than survival-probability estimates.
    issues = []
    if release_truncated:
        issues.append('release moved to model 3 km/s threshold')
    if have_fit and v_end < 3000.0:
        issues.append('speed fit entirely below luminous threshold')
    duration = t_last - t0_obs
    if n_obs < 10:
        issues.append(f'few centroid observations ({n_obs})')
    if duration < 0.8:
        issues.append(f'short luminous track ({duration:.2f} s)')
    if ref is not None:
        m_med, m_lo, m_hi, m_crit = (ref['m_fade_kg'], ref['m_fade_lo'],
                                   ref['m_fade_hi'], ref['m_crit_kg'])
        if not np.isfinite(m_med):
            # zero/negative fitted deceleration makes the drag inversion
            # diverge — the mass is not determinable at all
            issues.append('deceleration non-invertible')
            m_med = 0.0
        else:
            if not (np.isfinite(m_hi) and m_hi > 0) or                     m_hi / max(m_lo, 1e-12) > 20:
                issues.append('deceleration poorly constrained')
            if v_end > 4000.0 and m_med > 30 * max(m_crit, 1e-9):
                issues.append('inconsistent with fade-out')
            if m_med > 100.0:
                issues.append('implausibly large surviving mass')
    else:
        m_med = m_crit = 0.0
        issues.append('no speed/deceleration fit')
    reliability = 'unreliable' if any(
        i in ('inconsistent with fade-out', 'implausibly large surviving mass',
              'no speed/deceleration fit', 'speed fit entirely below luminous threshold',
              'deceleration non-invertible') for i in issues) else \
        ('marginal' if issues else 'ok')
    if reliability != 'ok':
        logging.warning(f'darkflight: mass estimate {reliability} '
                        f'({"; ".join(issues)})')

    # 'deceleration non-invertible' no longer aborts the simulation:
    # a constant-velocity fit yields no mass estimate, but the event may
    # still end low enough to drop meteorites (e.g. a linear fit was
    # chosen over noisy data). The fall-line/mass-grid scenarios carry
    # conditional paths; an unknown mass is not replaced by a nominal body.

    # --- Photometric cross-check (star-calibrated light curve) ------------------
    photometry = None
    try:
        import photometry as ph
        photometry = ph.event_photometry(event_dir, resdat=resdat,
                                         plot_data=fbspd_plot_data)
    except Exception as e:
        logging.debug(f'darkflight: photometry failed: {e}')

    magnitude_input = None
    mag_anchor = None
    # characteristic speed for the survival fraction: the fitted start
    # speed sets the total ablation budget; fall back to end speed
    v_char = v_end
    if have_fit:
        v0 = float(v_of_t(t0_obs))
        if np.isfinite(v0) and v0 > 0:
            v_char = v0
    if peak_mag is not None:
        magnitude_input = {'peak_mag': float(peak_mag), 'distance_km': 100.0,
                           'source': 'manual_peak_magnitude'}
        # Order-of-magnitude mass anchor: a brief flash term (peak
        # magnitude held only ~MAG_FLASH_T_EFF_S seconds) plus the
        # sustained-phase mass loss from the measured light curve when a
        # calibrated photometric integral exists; an assumed survival
        # fraction then gives what reached dark flight.
        m_flash = _mag_to_photometric_mass(
            peak_mag, v_end, MAG_FLASH_T_EFF_S)
        m_sustained = ((photometry or {}).get('m_phot_kg') or 0.0)
        m_entry_est = m_flash + m_sustained
        f_lo, f_hi = _survival_fraction(v_char)
        mag_anchor = {
            'peak_mag': float(peak_mag),
            'm_entry_est_kg': m_entry_est,
            'm_flash_kg': m_flash,
            'm_sustained_kg': m_sustained,
            'm_surv_lo_kg': m_entry_est * f_lo,
            'm_surv_hi_kg': m_entry_est * f_hi,
            'm_fade_est_kg': math.sqrt(f_lo * f_hi) * m_entry_est,
            'survival_frac': [f_lo, f_hi],
            'v_char_ms': v_char,
            't_eff_s': MAG_FLASH_T_EFF_S,
            'method': 'peak_luminous_energy',
            'source': 'peak_magnitude',
            'order_of_magnitude': True}
        logging.info(f'darkflight: magnitude anchor peak={peak_mag:.1f} -> '
                     f'flash {m_flash:.3g} kg + sustained {m_sustained:.3g} kg'
                     f' = entry ~{m_entry_est:.3g} kg, surviving '
                     f'{m_entry_est*f_lo:.2g}-{m_entry_est*f_hi:.2g} kg '
                     f'(order-of-magnitude, not a dynamical constraint)')

    # Without a supplied peak magnitude the measured luminous-mass
    # integral still provides an order-of-magnitude mass scale; apply the
    # same survival fraction so the report can show a surviving-mass
    # estimate rather than only the ablated-mass proxy.
    phot_anchor = None
    m_phot = (photometry or {}).get('m_phot_kg')
    if m_phot and np.isfinite(m_phot) and m_phot > 0:
        f_lo, f_hi = _survival_fraction(v_char)
        phot_anchor = {
            'm_entry_est_kg': float(m_phot),
            'm_surv_lo_kg': float(m_phot) * f_lo,
            'm_surv_hi_kg': float(m_phot) * f_hi,
            'm_fade_est_kg': math.sqrt(f_lo * f_hi) * float(m_phot),
            'survival_frac': [f_lo, f_hi],
            'v_char_ms': v_char,
            'method': 'measured_luminous_integral',
            'source': 'photometric_light_curve',
            'order_of_magnitude': True}

    # Nominal scenario mass: only a constrained dynamical fit supplies mass;
    # otherwise retain hypothetical mass-grid paths without a mass estimate.
    m_est = float(m_med) if reliability == 'ok' and ref is not None and m_med > 0 else None
    mass_source = 'dynamic_constant_mass_fit' if m_est is not None else None
    mass_status = 'constrained' if m_est is not None else 'unconstrained'
    if m_est is not None:
        mc_mass_bounds = None
        mc_reference_mass = m_est
        mc_anchor = 'dynamic'
        mc_mass_prior = None
    elif mag_anchor is not None:
        mc_mass_bounds = None
        mc_reference_mass = mag_anchor['m_fade_est_kg']
        mc_anchor = 'magnitude'
        mc_mass_prior = {'source': 'magnitude_anchor',
                         'peak_mag': mag_anchor['peak_mag'],
                         'median_kg': mc_reference_mass,
                         'lo_kg': mag_anchor['m_surv_lo_kg'],
                         'hi_kg': mag_anchor['m_surv_hi_kg'],
                         'stage': 'end_of_luminous_flight'}
        logging.info('darkflight: MC anchored on magnitude-derived '
                     f'surviving-mass estimate ~{mc_reference_mass:.3g} kg '
                     '(order-of-magnitude)')
    else:
        mc_mass_bounds = (0.005, 5.0)
        mc_reference_mass = math.sqrt(mc_mass_bounds[0] * mc_mass_bounds[1])
        mc_anchor = 'hypothetical'
        mc_mass_prior = {'source': 'conservative_assumption',
                         'distribution': 'stratified_log_uniform',
                         'min_kg': mc_mass_bounds[0],
                         'max_kg': mc_mass_bounds[1],
                         'median_kg': mc_reference_mass,
                         'stage': 'end_of_luminous_flight'}
    if m_est is None and mag_anchor is None:
        logging.warning('darkflight: surviving mass is not constrained; '
                        'showing hypothetical 5 g–5 kg release-mass tracks')
        if mc_runs > 0:
            logging.info('darkflight: modelled MC samples fragment masses '
                         'log-uniformly over 5 g–5 kg at end of luminous flight; '
                         'this is not a measured total surviving mass')

    # Entry-mass back-projection (needs height vs time along track)
    entry_estimates = []
    if have_fit and m_est is not None:
        for e in mass_estimates:
            m_ent = entry_mass_estimate(e['m_fade_kg'], e['rho'], e['A'],
                                        v_of_t, h_of_t, (t0_obs, t_last),
                                        atm=atm)
            entry_estimates.append({'rho': e['rho'], 'A': e['A'],
                                    'm_entry_kg': m_ent,
                                    'm_fade_kg': e['m_fade_kg'],
                                    'm_crit_kg': e['m_crit_kg'],
                                    'm_fade_lo': e['m_fade_lo'],
                                    'm_fade_hi': e['m_fade_hi']})

    # Ground height near the nominal landing point
    h_ground = ground_elevation(end_lat, end_lon)

    v0_vec = track_dir * v_end
    _, rho_end, _ = atm.at(max(end_h, 0.0))
    dyn_press = cd_hypersonic(1.4) * rho_end * v_end ** 2
    scenarios = build_scenarios(m_est, rho_m=rho_ref or 3500.0,
                                dyn_press=dyn_press)

    # Parallel propagation: each run is independent and CPU-bound.
    # Start workers only when more than half the CPUs are idle.
    jobs = [(si, run, r_end, v0_vec, atm, h_ground, sc['erode'])
            for si, sc in enumerate(scenarios)
            for run in sc['runs']]
    own_pool = False
    try:
        allow_processes = getattr(propagate, '__module__', None) == __name__
        results_list = _parallel_map(
            _prop_worker, jobs, pool=pool,
            allow_processes=allow_processes)
        for si, run, res in results_list:
            scenarios[si].setdefault('results', []).append(
                {'m': run['m'], 'rho': run['rho'],
                 'impact': res['impact'],
                 'lon': res['lon'], 'lat': res['lat'],
                 'h': res['h'], 'v': res['v'],
                 'mass_t': res['m']})

        # Monte Carlo
        mc_impacts = []
        mc_attempted = 0
        if mc_runs > 0 and mc_reference_mass > 0:
            # if the S5 breakup scenario fired, the real fall is a
            # fragment strewn field — draw MC masses from its spectrum
            # instead of perturbing a single nominal body
            m_samples = None
            if (mc_anchor in ('dynamic', 'magnitude')
                    and dyn_press > compressive_strength(rho_ref or 3500.0)):
                m_samples = fragmentation_spectrum(
                    mc_reference_mass,
                    severity=dyn_press / compressive_strength(rho_ref or 3500.0))
            if m_samples is not None:
                m_samples = m_samples[m_samples > 0]
            mc_attempted = mc_runs
            mc_impacts = [i for i in monte_carlo(
                r_end, v0_vec, mc_reference_mass, rho_ref or 3500.0, 1.4, wind_csv, h_ground,
                n_runs=mc_runs, seed=seed, pool=pool,
                lon=end_lon, lat=end_lat, event_dt=event_dt,
                mass_samples=m_samples, mass_bounds=mc_mass_bounds)
                if i.get('landed', True)]
    finally:
        if own_pool:
            pool.shutdown()

    results = {
        'event_dir': str(event_dir),
        'end_state': {'lon': end_lon, 'lat': end_lat, 'h_m': end_h,
                      'v_end_ms': v_end, 'a_end_ms2': a_end},
        'wind_used': atm.has_profile,
        'h_ground_m': h_ground,
        'mass_estimates': mass_estimates,
        'entry_estimates': entry_estimates,
        'm_est_ref_kg': m_est,
        'mass_source': mass_source,
        'mass_status': mass_status,
        'density_ref_kg_m3': float(rho_ref or 3500.0),
        'photometric': photometry,
        'mag_anchor': mag_anchor,
        'phot_anchor': phot_anchor,
        'magnitude_input': magnitude_input,
        'model_assumptions': ['flat ground at release-site elevation',
                              'heuristic strength and fragmentation distribution',
                              'MC samples independent fragments, not complete falls',
                              'buffered MC outline is not a confidence contour'] + (
            ['modelled MC samples fragment masses log-uniformly over '
             '5 g–5 kg at end of luminous flight']
            if mc_anchor == 'hypothetical' else
            [f'entry/surviving mass scale anchored on supplied peak magnitude '
             f'{peak_mag:.1f} (flash {MAG_FLASH_T_EFF_S} s plus measured '
             f'luminous-mass integral, survival fraction '
             f'{mag_anchor["survival_frac"][0]:.2f}–'
             f'{mag_anchor["survival_frac"][1]:.2f} at '
             f'{v_char/1000:.1f} km/s); order-of-magnitude only']
            if mc_anchor == 'magnitude' else []),
        'reliability': reliability,
        'issues': issues,
        'n_obs': int(n_obs), 'track_duration_s': float(duration),
        'scenarios': [{'name': s['name'], 'label': s['label'],
                       'hypothetical': s.get('hypothetical', False),
                       'impacts': [{'m': r['m'], **r['impact']}
                                   for r in s['results']]}
                      for s in scenarios],
        'mc': {'n': len(mc_impacts), 'attempted': mc_attempted,
               'requested': mc_runs, 'impacts': mc_impacts,
               'hypothetical': mc_anchor == 'hypothetical',
               'anchor': mc_anchor, 'mass_prior': mc_mass_prior,
               'skipped_reason': 'disabled' if mc_runs == 0 else None},
    }
    write_json(results, event_dir / 'darkflight.json')
    write_geojson(scenarios, event_dir / 'darkflight.geojson')
    write_kml(scenarios, (end_lon, end_lat, end_h),
              event_dir / 'darkflight.kml', mc_impacts=mc_impacts,
              mc_assumed=m_est is None)
    for file_prefix, tr in (lang_files or {'': {}}).items():
        try:
            write_map(scenarios, (end_lon, end_lat, end_h), mc_impacts,
                      event_dir / f'{file_prefix}map_darkflight.svg',
                      translations=tr, mc_assumed=m_est is None)
        except Exception as e:
            logging.warning(f'darkflight map failed: {e}')
        try:
            write_map3d(scenarios, (end_lon, end_lat, end_h), mc_impacts,
                        event_dir / f'{file_prefix}darkflight_map3d.html',
                        translations=tr, mc_assumed=m_est is None,
                        start_llh=(start_lon, start_lat, start_h))
        except Exception as e:
            logging.warning(f'darkflight 3d map failed: {e}')
    return results


def _weighted_median(values, weights=None):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    values = values[order]
    if weights is None:
        weights = np.ones_like(values)
    weights = np.asarray(weights, dtype=float)[order]
    total = float(np.sum(weights))
    if not values.size or not np.isfinite(total) or total <= 0:
        return float(np.median(values)) if values.size else math.nan
    idx = int(np.searchsorted(np.cumsum(weights), 0.5 * total, side='left'))
    return float(values[min(idx, len(values) - 1)])


def _camera_aligned_model_times(camera, plot_data):
    """Map camera trail points to fitted model times using merged station IDs."""
    merged = plot_data.get('final_merged_data') or {}
    station_ids_raw = plot_data.get('station_id_array')
    reltime_raw = merged.get('reltime')
    color_raw = merged.get('color_time')
    timestamps_raw = camera.get('timestamps')
    if (station_ids_raw is None or reltime_raw is None or color_raw is None
            or timestamps_raw is None):
        return None, None
    station_ids = np.asarray(station_ids_raw, dtype=int)
    reltime = np.asarray(reltime_raw, dtype=float)
    color_time = np.asarray(color_raw, dtype=float)
    timestamps = np.asarray(timestamps_raw, dtype=float)
    if (not station_ids.size or reltime.size != station_ids.size
            or color_time.size != station_ids.size or timestamps.size < 3
            or not np.all(np.isfinite(timestamps))):
        return None, None
    candidates = []
    details = plot_data.get('station_error_details') or []
    indices = [detail.get('source_index', detail.get('index'))
               for detail in details]
    for index in dict.fromkeys(i for i in indices if i is not None):
        selected = station_ids == int(index)
        if int(np.sum(selected)) != timestamps.size:
            continue
        source_color = color_time[selected]
        source_time = reltime[selected]
        if (not np.all(np.isfinite(source_color))
                or not np.all(np.isfinite(source_time))
                or source_color.size < 3 or np.ptp(source_color) <= 0
                or np.ptp(source_time) <= 0):
            continue
        timing_error = ((timestamps - timestamps[0]) -
                        (source_color - source_color[0]))
        score = float(np.median(np.abs(timing_error)))
        if score <= 0.15 and np.all(np.diff(source_time) > 0):
            candidates.append((score, index, source_time))
    if not candidates:
        return None, None
    _, index, model_times = min(candidates)
    return model_times, int(index)


def _observed_light_curve_samples(photometry, plot_data, resdat=None):
    """Extract range-corrected observed curves mapped to forward-model time."""
    from fbspd_merge import expfunc
    if not photometry or not photometry.get('per_cam'):
        return [], 'no calibrated camera light curves'
    params = np.asarray(plot_data['final_params'], dtype=float)
    times = np.asarray(plot_data['final_merged_data']['reltime'], dtype=float)
    if params.shape != (4,) or times.ndim != 1 or times.size < 2:
        return [], 'invalid luminous-track time fit'
    position = np.asarray(expfunc(times, *params), dtype=float)
    span = float(position[-1] - position[0])
    if (not np.all(np.isfinite(position)) or not np.isfinite(span)
            or span <= 0 or np.any(np.diff(position) < 0)):
        return [], 'non-increasing fitted luminous-track distance'
    position = (position - position[0]) / span
    model_time = times - float(times[0])
    r_start = direction = None
    if resdat is not None:
        r_start = llh2ecef(float(resdat.long1[0]), float(resdat.lat1[0]),
                           float(resdat.height[0]) * 1000.0)
        r_end = llh2ecef(float(resdat.long1[1]), float(resdat.lat1[1]),
                         float(resdat.height[1]) * 1000.0)
        direction = r_end - r_start
        direction /= np.linalg.norm(direction)
    model_position_0 = float(expfunc(0.0, *params))
    samples = []
    for camera in photometry.get('per_cam', []):
        aligned_times, source_index = _camera_aligned_model_times(camera, plot_data)
        track_raw = camera.get('track_frac')
        track_frac = np.asarray(track_raw if track_raw is not None else [],
                                dtype=float)
        if track_frac.ndim != 1:
            track_frac = np.array([])
        if aligned_times is None:
            point_times = np.full(track_frac.shape, np.nan)
            valid_frac = (np.isfinite(track_frac) &
                          (track_frac >= 0.0) & (track_frac <= 1.0))
            point_times[valid_frac] = np.interp(
                track_frac[valid_frac], position, model_time)
            time_source = 'track_fraction'
        else:
            point_times = np.asarray(aligned_times, dtype=float) - times[0]
            time_source = 'station_alignment'
        range_raw = camera.get('ranges_km')
        distances = np.asarray(range_raw if range_raw is not None else [],
                               dtype=float)
        if (r_start is not None and direction is not None
                and camera.get('lat') is not None and camera.get('lon') is not None
                and camera.get('elev') is not None):
            observer = llh2ecef(float(camera['lon']), float(camera['lat']),
                                float(camera['elev']))
            positions = (r_start + np.outer(
                (expfunc(point_times, *params) - model_position_0) * 1000.0,
                direction))
            distances = np.linalg.norm(positions - observer, axis=1) / 1000.0
        if distances.size < point_times.size:
            distances = np.r_[distances,
                              np.full(point_times.size - distances.size, np.nan)]
        finite_ranges = distances[np.isfinite(distances) & (distances > 0)]
        default_range = (float(np.median(finite_ranges))
                         if finite_ranges.size else 100.0)

        def usable_rows(values, saturated, calibrated, source):
            rows = []
            for index, (value, distance, sat) in enumerate(
                    zip(values, distances, saturated)):
                if index >= len(point_times):
                    break
                if (value is None or sat is None
                        or not np.isfinite(value)
                        or not np.isfinite(point_times[index])):
                    continue
                if calibrated:
                    flux = (1500.0 * (distance / 100.0) ** 2 *
                            10.0 ** (-0.4 * value))
                else:
                    flux = float(value) * (distance / 100.0) ** 2
                if np.isfinite(flux) and flux > 0:
                    rows.append((float(point_times[index]), float(flux), bool(sat)))
            rows = sorted(rows, key=lambda row: (row[0], row[1], row[2]))
            return rows if len(rows) >= 3 and rows[-1][0] > rows[0][0] else []

        distances = np.where(np.isfinite(distances) & (distances > 0),
                             distances, default_range)
        video = camera.get('video') or {}
        event_sat = [b is not None and float(b) >= 250.0
                     for b in (camera.get('brightness') or [])]
        candidates = [
            ('video', video.get('m_app') or [], video.get('sat') or [], True),
            ('video_flux', video.get('flux') or [], video.get('sat') or [], False),
            ('event_txt', camera.get('m_app') or [], event_sat, True),
            ('event_brightness', camera.get('brightness') or [], event_sat, False)]
        rows = source = None
        for name, values, saturated, calibrated in candidates:
            saturated = list(saturated)
            saturated += [False] * max(0, point_times.size - len(saturated))
            rows = usable_rows(values, saturated, calibrated, name)
            if rows:
                source = name
                break
        if not rows:
            continue
        confidence_weight = {'medium': 1.0, 'low': 0.4}.get(
            camera.get('confidence'), 0.2)
        if 'saturated' in (camera.get('warnings') or []):
            confidence_weight *= 0.5
        if source.endswith('flux') or source.endswith('brightness'):
            confidence_weight *= 0.5
        t = np.asarray([row[0] for row in rows], dtype=float)
        flux = np.asarray([row[1] for row in rows], dtype=float)
        saturated = np.asarray([row[2] for row in rows], dtype=bool)
        coverage = min(1.0, float(np.ptp(t) / model_time[-1]))
        coverage_weight = 0.2 + 0.8 * coverage
        samples.append({'cam': camera.get('cam'), 'source': source,
                        'confidence': camera.get('confidence'),
                        'confidence_weight': confidence_weight,
                        'coverage_fraction': coverage,
                        'coverage_weight': coverage_weight,
                        'weight': confidence_weight * coverage_weight,
                        'time_source': time_source,
                        'source_index': source_index,
                        'warnings': camera.get('warnings') or [],
                        't': t, 'flux': flux, 'sat': saturated})
    if not samples:
        return [], 'no camera light curve with three usable points'
    return samples, None


def _observed_peak_magnitude_estimate(observed):
    """Infer an absolute peak only from reliable, unsaturated magnitudes.

    Saturated or flux-proxy curves constrain the shape, but their peak is a
    lower bound rather than a measurement. Requiring multiple independent
    usable peaks avoids transferring a single calibration error into a
    spurious mass inversion.
    """
    peaks = []
    cameras = []
    for camera in observed or []:
        if (camera.get('source') not in ('video', 'event_txt')
                or camera.get('confidence') != 'medium'
                or 'saturated' in (camera.get('warnings') or [])):
            continue
        usable = [(f, s) for f, s in zip(camera.get('flux', []),
                                         camera.get('sat', []))
                  if np.isfinite(f) and f > 0]
        unsaturated = [float(f) for f, s in usable if not s]
        if len(unsaturated) < 3:
            continue
        peak = max(unsaturated)
        peak_index = int(np.argmax([f for f, _ in usable]))
        if usable[peak_index][1]:
            continue
        peaks.append(float(peak))
        cameras.append({'cam': camera.get('cam'),
                        'peak_absolute_magnitude': float(
                            -2.5 * math.log10(peak / 1500.0)),
                        'n_points': int(len(usable))})
    if len(peaks) < 2:
        return None
    peak_mags = np.asarray([camera['peak_absolute_magnitude']
                            for camera in cameras])
    if np.ptp(peak_mags) > 1.5:
        return None
    peak_flux = float(np.median(peaks))
    return {'absolute_magnitude_at_100_km': float(
                -2.5 * math.log10(peak_flux / 1500.0)),
            'source': 'calibrated_unsaturated_camera_light_curves',
            'n_cameras': len(peaks), 'cameras': cameras,
            'peak_magnitude_spread': float(np.ptp(peak_mags)),
            'method': 'median of independent camera peak fluxes at 100 km'}


def _post_observed_luminous_penalty(model_t, model_power, observed_end):
    """Heuristic penalty for model light after the observed disappearance."""
    positive = np.flatnonzero(model_power > 0)
    if not positive.size:
        return {'post_observed_duration_s': 0.0,
                'post_observed_peak_fraction': 0.0,
                'post_observed_energy_fraction': 0.0,
                'peak_time_s': None, 'peak_delay_s': None,
                'model_light_end_s': None,
                'terminal_mismatch': 0.0}
    peak_index = int(np.argmax(model_power))
    peak_time = float(model_t[peak_index])
    model_end = float(model_t[positive[-1]])
    post_index = np.flatnonzero(model_t > observed_end)
    if not post_index.size:
        post_peak_fraction = post_energy_fraction = 0.0
        post_duration = 0.0
    else:
        post_t = np.r_[observed_end, model_t[post_index]]
        post_power = np.r_[
            np.interp(observed_end, model_t, model_power),
            model_power[post_index]]
        total_energy = float(np.sum(np.diff(model_t) *
                                    (model_power[:-1] + model_power[1:]) * 0.5))
        post_energy = float(np.sum(np.diff(post_t) *
                                   (post_power[:-1] + post_power[1:]) * 0.5))
        post_peak_fraction = float(np.max(post_power) /
                                   max(model_power[peak_index], 1e-300))
        post_energy_fraction = float(np.clip(
            post_energy / max(total_energy, 1e-300), 0.0, 1.0))
        post_duration = max(0.0, model_end - observed_end)
    return {'post_observed_duration_s': float(post_duration),
            'post_observed_peak_fraction': post_peak_fraction,
            'post_observed_energy_fraction': post_energy_fraction,
            'peak_time_s': peak_time,
            'peak_delay_s': peak_time - float(observed_end),
            'model_light_end_s': model_end,
            'terminal_mismatch': float(
                0.5 * (post_peak_fraction + post_energy_fraction))}


def _light_curve_shape_scores(experiments, observed, sigma=0.35):
    """Score normalized log-flux shapes and convert to relative weights."""
    cap = 3.0
    terminal_weight = 1.0
    observed_end = max(float(camera['t'][-1]) for camera in observed) if observed else None
    scores = []
    for index, experiment in enumerate(experiments):
        curve = experiment.get('light_curve') or {}
        model_t = np.asarray(curve.get('t_s', []), dtype=float)
        model_power = np.asarray(curve.get('power_w', []), dtype=float)
        valid_model = (model_t.size == model_power.size and model_t.size >= 2
                       and np.all(np.isfinite(model_t))
                       and np.all(np.isfinite(model_power))
                       and np.all(np.diff(model_t) > 0))
        weighted_sse = 0.0
        effective_points = 0.0
        all_residuals = []
        all_weights = []
        per_camera = []
        if valid_model:
            for camera in observed:
                predicted = np.interp(camera['t'], model_t, model_power,
                                      left=0.0, right=0.0)
                in_window = ((camera['t'] >= model_t[0]) &
                             (camera['t'] <= model_t[-1]))
                positive = in_window & (predicted > 0.0)
                raw = np.full(camera['t'].shape, cap)
                raw[positive] = np.log10(camera['flux'][positive] /
                                         predicted[positive])
                saturated = np.asarray(camera.get(
                    'sat', np.zeros(camera['t'].shape)), dtype=bool)
                unsaturated = ~saturated
                scale_points = positive & unsaturated
                offset = (_weighted_median(raw[scale_points])
                          if np.any(scale_points) else 0.0)
                residual = raw - offset
                residual[saturated] = np.maximum(
                    residual[saturated], 0.0)
                residual = np.clip(residual, -cap, cap)
                residual[~in_window] = cap
                if not np.all(np.isfinite(residual)):
                    continue
                rms = float(np.sqrt(np.mean(residual ** 2)))
                # A camera's cadence should not dominate the joint score:
                # use its mean shape residual, with only a logarithmic
                # sample-count contribution.
                weight = float(camera['weight']) * math.log1p(residual.size)
                weighted_sse += weight * float(np.mean(residual ** 2))
                effective_points += weight
                all_residuals.extend(residual.tolist())
                all_weights.extend(np.repeat(weight / residual.size,
                                             residual.size))
                per_camera.append({'cam': camera['cam'], 'source': camera['source'],
                                   'n_points': int(residual.size),
                                   'n_saturated_lower_bound': int(np.sum(saturated)),
                                   'coverage_fraction': camera['coverage_fraction'],
                                   'time_source': camera['time_source'],
                                   'score_weight': weight,
                                   'rms_log10': rms})
        terminal = (_post_observed_luminous_penalty(
            model_t, model_power, observed_end)
            if valid_model and observed_end is not None else {})
        terminal_penalty_sse = terminal_weight * terminal.get(
            'terminal_mismatch', 0.0)
        weighted_sse += terminal_penalty_sse
        if valid_model and effective_points > 0 and all_residuals:
            rms_total = math.sqrt(weighted_sse / effective_points)
            median_abs = _weighted_median(np.abs(all_residuals), all_weights)
            log_likelihood = -0.5 * weighted_sse / (sigma ** 2)
        else:
            rms_total = median_abs = None
            log_likelihood = -math.inf
        scores.append({'index': index, 'model': experiment.get('model'),
                       'available': valid_model and bool(observed)
                       and effective_points > 0,
                       'log_likelihood': log_likelihood,
                       'rms_log10': rms_total,
                       'median_abs_log10': median_abs,
                       **terminal,
                       'terminal_penalty_sse': terminal_penalty_sse,
                       'n_points': len(all_residuals),
                       'n_cameras': len(per_camera), 'per_camera': per_camera})
    finite = [s['log_likelihood'] for s in scores if math.isfinite(s['log_likelihood'])]
    if observed and finite:
        maximum = max(finite)
        weights = [math.exp(s['log_likelihood'] - maximum)
                   if math.isfinite(s['log_likelihood']) else 0.0
                   for s in scores]
        total = math.fsum(weights)
        ranks = sorted((i for i, s in enumerate(scores)
                        if math.isfinite(s['log_likelihood'])),
                       key=lambda i: (-(scores[i]['log_likelihood']), i))
        for score in scores:
            score['shape_rank'] = None
        for rank, index in enumerate(ranks, 1):
            scores[index]['shape_rank'] = rank
        for score, weight in zip(scores, weights):
            score['shape_weight'] = (weight / total
                                     if total and score['available'] else None)
        terminal_tau_s = 0.5
        preference_scores = []
        for score in scores:
            post_light = score.get('post_observed_duration_s')
            terminal_score = (math.exp(-post_light / terminal_tau_s)
                              if post_light is not None else 0.0)
            density_prior = float(
                experiments[score['index']].get('density_prior_weight', 1.0))
            if not np.isfinite(density_prior) or density_prior < 0:
                density_prior = 0.0
            score['terminal_match_tau_s'] = terminal_tau_s
            score['terminal_match_score'] = terminal_score
            score['density_prior_weight'] = density_prior
            preference_scores.append(
                (score['shape_weight'] or 0.0) * terminal_score
                * density_prior)
        preference_total = math.fsum(preference_scores)
        preference_ranks = sorted(
            (i for i, value in enumerate(preference_scores) if value > 0),
            key=lambda i: (-preference_scores[i], i))
        for score in scores:
            score['preference_rank'] = None
        for rank, index in enumerate(preference_ranks, 1):
            scores[index]['preference_rank'] = rank
        for score, value in zip(scores, preference_scores):
            score['model_preference_weight'] = (
                value / preference_total
                if preference_total and score['available'] else None)
    else:
        for score in scores:
            score['shape_weight'] = None
            score['shape_rank'] = None
            score['terminal_match_tau_s'] = 0.5
            score['terminal_match_score'] = None
            score['density_prior_weight'] = float(
                experiments[score['index']].get('density_prior_weight', 1.0))
            score['preference_rank'] = None
            score['model_preference_weight'] = None
    return scores


def _light_curve_comparison_metadata(photometry, observed, reason):
    return {'available': bool(observed),
            'unavailable_reason': None if observed else reason,
            'method': ('range-corrected normalized log10 flux shape; absolute '
                       'scale removed by a per-camera median offset, '
                       'coverage-weighted camera contributions, one-sided '
                       'saturated lower-bound residuals, and a heuristic '
                       'post-observed luminous-persistence penalty'),
            'likelihood': ('Gaussian shape residual with uniform prior across '
                           'experiments; normalized to relative weights'),
            'sigma_log10': 0.35, 'residual_cap_log10': 3.0,
            'observed_light_end_s': (float(max(c['t'][-1] for c in observed))
                                   if observed else None),
            'terminal_penalty': ('half of post-observed peak fraction plus '
                                 'post-observed energy fraction, weighted 1.0; '
                                 'a screening heuristic, not a detection likelihood'),
            'time_basis': ('merged station time alignment, with fitted '
                           'along-track fraction fallback'),
            'n_cameras': len(observed),
            'cameras': [{'cam': c['cam'], 'source': c['source'],
                         'confidence': c['confidence'],
                         'time_source': c['time_source'],
                         'source_index': c['source_index'],
                         'n_points': int(len(c['t'])),
                         'n_saturated_lower_bound': int(np.sum(c['sat'])),
                         'coverage_fraction': c['coverage_fraction'],
                         'coverage_weight': c['coverage_weight'],
                         'confidence_weight': c['confidence_weight'],
                         'weight': c['weight'],
                         'start_s': float(c['t'][0]), 'end_s': float(c['t'][-1]),
                         'duration_s': float(np.ptp(c['t']))}
                        for c in observed],
            'photometry_mass_cams': photometry.get('n_cams') if photometry else 0}


def _solve_fragmentation_mass_for_magnitude(model, propagate_one, target_mag,
                                            speed_ms, duration_s):
    """Solve log(mass) so a model reproduces the supplied absolute peak magnitude."""
    from scipy.optimize import brentq
    min_log, max_log = math.log(1e-9), math.log(1e9)
    try:
        seed = _mag_to_photometric_mass(target_mag, speed_ms, max(duration_s, 1e-3))
    except (TypeError, ValueError, OverflowError):
        seed = 1.0
    mid_log = min(max(math.log(seed), min_log), max_log)
    cache = {}

    def evaluate(log_mass):
        if log_mass in cache:
            return cache[log_mass][2]
        mass = float(math.exp(log_mass))
        experiment = propagate_one(mass)
        magnitude = experiment.get('peak_absolute_magnitude')
        if magnitude is None:
            residual = math.inf
        elif not np.isfinite(magnitude):
            raise ValueError(f'{model} returned a nonfinite peak magnitude')
        else:
            residual = float(magnitude) - target_mag
        cache[log_mass] = (mass, experiment, residual)
        return residual

    mid_residual = evaluate(mid_log)
    step = math.log(4.0)
    if mid_residual > 0.0:
        lower_log, lower_residual = mid_log, mid_residual
        upper_log, upper_residual = mid_log, None
        while upper_log < max_log:
            upper_log = min(upper_log + step, max_log)
            upper_residual = evaluate(upper_log)
            if upper_residual <= 0.0:
                break
        else:
            raise ValueError(f'No {model} entry mass up to 1e9 kg reaches magnitude {target_mag}')
    elif mid_residual < 0.0:
        upper_log, upper_residual = mid_log, mid_residual
        lower_log, lower_residual = mid_log, None
        while lower_log > min_log:
            lower_log = max(lower_log - step, min_log)
            lower_residual = evaluate(lower_log)
            if lower_residual >= 0.0:
                break
        else:
            raise ValueError(f'Even a 1e-9 kg {model} object is brighter than magnitude {target_mag}')
    else:
        lower_log = upper_log = mid_log
        lower_residual = upper_residual = 0.0

    try:
        root_log = (mid_log if mid_residual == 0.0 else
                    brentq(evaluate, lower_log, upper_log,
                           xtol=1e-7, rtol=1e-10, maxiter=40))
        root_residual = evaluate(root_log)
    except Exception as exc:
        best = min(cache.values(), key=lambda item: abs(item[2]))
        if not math.isfinite(best[2]) or abs(best[2]) > 0.02:
            raise ValueError(f'Could not invert {model} peak magnitude {target_mag}: {exc}')
        root_log = math.log(best[0])
        root_residual = best[2]
    mass, experiment, residual = cache[root_log]
    if not math.isfinite(residual) or abs(residual) > 0.02:
        raise ValueError(f'Could not match {model} peak magnitude {target_mag}; '
                         f'nearest residual is {residual:.3g} mag')
    return experiment, {'target_absolute_magnitude_at_100_km': target_mag,
                        'solved_entry_mass_kg': mass,
                        'residual_model_minus_target_mag': residual,
                        'method': 'log-mass bracket plus brentq',
                        'function_evaluations': len(cache),
                        'bracket_kg': [math.exp(lower_log), math.exp(upper_log)]}


def run_fragmentation_models(event_dir, resdat, plot_data, output_dir,
                             entry_masses, models, rho_m=3500.0,
                             peak_mag=None, density_priors=None,
                             h_ground=None, lang_files=None,
                             mc_runs=0, seed=0, pool=None, **model_options):
    """Forward-model assumed masses at the FIRST observed point, not at fade-out.

    These are sensitivity experiments, not fitted mass or survival estimates.
    If entry_masses is omitted, peak_mag is solved into one conditional entry
    mass per model. With neither input, a deceleration-derived mass is
    back-projected to the first observed point for each material density.
    Optional density_priors weight the normalized terminal preference only;
    they do not alter propagation, mass estimates or shape likelihoods.
    With output_dir=None, updates event dark-flight maps, statistics and exports.
    An explicit separate directory preserves the event's published products.
    Existing dark-flight reports are never read as mass priors.
    """
    from datetime import datetime, timezone
    from fbspd_merge import expfunc_1stder, expfunc_2ndder

    publish = output_dir is None
    event_dir = Path(event_dir)
    output_dir = event_dir if publish else Path(output_dir)
    models = list(dict.fromkeys(models))
    if 'all' in models:
        models = ['single', 'discrete', 'disruption', 'erosion']
    if not models or any(m not in ('single', 'discrete', 'disruption', 'erosion') for m in models):
        raise ValueError('Select single, discrete, disruption, erosion, or all')
    invert_mass = entry_masses is None and peak_mag is not None
    estimate_entry_mass = entry_masses is None and peak_mag is None
    masses = None
    if entry_masses is not None:
        masses = np.asarray(entry_masses, dtype=float)
        if masses.ndim != 1 or not masses.size or not np.all(np.isfinite(masses) & (masses > 0)):
            raise ValueError('Entry masses must be finite positive kg')
    if isinstance(rho_m, (int, float, np.number)):
        densities = [float(rho_m)]
    else:
        densities = [float(value) for value in rho_m]
    densities = list(dict.fromkeys(densities))
    if (not densities or any(not np.isfinite(density) or density <= 0
                             for density in densities)):
        raise ValueError('Densities must be positive finite kg/m3')
    if density_priors is None:
        prior_weights = np.ones(len(densities), dtype=float)
    else:
        prior_weights = np.asarray(
            [float(value) for value in density_priors], dtype=float)
        if prior_weights.size != len(densities):
            raise ValueError(
                '--density-priors must provide one nonnegative weight per '
                '--density value')
    if (not np.all(np.isfinite(prior_weights))
            or np.any(prior_weights < 0)
            or not np.any(prior_weights > 0)):
        raise ValueError('Density priors must be nonnegative and not all zero')
    prior_weights /= float(np.sum(prior_weights))
    density_prior_by_density = dict(zip(densities, prior_weights.tolist()))
    if peak_mag is not None and not np.isfinite(peak_mag):
        raise ValueError('Comparison magnitude must be finite')
    if invert_mass and peak_mag >= -1.0:
        hint = (f' — did you mean {-abs(peak_mag):.1f}?' if peak_mag > 0 else '')
        raise ValueError('Magnitude inversion requires a magnitude brighter than -1 '
                         f'(got {peak_mag}){hint}')
    if plot_data is None or plot_data.get('final_params') is None:
        raise ValueError('Forward entry requires the observed speed fit; no nominal speed is substituted')
    params = np.asarray(plot_data['final_params'], dtype=float)
    times = np.asarray(plot_data['final_merged_data']['reltime'], dtype=float)
    if (params.shape != (4,) or not np.all(np.isfinite(params)) or times.ndim != 1
            or times.size < 2 or not np.all(np.isfinite(times)) or np.ptp(times) <= 0):
        raise ValueError('Invalid speed-fit parameters or observed times')
    endpoints = np.array([[resdat.long1[i], resdat.lat1[i], resdat.height[i] * 1000.]
                          for i in range(2)], dtype=float)
    if (not np.all(np.isfinite(endpoints)) or np.any(np.abs(endpoints[:, 0]) > 180)
            or np.any(np.abs(endpoints[:, 1]) > 90)
            or not 0 <= endpoints[1, 2] < endpoints[0, 2] <= 120000.):
        raise ValueError('Forward entry requires a descending, finite observed trajectory')
    r_start, r_end = [llh2ecef(*point) for point in endpoints]
    direction = r_end - r_start
    direction /= np.linalg.norm(direction)
    speed = float(expfunc_1stder(float(times.min()), *params)) * 1000.
    if not 3000 < speed <= 72000:
        raise ValueError('Fitted entry speed must be between 3 and 72 km/s')
    if h_ground is None:
        h_ground = ground_elevation(endpoints[1, 1], endpoints[1, 0])
    if not np.isfinite(h_ground) or h_ground >= endpoints[1, 2]:
        raise ValueError('Ground height must be finite and below the observed endpoint')
    try:
        event_dt = datetime.strptime(f'{event_dir.parent.name} {event_dir.name}',
                                     '%Y%m%d %H%M%S').replace(tzinfo=timezone.utc)
    except ValueError:
        event_dt = None
    atm = WindAtmosphere(event_dir / 'wind_profile.csv', lon=endpoints[0, 0],
                         lat=endpoints[0, 1], event_dt=event_dt)
    observed_photometry = None
    try:
        import photometry as ph
        observed_photometry = ph.event_photometry(
            event_dir, resdat=resdat, plot_data=plot_data)
    except Exception as e:
        logging.debug(f'forward photometry failed: {e}')
        observed, observed_reason = [], 'photometry unavailable'
    else:
        observed, observed_reason = _observed_light_curve_samples(
            observed_photometry, plot_data, resdat)
    observed_peak_estimate = _observed_peak_magnitude_estimate(observed)
    peak_mag_source = 'supplied' if peak_mag is not None else None
    if estimate_entry_mass and observed_peak_estimate is not None:
        peak_mag = observed_peak_estimate['absolute_magnitude_at_100_km']
        if np.isfinite(peak_mag) and peak_mag < -1.0:
            invert_mass = True
            estimate_entry_mass = False
            peak_mag_source = 'observed_light_curve'
            logging.info('Forward entry: inferred peak magnitude %.2f from '
                         '%d unsaturated calibrated camera light curves',
                         peak_mag, observed_peak_estimate['n_cameras'])
        else:
            observed_peak_estimate = None
    estimated_masses = {}
    mass_estimates = []
    if estimate_entry_mass:
        try:
            heights = np.asarray(
                plot_data['final_merged_data']['height'], dtype=float)
        except (KeyError, TypeError, ValueError):
            heights = np.array([])
        fit_error = ('Forward fragmentation needs --entry-mass, --mag, or a '
                     'usable speed/deceleration-height fit')
        if (heights.shape != times.shape or len(heights) < 2
                or not np.all(np.isfinite(heights))):
            raise ValueError(fit_error)
        height_fit = np.polyfit(times, heights, 1) if len(heights) > 2 else None

        def v_auto(t):
            return float(expfunc_1stder(t, *params)) * 1000.0

        def a_auto(t):
            return -float(expfunc_2ndder(t, *params)) * 1000.0

        def h_auto(t):
            if height_fit is not None:
                return float(np.polyval(height_fit, t)) * 1000.0
            return endpoints[1, 2]

        mass_estimates = estimate_mass(
            v_auto, a_auto, h_auto, (float(times.min()), float(times.max())),
            rho_grid=densities, A=1.4, atm=atm, params=params,
            pcov=plot_data.get('pcov'))
        estimates_by_density = {float(estimate['rho']): estimate
                                for estimate in mass_estimates}
        for density in densities:
            estimate = estimates_by_density.get(float(density))
            if estimate is None:
                raise ValueError(fit_error)
            m_fade = float(estimate.get('m_fade_kg', np.inf))
            if not np.isfinite(m_fade) or m_fade <= 0:
                raise ValueError(fit_error)
            m_entry = float(entry_mass_estimate(
                m_fade, density, 1.4, v_auto, h_auto,
                (float(times.min()), float(times.max())), atm=atm))
            if not np.isfinite(m_entry) or m_entry <= 0:
                raise ValueError(fit_error)
            estimate['m_entry_kg'] = m_entry
            estimated_masses[float(density)] = m_entry
            estimate['warnings'] = []
            m_lo = float(estimate.get('m_fade_lo') or 0.)
            m_hi = float(estimate.get('m_fade_hi') or np.inf)
            if (not np.isfinite(m_hi) or m_lo <= 0
                    or m_hi / max(m_lo, 1e-12) > 20.):
                estimate['warnings'].append(
                    'deceleration mass poorly constrained')
            end_speed = float(expfunc_1stder(
                float(times.max()), *params)) * 1000.
            if (end_speed > 4000. and estimate.get('m_crit_kg')
                    and m_fade > 30. * estimate['m_crit_kg']):
                estimate['warnings'].append(
                    'deceleration mass inconsistent with luminous fade-out')
            if estimate['warnings']:
                logging.warning('Forward entry: %s at density %.4g kg/m3',
                                '; '.join(estimate['warnings']), density)
            logging.info('Forward entry: inferred %.3g kg at first observed '
                         'point from the deceleration fit (%.3g kg at fade, '
                         'density %.4g kg/m3)', m_entry, m_fade, density)
    output_names = ('fragmentation.json', 'fragmentation_summary.csv', 'fragmentation.geojson')
    if not publish and output_dir.resolve() == event_dir.resolve():
        raise ValueError('Omit --fragmentation-output to update the event, or use a separate directory')
    if not publish and any((output_dir / name).exists() for name in output_names):
        raise FileExistsError('Fragmentation output already exists; choose a new directory')
    def add_experiment(experiment, density):
        experiment['density_kg_m3'] = float(density)
        experiment['density_prior_weight'] = density_prior_by_density[
            float(density)]
        experiment['experiment_index'] = len(experiments)
        experiment['experiment_id'] = (
            f'{experiment["model"]}_rho{density:g}')
        experiments.append(experiment)

    experiment_jobs = []
    job_inputs = []
    duration = float(np.ptp(times))
    if invert_mass:
        for model in models:
            for density in densities:
                job_inputs.append((None, model, density))
    elif estimate_entry_mass:
        for model in models:
            for density in densities:
                job_inputs.append((estimated_masses[float(density)],
                                   model, density))
    else:
        for mass in masses:
            for model in models:
                for density in densities:
                    job_inputs.append((float(mass), model, density))
    for index, (mass, model, density) in enumerate(job_inputs):
        experiment_jobs.append(
            (index, mass, model, density, r_start, direction, speed, atm,
             float(h_ground), model_options,
             peak_mag if invert_mass else None, duration))
    allow_processes = getattr(
        propagate_fragmentation, '__module__', None) == __name__
    experiment_results = _parallel_map(
        _forward_experiment_worker, experiment_jobs, pool=pool,
        allow_processes=allow_processes)

    experiments = []
    for job_index, (mass, model, density) in enumerate(job_inputs):
        index, experiment, inversion, error = experiment_results[job_index]
        if error is not None:
            if not invert_mass:
                raise RuntimeError(
                    f'{model} forward propagation failed at density '
                    f'{density:g} kg/m3: {error}')
            logging.warning('Forward entry: %s magnitude inversion '
                            'unavailable at density %.4g kg/m3: %s',
                            model, density, error)
            experiment = {
                'model': model,
                'parameters': {'rho_m': float(density)},
                'mass_budget': {
                    'initial_kg': None, 'ablated_kg': None,
                    'landed_kg': None, 'unresolved_kg': None,
                    'timeout_kg': None, 'balance_error_kg': None},
                'fragments': [], 'largest_landed_kg': None,
                'peak_absolute_magnitude': None,
                'warnings': ['magnitude inversion failed'],
                'inversion_failed': True,
                'entry_mass_inversion': {
                    'target_absolute_magnitude_at_100_km': peak_mag,
                    'solved_entry_mass_kg': None,
                    'failed': True, 'error': error,
                    'method': 'log-mass bracket plus brentq'},
                'observed_peak_comparison': {
                    'absolute_magnitude_at_100_km': peak_mag,
                    'peak_magnitude_source': peak_mag_source,
                    'model_minus_observed_mag': None,
                    'used_to_fit_parameters': False}}
            add_experiment(experiment, density)
            continue
        if invert_mass:
            logging.info('Forward entry: solved %s entry mass for peak '
                         'magnitude %.2f at density %.4g kg/m3',
                         model, peak_mag, density)
            experiment['entry_mass_inversion'] = inversion
            predicted = experiment.get('peak_absolute_magnitude')
            experiment['observed_peak_comparison'] = {
                'absolute_magnitude_at_100_km': peak_mag,
                'peak_magnitude_source': peak_mag_source,
                'model_minus_observed_mag': (predicted - peak_mag
                    if predicted is not None else None),
                'used_to_fit_parameters': True}
        else:
            logging.info('Forward entry: %s, %.3g kg %sat %.1f km, '
                         '%.2f km/s, density %.4g kg/m3',
                         model, mass,
                         'estimated ' if estimate_entry_mass else '',
                         endpoints[0, 2] / 1000., speed / 1000., density)
            if estimate_entry_mass:
                estimate = estimates_by_density[float(density)]
                experiment['entry_mass_estimate'] = {
                    'method': 'observed_deceleration_drag_fit_with_ablation_back_projection',
                    'm_entry_kg': mass,
                    'm_fade_kg': estimate['m_fade_kg'],
                    'm_fade_lo': estimate.get('m_fade_lo'),
                    'm_fade_hi': estimate.get('m_fade_hi'),
                    'm_crit_kg': estimate.get('m_crit_kg'),
                    'warnings': estimate.get('warnings', []),
                    'density_kg_m3': float(density),
                    'conditional_input': True}
            predicted = experiment.get('peak_absolute_magnitude')
            experiment['observed_peak_comparison'] = {
                'absolute_magnitude_at_100_km': peak_mag,
                'peak_magnitude_source': peak_mag_source,
                'model_minus_observed_mag': (predicted - peak_mag
                    if predicted is not None and peak_mag is not None else None),
                'used_to_fit_parameters': False}
        add_experiment(experiment, density)
    shape_scores = _light_curve_shape_scores(experiments, observed)
    preference_unavailable_reason = None
    if estimate_entry_mass and any(
            estimate.get('warnings') for estimate in mass_estimates):
        preference_unavailable_reason = (
            'automatic deceleration-derived entry mass is poorly constrained '
            'or inconsistent with luminous fade-out')
        for score in shape_scores:
            score['preference_rank'] = None
            score['model_preference_weight'] = None
    for experiment, score in zip(experiments, shape_scores):
        experiment['light_curve_comparison'] = score
    mc_impacts = _forward_mc_impacts_parallel(
        experiments, event_dir / 'wind_profile.csv', h_ground, mc_runs,
        seed, event_dt=event_dt, pool=pool)
    comparison = _light_curve_comparison_metadata(
        observed_photometry, observed, observed_reason)
    if observed_peak_estimate is not None:
        comparison['observed_peak_magnitude_estimate'] = observed_peak_estimate
    if preference_unavailable_reason is not None:
        comparison['model_preference'] = {
            'available': False,
            'unavailable_reason': preference_unavailable_reason,
            'interpretation': ('shape diagnostics depend on the uncertain '
                               'entry-mass estimate')}
    finite_scores = [s for s in shape_scores
                     if math.isfinite(s['log_likelihood'])]
    if finite_scores:
        best = max(finite_scores, key=lambda s: s['log_likelihood'])
        comparison['best_shape_experiment'] = {
            'index': best['index'], 'model': experiments[best['index']]['model'],
            'initial_mass_kg': experiments[best['index']]['mass_budget']['initial_kg'],
            'density_kg_m3': experiments[best['index']]['density_kg_m3'],
            'shape_weight': best['shape_weight']}
    preference_scores = [s for s in shape_scores
                         if s.get('model_preference_weight') is not None]
    if preference_scores:
        best_preference = max(
            preference_scores, key=lambda s: s['model_preference_weight'])
        comparison['model_preference'] = {
            'formula': ('normalized shape_weight * '
                        'exp(-post_observed_duration_s / terminal_match_tau_s) '
                        '* density_prior_weight'),
            'terminal_match_tau_s': best_preference['terminal_match_tau_s'],
            'interpretation': ('terminal-dominated heuristic preference, not a '
                               'calibrated physical probability')}
        comparison['best_preference_experiment'] = {
            'index': best_preference['index'],
            'model': experiments[best_preference['index']]['model'],
            'initial_mass_kg': experiments[
                best_preference['index']]['mass_budget']['initial_kg'],
            'density_kg_m3': experiments[
                best_preference['index']]['density_kg_m3'],
            'density_prior_weight': best_preference['density_prior_weight'],
            'model_preference_weight':
                best_preference['model_preference_weight']}
    import hashlib
    report = {
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'event_dir': str(event_dir.resolve()),
        'initial_state': {'lon': endpoints[0, 0], 'lat': endpoints[0, 1],
                          'h_m': endpoints[0, 2], 'speed_ms': speed,
                          'mass_stage': 'first_observed_point'},
        'observed_end': {'lon': endpoints[1, 0], 'lat': endpoints[1, 1],
                         'h_m': endpoints[1, 2],
                         'speed_ms': float(expfunc_1stder(float(times.max()), *params)) * 1000.,
                         'duration_s': float(np.ptp(times))},
        'wind_used': atm.has_profile, 'h_ground_m': h_ground,
        'densities_kg_m3': densities,
        'density_priors': [{'density_kg_m3': density,
                            'normalized_weight': weight,
                            'explicit': density_priors is not None}
                           for density, weight in zip(
                               densities, prior_weights.tolist())],
        'entry_mass_source': (
            'observed_light_curve_magnitude_inversion'
            if invert_mass and peak_mag_source == 'observed_light_curve' else
            'magnitude_inversion' if invert_mass else
            'deceleration_back_projection'
            if estimate_entry_mass else 'supplied'),
        'entry_mass_estimates': mass_estimates if estimate_entry_mass else [],
        'light_curve_comparison': comparison,
        'mc': {'n': len(mc_impacts),
               'requested_per_experiment': int(mc_runs),
               'requested_total': int(mc_runs) * len(experiments),
               'attempted': int(math.fsum(e['monte_carlo']['attempted']
                                          for e in experiments)),
               'impacts': mc_impacts, 'conditional': True,
               'method': 'legacy-style disruption mass spectra at landed cohort release states',
               'interpretation': ('Modelled what-if mass spectra for each model; '
                                  'not stochastic fragmentation draws or model probabilities')},
        'interpretation': 'Modelled forward sensitivity models, not a fit or survival probability. '
                          'Mass before the first observed point is not reconstructed. '
                          + (('Entry masses are solved independently for each model to reproduce the '
                              'supplied absolute peak magnitude. '
                              if peak_mag_source == 'supplied' else
                              'Entry masses are solved independently for each model to reproduce the '
                              'absolute peak magnitude inferred from unsaturated calibrated camera '
                              'light curves. ') if invert_mass else '')
                          + ('Entry masses are inferred by least-squares drag fitting to the '
                             'observed deceleration-versus-altitude curve, then back-projected '
                             'with ablation to the first observed point; they are modelled '
                             'estimates, not reconstructed pre-atmospheric masses. '
                             if estimate_entry_mass else '')
                          + ('Light-curve shape weights are relative diagnostics, not probabilities '
                             'that a model is physically true. They include a heuristic penalty '
                             'for model light predicted after the observed disappearance. '
                             'Model preference percentages add a stronger terminal-persistence '
                             'weighting and are heuristic, not calibrated physical probabilities. '
                             if comparison['available'] else '')
                          + ('User-supplied density priors multiply the terminal preference '
                             'only; they do not change propagation, mass inference or '
                             'light-curve shape likelihoods. '
                             if density_priors is not None else '')
                          + ('Monte Carlo points are legacy-style modelled mass-spectrum '
                             'realizations for each model, not stochastic fragmentation draws. '
                             if mc_impacts else '')
                          + 'Absolute magnitude refers to distance 100 km, not altitude.',
        'experiments': experiments}
    if publish:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='.darkflight-', dir=event_dir) as temporary:
            staged = Path(temporary)
            _publish_fragmentation(report, staged, lang_files)
            _write_fragmentation_products(report, staged)
            _install_fragmentation_outputs(staged, event_dir)
    else:
        _write_fragmentation_products(report, output_dir)
    return report


def _install_fragmentation_outputs(staged, event_dir):
    staged_names = {product.name for product in staged.iterdir()}
    for product in staged.iterdir():
        product.replace(event_dir / product.name)
    for pattern in ('map_darkflight.*', '*_map_darkflight.*',
                    'darkflight_map3d.html', '*_darkflight_map3d.html',
                    'darkflight.json', 'darkflight.geojson', 'darkflight.kml',
                    'fragmentation.json', 'fragmentation_summary.csv',
                    'fragmentation.geojson', 'darkflight_table.html',
                    '*_darkflight_table.html'):
        for old in event_dir.glob(pattern):
            if old.name not in staged_names:
                try:
                    old.unlink()
                except OSError:
                    pass


def _write_fragmentation_products(report, output_dir):
    experiments = report['experiments']
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, output_dir / 'fragmentation.json')
    with (output_dir / 'fragmentation_summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'density_kg_m3', 'density_prior_weight',
                         'initial_kg', 'ablated_kg', 'landed_kg', 'unresolved_kg',
                         'timeout_kg', 'balance_error_kg', 'largest_landed_kg',
                         'peak_absolute_magnitude', 'shape_weight',
                         'post_observed_light_s', 'terminal_match_score',
                         'model_preference_weight',
                         'mc_requested', 'mc_attempted', 'mc_n'])
        for result in experiments:
            budget = result['mass_budget']
            comparison = result.get('light_curve_comparison') or {}
            shape = comparison.get('shape_weight')
            mc = result.get('monte_carlo') or {}
            writer.writerow([result['model'], result.get('density_kg_m3'),
                             result.get('density_prior_weight')]
                            + [budget[k] for k in (
                'initial_kg', 'ablated_kg', 'landed_kg', 'unresolved_kg', 'timeout_kg', 'balance_error_kg')]
                + [result['largest_landed_kg'], result['peak_absolute_magnitude'],
                   '' if shape is None else shape,
                   comparison.get('post_observed_duration_s'),
                   comparison.get('terminal_match_score'),
                   comparison.get('model_preference_weight'),
                   mc.get('requested'), mc.get('attempted'), mc.get('n')])
    features = []
    for result in experiments:
        for fragment in result['fragments']:
            impact = fragment.get('impact')
            if impact and impact.get('landed'):
                features.append({'type': 'Feature',
                    'geometry': {'type': 'Point', 'coordinates': [impact['lon'], impact['lat']]},
                    'properties': {'model': result['model'],
                        'density_kg_m3': result.get('density_kg_m3'),
                        'experiment_index': result.get('experiment_index'),
                        'initial_parent_mass_kg': result['mass_budget']['initial_kg'],
                        'fragment_mass_kg': impact['m'],
                        'multiplicity': fragment['multiplicity'], 'conditional': True}})
        for impact in (result.get('monte_carlo') or {}).get('impacts', []):
            trajectory = impact.get('trajectory') or {}
            if len(trajectory.get('lon', ())) >= 2:
                features.append({'type': 'Feature',
                    'geometry': {'type': 'LineString',
                                 'coordinates': [[lo, la, hh] for lo, la, hh in zip(
                                     trajectory['lon'], trajectory['lat'],
                                     trajectory['h'])]},
                    'properties': {'model': result['model'], 'monte_carlo': True,
                        'density_kg_m3': impact.get('density_kg_m3'),
                        'experiment_index': impact.get('experiment_index'),
                        'feature_kind': 'path',
                        'fragment_id': impact.get('fragment_id'),
                        'multiplicity': impact.get('multiplicity'), 'conditional': True}})
            features.append({'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [impact['lon'], impact['lat']]},
                'properties': {'model': result['model'], 'monte_carlo': True,
                    'density_kg_m3': impact.get('density_kg_m3'),
                    'experiment_index': impact.get('experiment_index'),
                    'fragment_id': impact.get('fragment_id'),
                    'fragment_mass_kg': impact['m'],
                    'multiplicity': impact.get('multiplicity'), 'conditional': True}})
    write_json({'type': 'FeatureCollection', 'features': features}, output_dir / 'fragmentation.geojson')
    return report


def _publish_fragmentation(report, output_dir, lang_files=None):
    """Render conditional, mass-conserving experiments as current event products."""
    import html
    scenarios = []
    for index, experiment in enumerate(report['experiments']):
        mass = experiment['mass_budget']['initial_kg']
        runs = []
        for fragment in experiment['fragments']:
            impact = fragment.get('impact')
            if not impact or not impact.get('landed'):
                continue
            trajectory = fragment.get('trajectory') or {
                key: [impact[key]] for key in ('lon', 'lat', 'h')}
            runs.append({'m': fragment.get('plot_release_mass_kg', impact['m']),
                         'rho': experiment.get('parameters', {}).get('rho_m'),
                         'multiplicity': fragment['multiplicity'], 'impact': impact,
                         **trajectory})
        density = experiment.get('density_kg_m3')
        density_text = (f', {_localized_number(density, ".0f", {})} kg/m3'
                        if density is not None else '')
        mass_text = 'unresolved mass' if mass is None else f'{mass:g} kg'
        scenarios.append({'name': f'F{index}_{experiment["model"]}',
                          'label': f'{experiment["model"]}, {mass_text} at first observation'
                                   f'{density_text} (modelled)',
                          'model': experiment['model'],
                          'experiment_index': experiment.get('experiment_index', index),
                          'density_kg_m3': experiment.get('density_kg_m3'),
                          'initial_mass_kg': mass,
                          'hypothetical': True, 'conditional': True, 'erode': True,
                          'results': runs, 'mass_budget': experiment['mass_budget'],
                          'fragments': experiment['fragments']})
    end = report['observed_end']
    end_llh = (end['lon'], end['lat'], end['h_m'])
    start = report.get('initial_state')
    start_llh = ((start['lon'], start['lat'], start['h_m'])
                 if start is not None else None)
    report.update(analysis_mode='fragmentation', publication_mode='event',
                  end_state={**end, 'v_end_ms': end['speed_ms']},
                  mass_status='assumed', reliability='conditional',
                  scenarios=[{'name': s['name'], 'label': s['label'], 'conditional': True,
                              'model': s['model'],
                              'experiment_index': s['experiment_index'],
                              'density_kg_m3': s['density_kg_m3'],
                              'initial_mass_kg': s['initial_mass_kg'],
                              'mass_budget': s['mass_budget'],
                              'impacts': [{**r['impact'], 'release_mass_kg': r['m'],
                                           'multiplicity': r['multiplicity']}
                                          for r in s['results']]} for s in scenarios])
    mc_impacts = (report.get('mc') or {}).get('impacts') or []
    preferred_index = next(
        (index for index, experiment in enumerate(report['experiments'])
         if (experiment.get('light_curve_comparison') or {}).get(
             'preference_rank') == 1), None)
    if preferred_index is None:
        preferred_index = next(
            (index for index, experiment in enumerate(report['experiments'])
             if (experiment.get('light_curve_comparison') or {}).get(
                 'shape_rank') == 1), None)
    preferred_experiment = (report['experiments'][preferred_index]
                            if preferred_index is not None else None)
    interactive_scenarios = ([s for s in scenarios
                              if s['experiment_index'] == preferred_index]
                             if preferred_index is not None else scenarios)
    interactive_mc = ([impact for impact in mc_impacts
                       if impact.get('experiment_index') == preferred_index or
                       ('experiment_index' not in impact and
                        impact.get('model') == preferred_experiment['model'])]
                      if preferred_index is not None else mc_impacts)
    report['interactive_model'] = (
        preferred_experiment['model'] if preferred_experiment else None)
    report['interactive_density_kg_m3'] = (
        preferred_experiment.get('density_kg_m3')
        if preferred_experiment else None)
    event_wind_csv = (Path(report['event_dir']) / 'wind_profile.csv'
                      if report.get('event_dir') else None)
    write_geojson(scenarios, output_dir / 'darkflight.geojson',
                  mc_impacts=mc_impacts)
    write_kml(scenarios, end_llh, output_dir / 'darkflight.kml',
              mc_impacts=mc_impacts, mc_assumed=True)
    has_impacts = any(s['results'] for s in interactive_scenarios)
    for prefix, translations in (lang_files or load_language_files()).items():
        title = translations.get(
            'df_frag_map_title' if has_impacts else 'df_frag_map_no_impacts',
            'Modelled fragmentation impacts' if has_impacts else
            'Modelled fragmentation: no ground impacts')
        plot_translations = {**translations, 'dark_flight': title}
        svg = output_dir / f'{prefix}map_darkflight.svg'
        try:
            write_map(interactive_scenarios, end_llh, interactive_mc, svg,
                      title=title, translations=plot_translations,
                      mc_assumed=True, start_llh=start_llh,
                      observed_duration_s=report['observed_end'].get(
                          'duration_s'))
        except Exception as e:
            logging.warning(f'fragmentation map failed: {e}')
            svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="800" '
                           'height="120"><text x="20" y="60">'
                           + html.escape(title) + '</text></svg>', encoding='utf-8')
        map3d = output_dir / f'{prefix}darkflight_map3d.html'
        try:
            map3d_ok = write_map3d(
                interactive_scenarios, end_llh, interactive_mc, map3d,
                translations=plot_translations, start_llh=start_llh,
                mc_assumed=True, wind_csv=event_wind_csv,
                observed_duration_s=report['observed_end'].get('duration_s'))
        except Exception as e:
            logging.warning(f'fragmentation 3d map failed: {e}')
            map3d_ok = False
        if map3d_ok is False:
            unavailable = translations.get(
                'df_frag_3d_unavailable',
                '3D renderer unavailable. See the current 2D map and statistics.')
            map3d.write_text('<!doctype html><meta charset="utf-8"><p>'
                             + html.escape(title) + '</p><p>'
                             + html.escape(unavailable) + '</p>', encoding='utf-8')
        _write_fragmentation_table(report, output_dir / f'{prefix}darkflight_table.html', translations)
    write_json(report, output_dir / 'darkflight.json')


def _fragmentation_warning_text(warning, translations):
    import re
    if warning == 'magnitude inversion failed':
        return translations.get(
            'df_frag_warn_inversion',
            'Peak-magnitude mass inversion failed for this experiment; it is '
            'shown as unavailable rather than assigned a fallback mass.')
    match = re.fullmatch(r'Cohort (\d+): generation cap (\d+) reached; continuing intact',
                         warning)
    if match:
        return translations.get(
            'df_frag_warn_generation',
            'Cohort {cohort}: generation cap {limit} reached; continuing intact').format(
                cohort=match.group(1), limit=match.group(2))
    return warning


def _write_fragmentation_table(report, path, translations):
    """Keep alternative experiments separate; unresolved solids are not impacts."""
    import html
    translations = translations or {}
    def tr(key, fallback):
        return translations.get(key, fallback)
    def number(value, spec='.4g'):
        return '—' if value is None else _localized_number(value, spec, translations)

    def mass_number(value, outcome_scale=0):
        if value is None:
            return '—'
        if value == 0 or abs(value) < 0.005:
            text = '0'
        elif abs(value) < 1.0:
            text = f'{float(value):.3f}'
        else:
            precision = 2 - int(math.floor(math.log10(abs(value))))
            precision = max(precision, int(outcome_scale))
            rounded = round(float(value), precision)
            text = (f'{rounded:.{precision}f}' if precision > 0
                    else f'{rounded:.0f}')
        dec = translations.get('dec_sep', '.')
        return text.replace('.', dec) if dec != '.' else text

    comparison_metadata = report.get('light_curve_comparison') or {}
    include_shape = bool(comparison_metadata) or any(
        (experiment.get('light_curve_comparison') or {}).get(
            'shape_weight') is not None
        for experiment in report['experiments'])
    columns = [('model', 'Model'),
               ('initial', 'First-observed (kg)'),
               ('ablated', 'Ablated (kg)'),
               ('landed', 'Landed (kg)'),
               ('unresolved', 'Mass with unresolved fate (kg)'),
               ('largest', 'Largest landed piece (kg)'),
               ('peak_mag', 'Peak absolute magnitude'),
               ('density', 'Density (kg/m³)')]
    if include_shape:
        columns.extend([('shape', 'Shape weight'),
                        ('peak_delay', 'Peak delay (s)'),
                        ('post_light', 'Post-observed light (s)'),
                        ('preference', 'Model preference')])
    header_parts = []
    for key, title in columns:
        label = html.escape(tr(f'df_frag_h_{key}', title))
        content = (label if key == 'model' else
                   '<span style="writing-mode:vertical-rl;'
                   'transform:rotate(180deg);display:inline-block;'
                   'white-space:normal;line-height:1.05;'
                   'inline-size:120px">'
                   + label + '</span>')
        header_parts.append(
            '<th style="font-size:.85em;cursor:help;'
            'vertical-align:bottom;min-height:120px;'
            'min-width:1.8em;padding:2px 1px;overflow:visible" title="'
            + html.escape(tr(f'df_frag_help_{key}', title), quote=True)
            + '">' + content + '</th>')
    header_cells = ''.join(header_parts)
    rows = []
    ranked_experiments = sorted(
        enumerate(report['experiments']),
        key=lambda item: (
            (item[1].get('light_curve_comparison') or {}).get(
                'preference_rank') is None,
            (item[1].get('light_curve_comparison') or {}).get(
                'preference_rank', float('inf')),
            (item[1].get('light_curve_comparison') or {}).get(
                'shape_rank') is None,
            (item[1].get('light_curve_comparison') or {}).get(
                'shape_rank', float('inf')),
            item[0]))
    for _, experiment in ranked_experiments:
        budget = experiment['mass_budget']
        parameters = experiment.get('parameters', {})
        model = tr(f'df_frag_model_{experiment["model"]}', experiment['model'])
        outcome_mass = max(float(budget.get('landed_kg') or 0.),
                           float(budget.get('unresolved_kg') or 0.))
        outcome_scale = (max(0, -int(math.floor(math.log10(outcome_mass))))
                         if outcome_mass > 0 else 0)
        values = [model] + [mass_number(budget[k], outcome_scale) for k in
            ('initial_kg', 'ablated_kg', 'landed_kg', 'unresolved_kg')]
        values += [mass_number(experiment['largest_landed_kg'], outcome_scale),
                   number(experiment['peak_absolute_magnitude'], '.1f'),
                   number(parameters.get('rho_m'), '.0f')]
        if include_shape:
            comparison = experiment.get('light_curve_comparison') or {}
            weight = comparison.get('shape_weight')
            values.append('—' if weight is None else
                          f'{weight * 100.:.1f}%'.replace('.', translations.get('dec_sep', '.')))
            values += [number(comparison.get('peak_delay_s'), '.2f'),
                       number(comparison.get('post_observed_duration_s'), '.2f')]
            preference = comparison.get('model_preference_weight')
            values.append('—' if preference is None else
                          f'{preference * 100.:.1f}%'.replace(
                              '.', translations.get('dec_sep', '.')))
        row_attr = ''
        if (experiment.get('light_curve_comparison') or {}).get(
                'preference_rank') == 1:
            row_attr = (' class="df-frag-best" title="'
                        + html.escape(tr('df_frag_best_row',
                            'Highest terminal-dominated model preference'),
                                        quote=True) + '"')
        rows.append('<tr' + row_attr + '>' + ''.join(
            '<td>' + html.escape(v) + '</td>' for v in values) + '</tr>')
    warnings = [_fragmentation_warning_text(warning, translations)
                for experiment in report['experiments']
                for warning in experiment.get('warnings', [])]
    if comparison_metadata.get('available') is False:
        warnings.append(tr(
            'df_frag_light_unavailable',
            'Model comparison unavailable: no usable observed camera light curve.'))
    preference_info = ((report.get('light_curve_comparison') or {})
                       .get('model_preference') or {})
    if preference_info.get('available') is False:
        warnings.append(tr(
            'df_frag_pref_unavailable',
            'No reliable model preference: the automatic entry-mass estimate is '
            'poorly constrained. Supply --mag or --entry-mass for a modelled comparison.'))
    text = ('<div class="text-section">'
            + html.escape(tr('dark_flight', 'Dark flight'))
            + ' / ' + html.escape(tr('df_frag_section', 'fragmentation'))
            + '</div>'
            + ('<p>' + html.escape(tr('df_frag_no_table_impacts',
                'No modelled ground impacts in these scenarios.')) + '</p>'
               if not any((e['mass_budget'].get('landed_kg') or 0) > 0
                          for e in report['experiments']) else '')
            + '<style>.dftable td{text-align:center;padding:1px 3px}'
            + '.dftable tr.df-frag-best{background-color:#fff3bf;'
            + 'color:#202124;font-weight:600}body.theme-dark '
            + '.dftable tr.df-frag-best{background-color:#5f4b00;'
            + 'color:#ffe9a8}</style>'
            + '<div class="df-table-wrap" style="overflow-x:auto">'
            + '<table class="dftable" style="font-size:14px"><tr>'
            + header_cells
            + '</tr>' + ''.join(rows) + '</table></div>'
            + '''<script>(function(){var s=document.currentScript,w=s.previousElementSibling,t=w&&w.querySelector("table");if(!t)return;function f(){var size=14;t.style.fontSize="14px";while(t.scrollWidth>w.clientWidth&&size>6){size-=.25;t.style.fontSize=size+"px"}}requestAnimationFrame(f);if(window.ResizeObserver)new ResizeObserver(f).observe(w);else addEventListener("resize",f)})();</script>'''
            + ''.join('<p>' + html.escape(w) + '</p>' for w in warnings)
            + '<a href="darkflight.kml">KML</a> | <a href="darkflight.geojson">GeoJSON</a> | '
            '<a href="darkflight.json">JSON</a> | <a href="fragmentation_summary.csv">CSV</a>')
    Path(path).write_text(text, encoding='utf-8')


def _density_list(value):
    try:
        densities = [float(part.strip()) for part in str(value).split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'density must be a positive number or comma-separated list') from exc
    if (not densities or any(not math.isfinite(density) or density <= 0
                             for density in densities)):
        raise argparse.ArgumentTypeError(
            'density values must be positive finite kg/m3')
    return densities


def _density_prior_list(value):
    try:
        priors = [float(part.strip()) for part in str(value).split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'density priors must be nonnegative numbers') from exc
    if (not priors or any(not math.isfinite(prior) or prior < 0
                          for prior in priors)
            or not any(prior > 0 for prior in priors)):
        raise argparse.ArgumentTypeError(
            'density priors must be nonnegative and not all zero')
    return priors


def parse_args(argv=None, default_mc=300):
    ap = argparse.ArgumentParser(
        description='Dark-flight propagator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Forward fragmentation models:
  single      intact-object control; no fragmentation or erosion release
  discrete    equal-mass daughter cascade; daughter strength scales with mass
  disruption  one-shot release into a power-law grain spectrum
  erosion     continuous grain release; grains propagate independently
  all         run all four models

Mass inputs are modelled assumptions in kilograms. If --entry-mass is
omitted and --mag is supplied, an entry mass is solved independently for each
selected model to reproduce that absolute peak magnitude. If neither is
supplied, a reliable unsaturated calibrated camera peak is used when at least
two cameras agree; otherwise an entry mass is inferred from observed
deceleration versus altitude and back-projected to the first observed point.
--density-priors supplies relative prior weights aligned with --density;
they affect model preference only, not propagation or mass inference.''')
    if hasattr(ap, 'color'):
        ap.color = False
    ap.add_argument('event_dir')
    ap.add_argument('--mc', type=int, default=default_mc,
                    help='Monte Carlo dark-flight realizations per modelled scenario; 0 disables')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mag', type=float, default=None,
                    help='Peak visual magnitude normalised to 100 km '
                         '(absolute magnitude; more negative is brighter). '
                         'For forward models it solves '
                         'entry mass when --entry-mass is omitted; otherwise '
                         'it is only a comparison. Without --entry-mass or '
                         '--mag, reliable camera photometry or deceleration '
                         'supplies an estimated mass.')
    ap.add_argument('--density', type=_density_list, default=None,
                    help='Reference meteoroid density in kg/m3; accepts a '
                         'comma-separated list for forward models (e.g. '
                         '3500,7000). Sets the modelled material density; '
                         'the legacy dynamic mass grid still spans all densities.')
    ap.add_argument('--density-priors', type=_density_prior_list, default=None,
                    help='Optional nonnegative model-preference prior weights '
                         'aligned with --density values (e.g. 0.8,0.2); '
                         'normalized before use')
    ap.add_argument('--fragmentation-model', nargs='+',
                    choices=['single', 'discrete', 'disruption', 'erosion', 'all'],
                    help='Opt-in forward entry models; default remains legacy dark flight')
    ap.add_argument('--entry-mass', type=float, nargs='+', metavar='KG',
                    help='Assumed mass in kilograms (kg) at FIRST observed altitude; '
                         'optional with --mag or automatic deceleration inference, '
                         'never surviving mass')
    ap.add_argument('--fragmentation-output', type=Path,
                    help='Optional separate output directory for forward-model products; '
                         'omit to update the event\'s dark-flight products')
    ap.add_argument('--breakup-height', type=float,
                    help='Force first breakup/erosion onset at this altitude in km; otherwise use strength')
    ap.add_argument('--strength', type=float, default=1.,
                    help='Initial strength in MPa, pressure convention rho_air*v_relative^2')
    ap.add_argument('--fragment-count', type=int, default=8,
                    help='Equal daughters per discrete breakup')
    ap.add_argument('--strength-exponent', type=float, default=0.3,
                    help='Daughter strength scaling exponent S_child/S_parent=(m_parent/m_child)^alpha')
    ap.add_argument('--erosion-coeff', type=float, default=1e-7,
                    help='Erosion coefficient eta in s^2/m^2: dm/dt=-eta*S*rho_air*v^3/2')
    ap.add_argument('--grain-min', type=float, default=1e-6, help='Minimum grain mass in kg')
    ap.add_argument('--grain-max', type=float, default=1e-3, help='Maximum grain mass in kg')
    ap.add_argument('--grain-bins', type=int, default=6, help='Logarithmic grain mass bins')
    ap.add_argument('--mass-index', type=float, default=2., help='Grain number spectrum dN/dm proportional to m^-s')
    ap.add_argument('--ablation-scale', type=float, default=1., help='Multiplier on existing material ablation coefficient')
    ap.add_argument('--luminous-efficiency', type=float, default=0.03,
                    help='Constant fraction of dissipated energy emitted optically; forward models only')
    ap.add_argument('--entry-max-step', type=float, default=0.05, help='Maximum luminous integration step in seconds')
    ap.add_argument('--entry-record-dt', type=float, default=0.02, help='Light-curve exposure duration in seconds')
    ap.add_argument('--erosion-release-dt', type=float, default=0.005,
                    help='Maximum erosion grain-release packet interval in seconds; refine for convergence')
    ap.add_argument('--entry-max-evaluations', type=int, default=4000000,
                    help='Explicit per-model rate-evaluation work limit; increase for convergence studies')
    ap.add_argument('--max-generations', type=int, default=12, help='Discrete cascade generation cap (reported when reached)')
    ap.add_argument('--ground-height', type=float, help='Flat ground height in m for forward models; otherwise query elevation')
    args = ap.parse_args(argv)
    if not args.fragmentation_model and (args.entry_mass is not None or args.fragmentation_output is not None):
        ap.error('--entry-mass/--fragmentation-output require --fragmentation-model')
    if (args.density is not None and len(args.density) > 1
            and not args.fragmentation_model):
        ap.error('multiple --density values require --fragmentation-model')
    if args.density_priors is not None:
        expected = len(args.density) if args.density is not None else 1
        if not args.fragmentation_model:
            ap.error('--density-priors requires --fragmentation-model')
        if len(args.density_priors) != expected:
            ap.error('--density-priors must provide one weight per --density value')
    args.event_dir = Path(args.event_dir).resolve()
    if args.fragmentation_output is not None:
        args.fragmentation_output = args.fragmentation_output.resolve()
    return args


def run_from_args(args, analysis_results=None, lang_files=None):
    event_dir = Path(args.event_dir)
    if analysis_results is not None:
        resdat = analysis_results['resdat']
        results_fb = analysis_results.get('fbspd_results')
        plot_data = analysis_results.get('fbspd_plot_data')
    else:
        res_files = list(event_dir.glob('obs_*.res'))
        if not res_files:
            raise FileNotFoundError('No .res file in ' + str(event_dir))
        from fbspd_merge import readres
        resdat = readres(str(res_files[0]))
        import pickle
        pkl = event_dir / '_fbspd_plot_data.pkl'
        plot_data = results_fb = None
        if pkl.exists():
            with pkl.open('rb') as f:
                results_fb, plot_data = pickle.load(f)
    if args.fragmentation_model:
        return run_fragmentation_models(event_dir, resdat, plot_data,
            args.fragmentation_output, args.entry_mass, args.fragmentation_model,
            rho_m=args.density if args.density is not None else [3500.],
            peak_mag=args.mag, density_priors=args.density_priors,
            h_ground=args.ground_height,
            strength_pa=args.strength * 1e6,
            breakup_height_m=None if args.breakup_height is None else args.breakup_height * 1000.,
            fragment_count=args.fragment_count, strength_exponent=args.strength_exponent,
            erosion_coeff=args.erosion_coeff, grain_min=args.grain_min, grain_max=args.grain_max,
            grain_bins=args.grain_bins, mass_index=args.mass_index,
            ablation_scale=args.ablation_scale, luminous_efficiency=args.luminous_efficiency,
            max_step=args.entry_max_step, record_dt=args.entry_record_dt,
            max_generations=args.max_generations, erosion_release_dt=args.erosion_release_dt,
            max_evaluations=args.entry_max_evaluations, lang_files=lang_files,
            mc_runs=args.mc, seed=args.seed)
    wind_csv = event_dir / 'wind_profile.csv'
    if lang_files is None:
        lang_files = load_language_files()
    return run_darkflight(event_dir, resdat, results_fb, plot_data,
                   wind_csv if wind_csv.exists() else None,
                   mc_runs=args.mc, seed=args.seed,
                   lang_files=lang_files, peak_mag=args.mag,
                   rho_ref=(args.density[0] if args.density else None))


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    report = run_from_args(args)
    if args.fragmentation_model:
        for result in report['experiments']:
            budget = result['mass_budget']
            def cli_mass(value):
                return 'unresolved' if value is None else f'{value:.4g} kg'
            print(f'{result["model"]}@{result.get("density_kg_m3", 0):.0f}: '
                  f'initial={cli_mass(budget["initial_kg"])}, '
                  f'landed={cli_mass(budget["landed_kg"])}, '
                  f'unresolved={cli_mass(budget["unresolved_kg"])}, '
                  f'peak M={result["peak_absolute_magnitude"]}')


if __name__ == '__main__':
    main()
