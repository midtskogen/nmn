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
    (Sansom 2019 / DFN)."""
    cd_hyp = cd_hypersonic(A)
    if rho_m > 5000:
        return 0.07e-6 * cd_hyp
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
def _dragcoeff_nb(vel, temp, rho_a, A):
    if vel < 1e-6:
        vel = 1e-6
    mu_a = 18.27e-6 * (291.15 + 120.0) / (temp + 120.0) \
        * (temp / 291.15) ** 1.5
    sos = 331.3 * math.sqrt(temp / 273.15)
    mach = vel / sos
    re = rho_a * vel * 0.1 / mu_a
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
              erode=True, record_dt=0.5, wind_offset_enu=None):
    """Integrate a meteoroid to the ground.

    r0_ecef, v0_ecef: ECEF state [m], [m/s];  m0 [kg]; rho_m [kg/m3];
    A_shape: shape parameter (sphere 1.21 / cylinder 1.6 / brick 2.7);
    atm: WindAtmosphere; h_ground [m]; erode: enable ablation.

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
    if not np.isfinite(h_ground) or ecef2llh(r0_ecef)[2] <= h_ground:
        raise ValueError('Release position must be above finite ground height')
    c_ml = ablation_coeff(rho_m, A_shape) if erode else 0.0

    # pack the atmosphere once — the jitted dynamics sees only arrays
    if atm.has_profile:
        ph, pT, prho = atm.h, atm.T, atm.rho_p
        pwe, pwn = atm.we, atm.wn
    else:
        ph = pT = prho = pwe = pwn = np.zeros(1)
    if atm._msis is not None:
        mh, mrho, mT = [np.asarray(a, dtype=np.float64)
                        for a in atm._msis]
        has_msis = True
    else:
        mh = mrho = mT = np.zeros(1)
        has_msis = False
    dyn_args = (c_ml, A_shape, rho_m,
                atm.has_profile, ph, pT, prho, pwe, pwn,
                has_msis, mh, mrho, mT, atm._rho_scale, wind_offset)

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
        C = np.array([dragcoeff(max(v, 1.0), atm.at(max(h, 0.0))[2],
                                atm.at(max(h, 0.0))[1], A)
                      * A * atm.at(max(h, 0.0))[1] * v ** 2
                      / (2.0 * rho_m ** (2. / 3))
                      for v, h in zip(vs, hs)])
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
                    C_s = np.array([dragcoeff(v, atm.at(max(h, 0.0))[2],
                                              atm.at(max(h, 0.0))[1], A)
                                    * A * atm.at(max(h, 0.0))[1] * v ** 2
                                    / (2.0 * rho_m ** (2. / 3))
                                    for v, h in zip(v_s, hs)])
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
        rhos = np.array([atm.at(max(h, 0.0))[1] for h in hs])
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
    sc.append({'name': 'S0_intact', 'label': 'Single body (estimated mass)',
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


def sc_label(sc, t):
    """Localised scenario label; falls back to the built-in English label."""
    name = sc['name']
    import re as _re
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


def monte_carlo(r0, v0_vec, m0, rho_m, A, csv_path, h_ground, n_runs=300,
                pos_err=100.0, vel_frac_err=0.05, dir_err_deg=0.1,
                mass_err=0.3, rho_err=500.0, shape_err=0.15,
                wind_err=2.0, seed=0, pool=None,
                lon=None, lat=None, event_dt=None,
                mass_samples=None, mass_bounds=None):
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
    if pool is not None:
        return pool.map(_mc_worker, jobs)
    return [_mc_worker(j) for j in jobs]


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


def write_geojson(scenarios, path):
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
                               'release_mass_kg': run['m'],
                               'hypothetical': sc.get('hypothetical', False),
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
        description = ('<description>Hypothetical mass-grid trajectory; '
                       'not evidence of meteorite survival.</description>'
                       if sc.get('hypothetical', False) else '')
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
        area_name = ('Conditional fall area (MC)' if mc_assumed
                     else 'Fall area (MC)')
        area_desc = ('<description>Monte Carlo samples fragment masses '
                     'over 5 g–5 kg at the end of luminous flight; '
                     'not a measured surviving mass.</description>'
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
        # individual MC impact points — one placemark, terrain-clamped
        if mc_impacts:
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


def write_map(scenarios, end_llh, mc_impacts, out_svg, title='',
              translations=None, mc_assumed=False):
    """Fall-area map: Kartverket basemap, impact markers, MC ellipse."""
    translations = translations or {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lons_all, lats_all = [end_llh[0]], [end_llh[1]]
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
    trplot([end_llh[0]], [end_llh[1]], 'r*', ms=14)

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
    cmap_l = plt.cm.viridis
    for si, sc in enumerate(plot_sc):
        if sc['name'] == 'S1_fallline':
            continue
        col = cmap_l(si / max(len(plot_sc) - 1, 1))
        handles.append(Line2D([], [], marker='o', ls='', color=col, ms=7))
        labels.append(sc_label(sc, translations))
    handles.append(Line2D([], [], marker='*', ls='', color='red', ms=12))
    labels.append(translations.get('df_end_luminous', 'End of luminous path'))
    if mc_impacts:
        handles.append(Line2D([], [], marker='.', ls='', color='red',
                              ms=7))
        labels.append(translations.get(
            'df_mc_assumed' if mc_assumed else 'df_mc',
            'Monte Carlo (assumed masses)' if mc_assumed else 'Monte Carlo'))
    ax.legend(handles, labels, fontsize=7, loc='upper center',
              bbox_to_anchor=(0.5, -0.04), ncol=2, framealpha=0.9)
    if not pc:
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    if title:
        ax.set_title(title)
    plt.savefig(out_svg, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def write_map3d(scenarios, end_llh, mc_impacts, out_html,
                translations=None, mc_assumed=False):
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
    for sc in scenarios:
        for run in sc['results']:
            lons_all += list(run['lon']) + [run['impact']['lon']]
            lats_all += list(run['lat']) + [run['impact']['lat']]
    for imp in (mc_impacts or []):
        lons_all.append(imp['lon']); lats_all.append(imp['lat'])
    pad = 0.15
    lon_min, lon_max = min(lons_all) - pad, max(lons_all) + pad
    lat_min, lat_max = min(lats_all) - pad, max(lats_all) + pad

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
            step = max(1, len(run['lon']) // 300)
            lon_d = run['lon'][::step]; lat_d = run['lat'][::step]
            h_d = list(run['h'])[::step]
            x, y = project_points(lon_d, lat_d)
            traces.append(go.Scatter3d(
                x=x, y=y, z=[hh / 1000.0 for hh in h_d],
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
                    'Conditional fall area (MC)' if mc_assumed
                    else 'Fall area (MC)'),
                showscale=False, hoverinfo='skip'))
            traces.append(go.Scatter3d(
                x=list(sx), y=list(sy), z=[0.02] * len(sx),
                mode='lines', line=dict(color='red', width=3),
                name='Fall area edge', showlegend=False,
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
                    name='MC hatch', showlegend=False,
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
                                              'Height (km)')),
            aspectmode='data', dragmode='turntable',
            camera=dict(up=dict(x=0, y=0, z=1), center=center, eye=eye)),
        margin=dict(l=0, r=0, b=0, t=40)))
    out_html = Path(out_html)
    fig3.write_html(str(out_html), include_plotlyjs='cdn')
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
        wind_csv = out_html.parent / 'wind_profile.csv'
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
def _clean_outputs(event_dir):
    """Remove generated dark-flight artifacts (maps, tables, exports)."""
    for pat in ('map_darkflight.*', '*_map_darkflight.*',
                'darkflight_map3d.html', '*_darkflight_map3d.html',
                'darkflight.geojson', 'darkflight.kml',
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
            logging.info('darkflight: conditional MC samples fragment masses '
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
    # Numba cache=True means workers reuse the compiled code.
    jobs = [(si, run, r_end, v0_vec, atm, h_ground, sc['erode'])
            for si, sc in enumerate(scenarios)
            for run in sc['runs']]
    own_pool = pool is None and (len(jobs) > 4 or mc_runs > 0)
    if own_pool:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(
            max_workers=min(os.cpu_count() or 4, 16))
    try:
        if pool is not None:
            results_list = list(pool.map(_prop_worker, jobs))
        else:
            results_list = [_prop_worker(j) for j in jobs]
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
            ['conditional MC samples fragment masses log-uniformly over '
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
                        translations=tr, mc_assumed=m_est is None)
        except Exception as e:
            logging.warning(f'darkflight 3d map failed: {e}')
    return results


def main():
    ap = argparse.ArgumentParser(description='Dark-flight propagator')
    ap.add_argument('event_dir')
    ap.add_argument('--mc', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mag', type=float, default=None,
                    help='Peak visual magnitude normalised to 100 km '
                         '(absolute magnitude). Recorded as brightness only; '
                         'does not determine total or surviving mass.')
    ap.add_argument('--density', type=float, default=None,
                    help='Reference meteoroid density in kg/m3 '
                         '(e.g. 3500 chondrite, 7000 iron). Sets the '
                         'scenario material density; the dynamic mass '
                         'grid still spans '
                         'all densities.')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    from fbspd_merge import readres, calculate_speed_profile
    event_dir = Path(args.event_dir)
    res_files = list(event_dir.glob('obs_*.res'))
    if not res_files:
        sys.exit('No .res file in ' + str(event_dir))
    resdat = readres(str(res_files[0]))
    import pickle
    pkl = event_dir / '_fbspd_plot_data.pkl'
    plot_data = results_fb = None
    if pkl.exists():
        with pkl.open('rb') as f:
            results_fb, plot_data = pickle.load(f)
    wind_csv = event_dir / 'wind_profile.csv'
    # same per-language files as fetch.py produces
    lang_files = {'': {'dec_sep': '.'}}
    loc_dir = Path(__file__).resolve().parent.parent / 'server' / 'loc'
    for lang in ('en', 'cs', 'de', 'fi', 'lv'):
        lf = loc_dir / f'{lang}.json'
        if lf.exists():
            t = json.loads(lf.read_text(encoding='utf-8'))
            t['dec_sep'] = ',' if lang != 'en' else '.'
            lang_files[f'{lang}_'] = t
    if (loc_dir / 'nb.json').exists():
        t = json.loads((loc_dir / 'nb.json').read_text(encoding='utf-8'))
        t['dec_sep'] = ','
        lang_files[''] = t
    run_darkflight(event_dir, resdat, results_fb, plot_data,
                   wind_csv if wind_csv.exists() else None,
                   mc_runs=args.mc, seed=args.seed,
                   lang_files=lang_files, peak_mag=args.mag,
                   rho_ref=args.density)


if __name__ == '__main__':
    main()
