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
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# --- Constants (SI) -----------------------------------------------------------
MU_E = 3.986005e14            # Earth gravitational parameter [m3/s2]
OMEGA = 7.292115e-5           # Earth rotation rate [rad/s]
OMEGA_VEC = np.array([0.0, 0.0, OMEGA])
A_EQ = 6378137.0              # WGS-84 semi-major axis [m]
E2 = 6.69437999014e-3         # WGS-84 first eccentricity squared
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
class WindAtmosphere:
    """1-D vertical profile (Open-Meteo wind_profile.csv) over US76.

    CSV columns: Height_m, Temp_K, Pressure_Pa, WindSpeed_ms, WindDir_deg
    (direction wind blows *from*, meteorological convention).
    Above the profile top: US76 density, wind held at the top-layer value.
    Below the profile bottom: bottom-layer values.
    """

    def __init__(self, csv_path=None):
        self.has_profile = False
        self._rho_scale = 1.0
        if csv_path and Path(csv_path).exists():
            rows = []
            with open(csv_path) as f:
                for row in csv.reader(f):
                    if not row or row[0].startswith('#'):
                        continue
                    rows.append([float(x) for x in row[:5]])
            if len(rows) >= 4:
                d = np.asarray(sorted(rows, key=lambda r: r[0]))
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
                # blend US76 density to match the profile top
                _, _, rho_top_us76 = us76(self.h_max)
                if rho_top_us76 > 0:
                    self._rho_scale = self.rho_p[-1] / rho_top_us76
                self.has_profile = True

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
        # above profile (or no profile): US76, wind = last layer or zero
        T, _, rho = us76(h_m)
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


# --- Propagator ---------------------------------------------------------------
def propagate(r0_ecef, v0_ecef, m0, rho_m, A_shape, atm, h_ground,
              erode=True, record_dt=0.5):
    """Integrate a meteoroid to the ground.

    r0_ecef, v0_ecef: ECEF state [m], [m/s];  m0 [kg]; rho_m [kg/m3];
    A_shape: shape parameter (sphere 1.21 / cylinder 1.6 / brick 2.7);
    atm: WindAtmosphere; h_ground [m]; erode: enable ablation.

    Returns dict with time series (lon, lat, h, v, m) + impact dict.
    """
    c_ml = ablation_coeff(rho_m, A_shape) if erode else 0.0

    def dynamics(t, X):
        r, v, m = X[:3], X[3:6], X[6]
        lon, lat, h = ecef2llh(r)
        wind_enu, rho_a, T = atm.at(max(h, 0.0))
        w_ecef = enu2ecef_mat(lat, lon) @ wind_enu
        v_rel = v - w_ecef
        vmag = np.linalg.norm(v_rel)
        a_grav = -MU_E * r / np.linalg.norm(r) ** 3
        a_cor = -2.0 * np.cross(OMEGA_VEC, v)
        a_cf = -np.cross(OMEGA_VEC, np.cross(OMEGA_VEC, r))
        m_eff = max(m, 1e-6)
        if vmag > 1e-9:
            cd = dragcoeff(vmag, T, rho_a, A_shape)
            a_drag = -cd * A_shape * rho_a * vmag * v_rel \
                     / (2 * m_eff ** (1. / 3) * rho_m ** (2. / 3))
        else:
            a_drag = np.zeros(3)
        dm = -c_ml * A_shape * rho_a * vmag ** 3 * m_eff ** (2. / 3) \
             / (2 * rho_m ** (2. / 3)) if vmag > 1e-9 else 0.0
        return np.hstack([v, a_grav + a_cor + a_cf + a_drag, dm])

    def hit_ground(t, X):
        _, _, h = ecef2llh(X[:3])
        return h - h_ground
    hit_ground.terminal = True
    hit_ground.direction = -1

    def dust(t, X):
        return X[6] - 1e-3
    dust.terminal = True
    dust.direction = -1

    X0 = np.hstack([r0_ecef, v0_ecef, m0])
    t_eval = np.arange(0, 1800, record_dt)
    sol = solve_ivp(dynamics, (0, 1800), X0, method='DOP853',
                    dense_output=True, t_eval=t_eval,
                    events=[hit_ground, dust],
                    rtol=1e-7, atol=1e-9)

    t_end = sol.t_events[0][0] if sol.t_events[0].size else \
        (sol.t_events[1][0] if sol.t_events[1].size else sol.t[-1])
    X_end = sol.y_events[0][0] if sol.y_events[0].size else sol.y[:, -1]

    n = max(int(t_end / record_dt) + 2, 2)
    ts = np.linspace(0, t_end, n)
    Xs = sol.sol(ts)
    path = [ecef2llh(Xs[:, i]) for i in range(Xs.shape[1])]
    lon_arr = np.array([p[0] for p in path])
    lat_arr = np.array([p[1] for p in path])
    h_arr = np.array([p[2] for p in path])
    v_arr = np.linalg.norm(Xs[3:6], axis=0)

    lon_i, lat_i, h_i = ecef2llh(X_end[:3])
    return {
        't': ts, 'lon': lon_arr, 'lat': lat_arr, 'h': h_arr,
        'v': v_arr, 'm': Xs[6],
        'impact': {'lon': lon_i, 'lat': lat_i, 'h': h_i,
                   'v': float(np.linalg.norm(X_end[3:6])),
                   'm': float(X_end[6]), 't': float(t_end),
                   'landed': bool(sol.t_events[0].size)},
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
                draws = np.random.default_rng(rng.integers(1 << 30))
                for p in draws.multivariate_normal(params, pcov, size=200):
                    a_s = np.abs(expfunc_2ndder(ts, *p))
                    m13i = np.sum(C * a_s) / np.sum(C * C)
                    if m13i > 0:
                        m_samples.append((1.0 / m13i) ** 3)
            except Exception:
                pass
        if len(m_samples) > 10:
            # Clip at a hard meteoroid-plausibility ceiling but report the
            # range as unbounded (hi=inf) when the percentile saturates,
            # so the UI shows 'poorly constrained' instead of '10 t–10 t'.
            cl = np.clip(m_samples, 1e-6, 1e4)
            lo, hi = np.percentile(cl, [16, 84])
            if np.percentile(m_samples, 84) >= 1e4:
                hi = np.inf
        else:
            lo, hi = m_med * 0.5, m_med * 2.0

        # critical entry mass: ablation integral along track
        B = ablation_coeff(rho_m, A) * A / (2 * rho_m ** (2. / 3))
        rhos = np.array([atm.at(max(h, 0.0))[1] for h in hs])
        integral = np.trapz(rhos * vs ** 3, ts)
        m_crit = (B / 3 * integral) ** 3

        out.append({'rho': rho_m, 'A': A, 'm_fade_kg': float(m_med),
                    'm_fade_lo': float(lo), 'm_fade_hi': float(hi),
                    'm_crit_kg': float(m_crit)})
    return out


def entry_mass_estimate(m_fade, rho_m, A, v_of_t, h_of_t, t_range, atm=None):
    """Back-integrate ablation over the luminous track.

    dM^(1/3)/dτ = (B/3) rhoa v^3 backward in time, B = c_ml*A/(2 rho_m^(2/3))
    => M_entry^(1/3) = M_fade^(1/3) + (B/3) ∫ rhoa v^3 dt   (end -> start)
    """
    if atm is None:
        atm = WindAtmosphere()
    B = ablation_coeff(rho_m, A) * A / (2 * rho_m ** (2. / 3))
    ts = np.linspace(t_range[0], t_range[1], 400)
    vs = np.array([v_of_t(t) for t in ts])
    hs = np.array([h_of_t(t) for t in ts])
    rhos = np.array([atm.at(max(h, 0.0))[1] for h in hs])
    integral = np.trapz(rhos * vs ** 3, ts)
    m13 = m_fade ** (1. / 3) + (B / 3) * integral
    return m13 ** 3


def entry_mass_estimate(m_fade, rho_m, A, v_of_t, h_of_t, t_range, atm=None):
    """Back-integrate ablation over the luminous track.

    dM^(1/3)/dτ = (B/3) rhoa v^3 backward in time, B = c_ml*A/(2 rho_m^(2/3))
    => M_entry^(1/3) = M_fade^(1/3) + (B/3) ∫ rhoa v^3 dt   (end -> start)
    """
    if atm is None:
        atm = WindAtmosphere()
    B = ablation_coeff(rho_m, A) * A / (2 * rho_m ** (2. / 3))
    ts = np.linspace(t_range[0], t_range[1], 400)
    vs = np.array([v_of_t(t) for t in ts])
    hs = np.array([h_of_t(t) for t in ts])
    rhos = np.array([atm.at(max(h, 0.0))[1] for h in hs])
    integral = np.trapz(rhos * vs ** 3, ts)
    m13 = m_fade ** (1. / 3) + (B / 3) * integral
    return m13 ** 3


# --- Fragmentation scenarios ---------------------------------------------------
def build_scenarios(m_est, rho_m=3500.0, A=1.4, masses_grid=None):
    """Return list of scenario dicts:
    {'name', 'label', 'runs': [{'m':..,'rho':..,'A':..}], 'erode'}
    m_est: estimated surviving mass [kg] (reference density/shape).
    """
    if masses_grid is None:
        masses_grid = np.logspace(-3, np.log10(5.0), 16)  # 1 g .. 5 kg
    sc = []
    sc.append({'name': 'S0_intact', 'label': 'Single body (estimated mass)',
               'runs': [{'m': m_est, 'rho': rho_m, 'A': A}], 'erode': True})
    sc.append({'name': 'S1_fallline', 'label': 'Fall line (mass grid)',
               'runs': [{'m': m, 'rho': rho_m, 'A': A} for m in masses_grid],
               'erode': True})
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
    # power-law dN/dM ~ M^-2, fragments down to 1 g (inverse-CDF sampling)
    m_min = 0.001
    u = np.linspace(0.05, 0.95, 60)
    ms = 1.0 / (1.0 / m_min - u * (1.0 / m_min - 1.0 / m_est))
    sc.append({'name': 'S4_powerlaw', 'label': 'Power-law fragment swarm',
               'runs': [{'m': m, 'rho': rho_m, 'A': A} for m in ms],
               'erode': True})
    # no-erosion variants of the interesting scenarios
    for name in ('S0_intact', 'S3_equal_4'):
        base = next(s for s in sc if s['name'] == name)
        sc.append({**base, 'name': name + '_noerosion',
                   'label': base['label'] + ' (no ablation)', 'erode': False})
    # nothing smaller than 1 gram is shown — drop sub-gram runs and
    # scenarios left empty by the cut
    for s in sc:
        s['runs'] = [r for r in s['runs'] if r['m'] >= 1e-3]
    return [s for s in sc if s['runs']]


def sc_label(sc, t):
    """Localised scenario label; falls back to the built-in English label."""
    name = sc['name']
    import re as _re
    if name.endswith('_noerosion'):
        base = sc_label({**sc, 'name': name[:-10]}, t)
        return base + ' ' + t.get('df_sc_noabl', '(no ablation)')
    key = {'S0_intact': 'df_sc_intact', 'S1_fallline': 'df_sc_fallline',
           'S4_powerlaw': 'df_sc_powerlaw'}.get(name)
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
def _mc_worker(job):
    r0, v0, m, rho, A, csv_path, h_ground, erode = job
    atm = WindAtmosphere(csv_path)
    return propagate(r0, v0, m, rho, A, atm, h_ground, erode=erode,
                     record_dt=1.0)['impact']


def monte_carlo(r0, v0_vec, m0, rho_m, A, csv_path, h_ground, n_runs=300,
                pos_err=100.0, vel_frac_err=0.05, dir_err_deg=0.1,
                mass_err=0.3, rho_err=500.0, shape_err=0.15,
                wind_err=2.0, seed=0, pool=None):
    """Perturbed runs around nominal state. Returns list of impact dicts."""
    rng = np.random.default_rng(seed)
    jobs = []
    for _ in range(n_runs):
        r_p = r0 + rng.normal(0, pos_err, 3)
        v_p = v0_vec * rng.normal(1.0, vel_frac_err)
        # small angular perturbation of direction
        ortho = np.cross(v0_vec, [1, 0, 0])
        if np.linalg.norm(ortho) < 1e-9:
            ortho = np.cross(v0_vec, [0, 1, 0])
        ortho /= np.linalg.norm(ortho)
        d1 = rng.normal(0, np.radians(dir_err_deg)) * np.linalg.norm(v0_vec)
        d2 = rng.normal(0, np.radians(dir_err_deg)) * np.linalg.norm(v0_vec)
        ortho2 = np.cross(v0_vec, ortho); ortho2 /= np.linalg.norm(ortho2)
        v_p = v_p + d1 * ortho + d2 * ortho2
        m_p = max(1e-3, m0 * rng.uniform(1 - mass_err, 1 + mass_err))
        rho_p = max(500.0, rng.normal(rho_m, rho_err))
        A_p = float(np.clip(rng.normal(A, shape_err), 0.9, 3.0))
        jobs.append((r_p, v_p, m_p, rho_p, A_p, csv_path, h_ground, True))
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
        if isinstance(o, (np.floating, np.integer)):
            v = float(o)
            return v if np.isfinite(v) else None
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        if isinstance(o, np.ndarray):
            return [conv(x) for x in o.tolist()]
        raise TypeError
    Path(path).write_text(json.dumps(results, indent=1, default=conv),
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
                'properties': {'scenario': sc['name'], 'mass_kg': run['m'],
                               'impact_speed_ms': imp['v']}})
    Path(path).write_text(json.dumps(
        {'type': 'FeatureCollection', 'features': feats}), encoding='utf-8')


def write_kml(scenarios, end_llh, path):
    parts = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             f'<name>Dark flight</name>',
             '<Folder><name>Fall paths</name>']
    colours = ['ff1400ff', 'ff00a5ff', 'ff00ff00', 'ffaa00ff', 'ffffaa00']
    for si, sc in enumerate(scenarios):
        col = colours[si % len(colours)]
        for run in sc['results']:
            path_pts = ' '.join(
                f'{lo:.6f},{la:.6f},{hh:.0f}'
                for lo, la, hh in zip(run['lon'], run['lat'], run['h']))
            parts.append(
                f'<Placemark><name>{sc["name"]} {run["m"]*1000:.1f}g</name>'
                f'<Style><LineStyle><color>{col}</color><width>1.5</width></LineStyle>'
                f'</Style><LineString><altitudeMode>absolute</altitudeMode>'
                f'<coordinates>{path_pts}</coordinates></LineString></Placemark>')
            imp = run['impact']
            if not imp.get('landed', True):
                continue
            parts.append(
                f'<Placemark><name>{sc["name"]} impact {run["m"]*1000:.1f}g</name>'
                f'<Point><coordinates>{imp["lon"]:.6f},{imp["lat"]:.6f},'
                f'{imp["h"]:.0f}</coordinates></Point></Placemark>')
    parts.append('</Folder>')
    parts.append('</Document></kml>')
    Path(path).write_text(''.join(parts), encoding='utf-8')


def write_map(scenarios, end_llh, mc_impacts, out_svg, title='',
              translations=None):
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
                m_kg = run['m']
                m_txt = (f'{m_kg:.3g} kg' if m_kg >= 1
                         else f'{m_kg * 1000:.3g} g')
                dec = translations.get('dec_sep', '.')
                if dec != '.':
                    m_txt = m_txt.replace('.', dec)
                labels2draw.append((run['impact']['lon'],
                                    run['impact']['lat'], m_txt, col))
    trplot([end_llh[0]], [end_llh[1]], 'r*', ms=14)

    # collision-aware placement: measure real text bboxes with the
    # renderer, try candidate offsets in order, first fit wins
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed_boxes = []
    # reserve space around every impact marker — a label must not cover
    # its own or any other point
    from matplotlib.transforms import Bbox
    to_disp = pc._as_mpl_transform(ax) if pc else ax.transData
    for lon, lat, txt, col in labels2draw:
        px, py = to_disp.transform((lon, lat))
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
            a = ax.annotate(txt, xy=(lon, lat), xytext=(dx, dy),
                            textcoords='offset points', ha=ha,
                            fontsize=6.5, color=col,
                            arrowprops=dict(arrow_kw, color=col),
                            transform=pc if pc else ax.transData)
            bb = a.get_window_extent(renderer).expanded(1.05, 1.2)
            if any(bb.overlaps(b) for b in placed_boxes):
                a.remove()
                continue
            placed_boxes.append(bb)
            break
    if mc_impacts:
        kw = {'transform': pc} if pc else {}
        ax.scatter([i['lon'] for i in mc_impacts],
                   [i['lat'] for i in mc_impacts], s=2, c='magenta',
                   alpha=0.4, **kw)
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
        handles.append(Line2D([], [], marker='.', ls='', color='magenta',
                              ms=7))
        labels.append('Monte Carlo')
    ax.legend(handles, labels, fontsize=7, loc='upper center',
              bbox_to_anchor=(0.5, -0.04), ncol=2, framealpha=0.9)
    if not pc:
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    if title:
        ax.set_title(title)
    plt.savefig(out_svg, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def write_map3d(scenarios, end_llh, mc_impacts, out_html,
                translations=None):
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
        if sc['results']:
            imp = sc['results'][0]['impact']
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
        uniq, seen_m = [], set()
        for run in sc['results']:
            if run['m'] not in seen_m:
                seen_m.add(run['m'])
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
                m_kg = run['m']
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
                    textfont=dict(size=9, color=col),
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
            marker=dict(size=3, color='magenta', opacity=0.45),
            name='Monte Carlo'))

    scene_dx = (x_max_m - x_min_m) / 1000.0
    scene_dy = (y_max_m - y_min_m) / 1000.0
    half = max(scene_dx, scene_dy) / 2.0 or 1.0
    cx = (x_min_m + x_max_m) / 2000.0
    cy = (y_min_m + y_max_m) / 2000.0
    center = dict(x=float((ex[0] - cx) / half),
                  y=float((ey[0] - cy) / half), z=0.0)
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


def run_darkflight(event_dir, resdat, fbspd_results=None, fbspd_plot_data=None,
                   wind_csv=None, mc_runs=300, seed=0, pool=None,
                   verbose=False, lang_files=None):
    """Top-level: compute dark flight for an event directory.

    resdat: ResData (from fbspd_merge.readres) with track start/end.
    fbspd_plot_data: needs 'final_params' + merged reltime for end speed.
    Returns results dict (also written to darkflight.json).
    """
    event_dir = Path(event_dir)
    end_lon = float(resdat.long1[1]); end_lat = float(resdat.lat1[1])
    end_h = float(resdat.height[1]) * 1000.0   # km -> m
    start_lon = float(resdat.long1[0]); start_lat = float(resdat.lat1[0])
    start_h = float(resdat.height[0]) * 1000.0

    # Track direction at end point (straight-line assumption)
    r_start = llh2ecef(start_lon, start_lat, start_h)
    r_end = llh2ecef(end_lon, end_lat, end_h)
    track_dir = r_end - r_start
    track_dir /= np.linalg.norm(track_dir)

    # End speed & deceleration from fbspd fit
    have_fit = (fbspd_plot_data is not None
                and fbspd_plot_data.get('final_params') is not None)
    pcov = None
    if have_fit:
        from fbspd_merge import expfunc_1stder, expfunc_2ndder
        params = fbspd_plot_data['final_params']
        pcov = fbspd_plot_data.get('pcov')
        ts_obs = fbspd_plot_data['final_merged_data']['reltime']
        hs_obs = fbspd_plot_data['final_merged_data']['height']
        t0_obs = float(np.min(ts_obs)); t_last = float(np.max(ts_obs))
        n_obs = len(ts_obs)

        def v_of_t(t):
            return float(expfunc_1stder(t, *params)) * 1000.0   # km/s -> m/s

        def a_of_t(t):
            return float(abs(expfunc_2ndder(t, *params))) * 1000.0

        hfit = np.polyfit(ts_obs, hs_obs, 1) if len(ts_obs) > 2 else None

        def h_of_t(t):
            if hfit is not None:
                return float(np.polyval(hfit, t)) * 1000.0
            return end_h

        v_end = v_of_t(t_last)
        a_end = a_of_t(t_last)
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
            'm_est_ref_kg': 0.0,
            'reliability': 'unreliable', 'issues': issues_pre,
            'n_obs': int(n_obs if have_fit else 0),
            'track_duration_s': float(t_last - t0_obs),
            'scenarios': [], 'mc': {'n': 0, 'impacts': []},
        }
        write_json(results, event_dir / 'darkflight.json')
        _clean_outputs(event_dir)
        return results

    atm = WindAtmosphere(wind_csv)

    # Surviving-mass estimates over the density grid (whole-track fit)
    mass_estimates = []
    if have_fit:
        mass_estimates = estimate_mass(v_of_t, a_of_t, h_of_t,
                                       (t0_obs, t_last),
                                       params=params, pcov=pcov)
    ref = next((e for e in mass_estimates if e['rho'] == 3500), None)

    # --- Reliability assessment -------------------------------------------------
    # A meteor that fades while still fast (luminous regime ends ~3-4 km/s)
    # must have a small surviving mass: it disappeared because it fully
    # ablated, not because it slowed below the detection threshold.
    # Deceleration-inferred mass >> critical ablation mass therefore flags
    # an inconsistent/unreliable input (usually a bad trajectory or a
    # poorly constrained deceleration fit).
    issues = []
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
        elif np.isfinite(m_hi) and m_hi > 0:
            if m_hi / max(m_lo, 1e-12) > 20:
                issues.append('deceleration poorly constrained')
            if v_end > 4000.0 and m_med > 30 * max(m_crit, 1e-9):
                issues.append('inconsistent with fade-out')
            if m_med > 100.0:
                issues.append('implausibly large surviving mass')
        else:
            issues.append('deceleration poorly constrained')
    else:
        m_med = m_crit = 0.0
        issues.append('no speed/deceleration fit')
    reliability = 'unreliable' if any(
        i in ('inconsistent with fade-out', 'implausibly large surviving mass',
              'no speed/deceleration fit',
              'deceleration non-invertible') for i in issues) else \
        ('marginal' if issues else 'ok')
    if reliability != 'ok':
        logging.warning(f'darkflight: mass estimate {reliability} '
                        f'({"; ".join(issues)})')

    if 'deceleration non-invertible' in issues:
        # zero/negative fitted deceleration means the meteor was still
        # flying ballistically at fade-out — dark-flight propagation is
        # meaningless, write a stub and skip all maps/scenarios
        logging.info('darkflight: deceleration ~0 — skipping simulation')
        results = {
            'end_state': {'lon': end_lon, 'lat': end_lat, 'h_m': end_h,
                          'v_end_ms': v_end, 'a_end_ms2': a_end},
            'mass_estimates': [], 'entry_estimates': [],
            'm_est_ref_kg': None,
            'photometric': None,
            'reliability': 'unreliable', 'issues': issues,
            'n_obs': int(n_obs), 'track_duration_s': float(duration),
            'scenarios': [], 'mc': {'n': 0, 'impacts': []},
        }
        write_json(results, event_dir / 'darkflight.json')
        _clean_outputs(event_dir)
        return results

    # --- Photometric cross-check (star-calibrated light curve) ------------------
    photometry = None
    try:
        import photometry as ph
        photometry = ph.event_photometry(event_dir, resdat=resdat,
                                         plot_data=fbspd_plot_data)
    except Exception as e:
        logging.debug(f'darkflight: photometry failed: {e}')

    m_phot = photometry.get('m_phot_kg') if photometry else None
    if m_phot and ref is not None and np.isfinite(m_med) and m_med > 0:
        if m_med / m_phot > 50:
            issues.append(
                f'dynamic mass {m_med:.3g} kg >> photometric '
                f'{m_phot:.3g} kg — likely bad track/fit')
            if reliability == 'ok':
                reliability = 'unreliable'

    # Nominal scenario mass: photometric or fade-consistent mass when the
    # deceleration fit is unreliable, else the deceleration-inferred estimate
    if reliability == 'unreliable':
        m_est = float(np.clip(m_phot or (m_crit if m_crit > 0 else 1.0),
                              1e-4, 100.0))
    else:
        m_est = float(np.clip(m_med if ref else 1.0, 1e-4, 100.0))

    # Entry-mass back-projection (needs height vs time along track)
    entry_estimates = []
    if have_fit:
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
    scenarios = build_scenarios(m_est)
    for sc in scenarios:
        sc['results'] = []
        for run in sc['runs']:
            res = propagate(r_end, v0_vec, run['m'], run['rho'], run['A'],
                            atm, h_ground, erode=sc['erode'])
            sc['results'].append({'m': run['m'], 'rho': run['rho'],
                                  'impact': res['impact'],
                                  'lon': res['lon'], 'lat': res['lat'],
                                  'h': res['h'], 'v': res['v'],
                                  'mass_t': res['m']})

    # Monte Carlo
    mc_impacts = []
    if mc_runs > 0:
        mc_impacts = [i for i in monte_carlo(
            r_end, v0_vec, m_est, 3500.0, 1.4, wind_csv, h_ground,
            n_runs=mc_runs, seed=seed, pool=pool)
            if i.get('landed', True)]

    results = {
        'event_dir': str(event_dir),
        'end_state': {'lon': end_lon, 'lat': end_lat, 'h_m': end_h,
                      'v_end_ms': v_end, 'a_end_ms2': a_end},
        'wind_used': atm.has_profile,
        'h_ground_m': h_ground,
        'mass_estimates': mass_estimates,
        'entry_estimates': entry_estimates,
        'm_est_ref_kg': m_est,
        'reliability': reliability,
        'issues': issues,
        'n_obs': int(n_obs), 'track_duration_s': float(duration),
        'scenarios': [{'name': s['name'], 'label': s['label'],
                       'impacts': [{'m': r['m'], **r['impact']}
                                   for r in s['results']]}
                      for s in scenarios],
        'mc': {'n': len(mc_impacts), 'impacts': mc_impacts},
    }
    write_json(results, event_dir / 'darkflight.json')
    write_geojson(scenarios, event_dir / 'darkflight.geojson')
    write_kml(scenarios, (end_lon, end_lat, end_h),
              event_dir / 'darkflight.kml')
    for file_prefix, tr in (lang_files or {'': {}}).items():
        try:
            write_map(scenarios, (end_lon, end_lat, end_h), mc_impacts,
                      event_dir / f'{file_prefix}map_darkflight.svg',
                      translations=tr)
        except Exception as e:
            logging.warning(f'darkflight map failed: {e}')
        try:
            write_map3d(scenarios, (end_lon, end_lat, end_h), mc_impacts,
                        event_dir / f'{file_prefix}darkflight_map3d.html',
                        translations=tr)
        except Exception as e:
            logging.warning(f'darkflight 3d map failed: {e}')
    return results


def main():
    ap = argparse.ArgumentParser(description='Dark-flight propagator')
    ap.add_argument('event_dir')
    ap.add_argument('--mc', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
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
                   lang_files=lang_files)


if __name__ == '__main__':
    main()
