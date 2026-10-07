#!/usr/bin/env python3
"""Sanity tests for darkflight.py physics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from darkflight import (WindAtmosphere, propagate, estimate_mass,
                        ablation_coeff, dragcoeff, llh2ecef, ecef2llh,
                        us76)


def test_us76():
    T, P, rho = us76(0.0)
    assert abs(rho - 1.225) < 0.01
    T, P, rho = us76(20000.0)
    assert 0.08 < rho < 0.10


def test_geodesy_roundtrip():
    lon, lat, h = 10.5, 61.2, 25000.0
    r = llh2ecef(lon, lat, h)
    lon2, lat2, h2 = ecef2llh(r)
    assert abs(lon - lon2) < 1e-8 and abs(lat - lat2) < 1e-8
    assert abs(h - h2) < 1.0


def test_vertical_drop_terminal_velocity():
    """A 1 kg sphere-ish chondrite dropped from 30 km should reach the
    ground with a plausible terminal velocity (tens of m/s)."""
    atm = WindAtmosphere()
    r0 = llh2ecef(10.0, 60.0, 30000.0)
    v0 = np.zeros(3)  # released at rest relative to ground
    res = propagate(r0, v0, 1.0, 3500.0, 1.21, atm, 0.0, erode=False)
    imp = res['impact']
    assert 20.0 < imp['v'] < 120.0, f"impact v={imp['v']}"
    assert imp['t'] > 100.0          # several minutes of fall
    lon_i, lat_i, _ = ecef2llh_from_imp(imp)
    # zero wind: lands almost directly below release point
    assert abs(imp['lon'] - 10.0) < 0.02 and abs(imp['lat'] - 60.0) < 0.02


def test_mass_inversion():
    """Synthetic body: compute its deceleration, then recover mass."""
    atm = WindAtmosphere()
    m_true, rho_m, A = 0.7, 3500.0, 1.6
    h, v = 28000.0, 4500.0
    _, rho_a, T = atm.at(h)
    cd = dragcoeff(v, T, rho_a, A)
    a = cd * A * rho_a * v ** 2 / (2 * m_true ** (1. / 3) * rho_m ** (2. / 3))
    est = estimate_mass(v, a, h, rho_grid=[3500], A=A, atm=atm)
    assert abs(est[0]['m_fade_kg'] - m_true) / m_true < 0.10


def test_wind_drift():
    """Downwind drift: light fragments should drift more than heavy ones."""
    import csv, tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
        w = csv.writer(f)
        for h in np.arange(0, 31000, 1000):
            T, P, _ = us76(h)
            w.writerow([h, T, P, 20.0, 270.0])  # wind FROM west -> to east
        path = f.name
    atm = WindAtmosphere(path)
    r0 = llh2ecef(10.0, 60.0, 25000.0)
    v0 = np.array([0.0, 0.0, -4000.0])
    light = propagate(r0, v0, 0.01, 3500.0, 1.4, atm, 0.0)['impact']
    heavy = propagate(r0, v0, 2.0, 3500.0, 1.4, atm, 0.0)['impact']
    assert light['lon'] > 10.0 and heavy['lon'] > 10.0
    assert light['lon'] > heavy['lon'], \
        f"light {light['lon']} should drift further than heavy {heavy['lon']}"
    assert light['m'] < 0.011 and heavy['m'] < 2.01


def test_ablation_reduces_mass():
    atm = WindAtmosphere()
    r0 = llh2ecef(10.0, 60.0, 30000.0)
    v0 = np.array([500.0, 0.0, -3500.0])
    res_e = propagate(r0, v0, 0.5, 1500.0, 1.4, atm, 0.0, erode=True)
    res_n = propagate(r0, v0, 0.5, 1500.0, 1.4, atm, 0.0, erode=False)
    assert res_e['impact']['m'] <= res_n['impact']['m'] + 1e-9


def ecef2llh_from_imp(imp):
    return imp['lon'], imp['lat'], imp['h']


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
