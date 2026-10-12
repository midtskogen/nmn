#!/usr/bin/env python3
"""Sanity tests for darkflight.py physics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import math

import pytest

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
    est = estimate_mass(lambda t: v, lambda t: a, lambda t: h,
                        (0.0, 0.5), rho_grid=[3500], A=A, atm=atm)
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


def test_ablation_endpoint_is_exact():
    import darkflight as df
    r0 = llh2ecef(10.0, 60.0, 35000.0)
    v0 = df.enu2ecef_mat(60.0, 10.0) @ np.array([0.0, 0.0, -20000.0])
    res = propagate(r0, v0, 0.005, 1500.0, 1.4, WindAtmosphere(), 0.0,
                    record_dt=10.0)
    assert not res['impact']['landed']
    assert abs(res['impact']['m'] - 0.001) < 1e-8
    assert abs(res['impact']['m'] - res['m'][-1]) < 1e-10
    assert abs(res['impact']['h'] - res['h'][-1]) < 1e-5


def test_invalid_propagation_inputs():
    r0 = llh2ecef(10.0, 60.0, 1000.0)
    for m, rho, shape, dt in [(0, 3500, 1.4, 1), (1, -1, 1.4, 1),
                               (1, 3500, 0, 1), (1, 3500, 1.4, 0)]:
        try:
            propagate(r0, np.zeros(3), m, rho, shape, WindAtmosphere(),
                      0.0, record_dt=dt)
        except ValueError:
            continue
        raise AssertionError('invalid physical input accepted')


def test_json_nonfinite_values():
    import json
    import tempfile
    import darkflight as df
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'result.json'
        df.write_json({'a': float('inf'), 'b': [float('nan'), np.float64(-np.inf)],
                       'c': np.array([1., np.nan]), 'ok': np.bool_(True)}, path)
        result = json.loads(path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
        assert result == {'a': None, 'b': [None, None], 'c': [1., None], 'ok': True}


def test_mass_covariance_units():
    params = np.array([20., -10., 0., 0.])
    est = estimate_mass(lambda t: (20. - 10. * t) * 1000., lambda t: 10000.,
                        lambda t: 30000., (0., 1.), rho_grid=[3500],
                        params=params, pcov=np.zeros((4, 4)))[0]
    assert np.isclose(est['m_fade_lo'], est['m_fade_kg'], rtol=1e-10)
    assert np.isclose(est['m_fade_hi'], est['m_fade_kg'], rtol=1e-10)


def test_mc_zero_velocity_and_wind_uncertainty():
    import darkflight as df
    from unittest.mock import patch
    r0 = llh2ecef(10., 60., 1000.)
    captured = []
    def capture(*args, **kwargs):
        captured.append((args[1], kwargs.get('wind_offset_enu')))
        return {'impact': {'landed': True}}
    with patch.object(df, 'propagate', side_effect=capture):
        df.monte_carlo(r0, np.zeros(3), 1., 3500., 1.4, None, 0.,
                       n_runs=3, seed=42, wind_err=2.)
    assert all(np.all(np.isfinite(v)) for v, w in captured)
    assert all(w is not None and np.linalg.norm(w[:2]) > 0 for v, w in captured)
    assert all(w[2] == 0 for v, w in captured)


def test_fragmentation_mass_budget_and_empty_anchor():
    import darkflight as df
    assert df.fragmentation_spectrum(0.).size == 0
    for index in (1., 2.):
        fragments = df.fragmentation_spectrum(10., mass_index=index, severity=3.)
        assert np.all(np.isfinite(fragments)) and np.all(fragments > 0)
        assert 0 < fragments.sum() < 10.
    assert df.build_scenarios(0.) == []
    scenarios = df.build_scenarios(1., fragment_masses=[0.2, 0.3, 0.001], dyn_press=1e8)
    s5 = next(s for s in scenarios if s['name'] == 'S5_strength')
    assert [r['m'] for r in s5['runs']] == [0.2, 0.3]
    s4 = next(s for s in scenarios if s['name'] == 'S4_powerlaw')
    assert sum(r['m'] for r in s4['runs']) <= 1. + 1e-12


def test_mc_outline_contains_sparse_cloud_without_shapely():
    import darkflight as df
    from matplotlib.path import Path as PlotPath
    from unittest.mock import patch
    for points in ([[0., 0.]], [[0., 0.], [1., 0.]],
                   [[0., 0.], [1., 0.], [2., 1e-8]],
                   [[0., 0.], [3., 0.], [0., 1.], [0.1, 0.2]]):
        with patch.dict(sys.modules, {'shapely.geometry': None}):
            ring = df._mc_outline(points, pad_min=0.15)
        assert np.array_equal(ring[0], ring[-1])
        assert PlotPath(ring).contains_points(points).all()
        assert np.ptp(ring[:, 0]) < np.ptp(np.array(points)[:, 0]) + 2.


def test_exports_use_landed_mass():
    import darkflight as df
    import tempfile
    import json
    import xml.etree.ElementTree as ET
    run = {'m': 1., 'rho': 3500., 'lon': [10., 10.01], 'lat': [60., 60.01],
           'h': [1000., 0.], 'impact': {'lon': 10.01, 'lat': 60.01, 'h': 0.,
                                        'v': 30., 'm': 0.02, 'landed': True}}
    scenarios = [{'name': 'S1_hypothetical', 'results': [run], 'hypothetical': True}]
    with tempfile.TemporaryDirectory() as td:
        df.write_geojson(scenarios, Path(td) / 'df.geojson')
        props = json.loads((Path(td) / 'df.geojson').read_text())['features'][0]['properties']
        assert props['mass_kg'] == 0.02 and props['release_mass_kg'] == 1.
        assert props['hypothetical'] is True
        df.write_kml(scenarios, (10., 60., 1000.), Path(td) / 'df.kml')
        root = ET.parse(Path(td) / 'df.kml')
        ns = {'k': 'http://www.opengis.net/kml/2.2'}
        names = [p.find('k:name', ns).text for p in root.findall('.//k:Placemark', ns)
                 if p.find('k:Point', ns) is not None]
        assert names == ['20.0 g']
        assert all('not evidence' in p.find('k:description', ns).text
                   for p in root.findall('.//k:Placemark', ns))


def test_survival_fraction_decreases_with_speed():
    import darkflight as df
    lo_slow, hi_slow = df._survival_fraction(12000.)
    lo_fast, hi_fast = df._survival_fraction(60000.)
    assert lo_slow > lo_fast and hi_slow > hi_fast
    assert df._survival_fraction(15000.) == (0.05, 0.30)
    for v in (3000., 10000., 15000., 30000., 70000.):
        lo, hi = df._survival_fraction(v)
        assert 0 < lo <= hi <= 0.60


def test_peak_magnitude_anchors_estimate():
    import darkflight as df
    import photometry
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    class SerialPool:
        def map(self, fn, jobs):
            return list(map(fn, jobs))
    def fake_propagate(r0, v0, mass, rho, shape, atm, ground, **kwargs):
        return {'impact': {'lon': 10., 'lat': 60., 'h': 0., 'v': 40.,
                           'm': mass / 2, 'landed': True},
                'lon': np.array([10.]), 'lat': np.array([60.]),
                'h': np.array([0.]), 'v': np.array([40.]), 'm': np.array([mass / 2])}
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.], height=[50., 30.])
    plot = {'final_params': np.array([20., -0.1, 0., 0.]),
            'final_merged_data': {'reltime': np.linspace(0., 2., 20),
                                  'height': np.linspace(50., 30., 20)}}
    with tempfile.TemporaryDirectory() as td, \
            patch.object(df, 'ground_elevation', return_value=0.), \
            patch.object(df, 'write_map'), patch.object(df, 'write_map3d'), \
            patch.object(photometry, 'event_photometry', return_value=None), \
            patch.object(df, 'propagate', side_effect=fake_propagate), \
            patch.object(df, 'monte_carlo', return_value=[]) as mc:
        results = df.run_darkflight(td, resdat, fbspd_plot_data=plot, peak_mag=-16.,
                                    rho_ref=7000., mc_runs=3, pool=SerialPool())
        assert results['m_est_ref_kg'] is None
        assert results['mass_source'] is None and results['mass_status'] == 'unconstrained'
        anchor = results['mag_anchor']
        v_end = results['end_state']['v_end_ms']
        m_entry = df._mag_to_photometric_mass(-16., v_end, df.MAG_FLASH_T_EFF_S)
        f_lo, f_hi = df._survival_fraction(anchor['v_char_ms'])
        assert anchor['m_entry_est_kg'] == pytest.approx(m_entry)
        assert anchor['m_surv_lo_kg'] == pytest.approx(f_lo * m_entry)
        assert anchor['m_surv_hi_kg'] == pytest.approx(f_hi * m_entry)
        assert anchor['m_fade_est_kg'] == pytest.approx(
            math.sqrt(f_lo * f_hi) * m_entry)
        assert results['magnitude_input']['peak_mag'] == -16.
        assert results['mc']['attempted'] == 3 and results['mc']['n'] == 0
        assert results['mc']['skipped_reason'] is None
        assert results['mc']['anchor'] == 'magnitude'
        assert results['mc']['hypothetical'] is False
        assert results['mc']['mass_prior']['source'] == 'magnitude_anchor'
        assert mc.call_args.kwargs['mass_bounds'] is None
        assert mc.call_args.args[2] == pytest.approx(anchor['m_fade_est_kg'])
        assert mc.call_args.args[3] == 7000.
        assert [s['name'] for s in results['scenarios']] == ['S1_hypothetical']
        assert all(s['hypothetical'] for s in results['scenarios'])
        # anchor scales with luminosity: 1 mag -> factor 10^0.4
        for magnitude in (-22., -14.):
            other = df.run_darkflight(td, resdat, fbspd_plot_data=plot, peak_mag=magnitude,
                                      rho_ref=7000., mc_runs=3, pool=SerialPool())
            assert other['m_est_ref_kg'] is None
            assert other['mag_anchor']['m_entry_est_kg'] == pytest.approx(
                m_entry * 10.0 ** (-0.4 * (magnitude + 16.)))
            assert other['mc']['anchor'] == 'magnitude'
            assert other['scenarios'] == results['scenarios']
        # a non-negative peak magnitude is meaningless here (likely a
        # sign error like '--mag 16') and must be rejected
        for magnitude in (20., 0., -0.5):
            with pytest.raises(ValueError):
                df.run_darkflight(td, resdat, fbspd_plot_data=plot, peak_mag=magnitude,
                                  rho_ref=7000., mc_runs=3, pool=SerialPool())
        # no magnitude -> back to the hypothetical 5 g-5 kg prior
        other = df.run_darkflight(td, resdat, fbspd_plot_data=plot,
                                  rho_ref=7000., mc_runs=3, pool=SerialPool())
        assert other['mag_anchor'] is None
        assert other['mc']['anchor'] == 'hypothetical'
        assert other['mc']['hypothetical']
        assert other['mc']['mass_prior']['min_kg'] == 0.005
        assert mc.call_args.kwargs['mass_bounds'] == (0.005, 5.)
        estimates = [{'rho': 7000., 'A': 1.4, 'm_fade_kg': 0.3,
                      'm_fade_lo': 0.2, 'm_fade_hi': 0.4, 'm_crit_kg': 1.}]
        with patch.object(df, 'estimate_mass', return_value=estimates):
            constrained = df.run_darkflight(td, resdat, fbspd_plot_data=plot, peak_mag=-16.,
                                            rho_ref=7000., mc_runs=3, pool=SerialPool())
            assert constrained['mass_status'] == 'constrained'
            assert constrained['m_est_ref_kg'] == 0.3
            assert constrained['mass_source'] == 'dynamic_constant_mass_fit'
            assert mc.call_args.args[3] == 7000.
            assert constrained['mc']['attempted'] == 3
            assert not constrained['mc']['hypothetical']
            assert mc.call_args.kwargs['mass_bounds'] is None
            estimates[0]['m_fade_hi'] = np.inf
            marginal = df.run_darkflight(td, resdat, fbspd_plot_data=plot, peak_mag=-16.,
                                         rho_ref=7000., mc_runs=3, pool=SerialPool())
            assert marginal['reliability'] == 'marginal'
            assert marginal['m_est_ref_kg'] is None and marginal['mc']['attempted'] == 3
            # a weak fit keeps the magnitude anchor, not the 5 g-5 kg prior
            assert marginal['mc']['anchor'] == 'magnitude'
            assert not marginal['mc']['hypothetical']
        with patch.object(photometry, 'event_photometry', return_value={'m_phot_kg': 1000.}):
            calibrated = df.run_darkflight(td, resdat, fbspd_plot_data=plot,
                                           rho_ref=7000., mc_runs=3, pool=SerialPool())
            assert calibrated['photometric']['m_phot_kg'] == 1000.
            # the measured integral also anchors a surviving-mass estimate
            f_lo, f_hi = df._survival_fraction(
                calibrated['phot_anchor']['v_char_ms'])
            assert calibrated['phot_anchor']['m_surv_lo_kg'] == pytest.approx(
                f_lo * 1000.)
            assert calibrated['phot_anchor']['m_surv_hi_kg'] == pytest.approx(
                f_hi * 1000.)
            # but the MC fall area keeps the broad hypothetical prior
            assert calibrated['mc']['anchor'] == 'hypothetical'
            assert calibrated['m_est_ref_kg'] is None


def test_truncated_release_moves_with_time():
    import darkflight as df
    import photometry
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    class SerialPool:
        def map(self, fn, jobs):
            return []
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.], height=[50., 20.])
    plot = {'final_params': np.array([10., -4., 0., 0.]),
            'final_merged_data': {'reltime': np.linspace(0., 2., 20),
                                  'height': np.linspace(50., 20., 20)}}
    with tempfile.TemporaryDirectory() as td, \
            patch.object(df, 'ground_elevation', return_value=0.), \
            patch.object(df, 'write_map'), patch.object(df, 'write_map3d'), \
            patch.object(df, 'write_kml'), patch.object(df, 'write_geojson'), \
            patch.object(df, 'build_scenarios', return_value=[]), \
            patch.object(photometry, 'event_photometry', return_value=None):
        result = df.run_darkflight(td, resdat, fbspd_plot_data=plot, mc_runs=0, pool=SerialPool())
        plot['final_params'] = np.array([2., -0.1, 0., 0.])
        slow = df.run_darkflight(td, resdat, fbspd_plot_data=plot, mc_runs=0, pool=SerialPool())
        assert slow['end_state']['v_end_ms'] == 1800.
        assert slow['reliability'] == 'unreliable'
    assert result['end_state']['h_m'] > 20000.
    assert result['end_state']['v_end_ms'] == 3000.
    assert np.isclose(result['track_duration_s'], 1.75)


def test_subthreshold_mass_and_solver_failure():
    import darkflight as df
    from unittest.mock import patch
    from types import SimpleNamespace
    r0 = llh2ecef(10., 60., 1000.)
    result = propagate(r0, np.zeros(3), 0.0005, 3500., 1.4, WindAtmosphere(), 0.)
    assert result['impact']['termination'] == 'dust'
    assert result['impact']['m'] == 0.0005 and result['impact']['t'] == 0.
    with patch.object(df, 'solve_ivp', return_value=SimpleNamespace(success=False, message='test failure')):
        try:
            propagate(r0, np.zeros(3), 1., 3500., 1.4, WindAtmosphere(), 0.)
        except RuntimeError as error:
            assert 'test failure' in str(error)
        else:
            raise AssertionError('failed solve returned a prediction')


def test_mc_process_pool_reproducibility():
    import darkflight as df
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    r0 = llh2ecef(10., 60., 1000.)
    args = (r0, np.zeros(3), 1., 3500., 1.4, None, 0.)
    serial = df.monte_carlo(*args, n_runs=3, seed=17)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        parallel = list(df.monte_carlo(*args, n_runs=3, seed=17, pool=pool))
    assert serial == parallel
    assert all(i['termination'] == 'ground' for i in serial)


def test_rendered_map_geometry_offline():
    import darkflight as df
    import tempfile
    try:
        import cartopy.crs as ccrs
        import plotly.graph_objects as go
        from shapely.geometry import Polygon, Point
    except ImportError as error:
        from unittest import SkipTest
        raise SkipTest(f'Optional map dependency unavailable: {error}')
    from types import SimpleNamespace
    from unittest.mock import patch
    import xml.etree.ElementTree as ET
    clouds = [{'lon': 10., 'lat': 60.}, {'lon': 10.04, 'lat': 60.02},
              {'lon': 10.02, 'lat': 60.04}]
    run = {'m': 1., 'rho': 3500., 'lon': [10.1, 10.02], 'lat': [60.1, 60.02],
           'h': [30000., 0.], 'impact': {'lon': 10.02, 'lat': 60.02, 'h': 0.,
                                        'v': 30., 'm': 0.02, 'landed': True}}
    scenarios = [{'name': 'S0_intact', 'label': 'Intact', 'results': [run]}]
    def tiles(lons, lats, **kwargs):
        pts = ccrs.UTM(32).transform_points(ccrs.PlateCarree(),
                    np.array([lons[0], lons[1]]), np.array([lats[0], lats[1]]))
        return np.full((10, 10, 3), 240, dtype=np.uint8), [pts[0, 0], pts[1, 0], pts[0, 1], pts[1, 1]]
    helpers = SimpleNamespace(_fetch_kartverket_topo=tiles,
                              darken_blacks=lambda image, _: image,
                              _rotation_controls_html=lambda *args: '',
                              _wind_overlay_html=lambda *args: '')
    figures = []
    original = go.Figure.write_html
    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)
    with tempfile.TemporaryDirectory() as td, patch.dict(sys.modules, {'metrack': helpers}):
        df.write_map(scenarios, (10.1, 60.1, 30000.), clouds, Path(td) / 'map.svg')
        ET.parse(Path(td) / 'map.svg')
        with patch.object(helpers, '_fetch_kartverket_topo', return_value=None), \
                patch('cartopy.io.img_tiles.OSM', side_effect=RuntimeError('offline')):
            df.write_map(scenarios, (10.1, 60.1, 30000.), clouds, Path(td) / 'fallback.svg')
            ET.parse(Path(td) / 'fallback.svg')
        with patch.object(go.Figure, 'write_html', capture):
            assert df.write_map3d(scenarios, (10.1, 60.1, 30000.), clouds, Path(td) / 'map.html')
        df.write_kml(scenarios, (10.1, 60.1, 30000.), Path(td) / 'map.kml', clouds)
        root = ET.parse(Path(td) / 'map.kml')
        ns = {'k': 'http://www.opengis.net/kml/2.2'}
        coords = root.find('.//k:Polygon//k:coordinates', ns).text
        polygon = Polygon([tuple(map(float, p.split(',')[:2])) for p in coords.split()])
        assert all(polygon.covers(Point(c['lon'], c['lat'])) for c in clouds)
        hatches = root.findall('.//k:MultiGeometry/k:LineString/k:coordinates', ns)
        assert hatches and all(len(h.text.split()) >= 2 for h in hatches)
        mc_pts = root.findall('.//k:Placemark/k:MultiGeometry/k:Point', ns)
        assert len(mc_pts) == len(clouds)
        assert root.find('.//k:MultiGeometry/k:LineString/k:tessellate', ns) is not None
        assert root.find('.//k:Polygon//k:LinearRing/k:tessellate', ns) is not None
    traces = figures[0].data
    edge = next(t for t in traces if t.name == 'Fall area edge')
    mc = next(t for t in traces if t.name == 'Monte Carlo')
    polygon = Polygon(zip(edge.x, edge.y))
    assert all(polygon.covers(Point(x, y)) for x, y in zip(mc.x, mc.y))
    mesh = next(t for t in traces if t.type == 'mesh3d')
    assert mesh.hoverinfo == 'skip' and mc.hoverinfo == 'skip'
    x, y = np.asarray(mesh.x), np.asarray(mesh.y)
    for i, j, k in zip(mesh.i, mesh.j, mesh.k):
        assert abs((x[j] - x[i]) * (y[k] - y[i]) - (y[j] - y[i]) * (x[k] - x[i])) > 1e-12
    assert figures[0].layout.scene.aspectmode == 'data'


def test_unavailable_covariance_is_not_an_error_bar():
    for covariance in (None, np.eye(2), np.full((4, 4), np.nan), -np.eye(4)):
        est = estimate_mass(lambda t: 10000., lambda t: 1000., lambda t: 30000.,
                            (0., 1.), rho_grid=[3500], params=[10., -1., 0., 0.],
                            pcov=covariance)[0]
        assert est['m_fade_lo'] == 0. and np.isinf(est['m_fade_hi'])
        assert est['uncertainty_source'] == 'unavailable'


def test_geodesy_poles():
    import darkflight as df
    for lat in (-90., 90.):
        r = llh2ecef(0., lat, 1000.)
        lon2, lat2, h2 = ecef2llh(r)
        lon3, lat3, h3 = df._ecef2llh_nb(*r)
        assert abs(lat2 - lat) < 1e-9 and abs(h2 - 1000.) < 1e-6
        assert abs(np.degrees(lat3) - lat) < 1e-9 and abs(h3 - 1000.) < 1e-6


def test_invalid_fit_keeps_json_and_removes_stale_outputs():
    import darkflight as df
    import tempfile
    import json
    from types import SimpleNamespace
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.], height=[50., -20.])
    with tempfile.TemporaryDirectory() as td:
        table = Path(td) / 'darkflight_table.html'
        table.write_text('stale')
        result = df.run_darkflight(td, resdat, mc_runs=7)
        stored = json.loads((Path(td) / 'darkflight.json').read_text())
        assert not table.exists() and stored['reliability'] == 'unreliable'
        assert result['mc']['requested'] == 7 and result['mc']['attempted'] == 0
        assert result['mass_source'] is None and result['mag_anchor'] is None


def test_mc_rejects_empty_spectrum():
    import darkflight as df
    try:
        df.monte_carlo(llh2ecef(10., 60., 1000.), np.zeros(3), 1., 3500.,
                       1.4, None, 0., n_runs=1, mass_samples=[])
    except ValueError:
        pass
    else:
        raise AssertionError('Empty spectrum silently became a nominal body')


def test_report_does_not_label_ablation_as_impact():
    import ast
    import logging
    import re
    import tempfile
    source = Path(__file__).resolve().parents[1] / 'server' / 'fetch.py'
    tree = ast.parse(source.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('_fmt_mass', '_append_darkflight_table')]
    namespace = {'np': np, 'Path': Path, 'logging': logging, 're': re,
                 'load_translations': lambda lang: {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    assert namespace['_fmt_mass'](None) == 'unreliable'
    estimates = [{'rho': rho, 'm_entry_kg': mass, 'm_fade_kg': mass,
                  'm_fade_lo': None, 'm_fade_hi': None}
                 for rho, mass in ((3500., 999.), (7000., 2.))]
    result = {'end_state': {'lon': 10., 'lat': 60., 'h_m': 30000., 'v_end_ms': 3000.},
              'reliability': 'ok', 'entry_estimates': estimates,
              'density_ref_kg_m3': 7000.,
              'scenarios': [{'name': 'S0_intact', 'impacts': [{'landed': False}]}],
              'mc': {'impacts': []}}
    with tempfile.TemporaryDirectory() as td:
        namespace['_append_darkflight_table'](Path(td), result, ['en'], 'en')
        html = (Path(td) / 'darkflight_table.html').read_text()
        assert 'Predicted impact' not in html
        assert '~2 kg' in html and '999' not in html
        import json
        locales = ['en', 'nb', 'de', 'fi', 'cs', 'lv']
        translations = {lang: json.loads((source.parent / 'loc' / f'{lang}.json').read_text())
                        for lang in locales}
        namespace['load_translations'] = translations.get
        result.update(mass_status='unconstrained', m_est_ref_kg=None,
                      magnitude_input={'peak_mag': -16.5},
                      mag_anchor={'peak_mag': -16.5, 'm_fade_kg': 1094.},
                      photometric={'m_phot_kg': 10.})
        result['scenarios'][0]['impacts'] = [{'landed': True, 'm': 1000.,
                                              'lon': 10., 'lat': 60., 'v': 50.}]
        result['mc']['impacts'] = [{'lon': 10., 'lat': 60.}]
        for reliability in ('unreliable', 'marginal'):
            result['reliability'] = reliability
            namespace['_append_darkflight_table'](Path(td), result, locales, 'nb')
            for lang in locales:
                prefix = '' if lang == 'nb' else lang + '_'
                html = (Path(td) / f'{prefix}darkflight_table.html').read_text()
                t = translations[lang]
                assert t['df_mass_unknown'] in html
                assert t['df_peak_mag_note'] in html
                assert t['df_mass_unknown_note'] in html
                assert t['df_mass_kind_phot'] in html
                assert t['df_impact'] not in html
                if t['df_fall_area'] != t['df_fall_area_assumed']:
                    assert f'>{t["df_fall_area"]}:' not in html
                assert t['df_fall_area_assumed'] in html
                assert t['df_mc_assumed_note'] in html
                assert ('-16.5' if lang == 'en' else '-16,5') in html
                assert '1094' not in html and '1.09 t' not in html and '1,09 t' not in html
        # a populated magnitude anchor shows the estimate and replaces the
        # proxy row and the "not constrained" line
        result.update(mag_anchor={'peak_mag': -16.5, 'm_entry_est_kg': 1094.,
                                  'm_surv_lo_kg': 54.7, 'm_surv_hi_kg': 328.,
                                  'm_fade_est_kg': 134.})
        result['mc']['anchor'] = 'magnitude'
        result['mc']['hypothetical'] = False
        result['reliability'] = 'unreliable'
        namespace['_append_darkflight_table'](Path(td), result, locales, 'nb')
        for lang in locales:
            prefix = '' if lang == 'nb' else lang + '_'
            html = (Path(td) / f'{prefix}darkflight_table.html').read_text()
            t = translations[lang]
            assert t['df_entry_mass'] in html
            assert t['df_surviving_mass'] in html
            assert t['df_fall_area_mag'] in html
            assert t['df_mass_unknown'] not in html
            assert t['df_mass_kind_phot'] not in html
        en_html = (Path(td) / 'en_darkflight_table.html').read_text()
        assert '1.09 t' in en_html
        # with both anchors the report shows geometric-mean estimates
        result['phot_anchor'] = {'m_entry_est_kg': 10., 'm_surv_lo_kg': 0.5,
                                 'm_surv_hi_kg': 3., 'm_fade_est_kg': 1.22}
        namespace['_append_darkflight_table'](Path(td), result, locales, 'nb')
        en_html = (Path(td) / 'en_darkflight_table.html').read_text()
        assert '105 kg' in en_html       # sqrt(1094 kg * 10 kg)
        assert '12.8 kg' in en_html      # sqrt(134 kg * 1.22 kg)
        # without --mag the photometric integral still gives a surviving
        # estimate instead of the ablated-mass proxy
        result.pop('mag_anchor', None)
        result['mc'].pop('anchor', None)
        result['mc']['hypothetical'] = True
        namespace['_append_darkflight_table'](Path(td), result, locales, 'nb')
        for lang in locales:
            prefix = '' if lang == 'nb' else lang + '_'
            html = (Path(td) / f'{prefix}darkflight_table.html').read_text()
            t = translations[lang]
            assert t['df_surviving_mass'] in html
            assert t['df_mass_unknown'] not in html
            assert t['df_mass_kind_phot'] not in html
        result.pop('phot_anchor', None)
        # a lone anchor below the 5 g floor reports 0 g; combined with a
        # larger anchor the lower boundary is clamped to 5 g instead
        result['phot_anchor'] = {'m_entry_est_kg': 1e-4, 'm_surv_lo_kg': 1e-6,
                                 'm_surv_hi_kg': 1e-4, 'm_fade_est_kg': 1e-5}
        namespace['_append_darkflight_table'](Path(td), result, locales, 'nb')
        html = (Path(td) / 'en_darkflight_table.html').read_text()
        assert '0 g' in html
        result['mag_anchor'] = {'peak_mag': -16.5, 'm_entry_est_kg': 1094.,
                                'm_surv_lo_kg': 54.7, 'm_surv_hi_kg': 328.,
                                'm_fade_est_kg': 134.}
        namespace['_append_darkflight_table'](Path(td), result, locales, 'nb')
        html = (Path(td) / 'en_darkflight_table.html').read_text()
        # entry = sqrt(1094 kg * 0.005 kg floor); surviving applies the
        # geometric-mean survival fraction (0.1225 and 0.10 -> 0.1107)
        assert '259 g' in html
        result.pop('mag_anchor', None)
        result['mc']['impacts'] = []
        result['mc']['attempted'] = 5
        result['mc']['hypothetical'] = True
        result['mc'].pop('anchor', None)
        result.pop('phot_anchor', None)
        namespace['_append_darkflight_table'](Path(td), result, locales, 'nb')
        for lang in locales:
            prefix = '' if lang == 'nb' else lang + '_'
            html = (Path(td) / f'{prefix}darkflight_table.html').read_text()
            t = translations[lang]
            assert t['df_mc_no_impacts'] in html
            assert t['df_fall_area_assumed'] not in html


def test_conservative_mc_mass_prior():
    import darkflight as df
    class JobPool:
        def map(self, fn, jobs):
            return list(jobs)
    args = (llh2ecef(10., 60., 1000.), np.zeros(3), 0.01, 3500., 1.4, None, 0.)
    jobs = df.monte_carlo(*args, n_runs=3000, seed=3, pool=JobPool(),
                          mass_bounds=(0.005, 5.), mass_err=0.9)
    masses = np.array([job[2] for job in jobs])
    assert masses.min() >= 0.005 and masses.max() <= 5.
    assert np.isclose(np.median(masses), np.sqrt(0.005 * 5.), rtol=0.1)
    for lo, hi in ((0.005, 0.05), (0.05, 0.5), (0.5, 5.)):
        frac = ((masses >= lo) & (masses < hi)).mean()
        assert abs(frac - 1. / 3.) < 0.05
    for bounds in ((0., 1.), (1., 0.1), (1., np.inf)):
        try:
            df.monte_carlo(*args, n_runs=0, mass_bounds=bounds)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid mass prior accepted')


def test_mc_ring_trims_outliers():
    import darkflight as df
    from unittest import SkipTest
    try:
        from shapely.geometry import Polygon, Point
    except ImportError as error:
        raise SkipTest(f'shapely unavailable: {error}')
    rng = np.random.default_rng(0)
    impacts = [{'lon': float(10 + rng.normal(0, 0.01)),
                'lat': float(60 + rng.normal(0, 0.01))}
               for _ in range(95)]
    outliers = [{'lon': 11.5, 'lat': 60.}, {'lon': 8.5, 'lat': 60.5},
                {'lon': 10., 'lat': 61.5}, {'lon': 10.5, 'lat': 58.5},
                {'lon': 9., 'lat': 59.}]
    ring = df._mc_ring_ll(impacts + outliers)
    poly = Polygon(ring)
    inside = sum(poly.covers(Point(i['lon'], i['lat']))
                 for i in impacts + outliers)
    assert inside >= 85
    assert not any(poly.covers(Point(o['lon'], o['lat']))
                   for o in outliers)
    full = Polygon(df._mc_ring_ll(impacts + outliers, coverage=1.0))
    assert poly.area < 0.1 * full.area
    try:
        df._mc_ring_ll(impacts, coverage=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid ring coverage accepted')


def ecef2llh_from_imp(imp):
    return imp['lon'], imp['lat'], imp['h']


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    from unittest import SkipTest
    for t in tests:
        try:
            t()
        except SkipTest as error:
            print(f"SKIP {t.__name__}: {error}")
        else:
            print(f"PASS {t.__name__}")
