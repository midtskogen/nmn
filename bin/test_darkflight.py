#!/usr/bin/env python3
"""Sanity tests for darkflight.py physics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import math
import os
import re

import pytest

import numpy as np
from darkflight import (WindAtmosphere, propagate, estimate_mass,
                        ablation_coeff, dragcoeff, llh2ecef, ecef2llh,
                        us76)


@pytest.fixture
def fetch_cli(tmp_path):
    import argparse
    import ast
    import atexit
    import datetime
    import errno
    import fcntl
    import logging
    import os
    import time
    import darkflight as df
    from types import SimpleNamespace
    from unittest.mock import Mock
    source = Path(__file__).resolve().parents[1] / 'server' / 'fetch.py'
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in
             ('main', '_meteorite_darkflight_options', '_run_darkflight',
              '_run_darkflight_only', 'process_event_with_lock')]
    namespace = dict(argparse=argparse, datetime=datetime, errno=errno, fcntl=fcntl,
                     logging=logging, os=os, time=time, sys=sys, Path=Path,
                     atexit=SimpleNamespace(register=Mock()), __file__=str(source),
                     open=lambda path, *a, **kw: open(tmp_path / 'startup.log', *a, **kw),
                     darkflight=df, DARKFLIGHT_AVAILABLE=True,
                     SUPPORTED_LANGS=['nb', 'en'], DEFAULT_LANG='nb',
                     load_translations=lambda lang: {},
                     Config=SimpleNamespace(SVG_DARKFLIGHT_DPI=100))
    for name in ('setup_logging', 'restore_terminal_state', 'process_event',
                 'svg_to_jpg', '_append_darkflight_table', 'fetch_data'):
        namespace[name] = Mock()
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def test_fetch_darkflight_only_forwards_arguments_without_reprocessing(fetch_cli, tmp_path, monkeypatch):
    from unittest.mock import Mock
    run = Mock()
    monkeypatch.setitem(fetch_cli, '_run_darkflight_only', run)
    marker = tmp_path / 'index.php'
    marker.write_text('preserve')
    monkeypatch.setattr(sys, 'argv', ['fetch.py', str(tmp_path), '--darkflight-only',
        '--darkflight-args', '--mag', '-18', '--density', '2500', '--mc', '17', '--seed', '42'])
    fetch_cli['main']()
    event, options = run.call_args.args
    assert event == tmp_path
    assert (options.mag, options.density, options.mc, options.seed) == (-18., [2500.], 17, 42)
    fetch_cli['process_event'].assert_not_called()
    fetch_cli['fetch_data'].assert_not_called()
    assert marker.read_text() == 'preserve'
    assert fetch_cli['setup_logging'].call_args.args[0].name == 'darkflight_only.log'


def test_meteorite_darkflight_defaults_preserve_explicit_choices(fetch_cli, tmp_path):
    import darkflight as df
    options = fetch_cli['_meteorite_darkflight_options'](
        tmp_path, None, fast=True)
    assert options.fragmentation_model == ['all']
    assert options.density == [3500., 7400.]
    assert options.mc == 0

    explicit = df.parse_args([
        str(tmp_path), '--fragmentation-model', 'single',
        '--density', '2500', '--mc', '7'], default_mc=0)
    merged = fetch_cli['_meteorite_darkflight_options'](
        tmp_path, explicit, fast=True)
    assert merged.fragmentation_model == ['single']
    assert merged.density == [2500.]
    assert merged.mc == 7
    assert explicit.density == [2500.]

    model_only = df.parse_args(
        [str(tmp_path), '--fragmentation-model', 'disruption'],
        default_mc=0)
    merged = fetch_cli['_meteorite_darkflight_options'](
        tmp_path, model_only, fast=True)
    assert merged.fragmentation_model == ['disruption']
    assert merged.density == [3500., 7400.]


def test_fetch_forward_options_use_shared_parser_and_absolute_paths(fetch_cli, tmp_path, monkeypatch):
    from unittest.mock import Mock
    run = Mock()
    monkeypatch.setitem(fetch_cli, '_run_darkflight_only', run)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['fetch.py', '.', '--darkflight-only', '--darkflight-args',
        '--fragmentation-model', 'all', '--entry-mass', '100', '300', '--breakup-height', '40',
        '--fragmentation-output', 'output with spaces', '--erosion-release-dt', '0.0025'])
    fetch_cli['main']()
    options = run.call_args.args[1]
    assert options.entry_mass == [100., 300.]
    assert options.fragmentation_model == ['all'] and options.breakup_height == 40.
    assert options.fragmentation_output == tmp_path / 'output with spaces'
    assert options.erosion_release_dt == 0.0025
    run.reset_mock()
    monkeypatch.setattr(sys, 'argv', ['fetch.py', '.', '--darkflight-only',
        '--darkflight-args', '--fragmentation-model', 'single', '--entry-mass', '300'])
    fetch_cli['main']()
    options = run.call_args.args[1]
    assert options.fragmentation_model == ['single']
    assert options.entry_mass == [300.] and options.fragmentation_output is None
    run.reset_mock()
    monkeypatch.setattr(sys, 'argv', ['fetch.py', '.', '--darkflight-only',
        '--darkflight-args', '--fragmentation-model', 'disruption', '--mag', '-18.74'])
    fetch_cli['main']()
    options = run.call_args.args[1]
    assert options.fragmentation_model == ['disruption']
    assert options.entry_mass is None and options.mag == -18.74


@pytest.mark.parametrize('extra', [
    ['--mag', '-18'], ['--darkflight-args', '--unknown-darkflight-option'],
    ['--darkflight-only', '--infrasound'], ['--darkflight-only', '--all'],
    ['--darkflight-only', '--no-refraction'],
    ['10109', '/meteor/cam1/events/20260423/021525', '--darkflight-only'],
])
def test_fetch_invalid_options_fail_before_processing(fetch_cli, tmp_path, monkeypatch, extra):
    monkeypatch.setattr(sys, 'argv', ['fetch.py', str(tmp_path), *extra])
    with pytest.raises(SystemExit) as exited:
        fetch_cli['main']()
    assert exited.value.code == 2
    fetch_cli['setup_logging'].assert_not_called()
    fetch_cli['process_event'].assert_not_called()
    fetch_cli['fetch_data'].assert_not_called()


def test_fetch_darkflight_only_missing_cache_returns_failure(fetch_cli, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['fetch.py', str(tmp_path), '--darkflight-only'])
    with pytest.raises(SystemExit) as exited:
        fetch_cli['main']()
    assert exited.value.code == 1
    fetch_cli['process_event'].assert_not_called()


def test_fetch_darkflight_only_uses_existing_lock_and_inputs(fetch_cli, tmp_path, monkeypatch):
    import fcntl
    from unittest.mock import Mock
    (tmp_path / 'obs_test.res').write_text('existing trajectory')
    (tmp_path / '_fbspd_plot_data.pkl').write_bytes(b'existing fit')
    run = Mock(return_value={'scenarios': []})
    monkeypatch.setitem(fetch_cli, '_run_darkflight', run)
    options = object()
    assert fetch_cli['_run_darkflight_only'](tmp_path, options) == {'scenarios': []}
    run.assert_called_once_with(tmp_path, options)
    with (tmp_path / '.process_event.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    (tmp_path / '.no_trajectory').touch()
    with pytest.raises(ValueError, match='no_trajectory'):
        fetch_cli['_run_darkflight_only'](tmp_path, options)


@pytest.mark.parametrize('forward', [False, True])
def test_fetch_darkflight_dispatch_preserves_fresh_analysis(fetch_cli, tmp_path, monkeypatch, forward):
    import darkflight as df
    from unittest.mock import Mock
    argv = [str(tmp_path), '--mag', '-17', '--mc', '7', '--seed', '3']
    if forward:
        argv += ['--fragmentation-model', 'single', '--entry-mass', '300',
                 '--fragmentation-output', str(tmp_path / 'experiment')]
    options = df.parse_args(argv)
    analysis = {'resdat': object(), 'fbspd_plot_data': None, 'fbspd_results': None}
    run = Mock(return_value={'scenarios': [{'name': 'test'}]})
    monkeypatch.setattr(df, 'run_from_args', run)
    (tmp_path / 'map_darkflight.svg').touch()
    fetch_cli['_run_darkflight'](tmp_path, options, analysis)
    assert run.call_args.kwargs['analysis_results'] is analysis
    assert run.call_args.args[0].mc == 7
    assert run.call_args.args[0].seed == 3
    if forward:
        fetch_cli['svg_to_jpg'].assert_not_called()
        fetch_cli['_append_darkflight_table'].assert_not_called()
    else:
        fetch_cli['svg_to_jpg'].assert_called_once()
        fetch_cli['_append_darkflight_table'].assert_called_once()
        assert run.call_args.kwargs['lang_files']['en_']['dec_sep'] == '.'
    assert run.call_args.kwargs['lang_files']['en_']['plotly_locale'] == 'en'


def test_fetch_forward_default_output_refreshes_event_jpgs(fetch_cli, tmp_path, monkeypatch):
    import darkflight as df
    from unittest.mock import Mock
    options = df.parse_args([str(tmp_path), '--fragmentation-model', 'single',
                             '--entry-mass', '300'])
    run = Mock(return_value={'analysis_mode': 'fragmentation', 'scenarios': []})
    monkeypatch.setattr(df, 'run_from_args', run)
    (tmp_path / 'map_darkflight.svg').touch()
    fetch_cli['_run_darkflight'](tmp_path, options)
    assert run.call_args.kwargs['lang_files']['en_']['dec_sep'] == '.'
    assert run.call_args.kwargs['lang_files']['en_']['plotly_locale'] == 'en'
    fetch_cli['svg_to_jpg'].assert_called_once_with(
        tmp_path / 'map_darkflight.svg', tmp_path / 'map_darkflight.jpg', 100)
    fetch_cli['_append_darkflight_table'].assert_not_called()


def test_fetch_full_reprocess_and_lock_forward_options(fetch_cli, tmp_path, monkeypatch):
    import datetime
    event = tmp_path / '20260423' / '021525'
    event.mkdir(parents=True)
    monkeypatch.setattr(sys, 'argv', ['fetch.py', str(event), '--fast', '--density', '3500',
                                    '--no-refraction',
                                    '--darkflight-args', '--mag', '-18', '--density', '2500'])
    with pytest.raises(SystemExit) as exited:
        fetch_cli['main']()
    assert exited.value.code == 0
    options = fetch_cli['process_event'].call_args.kwargs['darkflight_options']
    assert fetch_cli['process_event'].call_args.kwargs['refraction'] is False
    assert options.mc == 0 and options.mag == -18 and options.density == [2500.]
    fetch_cli['process_event'].reset_mock()
    fetch_cli['process_event_with_lock'](event, datetime.datetime(2026, 4, 23), darkflight_options=options)
    assert fetch_cli['process_event'].call_args.kwargs['darkflight_options'] is options


@pytest.mark.parametrize('forward', [False, True])
def test_shared_darkflight_dispatch_uses_fresh_analysis(tmp_path, monkeypatch, forward):
    import darkflight as df
    from unittest.mock import Mock
    (tmp_path / '_fbspd_plot_data.pkl').write_bytes(b'stale invalid cache must not be read')
    argv = [str(tmp_path), '--mc', '13', '--seed', '4', '--mag', '-18', '--density', '2500']
    if forward:
        argv += ['--fragmentation-model', 'erosion', '--entry-mass', '300',
                 '--fragmentation-output', str(tmp_path / 'out'), '--breakup-height', '40',
                 '--strength', '0.5', '--grain-bins', '12', '--entry-max-evaluations', '8000000']
    args = df.parse_args(argv, default_mc=0)
    resdat = object()
    analysis = {'resdat': resdat, 'fbspd_results': None, 'fbspd_plot_data': None}
    legacy, fragmentation = Mock(return_value={'legacy': True}), Mock(return_value={'forward': True})
    monkeypatch.setattr(df, 'run_darkflight', legacy)
    monkeypatch.setattr(df, 'run_fragmentation_models', fragmentation)
    translations = {'en_': {'dec_sep': '.'}}
    result = df.run_from_args(args, analysis_results=analysis, lang_files=translations)
    if forward:
        assert result == {'forward': True}
        legacy.assert_not_called()
        assert fragmentation.call_args.args[1] is resdat
        assert fragmentation.call_args.args[2] is None
        assert fragmentation.call_args.kwargs['breakup_height_m'] == 40000.
        assert fragmentation.call_args.kwargs['strength_pa'] == 500000.
        assert fragmentation.call_args.kwargs['grain_bins'] == 12
        assert fragmentation.call_args.kwargs['max_evaluations'] == 8000000
        assert fragmentation.call_args.kwargs['rho_m'] == [2500.]
        args = df.parse_args([
            str(tmp_path), '--fragmentation-model', 'single',
            '--entry-mass', '100', '--density', '3500,7000',
            '--density-priors', '0.8,0.2'])
        assert args.density == [3500., 7000.]
        assert args.density_priors == [.8, .2]
        with pytest.raises(SystemExit):
            df.parse_args([
                str(tmp_path), '--fragmentation-model', 'single',
                '--density', '3500', '--density-priors', '0.8,0.2'])
    else:
        assert result == {'legacy': True}
        fragmentation.assert_not_called()
        assert legacy.call_args.args[1] is resdat
        assert legacy.call_args.args[2:4] == (None, None)
        assert legacy.call_args.kwargs['lang_files'] is translations
        assert legacy.call_args.kwargs['mc_runs'] == 13
        assert legacy.call_args.kwargs['seed'] == 4
        assert legacy.call_args.kwargs['peak_mag'] == -18
        assert legacy.call_args.kwargs['rho_ref'] == 2500


def test_shared_darkflight_dispatch_loads_saved_inputs(tmp_path, monkeypatch):
    import darkflight as df
    import fbspd_merge
    import pickle
    from unittest.mock import Mock
    args = df.parse_args([str(tmp_path)], default_mc=0)
    with pytest.raises(FileNotFoundError, match='No .res'):
        df.run_from_args(args)
    (tmp_path / 'obs_test.res').write_text('saved trajectory')
    saved = ({'speed': 20.}, {'final_params': [20., 0., 0., 0.]})
    (tmp_path / '_fbspd_plot_data.pkl').write_bytes(pickle.dumps(saved))
    readres = Mock(return_value=object())
    run = Mock(return_value={'ok': True})
    monkeypatch.setattr(fbspd_merge, 'readres', readres)
    monkeypatch.setattr(df, 'run_darkflight', run)
    assert df.run_from_args(args, lang_files={}) == {'ok': True}
    assert run.call_args.args[1] is readres.return_value
    assert run.call_args.args[2:4] == saved
    assert run.call_args.kwargs['mc_runs'] == 0
    assert run.call_args.kwargs['lang_files'] == {}


def test_forward_saved_inputs_default_to_event_publication(tmp_path, monkeypatch):
    import darkflight as df
    import fbspd_merge
    import pickle
    from unittest.mock import Mock
    (tmp_path / 'obs_test.res').write_text('saved trajectory')
    saved = ({'speed': 20.}, {'final_params': [20., 0., 0., 0.]})
    (tmp_path / '_fbspd_plot_data.pkl').write_bytes(pickle.dumps(saved))
    args = df.parse_args([str(tmp_path), '--fragmentation-model', 'single',
                          '--entry-mass', '300'])
    readres = Mock(return_value=object())
    run = Mock(return_value={'ok': True})
    monkeypatch.setattr(fbspd_merge, 'readres', readres)
    monkeypatch.setattr(df, 'run_fragmentation_models', run)
    assert df.run_from_args(args, lang_files={'': {}}) == {'ok': True}
    assert run.call_args.args[0] == tmp_path
    assert run.call_args.args[1] is readres.return_value
    assert run.call_args.args[2] == saved[1]
    assert run.call_args.args[3] is None
    assert run.call_args.kwargs['lang_files'] == {'': {}}


def test_help_uses_terminal_default_colours(monkeypatch, capsys):
    import darkflight as df
    monkeypatch.setenv('FORCE_COLOR', '1')
    monkeypatch.setenv('PYTHON_COLORS', '1')
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setattr(sys, 'argv', ['darkflight.py', '--help'])
    with pytest.raises(SystemExit) as exited:
        df.main()
    assert exited.value.code == 0
    output = capsys.readouterr()
    assert 'Dark-flight propagator' in output.out
    assert '--fragmentation-model' in output.out
    assert '\x1b' not in output.out + output.err


@pytest.mark.parametrize('model', ['single', 'discrete', 'disruption', 'erosion'])
def test_forward_fragmentation_mass_conservation(model):
    import darkflight as df
    r = llh2ecef(10., 60., 10000.)
    v = df.enu2ecef_mat(60., 10.) @ np.array([0., 0., -5000.])
    result = df.propagate_fragmentation(r, v, 1., 3500., 1.4, WindAtmosphere(), 0.,
        model=model, breakup_height_m=9990., strength_pa=1e12,
        ablation_scale=0., max_step=0.1, record_dt=0.05, grain_bins=2)
    budget = result['mass_budget']
    assert budget['initial_kg'] == 1.
    assert budget['ablated_kg'] == pytest.approx(0., abs=1e-7)
    assert sum(budget[k] for k in ('ablated_kg', 'landed_kg', 'unresolved_kg', 'timeout_kg')) == pytest.approx(1., abs=1e-6)
    assert abs(budget['balance_error_kg']) < 1e-6
    assert all(budget[k] >= -1e-9 for k in ('ablated_kg', 'landed_kg', 'unresolved_kg', 'timeout_kg'))
    for trajectory in (fragment['trajectory'] for fragment in result['fragments']
                       if fragment.get('trajectory')):
        assert trajectory['h'][0] == pytest.approx(10000.)
        assert trajectory['h'][-1] == pytest.approx(0.)
        assert np.all(np.isfinite(trajectory['lon']))
    if model == 'disruption':
        assert budget['landed_kg'] == 0.
        assert budget['unresolved_kg'] == pytest.approx(1., abs=1e-6)
    if model == 'single':
        assert result['largest_landed_kg'] == pytest.approx(1.)
    if model == 'discrete':
        assert result['largest_landed_kg'] == pytest.approx(1. / 8.)


def test_forward_light_curve_shape_prefers_matching_model():
    import darkflight as df
    plot = {'final_params': [10., 0., 0., 0.],
            'final_merged_data': {'reltime': np.array([0., 2., 5., 8., 10.]),
                                  'color_time': np.array([0., 1., 4., 7., 9.])},
            'station_id_array': np.array([9, 3, 3, 3, 9]),
            'station_error_details': [{'source_index': 3, 'source_name': 'TST',
                                       'n_points': 3}]}
    photometry = {'per_cam': [{'cam': 'cam1', 'confidence': 'medium',
        'warnings': [], 'm_app': [-2.5, 0., -2.5],
        'brightness': [260., 220., 260.],
        'timestamps': [101., 104., 107.], 'ranges_km': [100., 100., 100.],
        'track_frac': [.2, .5, .8], 'video': None},
       {'cam': 'cam2', 'confidence': 'medium', 'warnings': [],
        'm_app': [-2.5, 0., -2.5], 'brightness': [100., 220., 100.],
        'timestamps': [2., 5., 8.], 'ranges_km': [100., 100., 100.],
        'track_frac': [.2, .5, .8],
        'video': {'m_app': [0., 0.], 'sat': [False, False]}},
       {'cam': 'cam3', 'confidence': 'none', 'warnings': [],
        'm_app': [None, None, None], 'brightness': [None, None, None],
        'timestamps': [101., 104., 107.], 'ranges_km': [100., 100., 100.],
        'track_frac': [.2, .5, .8],
        'video': {'flux': [10., 1., 10.], 'sat': [False, False, False]}}]}
    observed, reason = df._observed_light_curve_samples(photometry, plot)
    assert reason is None and len(observed) == 3
    assert observed[0]['t'].tolist() == [2., 5., 8.]
    assert observed[0]['time_source'] == 'station_alignment'
    assert observed[0]['source_index'] == 3
    assert observed[0]['coverage_fraction'] == pytest.approx(.6)
    assert observed[0]['weight'] == pytest.approx(.68)
    assert np.sum(observed[0]['sat']) == 2
    assert observed[1]['source'] == 'event_txt'
    assert observed[2]['source'] == 'video_flux'
    matching = {'model': 'single', 'light_curve':
              {'t_s': [0., 2., 5., 8., 10.],
               'power_w': [5., 10., 1., 10., 5.]}}
    mismatch = {'model': 'disruption', 'light_curve':
                {'t_s': [0., 2., 5., 8., 10.],
                 'power_w': [5., 1., 10., 1., 5.]}}
    short = {'model': 'erosion', 'light_curve':
             {'t_s': [0., 2., 5.], 'power_w': [5., 10., 1.]}}
    missing = {'model': 'discrete'}
    scores = df._light_curve_shape_scores(
        [matching, mismatch, short, missing], observed)
    assert scores[0]['shape_rank'] == 1
    assert scores[0]['shape_weight'] > .999
    assert scores[1]['shape_weight'] < .001
    assert scores[2]['shape_weight'] < .001
    assert scores[3]['shape_weight'] is None and scores[3]['shape_rank'] is None
    assert scores[1]['rms_log10'] > scores[0]['rms_log10']


def test_forward_density_priors_weight_terminal_preference():
    import darkflight as df
    observed = [{'cam': 'cam1', 'source': 'event_txt',
                 'confidence': 'medium', 'confidence_weight': 1.,
                 'coverage_fraction': 1., 'coverage_weight': 1.,
                 'weight': 1., 'time_source': 'track_fraction',
                 'source_index': 0, 't': np.array([2., 5., 8.]),
                 'flux': np.array([10., 20., 10.]),
                 'sat': np.zeros(3, dtype=bool)}]
    experiments = [
        {'model': 'single', 'density_prior_weight': .8,
         'light_curve': {'t_s': np.array([0., 2., 5., 8.]),
                         'power_w': np.array([1., 10., 20., 10.])}},
        {'model': 'disruption', 'density_prior_weight': .2,
         'light_curve': {'t_s': np.array([0., 2., 5., 8.]),
                         'power_w': np.array([1., 10., 20., 10.])}}]
    scores = df._light_curve_shape_scores(experiments, observed)
    assert [score['shape_weight'] for score in scores] == pytest.approx(
        [.5, .5])
    assert [score['density_prior_weight'] for score in scores] == [
        .8, .2]
    assert [score['model_preference_weight'] for score in scores] == \
        pytest.approx([.8, .2])
    assert [score['preference_rank'] for score in scores] == [1, 2]


def test_forward_light_curve_terminal_persistence_is_penalized():
    import darkflight as df
    penalty = df._post_observed_luminous_penalty(
        np.array([0., 2., 4., 6.]), np.array([1., 2., 8., 4.]), 2.)
    assert penalty['peak_delay_s'] == pytest.approx(2.)
    assert penalty['post_observed_peak_fraction'] == pytest.approx(1.)
    assert penalty['post_observed_energy_fraction'] > .7
    assert penalty['terminal_mismatch'] > .85


def test_forward_light_curve_coverage_reduces_camera_weight():
    import darkflight as df
    plot = {'final_params': [10., 0., 0., 0.],
            'final_merged_data': {'reltime': np.array([0., 10.])}}
    photometry = {'per_cam': [{'cam': 'short', 'confidence': 'medium',
        'warnings': [], 'm_app': [0., 0., 0.],
        'brightness': [100., 100., 100.], 'timestamps': [4.9, 5., 5.1],
        'ranges_km': [100., 100., 100.],
        'track_frac': [.49, .5, .51], 'video': None}]}
    observed, reason = df._observed_light_curve_samples(photometry, plot)
    assert reason is None and len(observed) == 1
    assert observed[0]['coverage_fraction'] == pytest.approx(.02)
    assert observed[0]['weight'] == pytest.approx(.2 + .8 * .02)


def test_forward_light_curve_track_fraction_rejects_out_of_range():
    import darkflight as df
    plot = {'final_params': [10., 0., 0., 0.],
            'final_merged_data': {'reltime': np.array([0., 10.])}}
    photometry = {'per_cam': [{'cam': 'outside', 'confidence': 'medium',
        'warnings': [], 'm_app': [0., 0., 0.],
        'brightness': [100., 100., 100.], 'timestamps': [0., 1., 2.],
        'ranges_km': [100., 100., 100.], 'track_frac': [-0.2, 0.5, 1.2],
        'video': None}]}
    observed, reason = df._observed_light_curve_samples(photometry, plot)
    assert observed == []
    assert reason == 'no camera light curve with three usable points'


def test_forward_light_curve_comparison_report_integration(tmp_path):
    import json
    import photometry
    import darkflight as df
    from types import SimpleNamespace
    from unittest.mock import patch
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.],
                             height=[80., 30.])
    plot = {'final_params': np.array([10., 0., 0., 0.]),
            'final_merged_data': {'reltime': np.array([0., 10.])}}
    phot = {'per_cam': [{'cam': 'cam1', 'confidence': 'medium',
        'warnings': [], 'm_app': [-2.5, 0., -2.5],
        'brightness': [100., 220., 100.],
        'timestamps': [2., 5., 8.], 'ranges_km': [100., 100., 100.],
        'track_frac': [.2, .5, .8], 'video': None}], 'n_cams': 0}
    curves = {'single': [5., 10., 1., 10.],
              'disruption': [5., 1., 10., 1.]}
    def fake(r, v, mass, density, shape, atm, ground, **options):
        power = curves[options['model']]
        return {'model': options['model'],
                'parameters': {'rho_m': density},
                'mass_budget': {'initial_kg': mass, 'ablated_kg': 0.,
                    'landed_kg': mass, 'unresolved_kg': 0., 'timeout_kg': 0.,
                    'balance_error_kg': 0.},
                'fragments': ([{'id': 0, 'multiplicity': 1.,
                    'final_mass_kg': mass, 'total_final_mass_kg': mass,
                    'impact': {'lon': 15., 'lat': 59., 'h': 0., 'v': 100.,
                               'm': mass, 't': 10., 'landed': True,
                               'termination': 'ground'},
                    'dark_release': {'r_ecef': [6371000., 0., 0.],
                                     'v_ecef': [0., 0., -3000.],
                                     'mass_kg': mass, 't_s': 10.}}]
                    if options['model'] == 'single' else []),
                'largest_landed_kg': mass, 'peak_absolute_magnitude': -12.,
                'warnings': [],
                'light_curve': {'t_s': [0., 2., 5., 8.],
                                'power_w': power}}
    def fake_mc(r, v, mass, rho, shape, wind_csv, h_ground, n_runs, **kwargs):
        assert kwargs.get('mass_samples') is None
        return [{'impact': {'lon': 15., 'lat': 59., 'h': 0., 'v': 100.,
                            'm': float(mass * (0.7 + 0.1 * i)),
                            't': 10., 'landed': True,
                            'termination': 'ground'},
                 'lon': np.array([15., 15.01]), 'lat': np.array([59., 59.01]),
                 'h': np.array([1000., 0.])}
                for i in range(n_runs)]
    with patch.object(df, 'propagate_fragmentation', side_effect=fake), \
            patch.object(df, 'monte_carlo', side_effect=fake_mc), \
            patch.object(df, 'ground_elevation', return_value=0.), \
            patch.object(photometry, 'event_photometry', return_value=phot):
        report = df.run_fragmentation_models(
            tmp_path / '20260423' / '021525', resdat, plot,
            tmp_path / 'forward', [1.], ['single', 'disruption'],
            h_ground=0., mc_runs=4)
    comparison = report['light_curve_comparison']
    assert comparison['available'] and comparison['n_cameras'] == 1
    assert comparison['cameras'][0]['coverage_fraction'] == pytest.approx(.6)
    assert comparison['best_shape_experiment']['model'] == 'single'
    scores = [e['light_curve_comparison'] for e in report['experiments']]
    assert scores[0]['shape_weight'] > .99
    assert scores[1]['shape_weight'] < .01
    assert scores[0]['shape_rank'] == 1
    assert scores[0]['preference_rank'] == 1
    assert scores[0]['terminal_match_score'] == pytest.approx(1.)
    assert scores[0]['model_preference_weight'] == pytest.approx(
        scores[0]['shape_weight'])
    assert report['experiments'][0]['monte_carlo']['n'] == 4
    assert len({impact['m'] for impact in
                report['experiments'][0]['monte_carlo']['impacts']}) > 1
    assert report['experiments'][0]['monte_carlo']['mass_sampling'] == \
        'mass perturbation around the intact release mass'
    assert report['experiments'][1]['monte_carlo']['n'] == 0
    assert report['mc']['n'] == 4 and report['mc']['attempted'] == 4
    stored = json.loads((tmp_path / 'forward' /
                         'fragmentation.json').read_text())
    assert stored['light_curve_comparison']['best_shape_experiment']['model'] == 'single'
    geojson = json.loads((tmp_path / 'forward' /
                          'fragmentation.geojson').read_text())
    assert len(geojson['features']) == 9
    assert sum(f['geometry']['type'] == 'LineString'
               for f in geojson['features']) == 4
    assert 'shape_weight' in (tmp_path / 'forward' /
                            'fragmentation_summary.csv').read_text()


def test_forward_mc_paths_prepend_release_approach_and_single_uses_release_mass():
    import darkflight as df
    from unittest.mock import patch
    release_r = llh2ecef(8.5, 56.8, 20000.)
    fragment = {'id': 0, 'multiplicity': 1., 'kind': 'parent',
                'final_mass_kg': .1, 'total_final_mass_kg': .1,
                'impact': {'lon': 8.7, 'lat': 56.6, 'h': 0., 'v': 40.,
                           'm': .1, 't': 20., 'landed': True,
                           'termination': 'ground'},
                'dark_release': {'r_ecef': release_r.tolist(),
                                 'v_ecef': [0., 0., -3000.],
                                 'mass_kg': .1, 't_s': 10.},
                'trajectory': {'lon': [8.0, 8.2, 8.5],
                               'lat': [57.0, 56.9, 56.8],
                               'h': [40000., 30000., 20000.]}}
    experiments = [
        {'model': 'single', 'fragments': [fragment],
         'parameters': {'rho_m': 3500., 'A_shape': 1.4}},
        {'model': 'disruption', 'fragments': [fragment],
         'parameters': {'rho_m': 3500., 'A_shape': 1.4,
                        'mass_index': 2.}}]
    calls = []
    def fake_mc(*args, **kwargs):
        calls.append(kwargs)
        return [{'impact': {'lon': 8.7, 'lat': 56.6, 'h': 0., 'v': 40.,
                            'm': .1, 't': 20., 'landed': True,
                            'termination': 'ground'},
                 'lon': np.array([8.5, 8.6]), 'lat': np.array([56.8, 56.7]),
                 'h': np.array([20000., 1000.])}]
    with patch.object(df, 'monte_carlo', side_effect=fake_mc):
        impacts = df._forward_mc_impacts(experiments, None, 0., 1, 0)
    assert calls[0]['mass_samples'] is None
    assert calls[1]['mass_samples'] is not None
    single, disruption = impacts
    assert single['spectrum_parent_mass_kg'] is None
    assert single['release_parent_mass_kg'] == pytest.approx(.1)
    assert disruption['spectrum_parent_mass_kg'] == pytest.approx(.1)
    assert single['trajectory']['lon'] == pytest.approx([8.0, 8.2, 8.5, 8.5, 8.6])
    assert single['trajectory']['h'][-1] == pytest.approx(1000.)


def test_forward_grain_aerodynamics_and_erosion_energy():
    import darkflight as df
    assert df._dragcoeff_nb(20000., 200., 1e-6, 1.4, 1e-4) == pytest.approx(df._cd_fm_nb(20000.))
    assert df._dragcoeff_nb(20000., 200., 1e-6, 1.4) == pytest.approx(df.dragcoeff(20000., 200., 1e-6, 1.4))
    assert df._dragcoeff_nb(20000., 200., 1e-6, 1.4, 1e-4) != pytest.approx(df._dragcoeff_nb(20000., 200., 1e-6, 1.4))
    empty = np.zeros(1)
    atmosphere = (False, empty, empty, empty, empty, empty, False, empty, empty, empty, 1.)
    r = llh2ecef(10., 60., 40000.)
    v = df.enu2ecef_mat(60., 10.) @ np.array([0., 0., -20000.])
    state = np.r_[r, v, 100., 0., 0.]
    ablation = df._fragmentation_rates_nb(state, 1e-8, 0., 1.4, 3500., atmosphere)
    erosion = df._fragmentation_rates_nb(state, 1e-8, 1e-7, 1.4, 3500., atmosphere)
    assert erosion[4] == ablation[4]
    assert erosion[0][8] == ablation[0][8]
    assert erosion[0][6] + erosion[0][7] == pytest.approx(ablation[0][6])
    assert erosion[0][7] > 0.


def test_forward_generation_limit_is_explicit():
    import darkflight as df
    r = llh2ecef(10., 60., 10000.)
    v = df.enu2ecef_mat(60., 10.) @ np.array([0., 0., -5000.])
    result = df.propagate_fragmentation(r, v, 1., 3500., 1.4, WindAtmosphere(), 0.,
        model='discrete', strength_pa=1., strength_exponent=0., max_generations=1,
        ablation_scale=0., max_step=0.1)
    assert result['generation_cap_reached'] and result['warnings']
    assert result['mass_budget']['landed_kg'] == pytest.approx(1.)
    assert result['largest_landed_kg'] == pytest.approx(1. / 8.)


def test_forward_light_curve_resolves_short_grain_lifetimes():
    import darkflight as df
    r = llh2ecef(10., 60., 10000.)
    v = df.enu2ecef_mat(60., 10.) @ np.array([0., 0., -5000.])
    result = df.propagate_fragmentation(r, v, 1., 3500., 1.4, WindAtmosphere(), 0.,
        model='disruption', strength_pa=1., ablation_scale=0.,
        max_step=0.05, record_dt=1., grain_bins=2)
    energy = sum(result['light_curve']['power_w'])
    assert 0 < energy < 0.03 * 0.5 * np.dot(v, v) * 1.01


def test_forward_unbroken_matches_darkflight_without_ablation():
    import darkflight as df
    r = llh2ecef(10., 60., 10000.)
    v = df.enu2ecef_mat(60., 10.) @ np.array([1000., 0., -5000.])
    result = df.propagate_fragmentation(r, v, 1., 3500., 1.4, WindAtmosphere(), 0.,
                                       ablation_scale=0., max_step=0.1)
    baseline = df.propagate(r, v, 1., 3500., 1.4, WindAtmosphere(), 0., erode=False)['impact']
    fragment = result['fragments'][0]
    impact = fragment['impact']
    trajectory = fragment['trajectory']
    assert trajectory['h'][0] == pytest.approx(10000.)
    assert trajectory['h'][-1] == pytest.approx(0.)
    assert max(trajectory['h']) == pytest.approx(10000.)
    assert min(np.diff(trajectory['h'])) < 0
    assert impact['lon'] == pytest.approx(baseline['lon'], abs=1e-5)
    assert impact['lat'] == pytest.approx(baseline['lat'], abs=1e-5)
    assert impact['m'] == pytest.approx(baseline['m'])


@pytest.mark.parametrize('options', [
    {'model': 'invented'}, {'strength_pa': 0.}, {'erosion_coeff': -1.},
    {'grain_min': 0.}, {'grain_max': 1e-8}, {'grain_bins': 0},
    {'fragment_count': 1}, {'luminous_efficiency': 2.}, {'max_step': 0.},
    {'breakup_height_m': 20000.}, {'ablation_scale': float('nan')},
    {'erosion_release_dt': 0.}, {'max_evaluations': 0},
])
def test_forward_rejects_invalid_parameters(options):
    import darkflight as df
    with pytest.raises(ValueError):
        df.propagate_fragmentation(llh2ecef(10., 60., 10000.), np.array([0., 0., -5000.]),
                                  1., 3500., 1.4, WindAtmosphere(), 0., **options)


def test_forward_work_limit_raises_instead_of_returning_partial_mass():
    import darkflight as df
    with pytest.raises(RuntimeError, match='work cap'):
        df.propagate_fragmentation(llh2ecef(10., 60., 10000.), np.array([0., 0., -5000.]),
                                  1., 3500., 1.4, WindAtmosphere(), 0., max_evaluations=1)


def test_forward_cli_orchestration_keeps_mass_and_magnitude_independent(tmp_path):
    import darkflight as df
    from types import SimpleNamespace
    from unittest.mock import patch
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.], height=[80., 30.])
    plot = {'final_params': np.array([20., -0.1, 0., 0.]),
            'final_merged_data': {'reltime': np.linspace(0., 2., 20)}}
    event = tmp_path / '20260423' / '021525'
    event.mkdir(parents=True)
    marker = event / 'darkflight.json'
    marker.write_text('untouched')
    def fake(r, v, mass, density, shape, atm, ground, **options):
        return {'model': options['model'], 'mass_budget': {'initial_kg': mass,
            'ablated_kg': mass, 'landed_kg': 0., 'unresolved_kg': 0.,
            'timeout_kg': 0., 'balance_error_kg': 0.}, 'fragments': [],
            'largest_landed_kg': 0., 'peak_absolute_magnitude': -17.}
    with patch.object(df, 'propagate_fragmentation', side_effect=fake) as forward, \
            patch.object(df, '_msis_profile', return_value=None):
        first = df.run_fragmentation_models(event, resdat, plot, tmp_path / 'one',
                            [100., 300.], ['all'], peak_mag=-17., h_ground=0.)
        second = df.run_fragmentation_models(event, resdat, plot, tmp_path / 'two',
                            [100., 300.], ['all'], peak_mag=-18., h_ground=0.)
        assert len(first['experiments']) == 8
        assert [e['mass_budget'] for e in first['experiments']] == [e['mass_budget'] for e in second['experiments']]
        assert forward.call_args.args[2] == 300.
        assert np.linalg.norm(forward.call_args.args[1]) == pytest.approx(20000.)
        assert first['initial_state']['mass_stage'] == 'first_observed_point'
        assert first['experiments'][0]['observed_peak_comparison']['model_minus_observed_mag'] == 0.
        assert second['experiments'][0]['observed_peak_comparison']['model_minus_observed_mag'] == 1.
        with pytest.raises(FileExistsError):
            df.run_fragmentation_models(event, resdat, plot, tmp_path / 'one',
                                         [100.], ['single'], h_ground=0.)
        with pytest.raises(ValueError):
            df.run_fragmentation_models(event, resdat, None, tmp_path / 'three',
                                         [100.], ['single'], h_ground=0.)
        with pytest.raises(ValueError):
            df.run_fragmentation_models(event, resdat, plot, event,
                                         [100.], ['single'], h_ground=0.)
    assert marker.read_text() == 'untouched'


def test_idle_cpu_workers_requires_majority_idle(monkeypatch):
    import darkflight as df
    monkeypatch.setattr(os, 'sched_getaffinity', lambda pid: set(range(8)))
    monkeypatch.setattr(os, 'getloadavg', lambda: (5.0, 4.0, 3.0))
    assert df._idle_cpu_workers(8) == 0
    monkeypatch.setattr(os, 'getloadavg', lambda: (3.0, 2.0, 1.0))
    assert df._idle_cpu_workers(8) == 5
    assert df._idle_cpu_workers(3, max_workers=16) == 3
    assert df._idle_cpu_workers(20, max_workers=4) == 4
    assert df._idle_cpu_workers(255, min_tasks=256) == 0
    assert df._idle_cpu_workers(256, min_tasks=256) == 5


def test_forward_experiment_jobs_use_supplied_pool(tmp_path, monkeypatch):
    import darkflight as df
    from types import SimpleNamespace
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.],
                             height=[80., 30.])
    plot = {'final_params': np.array([20., -0.1, 0., 0.]),
            'final_merged_data': {'reltime': np.linspace(0., 2., 20)}}
    event = tmp_path / '20260423' / '021525'
    event.mkdir(parents=True)

    def fake(r, v, mass, density, shape, atm, ground, **options):
        return {'model': options['model'],
                'parameters': {'rho_m': density},
                'mass_budget': {'initial_kg': mass, 'ablated_kg': 0.,
                    'landed_kg': 0., 'unresolved_kg': mass,
                    'timeout_kg': 0., 'balance_error_kg': 0.},
                'fragments': [], 'largest_landed_kg': 0.,
                'peak_absolute_magnitude': -18., 'warnings': []}

    class RecordingPool:
        def __init__(self):
            self.calls = []
        def map(self, worker, jobs):
            jobs = list(jobs)
            self.calls.append((worker, jobs))
            return [worker(job) for job in jobs]

    pool = RecordingPool()
    monkeypatch.setattr(df, 'propagate_fragmentation', fake)
    report = df.run_fragmentation_models(
        event, resdat, plot, tmp_path / 'parallel', [1., 2.],
        ['single', 'discrete'], rho_m=[3500., 7000.],
        h_ground=0., pool=pool)
    assert len(pool.calls) == 1
    worker, jobs = pool.calls[0]
    assert worker is df._forward_experiment_worker and len(jobs) == 8
    assert [(job[1], job[2], job[3]) for job in jobs] == [
        (1., 'single', 3500.), (1., 'single', 7000.),
        (1., 'discrete', 3500.), (1., 'discrete', 7000.),
        (2., 'single', 3500.), (2., 'single', 7000.),
        (2., 'discrete', 3500.), (2., 'discrete', 7000.)]
    assert len(report['experiments']) == 8


def test_disintegration_burst_follows_prior_flight_direction():
    import darkflight as df
    scenario = {'mass_budget': {'landed_kg': 0.}, 'fragments': []}
    start = (16.4, 60.0, 90000.)
    end = (15.4, 59.4, 35000.)
    origins, endpoints = df._disintegration_burst_points(
        scenario, start, end, n_rays=48)
    assert origins.shape == endpoints.shape == (48, 3)
    assert np.all(origins == np.asarray(end))
    assert np.all(endpoints[:, 2] < end[2])
    flight = llh2ecef(*end) - llh2ecef(*start)
    flight /= np.linalg.norm(flight)
    displacement = np.array([llh2ecef(*point) for point in endpoints]) - llh2ecef(*end)
    alignment = displacement @ flight / np.linalg.norm(displacement, axis=1)
    assert np.all(alignment > 0.98)


def test_forward_density_list_creates_separate_experiments(tmp_path):
    import darkflight as df
    from types import SimpleNamespace
    from unittest.mock import patch
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.],
                             height=[80., 30.])
    plot = {'final_params': np.array([20., -0.1, 0., 0.]),
            'final_merged_data': {'reltime': np.linspace(0., 2., 20)}}
    event = tmp_path / '20260423' / '021525'
    event.mkdir(parents=True)

    def fake(r, v, mass, density, shape, atm, ground, **options):
        return {'model': options['model'],
                'parameters': {'rho_m': density},
                'mass_budget': {'initial_kg': mass, 'ablated_kg': 0.,
                    'landed_kg': 0., 'unresolved_kg': mass,
                    'timeout_kg': 0., 'balance_error_kg': 0.},
                'fragments': [], 'largest_landed_kg': 0.,
                'peak_absolute_magnitude': -18., 'warnings': []}

    with patch.object(df, 'propagate_fragmentation', side_effect=fake) as calls:
        report = df.run_fragmentation_models(
            event, resdat, plot, tmp_path / 'density-grid', [100.],
            ['single', 'disruption'], rho_m=[3500., 7000.],
            density_priors=[8., 2.], h_ground=0.)
    assert [call.args[3] for call in calls.call_args_list] == [
        3500., 7000., 3500., 7000.]
    assert report['densities_kg_m3'] == [3500., 7000.]
    assert [experiment['density_kg_m3']
            for experiment in report['experiments']] == [
                3500., 7000., 3500., 7000.]
    assert [experiment['experiment_index']
            for experiment in report['experiments']] == [0, 1, 2, 3]
    assert [experiment['density_prior_weight']
            for experiment in report['experiments']] == [
                .8, .2, .8, .2]
    assert report['density_priors'][0]['normalized_weight'] == pytest.approx(.8)
    assert 'density prior' in report['interpretation']
    summary = (tmp_path / 'density-grid' /
               'fragmentation_summary.csv').read_text()
    assert 'density_kg_m3' in summary and '3500.0' in summary
    assert '7000.0' in summary


def test_forward_estimates_entry_mass_from_deceleration(tmp_path):
    import darkflight as df
    assert df.parse_args([str(tmp_path), '--fragmentation-model', 'single']).entry_mass is None
    from types import SimpleNamespace
    from unittest.mock import patch
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.],
                             height=[80., 30.])
    plot = {'final_params': np.array([20., -0.1, 0., 0.]),
            'final_merged_data': {
                'reltime': np.linspace(0., 2., 20),
                'height': np.linspace(80., 30., 20)}}
    event = tmp_path / '20260423' / '021525'
    event.mkdir(parents=True)
    estimates = [
        {'rho': 3500., 'A': 1.4, 'm_fade_kg': 2., 'm_fade_lo': 1.,
         'm_fade_hi': 4., 'm_crit_kg': .5},
        {'rho': 7000., 'A': 1.4, 'm_fade_kg': 4., 'm_fade_lo': 2.,
         'm_fade_hi': 8., 'm_crit_kg': .7}]

    def fake_entry(m_fade, density, shape, v_of_t, h_of_t, t_range, atm=None):
        return m_fade * 10. + density / 7000.

    def fake(r, v, mass, density, shape, atm, ground, **options):
        return {'model': options['model'],
                'parameters': {'rho_m': density},
                'mass_budget': {'initial_kg': mass, 'ablated_kg': 0.,
                    'landed_kg': 0., 'unresolved_kg': mass,
                    'timeout_kg': 0., 'balance_error_kg': 0.},
                'fragments': [], 'largest_landed_kg': 0.,
                'peak_absolute_magnitude': -18., 'warnings': []}

    with patch.object(df, 'estimate_mass', return_value=estimates) as fitted, \
            patch.object(df, 'entry_mass_estimate', side_effect=fake_entry), \
            patch.object(df, 'propagate_fragmentation', side_effect=fake) as calls:
        report = df.run_fragmentation_models(
            event, resdat, plot, tmp_path / 'deceleration', None,
            ['single', 'disruption'], rho_m=[3500., 7000.], h_ground=0.)
    assert fitted.call_args.kwargs['rho_grid'] == [3500., 7000.]
    assert report['entry_mass_source'] == 'deceleration_back_projection'
    assert [call.args[2] for call in calls.call_args_list] == [
        20.5, 41., 20.5, 41.]
    assert [e['density_kg_m3'] for e in report['experiments']] == [
        3500., 7000., 3500., 7000.]
    assert [e['entry_mass_estimate']['m_entry_kg']
            for e in report['experiments']] == [20.5, 41., 20.5, 41.]
    assert report['entry_mass_estimates'][0]['m_entry_kg'] == 20.5


def test_forward_auto_uses_reliable_unsaturated_peak(tmp_path, monkeypatch):
    import darkflight as df
    import photometry
    from types import SimpleNamespace
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.],
                             height=[80., 30.])
    plot = {'final_params': np.array([20., -0.1, 0., 0.]),
            'final_merged_data': {'reltime': np.linspace(0., 2., 20)}}
    event = tmp_path / '20260423' / '021525'
    event.mkdir(parents=True)
    peak_flux = 1500. * 10. ** (18. * .4)
    observed = [
        {'cam': name, 'source': 'video', 'confidence': 'medium',
         'warnings': [], 't': np.linspace(0., 2., 5),
         'flux': np.array([1., 10., peak_flux, 10., 1.]),
         'sat': np.zeros(5, dtype=bool),
         'coverage_fraction': 1., 'coverage_weight': 1.,
         'confidence_weight': 1., 'weight': 1.,
         'time_source': 'station_alignment', 'source_index': index}
        for index, name in enumerate(('cam1', 'cam2'))]
    monkeypatch.setattr(photometry, 'event_photometry',
                        lambda *args, **kwargs: {'per_cam': observed})
    monkeypatch.setattr(df, '_observed_light_curve_samples',
                        lambda *args, **kwargs: (observed, None))

    def fake(r, v, mass, density, shape, atm, ground, **options):
        return {'model': options['model'],
                'parameters': {'rho_m': density},
                'mass_budget': {'initial_kg': mass, 'ablated_kg': 0.,
                    'landed_kg': 0., 'unresolved_kg': mass,
                    'timeout_kg': 0., 'balance_error_kg': 0.},
                'fragments': [], 'largest_landed_kg': 0.,
                'peak_absolute_magnitude': -18., 'warnings': []}

    monkeypatch.setattr(df, 'propagate_fragmentation', fake)
    monkeypatch.setattr(df, '_solve_fragmentation_mass_for_magnitude',
        lambda model, propagate, target, speed, duration:
            (propagate({'single': 10., 'disruption': 20.}[model]),
             {'target_magnitude': target}))
    report = df.run_fragmentation_models(
        event, resdat, plot, tmp_path / 'photometric', None,
        ['single', 'disruption'], h_ground=0.)
    estimate = report['light_curve_comparison'][
        'observed_peak_magnitude_estimate']
    assert estimate['absolute_magnitude_at_100_km'] == pytest.approx(-18.)
    assert report['entry_mass_source'] == (
        'observed_light_curve_magnitude_inversion')
    assert [e['mass_budget']['initial_kg']
            for e in report['experiments']] == [10., 20.]
    assert all(e['observed_peak_comparison']['peak_magnitude_source'] ==
               'observed_light_curve' for e in report['experiments'])


def test_uncertain_deceleration_estimate_suppresses_model_preference(
        tmp_path, monkeypatch):
    import darkflight as df
    import photometry
    from types import SimpleNamespace
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.],
                             height=[80., 30.])
    plot = {'final_params': np.array([20., -0.1, 0., 0.]),
            'final_merged_data': {
                'reltime': np.linspace(0., 2., 20),
                'height': np.linspace(80., 30., 20)}}
    event = tmp_path / '20260423' / '021525'
    event.mkdir(parents=True)
    estimates = [{'rho': 3500., 'A': 1.4, 'm_fade_kg': 2.,
                  'm_fade_lo': 1., 'm_fade_hi': 100., 'm_crit_kg': .01}]
    observed = [{'cam': 'cam1', 'source': 'video_flux',
                 'confidence': 'low', 'warnings': ['saturated'],
                 't': np.linspace(0., 2., 5), 'flux': np.ones(5),
                 'sat': np.ones(5, dtype=bool), 'coverage_fraction': 1.,
                 'coverage_weight': 1., 'confidence_weight': 1.,
                 'weight': 1.,
                 'time_source': 'track_fraction', 'source_index': 0}]
    monkeypatch.setattr(photometry, 'event_photometry',
                        lambda *args, **kwargs: {'per_cam': observed})
    monkeypatch.setattr(df, 'estimate_mass', lambda *args, **kwargs: estimates)
    monkeypatch.setattr(df, 'entry_mass_estimate', lambda *a, **k: 20.)
    monkeypatch.setattr(df, '_observed_light_curve_samples',
                        lambda *args, **kwargs: (observed, None))
    monkeypatch.setattr(df, '_light_curve_shape_scores', lambda experiments, obs: [
        {'index': 0, 'model': 'single', 'available': True,
         'log_likelihood': 0., 'rms_log10': 0., 'median_abs_log10': 0.,
         'shape_weight': 1., 'shape_rank': 1,
         'post_observed_duration_s': 0., 'terminal_match_tau_s': .5,
         'terminal_match_score': 1., 'preference_rank': 1,
         'model_preference_weight': 1., 'per_camera': []}])
    monkeypatch.setattr(df, 'propagate_fragmentation',
        lambda r, v, mass, density, shape, atm, ground, **options: {
            'model': 'single', 'parameters': {'rho_m': density},
            'mass_budget': {'initial_kg': mass, 'ablated_kg': 0.,
                'landed_kg': 0., 'unresolved_kg': mass,
                'timeout_kg': 0., 'balance_error_kg': 0.},
            'fragments': [], 'largest_landed_kg': 0.,
            'peak_absolute_magnitude': -12., 'warnings': []})
    report = df.run_fragmentation_models(
        event, resdat, plot, tmp_path / 'uncertain', None, ['single'],
        h_ground=0.)
    score = report['experiments'][0]['light_curve_comparison']
    assert score['model_preference_weight'] is None
    assert score['preference_rank'] is None
    preference = report['light_curve_comparison']['model_preference']
    assert preference['available'] is False
    assert 'poorly constrained' in preference['unavailable_reason']


def test_failed_magnitude_inversion_keeps_other_experiments(tmp_path, monkeypatch):
    import darkflight as df
    import json
    from types import SimpleNamespace
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.],
                             height=[80., 30.])
    plot = {'final_params': np.array([20., -0.1, 0., 0.]),
            'final_merged_data': {'reltime': np.linspace(0., 2., 20)}}
    event = tmp_path / '20260423' / '021525'
    event.mkdir(parents=True)

    def solve(model, propagate, target, speed, duration):
        if model == 'erosion':
            raise RuntimeError(
                'Fragmentation rate-evaluation work cap exceeded (4,000,000)')
        return ({'model': model,
                 'parameters': {'rho_m': 3500.},
                 'mass_budget': {'initial_kg': 20., 'ablated_kg': 0.,
                     'landed_kg': 0., 'unresolved_kg': 20.,
                     'timeout_kg': 0., 'balance_error_kg': 0.},
                 'fragments': [], 'largest_landed_kg': 0.,
                 'peak_absolute_magnitude': target, 'warnings': []},
                {'solved_entry_mass_kg': 20.})

    monkeypatch.setattr(df, '_solve_fragmentation_mass_for_magnitude', solve)
    monkeypatch.setattr(df, '_parallel_map', lambda worker, jobs, **kwargs:
                        [worker(job) for job in jobs])
    report = df.run_fragmentation_models(
        event, resdat, plot, tmp_path / 'partial-inversion', None,
        ['single', 'erosion'], peak_mag=-18., h_ground=0.)
    assert len(report['experiments']) == 2
    assert report['experiments'][0]['mass_budget']['initial_kg'] == 20.
    failed = report['experiments'][1]
    assert failed['inversion_failed']
    assert failed['mass_budget']['initial_kg'] is None
    assert failed['warnings'] == ['magnitude inversion failed']
    stored = json.loads((tmp_path / 'partial-inversion' /
                         'fragmentation.json').read_text())
    assert stored['experiments'][1]['inversion_failed']


def test_forward_magnitude_solves_entry_mass_per_model(tmp_path):
    import darkflight as df
    from types import SimpleNamespace
    from unittest.mock import patch
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.], height=[80., 30.])
    plot = {'final_params': np.array([20., -0.1, 0., 0.]),
            'final_merged_data': {'reltime': np.linspace(0., 2., 20)}}
    event = tmp_path / '20260423' / '021525'
    event.mkdir(parents=True)
    reference = {'single': 100., 'disruption': 300.}

    def fake(r, v, mass, density, shape, atm, ground, **options):
        model = options['model']
        peak = -18.74 - 2.5 * math.log10(mass / reference[model])
        return {'model': model, 'parameters': {'rho_m': density,
                'luminous_efficiency': 0.03},
                'mass_budget': {'initial_kg': mass, 'ablated_kg': 0.,
                    'landed_kg': 0., 'unresolved_kg': mass, 'timeout_kg': 0.,
                    'balance_error_kg': 0.}, 'fragments': [],
                'largest_landed_kg': 0., 'peak_absolute_magnitude': peak,
                'warnings': []}

    with patch.object(df, 'propagate_fragmentation', side_effect=fake), \
            patch.object(df, '_msis_profile', return_value=None):
        report = df.run_fragmentation_models(event, resdat, plot,
            tmp_path / 'magnitude-solution', None, ['single', 'disruption'],
            peak_mag=-18.74, h_ground=0.)
        assert report['entry_mass_source'] == 'magnitude_inversion'
        assert [e['mass_budget']['initial_kg'] for e in report['experiments']] == pytest.approx(
            [100., 300.], rel=1e-6)
        for experiment in report['experiments']:
            inversion = experiment['entry_mass_inversion']
            assert inversion['solved_entry_mass_kg'] == pytest.approx(
                reference[experiment['model']], rel=1e-6)
            assert experiment['peak_absolute_magnitude'] == pytest.approx(-18.74, abs=1e-6)
            assert experiment['observed_peak_comparison']['used_to_fit_parameters']
        with pytest.raises(ValueError,
                           match='usable speed/deceleration-height fit'):
            df.run_fragmentation_models(event, resdat, plot, tmp_path / 'missing',
                                        None, ['single'], h_ground=0.)
        with pytest.raises(ValueError, match='brighter than -1.*did you mean -18.7'):
            df.run_fragmentation_models(event, resdat, plot, tmp_path / 'positive',
                                        None, ['single'], peak_mag=18.74,
                                        h_ground=0.)


def test_forward_default_output_publishes_event_products(tmp_path):
    import darkflight as df
    import json
    from types import SimpleNamespace
    from unittest.mock import patch
    resdat = SimpleNamespace(long1=[10., 10.1], lat1=[60.1, 60.], height=[80., 30.])
    plot = {'final_params': np.array([20., -0.1, 0., 0.]),
            'final_merged_data': {'reltime': np.linspace(0., 2., 20)}}
    event = tmp_path / '20260423' / '021525'
    event.mkdir(parents=True)
    (event / 'darkflight.json').write_text('previous report')
    (event / 'en_map_darkflight.svg').write_text('stale localized map')
    (event / 'en_map_darkflight.jpg').write_text('stale localized jpg')
    (event / 'en_darkflight_table.html').write_text('stale localized table')
    (event / 'index.php').write_text('preserve')

    def fake(r, v, mass, density, shape, atm, ground, **options):
        landed = mass * 0.75
        return {'model': options['model'],
                'parameters': {'rho_m': density, 'luminous_efficiency': 0.03},
                'mass_budget': {'initial_kg': mass, 'ablated_kg': mass - landed,
                    'landed_kg': landed, 'unresolved_kg': 0., 'timeout_kg': 0.,
                    'balance_error_kg': 0.},
                'fragments': [{'id': 0, 'parent_id': None, 'kind': 'parent',
                    'weight': 1., 'multiplicity': 1., 'generation': 0,
                    'strength_pa': 1e6, 't_birth_s': 0., 't_end_s': 120.,
                    'initial_mass_kg': mass, 'final_mass_kg': landed,
                    'total_final_mass_kg': landed, 'ablated_kg': mass - landed,
                    'eroded_kg': 0., 'outcome': 'landed', 'termination': 'dark_ground',
                    'generation_cap_reached': False,
                    'impact': {'lon': 10.01, 'lat': 60., 'h': 0., 'v': 20.,
                               'm': landed, 'landed': True},
                    'plot_release_mass_kg': landed,
                    'trajectory': {'lon': [10.1, 10.01], 'lat': [60., 60.],
                                   'h': [30000., 0.]}}],
                'largest_landed_kg': landed, 'peak_absolute_magnitude': -17.,
                'warnings': []}

    def fake_map(scenarios, end_llh, mc_impacts, path, **kwargs):
        Path(path).write_text('<svg>modelled fragmentation</svg>')

    with patch.object(df, 'propagate_fragmentation', side_effect=fake), \
            patch.object(df, '_msis_profile', return_value=None), \
            patch.object(df, 'write_map') as map_writer, \
            patch.object(df, 'write_map3d', return_value=False):
        map_writer.side_effect = fake_map
        report = df.run_fragmentation_models(event, resdat, plot, None, [100.],
            ['single'], peak_mag=-18., h_ground=0.,
            lang_files={'': {'dec_sep': '.'}})
        assert 'modelled fragmentation' in (event / 'map_darkflight.svg').read_text()
        map_writer.side_effect = RuntimeError('map renderer unavailable')
        report = df.run_fragmentation_models(event, resdat, plot, None, [100.],
            ['single'], peak_mag=-18., h_ground=0.,
            lang_files={'': {'dec_sep': '.'}})
    assert report['publication_mode'] == 'event'
    assert report['analysis_mode'] == 'fragmentation'
    assert (event / 'index.php').read_text() == 'preserve'
    assert json.loads((event / 'darkflight.json').read_text())['analysis_mode'] == 'fragmentation'
    assert json.loads((event / 'fragmentation.json').read_text())['publication_mode'] == 'event'
    assert json.loads((event / 'darkflight.geojson').read_text())['features']
    assert '<svg xmlns=' in (event / 'map_darkflight.svg').read_text()
    assert 'modelled fragmentation' not in (event / 'map_darkflight.svg').read_text()
    assert '3D renderer unavailable' in (event / 'darkflight_map3d.html').read_text()
    table = (event / 'darkflight_table.html').read_text()
    assert '<table' in table and table.count('<th') == 12
    assert 'Model comparison unavailable' in table
    assert not (event / 'en_map_darkflight.svg').exists()
    assert not (event / 'en_map_darkflight.jpg').exists()
    assert not (event / 'en_darkflight_table.html').exists()
    assert 'landed_kg' in (event / 'fragmentation_summary.csv').read_text()
    assert json.loads((event / 'fragmentation.geojson').read_text())['features']
    assert not list(event.glob('.darkflight-*'))


def test_fragmentation_products_localize_all_languages(tmp_path):
    import darkflight as df
    import html
    import json
    from unittest.mock import patch
    required = ('df_luminous_path', 'df_fall_area_edge', 'df_mc_hatch',
                'df_frag_section',
                'df_frag_model_single', 'df_frag_model_discrete',
                'df_frag_model_disruption', 'df_frag_model_erosion',
                'df_frag_scenario', 'df_frag_map_title',
                'df_frag_map_no_impacts', 'df_frag_3d_unavailable',
                'df_frag_interpretation', 'df_frag_inversion_note',
                'df_frag_table_note', 'df_frag_unresolved_note',
                'df_frag_maps_note', 'df_frag_no_table_impacts',
                'df_frag_h_model', 'df_frag_h_initial', 'df_frag_h_ablated',
                'df_frag_h_landed', 'df_frag_h_unresolved',
                'df_frag_h_largest', 'df_frag_h_peak_mag', 'df_frag_h_density',
                'df_frag_h_shape', 'df_frag_warn_generation',
                'df_frag_shape_note', 'df_frag_mc_note',
                'df_frag_h_peak_delay', 'df_frag_help_peak_delay',
                'df_frag_h_post_light', 'df_frag_help_post_light',
                'df_frag_h_preference', 'df_frag_help_preference',
                'df_frag_preference_note', 'df_frag_best_row',
                'df_frag_light_unavailable', 'df_frag_disintegration',
                'df_frag_help_model', 'df_frag_help_initial',
                'df_frag_help_ablated', 'df_frag_help_landed',
                'df_frag_help_unresolved',
                'df_frag_help_largest', 'df_frag_help_peak_mag',
                'df_frag_help_density', 'df_frag_help_shape')
    lang_files = df.load_language_files()
    assert set(lang_files) == {'', 'en_', 'de_', 'cs_', 'fi_', 'lv_'}
    assert lang_files['']['plotly_locale'] == 'no'
    for translations in lang_files.values():
        missing = [key for key in required if not translations.get(key)]
        assert not missing
        scenario = {'name': 'F0_disruption', 'label': 'fallback',
                  'model': 'disruption', 'initial_mass_kg': 1.25,
                  'conditional': True}
        label = df.sc_label(scenario, translations)
        assert translations['df_frag_model_disruption'] in label
        expected_mass = '1,25' if translations['dec_sep'] == ',' else '1.25'
        assert expected_mass in label

    experiment = {'model': 'disruption',
        'parameters': {'rho_m': 3500., 'luminous_efficiency': 0.03},
        'mass_budget': {'initial_kg': 9876., 'ablated_kg': 1234.,
            'landed_kg': 0., 'unresolved_kg': .567, 'timeout_kg': 0.,
            'balance_error_kg': 0.}, 'fragments': [
                {'id': 0, 'kind': 'parent', 'outcome': 'disrupted',
                 't_end_s': 1.0, 'multiplicity': 1.,
                 'total_final_mass_kg': 1.2, 'trajectory': None},
                {'id': 1, 'kind': 'grain', 'outcome': 'unresolved',
                 't_end_s': 1.5, 'multiplicity': 1000.,
                 'total_final_mass_kg': .05, 'trajectory': None}],
        'largest_landed_kg': .0049, 'peak_absolute_magnitude': -12.5,
        'density_kg_m3': 3500., 'experiment_index': 0,
        'light_curve_comparison': {'shape_weight': .73, 'shape_rank': 1,
            'model_preference_weight': .88, 'preference_rank': 1},
        'warnings': ['Cohort 7: generation cap 12 reached; continuing intact']}
    other = {**experiment, 'model': 'single', 'experiment_index': 1,
        'largest_landed_kg': .0299,
        'light_curve_comparison': {'shape_weight': .1, 'shape_rank': 3,
            'model_preference_weight': .01, 'preference_rank': 3}}
    other_density = {**experiment, 'density_kg_m3': 7000.,
        'experiment_index': 2,
        'light_curve_comparison': {'shape_weight': .2, 'shape_rank': 2,
            'model_preference_weight': .12, 'preference_rank': 2}}
    report = {'initial_state': {'lon': 9.8, 'lat': 60.2, 'h_m': 90000.,
                                 'speed_ms': 12000.},
              'observed_end': {'lon': 10., 'lat': 60., 'h_m': 30000.,
                               'speed_ms': 3000., 'duration_s': 2.},
              'experiments': [experiment, other, other_density],
              'entry_mass_source': 'magnitude_inversion',
              'mc': {'n': 2, 'impacts': [
                  {'lon': 10., 'lat': 60., 'h': 0., 'v': 50., 'm': .1,
                   'landed': True, 'model': 'disruption'},
                  {'lon': 10.1, 'lat': 60.1, 'h': 0., 'v': 50., 'm': .2,
                   'landed': True, 'model': 'single'}]},
              'interpretation': 'English fallback that must not leak'}
    captured_titles, captured_starts = [], []
    captured_scenarios, captured_mc = [], []
    captured_static_scenarios = []
    def fake_map(scenarios, end_llh, mc_impacts, path, **kwargs):
        captured_static_scenarios.append([s['name'] for s in scenarios])
        captured_titles.append(kwargs['title'])
        Path(path).write_text('<svg/>')
    def fake_map3d(scenarios, end_llh, mc_impacts, path, **kwargs):
        captured_scenarios.append([scenario['name'] for scenario in scenarios])
        captured_mc.append([impact['model'] for impact in mc_impacts])
        captured_starts.append(kwargs['start_llh'])
        return False
    with patch.object(df, 'write_map', side_effect=fake_map), \
            patch.object(df, 'write_map3d', side_effect=fake_map3d):
        df._publish_fragmentation(report, tmp_path, lang_files)
    expected_titles = [t['df_frag_map_no_impacts'] for t in lang_files.values()]
    assert captured_titles == expected_titles
    assert captured_starts == [(9.8, 60.2, 90000.)] * len(lang_files)
    assert captured_scenarios == [['F0_disruption']] * len(lang_files)
    assert captured_static_scenarios == [['F0_disruption']] * len(lang_files)
    assert captured_mc == [['disruption']] * len(lang_files)
    assert report['interactive_model'] == 'disruption'
    assert report['interactive_density_kg_m3'] == 3500.
    for prefix, translations in lang_files.items():
        table = (tmp_path / f'{prefix}darkflight_table.html').read_text()
        assert translations['df_frag_interpretation'] not in table
        assert 'English fallback that must not leak' not in table
        assert translations['df_frag_table_note'] not in table
        assert translations['df_frag_inversion_note'] not in table
        assert translations['df_frag_unresolved_note'] not in table
        assert translations['df_frag_maps_note'] not in table
        assert translations['df_frag_no_table_impacts'] in table
        assert translations['df_frag_h_model'] in table
        assert translations['df_frag_model_disruption'] in table
        assert 'font-size:.85em' in table and 'title="' in table
        assert 'class="df-table-wrap"' in table and 'font-size:14px' in table
        assert 'ResizeObserver' in table
        assert '.dftable td{text-align:center' in table
        assert 'class="df-frag-best"' in table
        assert 'background-color:#fff3bf' in table
        assert 'body.theme-dark .dftable tr.df-frag-best' in table
        assert html.escape(translations['df_frag_best_row'], quote=True) in table
        for header_key in ('peak_delay', 'post_light', 'preference'):
            assert all(word in table for word in
                       translations[f'df_frag_h_{header_key}'].split())
        table_markup = table.split('<script', 1)[0]
        expected_ablated = '1234,0' if translations['dec_sep'] == ',' else '1234.0'
        expected_initial = '9876,0' if translations['dec_sep'] == ',' else '9876.0'
        assert expected_ablated in table and expected_initial in table and 'e+' not in table_markup
        assert '.0049' not in table and ',0049' not in table
        expected_grams = '0,030' if translations['dec_sep'] == ',' else '0.030'
        assert expected_grams in table
        assert 'writing-mode:vertical-rl' in table
        assert 'min-height:120px' in table
        assert 'inline-size:120px' in table and 'white-space:normal' in table
        assert translations['df_frag_shape_note'] not in table
        assert translations['df_frag_preference_note'] not in table
        assert html.escape(translations['df_frag_mc_note']) not in table
        assert table.count('<th') == 12
        cells = re.findall(r'<td[^>]*>(.*?)</td>', table)
        assert cells[::12] == [translations['df_frag_model_disruption'],
                               translations['df_frag_model_disruption'],
                               translations['df_frag_model_single']]
        if prefix == '':
            assert 'Mørk flukt / fragmentering' in table
        for key in ('model', 'initial', 'ablated', 'landed', 'unresolved',
                    'largest', 'peak_mag', 'density', 'shape'):
            assert translations[f'df_frag_help_{key}'] in table
            assert len(translations[f'df_frag_help_{key}']) > len(translations[f'df_frag_h_{key}'])
        warning = translations['df_frag_warn_generation'].format(cohort=7, limit=12)
        assert warning in table
        expected_shape = '73,0%' if translations['dec_sep'] == ',' else '73.0%'
        assert expected_initial in table and expected_shape in table
        fallback3d = (tmp_path / f'{prefix}darkflight_map3d.html').read_text()
        assert translations['df_frag_3d_unavailable'] in fallback3d
    stored = json.loads((tmp_path / 'darkflight.json').read_text())
    assert stored['experiments'][0]['model'] == 'disruption'


def test_fragmentation_table_explains_unavailable_light_curve(tmp_path):
    import darkflight as df
    experiment = {'model': 'single', 'parameters': {'rho_m': 3500.},
        'mass_budget': {'initial_kg': 10., 'ablated_kg': 9.,
            'landed_kg': .5, 'unresolved_kg': .5, 'timeout_kg': 0.},
        'largest_landed_kg': .5, 'peak_absolute_magnitude': -18.,
        'warnings': [],
        'light_curve_comparison': {'shape_weight': None,
            'model_preference_weight': None}}
    report = {'experiments': [experiment],
              'light_curve_comparison': {
                  'available': False,
                  'unavailable_reason':
                      'no camera light curve with three usable points'}}
    path = tmp_path / 'table.html'
    df._write_fragmentation_table(report, path,
                                  df.load_language_files()['en_'])
    table = path.read_text()
    assert 'Model preference' in table
    assert 'Shape weight' in table
    assert table.count('<th') == 12
    assert ('Model comparison unavailable: no usable observed camera '
            'light curve.') in table


def test_fragmentation_table_scales_budget_to_outcome_mass(tmp_path):
    import darkflight as df
    experiment = {'model': 'single', 'parameters': {'rho_m': 7400.},
        'mass_budget': {'initial_kg': 8502.683119375732,
            'ablated_kg': 8502.161415551303,
            'landed_kg': 0.5217038244286959,
            'unresolved_kg': 0., 'timeout_kg': 0.},
        'largest_landed_kg': 0.5217038244286959,
        'peak_absolute_magnitude': -18.5,
        'warnings': [],
        'light_curve_comparison': {'shape_weight': None,
            'model_preference_weight': None}}
    report = {'experiments': [experiment], 'light_curve_comparison': {}}
    path = tmp_path / 'table.html'
    df._write_fragmentation_table(report, path,
                                  df.load_language_files()['en_'])
    cells = re.findall(r'<td[^>]*>(.*?)</td>', path.read_text())
    assert cells[1:4] == ['8502.7', '8502.2', '0.522']


def test_fbspd_terminal_bounded_exponential_beats_constant_deceleration():
    import fbspd_merge as fb
    t = np.linspace(0.0, 5.4, 90)
    true = np.array([23.5, -0.0054, 1.62, -2.2])
    sig = np.full_like(t, 0.12)
    rng = np.random.default_rng(7)
    pos = fb.expfunc(t, *true) + rng.normal(0.0, sig * 0.15)
    params, n_ok, pcov, cost = fb._fit_merged_data_with_cost(
        t, pos, sig, debug=False, fscale=0.1)
    assert n_ok == len(t) and params[2] > 0.0
    assert fb.expfunc_1stder(t[-1], *params) >= 0.0
    quad_pos = np.polyval(np.polyfit(t, pos, 2), t)
    assert abs(pos[-1] - fb.expfunc(t[-1], *params)) < abs(pos[-1] - quad_pos[-1])
    assert pcov is not None and pcov.shape == (4, 4)


def test_fbspd_refraction_correction_uses_delta_z_observer_height():
    import fbspd_merge as fb
    cendat = fb.CenData()
    cendat.ndata = 1
    cendat.cenalt = np.array([45.0])
    cendat.cenaz = np.array([0.0])
    cendat.censig = np.array([1.0])
    cendat.site_info = {'name': 'TEST', 'lon': 0., 'lat': 0., 'height': 0.}
    site_pos = fb.lonlat2xyz(0., 0., 0.)
    track_p1 = fb.lonlat2xyz(0., 0.1, 10.)
    track_p2 = fb.lonlat2xyz(0., 0.1, 30.)
    track_vec = (track_p2 - track_p1) / np.linalg.norm(track_p2 - track_p1)
    raw = fb._process_station(cendat, site_pos, track_p1, track_vec,
                              refraction=False)
    corrected = fb._process_station(cendat, site_pos, track_p1, track_vec,
                                    refraction=True)
    assert raw['refraction_deg'] == pytest.approx([0.])
    assert corrected['refraction_deg'][0] == pytest.approx(
        1.02 / np.tan(np.radians(45 + 10.3 / 50.11)) / 60)
    assert corrected['height'][0] < raw['height'][0]
    assert not np.allclose(raw['pos'], corrected['pos'])


def test_delta_z_matches_published_ray_tracing_examples():
    import refraction as refr
    assert refr.delta_z_m(0.) == pytest.approx(2040., abs=3.)
    assert refr.delta_z_m(1.) == pytest.approx(1153., abs=2.)
    assert refr.delta_z_m(10.) == pytest.approx(71., abs=2.)
    assert refr.delta_z_m(20.) == pytest.approx(20., abs=2.)
    assert refr.delta_z_m(0., 1000.) == pytest.approx(1831., abs=2.)
    assert refr.delta_z_m(-0.5, 1000.) == pytest.approx(2530., abs=2.)
    assert refr.refraction_deg_apparent_minus_true(-0.5) == pytest.approx(
        0.5980, abs=1e-4)


def test_metrack_delta_z_offsets_preserve_reported_site_height():
    import metrack as mt
    obs = {'altitudes': np.array([10.]), 'heights_m': np.array([0.04])}
    corrected = mt._refraction_correct_obs_data(obs)
    assert corrected['altitudes'][0] == pytest.approx(
        10. - mt.refraction_deg_apparent_minus_true(10.))
    assert corrected['site_height_offsets_km'][0] == pytest.approx(
        mt.delta_z_m(10., 40.) / 1000.)
    assert corrected['heights_m'][0] == pytest.approx(0.04)
    assert mt._effective_site_heights_km(corrected)[0] == pytest.approx(
        0.04 + corrected['site_height_offsets_km'][0])


def test_metrack_ecliptic_angles_are_pickleable_numbers():
    import metrack as mt
    import pickle
    lon, lat = mt.radec_to_ecliptic('4:00:00', '-63:00:00')
    assert isinstance(lon, float) and isinstance(lat, float)
    assert pickle.loads(pickle.dumps((lon, lat))) == pytest.approx((lon, lat))


def test_luminous_track_sampling_covers_entire_segment():
    import darkflight as df
    lon, lat, h = df._straight_track_llh((9.8, 60.2, 90000.),
                                       (10.1, 60.0, 30000.), n=64)
    assert len(lon) == len(lat) == len(h) == 64
    assert lon[0] == pytest.approx(9.8) and lon[-1] == pytest.approx(10.1)
    assert lat[0] == pytest.approx(60.2) and lat[-1] == pytest.approx(60.0)
    assert h[0] == pytest.approx(90000.) and h[-1] == pytest.approx(30000.)
    assert np.all(np.diff(h) < 0)


def test_track_plot_clips_to_40_km_with_interpolated_boundary():
    import darkflight as df
    lon, lat, h = df._clip_track_max_h([0., 2., 4.], [0., 2., 4.],
                                       [80000., 20000., 0.])
    assert h[0] == pytest.approx(40000.)
    assert lon[0] == pytest.approx(4. / 3.) and lat[0] == pytest.approx(4. / 3.)
    assert np.all(h <= 40000.) and h[-1] == 0.


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


def test_ablation_coeff_iron_density_ramp():
    import darkflight as df
    cd = df.cd_hypersonic(1.4)
    old_coeff = 0.07e-6 * cd
    assert ablation_coeff(7400., 1.4) == pytest.approx(5.50e-8)
    assert ablation_coeff(6950., 1.4) == pytest.approx(
        old_coeff + 0.5 * (5.50e-8 - old_coeff))
    assert ablation_coeff(6500., 1.4) == pytest.approx(old_coeff)
    assert ablation_coeff(6000., 1.4) == pytest.approx(old_coeff)


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


def test_propagate_accepts_ablation_scale():
    atm = WindAtmosphere()
    r0 = llh2ecef(10.0, 60.0, 30000.0)
    v0 = np.array([500.0, 0.0, -3500.0])
    res = propagate(r0, v0, 0.5, 1500.0, 1.4, atm, 0.0,
                    erode=True, ablation_scale=0.0)
    assert np.ptp(res['m']) == pytest.approx(0.0)


def test_fit_ablation_score_helpers():
    import fit_ablation as fa
    args = fa.parse_args([
        'event', '--density', '7400', '--recovered-mass', '13.8',
        '--entry-mass', '100'])
    options = fa.model_options_from_args(args, 5.5e-8, 1.1e-7)
    assert options['ablation_scale'] == pytest.approx(0.5)
    exact = fa.mass_score({'largest_landed_kg': 13.8}, args)
    assert exact['chi2'] == pytest.approx(0.0)
    missing = fa.mass_score({'largest_landed_kg': 0.0}, args)
    assert missing['chi2'] == pytest.approx(100.0)
    assert fa._boundary_hit(8e-8, 1e-9, 8e-8) == 'upper'
    assert fa._boundary_hit(1e-9, 1e-9, 8e-8) == 'lower'
    assert fa._boundary_hit(5e-8, 1e-9, 8e-8) is None


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
        burst_scenarios = scenarios + [
            {**scenarios[0], 'name': 'S1_fallline', 'label': 'Other'}]
        with patch.object(go.Figure, 'write_html', capture):
            assert df.write_map3d(burst_scenarios, (10.1, 60.1, 30000.), clouds,
                                  Path(td) / 'map.html',
                                  start_llh=(10.0, 60.2, 90000.))
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
    legend_names = [t.name for t in traces
                    if t.name and t.showlegend is not False]
    assert len(legend_names) == len(set(legend_names))
    luminous = next(t for t in traces if t.name == 'Luminous trajectory')
    assert luminous.z[0] == pytest.approx(40.)
    assert luminous.z[-1] == pytest.approx(30.)
    assert figures[0].layout.scene.zaxis.range == (0., 40.)
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
        fragmentation = Path(td) / 'fragmentation.json'
        table.write_text('stale')
        fragmentation.write_text('stale')
        result = df.run_darkflight(td, resdat, mc_runs=7)
        stored = json.loads((Path(td) / 'darkflight.json').read_text())
        assert not table.exists() and not fragmentation.exists()
        assert stored['reliability'] == 'unreliable'
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
