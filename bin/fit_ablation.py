#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit an event-specific ablation coefficient.

This diagnostic scans an absolute mass-loss coefficient sigma and evaluates the
same forward-entry model used by darkflight.py.  For every candidate it either
uses a fixed first-observed mass or solves mass against a peak magnitude, then
combines:

  * speed/deceleration residuals against the fbspd luminous-track fit,
  * normalized observed-camera light-curve residuals,
  * an optional broad recovered/largest-fragment mass constraint.

The result is intentionally an event diagnostic, not an automatic replacement
for the density-class coefficient in darkflight.ablation_coeff().
"""

import argparse
import json
import logging
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import darkflight as df
import fbspd_merge as fb


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='Fit sigma to dynamics, camera light curve and an optional mass constraint.')
    ap.add_argument('event_dir', type=Path)
    ap.add_argument('--density', type=float, required=True,
                    help='Meteoroid density in kg/m3.')
    ap.add_argument('--model', choices=['single', 'discrete', 'disruption', 'erosion'],
                    default='single', help='Forward fragmentation model.')
    ap.add_argument('--entry-mass', type=float,
                    help='Fixed first-observed mass in kg; otherwise solve against --mag.')
    ap.add_argument('--mag', type=float,
                    help='Target absolute peak magnitude at 100 km for modelled mass inversion.')
    ap.add_argument('--sigma-min', type=float, default=1e-9,
                    help='Minimum sigma in s2/m2. Default: 1e-9')
    ap.add_argument('--sigma-max', type=float, default=8e-8,
                    help='Maximum sigma in s2/m2. Default: 8e-8')
    ap.add_argument('--grid', type=int, default=9,
                    help='Coarse logarithmic sigma grid points. Default: 9')
    ap.add_argument('--refine-grid', type=int, default=5,
                    help='Local refinement points around the best candidate. Default: 5')
    ap.add_argument('--recovered-mass', type=float,
                    help='Optional recovered/largest-fragment mass in kg.')
    ap.add_argument('--recovered-mass-log-sigma', type=float, default=0.5,
                    help='Natural-log uncertainty used for --recovered-mass. Default: 0.5')
    ap.add_argument('--speed-sigma', type=float, default=500.0,
                    help='Speed-residual scale in m/s. Default: 500')
    ap.add_argument('--accel-sigma', type=float, default=100.0,
                    help='Deceleration-residual scale in m/s2. Default: 100')
    ap.add_argument('--light-weight', type=float, default=1.0)
    ap.add_argument('--dynamic-weight', type=float, default=1.0)
    ap.add_argument('--mass-weight', type=float, default=1.0)
    ap.add_argument('--strength', type=float, default=1.0,
                    help='Fragmentation strength in MPa. Default: 1.0')
    ap.add_argument('--breakup-height', type=float,
                    help='Optional first breakup height in km.')
    ap.add_argument('--fragment-count', type=int, default=8)
    ap.add_argument('--strength-exponent', type=float, default=0.3)
    ap.add_argument('--erosion-coeff', type=float, default=1e-7)
    ap.add_argument('--luminous-efficiency', type=float, default=0.03)
    ap.add_argument('--entry-max-step', type=float, default=0.05)
    ap.add_argument('--entry-record-dt', type=float, default=0.02)
    ap.add_argument('--entry-max-evaluations', type=int, default=4000000)
    ap.add_argument('--output', type=Path,
                    help='Optional JSON report path; the event products are never overwritten.')
    args = ap.parse_args(argv)
    if args.sigma_min <= 0 or args.sigma_max <= args.sigma_min:
        ap.error('--sigma-min must be positive and below --sigma-max')
    if args.grid < 2:
        ap.error('--grid must be at least 2')
    if args.refine_grid < 0:
        ap.error('--refine-grid must be nonnegative')
    if args.entry_mass is not None and args.entry_mass <= 0:
        ap.error('--entry-mass must be positive')
    if args.recovered_mass is not None and args.recovered_mass <= 0:
        ap.error('--recovered-mass must be positive')
    if args.recovered_mass_log_sigma <= 0:
        ap.error('--recovered-mass-log-sigma must be positive')
    return args


def load_event_inputs(event_dir):
    """Load trajectory, speed profile, atmosphere and observed camera samples."""
    event_dir = Path(event_dir).resolve()
    res_files = sorted(event_dir.glob('obs_*.res'))
    if not res_files:
        raise FileNotFoundError('No obs_*.res file in ' + str(event_dir))
    resdat = fb.readres(str(res_files[0]))
    pkl = event_dir / '_fbspd_plot_data.pkl'
    if not pkl.exists():
        raise FileNotFoundError('No _fbspd_plot_data.pkl in ' + str(event_dir))
    with pkl.open('rb') as f:
        _, plot_data = pickle.load(f)
    if not plot_data or plot_data.get('final_params') is None:
        raise ValueError('Missing fitted speed profile in _fbspd_plot_data.pkl')

    params = np.asarray(plot_data['final_params'], dtype=float)
    merged = plot_data.get('final_merged_data') or {}
    times = np.asarray(merged.get('reltime', []), dtype=float)
    if (params.shape != (4,) or times.size < 2 or not np.all(np.isfinite(params))
            or not np.all(np.isfinite(times)) or np.ptp(times) <= 0):
        raise ValueError('Invalid speed-fit parameters or observed times')
    endpoints = np.array([[resdat.long1[i], resdat.lat1[i], resdat.height[i] * 1000.]
                          for i in range(2)], dtype=float)
    if not 0 <= endpoints[1, 2] < endpoints[0, 2] <= 120000.:
        raise ValueError('Forward fit requires a descending observed trajectory')
    r_start, r_end = [df.llh2ecef(*point) for point in endpoints]
    direction = r_end - r_start
    direction /= np.linalg.norm(direction)
    speed = float(fb.expfunc_1stder(float(times.min()), *params)) * 1000.0

    from datetime import datetime, timezone
    try:
        event_dt = datetime.strptime(
            f'{event_dir.parent.name} {event_dir.name}',
            '%Y%m%d %H%M%S').replace(tzinfo=timezone.utc)
    except ValueError:
        event_dt = None
    wind_csv = event_dir / 'wind_profile.csv'
    atm = df.WindAtmosphere(
        wind_csv if wind_csv.exists() else None,
        lon=endpoints[0, 0], lat=endpoints[0, 1], event_dt=event_dt)
    ground = df.ground_elevation(endpoints[1, 1], endpoints[1, 0])

    observed_photometry = None
    try:
        import photometry as ph
        observed_photometry = ph.event_photometry(
            event_dir, resdat=resdat, plot_data=plot_data)
        observed, observed_reason = df._observed_light_curve_samples(
            observed_photometry, plot_data, resdat)
    except Exception as exc:
        observed, observed_reason = [], f'photometry unavailable: {exc}'
    peak_estimate = df._observed_peak_magnitude_estimate(observed)

    return {
        'event_dir': event_dir, 'resdat': resdat, 'plot_data': plot_data,
        'params': params, 'times': times, 'endpoints': endpoints,
        'r_start': r_start, 'direction': direction, 'speed': speed,
        'ground': ground, 'atm': atm, 'observed_photometry': observed_photometry,
        'observed': observed, 'observed_reason': observed_reason,
        'peak_estimate': peak_estimate,
    }


def model_options_from_args(args, sigma, base_sigma):
    return {
        'strength_pa': args.strength * 1e6,
        'breakup_height_m': (None if args.breakup_height is None
                             else args.breakup_height * 1000.),
        'fragment_count': args.fragment_count,
        'strength_exponent': args.strength_exponent,
        'erosion_coeff': args.erosion_coeff,
        'ablation_scale': float(sigma) / float(base_sigma),
        'luminous_efficiency': args.luminous_efficiency,
        'max_step': args.entry_max_step,
        'record_dt': args.entry_record_dt,
        'max_evaluations': args.entry_max_evaluations,
    }


def dynamic_score(inputs, args, mass, sigma, base_sigma):
    """Compare modeled ground-relative speed and deceleration with fbspd."""
    if args.model != 'single':
        return {'available': False,
                'unavailable_reason': 'dynamic score currently supports the intact single-object model'}
    scale = float(sigma) / float(base_sigma)
    path = df.propagate(
        inputs['r_start'], inputs['direction'] * inputs['speed'], mass,
        args.density, 1.4, inputs['atm'], inputs['ground'], erode=True,
        record_dt=min(args.entry_record_dt, 0.02), ablation_scale=scale)
    model_t = np.asarray(path['t'], dtype=float)
    model_v = np.asarray(path['v'], dtype=float)
    if model_t.size < 2 or np.ptp(model_t) <= 0:
        raise RuntimeError('Dynamic propagation produced no usable speed history')
    rel_times = np.asarray(inputs['times'], dtype=float) - float(inputs['times'].min())
    in_window = rel_times <= float(model_t[-1])
    coverage = float(np.mean(in_window))
    speed_obs = np.asarray(fb.expfunc_1stder(inputs['times'], *inputs['params']),
                           dtype=float) * 1000.0
    accel_obs = -np.asarray(fb.expfunc_2ndder(inputs['times'], *inputs['params']),
                            dtype=float) * 1000.0
    speed_model = np.interp(rel_times, model_t, model_v,
                            left=np.nan, right=np.nan)
    accel_model = -np.gradient(model_v, model_t)
    accel_model = np.interp(rel_times, model_t, accel_model,
                            left=np.nan, right=np.nan)
    speed_residual = speed_model - speed_obs
    accel_residual = accel_model - accel_obs
    speed_chi = float(np.nanmean((speed_residual[in_window] /
                                  args.speed_sigma) ** 2))
    accel_chi = float(np.nanmean((accel_residual[in_window] /
                                  args.accel_sigma) ** 2))
    missing_penalty = float(10.0 * (1.0 - coverage))
    return {
        'available': coverage > 0,
        'coverage_fraction': coverage,
        'speed_chi2': speed_chi,
        'accel_chi2': accel_chi,
        'missing_coverage_penalty': missing_penalty,
        'chi2': speed_chi + accel_chi + missing_penalty,
        'speed_rms_ms': float(np.sqrt(np.nanmean(speed_residual[in_window] ** 2))),
        'accel_rms_ms2': float(np.sqrt(np.nanmean(accel_residual[in_window] ** 2))),
        'speed_model_end_ms': float(speed_model[-1]),
        'speed_observed_end_ms': float(speed_obs[-1]),
        'accel_model_end_ms2': float(accel_model[-1]),
        'accel_observed_end_ms2': float(accel_obs[-1]),
    }


def light_curve_score(experiment, observed):
    score = df._light_curve_shape_scores([experiment], observed)[0]
    available = bool(score['available'] and math.isfinite(score['log_likelihood']))
    return {
        'available': available,
        'log_likelihood': score['log_likelihood'] if available else None,
        'chi2': -2.0 * score['log_likelihood'] if available else None,
        'rms_log10': score['rms_log10'],
        'median_abs_log10': score['median_abs_log10'],
        'terminal_mismatch': score.get('terminal_mismatch'),
    }


def mass_score(experiment, args):
    if args.recovered_mass is None:
        return {'available': False, 'chi2': 0.0}
    largest = experiment.get('largest_landed_kg')
    if largest is None or not np.isfinite(largest) or largest <= 0:
        residual = 10.0
    else:
        residual = (math.log(largest / args.recovered_mass) /
                    args.recovered_mass_log_sigma)
    return {'available': True,
            'largest_landed_kg': largest,
            'target_landed_kg': args.recovered_mass,
            'residual_log_sigma': residual,
            'chi2': residual ** 2}


def _boundary_hit(sigma, lower, upper, rel_tol=1e-6):
    """Classify whether a grid optimum is constrained by an outer bound."""
    if math.isclose(float(sigma), float(lower), rel_tol=rel_tol, abs_tol=1e-18):
        return 'lower'
    if math.isclose(float(sigma), float(upper), rel_tol=rel_tol, abs_tol=1e-18):
        return 'upper'
    return None


def evaluate_sigma(inputs, args, sigma, base_sigma):
    options = model_options_from_args(args, sigma, base_sigma)

    def propagate_one(mass):
        return df.propagate_fragmentation(
            inputs['r_start'], inputs['direction'] * inputs['speed'],
            float(mass), args.density, 1.4, inputs['atm'], inputs['ground'],
            model=args.model, **options)

    if args.entry_mass is not None:
        mass = float(args.entry_mass)
        experiment = propagate_one(mass)
        inversion = {'solved_entry_mass_kg': None,
                     'fixed_entry_mass_kg': mass}
    else:
        experiment, inversion = df._solve_fragmentation_mass_for_magnitude(
            args.model, propagate_one, inputs['target_mag'],
            inputs['speed'], float(np.ptp(inputs['times'])))
        mass = float(inversion['solved_entry_mass_kg'])

    dynamic = dynamic_score(inputs, args, mass, sigma, base_sigma)
    light = light_curve_score(experiment, inputs['observed'])
    mass_fit = mass_score(experiment, args)
    dynamic_chi = dynamic.get('chi2', 0.0) if dynamic.get('available') else 0.0
    light_chi = light.get('chi2') or 0.0
    total = (args.dynamic_weight * dynamic_chi +
             args.light_weight * light_chi +
             args.mass_weight * mass_fit['chi2'])
    budget = experiment.get('mass_budget') or {}
    return {
        'sigma_s2_m2': float(sigma),
        'ablation_scale_vs_current': float(sigma) / float(base_sigma),
        'entry_mass_kg': mass,
        'landed_kg': budget.get('landed_kg'),
        'unresolved_kg': budget.get('unresolved_kg'),
        'largest_landed_kg': experiment.get('largest_landed_kg'),
        'peak_absolute_magnitude': experiment.get('peak_absolute_magnitude'),
        'total_score': float(total),
        'dynamic': dynamic,
        'light_curve': light,
        'mass_fit': mass_fit,
        'entry_mass_inversion': inversion,
    }


def fit_ablation(inputs, args):
    """Run coarse log-sigma grid plus one local refinement pass."""
    base_sigma = df.ablation_coeff(args.density, 1.4)
    if base_sigma <= 0 or not np.isfinite(base_sigma):
        raise ValueError('Cannot scan an ablation coefficient below the material class floor')
    if args.entry_mass is None:
        target_mag = args.mag
        if target_mag is None and inputs['peak_estimate'] is not None:
            target_mag = inputs['peak_estimate'][
                'absolute_magnitude_at_100_km']
        if target_mag is None or not np.isfinite(target_mag) or target_mag >= -1.0:
            raise ValueError('Supply --entry-mass or a usable --mag brighter than -1')
        inputs['target_mag'] = float(target_mag)
    candidates = []
    log_min, log_max = math.log(args.sigma_min), math.log(args.sigma_max)
    for sigma in np.exp(np.linspace(log_min, log_max, args.grid)):
        try:
            candidates.append(evaluate_sigma(inputs, args, float(sigma), base_sigma))
        except Exception as exc:
            candidates.append({'sigma_s2_m2': float(sigma),
                               'total_score': float('inf'),
                               'error': f'{type(exc).__name__}: {exc}'})
        logging.info('sigma %.3g -> %s', sigma,
                     ('error' if 'error' in candidates[-1]
                      else f"score {candidates[-1]['total_score']:.3f}"))
    finite = [c for c in candidates if math.isfinite(c['total_score'])]
    if not finite:
        raise RuntimeError('No ablation-coefficient candidate evaluated successfully')
    best = min(finite, key=lambda c: c['total_score'])
    index = next(i for i, candidate in enumerate(candidates) if candidate is best)
    lo_index = max(0, index - 1)
    hi_index = min(len(candidates) - 1, index + 1)
    lo = math.log(candidates[lo_index]['sigma_s2_m2'])
    hi = math.log(candidates[hi_index]['sigma_s2_m2'])
    for sigma in np.exp(np.linspace(lo, hi, args.refine_grid + 2)[1:-1]):
        if any(abs(candidate['sigma_s2_m2'] / sigma - 1.0) < 1e-12
               for candidate in candidates):
            continue
        try:
            candidates.append(evaluate_sigma(inputs, args, float(sigma), base_sigma))
        except Exception as exc:
            candidates.append({'sigma_s2_m2': float(sigma),
                               'total_score': float('inf'),
                               'error': f'{type(exc).__name__}: {exc}'})
    candidates.sort(key=lambda c: (not math.isfinite(c['total_score']),
                                   c['total_score'],
                                   c['sigma_s2_m2']))
    finite = [c for c in candidates if math.isfinite(c['total_score'])]
    best = min(finite, key=lambda c: c['total_score'])
    boundary = _boundary_hit(best['sigma_s2_m2'], args.sigma_min,
                           args.sigma_max)
    return {'event_dir': str(inputs['event_dir']), 'model': args.model,
            'density_kg_m3': args.density,
            'current_sigma_s2_m2': float(base_sigma),
            'target_peak_magnitude': inputs.get('target_mag'),
            'solution_status': ('boundary_limited' if boundary else 'interior'),
            'accepted': boundary is None,
            'boundary': boundary,
            'photometry': {'available': bool(inputs['observed']),
                           'reason': inputs['observed_reason'],
                           'n_cameras': len(inputs['observed'])},
            'weights': {'light': args.light_weight,
                        'dynamic': args.dynamic_weight,
                        'mass': args.mass_weight},
            'best': best, 'candidates': candidates}


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    inputs = load_event_inputs(args.event_dir)
    report = fit_ablation(inputs, args)
    label = ('best sigma' if report['accepted']
             else f"boundary-limited {report['boundary']} candidate")
    print(f"{label}: {report['best']['sigma_s2_m2']:.6g} s^2/m^2 "
          f"(scale {report['best']['ablation_scale_vs_current']:.4g})")
    if not report['accepted']:
        print('No accepted interior solution: extend the physical range or '
              'treat all candidates as diagnostics.')
    print(f"entry mass: {report['best']['entry_mass_kg']:.4g} kg, "
          f"landed: {report['best']['landed_kg']:.4g} kg, "
          f"largest: {report['best']['largest_landed_kg']:.4g} kg")
    print(f"total score: {report['best']['total_score']:.4g}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f'wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
