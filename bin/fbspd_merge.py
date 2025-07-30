#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fbspd_merge.py: Fits speed and acceleration profiles to meteor centroid data.

This script analyzes meteor trajectory data from multiple observation stations.
It reads trajectory solutions (.res), station coordinates (.dat), and centroid
observations (.cen) to compute a merged speed and acceleration profile for
a meteor event. This approach is robust to time-shift errors between stations.
"""

import argparse
import sys
import itertools
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares, minimize, brute
from scipy.stats import norm, median_abs_deviation
from scipy.interpolate import interp1d

# WGS-84 Ellipsoid Parameters
EARTH_RADIUS = 6371.0  # Mean radius in km
FLATTENING = 1.0 / 298.257223563  # Earth flattening factor (WGS-84)
E_SQUARED = 2 * FLATTENING - FLATTENING**2  # Eccentricity squared

# --- Configuration Constants ---
TIME_OFFSET_BRUTE_STEP = 0.25
MIN_POINTS_FOR_EXP_FIT = 4
XYZ_CONVERGENCE_TOLERANCE = 1e-12
PARALLEL_LINES_TOLERANCE = 1e-9
MIN_POSITIONAL_SIGMA = 1e-9
DAT_STATION_NAME_START_COLUMN = 12
CAMERA_DEGREES_PER_PIXEL = 0.25

class ResData:
    def __init__(self):
        self.ndata, self.long1, self.lat1, self.long2, self.lat2, self.height, self.desc = 0, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

class CenData:
    def __init__(self):
        self.ndata, self.seqid, self.reltime, self.cenalt, self.cenaz, self.censig, self.sitestr, self.datestr, self.timestr, self.site_info = 0, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), {}

def readres(inname: str) -> ResData:
    res_path, data = Path(inname), {"long1": [], "lat1": [], "long2": [], "lat2": [], "height": [], "desc": []}
    with res_path.open('r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            words = line.split()
            data["long1"].append(float(words[0])); data["lat1"].append(float(words[1])); data["long2"].append(float(words[2])); data["lat2"].append(float(words[3])); data["height"].append(float(words[4])); data["desc"].append(words[5])
    res = ResData()
    for k, v in data.items(): setattr(res, k, np.array(v))
    res.ndata = len(data["long1"])
    return res

def readcen(inname: str) -> CenData:
    cen_path, data = Path(inname), {"seqid": [], "reltime": [], "cenalt": [], "cenaz": [], "censig": [], "sitestr": [], "datestr": [], "timestr": []}
    with cen_path.open('r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            words = line.split()
            data["seqid"].append(int(words[0])); data["reltime"].append(float(words[1])); data["cenalt"].append(float(words[2])); data["cenaz"].append(float(words[3])); data["censig"].append(float(words[4])); data["sitestr"].append(words[5]); data["datestr"].append(words[6]); data["timestr"].append(words[7])
    cen = CenData()
    cen.ndata = len(data["seqid"])
    for k, v in data.items(): setattr(cen, k, np.array(v))
    return cen

def get_sitecoord_fromdat(inname: str) -> List[Tuple[str, float, float, float]]:
    dat_path, sitecoords = Path(inname), []
    with dat_path.open('r') as f:
        for line in f:
            words = line.split()
            if not words or words[0].startswith('#') or words[0] == 'borders': continue
            lon, lat, height_km, last_name_idx = float(words[0]), float(words[1]), 0.0, len(words)
            try:
                height_km = float(words[-1]) / 1000.0; last_name_idx = -2
            except (ValueError, IndexError): last_name_idx = -1
            name = " ".join(words[DAT_STATION_NAME_START_COLUMN:last_name_idx])
            sitecoords.append((name, lon, lat, height_km))
    return sitecoords

def lonlat2xyz(lon_deg: float, lat_deg: float, height_km: float = 0) -> np.ndarray:
    lat_rad, lon_rad = np.radians(lat_deg), np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    N = EARTH_RADIUS / np.sqrt(1 - E_SQUARED * sin_lat**2)
    return np.array([(N + height_km) * cos_lat * np.cos(lon_rad), (N + height_km) * cos_lat * np.sin(lon_rad), (N * (1 - E_SQUARED) + height_km) * sin_lat])

def xyz2lonlat(v: np.ndarray) -> Tuple[float, float, float]:
    p, lon_deg = np.sqrt(v[0]**2 + v[1]**2), np.degrees(np.arctan2(v[1], v[0]))
    lat_rad = np.arctan2(v[2], p * (1.0 - E_SQUARED))
    for _ in range(10):
        sin_lat = np.sin(lat_rad)
        N = EARTH_RADIUS / np.sqrt(1.0 - E_SQUARED * sin_lat**2)
        height_km = p / np.cos(lat_rad) - N
        lat_rad_new = np.arctan2(v[2], p * (1.0 - E_SQUARED * N / (N + height_km)))
        if abs(lat_rad_new - lat_rad) < XYZ_CONVERGENCE_TOLERANCE: break
        lat_rad = lat_rad_new
    return lon_deg, np.degrees(lat_rad), height_km

def altaz2xyz(alt_deg: float, az_deg: float, obs_lon: float, obs_lat: float) -> np.ndarray:
    alt_rad, az_rad, lat_rad, lon_rad = np.radians(alt_deg), np.radians(az_deg), np.radians(obs_lat), np.radians(obs_lon)
    v_enu = np.array([np.sin(az_rad) * np.cos(alt_rad), np.cos(az_rad) * np.cos(alt_rad), np.sin(alt_rad)])
    sin_lat, cos_lat, sin_lon, cos_lon = np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)
    R = np.array([[-sin_lon, -cos_lon * sin_lat, cos_lon * cos_lat], [cos_lon, -sin_lon * sin_lat, sin_lon * cos_lat], [0, cos_lat, sin_lat]])
    return R @ v_enu

def linfunc(x: float, a: float, b: float) -> float: return a * x + b
def expfunc(t: np.ndarray, v0: float, accel0: float, k: float, p0: float) -> np.ndarray:
    if abs(k) < 1e-6: return p0 + v0 * t + 0.5 * accel0 * t**2
    return p0 + (v0 - accel0 / k) * t + (accel0 / (k**2)) * (np.exp(k * t) - 1)
def expfunc_1stder(t: np.ndarray, v0: float, accel0: float, k: float, p0: float) -> np.ndarray:
    if abs(k) < 1e-6: return v0 + accel0 * t
    return (v0 - accel0 / k) + (accel0 / k) * np.exp(k * t)
def expfunc_2ndder(t: np.ndarray, v0: float, accel0: float, k: float, p0: float) -> np.ndarray:
    if abs(k) < 1e-6: return np.full_like(t, accel0)
    return accel0 * np.exp(k * t)

def _calculate_quadratic_fit_error(offsets: np.ndarray, obs_list: List[Dict]) -> float:
    if not obs_list: return 1e12
    all_time, all_pos = [obs_list[0]['reltime']], [obs_list[0]['pos']]
    for i, obs in enumerate(obs_list[1:]): all_time.append(obs['reltime'] + offsets[i]); all_pos.append(obs['pos'])
    time_arr, pos_arr = np.concatenate(all_time), np.concatenate(all_pos)
    sort_idx = np.argsort(time_arr)
    time_arr, pos_arr = time_arr[sort_idx], pos_arr[sort_idx]
    try: return np.sum((pos_arr - np.polyval(np.polyfit(time_arr, pos_arr, 2), time_arr))**2)
    except (np.linalg.LinAlgError, ValueError): return 1e12

def _calculate_linear_fit_error(offsets: np.ndarray, obs_list: List[Dict]) -> float:
    """More stable cost function for sparse data, using a linear fit."""
    if not obs_list: return 1e12
    all_time, all_pos = [obs_list[0]['reltime']], [obs_list[0]['pos']]
    for i, obs in enumerate(obs_list[1:]): all_time.append(obs['reltime'] + offsets[i]); all_pos.append(obs['pos'])
    time_arr, pos_arr = np.concatenate(all_time), np.concatenate(all_pos)
    sort_idx = np.argsort(time_arr)
    time_arr, pos_arr = time_arr[sort_idx], pos_arr[sort_idx]
    try: return np.sum((pos_arr - np.polyval(np.polyfit(time_arr, pos_arr, 1), time_arr))**2)
    except (np.linalg.LinAlgError, ValueError): return 1e12

def find_offsets_by_physical_fit(station_obs: List[Dict], debug: bool = False) -> np.ndarray:
    num_stations = len(station_obs)
    if num_stations < 2: return np.zeros(num_stations)
    original_station_order_map = {obs['site_info']['name']: i for i, obs in enumerate(station_obs)}
    station_obs.sort(key=lambda obs: obs['reltime'].min())
    ref_obs, non_ref_obs = station_obs[0], station_obs[1:]
    ref_name = ref_obs.get('site_info', {}).get('name', 'Unknown')
    if debug: print(f"Anchoring time offsets to reference station: '{ref_name}'")
    
    A_rows, b_vals, num_unknowns = [], [], num_stations - 1
    
    if num_stations >= 3:
        all_triplets = list(itertools.combinations(range(num_stations), 3))
        if debug: print(f"Using robust triplet method. Performing {len(all_triplets)} 2D brute-force searches.")
        for idx_i, idx_j, idx_k in all_triplets:
            s_i, s_j, s_k = station_obs[idx_i], station_obs[idx_j], station_obs[idx_k]
            def get_range(s_ref, s_off): return slice(s_ref['reltime'].min()-s_off['reltime'].max(), s_ref['reltime'].max()-s_off['reltime'].min(), TIME_OFFSET_BRUTE_STEP)
            range_j, range_k = get_range(s_i, s_j), get_range(s_i, s_k)
            offsets = brute(lambda p: _calculate_quadratic_fit_error(p, [s_i, s_j, s_k]), (range_j, range_k), finish=None)
            for offset, s_idx_off in [(offsets[0], idx_j), (offsets[1], idx_k)]:
                row = np.zeros(num_unknowns)
                if idx_i == 0: row[s_idx_off - 1] = 1.0
                elif s_idx_off == 0: row[idx_i - 1] = -1.0
                else: row[s_idx_off - 1], row[idx_i - 1] = 1.0, -1.0
                A_rows.append(row); b_vals.append(offset)
    else: # num_stations == 2
        if debug: print("Fewer than 3 stations. Using robust linear fit for pairwise search.")
        s_i, s_j = station_obs[0], station_obs[1]
        brute_range = slice(s_i['reltime'].min() - s_j['reltime'].max(), s_i['reltime'].max() - s_j['reltime'].min(), TIME_OFFSET_BRUTE_STEP)
        offset_j_vs_i = brute(lambda p: _calculate_linear_fit_error([p], [s_i, s_j]), (brute_range,), finish=None)
        row = np.zeros(num_unknowns); row[0] = 1.0
        A_rows.append(row); b_vals.append(offset_j_vs_i)

    coarse_offsets, _, _, _ = np.linalg.lstsq(np.array(A_rows), np.array(b_vals), rcond=None)
    if debug: print(f"Least-squares on sub-problems found initial guess: {coarse_offsets}")
    result = minimize(lambda p: _calculate_quadratic_fit_error(p, [ref_obs] + non_ref_obs), x0=coarse_offsets, method='L-BFGS-B')
    fine_offsets = result.x if result.success else coarse_offsets
    if not result.success: print(f"Warning: Fine-grained offset optimization may have failed: {result.message}")
    
    final_offsets_map = {ref_name: 0.0}
    for i, obs in enumerate(non_ref_obs): final_offsets_map[obs['site_info']['name']] = fine_offsets[i]
    final_offsets_array = np.zeros(num_stations)
    for name, offset in final_offsets_map.items(): final_offsets_array[original_station_order_map[name]] = offset
    return final_offsets_array

def _refine_offsets_by_projection(station_obs: List[Dict], initial_offsets: np.ndarray, debug: bool = False) -> np.ndarray:
    if len(station_obs) < 2: return initial_offsets
    
    merged_data, station_id_arr = _merge_and_sort_station_data(station_obs, initial_offsets)
    fit_params, n_ok, _ = _fit_merged_data(merged_data['reltime'], merged_data['pos'], merged_data['sig'], debug, 0.1)
    if n_ok == 0: return initial_offsets

    residuals = merged_data['pos'] - expfunc(merged_data['reltime'], *fit_params)
    mad = median_abs_deviation(residuals, scale='normal')
    if mad < 1e-9: return initial_offsets
    
    is_point_outlier = np.abs(residuals) > 3 * mad
    
    inlier_station_idxs, outlier_station_idxs = set(), set()
    for i in range(len(station_obs)):
        station_mask = (station_id_arr == i)
        if np.sum(station_mask) > 0 and np.mean(is_point_outlier[station_mask]) > 0.5: outlier_station_idxs.add(i)
        else: inlier_station_idxs.add(i)

    if not outlier_station_idxs or not inlier_station_idxs:
        if debug: print("Projection Timing: No clear inlier/outlier split found. Using initial solution.")
        return initial_offsets

    if debug:
        outlier_names = [station_obs[i]['site_info']['name'] for i in outlier_station_idxs]
        print(f"Projection Timing: Identified {len(outlier_names)} potential outlier stations for re-timing: {outlier_names}")

    inlier_obs = [station_obs[i] for i in inlier_station_idxs]
    inlier_offsets = initial_offsets[list(inlier_station_idxs)]
    core_merged_data, _ = _merge_and_sort_station_data(inlier_obs, inlier_offsets)
    core_params, n_ok, _ = _fit_merged_data(core_merged_data['reltime'], core_merged_data['pos'], core_merged_data['sig'], debug, 0.1)
    if n_ok == 0: return initial_offsets
    
    t_dense = np.linspace(core_merged_data['reltime'].min(), core_merged_data['reltime'].max(), 2000)
    pos_dense = expfunc(t_dense, *core_params)
    sort_idx = np.argsort(pos_dense)
    time_from_position = interp1d(pos_dense[sort_idx], t_dense[sort_idx], bounds_error=False, fill_value="extrapolate")

    final_offsets = np.copy(initial_offsets)
    for idx in outlier_station_idxs:
        outlier_station = station_obs[idx]
        if outlier_station['pos'].size == 0: continue
        
        corrected_time = time_from_position(outlier_station['pos'])
        
        if np.isnan(corrected_time).any():
            if debug: print(f"  -> Failed to re-time '{outlier_station['site_info']['name']}' due to extrapolation error.")
            continue

        new_offset = np.mean(corrected_time - outlier_station['reltime'])
        final_offsets[idx] = new_offset
        if debug: print(f"  -> Successfully re-timed '{outlier_station['site_info']['name']}'. New offset: {new_offset:.2f}s")
        
    return final_offsets

def _merge_and_sort_station_data(station_obs: List[Dict], time_offsets: Optional[np.ndarray] = None) -> Tuple[Dict, np.ndarray]:
    if not station_obs: return {}, np.array([])
    
    if time_offsets is None: time_offsets = np.zeros(len(station_obs))
    
    all_reltime, all_colortime, all_pos, all_height, all_sig, all_station_ids = [], [], [], [], [], []
    for i, (obs, offset) in enumerate(zip(station_obs, time_offsets)):
        num_pts = len(obs['reltime'])
        if num_pts == 0: continue
        all_reltime.append(obs['reltime'] + offset)
        all_colortime.append(obs['reltime'])
        all_pos.append(obs['pos'])
        all_height.append(obs['height'])
        all_sig.append(obs['sig'])
        all_station_ids.append(np.full(num_pts, i, dtype=int))

    if not all_reltime: return {k: np.array([]) for k in ['reltime', 'color_time', 'pos', 'height', 'sig']}, np.array([])
    
    merged_data = {
        'reltime': np.concatenate(all_reltime), 'color_time': np.concatenate(all_colortime),
        'pos': np.concatenate(all_pos), 'height': np.concatenate(all_height),
        'sig': np.concatenate(all_sig),
    }
    station_id_arr = np.concatenate(all_station_ids)
    
    if merged_data['reltime'].size == 0: return merged_data, station_id_arr
    
    sort_idx = np.argsort(merged_data['reltime'])
    for key in merged_data: merged_data[key] = merged_data[key][sort_idx]
    station_id_arr = station_id_arr[sort_idx]
        
    return merged_data, station_id_arr

def _process_station(cendat: CenData, site_pos: np.ndarray, track_p1: np.ndarray,
                     track_vec_norm: np.ndarray) -> dict:
    site_lon, site_lat = cendat.site_info['lon'], cendat.site_info['lat']
    los_vecs = np.array([altaz2xyz(alt, az, site_lon, site_lat) for alt, az in zip(cendat.cenalt, cendat.cenaz)])
    p21 = track_p1 - site_pos
    m = np.cross(track_vec_norm, los_vecs)
    m2 = np.einsum('ij,ij->i', m, m)
    m2[m2 < PARALLEL_LINES_TOLERANCE] = PARALLEL_LINES_TOLERANCE
    R = np.cross(p21, m / m2[:, np.newaxis])
    t1 = np.einsum('ij,j->i', R, track_vec_norm)
    t2 = np.einsum('ij,ij->i', R, los_vecs)
    cross_points1 = site_pos + t1[:, np.newaxis] * los_vecs
    cross_points2 = track_p1 + t2[:, np.newaxis] * track_vec_norm
    intersec_on_track = (cross_points1 + cross_points2) / 2.0
    pos = np.einsum('ij,j->i', intersec_on_track - track_p1, track_vec_norm)
    height = np.array([xyz2lonlat(v)[2] for v in intersec_on_track])
    sitedist = np.linalg.norm(intersec_on_track - site_pos, axis=1)
    ang_err = np.radians(CAMERA_DEGREES_PER_PIXEL * cendat.censig)
    sig = sitedist * np.tan(ang_err)
    sig[ang_err <= 0] = MIN_POSITIONAL_SIGMA
    return {"pos": pos, "height": height, "sig": sig, "site_info": cendat.site_info}


def _fit_merged_data(reltime: np.ndarray, pos: np.ndarray, sig: np.ndarray,
                     debug: bool, fscale: float) -> Tuple[Optional[np.ndarray], int, Optional[np.ndarray]]:
    n_pts = len(reltime)
    if n_pts < MIN_POINTS_FOR_EXP_FIT:
        print("Error: Not enough data points to perform a fit.")
        return None, 0, None

    def residual_func(params, x, y, sig, model_func): return (y - model_func(x, *params)) / sig
    def get_pcov(res, n, p):
        if n <= p: return None
        try:
            mse = 2 * res.cost / (n - p)
            return np.linalg.pinv(res.jac.T @ res.jac) * mse
        except (np.linalg.LinAlgError, ZeroDivisionError): return None

    try:
        lin_params_guess, _ = curve_fit(linfunc, reltime, pos, sigma=1./sig)
        v0_guess, p0_guess = lin_params_guess[0], lin_params_guess[1]
    except RuntimeError: v0_guess, p0_guess = 40.0, pos[0] if len(pos) > 0 else 0

    exp_params_guess = [v0_guess, -10.0, 1.0, p0_guess]
    exp_bounds = ([8.0, -500.0, 1e-6, -np.inf], [73.0, -1e-9, 10.0, np.inf])

    res_exp = least_squares(residual_func, exp_params_guess, loss='soft_l1', f_scale=fscale, args=(reltime, pos, sig, expfunc), bounds=exp_bounds, max_nfev=5000)

    res_lin = None
    try:
        lin_params_initial_guess, _ = curve_fit(linfunc, reltime, pos, sigma=1./sig)
        res_lin = least_squares(residual_func, lin_params_initial_guess, loss='soft_l1', f_scale=fscale, args=(reltime, pos, sig, linfunc))
    except (RuntimeError, ValueError):
        if not res_exp.success:
            print("Error: Both exponential and linear fits failed.")
            return None, 0, None

    if res_exp.success and (res_lin is None or not res_lin.success or res_exp.cost < res_lin.cost):
        if debug: print("Selected robust exponential model on all points.")
        final_params, pcov = res_exp.x, get_pcov(res_exp, n_pts, len(res_exp.x))
        return final_params, n_pts, pcov
    
    if res_lin and res_lin.success:
        if debug: print("Selected robust linear model on all points as fallback.")
        lin_params = res_lin.x
        final_params = np.array([lin_params[0], 0, 0, lin_params[1]])
        lin_pcov = get_pcov(res_lin, n_pts, len(lin_params))
        final_pcov = None
        if lin_pcov is not None:
            final_pcov = np.zeros((4, 4)); final_pcov[0, 0], final_pcov[3, 3] = lin_pcov[0, 0], lin_pcov[1, 1]; final_pcov[0, 3], final_pcov[3, 0] = lin_pcov[0, 1], lin_pcov[1, 0]
        return final_params, n_pts, final_pcov

    print("Error: Could not find a successful fit for the data.")
    return None, 0, None


def _generate_plots(data: dict, params: np.ndarray, 
                    lower_params: Optional[np.ndarray], upper_params: Optional[np.ndarray], 
                    n_ok: int, doplot: str, debug: bool = False, sigma_level: float = 1.0,
                    station_id_array: Optional[np.ndarray] = None, exclude_station_idx: Optional[int] = None):
    plot_mask = np.full(len(data['reltime']), True)
    if exclude_station_idx is not None and station_id_array is not None:
        plot_mask = (station_id_array != exclude_station_idx)

    reltime_plot, pos_plot, color_time_plot, sig_plot = data['reltime'][plot_mask], data['pos'][plot_mask], data['color_time'][plot_mask], data['sig'][plot_mask]
    residuals_plot = (pos_plot - expfunc(reltime_plot, *params))

    t_fit = np.linspace(data['reltime'].min(), data['reltime'].max(), 300)
    fit_speed, fit_accel, fit_pos = expfunc_1stder(t_fit, *params), expfunc_2ndder(t_fit, *params), expfunc(t_fit, *params)
    
    try:
        avg_time_step = np.mean(np.diff(np.unique(data['reltime'])))
        window_size = min(max(3, int(0.15 / avg_time_step)), len(data['reltime']) - 1)
        if window_size % 2 == 0: window_size += 1
        sig_smoothed = np.convolve(data['sig'], np.ones(window_size)/window_size, mode='same')[plot_mask]
    except (ValueError, IndexError): sig_smoothed = sig_plot

    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig1.suptitle("Atmosfærisk bane", fontsize=16, fontstyle="oblique")
    sc = ax1.scatter(reltime_plot, pos_plot, c=color_time_plot, cmap='viridis', s=10, label='Enkeltobservasjoner')
    ax1.plot(t_fit, fit_pos, 'r-', label='Estimert bane', linewidth=2)
    ax1.set_ylabel('Posisjonen langs banen [km]'); ax1.legend()
    fig1.subplots_adjust(right=0.85, top=0.92)
    cbar_ax = fig1.add_axes([0.88, 0.11, 0.03, 0.77])
    cbar = fig1.colorbar(sc, cax=cbar_ax); cbar.set_label('Tid [s]')
    ax2.axhline(0, color='r', linestyle='--', linewidth=1.5, zorder=4)
    ax2.scatter(reltime_plot, residuals_plot, c=color_time_plot, cmap='viridis', s=10, zorder=5)
    ax2.fill_between(reltime_plot, -sig_smoothed*sigma_level, sig_smoothed*sigma_level, color='blue', alpha=0.3, label=f'±{sigma_level:.0f}σ usikkerhet')
    ax2.set_ylabel('Residualer [km]'); ax2.set_xlabel('Tid [s]'); ax2.legend(loc='upper right')
    if doplot in ['save', 'both']: plt.savefig("posvstime.svg", bbox_inches='tight', pad_inches=0.05)

    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    fig2.suptitle("     Dynamisk analyse", fontsize=16, fontstyle="oblique")
    ax3.plot(t_fit, fit_speed, 'b-', zorder=10, label='Beste estimat'); ax3.set_ylim(bottom=0)
    ax3.set_ylabel('Hastighet [km/s]'); ax3.set_title('Hastighetsprofil')
    ax4.plot(t_fit, fit_accel, 'b-', zorder=10, label='Beste estimat')
    ax4.set_ylabel('Akselerasjon [km/s²]'); ax4.set_xlabel('Tid [s]'); ax4.set_title('Akselerasjonsprofil')

    if lower_params is not None and upper_params is not None:
        lower_speed, upper_speed = expfunc_1stder(t_fit, *lower_params), expfunc_1stder(t_fit, *upper_params)
        lower_accel, upper_accel = expfunc_2ndder(t_fit, *lower_params), expfunc_2ndder(t_fit, *upper_params)
        ax3.fill_between(t_fit, lower_speed, upper_speed, color='blue', alpha=0.2, label=f'±{sigma_level:.0f}σ usikkerhet')
        ax4.fill_between(t_fit, lower_accel, np.minimum(upper_accel, 0), color='blue', alpha=0.2, label=f'±{sigma_level:.0f}σ usikkerhet')
    ax3.legend(); ax4.legend()

    if doplot in ['save', 'both']: plt.savefig("spd_acc.svg", bbox_inches='tight', pad_inches=0.05)
    if doplot in ['show', 'both']: plt.show()

def fbspd(resname: str, cennames: List[str], datname: str,
          doplot: str = '', posdata: bool = False, debug: bool = False,
          fscale: float = 0.1, sigma_level: float = 1.0,
          seed: Optional[int] = None, num_simulations: int = 1000) -> Tuple[bool, float]:

    if seed is not None: np.random.seed(seed)
    if doplot and matplotlib.get_backend() != 'agg' and 'show' not in doplot: matplotlib.use('agg')

    resdat = readres(resname)
    all_sitedata = get_sitecoord_fromdat(datname) if datname else []
    all_cendat = [readcen(f) for f in cennames if Path(f).exists()]
    if not all_cendat:
        print("Error: No valid centroid data could be loaded."), sys.exit(1)

    path_p1 = lonlat2xyz(resdat.long1[0], resdat.lat1[0], resdat.height[0])
    path_p2 = lonlat2xyz(resdat.long1[1], resdat.lat1[1], resdat.height[1])
    path_vec_norm = (path_p2 - path_p1) / np.linalg.norm(path_p2 - path_p1)

    station_obs = []
    for cendat in all_cendat:
        site_info = next((s for s in all_sitedata if s[0] == cendat.sitestr[0]), None)
        if not site_info:
            print(f"Warning: No site coords for {cendat.sitestr[0]}. Skipping.")
            continue
        name, lon, lat, height = site_info
        cendat.site_info = {'name': name, 'lon': lon, 'lat': lat, 'height': height}
        processed_data = _process_station(cendat, lonlat2xyz(lon, lat, height), path_p1, path_vec_norm)
        processed_data['reltime'] = cendat.reltime
        station_obs.append(processed_data)

    if not station_obs:
        print("Error: Could not process any station data."), sys.exit(1)

    initial_offsets = find_offsets_by_physical_fit(station_obs, debug=debug)
    final_offsets = _refine_offsets_by_projection(station_obs, initial_offsets, debug=debug)

    final_merged_data, station_id_array = _merge_and_sort_station_data(station_obs, final_offsets)
    
    if final_merged_data['reltime'].size == 0:
        print("Error: No data points remaining after processing."), sys.exit(1)
        
    first_obs_time = final_merged_data['reltime'][0]
    final_merged_data['reltime'] -= first_obs_time
    if debug: print(f"Time axis shifted by {-first_obs_time:.4f}s to set t=0 at first observation.")

    final_params, n_ok, pcov = _fit_merged_data(final_merged_data['reltime'], final_merged_data['pos'], final_merged_data['sig'], debug, fscale)
    if n_ok == 0:
        print("Error: Final fit failed."), sys.exit(1)
    
    print("\n--- Per-Station Fit Error ---")
    station_errors, worst_station_idx = [], None
    if len(station_obs) > 1:
        for i, station in enumerate(station_obs):
            if len(station['pos']) == 0: continue
            offset = final_offsets[i]
            shifted_time = station['reltime'] + offset - first_obs_time
            pred_pos = expfunc(shifted_time, *final_params)
            mse = np.mean((station['pos'] - pred_pos)**2)
            station_errors.append({'name': station['site_info']['name'], 'index': i, 'mse': mse})
        
        station_errors.sort(key=lambda x: x['mse'], reverse=True)
        for s in station_errors: print(f"  Station: {s['name']:<15} MSE: {s['mse']:.4f} km^2")
        
        if station_errors:
            print(f"\n-> Station with highest error is '{station_errors[0]['name']}'.")
            if len(station_errors) > 2:
                worst_mse = station_errors[0]['mse']
                other_mses = np.array([s['mse'] for s in station_errors[1:]])
                threshold = np.mean(other_mses) + 10 * np.std(other_mses)
                if debug: print(f"   Outlier exclusion check: Worst MSE={worst_mse:.4f}, Threshold={threshold:.4f}")
                if worst_mse > threshold:
                    worst_station_idx = station_errors[0]['index']
    
    initial_speed = expfunc_1stder(0.0, *final_params)
    initial_speed_uncertainty, lower_bound_params, upper_bound_params = 0.0, None, None

    if pcov is not None and not np.isnan(pcov).any():
        try:
            param_samples = np.random.multivariate_normal(final_params, pcov, size=num_simulations)
            param_samples = param_samples[param_samples[:, 1] <= 0]
            if len(param_samples) > 10:
                std_v0, std_accel0 = np.std(param_samples[:, 0]), np.std(param_samples[:, 1])
                initial_speed_uncertainty = std_v0 * sigma_level
                lower_bound_params, upper_bound_params = np.copy(final_params), np.copy(final_params)
                lower_bound_params[0] -= std_v0 * sigma_level
                lower_bound_params[1] -= std_accel0 * sigma_level
                upper_bound_params[0] += std_v0 * sigma_level
                upper_bound_params[1] = min(final_params[1] + std_accel0 * sigma_level, 0.0)
        except (np.linalg.LinAlgError, ValueError) as e: print(f"Warning: Could not perform uncertainty simulation: {e}")

    print(f"\nFit successful using {n_ok} data points.")
    print(f"Initial speed (v_i) [{sigma_level}-sigma]: {initial_speed:.3f} ± {initial_speed_uncertainty:.3f} km/s")

    if doplot:
        if worst_station_idx is not None:
             print(f"\nGenerating plots. Excluding significant outlier '{station_obs[worst_station_idx]['site_info']['name']}' for clarity.")
        else:
            print("\nGenerating plots for all stations (no significant outliers found).")
        _generate_plots(final_merged_data, final_params, lower_bound_params, upper_bound_params, n_ok, doplot, debug, sigma_level,
                        station_id_array=station_id_array, exclude_station_idx=worst_station_idx)
    
    if posdata:
        print("\nTime [s]  Height [km]  Position [km]  Speed [km/s]")
        speeds = expfunc_1stder(final_merged_data['reltime'], *final_params)
        for i in range(len(final_merged_data['reltime'])):
            print(f"{final_merged_data['reltime'][i]:>8.3f}  {final_merged_data['height'][i]:>10.3f}  {final_merged_data['pos'][i]:>12.3f}  {speeds[i]:>11.3f}")
        
    return True, initial_speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit speed and acceleration profiles to meteor centroid data.')
    parser.add_argument('-r', '--res', dest='resname', required=True, help='Name of input file with .res extension.')
    parser.add_argument('-d', '--dat', dest='datname', default='', help='Name of input file with .dat extension.')
    parser.add_argument('-c', '--cen', dest='cennames', required=True, type=lambda s: s.split(','), help='Comma-separated list of input files with centroid data.')
    parser.add_argument('-p', '--posdata', dest='posdata', action='store_true', help='Output final time, height, position, and speed data to console.')
    parser.add_argument('-o', '--output', dest='output', default='', choices=['', 'show', 'save', 'both'], help='show: Display graphics, save: save graphics to SVG, both: Display and save.')
    parser.add_argument('-v', '--verbose', dest='debug', action="store_true", help='Provide additional debugging output.')
    parser.add_argument('--fscale', dest='fscale', type=float, default=0.1, help='Robust loss function scale. Smaller is more robust. Default: 0.1')
    parser.add_argument('--uncertainty-sigma', dest='sigma_level', type=float, default=1.0, help='Sigma level for uncertainty reporting (e.g., 1.0, 3.0). Default: 1.0')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='Random seed for reproducibility of uncertainty calculations. Default: None')
    parser.add_argument('--sims', dest='num_simulations', type=int, default=1000, help='Number of Monte Carlo simulations for uncertainty calculation. Default: 1000')
    args = parser.parse_args()

    if args.cennames and not args.cennames[0]: args.cennames.pop(0)
    if not args.resname or not args.cennames: parser.print_help(), sys.exit(1)

    fbspd(resname=args.resname, cennames=args.cennames, datname=args.datname, doplot=args.output, posdata=args.posdata, debug=args.debug, fscale=args.fscale, sigma_level=args.sigma_level, seed=args.seed, num_simulations=args.num_simulations)
