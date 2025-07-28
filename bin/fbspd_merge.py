#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fbspd_merge.py: Fits speed and acceleration profiles to meteor centroid data.

This script analyzes meteor trajectory data from multiple observation stations.
It reads trajectory solutions (.res), station coordinates (.dat), and centroid
observations (.cen) to compute a merged speed and acceleration profile for
a meteor event.

The main functions `readres` and `fbspd` are designed to be callable from
external scripts.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares, brute


# WGS-84 Ellipsoid Parameters
EARTH_RADIUS = 6371.0  # Mean radius in km
FLATTENING = 1.0 / 298.257223563  # Earth flattening factor (WGS-84)
E_SQUARED = 2 * FLATTENING - FLATTENING**2  # Eccentricity squared


# --- Data Structures ---

class ResData:
    """Container for data from a .res file."""
    def __init__(self):
        self.ndata: int = 0
        self.long1: np.ndarray = np.array([])
        self.lat1: np.ndarray = np.array([])
        self.long2: np.ndarray = np.array([])
        self.lat2: np.ndarray = np.array([])
        self.height: np.ndarray = np.array([])
        self.desc: np.ndarray = np.array([])


class CenData:
    """Container for data from a centroid file."""
    def __init__(self):
        self.ndata: int = 0
        self.seqid: np.ndarray = np.array([])
        self.reltime: np.ndarray = np.array([])
        self.cenalt: np.ndarray = np.array([])
        self.cenaz: np.ndarray = np.array([])
        self.censig: np.ndarray = np.array([])
        self.sitestr: np.ndarray = np.array([])
        self.datestr: np.ndarray = np.array([])
        self.timestr: np.ndarray = np.array([])
        self.site_info: dict = {}


# --- File Reading Functions ---

def readres(inname: str) -> ResData:
    """Reads data from a .res file and returns a ResData object."""
    res_path = Path(inname)
    data = {
        "long1": [], "lat1": [], "long2": [], "lat2": [], "height": [], "desc": []
    }

    with res_path.open('r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            words = line.split()
            data["long1"].append(float(words[0]))
            data["lat1"].append(float(words[1]))
            data["long2"].append(float(words[2]))
            data["lat2"].append(float(words[3]))
            data["height"].append(float(words[4]))
            data["desc"].append(words[5])

    res = ResData()
    res.ndata=len(data["long1"])
    res.long1=np.array(data["long1"])
    res.lat1=np.array(data["lat1"])
    res.long2=np.array(data["long2"])
    res.lat2=np.array(data["lat2"])
    res.height=np.array(data["height"])
    res.desc=np.array(data["desc"])
    return res


def readcen(inname: str) -> CenData:
    """Reads data from a centroid file and returns a CenData object."""
    cen_path = Path(inname)
    data = {
        "seqid": [], "reltime": [], "cenalt": [], "cenaz": [],
        "censig": [], "sitestr": [], "datestr": [], "timestr": []
    }

    with cen_path.open('r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            words = line.split()
            data["seqid"].append(int(words[0]))
            data["reltime"].append(float(words[1]))
            data["cenalt"].append(float(words[2]))
            data["cenaz"].append(float(words[3]))
            data["censig"].append(float(words[4]))
            data["sitestr"].append(words[5])
            data["datestr"].append(words[6])
            data["timestr"].append(words[7])

    cen = CenData()
    cen.ndata = len(data["seqid"])
    for k, v in data.items():
        setattr(cen, k, np.array(v))
    return cen


def get_sitecoord_fromdat(inname: str) -> List[Tuple[str, float, float, float]]:
    """Extracts site coordinates from a .dat file."""
    dat_path = Path(inname)
    sitecoords = []
    
    with dat_path.open('r') as f:
        for line in f:
            words = line.split()
            if not words or words[0].startswith('#') or words[0] == 'borders':
                continue

            lon, lat = float(words[0]), float(words[1])
            
            # Find the station name, which can contain spaces
            name_parts = []
            first_numeric_idx = 12
            for i, word in enumerate(words[12:]):
                try:
                    float(word)
                    first_numeric_idx = 12 + i
                    break
                except ValueError:
                    name_parts.append(word)
            
            name = " ".join(name_parts)
            
            # Observer height is optional, default to 0
            try:
                obs_height_m = float(words[first_numeric_idx + 1])
                height_km = obs_height_m / 1000.0
            except (IndexError, ValueError):
                height_km = 0.0
                
            sitecoords.append((name, lon, lat, height_km))
            
    return sitecoords


# --- Coordinate and Vector Functions ---

def lonlat2xyz(lon_deg: float, lat_deg: float, height_km: float = 0) -> np.ndarray:
    """Converts geographic lon/lat/height (WGS-84) to cartesian ECEF coordinates."""
    lat_rad, lon_rad = np.radians(lat_deg), np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    
    N = EARTH_RADIUS / np.sqrt(1 - E_SQUARED * sin_lat**2)
    
    x = (N + height_km) * cos_lat * np.cos(lon_rad)
    y = (N + height_km) * cos_lat * np.sin(lon_rad)
    z = (N * (1 - E_SQUARED) + height_km) * sin_lat
    
    return np.array([x, y, z])


def xyz2lonlat(v: np.ndarray) -> Tuple[float, float, float]:
    """Converts cartesian ECEF coordinates to geographic lon/lat/height (WGS-84)."""
    p = np.sqrt(v[0]**2 + v[1]**2)
    lon_deg = np.degrees(np.arctan2(v[1], v[0]))
    
    # Iterative method to find latitude and height
    lat_rad = np.arctan2(v[2], p * (1.0 - E_SQUARED))
    
    for _ in range(10): # Typically converges very quickly
        sin_lat = np.sin(lat_rad)
        N = EARTH_RADIUS / np.sqrt(1.0 - E_SQUARED * sin_lat**2)
        height_km = p / np.cos(lat_rad) - N
        lat_rad_new = np.arctan2(v[2], p * (1.0 - E_SQUARED * N / (N + height_km)))
        if abs(lat_rad_new - lat_rad) < 1e-12:
            break
        lat_rad = lat_rad_new
        
    return lon_deg, np.degrees(lat_rad), height_km


def altaz2xyz(alt_deg: float, az_deg: float, obs_lon: float, obs_lat: float) -> np.ndarray:
    """Converts local alt/az to a unit vector in the ECEF frame."""
    alt_rad, az_rad = np.radians(alt_deg), np.radians(az_deg)
    lat_rad, lon_rad = np.radians(obs_lat), np.radians(obs_lon)

    # Vector in local ENU (East-North-Up) frame
    v_enu = np.array([
        np.sin(az_rad) * np.cos(alt_rad),
        np.cos(az_rad) * np.cos(alt_rad),
        np.sin(alt_rad)
    ])

    # Rotation matrix to convert ENU vector to ECEF frame
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    R = np.array([
        [-sin_lon, -cos_lon * sin_lat, cos_lon * cos_lat],
        [ cos_lon, -sin_lon * sin_lat, sin_lon * cos_lat],
        [       0,           cos_lat,           sin_lat]
    ])
    
    return R @ v_enu


def closest_point(p1: np.ndarray, p2: np.ndarray, u1: np.ndarray, u2: np.ndarray,
                  return_points: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Finds the closest point(s) between two lines in 3D space."""
    p21 = p2 - p1
    m = np.cross(u2, u1)
    m2 = np.dot(m, m)

    if m2 < 1e-9:  # Lines are parallel
        closest_p2_on_line1 = p1 + np.dot(p21, u1) * u1
        if return_points:
            return closest_p2_on_line1, p2
        return (closest_p2_on_line1 + p2) / 2.0

    R = np.cross(p21, m / m2)
    t1 = np.dot(R, u2)
    t2 = np.dot(R, u1)

    cross_1 = p1 + t1 * u1
    cross_2 = p2 + t2 * u2

    if return_points:
        return cross_1, cross_2
    return (cross_1 + cross_2) / 2.0


def dist_line_line(p1: np.ndarray, u1: np.ndarray, p2: np.ndarray, u2: np.ndarray) -> float:
    """Calculates the minimum distance between two lines."""
    pq = closest_point(p1, p2, u1, u2, return_points=True)
    dist_vec = pq[0] - pq[1]
    return np.sqrt(np.dot(dist_vec, dist_vec))


# --- Fitting Functions ---

def linfunc(x: float, a: float, b: float) -> float:
    """Linear function for fitting."""
    return a * x + b


def expfunc(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """
    Exponential-linear function for fitting meteor dynamics.
    The absolute values constrain the fit to a physically meaningful regime
    (i.e., deceleration).
    """
    return -abs(a) * np.exp(abs(b) * x) + abs(a) * abs(b) * x + abs(a) + c * x + d


def expfunc_1stder(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """First derivative of expfunc (speed)."""
    return -abs(a) * abs(b) * np.exp(abs(b) * x) + abs(a) * abs(b) + c


def expfunc_2ndder(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Second derivative of expfunc (acceleration)."""
    return -abs(a) * abs(b)**2 * np.exp(abs(b) * x)


def try_expfit(x: np.ndarray, y: np.ndarray, weights: np.ndarray, guess: List[float]) -> Tuple[Optional[np.ndarray], bool]:
    """Tries to fit the exp-lin function with a given initial guess."""
    try:
        params, _ = curve_fit(expfunc, x, y, p0=guess, sigma=1./weights)
    except (RuntimeError, ValueError):
        return None, False

    # Check for physically plausible results
    speed = expfunc_1stder(x, *params)
    accel = expfunc_2ndder(x, *params)

    if np.min(speed) < -1 or np.max(accel) > 1e-3: # Allow for tiny positive accel
        return None, False
        
    return params, True


def guess_expfit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
    """Tries fitting with different initial guesses, returning the first success."""
    guesslist = [
        [0.1, 1.0, 20.0, 1.0], [0.1, 1.0, 20.0, 0.0], [1.0, 1.0, 25.0, 0.0],
        [1e-3, 1.0, 25.0, 0.0], [0.1, 1.0, 10.0, 0.0], [0.1, 1.0, 30.0, 0.0],
    ]
    
    current_x, current_y, current_weights = np.copy(x), np.copy(y), np.copy(weights)

    # If fitting fails, iteratively remove the last point and retry
    while len(current_x) > 4:
        for guess in guesslist:
            params, success = try_expfit(current_x, current_y, current_weights, guess)
            if success:
                return params, len(current_x)
        current_x, current_y, current_weights = current_x[:-1], current_y[:-1], current_weights[:-1]

    return None, 0


# --- Time Series Alignment ---

def _chainlength(time_series: List[np.ndarray], pos_series: List[np.ndarray], offsets: np.ndarray) -> float:
    """Calculates squared distance of a time-sorted chain of points."""
    if not any(len(ts) > 0 for ts in time_series):
        return 1e10

    timearr = np.concatenate([ts + offset for ts, offset in zip(time_series, offsets)])
    posarr = np.concatenate(pos_series)
    
    if len(timearr) < 2:
        return 1e10
        
    sortidx = np.argsort(timearr)
    sorted_time = timearr[sortidx]
    sorted_pos = posarr[sortidx]

    seglengths_sq = (sorted_time[1:] - sorted_time[:-1])**2 + (sorted_pos[1:] - sorted_pos[:-1])**2
    return seglengths_sq.sum()


def minimize_chainlength(all_time_arrays: List[np.ndarray], all_pos_arrays: List[np.ndarray]) -> np.ndarray:
    """Finds time offsets between series by minimizing the chain length."""
    n_series = len(all_time_arrays)
    if n_series < 2:
        return np.zeros(n_series)

    best_offsets = np.zeros(n_series)
    # Anchor the first series and find best offset for each subsequent series
    for i in range(1, n_series):
        # Define the objective function to be minimized.
        # brute passes a 1-element array, so we extract the float with offset[0].
        def objective_func(offset: np.ndarray) -> float:
            return _chainlength(
                [all_time_arrays[0], all_time_arrays[i]],
                [all_pos_arrays[0], all_pos_arrays[i]],
                np.array([0, offset[0]])
            )

        # Use brute-force grid search, which is robust for this non-smooth function.
        # The 'ranges' tuple defines the search space [start, stop, step-size].
        # We explicitly disable a finishing step to ensure only grid points are tested.
        ranges = (slice(-5, 5, 0.01),)
        result = brute(objective_func, ranges, finish=None)
        best_offsets[i] = result

    return best_offsets


# --- Main Logic and Plotting ---

def _process_station(cendat: CenData, site_pos: np.ndarray, track_p1: np.ndarray,
                     track_vec_norm: np.ndarray) -> dict:
    """Projects one station's observations onto the meteor track."""
    site_lon, site_lat = cendat.site_info['lon'], cendat.site_info['lat']

    # Call altaz2xyz for each observation and stack into an (N, 3) array.
    los_vecs = np.array([
        altaz2xyz(alt, az, site_lon, site_lat)
        for alt, az in zip(cendat.cenalt, cendat.cenaz)
    ])

    # Vectorized geometry calculations
    p21 = track_p1 - site_pos
    m = np.cross(track_vec_norm, los_vecs)
    m2 = np.einsum('ij,ij->i', m, m)

    # Avoid division by zero for parallel lines
    m2[m2 < 1e-9] = 1e-9

    R = np.cross(p21, m / m2[:, np.newaxis])
    t1 = np.einsum('ij,j->i', R, track_vec_norm)
    
    # The subscript for los_vecs must be 'ij' to match its 2D shape.
    t2 = np.einsum('ij,ij->i', R, los_vecs)

    cross_points1 = site_pos + t1[:, np.newaxis] * los_vecs
    cross_points2 = track_p1 + t2[:, np.newaxis] * track_vec_norm

    intersec_on_track = (cross_points1 + cross_points2) / 2.0
    dist_off_track = np.linalg.norm(cross_points1 - cross_points2, axis=1)

    # Calculate position, height, and uncertainty in a vectorized way
    pos = np.einsum('ij,j->i', intersec_on_track - track_p1, track_vec_norm)
    height = np.array([xyz2lonlat(v)[2] for v in intersec_on_track])

    sitedist = np.linalg.norm(intersec_on_track - site_pos, axis=1)
    ang_err = np.radians(0.25 * cendat.censig)
    sig = sitedist * np.tan(ang_err)
    sig[ang_err <= 0] = 1e-9 # Handle non-positive error values

    return {"pos": pos, "height": height, "dist_off_track": dist_off_track, "sig": sig}


def _fit_merged_data(reltime: np.ndarray, pos: np.ndarray, sig: np.ndarray,
                     debug: bool) -> Tuple[np.ndarray, int]:
    """
    Fits the merged data using a combined two-stage robust fitting process:
    1. Iterative outlier rejection to remove egregious errors.
    2. A robust loss function ('soft_l1') for the final fit on cleaned data.
    """
    # --- STAGE 1: Iterative Outlier Rejection ---

    # Perform an initial baseline fit using all data
    initial_params, _ = guess_expfit(reltime, pos, sig)
    if initial_params is None:
        try:
            lin_params_initial, _ = curve_fit(linfunc, reltime, pos, sigma=1./sig)
            initial_params = np.array([0, 0, lin_params_initial[0], lin_params_initial[1]])
        except RuntimeError:
            print("Error: Initial baseline fit failed. Cannot perform outlier rejection.")
            return np.array([0, 0, 0, 0]), 0
    
    # Calculate standardized residuals and identify inliers (non-outliers)
    fit_values = expfunc(reltime, *initial_params)
    residuals = (pos - fit_values) / sig
    inlier_mask = np.abs(residuals) < 3.0
    
    num_outliers = np.sum(~inlier_mask)
    if debug and num_outliers > 0:
        print(f"Outlier rejection: Identified and removed {num_outliers} of {len(reltime)} points.")

    if np.sum(inlier_mask) < 5:
        print("Warning: Outlier rejection left too few points. Fitting all data.")
        inlier_mask = np.ones_like(reltime, dtype=bool)

    reltime_inliers = reltime[inlier_mask]
    pos_inliers = pos[inlier_mask]
    sig_inliers = sig[inlier_mask]

    # --- STAGE 2: Final Fit with Robust Loss Function ---

    # Helper function to define the residual calculation for least_squares
    def residual_func(params, x, y, sig, model_func):
        return (y - model_func(x, *params)) / sig

    # --- Robust Exponential Fit ---
    exp_params_guess, _ = guess_expfit(reltime_inliers, pos_inliers, sig_inliers)
    exp_params = None
    if exp_params_guess is not None:
        res_exp = least_squares(
            residual_func,
            exp_params_guess,
            loss='soft_l1',
            f_scale=0.1,
            args=(reltime_inliers, pos_inliers, sig_inliers, expfunc)
        )
        exp_params = res_exp.x

    # --- Robust Linear Fit ---
    lin_params_guess, _ = curve_fit(linfunc, reltime_inliers, pos_inliers, sigma=1./sig_inliers)
    res_lin = least_squares(
        residual_func,
        lin_params_guess,
        loss='soft_l1',
        f_scale=0.1,
        args=(reltime_inliers, pos_inliers, sig_inliers, linfunc)
    )
    lin_params = res_lin.x

    # --- Compare Models and Return Best Fit ---
    if exp_params is not None:
        # Compare chi-squared of the two robust fits to see which model is better
        exp_fit_vals = expfunc(reltime_inliers, *exp_params)
        lin_fit_vals = linfunc(reltime_inliers, *lin_params)
        
        chi2_exp = np.sum(((pos_inliers - exp_fit_vals) / sig_inliers)**2)
        chi2_lin = np.sum(((pos_inliers - lin_fit_vals) / sig_inliers)**2)

        if chi2_exp < chi2_lin:
            if debug: print("Selected robust exponential model.")
            return exp_params, len(reltime_inliers)

    # Fallback to the robust linear fit if the exponential model failed or was worse
    if debug: print("Selected robust linear model as fallback.")
    return np.array([0, 0, lin_params[0], lin_params[1]]), len(reltime_inliers)


def _generate_plots(data: dict, params: np.ndarray, n_ok: int, doplot: str, resname: str):
    """Generates and saves/shows output plots."""
    reltime, pos, sig = data['reltime'], data['pos'], data['sig']
    
    fit_pos = expfunc(reltime, *params)
    fit_dev = pos - fit_pos
    fit_speed = expfunc_1stder(reltime, *params)
    fit_accel = expfunc_2ndder(reltime, *params)

    site_code = Path(resname).stem
    
    # Figure 1: Position vs. Time
    plt.figure(1, figsize=(10, 8))
    plt.suptitle(f"Trajectory Fit for {site_code}", fontsize=16)

    ax1 = plt.subplot(211)
    ax1.errorbar(reltime, pos, yerr=sig, fmt='b.', label='Observations')
    if n_ok < len(reltime):
        ax1.plot(reltime[n_ok:], pos[n_ok:], 'y.', label='Excluded')
    ax1.plot(reltime, fit_pos, 'r-', label='Fit')
    ax1.set_ylabel('Position along track [km]')
    ax1.set_xlabel('Time [s]')
    ax1.set_title('Position vs. Time')
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.errorbar(reltime, fit_dev, yerr=sig, fmt='b.', label='Residuals')
    if n_ok < len(reltime):
        ax2.plot(reltime[n_ok:], fit_dev[n_ok:], 'y.')
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_ylabel('Residual [km]')
    ax2.set_xlabel('Time [s]')
    ax2.set_title('Fit Residuals')
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if doplot in ['save', 'both']:
        plt.savefig(f"{site_code}_pos_fit.svg")

    # Figure 2: Speed and Acceleration
    plt.figure(2, figsize=(10, 8))
    plt.suptitle(f"Derived Dynamics for {site_code}", fontsize=16)
    
    ax3 = plt.subplot(211)
    ax3.plot(reltime, fit_speed, 'b-')
    ax3.set_ylim(bottom=0)
    ax3.set_ylabel('Speed [km/s]')
    ax3.set_xlabel('Time [s]')
    ax3.set_title('Speed Profile')
    ax3.grid(True)

    ax4 = plt.subplot(212, sharex=ax3)
    ax4.plot(reltime, fit_accel, 'b-')
    ax4.set_ylabel('Acceleration [km/s²]')
    ax4.set_xlabel('Time [s]')
    ax4.set_title('Acceleration Profile')
    ax4.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if doplot in ['save', 'both']:
        plt.savefig(f"{site_code}_dyn_profile.svg")

    if doplot in ['show', 'both']:
        plt.show()


def fbspd(resname: str, cennames: List[str], datname: str,
          doplot: str = '', posdata: bool = False, debug: bool = False) -> Tuple[bool, float]:
    """
    Determines speed and acceleration profiles for meteors from trajectory
    parameters and merged multi-station centroid files.
    """
    if doplot == 'save' and matplotlib.get_backend() != 'agg':
        matplotlib.use('agg')

    # 1. Load input data
    resdat = readres(resname)
    all_sitedata = get_sitecoord_fromdat(datname) if datname else []
    
    all_cendat = []
    for cen_file in cennames:
        try:
            all_cendat.append(readcen(cen_file))
        except FileNotFoundError:
            print(f"Warning: Centroid file not found, skipping: {cen_file}")
            
    if not all_cendat:
        print("Error: No valid centroid data could be loaded.")
        return False, 0.0

    # Sort centroid data by number of points (longest first)
    all_cendat.sort(key=lambda c: c.ndata, reverse=True)

    # 2. Define meteor trajectory from .res file
    path_p1 = lonlat2xyz(resdat.long1[0], resdat.lat1[0], resdat.height[0])
    path_p2 = lonlat2xyz(resdat.long1[1], resdat.lat1[1], resdat.height[1])
    path_vec = path_p2 - path_p1
    path_vec_norm = path_vec / np.linalg.norm(path_vec)

    # 3. Process each station's data
    station_obs = []
    for cendat in all_cendat:
        site_info = next((s for s in all_sitedata if s[0] == cendat.sitestr[0]), None)
        if not site_info:
            print(f"Warning: No site coords for {cendat.sitestr[0]}. Skipping.")
            continue
        
        name, lon, lat, height = site_info
        cendat.site_info = {'lon': lon, 'lat': lat, 'height': height}
        site_pos = lonlat2xyz(lon, lat, height)
        
        processed_data = _process_station(cendat, site_pos, path_p1, path_vec_norm)
        processed_data['reltime'] = cendat.reltime
        station_obs.append(processed_data)

    if not station_obs:
        print("Error: Could not process any station data.")
        return False, 0.0

    # 4. Merge data from all stations
    all_pos_arrays = [obs['pos'] for obs in station_obs]
    all_time_arrays = [obs['reltime'] for obs in station_obs]
    
    time_offsets = minimize_chainlength(all_time_arrays, all_pos_arrays)
    if debug: print(f"Determined time offsets: {time_offsets}")

    merged_data = {
        'reltime': np.concatenate([obs['reltime'] + offset for obs, offset in zip(station_obs, time_offsets)]),
        'pos': np.concatenate([obs['pos'] for obs in station_obs]),
        'height': np.concatenate([obs['height'] for obs in station_obs]),
        'sig': np.concatenate([obs['sig'] for obs in station_obs]),
    }

    # Sort merged data by time
    sort_idx = np.argsort(merged_data['reltime'])
    for key in merged_data:
        merged_data[key] = merged_data[key][sort_idx]

    # 5. Fit the merged data
    # First fit to find the time adjustment
    try:
        lin_params, _ = curve_fit(linfunc, merged_data['reltime'], merged_data['pos'], sigma=1./merged_data['sig'])
    except RuntimeError:
        print("Error: Initial linear fit for time adjustment failed. Cannot proceed.")
        return False, 0.0

    adjtime = -lin_params[1] / lin_params[0] if lin_params[0] != 0 else 0
    merged_data['reltime'] += adjtime
    if debug: print(f"Time axis adjusted by: {adjtime:.4f} s")
    
    # Final fit on time-adjusted data
    final_params, n_ok = _fit_merged_data(
        merged_data['reltime'], merged_data['pos'], merged_data['sig'], debug
    )
    
    if n_ok == 0:
        print("Error: Final fit failed for all data points.")
        return False, 0.0

    # 6. Output results
    initial_speed = expfunc_1stder(0.0, *final_params)
    print(f"Fit successful using {n_ok} of {len(merged_data['reltime'])} points.")
    print(f"Initial speed (v_i): {initial_speed:.3f} km/s")

    if doplot:
        _generate_plots(merged_data, final_params, n_ok, doplot, resname)

    if posdata:
        print("\nTime [s]  Height [km]  Position [km]  Speed [km/s]")
        speeds = expfunc_1stder(merged_data['reltime'], *final_params)
        for i in range(len(merged_data['reltime'])):
            print(f"{merged_data['reltime'][i]:>8.3f}  "
                  f"{merged_data['height'][i]:>10.3f}  "
                  f"{merged_data['pos'][i]:>12.3f}  "
                  f"{speeds[i]:>11.3f}")

    return True, initial_speed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fit speed and acceleration profiles to meteor centroid data.',
        epilog="Example: fbspd_merge.py -r meteor.res -d meteor.dat -c cen1.txt,cen2.txt -o show"
    )
    parser.add_argument(
        '-r', '--res', dest='resname', required=True,
        help='Name of input file with .res extension.'
    )
    parser.add_argument(
        '-d', '--dat', dest='datname', default='',
        help='Name of input file with .dat extension.'
    )
    parser.add_argument(
        '-c', '--cen', dest='cennames', required=True, type=lambda s: s.split(','),
        help='Comma-separated list of input files with centroid data.'
    )
    parser.add_argument(
        '-p', '--posdata', dest='posdata', action='store_true',
        help='Output final time, height, position, and speed data to console.'
    )
    parser.add_argument(
        '-o', '--output', dest='output', default='', choices=['', 'show', 'save', 'both'],
        help='show: Display graphics, save: save graphics to SVG, both: Display and save.'
    )
    parser.add_argument(
        '-v', '--verbose', dest='debug', action="store_true",
        help='Provide additional debugging output.'
    )

    args = parser.parse_args()

    # Handle the leading comma from the command format: "-c,file1,file2"
    if args.cennames and not args.cennames[0]:
        args.cennames.pop(0)

    if not args.resname or not args.cennames:
        parser.print_help()
        sys.exit(1)

    fbspd(
        resname=args.resname,
        cennames=args.cennames,
        datname=args.datname,
        doplot=args.output,
        posdata=args.posdata,
        debug=args.debug
    )
