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
from typing import List, Tuple, Optional, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares, brute

# WGS-84 Ellipsoid Parameters
EARTH_RADIUS = 6371.0  # Mean radius in km
FLATTENING = 1.0 / 298.257223563  # Earth flattening factor (WGS-84)
E_SQUARED = 2 * FLATTENING - FLATTENING**2  # Eccentricity squared

# --- Configuration Constants ---

# Hardware and observation parameters
CAMERA_DEGREES_PER_PIXEL = 0.25  # Assumed angular resolution for error estimation

# File format parameters
DAT_STATION_NAME_START_COLUMN = 12  # Column where the station name starts in .dat files

# Fitting and model parameters
TIME_OFFSET_SEARCH_RANGE = (-5.0, 5.0)  # Search range for time offsets in seconds
TIME_OFFSET_SEARCH_STEP = 0.01          # Step size for the brute-force time offset search
MIN_INLIERS_FOR_ROBUST_FIT = 5          # Min points required for a fit to be attempted
MIN_POINTS_FOR_EXP_FIT = 4              # Min points to attempt an exponential fit (must match # of params)

# Physical plausibility checks for fitting
FIT_MAX_POS_ACCELERATION = 1e-3  # Allow for tiny positive acceleration from noise
FIT_MIN_REASONABLE_SPEED = -1.0  # Allow for small negative speed from noise

# Numerical tolerance parameters
XYZ_CONVERGENCE_TOLERANCE = 1e-12   # For iterative xyz2lonlat conversion
PARALLEL_LINES_TOLERANCE = 1e-9     # For geometry checks to avoid division by zero
MIN_POSITIONAL_SIGMA = 1e-9         # A small floor value for position uncertainty


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

            # Parse from the end of the line, which is more predictable
            height_km = 0.0
            last_name_idx = len(words)

            try:
                # Check if the last word is the optional observer height
                obs_height_m = float(words[-1])
                height_km = obs_height_m / 1000.0
                last_name_idx = -2 # Name ends before the last two numbers
            except (ValueError, IndexError):
                last_name_idx = -1 # Name ends before the last number

            # The station name is between the start column and the first number from the end
            name = " ".join(words[DAT_STATION_NAME_START_COLUMN:last_name_idx])

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
        if abs(lat_rad_new - lat_rad) < XYZ_CONVERGENCE_TOLERANCE:
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


# --- Fitting Functions ---

def linfunc(x: float, a: float, b: float) -> float:
    """Linear function for fitting."""
    return a * x + b


def expfunc(t: np.ndarray, v0: float, accel0: float, k: float, p0: float) -> np.ndarray:
    """
    Position function parameterized by initial speed (v0), initial acceleration (accel0),
    an exponential rate (k), and initial position (p0).
    Note: accel0 is expected to be negative. k must be positive.
    """
    # To avoid division by zero if k is near zero, handle as a linear fit (Taylor expansion)
    if abs(k) < 1e-6:
        return p0 + v0 * t + 0.5 * accel0 * t**2
    
    term1 = v0 - accel0 / k
    term2 = accel0 / (k**2)
    return p0 + term1 * t + term2 * (np.exp(k * t) - 1)

def expfunc_1stder(t: np.ndarray, v0: float, accel0: float, k: float, p0: float) -> np.ndarray:
    """First derivative (Speed) of the new model."""
    if abs(k) < 1e-6:
        return v0 + accel0 * t
    return (v0 - accel0 / k) + (accel0 / k) * np.exp(k * t)

def expfunc_2ndder(t: np.ndarray, v0: float, accel0: float, k: float, p0: float) -> np.ndarray:
    """Second derivative (Acceleration) of the new model."""
    if abs(k) < 1e-6:
        return np.full_like(t, accel0)
    return accel0 * np.exp(k * t)


def try_expfit(x: np.ndarray, y: np.ndarray, weights: np.ndarray, guess: List[float]) -> Tuple[Optional[np.ndarray], bool]:
    """Tries to fit the exp-lin function with a given initial guess."""
    # Bounds for parameters (a, b, c, d). Limits c (speed) and b (deceleration) to sane ranges.
    bounds = ([0, 0, 5, -np.inf], [np.inf, 10, 100, np.inf])
    try:
        # Add bounds and increase max iterations for better convergence
        params, _ = curve_fit(expfunc, x, y, p0=guess, sigma=1./weights, bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError):
        return None, False

    # Check for physically plausible results
    speed = expfunc_1stder(x, *params)
    accel = expfunc_2ndder(x, *params)

    if np.min(speed) < FIT_MIN_REASONABLE_SPEED or np.max(accel) > FIT_MAX_POS_ACCELERATION:
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
    while len(current_x) > MIN_POINTS_FOR_EXP_FIT:
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
        search_start, search_end = TIME_OFFSET_SEARCH_RANGE
        ranges = (slice(search_start, search_end, TIME_OFFSET_SEARCH_STEP),)
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
    m2[m2 < PARALLEL_LINES_TOLERANCE] = PARALLEL_LINES_TOLERANCE

    R = np.cross(p21, m / m2[:, np.newaxis])
    t1 = np.einsum('ij,j->i', R, track_vec_norm)
    t2 = np.einsum('ij,ij->i', R, los_vecs)

    cross_points1 = site_pos + t1[:, np.newaxis] * los_vecs
    cross_points2 = track_p1 + t2[:, np.newaxis] * track_vec_norm

    intersec_on_track = (cross_points1 + cross_points2) / 2.0
    
    # Calculate position, height, and uncertainty in a vectorized way
    pos = np.einsum('ij,j->i', intersec_on_track - track_p1, track_vec_norm)
    height = np.array([xyz2lonlat(v)[2] for v in intersec_on_track])

    sitedist = np.linalg.norm(intersec_on_track - site_pos, axis=1)
    ang_err = np.radians(CAMERA_DEGREES_PER_PIXEL * cendat.censig)
    sig = sitedist * np.tan(ang_err)
    sig[ang_err <= 0] = MIN_POSITIONAL_SIGMA # Handle non-positive error values

    return {"pos": pos, "height": height, "sig": sig}


def _fit_merged_data(reltime: np.ndarray, pos: np.ndarray, sig: np.ndarray,
                     debug: bool, fscale: float) -> Tuple[Optional[np.ndarray], int, Optional[np.ndarray]]:
    """
    Performs a single robust fit on all data points using a physically-motivated
    model to ensure stable uncertainty propagation.
    """
    n_pts = len(reltime)
    if n_pts < MIN_POINTS_FOR_EXP_FIT:
        print("Error: Not enough data points to perform a fit.")
        return None, 0, None

    def residual_func(params, x, y, sig, model_func):
        return (y - model_func(x, *params)) / sig

    def get_pcov(res, n, p):
        if n <= p: return None
        try:
            mse = 2 * res.cost / (n - p)
            return np.linalg.pinv(res.jac.T @ res.jac) * mse
        except (np.linalg.LinAlgError, ZeroDivisionError):
            return None

    # --- Initial Guess and Bounds for New Physical Parameters ---
    try:
        # Use a simple linear fit to get an initial guess for v0 and p0
        lin_params_guess, _ = curve_fit(linfunc, reltime, pos, sigma=1./sig)
        v0_guess = lin_params_guess[0]
        p0_guess = lin_params_guess[1]
    except RuntimeError:
        # Fallback if the initial linear guess fails
        v0_guess = 40.0
        p0_guess = pos[0] if len(pos) > 0 else 0

    # Make physically reasonable guesses for acceleration and rate
    accel0_guess = -10.0  # A reasonable starting deceleration in km/s^2
    k_guess = 1.0         # A reasonable exponential rate
    
    exp_params_guess = [v0_guess, accel0_guess, k_guess, p0_guess]

    # Define physical bounds for the new parameters (v0, accel0, k, p0)
    # v0: 10-73 km/s, accel0: -500 to almost 0, k: >0 to 10, p0: no bound
    exp_bounds = ([8.0, -500.0, 1e-6, -np.inf], [73.0, -1e-9, 10.0, np.inf])

    # --- Attempt Robust Exponential Fit
    res_exp = least_squares(
        residual_func, exp_params_guess, loss='soft_l1', f_scale=fscale,
        args=(reltime, pos, sig, expfunc),
        bounds=exp_bounds, max_nfev=5000
    )

    # --- Attempt Linear Fit on ALL data as a fallback ---
    res_lin = None
    try:
        lin_params_initial_guess, _ = curve_fit(linfunc, reltime, pos, sigma=1./sig)
        res_lin = least_squares(
            residual_func, lin_params_initial_guess, loss='soft_l1', f_scale=fscale,
            args=(reltime, pos, sig, linfunc)
        )
    except (RuntimeError, ValueError):
        if not res_exp.success:
            print("Error: Both exponential and linear fits failed.")
            return None, 0, None

    # --- Compare Models and Select the Best One ---
    # Choose exponential if its fit succeeded and its cost is lower than the linear fit's cost
    if res_exp.success and (res_lin is None or not res_lin.success or res_exp.cost < res_lin.cost):
        if debug: print("Selected robust exponential model on all points.")
        final_params = res_exp.x
        final_pcov = get_pcov(res_exp, n_pts, len(final_params))
        return final_params, n_pts, final_pcov
    
    # Otherwise, fallback to the linear model
    if res_lin and res_lin.success:
        if debug: print("Selected robust linear model on all points as fallback.")
        lin_params = res_lin.x
        # Convert linear params [slope, intercept] to the 4-param format [v0, accel0, k, p0]
        # A linear model has zero acceleration.
        final_params = np.array([lin_params[0], 0, 0, lin_params[1]])
        
        lin_pcov = get_pcov(res_lin, n_pts, len(lin_params))
        final_pcov = None
        if lin_pcov is not None:
            # Map the 2x2 covariance of (v0, p0) into the 4x4 matrix
            final_pcov = np.zeros((4, 4))
            final_pcov[0, 0] = lin_pcov[0, 0]  # var(v0)
            final_pcov[3, 3] = lin_pcov[1, 1]  # var(p0)
            final_pcov[0, 3] = lin_pcov[0, 1]  # cov(v0, p0)
            final_pcov[3, 0] = lin_pcov[1, 0]  # cov(p0, v0)
            
        return final_params, n_pts, final_pcov

    # If we reach here, no fit was successful
    print("Error: Could not find a successful fit for the data.")
    return None, 0, None


def _generate_plots(data: dict, params: np.ndarray, param_samples: Optional[np.ndarray], n_ok: int, doplot: str, resname: str, debug: bool = False, sigma_level: float = 1.0):
    """
    Generates plots with a smoothed, solid, light blue residual plot region.
    """
    reltime, pos, sig = data['reltime'], data['pos'], data['sig']

    # --- Main Calculations ---
    t_fit = np.linspace(reltime.min(), reltime.max(), 300)
    fit_pos = expfunc(t_fit, *params)
    fit_speed = expfunc_1stder(t_fit, *params)
    fit_accel = expfunc_2ndder(t_fit, *params)
    residuals = pos - expfunc(reltime, *params)

    # --- Smooth the Observational Uncertainty for Plotting ---
    smoothing_window_sec = 0.15
    avg_time_step = np.mean(np.diff(reltime))
    window_size = int(smoothing_window_sec / avg_time_step)
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(3, window_size)
    
    kernel = np.ones(window_size) / window_size
    sig_smoothed = np.convolve(sig, kernel, mode='same')

    # --- Figure 1: Position vs. Time ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig1.suptitle(f"Atmosfærisk bane", fontsize=16, fontstyle="oblique")

    sc = ax1.scatter(reltime, pos, c=reltime, cmap='viridis', s=10, label='Enkeltobservasjoner')
    ax1.plot(t_fit, fit_pos, 'r-', label='Estimert bane', linewidth=2)
    
    ax1.set_ylabel('Posisjonen langs banen [km]')
    ax1.legend()
    
    fig1.subplots_adjust(right=0.85, top=0.92)
    cbar_ax = fig1.add_axes([0.88, 0.11, 0.03, 0.77])
    cbar = fig1.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Tid [s]')

    
    ax2.axhline(0, color='r', linestyle='--', linewidth=1.5, zorder=4)
    ax2.scatter(reltime, residuals, c=reltime, cmap='viridis', s=10, zorder=5)
    ax2.fill_between(reltime, -sig_smoothed * sigma_level, sig_smoothed * sigma_level, color='blue', alpha=0.3, label=f'±{sigma_level:.0f}σ usikkerhet')
    
    ax2.set_ylabel('Residualer [km]')
    ax2.set_xlabel('Tid [s]')
    ax2.legend(loc='upper right')

    if doplot in ['save', 'both']:
        plt.savefig("posvstime.svg", bbox_inches='tight', pad_inches=0.05)

    # --- Figure 2: Speed and Acceleration ---
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    fig2.suptitle(f"     Dynamisk analyse", fontsize=16, fontstyle="oblique")
    
    ax3.plot(t_fit, fit_speed, 'b-')
    ax3.set_ylim(bottom=0)
    ax3.set_ylabel('Hastighet [km/s]')
    ax3.set_title('Hastighetsprofil')

    ax4.plot(t_fit, fit_accel, 'b-')
    ax4.set_ylabel('Aksellerasjon [km/s²]')
    ax4.set_xlabel('Tid [s]')
    ax4.set_title('Aksellerasjonsprofil')

    if param_samples is not None:
        speed_samples = np.array([expfunc_1stder(t_fit, *p) for p in param_samples])
        accel_samples = np.array([expfunc_2ndder(t_fit, *p) for p in param_samples])
        speed_std = np.std(speed_samples, axis=0) * sigma_level
        accel_std = np.std(accel_samples, axis=0) * sigma_level
        
        t0_index = np.argmin(np.abs(t_fit))

        # --- Constrain initial uncertainty for both plots ---
        for std_arr in [speed_std, accel_std]:
            std_at_t0 = std_arr[t0_index]
            pre_t0_std = std_arr[:t0_index]
            if len(pre_t0_std) > 0:
                capped_pre_t0 = np.minimum.accumulate(pre_t0_std[::-1])[::-1]
                capped_pre_t0[capped_pre_t0 > std_at_t0] = std_at_t0
                std_arr[:t0_index] = capped_pre_t0
        
        # --- Logic for Speed Plot ---
        lower_speed_bound = fit_speed - speed_std
        upper_speed_bound = fit_speed + speed_std
        # Enforce that the upper bound is non-increasing
        final_upper_bound = np.minimum.accumulate(upper_speed_bound)
        ax3.fill_between(t_fit, lower_speed_bound, final_upper_bound, color='blue', alpha=0.2, label=f'±{sigma_level:.0f}σ usikkerhet')
        
        # --- Logic for Acceleration Plot ---
        lower_accel_bound = fit_accel - accel_std
        upper_accel_bound = fit_accel + accel_std
        # Enforce the single most important constraint: acceleration cannot be positive.
        final_upper_accel_bound = np.minimum(upper_accel_bound, 0)
        ax4.fill_between(t_fit, lower_accel_bound, final_upper_accel_bound, color='blue', alpha=0.2, label=f'±{sigma_level:.0f}σ usikkerhet')
        
        ax3.legend()
        ax4.legend()

    if doplot in ['save', 'both']:
        plt.savefig("spd_acc.svg", bbox_inches='tight', pad_inches=0.05)

    if doplot in ['show', 'both']:
        plt.show()

        
def fbspd(resname: str, cennames: List[str], datname: str,
          doplot: str = '', posdata: bool = False, debug: bool = False,
          fscale: float = 0.1, sigma_level: float = 1.0,
          seed: Optional[int] = None,
          num_simulations: int = 500) -> Tuple[bool, float]:
    """
    Determines speed and acceleration profiles for meteors from trajectory
    parameters and merged multi-station centroid files.
    """

    if seed is not None:
        np.random.seed(seed)
    
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

    try:
        merged_data = {
            'reltime': np.concatenate([obs['reltime'] + offset for obs, offset in zip(station_obs, time_offsets)]),
            'pos': np.concatenate([obs['pos'] for obs in station_obs]),
            'height': np.concatenate([obs['height'] for obs in station_obs]),
            'sig': np.concatenate([obs['sig'] for obs in station_obs]),
        }
        if merged_data['reltime'].size == 0:
            raise ValueError("All station data resulted in zero valid points after processing.")

    except ValueError as e:
        print(f"\nError: Could not merge station data. This can happen with inconsistent or empty input files.", file=sys.stderr)
        print(f"Underlying Error: {e}", file=sys.stderr)
        return False, 0.0

    # Sort merged data by time
    sort_idx = np.argsort(merged_data['reltime'])
    for key in merged_data:
        merged_data[key] = merged_data[key][sort_idx]

    # 5. Fit the merged data
    try:
        lin_params, _ = curve_fit(linfunc, merged_data['reltime'], merged_data['pos'], sigma=1./merged_data['sig'])
    except RuntimeError:
        print("Error: Initial linear fit for time adjustment failed. Cannot proceed.")
        return False, 0.0

    adjtime = -lin_params[1] / lin_params[0] if lin_params[0] != 0 else 0
    merged_data['reltime'] += adjtime
    if debug: print(f"Time axis adjusted by: {adjtime:.4f} s")

    # Final fit on time-adjusted data
    final_params, n_ok, pcov = _fit_merged_data(
        merged_data['reltime'], merged_data['pos'], merged_data['sig'], debug, fscale
    )

    if n_ok == 0:
        print("Error: Final fit failed for all data points.")
        return False, 0.0

    # --- Uncertainty Simulation ---
    param_samples = None
    initial_speed_uncertainty = 0.0
    if pcov is not None and not np.isnan(pcov).any(): # Check for valid covariance matrix
        try:
            # Ensure covariance matrix is positive semi-definite for sampling
            eigenvalues = np.linalg.eigvalsh(pcov)
            if np.min(eigenvalues) < -1e-12: # Allow for small negative noise
                jitter = abs(np.min(eigenvalues)) + 1e-9
                pcov += np.eye(pcov.shape[0]) * jitter
            
            # Check for non-finite values before sampling
            if not np.isfinite(pcov).all():
                raise np.linalg.LinAlgError("Covariance matrix contains non-finite values.")

            param_samples = np.random.multivariate_normal(final_params, pcov, size=num_simulations)

            # Keep only samples where accel0 (the second parameter, index 1) is negative.
            valid_samples = param_samples[param_samples[:, 1] <= 0]

            # Check if any valid samples remain
            if len(valid_samples) < 10: # Use a threshold to ensure enough samples for a stable estimate
                print("Warning: Too few valid samples for uncertainty estimate. Disabling bands.")
                param_samples = None
            else:
                param_samples = valid_samples
            
            # Check for bad samples
            if not np.isfinite(param_samples).all():
                print("Warning: Numerical instability in uncertainty sampling produced non-finite values. Disabling uncertainty bands.")
                param_samples = None # Reset to disable plotting uncertainty

            if param_samples is not None:
                initial_speed_samples = [expfunc_1stder(0.0, *p) for p in param_samples]
                initial_speed_uncertainty = np.std(initial_speed_samples) * sigma_level
        except np.linalg.LinAlgError as e:
            print(f"Warning: Could not estimate uncertainties due to numerical instability: {e}")
            param_samples = None # Ensure it's None on failure
            
    # 6. Output results
    initial_speed = expfunc_1stder(0.0, *final_params)
    print(f"Fit successful using {n_ok} data points.")
    print(f"Initial speed (v_i) [{sigma_level}-sigma]: {initial_speed:.3f} ± {initial_speed_uncertainty:.3f} km/s")

    if doplot:
        _generate_plots(merged_data, final_params, param_samples, n_ok, doplot, resname, debug, sigma_level)

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
    parser.add_argument(
        '--fscale', dest='fscale', type=float, default=0.1,
        help='Robust loss function scale. Smaller is more robust. Default: 0.1'
    )
    parser.add_argument(
        '--uncertainty-sigma', dest='sigma_level', type=float, default=1.0,
        help='Sigma level for uncertainty reporting (e.g., 1.0, 3.0). Default: 1.0'
    )
    parser.add_argument(
        '--seed', dest='seed', type=int, default=None,
        help='Random seed for reproducibility of uncertainty calculations. Default: None'
    )
    parser.add_argument(
        '--sims', dest='num_simulations', type=int, default=1000,
        help='Number of Monte Carlo simulations for uncertainty calculation. Default: 1000'
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
        debug=args.debug,
        fscale=args.fscale,
        sigma_level=args.sigma_level,
        seed=args.seed,
        num_simulations=args.num_simulations
    )
