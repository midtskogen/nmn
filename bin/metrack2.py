#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrack: Atmospheric Meteor Trajectory Fitting Tool

This script reads a data file describing a set of meteor observations, calculates
the meteor's trajectory through the atmosphere using a robust RANSAC-based
fitting method, and outputs the results. It can be run as a standalone script
or imported as a module in other tools.
"""

import argparse
import configparser
import datetime
import os
import sys
import time
import random
import itertools
from pathlib import Path
from argparse import Namespace

import numpy as np

# --- Optional Dependency Handling ---
def _check_and_import_dependencies():
    """
    Checks for optional libraries, imports them, and provides clear feedback.
    Returns a dictionary indicating which libraries are available.
    """
    deps = {
        'cairosvg': 'cairosvg', 'cartopy': 'cartopy', 'ephem': 'ephem',
        'matplotlib': 'matplotlib', 'Pillow': 'PIL', 'scipy': 'scipy',
        'scour': 'scour', 'showerassoc': 'showerassoc',
    }
    available = {}
    missing = []
    
    for name, import_name in deps.items():
        try:
            available[name] = __import__(import_name)
        except ImportError:
            missing.append(name)
    
    if missing:
        print("Warning: One or more optional libraries are not installed. Functionality will be limited.")
        for name in missing:
            pip_name = 'pyephem' if name == 'ephem' else 'Pillow' if name == 'Pillow' else name
            print(f"  - To install {name}, run: 'pip install {pip_name}'")
    
    return available

AVAILABLE_LIBS = _check_and_import_dependencies()

# Import modules from available libraries
if 'cartopy' in AVAILABLE_LIBS:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io.img_tiles import OSM
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
if 'matplotlib' in AVAILABLE_LIBS:
    from matplotlib import pylab
if 'scipy' in AVAILABLE_LIBS:
    from scipy.optimize import fmin_powell

# --- Constants ---
EARTH_RADIUS_KM = 6371.0
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_E_SQ = 2 * WGS84_FLATTENING - WGS84_FLATTENING**2

class MetrackInfo:
    """Structure for storing information about a fitted meteor track."""
    def __init__(self):
        self.date = ''
        self.timestamp = 0.
        self.error = 1e9
        self.fit_quality = 0.
        self.start_height = 0.
        self.end_height = 0.
        self.ground_track = 0.
        self.course = 0.
        self.incidence = 0.
        self.speed = 0.
        self.radiant_ra = 0.
        self.radiant_dec = 0.
        self.radiant_ecllong = 0.
        self.radiant_ecllat = 0.
        self.shower = 'N/A'
        self.inlier_stations = []

# --- Coordinate and Geometric Transformations ---
def lonlat2xyz(lon_deg, lat_deg, height_km=0):
    lat_rad, lon_rad = np.radians(lat_deg), np.radians(lon_deg)
    N = EARTH_RADIUS_KM / np.sqrt(1 - WGS84_E_SQ * np.sin(lat_rad)**2)
    x = (N + height_km) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + height_km) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - WGS84_E_SQ) + height_km) * np.sin(lat_rad)
    return np.array([x, y, z])

def xyz2lonlat(v):
    p = np.sqrt(v[0]**2 + v[1]**2)
    lon_deg = np.degrees(np.arctan2(v[1], v[0]))
    lat_rad_old = np.arctan2(v[2], (1.0 - WGS84_E_SQ) * p) if p > 1e-9 else np.pi/2 * np.sign(v[2])
    # Iterative method for latitude, 10 steps is sufficient for convergence.
    for _ in range(10):
        sin_lat = np.sin(lat_rad_old)
        N = EARTH_RADIUS_KM / np.sqrt(1.0 - WGS84_E_SQ * sin_lat**2)
        height_km = p / np.cos(lat_rad_old) - N if abs(np.cos(lat_rad_old)) > 1e-9 else v[2] / sin_lat - N * (1-WGS84_E_SQ)
        lat_rad_new = np.arctan2(v[2], p * (1.0 - WGS84_E_SQ * N / (N + height_km)))
        if abs(lat_rad_new - lat_rad_old) < 1e-10: break
        lat_rad_old = lat_rad_new
    return lon_deg, np.degrees(lat_rad_new), height_km

def rotation_matrix(axis, theta):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc-ad), 2*(bd+ac)],
                     [2*(bc+ad), aa+cc-bb-dd, 2*(cd-ab)],
                     [2*(bd-ac), 2*(cd+ab), aa+dd-bb-cc]])

def altaz2xyz(alt_deg, az_deg, lon_deg, lat_deg):
    v = np.array([-1., 0., 0.])
    v = np.dot(rotation_matrix(np.array([0,1,0]), -np.radians(alt_deg)), v)
    v = np.dot(rotation_matrix(np.array([0,0,1]), np.radians(az_deg)), v)
    v = np.dot(rotation_matrix(np.array([0,1,0]), np.radians(lat_deg-90)), v)
    v = np.dot(rotation_matrix(np.array([0,0,1]), -np.radians(lon_deg)), v)
    return v

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))

def dist_line_line(p1, u1, p2, u2):
    """Calculates the shortest distance between two lines in 3D space."""
    a = np.array([[np.dot(u1, u1), -np.dot(u1, u2)], [np.dot(u2, u1), -np.dot(u2, u2)]])
    b = np.array([np.dot(u1, (p2 - p1)), np.dot(u2, (p2 - p1))])
    try:
        # Solve for the parameters s and t that give the closest points
        s, t = np.linalg.solve(a, b)
        return np.linalg.norm((p1 + s * u1) - (p2 + t * u2))
    except np.linalg.LinAlgError:
        # Lines are parallel, calculate distance from a point to the line
        return np.linalg.norm(np.cross(u1, p2 - p1))

def closest_point(p1, p2, u1, u2, return_points=False):
    p21 = p2 - p1
    m = np.cross(u2, u1)
    m2 = float(np.dot(m, m))
    if m2 < 1e-9: return (p1 + p2)/2.0
    R = np.cross(p21, m / m2)
    t1, t2 = np.dot(R, u2), np.dot(R, u1)
    cross_1, cross_2 = p1 + t1 * u1, p2 + t2 * u2
    return (cross_1, cross_2) if return_points else (cross_1 + cross_2) / 2.0

def intersec_line_plane(line_ref, line_vec, plane_ref, plane_vec):
    dot_product = np.dot(plane_vec, line_vec)
    if abs(dot_product) < 1e-9: return None
    factor = -np.dot(plane_vec, line_ref - plane_ref) / dot_product
    return line_ref + line_vec * factor

def altaz_to_radec(longitude, latitude, alt, az, timestamp):
    if 'ephem' not in AVAILABLE_LIBS: return 0, 0
    ephem = AVAILABLE_LIBS['ephem']
    obs = ephem.Observer()
    obs.long, obs.lat = str(longitude), str(latitude)
    obs.pressure, obs.epoch = 0, ephem.J2000
    obs.date = ephem.Date(datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc))
    return obs.radec_of(str(az), str(alt))

def radec_to_ecliptic(ra, dec):
    if 'ephem' not in AVAILABLE_LIBS: return 0, 0
    ephem = AVAILABLE_LIBS['ephem']
    equ = ephem.Equatorial(ra, dec, epoch=ephem.J2000)
    ecl = ephem.Ecliptic(equ)
    return ecl.lon, ecl.lat

# --- Core Logic, I/O, and Plotting ---
def clean_svg(svg_path):
    """Optimizes an SVG file in place using the Scour library."""
    if 'scour' not in AVAILABLE_LIBS:
        print("Warning: Scour library not installed. Cannot clean SVG.")
        return
    try:
        from scour import scour
        options = Namespace(remove_metadata=True, strip_comments=True, enable_viewboxing=True, indent_type='none', simple_colors=False, style_to_xml=True, group_collapse=True, create_groups=True, keep_editor_data=False, keep_defs=False, renderer_workaround=True, strip_xml_prolog=False, remove_titles=True, remove_descriptions=True, remove_ids='all', protect_ids_noninkscape=False, digits=5)
        with open(svg_path, 'r+', encoding='utf-8') as f:
            in_svg = f.read()
            f.seek(0)
            cleaned_svg = scour.scourString(in_svg, options)
            f.write(cleaned_svg)
            f.truncate()
    except Exception as e:
        print(f"An error occurred while cleaning SVG {svg_path}: {e}")

def plot_height(track_start, track_end, cross_pos, obs_data, inlier_indices, options):
    """Shows the vertical path of a track, distinguishing inliers and outliers."""
    if 'matplotlib' not in AVAILABLE_LIBS: return
    n_steps, n_obs = 100, len(obs_data['names']) // 2
    start_lon, start_lat, _ = xyz2lonlat(track_start)
    
    pylab.figure(figsize=(10, 8))
    
    # Plot the fitted track
    x_track = np.linspace(0, 1, n_steps)
    track_points = track_start + x_track[:, np.newaxis] * (track_end - track_start)
    lons, lats, heights = zip(*[xyz2lonlat(p) for p in track_points])
    ground_dists = haversine(start_lon, start_lat, np.array(lons), np.array(lats))
    pylab.plot(ground_dists, heights, 'g-', label='Estimert bane', zorder=1)

    # Plot observation start and end points and add labels in a single loop
    for i in range(n_obs):
        start_point, end_point = cross_pos[i], cross_pos[i + n_obs]
        points = [start_point, end_point]
        heights = [xyz2lonlat(p)[2] for p in points]
        dists = [haversine(start_lon, start_lat, *xyz2lonlat(p)[:2]) for p in points]
        
        is_inlier = i in inlier_indices
        color, zorder = ('r', 3) if is_inlier else ('k', 2)
        mfc = color if is_inlier else 'none'
        label = 'Tellende stasjon' if is_inlier else 'Avvikende stasjon'
        pylab.plot(dists, heights, color=color, marker='o', mfc=mfc, linestyle='None', label=label, zorder=zorder)
    
    # After plotting all points, get axis limits for dynamic text offset
    ymin, ymax = pylab.gca().get_ylim()
    y_offset = (ymax - ymin) * 0.015  # 1.5% of the plot height

    for i in range(n_obs):
        start_dist = haversine(start_lon, start_lat, *xyz2lonlat(cross_pos[i])[:2])
        start_height = xyz2lonlat(cross_pos[i])[2]
        end_dist = haversine(start_lon, start_lat, *xyz2lonlat(cross_pos[i+n_obs])[:2])
        end_height = xyz2lonlat(cross_pos[i+n_obs])[2]
        pylab.text(start_dist, start_height + y_offset, obs_data['names'][i], ha='center', va='bottom', fontsize=8)
        pylab.text(end_dist, end_height - y_offset, obs_data['names'][i], ha='center', va='top', fontsize=8)

    pylab.xlabel('Ground Distance (km)'); pylab.ylabel('Height (km)')
    handles, labels = pylab.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    pylab.legend(by_label.values(), by_label.keys())
    
    if 'save' in options['doplot']: pylab.savefig('height.svg'); clean_svg('height.svg')
    if 'show' in options['doplot']: pylab.show()
    pylab.close()

def plot_map(track_start, track_end, cross_pos, obs_data, inlier_indices, options):
    """Shows a map of the track and lines of sight, distinguishing inliers and outliers."""
    if 'cartopy' not in AVAILABLE_LIBS or 'matplotlib' not in AVAILABLE_LIBS:
        print("Cannot plot map, missing 'cartopy' or 'matplotlib'.")
        return
    
    site_lons, site_lats = obs_data['longitudes'], obs_data['latitudes']
    n_obs = len(site_lons) // 2
    if options['autoborders']:
        all_lons, all_lats = list(site_lons), list(site_lats)
        if not options['azonly'] and cross_pos:
            los_lons, los_lats, _ = zip(*[xyz2lonlat(p) for p in cross_pos])
            all_lons.extend(los_lons); all_lats.extend(los_lats)
        lon_left, lon_right = min(all_lons) - 1, max(all_lons) + 1
        lat_bot, lat_top = min(all_lats) - 0.5, max(all_lats) + 0.5
    else:
        lon_left, lon_right, lat_bot, lat_top = options['borders']

    pylab.figure(figsize=(10, 10))
    proj = ccrs.Gnomonic(central_longitude=np.mean([lon_left, lon_right]), central_latitude=np.mean([lat_bot, lat_top]))
    ax = pylab.axes(projection=proj)
    ax.set_extent([lon_left, lon_right, lat_bot, lat_top], crs=ccrs.PlateCarree())
    
    resolution = {'c': '110m', 'l': '50m', 'i': '10m', 'h': '10m', 'f': '10m'}.get(options['mapres'], '10m')
    lat_span = abs(lat_top - lat_bot)
    zoom_level = int(np.log2(360 / (lat_span + 1)))
    zoom_level = max(6, min(zoom_level, 9))
    
    try:
        ax.add_image(OSM(), zoom_level)
        ax.add_feature(cfeature.LAKES, edgecolor='royalblue', linewidth=0.5, facecolor='none')
    except Exception: ax.add_feature(cfeature.LAND); ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution)); ax.add_feature(cfeature.BORDERS.with_scale(resolution))
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xformatter, gl.yformatter = LongitudeFormatter(), LatitudeFormatter()

    # Determine unique stations and their inlier status
    unique_stations = {obs_data['names'][i]: (site_lons[i], site_lats[i]) for i in range(n_obs)}
    station_is_inlier = {name: False for name in unique_stations}
    for i in inlier_indices: station_is_inlier[obs_data['names'][i]] = True

    # Plot each unique station marker once
    for name, (lon, lat) in unique_stations.items():
        if station_is_inlier[name]:
            ax.plot(lon, lat, 'r*', markersize=10, transform=ccrs.PlateCarree(), label='Tellende stasjon')
        else:
            ax.plot(lon, lat, 'ko', markersize=8, mfc='none', transform=ccrs.PlateCarree(), label='Avvikende stasjon')
        ax.text(lon + 0.05, lat + 0.05, name, color='black', transform=ccrs.PlateCarree())

    # Plot lines of sight for all observations
    if not options['azonly'] and cross_pos:
        for i in range(n_obs):
            start_los_lon, start_los_lat, _ = xyz2lonlat(cross_pos[i])
            end_los_lon, end_los_lat, _ = xyz2lonlat(cross_pos[i + n_obs])
            linestyle = '-' if i in inlier_indices else '--'
            ax.plot([site_lons[i], start_los_lon], [site_lats[i], start_los_lat], color='#5499c7', transform=ccrs.PlateCarree(), linestyle=linestyle)
            ax.plot([site_lons[i], end_los_lon], [site_lats[i], end_los_lat], color='#1a5276', transform=ccrs.PlateCarree(), linestyle=linestyle)

    # Plot the final fitted track
    if not options['azonly'] and track_start is not None:
        start_lon, start_lat, _ = xyz2lonlat(track_start)
        end_lon, end_lat, _ = xyz2lonlat(track_end)
        ax.plot([start_lon, end_lon], [start_lat, end_lat], color='blue', linewidth=2, transform=ccrs.PlateCarree(), label='Estimert bane')
        ax.annotate('', xy=(end_lon, end_lat), xytext=(start_lon, start_lat), arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->'), transform=ccrs.PlateCarree())

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    if 'save' in options['doplot']: pylab.savefig('map.svg', bbox_inches='tight', dpi=150); clean_svg('map.svg')
    if 'show' in options['doplot']: pylab.show()
    pylab.close()

def chisq_of_fit(track_params, los_refs, los_vecs, weights):
    track_ref, track_vec = track_params[:3], track_params[3:]
    n_obs = len(weights)
    return sum((weights[i % n_obs] * dist_line_line(track_ref, track_vec, ref, vec))**2 for i, (ref, vec) in enumerate(zip(los_refs, los_vecs)))

def fit_track(obs_data, optimize=True):
    n_obs = len(obs_data['longitudes']) // 2
    if n_obs < 2 or len(set(zip(obs_data['latitudes'][:n_obs], obs_data['longitudes'][:n_obs]))) < 2:
        return None, None, [], float('inf'), 0

    pos_vectors = [lonlat2xyz(lon, lat, h) for lon, lat, h in zip(obs_data['longitudes'], obs_data['latitudes'], obs_data['heights_m'])]
    los_vectors = [altaz2xyz(alt, az, lon, lat) for alt, az, lon, lat in zip(obs_data['altitudes'], obs_data['azimuths'], obs_data['longitudes'], obs_data['latitudes'])]
    
    # --- Initial Guess for Track Vector ---
    plane_normals = [np.cross(los_vectors[i], los_vectors[i + n_obs]) for i in range(n_obs)]
    plane_normals = [n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else n for n in plane_normals]
    
    dir_vectors, vec_weights = [], []
    for i, j in itertools.combinations(range(n_obs), 2):
        this_vec = np.cross(plane_normals[j], plane_normals[i])
        norm = np.linalg.norm(this_vec)
        if norm > 1e-9:
            dir_vectors.append(this_vec / norm)
            vec_weights.append(norm * obs_data['weights'][i] * obs_data['weights'][j])

    fit_quality = sum(w for w in vec_weights if not np.isnan(w))
    if fit_quality < 0.02: optimize = False

    # --- Initial Guess for Track Position and Direction ---
    if optimize:
        start_coords = [p for i, j in itertools.combinations(range(n_obs), 2) for p in (intersec_line_plane(pos_vectors[i], los_vectors[i], pos_vectors[j], plane_normals[j]), intersec_line_plane(pos_vectors[j], los_vectors[j], pos_vectors[i], plane_normals[i])) if p is not None]
        end_coords = [p for i, j in itertools.combinations(range(n_obs), 2) for p in (intersec_line_plane(pos_vectors[i+n_obs], los_vectors[i+n_obs], pos_vectors[j], plane_normals[j]), intersec_line_plane(pos_vectors[j+n_obs], los_vectors[j+n_obs], pos_vectors[i], plane_normals[i])) if p is not None]
        start_coord = np.average(start_coords, axis=0) if start_coords else np.zeros(3)
        end_coord = np.average(end_coords, axis=0) if end_coords else np.zeros(3)
        guess_fit = end_coord - start_coord
        best_fit_dir = np.average(dir_vectors, axis=0, weights=vec_weights if any(vec_weights) else None) if dir_vectors else guess_fit
        if np.dot(guess_fit, best_fit_dir) < 0: best_fit_dir = -best_fit_dir
    else:
        start_points = [closest_point(pos_vectors[i], pos_vectors[j], los_vectors[i], los_vectors[j]) for i, j in itertools.combinations(range(n_obs), 2)]
        end_points = [closest_point(pos_vectors[i+n_obs], pos_vectors[j+n_obs], los_vectors[i+n_obs], los_vectors[j+n_obs]) for i,j in itertools.combinations(range(n_obs), 2)]
        start_coord, end_coord = np.mean(start_points, axis=0), np.mean(end_points, axis=0)
        best_fit_dir = end_coord - start_coord

    if np.linalg.norm(best_fit_dir) < 1e-9: return None, None, [], float('inf'), fit_quality
    best_fit_dir /= np.linalg.norm(best_fit_dir)
    
    # --- Optimization ---
    initial_params = np.concatenate([start_coord, best_fit_dir])
    if optimize and 'scipy' in AVAILABLE_LIBS:
        final_params, _, _, _, _, flag = fmin_powell(chisq_of_fit, initial_params, args=(pos_vectors, los_vectors, obs_data['weights']), disp=False, full_output=True)
        if flag != 0: final_params = initial_params
    else: final_params = initial_params
    
    track_chi2 = chisq_of_fit(final_params, pos_vectors, los_vectors, obs_data['weights'])
    start_coord, best_fit_dir = final_params[:3], final_params[3:]

    if np.linalg.norm(best_fit_dir) < 1e-9: return None, None, [], float('inf'), fit_quality
    best_fit_dir /= np.linalg.norm(best_fit_dir)
    if np.dot(initial_params[3:], best_fit_dir) < 0: best_fit_dir = -best_fit_dir

    # --- Finalize Track Geometry ---
    cross_positions, dists = [], []
    for i in range(len(obs_data['longitudes'])):
        los_pos, track_pos = closest_point(pos_vectors[i], start_coord, los_vectors[i], best_fit_dir, return_points=True)
        cross_positions.append(los_pos)
        dists.append(np.linalg.norm(track_pos - start_coord) * np.sign(np.dot(track_pos - start_coord, best_fit_dir)))
    
    track_start = start_coord + min(dists) * best_fit_dir
    track_end = start_coord + max(dists) * best_fit_dir
    return track_start, track_end, cross_positions, track_chi2, fit_quality

def _create_subset_obs_data(full_obs_data, indices):
    """Creates a new obs_data dictionary containing only data for the given station indices."""
    n_full = len(full_obs_data['names']) // 2
    all_indices = list(indices) + [i + n_full for i in indices]
    subset = {key: value[all_indices] if key != 'durations' else value[list(indices)] for key, value in full_obs_data.items() if isinstance(value, np.ndarray)}
    subset['names'] = [full_obs_data['names'][i] for i in all_indices]
    return subset

def robust_fit_with_ransac(obs_data, raw_data, options):
    """
    Performs a robust trajectory fit using a multi-run RANSAC approach.

    This algorithm works in two main phases:
    1.  **Candidate Generation:** It performs multiple independent RANSAC runs.
        Each run iterates many times, randomly sampling minimal sets of
        observations to generate a trial trajectory. It then finds the
        "consensus set" (all observations consistent with this trial trajectory)
        and keeps the largest, lowest-error set it finds in that run.
        The best set from each run becomes a "candidate model".

    2.  **Final Selection (Run-off):** After collecting unique candidate sets
        from all the runs, it performs a final, high-quality *optimized* fit
        on each one. It then selects the absolute best model based on a clear
        hierarchy:
            a. The model with the most inliers is preferred.
            b. If multiple models have the same (maximum) number of inliers,
               the one with the lowest final fit error is chosen.

    This multi-run, two-phase approach is highly robust against local minima
    and is much more likely to find the true global optimum solution than a
    single RANSAC run.
    """
    random.seed(options['seed'])
    num_stations = len(raw_data['names'])
    if num_stations < 2:
        print("Not enough stations for RANSAC, falling back to simple fit.")
        return fit_track(obs_data, optimize=options['optimize']), obs_data, list(range(num_stations))

    pos_vectors = [lonlat2xyz(lon, lat, h) for lon, lat, h in zip(obs_data['longitudes'], obs_data['latitudes'], obs_data['heights_m'])]
    los_vectors = [altaz2xyz(alt, az, lon, lat) for alt, az, lon, lat in zip(obs_data['altitudes'], obs_data['azimuths'], obs_data['longitudes'], obs_data['latitudes'])]

    if options['debug_ransac']: print("--- RANSAC Debugging Enabled ---")
    candidate_sets = set()

    # --- Phase 1: Candidate Generation via Multiple RANSAC Runs ---
    for run in range(options['ransac_runs']):
        best_inlier_indices_this_run, best_fit_error_this_run = set(), float('inf')
        for i in range(options['ransac_iterations']):
            # Ensure the minimal sample is from two different locations
            while True:
                sample_indices = random.sample(range(num_stations), 2)
                loc1 = (raw_data['latitudes'][sample_indices[0]], raw_data['longitudes'][sample_indices[0]])
                loc2 = (raw_data['latitudes'][sample_indices[1]], raw_data['longitudes'][sample_indices[1]])
                if loc1 != loc2: break
            
            candidate_start, candidate_end, _, _, _ = fit_track(_create_subset_obs_data(obs_data, sample_indices), optimize=False)
            if candidate_start is None: continue

            track_ref, track_vec = candidate_start, (candidate_end - candidate_start)
            track_vec /= np.linalg.norm(track_vec)
            
            # Find consensus set
            current_inlier_indices = set(sample_indices)
            for j in range(num_stations):
                if j in sample_indices: continue
                dist = (dist_line_line(track_ref, track_vec, pos_vectors[j], los_vectors[j]) + dist_line_line(track_ref, track_vec, pos_vectors[j + num_stations], los_vectors[j + num_stations])) / 2.0
                if dist < options['ransac_threshold']: current_inlier_indices.add(j)
            
            # If this model is potentially better, check its non-optimized error
            if len(current_inlier_indices) >= len(best_inlier_indices_this_run):
                _, _, _, current_chi2, _ = fit_track(_create_subset_obs_data(obs_data, current_inlier_indices), optimize=False)
                if np.isfinite(current_chi2) and (len(current_inlier_indices) > len(best_inlier_indices_this_run) or current_chi2 < best_fit_error_this_run):
                    best_fit_error_this_run = current_chi2
                    best_inlier_indices_this_run = current_inlier_indices
        
        if best_inlier_indices_this_run: candidate_sets.add(frozenset(best_inlier_indices_this_run))

    # --- Phase 2: Final Selection from Candidates ---
    if options['debug_ransac']: print(f"\n--- RANSAC Final Run-off from {options['ransac_runs']} runs ---\nFound {len(candidate_sets)} unique candidate sets to evaluate.")
    
    best_final_model, best_final_model_score = None, (0, float('inf'))
    for k, indices_set in enumerate(candidate_sets):
        if len(indices_set) < 2: continue
        inlier_obs_data = _create_subset_obs_data(obs_data, indices_set)
        fit_results = fit_track(inlier_obs_data, optimize=True)
        final_error, final_quality = fit_results[3], fit_results[4]
        current_score = (-len(indices_set), (final_error + 1) / (final_quality + 1e-9))

        if options['debug_ransac']:
            inlier_names = sorted([raw_data['names'][i] for i in indices_set])
            print(f"Candidate {k+1}: {len(indices_set)} inliers, final_err={final_error:.2f}, quality={final_quality:.2f}, score={current_score[1]:.2f} -> {inlier_names}")

        if current_score < best_final_model_score:
            best_final_model_score = current_score
            best_final_model = (fit_results, inlier_obs_data, sorted(list(indices_set)))
    
    if best_final_model is None:
        print("RANSAC failed to find a valid model. Falling back to simple fit on all data.")
        return fit_track(obs_data, optimize=True), obs_data, list(range(num_stations))

    # --- Phase 3: "All-In" Sanity Check ---
    all_in_fit_results = fit_track(obs_data, optimize=True)
    all_in_error = all_in_fit_results[3]
    if options['debug_ransac']:
        print(f"\n--- Final Sanity Check ---\nBest RANSAC result score: {best_final_model_score[1]:.2f} with {-best_final_model_score[0]} inliers.\nAll-in fit error: {all_in_error:.2f} with {num_stations} inliers.")
    
    if all_in_error < options['all_in_tolerance']:
        if options['debug_ransac']: print("All-in error is below tolerance. Using all stations.")
        final_fit_results, final_obs_data, final_inlier_indices = all_in_fit_results, obs_data, list(range(num_stations))
    else:
        if options['debug_ransac']: print("All-in error is too high. Using best RANSAC result.")
        final_fit_results, final_obs_data, final_inlier_indices = best_final_model

    unique_inlier_names = {raw_data['names'][i] for i in final_inlier_indices}
    print(f"Final solution uses {len(final_inlier_indices)} observations from {len(unique_inlier_names)} unique stations.")
    return final_fit_results, final_obs_data, final_inlier_indices

def _load_and_prepare_data(filepath):
    """Parses the input file and prepares data arrays for computation."""
    station_data_list = []
    borders = None
    with open(filepath, 'r') as f:
        for line in f:
            words = line.split()
            if not words or line.startswith('#'): continue
            if words[0].lower() == 'borders':
                borders = [float(w) for w in words[1:5]]
                continue
            
            def is_number(s):
                try: float(s); return True
                except ValueError: return False
            
            idx = 12
            while idx < len(words) and not is_number(words[idx]): idx += 1
            
            station_data_list.append({
                'longitudes': float(words[0]), 'latitudes': float(words[1]), 'az_start': float(words[2]),
                'az_end': float(words[3]), 'alt_start': float(words[4]), 'alt_end': float(words[5]),
                'weight': float(words[6]), 'duration': float(words[7]), 'length': float(words[8]),
                'color': tuple(map(int, words[9:12])), 'names': " ".join(words[12:idx]),
                'timestamp': float(words[idx]) if idx < len(words) else None,
                'height_m': float(words[idx+1]) / 1000.0 if idx+1 < len(words) else 0.0
            })

    if not station_data_list: # Handle empty input file
        return {}, {}, None

    raw_data = {key: [s[key] for s in station_data_list] for key in station_data_list[0]}
    
    lons, lats, heights_m, weights, names = raw_data['longitudes'], raw_data['latitudes'], raw_data['height_m'], raw_data['weight'], raw_data['names']
    
    full_obs_data = {
        'longitudes': np.array(lons * 2), 'latitudes': np.array(lats * 2), 'heights_m': np.array(heights_m * 2),
        'azimuths': np.array(raw_data['az_start'] + raw_data['az_end']),
        'altitudes': np.array(raw_data['alt_start'] + raw_data['alt_end']),
        'weights': np.array(weights * 2), 'durations': np.array(raw_data['duration']), 'names': names * 2
    }
    return raw_data, full_obs_data, borders

def print_results(info):
    print("\n--- Metrack Analysis Results ---")
    print(f"[Track] Start/End Height:   {info.start_height:.2f} / {info.end_height:.2f} km")
    print(f"[Track] Ground Track:       {info.ground_track:.2f} km")
    print(f"[Track] Course / Incidence: {info.course:.2f} / {info.incidence:.2f} deg")
    if info.speed > 0: print(f"[Track] Avg. Speed:         {info.speed:.1f} km/s")
    unique_inlier_names = sorted(list(set(info.inlier_stations)))
    print(f"[Fit]   Inliers:            {len(info.inlier_stations)} observations from {len(unique_inlier_names)} unique stations")
    print(f"[Fit]   Inlier Stations:    {', '.join(unique_inlier_names)}")
    print(f"[Fit]   Error / Quality:    {info.error:.2f} / {info.fit_quality:.2f}")
    print(f"[Radiant] RA / Dec:         {info.radiant_ra:.3f} / {info.radiant_dec:.3f} deg")
    print(f"[Radiant] Ecl. Lon / Lat:   {info.radiant_ecllong:.3f} / {info.radiant_ecllat:.3f} deg")
    print(f"[Radiant] Shower Assoc:     {info.shower}")
    print(f"[Time]    Date (UTC):       {info.date}")
    print("--------------------------------\n")
    
def write_stat_file(info, in_name):
    output_path = Path(in_name).with_suffix('.stat')
    config = configparser.ConfigParser()
    unique_inlier_names = sorted(list(set(info.inlier_stations)))
    config['track'] = {'startheight':f'{info.start_height:.1f} km','endheight':f'{info.end_height:.1f} km','groundtrack':f'{info.ground_track:.1f} km','course':f'{info.course:.1f} deg','incidence':f'{info.incidence:.1f} deg', 'speed':f'{info.speed:.1f} km/s'}
    config['fit'] = {'error':f'{info.error:.1f}','quality':f'{info.fit_quality:.2f}', 'inliers': ', '.join(unique_inlier_names)}
    config['radiant'] = {'ra':f'{info.radiant_ra:.2f} deg','dec':f'{info.radiant_dec:.2f} deg','ecl_long':f'{info.radiant_ecllong:.2f} deg','ecl_lat':f'{info.radiant_ecllat:.2f} deg','shower':info.shower}
    config['date'] = {'timestamp':f'{info.timestamp}','date':info.date}
    with open(output_path, 'w') as f: config.write(f)
    print(f"Results saved to {output_path}")

def write_res_file(track_start, track_end, cross_pos, obs_data, in_name):
    """Writes the detailed results to a .res file."""
    output_path = Path(in_name).with_suffix('.res')
    start_lon, start_lat, start_height = xyz2lonlat(track_start)
    end_lon, end_lat, end_height = xyz2lonlat(track_end)
    with open(output_path, 'w') as f:
        f.write(f'{start_lon:8.4f} {start_lat:10.6f} {start_lon:8.4f} {start_lat:8.4f} {start_height:6.1f} Start\n')
        f.write(f'{end_lon:8.4f} {end_lat:10.6f} {end_lon:8.4f} {end_lat:8.4f} {end_height:6.1f} End\n')
        for i in range(len(cross_pos)):
            cross_lon, cross_lat, cross_height = xyz2lonlat(cross_pos[i])
            site_lon, site_lat, site_name = obs_data['longitudes'][i], obs_data['latitudes'][i], obs_data['names'][i]
            site_height_km = obs_data['heights_m'][i]
            f.write(f'{site_lon:6.5f} {site_lat:6.5f} {cross_lon:8.5f} {cross_lat:8.5f} {cross_height:6.3f} {site_name} {site_height_km:6.3f}\n')
    print(f"Detailed results saved to {output_path}")

def metrack(inname, doplot='', accept_err=0, mapres='i', azonly=False, autoborders=False, 
            timestamp=None, optimize=True, writestat=True, use_ransac=True, ransac_threshold=1.0, 
            ransac_iterations=10, ransac_runs=100, seed=0, debug_ransac=False, all_in_tolerance=1.0, **kwargs):
    """
    Main function to run the meteor track analysis.
    This is the primary entry point for both script and module usage.
    """
    options = locals().copy() # Capture all arguments in a dictionary
    options.pop('kwargs', None); options.pop('accept_err', None)
    
    raw_data, full_obs_data, borders = _load_and_prepare_data(inname)
    if not raw_data:
        print(f"Error: No valid data found in {inname}")
        return MetrackInfo()

    if borders and not autoborders: options['borders'] = borders
    else: options['borders'] = None
    
    if azonly:
        print("Azimuth-only mode: Plotting observation azimuths without fitting.")
        if doplot: plot_map(None, None, None, full_obs_data, list(range(len(raw_data['names']))), options)
        return MetrackInfo()

    if use_ransac:
        (track_start, track_end, cross_pos_inliers, track_err, fit_quality), inlier_obs_data, inlier_indices = robust_fit_with_ransac(full_obs_data, raw_data, options)
    else:
        print("RANSAC disabled. Using standard fit on all data.")
        (track_start, track_end, cross_pos_inliers, track_err, fit_quality) = fit_track(full_obs_data, optimize=optimize)
        inlier_obs_data, inlier_indices = full_obs_data, list(range(len(raw_data['names'])))

    if track_start is None:
        print("Could not determine a valid track. Aborting.")
        return MetrackInfo()

    info = MetrackInfo()
    info.inlier_stations = [raw_data['names'][i] for i in inlier_indices]
    info.error, info.fit_quality = track_err, fit_quality
    start_lon, start_lat, info.start_height = xyz2lonlat(track_start)
    end_lon, end_lat, info.end_height = xyz2lonlat(track_end)
    
    if info.end_height > info.start_height:
        track_start, track_end = track_end, track_start
        start_lon, start_lat, info.start_height = xyz2lonlat(track_start)
        end_lon, end_lat, info.end_height = xyz2lonlat(track_end)

    info.ground_track = haversine(start_lon, start_lat, end_lon, end_lat)

    dir_vec = track_end - track_start
    dir_vec_rot_lon = np.dot(rotation_matrix(np.array([0,0,1]), np.radians(start_lon)), dir_vec)
    dir_vec_local = np.dot(rotation_matrix(np.array([0,1,0]), np.radians(90 - start_lat)), dir_vec_rot_lon)

    info.course = np.degrees(np.arctan2(dir_vec_local[1], -dir_vec_local[0])) % 360
    info.incidence = -np.degrees(np.arctan2(dir_vec_local[2], np.hypot(dir_vec_local[0], dir_vec_local[1])))

    valid_times = [t for t in raw_data['timestamp'] if t]
    info.timestamp = timestamp if timestamp is not None else np.mean(valid_times) if valid_times else time.time()
    info.date = time.asctime(time.gmtime(info.timestamp))
    
    ra, dec = altaz_to_radec(start_lon, start_lat, -info.incidence, (info.course + 180) % 360, info.timestamp)
    info.radiant_ra, info.radiant_dec = np.degrees(float(ra)), np.degrees(float(dec))
    info.radiant_ecllong, info.radiant_ecllat = radec_to_ecliptic(ra, dec)
    
    airspeeds = [np.linalg.norm(cross_pos_inliers[i + len(inlier_obs_data['durations'])] - cross_pos_inliers[i]) / d for i, d in enumerate(inlier_obs_data['durations']) if d > 0]
    if airspeeds: info.speed = np.mean(airspeeds)

    if 'showerassoc' in AVAILABLE_LIBS:
        info.shower, _ = AVAILABLE_LIBS['showerassoc'].showerassoc(info.radiant_ra, info.radiant_dec, info.speed, time.strftime("%Y-%m-%d", time.localtime(info.timestamp)))
    
    print_results(info)
    if writestat: write_stat_file(info, inname)
    write_res_file(track_start, track_end, cross_pos_inliers, inlier_obs_data, inname)
    
    if doplot:
        final_track_vec = (track_end - track_start)
        final_track_vec /= np.linalg.norm(final_track_vec)
        all_pos_vectors = [lonlat2xyz(lon, lat, h) for lon, lat, h in zip(full_obs_data['longitudes'], full_obs_data['latitudes'], full_obs_data['heights_m'])]
        all_los_vectors = [altaz2xyz(alt, az, lon, lat) for alt, az, lon, lat in zip(full_obs_data['altitudes'], full_obs_data['azimuths'], full_obs_data['longitudes'], full_obs_data['latitudes'])]
        cross_pos_all = [closest_point(pos, track_start, los, final_track_vec, return_points=True)[0] for pos, los in zip(all_pos_vectors, all_los_vectors)]

        plot_height(track_start, track_end, cross_pos_all, full_obs_data, inlier_indices, options)
        plot_map(track_start, track_end, cross_pos_all, full_obs_data, inlier_indices, options)

    return info

def main():
    """Parses command-line arguments and runs the analysis."""
    parser = argparse.ArgumentParser(description='Atmospheric meteor trajectory fitting.', epilog='Example: ./metrack.py obs_20120403.dat -o both -m i --borders-auto')
    parser.add_argument('inname', nargs='?', default=None, help='Name of input file with observation data.')
    parser.add_argument('-o', '--output', choices=['show', 'save', 'both'], default='', dest='doplot', help='Graphics output mode.')
    parser.add_argument('-m', '--map', dest='mapres', default='i', choices=['c','l','i','h','f'], help='Map detail level.')
    parser.add_argument('-a', '--azonly', action='store_true', help='Plot azimuths only; no fitting.')
    parser.add_argument('-b', '--borders-auto', dest='autoborders', action='store_true', help='Automatically determine map boundaries.')
    parser.add_argument('-t', '--timestamp', type=float, help="Force a specific Unix timestamp.")
    parser.add_argument('--no-opt', action='store_false', dest='optimize', default=True, help="Turn off optimization.")
    parser.add_argument('--no-write-stat', action='store_false', dest='writestat', default=True, help="Do not write a .stat file.")
    parser.add_argument('--no-ransac', action='store_false', dest='use_ransac', default=True, help="Disable RANSAC robust fitting.")
    parser.add_argument('--ransac-threshold', type=float, default=1.0, help="RANSAC inlier distance threshold in km.")
    parser.add_argument('--ransac-iterations', type=int, default=10, help="Number of RANSAC iterations per run.")
    parser.add_argument('--ransac-runs', type=int, default=100, help="Number of independent RANSAC runs.")
    parser.add_argument('--all-in-tolerance', type=float, default=1.0, help="Error tolerance to accept a fit with all stations.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for RANSAC.")
    parser.add_argument('--debug-ransac', action='store_true', help="Enable RANSAC debug prints.")
    
    args = parser.parse_args()

    if not args.inname:
        parser.error("No input file specified.")
    if not Path(args.inname).is_file():
        parser.error(f"Input file not found: {args.inname}")
    if args.azonly and not args.doplot:
        args.doplot = 'show'
    
    metrack_args = vars(args)
    metrack_args['accept_err'] = 0
    metrack(**metrack_args)

if __name__ == "__main__":
    main()
