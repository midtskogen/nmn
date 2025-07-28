#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metrack: Atmospheric Meteor Trajectory Fitting Tool

This script reads a data file describing a set of meteor observations, calculates
the meteor's trajectory through the atmosphere, and outputs the results.
It can be run as a standalone script or imported as a module in other tools.
"""

import argparse
import configparser
import datetime
import os
import sys
import time
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
    a = np.array([[np.dot(u1, u1), -np.dot(u1, u2)], [np.dot(u2, u1), -np.dot(u2, u2)]])
    b = np.array([np.dot(u1, (p2 - p1)), np.dot(u2, (p2 - p1))])
    try:
        s, t = np.linalg.solve(a, b)
        return np.linalg.norm((p1 + s * u1) - (p2 + t * u2))
    except np.linalg.LinAlgError:
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
        
        # Create a minimal options object mimicking argparse.Namespace
        # This is compatible with modern Scour versions programmatically.
        options = Namespace(
            remove_metadata=True,
            strip_comments=True,
            enable_viewboxing=True,
            indent_type='none',
            simple_colors=False,
            style_to_xml=True,
            group_collapse=True,
            create_groups=True,
            keep_editor_data=False,
            keep_defs=False,
            renderer_workaround=True,
            strip_xml_prolog=False,
            remove_titles=True,
            remove_descriptions=True,
            remove_ids='all',
            protect_ids_noninkscape=False,
            digits=5
        )
        
        with open(svg_path, 'r+', encoding='utf-8') as f:
            in_svg = f.read()
            f.seek(0)
            cleaned_svg = scour.scourString(in_svg, options)
            f.write(cleaned_svg)
            f.truncate()
    except Exception as e:
        print(f"An error occurred while cleaning SVG {svg_path}: {e}")

def plot_height(track_start, track_end, cross_pos, obs_data, options):
    """Shows the vertical path of a track and the corresponding observations."""
    if 'matplotlib' not in AVAILABLE_LIBS: return
    
    n_steps = 100
    site_names = obs_data['names']
    start_lon, start_lat, _ = xyz2lonlat(track_start)
    los_heights = [xyz2lonlat(p)[2] for p in cross_pos]
    los_dists = [haversine(start_lon, start_lat, *xyz2lonlat(p)[:2]) for p in cross_pos]

    x_track = np.zeros(n_steps)
    y_track = np.zeros(n_steps)
    for i in range(n_steps):
        track_pos = track_start + float(i) / (n_steps - 1) * (track_end - track_start)
        lon, lat, height = xyz2lonlat(track_pos)
        x_track[i] = haversine(start_lon, start_lat, lon, lat)
        y_track[i] = height

    pylab.figure(figsize=(10, 8))
    pylab.plot(los_dists, los_heights, 'ro')
    pylab.plot(x_track, y_track, 'g-')
    for i in range(len(site_names)):
        pylab.text(los_dists[i], los_heights[i], site_names[i])
    pylab.xlabel('Ground Distance (km)')
    pylab.ylabel('Height (km)')
    
    if 'save' in options.output:
        pylab.savefig('height.svg')
        clean_svg('height.svg')
    if 'show' in options.output:
        pylab.show()
    pylab.close()

def plot_map(track_start, track_end, cross_pos, obs_data, options):
    """Shows a map of the track and lines of sight using Cartopy."""
    if 'cartopy' not in AVAILABLE_LIBS or 'matplotlib' not in AVAILABLE_LIBS:
        print("Cannot plot map, missing 'cartopy' or 'matplotlib'.")
        return
    
    site_lons, site_lats, site_names = obs_data['longitudes'], obs_data['latitudes'], obs_data['names']
    n_obs = len(site_lons) // 2
    if options.autoborders:
        all_lons, all_lats = list(site_lons), list(site_lats)
        if not options.azonly and cross_pos:
            los_lons, los_lats, _ = zip(*[xyz2lonlat(p) for p in cross_pos])
            all_lons.extend(los_lons); all_lats.extend(los_lats)
        lon_left, lon_right = min(all_lons) - 1, max(all_lons) + 1
        lat_bot, lat_top = min(all_lats) - 0.5, max(all_lats) + 0.5
    else:
        lon_left, lon_right, lat_bot, lat_top = options.borders

    pylab.figure(figsize=(10, 10))
    proj = ccrs.Gnomonic(central_longitude=np.mean([lon_left, lon_right]), central_latitude=np.mean([lat_bot, lat_top]))
    ax = pylab.axes(projection=proj)
    ax.set_extent([lon_left, lon_right, lat_bot, lat_top], crs=ccrs.PlateCarree())
    
    resolution_map = {'c': '110m', 'l': '50m', 'i': '10m', 'h': '10m', 'f': '10m'}
    resolution = resolution_map.get(options.mapres, '10m')
    lat_span = abs(lat_top - lat_bot)
    zoom_level = int(np.log2(360 / (lat_span + 1))) if lat_span > 0 else 6
    zoom_level = max(5, min(zoom_level, 10))
    
    try: ax.add_image(OSM(), zoom_level)
    except Exception: ax.add_feature(cfeature.LAND); ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution))
    ax.add_feature(cfeature.BORDERS.with_scale(resolution))
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.xformatter, gl.yformatter = LongitudeFormatter(), LatitudeFormatter()

    startcol = '#5499c7'
    endcol = '#1a5276'
    arrowcol = 'blue'
    textcol = 'black'

    for i in range(n_obs):
        ax.plot(site_lons[i], site_lats[i], 'r*', markersize=10, transform=ccrs.PlateCarree())
        ax.text(site_lons[i] + 0.05, site_lats[i] + 0.05, site_names[i], color=textcol, transform=ccrs.PlateCarree())
        if not options.azonly and cross_pos:
            start_los_lon, start_los_lat, _ = xyz2lonlat(cross_pos[i])
            end_los_lon, end_los_lat, _ = xyz2lonlat(cross_pos[i + n_obs])
            ax.plot([site_lons[i], start_los_lon], [site_lats[i], start_los_lat], color=startcol, transform=ccrs.PlateCarree())
            ax.plot([site_lons[i], end_los_lon], [site_lats[i], end_los_lat], color=endcol, transform=ccrs.PlateCarree())

    if not options.azonly and track_start is not None:
        start_lon, start_lat, _ = xyz2lonlat(track_start)
        end_lon, end_lat, _ = xyz2lonlat(track_end)
        ax.plot([start_lon, end_lon], [start_lat, end_lat], color=arrowcol, linewidth=2, transform=ccrs.PlateCarree())
        ax.annotate('', xy=(end_lon, end_lat), xytext=(start_lon, start_lat), arrowprops=dict(facecolor=arrowcol, edgecolor=arrowcol, arrowstyle='->'), transform=ccrs.PlateCarree())

    if 'save' in options.output:
        pylab.savefig('map.svg', bbox_inches='tight', dpi=150)
        clean_svg('map.svg')
    if 'show' in options.output:
        pylab.show()
    pylab.close()

def chisq_of_fit(track_params, los_refs, los_vecs, weights):
    track_ref, track_vec = track_params[:3], track_params[3:]
    n_obs = len(weights)
    chisq = sum((weights[i % n_obs] * dist_line_line(track_ref, track_vec, ref, vec))**2 for i, (ref, vec) in enumerate(zip(los_refs, los_vecs)))
    return chisq

def fit_track(obs_data, optimize=True):
    n_obs = len(obs_data['longitudes']) // 2
    pos_vectors = [lonlat2xyz(lon, lat, h) for lon, lat, h in zip(obs_data['longitudes'], obs_data['latitudes'], obs_data['heights_m']) ]
    los_vectors = [altaz2xyz(alt, az, lon, lat) for alt, az, lon, lat in zip(obs_data['altitudes'], obs_data['azimuths'], obs_data['longitudes'], obs_data['latitudes'])]
    plane_normals = [np.cross(los_vectors[i], los_vectors[i + n_obs]) for i in range(n_obs)]
    plane_normals = [n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else n for n in plane_normals]

    dir_vectors, vec_weights = [], []
    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            this_vec = np.cross(plane_normals[j], plane_normals[i])
            norm = np.linalg.norm(this_vec)
            if norm > 1e-9:
                dir_vectors.append(this_vec / norm)
                vec_weights.append(norm * obs_data['weights'][i] * obs_data['weights'][j])

    fit_quality = sum(w for w in vec_weights if not np.isnan(w))
    if fit_quality < 0.02:
        print("Warning: Small angle between observed directions. Disabling optimization.")
        optimize = False

    if optimize:
        start_coords = [p for i in range(n_obs) for j in range(i + 1, n_obs) for p in [intersec_line_plane(pos_vectors[i], los_vectors[i], pos_vectors[j], plane_normals[j]), intersec_line_plane(pos_vectors[j], los_vectors[j], pos_vectors[i], plane_normals[i])] if p is not None]
        end_coords = [p for i in range(n_obs) for j in range(i + 1, n_obs) for p in [intersec_line_plane(pos_vectors[i+n_obs], los_vectors[i+n_obs], pos_vectors[j], plane_normals[j]), intersec_line_plane(pos_vectors[j+n_obs], los_vectors[j+n_obs], pos_vectors[i], plane_normals[i])] if p is not None]
        start_coord = np.average(start_coords, axis=0) if start_coords else np.zeros(3)
        end_coord = np.average(end_coords, axis=0) if end_coords else np.zeros(3)
        guess_fit = end_coord - start_coord
        best_fit_dir = np.average(dir_vectors, axis=0, weights=vec_weights if any(vec_weights) else None) if dir_vectors else guess_fit
        if np.dot(guess_fit, best_fit_dir) < 0: best_fit_dir = -best_fit_dir
    else:
        start_points = [closest_point(pos_vectors[i], pos_vectors[j], los_vectors[i], los_vectors[j]) for i in range(n_obs) for j in range(i + 1, n_obs)]
        end_points = [closest_point(pos_vectors[i+n_obs], pos_vectors[j+n_obs], los_vectors[i+n_obs], los_vectors[j+n_obs]) for i in range(n_obs) for j in range(i + 1, n_obs)]
        start_coord, end_coord = np.mean(start_points, axis=0), np.mean(end_points, axis=0)
        best_fit_dir = end_coord - start_coord
    
    best_fit_dir /= np.linalg.norm(best_fit_dir)
    initial_params = np.concatenate([start_coord, best_fit_dir])
    
    if optimize and 'scipy' in AVAILABLE_LIBS:
        final_params, _, _, _, _, flag = fmin_powell(chisq_of_fit, initial_params, args=(pos_vectors, los_vectors, obs_data['weights']), disp=False, full_output=True)
        if flag != 0: final_params = initial_params
    else:
        final_params = initial_params

    track_chi2 = chisq_of_fit(final_params, pos_vectors, los_vectors, obs_data['weights'])
    start_coord, best_fit_dir = final_params[:3], final_params[3:]
    best_fit_dir /= np.linalg.norm(best_fit_dir)
    if np.dot(initial_params[3:], best_fit_dir) < 0: best_fit_dir = -best_fit_dir

    cross_positions, dists = [], []
    for i in range(len(obs_data['longitudes'])):
        los_pos, track_pos = closest_point(pos_vectors[i], start_coord, los_vectors[i], best_fit_dir, return_points=True)
        cross_positions.append(los_pos)
        dists.append(np.linalg.norm(track_pos - start_coord) * np.sign(np.dot(track_pos - start_coord, best_fit_dir)))
    
    track_start = start_coord + min(dists) * best_fit_dir
    track_end = start_coord + max(dists) * best_fit_dir
    return track_start, track_end, cross_positions, track_chi2, fit_quality

def _parse_input_file(filepath):
    data = {'longitudes':[],'latitudes':[],'azimuth_start':[],'azimuth_end':[],'altitude_start':[],'altitude_end':[],'weights':[],'durations':[],'lengths':[],'colors':[],'names':[],'timestamps':[],'heights_m':[]}
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
            data['longitudes'].append(float(words[0])); data['latitudes'].append(float(words[1]))
            data['azimuth_start'].append(float(words[2])); data['azimuth_end'].append(float(words[3]))
            data['altitude_start'].append(float(words[4])); data['altitude_end'].append(float(words[5]))
            data['weights'].append(float(words[6])); data['durations'].append(float(words[7]))
            data['lengths'].append(float(words[8])); data['colors'].append(tuple(map(int, words[9:12])))
            name_parts, idx = [], 12
            while idx < len(words) and not is_number(words[idx]):
                name_parts.append(words[idx]); idx += 1
            data['names'].append(" ".join(name_parts))
            data['timestamps'].append(float(words[idx]) if idx < len(words) else None); idx += 1
            data['heights_m'].append(float(words[idx]) / 1000.0 if idx < len(words) else 0.0)
    return data, borders

def _prepare_data_arrays(raw_data):
    return {'longitudes': np.array(raw_data['longitudes'] * 2),'latitudes': np.array(raw_data['latitudes'] * 2),'heights_m': np.array(raw_data['heights_m'] * 2),'azimuths': np.array(raw_data['azimuth_start'] + raw_data['azimuth_end']),'altitudes': np.array(raw_data['altitude_start'] + raw_data['altitude_end']),'weights': np.array(raw_data['weights'] * 2),'durations': np.array(raw_data['durations']),'names': raw_data['names'] * 2,}

def print_results(info):
    print("\n--- Metrack Analysis Results ---")
    print(f"[Track] Start/End Height: {info.start_height:7.2f} / {info.end_height:7.2f} km")
    print(f"[Track] Ground Track:     {info.ground_track:7.2f} km")
    print(f"[Track] Course / Incidence: {info.course:7.2f} / {info.incidence:7.2f} deg")
    if info.speed > 0: print(f"[Track] Avg. Speed:       {info.speed:6.1f} km/s")
    print(f"[Fit]   Error / Quality:    {info.error:7.2f} / {info.fit_quality:7.2f}")
    print(f"[Radiant] RA / Dec:         {info.radiant_ra:7.3f} / {info.radiant_dec:7.3f} deg")
    print(f"[Radiant] Ecl. Lon / Lat:   {info.radiant_ecllong:7.3f} / {info.radiant_ecllat:7.3f} deg")
    print(f"[Radiant] Shower Assoc:     {info.shower}")
    print(f"[Time]    Date (UTC):       {info.date}")
    print("--------------------------------\n")
    
def write_stat_file(info, in_name):
    output_path = Path(in_name).with_suffix('.stat')
    config = configparser.ConfigParser()
    config['track'] = {'startheight':f'{info.start_height:.1f} km','endheight':f'{info.end_height:.1f} km','groundtrack':f'{info.ground_track:.1f} km','course':f'{info.course:.1f} deg','incidence':f'{info.incidence:.1f} deg','speed':f'{info.speed:.1f} km/s'}
    config['fit'] = {'error':f'{info.error:.1f}','quality':f'{info.fit_quality:.2f}'}
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
            f.write(f'{site_lon:6.2f} {site_lat:6.2f} {cross_lon:8.4f} {cross_lat:8.4f} {cross_height:6.1f} {site_name}\n')
    print(f"Detailed results saved to {output_path}")

def metrack(inname, doplot='', accept_err=0, mapres='i', azonly=False, autoborders=False, 
            timestamp=None, optimize=True, writestat=False, **kwargs):
    """
    Main function to run the meteor track analysis.
    This is the primary entry point for both script and module usage.
    `accept_err` is a deprecated argument kept for backward compatibility.
    """
    options = Namespace(
        input_file=inname, output=doplot, mapres=mapres, azonly=azonly,
        autoborders=autoborders, timestamp=timestamp, optimize=optimize,
        writestat=writestat, borders=None
    )
    
    raw_data, borders = _parse_input_file(options.input_file)
    if borders and not options.autoborders:
        options.borders = borders
    obs_data = _prepare_data_arrays(raw_data)
    
    if options.azonly:
        print("Azimuth-only mode: Plotting observation azimuths without fitting.")
        if options.output:
            plot_map(None, None, None, obs_data, options)
        return MetrackInfo()

    track_start, track_end, cross_pos, track_err, fit_quality = fit_track(obs_data, optimize=options.optimize)

    info = MetrackInfo()
    info.error, info.fit_quality = track_err, fit_quality
    start_lon, start_lat, info.start_height = xyz2lonlat(track_start)
    end_lon, end_lat, info.end_height = xyz2lonlat(track_end)
    if info.end_height > info.start_height:
        print("WARNING: Start height is lower than end height. Results might be inverted.")
    info.ground_track = haversine(start_lon, start_lat, end_lon, end_lat)

    dir_vec = track_end - track_start
    dir_vec_rot_lon = np.dot(rotation_matrix(np.array([0,0,1]), np.radians(start_lon)), dir_vec)
    dir_vec_local = np.dot(rotation_matrix(np.array([0,1,0]), np.radians(90 - start_lat)), dir_vec_rot_lon)

    info.course = np.degrees(np.arctan2(dir_vec_local[1], -dir_vec_local[0])) % 360
    info.incidence = -np.degrees(np.arctan2(dir_vec_local[2], np.hypot(dir_vec_local[0], dir_vec_local[1])))

    valid_times = [t for t in raw_data['timestamps'] if t]
    info.timestamp = options.timestamp if options.timestamp is not None else np.mean(valid_times) if valid_times else time.time()
    info.date = time.asctime(time.gmtime(info.timestamp))
    
    ra, dec = altaz_to_radec(start_lon, start_lat, -info.incidence, (info.course + 180) % 360, info.timestamp)
    ecl_lon, ecl_lat = radec_to_ecliptic(ra, dec)
    info.radiant_ra, info.radiant_dec = np.degrees(float(ra)), np.degrees(float(dec))
    info.radiant_ecllong, info.radiant_ecllat = np.degrees(float(ecl_lon)), np.degrees(float(ecl_lat))
    
    airspeeds = [np.linalg.norm(cross_pos[i + len(raw_data['durations'])] - cross_pos[i]) / d for i, d in enumerate(raw_data['durations']) if d > 0]
    if airspeeds: info.speed = np.mean(airspeeds)
    
    if 'showerassoc' in AVAILABLE_LIBS:
        showerassoc = AVAILABLE_LIBS['showerassoc']
        info.shower, _ = showerassoc.showerassoc(info.radiant_ra, info.radiant_dec, info.speed,
                                                 time.strftime("%Y-%m-%d", time.localtime(info.timestamp)))
    print_results(info)
    if options.writestat:
        write_stat_file(info, options.input_file)
    
    write_res_file(track_start, track_end, cross_pos, obs_data, options.input_file)

    if options.output:
        plot_height(track_start, track_end, cross_pos, obs_data, options)
        plot_map(track_start, track_end, cross_pos, obs_data, options)

    return info

def main():
    """Parses command-line arguments and runs the analysis."""
    parser = argparse.ArgumentParser(
        description='Atmospheric meteor trajectory fitting.',
        epilog='Example: ./metrack.py obs_20120403.dat -o both -m i --borders-auto'
    )
    parser.add_argument('input_file', nargs='?', help='Name of input file with observation data.')
    parser.add_argument('-o', '--output', choices=['show', 'save', 'both'], default='', help='Graphics output mode.')
    parser.add_argument('-m', '--map', dest='mapres', default='i', choices=['c','l','i','h','f'], help='Map detail level.')
    parser.add_argument('-e', '--err', dest='errlim', type=float, default=1.0, help='Deprecated option, kept for compatibility.')
    parser.add_argument('-a', '--azonly', action='store_true', help='Plot azimuths only; no fitting.')
    parser.add_argument('-b', '--borders-auto', dest='autoborders', action='store_true', help='Automatically determine map boundaries.')
    parser.add_argument('-t', '--timestamp', type=float, help="Force a specific Unix timestamp.")
    parser.add_argument('--no-opt', action='store_false', dest='optimize', default=True, help="Turn off optimization.")
    parser.add_argument('--no-write-stat', action='store_false', dest='writestat', default=True, help="Do not write a .stat file.")
    
    options = parser.parse_args()

    if not options.input_file:
        parser.error("No input file specified.")
    if not Path(options.input_file).is_file():
        parser.error(f"Input file not found: {options.input_file}")
        
    if options.azonly and not options.output:
        options.output = 'show'

    metrack(
        inname=options.input_file,
        doplot=options.output,
        accept_err=options.errlim,
        mapres=options.mapres,
        azonly=options.azonly,
        autoborders=options.autoborders,
        timestamp=options.timestamp,
        optimize=options.optimize,
        writestat=options.writestat
    )

if __name__ == "__main__":
    main()
