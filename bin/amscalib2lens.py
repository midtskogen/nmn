#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Creates a Hugin .pto file from an AMS (AllSky Meteor Software)
JSON calibration file. This script generates control points by correlating
pixel coordinates of stars from the input file with their calculated
celestial positions (Azimuth/Altitude) for a given time and location.

Usage: amscalib2lens.py <AMS calibration json file> <pto file>
       amscalib2lens.py /mnt/ams2/cal/freecal/2022_05_06_00_32_20_000_011193/2022_05_06_00_32_20_000_011193-stacked-calparams.json lens.pto
"""

import argparse
import configparser
import copy
import glob
import json
import math
import os
import subprocess
from datetime import datetime, UTC
import io

import ephem  # For calculating celestial object positions
from stars import cat  # Star catalog for position cross-referencing


import sys
from pathlib import Path

# Ensure local project modules are importable even when this script is executed via symlink
_SCRIPT_PATH = Path(__file__).resolve()
_PROJECT_DIR = None
for _cand in (_SCRIPT_PATH.parent, *_SCRIPT_PATH.parents):
    if (_cand / 'bin').is_dir() and (_cand / 'server').is_dir():
        _PROJECT_DIR = _cand
        break
if _PROJECT_DIR is not None:
    _BIN_DIR = _PROJECT_DIR / 'bin'
    _SRC_DIR = _PROJECT_DIR / 'src'
    for _p in (_BIN_DIR, _SRC_DIR, _PROJECT_DIR):
        if _p.exists():
            _ps = str(_p)
            if _ps not in sys.path:
                sys.path.insert(0, _ps)

def _find_config_path(args):
    """Select configuration file: -c if given, otherwise /etc/meteor.cfg,
    otherwise the amscams as6.json if it exists."""
    if args.config:
        return args.config
    default_cfg = '/etc/meteor.cfg'
    if os.path.isfile(default_cfg) and os.access(default_cfg, os.R_OK):
        return default_cfg
    fallback_json = '/home/ams/amscams/conf/as6.json'
    if os.path.isfile(fallback_json) and os.access(fallback_json, os.R_OK):
        return fallback_json
    return None


def _read_config(config_path):
    """Read the selected config file.

    INI files are parsed with configparser. JSON files (e.g. as6.json)
    are translated into a ConfigParser with the relevant site fields
    placed in an [astronomy] section so the rest of the code can use
    them uniformly.
    """
    config = configparser.ConfigParser()
    if not config_path or not os.path.exists(config_path):
        return config
    if config_path.endswith('.json'):
        try:
            with open(config_path) as f:
                data = json.load(f)
            if 'site' in data:
                site = data['site']
                if not config.has_section('astronomy'):
                    config.add_section('astronomy')
                if 'device_lat' in site:
                    config.set('astronomy', 'latitude', str(site['device_lat']))
                if 'device_lng' in site:
                    config.set('astronomy', 'longitude', str(site['device_lng']))
                if 'device_alt' in site:
                    config.set('astronomy', 'elevation', str(site['device_alt']))
                # sensible defaults for amscams installs without these fields
                if not config.has_option('astronomy', 'temperature'):
                    config.set('astronomy', 'temperature', '10')
                if not config.has_option('astronomy', 'pressure'):
                    config.set('astronomy', 'pressure', '1010')
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not parse JSON config '{config_path}': {e}")
    else:
        config.read(config_path)
    return config


def setup_observer(args, config, calib_data, verbose=False):
    """
    Initializes and configures an ephem.Observer object with location,
    elevation, and atmospheric data, following a specific priority order.
    """
    obs = ephem.Observer()

    def warn_if_overriding(old_val, new_val, name, new_source, old_source):
        if old_val is not None and str(old_val) != str(new_val):
            print(f"Warning: {name} from {new_source} ('{new_val}') is overriding value from {old_source} ('{old_val}').")

    lat_val, lon_val, ele_val = None, None, None
    lat_source, lon_source, ele_source = "Nothing", "Nothing", "Nothing"

    # 1. Baseline from JSON data (Lowest Priority)
    if 'device_lat' in calib_data:
        lat_val, lat_source = str(calib_data['device_lat']), "JSON"
    elif 'site_lat' in calib_data:
        lat_val, lat_source = str(calib_data['site_lat']), "JSON"
    if 'device_lon' in calib_data:
        lon_val, lon_source = str(calib_data['device_lon']), "JSON"
    elif 'site_lng' in calib_data:
        lon_val, lon_source = str(calib_data['site_lng']), "JSON"
    if 'device_alt' in calib_data:
        ele_val, ele_source = float(calib_data['device_alt']), "JSON"
    elif 'site_alt' in calib_data:
        ele_val, ele_source = float(calib_data['site_alt']), "JSON"

    # 2. Override with config file data (Medium Priority)
    if config.has_section('astronomy'):
        conf_lat = config.get('astronomy', 'latitude', fallback=None)
        conf_lon = config.get('astronomy', 'longitude', fallback=None)
        conf_ele = config.getfloat('astronomy', 'elevation', fallback=None)

        if conf_lat is not None:
            warn_if_overriding(lat_val, conf_lat, "Latitude", "Config File", lat_source)
            lat_val, lat_source = conf_lat, "Config File"
        if conf_lon is not None:
            warn_if_overriding(lon_val, conf_lon, "Longitude", "Config File", lon_source)
            lon_val, lon_source = conf_lon, "Config File"
        if conf_ele is not None:
            warn_if_overriding(ele_val, conf_ele, "Elevation", "Config File", ele_source)
            ele_val, ele_source = conf_ele, "Config File"

    # 3. Override with command-line arguments (Highest Priority)
    if args.latitude is not None:
        warn_if_overriding(lat_val, args.latitude, "Latitude", "Command-line", lat_source)
        lat_val = str(args.latitude)
    if args.longitude is not None:
        warn_if_overriding(lon_val, args.longitude, "Longitude", "Command-line", lon_source)
        lon_val = str(args.longitude)
    if args.elevation is not None:
        warn_if_overriding(ele_val, args.elevation, "Elevation", "Command-line", ele_source)
        ele_val = args.elevation

    # Final validation: ensure we have a complete location
    if lat_val is None or lon_val is None or ele_val is None:
        missing = []
        if lat_val is None: missing.append("Latitude")
        if lon_val is None: missing.append("Longitude")
        if ele_val is None: missing.append("Elevation")
        raise ValueError(f"Observer location is incomplete. Missing: {', '.join(missing)}. Provide the location via the JSON file, a config file (-c), or command-line arguments.")

    obs.lat = lat_val
    obs.lon = lon_val
    obs.elevation = ele_val

    # Handle temperature and pressure
    obs.temp = float(config.get('astronomy', 'temperature', fallback=10))
    obs.pressure = float(config.get('astronomy', 'pressure', fallback=1010))
    if args.temperature is not None: obs.temp = args.temperature
    if args.pressure is not None: obs.pressure = args.pressure

    # Handle timestamp
    timestamp = args.timestamp if hasattr(args, 'timestamp') and args.timestamp else None
    if not timestamp:
        try:
            fname = getattr(args, 'amscalib', '')
            fname_parts = os.path.basename(fname).split('_')
            dt_str = f"{fname_parts[0]}-{fname_parts[1]}-{fname_parts[2]} {fname_parts[3]}:{fname_parts[4]}:{fname_parts[5]}"
            
            naive_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            aware_dt = naive_dt.replace(tzinfo=UTC)
            timestamp = aware_dt.timestamp()

        except (IndexError, ValueError, AttributeError):
            timestamp = datetime.now(UTC).timestamp()
            print("Warning: Could not parse timestamp from filename. Using current time.")

    # Create a datetime object from the final timestamp
    final_dt_utc = datetime.fromtimestamp(float(timestamp), UTC)
    
    # Set the observer date for ephem calculations
    obs.date = final_dt_utc.strftime('%Y-%m-%d %H:%M:%S')

    # If verbose, print the observer details using the new formats
    if verbose:
        print("\n--- Observer Details ---")
        print(f"Timestamp for calculation: {final_dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Latitude: {math.degrees(obs.lat):.5f}")
        print(f"Longitude: {math.degrees(obs.lon):.5f}")
        print(f"Elevation: {int(obs.elevation)} m")
        
    return obs


def _get_pto_scaffold(width, height, calib_data, date_str):
    """
    Generates the main panorama and image lines (p-line and i-lines) for the
    .pto file as a string.
    """
    pixel_scale = float(calib_data.get('pixel_scale', calib_data.get('pixscale', 0)))
    if pixel_scale == 0:
        raise ValueError("Could not find 'pixel_scale' or 'pixscale' in JSON file.")
        
    fov_orig_calc = width * pixel_scale / 3600
    center_az = calib_data.get('center_az', 180)
    center_el = calib_data.get('center_el', 0)
    position_angle = calib_data.get('position_angle', 0)

    with io.StringIO() as header_stream:
        header_stream.write("# hugin project file\n")
        header_stream.write("#hugin_ptoversion 2\n")
        header_stream.write('p f2 w36000 h18000 v360 E0 R0 n"TIFF_m c:LZW"\n')
        header_stream.write("m g1 i0 m2 p0.00784314\n\n")
        header_stream.write("# image lines\n")
        header_stream.write("#-hugin cropFactor=1\n")
        header_stream.write(f"i w{width} h{height} f3 v{fov_orig_calc} Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r{position_angle} p{center_el} y{center_az - 180} TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 Vm5\n")
        header_stream.write('i w36000 h18000 f4 v360 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p0 y0 TrX0 TrY0 TrZ0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0 Vm5 n"dummy.jpg"\n\n')
        header_stream.write("# specify variables that should be optimized\n")
        header_stream.write("v v0\nv r0\nv p0\nv y0\nv a0\nv b0\nv c0\nv d0\nv e0\nv\n\n")
        header_stream.write(f"# {date_str}\n")
        header_stream.write("# control points\n")
        return header_stream.getvalue()


def _get_control_points(calib_data, observer, verbose=False):
    """
    Generates control point lines from the provided list of stars.
    """
    control_points = []
    # This function now expects a pre-filtered list of stars
    for star_data in calib_data.get('cat_image_stars', []):
        dcname, mag, ra, dec, _, _, match_dist, _, _, _, _, _, _, six, siy, _, _ = star_data
        
        if verbose:
            # The initial check is now done elsewhere, but we can still announce the re-verification
            print(f"\n- Verifying star from JSON: '{dcname if dcname else 'Unnamed'}' at pixel (X:{six}, Y:{siy})")

        ra_hours = ra * 24 / 360
        best_match_body = None
        best_match_name = "Unknown"
        min_separation = 99999

        for (cat_ra, pmra, cat_dec, pmdec, cat_mag, name) in cat:
            if abs(mag - cat_mag) > 0.2:
                continue

            json_star = ephem.FixedBody()
            json_star._ra, json_star._dec, json_star._epoch = str(ra_hours), str(dec), ephem.J2000
            json_star.compute(observer)

            catalog_star = ephem.FixedBody()
            catalog_star._ra, catalog_star._pmra, catalog_star._dec, catalog_star._pmdec, catalog_star._epoch = str(cat_ra), pmra, str(cat_dec), pmdec, ephem.J2000
            catalog_star.compute(observer)
            
            separation = float(repr(ephem.separation(json_star, catalog_star)))
            if separation < min_separation:
                min_separation, best_match_body, best_match_name = separation, catalog_star, name

        if min_separation < 0.0001:
            best_match_body.compute(observer)
            az = math.degrees(float(repr(best_match_body.az)))
            alt = math.degrees(float(repr(best_match_body.alt)))

            json_star.compute(observer)
            json_az = math.degrees(float(repr(json_star.az)))
            json_alt = math.degrees(float(repr(json_star.alt)))

            if verbose:
                print(f"  - Re-verification match found: '{best_match_name}' (Separation: {min_separation:.6f} rad)")
                print(f"  - JSON Pos (X:{six}, Y:{siy} -> Az/Alt): {json_az:.4f}° / {json_alt:.4f}°")
                print(f"  - Ephem Pos (Catalog -> Az/Alt):  {az:.4f}° / {alt:.4f}°")

            if alt > 1:
                pano_x = az * 100
                pano_y = (90 - alt) * 100
                comment = f"Star: {best_match_name}, JSON_px:({six},{siy}), JSON_pos:({json_az:.2f},{json_alt:.2f})"
                control_points.append(f'c n0 N1 x{six} y{siy} X{pano_x:.4f} Y{pano_y:.4f} t0 # {comment}\n')
        
        elif verbose:
            print(f"  - No close re-verification match found in script's catalog (min separation: {min_separation:.6f} rad).")
            print("  -> REJECTED.")
    
    return control_points


def generate_pto_from_json(calib_data, observer, width, height, match_dist_limit, verbose=False):
    """
    Selects the best stars and generates the full content of a Hugin .pto file.
    """
    # --- New Star Selection Logic ---
    raw_star_list = calib_data.get('cat_image_stars', [])
    final_star_list = []

    if raw_star_list:
        # 1. Filter stars by the user-defined quality limit
        good_stars = [s for s in raw_star_list if s[6] <= match_dist_limit]

        # 2. If enough good stars are found, use them.
        if len(good_stars) >= 3:
            final_star_list = good_stars
            if verbose:
                print(f"Info: Found {len(final_star_list)} stars within the match distance limit of {match_dist_limit}.")
        # 3. Otherwise, fall back to using the absolute 3 best stars.
        else:
            raw_star_list.sort(key=lambda s: s[6])
            final_star_list = raw_star_list[:3]
            print(f"Warning: Only {len(good_stars)} stars met the quality limit of {match_dist_limit}.")
            print("Falling back to using the 3 best-matched stars to attempt a solution.")
    
    # Create a new dictionary with the final curated list of stars
    curated_calib_data = calib_data.copy()
    curated_calib_data['cat_image_stars'] = final_star_list

    # --- Generation ---
    scaffold = _get_pto_scaffold(width, height, curated_calib_data, observer.date)
    # The 'match_dist_limit' is no longer passed as filtering is complete
    control_points = _get_control_points(curated_calib_data, observer, verbose=verbose)

    # Final validation remains a good safeguard
    if len(control_points) < 3:
        raise ValueError(f"Only {len(control_points)} valid control points could be generated. At least 3 are required.")

    return scaffold + "".join(control_points)


def _load_as6_json(args):
    """Load the amscams as6.json for camera/cams_id mapping."""
    path = None
    if args.config and args.config.endswith('.json') and os.path.exists(args.config):
        path = args.config
    if path is None:
        default = '/home/ams/amscams/conf/as6.json'
        if os.path.exists(default):
            path = default
    if path is None:
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not load as6.json from '{path}': {e}", file=sys.stderr)
    return None


def _find_latest_json_for_camera(cams_id):
    """Find the most recent calibration JSON for a given cams_id in freecal."""
    base = '/mnt/ams2/cal/freecal'
    if not os.path.isdir(base):
        return None, None, None
    candidates = []
    for root in os.listdir(base):
        parts = root.rsplit('_', 1)
        if len(parts) != 2 or parts[1] != cams_id:
            continue
        if len(root) < 19 or root[4] != '_' or root[7] != '_' or root[10] != '_' or root[13] != '_' or root[16] != '_':
            continue
        dir_path = os.path.join(base, root)
        if not os.path.isdir(dir_path):
            continue
        json_file = os.path.join(dir_path, f'{root}-stacked-calparams.json')
        if not os.path.exists(json_file):
            json_file = os.path.join(dir_path, f'{root}-calparams.json')
        if not os.path.exists(json_file):
            continue
        try:
            dt = datetime.strptime(root[:19], '%Y_%m_%d_%H_%M_%S')
        except ValueError:
            continue
        candidates.append((dt, json_file, root))
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda x: x[0], reverse=True)
    dt, json_file, root = candidates[0]
    return json_file, root, dt.timestamp()


def _lens_pto_name_from_root(root):
    """Return lens-YYYYMMDD.pto from a freecal root timestamp."""
    dt = datetime.strptime(root[:19], '%Y_%m_%d_%H_%M_%S')
    return f'lens-{dt.year:04d}{dt.month:02d}{dt.day:02d}.pto'


def convert_json_to_pto(args):
    """Convert a single AMS JSON calibration file into a Hugin .pto file.

    Uses args.amscalib for input, args.ptofile for output, and the location
    and timestamp settings from args.
    """
    try:
        with open(args.amscalib) as f:
            calib_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not read or decode JSON from '{args.amscalib}': {e}")
        return False

    config_path = _find_config_path(args)
    if args.config and config_path != args.config:
        print(f"Error: Config file not found: {args.config}")
        return False
    config = _read_config(config_path)

    try:
        observer = setup_observer(args, config, calib_data, verbose=args.verbose)
    except ValueError as e:
        print(f"Error: Could not determine observer location. {e}")
        return False

    if 'cal_params' in calib_data:
        cal_params_data = calib_data['cal_params']
    else:
        cal_params_data = calib_data

    width = int(cal_params_data.get('imagew', args.width))
    height = int(cal_params_data.get('imageh', args.height))

    try:
        if args.verbose:
            print("\n--- Verifying Stars for Control Points ---")

        pto_content = generate_pto_from_json(cal_params_data, observer, width, height,
                                             match_dist_limit=args.match_dist, verbose=args.verbose)

        with open(args.ptofile, 'w') as ptofile_handle:
            ptofile_handle.write(pto_content)
        print(f"Successfully generated initial .pto file: {args.ptofile}")

        if args.verbose:
            print("\n--- Generated .pto File Content ---")
            print(pto_content.strip())
            print("-----------------------------------")

    except (IOError, ValueError) as e:
        print(f"Error generating or writing PTO file: {e}")
        return False

    cp_count = 0
    try:
        with open(args.ptofile) as f:
            for line in f:
                if line.startswith('c '):
                    cp_count += 1
    except Exception as e:
        print(f"\nWarning: could not count control points: {e}")

    if cp_count < args.min_cp:
        print(f"Skipping autooptimiser: only {cp_count} control point(s), below limit {args.min_cp}.")
        return True

    try:
        print("Running Hugin's autooptimiser...")
        proc = subprocess.run(['autooptimiser', '-n', args.ptofile, '-o', args.ptofile],
                              capture_output=True, text=True)

        if proc.returncode == 0:
            print("Optimization complete.")
            if args.verbose:
                print("\n--- Autooptimiser Output ---")
                print(proc.stdout if proc.stdout.strip() else "(No standard output)")
                if proc.stderr.strip():
                    print("\n--- Autooptimiser Errors/Warnings ---")
                    print(proc.stderr)
        else:
            print(f"\nError during optimization (exit code {proc.returncode}).")
            print(proc.stderr)

    except FileNotFoundError:
        print("\nError: 'autooptimiser' command not found.")
        print("Please ensure Hugin command-line tools are installed and in your system's PATH.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during optimization: {e}")

    return True


def fix_missing_lens_files(args):
    """Find missing /meteor/camX/lens.pto files and create them from the latest JSON."""
    as6 = _load_as6_json(args)
    if as6 is None or 'cameras' not in as6:
        print("Error: --fix-missing requires as6.json to map camera numbers to cams_id.", file=sys.stderr)
        return

    existing = 0
    skipped = 0
    missing = 0
    created = 0

    for cam_key in sorted(as6['cameras'].keys()):
        cam_info = as6['cameras'][cam_key]
        cams_id = cam_info.get('cams_id')
        if not cams_id:
            continue
        digits = ''.join(ch for ch in cam_key if ch.isdigit())
        if not digits:
            continue
        cam_num = int(digits)

        lens_dir = f'/meteor/cam{cam_num}'
        lens_link = f'{lens_dir}/lens.pto'

        # If the link already points to an existing target, nothing to do.
        if os.path.exists(lens_link):
            existing += 1
            if args.verbose:
                print(f"  {cam_key}: {lens_link} already exists")
            continue

        json_file, root, timestamp = _find_latest_json_for_camera(cams_id)
        if json_file is None:
            if args.verbose:
                print(f"  {cam_key}: no calibration JSON found for cams_id {cams_id}")
            skipped += 1
            continue

        target_name = _lens_pto_name_from_root(root)
        target_path = f'{lens_dir}/{target_name}'

        missing += 1
        if args.dryrun:
            print(f"Would create {lens_link} -> {target_name} from {json_file}")
            continue

        # Generate the dated .pto file if it does not already exist.
        if not os.path.exists(target_path):
            camera_args = copy.copy(args)
            camera_args.amscalib = json_file
            camera_args.ptofile = target_path
            camera_args.timestamp = timestamp
            print(f"Generating {target_path} from {json_file}")
            if not convert_json_to_pto(camera_args):
                continue

        # Ensure the lens.pto symlink points to the dated file.
        if os.path.islink(lens_link) or os.path.exists(lens_link):
            try:
                os.unlink(lens_link)
            except OSError as e:
                print(f"Error removing old {lens_link}: {e}", file=sys.stderr)
                continue
        try:
            os.symlink(target_name, lens_link)
            created += 1
            print(f"Created {lens_link} -> {target_name}")
        except OSError as e:
            print(f"Error creating symlink {lens_link}: {e}", file=sys.stderr)

    action = "would create" if args.dryrun else "created"
    print(f"\nSummary: {missing} missing, {existing} already exist, {skipped} no JSON, {created if not args.dryrun else missing} {action}.")


def main():
    """
    Main execution function for standalone script usage.
    """
    parser = argparse.ArgumentParser(
        description='Convert AMS calibration into a Hugin/panotools pto file.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('amscalib', nargs='?', help='AMS calibration json file')
    parser.add_argument('ptofile', nargs='?', help='Output Hugin .pto file')
    parser.add_argument('-W', '--width', type=int, default=1920, help='Image width (default: 1920)')
    parser.add_argument('-H', '--height', type=int, default=1080, help='Image height (default: 1080)')
    parser.add_argument('-d', '--match_dist', type=float, default=0.2, help='Maximum allowed match distance (default: 0.2)')
    parser.add_argument('-c', '--config', help='Meteor config file (default: /etc/meteor.cfg, or /home/ams/amscams/conf/as6.json as fallback)')
    parser.add_argument('-T', '--timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00 UTC)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed script output.')
    parser.add_argument('-x', '--longitude', type=float, help='Observer longitude')
    parser.add_argument('-y', '--latitude', type=float, help='Observer latitude')
    parser.add_argument('-e', '--elevation', type=float, help='Observer elevation (m)')
    parser.add_argument('-t', '--temperature', type=float, help='Observer temperature (C, for refraction)')
    parser.add_argument('-p', '--pressure', type=float, help='Observer air pressure (hPa, for refraction)')
    parser.add_argument('--min-cp', type=int, default=15,
                        help='Minimum control points that must be present in the generated initial PTO '
                             'before autooptimiser is run (default: 15). If the count is lower, the '
                             'initial PTO is kept unchanged and autooptimiser is skipped.')
    parser.add_argument('--fix-missing', action='store_true', dest='fix_missing',
                        help='Detect missing /meteor/camX/lens.pto files and create them from the most recent JSON for each camera.')
    parser.add_argument('--dryrun', action='store_true',
                        help='With --fix-missing, only print which symlinks/files would be created.')
    args = parser.parse_args()

    if args.fix_missing:
        if args.amscalib or args.ptofile:
            print("Error: --fix-missing does not take amscalib or ptofile arguments.", file=sys.stderr)
            sys.exit(1)
        fix_missing_lens_files(args)
        return

    if not args.amscalib or not args.ptofile:
        parser.print_usage(sys.stderr)
        sys.exit(1)

    if not convert_json_to_pto(args):
        sys.exit(1)


if __name__ == '__main__':
    if 'cat' not in globals():
         print("Error: Could not find the 'cat' variable from 'stars.py'.")
         print("Please ensure 'stars.py' is in the same directory.")
    else:
         main()
