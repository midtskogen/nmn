#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetches and processes meteor observation data for the Norsk Meteornettverk. (Multilingual Version)

This script performs the following actions for each supported language:
1.  Fetches event data from a remote station using rsync.
2.  Processes the raw data to calculate trajectory, orbit, and other metrics.
3.  Generates scientific plots and summary images (SVG to JPG) with translated text.
4.  Produces language-specific HTML and KML reports for web presentation.
5.  Optionally posts a notification to social media.
Usage:
    python3 fetch.py <station_name> <ssh_port> <remote_dir>
    python3 fetch.py <local_event_directory> [--fast] [--all] [--origcen]
"""

import argparse
import atexit
import configparser
import datetime
import glob
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Third-party library imports
import numpy as np
import pytz
from dateutil.tz import tzlocal

# Use a non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import cairosvg
    from PIL import Image, ImageOps
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False
    logging.warning("Libs cairosvg or Pillow not found. SVG conversion will be skipped.")

# Local script imports from the same project
from fb2kml import fb2kml
from fbspd_merge import readres, calculate_speed_profile, generate_speed_plots
from metrack import calculate_trajectory, generate_plots as generate_metrack_plots, write_res_file
from orbit import calc_azalt, orbit
import reverse_geocode

try:
    import windprofile
    WINDPROFILE_AVAILABLE = True
except ImportError:
    logging.warning("Module 'windprofile.py' not found. Wind profile generation will be skipped.")
    WINDPROFILE_AVAILABLE = False


def _collect_infra_sites(event_dir: Path):
    sites = []
    try:
        for p in sorted(event_dir.glob('*/cam*/infra.txt')):
            try:
                cfg = configparser.ConfigParser()
                cfg.read(p)
                if not cfg.has_section('arrival') or not cfg.has_option('arrival', 'time'):
                    continue
                station = cfg.get('station', 'name', fallback=p.parent.parent.name)
                code = cfg.get('station', 'code', fallback=str(station)[:3].upper()).strip()
                lat = cfg.getfloat('station', 'latitude', fallback=None)
                lon = cfg.getfloat('station', 'longitude', fallback=None)
                elev_m = cfg.getfloat('station', 'elevation', fallback=0.0)
                if lat is None or lon is None:
                    continue
                sites.append({'code': code, 'station': station, 'lat': float(lat), 'lon': float(lon), 'elev_m': float(elev_m)})
            except Exception:
                continue
    except Exception:
        return []

    # Deduplicate by code
    uniq = {}
    for s in sites:
        c = str(s.get('code') or '').strip()
        if c and c not in uniq:
            uniq[c] = s
    return list(uniq.values())


def _fetch_infra_wind_profiles(event_dir: Path, event_timestamp: float):
    if not WINDPROFILE_AVAILABLE:
        return
    try:
        sites = _collect_infra_sites(event_dir)
        if not sites:
            return
        for s in sites:
            try:
                code = str(s.get('code') or '').strip() or 'XXX'
                out_csv = event_dir / f"wind_profile_{code}.csv"
                if out_csv.exists() and out_csv.stat().st_size > 50:
                    continue
                ok, matched_time = windprofile.get_wind_profile(
                    float(s['lat']),
                    float(s['lon']),
                    float(event_timestamp),
                    str(out_csv),
                )
                if ok:
                    logging.info(f"Fetched wind profile for infrasound site {code} ({matched_time}) -> {out_csv}")
                else:
                    logging.info(f"No wind profile available for infrasound site {code}")
            except Exception as e:
                logging.warning(f"Failed to fetch wind profile for infrasound site {s.get('code')}: {e}")
    except Exception:
        return


def _pick_infra_fit_for_mapping(infra_fit: dict, min_elev_km: float = 5.0, max_elev_km: float = 80.0):
    try:
        if not infra_fit or 'lat' not in infra_fit or 'lon' not in infra_fit:
            return None, None

        def _in_range(d: dict) -> bool:
            try:
                elev_km = float(d.get('elev_m', 0.0)) / 1000.0
                return bool(min_elev_km <= elev_km <= max_elev_km)
            except Exception:
                return False

        if _in_range(infra_fit):
            return infra_fit, 'primary'

        base = infra_fit.get('base_solution') if isinstance(infra_fit.get('base_solution'), dict) else None
        refined = infra_fit.get('refined_solution') if isinstance(infra_fit.get('refined_solution'), dict) else None

        fallback = None
        reason = None
        if base and _in_range(base):
            fallback = base
            reason = 'base_solution'
        elif refined and _in_range(refined):
            fallback = refined
            reason = 'refined_solution'

        if fallback is None:
            return None, None

        out = dict(infra_fit)
        for k in ('lat', 'lon', 'elev_m', 't0_unix'):
            if k in fallback:
                out[k] = fallback.get(k)
        return out, reason
    except Exception:
        return None, None


class Config:
    """Configuration constants for the script."""
    # Base directory for all meteor data on the local server
    BASE_HTTP_DIR = Path(__file__).resolve().parents[1]
    METEOR_DATA_DIR = BASE_HTTP_DIR / 'meteor'
    BIN_DIR = BASE_HTTP_DIR / 'bin'

    # Bandwidth limit for rsync in KB/s
    BW_LIMIT_DEFAULT = '1000'
    BW_LIMIT_SLOW = '150'

    # Number of times to retry rsync on failure
    RSYNC_RETRIES = 5
    RETRY_DELAY_SECONDS = 60

    # Directory search window for merging events (in seconds)
    EVENT_MERGE_WINDOW = 8

    # Image processing settings
    SVG_DEFAULT_DPI = 300
    SVG_MAP_DPI = 80
    SVG_ORBIT_DPI = 100
    
    # Tweet settings
    MAX_EVENT_AGE_FOR_TWEET_HOURS = 4.0


# --- Internationalization (i18n) Setup ---
SUPPORTED_LANGS = ['nb', 'en', 'de', 'cs', 'fi']
DEFAULT_LANG = 'nb'
LOC_DIR = Config.BIN_DIR / 'loc'

def load_translations(lang_code: str) -> dict:
    """Loads the translation dictionary for a given language, with fallback to default."""
    default_path = LOC_DIR / f"{DEFAULT_LANG}.json"
    lang_path = LOC_DIR / f"{lang_code}.json"

    translations = {}
    if default_path.exists():
        with default_path.open('r', encoding='utf-8') as f:
            try:
                translations = json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing default translations {default_path}: {e}")
                
    if lang_code != DEFAULT_LANG and lang_path.exists():
        with lang_path.open('r', encoding='utf-8') as f:
            try:
                translations.update(json.load(f))
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing language file {lang_path}: {e}")
            
    return translations


def setup_logging(log_file_path: Path):
    """Configures logging to file and console."""
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
        filename=log_file_path,
        filemode='a'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(console_handler)


def restore_terminal_state():
    """
    Restores the terminal settings to a sane state on exit.
    This fixes issues where subprocesses (like oysttyer or ssh) disable
    terminal echo and fail to restore it.
    """
    try:
        # 'stty sane' restores terminal settings (including echo)
        subprocess.run(['stty', 'sane'], check=False)
    except Exception:
        # If stty isn't available or fails (e.g. cron job), ignore it.
        pass


def run_command(command, cwd=None, check=False, shell=False, stream_output=False):
    """Runs an external command using subprocess.run."""
    logging.info(f"Running command: {' '.join(command) if isinstance(command, list) else command}")
    
    capture_props = {} if stream_output else {'capture_output': True, 'text': True}

    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            check=check,
            shell=shell,
            **capture_props
        )
        if not stream_output:
            if process.stdout:
                logging.info(process.stdout)
            if process.stderr:
                logging.warning(process.stderr)
        return process
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}: {e.cmd}")
        if not stream_output:
            logging.error(f"STDOUT: {e.stdout}")
            logging.error(f"STDERR: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred while running command: {command}. Error: {e}")
        raise


def fetch_data(station: str, port: str, remote_dir: str, local_dir: Path) -> bool:
    """Fetches data from a remote station using rsync with retries."""
    local_dir.mkdir(parents=True, exist_ok=True)
    speed = Config.BW_LIMIT_SLOW if station == 'kristiansand' else Config.BW_LIMIT_DEFAULT
    host = station if port == '0' else '192.168.1.10'
    ssh_port = '22' if port == '0' else port

    rsync_cmd = [
        'rsync', '-av', '--exclude', 'frame-*', f'--bwlimit={speed}',
        '--progress', '-e', f'ssh -o StrictHostKeyChecking=no -p {ssh_port}',
        f'meteor@{host}:{remote_dir}/', str(local_dir)
    ]

    for attempt in range(Config.RSYNC_RETRIES):
        logging.info(f"Rsync attempt {attempt + 1}/{Config.RSYNC_RETRIES}...")
        result = run_command(rsync_cmd, stream_output=True)
        if result.returncode == 0:
            logging.info("Rsync completed successfully.")
            ssh_cmd = ['ssh', '-o', 'StrictHostKeyChecking=no', '-p', ssh_port,
                       f'meteor@{host}', 'touch', f'{remote_dir}/uploaded']
            run_command(ssh_cmd)
            return True
        logging.warning(f"Rsync failed. Retrying in {Config.RETRY_DELAY_SECONDS} seconds...")
        time.sleep(Config.RETRY_DELAY_SECONDS)

    logging.error("Rsync failed after multiple retries. Network error.")
    return False


def set_permissions(directory: Path):
    """Recursively sets directory and file permissions."""
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            try:
                Path(root, d).chmod(0o775)
            except OSError as e:
                logging.warning(f"Could not set permissions on dir {root}/{d}: {e}")
        for f in files:
            try:
                Path(root, f).chmod(0o664)
            except OSError as e:
                logging.warning(f"Could not set permissions on file {root}/{f}: {e}")


def to_cartesian(az: float, alt: float) -> list:
    """Converts Azimuth/Altitude to Cartesian coordinates."""
    az_rad = np.radians(az)
    alt_rad = np.radians(alt)
    return [
        np.cos(alt_rad) * np.cos(az_rad),
        np.cos(alt_rad) * np.sin(az_rad),
        np.sin(alt_rad)
    ]


def from_cartesian(p: list) -> tuple:
    """Converts Cartesian coordinates to Azimuth/Altitude."""
    az = np.degrees(np.arctan2(p[1], p[0]))
    alt = np.degrees(np.arcsin(p[2]))
    return (np.fmod(az + 360, 360), alt)


def create_centroid_file(local_dir: Path):
    """Reads event data, recalculates coordinates, and writes centroid2.txt."""
    event_file = local_dir / 'event.txt'
    centroid_file = local_dir / 'centroid2.txt'
    
    if not event_file.is_file():
        # This is expected for some directories, silent return or debug log only
        return

    obs = configparser.ConfigParser()
    try:
        obs.read(event_file)

        coordinates = obs.get('trail', 'coordinates').split()
        timestamps = obs.get('trail', 'timestamps').split()
        az1_str, alt1_str = obs.get('summary', 'startpos').split()
        az2_str, alt2_str = obs.get('summary', 'endpos').split()
    except Exception as e:
        logging.error(f"Failed to parse event.txt file {event_file}: {e}")
        return

    p1 = to_cartesian(float(az1_str), float(alt1_str))
    p2 = to_cartesian(float(az2_str), float(alt2_str))

    coordinates2 = []
    for c in coordinates:
        try:
            if ',' not in c:
                raise ValueError("missing comma")
            az3_str, alt3_str = c.split(',', 1)
            p3 = to_cartesian(float(az3_str), float(alt3_str))
            t = np.cross(np.cross(p1, p2), np.cross(p3, np.cross(p1, p2)))
            norm_t = np.linalg.norm(t)
            if norm_t > 0:
                coordinates2.append(from_cartesian(t / norm_t))
            else:
                coordinates2.append((0, 0))
        except Exception as e:
            logging.warning(f"Skipping malformed coordinate token '{c}' in {event_file}: {e}")
            coordinates2.append((0, 0))

    try:
        # Attempt to find the observation file to get the station code
        code = obs.get('station', 'code', fallback='').strip()
        if not code:
            obs_file = next(local_dir.glob('*[0-9][0-9].txt'))
            code = obs_file.read_text().split()[12]
    except (StopIteration, IndexError, configparser.NoOptionError, configparser.NoSectionError):
        # Fallback: try to guess code from directory name if file parse fails
        code = local_dir.parent.name.upper()
        if len(code) > 3: code = code[:3]

    try:
        with centroid_file.open('w', encoding='utf-8') as f:
            for i, (t_str, (az, alt)) in enumerate(zip(timestamps, coordinates2)):
                t_float = float(t_str)
                dt = t_float - float(timestamps[0])
                gm_time = time.gmtime(t_float)
                
                main_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', gm_time)
                fractional_second = f"{t_float % 1:.2f}".split('.')[1]
                
                output_line = (
                    f"{i} {dt:.2f} {alt:.2f} {az:.2f} 1.0 {code} "
                    f"{main_timestamp}.{fractional_second} UTC"
                )
                f.write(output_line + '\n')
        logging.info(f"Generated centroid2.txt for {local_dir}")
    except Exception as e:
        logging.error(f"Failed to write centroid2.txt in {local_dir}: {e}")


def find_and_merge_event_directory(base_date: datetime.datetime, station: str, current_local_dir: Path):
    """Searches for an existing event directory and merges the current data into it."""
    date_str = base_date.strftime('%Y%m%d')
    time_str = base_date.strftime('%H%M%S')
    
    for i in range(-Config.EVENT_MERGE_WINDOW, Config.EVENT_MERGE_WINDOW + 1):
        if i == 0:
            continue
            
        target_date = base_date + datetime.timedelta(seconds=i)
        target_dir = Config.METEOR_DATA_DIR / target_date.strftime('%Y%m%d/%H%M%S')

        if target_dir.is_dir():
            logging.info(f"Merging event into existing directory: {target_dir}")
            
            station_dest_dir = target_dir / station / current_local_dir.name
            station_dest_dir.mkdir(parents=True, exist_ok=True)
            
            for f in current_local_dir.glob('*'):
                destination_file = station_dest_dir / f.name
                try:
                    os.replace(f, destination_file)
                except OSError as e:
                    logging.error(f"Could not move {f} to {destination_file}: {e}")
            
            try:
                shutil.rmtree(current_local_dir.parent.parent)
            except OSError as e:
                logging.error(f"Could not remove original event dir {current_local_dir.parent.parent}: {e}")

            return target_dir

    return Config.METEOR_DATA_DIR / date_str / time_str


def svg_to_jpg(svg_path: Path, jpg_path: Path, dpi: int):
    """Converts an SVG file to JPG using CairoSVG and Pillow."""
    if not LIBS_AVAILABLE:
        logging.warning("CairoSVG or Pillow not installed. Skipping SVG to JPG conversion.")
        return
    if not svg_path.exists():
        logging.warning(f"SVG file not found, cannot convert: {svg_path}")
        return

    try:
        png_data = cairosvg.svg2png(url=str(svg_path), dpi=dpi)
        png_image = Image.open(io.BytesIO(png_data))

        bbox = png_image.getbbox()
        trimmed_image = png_image.crop(bbox) if bbox else png_image
        bordered_image = ImageOps.expand(trimmed_image, border=10, fill='white')

        final_image = Image.new("RGB", bordered_image.size, "white")
        if bordered_image.mode == 'RGBA':
            final_image.paste(bordered_image, (0, 0), bordered_image.split()[3])
        else:
            final_image.paste(bordered_image, (0, 0))
        
        final_image.save(jpg_path, 'JPEG', quality=95)
        logging.info(f"Successfully converted {svg_path.name} to {jpg_path.name}")

    except Exception as e:
        logging.error(f"Error during SVG to JPG conversion for {svg_path.name}: {e}")


def get_location_from_coords(lat, lon) -> str:
    """Reverse geocodes coordinates to a municipality name using the reverse-geocode module."""
    municipality = ''
    try:
        coordinates = [(lat, lon)]
        results = reverse_geocode.search(coordinates)

        if results:
            municipality = results[0].get('city', '')
        else:
            logging.warning(f"Reverse geocode returned no results for {coordinates[0]}.")

    except Exception as e:
        logging.error(f"An error occurred while using the reverse_geocode module: {e}")

    return municipality.strip()


def generate_triangulation_html_report(output_path: Path, resdat, orbit_data, placename, translations: dict, lang: str):
    """Generates the language-specific HTML tables that depend on triangulation results."""
    
    with output_path.open('w', encoding='utf-8') as f:
        table1 = f"""
<b>{translations.get("atmospheric_trajectory", "Meteor's Atmospheric Trajectory")}</b>:<br>
<table border=1>
    <tr><td>{translations.get("start_height", "Start height")}:</td><td> {resdat.height[0]:.1f} km</td></tr>
    <tr><td>{translations.get("end_height", "End height")}:</td><td> {resdat.height[1]:.1f} km</td></tr>
    <tr><td>{translations.get("start_position", "Start position")}:</td><td> {resdat.lat1[0]:.3f}N {resdat.long1[0]:.3f}E</td></tr>
    <tr><td>{translations.get("end_position", "End position")}:</td><td> {resdat.lat1[1]:.3f}N {resdat.long1[1]:.3f}E</td></tr>
    <tr><td>{translations.get("direction", "Direction")}:</td><td> {np.fmod(orbit_data['az'] + 360, 360):.1f}°</td></tr>
    <tr><td>{translations.get("inclination_angle", "Inclination angle")}:</td><td> {orbit_data['alt']:.1f}°</td></tr>
"""
        if orbit_data['entry_speed'] > 0:
            table1 += f"    <tr><td>{translations.get('entry_speed', 'Entry speed')}:</td><td> {orbit_data['entry_speed']:.1f} km/s</td></tr>\n"

        if orbit_data['valid']:
            ramin = int(orbit_data['ra'] * 24 * 60 / 360)
            # The showername passed in orbit_data should now be correctly translated
            shower_name = translations.get("sporadic", "sporadic") if not orbit_data.get('showername') else orbit_data['showername']
            table1 += f"""
    <tr><td>{translations.get("radiant_ra", "Radiant R.A.")}:</td><td> {ramin // 60:02d}:{ramin % 60:02d} ({orbit_data['ra']:.1f}°)</td></tr>
    <tr><td>{translations.get("radiant_dec", "Radiant Dec.")}:</td><td> {orbit_data['dec']:.1f}°</td></tr>
    <tr><td>{translations.get("meteor_shower", "Meteor Shower")}:</td><td align=center> {shower_name}</td></tr>
"""
        table1 += "</table>"
        
        # Use comma as decimal separator for languages other than English
        if lang != 'en':
            table1 = table1.replace('.', ',')
        
        f.write('<table><tr><td valign=top>\n')
        f.write(table1)
        f.write('\n</td><td valign=top>\n')

        if orbit_data['valid']:
            table2 = f"""
<b>{translations.get("orbital_elements", "Meteoroid's Orbital Elements")}</b>:<br>
<table border=1>
    <tr><td>{translations.get("perihelion_dist", "Perihelion distance")}:</td><td> {orbit_data['rp']:.3f} AU</td></tr>
    <tr><td>{translations.get("eccentricity", "Eccentricity")}:</td><td> {orbit_data['ecc']:.3f}</td></tr>
    <tr><td>{translations.get("inclination", "Inclination")}:</td><td> {orbit_data['inc']:.1f}°</td></tr>
    <tr><td>{translations.get("longitude_node", "Long. of Asc. Node")}:</td><td> {orbit_data['lnode']:.1f}°</td></tr>
    <tr><td>{translations.get("arg_periapsis", "Argument of Perihelion")}:</td><td> {orbit_data['argp']:.1f}°</td></tr>
    <tr><td>{translations.get("mean_anomaly", "Mean Anomaly")}:</td><td> {orbit_data['m0']:.1f}°</td></tr>
    <tr><td>{translations.get("epoch", "Epoch")}:</td><td> {orbit_data['t0']}</td></tr>
</table>
"""
            if lang != 'en':
                table2 = table2.replace('.', ',')
            f.write(table2)
    
        f.write('</td></tr></table>')


def generate_station_html_report(output_path: Path, event_dir: Path, translations: dict):
    """Generates the language-specific station HTML file with full details."""
    with output_path.open('w', encoding='utf-8') as f:
        station_files = sorted(event_dir.glob('*/*/event.txt'))
        for event_file in station_files:
            station = event_file.parent.parent.name
            cam = event_file.parent.name
            location = station.title()

            cfg = configparser.ConfigParser()
            try:
                cfg.read(event_file)
                ts_float = float(cfg.get('trail', 'timestamps').split()[0])
            except Exception as e:
                logging.warning(f"Could not parse event file {event_file}: {e}")
                continue
                
            ts_utc = datetime.datetime.fromtimestamp(ts_float, tz=pytz.utc)
            ts_str = ts_utc.strftime('%Y%m%d%H%M%S')
            
            code = ''
            try:
                code = cfg.get('station', 'code', fallback='').strip()
            except Exception:
                code = ''

            if not code:
                obs_txt_file = event_file.parent / f"{station}-{ts_str}.txt"
                if obs_txt_file.exists():
                    try:
                        code = obs_txt_file.read_text().split()[12]
                    except IndexError:
                        pass

            html_template = """
<div class="container">
  <div class="column">
<h1>{location} ({code}) {cam}</h1>
<?php
$webm_path = "{station}/{cam}/fireball_neg.webm";
$jpg_path = "{station}/{cam}/fireball.jpg";
$webm_url = "{url_base}/{cam}/fireball_neg.webm";
$webm_url2 = "{url_base}/{cam}/fireball_orig.webm";
$jpg_url = "{url_base}/{cam}/fireball.jpg";

// Logic to display language-specific brightness plot with a fallback to default
$b_prefix = ($lang === '{default_lang_code}') ? '' : substr($lang, 0, 2) . '_';
$specific_brightness_path = "{station}/{cam}/" . $b_prefix . "brightness.jpg";
$specific_brightness_url = "{url_base}/{cam}/" . $b_prefix . "brightness.jpg";
$default_brightness_path = "{station}/{cam}/brightness.jpg";
$default_brightness_url = "{url_base}/{cam}/brightness.jpg";

$display_brightness_path = null;
$display_brightness_url = null;

if (file_exists($specific_brightness_path)) {{
    $display_brightness_path = $specific_brightness_path;
    $display_brightness_url = $specific_brightness_url;
}} elseif (file_exists($default_brightness_path)) {{
    $display_brightness_path = $default_brightness_path;
    $display_brightness_url = $default_brightness_url;
}}

$preview_img_path = null;
$preview_img_url = null;
$preview_href_url = null;

if (file_exists("{station}/{cam}/{station_ts}-gnomonic-grid.jpg")) {{
    $preview_img_path = "{station}/{cam}/{station_ts}-gnomonic-grid.jpg";
    $preview_img_url = "{url_base}/{cam}/{station_ts}-gnomonic-grid.jpg";
    if (file_exists("{station}/{cam}/{station_ts}-gnomonic.mp4")) {{
        $preview_href_url = "{url_base}/{cam}/{station_ts}-gnomonic.mp4";
    }} else {{
        $preview_href_url = $preview_img_url;
    }}
}} elseif (file_exists("{station}/{cam}/{station_ts}-grid.jpg")) {{
    $preview_img_path = "{station}/{cam}/{station_ts}-grid.jpg";
    $preview_img_url = "{url_base}/{cam}/{station_ts}-grid.jpg";
    if (file_exists("{station}/{cam}/{station_ts}-grid.mp4")) {{
        $preview_href_url = "{url_base}/{cam}/{station_ts}-grid.mp4";
    }} else {{
        $preview_href_url = $preview_img_url;
    }}
}} elseif (file_exists("{station}/{cam}/{station_ts}.jpg")) {{
    $preview_img_path = "{station}/{cam}/{station_ts}.jpg";
    $preview_img_url = "{url_base}/{cam}/{station_ts}.jpg";
    if (file_exists("{station}/{cam}/{station_ts}-orig.mp4")) {{
        $preview_href_url = "{url_base}/{cam}/{station_ts}-orig.mp4";
    }} elseif (file_exists("{station}/{cam}/{station_ts}.mp4")) {{
        $preview_href_url = "{url_base}/{cam}/{station_ts}.mp4";
    }} else {{
        $preview_href_url = $preview_img_url;
    }}
}}
?>
    <div style="text-align: center;">
        <?php if (file_exists($webm_path)) {{ ?>
        <a href="<?php echo $webm_url2; ?>"><video autoplay loop muted playsinline style="max-width: 800px; width: 100%; height: auto; border: 1px solid black;"><source src="<?php echo $webm_url; ?>" type="video/webm"></video></a><br>
        <?php }} elseif (file_exists($jpg_path)) {{ ?>
        <a href="<?php echo $jpg_url; ?>"><img src="<?php echo $jpg_url; ?>" style="max-width: 800px; width: 100%; height: auto;" alt="fireball"><br></a>
        <?php }} ?>
    </div>
<table><tr><td valign=top>
<?php if ($preview_img_path !== null) {{ ?><a href="<?php echo $preview_href_url; ?>"><img src="<?php echo $preview_img_url; ?>" width=768 alt="preview"></a><?php }} ?>
</td>
<td valign=top>
<table border=1>
<tr><td><b>{videos_header}</b><br>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic.mp4">{gnomonic_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic-grid.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic-grid.mp4">{gnomonic_with_coords_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-orig.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-orig.mp4">{original_label}</a><br> <?php }} elseif (file_exists("{station}/{cam}/{station_ts}.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}.mp4">{original_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-grid.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-grid.mp4">{original_with_coords_label}</a><br> <?php }} ?>
</td></tr>
<tr><td><b>{images_header}</b><br>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic.jpg">{gnomonic_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic-grid.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic-grid.jpg">{gnomonic_with_coords_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic-grid-uncorr.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic-grid-uncorr.jpg">{gnomonic_uncorrected_with_coords_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic-labels.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic-labels.jpg">{gnomonic_with_labels_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic-labels-uncorr.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic-labels-uncorr.jpg">{gnomonic_uncorrected_with_labels_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}.jpg">{original_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-grid.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-grid.jpg">{original_with_coords_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-mask.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-mask.jpg">{original_with_mask_label}</a><br> <?php }} ?>
</td></tr>
<tr><td><b>{text_files_header}</b><br>
<?php if (file_exists("{station}/{cam}/event.txt")) {{ ?>• <a href="{url_base}/{cam}/event.txt">{detection_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}.txt")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}.txt">{observation_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/centroid2.txt")) {{ ?>• <a href="{url_base}/{cam}/centroid2.txt">{coordinates_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/stderr.txt")) {{ ?>• <a href="{url_base}/{cam}/stderr.txt">{error_messages_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/report.log")) {{ ?>• <a href="{url_base}/{cam}/report.log">{log_file_label}</a><br> <?php }} ?>
</td></tr></table>
<?php if ($display_brightness_path !== null) {{ ?><a href="<?php echo $display_brightness_url; ?>"><img src="<?php echo $display_brightness_url; ?>" width=400 alt="{brightness_label}"><br></a> <?php }} ?>
</td></tr></table>
</p>
</div></div>
            """
            
            url_base_path = f'/meteor/{event_dir.parent.name}/{event_dir.name}/{station}'
            station_timestamp_str = f"{station}-{ts_str}"

            f.write(html_template.format(
                url_base=url_base_path, station=station, cam=cam, station_ts=station_timestamp_str,
                code=code, location=location, default_lang_code=DEFAULT_LANG,
                videos_header=translations.get("videos", "Videos:"),
                images_header=translations.get("images", "Images:"),
                text_files_header=translations.get("text_files", "Text Files:"),
                gnomonic_label=translations.get("gnomonic", "Gnomonic"),
                gnomonic_with_coords_label=translations.get("gnomonic_with_coords", "Gnomonic with coordinates"),
                original_label=translations.get("original", "Original"),
                original_with_coords_label=translations.get("original_with_coords", "Original with coordinates"),
                gnomonic_uncorrected_with_coords_label=translations.get("gnomonic_uncorrected_with_coords", "Uncorrected gnomonic with coordinates"),
                gnomonic_with_labels_label=translations.get("gnomonic_with_labels", "Gnomonic with labels"),
                gnomonic_uncorrected_with_labels_label=translations.get("gnomonic_uncorrected_with_labels", "Uncorrected gnomonic with labels"),
                original_with_mask_label=translations.get("original_with_mask", "Original with mask"),
                detection_label=translations.get("detection", "Detection"),
                observation_label=translations.get("observation", "Observation"),
                coordinates_label=translations.get("coordinates", "Coordinates"),
                error_messages_label=translations.get("error_messages", "Error Messages"),
                log_file_label=translations.get("log_file", "Log"),
                brightness_label=translations.get("brightness", "Brightness")
            ))


def generate_brightness_plots(event_dir: Path, timestamps: list, brightness: list):
    """Generates translated brightness plots for all supported languages."""
    logging.info(f"Generating multilingual brightness plots for {event_dir}...")
    if not timestamps or not brightness:
        logging.warning("Warning: No data available to generate brightness plot.")
        return

    time_axis = [t - timestamps[0] for t in timestamps]

    for lang in SUPPORTED_LANGS:
        translations = load_translations(lang)
        file_prefix = '' if lang == DEFAULT_LANG else f'{lang}_'
        
        svg_path = event_dir / f'{file_prefix}brightness.svg'
        jpg_path = event_dir / f'{file_prefix}brightness.jpg'

        try:
            plt.figure()
            plt.plot(time_axis, brightness)
            plt.xlabel(translations.get("plot_time_x_label", "Time [s]"))
            plt.ylabel(translations.get("brightness", "Brightness"))
            plt.title(translations.get("brightness_plot_title", "Brightness over Time"))
            
            plt.savefig(svg_path)
            plt.savefig(jpg_path)
            plt.close()
            logging.info(f"  - Saved {svg_path.name} and {jpg_path.name}")
        except Exception as e:
            logging.error(f"Error generating plot for language '{lang}': {e}")


def send_tweet(event_dir: Path, date: datetime.datetime, placename: str, showername_sg: str, count: int, first: bool, height_valid: bool, translations: dict):
    """Constructs and sends a tweet if conditions are met."""
    if not (count >= 2 and first and height_valid and len(sys.argv) == 4):
        logging.info("Conditions for tweeting not met (count>=2, first_run, height_valid, is_fetch_run). Skipping.")
        return
    
    meteor_registered = translations.get('tweet_meteor_registered', 'Meteor registrert')
    shower_registered_template = translations.get('tweet_shower_registered', '{shower} registrert')
    over_str = translations.get('tweet_over', 'over')
    
    tweet_base = shower_registered_template.format(shower=showername_sg.capitalize()) if showername_sg else meteor_registered
    tweet_str = f"{tweet_base} {over_str} {placename} " if placename else f"{tweet_base} "

    utc_time = date.replace(tzinfo=pytz.utc)
    local_time = utc_time.astimezone(tzlocal())
    zone_name = local_time.strftime('%Z')
    
    if zone_name == 'CET':
        zone = translations.get('timezone_cet', 'norsk normaltid')
    elif zone_name == 'CEST':
        zone = translations.get('timezone_cest', 'norsk sommertid')
    else:
        zone = zone_name
    
    tweet_str += local_time.strftime(f'%Y-%m-%d %H:%M:%S {zone}')
    link = f"https://norskmeteornettverk.no/meteor/{utc_time.strftime('%Y%m%d/%H%M%S')}/"
    
    full_tweet = f'-status="{tweet_str} {link}"'
    logging.info(f"Prepared tweet: {full_tweet}")
    
    key_solobs = '/var/www/.oysttyerkey-solobs'
    key_main = '/var/www/.oysttyerkey'
    
    for key in [key_solobs, key_main]:
        cmd = f'ulimit -t 100; /usr/bin/oysttyer -ssl -keyf={key} -silent {full_tweet}'
        try:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logging.info(f"Successfully sent tweet using key {key}.")
        except Exception as e:
            logging.error(f"Failed to send tweet using key {key}: {e}")


def process_event(event_dir: Path, date: datetime.datetime, fast: bool = False, all_stations: bool = False, use_orig_cen: bool = False, infrasound_only: bool = False, verbose: bool = False):
    """Main processing logic for a meteor event."""
    logging.info(f"Processing event in directory: {event_dir}")
    obs_filename = f"obs_{date.strftime('%Y-%m-%d_%H:%M:%S')}.txt"
    obs_filepath = event_dir / obs_filename
    res_filename = obs_filepath.with_suffix('.res')

    if infrasound_only:
        logging.info("--- INFRASOUND MODE: Skipping pre-processing and running only infrasound fit. ---")
        original_cwd = Path.cwd()
        infra_fit = None
        try:
            os.chdir(event_dir)
        except Exception as e:
            logging.error(f"Failed to change directory to {event_dir}: {e}")
            return

        try:
            infra_files = list(event_dir.glob('*/cam*/infra.txt'))
            if len(infra_files) >= 3:
                _fetch_infra_wind_profiles(event_dir, date.timestamp())
                infra_out = event_dir / 'infra_fit.json'
                infra_report = event_dir / 'infra_fit_report.txt'
                cmd = [
                    sys.executable,
                    str(Config.BIN_DIR / 'infra_fit.py'),
                    str(event_dir),
                    '--output', str(infra_out),
                    '--report', str(infra_report),
                ]
                if verbose:
                    cmd.append('--verbose')
                if verbose:
                    env = os.environ.copy()
                    env.setdefault('OMP_NUM_THREADS', '1')
                    env.setdefault('OPENBLAS_NUM_THREADS', '1')
                    env.setdefault('MKL_NUM_THREADS', '1')
                    env.setdefault('NUMEXPR_NUM_THREADS', '1')
                    p = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
                    try:
                        if p.stdout is not None:
                            for line in p.stdout:
                                try:
                                    print(line, end='')
                                except Exception:
                                    pass
                    finally:
                        proc = p.wait()
                    class _R:
                        pass
                    rr = _R()
                    rr.returncode = int(proc)
                    proc = rr
                else:
                    env = os.environ.copy()
                    env.setdefault('OMP_NUM_THREADS', '1')
                    env.setdefault('OPENBLAS_NUM_THREADS', '1')
                    env.setdefault('MKL_NUM_THREADS', '1')
                    env.setdefault('NUMEXPR_NUM_THREADS', '1')
                    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

                if proc.returncode != 0:
                    if verbose:
                        logging.warning(
                            "Infrasound fit failed. "
                            f"returncode={proc.returncode}"
                        )
                    else:
                        stderr_snip = (proc.stderr or '').strip()[:400]
                        stdout_snip = (proc.stdout or '').strip()[:400]
                        logging.warning(
                            "Infrasound fit failed. "
                            f"returncode={proc.returncode} stdout={stdout_snip} stderr={stderr_snip}"
                        )
                    if infra_report.exists():
                        logging.warning(f"Infrasound fit report: {infra_report}")
                else:
                    logging.info(f"Infrasound fit JSON: {infra_out}")
                    if infra_report.exists():
                        logging.info(f"Infrasound fit report: {infra_report}")
                    try:
                        infra_fit = json.loads(infra_out.read_text(encoding='utf-8'))
                    except Exception:
                        infra_fit = None
            else:
                logging.info("No/insufficient infra.txt files found (need >=3). Skipping infrasound fit.")
        except Exception as e:
            logging.warning(f"Infrasound fit step failed: {e}", exc_info=True)

        # Regenerate maps (best-effort) using existing obs_*.txt and metrack trajectory solve.
        try:
            if not obs_filepath.exists():
                logging.warning(f"Observation file not found, cannot regenerate maps: {obs_filepath}")
                return

            metrack_opts = {
                'timestamp': date.timestamp(),
                'optimize': True,
                'use_ransac': True,
                'seed': 0,
                'ransac_threshold': 1.0,
                'ransac_iterations': 10,
                'ransac_runs': 100,
                'debug_ransac': False,
                'all_in_tolerance': 1.0
            }

            metrack_info, metrack_plot_data = calculate_trajectory(str(obs_filepath), **metrack_opts)
            if not metrack_plot_data:
                raise ValueError('Metrack calculation returned no plot data')

            for lang in SUPPORTED_LANGS:
                translations = load_translations(lang)
                file_prefix = '' if lang == DEFAULT_LANG else f'{lang}_'

                plot_opts = {
                    'doplot': 'save',
                    'interactive': True,
                    'autoborders': True,
                    'azonly': False,
                    'mapres': 'i'
                }

                infra_fit_for_maps, infra_pick_reason = _pick_infra_fit_for_mapping(infra_fit)
                if infra_fit_for_maps and 'lat' in infra_fit_for_maps and 'lon' in infra_fit_for_maps:
                    try:
                        elev_km = float(infra_fit_for_maps.get('elev_m', 0.0)) / 1000.0
                        if 5.0 <= elev_km <= 80.0:
                            if infra_pick_reason and infra_pick_reason != 'primary':
                                logging.info(
                                    f"Infrasound fit elevation out of range; using {infra_pick_reason} for map marker. "
                                    f"elev_km={elev_km:.3f}"
                                )
                            extra_markers = []
                            extra_markers.append({
                                'kind': 'infra_source',
                                'lat': float(infra_fit_for_maps['lat']),
                                'lon': float(infra_fit_for_maps['lon']),
                                'elev_m': float(infra_fit_for_maps.get('elev_m', 0.0)),
                                'label': (lambda _t0: (translations.get('infrasound_source', 'Infrasound source') + (f" {_t0}" if _t0 else "")))(
                                    (datetime.datetime.fromtimestamp(float(infra_fit_for_maps.get('t0_unix')), tz=datetime.timezone.utc).strftime('%H:%M:%S')
                                     if infra_fit_for_maps.get('t0_unix') is not None else "")
                                ),
                                'color': '#006400',
                                'label_color': 'black',
                                'symbol': 'o',
                                'plotly_symbol': 'circle',
                                'size': 7,
                                'drop_to_ground': True,
                            })

                            infra_sites = infra_fit.get('stations_all') or infra_fit.get('stations') or []
                            for st in infra_sites:
                                try:
                                    st_code = str(st.get('code', '')).strip()
                                    st_name = str(st.get('station', '')).strip()
                                    st_label = st_code or st_name
                                    is_inlier = bool(st.get('is_inlier', True))
                                    extra_markers.append({
                                        'kind': 'infra_detection',
                                        'lat': float(st.get('lat')),
                                        'lon': float(st.get('lon')),
                                        'elev_m': float(st.get('elev_m', 0.0)),
                                        'label': st_label,
                                        'color': '#006400' if is_inlier else '#777777',
                                        'label_color': '#006400' if is_inlier else '#777777',
                                        'symbol': '^' if is_inlier else 'o',
                                        'plotly_symbol': 'triangle-up' if is_inlier else 'circle',
                                        'size': 5 if is_inlier else 4,
                                        'drop_to_ground': False,
                                        'opacity': 0.9 if is_inlier else 0.55,
                                    })
                                except Exception:
                                    continue

                            plot_opts['extra_markers'] = extra_markers

                            paths = []
                            for ring in ((infra_fit.get('isochrones') if isinstance(infra_fit, dict) else None) or (infra_fit_for_maps.get('isochrones') or [])):
                                try:
                                    paths.append({
                                        'label': f"Isochrone {ring.get('code','')}",
                                        'color': '#006400',
                                        'lats': ring.get('lats'),
                                        'lons': ring.get('lons'),
                                        'elev_km': float(ring.get('alt_km', elev_km)),
                                        'station_lat': ring.get('station_lat'),
                                        'station_lon': ring.get('station_lon'),
                                        'station_elev_m': ring.get('station_elev_m'),
                                    })
                                except Exception:
                                    continue
                            if paths:
                                plot_opts['extra_paths'] = paths
                    except Exception:
                        pass

                generate_metrack_plots(metrack_info, metrack_plot_data, plot_opts, translations=translations, output_prefix=file_prefix)

                for svg_name in ["height.svg", "map.svg"]:
                    svg_path = event_dir / f"{file_prefix}{svg_name}"
                    if svg_path.exists():
                        svg_to_jpg(svg_path, svg_path.with_suffix('.jpg'), Config.SVG_MAP_DPI)
        except Exception as e:
            logging.warning(f"Failed to regenerate maps in --infrasound mode: {e}", exc_info=True)
        finally:
            try:
                os.chdir(original_cwd)
            except Exception:
                pass
        return
    
    if not fast:
        # Run process.py for ALL event.txt files in PARALLEL
        report_script = Config.BIN_DIR / 'process.py'
        event_files = sorted(event_dir.glob('*/*/event.txt'))
        
        if not event_files:
            logging.warning("No station event.txt files found. Skipping process.py.")
        
        processes = [] # List to hold Popen objects
        for event_file in event_files:
            report_log = event_file.parent / "report.log" # Generic log name
            report_cmd = [sys.executable, str(report_script), str(event_file)]
            logging.info(f"--- Spawning {report_script.name} for {event_file} ---")
            try:
                with report_log.open('w') as log_file:
                   proc = subprocess.Popen(report_cmd, stdout=log_file, stderr=log_file)
                   processes.append(proc)
                   # Wait 3 seconds before spawning the next process to stagger load
                   time.sleep(3)
            except Exception as e:
                logging.error(f"Failed to spawn process.py for {event_file}: {e}")
                
        if processes:
            logging.info(f"Waiting for {len(processes)} station processing jobs to complete...")
            for proc in processes:
                proc.wait()
                
                if proc.returncode == 0:
                    logging.info(f"Job for {proc.args[2]} finished successfully (exit code 0).")
                elif proc.returncode == 1:
                    logging.info(f"Job for {proc.args[2]} was discarded (exit code 1).")
                else:
                    logging.warning(f"Job for {proc.args[2]} failed with exit code {proc.returncode}.")
            
            logging.info("All station processing jobs finished.")
    else:
        # --fast mode: Skip process.py, but regenerate brightness plots manually
        logging.info("--- FAST MODE: Skipping process.py (station video/classification). ---")
        logging.info("--- FAST MODE: Regenerating station brightness plots... ---")
        station_event_files = sorted(event_dir.glob('*/*/event.txt'))
        
        if not station_event_files:
            logging.warning("--- FAST MODE: No station event.txt files found to regenerate brightness plots. ---")

        for event_file in station_event_files:
            try:
                cfg = configparser.ConfigParser()
                cfg.read(event_file)
                timestamps = [float(t) for t in cfg.get('trail', 'timestamps').split()]
                brightness = [float(b) for b in cfg.get('trail', 'brightness').split()]
                # The plot function saves plots in the event_file's parent directory
                generate_brightness_plots(event_file.parent, timestamps, brightness) 
            except Exception as e:
                logging.warning(f"Failed to regenerate brightness plot for {event_file}: {e}")

    # --- CRITICAL FIX: Ensure centroid2.txt exists for ALL stations ---
    # In reprocessing mode (or if files were deleted), these might be missing.
    # We iterate all subdirectories that look like station folders and regenerate them.
    # This also ensures we have the option to use centroid2.txt even if requested to use orig.
    logging.info("Ensuring centroid2.txt exists for all stations...")
    for station_dir in event_dir.glob('*/*/'):
        if (station_dir / 'event.txt').exists():
             try:
                 create_centroid_file(station_dir)
             except Exception as e:
                 logging.error(f"Failed to generate centroid2.txt in {station_dir}: {e}", exc_info=True)
    # -----------------------------------------------------------------

    station_codes, station_name_to_code = set(), {}
    try:
        with obs_filepath.open('w', encoding='utf-8') as outfile:
            for obs_file in sorted(event_dir.glob('*/*/*[0-9][0-9].txt')):
                content = obs_file.read_text().strip()
                if not content: continue
                
                outfile.write(content + '\n')
                station_dir_name = obs_file.relative_to(event_dir).parts[0]
                try:
                    code = content.split()[12]
                    station_codes.add(code)
                    if station_dir_name not in station_name_to_code:
                        station_name_to_code[station_dir_name] = code
                except IndexError:
                    logging.warning(f"Could not parse station code from {obs_file}")
                    continue
    except Exception as e:
        logging.error(f"Failed to create observation file {obs_filepath}: {e}")
        return

    original_cwd = Path.cwd()
    try:
        os.chdir(event_dir)
    except Exception as e:
        logging.error(f"Failed to change directory to {event_dir}: {e}")
        return
        
    logging.info(f"Changed working directory to: {event_dir}")
    if not Path('index.php').exists():
        try: Path('index.php').symlink_to('../../report.php')
        except OSError as e:
             logging.warning(f"Could not create symlink index.php: {e}")

    is_multistation = len(station_codes) > 1
    analysis_results = {}

    if is_multistation:
        try:
            logging.info(f"Multi-station event ({len(station_codes)} stations). Performing core analysis...")
            
            metrack_opts = {
                'timestamp': date.timestamp(), 
                'optimize': True, 
                'use_ransac': not all_stations, # Disable RANSAC if all_stations is True
                'seed': 0,
                'ransac_threshold': 1.0,
                'ransac_iterations': 10,
                'ransac_runs': 100,
                'debug_ransac': False,
                'all_in_tolerance': 1.0
            }
            fbspd_opts = {'debug': True, 'seed': 0}
            
            metrack_info, metrack_plot_data = calculate_trajectory(str(obs_filepath), **metrack_opts)
            if not metrack_plot_data: raise ValueError("Metrack calculation returned no plot data.")
            
            write_res_file(metrack_plot_data['track_start'], metrack_plot_data['track_end'],
                           metrack_plot_data['cross_pos_inliers'], metrack_plot_data['inlier_obs_data'], str(obs_filepath))
            
            resdat = readres(str(res_filename))
            if not resdat: raise ValueError("readres() failed or returned no data.")
                
            fb2kml(str(res_filename))
            
            inlier_codes = set(metrack_info.inlier_stations)
            
            # Use original centroid.txt if flag is set, otherwise default to centroid2.txt
            target_centroid_file = 'centroid.txt' if use_orig_cen else 'centroid2.txt'
            if use_orig_cen:
                logging.info("Option --origcen selected: Using original 'centroid.txt' for speed profile analysis.")
            
            # Note: station_name_to_code maps 'dir_name' -> 'STATION_CODE'.
            # metrack returns 'STATION_CODE'.
            # p.parts[-3] gives us the directory name of the station (e.g. 'osl').
            # We must map that dir name to a code to check against inlier_codes.
            centroid_files = [str(p) for p in event_dir.glob(f'*/*/{target_centroid_file}') 
                              if station_name_to_code.get(p.parts[-3]) in inlier_codes]

            entry_speed = None
            fbspd_plot_data = None
            try:
                fbspd_results, fbspd_plot_data = calculate_speed_profile(
                    str(res_filename), centroid_files, str(obs_filepath), **fbspd_opts
                )
                if fbspd_results and 'initial_speed' in fbspd_results:
                    entry_speed = fbspd_results['initial_speed']
                else:
                    logging.warning("FBSPD calculation returned no results. Continuing without speed profile.")
            except Exception as e:
                logging.warning(f"FBSPD calculation failed. Continuing without speed profile: {e}", exc_info=True)

            az, alt = calc_azalt(resdat.lat1[0], resdat.long1[0], resdat.height[0], resdat.lat1[1], resdat.long1[1], resdat.height[1])
            placename = get_location_from_coords(resdat.lat1[1], resdat.long1[1])
            if placename: 
                try: Path('location.txt').write_text(placename, encoding='utf-8')
                except OSError as e: logging.warning(f"Could not write location.txt: {e}")
            
            # --- Core Analysis Data Storage ---
            orbit_data_base = {'valid': False, 'entry_speed': entry_speed, 'az': az, 'alt': alt}
            if entry_speed is not None and entry_speed > 9.8:
                # Capture the raw shower name (likely in default language/Norwegian)
                ra, dec, (rp, ecc, inc, lnode, argp, m0, t0), raw_shower_name, _, valid = orbit(
                    True, entry_speed, 0, str(res_filename), date.strftime('%Y-%m-%d'), date.strftime('%H:%M:%S'), doplot=''
                )
                orbit_data_base.update({'ra': ra, 'dec': dec, 'rp': rp, 'ecc': ecc, 'inc': inc, 
                                   'lnode': lnode, 'argp': argp, 'm0': m0, 't0': t0,
                                   'showername': raw_shower_name, # Store the raw name
                                   'valid': valid})

            analysis_results = {
                'metrack_info': metrack_info, 'metrack_plot_data': metrack_plot_data, 'fbspd_plot_data': fbspd_plot_data,
                'orbit_data': orbit_data_base,
                'resdat': resdat, 
                'placename': placename,
                'first_run': not any(p.name.endswith('map.jpg') for p in event_dir.glob('*.jpg'))
            }

            logging.info("Core analysis successful.")

        except Exception as e:
            logging.error(f"Core analysis failed: {e}", exc_info=True)
            is_multistation = False
    else:
        logging.info("Single-station event. Skipping core analysis.")


    # --- Language-specific output generation loop ---
    
    # 1. Build a Reverse Lookup Map (Name -> Code) using the default language (nb)
    #    This handles cases where orbit() returns a localized name (e.g. "Leonidene") instead of a code.
    nb_translations = load_translations('nb')
    name_to_code = {}
    if 'showers' in nb_translations:
        for code, name in nb_translations['showers'].items():
            name_to_code[name.lower()] = code

    for lang in SUPPORTED_LANGS:
        try:
            logging.info(f"--- Generating outputs for language: [{lang}] ---")
            translations = load_translations(lang)
            file_prefix = '' if lang == DEFAULT_LANG else f'{lang}_'

            generate_station_html_report(event_dir / f"{file_prefix}stations.html", event_dir, translations)

            current_orbit_data = {'valid': False}

            if is_multistation and analysis_results:
                logging.info(f"Generating translated plots and reports for [{lang}]")
                
                # --- Resolve Shower Name Logic ---
                current_orbit_data = analysis_results['orbit_data'].copy() 
                
                raw_shower_name = current_orbit_data.get('showername', '')
                shower_code = None
                
                # 1. Check if the raw name is already a code
                if raw_shower_name in translations.get('showers', {}):
                    shower_code = raw_shower_name
                # 2. Use the reverse map to find the code from the Norwegian/default name
                elif raw_shower_name and raw_shower_name.lower() in name_to_code:
                    shower_code = name_to_code[raw_shower_name.lower()]
                
                # Resolve the Final Name for the current language
                final_shower_name = ""
                if shower_code:
                    final_shower_name = translations.get('showers', {}).get(shower_code, raw_shower_name)
                else:
                    # Fallback logic
                    if not raw_shower_name or raw_shower_name.lower() in ['sporadic', 'spo', '']:
                         final_shower_name = translations.get('sporadic', 'sporadic')
                    else:
                         final_shower_name = raw_shower_name

                current_orbit_data['showername'] = final_shower_name
                current_orbit_data['showername_sg'] = final_shower_name 
                
                plot_opts = {
                    'doplot': 'save', 
                    'interactive': True, 
                    'autoborders': True,
                    'azonly': False,
                    'mapres': 'i'
                }

                try:
                    infra_fit = analysis_results.get('infra_fit')
                    infra_fit_for_maps, infra_pick_reason = _pick_infra_fit_for_mapping(infra_fit)
                    if infra_fit_for_maps and 'lat' in infra_fit_for_maps and 'lon' in infra_fit_for_maps:
                        elev_km = float(infra_fit_for_maps.get('elev_m', 0.0)) / 1000.0
                        if 5.0 <= elev_km <= 80.0:
                            if infra_pick_reason and infra_pick_reason != 'primary':
                                logging.info(
                                    f"Infrasound fit elevation out of range; using {infra_pick_reason} for map marker. "
                                    f"elev_km={elev_km:.3f}"
                                )
                            extra_markers = []

                            extra_markers.append({
                                'kind': 'infra_source',
                                'lat': float(infra_fit_for_maps['lat']),
                                'lon': float(infra_fit_for_maps['lon']),
                                'elev_m': float(infra_fit_for_maps.get('elev_m', 0.0)),
                                'label': (lambda _t0: (translations.get('infrasound_source', 'Infrasound source') + (f" {_t0}" if _t0 else "")))(
                                    (datetime.datetime.fromtimestamp(float(infra_fit_for_maps.get('t0_unix')), tz=datetime.timezone.utc).strftime('%H:%M:%S')
                                     if infra_fit_for_maps.get('t0_unix') is not None else "")
                                ),
                                'color': '#006400',
                                'label_color': 'black',
                                'symbol': 'o',
                                'plotly_symbol': 'circle',
                                'size': 7,
                                'drop_to_ground': True,
                            })

                            # Detection locations (include outliers too)
                            infra_sites = infra_fit.get('stations_all') or infra_fit.get('stations') or []
                            for st in infra_sites:
                                try:
                                    st_code = str(st.get('code', '')).strip()
                                    st_name = str(st.get('station', '')).strip()
                                    st_label = st_code or st_name
                                    is_inlier = bool(st.get('is_inlier', True))
                                    extra_markers.append({
                                        'kind': 'infra_detection',
                                        'lat': float(st.get('lat')),
                                        'lon': float(st.get('lon')),
                                        'elev_m': 0.0,
                                        'label': st_label,
                                        'color': '#006400' if is_inlier else '#777777',
                                        'label_color': '#006400' if is_inlier else '#777777',
                                        'symbol': '^' if is_inlier else 'o',
                                        'plotly_symbol': 'triangle-up' if is_inlier else 'circle',
                                        'size': 5 if is_inlier else 4,
                                        'drop_to_ground': False,
                                        'opacity': 0.9 if is_inlier else 0.55,
                                    })
                                except Exception:
                                    continue

                            plot_opts['extra_markers'] = extra_markers

                            # Wind-corrected rings (isochrones)
                            paths = []
                            for ring in ((infra_fit.get('isochrones') if isinstance(infra_fit, dict) else None) or (infra_fit_for_maps.get('isochrones') or [])):
                                try:
                                    paths.append({
                                        'label': f"Isochrone {ring.get('code','')}",
                                        'color': '#006400',
                                        'lats': ring.get('lats'),
                                        'lons': ring.get('lons'),
                                        'elev_km': float(ring.get('alt_km', elev_km)),
                                        'station_lat': ring.get('station_lat'),
                                        'station_lon': ring.get('station_lon'),
                                        'station_elev_m': ring.get('station_elev_m'),
                                    })
                                except Exception:
                                    continue
                            if paths:
                                plot_opts['extra_paths'] = paths
                except Exception:
                    pass
            
            metrack_info = analysis_results.get('metrack_info') if isinstance(analysis_results, dict) else None
            metrack_plot_data = analysis_results.get('metrack_plot_data') if isinstance(analysis_results, dict) else None
            if metrack_info is not None and metrack_plot_data is not None:
                generate_metrack_plots(metrack_info, metrack_plot_data,
                                       plot_opts, translations=translations, output_prefix=file_prefix)
            else:
                logging.info("Skipping metrack plots (no metrack data available).")

            if analysis_results.get('fbspd_plot_data') is not None:
                generate_speed_plots(analysis_results['fbspd_plot_data'], 
                                     translations=translations, output_prefix=file_prefix)
            else:
                logging.info("Skipping speed plots (no FBSPD plot data available).")

            if is_multistation and analysis_results and current_orbit_data.get('valid'):
                orbit(True, current_orbit_data['entry_speed'], 0, str(res_filename), 
                      date.strftime('%Y-%m-%d'), date.strftime('%H:%M:%S'), 'save', 
                      interactive=True, translations=translations, output_prefix=file_prefix)
            
            dpi_map = {'map.svg': Config.SVG_MAP_DPI, 'orbit.svg': Config.SVG_ORBIT_DPI}
            for svg_name in ["posvstime.svg", "spd_acc.svg", "orbit.svg", "height.svg", "map.svg"]:
                svg_path = event_dir / f"{file_prefix}{svg_name}"
                if svg_path.exists():
                    dpi = dpi_map.get(svg_name, Config.SVG_DEFAULT_DPI)
                    svg_to_jpg(svg_path, svg_path.with_suffix('.jpg'), dpi)
            
            if is_multistation and analysis_results:
                generate_triangulation_html_report(event_dir / f"{file_prefix}tables.html", 
                                                   analysis_results['resdat'], 
                                                   current_orbit_data, 
                                                   analysis_results['placename'], 
                                                   translations, lang)
        
        except Exception as e:
            logging.error(f"Failed to generate output for language '{lang}': {e}", exc_info=True)

    if is_multistation and analysis_results:
        if WINDPROFILE_AVAILABLE:
            try:
                end_height_km = analysis_results['resdat'].height[1]
                if 10 <= end_height_km <= 40:
                    wind_csv_path = event_dir / "wind_profile.csv"
                    success = False
                    matched_time = None
                    
                    logging.info(f"End altitude {end_height_km:.1f} km is between 10-40 km. Fetching wind profile.")
                    end_lat = analysis_results['resdat'].lat1[1]
                    end_lon = analysis_results['resdat'].long1[1]
                    event_timestamp = date.timestamp()
                    
                    success, matched_time = windprofile.get_wind_profile(
                        end_lat, 
                        end_lon, 
                        event_timestamp, 
                        str(wind_csv_path)
                    )

                    if success:
                        if not fast:
                            logging.info(f"Successfully fetched wind data for {matched_time}")
                        
                        for lang in SUPPORTED_LANGS:
                            translations = load_translations(lang)
                            file_prefix = '' if lang == DEFAULT_LANG else f'{lang}_'
                            
                            windprofile.generate_wind_plot(
                                str(wind_csv_path), 
                                translations, 
                                matched_time, 
                                file_prefix
                            )
                            
                            svg_path = event_dir / f"{file_prefix}wind_profile.svg"
                            if svg_path.exists():
                                svg_to_jpg(svg_path, svg_path.with_suffix('.jpg'), Config.SVG_DEFAULT_DPI)
                            else:
                                logging.warning(f"Wind profile SVG not found: {svg_path}")
                    elif not fast:
                        logging.warning("Failed to get wind profile data.")
                else:
                    logging.info(f"End altitude {end_height_km:.1f} km is outside 10-40 km range. Skipping wind profile.")

            except Exception as e:
                logging.error(f"Failed to generate wind profile: {e}", exc_info=True)

        # --- Infrasound fit (after wind profile if available) ---
        try:
            infra_files = list(event_dir.glob('*/cam*/infra.txt'))
            if len(infra_files) >= 3:
                _fetch_infra_wind_profiles(event_dir, date.timestamp())
                infra_out = event_dir / 'infra_fit.json'
                infra_report = event_dir / 'infra_fit_report.txt'
                cmd = [
                    sys.executable,
                    str(Config.BIN_DIR / 'infra_fit.py'),
                    str(event_dir),
                    '--output', str(infra_out),
                    '--report', str(infra_report),
                ]
                if verbose:
                    cmd.append('--verbose')
                if verbose:
                    env = os.environ.copy()
                    env.setdefault('OMP_NUM_THREADS', '1')
                    env.setdefault('OPENBLAS_NUM_THREADS', '1')
                    env.setdefault('MKL_NUM_THREADS', '1')
                    env.setdefault('NUMEXPR_NUM_THREADS', '1')
                    p = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
                    try:
                        if p.stdout is not None:
                            for line in p.stdout:
                                try:
                                    print(line, end='')
                                except Exception:
                                    pass
                    finally:
                        rc = p.wait()
                    class _R:
                        pass
                    rr = _R()
                    rr.returncode = int(rc)
                    proc = rr
                else:
                    env = os.environ.copy()
                    env.setdefault('OMP_NUM_THREADS', '1')
                    env.setdefault('OPENBLAS_NUM_THREADS', '1')
                    env.setdefault('MKL_NUM_THREADS', '1')
                    env.setdefault('NUMEXPR_NUM_THREADS', '1')
                    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if proc.returncode != 0:
                    if verbose:
                        msg = (
                            "Infrasound fit failed. Continuing without infrasound source. "
                            f"returncode={proc.returncode}"
                        )
                    else:
                        stderr_snip = (proc.stderr or '').strip()[:400]
                        stdout_snip = (proc.stdout or '').strip()[:400]
                        msg = (
                            "Infrasound fit failed. Continuing without infrasound source. "
                            f"returncode={proc.returncode} "
                            f"stdout={stdout_snip} "
                            f"stderr={stderr_snip}"
                        )
                    logging.warning(msg)
                    if infra_report.exists():
                        logging.warning(f"Infrasound fit report: {infra_report}")
                elif infra_out.exists():
                    try:
                        analysis_results['infra_fit'] = json.loads(infra_out.read_text(encoding='utf-8'))
                        if analysis_results['infra_fit'].get('summary'):
                            logging.info(f"Infrasound fit: {analysis_results['infra_fit'].get('summary')}")
                        if infra_report.exists():
                            logging.info(f"Infrasound fit report: {infra_report}")
                    except Exception as e:
                        logging.warning(f"Could not read {infra_out}: {e}")

                    # Re-generate KML so it can include the infrasound source placemark.
                    try:
                        fb2kml(str(res_filename))
                    except Exception as e:
                        logging.warning(f"Failed to regenerate KML after infrasound fit: {e}")

                    # Re-generate maps so marker is present even if outputs were already generated.
                    try:
                        for lang in SUPPORTED_LANGS:
                            translations = load_translations(lang)
                            file_prefix = '' if lang == DEFAULT_LANG else f'{lang}_'

                            plot_opts = {
                                'doplot': 'save',
                                'interactive': True,
                                'autoborders': True,
                                'azonly': False,
                                'mapres': 'i',
                            }
                            try:
                                infra_fit = analysis_results.get('infra_fit')
                                if infra_fit and 'lat' in infra_fit and 'lon' in infra_fit:
                                    elev_km = float(infra_fit.get('elev_m', 0.0)) / 1000.0
                                    if 5.0 <= elev_km <= 80.0:
                                        extra_markers = []

                                        extra_markers.append({
                                            'kind': 'infra_source',
                                            'lat': float(infra_fit['lat']),
                                            'lon': float(infra_fit['lon']),
                                            'elev_m': float(infra_fit.get('elev_m', 0.0)),
                                            'label': (lambda _t0: (translations.get('infrasound_source', 'Infrasound source') + (f" {_t0}" if _t0 else "")))(
                                                (datetime.datetime.fromtimestamp(float(infra_fit.get('t0_unix')), tz=datetime.timezone.utc).strftime('%H:%M:%S')
                                                 if infra_fit.get('t0_unix') is not None else "")
                                            ),
                                            'color': '#006400',
                                            'label_color': 'black',
                                            'symbol': 'o',
                                            'plotly_symbol': 'circle',
                                            'size': 7,
                                            'drop_to_ground': True,
                                        })

                                        infra_sites = infra_fit.get('stations_all') or infra_fit.get('stations') or []
                                        for st in infra_sites:
                                            try:
                                                st_code = str(st.get('code', '')).strip()
                                                st_name = str(st.get('station', '')).strip()
                                                st_label = st_code or st_name
                                                is_inlier = bool(st.get('is_inlier', True))
                                                extra_markers.append({
                                                    'kind': 'infra_detection',
                                                    'lat': float(st.get('lat')),
                                                    'lon': float(st.get('lon')),
                                                    'elev_m': 0.0,
                                                    'label': st_label,
                                                    'color': '#006400' if is_inlier else '#777777',
                                                    'label_color': '#006400' if is_inlier else '#777777',
                                                    'symbol': '^' if is_inlier else 'o',
                                                    'plotly_symbol': 'triangle-up' if is_inlier else 'circle',
                                                    'size': 5 if is_inlier else 4,
                                                    'drop_to_ground': False,
                                                    'opacity': 0.9 if is_inlier else 0.55,
                                                })
                                            except Exception:
                                                continue

                                        plot_opts['extra_markers'] = extra_markers

                                        paths = []
                                        for ring in (infra_fit.get('isochrones') or []):
                                            try:
                                                paths.append({
                                                    'label': f"Isochrone {ring.get('code','')}",
                                                    'color': '#006400',
                                                    'lats': ring.get('lats'),
                                                    'lons': ring.get('lons'),
                                                    'elev_km': float(ring.get('alt_km', elev_km)),
                                                    'station_lat': ring.get('station_lat'),
                                                    'station_lon': ring.get('station_lon'),
                                                    'station_elev_m': ring.get('station_elev_m'),
                                                })
                                            except Exception:
                                                continue
                                        if paths:
                                            plot_opts['extra_paths'] = paths
                            except Exception:
                                pass

                            generate_metrack_plots(
                                analysis_results['metrack_info'],
                                analysis_results['metrack_plot_data'],
                                plot_opts,
                                translations=translations,
                                output_prefix=file_prefix,
                            )

                            svg_path = event_dir / f"{file_prefix}map.svg"
                            if svg_path.exists():
                                svg_to_jpg(svg_path, svg_path.with_suffix('.jpg'), Config.SVG_MAP_DPI)
                    except Exception as e:
                        logging.warning(f"Failed to regenerate maps after infrasound fit: {e}")
            else:
                logging.info("No/insufficient infra.txt files found (need >=3). Skipping infrasound fit.")
        except Exception as e:
            logging.warning(f"Infrasound fit step failed. Continuing: {e}", exc_info=True)


    if is_multistation and analysis_results:
        event_time_utc = date.replace(tzinfo=pytz.utc)
        current_time_utc = datetime.datetime.now(pytz.utc)
        event_age = current_time_utc - event_time_utc
        max_age_seconds = Config.MAX_EVENT_AGE_FOR_TWEET_HOURS * 3600

        if event_age.total_seconds() > max_age_seconds:
            logging.info(f"Event is older than {Config.MAX_EVENT_AGE_FOR_TWEET_HOURS} hours ({event_age}). Skipping tweet.")
        else:
            resdat = analysis_results['resdat']
            height_valid = 10 < resdat.height[0] <= 150 and 10 <= resdat.height[1] <= 150
            
            # For tweeting, we use the 'nb' name. If raw name is 'nb', we use that.
            # If raw name is code, we translate.
            raw_shower_name = analysis_results['orbit_data'].get('showername', '')
            showername_sg = raw_shower_name
            
            # Optional: Ensure tweet uses 'nb' explicitly if raw wasn't nb
            if raw_shower_name and raw_shower_name in nb_translations.get('showers', {}):
                 showername_sg = nb_translations['showers'][raw_shower_name]

            send_tweet(event_dir, date, analysis_results['placename'], 
                       showername_sg, 
                       len(station_codes), analysis_results['first_run'], height_valid, nb_translations)

    elif not is_multistation:
        logging.info("Not enough stations for triangulation. Skipping tweet.")
        
    os.chdir(original_cwd)
    logging.info(f"Returned to original directory: {original_cwd}")


def main():
    """Main script execution flow."""
    
    # --- Register cleanup to fix terminal echo on exit ---
    atexit.register(restore_terminal_state)
    
    parser = argparse.ArgumentParser(
        description="Fetch and process meteor data, or reprocess an existing event directory.",
        usage="""
    To fetch:     python3 fetch.py <station> <port> <remote_dir>
    To reprocess: python3 fetch.py <local_event_directory> [--fast] [--all] [--origcen]
        """
    )
    parser.add_argument("arg1", help="Station name OR path to local event directory for reprocessing.")
    parser.add_argument("port", nargs='?', help="SSH port (if fetching).")
    parser.add_argument("remote_dir", nargs='?', help="Remote directory path (if fetching).")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="For reprocessing mode: skips slow video processing (process.py) "
             "and wind data fetching, and only regenerates analysis, plots, and tables."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Skip elimination of outlier stations (disables RANSAC)."
    )
    parser.add_argument(
        "--origcen",
        action="store_true",
        help="Use original centroid.txt data instead of calculated centroid2.txt."
    )
    parser.add_argument(
        "--infrasound",
        action="store_true",
        help="For reprocessing mode: skip pre-processing and only run infrasound fitting (infra_fit.py) for the event."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (passes --verbose to infra_fit.py and prints progress)."
    )
    
    args = parser.parse_args()

    is_reprocessing = args.port is None and args.remote_dir is None

    if is_reprocessing:
        # --- Reprocessing Mode ---
        final_event_dir = Path(args.arg1).resolve()
        if not final_event_dir.is_dir():
            sys.exit(f"Error: Directory not found: {final_event_dir}")
        
        log_name = 'reprocess_fast.log' if args.fast else 'reprocess.log'
        log_path = final_event_dir / log_name
        setup_logging(log_path)
        logging.info(f"--- Starting REPROCESSING{' (FAST MODE)' if args.fast else ''} for event directory: {final_event_dir} ---")
        
        try:
            date_str = final_event_dir.parent.name
            time_str = final_event_dir.name
            processing_date = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        except (ValueError, IndexError):
            logging.error(f"Could not deduce date from path '{final_event_dir}'. Path must end in '<YYYYMMDD>/<HHMMSS>'.")
            sys.exit(f"Could not deduce date from path '{final_event_dir}'. Path must end in '<YYYYMMDD>/<HHMMSS>'.")
        
        if not (final_event_dir / 'index.php').exists():
            try: (final_event_dir / 'index.php').symlink_to('../../report.php')
            except OSError as e:
                logging.warning(f"Could not create index.php symlink: {e}")
        
        try:
            process_event(final_event_dir, processing_date, fast=args.fast, all_stations=args.all, use_orig_cen=args.origcen, infrasound_only=args.infrasound, verbose=args.verbose)
        except Exception as e:
            logging.critical(f"A critical error occurred during reprocessing: {e}", exc_info=True)
        finally:
            logging.info("--- Reprocessing script finished. ---")
        sys.exit(0)
        # --- End Reprocessing Mode ---


    # --- Fetch Mode ---
    if args.port is None or args.remote_dir is None:
         parser.error("Missing arguments. Both port and remote_dir are required for fetching.")
    
    if args.fast:
        logging.warning("--fast option is only supported for reprocessing mode. Ignoring.")
    
    if args.origcen:
        logging.warning("--origcen option is only relevant for reprocessing mode logic, but will be passed along.")

    station = re.sub(r'\W', '', args.arg1)
    port = re.sub(r'\D', '', args.port)
    remote_cleaned = re.sub(r'[^a-zA-Z0-9_/]', '', args.remote_dir)
    
    try:
        dir_parts = [p for p in remote_cleaned.split('/') if p]
        date_str, time_str, cam_name = dir_parts[3], dir_parts[4], dir_parts[1]
    except IndexError:
        logging.error(f"Could not parse remote directory structure: {remote_cleaned}")
        sys.exit(f"Could not parse remote directory structure: {remote_cleaned}")
        
    local_dir = Config.METEOR_DATA_DIR / date_str / time_str / station / cam_name
    
    log_path = local_dir.parent / 'fetch.log'
    setup_logging(log_path)
    logging.info(f"--- Starting FETCH for station {station}, remote dir {args.remote_dir} ---")

    if not fetch_data(station, port, args.remote_dir, local_dir):
        logging.error("Failed to fetch data. Exiting.")
        sys.exit("Failed to fetch data.")
        
    try:
        set_permissions(Config.METEOR_DATA_DIR / date_str)
    except Exception as e:
        logging.warning(f"Failed to set permissions on {Config.METEOR_DATA_DIR / date_str}: {e}")
        
    date_obj = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
    
    create_centroid_file(local_dir)
    final_event_dir = find_and_merge_event_directory(date_obj, station, local_dir)
    
    log_path = final_event_dir / 'process.log'
    setup_logging(log_path)
    logging.info(f"--- Event merged into {final_event_dir}. Starting main processing. ---")
    
    proc_date_str = final_event_dir.parent.name
    proc_time_str = final_event_dir.name
    processing_date = datetime.datetime.strptime(proc_date_str + proc_time_str, '%Y%m%d%H%M%S')
    
    try:
        process_event(final_event_dir, processing_date, fast=False, all_stations=args.all, use_orig_cen=args.origcen, infrasound_only=args.infrasound, verbose=args.verbose)
    except Exception as e:
        logging.critical(f"A critical error occurred during event processing: {e}", exc_info=True)
    finally:
        logging.info("--- Script finished. ---")


if __name__ == '__main__':
    fetch_foreign_script = Path(__file__).parent / 'fetch_foreign.sh'
    try:
        argv = sys.argv[1:]
        legacy_remote_fetch_mode = (len(argv) == 3 and all((not str(a).startswith('-')) for a in argv))
    except Exception:
        legacy_remote_fetch_mode = False
    if legacy_remote_fetch_mode and fetch_foreign_script.exists():
        try:
            run_command([str(fetch_foreign_script)])
        except Exception as e:
            logging.warning(f"Failed to run fetch_foreign.sh: {e}")
    main()
