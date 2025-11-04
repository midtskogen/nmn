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
    python3 fetch.py <local_event_directory> (for reprocessing)
"""

import argparse
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

try:
    import cairosvg
    from PIL import Image, ImageOps
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

# Local script imports from the same project
from fb2kml import fb2kml
from fbspd_merge import readres, calculate_speed_profile, generate_speed_plots
from metrack import calculate_trajectory, generate_plots as generate_metrack_plots, write_res_file
from orbit import calc_azalt, orbit
import reverse_geocode


class Config:
    """Configuration constants for the script."""
    # Base directory for all meteor data on the local server
    BASE_HTTP_DIR = Path('/home/httpd/norskmeteornettverk.no')
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

# --- Internationalization (i18n) Setup ---
SUPPORTED_LANGS = ['nb', 'en', 'de', 'cs']
DEFAULT_LANG = 'nb'
LOC_DIR = Config.BIN_DIR / 'loc'

def load_translations(lang_code: str) -> dict:
    """Loads the translation dictionary for a given language, with fallback to default."""
    default_path = LOC_DIR / f"{DEFAULT_LANG}.json"
    lang_path = LOC_DIR / f"{lang_code}.json"

    translations = {}
    if default_path.exists():
        with default_path.open('r', encoding='utf-8') as f:
            translations = json.load(f)

    if lang_code != DEFAULT_LANG and lang_path.exists():
        with lang_path.open('r', encoding='utf-8') as f:
            translations.update(json.load(f))
            
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
    host = station if port == '0' else 'localhost'
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
            Path(root, d).chmod(0o775)
        for f in files:
            Path(root, f).chmod(0o664)


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
        logging.warning(f"Event file not found: {event_file}")
        return

    obs = configparser.ConfigParser()
    obs.read(event_file)

    coordinates = obs.get('trail', 'coordinates').split()
    timestamps = obs.get('trail', 'timestamps').split()
    az1_str, alt1_str = obs.get('summary', 'startpos').split()
    az2_str, alt2_str = obs.get('summary', 'endpos').split()

    p1 = to_cartesian(float(az1_str), float(alt1_str))
    p2 = to_cartesian(float(az2_str), float(alt2_str))

    coordinates2 = []
    for c in coordinates:
        az3_str, alt3_str = c.split(',')
        p3 = to_cartesian(float(az3_str), float(alt3_str))
        t = np.cross(np.cross(p1, p2), np.cross(p3, np.cross(p1, p2)))
        norm_t = np.linalg.norm(t)
        if norm_t > 0:
            coordinates2.append(from_cartesian(t / norm_t))
        else:
            coordinates2.append((0,0))

    try:
        obs_file = next(local_dir.glob('*[0-9][0-9].txt'))
        code = obs_file.read_text().split()[12]
    except (StopIteration, IndexError):
        logging.error("Could not find an observation file or parse the station code.")
        return

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

            shutil.rmtree(current_local_dir.parent.parent)

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
            shower_name = translations.get("sporadic", "sporadic") if not orbit_data['showername'] else orbit_data['showername']
            table1 += f"""
    <tr><td>{translations.get("radiant_ra", "Radiant R.A.")}:</td><td> {ramin // 60:02d}:{ramin % 60:02d} ({orbit_data['ra']:.1f}°)</td></tr>
    <tr><td>{translations.get("radiant_dec", "Radiant Dec.")}:</td><td> {orbit_data['dec']:.1f}°</td></tr>
    <tr><td>{translations.get("meteor_shower", "Meteor Shower")}:</td><td align=center> {shower_name}</td></tr>
"""
        table1 += "</table>"
        
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
            cfg.read(event_file)
            ts_float = float(cfg.get('trail', 'timestamps').split()[0])
            ts_utc = datetime.datetime.fromtimestamp(ts_float, tz=pytz.utc)
            ts_str = ts_utc.strftime('%Y%m%d%H%M%S')
            
            code = ''
            obs_txt_file = event_file.parent / f"{station}-{ts_str}.txt"
            if obs_txt_file.exists():
                try: code = obs_txt_file.read_text().split()[12]
                except IndexError: pass
            
            f.write('</td></tr></table>')

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
$b_prefix = ($lang === '{default_lang_code}') ? '' : substr($lang, 0, 2) . '_';
$brightness_jpg_path = "{station}/{cam}/" . $b_prefix . "brightness.jpg";
$brightness_jpg_url = "{url_base}/{cam}/" . $b_prefix . "brightness.jpg";
?>
    <div style="text-align: center;">
        <?php if (file_exists($webm_path)) {{ ?>
        <a href="<?php echo $webm_url2; ?>"><video autoplay loop muted playsinline style="max-width: 800px; width: 100%; height: auto; border: 1px solid black;"><source src="<?php echo $webm_url; ?>" type="video/webm"></video></a><br>
        <?php }} elseif (file_exists($jpg_path)) {{ ?>
        <a href="<?php echo $jpg_url; ?>"><img src="<?php echo $jpg_url; ?>" style="max-width: 800px; width: 100%; height: auto;" alt="fireball"><br></a>
        <?php }} ?>
    </div>
<table><tr><td valign=top>
<a href="{url_base}/{cam}/{station_ts}-gnomonic.mp4"><img src="{url_base}/{cam}/{station_ts}-gnomonic-grid.jpg" width=768 alt="gnomonic"></a>
</td>
<td valign=top>
<table border=1>
<tr><td><b>{videos_header}</b><br>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic.mp4">{gnomonic_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic-grid.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic-grid.mp4">{gnomonic_with_coords_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}.mp4">{original_label}</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}_hevc.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}_hevc.mp4">{original_hevc_label}</a><br> <?php }} ?>
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
<?php if (file_exists($brightness_jpg_path)) {{ ?><a href="<?php echo $brightness_jpg_url; ?>"><img src="<?php echo $brightness_jpg_url; ?>" width=400 alt="brightness"><br></a> <?php }} ?>
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
                original_hevc_label=translations.get("original_hevc", "Original (HEVC)"),
                original_with_coords_label=translations.get("original_with_coords", "Original with coordinates"),
                gnomonic_uncorrected_with_coords_label=translations.get("gnomonic_uncorrected_with_coords", "Uncorrected gnomonic with coordinates"),
                gnomonic_with_labels_label=translations.get("gnomonic_with_labels", "Gnomonic with labels"),
                gnomonic_uncorrected_with_labels_label=translations.get("gnomonic_uncorrected_with_labels", "Uncorrected gnomonic with labels"),
                original_with_mask_label=translations.get("original_with_mask", "Original with mask"),
                detection_label=translations.get("detection", "Detection"),
                observation_label=translations.get("observation", "Observation"),
                coordinates_label=translations.get("coordinates", "Coordinates"),
                error_messages_label=translations.get("error_messages", "Error Messages"),
                log_file_label=translations.get("log_file", "Log")
            ))


def send_tweet(event_dir: Path, date: datetime.datetime, placename: str, showername_sg: str, count: int, first: bool, height_valid: bool, translations: dict):
    """Constructs and sends a tweet if conditions are met."""
    if not (count >= 2 and first and height_valid and len(sys.argv) == 4):
        logging.info("Conditions for tweeting not met. Skipping.")
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
            run_command(cmd, shell=True)
        except Exception as e:
            logging.error(f"Failed to send tweet using key {key}: {e}")


def process_event(event_dir: Path, date: datetime.datetime):
    """Main processing logic for a meteor event."""
    logging.info(f"Processing event in directory: {event_dir}")
    obs_filename = f"obs_{date.strftime('%Y-%m-%d_%H:%M:%S')}.txt"
    obs_filepath = event_dir / obs_filename
    res_filename = obs_filepath.with_suffix('.res')
    
    station_codes, station_name_to_code = set(), {}
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
                continue
    
    original_cwd = Path.cwd()
    os.chdir(event_dir)
    logging.info(f"Changed working directory to: {event_dir}")
    if not Path('index.php').exists():
        try: Path('index.php').symlink_to('../../report.php')
        except OSError: pass

    is_multistation = len(station_codes) > 1
    analysis_results = {}

    if is_multistation:
        try:
            logging.info(f"Multi-station event ({len(station_codes)} stations). Performing core analysis...")
            
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
            fbspd_opts = {'debug': True, 'seed': 0}
            
            metrack_info, metrack_plot_data = calculate_trajectory(str(obs_filepath), **metrack_opts)
            if not metrack_plot_data: raise ValueError("Metrack calculation failed.")
            
            write_res_file(metrack_plot_data['track_start'], metrack_plot_data['track_end'],
                           metrack_plot_data['cross_pos_inliers'], metrack_plot_data['inlier_obs_data'], str(obs_filepath))
            
            resdat = readres(str(res_filename))
            fb2kml(str(res_filename))
            
            inlier_codes = set(metrack_info.inlier_stations)
            centroid_files = [str(p) for p in event_dir.glob('*/*/centroid2.txt') if station_name_to_code.get(p.parts[-3]) in inlier_codes]
            fbspd_results, fbspd_plot_data = calculate_speed_profile(str(res_filename), centroid_files, str(obs_filepath), **fbspd_opts)
            if not fbspd_results: raise ValueError("FBSPD calculation failed.")
            entry_speed = fbspd_results['initial_speed']

            az, alt = calc_azalt(resdat.lat1[0], resdat.long1[0], resdat.height[0], resdat.lat1[1], resdat.long1[1], resdat.height[1])
            placename = get_location_from_coords(resdat.lat1[1], resdat.long1[1])
            if placename: Path('location.txt').write_text(placename, encoding='utf-8')
            
            orbit_data = {'valid': False, 'entry_speed': entry_speed, 'az': az, 'alt': alt}
            if entry_speed > 9.8:
                ra, dec, (rp, ecc, inc, lnode, argp, m0, t0), s_name, s_name_sg, valid = orbit(
                    True, entry_speed, 0, str(res_filename), date.strftime('%Y-%m-%d'), date.strftime('%H:%M:%S'), doplot=''
                )
                orbit_data.update({'ra': ra, 'dec': dec, 'rp': rp, 'ecc': ecc, 'inc': inc, 
                                   'lnode': lnode, 'argp': argp, 'm0': m0, 't0': t0,
                                   'showername': s_name, 'showername_sg': s_name_sg, 'valid': valid})

            analysis_results = {
                'metrack_info': metrack_info, 'metrack_plot_data': metrack_plot_data, 'fbspd_plot_data': fbspd_plot_data,
                'orbit_data': orbit_data, 'resdat': resdat, 'placename': placename,
                'first_run': not any(p.name.endswith('map.jpg') for p in event_dir.glob('*.jpg'))
            }
            logging.info("Core analysis successful.")

        except Exception as e:
            logging.error(f"Core analysis failed: {e}", exc_info=True)
            is_multistation = False

    for lang in SUPPORTED_LANGS:
        try:
            logging.info(f"--- Generating outputs for language: [{lang}] ---")
            translations = load_translations(lang)
            file_prefix = '' if lang == DEFAULT_LANG else f'{lang}_'

            generate_station_html_report(event_dir / f"{file_prefix}stations.html", event_dir, translations)

            if is_multistation and analysis_results:
                logging.info(f"Generating translated plots and reports for [{lang}]")
                
                plot_opts = {
                    'doplot': 'save', 
                    'interactive': True, 
                    'autoborders': True,
                    'azonly': False,
                    'mapres': 'i'
                }
                
                generate_metrack_plots(analysis_results['metrack_info'], analysis_results['metrack_plot_data'], 
                                       plot_opts, translations=translations, output_prefix=file_prefix)

                generate_speed_plots(analysis_results['fbspd_plot_data'], 
                                     translations=translations, output_prefix=file_prefix)

                if analysis_results['orbit_data']['valid']:
                     orbit(True, analysis_results['orbit_data']['entry_speed'], 0, str(res_filename), 
                           date.strftime('%Y-%m-%d'), date.strftime('%H:%M:%S'), 'save', 
                           interactive=True, translations=translations, output_prefix=file_prefix)
                
                dpi_map = {'map.svg': Config.SVG_MAP_DPI, 'orbit.svg': Config.SVG_ORBIT_DPI}
                for svg_name in ["posvstime.svg", "spd_acc.svg", "orbit.svg", "height.svg", "map.svg"]:
                    svg_path = event_dir / f"{file_prefix}{svg_name}"
                    if svg_path.exists():
                        dpi = dpi_map.get(svg_name, Config.SVG_DEFAULT_DPI)
                        svg_to_jpg(svg_path, svg_path.with_suffix('.jpg'), dpi)
                
                generate_triangulation_html_report(event_dir / f"{file_prefix}tables.html", analysis_results['resdat'], analysis_results['orbit_data'], analysis_results['placename'], translations, lang)
        
        except Exception as e:
            logging.error(f"Failed to generate output for language '{lang}': {e}", exc_info=True)

    if is_multistation and analysis_results:
        resdat = analysis_results['resdat']
        height_valid = 10 < resdat.height[0] <= 150 and 10 <= resdat.height[1] <= 150
        nb_translations = load_translations('nb')
        send_tweet(event_dir, date, analysis_results['placename'], analysis_results['orbit_data'].get('showername_sg', ''), len(station_codes), analysis_results['first_run'], height_valid, nb_translations)
    elif not is_multistation:
        logging.info("Not enough stations for triangulation. Skipping tweet.")
        
    os.chdir(original_cwd)
    logging.info(f"Returned to original directory: {original_cwd}")


def main():
    """Main script execution flow."""
    if len(sys.argv) == 2:
        final_event_dir = Path(sys.argv[1]).resolve()
        if not final_event_dir.is_dir():
            sys.exit(f"Error: Directory not found: {final_event_dir}")
        setup_logging(final_event_dir / 'reprocess.log')
        logging.info(f"Starting reprocessing for event directory: {final_event_dir}")
        try:
            date_str = final_event_dir.parent.name
            time_str = final_event_dir.name
            processing_date = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
        except (ValueError, IndexError):
            sys.exit(f"Could not deduce date from path '{final_event_dir}'. Path must end in '<YYYYMMDD>/<HHMMSS>'.")
        
        if not (final_event_dir / 'index.php').exists():
            try: (final_event_dir / 'index.php').symlink_to('../../report.php')
            except OSError: pass
        
        try:
            process_event(final_event_dir, processing_date)
        except Exception as e:
            logging.critical(f"A critical error occurred during reprocessing: {e}", exc_info=True)
        finally:
            logging.info("Reprocessing script finished.")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Fetch and process meteor data, or reprocess an existing event directory.",
        usage="""
    To fetch: python3 fetch.py <station> <port> <remote_dir>
    To reprocess: python3 fetch.py <local_event_directory>
        """
    )
    parser.add_argument("station", nargs='?', help="Name of station.")
    parser.add_argument("port", nargs='?', help="SSH port.")
    parser.add_argument("remote_dir", nargs='?', help="Remote directory path.")
    if len(sys.argv) != 4: parser.print_help(); sys.exit(1)
    
    args = parser.parse_args()
    station = re.sub(r'\W', '', args.station)
    port = re.sub(r'\D', '', args.port)
    remote_cleaned = re.sub(r'[^a-zA-Z0-9_/]', '', args.remote_dir)
    dir_parts = [p for p in remote_cleaned.split('/') if p]
    date_str, time_str, cam_name = dir_parts[3], dir_parts[4], dir_parts[1]
    
    local_dir = Config.METEOR_DATA_DIR / date_str / time_str / station / cam_name
    setup_logging(local_dir.parent / 'fetch.log')

    if not fetch_data(station, port, args.remote_dir, local_dir):
        sys.exit("Failed to fetch data.")
        
    set_permissions(Config.METEOR_DATA_DIR / date_str)
    date_obj = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')

    report_log = local_dir / 'report.log'
    report_script = Config.BIN_DIR / 'process.py'
    report_cmd = [sys.executable, str(report_script), str(local_dir / 'event.txt')]

    logging.info(f"--- Running {report_script.name} for {local_dir / 'event.txt'} ---")
    with report_log.open('w') as log_file:
       subprocess.call(report_cmd, stdout=log_file, stderr=log_file)
    if not local_dir.exists():
        logging.info("--- Event was discarded by process.py ---")
        sys.exit(0)

    create_centroid_file(local_dir)
    final_event_dir = find_and_merge_event_directory(date_obj, station, local_dir)
    proc_date_str = final_event_dir.parent.name
    proc_time_str = final_event_dir.name
    processing_date = datetime.datetime.strptime(proc_date_str + proc_time_str, '%Y%m%d%H%M%S')
    
    try:
        process_event(final_event_dir, processing_date)
    except Exception as e:
        logging.critical(f"A critical error occurred during event processing: {e}", exc_info=True)
    finally:
        logging.info("Script finished.")


if __name__ == '__main__':
    fetch_foreign_script = Path(__file__).parent / 'fetch_foreign.sh'
    if len(sys.argv) == 4 and fetch_foreign_script.exists():
        run_command([str(fetch_foreign_script)])
    main()
