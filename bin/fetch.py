#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetches and processes meteor observation data for the Norsk Meteornettverk.

This script performs the following actions:
1.  Fetches event data from a remote station using rsync.
2.  Processes the raw data to calculate trajectory, orbit, and other metrics.
3.  Generates scientific plots and summary images (SVG to JPG).
4.  Produces HTML and KML reports for web presentation.
5.  Optionally posts a notification to social media.

Usage:
    python3 fetch.py <station_name> <ssh_port> <remote_dir>

Example:
    python3 fetch.py harestua 10003 /meteor/cam2/events/20150819/224804
"""

import argparse
import configparser
import datetime
import glob
import io
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
import numpy
import pytz
from dateutil.tz import tzlocal

# Attempt to import optional libraries for image processing
try:
    import cairosvg
    from PIL import Image, ImageOps
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

# Local script imports from the same project
from fb2kml import fb2kml
from fbspd_merge import fbspd, readres
from metrack import metrack
from orbit import calc_azalt, orbit
import reverse_geocode


class Config:
    """Configuration constants for the script."""
    # Base directory for all meteor data on the local server
    BASE_HTTP_DIR = Path('/home/httpd/norskmeteornettverk.no')
    METEOR_DATA_DIR = BASE_HTTP_DIR / 'meteor'

    # Bandwidth limit for rsync in KB/s
    # Kristiansand has a slower connection
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
    """
    Runs an external command using subprocess.run.
    If stream_output is True, output is printed in real-time.
    """
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
    """
    Fetches data from a remote station using rsync with retries.
    """
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
    az_rad = math.radians(az)
    alt_rad = math.radians(alt)
    return [
        math.cos(alt_rad) * math.cos(az_rad),
        math.cos(alt_rad) * math.sin(az_rad),
        math.sin(alt_rad)
    ]


def from_cartesian(p: list) -> tuple:
    """Converts Cartesian coordinates to Azimuth/Altitude."""
    az = math.degrees(math.atan2(p[1], p[0]))
    alt = math.degrees(math.asin(p[2]))
    return (math.fmod(az + 360, 360), alt)


def create_centroid_file(local_dir: Path):
    """
    Reads event data, recalculates coordinates, and writes centroid2.txt.
    """
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
        t = numpy.cross(numpy.cross(p1, p2), numpy.cross(p3, numpy.cross(p1, p2)))
        coordinates2.append(from_cartesian(t / numpy.linalg.norm(t)))

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
    """
    Searches for an existing event directory and merges the current data into it.
    """
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
    """
    Converts an SVG file to JPG using CairoSVG and Pillow.
    """
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
    """
    Reverse geocodes coordinates to a municipality name using the reverse-geocode module.
    This function is designed to continue gracefully on failure.
    """
    municipality = ''
    try:
        # The new library expects a list of coordinate tuples
        coordinates = [(lat, lon)]
        results = reverse_geocode.search(coordinates)

        if results:
            # 'city' is the key for the municipality in this library
            municipality = results[0].get('city', '')
        else:
            logging.warning(f"Reverse geocode returned no results for {coordinates[0]}.")

    except Exception as e:
        logging.error(f"An error occurred while using the reverse_geocode module: {e}")

    return municipality.strip()


def generate_triangulation_html_report(event_dir: Path, resdat, orbit_data, placename):
    """Generates the HTML tables that depend on triangulation results."""
    
    with (event_dir / 'tables.html').open('w', encoding='utf-8') as f:
        table1 = f"""
<b>Meteorens atmosfæriske bane</b>:<br>
<table border=1>
    <tr><td>Starthøgde:</td><td> {resdat.height[0]:.1f} km</td></tr>
    <tr><td>Slutthøgde:</td><td> {resdat.height[1]:.1f} km</td></tr>
    <tr><td>Startposisjon:</td><td> {resdat.lat1[0]:.3f}N {resdat.long1[0]:.3f}E</td></tr>
    <tr><td>Sluttposisjon:</td><td> {resdat.lat1[1]:.3f}N {resdat.long1[1]:.3f}E</td></tr>
    <tr><td>Retning:</td><td> {math.fmod(orbit_data['az'] + 360, 360):.1f}°</td></tr>
    <tr><td>Fallvinkel:</td><td> {orbit_data['alt']:.1f}°</td></tr>
"""
        if orbit_data['entry_speed'] > 0:
            table1 += f"    <tr><td>Inngangshastighet:</td><td> {orbit_data['entry_speed']:.1f} km/s</td></tr>\n"

        if orbit_data['valid']:
            ramin = int(orbit_data['ra'] * 24 * 60 / 360)
            table1 += f"""
    <tr><td>Radiantens rektascensjon:</td><td> {ramin // 60:02d}:{ramin % 60:02d} ({orbit_data['ra']:.1f}°)</td></tr>
    <tr><td>Radiantens deklinasjon:</td><td> {orbit_data['dec']:.1f}°</td></tr>
    <tr><td>Meteorsverm:</td><td align=center> {'sporadisk' if not orbit_data['showername'] else orbit_data['showername']}</td></tr>
"""
        table1 += "</table>"
        table1 = table1.replace('.', ',')
        
        f.write('<table><tr><td valign=top>\n')
        f.write(table1)
        f.write('\n</td><td valign=top>\n')

        if orbit_data['valid']:
            table2 = f"""
<b>Meteoroidens baneelement:</b><br>
<table border=1>
    <tr><td>Perihelavstand:</td><td> {orbit_data['rp']:.3f} AU</td></tr>
    <tr><td>Eksentrisitet:</td><td> {orbit_data['ecc']:.3f}</td></tr>
    <tr><td>Inklinasjon:</td><td> {orbit_data['inc']:.1f}°</td></tr>
    <tr><td>Knutelengde:</td><td> {orbit_data['lnode']:.1f}°</td></tr>
    <tr><td>Perihelargument:</td><td> {orbit_data['argp']:.1f}°</td></tr>
    <tr><td>Midlere anomali:</td><td> {orbit_data['m0']:.1f}°</td></tr>
    <tr><td>Epoke:</td><td> {orbit_data['t0']}</td></tr>
</table>
"""
            table2 = table2.replace('.', ',')
            f.write(table2)
        
        f.write('</td></tr></table>')


def generate_station_html_report(event_dir: Path):
    """Generates the station-specific HTML file."""

    with (event_dir / 'stations.html').open('w', encoding='utf-8') as f:
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
                try:
                    code = obs_txt_file.read_text().split()[12]
                except IndexError:
                    pass
            
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

if (file_exists($webm_path)) {{
?>
    <div style="text-align: center;">
        <a href="<?php echo $webm_url2; ?>"><video autoplay loop muted playsinline style="max-width: 800px; width: 100%; height: auto; border: 1px solid black;"><source src="<?php echo $webm_url; ?>" type="video/webm"></video></a><br>
    </div>
<?php
}} elseif (file_exists($jpg_path)) {{
?>
    <div style="text-align: center;">
        <a href="<?php echo $jpg_url; ?>"><img src="<?php echo $jpg_url; ?>" style="max-width: 800px; width: 100%; height: auto;" alt="ildkule"><br>
    </div>
<?php
}}
?>
<table><tr><td valign=top>
<a href="{url_base}/{cam}/{station_ts}-gnomonic.mp4"><img src="{url_base}/{cam}/{station_ts}-gnomonic-grid.jpg" width=768 alt="gnomonisk"></a>
</td>
<td valign=top>
<table border=1>
<tr><td><b>Videoer:</b></br>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic.mp4">Gnomonisk</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic-grid.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic-grid.mp4">Gnomonisk med koordinater</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}.mp4">Original</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-grid.mp4")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-grid.mp4">Original med koordinater</a><br> <?php }} ?>
</td></tr>
<tr><td><b>Bilder:</b><br>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic.jpg">Gnomonisk</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic-grid.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic-grid.jpg">Gnomonisk, koordinater</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic-grid-uncorr.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic-grid-uncorr.jpg">Ukorrigert gnomonisk med koordinater</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic-labels.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic-labels.jpg">Gnomonisk med annotering</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-gnomonic-labels-uncorr.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-gnomonic-labels-uncorr.jpg">Ukorrigert gnomonisk med annotering</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}.jpg">Original</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-grid.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-grid.jpg">Original med koordinater</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}-mask.jpg")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}-mask.jpg">Original med maske</a><br> <?php }} ?>
</td></tr>
<tr><td><b>Tekstfiler:</b><br>
<?php if (file_exists("{station}/{cam}/event.txt")) {{ ?>• <a href="{url_base}/{cam}/event.txt">Deteksjon</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/{station_ts}.txt")) {{ ?>• <a href="{url_base}/{cam}/{station_ts}.txt">Observasjon</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/centroid2.txt")) {{ ?>• <a href="{url_base}/{cam}/centroid2.txt">Koordinater</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/lens.pto")) {{ ?>• <a href="{url_base}/{cam}/lens.pto">Original til ekvirektangulær projeksjon</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/gnomonic.pto")) {{ ?>• <a href="{url_base}/{cam}/gnomonic.pto">Original til gnonomisk projeksjon</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/gnomonic_grid.pto")) {{ ?>• <a href="{url_base}/{cam}/gnomonic_grid.pto">Gnonomisk til ekvirektangulær projeksjon</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/gnomonic_corr_grid.pto")) {{ ?>• <a href="{url_base}/{cam}/gnomonic_corr_grid.pto">Korrigert gnonomisk til ekvirektangulær projeksjon</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/stderr.txt")) {{ ?>• <a href="{url_base}/{cam}/stderr.txt">Feilmeldinger</a><br> <?php }} ?>
<?php if (file_exists("{station}/{cam}/report.log")) {{ ?>• <a href="{url_base}/{cam}/report.log">Logg</a><br> <?php }} ?>
</td></tr></table>
<?php if (file_exists("{station}/{cam}/brightness.jpg")) {{ ?><a href="{url_base}/{cam}/brightness.jpg"><img src="{url_base}/{cam}/brightness.jpg" width=400 alt="lyssyrke"><br></a> <?php }} ?>
</td></tr></table>
</p>
</div></div>
            """

            # Construct the correct base URL path and the combined station-timestamp string
            url_base_path = f'/meteor/{event_dir.parent.name}/{event_dir.name}/{station}'
            station_timestamp_str = f"{station}-{ts_str}"

            # Format the template with the correct values and write to the file
            f.write(html_template.format(
                url_base=url_base_path,
                station=station,
                cam=cam,
                station_ts=station_timestamp_str,
                code=code,
                location=location
            ))


def send_tweet(event_dir: Path, date: datetime.datetime, placename: str, showername_sg: str, count: int, first: bool, height_valid: bool):
    """Constructs and sends a tweet if conditions are met."""
    if not (count == 2 and first and height_valid and len(sys.argv) == 4):
        logging.info("Conditions for tweeting not met. Skipping.")
        return
        
    tweet_base = 'Meteor registrert' if not showername_sg else f'{showername_sg.capitalize()} registrert'
    tweet_str = f"{tweet_base} over {placename} " if placename else f"{tweet_base} "

    utc_time = date.replace(tzinfo=pytz.utc)
    local_time = utc_time.astimezone(tzlocal())
    zone = local_time.strftime('%Z')
    if zone == 'CET':
        zone = 'norsk normaltid'
    elif zone == 'CEST':
        zone = 'norsk sommertid'
    
    tweet_str += local_time.strftime(f'%Y-%m-%d %H:%M:%S {zone}')
    link = f"{Config.BASE_HTTP_DIR.name}/meteor/{utc_time.strftime('%Y%m%d/%H%M%S')}/"
    
    full_tweet = f'-status="{tweet_str} {link}"'
    logging.info(f"Prepared tweet: {full_tweet}")
    
    key_solobs = '/var/www/.oysttyerkey-solobs'
    key_main = '/var/www/.oysttyerkey'
    
    cmd1 = f'ulimit -t 100; /usr/bin/oysttyer -ssl -keyf={key_solobs} -silent {full_tweet}'
    cmd2 = f'ulimit -t 100; /usr/bin/oysttyer -ssl -keyf={key_main} -silent {full_tweet}'

    try:
        run_command(cmd1, shell=True)
    except Exception as e:
        logging.error(f"Failed to send tweet via solobs account: {e}")
        
    try:
        run_command(cmd2, shell=True)
    except Exception as e:
        logging.error(f"Failed to send tweet via main account: {e}")


def process_event(event_dir: Path, date: datetime.datetime):
    """Main processing logic for a given meteor event."""
    logging.info(f"Processing event in directory: {event_dir}")
    
    obs_filename = f"obs_{date.strftime('%Y-%m-%d_%H:%M:%S')}.txt"
    obs_filepath = event_dir / obs_filename
    station_codes = set()
    station_name_to_code = {} 
    
    with obs_filepath.open('w', encoding='utf-8') as outfile:
        for obs_file_absolute in sorted(event_dir.glob('*/*/*[0-9][0-9].txt')):
            content = obs_file_absolute.read_text().strip()
            if not content: continue
            outfile.write(content + '\n')
            
            relative_path = obs_file_absolute.relative_to(event_dir)
            station_dir_name = relative_path.parts[0]
            
            try:
                code = content.split()[12]
                station_codes.add(code)
                if station_dir_name not in station_name_to_code:
                    station_name_to_code[station_dir_name] = code
            except IndexError:
                continue
    
    logging.info(f"Created station name-to-code map: {station_name_to_code}")

    original_cwd = Path.cwd()
    os.chdir(event_dir)
    logging.info(f"Changed working directory to: {event_dir}")

    if not Path('index.php').exists():
        try:
            Path('index.php').symlink_to('../../report.php')
        except:
            pass

    # Always generate the station-specific HTML report
    generate_station_html_report(event_dir)

    if len(station_codes) <= 1:
        logging.info(f"Not enough stations ({len(station_codes)}) for triangulation. Aborting full analysis.")
        os.chdir(original_cwd)
        return

    # This block now only runs for multi-station events
    try:
        first_run = not Path('map.jpg').exists()
        res_filename = obs_filename.replace('.txt', '.res')
        
        # 1. Run metrack to generate the .res file
        track_info = metrack(str(obs_filepath), 'save', 1.0, 'h', False, True, timestamp=None, optimize=True, writestat=True, interactive=True)
        
        # Optional: Rerun metrack if results are out of bounds
        try:
            stat_filepath = obs_filepath.with_suffix('.stat')
            config = configparser.ConfigParser()
            config.read(stat_filepath)
            start_h = float(config.get('track', 'startheight').split()[0])
            end_h = float(config.get('track', 'endheight').split()[0])
            if start_h > 200 or end_h < 10:
                logging.info("Rerunning Metrack with optimization disabled.")
                metrack(str(obs_filepath), 'save', 1.0, 'h', False, True, timestamp=None, optimize=False, writestat=True)
        except Exception as e:
            logging.warning(f"Could not check or re-run Metrack: {e}")

        # 2. Process the .res file and other data
        fb2kml(res_filename)
        print(track_info)
        try:
            inlier_codes = set(track_info.inlier_stations)
            logging.info(f"Metrack identified inlier codes: {inlier_codes}")

            all_centroid_paths = glob.glob('*/*/centroid2.txt')
            
            centroid_files = []
            for file_path_str in all_centroid_paths:
                station_dir_name = Path(file_path_str).parts[0]
                station_code = station_name_to_code.get(station_dir_name)
                
                if station_code in inlier_codes:
                    centroid_files.append(file_path_str)
                else:
                    logging.info(f"Excluding outlier centroid file (station: {station_dir_name}, code: {station_code}): {file_path_str}")

            if centroid_files:
                abs_res_path = str(event_dir / res_filename)
                abs_cen_paths_str = ','.join([str(event_dir / p) for p in centroid_files])
                logging.info(f"To reproduce, run: fbspd_merge.py -r {abs_res_path} -c {abs_cen_paths_str} -d {str(obs_filepath)} -o save -v")
                _, entry_speed = fbspd(res_filename, centroid_files, str(obs_filepath), doplot='save', debug=True)
            else:
                logging.warning("No valid centroid files from inlier stations to calculate speed.")
                entry_speed = 0

        except Exception as e:
            logging.error(f"Failed to calculate entry speed with fbspd: {e}")
            entry_speed = 0

        orbit_data = {'valid': False, 'entry_speed': entry_speed}
        if entry_speed > 9.8:
            ra, dec, (rp, ecc, inc, lnode, argp, m0, t0), s_name, s_name_sg, valid = orbit(
                True, entry_speed, 0, res_filename, 
                date.strftime('%Y-%m-%d'), date.strftime('%H:%M:%S'), 'save', interactive=True
            )
            orbit_data.update({
                'ra': ra, 'dec': dec, 'rp': rp, 'ecc': ecc, 'inc': inc, 
                'lnode': lnode, 'argp': argp, 'm0': m0, 't0': t0.rsplit(' ', 1)[0],
                'showername': s_name, 'showername_sg': s_name_sg, 'valid': valid
            })

        svg_to_jpg(Path('posvstime.svg'), Path('posvstime.jpg'), Config.SVG_DEFAULT_DPI)
        svg_to_jpg(Path('spd_acc.svg'), Path('spd_acc.jpg'), Config.SVG_DEFAULT_DPI)
        svg_to_jpg(Path('orbit.svg'), Path('orbit.jpg'), Config.SVG_ORBIT_DPI)
        svg_to_jpg(Path('height.svg'), Path('height.jpg'), Config.SVG_DEFAULT_DPI)
        if Path('map.svg').exists():
            svg_to_jpg(Path('map.svg'), Path('map.jpg'), Config.SVG_MAP_DPI)

        # 3. NOW it's safe to read the .res file
        resdat = readres(res_filename)
        az, alt = calc_azalt(resdat.lat1[0], resdat.long1[0], resdat.height[0], resdat.lat1[1], resdat.long1[1], resdat.height[1])
        orbit_data.update({'az': az, 'alt': alt})

        placename = get_location_from_coords(resdat.lat1[1], resdat.long1[1])
        if placename:
            Path('location.txt').write_text(placename, encoding='utf-8')

        # 4. Generate the final reports using the processed data
        generate_triangulation_html_report(event_dir, resdat, orbit_data, placename)
        
        height_valid = 10 < resdat.height[0] <= 150 and 10 <= resdat.height[1] <= 150
        send_tweet(event_dir, date, placename, orbit_data.get('showername_sg', ''), len(station_codes), first_run, height_valid)
        
    finally:
        os.chdir(original_cwd)
        logging.info(f"Returned to original directory: {original_cwd}")


def main():
    """Main script execution flow."""
    # New logic: Check for a single argument to reprocess an existing event directory
    if len(sys.argv) == 2:
        final_event_dir = Path(sys.argv[1])
        if not final_event_dir.is_absolute():
            final_event_dir = Path.cwd() / final_event_dir
        if not final_event_dir.is_dir():
            print(f"Error: Directory not found: {final_event_dir}", file=sys.stderr)
            sys.exit(1)

        # Set up logging within the event directory for this reprocessing job
        setup_logging(final_event_dir / 'reprocess.log')
        logging.info(f"Starting reprocessing for event directory: {final_event_dir}")

        # Deduce the processing date from the directory path structure (.../YYYYMMDD/HHMMSS)
        try:
            proc_date_str = final_event_dir.parent.name
            proc_time_str = final_event_dir.name
            processing_date = datetime.datetime.strptime(proc_date_str + proc_time_str, '%Y%m%d%H%M%S')
            logging.info(f"Deduced processing date: {processing_date}")
        except (ValueError, IndexError):
            logging.critical(
                f"Could not deduce date from path '{final_event_dir}'. "
                f"Path must end in the format '<YYYYMMDD>/<HHMMSS>'."
            )
            sys.exit(1)

        if not Path(str(final_event_dir) + '/index.php').exists():
            try:
                Path(str(final_event_dir) + '/index.php').symlink_to('../../report.php')
            except:
                pass

        # Run the main processing function
        try:
            process_event(final_event_dir, processing_date)
        except Exception as e:
            logging.critical(f"A critical error occurred during event reprocessing: {e}", exc_info=True)
        finally:
            logging.info("Reprocessing script finished.")
        sys.exit(0)

    # Original logic for fetching data with three arguments
    parser = argparse.ArgumentParser(
        description="Fetch and process meteor data, or reprocess an existing event directory.",
        usage="""
    To fetch: python3 fetch.py <station> <port> <remote_dir>
    To reprocess: python3 fetch.py <local_event_directory>
        """
    )
    parser.add_argument("station", nargs='?', help="Name of the observation station (e.g., 'harestua').")
    parser.add_argument("port", nargs='?', help="SSH port for the connection ('0' for default port 22).")
    parser.add_argument("remote_dir", nargs='?', help="Remote directory path of the event.")

    # Exit if the argument count is incorrect (not 1 for reprocess, not 3 for fetch)
    if len(sys.argv) != 4:
        print("Invalid number of arguments.")
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    station = re.sub(r'\W', '', args.station)
    port = re.sub(r'\D', '', args.port)
    remote_dir_cleaned = re.sub(r'[^a-zA-Z0-9_/]', '', args.remote_dir)
    dir_parts = [part for part in remote_dir_cleaned.split('/') if part]

    date_str, time_str = dir_parts[3], dir_parts[4]
    cam_name = dir_parts[1]

    local_dir = Config.METEOR_DATA_DIR / date_str / time_str / station / cam_name
    setup_logging(local_dir.parent / 'fetch.log')

    if not fetch_data(station, port, args.remote_dir, local_dir):
        sys.exit("Failed to fetch data. Exiting.")

    set_permissions(Config.METEOR_DATA_DIR / date_str)

    date_obj = datetime.datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')

    report_log = local_dir / 'report.log'
    report_script = Config.BASE_HTTP_DIR / 'bin/process.py'
    report_cmd = [sys.executable, str(report_script), str(local_dir / 'event.txt')]

    logging.info(f"--- Running process.py for {local_dir / 'event.txt'} ---")
    with report_log.open('w') as log_file:
       subprocess.call(report_cmd, stdout=log_file, stderr=log_file)

    if not Path(local_dir).exists():
        logging.info("--- Event was discarded ---")
        exit(0)

    if Path(report_log).exists():
        print(report_log.read_text())
    logging.info("--- Finished process.py ---")

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
