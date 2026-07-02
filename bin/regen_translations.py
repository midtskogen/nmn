#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate missing language-specific output files for a single meteor event directory.

Usage (run from any directory):
    ../bin/regen_translations.py 20260629/224825/
    ../bin/regen_translations.py 20260629/224825/ --lang lv
    ../bin/regen_translations.py 20260629/224825/ --lang de
    ../bin/regen_translations.py /absolute/path/to/meteor/20260629/224825/

Generates only files that do not already exist (use --force to regenerate):
  - {lang}_stations.html         (always, from event.txt files)
  - {lang}_brightness.jpg/svg    (per station/cam, from event.txt brightness data)
  - {lang}_tables.html           (if .res file present)
  - {lang}_posvstime.svg/jpg     (if metrack plot data available)
  - {lang}_height.svg/jpg        (if metrack plot data available)
  - {lang}_spd_acc.svg/jpg       (if fbspd data available)
  - {lang}_map.svg/jpg           (if metrack plot data available)
  - {lang}_orbit.svg/jpg         (if orbit data available)

Note: for 'nb' (Norwegian Bokmål) the prefix is empty (no language prefix).
"""

import argparse
import configparser
import datetime
import json
import logging
import os
import re
import sys
from pathlib import Path

# Ensure project modules are importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_SERVER_DIR = _PROJECT_DIR / 'server'
_BIN_DIR = _PROJECT_DIR / 'bin'
_SRC_DIR = _PROJECT_DIR / 'src'
for _p in (_BIN_DIR, _SRC_DIR, _SERVER_DIR, _PROJECT_DIR):
    if _p.exists():
        sys.path.insert(0, str(_p))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import cairosvg
    from PIL import Image, ImageOps
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

from fbspd_merge import readres, calculate_speed_profile, generate_speed_plots
from metrack import calculate_trajectory, generate_plots as generate_metrack_plots
from orbit import calc_azalt, orbit

# ── i18n ─────────────────────────────────────────────────────────────────────

DEFAULT_LANG = 'nb'
SUPPORTED_LANGS = ['nb', 'en', 'de', 'cs', 'fi', 'lv']

_LOC_DIR_CANDIDATES = [
    _BIN_DIR / 'loc',
    _SERVER_DIR / 'loc',
    _PROJECT_DIR / 'bin' / 'loc',
]
LOC_DIR = next((p for p in _LOC_DIR_CANDIDATES if p.is_dir()), _LOC_DIR_CANDIDATES[0])


def lang_prefix(lang: str) -> str:
    """Return the file prefix for a given language code (empty for 'nb')."""
    return '' if lang == DEFAULT_LANG else f'{lang}_'


def decimal_sep_for_lang(lang: str) -> bool:
    """Return True if language uses comma as decimal separator."""
    return lang not in ('en', 'nb')


def load_translations(lang: str) -> dict:
    t = {}
    default_path = LOC_DIR / f'{DEFAULT_LANG}.json'
    lang_path = LOC_DIR / f'{lang}.json'
    if default_path.exists():
        with default_path.open(encoding='utf-8') as f:
            t = json.load(f)
    if lang != DEFAULT_LANG and lang_path.exists():
        with lang_path.open(encoding='utf-8') as f:
            t.update(json.load(f))
    return t


# ── SVG → JPG ────────────────────────────────────────────────────────────────

SVG_DEFAULT_DPI = 300
SVG_MAP_DPI = 80
SVG_ORBIT_DPI = 100


def svg_to_jpg(svg_path: Path, jpg_path: Path, dpi: int = SVG_DEFAULT_DPI):
    if not LIBS_AVAILABLE:
        logging.warning("cairosvg/Pillow not available, skipping SVG→JPG conversion.")
        return
    try:
        png_data = cairosvg.svg2png(url=str(svg_path), dpi=dpi)
        img = Image.open(__import__('io').BytesIO(png_data)).convert('RGB')
        img.save(str(jpg_path), 'JPEG', quality=90)
        logging.info(f"  Converted {svg_path.name} → {jpg_path.name}")
    except Exception as e:
        logging.warning(f"SVG→JPG failed for {svg_path}: {e}")


# ── Station display names ─────────────────────────────────────────────────────

def load_station_display_names() -> dict:
    display_names = {}
    stations_candidates = [
        _BIN_DIR / 'stations.json',
        _SERVER_DIR / 'stations.json',
        _PROJECT_DIR / 'bin' / 'stations.json',
    ]
    for stations_file in stations_candidates:
        if stations_file.exists():
            try:
                with stations_file.open(encoding='utf-8') as f:
                    data = json.load(f)
                for entry in data.values():
                    s = entry.get('station', {})
                    name = s.get('name', '')
                    if name:
                        display_names[name] = s.get('display_name', name.title())
            except Exception:
                pass
            break
    return display_names


_STATION_DISPLAY_NAMES = load_station_display_names()


# ── Brightness plots ──────────────────────────────────────────────────────────

def generate_lv_brightness(event_dir: Path, translations: dict, prefix: str, force: bool = False):
    """Generate {prefix}brightness.svg/jpg for every station/cam under event_dir."""
    for event_file in sorted(event_dir.glob('*/*/event.txt')):
        cam_dir = event_file.parent
        svg_path = cam_dir / f'{prefix}brightness.svg'
        jpg_path = cam_dir / f'{prefix}brightness.jpg'
        if not force and svg_path.exists() and jpg_path.exists():
            logging.info(f"  Skip brightness (exists): {svg_path.relative_to(event_dir)}")
            continue

        cfg = configparser.ConfigParser()
        try:
            cfg.read(event_file)
            timestamps = [float(t) for t in cfg.get('trail', 'timestamps').split()]
            brightness = [float(b) for b in cfg.get('trail', 'brightness').split()]
        except Exception as e:
            logging.warning(f"  Cannot read brightness from {event_file}: {e}")
            continue

        time_axis = [t - timestamps[0] for t in timestamps]
        try:
            plt.figure()
            plt.plot(time_axis, brightness)
            plt.xlabel(translations.get('plot_time_x_label', 'Time [s]'))
            plt.ylabel(translations.get('brightness', 'Brightness'))
            plt.title(translations.get('brightness_plot_title', 'Brightness vs time'))
            plt.tight_layout()
            plt.savefig(str(svg_path))
            plt.savefig(str(jpg_path))
            plt.close()
            logging.info(f"  Generated {svg_path.relative_to(event_dir)}")  # noqa: E501
        except Exception as e:
            plt.close('all')
            logging.warning(f"  Brightness plot failed for {cam_dir}: {e}")


# ── stations.html ─────────────────────────────────────────────────────────────

def generate_lv_stations(event_dir: Path, translations: dict, prefix: str, force: bool = False):
    output_path = event_dir / f'{prefix}stations.html'
    if not force and output_path.exists():
        logging.info(f"  Skip {prefix}stations.html (exists)")
        return

    with output_path.open('w', encoding='utf-8') as f:
        station_files = sorted(event_dir.glob('*/*/event.txt'))
        for event_file in station_files:
            station = event_file.parent.parent.name
            cam = event_file.parent.name
            location = _STATION_DISPLAY_NAMES.get(station, station.title())

            cfg = configparser.ConfigParser()
            try:
                cfg.read(event_file)
                ts_float = float(cfg.get('trail', 'timestamps').split()[0])
            except Exception as e:
                logging.warning(f"  Cannot parse {event_file}: {e}")
                continue

            import pytz
            ts_utc = datetime.datetime.fromtimestamp(ts_float, tz=pytz.utc)
            ts_str = ts_utc.strftime('%Y%m%d%H%M%S')

            code = ''
            try:
                code = cfg.get('station', 'code', fallback='').strip()
            except Exception:
                pass
            if not code:
                obs_txt = event_file.parent / f"{station}-{ts_str}.txt"
                if obs_txt.exists():
                    try:
                        code = obs_txt.read_text().split()[12]
                    except IndexError:
                        pass

            url_base = f'/meteor/{event_dir.parent.name}/{event_dir.name}/{station}'
            station_ts = f"{station}-{ts_str}"

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

            f.write(html_template.format(
                url_base=url_base, station=station, cam=cam, station_ts=station_ts,
                code=code, location=location, default_lang_code=DEFAULT_LANG,
                videos_header=translations.get('videos', 'Videos:'),
                images_header=translations.get('images', 'Images:'),
                text_files_header=translations.get('text_files', 'Text Files:'),
                gnomonic_label=translations.get('gnomonic', 'Gnomonic'),
                gnomonic_with_coords_label=translations.get('gnomonic_with_coords', 'Gnomonic with coordinates'),
                original_label=translations.get('original', 'Original'),
                original_with_coords_label=translations.get('original_with_coords', 'Original with coordinates'),
                gnomonic_uncorrected_with_coords_label=translations.get('gnomonic_uncorrected_with_coords', 'Uncorrected gnomonic with coordinates'),
                gnomonic_with_labels_label=translations.get('gnomonic_with_labels', 'Gnomonic with labels'),
                gnomonic_uncorrected_with_labels_label=translations.get('gnomonic_uncorrected_with_labels', 'Uncorrected gnomonic with labels'),
                original_with_mask_label=translations.get('original_with_mask', 'Original with mask'),
                detection_label=translations.get('detection', 'Detection'),
                observation_label=translations.get('observation', 'Observation'),
                coordinates_label=translations.get('coordinates', 'Coordinates'),
                error_messages_label=translations.get('error_messages', 'Error Messages'),
                log_file_label=translations.get('log_file', 'Log'),
                brightness_label=translations.get('brightness', 'Brightness'),
            ))

    logging.info(f"  Generated {prefix}stations.html")


# ── tables.html (triangulation) ───────────────────────────────────────────────

def generate_lv_tables(event_dir: Path, resdat, orbit_data: dict,
                       placename: str, translations: dict, prefix: str,
                       use_decimal_comma: bool = True, force: bool = False):
    output_path = event_dir / f'{prefix}tables.html'
    if not force and output_path.exists():
        logging.info(f"  Skip {prefix}tables.html (exists)")
        return

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
        if orbit_data.get('entry_speed', 0) > 0:
            table1 += f"    <tr><td>{translations.get('entry_speed', 'Entry speed')}:</td><td> {orbit_data['entry_speed']:.1f} km/s</td></tr>\n"

        if orbit_data.get('valid'):
            ramin = int(orbit_data['ra'] * 24 * 60 / 360)
            shower_name = orbit_data.get('showername') or translations.get('sporadic', 'sporadic')
            table1 += f"""
    <tr><td>{translations.get("radiant_ra", "Radiant R.A.")}:</td><td> {ramin // 60:02d}:{ramin % 60:02d} ({orbit_data['ra']:.1f}°)</td></tr>
    <tr><td>{translations.get("radiant_dec", "Radiant Dec.")}:</td><td> {orbit_data['dec']:.1f}°</td></tr>
    <tr><td>{translations.get("meteor_shower", "Meteor Shower")}:</td><td align=center> {shower_name}</td></tr>
"""
        table1 += "</table>"
        if use_decimal_comma:
            table1 = re.sub(r'(?<=\d)\.(?=\d)', ',', table1)

        f.write('<table><tr><td valign=top>\n')
        f.write(table1)
        f.write('\n</td><td valign=top>\n')

        if orbit_data.get('valid'):
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
            if use_decimal_comma:
                table2 = re.sub(r'(?<=\d)\.(?=\d)', ',', table2)
            f.write(table2)

        f.write('</td></tr></table>')

    logging.info(f"  Generated {prefix}tables.html")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description='Generate missing language-specific output files for a meteor event directory.'
    )
    parser.add_argument('event_dir', type=Path,
                        help='Path to the event directory, e.g. 20260629/224825/')
    parser.add_argument('--lang', default='lv', choices=SUPPORTED_LANGS,
                        help='Language code to generate (default: lv)')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if files already exist')
    args = parser.parse_args()

    LANG = args.lang
    PREFIX = lang_prefix(LANG)
    use_decimal_comma = decimal_sep_for_lang(LANG)

    event_dir = args.event_dir.resolve()
    if not event_dir.is_dir():
        sys.exit(f"Error: {event_dir} is not a directory")

    # Change to event_dir so relative paths in metrack/fbspd work correctly
    os.chdir(event_dir)

    translations = load_translations(LANG)
    logging.info(f"Processing: {event_dir} (lang={LANG}, prefix={PREFIX!r})")

    # 1. Always: {prefix}stations.html
    generate_lv_stations(event_dir, translations, prefix=PREFIX, force=args.force)

    # 2. Always: {prefix}brightness.svg/jpg per station/cam
    generate_lv_brightness(event_dir, translations, prefix=PREFIX, force=args.force)

    # 3. Find .res file — needed for triangulation outputs
    res_files = list(event_dir.glob('obs_*.res'))
    if not res_files:
        logging.info("No .res file found — skipping triangulation outputs.")
        return

    res_path = res_files[0]
    obs_path = res_path.with_suffix('.txt')
    if not obs_path.exists():
        logging.info(f"No matching obs .txt for {res_path.name} — skipping triangulation outputs.")
        return

    # Parse date from obs filename: obs_YYYY-MM-DD_HH:MM:SS.txt
    try:
        stem = res_path.stem  # obs_2026-06-29_22:48:25
        date_str = stem[4:]   # 2026-06-29_22:48:25
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d_%H:%M:%S')
    except ValueError as e:
        logging.warning(f"Cannot parse date from {res_path.name}: {e}")
        return

    # 4. Trajectory plots (posvstime, height, map, spd_acc)
    # Plot_data caches avoid re-running the slow trajectory calculation.
    metrack_cache = event_dir / '_metrack_plot_data.pkl'
    fbspd_cache   = event_dir / '_fbspd_plot_data.pkl'

    metrack_plots_needed = [PREFIX + n for n in ['height.svg', 'map.svg']]
    fbspd_plots_needed   = [PREFIX + n for n in ['posvstime.svg', 'spd_acc.svg']]
    plots_missing = (args.force  # noqa
                     or any(not (event_dir / p).exists() for p in metrack_plots_needed)
                     or any(not (event_dir / p).exists() for p in fbspd_plots_needed))

    orbit_data = {'valid': False}
    metrack_info = None
    metrack_plot_data = None
    fbspd_results = None
    fbspd_plot_data = None

    import pickle

    if plots_missing:
        # Try to load cached plot_data first (written by fetch.py on first run)
        if metrack_cache.exists() and not args.force:
            try:
                with metrack_cache.open('rb') as f:
                    metrack_info, metrack_plot_data = pickle.load(f)
                logging.info("  Loaded metrack plot_data from cache — skipping trajectory calculation.")
            except Exception as e:
                logging.warning(f"  Cache load failed ({e}), recalculating...")
                metrack_info = metrack_plot_data = None

        if metrack_plot_data is None:
            logging.info("  Computing trajectory (no cache found)...")
            try:
                metrack_opts = {
                    'timestamp': date.timestamp(),
                    'optimize': True,
                    'use_ransac': True,
                    'seed': 0,
                    'ransac_threshold': 1.0,
                    'ransac_iterations': 10,
                    'ransac_runs': 100,
                    'debug_ransac': False,
                    'all_in_tolerance': 1.0,
                }
                metrack_info, metrack_plot_data = calculate_trajectory(str(obs_path), **metrack_opts)
                if metrack_plot_data:
                    try:
                        with metrack_cache.open('wb') as f:
                            pickle.dump((metrack_info, metrack_plot_data), f)
                        logging.info("  Saved metrack plot_data cache.")
                    except Exception as e:
                        logging.warning(f"  Could not save metrack cache: {e}")
            except Exception as e:
                logging.warning(f"  Trajectory calculation failed: {e}")

        if metrack_plot_data:
            plot_opts = {
                'doplot': 'save',
                'interactive': True,
                'autoborders': True,
                'azonly': False,
                'mapres': 'i',
            }
            generate_metrack_plots(metrack_info, metrack_plot_data,
                                   plot_opts, translations=translations,
                                   output_prefix=PREFIX)
            logging.info("  Generated metrack plots (posvstime, height, map)")

            for svg_name, dpi in [('posvstime.svg', SVG_DEFAULT_DPI),
                                  ('height.svg', SVG_DEFAULT_DPI),
                                  ('map.svg', SVG_MAP_DPI)]:
                svg_path = event_dir / f'{PREFIX}{svg_name}'
                if svg_path.exists():
                    svg_to_jpg(svg_path, svg_path.with_suffix('.jpg'), dpi)

        # Speed/acceleration plot
        if fbspd_cache.exists() and not args.force:
            try:
                with fbspd_cache.open('rb') as f:
                    cached = pickle.load(f)
                fbspd_results, fbspd_plot_data = cached
                logging.info("  Loaded fbspd plot_data from cache.")
            except Exception as e:
                logging.warning(f"  fbspd cache load failed ({e}), recalculating...")
                fbspd_results = fbspd_plot_data = None

        if fbspd_plot_data is None:
            try:
                centroid_files = [str(p) for p in event_dir.glob('*/*/centroid2.txt')]
                fbspd_results, fbspd_plot_data = calculate_speed_profile(
                    str(res_path), centroid_files, str(obs_path), debug=False, seed=0
                )
                if fbspd_plot_data:
                    try:
                        with fbspd_cache.open('wb') as f:
                            pickle.dump((fbspd_results, fbspd_plot_data), f)
                        logging.info("  Saved fbspd plot_data cache.")
                    except Exception as e:
                        logging.warning(f"  Could not save fbspd cache: {e}")
            except Exception as e:
                logging.warning(f"  Speed calculation failed: {e}")

        if fbspd_plot_data:
            generate_speed_plots(fbspd_plot_data,
                                 translations=translations,
                                 output_prefix=PREFIX)
            for svg_name, dpi in [('posvstime.svg', SVG_DEFAULT_DPI),
                                  ('spd_acc.svg',   SVG_DEFAULT_DPI)]:
                svg_path = event_dir / f'{PREFIX}{svg_name}'
                if svg_path.exists():
                    svg_to_jpg(svg_path, svg_path.with_suffix('.jpg'), dpi)
            logging.info("  Generated speed plots")

        if fbspd_results:
            try:
                entry_speed = fbspd_results['initial_speed']
                resdat = readres(str(res_path))
                az, alt = calc_azalt(resdat.lat1[0], resdat.long1[0], resdat.height[0],
                                     resdat.lat1[1], resdat.long1[1], resdat.height[1])
                orbit_data.update({'entry_speed': entry_speed, 'az': az, 'alt': alt})

                if entry_speed > 9.8:
                    ra, dec, (rp, ecc, inc, lnode, argp, m0, t0), s_name, s_name_sg, valid = orbit(
                        True, entry_speed, 0, str(res_path),
                        date.strftime('%Y-%m-%d'), date.strftime('%H:%M:%S'), doplot=''
                    )
                    shower_name = translations.get('showers', {}).get(s_name, s_name) if s_name else translations.get('sporadic', 'sporadic')
                    orbit_data.update({
                        'ra': ra, 'dec': dec, 'rp': rp, 'ecc': ecc,
                        'inc': inc, 'lnode': lnode, 'argp': argp,
                        'm0': m0, 't0': t0,
                        'showername': shower_name,
                        'valid': valid,
                    })
            except Exception as e:
                logging.warning(f"  Orbit data extraction failed: {e}")

    else:
        logging.info("  All trajectory plots already exist — skipping.")
        # Still need orbit_data for tables.html — read from .res
        try:
            resdat = readres(str(res_path))
            az, alt = calc_azalt(resdat.lat1[0], resdat.long1[0], resdat.height[0],
                                 resdat.lat1[1], resdat.long1[1], resdat.height[1])
            orbit_data.update({'az': az, 'alt': alt, 'entry_speed': 0})
        except Exception as e:
            logging.warning(f"  Could not read .res for tables: {e}")

    # 5. Orbit plot
    orbit_svg = event_dir / f'{PREFIX}orbit.svg'  # noqa
    if orbit_data.get('valid') and (args.force or not orbit_svg.exists()):
        logging.info("  Generating orbit plot...")
        try:
            orbit(True, orbit_data['entry_speed'], 0, str(res_path),
                  date.strftime('%Y-%m-%d'), date.strftime('%H:%M:%S'), 'save',
                  interactive=True, translations=translations, output_prefix=PREFIX)
            if orbit_svg.exists():
                svg_to_jpg(orbit_svg, orbit_svg.with_suffix('.jpg'), SVG_ORBIT_DPI)
                logging.info("  Generated orbit plot")
        except Exception as e:
            logging.warning(f"  Orbit plot failed: {e}")

    # 6. tables.html — needs resdat
    tables_needed = args.force or not (event_dir / f'{PREFIX}tables.html').exists()
    if tables_needed and orbit_data.get('az') is not None:
        try:
            resdat = readres(str(res_path))
            placename_file = event_dir / 'location.txt'
            placename = placename_file.read_text(encoding='utf-8').strip() if placename_file.exists() else ''
            generate_lv_tables(event_dir, resdat, orbit_data, placename, translations,
                               prefix=PREFIX, use_decimal_comma=use_decimal_comma, force=args.force)
        except Exception as e:
            logging.warning(f"  {PREFIX}tables.html generation failed: {e}")

    logging.info("Done.")


if __name__ == '__main__':
    main()
