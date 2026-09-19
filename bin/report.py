#!/usr/bin/env python3
"""
Processes a meteor event, classifies it using a CNN, and reports it to the
Norsk Meteornettverk server if it meets the probability threshold.

Usage:
    python report.py <event.txt>
"""

import base64
import configparser
import datetime
import calendar
import fcntl
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

# Third-party libraries

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

try:
    import ephem
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dateutil import parser
except ImportError as e:
    print(f"Error: Missing required library. {e}", file=sys.stderr)
    print("Please install it using: pip install pyephem matplotlib python-dateutil", file=sys.stderr)
    sys.exit(1)


# --- Constants ---
METEOR_PROBABILITY_THRESHOLD = 0.5
SSH_TUNNEL_CONFIG_PATH = '/etc/default/ssh_tunnel'
REMOTE_REPORT_URL = "https://norskmeteornettverk.no/ssh/report.php"
REMOTE_UPLOAD_URL = "https://norskmeteornettverk.no/ssh/upload.php"
# Same exclude set as the server's rsync pull in server/fetch.py — push must
# deliver exactly the files a pull would have fetched.
PUSH_EXCLUDES = [
    'frame-*', '*.pkl', '*.php', '*.pht*', '*.phar', '*.cgi', '*.pl',
    '*.shtml', '*.py', '*.pyc', '*.pyo', '*.sh', '.htaccess', '.htpasswd',
]
# Persistent report queue: a ping that fails goes here and is retried by a
# detached drainer every REPORT_RETRY_INTERVAL seconds until the server
# acknowledges it — server downtime must not lose detections.
REPORT_QUEUE_PATH = Path.home() / '.nmn_report_queue'
REPORT_RETRY_INTERVAL = 600
# Assume processing scripts are in the user's bin directory
METEORCROP_PATH = Path.home() / "bin" / "meteorcrop.py"
PREDICT_PATH = Path.home() / "bin" / "predict.py"

# --- i18n ---
SUPPORTED_LANGS = ['nb', 'en', 'de', 'cs', 'fi']
DEFAULT_LANG = 'nb'
LOC_DIR = Path(__file__).resolve().parent.parent / 'server' / 'loc'


def load_translations(lang_code: str) -> dict:
    """Loads translation dict for lang_code, falling back to DEFAULT_LANG."""
    translations: dict = {}
    default_path = LOC_DIR / f"{DEFAULT_LANG}.json"
    if default_path.exists():
        with default_path.open('r', encoding='utf-8') as f:
            translations = json.load(f)
    if lang_code != DEFAULT_LANG:
        lang_path = LOC_DIR / f"{lang_code}.json"
        if lang_path.exists():
            with lang_path.open('r', encoding='utf-8') as f:
                translations.update(json.load(f))
    return translations


def load_config(event_file_path: Path) -> configparser.ConfigParser:
    """Loads configuration from the event file and system/user config files."""
    config = configparser.ConfigParser()
    config.read(event_file_path)
    station_config_paths = ['/etc/meteor.cfg', Path.home() / 'meteor.cfg']
    config.read(station_config_paths)
    return config


def _fs_safe(value: str) -> str:
    """Return a filesystem/URL-path-safe token from a config-supplied string.

    The station name comes from meteor.cfg and is used in local filenames and
    in the remote upload path — it must not contain separators or dots that
    could traverse directories.
    """
    import re
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).lstrip('.-')
    return cleaned or 'unnamed'


def haversine_arc(az1: float, alt1: float, az2: float, alt2: float) -> float:
    """Calculates the angular separation (arc) in degrees between two points."""
    x1, x2 = math.radians(az1), math.radians(az2)
    y1, y2 = math.radians(alt1), math.radians(alt2)
    delta_lat = (y2 - y1) / 2
    delta_lon = (x2 - x1) / 2
    a = math.sin(delta_lat)**2 + math.cos(y1) * math.cos(y2) * math.sin(delta_lon)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return math.degrees(c)


def get_sun_position(config: configparser.ConfigParser, start_dt: datetime.datetime) -> Tuple[float, float]:
    """Computes the sun's altitude and azimuth for the event time."""
    observer = ephem.Observer()
    observer.lat = config.get('astronomy', 'latitude')
    observer.lon = config.get('astronomy', 'longitude')
    observer.elevation = config.getfloat('astronomy', 'elevation')
    observer.date = start_dt
    sun = ephem.Sun()
    sun.compute(observer)
    return math.degrees(sun.alt), math.degrees(sun.az)


@contextmanager
def acquire_lock():
    """Acquires a system-wide lock to prevent multiple instances from running."""
    lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    lock_name = f'\0{Path(__file__).name}'
    try:
        while True:
            try:
                lock_socket.bind(lock_name)
                print("Lock acquired.")
                yield
                break
            except socket.error:
                print("Another instance is running. Waiting...")
                time.sleep(5)
    finally:
        lock_socket.close()
        print("Lock released.")


def _report_token() -> str:
    """Optional shared secret: the server validates inputs strictly; a token
    lets it distinguish real stations."""
    try:
        return Path('/etc/default/nmn_report_token').read_text().strip() or \
               (Path.home() / '.nmn_report_token').read_text().strip()
    except OSError:
        return ''


def _report_url(station_name: str, port: str, event_dir) -> str:
    return REMOTE_REPORT_URL + '?' + urllib.parse.urlencode({
        'station': station_name, 'port': port, 'dir': str(event_dir),
        'token': _report_token()})


def _ping(url: str, station_name: str = '', event_dir=None) -> Tuple[str, str]:
    """Single report ping. Returns (http_code, body); '000' on failure.

    When station_name/event_dir are given the ping is signed like an
    upload: the signature covers "<station>\t<dir>", which is all the
    endpoint needs to authenticate the request.
    """
    headers = []
    if station_name and event_dir is not None:
        norm_dir = '/' + str(event_dir).strip('/')
        fd, payload_path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as pf:
                pf.write(f'{station_name}\t{norm_dir}'.encode())
            sig = _sign_file(Path(payload_path))
        finally:
            Path(payload_path).unlink(missing_ok=True)
            Path(payload_path + '.sig').unlink(missing_ok=True)
        if sig:
            headers = ['-H', f'X-NMN-Sig: {sig}']
    try:
        r = subprocess.run(
            ['curl', '-s', '-w', '\n%{http_code}', '--max-time', '30']
            + headers + [url],
            capture_output=True, text=True, timeout=60)
        body, _, code = r.stdout.rpartition('\n')
        return code, body.strip()
    except Exception as e:
        return '000', str(e)


def _sign_file(path: Path) -> str:
    """ssh-keygen sign a file with the station's tunnel key (base64 sig).

    Stations share one keypair, so the signature proves the upload came
    from a station without transmitting any secret.
    """
    for name in ('id_rsa', 'id_ed25519', 'id_ecdsa', 'id_dsa'):
        key = Path.home() / '.ssh' / name
        if not key.is_file():
            continue
        try:
            subprocess.run(
                ['ssh-keygen', '-Y', 'sign', '-f', str(key),
                 '-n', 'nmn-upload', str(path)],
                check=True, capture_output=True, timeout=30)
            return base64.b64encode(
                Path(str(path) + '.sig').read_bytes()).decode()
        except Exception:
            continue
    return ''


def _push_event(station_name: str, event_dir) -> Tuple[str, str]:
    """Push the event dir to the server as a tar.gz over HTTPS.

    The payload "<dir>\n<tar>" is signed with the station ssh key; the
    server verifies it against its allowed_signers before unpacking, so
    nothing but a real station can hand data to the event tree.
    Returns (http_code, body); '000' on transport failure.
    """
    norm_dir = '/' + str(event_dir).strip('/')
    url = REMOTE_UPLOAD_URL + '?' + urllib.parse.urlencode({
        'station': station_name, 'dir': norm_dir})
    tar_path = payload_path = None
    try:
        fd, tar_path = tempfile.mkstemp(suffix='.tar.gz')
        os.close(fd)
        tar_cmd = ['tar', 'czf', tar_path, '-C', str(event_dir)]
        for pat in PUSH_EXCLUDES:
            tar_cmd += ['--exclude', pat]
        tar_cmd.append('.')
        subprocess.run(tar_cmd, check=True, timeout=300,
                       stderr=subprocess.DEVNULL)
        fd, payload_path = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as pf, open(tar_path, 'rb') as tf:
            pf.write(norm_dir.encode() + b'\n')
            shutil.copyfileobj(tf, pf)
        headers = ['-H', 'Content-Type: application/octet-stream',
                   '-H', f'X-NMN-Token: {_report_token()}']
        sig = _sign_file(Path(payload_path))
        if sig:
            headers += ['-H', f'X-NMN-Sig: {sig}']
        r = subprocess.run(
            ['curl', '-s', '-w', '\n%{http_code}', '--max-time', '600',
             '-X', 'POST', '--data-binary', '@' + tar_path] + headers + [url],
            capture_output=True, text=True, timeout=660)
        body, _, code = r.stdout.rpartition('\n')
        return code, body.strip()
    except Exception as e:
        return '000', str(e)
    finally:
        for p in (tar_path, payload_path):
            if p:
                Path(p).unlink(missing_ok=True)
                sig_p = Path(str(p) + '.sig')
                sig_p.unlink(missing_ok=True)


def _deliver_report(station_name: str, port: str, event_dir) -> str:
    """Deliver a report: push the data over HTTPS, fall back to a pull ping.

    Returns the final http code.  Either transport delivers the event:
    push hands the files to upload.php directly; the ping tells the server
    to pull the dir over the ssh tunnel.
    """
    code, _ = _push_event(station_name, event_dir)
    if code.startswith('2'):
        return code
    ping_code, _ = _ping(_report_url(station_name, port, event_dir),
                         station_name, event_dir)
    return code if code != '000' else ping_code


def _enqueue_report(station_name: str, port: str, event_dir) -> None:
    """Persist a failed report so it is retried until the server acks it."""
    entry = f'{station_name}\t{port}\t{event_dir}\n'
    try:
        with open(REPORT_QUEUE_PATH, 'a+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            if entry not in f.readlines():
                f.write(entry)
    except OSError as e:
        print(f"WARNING: could not queue report for {event_dir}: {e}",
              file=sys.stderr)


def _drain_queue() -> None:
    """One pass over the retry queue: ping each entry once, drop successes."""
    try:
        with open(REPORT_QUEUE_PATH, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            entries = f.readlines()
            remaining = []
            for line in entries:
                parts = line.rstrip('\n').split('\t')
                if len(parts) != 3:
                    continue
                st, pt, dr = parts
                code = _deliver_report(st, pt, dr)
                if code.startswith('2'):
                    print(f"Queued report delivered: {dr}")
                elif not code.startswith('4'):
                    remaining.append(line)  # transient failure: retry later
            if remaining:
                f.seek(0)
                f.truncate()
                f.writelines(remaining)
            else:
                REPORT_QUEUE_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def _spawn_queue_drainer() -> None:
    """Start a detached retry process if one isn't already running."""
    try:
        subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), '--drain-queue'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as e:
        print(f"WARNING: could not spawn report queue drainer: {e}",
              file=sys.stderr)


def drain_report_queue() -> None:
    """Detached drainer: keeps retrying queued reports until all are acked.

    Runs under its own singleton (separate from the main report lock) so it
    neither blocks nor multiplies.  A pass runs every REPORT_RETRY_INTERVAL
    seconds — infrequent enough to not hammer an unavailable server.
    """
    lock_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        lock_socket.bind('\0' + Path(__file__).name + '-drain')
    except socket.error:
        return  # another drainer is already running
    try:
        while REPORT_QUEUE_PATH.exists():
            _drain_queue()
            if not REPORT_QUEUE_PATH.exists():
                break
            time.sleep(REPORT_RETRY_INTERVAL)
    finally:
        lock_socket.close()


def run_video_creation(
    config: configparser.ConfigParser,
    event_file_path: Path,
    start_timestamp: float,
    video_name: str,
    nologos: bool = False,
    credit: str = "",
    creditpos: str = "lower-right",
    creditsize: int = 24,
    creditfont: str = "Helvetica",
    logo_sequence: Optional[list] = None,
) -> Optional[list]:
    """Runs makevideos.py script and returns its output."""
    duration = math.ceil(config.getfloat('trail', 'duration'))
    # Use the event directory itself as the video directory
    video_dir = event_file_path.parents[3]
    event_dir = event_file_path.parent
    makevideos_script = Path.home() / 'bin' / 'makevideos.py'
    command = [
        str(makevideos_script), "--client",
        "--video-dir", str(video_dir),
        "--start", str(int(round(start_timestamp))),
        "--length", str(duration),
        video_name
    ]
    if logo_sequence:
        command[1:1] = list(logo_sequence)
    if credit:
        command.insert(1, "--credit")
        command.insert(2, credit)
        command.insert(3, "--creditpos")
        command.insert(4, creditpos)
        command.insert(5, "--creditsize")
        command.insert(6, str(creditsize))
        command.insert(7, "--creditfont")
        command.insert(8, creditfont)
    elif nologos:
        command.insert(1, "--nologos")
    print(f"Running command: {' '.join(command)}")
    try:
        proc = subprocess.run(command, cwd=event_dir, capture_output=True, text=True, check=True)
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith("AZALT:"):
                return line.split()[1:]
        print(f"Error: makevideos.py did not output an AZALT: line.\nStdout: {proc.stdout}", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running makevideos.py: {e}\nStderr: {e.stderr}", file=sys.stderr)
        return None


def generate_reports(config: configparser.ConfigParser, video_output: list, event_dir: Path, video_name: str, start_timestamp: float):
    """Generates metrack, centroid, light data files and a brightness plot."""
    original_arc = config.getfloat('trail', 'arc')
    original_duration = config.getfloat('trail', 'duration')
    start_az, start_alt = float(video_output[0]), float(video_output[1])
    end_az, end_alt = float(video_output[2]), float(video_output[3])
    recalibrated_arc = haversine_arc(start_az, start_alt, end_az, end_alt)
    duration = original_duration * (recalibrated_arc / original_arc) if original_arc > 0 else 0
    lon = config.get('astronomy', 'longitude')
    lat = config.get('astronomy', 'latitude')
    elevation = config.get('astronomy', 'elevation')
    station_code = config.get('station', 'code')
    metrack_data = [
        lon, lat, str(start_az), str(end_az), str(start_alt), str(end_alt),
        '1', str(round(duration, 2)), '400', '128', '255', '255',
        station_code, str(round(start_timestamp, 2)), elevation
    ]
    (event_dir / f"{video_name}.txt").write_text(' '.join(metrack_data) + '\n')
    print(f"Generated {video_name}.txt")

    timestamps = [float(t) for t in config.get('trail', 'timestamps').split()]
    coordinates = config.get('trail', 'coordinates').split()
    brightness = config.get('trail', 'brightness').split()
    frame_brightness = config.get('trail', 'frame_brightness').split()
    size = config.get('trail', 'size').split()

    n = len(timestamps)
    for name, lst in (('coordinates', coordinates), ('brightness', brightness),
                      ('frame_brightness', frame_brightness), ('size', size)):
        if len(lst) != n:
            print(f"Error: trail.{name} has {len(lst)} entries but trail.timestamps has {n}; "
                  "skipping centroid/light report generation", file=sys.stderr)
            return

    with open(event_dir / 'centroid.txt', 'w') as f_centroid, \
         open(event_dir / 'light.txt', 'w') as f_light:
        for i, t in enumerate(timestamps):
            time_offset = round(t - timestamps[0], 2)
            az, alt = coordinates[i].split(',')
            gm_time = time.gmtime(t)
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', gm_time)
            ms = f"{round(t - math.floor(t), 2):.2f}"[2:]
            f_centroid.write(f"{i} {time_offset} {alt} {az} 1.0 {station_code} {time_str}.{ms} UTC\n")
            f_light.write(f"{time_offset} {brightness[i]} {size[i]} {frame_brightness[i]}\n")
    print("Generated centroid.txt and light.txt")

    time_points = [t - timestamps[0] for t in timestamps]
    brightness_floats = list(map(float, brightness))
    for lang in SUPPORTED_LANGS:
        tr = load_translations(lang)
        prefix = '' if lang == DEFAULT_LANG else f'{lang}_'
        fig, ax = plt.subplots()
        ax.plot(time_points, brightness_floats)
        ax.set_xlabel(tr.get('plot_time_x_label', 'Time [s]'))
        ax.set_ylabel(tr.get('brightness', 'Brightness'))
        ax.set_title(tr.get('brightness_plot_title', 'Brightness vs time'))
        fig.savefig(event_dir / f'{prefix}brightness.svg')
        fig.savefig(event_dir / f'{prefix}brightness.jpg')
        plt.close(fig)
    print("Generated brightness plots (all languages).")


def get_meteor_probability(event_dir: Path) -> float:
    """
    Runs meteorcrop and predict scripts to determine the meteor probability.
    Returns the probability score as a float.
    """
    print("\n--- Starting Meteor Classification ---")
    try:
        print(f"Running meteorcrop in '{event_dir}'...")
        crop_cmd = [sys.executable, str(METEORCROP_PATH), "--mode", "both", str(event_dir)]
        subprocess.run(crop_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running meteorcrop.py: {e}\nStderr: {e.stderr}", file=sys.stderr)
        return 0.0

    fireball_jpg_path = event_dir / "fireball_orig.jpg"
    if not fireball_jpg_path.is_file():
        print(f"Error: meteorcrop did not produce '{fireball_jpg_path.name}'.", file=sys.stderr)
        return 0.0

    try:
        print(f"Running predict on '{fireball_jpg_path.name}'...")
        predict_cmd = [sys.executable, str(PREDICT_PATH), str(fireball_jpg_path)]
        result = subprocess.run(predict_cmd, check=True, capture_output=True, text=True)
        
        output_line = result.stdout.strip()
        probability_str = output_line.split(' ')[-1]
        probability = float(probability_str)
        print(f"Meteor probability: {probability:.4f}")
        return probability
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        print(f"Error getting probability from predict.py: {e}", file=sys.stderr)
        if isinstance(e, subprocess.CalledProcessError):
            print(f"Stderr: {e.stderr}", file=sys.stderr)
        return 0.0


def update_event_file(config: configparser.ConfigParser, video_output: list, event_file_path: Path, probability: float, start_dt: datetime.datetime):
    """Adds a [summary] section to the event file with key results."""
    original_duration = config.getfloat('trail', 'duration')
    original_arc = config.getfloat('trail', 'arc')
    start_az, start_alt = float(video_output[0]), float(video_output[1])
    end_az, end_alt = float(video_output[2]), float(video_output[3])
    recalibrated_arc = haversine_arc(start_az, start_alt, end_az, end_alt)
    duration = original_duration * (recalibrated_arc / original_arc) if original_arc > 0 else 0
    sun_alt, _ = get_sun_position(config, start_dt)

    if not config.has_section('summary'):
        config.add_section('summary')

    config.set('summary', 'latitude', config.get('astronomy', 'latitude'))
    config.set('summary', 'longitude', config.get('astronomy', 'longitude'))
    config.set('summary', 'elevation', config.get('astronomy', 'elevation'))
    config.set('summary', 'timestamp', config.get('video', 'start'))
    config.set('summary', 'startpos', f"{start_az} {start_alt}")
    config.set('summary', 'endpos', f"{end_az} {end_alt}")
    config.set('summary', 'duration', str(round(duration, 2)))
    config.set('summary', 'sunalt', str(round(sun_alt, 1)))
    config.set('summary', 'recalibrated', "0" if sun_alt > -10 else "1")
    config.set('summary', 'meteor_probability', f"{probability:.6f}")

    with open(event_file_path, 'w') as configfile:
        config.write(configfile)
    print(f"Updated event file with summary and probability: {event_file_path.name}")


def upload_results(config: configparser.ConfigParser, event_dir: Path):
    """Uploads the event directory to the server and pings the report URL."""
    try:
        with open(SSH_TUNNEL_CONFIG_PATH) as f:
            port = next((line.strip().split('=')[1] for line in f if line.startswith('PORT=')), '0')
    except FileNotFoundError:
        port = '0'

    station_name = _fs_safe(config.get('station', 'name'))
    if int(port) == 0:
        print("SSH tunnel port is 0, using lftp to upload...")
        remote_path = f"upload/meteor/{station_name}/"
        # lftp -e interprets ';', quotes etc. as its own command syntax —
        # quote both paths so metacharacters can't inject lftp commands.
        lftp_cmd = 'mirror -R {} {}'.format(
            "'" + str(event_dir).replace("'", "'\\''") + "'",
            "'" + remote_path.replace("'", "'\\''") + "'")
        command = ['lftp', '-e', lftp_cmd, 'norskmeteornettverk.no']
        subprocess.run(command)
    else:
        print(f"SSH tunnel is active on port {port}. Not using lftp.")

    # Preferred path: push the event dir straight to the server over HTTPS
    # (the only outbound channel stations have).  On failure fall back to
    # the pull ping, which asks the server to rsync the dir over the ssh
    # tunnel.  If neither reaches the server the report is queued and the
    # detached drainer keeps retrying until the server acknowledges it.
    code, body = _push_event(station_name, event_dir)
    print(f"Event push: HTTP {code} {body}")
    if not code.startswith('2'):
        if code != '404':  # 404 = upload.php not deployed yet; go straight to ping
            print(f"Push failed (HTTP {code}), falling back to pull ping")
        url = _report_url(station_name, port, event_dir)
        for attempt in range(3):
            code, body = _ping(url, station_name, event_dir)
            print(f"Report ping attempt {attempt + 1}/3: HTTP {code} {body}")
            if code.startswith('2') or code.startswith('4'):
                break
            time.sleep(30)
    if code.startswith('2'):
        _drain_queue()  # server is reachable: flush any queued reports
    elif code.startswith('4'):
        print(f"WARNING: report denied (HTTP {code}): {body}",
              file=sys.stderr)
    else:
        print(f"WARNING: server unreachable (HTTP {code}); queued for retry",
              file=sys.stderr)
        _enqueue_report(station_name, port, event_dir)
        _spawn_queue_drainer()
    print("Upload and reporting complete.")


def main():
    """Main execution function."""
    nologos = False
    credit = ""
    creditpos = "lower-right"
    creditsize = 24
    creditfont = "Helvetica"
    logo_sequence = []
    argv = sys.argv[1:]
    if '--drain-queue' in argv:
        drain_report_queue()
        sys.exit(0)
    usage = f"Usage: {sys.argv[0]} [--nologos] [--credit <string> [--creditpos <pos>] [--creditsize <size>] [--creditfont <font>]] [--logo <file> [--logopos <pos>]]... <event.txt>"

    remaining = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--nologos":
            nologos = True
            i += 1
            continue

        if a == "--credit":
            if i + 1 >= len(argv):
                print(usage)
                sys.exit(1)
            credit = argv[i + 1]
            i += 2
            continue

        if a == "--creditpos":
            if i + 1 >= len(argv):
                print(usage)
                sys.exit(1)
            creditpos = argv[i + 1]
            i += 2
            continue

        if a == "--creditsize":
            if i + 1 >= len(argv):
                print(usage)
                sys.exit(1)
            try:
                creditsize = int(argv[i + 1])
            except ValueError:
                print(f"Invalid --creditsize value: {argv[i + 1]}")
                sys.exit(1)
            i += 2
            continue

        if a == "--creditfont":
            if i + 1 >= len(argv):
                print(usage)
                sys.exit(1)
            creditfont = argv[i + 1]
            i += 2
            continue

        if a == "--logo":
            if i + 1 >= len(argv):
                print(usage)
                sys.exit(1)
            logo_sequence.extend(["--logo", argv[i + 1]])
            i += 2
            continue

        if a == "--logopos":
            if i + 1 >= len(argv):
                print(usage)
                sys.exit(1)
            logo_sequence.extend(["--logopos", argv[i + 1]])
            i += 2
            continue

        remaining.append(a)
        i += 1

    argv = remaining

    if len(argv) != 1:
        print(usage)
        sys.exit(1)

    event_file_path = Path(argv[0])
    if not event_file_path.is_file():
        print(f"Error: File not found at {event_file_path}", file=sys.stderr)
        sys.exit(1)

    event_dir = event_file_path.parent
    config = load_config(event_file_path)

    # --- PARSE THE DATE ONCE AND REUSE ---
    video_start_str = config.get('video', 'start').split(' (')[0]
    start_dt = parser.parse(video_start_str)
    start_timestamp = calendar.timegm(start_dt.utctimetuple()) + start_dt.microsecond / 1_000_000.0
    # ---

    station_name = _fs_safe(config.get('station', 'name'))
    event_timestamp_str = start_dt.strftime('%Y%m%d%H%M%S')
    video_name = f"{station_name}-{event_timestamp_str}"

    with acquire_lock():
        video_output = run_video_creation(
            config,
            event_file_path,
            start_timestamp,
            video_name,
            nologos=nologos,
            credit=credit,
            creditpos=creditpos,
            creditsize=creditsize,
            creditfont=creditfont,
            logo_sequence=logo_sequence,
        )
        if not video_output:
            sys.exit(1)

        generate_reports(config, video_output, event_dir, video_name, start_timestamp)
        
        probability = get_meteor_probability(event_dir)
        
        update_event_file(config, video_output, event_file_path, probability, start_dt)

        threshold = config.getfloat('classification', 'threshold', fallback=METEOR_PROBABILITY_THRESHOLD)

        if config.getfloat('trail', 'manual', fallback=0) > 0:
            print(f"Manually reduced. Reporting to server.")
            upload_results(config, event_dir)
        elif probability >= threshold:
            print(f"Probability ({probability:.4f}) is >= {threshold}. Reporting to server.")
            upload_results(config, event_dir)
        else:
            print(f"Probability ({probability:.4f}) is < {threshold}. Not reporting to server.")


if __name__ == "__main__":
    main()
