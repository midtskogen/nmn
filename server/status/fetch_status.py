#!/usr/bin/env python3
"""
Fetch AS7 health status from all stations via SSH.
Stores logs in SQLite database for 1-month history.
"""

import os
import sys
import json
import re
import subprocess
import sqlite3
import time
import smtplib
from email.mime.text import MIMEText
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Configuration
STATUS_DIR = Path(__file__).parent.resolve()
DB_PATH = STATUS_DIR / "status.db"
CAMERAS_JSON = STATUS_DIR.parent / "nmn" / "server" / "data" / "cameras.json"
LOG_FILE = "/var/log/as7health.log"
MAX_HISTORY_PER_STATION = 186  # 6 fetches/day × 31 days = ~1 month
SSH_TIMEOUT = 30
SSH_TIMEOUT_BACKUP = 5
SSH_USER = "meteor"

# RaspberryShake API
FDSN_STATION_URL = "https://data.raspberryshake.org/fdsnws/station/1/query"
FDSN_DATA_URL = "https://data.raspberryshake.org/fdsnws/dataselect/1/query"


def init_database():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stations (
            id TEXT PRIMARY KEY,
            name TEXT,
            display_name TEXT,
            code TEXT,
            latitude REAL,
            longitude REAL,
            elevation REAL,
            country TEXT,
            geophone_id TEXT,
            infrasound_id TEXT
        )
    ''')

    # Migration: add display_name column if the database was created before it existed.
    try:
        cursor.execute("ALTER TABLE stations ADD COLUMN display_name TEXT")
    except sqlite3.OperationalError:
        pass
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS status_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            log_content TEXT,
            successes INTEGER DEFAULT 0,
            warnings INTEGER DEFAULT 0,
            failures INTEGER DEFAULT 0,
            total INTEGER DEFAULT 0,
            issues_json TEXT,
            fetch_success BOOLEAN DEFAULT 0,
            error_message TEXT,
            backup_pc_ok BOOLEAN,
            geophone_ok BOOLEAN,
            infrasound_ok BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (station_id) REFERENCES stations(id)
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_logs_station_time 
        ON status_logs(station_id, timestamp DESC)
    ''')
    
    conn.commit()
    conn.close()


def check_raspberryshake(geophone_id=None, infrasound_id=None, minutes=1440):
    """Check RaspberryShake station data availability.
    
    Returns tuple (geophone_ok, infrasound_ok) where each is:
    - True: data received in last 'minutes' (default 24 hours to account for FDSN upload delays)
    - False: no recent data
    - None: station not configured
    """
    geophone_ok = None
    infrasound_ok = None
    
    # Calculate time range - shift window 2 hours back to account for upload delays
    end_time = datetime.now(timezone.utc) - timedelta(hours=2)
    start_time = end_time - timedelta(minutes=minutes)
    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
    end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Check geophone (EHZ channel)
    if geophone_id:
        try:
            params = {
                "net": "AM",
                "sta": geophone_id,
                "cha": "EHZ",
                "starttime": start_str,
                "endtime": end_str,
                "format": "miniseed"
            }
            resp = requests.get(FDSN_DATA_URL, params=params, timeout=15)
            # Debug logging
            print(f"  [RShake Debug] {geophone_id} EHZ: status={resp.status_code}, size={len(resp.content)} bytes", flush=True)
            print(f"  [RShake Debug] URL: {resp.url}", flush=True)
            geophone_ok = resp.status_code == 200 and len(resp.content) > 100
        except Exception as e:
            print(f"  [RShake Debug] {geophone_id} EHZ: ERROR {e}", flush=True)
            geophone_ok = False
    
    # Small delay between requests to avoid rate limiting
    time.sleep(0.5)
    
    # Check infrasound (HDF channel)
    if infrasound_id:
        try:
            params = {
                "net": "AM",
                "sta": infrasound_id,
                "cha": "HDF",
                "starttime": start_str,
                "endtime": end_str,
                "format": "miniseed"
            }
            resp = requests.get(FDSN_DATA_URL, params=params, timeout=15)
            # Debug logging
            print(f"  [RShake Debug] {infrasound_id} HDF: status={resp.status_code}, size={len(resp.content)} bytes", flush=True)
            print(f"  [RShake Debug] URL: {resp.url}", flush=True)
            infrasound_ok = resp.status_code == 200 and len(resp.content) > 100
        except Exception as e:
            print(f"  [RShake Debug] {infrasound_id} HDF: ERROR {e}", flush=True)
            infrasound_ok = False
    
    return (geophone_ok, infrasound_ok)


def sync_stations():
    """Sync stations from cameras.json to database."""
    with open(CAMERAS_JSON, 'r') as f:
        cameras = json.load(f)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for station_id, data in cameras.items():
        if not station_id.startswith("ams"):
            continue
        
        station = data.get('station', {})
        astro = data.get('astronomy', {})
        
        # Get infrasound_id - same as geophone_id for RaspberryShake
        # Only store if it exists and is not empty
        infrasound_id = station.get('infrasound_id')
        if infrasound_id and infrasound_id.strip():
            geophone_id = infrasound_id  # Same station code for both sensors
        else:
            geophone_id = None
            infrasound_id = None
        
        cursor.execute('''
            INSERT OR REPLACE INTO stations 
            (id, name, display_name, code, latitude, longitude, elevation, country, geophone_id, infrasound_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            station_id, station.get('name'), station.get('display_name'), station.get('code'),
            astro.get('latitude'), astro.get('longitude'), astro.get('elevation'),
            data.get('country'),
            geophone_id,
            infrasound_id
        ))
    
    conn.commit()
    conn.close()
    return len([s for s in cameras if s.startswith("ams")])


def parse_log_content(content):
    """Parse as7health.log content to extract status info."""
    lines = content.split('\n')
    
    total = 0
    successes = 0
    warnings = 0
    failures = 0
    issues = []
    timestamp = None
    
    in_issues_section = False
    
    for line in lines:
        # Extract timestamp from log
        if line.startswith('## '):
            timestamp = line.replace('## ', '').strip()
        
        # Summary section (case-insensitive matching)
        if re.search(r'Total checks?:', line, re.IGNORECASE):
            match = re.search(r'Total checks?:\s*(\d+)', line, re.IGNORECASE)
            if match:
                total = int(match.group(1))
        elif re.search(r'Successful:', line, re.IGNORECASE):
            match = re.search(r'Successful:\s*(\d+)', line, re.IGNORECASE)
            if match:
                successes = int(match.group(1))
        elif 'Warnings:' in line:
            match = re.search(r'Warnings:\s*(\d+)', line)
            if match:
                warnings = int(match.group(1))
        elif 'Failures:' in line:
            match = re.search(r'Failures:\s*(\d+)', line)
            if match:
                failures = int(match.group(1))
        
        # Issues section
        if 'Detailed Issues and Recommendations' in line:
            in_issues_section = True
            continue
        
        if in_issues_section:
            if line.startswith('[') and ('WARN' in line or 'FAIL' in line):
                issue_type = 'fail' if '[ FAIL ]' in line else 'warning'
                # Clean up the line
                clean = re.sub(r'\x1b\[[0-9;]*m', '', line)  # Remove ANSI codes
                clean = re.sub(r'\[ (WARN|FAIL|INFO) \]', '', clean).strip()
                if clean and len(clean) > 3:
                    issues.append({'type': issue_type, 'text': clean})
    
    return {
        'timestamp': timestamp,
        'total': total,
        'successes': successes,
        'warnings': warnings,
        'failures': failures,
        'issues': issues[:5]  # Limit to top 5
    }


def check_backup_pc(station_id):
    """Check if backup PC is reachable."""
    backup_host = f"{station_id}b"
    try:
        cmd = [
            "ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            f"{SSH_USER}@{backup_host}", "echo ok"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=SSH_TIMEOUT_BACKUP)
        return result.returncode == 0 and "ok" in result.stdout
    except:
        return False


def fetch_station(station_id):
    """Fetch status from a single station."""
    host = station_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get RaspberryShake IDs from database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT geophone_id, infrasound_id FROM stations WHERE id = ?', (station_id,))
    row = cursor.fetchone()
    conn.close()
    
    geophone_id = row[0] if row else None
    infrasound_id = row[1] if row else None
    
    try:
        # Use ssh cat instead of scp for reliable stdout capture
        cmd = [
            "ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
            f"{SSH_USER}@{host}", f"cat {LOG_FILE}"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=SSH_TIMEOUT)
        
        if result.returncode != 0:
            cmd[3] = f"root@{host}"
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=SSH_TIMEOUT)
        
        backup_pc_ok = check_backup_pc(station_id)
        geophone_ok, infrasound_ok = check_raspberryshake(geophone_id, infrasound_id)
        
        if result.returncode == 0:
            log_content = result.stdout
            parsed = parse_log_content(log_content)
            
            store_log(station_id, timestamp, log_content, parsed, True, None, backup_pc_ok, geophone_ok, infrasound_ok)
            
            return {
                "station": station_id,
                "success": True,
                "timestamp": timestamp,
                "backup_pc_ok": backup_pc_ok,
                "geophone_ok": geophone_ok,
                "infrasound_ok": infrasound_ok,
                "warnings": parsed['warnings'],
                "failures": parsed['failures']
            }
        else:
            store_log(station_id, timestamp, None, None, False, result.stderr.strip(), backup_pc_ok, geophone_ok, infrasound_ok)
            return {
                "station": station_id,
                "success": False,
                "timestamp": timestamp,
                "error": result.stderr.strip(),
                "backup_pc_ok": backup_pc_ok,
                "geophone_ok": geophone_ok,
                "infrasound_ok": infrasound_ok
            }
            
    except subprocess.TimeoutExpired:
        backup_pc_ok = check_backup_pc(station_id)
        geophone_ok, infrasound_ok = check_raspberryshake(geophone_id, infrasound_id)
        error_msg = f"SSH timeout after {SSH_TIMEOUT}s"
        store_log(station_id, timestamp, None, None, False, error_msg, backup_pc_ok, geophone_ok, infrasound_ok)
        return {
            "station": station_id,
            "success": False,
            "timestamp": timestamp,
            "error": error_msg,
            "backup_pc_ok": backup_pc_ok,
            "geophone_ok": geophone_ok,
            "infrasound_ok": infrasound_ok
        }
    except Exception as e:
        backup_pc_ok = check_backup_pc(station_id)
        geophone_ok, infrasound_ok = check_raspberryshake(geophone_id, infrasound_id)
        error_msg = str(e)
        store_log(station_id, timestamp, None, None, False, error_msg, backup_pc_ok, geophone_ok, infrasound_ok)
        return {
            "station": station_id,
            "success": False,
            "timestamp": timestamp,
            "error": error_msg,
            "backup_pc_ok": backup_pc_ok
        }


def store_log(station_id, timestamp, log_content, parsed, success, error, backup_ok, geophone_ok=None, infrasound_ok=None):
    """Store log entry in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    issues_json = json.dumps(parsed['issues'] if parsed else [])
    
    cursor.execute('''
        INSERT INTO status_logs 
        (station_id, timestamp, log_content, successes, warnings, failures, total,
         issues_json, fetch_success, error_message, backup_pc_ok, geophone_ok, infrasound_ok)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        station_id, timestamp, log_content,
        parsed['successes'] if parsed else 0,
        parsed['warnings'] if parsed else 0,
        parsed['failures'] if parsed else 0,
        parsed['total'] if parsed else 0,
        issues_json, success, error, backup_ok, geophone_ok, infrasound_ok
    ))
    
    # Cleanup old entries
    cursor.execute('''
        DELETE FROM status_logs 
        WHERE station_id = ? AND id NOT IN (
            SELECT id FROM status_logs 
            WHERE station_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        )
    ''', (station_id, station_id, MAX_HISTORY_PER_STATION))
    
    conn.commit()
    conn.close()


def write_summary_json(results):
    """Write summary for dashboard compatibility."""
    summary_file = STATUS_DIR / "last_fetch_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_stations": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def get_recent_statuses(station_id, n=3):
    """Get the N most recent status log entries for a station.
    
    Returns list of dicts with keys: failures, fetch_success, issues_json, timestamp.
    Ordered most recent first.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT failures, fetch_success, issues_json, timestamp
        FROM status_logs
        WHERE station_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (station_id, n))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def station_has_failure(entry):
    """Determine if a status entry represents a failure state.
    
    A station is in failure state if:
    - fetch_success is False (offline/unreachable), OR
    - failures > 0
    """
    if not entry['fetch_success']:
        return True
    return entry['failures'] > 0


def station_is_offline(entry):
    """Check if a status entry represents an offline/unreachable station."""
    return not entry['fetch_success']


def get_contact_emails(station_id):
    """Get contact_email list from cameras.json for a station."""
    try:
        with open(CAMERAS_JSON, 'r') as f:
            cameras = json.load(f)
        station_data = cameras.get(station_id, {})
        return station_data.get('station', {}).get('contact_email', [])
    except Exception:
        return []


def get_station_display_name(station_id):
    """Get a human-readable station name."""
    try:
        with open(CAMERAS_JSON, 'r') as f:
            cameras = json.load(f)
        station_data = cameras.get(station_id, {})
        st = station_data.get('station', {})
        return st.get('display_name') or st.get('name', station_id)
    except Exception:
        return station_id


def send_email(to_addresses, subject, body):
    """Send an email via local sendmail/SMTP."""
    if not to_addresses:
        return
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = 'Norsk meteornettverk <steinar@norskmeteornettverk.no>'
    msg['To'] = ', '.join(to_addresses)
    try:
        with smtplib.SMTP('localhost') as smtp:
            smtp.send_message(msg)
        print(f"  [Email] Sent to {msg['To']}: {subject}")
    except Exception as e:
        print(f"  [Email] Failed to send to {msg['To']}: {e}")


def check_and_notify(station_id):
    """Check for state transitions and send email notifications.
    
    Sends alert email when:
    - Last 2 checks both have failures, and the check before that had none
      (confirmed new failure)
    - Last check has no failures, and the check before that had failures
      (recovery) — but NOT if current state is offline/unknown
    """
    entries = get_recent_statuses(station_id, n=3)
    if len(entries) < 2:
        return
    
    current = entries[0]
    previous = entries[1]
    before_prev = entries[2] if len(entries) >= 3 else None
    
    contacts = get_contact_emails(station_id)
    if not contacts:
        return
    
    display_name = get_station_display_name(station_id)
    
    # Case 1: New failure confirmed (current and previous both have failures,
    # but the one before that was clean)
    if (station_has_failure(current) and station_has_failure(previous)
            and before_prev is not None and not station_has_failure(before_prev)):
        # Build failure description
        issues = []
        try:
            issues = json.loads(current.get('issues_json') or '[]')
        except Exception:
            pass
        
        if station_is_offline(current):
            feil_beskrivelse = "Stasjonen er ikke tilgjengelig (offline).\n"
        else:
            feil_beskrivelse = f"Antall feil: {current['failures']}\n"
            if issues:
                feil_beskrivelse += "\nFeil:\n"
                for issue in issues:
                    feil_beskrivelse += f"  - {issue.get('text', '')}\n"
        
        subject = f"[NMN] Feil på {display_name} ({station_id})"
        body = (
            f"Hei,\n\n"
            f"Stasjon {display_name} ({station_id}) har nå feil som vedvarer "
            f"over to påfølgende kontroller.\n\n"
            f"{feil_beskrivelse}\n"
            f"Siste sjekk: {current['timestamp']}\n\n"
            f"Du kan se mer informasjon på:\n"
            f"https://norskmeteornettverk.no/status/\n\n"
            f"Mvh,\nNorsk meteornettverk\n"
        )
        send_email(contacts, subject, body)
    
    # Case 2: Recovery (current is clean, and the last two checks both had failures)
    elif (not station_has_failure(current) and not station_is_offline(current)
              and station_has_failure(previous)
              and before_prev is not None and station_has_failure(before_prev)):
        subject = f"[NMN] {display_name} ({station_id}) er tilbake uten feil"
        body = (
            f"Hei,\n\n"
            f"Stasjon {display_name} ({station_id}) har ikke lenger noen feil.\n\n"
            f"Siste sjekk: {current['timestamp']}\n\n"
            f"Mvh,\nNorsk meteornettverk\n"
        )
        send_email(contacts, subject, body)



def test_single_station(station_id):
    """Test fetching from a single station."""
    print(f"Testing connection to {station_id}...")
    result = fetch_station(station_id)
    
    geo = result.get('geophone_ok')
    infra = result.get('infrasound_ok')
    geo_status = 'OK' if geo else ('FAIL' if geo == False else 'N/A')
    infra_status = 'OK' if infra else ('FAIL' if infra == False else 'N/A')
    
    if result["success"]:
        print(f"✓ Success!")
        print(f"  Backup PC: {'OK' if result['backup_pc_ok'] else 'FAIL'}")
        print(f"  Geophone: {geo_status}, Infrasound: {infra_status}")
        print(f"  Warnings: {result.get('warnings', 0)}, Failures: {result.get('failures', 0)}")
        return 0
    else:
        print(f"✗ Failed: {result['error']}")
        print(f"  Backup PC: {'OK' if result['backup_pc_ok'] else 'FAIL'}")
        print(f"  Geophone: {geo_status}, Infrasound: {infra_status}")
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        init_database()
        count = sync_stations()
        print(f"Synced {count} stations")
        stations = [r['station'] for r in get_all_stations()]
        if not stations:
            print("No stations found!")
            return 1
        return test_single_station(stations[0])
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting status fetch...")
    
    init_database()
    station_count = sync_stations()
    print(f"Loaded {station_count} stations from {CAMERAS_JSON}")
    
    stations = [r['station'] for r in get_all_stations()]
    
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_station, sid): sid for sid in stations}
        
        for future in as_completed(futures):
            station_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "✓" if result["success"] else "✗"
                backup = "b" if result.get("backup_pc_ok") else "-"
                print(f"  {status}{backup} {station_id}")
            except Exception as e:
                print(f"  ✗- {station_id}: Exception - {e}")
                results.append({
                    "station": station_id,
                    "success": False,
                    "error": str(e),
                    "backup_pc_ok": None
                })
    
    write_summary_json(results)
    
    # Check for state transitions and send email notifications
    print("\nChecking for status transitions...")
    for result in results:
        try:
            check_and_notify(result["station"])
        except Exception as e:
            print(f"  [Notify] Error checking {result['station']}: {e}")
    
    success_count = sum(1 for r in results if r["success"])
    print(f"\nFetch complete: {success_count}/{len(stations)} stations OK")
    
    return 0 if success_count == len(stations) else 1


def get_all_stations():
    """Get all stations from database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT id as station FROM stations ORDER BY id')
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


if __name__ == "__main__":
    sys.exit(main())
