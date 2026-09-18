#!/usr/bin/env python3
"""
Comprehensive API smoke tests for the NMN public API.

Usage examples:
    # Read-only endpoints only
    python3 test_api.py

    # Include CPU-heavy predictions
    python3 test_api.py --predictions

    # Include live station tests (creates SSH/SCP traffic and streams)
    NMN_API_KEY=ak_live_... python3 test_api.py --station ams173 --camera 1 --live-station-tests

Environment variables:
    NMN_API_BASE      Base URL (default: https://norskmeteornettverk.no/api/v1)
    NMN_API_KEY       API key for stateful endpoints
    NMN_TEST_STATION  Station ID for live tests (overridden by --station)
    NMN_TEST_CAMERA   Camera number for live tests (overridden by --camera)
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

DEFAULT_BASE = 'https://norskmeteornettverk.no/api/v1'


class ApiTester:
    def __init__(self, base_url, api_key=None):
        self.base = base_url.rstrip('/')
        self.api_key = api_key
        self.passed = 0
        self.failed = 0
        self.warnings = []

    def url(self, path):
        return f"{self.base}{path}"

    def request(self, method, path, headers=None, data=None, key=None, timeout=30):
        """Make an HTTP request and return (status, headers, parsed_json_or_text)."""
        h = dict(headers or {})
        if key:
            h['X-API-Key'] = key
        if data is not None and isinstance(data, dict):
            data = json.dumps(data).encode('utf-8')
            h.setdefault('Content-Type', 'application/json')
        req = urllib.request.Request(self.url(path), method=method, data=data, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode('utf-8')
                try:
                    return resp.status, dict(resp.headers), json.loads(body)
                except json.JSONDecodeError:
                    return resp.status, dict(resp.headers), body
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8')
            try:
                return e.code, dict(e.headers), json.loads(body)
            except json.JSONDecodeError:
                return e.code, dict(e.headers), body

    def log(self, status, msg):
        mark = 'PASS' if status else 'FAIL'
        print(f'[{mark}] {msg}')
        if status:
            self.passed += 1
        else:
            self.failed += 1

    def warn(self, msg):
        print(f'[WARN] {msg}')
        self.warnings.append(msg)

    def _expect_status(self, method, path, expected, key=None, data=None, timeout=30):
        status, headers, body = self.request(method, path, key=key, data=data, timeout=timeout)
        ok = status == expected
        detail = f'status={status}, expected={expected}'
        if isinstance(body, dict) and 'error' in body:
            detail += f", error={body['error']}"
        self.log(ok, f'{method} {path}: {detail}')
        return status, headers, body

    def _expect_json_list_or_dict(self, method, path):
        status, headers, body = self.request('GET', path)
        ok = status == 200 and isinstance(body, (list, dict))
        self.log(ok, f'GET {path}: status={status}, json={isinstance(body, (list, dict))}')
        return status, headers, body

    def _expect_data(self, method, path, expected_type=list, min_len=1, required_keys=None):
        """Expect a non-empty JSON response of a given type with optional required keys."""
        status, headers, body = self.request(method, path)
        type_ok = status == 200 and isinstance(body, expected_type)
        len_ok = False
        keys_ok = True
        detail = f'status={status}, type_ok={type_ok}'
        if type_ok:
            if isinstance(body, list):
                len_ok = len(body) >= min_len
                detail += f', len={len(body)}, min_len={min_len}'
            elif isinstance(body, dict):
                len_ok = len(body) >= min_len
                detail += f', keys={len(body)}'
                if required_keys:
                    missing = [k for k in required_keys if k not in body]
                    keys_ok = not missing
                    detail += f', missing_keys={missing}'
        self.log(type_ok and len_ok and keys_ok,
                 f'{method} {path}: {detail}')
        return status, headers, body

    def test_readonly(self):
        print('\n--- Read-only endpoints ---')
        self._expect_data('GET', '/stations', dict, min_len=1)
        self._expect_data('GET', '/cameras/fovs', (list, dict), min_len=1)
        self._expect_data('GET', '/kp', list, min_len=1)
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        self._expect_data('GET', f'/lightning?date={yesterday}', (list, dict), min_len=0)
        self._expect_data('GET', '/meteors', (list, dict), min_len=1)

    def test_station_stats(self, station):
        if not station:
            return
        print('\n--- Station stats ---')
        start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d')
        end = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self._expect_data('GET', f'/stations/{station}/stats?start_date={start}&end_date={end}', dict, min_len=1)

    def _poll_task(self, poll_path, timeout=60):
        """Poll a queued task until it completes, errors or times out."""
        deadline = time.time() + timeout
        last_status = None
        while time.time() < deadline:
            status, headers, body = self.request('GET', poll_path)
            if status != 200:
                return status, headers, body
            task_status = body.get('status') if isinstance(body, dict) else None
            last_status = task_status
            if task_status in ('complete', 'error', 'done'):
                return status, headers, body
            if task_status in ('queued', 'pending', 'running'):
                time.sleep(2)
                continue
            # Unknown status; return anyway
            return status, headers, body
        return 200, {}, {'status': 'timeout', 'last_status': last_status}

    def test_predictions(self, station=None):
        print('\n--- Queued predictions (CPU-intensive) ---')
        # Use a single station and small window to keep the test short.
        if not station:
            status, headers, body = self.request('GET', '/stations')
            if status == 200 and isinstance(body, list) and len(body) > 0:
                station = body[0]
            elif isinstance(body, dict) and body:
                station = list(body.keys())[0]
        params = '?days=1'
        if station:
            params += f'&station={station}'

        # Satellite passes
        status, headers, body = self.request('POST', f'/predict/passes{params}', timeout=10)
        ok = status == 202 and isinstance(body, dict) and 'task_id' in body
        self.log(ok, f"POST /predict/passes{params}: status={status}, task_id={body.get('task_id') if isinstance(body, dict) else None}")
        if ok:
            poll = f"/predict/passes/{body['task_id']}"
            s, h, b = self._poll_task(poll, timeout=180)
            final = b.get('status') if isinstance(b, dict) else b
            valid = s == 200 and isinstance(b, dict) and final in ('queued', 'pending', 'running', 'progress', 'complete', 'error', 'done')
            self.log(valid, f"Poll /predict/passes: status={s}, final={final}")
            if s == 200 and final == 'complete':
                data_ok = isinstance(b, (list, dict))
                if isinstance(b, list):
                    data_ok = data_ok and len(b) > 0
                self.log(data_ok, f"GET /predict/passes data: items={len(b) if isinstance(b, list) else 'n/a'}")
            elif valid and final != 'error':
                self.warn(f"Prediction still running (final={final}); check task {poll} later")

        # Aircraft crossings
        status, headers, body = self.request('POST', f'/predict/aircraft{params}', timeout=10)
        ok = status == 202 and isinstance(body, dict) and 'task_id' in body
        self.log(ok, f"POST /predict/aircraft{params}: status={status}, task_id={body.get('task_id') if isinstance(body, dict) else None}")
        if ok:
            poll = f"/predict/aircraft/{body['task_id']}"
            s, h, b = self._poll_task(poll, timeout=180)
            final = b.get('status') if isinstance(b, dict) else b
            valid = s == 200 and isinstance(b, dict) and final in ('queued', 'pending', 'running', 'progress', 'complete', 'error', 'done')
            self.log(valid, f"Poll /predict/aircraft: status={s}, final={final}")
            if s == 200 and final == 'complete':
                data_ok = isinstance(b, (list, dict))
                if isinstance(b, list):
                    data_ok = data_ok and len(b) > 0
                self.log(data_ok, f"GET /predict/aircraft data: items={len(b) if isinstance(b, list) else 'n/a'}")
            elif valid and final != 'error':
                self.warn(f"Aircraft prediction still running (final={final}); check task {poll} later")

    def test_auth_required(self):
        if not self.api_key:
            print('\n--- Auth checks (no key provided, skipping) ---')
            return
        print('\n--- Auth checks ---')
        # Missing key on protected endpoints should return 401.
        # Status reads for downloads/streams do not require a key by design.
        for method, path in [
            ('POST', '/downloads'),
            ('POST', '/streams'),
            ('GET', '/grids/ams173/1'),
            ('GET', '/annotations/ams173/1'),
            ('GET', '/admin/stats'),
        ]:
            if method == 'POST' and path == '/downloads':
                body_data = b'{}'
                h = {'Content-Type': 'application/json'}
            elif method == 'POST' and path == '/streams':
                body_data = b'station_id=ams173&camera_num=1&resolution=lowres'
                h = {'Content-Type': 'application/x-www-form-urlencoded'}
            else:
                body_data = None
                h = {}
            status, headers, body = self.request(method, path, headers=h, data=body_data)
            self.log(status == 401, f'{method} {path} without key: status={status}, expected=401')

    def test_downloads(self, station=None):
        if not self.api_key:
            return
        print('\n--- Downloads ---')
        # A minimal payload. It may fail later because the files do not exist on the station,
        # but the API should accept it and return a task id.
        files = []
        if station:
            yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
            files.append({'station_id': station, 'cam': 1, 'time': f'{yesterday}_20:00', 'file_type': 'image'})
        payload = {'files': files} if files else {'files': []}
        status, headers, body = self.request('POST', '/downloads', key=self.api_key, data=payload, timeout=10)
        ok = status == 202 and isinstance(body, dict) and 'task_id' in body
        self.log(ok, f"POST /downloads: status={status}, task_id={body.get('task_id') if isinstance(body, dict) else None}")
        if ok:
            poll = f"/downloads/{body['task_id']}"
            s, h, b = self._poll_task(poll, timeout=30)
            final = b.get('status') if isinstance(b, dict) else b
            self.log(s == 200, f"Poll /downloads: status={s}, final={final}")
            if s == 200 and isinstance(b, dict):
                self.log('status' in b, f"GET /downloads task body has 'status' key")
            # Attempt cancel (best-effort cleanup)
            self.request('DELETE', poll, key=self.api_key)

    def test_streams(self, station, camera):
        if not self.api_key or not station:
            return
        print('\n--- Live streams ---')
        data = f'station_id={station}&camera_num={camera}&resolution=lowres'
        status, headers, body = self.request(
            'POST', '/streams',
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data=data.encode('utf-8'),
            key=self.api_key,
            timeout=10
        )
        ok = status == 202 and isinstance(body, dict) and 'task_id' in body
        self.log(ok, f"POST /streams: status={status}, task_id={body.get('task_id') if isinstance(body, dict) else None}")
        if ok:
            poll = f"/streams/{body['task_id']}"
            s, h, b = self._poll_task(poll, timeout=20)
            final = b.get('status') if isinstance(b, dict) else b
            self.log(s == 200, f"Poll /streams: status={s}, final={final}")
            if s == 200 and isinstance(b, dict):
                self.log('status' in b, f"GET /streams task body has 'status' key")
            # Stop the stream immediately to avoid consuming quota
            stop_status, _, stop_body = self.request('DELETE', poll, key=self.api_key)
            self.log(stop_status == 200, f"DELETE /streams: status={stop_status}")

    def test_overlays(self, station, camera):
        if not self.api_key or not station:
            return
        print('\n--- Overlays ---')
        paths = [
            f'/grids/{station}/{camera}',
            f'/annotations/{station}/{camera}',
        ]
        for path in paths:
            status, headers, body = self.request('GET', path, key=self.api_key, timeout=30)
            ok = status == 200 and isinstance(body, dict)
            success = body.get('success') if isinstance(body, dict) else False
            url = body.get('grid_url') or body.get('annotation_url') if isinstance(body, dict) else None
            self.log(ok and success and bool(url),
                     f'GET {path}: status={status}, success={success}, url={bool(url)}')

        ts = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        for kind in ['grid', 'annotation', 'mask']:
            path = f'/archive/{kind}?station_id={station}&camera_num={camera}&timestamp={ts}'
            status, headers, body = self.request('GET', path, key=self.api_key, timeout=30)
            ok = status == 200 and isinstance(body, dict)
            success = body.get('success') if isinstance(body, dict) else False
            url = body.get('grid_url') or body.get('annotation_url') or body.get('mask_url') if isinstance(body, dict) else None
            msg = body.get('message') or body.get('error') if isinstance(body, dict) else None
            if ok and not success and ('not found' in str(msg).lower() or 'no ' in str(msg).lower() or msg is None):
                self.warn(f'GET {path}: success={success} ({msg}); no archive data for timestamp {ts}')
            else:
                self.log(ok and success and bool(url),
                         f'GET {path}: status={status}, success={success}, url={bool(url)}, msg={msg}')

    def test_enhance(self):
        if not self.api_key:
            return
        print('\n--- Image enhancement ---')
        payload = {'image': 'download/ams173_20260831_200000.jpg', 'filter': 30}
        status, headers, body = self.request('POST', '/enhance', key=self.api_key, data=payload, timeout=10)
        ok = status == 202 and isinstance(body, dict) and 'task_id' in body
        self.log(ok, f"POST /enhance: status={status}, task_id={body.get('task_id') if isinstance(body, dict) else None}")
        if ok:
            poll = f"/tasks/{body['task_id']}"
            s, h, b = self._poll_task(poll, timeout=30)
            final = b.get('status') if isinstance(b, dict) else b
            self.log(s == 200, f"Poll /tasks: status={s}, final={final}")
            if s == 200 and final == 'complete':
                has_image = isinstance(b, dict) and ('image' in b or 'data' in b)
                self.log(has_image, f"GET /tasks enhance data: has_image={has_image}")

    def test_admin_stats(self):
        if not self.api_key:
            return
        print('\n--- Admin stats ---')
        status, headers, body = self.request('GET', '/admin/stats', key=self.api_key)
        expected_keys = {'requests', 'top_ips', 'top_keys', 'endpoints', 'quota_hits', 'abuse', 'hour', 'day', 'week', 'queue'}
        keys_ok = status == 200 and isinstance(body, dict) and bool(expected_keys & set(body.keys()))
        detail = f'status={status}, json={isinstance(body, dict)}, keys={list(body.keys())[:5] if isinstance(body, dict) else None}'
        self.log(keys_ok, f"GET /admin/stats: {detail}")
        status, headers, body = self.request('GET', '/admin/stats?html=1', key=self.api_key)
        ct = headers.get('Content-Type', '')
        body_ok = status == 200 and 'text/html' in ct and len(body) > 200 if isinstance(body, str) else True
        self.log(body_ok,
                 f"GET /admin/stats?html=1: status={status}, content-type={ct}, body_len={len(body) if isinstance(body, str) else 'n/a'}")

    def test_docs(self):
        print('\n--- Documentation ---')
        for path in ['/docs', '/docs/swagger', '/docs/openapi.yaml']:
            status, headers, body = self.request('GET', path)
            ct = headers.get('Content-Type', '')
            ok = status == 200 and ('text/html' in ct or 'yaml' in ct or 'json' in ct or 'octet' in ct)
            body_len = len(body) if isinstance(body, str) else len(str(body))
            self.log(ok and body_len > 100,
                     f"GET {path}: status={status}, content-type={ct}, body_len={body_len}")

    def run(self, predictions=False, live_station_tests=False, station=None, camera=1):
        self.test_docs()
        self.test_readonly()
        self.test_station_stats(station)
        self.test_auth_required()

        if predictions:
            self.test_predictions(station)

        if self.api_key:
            self.test_admin_stats()

        if live_station_tests and self.api_key and station:
            self.test_downloads(station)
            self.test_streams(station, camera)
            self.test_overlays(station, camera)
            self.test_enhance()
        elif live_station_tests and not self.api_key:
            self.warn('Live station tests require an API key (set NMN_API_KEY)')
        elif live_station_tests and not station:
            self.warn('Live station tests require --station')

        print('\n--- Summary ---')
        print(f'Passed: {self.passed}')
        print(f'Failed: {self.failed}')
        if self.warnings:
            print('Warnings:')
            for w in self.warnings:
                print(f'  - {w}')
        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description='NMN API smoke tests')
    parser.add_argument('--base', default=os.environ.get('NMN_API_BASE', DEFAULT_BASE), help='API base URL')
    parser.add_argument('--key', default=os.environ.get('NMN_API_KEY'), help='API key')
    parser.add_argument('--station', default=os.environ.get('NMN_TEST_STATION'), help='Station ID for live tests')
    parser.add_argument('--camera', type=int, default=int(os.environ.get('NMN_TEST_CAMERA', '1')), help='Camera number')
    parser.add_argument('--predictions', action='store_true', help='Run CPU-heavy prediction tests')
    parser.add_argument('--live-station-tests', action='store_true', help='Run tests that contact real stations')
    args = parser.parse_args()

    tester = ApiTester(args.base, args.key)
    ok = tester.run(
        predictions=args.predictions,
        live_station_tests=args.live_station_tests,
        station=args.station,
        camera=args.camera
    )
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
