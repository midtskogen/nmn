# NMN Public API

This directory contains a REST API that mirrors the services provided by the `data/` web interface.

## Base URL

```text
https://norskmeteornettverk.no/api/v1
```

The entry point is `nmn/api/index.php`.  Configure your web server so that all requests under `/api/v1` are routed to that file.

## Documentation

- HTML reference: `GET /api/v1/docs`
- Swagger UI: `GET /api/v1/docs/swagger` (or `GET /api/v1/docs/index.html`)
- OpenAPI spec: `GET /api/v1/docs/openapi.yaml`

## Authentication

Read-only endpoints do not require a key.  Stateful or station-traffic endpoints require an API key.

Pass the key in the header:

```bash
curl -H 'X-API-Key: ak_live_...' https://norskmeteornettverk.no/api/v1/downloads
```

A key may also be sent in the POST body as `api_key` for clients that
cannot set headers.  Keys are **not** accepted in the URL query string,
because that would leak them into web-server and proxy logs, browser
history and Referer headers.

API keys are stored in `etc/api_keys.json` (outside the web root).  Each key can be restricted to specific endpoint groups and to a per-key rate limit.

## Rate limits

- Anonymous (read-only) requests: 100 per minute / 1000 per hour per IP.
- Authenticated (stateful) requests: 20 per minute / 200 per hour per key.
- Queued prediction jobs (`predict/passes`, `predict/aircraft`): 5 per
  minute / 40 per hour per IP, with `days` <= 31, <= 20 stations per
  request, a 45-day start/end range cap and a global pending-queue cap.
- Failed API-key authentications: 10 per minute / 60 per hour per IP,
  logged to the abuse log.
- Tasks created through the API record the owning key; only that key (or
  an `admin` key) may cancel/stop/transcode them afterwards.
- `etc/` must live OUTSIDE the document root.  Set `NMN_SECRETS_DIR` in
  the web server environment to point the API at it; `etc/.htaccess`
  denies direct access as a fallback.

When a limit is exceeded the API returns HTTP 429 with a `Retry-After` header.

## Bandwidth / station-traffic limits

The API shares the same daily limits as the web interface:

- Downloads: 2 GB total / 1 GB per remote station per day.
- Live streams: same per-IP stream-time quotas as the web UI.
- Grid / annotation / archive overlay fetches also count toward the daily download quota.

## CPU-intensive task queue

Endpoints that start heavy calculations (`POST /predict/passes`, `POST /predict/aircraft`, `POST /enhance`) return immediately with a `task_id` and `status: queued`.  A background worker runs them with bounded concurrency (`max_workers = max(1, floor(cpu_count / 2))`) so server load stays below half of the cores.

Poll the task status endpoint until `status` becomes `complete` or `error`.

## Endpoints

| Method | Endpoint | Auth | Notes |
|---|---|---|---|
| GET | `/stations` | none | List stations |
| GET | `/cameras/fovs` | none | Camera FOVs |
| GET | `/kp` | none | Aurora KP index |
| GET | `/lightning?date=YYYY-MM-DD` | none | Lightning strikes |
| GET | `/meteors` | none | Meteor events |
| GET | `/stations/{id}/stats` | none | Station usage stats |
| POST | `/predict/passes` | none | Queue satellite pass prediction |
| GET | `/predict/passes/{task_id}` | none | Poll prediction status |
| POST | `/predict/aircraft` | none | Queue aircraft crossing prediction |
| GET | `/predict/aircraft/{task_id}` | none | Poll prediction status |
| POST | `/downloads` | key | Start file download |
| GET | `/downloads/{task_id}` | key | Poll download status |
| DELETE | `/downloads/{task_id}` | key | Cancel/cleanup download |
| POST | `/streams` | key | Start live stream |
| GET | `/streams/{task_id}` | key | Poll stream status |
| DELETE | `/streams/{task_id}` | key | Stop stream |
| POST | `/streams/{task_id}/transcode` | key | Request H.264 transcode |
| GET | `/grids/{station_id}/{camera_num}` | key | Fetch grid overlay |
| GET | `/annotations/{station_id}/{camera_num}` | key | Fetch annotation overlay |
| GET | `/archive/grid?station_id=&camera_num=&timestamp=` | key | Archive grid overlay |
| GET | `/archive/annotation?station_id=&camera_num=&timestamp=` | key | Archive annotation overlay |
| GET | `/archive/mask?station_id=&camera_num=&timestamp=` | key | Archive mask overlay |
| POST | `/enhance` | key | Queue image enhancement |
| GET | `/tasks/{task_id}` | key/status | Generic task status |
| GET | `/admin/stats` | key (admin) | Traffic stats JSON/HTML |
| GET | `/docs` | none | Swagger UI |
| GET | `/docs/openapi.yaml` | none | OpenAPI specification |

## Example requests

### Read-only data

```bash
# List all stations
curl https://norskmeteornettverk.no/api/v1/stations

# Camera fields-of-view
curl https://norskmeteornettverk.no/api/v1/cameras/fovs

# Aurora KP index
curl https://norskmeteornettverk.no/api/v1/kp

# Lightning for a specific date
curl 'https://norskmeteornettverk.no/api/v1/lightning?date=2026-08-31'

# Recent meteor events
curl https://norskmeteornettverk.no/api/v1/meteors

# Station statistics for a date range
curl 'https://norskmeteornettverk.no/api/v1/stations/ams173/stats?start_date=2026-08-01&end_date=2026-08-31'
```

### Satellite passes and aircraft crossings

Passes and aircraft searches run in a background queue because they are CPU-intensive. They return a `task_id` immediately.

```bash
# Predict satellite passes for one station, last 3 days
curl -X POST 'https://norskmeteornettverk.no/api/v1/predict/passes?station=ams173&days=3'

# Predict passes for all stations in a time window
curl -X POST 'https://norskmeteornettverk.no/api/v1/predict/passes?start=2026-08-31T00:00:00Z&end=2026-09-03T00:00:00Z'

# Predict aircraft crossings
curl -X POST 'https://norskmeteornettverk.no/api/v1/predict/aircraft?station=ams173&days=2'
```

Response:

```json
{
  "task_id": "pass_task_1725...",
  "status": "queued",
  "poll_url": "/api/v1/predict/passes/pass_task_1725..."
}
```

Poll until `status` is `complete` or `error`:

```bash
curl https://norskmeteornettverk.no/api/v1/predict/passes/pass_task_1725...
```

A running task may return intermediate progress:

```json
{
  "status": "progress",
  "step": 42,
  "total": 100,
  "message": "status_calculating_for_station|processed=7,total=90"
}
```

### Downloads

Downloads require an API key. The payload is the same JSON format used by the web interface.

```bash
# payload.json
{
  "files": [
    {"station_id": "ams173", "cam": 1, "time": "2026-08-31_20:00", "file_type": "image"},
    {"station_id": "ams173", "cam": 1, "time": "2026-08-31_20:05", "file_type": "image"}
  ]
}
```

```bash
curl -X POST https://norskmeteornettverk.no/api/v1/downloads \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: ak_live_...' \
  -d @payload.json
```

Response:

```json
{
  "task_id": "master_task_1725...",
  "status": "pending",
  "poll_url": "/api/v1/downloads/master_task_1725..."
}
```

Poll and cancel:

```bash
curl -H 'X-API-Key: ak_live_...' https://norskmeteornettverk.no/api/v1/downloads/master_task_1725...
curl -X DELETE -H 'X-API-Key: ak_live_...' https://norskmeteornettverk.no/api/v1/downloads/master_task_1725...
```

### Live streams

```bash
# Start a low-resolution stream on camera 1
curl -X POST 'https://norskmeteornettverk.no/api/v1/streams' \
  -H 'X-API-Key: ak_live_...' \
  -d 'station_id=ams173&camera_num=1&resolution=lowres'

# Response
{"task_id": "stream_1725...", "status": "pending", "poll_url": "/api/v1/streams/stream_1725..."}

# Poll stream status
curl -H 'X-API-Key: ak_live_...' https://norskmeteornettverk.no/api/v1/streams/stream_1725...

# Request H.264 transcode for a browser that does not support HEVC
curl -X POST -H 'X-API-Key: ak_live_...' \
  https://norskmeteornettverk.no/api/v1/streams/stream_1725.../transcode

# Stop the stream
curl -X DELETE -H 'X-API-Key: ak_live_...' https://norskmeteornettverk.no/api/v1/streams/stream_1725...
```

### Overlays

Overlays are fetched from the remote stations and count against the same daily download quota as downloads.

```bash
# Live stream grid overlay
curl -H 'X-API-Key: ak_live_...' https://norskmeteornettverk.no/api/v1/grids/ams173/1

# Star annotation overlay for the live stream
curl -H 'X-API-Key: ak_live_...' https://norskmeteornettverk.no/api/v1/annotations/ams173/1

# Archive grid overlay for a specific timestamp
curl -H 'X-API-Key: ak_live_...' \
  'https://norskmeteornettverk.no/api/v1/archive/grid?station_id=ams173&camera_num=1&timestamp=2026-08-31T20:00:00Z'

# Archive annotation overlay
curl -H 'X-API-Key: ak_live_...' \
  'https://norskmeteornettverk.no/api/v1/archive/annotation?station_id=ams173&camera_num=1&timestamp=2026-08-31T20:00:00Z'

# Archive mask overlay
curl -H 'X-API-Key: ak_live_...' \
  'https://norskmeteornettverk.no/api/v1/archive/mask?station_id=ams173&camera_num=1&timestamp=2026-08-31T20:00:00Z'
```

Response (example):

```json
{"success": true, "grid_url": "download/grid_ams173_cam1.png"}
```

### Image enhancement

Enhancement is CPU-intensive and queued.

```bash
curl -X POST https://norskmeteornettverk.no/api/v1/enhance \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: ak_live_...' \
  -d '{"image": "download/ams173_20260831_200000.jpg", "filter": 30}'
```

Response:

```json
{
  "task_id": "api_task_1725...",
  "status": "queued",
  "poll_url": "/api/v1/tasks/api_task_1725..."
}
```

Poll the generic task endpoint:

```bash
curl -H 'X-API-Key: ak_live_...' https://norskmeteornettverk.no/api/v1/tasks/api_task_1725...
```

### Administration / statistics

```bash
# Traffic statistics as JSON
curl -H 'X-API-Key: ak_live_...' https://norskmeteornettverk.no/api/v1/admin/stats

# Traffic statistics as an HTML dashboard
curl -H 'X-API-Key: ak_live_...' 'https://norskmeteornettverk.no/api/v1/admin/stats?html=1'
```

### Error and rate-limit responses

Missing API key:

```json
{"error": "missing_api_key", "message": "This endpoint requires an API key in the X-API-Key header."}
```

Rate limit exceeded:

```json
{"error": "rate_limit_exceeded", "message": "Too many requests. Please slow down.", "retry_after": 42}
```

The response status is `429` and a `Retry-After` header is included.

## Logs and stats

Every API request is appended to `nmn/api/query_log.json` (NDJSON).  Abuse events go to `nmn/api/abuse_log.json`.  Visit `/api/v1/admin/stats?html=1` for a dashboard, or omit `?html=1` for JSON.

## Configuration files

- `etc/api_config.json` – CORS origins, default rate limits, quota limits, queue settings.
- `etc/api_keys.json` – API key store.

Both files must be readable by `www-data` and not directly web-accessible.

## Deployment

1. Ensure `nmn/api/` is group `www-data` and mode `2770` so the web server can write logs and the queue file.
2. Create `etc/api_config.json` and `etc/api_keys.json` (see examples in this repo).
3. Make the API directory reachable under the web root. For example, on `bolide`:
   ```bash
   cd html
   ln -s ~/norskmeteornettverk.no/nmn/api api
   ```
4. The included `nmn/api/.htaccess` routes requests such as `/api/v1/stations` to `index.php`. For this to work, Apache must allow `.htaccess` overrides and symbolic links in that directory. Add to the site configuration (inside the `<VirtualHost>` block, adapted to the real path):
   ```apache
   <Directory /home/steinar/html/api>
       Options +FollowSymLinks
       AllowOverride All
       Require all granted
   </Directory>
   ```
   If the site uses a different document root, adjust the path accordingly. Then reload Apache:
   ```bash
   sudo systemctl reload apache2
   ```
5. Test a public endpoint:
   ```bash
   curl -I https://norskmeteornettverk.no/api/v1/stations
   ```
   A working setup returns HTTP 200 with `Content-Type: application/json`.
6. Run the API test script:
   ```bash
   cd nmn/api
   # Read-only tests only (verifies endpoints return HTTP 200 and non-empty data)
   python3 test_api.py

   # Include CPU-heavy predictions (uses one station, 1 day)
   python3 test_api.py --predictions

   # Include live station tests (needs an API key and a real station)
   NMN_API_KEY=ak_live_... python3 test_api.py --station ams173 --camera 1 --live-station-tests
   ```
7. The first queued CPU task will auto-start the queue worker daemon.
