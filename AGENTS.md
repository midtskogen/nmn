# Project notes

## Event reprocessing caveats (learned 2026-09, meteor/20260826/014151)

- Detected trails can be *phantom*: the detector may lock onto a noise/star
  path while the real meteor is a bright streak elsewhere in the frame.
  Verify by sampling `-clean.jpg` pixel values along `positions` — real
  trail points sit well above background. A 2-station trajectory fit can
  still look plausible on a bogus track (two lines of sight almost always
  intersect), so a clean fit is not proof the inputs are good.
- To repair a phantom trail: measure the streak in the *gnomonic* image,
  back-project its endpoints to az/alt via `gnomonic_corr_grid.pto`
  (`map_image_to_pano` -> pano -> az/alt) and write `positions`,
  `coordinates`, `startpos`, `endpos` in event.txt, then reprocess.
- `recalibrated=1` events: `lens.pto` carries the stale orientation;
  `gnomonic_corr_grid.pto` is the star-calibrated (authoritative)
  image->az/alt mapping. `calculate_refined_endpoints()` mixes frames
  (positions -> lens.pto -> az/alt -> corr_grid), so its initial guess can
  land tens of px off the streak and refinetrack won't reach it.
- `<name>-gnomonic-clean.jpg` was previously snapshotted only-if-missing,
  so a stale render in the old view geometry survived reprocessing and
  broke meteorcrop rotation/centering (fixed: snapshot now refreshes
  every run).
- `metrack.py`: `_fit_good_enough()`/`is_plausible`/`_is_implausible_info`
  now reject `min(start_h, end_h) < 5 km` — a degenerate all-inlier fit
  diving to ~1 km altitude used to pass the `max < 10 km` check and skip
  subset evaluation entirely.
- `meteorcrop.py` video trimming: source videos are ~12 s ring-buffer clips
  that can start many seconds before the event (burned-in clock ≠ the
  `[video] start` event-window timestamp). `detect_meteor_activity()` now
  uses a whole-clip median baseline, anchors on the peak-brightness frame,
  and expands contiguously through a lower `EDGE_FRACTION` threshold with
  `GAP_TOLERANCE_FRAMES` flicker bridging — the old first/last-above-
  threshold logic both missed short (3-frame) meteors and let stray noise
  spikes stretch the window.

## stitcher.py / multiblend.py

- Full-360 equirect blends use wrap-aware horizontal handling end to end
  (`wrap_x`, auto-enabled for f=2/v>=360 non-fisheye output): pyramid kernels
  wrap horizontally, and per-camera footprint masks are eroded with
  horizontal wrap (`_erode_camera_mask`).  Without the wrap-aware erosion the
  seam columns lose true coverage and the blend smears fisheye-corner junk
  into a pale vertical stripe at the wrap point.  The seam-assignment disk
  cache is wrap-independent (only assignment, not pyramids, is stored).
- ams173 cron runs a SEPARATE copy at /home/meteor/nmn/bin (user meteor);
  fixes here must be copied/symlinked there to take effect.



## Sky mask pipeline (nmn/bin/)

- `automask.py` orchestrates: `make_equirect_mask.py` (equirect sky mask from
  a cam8 timelapse video) -> `make_camera_masks.py` (reverse-project to
  per-camera native masks for amscams/scan_stack.py).
- Mask algorithm (make_equirect_mask.py): daytime-frame statistics
  (day_mean / persistent local contrast dark_frac / persistent vertical
  gradient vgrad) -> Otsu+texture seeds -> cyclic DP horizon seam ->
  attached canopy components -> thin-structure (blackhat) tier for
  antennas/wires -> enclosed-sky fill. Pixel-space params scale with width
  relative to a 1280px reference.
- Verification: benchmark against a hand-drawn mask (e.g. ~/ref.png,
  4096x1168, white=foreground) with IoU; expect IoU(fg) ~0.94, agreement
  ~98.8%. Visual review: `--preview` writes a red-boundary overlay.
- Camera-mask geometry: stitcher.py crops the equirect canvas to content
  bottom-only, so a 1280x464 (SD) / 4096x1168 (HD) video frame matches PTO
  canvas rows 0..463/1167 directly; pass `--output-width 1280
  --output-height 848` for SD mode to make_camera_masks.py (PTO canvas
  size, not the cropped image size).
- Per-camera masks are always written 1920x1080 (scan_stack.py's working
  size): the camera->pano coordinate maps are bilinearly upsampled and the
  equirect mask resampled with cubic + midpoint threshold, giving smooth
  anti-aliased binary edges.  The install dir .../CAL/MASKS (incl. CAL) is
  created if missing.
- Runtimes (2026-08 workstation): SD ~14 s, HD hires ~1.7 min. Speed came
  from estimating the local background on a 1/8-downscaled frame (equivalent
  to a full-res wide Gaussian within ~1-2 gray levels) and a numba-JIT
  banded DP for the horizon seam (falls back to numpy if numba is absent).

## Satellite pass prediction (nmn/server/data/predict_sat.py)

- CRITICAL PERF DEPENDENCY: the `sgp4` PyPI package ships an optional
  compiled C accelerator; Debian's apt-packaged `python3-sgp4`
  (`/usr/lib/python3/dist-packages`) is pure-Python only
  (`sgp4.api.accelerated == False`). This makes `find_events()`'s rise/
  set search ~20x slower (profiled: 54s -> 2.6s cumulative for one
  station/7-day window) since skyfield calls `sgp4()` millions of times
  during the bisection search. Fixed system-wide on bolide (2026-08-21) as
  root: `pip3 install --break-system-packages --upgrade --force-reinstall sgp4`,
  which installs the manylinux wheel with the compiled extension into
  `/usr/local/lib/python3.12/dist-packages` (takes precedence over the apt
  copy in `/usr/lib/python3/dist-packages` in sys.path, for all users
  including www-data). Verify with
  `python3 -c "from sgp4.api import accelerated; print(accelerated)"`.
  (An earlier www-data-only workaround under `/var/www/.local/...` was
  used before the system-wide fix landed and has since been removed.)
  If this moves to a new host or the system package is reinstalled via
  apt, redo this or the satellite panel silently reverts to being ~2.5x
  slower.
- Camera FOV calibration (`cameras.json`, via `pto_mapper.get_pto_data_from_json`)
  is cached by `_load_cameras_json()` (keyed on path+mtime) and looked up
  once per station in `process_station()`, not once per (satellite, pass,
  camera) -- that used to mean thousands of redundant JSON re-reads/parses
  per station.
- `get_tle_data()`'s cache freshness check must NOT also require every
  `SATELLITES_OF_INTEREST` name to be present: a permanently
  decayed/deorbited satellite (e.g. LACROSSE 5, no source ever has a
  current TLE for it again) would otherwise fail that check forever,
  forcing a full CelesTrak re-fetch (and 403 rate-limit risk) on every
  single call regardless of cache age.
- Per-station pass results are cached independently under
  `cache/passes/{station_id}.json` (see `_station_cache_path`) so an
  on-demand request for one/a few stations doesn't recompute for the whole
  network (~90 stations); day/time-range filters (`_filter_by_days`,
  `_filter_by_range`) are cheap post-filters over that cache and must use
  interval-overlap semantics (`earliest_camera_utc`/`latest_camera_utc`),
  not just "earliest view in bounds", or a narrow window can wrongly drop
  passes that start before it but run through it.

## Aircraft prediction (`nmn/server/data/predict_flight.py`)

- Data source is the OpenSky Network API:
  - OAuth token endpoint: `https://auth.opensky-network.org/...`
  - Flight list: `https://opensky-network.org/api/flights/all`
  - Track detail: `https://opensky-network.org/api/tracks/all`
  - State vectors: `https://opensky-network.org/api/states/all`
- A 7-day rolling local archive of 2-hour flight-list chunks is kept in
  `cache/flight_archive` (`FLIGHT_ARCHIVE_HOURS = 7*24`,
  `FLIGHT_ARCHIVE_CHUNK_SECONDS = 2*3600`).
- Track cache `cache/flight_tracks/` lifetime is 24 h (`TRACK_CACHE_HOURS`);
  main results cache is 15 min (`AIRCRAFT_CACHE_LIFETIME_MINUTES`).
- **Archive path pinning:** The archive, flight DB, token cache, and log
  directories are pinned to the project-root runtime `data/` directory rather
  than the script directory or arbitrary `NMN_DATA_DIR`. This keeps runtime
  credentials/caches outside the public `nmn` git repository and prevents the
  split-cache bug where web and cron jobs built separate archives.
- OpenSky `/tracks/all` already returns the densest raw track available;
  typical point spacing is seconds. Multi-minute gaps are genuine ADS-B
  receiver coverage holes, not a lower detail tier. The code interpolates
  suitable small gaps to ~15 s spacing but deliberately skips gaps with
  large heading or altitude changes.
- A `--cron` mode refreshes the archive periodically. Keep the web cache and
  cron cache at the same canonical path so filtering is consistent across
  invocations.

## Public API (`nmn/api/`)

- New REST API entry point is `nmn/api/index.php`, served at `/api/v1`.
- It re-uses the same caches, locks and quota trackers as `server/data/index.php`
  by setting `NMN_DATA_DIR` to the canonical `nmn/server/data` directory.
- API-specific configuration and keys live in `nmn/etc/api_config.json` and
  `nmn/etc/api_keys.json` (outside the web root, mode 640, group `www-data`).
- CPU-intensive endpoints (`predict/passes`, `predict/aircraft`, `enhance`) are
  queued via `nmn/api/queue_worker.py` to keep the 1-minute load below about
  half of the available cores.
- The same download and stream-time quotas apply as for the web UI.

## Web runtime paths / lock directory (`nmn/server/data/index.php`, `prediction_utils.py`)

- `server/data/index.php` is the web entry point. It sets:
  - `NMN_DATA_DIR` to the directory through which the request arrived, so
    symlinks under `/var/www/html/data/` and the real source tree both work.
  - `NMN_LOCK_DIR` to the web-root `data/locks/` directory so `www-data` can
    write status files even when the source tree is not group-writable.
- `server/data/prediction_utils.py` and related scripts use those variables
  when present and fall back to the script's real directory when they are
  absent. This lets cron invocations and direct CLI runs share the same
  caches as the web front-end.
- If a station has an independent copy under `/home/meteor/nmn/bin`, fixes
  must be copied/symlinked there too.

## AS7 health status and email alerts (`nmn/server/status/fetch_status.py`)

- `nmn/bin/as7health.py` performs the health audit and writes `as7health.log`;
  it does **not** send email itself.
- `server/status/fetch_status.py` SSHes to stations, parses the log, stores
  one month of history in `server/status/status.db`, and sends notification
  emails on confirmed state transitions (new failures, recoveries).
- High-load classification: `as7health.py` now reports a load average above
  `4 × CPU cores` as a **warning** (`HIGH_LOAD_AVG_FAIL` type = `warning`)
  instead of a failure, so it no longer triggers a failure alert.
- Alert emails for new failures use `build_issue_summary()` to include:
  - failure count and warning count from the latest check,
  - up to 15 detailed issues,
  - failures listed first, labeled `[FEIL]`, warnings labeled `[ADVARSEL]`,
  - a clear message when a station is offline.
- `fetch_status.py:parse_log_content()` retains up to 20 parsed issues
  (`issues[:20]`) so the summary has enough detail without flooding the DB.

## Retraining the meteor classification model (`nmn/bin/classify.py`)

- The runtime classifier is `nmn/bin/predict.py` (called by `nmn/bin/report.py`).
  It loads the first file found from this list:
  - `meteor_efficientnet_b0_model_clustered.pth.zst` (compressed, clustered)
  - `meteor_efficientnet_b0_model.pth`
  searched in `nmn/model/` (script-relative) or `~/nmn/model/` only —
  the current directory is deliberately NOT searched because report.py
  invokes it inside station-populated event directories.
- To regenerate that model, use the `efficientnet` subcommand of
  `nmn/bin/classify.py`:
  ```
  python3 nmn/bin/classify.py efficientnet --cluster /path/to/meteors /path/to/wrongs
  ```
  - `positive_dir` should contain verified meteor images (`meteor` class).
  - `negative_dir` should contain false/non-meteor images (`non_meteor` class).
  - The script supports `.jpg/.jpeg/.png` for image mode.
  - `--cluster` applies K-Means weight clustering and zstd-compresses the
    result to `meteor_efficientnet_b0_model_clustered.pth.zst`.
  - `--balance` can generate synthetic negatives if positives outnumber
    negatives.
  - Default input size is 192×96 (width×height); `predict.py` uses the same
    size.
- Important caveat: `classify.py` trains **from pretrained ImageNet weights
  every time**. It does *not* load the existing
  `meteor_efficientnet_b0_model_clustered.pth.zst` and fine-tune it. To
  update the model you must pass the full, combined dataset (old verified
  meteors + newly collected false/wrong samples) in the two class
  directories.
- After training, copy the resulting artifact to where `predict.py` looks:
  ```
  cp meteor_efficientnet_b0_model_clustered.pth.zst nmn/model/
  ```
  or, on a deployed station, to `~/nmn/model/`.
- Evaluate before replacing the live model:
  ```
  python3 nmn/bin/classify.py evaluate -m meteor_efficientnet_b0_model_clustered.pth.zst /path/to/test/meteors /path/to/test/wrongs
  ```
  This prints precision/recall/F1 and the optimal threshold. If you only
  want to train the unclustered `.pth`, omit `--cluster`; if you want the
  full image+video+stacking ensemble, use `buildensemble` instead.
- For a guided, graphical workflow use `nmn/bin/retrain_meteor_model_gui.py`.
  It checks dependencies, helps select/split positive and negative image
  directories, trains EfficientNet-B0, sweeps K-Means cluster counts
  (64/128/256/512) to compare model size vs. F1/precision/recall, and
  installs the chosen model into `nmn/model/`.
- The GUI can also fetch training images directly from the report directories:
  verified meteors live under `<repo_root>/meteor/YYYYMMDD/HHMMSS/<station>/camN/`,
  while false detections moved by `nmn/server/false.py` are stored under
  `<repo_root>/wrongs/` (equivalent to `/var/www/html/wrongs/` on a deployed
  web-root host).  It collects a chosen file pattern (default `fireball_orig.jpg`)
  from every leaf directory, hardlinks/copies them into separate
  positive/negative output folders, and then uses those folders for the
  train/verify split.

## Security hardening (learned 2026-09-18)

- `/var/www/html/etc/` was inside the document root and publicly served
  api_keys.json, credentials.json (OpenSky secret) and config.json (Frost
  secret) for ~2 weeks. Blocked via webroot .htaccess (`/etc` deny);
  the credentials still must be rotated and the dir moved outside the
  webroot (`NMN_SECRETS_DIR` env var overrides the default lookup).
- `/var/www/html/ssh/report.php` was an UNAUTHENTICATED endpoint whose
  `?dir=` parameter let anyone rsync arbitrary remote paths from stations
  into the public meteor tree. Hardened in place on bolide (station
  whitelist, event-shaped dir check, port range, per-IP rate limit,
  dedupe). It is NOT in the repo — keep it in sync if it changes.
- The endpoint now REQUIRES a shared-secret token: stations send
  `&token=` read from `/etc/default/nmn_report_token` (fallback
  `~/.nmn_report_token`) in `bin/report.py`. Token deployed as root
  (password auth) on 2026-09-18 to 13 primary stations + 10 backup
  PCs (ams*b). Still pending (tunnels refused/offline): ams136
  (vasteras), ams174b, ams180b, ams135b — each needs
  `/etc/default/nmn_report_token` + updated `~/nmn/bin/report.py`
  when it comes back. Station ssh port map lives in
  `/var/www/.ssh/config` on bolide (host 192.168.2.10, ports 10xxx).
- `server/report.php` (public report form): uploads now map detected
  MIME to a fixed server-side extension (never the client's), base64
  images are magic-byte validated, submissions are per-IP rate limited,
  and the script writes a protective `.htaccess` into `reports/` that
  disables script handlers and MIME sniffing.
- `server/fetch_foreign.sh` piped remote HTTP data into awk `system()`
  with `curl -k` — remote/MITM command injection. Fixed: TLS verified,
  fields regex-validated, `print > file` instead of `echo` in system().
- `server/meteor/id_check.php` echoed raw `$_POST` (reflected XSS).
- API keys are header/POST-body only (`?api_key=` removed); failed auth
  is logged + throttled; task ids are `random_bytes` (not `uniqid`);
  API tasks carry an owner sidecar for stop/cancel authorization.
- `.htaccess` files are now tracked (nmn/, server/data/, api/, lang/) —
  do not remove; they protect secrets, locks, cache and log files when
  the repo tree is deployed inside a document root.
