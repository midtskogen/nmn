# Norsk meteornettverk (NMN) — AllSky7 Software

This repository contains the software that runs the Norsk meteornettverk (NMN), a network of AllSky7 (AMS/AllSky7) meteor camera stations in Norway. It handles station health monitoring, real-time meteor detection, video and image processing, orbit/trajectory calculation, web reporting, live all-sky imagery, and prediction of satellites and aircraft.

> **Path note:** The live stations and central server usually deploy this tree as `/home/meteor/nmn`. The examples below use that path; adjust to your local clone (e.g. `/home/steinar/norskmeteornettverk.no/nmn`) when working on this repository directly.

## Repository Layout

```
.
├── bin/               # Station-side command-line tools and offline processing scripts
├── server/            # Central web backend, public API, fetch/processing scripts
│   ├── data/          # Live web application (index.php, controller.py, JS, prediction scripts, assets)
│   ├── loc/           # Report/event translation files
│   └── status/        # Station health status aggregation and alerting
├── model/             # Machine-learning model files for meteor detection/classification
├── src/               # C source code for compiled helper programs
├── .clang-format      # Formatting rules for C/C++ code
└── ../AGENTS.md       # Project notes for contributors (repo root)
```

## Main Components

### `bin/` — Station Tools

Scripts used directly on each camera station and for offline processing:

#### Health & Configuration

- **`as7health.py`** — Comprehensive station health audit. Checks mounts, cloud archive, network, syslog, OOM events, segmentation faults, NMN directory/git/config state, and more.
  - Run without flags for a **station-only** check.
  - Run with `--nmn` for **NMN-wide** checks on the central server.
  - Issues are classified as `failure`, `warning`, or `info`. Critical high-load conditions are reported as warnings, not failures.
- **`setconf.sh`** — Station configuration helper (network/IP, logo, etc.).
- **`clean_mailbox.py`** / **`reduce_log_spam.py`** / **`cleanup.sh`** — Maintenance helpers.

#### Capture & Detection Pipeline

- **`cammon.py`** / **`amsmon.py`** / **`camip.py`** — Camera monitoring, IP discovery, and capture helpers.
- **`camconfig.py`** — Camera configuration utilities.
- **`process.sh`** + **`cammon.py`** + compiled `metdetect` — Legacy real-time detection loop. The current pipeline uses the Python tools below instead.
- **`metrack.py`** — RANSAC-based atmospheric meteor trajectory fitting.
- **`centroid2event.py`** — Maps celestial centroids from a Hugin PTO project into an NMN event file.
- **`ams2event.py`** — Converts AMS detection JSON files to the NMN event format.
- **`orbit.py`** — Calculates and plots meteor orbits (uses SPICE kernels).
- **`showerassoc.py`** — Associates observed meteors with known showers (uses SPICE).

#### Calibration & Geometry

- **`calibrate.py`** — Stellar calibration to produce a lens `.pto` file.
- **`calibrate.sh`** — Legacy shell wrapper for calibration (no longer maintained).
- **`amscalib2lens.py`** — Creates a Hugin `.pto` from an AMS JSON calibration file.
- **`pto2amscalib.py`** — Converts a Hugin `.pto` back into an AMS-style `*calparams.json`.
- **`pto_mapper.py`** — Coordinate transforms between panorama and individual camera pixels.
- **`findstar.py`** / **`astrometry.py`** / **`stars.py`** — Star catalog and astrometry helpers.

#### Image/Video Processing

- **`stitcher.py`** — All-sky image/video stitcher. Reprojects multiple camera views into a single panorama (equirectangular or fisheye), with vignetting correction (`--devignette`), exposure compensation, and seamless Laplacian/pyramid blending.
- **`multiblend.py`** — Multi-band blending backend used by `stitcher.py`.
- **`stitch.py`** *(in `server/data/`)* — Thin web wrapper around `stitcher.py`.
- **`makevideos.py`** — Event video processing: stacked images, grid overlays, and gnomonic projections.
- **`stack.py`** — Image/video stacking, with optional ffmpeg hardware acceleration.
- **`timelapse*.sh`** / **`stitch_latest.sh`** — Shell wrappers for nightly timelapse and live-stitch generation.

#### Sky-Mask Pipeline

- **`automask.py`** — Orchestrator for the full mask pipeline.
- **`make_equirect_mask.py`** — Builds a binary sky mask from an equirectangular timelapse video, separating sky from static foreground (trees, buildings, masts, wires).
- **`make_camera_masks.py`** — Reverse-projects the equirect mask into per-camera native 1920×1080 masks for `scan_stack.py`, and optionally produces a fisheye mask.
- **`draw_camera_boundaries.py`** / **`drawgrid.py`** — Boundary/grid visualization helpers.

#### Classification & Reporting

- **`classify.py`** — PyTorch meteor image/video classification (2D and 3D CNNs). Supports train, predict, and `buildensemble` modes, optional K-Means clustering, and synthetic balancing.
- **`process.py`** — Processes a single meteor event detection: validation, video/classification calls, Metrack data generation, and translated brightness plots.
- **`report.py`** — Classifies an event, produces plots/reports, and reports to the central NMN server if it passes the probability threshold.

#### Utilities

- **`fb2kml.py`** / **`fbspd_merge.py`** — Fireball/FBSPD report helpers.
- **`p100.py`** / **`imx291time.py`** / **`telnet_opener.py`** — Hardware/camera helpers.
- **`sunpos.py`** / **`altaz.py`** / **`findcoord.py`** / **`windprofile.py`** — Astronomical and atmospheric utilities.

### `server/` — Central Web Backend & Services

- **`server/fetch.py`** — Central data fetcher. Pulls observations from remote stations with rsync, computes trajectories and orbits, generates plots, and produces translated HTML/KML reports.
- **`server/loc_fetch.py`**, **`server/loc_metrack.py`**, **`server/loc_orbit.py`**, **`server/loc_process.py`**, **`server/loc_fbspd_merge.py`** — Localized (language-aware) wrappers around the tools above.
- **`server/merge_events.py`** — Merges multi-station observations into single events.
- **`server/classification.py`** — Keras/Tuner model training/tuning pipeline for meteor image classification. Distinct from the runtime classifier in `bin/classify.py`.
- **`server/false.py`** — Sanitises, copies, and cleans up suspect meteor data directories (e.g. rejected/false events).
- **`server/event_cleanup.py`** — Automated cleanup of old event data.
- **`server/filter.py`** — Filtering helpers.
- **`server/obs.html`** — Legacy/static "Report a meteor" page (separate from the live `server/data/index.php` site).

### `server/data/` — Live Web Application

- **`index.php`** — Web entry point. Sets `NMN_DATA_DIR`/`NMN_LOCK_DIR`, resolves language, and dispatches to `controller.py`.
- **`controller.py`** — Main live website controller; orchestrates video streams, stitching, predictions, quotas, etc.
- **`predict_sat.py`** — Satellite pass prediction using `skyfield`/`sgp4`. Per-station caches under `cache/passes/`.
- **`predict_flight.py`** — Aircraft track prediction using the OpenSky Network, with a 7-day rolling local archive.
- **`media_processor.py`** — Overlay generation, thumbnails, video probes, and aircraft/satellite track rendering.
- **`live_streamer.py`** — Live camera streaming with per-IP/per-station time quotas.
- **`stitch.py`** — Web-facing wrapper for `bin/stitcher.py`.
- **`api.js`** / **`main.js`** / **`map_handler.js`** / **`ui_manager.js`** / **`chart_handler.js`** / **`utils.js`** / **`calculations.js`** — Frontend JavaScript.
- **`style.css`** — Frontend styles.
- **`lang/*.json`** — UI translations.
- **`airline_codes.json`** / **`airline_codes.js`** — ICAO/IATA airline code lookups.
- **`prediction_utils.py`**, **`shared_utils.py`**, **`data_fetchers.py`** — Shared backend utilities.

### `server/status/` — Health Status Aggregation

- **`fetch_status.py`** — SSHes to every station, fetches/parses `as7health.log`, stores status history in SQLite, and sends email alerts on new failures and recoveries. This is the component that sends health email, not `as7health.py`.
- **`index.html`** — Station status dashboard.

### `model/` — Machine Learning

- **`meteor_efficientnet_b0_model_clustered.pth.zst`** — Compressed EfficientNet-B0 PyTorch model used for meteor/non-meteor classification.

### `src/` — Legacy C Helpers

- **`metdetect.c`** — Fast meteor detection helper.
- **`parsexy.c`** — Star/plate coordinate parser helper.

These C helpers and the shell scripts that invoked them (`bin/process.sh`, `bin/calibrate.sh`) are **legacy software that is no longer used or maintained** by the current pipeline. `bin/compile.sh` remains in the tree but is not part of active builds.

## Quick Start

### Station Health Check

On a station (deployed under `/home/meteor/nmn`):

```bash
# Station-only checks; run as root for a complete diagnosis
sudo python3 /home/meteor/nmn/bin/as7health.py

# Central NMN-wide checks; usually run on the central server
sudo python3 /home/meteor/nmn/bin/as7health.py --nmn
```

### All-Sky Image/Video Stitch

```bash
python3 /home/meteor/nmn/bin/stitcher.py --help

# Equirectangular output
python3 /home/meteor/nmn/bin/stitcher.py lens.pto image*.jpg out.jpg --equirect

# Fisheye output
python3 /home/meteor/nmn/bin/stitcher.py --fisheye image*.jpg out.mp4
```

### Calibration

```bash
python3 /home/meteor/nmn/bin/calibrate.py --help
python3 /home/meteor/nmn/bin/amscalib2lens.py --help
python3 /home/meteor/nmn/bin/pto2amscalib.py --help
```

### Sky-Mask Pipeline

```bash
python3 /home/meteor/nmn/bin/automask.py --help
```

### Real-Time Detection (legacy)

The original detection loop was built around the C helper `metdetect` and the shell wrappers `compile.sh` / `process.sh`. These are **legacy and no longer maintained**; the current pipeline uses the Python tooling instead.

## Dependencies

The code is written for Python 3 and uses a mix of standard and third-party libraries. There is no bundled `requirements.txt`; individual scripts check for optional dependencies at runtime and print install hints.

Commonly required libraries:

- NumPy, SciPy, Pillow/PIL
- Numba (used heavily in `stitcher.py` and `multiblend.py`)
- OpenCV (`cv2`)
- PyAV (`av`)
- PyTorch
- requests, paramiko
- `skyfield` + `sgp4` (satellite prediction; the PyPI wheel with compiled extension is strongly recommended)
- `spiceypy` (orbit determination)
- `ephem` / `pyephem`
- matplotlib, cartopy, plotly
- astropy, Wand (ImageMagick), `ffmpeg-python`
- zstandard, tifffile
- scikit-learn, KerasTuner
- tqdm

Optional enhancements: cairosvg, scour, scikit-image.

## Configuration & Runtime Notes

- Station-specific settings live outside this repository on each host (e.g. `/home/ams/amscams/conf/...`, `/etc/meteor.cfg`). `setconf.sh` and `as7health.py` check these files and report problems if they are missing or inconsistent.
- The web entry point (`server/data/index.php`) sets `NMN_DATA_DIR` and `NMN_LOCK_DIR` so that symlinks under `/var/www/html/data/` and direct cron invocations share the same cache/archive/log directories. Prediction scripts fall back to their own real directory when the variable is absent.
- Satellite performance: `predict_sat.py` is sensitive to the `sgp4` implementation. Debian's `python3-sgp4` package is pure-Python and ~20× slower than the PyPI wheel with the compiled extension. On production servers, ensure the compiled wheel is installed system-wide and takes precedence.
- Some stations run a separate copy of scripts under `/home/meteor/nmn/bin`; fixes must be copied or symlinked there to take effect on those hosts.
- Health email notifications are generated by `server/status/fetch_status.py`, not by `as7health.py`.

## Development Notes

- **Numba cache issues:** `multiblend.py` uses Numba with `cache=True`. If you encounter strange Numba errors after changing import paths, clear the cache directories (`nmn/bin/__pycache__` and Numba's own cache, normally `~/.cache/numba` or `$NUMBA_CACHE_DIR`).
- **C helpers:** `src/metdetect.c` and `src/parsexy.c`, and `bin/compile.sh`, are legacy code and are not rebuilt or used by the active pipeline.
- See `../AGENTS.md` for detailed operational notes on the mask pipeline, `sgp4` performance, web-root data directory pitfalls, and other contributor guidance.

---

*Note: This README was updated by an LLM and should be reviewed periodically as the codebase evolves.*
