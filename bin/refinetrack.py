#!/usr/bin/env python3
"""
refinetrack.py

This script refines the start and end coordinates of a meteor track in an image.
Given an image and an approximate start/end position, it analyzes the image to
find the most likely, precise path of the meteor.

The process involves:
1.  A broad search for the brightest path near the initial guess.
2.  A detailed scan along that path to find the longest continuous bright segment.
3.  Validation checks to ensure the refined track is reasonable.
4.  Trimming the track if it extends too far beyond the original estimate.
5.  Optionally, applying a directional 1D filter to the entire image and then
    estimating brightness at N points using a conditional flood-fill algorithm.
"""

# Import necessary libraries
import argparse
import math
import sys
import numpy as np
import numba
from numba import prange
from pathlib import Path
from PIL import Image
from collections import deque
from scipy.ndimage import rotate, convolve1d

# --- Helper & Algorithm Functions ---

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

def coords(s: str) -> tuple[float, float]:
    """Parses a string 'x,y' into a tuple of two floats."""
    try:
        x, y = map(float, s.split(','))
        return x, y
    except:
        raise argparse.ArgumentTypeError("Coordinates must be in 'x,y' format.")

def frange(start: float, end: float, step: float):
    """A float-range generator."""
    current = start
    while current < end:
        yield current
        current += step

def dist(start: tuple[float, float], end: tuple[float, float]) -> float:
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

def line(center: tuple[float, float], angle: float, length: float, steps: int) -> list[tuple[float, float]]:
    """Generates points for a line segment."""
    start_point = (center[0] - math.cos(angle) * length / 2,
                   center[1] - math.sin(angle) * length / 2)
    line_points = []
    if steps == 0: return []
    step_size = float(length) / steps
    for i in frange(0, length, step_size):
        x = start_point[0] + math.cos(angle) * i
        y = start_point[1] + math.sin(angle) * i
        line_points.append((x, y))
    return line_points

@numba.njit(cache=True, fastmath=True)
def _brightness_nb(img_y, start_x, start_y, end_x, end_y, steps):
    """JIT-compiled brightness sampling along a line using bilinear interpolation on luma."""
    length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    if steps == 0 or length == 0.0:
        return 0.0
    angle = math.atan2(end_y - start_y, end_x - start_x)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    step_size = length / steps
    h, w = img_y.shape
    total = 0.0
    i = 0.0
    while i < length:
        x = start_x + cos_a * i
        y = start_y + sin_a * i
        xi, yi = int(x), int(y)
        if 0 <= xi < w - 1 and 0 <= yi < h - 1:
            xf, yf = x - xi, y - yi
            p00 = img_y[yi, xi]
            p10 = img_y[yi, xi + 1]
            p01 = img_y[yi + 1, xi]
            p11 = img_y[yi + 1, xi + 1]
            p_top = p00 * (1.0 - xf) + p10 * xf
            p_bot = p01 * (1.0 - xf) + p11 * xf
            total += p_top * (1.0 - yf) + p_bot * yf
        i += step_size
    return total


@numba.njit(cache=True, fastmath=True)
def _phase1_search_nb(img_y, starts_x, starts_y, ends_x, ends_y, steps):
    """JIT-compiled Phase 1 exhaustive search across all start/end pairs."""
    n_s = len(starts_x)
    n_e = len(ends_x)
    best = -1.0
    best_si = 0
    best_ei = 0
    for si in range(n_s):
        for ei in range(n_e):
            val = _brightness_nb(img_y, starts_x[si], starts_y[si], ends_x[ei], ends_y[ei], steps)
            if val > best:
                best = val
                best_si = si
                best_ei = ei
    return best_si, best_ei


def brightness(img, start: tuple[float, float], end: tuple[float, float], steps: int) -> list[float]:
    """Calculates brightness values along a line using bilinear interpolation."""
    brightness_values = []
    length = dist(start, end)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    if steps == 0: return []
    step_size = length / steps
    for i in frange(0, length, step_size):
        x, y = start[0] + math.cos(angle) * i, start[1] + math.sin(angle) * i
        try:
            x_int, y_int = int(x), int(y)
            x_frac, y_frac = x - x_int, y - y_int
            p00 = sum(img[x_int, y_int][:2])
            p10 = sum(img[x_int + 1, y_int][:2])
            p01 = sum(img[x_int, y_int + 1][:2])
            p11 = sum(img[x_int + 1, y_int + 1][:2])
            p_top = p00 * (1 - x_frac) + p10 * x_frac
            p_bottom = p01 * (1 - x_frac) + p11 * x_frac
            brightness_values.append(p_top * (1 - y_frac) + p_bottom * y_frac)
        except (IndexError, TypeError):
            pass
    return brightness_values

@numba.njit(cache=True, fastmath=True)
def _track_nb(img_y, start_x, start_y, end_x, end_y, steps):
    """JIT-compiled track scan: returns arrays of (is_dim, last_x, last_y, x, y, is_very_bright)."""
    length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    if steps == 0 or length == 0.0:
        return (np.empty(0, np.bool_), np.empty(0, np.float64), np.empty(0, np.float64),
                np.empty(0, np.float64), np.empty(0, np.float64), np.empty(0, np.bool_))
    h, w = img_y.shape
    angle = math.atan2(end_y - start_y, end_x - start_x)
    perp = angle + math.pi / 2.0
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    cos_p, sin_p = math.cos(perp), math.sin(perp)
    step_size = length / steps
    n = int(length / step_size) + 1
    is_dim_arr       = np.empty(n, np.bool_)
    last_x_arr       = np.empty(n, np.float64)
    last_y_arr       = np.empty(n, np.float64)
    cur_x_arr        = np.empty(n, np.float64)
    cur_y_arr        = np.empty(n, np.float64)
    is_bright_arr    = np.empty(n, np.bool_)
    count = 0
    last_x, last_y = start_x, start_y
    i_f = 0.0
    while i_f < length:
        x = start_x + cos_a * i_f
        y = start_y + sin_a * i_f
        xi, yi = int(x), int(y)
        if 0 <= xi < w - 1 and 0 <= yi < h - 1:
            xf, yf = x - xi, y - yi
            p00 = img_y[yi, xi]; p10 = img_y[yi, xi + 1]
            p01 = img_y[yi + 1, xi]; p11 = img_y[yi + 1, xi + 1]
            pt_b = (p00*(1.0-xf)+p10*xf)*(1.0-yf) + (p01*(1.0-xf)+p11*xf)*yf
            # background: 30-point perpendicular sample
            bg_sum = 0.0
            bg_n = 0
            for k in range(30):
                bx = x + cos_p * (-15.0 + k)
                by = y + sin_p * (-15.0 + k)
                bxi, byi = int(bx), int(by)
                if 0 <= bxi < w - 1 and 0 <= byi < h - 1:
                    bg_sum += img_y[byi, bxi]
                    bg_n += 1
            bg_avg = bg_sum / bg_n if bg_n > 0 else pt_b
            is_dim = pt_b <= bg_avg
            is_bright = pt_b > bg_avg + 24.0
            if count < n:
                is_dim_arr[count]    = is_dim
                last_x_arr[count]    = last_x
                last_y_arr[count]    = last_y
                cur_x_arr[count]     = x
                cur_y_arr[count]     = y
                is_bright_arr[count] = is_bright
                count += 1
            if is_dim:
                last_x, last_y = x, y
        i_f += step_size
    return (is_dim_arr[:count], last_x_arr[:count], last_y_arr[:count],
            cur_x_arr[:count], cur_y_arr[:count], is_bright_arr[:count])


def track(img, start: tuple[float, float], end: tuple[float, float], steps: int) -> list:
    """Scans a line to determine if points are dimmer than their surroundings."""
    track_data = []
    length = dist(start, end)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    last_x, last_y = start
    if steps == 0: return []
    step_size = length / steps
    for i in frange(0, length, step_size):
        x, y = start[0] + math.cos(angle) * i, start[1] + math.sin(angle) * i
        try:
            x_int, y_int, x_frac, y_frac = int(x), int(y), x - int(x), y - int(y)
            p00, p10 = sum(img[x_int, y_int][:2]), sum(img[x_int + 1, y_int][:2])
            p01, p11 = sum(img[x_int, y_int + 1][:2]), sum(img[x_int + 1, y_int + 1][:2])
            p_top, p_bottom = p00 * (1 - x_frac) + p10 * x_frac, p01 * (1 - x_frac) + p11 * x_frac
            point_brightness = p_top * (1 - y_frac) + p_bottom * y_frac

            background_points = line((x, y), angle + math.pi / 2, 30, 30)
            background_avg = sum(sum(img[j[0], j[1]][:2]) for j in background_points) / len(background_points)

            is_dim = point_brightness <= background_avg
            is_very_bright = point_brightness > background_avg + 24
            track_data.append((is_dim, last_x, last_y, x, y, is_very_bright))
            if is_dim:
                last_x, last_y = x, y
        except (IndexError, TypeError, ZeroDivisionError):
            pass
    return track_data

def apply_directional_filter(image_obj, angle_deg, filter_width):
    """
    Applies a 1D smoothing filter to a 2D image along a specified angle.
    It expands the canvas to avoid cropping artifacts during rotation.
    """
    img_array = np.array(image_obj.convert('L'))
    original_shape = img_array.shape
    rotated_array = rotate(img_array, -angle_deg, reshape=True, mode='nearest')
    kernel = np.ones(filter_width) / filter_width
    filtered_rotated_array = convolve1d(rotated_array, kernel, axis=1, mode='nearest')
    final_rotated_back_array = rotate(filtered_rotated_array, angle_deg, reshape=True, mode='nearest')
    h_large, w_large = final_rotated_back_array.shape
    h_orig, w_orig = original_shape
    start_y = (h_large - h_orig) // 2
    start_x = (w_large - w_orig) // 2
    cropped_array = final_rotated_back_array[start_y : start_y + h_orig, start_x : start_x + w_orig]
    return Image.fromarray(cropped_array.astype(np.uint8)).convert('RGB')

def get_block_max_brightness(image_obj, center_xy):
    """Calculates the maximum brightness within a 3x3 block of pixels."""
    try:
        img_gray = image_obj.convert('L')
        img_array = np.array(img_gray)
        x, y = int(round(center_xy[0])), int(round(center_xy[1]))
        
        x_min, x_max = max(0, x - 1), min(img_array.shape[1], x + 2)
        y_min, y_max = max(0, y - 1), min(img_array.shape[0], y + 2)
        
        if x_min >= x_max or y_min >= y_max: return 0.0
        
        block = img_array[y_min:y_max, x_min:x_max]
        return np.max(block)
    except Exception:
        return 0.0

def get_block_average(image_obj, center_xy):
    """Calculates the average brightness of a 3x3 block of pixels."""
    try:
        img_gray = image_obj.convert('L')
        img_array = np.array(img_gray)
        x, y = int(round(center_xy[0])), int(round(center_xy[1]))
        x_min, x_max = max(0, x - 1), min(img_array.shape[1], x + 2)
        y_min, y_max = max(0, y - 1), min(img_array.shape[0], y + 2)
        if x_min >= x_max or y_min >= y_max: return 0.0
        block = img_array[y_min:y_max, x_min:x_max]
        return np.mean(block)
    except Exception:
        return 0.0

def estimate_saturation_brightness(image_obj, center_xy, track_angle_rad, spacing, tolerance_percent=0.05) -> float:
    """
    Performs a constrained flood-fill within a bounding box oriented along the track.
    """
    try:
        img_gray = image_obj.convert('L')
        img_array = np.array(img_gray)
        height, width = img_array.shape
        center_x, center_y = int(round(center_xy[0])), int(round(center_xy[1]))

        if not (0 <= center_x < width and 0 <= center_y < height): return 0.0
        
        start_value = float(img_array[center_y, center_x])
        tolerance_value = start_value * tolerance_percent
        min_val, max_val = start_value - tolerance_value, start_value + tolerance_value

        track_vec = np.array([math.cos(track_angle_rad), math.sin(track_angle_rad)])
        perp_vec = np.array([-math.sin(track_angle_rad), math.cos(track_angle_rad)])

        q, visited, area = deque([(center_x, center_y)]), set([(center_x, center_y)]), 0

        while q:
            x, y = q.popleft()
            area += 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    if (0 <= nx < width and 0 <= ny < height and
                            min_val <= img_array[ny, nx] <= max_val):
                        
                        disp_vec = np.array([nx - center_x, -(ny - center_y)])
                        
                        parallel_dist = np.dot(disp_vec, track_vec)
                        if abs(parallel_dist) > (spacing / 2.0):
                            continue

                        perp_dist = np.dot(disp_vec, perp_vec)
                        if abs(perp_dist) > 256:
                            continue
                        
                        q.append((nx, ny))
        return float(area)
    except Exception as e:
        print(f"Saturation brightness estimation failed: {e}", file=sys.stderr)
        return 0.0

# --- Main Execution ---

# 1. SETUP: Parse command-line arguments
parser = argparse.ArgumentParser(description="Find the accurate start and end positions of a meteor track.")
parser.add_argument('-r', '--radius', dest='radius', help='Search radius in pixels (default: 100)', default=100, type=int)
parser.add_argument('--frames', dest='frames', help='Number of stacked frames to estimate brightness for', type=int, default=None)
parser.add_argument(action='store', dest='img', help='Image file of the meteor')
parser.add_argument(action='store', dest='start', help="Approximate start position 'x,y'", type=coords)
parser.add_argument(action='store', dest='end', help="Approximate end position 'x,y'", type=coords)
args = parser.parse_args()

# Load image and initial properties
im = Image.open(args.img)
img_pixels = im.load()
# Convert to luma numpy array once — used by all JIT-compiled hot paths
img_y = np.array(im.convert('L'), dtype=np.float32)
x1, y1 = args.start
x2, y2 = args.end
initial_angle = math.atan2(y2 - y1, x2 - x1)
initial_length = dist(args.start, args.end)

# 2. PHASE 1: FAST APPROXIMATION (JIT-compiled exhaustive search)
start_search_line = line(args.start, initial_angle - math.pi / 2, args.radius * 2, args.radius)
end_search_line = line(args.end, initial_angle - math.pi / 2, args.radius * 2, args.radius)
starts_x = np.array([p[0] for p in start_search_line], dtype=np.float64)
starts_y = np.array([p[1] for p in start_search_line], dtype=np.float64)
ends_x   = np.array([p[0] for p in end_search_line],   dtype=np.float64)
ends_y   = np.array([p[1] for p in end_search_line],   dtype=np.float64)
steps_p1 = max(1, int(initial_length / 5))
best_si, best_ei = _phase1_search_nb(img_y, starts_x, starts_y, ends_x, ends_y, steps_p1)
best_start_approx = (float(starts_x[best_si]), float(starts_y[best_si]))
best_end_approx   = (float(ends_x[best_ei]),   float(ends_y[best_ei]))

# 3. PHASE 2: DETAILED REFINEMENT (JIT-compiled track scan)
refined_angle = math.atan2(best_end_approx[1] - best_start_approx[1], best_end_approx[0] - best_start_approx[0])
extended_start = (best_start_approx[0] + math.cos(refined_angle + math.pi) * initial_length * 10,
                  best_start_approx[1] + math.sin(refined_angle + math.pi) * initial_length * 10)
extended_end   = (best_end_approx[0] + math.cos(refined_angle) * initial_length * 10,
                  best_end_approx[1] + math.sin(refined_angle) * initial_length * 10)
extended_len = dist(extended_start, extended_end)
steps_p2 = max(1, int(extended_len * 10))
is_dim_a, lx_a, ly_a, cx_a, cy_a, ib_a = _track_nb(
    img_y, extended_start[0], extended_start[1], extended_end[0], extended_end[1], steps_p2)
first_bright_point, last_bright_point = None, None
for idx in range(len(ib_a)):
    if ib_a[idx]:
        last_bright_point = (float(cx_a[idx]), float(cy_a[idx]))
        if first_bright_point is None:
            first_bright_point = (float(cx_a[idx]), float(cy_a[idx]))
if first_bright_point: best_start_approx = first_bright_point
if last_bright_point:  best_end_approx   = last_bright_point

# 4. PHASE 3: FINAL PINPOINTING (JIT-compiled exhaustive search)
args.radius = max(80, args.radius)
start_search_line = line(best_start_approx, refined_angle - math.pi / 2, args.radius / 40, args.radius / 8)
end_search_line   = line(best_end_approx,   refined_angle - math.pi / 2, args.radius / 40, args.radius / 8)
starts_x = np.array([p[0] for p in start_search_line], dtype=np.float64)
starts_y = np.array([p[1] for p in start_search_line], dtype=np.float64)
ends_x   = np.array([p[0] for p in end_search_line],   dtype=np.float64)
ends_y   = np.array([p[1] for p in end_search_line],   dtype=np.float64)
steps_p3 = max(1, int(initial_length * 2))
best_si, best_ei = _phase1_search_nb(img_y, starts_x, starts_y, ends_x, ends_y, steps_p3)
best_start_approx = (float(starts_x[best_si]), float(starts_y[best_si]))
best_end_approx   = (float(ends_x[best_ei]),   float(ends_y[best_ei]))

# 5. FIND LONGEST BRIGHT SEGMENT (JIT-compiled track scan)
steps_p5 = max(1, int(dist(best_start_approx, best_end_approx) * 10))
is_dim_a5, lx_a5, ly_a5, cx_a5, cy_a5, ib_a5 = _track_nb(
    img_y, best_start_approx[0], best_start_approx[1],
    best_end_approx[0], best_end_approx[1], steps_p5)
run_length, bright_segments = 0, []
for idx in range(len(is_dim_a5)):
    if is_dim_a5[idx]:
        bright_segments.append((run_length, float(lx_a5[idx]), float(ly_a5[idx]), float(cx_a5[idx]), float(cy_a5[idx])))
        run_length = 0
    else:
        run_length += 1
if run_length > 0 and len(cx_a5) > 0:
    bright_segments.append((run_length, float(lx_a5[-1]), float(ly_a5[-1]), float(cx_a5[-1]), float(cy_a5[-1])))

# --- DECISION LOGIC & FINAL OUTPUT ---
if bright_segments:
    longest_segment = max(bright_segments, key=lambda x: x[0])
    candidate_start, candidate_end = (longest_segment[1], longest_segment[2]), (longest_segment[3], longest_segment[4])
    if initial_length > dist(candidate_start, candidate_end) * 1.5:
        candidate_start, candidate_end = best_start_approx, best_end_approx

    # --- VALIDATION AND LIMITING ---
    revert_to_original = False
    new_angle_deg = math.degrees(math.atan2(candidate_end[1] - candidate_start[1], candidate_end[0] - candidate_start[0]))
    angle_diff = abs(math.degrees(initial_angle) - new_angle_deg)
    if angle_diff > 180: angle_diff = 360 - angle_diff
    if angle_diff > 5: revert_to_original = True

    if revert_to_original:
        print(f'{args.start[0]:.2f},{args.start[1]:.2f} {args.end[0]:.2f},{args.end[1]:.2f}')
    else:
        final_start, final_end = candidate_start, candidate_end
        p1, p2 = np.array(candidate_start), np.array(candidate_end)
        line_vec, line_len = p2 - p1, np.linalg.norm(p2 - p1)
        # Initialize closest points for brightness estimation
        closest_pt_to_orig_start, closest_pt_to_orig_end = p1, p2
        if line_len > 0:
            line_unitvec = line_vec / line_len
            orig_start_pt, orig_end_pt = np.array(args.start), np.array(args.end)
            t_start = np.dot(orig_start_pt - p1, line_unitvec)
            closest_pt_to_orig_start = p1 + t_start * line_unitvec
            t_end = np.dot(orig_end_pt - p1, line_unitvec)
            closest_pt_to_orig_end = p1 + t_end * line_unitvec
            if np.dot(p1 - closest_pt_to_orig_start, line_unitvec) < 0 and np.linalg.norm(p1 - closest_pt_to_orig_start) > args.radius:
                final_start = tuple(closest_pt_to_orig_start - args.radius * line_unitvec)
            if np.dot(p2 - closest_pt_to_orig_end, line_unitvec) > 0 and np.linalg.norm(p2 - closest_pt_to_orig_end) > args.radius:
                final_end = tuple(closest_pt_to_orig_end + args.radius * line_unitvec)

        # Print final coordinates, keeping output on one line if --frames is used
        print(f'{final_start[0]:.2f},{final_start[1]:.2f} {final_end[0]:.2f},{final_end[1]:.2f}', end='')

        # --- OPTIONAL BRIGHTNESS ESTIMATION (DIRECTIONAL SMOOTHING + CONSTRAINED FLOOD-FILL) ---
        if args.frames and args.frames > 0:
            brightness_list = []
            seg_start, seg_end = tuple(closest_pt_to_orig_start), tuple(closest_pt_to_orig_end)
            segment_length = dist(seg_start, seg_end)
            dy, dx = seg_end[1] - seg_start[1], seg_end[0] - seg_start[0]
            track_angle_rad = math.atan2(-dy, dx)

            if segment_length > 1:
                spacing = segment_length / (args.frames - 1) if args.frames > 1 else 7.0
                filter_length = spacing * 2
                filter_width = max(3, int(round(filter_length)) // 2 * 2 + 1)
                
                print("\nApplying directional filter...", file=sys.stderr, end='', flush=True)
                filtered_image = apply_directional_filter(im, math.degrees(track_angle_rad), filter_width)
                print(" Done.", file=sys.stderr)
                
                try:
                    debug_filename = f"{Path(args.img).stem}-filtered.jpg"
                    filtered_image.save(debug_filename, quality=95)
                    print(f"Debug: Saved filtered image to {debug_filename}", file=sys.stderr)
                except Exception as e:
                    print(f"Debug: Could not save filtered image: {e}", file=sys.stderr)

                for i in range(args.frames):
                    t = i / (args.frames - 1) if args.frames > 1 else 0.5
                    point_to_measure = (seg_start[0] + t * (seg_end[0] - seg_start[0]),
                                        seg_start[1] + t * (seg_end[1] - seg_start[1]))
                    
                    # --- ADDITIVE BRIGHTNESS LOGIC ---
                    # 1. Check for saturation on the ORIGINAL image using the MAX pixel in a 3x3 block
                    max_brightness_orig = get_block_max_brightness(im, point_to_measure)
                    saturation_threshold = 255 * 0.95

                    # 2. Get the base brightness from the AVERAGE of the 3x3 block on the FILTERED image
                    base_brightness = get_block_average(filtered_image, point_to_measure)
                    brightness_val = base_brightness
                    
                    # 3. If the original was saturated, ADD the flood-fill area to the base brightness
                    if max_brightness_orig >= saturation_threshold:
                        saturation_area = estimate_saturation_brightness(filtered_image, point_to_measure, track_angle_rad, spacing)
                        brightness_val += saturation_area
                    
                    brightness_list.append(int(brightness_val))
            
            if len(brightness_list) != args.frames:
                brightness_list = [0] * args.frames

            brightness_str = " ".join(map(str, brightness_list))
            print(f' {brightness_str}')
        else:
            print()
else:
    print(f'{args.start[0]:.2f},{args.start[1]:.2f} {args.end[0]:.2f},{args.end[1]:.2f}')
