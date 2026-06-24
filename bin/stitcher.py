#!/usr/bin/env python3

import sys
import os
import numpy as np
import numba
from numba import prange
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import math
import argparse
import warnings
import datetime
import json
import re
import tempfile
import glob
try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None

# --- Dependency Imports with User-Friendly Error Handling ---

from pathlib import Path

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
    import pto_mapper
except ImportError as e:
    raise ImportError(
        "The required 'pto_mapper.py' module was not found. "
        "Please ensure 'pto_mapper.py' is in the same directory as this script."
    ) from e

try:
    from stack import enhance_filter
except ImportError as e:
    raise ImportError(
        "The 'stack.py' module was not found. "
        "Please ensure 'stack.py' (containing the enhancement filter) is in the same directory."
    ) from e

# The timestamp import is now deferred to the video processing function where it is needed.

try:
    import av
except ImportError:
    print("Warning: 'PyAV' library not found. Video processing functionality will be unavailable.", file=sys.stderr)
    av = None

try:
    import scipy.ndimage as ndimage
except ImportError:
    print("Warning: 'scipy' not found. Noise estimation will be unavailable. Run 'pip install scipy'", file=sys.stderr)
    ndimage = None

# Import multiblend for blending functionality
try:
    import multiblend
except ImportError as e:
    raise ImportError(
        "The 'multiblend' module is required. Please ensure multiblend.py is available."
    ) from e

# Global quiet flag. When True, all normal text output is suppressed.
_quiet = False


def _print(*args, **kwargs):
    """Print wrapper that respects the global _quiet flag."""
    if not _quiet:
        print(*args, **kwargs)


@numba.njit(parallel=True, fastmath=True, cache=True)
def _yuv420_to_rgb_kernel(y_flat, u_flat, v_flat, r_out, g_out, b_out, h, w, uv_w):
    """BT.601 YUV420 -> RGB in a single parallel pass. All arrays are flat (row-major)."""
    for yi in prange(h):
        uv_row = (yi >> 1) * uv_w
        y_row  = yi * w
        for xi in range(w):
            yv = np.float32(y_flat[y_row + xi])
            uv = np.float32(u_flat[uv_row + (xi >> 1)]) - np.float32(128)
            vv = np.float32(v_flat[uv_row + (xi >> 1)]) - np.float32(128)
            rv = yv + np.float32(1.402)  * vv
            gv = yv - np.float32(0.344136) * uv - np.float32(0.714136) * vv
            bv = yv + np.float32(1.772)  * uv
            r_out[y_row + xi] = np.uint8(min(np.float32(255), max(np.float32(0), rv)))
            g_out[y_row + xi] = np.uint8(min(np.float32(255), max(np.float32(0), gv)))
            b_out[y_row + xi] = np.uint8(min(np.float32(255), max(np.float32(0), bv)))

def yuv_to_rgb(y_plane, u_plane, v_plane):
    """Convert YUV420 planes to RGB using the standard BT.601 conversion (Numba JIT)."""
    h, w = y_plane.shape
    uv_h, uv_w = u_plane.shape
    r_out = np.empty((h, w), dtype=np.uint8)
    g_out = np.empty((h, w), dtype=np.uint8)
    b_out = np.empty((h, w), dtype=np.uint8)
    _yuv420_to_rgb_kernel(
        np.ascontiguousarray(y_plane).ravel(),
        np.ascontiguousarray(u_plane).ravel(),
        np.ascontiguousarray(v_plane).ravel(),
        r_out.ravel(), g_out.ravel(), b_out.ravel(),
        h, w, uv_w)
    return r_out, g_out, b_out

def create_image_info_from_yuv(y_plane, u_plane, v_plane, weight_map=None, xpos=0, ypos=0):
    """Create a multiblend ImageInfo object from YUV planes with optional weight-based mask."""
    r, g, b = yuv_to_rgb(y_plane, u_plane, v_plane)
    h, w = y_plane.shape
    mask = (weight_map > 1e-9) if weight_map is not None else np.ones((h, w), dtype=bool)
    return multiblend.ImageInfo(
        filename="", bpp=8, width=w, height=h, xpos=xpos, ypos=ypos,
        channels=[r, g, b], mask=mask,
    )

@numba.njit(parallel=True, fastmath=True, cache=True)
def _rgb_to_yuv420_kernel(r_flat, g_flat, b_flat, y_out, u_out, v_out, h, w):
    """BT.601 RGB -> YUV420 in a single parallel pass.
    y_out is h*w, u_out/v_out are (h//2)*(w//2)."""
    uv_w = w >> 1
    for yi in prange(h):
        row = yi * w
        for xi in range(w):
            rv = np.float32(r_flat[row + xi])
            gv = np.float32(g_flat[row + xi])
            bv = np.float32(b_flat[row + xi])
            yv = np.float32(0.299)*rv + np.float32(0.587)*gv + np.float32(0.114)*bv
            y_out[row + xi] = np.uint8(min(np.float32(255), max(np.float32(0), yv)))
            if (yi & 1) == 0 and (xi & 1) == 0:
                uv = -np.float32(0.169)*rv - np.float32(0.331)*gv + np.float32(0.5)*bv + np.float32(128)
                vv =  np.float32(0.5)*rv   - np.float32(0.419)*gv - np.float32(0.081)*bv + np.float32(128)
                idx = (yi >> 1) * uv_w + (xi >> 1)
                u_out[idx] = np.uint8(min(np.float32(255), max(np.float32(0), uv)))
                v_out[idx] = np.uint8(min(np.float32(255), max(np.float32(0), vv)))

def rgb_to_yuv(rgb_channels):
    """Convert RGB channels to YUV420 planes (Numba JIT)."""
    h, w = rgb_channels[0].shape
    y_out  = np.empty((h, w),          dtype=np.uint8)
    # Use ceiling dimensions for chroma planes so the kernel never writes
    # out of bounds when the luma height or width is odd.
    u_out  = np.empty(((h + 1) // 2, (w + 1) // 2), dtype=np.uint8)
    v_out  = np.empty(((h + 1) // 2, (w + 1) // 2), dtype=np.uint8)
    _rgb_to_yuv420_kernel(
        np.ascontiguousarray(rgb_channels[0]).ravel(),
        np.ascontiguousarray(rgb_channels[1]).ravel(),
        np.ascontiguousarray(rgb_channels[2]).ravel(),
        y_out.ravel(), u_out.ravel(), v_out.ravel(), h, w)
    return y_out, u_out, v_out


# --- Numba JIT-compiled Core Functions ---

@numba.njit(parallel=True, fastmath=True, cache=True)
def create_blend_weight_map(width, height):
    weights = np.empty((height, width), dtype=np.float32)
    norm = min(width, height) / 2.0
    if norm == 0: norm = 1.0
    
    for y in prange(height):
        dist_y = min(y, height - 1 - y)
        for x in range(width):
            dist_x = min(x, width - 1 - x)
            # Calculate the weight
            raw_weight = min(dist_x, dist_y) / norm
            # Manually clip the value to the [0.0, 1.0] range before assignment
            weights[y, x] = max(0.0, min(1.0, raw_weight))
            
    return weights

@numba.njit(parallel=True, fastmath=True, cache=True)
def _blur_padded_area_numba(plane, pad_t, pad_b, pad_l, pad_r, blur_kernel_size, noise_amplitude):
    """Applies a 2-pass blur and a final noise pass for a natural, textured effect."""
    h, w = plane.shape
    if pad_t <= 0 and pad_b <= 0 and pad_l <= 0 and pad_r <= 0:
        return plane.astype(np.uint8)

    # The first pass kernel is capped at a maximum of 16 pixels.
    pass1_kernel_size = min(blur_kernel_size, 16)
    if pass1_kernel_size < 1: pass1_kernel_size = 1

    # --- Pass 1: Smear edges into padding ---
    pass1_plane = plane.copy()

    # Pass 1: Top blur (vertical smear)
    if pad_t > 0:
        for c in prange(w):
            for r in range(pad_t):
                acc = 0.0
                for k in range(pass1_kernel_size):
                    y = min(r + k, h - 1)
                    acc += plane[y, c]
                pass1_plane[r, c] = acc / pass1_kernel_size

    # Pass 1: Bottom blur (vertical smear)
    if pad_b > 0:
        for c in prange(w):
            for r in range(h - pad_b, h):
                acc = 0.0
                for k in range(pass1_kernel_size):
                    y = max(r - k, 0)
                    acc += plane[y, c]
                pass1_plane[r, c] = acc / pass1_kernel_size

    # Pass 1: Left blur (horizontal smear)
    if pad_l > 0:
        for r in prange(h):
            for c in range(pad_l):
                acc = 0.0
                for k in range(pass1_kernel_size):
                    x = min(c + k, w - 1)
                    acc += plane[r, x]
                pass1_plane[r, c] = acc / pass1_kernel_size

    # Pass 1: Right blur (horizontal smear)
    if pad_r > 0:
        for r in prange(h):
            for c in range(w - pad_r, w):
                acc = 0.0
                for k in range(pass1_kernel_size):
                    x = max(c - k, 0)
                    acc += plane[r, x]
                pass1_plane[r, c] = acc / pass1_kernel_size

    # --- Pass 2: Smooth with a dynamic kernel size for a graduated blur ---
    blurred_plane = pass1_plane.copy()
    base_kernel_size = blur_kernel_size if blur_kernel_size > 0 else 1

    # Pass 2: Top pad (horizontal box blur with increasing kernel size)
    if pad_t > 0:
        for r in prange(pad_t):
            distance = pad_t - r
            dynamic_kernel_size = base_kernel_size + distance
            for c in range(w):
                acc = 0.0
                for k in range(dynamic_kernel_size):
                    x = max(0, min(c - dynamic_kernel_size // 2 + k, w - 1))
                    acc += pass1_plane[r, x]
                blurred_plane[r, c] = acc / dynamic_kernel_size

    # Pass 2: Bottom pad (horizontal box blur with increasing kernel size)
    if pad_b > 0:
        for r in prange(h - pad_b, h):
            distance = r - (h - pad_b - 1)
            dynamic_kernel_size = base_kernel_size + distance
            for c in range(w):
                acc = 0.0
                for k in range(dynamic_kernel_size):
                    x = max(0, min(c - dynamic_kernel_size // 2 + k, w - 1))
                    acc += pass1_plane[r, x]
                blurred_plane[r, c] = acc / dynamic_kernel_size

    # Pass 2: Left pad (vertical box blur with increasing kernel size)
    if pad_l > 0:
        for c in prange(pad_l):
            distance = pad_l - c
            dynamic_kernel_size = base_kernel_size + distance
            for r in range(h):
                acc = 0.0
                for k in range(dynamic_kernel_size):
                    y = max(0, min(r - dynamic_kernel_size // 2 + k, h - 1))
                    acc += pass1_plane[y, c]
                blurred_plane[r, c] = acc / dynamic_kernel_size

    # Pass 2: Right pad (vertical box blur with increasing kernel size)
    if pad_r > 0:
        for c in prange(w - pad_r, w):
            distance = c - (w - pad_r - 1)
            dynamic_kernel_size = base_kernel_size + distance
            for r in range(h):
                acc = 0.0
                for k in range(dynamic_kernel_size):
                    y = max(0, min(r - dynamic_kernel_size // 2 + k, h - 1))
                    acc += pass1_plane[y, c]
                blurred_plane[r, c] = acc / dynamic_kernel_size

    # --- Pass 3: Add slight noise to break up smoothness ---
    if noise_amplitude > 0:
        # Top region
        if pad_t > 0:
            for r in prange(pad_t):
                for c in range(w):
                    noise = np.random.uniform(-noise_amplitude, noise_amplitude)
                    blurred_plane[r, c] += noise
        # Bottom region
        if pad_b > 0:
            for r in prange(h - pad_b, h):
                for c in range(w):
                    noise = np.random.uniform(-noise_amplitude, noise_amplitude)
                    blurred_plane[r, c] += noise
        # Left region (excluding corners)
        if pad_l > 0:
            for c in prange(pad_l):
                for r in range(pad_t, h - pad_b):
                    noise = np.random.uniform(-noise_amplitude, noise_amplitude)
                    blurred_plane[r, c] += noise
        # Right region (excluding corners)
        if pad_r > 0:
            for c in prange(w - pad_r, w):
                for r in range(pad_t, h - pad_b):
                    noise = np.random.uniform(-noise_amplitude, noise_amplitude)
                    blurred_plane[r, c] += noise

    # --- Final Manual Clip and Type Conversion ---
    final_plane = np.empty_like(blurred_plane, dtype=np.uint8)
    for i in prange(h):
        for j in range(w):
            val = blurred_plane[i, j]
            if val < 0: final_plane[i, j] = 0
            elif val > 255: final_plane[i, j] = 255
            else: final_plane[i, j] = val

    return final_plane


@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def reproject_y(py, dw, dh, sw, map_y_idx, c01, c23, out_y):
    for yi in prange(dh):
        base_out = yi * dw
        base_map = yi * dw
        for xi in range(dw):
            idx = map_y_idx[base_map + xi]
            if idx < 0: continue
            weights01, weights23 = c01[base_map + xi], c23[base_map + xi]
            w0, w1, w2, w3 = (weights01 >> 8) & 0xFF, weights01 & 0xFF, (weights23 >> 8) & 0xFF, weights23 & 0xFF
            interpolated_value = (py[idx] * w0 + py[idx + 1] * w1 + py[idx + sw] * w2 + py[idx + sw + 1] * w3) >> 7
            out_y[base_out + xi] = interpolated_value

@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def reproject_float(p_float_src, dw, dh, sw, map_y_idx, c01, c23, out_float):
    for yi in prange(dh):
        base_out = yi * dw
        base_map = yi * dw
        for xi in range(dw):
            idx = map_y_idx[base_map + xi]
            if idx < 0: continue
            weights01, weights23 = c01[base_map + xi], c23[base_map + xi]
            w0, w1, w2, w3 = (weights01 >> 8) & 0xFF, weights01 & 0xFF, (weights23 >> 8) & 0xFF, weights23 & 0xFF
            interpolated_value = (p_float_src[idx] * w0 + p_float_src[idx + 1] * w1 + p_float_src[idx + sw] * w2 + p_float_src[idx + sw + 1] * w3) / 128.0
            out_float[base_out + xi] = interpolated_value

@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def reproject_uv(pu, pv, dw, dh, map_uv_idx, out_u, out_v):
    half_w, half_h = dw // 2, dh // 2
    for y_uv in prange(half_h):
        base_uv = y_uv * half_w
        for x_uv in range(half_w):
            coffset = map_uv_idx[base_uv + x_uv]
            if coffset >= 0:
                out_u[base_uv + x_uv], out_v[base_uv + x_uv] = pu[coffset], pv[coffset]
            else:
                out_u[base_uv + x_uv] = 128
                out_v[base_uv + x_uv] = 128

@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def compute_map_and_weights(coords_y, sw, sh, pad_t, pad_b, pad_l, pad_r):
    dh, dw, _ = coords_y.shape
    map_y_idx, c01, c23 = np.full(dh*dw,-1,dtype=np.int32), np.zeros(dh*dw,dtype=np.uint16), np.zeros(dh*dw,dtype=np.uint16)
    effective_sw, effective_sh = sw + pad_l + pad_r, sh + pad_t + pad_b
    for idx in prange(dh * dw):
        y_dest, x_dest = idx // dw, idx % dw
        dx_orig, dy_orig = coords_y[y_dest, x_dest, 0], coords_y[y_dest, x_dest, 1]
        if dx_orig <= -99999.0: continue
        dx, dy = dx_orig + pad_l, dy_orig + pad_t
        if 0 <= dx < effective_sw - 1 and 0 <= dy < effective_sh - 1:
            xi, yi = int(dx), int(dy)
            xf, yf = dx - xi, dy - yi
            c0, c1, c2, c3 = int((1-xf)*(1-yf)*128+.5), int(xf*(1-yf)*128+.5), int((1-xf)*yf*128+.5), int(xf*yf*128+.5)
            diff, sel = 128-(c0+c1+c2+c3), (1 if xf>=.5 else 0)+2*(1 if yf>=.5 else 0)
            if sel == 0: c0 += diff
            elif sel == 1: c1 += diff
            elif sel == 2: c2 += diff
            else: c3 += diff
            map_y_idx[idx], c01[idx], c23[idx] = yi*effective_sw+xi, (c0<<8)|c1, (c2<<8)|c3
    return map_y_idx, c01.reshape(dh, dw), c23.reshape(dh, dw)

@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def compute_uv_map(coords_uv, sw_uv, sh_uv, pad_uv_t, pad_uv_b, pad_uv_l, pad_uv_r):
    h_uv, w_uv, _ = coords_uv.shape
    map_uv_idx = np.full(h_uv * w_uv, -1, dtype=np.int32)
    effective_sw_uv, effective_sh_uv = sw_uv + pad_uv_l + pad_uv_r, sh_uv + pad_uv_t + pad_uv_b
    for idx in prange(h_uv * w_uv):
        y_uv, x_uv = idx // w_uv, idx % w_uv
        sx_uv_orig, sy_uv_orig = coords_uv[y_uv, x_uv, 0], coords_uv[y_uv, x_uv, 1]
        if sx_uv_orig <= -99999.0: continue
        sx_uv, sy_uv = sx_uv_orig + pad_uv_l, sy_uv_orig + pad_uv_t
        if 0 <= sx_uv < effective_sw_uv and 0 <= sy_uv < effective_sh_uv:
            map_uv_idx[idx] = int(sy_uv) * effective_sw_uv + int(sx_uv)
    return map_uv_idx.reshape(h_uv, w_uv)

def _map_one_image(args):
    img, pad, final_w, final_h, orig_w, orig_h, crop_offset_x, crop_offset_y, pano_proj_f, pano_hfov, pano_r, pano_s, padsides = args
    sw,sh,fov,src_proj_f = img.get('w'),img.get('h'),img.get('v'),int(img.get('f',0))
    if sw is None or sh is None: raise ValueError("Image must have width 'w' and height 'h'.")
    if fov is None: raise ValueError("Image must have HFOV 'v'.")
    fov_rad, src_focal = math.radians(fov), 0
    if src_proj_f == 0: src_focal = sw / (2*math.tan(fov_rad/2)) if fov_rad > 0 else 0
    elif src_proj_f == 3: src_focal = sw / fov_rad if fov_rad > 0 else 0
    else: raise ValueError(f"Unsupported source image projection f{src_proj_f}")

    src_norm_radius,y,p,r = min(sw,sh)/2.,img.get('y',0),img.get('p',0),-img.get('r',0)
    a,b,c,cx,cy = img.get('a',0),img.get('b',0),img.get('c',0),-img.get('d',0),img.get('e',0)

    R_pr = pto_mapper.create_pr_rotation_matrix(p,r)
    R_pr_inv = R_pr.T
    coords_y = np.empty((final_h, final_w, 2), dtype=np.float32)

    pto_mapper.calculate_source_coords(coords_y,final_w,final_h,orig_w,orig_h,crop_offset_x,crop_offset_y,pano_proj_f,pano_hfov,sw,sh,R_pr_inv,y,src_focal,src_norm_radius,a,b,c,cx,cy,src_proj_f,pano_r,pano_s)

    pad_t = pad if 'top' in padsides else 0
    pad_b = pad if 'bottom' in padsides else 0
    pad_l = pad if 'left' in padsides else 0
    pad_r = pad if 'right' in padsides else 0

    map_y_idx, c01, c23 = compute_map_and_weights(coords_y, sw, sh, pad_t, pad_b, pad_l, pad_r)
    coords_uv = coords_y[::2,::2]/2.
    map_uv_idx = compute_uv_map(coords_uv, sw//2, sh//2, pad_t//2, pad_b//2, pad_l//2, pad_r//2)

    return (map_y_idx, c01, c23, map_uv_idx, sw, sh)


def build_mappings(pto_file, pad, num_workers, padsides, is_video_output=False):
    try:
        global_options, images = pto_mapper.parse_pto_file(pto_file)
    except Exception as e:
        raise ValueError(f"Failed to parse PTO file '{pto_file}'. Reason: {e}")

    orig_w, orig_h = global_options.get('w'), global_options.get('h')
    if orig_w is None or orig_h is None: raise ValueError("PTO 'p' line must contain canvas width 'w' and height 'h'.")
    
    # Get panorama settings with safe defaults
    pano_s = global_options.get('s', 1.0)
    pano_proj_f = int(global_options.get('f', 2))
    pano_hfov = global_options.get('v')
    pano_r = global_options.get('r', 0.0)

    if pano_s <= 0: raise ValueError("PTO panorama scale factor 's' must be greater than 0.")
    if pano_hfov is None: raise ValueError("PTO 'p' line must have HFOV 'v' for projection calculations.")

    # Get crop coordinates. If 'S' line is missing, default to the full canvas size.
    crop_coords = global_options.get('S')
    if crop_coords:
        # If 'S' line exists, use its values in the correct L, T, R, B order
        left, top, right, bottom = crop_coords
    else:
        _print("INFO: No crop 'S' line found in PTO file. Using full canvas dimensions.")
        left, top, right, bottom = 0, 0, orig_w, orig_h

    # Calculate final dimensions based on crop and scale
    final_w = int(round((right - left) * pano_s))
    final_h = int(round((bottom - top) * pano_s))
    
    # --- CRITICAL VALIDATION WITH DETAILED FEEDBACK ---
    if final_w <= 0 or final_h <= 0:
        error_details = (
            f"Calculated final panorama dimensions are invalid: {final_w}x{final_h}.\n\n"
            f"🕵️ Here's how these dimensions were calculated from your PTO file:\n"
            f"  - Crop Box (Left, Right, Top, Bottom): ({left}, {right}, {top}, {bottom})\n"
            f"  - Scale Factor ('s'): {pano_s}\n"
            f"  - Width Formula: (Right - Left) * Scale = ({right} - {left}) * {pano_s} = {final_w}\n"
            f"  - Height Formula: (Bottom - Top) * Scale = ({bottom} - {top}) * {pano_s} = {final_h}\n\n"
            f"The error is because the crop width (Right - Left) or height (Bottom - Top) is zero or negative.\n"
            f"Please correct the 'S' line in '{os.path.basename(pto_file)}'."
        )
        raise ValueError(error_details)

    # Ensure coordinates are even for YUV processing compatibility
    left &= ~1; top &= ~1; right &= ~1; bottom &= ~1
    
    if is_video_output:
        if final_h % 2 != 0: final_h -= 1
        if final_w % 32 != 0:
            original_w = final_w
            final_w = ((original_w + 31) // 32) * 32
            _print(f"Warning: Output width {original_w} is not optimal for video. Adjusting to {final_w} for codec compatibility.", file=sys.stderr)

    crop_offset_x = left
    crop_offset_y = top

    global_options['final_w'], global_options['final_h'] = final_w, final_h

    task_args = [(img, pad, final_w, final_h, orig_w, orig_h, crop_offset_x, crop_offset_y, pano_proj_f, pano_hfov, pano_r, pano_s, padsides) for img in images]

    _print("Building projection maps...")
    all_mappings = [_map_one_image(args) for args in task_args]

    return all_mappings, global_options

def estimate_noise(image_plane):
    """
    Estimates the noise standard deviation of an image plane using the
    standard deviation of its Laplacian. Returns a value between 1.0 and 10.0.
    """
    if ndimage is None:
        # Fallback to a default value if scipy is not installed
        return 4.0

    # The Laplacian filter is sensitive to high-frequency noise
    laplacian = ndimage.laplace(image_plane.astype(np.float32))
    
    # The standard deviation of the Laplacian is a robust noise estimator
    noise_std = np.std(laplacian)
    
    # Clamp the value to a reasonable range to avoid extreme results
    return np.clip(noise_std, 1.0, 10.0)

def load_image_to_yuv(image_path, pad, padsides):
    # Add a compatibility check for different Pillow versions
    try:
        resample_filter = Image.Resampling.BICUBIC
    except AttributeError:
        # Fallback for older versions
        resample_filter = Image.BICUBIC

    try:
        img_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input image not found at path: {image_path}")
    except Exception as e:
        raise IOError(f"Could not open or read image file '{image_path}'. Reason: {e}")
        
    img_ycbcr = img_pil.convert("YCbCr")
    y, u, v = img_ycbcr.split()
    
    # Use the determined resampling filter
    u_resized = u.resize((img_pil.width // 2, img_pil.height // 2), resample_filter)
    v_resized = v.resize((img_pil.width // 2, img_pil.height // 2), resample_filter)
    
    pad_t = pad if 'top' in padsides else 0
    pad_b = pad if 'bottom' in padsides else 0
    pad_l = pad if 'left' in padsides else 0
    pad_r = pad if 'right' in padsides else 0

    pad_y_width = ((pad_t, pad_b), (pad_l, pad_r))
    pad_uv_width = ((pad_t // 2, pad_b // 2), (pad_l // 2, pad_r // 2))

    y_arr = np.array(y, np.uint8)
    padded_y = np.pad(y_arr, pad_y_width, mode='edge')
    padded_u = np.pad(np.array(u_resized, np.uint8), pad_uv_width, mode='edge')
    padded_v = np.pad(np.array(v_resized, np.uint8), pad_uv_width, mode='edge')

    pad_uv_t, pad_uv_b, pad_uv_l, pad_uv_r = pad_t//2, pad_b//2, pad_l//2, pad_r//2

    if pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0:
        noise_level = estimate_noise(y_arr) / 4.0
        blur_size = 96
        padded_y = _blur_padded_area_numba(padded_y.astype(np.float32), pad_t, pad_b, pad_l, pad_r, blur_size, noise_level)
        if pad_uv_t > 0 or pad_uv_b > 0 or pad_uv_l > 0 or pad_uv_r > 0:
            blur_size_uv = blur_size // 2
            padded_u = _blur_padded_area_numba(padded_u.astype(np.float32), pad_uv_t, pad_uv_b, pad_uv_l, pad_uv_r, blur_size_uv, noise_level)
            padded_v = _blur_padded_area_numba(padded_v.astype(np.float32), pad_uv_t, pad_uv_b, pad_uv_l, pad_uv_r, blur_size_uv, noise_level)

    target_h_y = img_pil.height + pad_t + pad_b
    target_h_uv = img_pil.height // 2 + pad_uv_t + pad_uv_b
    
    return (padded_y[:target_h_y, :], padded_u[:target_h_uv, :], padded_v[:target_h_uv, :], img_pil.width, img_pil.height)


def save_image_yuv420(y_plane, u_plane, v_plane, output_path):
    # --- Robustness Check ---
    if y_plane is None or y_plane.size == 0:
        raise ValueError("Cannot save image: The final luma (Y) plane is empty.")

    try:
        # Add a compatibility check for different Pillow versions
        try:
            resample_filter = Image.Resampling.BICUBIC
        except AttributeError:
            # Fallback for older versions
            resample_filter = Image.BICUBIC
            
        h, w = y_plane.shape
        y_img = Image.fromarray(y_plane,"L")
        u_img = Image.fromarray(u_plane,"L").resize((w,h), resample_filter)
        v_img = Image.fromarray(v_plane,"L").resize((w,h), resample_filter)
        Image.merge("YCbCr", (y_img,u_img,v_img)).convert("RGB").save(output_path, "JPEG", quality=95)
    except Exception as e:
        # Catch errors from Pillow (e.g., cannot write empty image) or filesystem (permission denied)
        raise IOError(f"Failed to save the final image to '{output_path}'. Reason: {e}")


_TIMESTAMP_BOX_HD = (0, 1040, 305, 1080)   # (x1, y1, x2, y2) for height >= 900
_TIMESTAMP_BOX_SD = (0,  430, 155,  448)   # for height < 900


def process_and_reproject_image(args):
    """Worker function to reproject a single image, writing to pre-allocated buffers."""
    (input_path, dw, dh, mapping, pad, padsides), out_buffers = args
    reproj_y, reproj_u, reproj_v, reproj_weights_y = out_buffers

    py, pu, pv, sw_orig, sh_orig = load_image_to_yuv(input_path, pad, padsides)

    pad_t = pad if 'top' in padsides else 0
    pad_b = pad if 'bottom' in padsides else 0
    pad_l = pad if 'left' in padsides else 0
    pad_r = pad if 'right' in padsides else 0

    x1, y1, x2, y2 = _TIMESTAMP_BOX_HD if sh_orig >= 900 else _TIMESTAMP_BOX_SD
    # Shift into padded image space
    py_y1, py_y2 = y1 + pad_t, y2 + pad_t
    py_x1, py_x2 = x1 + pad_l, x2 + pad_l
    py[py_y1:py_y2, py_x1:py_x2] = 0
    pu[py_y1//2:py_y2//2, py_x1//2:py_x2//2] = 128
    pv[py_y1//2:py_y2//2, py_x1//2:py_x2//2] = 128

    sw_padded, sh_padded = sw_orig + pad_l + pad_r, sh_orig + pad_t + pad_b

    # Weight map for the non-padded region only — computed on original size,
    # then embedded in a zero-padded canvas so padded pixels have zero weight
    # and never win seams, but their pixel data still fills the FOV extension.
    inner_weights = create_blend_weight_map(sw_orig, sh_orig)
    blend_weights_y = np.zeros((sh_padded, sw_padded), dtype=np.float32)
    blend_weights_y[pad_t:pad_t + sh_orig, pad_l:pad_l + sw_orig] = inner_weights
    blend_weights_y[y1 + pad_t:y2 + pad_t, x1 + pad_l:x2 + pad_l] = 0

    map_y_idx, c01, c23, map_uv_idx, _, _ = mapping
    
    # Initialize output buffers
    reproj_y.fill(0)
    reproj_u.fill(128)
    reproj_v.fill(128)
    reproj_weights_y.fill(0)

    # Reproject into shared buffers
    reproject_y(py.ravel(), dw, dh, py.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_y.ravel())
    reproject_uv(pu.ravel(), pv.ravel(), dw, dh, map_uv_idx.ravel(), reproj_u.ravel(), reproj_v.ravel())
    reproject_float(blend_weights_y.ravel(), dw, dh, blend_weights_y.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_weights_y.ravel())


def _round_up_16(x: int) -> int:
    """Round an integer up to the next multiple of 16."""
    return ((x + 15) // 16) * 16


def _precompile_numba_functions():
    """
    Call all Numba JIT functions with dummy data to force compilation 
    in a single thread, avoiding race conditions in thread pools.
    """
    _print("Pre-compiling JIT functions...")
    
    # Dummy arrays with correct types and minimal dimensions
    dw, dh = 8, 8
    sw_src, sh_src = 16, 16 # Larger than dest to avoid index errors
    
    map_y_idx = np.full(dw * dh, -1, dtype=np.int32)
    c01 = np.zeros(dw * dh, dtype=np.uint16)
    c23 = np.zeros(dw * dh, dtype=np.uint16)
    map_uv_idx = np.full((dw // 2) * (dh // 2), -1, dtype=np.int32)
    
    p_y = np.zeros(sw_src * sh_src, dtype=np.uint8)
    p_uv = np.zeros((sw_src // 2) * (sh_src // 2), dtype=np.uint8)
    p_float = np.zeros(sw_src * sh_src, dtype=np.float32)
    
    out_y = np.zeros(dw * dh, dtype=np.uint8)
    out_u = np.zeros((dw // 2) * (dh // 2), dtype=np.uint8)
    out_v = np.zeros((dw // 2) * (dh // 2), dtype=np.uint8)
    out_float = np.zeros(dw * dh, dtype=np.float32)

    # Call each function once to compile it
    _ = create_blend_weight_map(dw, dh)
    _ = _blur_padded_area_numba(np.zeros((32, 32), dtype=np.float32), 8, 8, 8, 8, 16, 4.0)
    reproject_y(p_y, dw, dh, sw_src, map_y_idx, c01, c23, out_y)
    reproject_uv(p_uv, p_uv, dw, dh, map_uv_idx, out_u, out_v)
    reproject_float(p_float, dw, dh, sw_src, map_y_idx, c01, c23, out_float)
    _yuv420_to_rgb_kernel(p_y, p_uv, p_uv,
        np.zeros(dw*dh,dtype=np.uint8), np.zeros(dw*dh,dtype=np.uint8), np.zeros(dw*dh,dtype=np.uint8),
        dh, dw, dw//2)
    _rgb_to_yuv420_kernel(p_y, p_y, p_y,
        np.zeros(dw*dh,dtype=np.uint8), np.zeros((dh//2)*(dw//2),dtype=np.uint8), np.zeros((dh//2)*(dw//2),dtype=np.uint8),
        dh, dw)

    _print("Pre-compilation complete.")

def reproject_images(pto_file, input_files, output_file, pad, num_cores, padsides, enhance, force_video_dims: bool = False, fisheye_mask: bool = False):
    mappings, global_options = build_mappings(pto_file, pad, num_cores, padsides, is_video_output=force_video_dims)
    final_w, final_h = global_options['final_w'], global_options['final_h']
    num_images = len(mappings)
    if len(input_files) != num_images:
        raise ValueError(f"Number of input files ({len(input_files)}) does not match the number of images in the PTO file ({num_images}).")

    # --- Start of Optimized Path for a Single Image ---
    if num_images == 1:
        _print("INFO: Single image detected, taking optimized path.")
        input_path = input_files[0]
        mapping = mappings[0]
        dw, dh = final_w, final_h

        py, pu, pv, _, _ = load_image_to_yuv(input_path, pad, padsides)
        map_y_idx, c01, c23, map_uv_idx, _, _ = mapping
    
        y_final = np.zeros((dh, dw), dtype=np.uint8)
        u_final = np.full((dh // 2, dw // 2), 128, dtype=np.uint8)
        v_final = np.full((dh // 2, dw // 2), 128, dtype=np.uint8)

        reproject_y(py.ravel(), dw, dh, py.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), y_final.ravel())
        reproject_uv(pu.ravel(), pv.ravel(), dw, dh, map_uv_idx.ravel(), u_final.ravel(), v_final.ravel())

        if enhance:
            _print("Applying enhancement filter...")
            seed_y = int.from_bytes(os.urandom(4), 'little')
            y_final = enhance_filter(y_final, t=12, log2sizex=5, log2sizey=5, dither=6, seed=seed_y)
            u_final = enhance_filter(u_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
            v_final = enhance_filter(v_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)

        save_image_yuv420(y_final, u_final, v_final, output_file)
        _print(f"✅ Success! Panoramic image saved to {output_file}")
        return # End execution here for the single-image case

    # --- Validate image dimensions before multi-image processing ---
    _print("Validating input image dimensions...")
    for i, input_path in enumerate(input_files):
        try:
            with Image.open(input_path) as img:
                actual_w, actual_h = img.size
        except Exception as e:
            raise FileNotFoundError(f"Could not open or read image file: '{input_path}'. Error: {e}")
        
        expected_w = mappings[i][4]
        expected_h = mappings[i][5]

        if actual_w != expected_w or actual_h != expected_h:
            raise ValueError(
                f"Dimension mismatch for '{os.path.basename(input_path)}'.\n"
                f"Image is {actual_w}x{actual_h}, but PTO file expects {expected_w}x{expected_h}."
            )

    # Eagerly compile Numba functions before entering the thread pool
    _precompile_numba_functions()

    _print("Reprojecting source images...")

    # Shared single-camera canvas buffers — reused for each camera in turn.
    # Peak memory = 1× canvas (Y+U/2+V/2 uint8 + weight float32) instead of N×.
    _buf_y  = np.empty((final_h, final_w), dtype=np.uint8)
    _buf_u  = np.empty((final_h // 2, final_w // 2), dtype=np.uint8)
    _buf_v  = np.empty((final_h // 2, final_w // 2), dtype=np.uint8)
    _buf_w  = np.empty((final_h, final_w), dtype=np.float32)

    # gap_map: True where no camera has weight > 0. Accumulate as uint8 count.
    gap_map_acc = np.zeros((final_h, final_w), dtype=np.uint8)

    from scipy.ndimage import distance_transform_edt as _edt, binary_erosion as _binary_erosion

    def _build_image_info(y_snap, u_snap, v_snap, w_snap, map_y_snap, cam_idx):
        """Build ImageInfo from snapshotted per-camera arrays (runs in a thread,
        overlapping with the next camera's Numba reprojection)."""
        mask = (map_y_snap >= 0) & (w_snap > 1e-9)
        mask = _binary_erosion(mask, iterations=2)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None
        r0 = int(np.argmax(rows))
        r1 = int(len(rows) - 1 - np.argmax(rows[::-1])) + 1
        c0 = int(np.argmax(cols))
        c1 = int(len(cols) - 1 - np.argmax(cols[::-1])) + 1

        r, g, b = yuv_to_rgb(y_snap, u_snap, v_snap)
        r_crop = r[r0:r1, c0:c1].copy()
        g_crop = g[r0:r1, c0:c1].copy()
        b_crop = b[r0:r1, c0:c1].copy()
        del r, g, b
        mask_crop = mask[r0:r1, c0:c1].copy()

        if not mask_crop.all():
            ds = 8
            H, W = mask_crop.shape
            ph = ((H + ds - 1) // ds) * ds
            pw = ((W + ds - 1) // ds) * ds
            solid_pad = np.zeros((ph, pw), dtype=bool)
            solid_pad[:H, :W] = mask_crop
            solid_ds = solid_pad[::ds, ::ds]
            ri_ds, ci_ds = _edt(~solid_ds, return_distances=False, return_indices=True)
            ri = np.repeat(np.repeat(ri_ds * ds, ds, axis=0), ds, axis=1)[:H, :W]
            ci = np.repeat(np.repeat(ci_ds * ds, ds, axis=0), ds, axis=1)[:H, :W]
            invalid = ~mask_crop
            r_crop[invalid] = r_crop[ri[invalid], ci[invalid]]
            g_crop[invalid] = g_crop[ri[invalid], ci[invalid]]
            b_crop[invalid] = b_crop[ri[invalid], ci[invalid]]
            del ri_ds, ci_ds, ri, ci, solid_pad, solid_ds

        _print(f"  cam {cam_idx+1}: bbox {c0},{r0}-{c1},{r1} ({c1-c0}x{r1-r0})")
        return multiblend.ImageInfo(
            filename="", bpp=8,
            width=c1 - c0, height=r1 - r0,
            xpos=c0, ypos=r0,
            channels=[r_crop, g_crop, b_crop],
            mask=mask_crop,
        )

    images = [None] * num_images
    pending_future = None
    pending_idx = None
    with ThreadPoolExecutor(max_workers=1) as post_pool:
        for i in range(num_images):
            mapping = mappings[i]
            mappings[i] = None
            process_and_reproject_image(
                ((input_files[i], final_w, final_h, mapping, pad, padsides),
                 (_buf_y, _buf_u, _buf_v, _buf_w))
            )

            gap_map_acc[_buf_w < 1e-9] += 1

            # Snapshot the shared buffers — these copies are consumed by the
            # background thread while the next reprojection runs on the originals.
            map_y_snap = mapping[0].reshape(final_h, final_w).copy()
            del mapping
            y_snap = _buf_y.copy()
            u_snap = _buf_u.copy()
            v_snap = _buf_v.copy()
            w_snap = _buf_w.copy()

            # Collect previous camera's result before submitting next.
            if pending_future is not None:
                result = pending_future.result()
                if result is not None:
                    images[pending_idx] = result

            pending_future = post_pool.submit(
                _build_image_info, y_snap, u_snap, v_snap, w_snap, map_y_snap, i)
            pending_idx = i

        if pending_future is not None:
            result = pending_future.result()
            if result is not None:
                images[pending_idx] = result

    images = [img for img in images if img is not None]

    del _buf_y, _buf_u, _buf_v, _buf_w
    _print("Reprojection complete.")

    _print("Blending with multiblend (graph-cut seams + exposure correction)...")

    # gap_map: pixel is a gap if ALL cameras had zero weight there.
    gap_map = gap_map_acc == num_images
    del gap_map_acc

    min_left, min_top, workwidth, workheight = multiblend.tighten(images)
    levels = multiblend.compute_levels(images, workwidth, workheight, False, 1_000_000, 0)
    _print(f"  {workwidth}x{workheight}, {levels} levels (tightened from {final_w}x{final_h})")

    _print("  seaming...")
    assignment, _ = multiblend.compute_seams(
        images, workwidth, workheight,
        simple_seam=False, content_seam=False,
        verbosity=0 if _quiet else 1,
        print_func=_print,
    )

    _print("  blending...")
    rgb_channels = multiblend.blend(
        images=images,
        assignment=assignment,
        workwidth=workwidth,
        workheight=workheight,
        levels=levels,
        workbpp=8,
        exposure_correct=True,
        saturation_correct=False,
        verbosity=0 if _quiet else 2,
        print_func=_print,
    )

    # Compute coverage in the tightened workspace (image xpos/ypos are relative
    # to the tightened bbox after tighten() shifted them).
    covered_tight = multiblend._coverage_mask(images, workwidth, workheight)
    del images, assignment, gap_map

    # Embed tightened result and coverage back into full canvas.
    if min_left > 0 or min_top > 0 or workwidth < final_w or workheight < final_h:
        full = [np.zeros((final_h, final_w), dtype=ch.dtype) for ch in rgb_channels]
        for c, ch in enumerate(rgb_channels):
            full[c][min_top:min_top + workheight, min_left:min_left + workwidth] = ch
        rgb_channels = full
        covered = np.zeros((final_h, final_w), dtype=bool)
        covered[min_top:min_top + workheight, min_left:min_left + workwidth] = covered_tight
    else:
        covered = covered_tight
    del covered_tight

    gap = ~covered
    del covered

    # Determine final crop height before gap fill so we don't process rows
    # that will be discarded. Round both dimensions up to a multiple of 16
    # for image/video compatibility.
    new_h = final_h
    if not fisheye_mask:
        row_has_content = np.any(~gap, axis=1)
        if row_has_content.any():
            last_row = int(len(row_has_content) - 1 - np.argmax(row_has_content[::-1]))
            new_h = last_row + 1
    new_h = _round_up_16(new_h)
    new_w = _round_up_16(final_w)
    if new_w != final_w or new_h != final_h:
        if fisheye_mask:
            _print(f"  will pad fisheye canvas: {final_w}x{final_h} -> {new_w}x{new_h}")
        else:
            _print(f"  will crop/pad canvas: {final_w}x{final_h} -> {new_w}x{new_h}")

    # Crop or pad gap and channels to the output size before filling.
    if new_h < final_h or new_w < final_w:
        gap = gap[:new_h, :new_w]
        rgb_channels = [ch[:new_h, :new_w] for ch in rgb_channels]
    elif new_h > final_h or new_w > final_w:
        _gap_h = np.zeros((new_h, new_w), dtype=bool)
        _gap_h[:final_h, :final_w] = gap
        if new_h > final_h:
            _gap_h[final_h:, :] = True
        if new_w > final_w:
            _gap_h[:, final_w:] = True
        gap = _gap_h
        rgb_channels = [np.pad(ch, ((0, max(0, new_h - final_h)), (0, max(0, new_w - final_w))), mode='edge') for ch in rgb_channels]

    n_gap = int(gap.sum())
    _print(f"  gap pixels: {n_gap}")

    if n_gap > 0:
        # Gap fill: EDT nearest-pixel + Gaussian smooth + feathered blend.
        # Processed one channel at a time to avoid large HxWx3 allocations.
        from scipy.ndimage import gaussian_filter, distance_transform_edt
        H, W = gap.shape
        S = 8
        sw, sh = max(1, W // S), max(1, H // S)
        sigma_s = 4
        feather_radius = max(1, round(20 * W / 4096))

        # Step 1: EDT index maps at 8x downscale (shared across all channels).
        ph = ((H + S - 1) // S) * S
        pw = ((W + S - 1) // S) * S
        gap_pad = np.zeros((ph, pw), dtype=bool)
        gap_pad[:H, :W] = gap
        gap_ds = gap_pad[::S, ::S];  del gap_pad
        ri_ds, ci_ds = distance_transform_edt(gap_ds, return_distances=False, return_indices=True)
        del gap_ds
        ri = np.repeat(np.repeat(ri_ds * S, S, axis=0), S, axis=1)[:H, :W];  del ri_ds
        ci = np.repeat(np.repeat(ci_ds * S, S, axis=0), S, axis=1)[:H, :W];  del ci_ds

        # Step 2: Feather weights from full-res EDT (shared across channels).
        dist = distance_transform_edt(~gap)
        blend_w = np.clip(dist / feather_radius, 0.0, 1.0).astype(np.float32);  del dist

        # Step 3: Process each channel independently.
        out_channels = []
        for ch in rgb_channels:
            ch_f = ch.astype(np.float32)
            # EDT fill: gap pixels get nearest valid pixel colour.
            ch_fill = ch_f.copy()
            ch_fill[gap] = ch_f[ri[gap], ci[gap]]
            # Gaussian smooth at downscale.
            src_u8 = ch_fill.clip(0, 255).astype(np.uint8)
            if _cv2 is not None:
                small = _cv2.resize(src_u8, (sw, sh), interpolation=_cv2.INTER_AREA).astype(np.float32)
            else:
                from PIL import Image as _PIL2
                small = np.array(_PIL2.fromarray(src_u8).resize((sw, sh), _PIL2.BOX)).astype(np.float32)
            blurred = gaussian_filter(small, sigma=sigma_s);  del small
            blurred_u8 = blurred.clip(0, 255).astype(np.uint8)
            if _cv2 is not None:
                full = _cv2.resize(blurred_u8, (W, H), interpolation=_cv2.INTER_LINEAR).astype(np.float32)
            else:
                from PIL import Image as _PIL2
                full = np.array(_PIL2.fromarray(blurred_u8).resize((W, H), _PIL2.BILINEAR)).astype(np.float32)
            del blurred, blurred_u8
            # Feathered blend.
            result = (ch_f * blend_w + full * (1.0 - blend_w))
            del ch_f, full, ch_fill
            np.clip(result, 0, 255, out=result)
            out_channels.append(result.astype(np.uint8));  del result

        del ri, ci, blend_w
        rgb_channels = out_channels
        _print(f"  gap fill done S={S} sigma_s={sigma_s} feather={feather_radius}px")

    del gap

    y_final, u_final, v_final = rgb_to_yuv(rgb_channels)

    if enhance:
        _print("Applying enhancement filter...")
        seed_y = int.from_bytes(os.urandom(4), 'little')
        y_final = enhance_filter(y_final, t=8, log2sizex=5, log2sizey=5, dither=6, seed=seed_y)
        u_final = enhance_filter(u_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
        v_final = enhance_filter(v_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)

    # Apply fisheye circular mask directly to YUV planes before saving —
    # avoids a second JPEG encode/decode cycle via ImageMagick.
    if fisheye_mask:
        h_y, w_y = y_final.shape
        cx, cy = w_y // 2, h_y // 2
        r = min(cx, cy)
        ys_y, xs_y = np.ogrid[:h_y, :w_y]
        outside_y = (xs_y - cx) ** 2 + (ys_y - cy) ** 2 > r * r
        y_final[outside_y] = 0
        # UV plane (half resolution)
        h_uv, w_uv = u_final.shape
        cx_uv, cy_uv = w_uv // 2, h_uv // 2
        r_uv = min(cx_uv, cy_uv)
        ys_uv, xs_uv = np.ogrid[:h_uv, :w_uv]
        outside_uv = (xs_uv - cx_uv) ** 2 + (ys_uv - cy_uv) ** 2 > r_uv * r_uv
        u_final[outside_uv] = 128
        v_final[outside_uv] = 128
        _print("Applied fisheye circular mask to YUV planes.")

    _print("Saving final image...")
    save_image_yuv420(y_final, u_final, v_final, output_file)
    _print(f"✅ Success! Panoramic image saved to {output_file}")

def worker_for_video_frame(args):
    """Worker function for video frames, writing to pre-allocated buffers."""
    (idx, frame, mapping, dw, dh, blend_weights_y_src, blend_weights_uv_src, pad, padsides), out_buffers = args
    reproj_y, reproj_u, reproj_v, reproj_weights_y, reproj_weights_uv = out_buffers

    if frame is None: return None
    if frame.format.name not in ("yuv420p", "yuvj420p"): frame = frame.reformat(format="yuv420p")

    sw_orig, sh_orig = frame.width, frame.height
    
    # Handle Y-plane stride (copy to make writable for timestamp erasure)
    py_buffer = np.asarray(frame.planes[0])
    py_stride = py_buffer.size // sh_orig
    py_src_orig = py_buffer.reshape(sh_orig, py_stride)[:, :sw_orig].copy()
    
    # Handle U-plane stride
    pu_buffer = np.asarray(frame.planes[1])
    pu_stride = pu_buffer.size // (sh_orig // 2)
    pu_src_orig = pu_buffer.reshape(sh_orig // 2, pu_stride)[:, :sw_orig // 2].copy()
    
    # Handle V-plane stride
    pv_buffer = np.asarray(frame.planes[2])
    pv_stride = pv_buffer.size // (sh_orig // 2)
    pv_src_orig = pv_buffer.reshape(sh_orig // 2, pv_stride)[:, :sw_orig // 2].copy()
    
    # Erase timestamp overlay (applied to all cameras, same as image path)
    x1, y1, x2, y2 = _TIMESTAMP_BOX_HD if sh_orig >= 900 else _TIMESTAMP_BOX_SD
    py_src_orig[y1:y2, x1:x2] = 0
    pu_src_orig[y1//2:y2//2, x1//2:x2//2] = 128
    pv_src_orig[y1//2:y2//2, x1//2:x2//2] = 128

    pad_t = pad if 'top' in padsides else 0
    pad_b = pad if 'bottom' in padsides else 0
    pad_l = pad if 'left' in padsides else 0
    pad_r = pad if 'right' in padsides else 0

    if pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0:
        noise_level = estimate_noise(py_src_orig) / 2.0
        pad_y_width = ((pad_t, pad_b), (pad_l, pad_r))
        pad_uv_width = ((pad_t // 2, pad_b // 2), (pad_l // 2, pad_r // 2))

        py_src_all = np.pad(py_src_orig, pad_y_width, mode='edge')
        pu_src_all = np.pad(pu_src_orig, pad_uv_width, mode='edge')
        pv_src_all = np.pad(pv_src_orig, pad_uv_width, mode='edge')

        blur_size = 96
        py_src_all = _blur_padded_area_numba(py_src_all.astype(np.float32), pad_t, pad_b, pad_l, pad_r, blur_size, noise_level)

        pad_uv_t, pad_uv_b, pad_uv_l, pad_uv_r = pad_t//2, pad_b//2, pad_l//2, pad_r//2
        blur_size_uv = blur_size // 2
        if pad_uv_t > 0 or pad_uv_b > 0 or pad_uv_l > 0 or pad_uv_r > 0:
            pu_src_all = _blur_padded_area_numba(pu_src_all.astype(np.float32), pad_uv_t, pad_uv_b, pad_uv_l, pad_uv_r, blur_size_uv, noise_level)
            pv_src_all = _blur_padded_area_numba(pv_src_all.astype(np.float32), pad_uv_t, pad_uv_b, pad_uv_l, pad_uv_r, blur_size_uv, noise_level)

        target_h_y = sh_orig + pad_t + pad_b
        target_h_uv = sh_orig // 2 + pad_uv_t + pad_uv_b
        py_src, pu_src, pv_src = py_src_all[:target_h_y, :], pu_src_all[:target_h_uv, :], pv_src_all[:target_h_uv, :]
    else:
        py_src, pu_src, pv_src = py_src_orig, pu_src_orig, pv_src_orig

    padded_sw_y = py_src.shape[1]
    map_y_idx, c01, c23, map_uv_idx, _, _ = mapping
    
    reproj_y.fill(0)
    reproj_u.fill(128)
    reproj_v.fill(128)
    reproj_weights_y.fill(0)
    reproj_weights_uv.fill(0)

    reproject_y(py_src.ravel(), dw, dh, padded_sw_y, map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_y.ravel())
    reproject_uv(pu_src.ravel(), pv_src.ravel(), dw, dh, map_uv_idx.ravel(), reproj_u.ravel(), reproj_v.ravel())
    reproject_float(blend_weights_y_src.ravel(), dw, dh, blend_weights_y_src.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_weights_y.ravel())

    # Average the smooth luma weights down to the chroma resolution.
    h, w = dh, dw
    reproj_weights_uv[:, :] = 0.25 * (reproj_weights_y[0:h:2, 0:w:2] +
                                      reproj_weights_y[1:h:2, 0:w:2] +
                                      reproj_weights_y[0:h:2, 1:w:2] +
                                      reproj_weights_y[1:h:2, 1:w:2])
    
    return idx

def _extract_timestamps_from_file(args):
    """
    Worker function for ThreadPoolExecutor. Extracts all timestamps from a single video file.
    This function is executed in a separate thread for each video file.

    If the container has a non-zero start time (Unix timestamp), the timestamps
    are derived directly from packet PTS values, which is much faster than decoding
    every frame and reading burned-in timestamps. Otherwise it falls back to OCR.
    """
    i, video_file, model = args
    # Ensure the absolute full path is printed for clarity.
    full_path = os.path.abspath(video_file)
    _print(f"\nAnalyzing timestamps for {full_path}...")

    timestamps = []

    try:
        with av.open(video_file) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base

            # Check for a non-zero container start time (Unix timestamp in seconds).
            container_start = container.start_time
            start_time_sec = container_start / av.time_base if container_start is not None else 0.0

            if start_time_sec > 0:
                start_dt = datetime.datetime.fromtimestamp(start_time_sec, tz=datetime.timezone.utc)
                _print(f"  Using container start time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}.{start_dt.microsecond:06d}")
                # Collect packet PTS values and derive absolute timestamps from them.
                packet_ts = []
                for packet in container.demux(stream):
                    if packet.pts is None:
                        continue
                    ts_seconds = packet.pts * time_base
                    packet_ts.append((packet.pts, ts_seconds))

                # Sort by PTS so frame indices are in display order.
                packet_ts.sort(key=lambda x: x[0])
                for frame_idx, (_, ts_seconds) in enumerate(packet_ts):
                    ts = datetime.datetime.fromtimestamp(float(ts_seconds), tz=datetime.timezone.utc)
                    timestamps.append((frame_idx, ts))
                _print(f"  Extracted {len(timestamps)} timestamps from container metadata.")
            else:
                # Fall back to reading burned-in timestamps from decoded frames.
                try:
                    from timestamp import get_timestamp
                except ImportError:
                    # Re-raise the ImportError. The main thread will catch this exception
                    # from the ThreadPoolExecutor and handle the user message and exit.
                    raise

                stream.thread_type = 'AUTO'
                frame_idx = 0
                for frame in container.decode(stream):
                    ts = None
                    try:
                        ts = get_timestamp(frame.to_image(), robust=False, model=model)
                        if ts is None:
                            ts = get_timestamp(frame.to_image(), robust=True, model=model)
                    except (ValueError, TypeError):
                        ts = None

                    if ts:
                        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                        progress_message = f"  -> Current Timestamp: {ts_str}".ljust(70)
                        sys.stdout.write(f'\r{progress_message}')
                        sys.stdout.flush()

                    timestamps.append((frame_idx, ts))
                    frame_idx += 1
                sys.stdout.write('\n')
                sys.stdout.flush()

    except (av.Error, IndexError) as e:
        sys.stdout.write('\n')
        _print(f"Warning: Could not process video '{full_path}': {e}", file=sys.stderr)

    return i, timestamps

def _estimate_and_fill_timestamps(all_timestamps):
    """
    Analyzes timestamps, calculates frame intervals using the median to resist outliers,
    and fills in missing values.
    """
    cleaned_timestamps = []
    estimated_intervals_sec = []

    for stream_timestamps in all_timestamps:
        # Calculate the MEDIAN interval from valid timestamps.
        # The median is robust to outliers, such as large time gaps.
        valid_diffs = np.diff([ts.timestamp() for _, ts in stream_timestamps if ts is not None])
        if len(valid_diffs) > 1:
            median_interval = np.median(valid_diffs)
            # Add a sanity check for the calculated interval
            if not (0.01 < median_interval < 10):
                median_interval = None
        else:
            median_interval = None
        
        if median_interval:
            estimated_intervals_sec.append(median_interval)

        filled_stream_ts = []
        last_valid_ts = None
        last_valid_idx = -1

        # Find first valid timestamp to start from
        for idx, ts in stream_timestamps:
            if ts:
                last_valid_ts = ts
                last_valid_idx = idx
                break
        
        if last_valid_ts is None:
            _print("Warning: No valid timestamps found in a stream. It will be ignored in synchronization.", file=sys.stderr)
            cleaned_timestamps.append([]) # Add empty list to maintain stream count
            continue

        # Forward fill and estimate missing timestamps
        for idx, ts in stream_timestamps:
            if ts:
                if median_interval and last_valid_idx != -1:
                    expected_ts = last_valid_ts + datetime.timedelta(seconds=(idx - last_valid_idx) * median_interval)
                    # If a timestamp deviates too much, treat it as invalid.
                    if abs((ts - expected_ts).total_seconds()) > median_interval * 5:
                         filled_stream_ts.append((idx, expected_ts))
                         continue
                
                filled_stream_ts.append((idx, ts))
                last_valid_ts = ts
                last_valid_idx = idx
            else: # ts is None, so we estimate
                 if median_interval and last_valid_idx != -1:
                    estimated_ts = last_valid_ts + datetime.timedelta(seconds=(idx - last_valid_idx) * median_interval)
                    filled_stream_ts.append((idx, estimated_ts))
                 else:
                     filled_stream_ts.append((idx, None))

        cleaned_timestamps.append(filled_stream_ts)
    
    # Calculate overall median interval for sync tolerance
    median_overall_interval = np.median(estimated_intervals_sec) if estimated_intervals_sec else (1/30.0) # Default to 30fps
    return cleaned_timestamps, median_overall_interval

def _find_synchronized_frames(timestamps_per_video, sync_tolerance_sec):
    """
    Finds groups of frames that are synchronized within the given tolerance.
    """
    num_videos = len(timestamps_per_video)
    if num_videos == 0: return []

    valid_streams_data = [(i, ts_list) for i, ts_list in enumerate(timestamps_per_video) if ts_list]
    if len(valid_streams_data) < num_videos:
        _print(f"Warning: Only {len(valid_streams_data)} of {num_videos} have valid timestamps for synchronization.", file=sys.stderr)
    if len(valid_streams_data) < 2:
        _print("Error: Synchronization requires at least two video streams with valid timestamps.", file=sys.stderr)
        return []
    
    stream_indices = [d[0] for d in valid_streams_data]
    ts_data = [d[1] for d in valid_streams_data]
    num_valid_streams = len(ts_data)

    pointers = [0] * num_valid_streams
    stream_lengths = [len(s) for s in ts_data]
    synchronized_frame_groups = []

    while all(p < l for p, l in zip(pointers, stream_lengths)):
        current_timestamps = [ts_data[i][pointers[i]][1] for i in range(num_valid_streams)]
        
        if any(ts is None for ts in current_timestamps):
            for i, ts in enumerate(current_timestamps):
                if ts is None: pointers[i] += 1
            continue

        max_ts = max(current_timestamps)

        if all((max_ts - ts).total_seconds() <= sync_tolerance_sec for ts in current_timestamps):
            frame_indices = [ts_data[i][pointers[i]][0] for i in range(num_valid_streams)]
            
            full_group = [-1] * num_videos
            for original_idx, frame_idx in zip(stream_indices, frame_indices):
                full_group[original_idx] = frame_idx
            synchronized_frame_groups.append(tuple(full_group))

            for i in range(num_valid_streams): pointers[i] += 1
        else:
            min_ts = min(current_timestamps)
            min_idx = current_timestamps.index(min_ts)
            pointers[min_idx] += 1
            
    return synchronized_frame_groups

def reproject_videos(pto_file, input_files, output_file, pad, num_cores, padsides, use_sync=False, model=None, save_sync_file=None, load_sync_file=None, enhance=False, fisheye_mask=False, max_frames=0, level_subsample=1):
    if av is None: raise ImportError("PyAV is not installed, but video processing was requested.")

    mappings, global_options = build_mappings(pto_file, pad, num_cores, padsides, is_video_output=True)
    final_w, final_h = global_options['final_w'], global_options['final_h']
    if final_w > 16384:
        raise ValueError(f"Output width {final_w} exceeds codec limits for H.264/libx264. PTO='{pto_file}'")
    num_images = len(mappings)
    if len(input_files) != num_images: raise ValueError("Number of videos does not match PTO.")
    
    _precompile_numba_functions()

    # --- Start of Single-Video Optimization ---
    if num_images == 1:
        _print("INFO: Single video detected, taking optimized path.")
        input_path = input_files[0]
        mapping = mappings[0]
        # Round down to multiples of 16 so the precomputed mapping still covers the
        # smaller output rectangle without out-of-bounds access.
        dw = (final_w // 16) * 16
        dh = (final_h // 16) * 16
        if dw != final_w or dh != final_h:
            _print(f"  rounding single-video output: {final_w}x{final_h} -> {dw}x{dh}")

        try:
            in_container = av.open(input_path)
            in_stream = in_container.streams.video[0]
            in_stream.thread_type = 'AUTO'
            total_frames = in_stream.frames if in_stream.frames > 0 else 0
            if max_frames > 0 and (total_frames == 0 or total_frames > max_frames):
                total_frames = max_frames

            out_container = av.open(output_file, mode='w')
            out_stream = out_container.add_stream("libx264", rate=in_stream.average_rate)
            out_stream.width, out_stream.height, out_stream.pix_fmt = dw, dh, 'yuv420p'
            out_stream.options = {"preset": "ultrafast", "crf": "28"}
        except av.AVError as e:
            raise IOError(f"PyAV Error: Could not open video files for processing. Check paths and file integrity.\nDetails: {e}")
        
        map_y_idx, c01, c23, map_uv_idx, _, _ = mapping
        frame_count = 0

        # Pre-compute pad dimensions once — they are constant across all frames.
        pad_t = pad if 'top' in padsides else 0; pad_b = pad if 'bottom' in padsides else 0
        pad_l = pad if 'left' in padsides else 0; pad_r = pad if 'right' in padsides else 0
        _needs_pad = pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0
        pad_y_width = ((pad_t, pad_b), (pad_l, pad_r))
        pad_uv_width = ((pad_t // 2, pad_b // 2), (pad_l // 2, pad_r // 2))

        # --- Create output frame once to improve performance and stability ---
        try:
            out_frame = av.VideoFrame(width=dw, height=dh, format='yuv420p')
            if not out_frame.planes or not out_frame.planes[0]:
                raise RuntimeError() # Will be caught below
        except Exception:
            raise RuntimeError(
                f"Failed to allocate video frame buffer with dimensions {dw}x{dh}.\n"
                "Please check system memory and ensure PTO parameters result in valid dimensions."
            )

        # Two buffer sets for double-buffering: while the encoder consumes buf[cur],
        # the Numba reprojection fills buf[nxt] in a background thread.
        # Unmapped pixels (outside the projection) are initialised once here and never
        # overwritten, because reproject_y/uv only writes to mapped pixel positions
        # and the projection map is constant across all frames.
        _bufs = [
            (np.zeros((dh, dw), dtype=np.uint8), np.full((dh//2, dw//2), 128, dtype=np.uint8), np.full((dh//2, dw//2), 128, dtype=np.uint8)),
            (np.zeros((dh, dw), dtype=np.uint8), np.full((dh//2, dw//2), 128, dtype=np.uint8), np.full((dh//2, dw//2), 128, dtype=np.uint8)),
        ]

        def _reproject_frame(frame, y_out, u_out, v_out):
            if frame.format.name not in ("yuv420p", "yuvj420p"):
                frame = frame.reformat(format="yuv420p")
            sw_f, sh_f = frame.width, frame.height
            py_buf = np.asarray(frame.planes[0])
            py_s = py_buf.reshape(sh_f, py_buf.size // sh_f)[:, :sw_f]
            pu_buf = np.asarray(frame.planes[1])
            pu_s = pu_buf.reshape(sh_f // 2, pu_buf.size // (sh_f // 2))[:, :sw_f // 2]
            pv_buf = np.asarray(frame.planes[2])
            pv_s = pv_buf.reshape(sh_f // 2, pv_buf.size // (sh_f // 2))[:, :sw_f // 2]
            if _needs_pad:
                py_s = np.pad(py_s, pad_y_width, mode='edge')
                pu_s = np.pad(pu_s, pad_uv_width, mode='edge')
                pv_s = np.pad(pv_s, pad_uv_width, mode='edge')
            reproject_y(py_s.ravel(), dw, dh, py_s.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), y_out.ravel())
            reproject_uv(pu_s.ravel(), pv_s.ravel(), dw, dh, map_uv_idx.ravel(), u_out.ravel(), v_out.ravel())
            return y_out, u_out, v_out

        def _encode_yuv(y, u, v, pts):
            if enhance:
                seed_y = int.from_bytes(os.urandom(4), 'little')
                y = enhance_filter(y, t=8, log2sizex=5, log2sizey=5, dither=6, seed=seed_y)
                u = enhance_filter(u, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
                v = enhance_filter(v, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
            out_frame.planes[0].update(y); out_frame.planes[1].update(u); out_frame.planes[2].update(v)
            out_frame.pts = pts
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)

        frame_iter = in_container.decode(in_stream)
        with ThreadPoolExecutor(max_workers=1) as _pipe_ex:
            cur = 0
            first_frame = next(frame_iter, None)
            if first_frame is not None:
                _fut = _pipe_ex.submit(_reproject_frame, first_frame, *_bufs[cur])

            for next_frame in frame_iter:
                y_final, u_final, v_final = _fut.result()
                nxt = 1 - cur
                # Kick off reprojection of next frame while we encode current.
                _fut = _pipe_ex.submit(_reproject_frame, next_frame, *_bufs[nxt])

                frame_count += 1
                if not _quiet and total_frames > 0 and (frame_count % 5 == 0 or frame_count == total_frames):
                    percent_done = (frame_count / total_frames) * 100
                    if sys.stderr.isatty():
                        bar_length = 40; filled_len = int(round(bar_length * frame_count / float(total_frames)))
                        bar = '█' * filled_len + '-' * (bar_length - filled_len)
                        sys.stderr.write(f'Stitching: [{bar}] {percent_done:.1f}% \r'); sys.stderr.flush()
                    else:
                        _print(f"PROGRESS:{percent_done:.1f}", file=sys.stderr, flush=True)

                _encode_yuv(y_final, u_final, v_final, frame_count - 1)
                cur = nxt

            # Encode the last frame.
            if first_frame is not None:
                y_final, u_final, v_final = _fut.result()
                frame_count += 1
                _encode_yuv(y_final, u_final, v_final, frame_count - 1)

        if not _quiet and total_frames > 0 and sys.stderr.isatty(): sys.stderr.write("\n"); sys.stderr.flush()
        for packet in out_stream.encode(): out_container.mux(packet)
        out_container.close(); in_container.close()
        _print(f"\n✅ Success! Panoramic video saved to {output_file}")
        return
    # --- End of Single-Video Optimization ---

    # --- Video Synchronization Pass ---
    synchronized_frame_groups = []
    if use_sync:
        if load_sync_file:
            _print(f"Loading sync map from {load_sync_file}...")
            try:
                with open(load_sync_file, 'r') as f:
                    synchronized_frame_groups = json.load(f)
                _print(f"Loaded {len(synchronized_frame_groups)} synchronized frame groups.")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(f"Could not load or parse sync file '{load_sync_file}'. Reason: {e}")
        else:
            _print("Starting Pass 1: Timestamp analysis (utilizing all available cores)...")
            try:
                with ThreadPoolExecutor(max_workers=num_cores) as executor:
                    raw_ts_data_unordered = list(executor.map(_extract_timestamps_from_file, [(i, f, model) for i, f in enumerate(input_files)]))
            except ImportError:
                raise ImportError("\nThe 'timestamp.py' module is required for the --sync feature but was not found.")
            
            raw_ts_data = [d for _, d in sorted(raw_ts_data_unordered)]

            _print("Estimating timestamps using robust median interval...")
            cleaned_ts_data, median_interval = _estimate_and_fill_timestamps(raw_ts_data)
            
            sync_tolerance = median_interval * 1.5
            _print(f"Calculated median frame interval: {median_interval:.3f}s. Using sync tolerance: {sync_tolerance:.3f}s")

            _print("Starting Pass 2: Finding synchronized frame groups...")
            synchronized_frame_groups = _find_synchronized_frames(cleaned_ts_data, sync_tolerance)

            if not synchronized_frame_groups:
                raise RuntimeError("Could not find any synchronized frames. Aborting.")
                
            _print(f"Found {len(synchronized_frame_groups)} synchronized frame groups to stitch.")

            if save_sync_file:
                _print(f"Saving sync map to {save_sync_file}...")
                with open(save_sync_file, 'w') as f:
                    json.dump(synchronized_frame_groups, f, indent=2)
                _print("Sync map saved.")

    # --- Precompute geometry (gap fill, crop) from weight maps ---------------
    # Weight maps are pure geometry — identical for every frame.
    # We precompute everything here so the per-frame loop has zero overhead.
    _print("Precomputing gap/crop geometry from weight maps...")
    _tmp_weights = np.zeros((final_h, final_w), dtype=np.float32)
    _tmp_blend_weights_y = []
    _tmp_pad_t = pad if 'top' in padsides else 0
    _tmp_pad_b = pad if 'bottom' in padsides else 0
    _tmp_pad_l = pad if 'left' in padsides else 0
    _tmp_pad_r = pad if 'right' in padsides else 0
    for i in range(num_images):
        sw_map, sh_map = mappings[i][4], mappings[i][5]
        sw_padded = sw_map + _tmp_pad_l + _tmp_pad_r
        sh_padded = sh_map + _tmp_pad_t + _tmp_pad_b
        bw = create_blend_weight_map(sw_padded, sh_padded)
        # Zero timestamp box in weight map so those pixels never win seams
        ts_x1, ts_y1, ts_x2, ts_y2 = _TIMESTAMP_BOX_HD if sh_map >= 900 else _TIMESTAMP_BOX_SD
        bw[ts_y1 + _tmp_pad_t:ts_y2 + _tmp_pad_t, ts_x1 + _tmp_pad_l:ts_x2 + _tmp_pad_l] = 0
        _tmp_blend_weights_y.append(bw)
        # Reproject blend-weight map to canvas to accumulate coverage
        map_y_idx, c01, c23 = mappings[i][0], mappings[i][1], mappings[i][2]
        _tmp_reproj = np.zeros((final_h, final_w), dtype=np.float32)
        reproject_float(bw.ravel(), final_w, final_h, bw.shape[1],
                        map_y_idx.ravel(), c01.ravel(), c23.ravel(), _tmp_reproj.ravel())
        _tmp_weights += _tmp_reproj
    geo_gap = _tmp_weights < 1e-9
    del _tmp_weights, _tmp_reproj

    from scipy.ndimage import gaussian_filter, distance_transform_edt as _geo_edt
    H_geo, W_geo = geo_gap.shape
    S_geo = 8
    feather_radius = max(1, round(20 * W_geo / 4096))
    # Gap-fill EDT geometry is computed after geo_crop_h / out_h is known so we
    # only process rows that will survive the final crop.

    # Per-camera bounding boxes, eroded masks, and inpaint maps (geometry-only)
    from scipy.ndimage import binary_erosion as _binary_erosion_cam
    cam_bboxes = []   # list of (r0, r1, c0, c1) or None
    cam_masks  = []   # list of eroded mask_crop or None  (passed to ImageInfo.mask)
    cam_inpaint = []  # list of (ri, ci, invalid_mask) or None  (for EDT inpaint)
    for i in range(num_images):
        bw = _tmp_blend_weights_y[i]
        _tmp_reproj_i = np.zeros((final_h, final_w), dtype=np.float32)
        map_y_idx_i, c01_i, c23_i = mappings[i][0], mappings[i][1], mappings[i][2]
        reproject_float(bw.ravel(), final_w, final_h, bw.shape[1],
                        map_y_idx_i.ravel(), c01_i.ravel(), c23_i.ravel(), _tmp_reproj_i.ravel())
        mask_i = _tmp_reproj_i > 1e-9
        rows_i = np.any(mask_i, axis=1); cols_i = np.any(mask_i, axis=0)
        if not rows_i.any():
            cam_bboxes.append(None); cam_masks.append(None); cam_inpaint.append(None); continue
        r0_i = int(np.argmax(rows_i))
        r1_i = int(len(rows_i) - np.argmax(rows_i[::-1]))
        c0_i = int(np.argmax(cols_i))
        c1_i = int(len(cols_i) - np.argmax(cols_i[::-1]))
        cam_bboxes.append((r0_i, r1_i, c0_i, c1_i))
        # Erode mask first, then derive bbox from eroded mask — identical to image path
        eroded_full_i = _binary_erosion_cam(mask_i, iterations=2)
        rows_e = np.any(eroded_full_i, axis=1); cols_e = np.any(eroded_full_i, axis=0)
        if not rows_e.any():
            # Erosion consumed entire mask — fall back to uneroded bbox with empty mask
            eroded_i = np.zeros((r1_i - r0_i, c1_i - c0_i), dtype=bool)
            cam_masks.append(eroded_i)
            cam_inpaint.append(None)
            continue
        r0_i = int(np.argmax(rows_e))
        r1_i = int(len(rows_e) - np.argmax(rows_e[::-1]))
        c0_i = int(np.argmax(cols_e))
        c1_i = int(len(cols_e) - np.argmax(cols_e[::-1]))
        cam_bboxes[-1] = (r0_i, r1_i, c0_i, c1_i)  # update bbox to eroded extent
        mask_crop_i = eroded_full_i[r0_i:r1_i, c0_i:c1_i]
        cam_masks.append(mask_crop_i)
        # Inpaint internal holes only (same as image path — border is already valid data)
        if not mask_crop_i.all():
            H_i, W_i = mask_crop_i.shape; ds_i = 8
            ph_i = ((H_i+ds_i-1)//ds_i)*ds_i; pw_i = ((W_i+ds_i-1)//ds_i)*ds_i
            sp = np.zeros((ph_i, pw_i), dtype=bool)
            sp[:H_i, :W_i] = mask_crop_i
            sd = sp[::ds_i, ::ds_i]
            ri_ds_i, ci_ds_i = _geo_edt(~sd, return_distances=False, return_indices=True)
            ri_i = np.repeat(np.repeat(ri_ds_i*ds_i, ds_i, axis=0), ds_i, axis=1)[:H_i, :W_i]
            ci_i = np.repeat(np.repeat(ci_ds_i*ds_i, ds_i, axis=0), ds_i, axis=1)[:H_i, :W_i]
            cam_inpaint.append((ri_i, ci_i, ~mask_crop_i))
        else:
            cam_inpaint.append(None)
    del _tmp_reproj_i

    # Compute tighten offsets once from geometry
    valid_bboxes = [b for b in cam_bboxes if b is not None]
    geo_min_top  = min(b[0] for b in valid_bboxes)
    geo_min_left = min(b[2] for b in valid_bboxes)
    geo_workwidth  = max(b[3]-b[2] + b[2]-geo_min_left for b in valid_bboxes)
    geo_workheight = max(b[1]-b[0] + b[0]-geo_min_top  for b in valid_bboxes)
    # Precompute tightened xpos/ypos for each camera
    cam_tight_pos = []
    for bbox in cam_bboxes:
        if bbox is None:
            cam_tight_pos.append(None)
        else:
            r0_i, r1_i, c0_i, c1_i = bbox
            cam_tight_pos.append((r0_i - geo_min_top, c0_i - geo_min_left))

    # Bottom-crop row (equirect only — fisheye video not supported)
    geo_crop_h = final_h
    if True:
        row_has_content = np.any(~geo_gap, axis=1)
        if row_has_content.any():
            geo_crop_h = int(len(row_has_content) - np.argmax(row_has_content[::-1]))
            if geo_crop_h < final_h:
                _print(f"  will crop canvas: {final_h} -> {geo_crop_h} rows")

    out_h = _round_up_16(geo_crop_h)  # actual encoded height
    out_w = _round_up_16(final_w)     # actual encoded width
    if out_h != geo_crop_h:
        _print(f"  will round height: {geo_crop_h} -> {out_h} rows")
    if out_w != final_w:
        _print(f"  will round width: {final_w} -> {out_w} columns")

    # Crop or pad gap-fill geometry to the actual output size so we don't
    # process rows/columns that will be discarded, and so padded areas are
    # filled by the gap-fill logic.
    if out_h > final_h or out_w > final_w:
        _gap_w = np.zeros((out_h, out_w), dtype=bool)
        _gap_w[:final_h, :final_w] = geo_gap
        _gap_w[final_h:, :] = True   # new rows are gaps
        _gap_w[:, final_w:] = True   # new columns are gaps
        geo_gap = _gap_w
    else:
        geo_gap = geo_gap[:out_h, :out_w]
    H_geo = out_h
    W_geo = out_w
    geo_sw = max(1, W_geo // S_geo)
    geo_sh = max(1, H_geo // S_geo)
    geo_sigma_s = 4
    # EDT index maps at 8× downscale for gap fill
    ph_g = ((H_geo + S_geo - 1) // S_geo) * S_geo
    pw_g = ((W_geo + S_geo - 1) // S_geo) * S_geo
    gap_pad_g = np.zeros((ph_g, pw_g), dtype=bool)
    gap_pad_g[:H_geo, :W_geo] = geo_gap
    gap_ds_g = gap_pad_g[::S_geo, ::S_geo]; del gap_pad_g
    ri_ds_g, ci_ds_g = _geo_edt(gap_ds_g, return_distances=False, return_indices=True); del gap_ds_g
    geo_ri = np.repeat(np.repeat(ri_ds_g * S_geo, S_geo, axis=0), S_geo, axis=1)[:H_geo, :W_geo]; del ri_ds_g
    geo_ci = np.repeat(np.repeat(ci_ds_g * S_geo, S_geo, axis=0), S_geo, axis=1)[:H_geo, :W_geo]; del ci_ds_g
    # Feather weights from full-res EDT
    geo_dist = _geo_edt(~geo_gap)
    geo_blend_w = np.clip(geo_dist / feather_radius, 0.0, 1.0).astype(np.float32); del geo_dist
    geo_n_gap = int(geo_gap.sum())
    geo_gap_idx = np.where(geo_gap)  # precomputed tuple for fast per-frame fill
    # Precompute the EDT source row/col at gap pixels (1D arrays) — used for fast fill
    geo_ri_gap = geo_ri[geo_gap_idx]  # shape (geo_n_gap,)
    geo_ci_gap = geo_ci[geo_gap_idx]  # shape (geo_n_gap,)
    _print(f"  gap pixels: {geo_n_gap}, feather: {feather_radius}px")

    # Precompute fisheye circular mask (constant geometry)
    if fisheye_mask:
        _fy, _fx = out_h, out_w
        _fcx, _fcy = _fx // 2, _fy // 2
        _fr = min(_fcx, _fcy)
        _fys, _fxs = np.ogrid[:_fy, :_fx]
        geo_outside_y = (_fxs - _fcx) ** 2 + (_fys - _fcy) ** 2 > _fr * _fr
        _fuvy, _fuvx = _fy // 2, _fx // 2
        _fuvcx, _fuvcy = _fuvx // 2, _fuvy // 2
        _fuvr = min(_fuvcx, _fuvcy)
        _fuvys, _fuvxs = np.ogrid[:_fuvy, :_fuvx]
        geo_outside_uv = (_fuvxs - _fuvcx) ** 2 + (_fuvys - _fuvcy) ** 2 > _fuvr * _fuvr
    else:
        geo_outside_y = geo_outside_uv = None

    # Preallocate gap-fill working buffers — one per channel for parallel fill
    _geo_fill_u8 = [np.empty((H_geo, W_geo), dtype=np.uint8) for _ in range(3)] if geo_n_gap > 0 else None

    # --- Stitching Pass ---
    _print("\nStarting stitching process...")
    try:
        in_containers = [av.open(f) for f in input_files]
        in_streams = [c.streams.video[0] for c in in_containers]
        for s in in_streams: s.thread_type = 'AUTO'

        out_container = av.open(output_file, mode='w')
        out_stream = out_container.add_stream("libx264", rate=in_streams[0].average_rate)
        out_stream.width, out_stream.height, out_stream.pix_fmt = out_w, out_h, 'yuv420p'
        out_stream.options = {"preset": "ultrafast", "crf": "28"}
    except av.AVError as e:
        raise IOError(f"PyAV Error: Could not open video files for processing. Check paths and file integrity.\nDetails: {e}")
        
    total_frames = 0
    if use_sync:
        total_frames = len(synchronized_frame_groups)
    else:
        frame_counts = [s.frames for s in in_streams if s.frames > 0]
        if frame_counts:
            total_frames = min(frame_counts)
    if max_frames > 0 and total_frames > max_frames:
        total_frames = max_frames

    cached_seam_weights, target_leveling_params, previous_leveling_params = None, None, None
    recalc_frame_number = 1
    
    # Seam caching for multiblend
    import tempfile
    seam_cache_file = tempfile.mktemp(suffix='_seam.png') if multiblend is not None else None
    
    frame_y_planes = np.empty((num_images, final_h, final_w), dtype=np.uint8)
    frame_u_planes = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.uint8)
    frame_v_planes = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.uint8)
    frame_weights_y = np.empty((num_images, final_h, final_w), dtype=np.float32)
    frame_weights_uv = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.float32)
    # Preallocate canvas buffers — reused every frame with .fill().
    # Only out_h rows and out_w columns are needed; the final crop is already applied to geometry.
    _canvas_r = np.empty((out_h, out_w), dtype=np.float32)
    _canvas_g = np.empty((out_h, out_w), dtype=np.float32)
    _canvas_b = np.empty((out_h, out_w), dtype=np.float32)

    blend_weights_y = _tmp_blend_weights_y  # already computed during geometry precompute
    pad_t = _tmp_pad_t; pad_b = _tmp_pad_b; pad_l = _tmp_pad_l; pad_r = _tmp_pad_r

    def _yuv_crop_inpaint(i):
        """Convert reprojected YUV planes to RGB, crop to eroded bbox, inpaint holes.
        Crops Y/U/V planes to the bbox before conversion to minimise work in yuv_to_rgb."""
        bbox = cam_bboxes[i]
        if bbox is None:
            return i, None
        r0, r1, c0, c1 = bbox
        # Align UV crop to even boundaries (YUV420 requirement)
        ur0, ur1 = r0 // 2, (r1 + 1) // 2
        uc0, uc1 = c0 // 2, (c1 + 1) // 2
        y_crop_src = frame_y_planes[i][r0:r1, c0:c1]
        u_crop_src = frame_u_planes[i][ur0:ur1, uc0:uc1]
        v_crop_src = frame_v_planes[i][ur0:ur1, uc0:uc1]
        r, g, b = yuv_to_rgb(y_crop_src, u_crop_src, v_crop_src)
        # yuv_to_rgb may produce (r1-r0) or (r1-r0+1) rows depending on UV upscale;
        # slice back to exact bbox height/width.
        h_bb, w_bb = r1 - r0, c1 - c0
        r_crop = r[:h_bb, :w_bb].copy()
        g_crop = g[:h_bb, :w_bb].copy()
        b_crop = b[:h_bb, :w_bb].copy()
        inpaint = cam_inpaint[i]
        if inpaint is not None:
            ri_i, ci_i, invalid_i = inpaint
            r_crop[invalid_i] = r_crop[ri_i[invalid_i], ci_i[invalid_i]]
            g_crop[invalid_i] = g_crop[ri_i[invalid_i], ci_i[invalid_i]]
            b_crop[invalid_i] = b_crop[ri_i[invalid_i], ci_i[invalid_i]]
        return i, (r_crop, g_crop, b_crop)

    frame_iters = [c.decode(s) for c, s in zip(in_containers, in_streams)]
    frame_count = 0
    loop_iterator = synchronized_frame_groups if use_sync else zip(*frame_iters)
    
    # --- Create output frame once to improve performance and stability ---
    try:
        out_frame = av.VideoFrame(width=out_w, height=out_h, format='yuv420p')
        if not out_frame.planes or not out_frame.planes[0]:
            raise RuntimeError()
    except Exception:
        raise RuntimeError(
            f"FATAL: Failed to allocate video frame buffer with dimensions {out_w}x{out_h}."
        )

    if geo_n_gap > 0:
        def _gap_fill_channel(ch_f, buf_u8):
            np.clip(ch_f, 0, 255, out=ch_f)
            np.copyto(buf_u8, ch_f, casting='unsafe')
            buf_u8[geo_gap_idx] = ch_f[geo_ri_gap, geo_ci_gap].astype(np.uint8)
            if _cv2 is not None:
                small = _cv2.resize(buf_u8, (geo_sw, geo_sh), interpolation=_cv2.INTER_AREA).astype(np.float32)
            else:
                from PIL import Image as _PIL2
                small = np.array(_PIL2.fromarray(buf_u8).resize((geo_sw, geo_sh), _PIL2.BOX)).astype(np.float32)
            blurred_u8 = gaussian_filter(small, sigma=geo_sigma_s).clip(0, 255).astype(np.uint8)
            if _cv2 is not None:
                full = _cv2.resize(blurred_u8, (W_geo, H_geo), interpolation=_cv2.INTER_LINEAR).astype(np.float32)
            else:
                from PIL import Image as _PIL2
                full = np.array(_PIL2.fromarray(blurred_u8).resize((W_geo, H_geo), _PIL2.BILINEAR)).astype(np.float32)
            result = ch_f * geo_blend_w + full * (1.0 - geo_blend_w)
            np.clip(result, 0, 255, out=result)
            return result.astype(np.uint8)

    # Cache exposure correction info to support --level-subsample.
    cached_exp_info = None

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        current_frame_indices = [-1] * num_images
        
        for group in loop_iterator:
            final_group_frames = [None] * num_images
            if use_sync:
                target_indices = group
                for i, target_idx in enumerate(target_indices):
                    if target_idx == -1: continue
                    frame = None
                    try:
                        while current_frame_indices[i] < target_idx:
                            frame = next(frame_iters[i])
                            current_frame_indices[i] += 1
                        if frame and current_frame_indices[i] == target_idx:
                             final_group_frames[i] = frame
                        else: raise StopIteration
                    except StopIteration:
                        _print(f"\nWarning: Stream {i} ended unexpectedly while seeking frame {target_idx}. Terminating.", file=sys.stderr)
                        group = None; break
                if group is None: break
            else:
                try: final_group_frames = list(group)
                except StopIteration: break
            
            if any(f is None for f in final_group_frames): continue
            frame_count += 1
            if max_frames > 0 and frame_count > max_frames: break
            
            if not _quiet and total_frames > 0 and (frame_count % 5 == 0 or frame_count == total_frames):
                percent_done = (frame_count / total_frames) * 100
                if sys.stderr.isatty():
                    bar_length = 40; filled_len = int(round(bar_length*frame_count/float(total_frames)))
                    bar = '█'*filled_len + '-'*(bar_length - filled_len)
                    sys.stderr.write(f'Stitching: [{bar}] {percent_done:.1f}% \r'); sys.stderr.flush()
                else: _print(f"PROGRESS:{percent_done:.1f}", file=sys.stderr, flush=True)

            worker_args = [
                ((i, final_group_frames[i], mappings[i], final_w, final_h, blend_weights_y[i], None, pad, padsides),
                 (frame_y_planes[i], frame_u_planes[i], frame_v_planes[i], frame_weights_y[i], frame_weights_uv[i]))
                for i in range(num_images) if final_group_frames[i] is not None
            ]
            list(executor.map(worker_for_video_frame, worker_args))

            # Build ImageInfo list: yuv_to_rgb + crop + inpaint per camera
            images = []
            for i in range(num_images):
                _, rgb_crops = _yuv_crop_inpaint(i)
                if rgb_crops is None:
                    continue
                r0, r1, c0, c1 = cam_bboxes[i]
                tight_ypos, tight_xpos = cam_tight_pos[i]
                images.append(multiblend.ImageInfo(
                    filename="", bpp=8, width=c1-c0, height=r1-r0,
                    xpos=tight_xpos, ypos=tight_ypos,
                    channels=list(rgb_crops),
                    mask=cam_masks[i],
                ))

            workwidth, workheight = geo_workwidth, geo_workheight
            # Seam computation — only on first frame, then reuse
            if frame_count == 1:
                _print("Computing seams with multiblend (first frame)...")
                levels = multiblend.compute_levels(images, workwidth, workheight, False, 1_000_000, 0)
                assignment, _ = multiblend.compute_seams(
                    images=images,
                    workwidth=workwidth,
                    workheight=workheight,
                    simple_seam=False,
                    content_seam=False,
                    verbosity=0,
                    print_func=_print,
                )
                if seam_cache_file:
                    try:
                        multiblend._save_seams_png(seam_cache_file, assignment, workwidth, workheight, 0, _print)
                    except Exception:
                        seam_cache_file = None
            else:
                if seam_cache_file and os.path.exists(seam_cache_file):
                    try:
                        assignment = multiblend._load_seams_png(seam_cache_file, workwidth, workheight, 0, _print)
                    except Exception:
                        assignment = np.zeros((workheight, workwidth), dtype=np.uint8)

            # Blend using multiblend with exposure correction. Recompute only
            # every level_subsample frames; reuse cached correction otherwise.
            recompute_exposure = (frame_count - 1) % level_subsample == 0
            blend_out_info = {}
            rgb_blended = multiblend.blend(
                images=images,
                assignment=assignment,
                workwidth=workwidth,
                workheight=workheight,
                levels=levels,
                workbpp=8,
                exposure_correct=True,
                saturation_correct=False,
                verbosity=0,
                print_func=_print,
                exposure_info=None if recompute_exposure else cached_exp_info,
                out_info=blend_out_info,
            )
            if recompute_exposure and 'exposure' in blend_out_info:
                cached_exp_info = blend_out_info['exposure']
            # Composite blended patch back onto preallocated canvas
            _canvas_r.fill(0); _canvas_g.fill(0); _canvas_b.fill(0)
            t, l = geo_min_top, geo_min_left
            _canvas_r[t:t+workheight, l:l+workwidth] = rgb_blended[0]
            _canvas_g[t:t+workheight, l:l+workwidth] = rgb_blended[1]
            _canvas_b[t:t+workheight, l:l+workwidth] = rgb_blended[2]
            canvas_r, canvas_g, canvas_b = _canvas_r, _canvas_g, _canvas_b

            # Gap fill using precomputed geometry (EDT + Gaussian smooth + feather)
            if geo_n_gap > 0:
                futs = [executor.submit(_gap_fill_channel, ch, buf)
                        for ch, buf in zip((canvas_r, canvas_g, canvas_b), _geo_fill_u8)]
                canvas_rgb = [f.result() for f in futs]
            else:
                canvas_rgb = [np.clip(c, 0, 255).astype(np.uint8) for c in (canvas_r, canvas_g, canvas_b)]

            y_final, u_final, v_final = rgb_to_yuv(canvas_rgb)

            # Apply fisheye circular mask
            if fisheye_mask:
                y_final[geo_outside_y] = 0
                u_final[geo_outside_uv] = 128
                v_final[geo_outside_uv] = 128

            if enhance:
                seed_y = int.from_bytes(os.urandom(4), 'little')
                y_final = enhance_filter(y_final, t=8, log2sizex=5, log2sizey=5, dither=6, seed=seed_y)
                u_final = enhance_filter(u_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
                v_final = enhance_filter(v_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
            
            # --- Update and encode the single, reused output frame ---
            out_frame.planes[0].update(y_final); out_frame.planes[1].update(u_final); out_frame.planes[2].update(v_final)
            # Set the Presentation Time Stamp (PTS)
            out_frame.pts = frame_count - 1
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)

    if not _quiet and total_frames > 0 and sys.stderr.isatty(): sys.stderr.write("\n"); sys.stderr.flush()

    for packet in out_stream.encode(): out_container.mux(packet)
    out_container.close(); [c.close() for c in in_containers]
    _print(f"\n✅ Success! Panoramic video saved to {output_file}")
    
    # Clean up seam cache file
    if seam_cache_file and os.path.exists(seam_cache_file):
        try:
            os.unlink(seam_cache_file)
        except:
            pass


def stitch(input_files, output_file, *, pto_file=None, projection='equirect',
           pad=0, padsides=None, enhance=False, fisheye_mask=False,
           force_video_dims=False, max_frames=0, level_subsample=1,
           sync=False, model=None, save_sync=None, load_sync=None,
           quiet=False, num_cores=None):
    """Stitch images or videos into a panoramic image or video.

    This is the public API entry point for programs that import stitcher.py.
    It mirrors the command-line interface and dispatches to either
    reproject_images() or reproject_videos() based on the input file types.

    Parameters
    ----------
    input_files : str or list[str]
        One or more input image or video files. All must be the same type.
    output_file : str
        Path for the output panorama.
    pto_file : str, optional
        Path to a Hugin .pto project file. If None, a PTO is generated from
        lens.pto files found relative to the inputs using ``projection``.
    projection : {'equirect', 'fisheye'}, optional
        Projection used when generating a PTO (default: 'equirect').
    pad : int, optional
        Pixels to pad source images before reprojection.
    padsides : str, set, or None, optional
        Sides to pad. None means all sides when pad > 0, otherwise none.
    enhance : bool, optional
        Apply the adaptive enhancement filter.
    fisheye_mask : bool, optional
        Apply a circular mask (fisheye output).
    force_video_dims : bool, optional
        Round output dimensions to multiples of 16 even for images.
    max_frames : int, optional
        Stop after encoding this many frames (video only).
    level_subsample : int, optional
        Recompute exposure correction every N frames (video only, default 1).
    sync : bool, optional
        Synchronize video streams by embedded timestamps.
    model : str, optional
        Timestamp model for synchronization.
    save_sync : str, optional
        JSON file to save the synchronization map.
    load_sync : str, optional
        JSON file to load a pre-computed synchronization map.
    quiet : bool, optional
        Suppress all text output.
    num_cores : int or None, optional
        Number of CPU cores to use. None uses all available cores.

    Raises
    ------
    ValueError
        On invalid arguments or mixed input types.
    FileNotFoundError
        If an input or PTO file is missing.
    RuntimeError
        If processing fails.
    ImportError
        If required modules are not available.
    """
    global _quiet
    _quiet = quiet

    if num_cores is None:
        try:
            num_cores = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cores = os.cpu_count() or 1
    numba.set_num_threads(num_cores)

    if isinstance(input_files, str):
        input_files = [input_files]
    input_files = list(input_files)

    # Expand glob patterns, preserving order and removing duplicates.
    expanded = []
    seen = set()
    for pattern in input_files:
        matches = glob.glob(pattern)
        if not matches:
            matches = [pattern]
        for f in matches:
            if f not in seen:
                seen.add(f)
                expanded.append(f)
    input_files = expanded

    if not input_files:
        raise ValueError("No input files specified.")

    if pto_file is None:
        if projection not in ('equirect', 'fisheye'):
            raise ValueError("projection must be 'equirect' or 'fisheye'")
        pto_file = generate_pto_from_lens_files(input_files, projection)
        if pto_file is None:
            raise RuntimeError("Failed to generate PTO file from lens.pto files.")
        auto_generated_pto = pto_file
    else:
        auto_generated_pto = None

    for f in [pto_file] + input_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Input file not found: {f}")

    is_image_input = all(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in input_files)
    is_video_input = all(f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')) for f in input_files)

    if not is_image_input and not is_video_input:
        raise ValueError("Input files must all be of the same type (either all images or all videos).")

    if padsides is None:
        padsides_set = {'top', 'bottom', 'left', 'right'} if pad > 0 else set()
    elif isinstance(padsides, str):
        padsides_set = set(s.strip() for s in padsides.split(',') if s.strip())
    else:
        padsides_set = set(padsides)

    if save_sync and load_sync:
        raise ValueError("save_sync and load_sync cannot be used at the same time.")
    if (save_sync or load_sync) and not sync:
        raise ValueError("save_sync and load_sync require sync=True.")

    if is_image_input:
        reproject_images(
            pto_file, input_files, output_file, pad, num_cores, padsides_set,
            enhance, force_video_dims=force_video_dims, fisheye_mask=fisheye_mask
        )
    else:
        if len(input_files) < 2 and sync:
            sync = False
        reproject_videos(
            pto_file, input_files, output_file,
            pad, num_cores, padsides_set, sync, model,
            save_sync_file=save_sync, load_sync_file=load_sync, enhance=enhance,
            fisheye_mask=fisheye_mask, max_frames=max_frames,
            level_subsample=level_subsample
        )

    if auto_generated_pto and os.path.exists(auto_generated_pto):
        try:
            os.unlink(auto_generated_pto)
        except Exception:
            pass


def extract_camera_number_from_path(path: str) -> int:
    """Extract camera number from a path like /meteor/cam1/20260621/07/full_00.jpg -> 1"""
    # Look for 'cam' followed by a number in the path
    match = re.search(r'/cam(\d+)/', path)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract camera number from path: {path}")


def find_lens_pto_for_image(image_path: str) -> str:
    """Find lens.pto file two directories up from the image path.
    
    For /meteor/cam1/20260621/07/full_00.jpg, look for /meteor/cam1/lens.pto
    """
    # Get the directory containing the image
    image_dir = os.path.dirname(os.path.abspath(image_path))
    # Go up two directories
    parent_dir = os.path.dirname(image_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    # Look for lens.pto in the grandparent directory
    lens_pto = os.path.join(grandparent_dir, "lens.pto")
    if os.path.exists(lens_pto):
        return lens_pto
    return None


def build_pto_header(w: int, h: int, projection: str) -> str:
    """Return the two-line PTO header for nona/hugin.
    projection: 'fisheye' (f3) or 'equirect' (f2)
    """
    f = 3 if projection == 'fisheye' else 2
    v = 190 if projection == 'fisheye' else 360
    return (f'p f{f} w{w} h{h} v{v} E0 R0 n"TIFF_m c:LZW"\n'
            f'm g1 i0 m2 p0.00784314\n')


def generate_pto_from_lens_files(input_files: list, projection: str) -> str:
    """Generate a PTO file from lens.pto files found relative to input files.
    
    Args:
        input_files: List of input image file paths
        projection: 'fisheye' or 'equirect'
    
    Returns:
        Path to the generated PTO file, or None on failure
    """
    # Define output dimensions based on projection
    if projection == 'fisheye':
        w, h = 4096, 4096
    else:  # equirect
        w, h = 4096, 2160
    
    # Find lens.pto files for each input
    lens_files = {}
    missing_lens = []
    for img_path in input_files:
        try:
            cam_num = extract_camera_number_from_path(img_path)
            lens_pto = find_lens_pto_for_image(img_path)
            if lens_pto is None:
                _print(f"Error: lens.pto not found for camera {cam_num} (image: {img_path})", file=sys.stderr)
                missing_lens.append(cam_num)
                continue
            lens_files[cam_num] = lens_pto
        except ValueError as e:
            _print(f"Error: {e}", file=sys.stderr)
            missing_lens.append("unknown")
            continue
    
    if not lens_files:
        _print("Error: No lens.pto files found for any input images", file=sys.stderr)
        return None
    
    if missing_lens:
        _print(f"Error: lens.pto files not found for cameras {missing_lens}. Cannot proceed without all calibration files.", file=sys.stderr)
        return None
    
    # Build PTO header
    header = build_pto_header(w, h, projection)
    
    # Detect actual input dimensions to scale lens calibration if needed.
    # All input files are assumed to have the same dimensions.
    actual_w, actual_h = None, None
    for img_path in input_files:
        try:
            with Image.open(img_path) as _im:
                actual_w, actual_h = _im.size
            break
        except Exception:
            pass
        # Fallback for video files: use av if available
        try:
            import av as _av
            with _av.open(img_path) as _vc:
                vs = _vc.streams.video[0]
                actual_w, actual_h = vs.width, vs.height
            break
        except Exception:
            continue

    # Build image lines from lens.pto files
    lines = [header]
    for cam_num in sorted(lens_files.keys()):
        lens_pto = lens_files[cam_num]
        try:
            with open(lens_pto, 'r') as f:
                for line in f:
                    if line.startswith('i ') or line.startswith('i\t'):
                        stripped = line.rstrip()
                        # Remove existing n"..." token
                        stripped = re.sub(r'\s+n"[^"]*"', '', stripped)

                        # Scale w/h/d/e if actual image differs from lens calibration.
                        if actual_w is not None:
                            cal_w_m = re.search(r'\bw(\d+)', stripped)
                            cal_h_m = re.search(r'\bh(\d+)', stripped)
                            if cal_w_m and cal_h_m:
                                cal_w = int(cal_w_m.group(1))
                                cal_h = int(cal_h_m.group(1))
                                if cal_w != actual_w or cal_h != actual_h:
                                    sx = actual_w / cal_w
                                    sy = actual_h / cal_h
                                    stripped = re.sub(r'\bw\d+', f'w{actual_w}', stripped)
                                    stripped = re.sub(r'\bh\d+', f'h{actual_h}', stripped)
                                    # Scale principal point offsets d/e (pixels).
                                    # Match standalone 'd' and 'e' tokens (space-preceded, not part of longer param names).
                                    stripped = re.sub(r'(?<=\s)d(-?[\d.]+)', lambda m: f'd{float(m.group(1))*sx:.6g}', stripped)
                                    stripped = re.sub(r'(?<=\s)e(-?[\d.]+)', lambda m: f'e{float(m.group(1))*sy:.6g}', stripped)

                        # Find the corresponding input file for this camera
                        for img_path in input_files:
                            try:
                                if extract_camera_number_from_path(img_path) == cam_num:
                                    img_rel = os.path.basename(img_path)
                                    lines.append(f'{stripped} n"{img_rel}"\n')
                                    break
                            except ValueError:
                                continue
                        break
        except OSError as e:
            _print(f"Warning: Could not read lens.pto for camera {cam_num}: {e}", file=sys.stderr)
            continue

    # Scale output canvas proportionally if input is not the calibration size.
    if actual_w is not None:
        # Derive calibration w from first lens.pto found
        cal_w_ref = None
        for lens_pto in lens_files.values():
            try:
                with open(lens_pto) as f:
                    for line in f:
                        if line.startswith('i '):
                            m = re.search(r'\bw(\d+)', line)
                            if m:
                                cal_w_ref = int(m.group(1))
                            break
            except Exception:
                pass
            if cal_w_ref:
                break
        if cal_w_ref and cal_w_ref != actual_w:
            scale = actual_w / cal_w_ref
            w = max(1, int(round(w * scale)))
            h = max(1, int(round(h * scale)))
            # Ensure even dimensions
            w = w & ~1; h = h & ~1
            _print(f"  Scaled output canvas to {w}x{h} for {actual_w}x{actual_h} input")
            # Rewrite header with new dimensions
            lines[0] = build_pto_header(w, h, projection)
    
    # Write PTO file to a temporary location
    pto_content = "".join(lines)
    pto_fd, pto_path = tempfile.mkstemp(suffix='.pto', prefix='auto_')
    try:
        with os.fdopen(pto_fd, 'w') as f:
            f.write(pto_content)
        _print(f"Generated PTO file with {len(lines)} lines for {len(lens_files)} cameras")

        if projection == 'fisheye':
            pto_data = pto_mapper.parse_pto_file(pto_path)
            pto_mapper.rotate_panorama(pto_data, yaw_deg=0, pitch_deg=-90, roll_deg=0)
            pto_mapper.write_pto_file(pto_data, pto_path)
            _print("Applied fisheye rotation (0,-90,0) via pto_mapper")

        return pto_path
    except Exception as e:
        _print(f"Error: Failed to write PTO file: {e}", file=sys.stderr)
        os.close(pto_fd)
        return None


def main():
    try:
        num_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cores = os.cpu_count() or 1
    numba.set_num_threads(num_cores)

    parser = argparse.ArgumentParser(
        description="Reproject and stitch images or videos into a panorama based on a Hugin .pto file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("pto_file", nargs='?', help="Path to the Hugin PTO project file. Required unless --fisheye or --equirect is specified.")
    parser.add_argument("input_files", nargs='+', help="One or more input image or video files (must all be same type).")
    parser.add_argument("output_file", help="Path for the output panoramic image or video.")
    parser.add_argument("--fisheye", action='store_true', help="Generate fisheye panorama (8192x8192). Automatically creates PTO from lens.pto files found two directories up from input files.")
    parser.add_argument("--equirect", action='store_true', help="Generate equirectangular panorama (3380x2240). Automatically creates PTO from lens.pto files found two directories up from input files.")
    parser.add_argument("--enhance", action='store_true', help="Apply an adaptive enhancement filter to reduce noise and artifacts.")
    parser.add_argument("--force-video-dims", action='store_true', help="Force codec-safe output dimensions (video rules) even when input files are images.")
    parser.add_argument("--quiet", action='store_true', help="Suppress all text output.")
    parser.add_argument("--pad", type=int, default=0, help="Pixels to pad source images before reprojection (extends edges with blurred content).")
    parser.add_argument("--padsides", type=str, default="", help="Comma-separated sides to pad: top,bottom,left,right (default: all sides if --pad > 0).")
    
    parser.add_argument("-n", "--max-frames", type=int, default=0, metavar="N",
        help="Stop after encoding N frames (video only). Useful for quick tests.")
    parser.add_argument("--level-subsample", type=int, default=1, metavar="N",
        help="Recompute exposure correction only every N frames in video mode (default: 1). Higher values are faster.")

    sync_group = parser.add_argument_group('Video Synchronization Options')
    sync_group.add_argument("--sync", action='store_true', help="Synchronize video streams by their embedded timestamps before stitching.")
    sync_group.add_argument("--model", type=str, default=None, help="Specify the model for timestamp extraction.")
    sync_group.add_argument("--save-sync", type=str, default=None, help="Save the synchronization map to a JSON file (requires --sync).")
    sync_group.add_argument("--load-sync", type=str, default=None, help="Load a pre-computed synchronization map from a JSON file (requires --sync).")
    
    args = parser.parse_args()

    global _quiet
    _quiet = args.quiet

    _print(f"INFO: Detected {num_cores} available CPU cores.")

    # If --fisheye or --equirect is used, shift arguments: pto_file should be None and the first input file should be moved from pto_file to input_files
    if (args.fisheye or args.equirect) and args.pto_file is not None:
        # Check if pto_file looks like an image file (not a .pto file)
        if not args.pto_file.lower().endswith('.pto'):
            # Move pto_file to the beginning of input_files
            args.input_files.insert(0, args.pto_file)
            args.pto_file = None

    # --- Argument Validation ---
    if args.save_sync and args.load_sync:
        _print("Error: --save-sync and --load-sync cannot be used at the same time.", file=sys.stderr); sys.exit(1)
    if (args.save_sync or args.load_sync) and not args.sync:
        _print("Error: --save-sync and --load-sync require the --sync flag to be enabled.", file=sys.stderr); sys.exit(1)
    if len(args.input_files) < 2:
        if args.sync: _print("INFO: --sync option ignored for a single input file."); args.sync = False
    if not args.input_files:
        _print("Error: No input files specified.", file=sys.stderr); sys.exit(1)

    # Validate --fisheye and --equirect options
    if args.fisheye and args.equirect:
        _print("Error: --fisheye and --equirect are mutually exclusive.", file=sys.stderr); sys.exit(1)
    if not args.pto_file and not (args.fisheye or args.equirect):
        _print("Error: Either pto_file or --fisheye/--equirect must be specified.", file=sys.stderr); sys.exit(1)
    if args.level_subsample < 1:
        _print("Error: --level-subsample must be a positive integer.", file=sys.stderr); sys.exit(1)

    # If --fisheye or --equirect is specified, generate PTO file from lens.pto files
    auto_generated_pto = None
    if args.fisheye or args.equirect:
        projection = 'fisheye' if args.fisheye else 'equirect'
        
        # Expand glob patterns in input_files
        expanded_input_files = []
        for pattern in args.input_files:
            matches = glob.glob(pattern)
            if matches:
                expanded_input_files.extend(matches)
            else:
                # If no matches, keep the original pattern (might be a literal path)
                expanded_input_files.append(pattern)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_input_files = []
        for f in expanded_input_files:
            if f not in seen:
                seen.add(f)
                unique_input_files.append(f)
        
        args.input_files = unique_input_files
        _print(f"Expanded input files to {len(args.input_files)} files")
        
        pto_file = generate_pto_from_lens_files(args.input_files, projection)
        if pto_file is None:
            _print("Error: Failed to generate PTO file from lens.pto files.", file=sys.stderr); sys.exit(1)
        args.pto_file = pto_file
        auto_generated_pto = pto_file
        _print(f"Generated PTO file: {pto_file}")

    for f in [args.pto_file] + args.input_files:
        if not os.path.exists(f):
            _print(f"Error: Input file not found: {f}", file=sys.stderr); sys.exit(1)

    is_image_input = all(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in args.input_files)
    is_video_input = all(f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')) for f in args.input_files)
    
    # --- Main Execution with Global Error Handling ---
    try:
        padsides = set(s.strip() for s in args.padsides.split(',') if s.strip()) if args.padsides else ({'top','bottom','left','right'} if args.pad > 0 else set())
        if is_image_input:
            reproject_images(args.pto_file, args.input_files, args.output_file, args.pad, num_cores, padsides, args.enhance, force_video_dims=args.force_video_dims, fisheye_mask=args.fisheye)
        elif is_video_input:
            reproject_videos(
                args.pto_file, args.input_files, args.output_file,
                args.pad, num_cores, padsides, args.sync, args.model,
                save_sync_file=args.save_sync, load_sync_file=args.load_sync, enhance=args.enhance,
                fisheye_mask=args.fisheye, max_frames=args.max_frames,
                level_subsample=args.level_subsample
            )
        else:
            _print("Error: Input files must all be of the same type (either all images or all videos).", file=sys.stderr)
            sys.exit(1)
    except (ValueError, FileNotFoundError, ImportError, IOError, RuntimeError, KeyboardInterrupt) as e:
        _print(f"\n❌ An error occurred during processing:\n{e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        _print(f"\n❌ An unexpected critical error occurred:\n{e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up auto-generated PTO file
        if auto_generated_pto and os.path.exists(auto_generated_pto):
            try:
                os.unlink(auto_generated_pto)
                _print(f"Cleaned up temporary PTO file: {auto_generated_pto}")
            except Exception as e:
                _print(f"Warning: Failed to clean up temporary PTO file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
