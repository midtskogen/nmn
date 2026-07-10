#!/usr/bin/env python3

import sys
import os
import shutil
import subprocess
import threading
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
import shlex
import tempfile
import hashlib
import time
import glob
import gc

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

# Lazy getters for heavy optional modules. Importing them at module load adds
# ~0.5 s to image-stitch startup even though they are only used for video,
# enhancement, or specific helpers.
_stack_mod = None

def _enhance_filter():
    global _stack_mod
    if _stack_mod is None:
        try:
            from stack import enhance_filter as _ef
            _stack_mod = _ef
        except ImportError as e:
            raise ImportError(
                "The 'stack.py' module was not found. "
                "Please ensure 'stack.py' (containing the enhancement filter) is in the same directory."
            ) from e
    return _stack_mod

_av_mod = None

def _av():
    global _av_mod
    if _av_mod is None:
        try:
            import av as _av_local
            _av_local.logging.set_level(_av_local.logging.ERROR)
            _av_mod = _av_local
        except ImportError:
            print("Warning: 'PyAV' library not found. Video processing functionality will be unavailable.", file=sys.stderr)
            _av_mod = False
    return _av_mod

_cv2_mod = None

def _cv2():
    global _cv2_mod
    if _cv2_mod is None:
        try:
            import cv2 as _cv2_local
            _cv2_mod = _cv2_local
        except ImportError:
            _cv2_mod = False
    return _cv2_mod

_zstd_mod = None

def _zstd():
    global _zstd_mod
    if _zstd_mod is None:
        try:
            import zstandard as _zstd_local
            _zstd_mod = _zstd_local
        except ImportError:
            _zstd_mod = False
    return _zstd_mod

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
def _build_vignette_gain(width, height, k1):
    """Build a per-pixel multiplicative gain LUT to undo radial vignetting.

    Model:  brightness(r) = 1 + k1*r^2
    where r is distance from centre normalised so that the corner = 1.
    Gain = 1 / brightness(r), clamped to [1, 4] to avoid blowup.

    Typical usage: k1 < 0 (darkening toward edges).  E.g. k1=-0.5
    gives ~33 % brightening at the corners.
    """
    gain = np.empty((height, width), dtype=np.float32)
    cx = width  * 0.5
    cy = height * 0.5
    # Work with r² directly — avoids a sqrt per pixel
    inv_r2max = 1.0 / max(cx * cx + cy * cy, 1.0)
    for y in prange(height):
        dy = y - cy
        dy2 = dy * dy
        for x in range(width):
            dx = x - cx
            r2 = (dx * dx + dy2) * inv_r2max      # normalised r²
            v = 1.0 + k1 * r2
            g = 1.0 / v if v > 0.25 else 4.0       # clamp gain ≤ 4
            if g < 1.0:
                g = 1.0   # never darken
            gain[y, x] = g
    return gain


@numba.njit(parallel=True, fastmath=True, cache=True)
def _apply_vignette_y(py, gain):
    """Multiply Y plane (uint8) by per-pixel gain in-place, clipping to 255."""
    h, w = py.shape
    for y in prange(h):
        for x in range(w):
            v = py[y, x] * gain[y, x]
            if v > 255.0:
                v = 255.0
            py[y, x] = np.uint8(v)


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
def reproject_y(py, dw, dh, sw, map_y_idx, c01, c23, out_y, fisheye_mask=None, crop_h=None, r0=0, r1=-1, c0=0, c1=-1):
    # r0:r1, c0:c1 optionally restrict iteration to the camera's valid bounding
    # box (pixels outside never receive kernel writes; callers pre-fill outputs).
    dh_limit = dh if crop_h is None else min(dh, crop_h)
    row_end = dh_limit if r1 < 0 else min(dh_limit, r1)
    col_end = dw if c1 < 0 else c1
    for yi in prange(r0, row_end):
        base_out = yi * dw
        base_map = yi * dw
        for xi in range(c0, col_end):
            if fisheye_mask is not None and fisheye_mask[base_out + xi]:
                out_y[base_out + xi] = 0
                continue
            idx = map_y_idx[base_map + xi]
            if idx < 0: continue
            weights01, weights23 = c01[base_map + xi], c23[base_map + xi]
            w0, w1, w2, w3 = (weights01 >> 8) & 0xFF, weights01 & 0xFF, (weights23 >> 8) & 0xFF, weights23 & 0xFF
            interpolated_value = (py[idx] * w0 + py[idx + 1] * w1 + py[idx + sw] * w2 + py[idx + sw + 1] * w3) >> 7
            out_y[base_out + xi] = interpolated_value

@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def reproject_float(p_float_src, dw, dh, sw, map_y_idx, c01, c23, out_float, fisheye_mask=None, crop_h=None, r0=0, r1=-1, c0=0, c1=-1):
    dh_limit = dh if crop_h is None else min(dh, crop_h)
    row_end = dh_limit if r1 < 0 else min(dh_limit, r1)
    col_end = dw if c1 < 0 else c1
    for yi in prange(r0, row_end):
        base_out = yi * dw
        base_map = yi * dw
        for xi in range(c0, col_end):
            if fisheye_mask is not None and fisheye_mask[base_out + xi]:
                out_float[base_out + xi] = 0.0
                continue
            idx = map_y_idx[base_map + xi]
            if idx < 0: continue
            weights01, weights23 = c01[base_map + xi], c23[base_map + xi]
            w0, w1, w2, w3 = (weights01 >> 8) & 0xFF, weights01 & 0xFF, (weights23 >> 8) & 0xFF, weights23 & 0xFF
            interpolated_value = (p_float_src[idx] * w0 + p_float_src[idx + 1] * w1 + p_float_src[idx + sw] * w2 + p_float_src[idx + sw + 1] * w3) / 128.0
            out_float[base_out + xi] = interpolated_value

@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def reproject_uv(pu, pv, dw, dh, map_uv_idx, out_u, out_v, fisheye_mask=None, crop_h=None, r0=0, r1=-1, c0=0, c1=-1):
    # r0:r1, c0:c1 are in UV (half-resolution) grid coordinates.
    half_w, half_h = dw // 2, dh // 2
    half_h_limit = half_h if crop_h is None else min(half_h, (crop_h + 1) // 2)
    row_end = half_h_limit if r1 < 0 else min(half_h_limit, r1)
    col_end = half_w if c1 < 0 else c1
    for y_uv in prange(r0, row_end):
        base_uv = y_uv * half_w
        for x_uv in range(c0, col_end):
            if fisheye_mask is not None and fisheye_mask[base_uv + x_uv]:
                out_u[base_uv + x_uv] = 128
                out_v[base_uv + x_uv] = 128
                continue
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

    # Projection maps are deterministic given the PTO content and parameters —
    # cache them on disk so repeated runs with the same geometry skip the
    # expensive per-pixel trig (~2s for 7 cameras). Arrays are cropped to the
    # valid-pixel bounding box, index maps are delta-encoded (near-constant
    # increments compress ~8x better), and everything is stored in an npz file;
    # full-size arrays are reconstructed on load. All transforms are lossless.
    # If zstandard is installed the npz is zstd-compressed (smaller than
    # uncompressed, faster decompression than zlib), otherwise it is stored
    # uncompressed for raw read speed.
    _MAP_CACHE_VERSION = 5
    cache_path = None
    try:
        import hashlib, re
        with open(pto_file, 'rb') as _f:
            _pto_bytes = _f.read()
        # Projection maps depend only on lens/output geometry, not on the
        # input image filenames. Strip the per-image n"..." / nfilename tokens
        # so the cache key stays stable across different minutes/files.
        _pto_norm = re.sub(rb'\s+n(?:"[^"]*"|[^\s"]+)', b'', _pto_bytes)
        _key_src = _pto_norm + repr((_MAP_CACHE_VERSION, pad, sorted(padsides), is_video_output)).encode()
        _key = hashlib.sha256(_key_src).hexdigest()[:24]
        _cache_dir = os.path.join(tempfile.gettempdir(), 'stitcher_map_cache')
        cache_path = os.path.join(_cache_dir, f'maps_{_key}.npz')
        _zst_path = cache_path + '.zst'
        _has_zstd = bool(_zstd())

        def _undelta(d, shape):
            """Invert np.diff(..., prepend=0) delta encoding (exact)."""
            return np.cumsum(d.ravel(), dtype=np.int64).astype(np.int32).reshape(shape)

        def _load_from_npz(_npz_path):
            with np.load(_npz_path) as _npz:
                n = int(_npz['n'])
                _mappings = []
                for i in range(n):
                    h, w, r0, c0 = (int(v) for v in _npz[f'ybox{i}'])
                    my_d = _npz[f'my{i}']
                    my_c = _undelta(my_d, my_d.shape)
                    my = np.full(h * w, -1, dtype=np.int32)
                    my.reshape(h, w)[r0:r0 + my_c.shape[0], c0:c0 + my_c.shape[1]] = my_c
                    c01 = np.zeros((h, w), dtype=np.uint16)
                    c23 = np.zeros((h, w), dtype=np.uint16)
                    c01[r0:r0 + my_c.shape[0], c0:c0 + my_c.shape[1]] = _npz[f'c01_{i}']
                    c23[r0:r0 + my_c.shape[0], c0:c0 + my_c.shape[1]] = _npz[f'c23_{i}']
                    uh, uw, ur0, uc0 = (int(v) for v in _npz[f'uvbox{i}'])
                    uv_d = _npz[f'uv{i}']
                    uv_c = _undelta(uv_d, uv_d.shape)
                    uv = np.full((uh, uw), -1, dtype=np.int32)
                    uv[ur0:ur0 + uv_c.shape[0], uc0:uc0 + uv_c.shape[1]] = uv_c
                    _mappings.append((my, c01, c23, uv,
                                    int(_npz[f'sw{i}']), int(_npz[f'sh{i}'])))
            return n, _mappings

        if _has_zstd and os.path.exists(_zst_path):
            _tmp_load = cache_path + f'.load{os.getpid()}.npz'
            try:
                with open(_zst_path, 'rb') as _src, open(_tmp_load, 'wb') as _dst:
                    _zstd().ZstdDecompressor().copy_stream(_src, _dst)
                n, all_mappings = _load_from_npz(_tmp_load)
                if n == len(images):
                    _print("Loaded projection maps from cache (zstd).")
                    return all_mappings, global_options
            finally:
                try:
                    os.unlink(_tmp_load)
                except OSError:
                    pass
        elif os.path.exists(cache_path):
            n, all_mappings = _load_from_npz(cache_path)
            if n == len(images):
                _print("Loaded projection maps from cache.")
                return all_mappings, global_options
    except Exception:
        cache_path = None

    _print("Building projection maps...")
    all_mappings = [_map_one_image(args) for args in task_args]

    if cache_path is not None:
        try:
            os.makedirs(_cache_dir, exist_ok=True)
            # Opportunistically prune stale entries (>10 days old) and cap the
            # cache at the 3 most recent geometries.
            _now = time.time()
            _entries = []
            for _fn in os.listdir(_cache_dir):
                _fp = os.path.join(_cache_dir, _fn)
                try:
                    _mt = os.path.getmtime(_fp)
                    if _now - _mt > 10 * 86400:
                        os.unlink(_fp)
                    else:
                        _entries.append((_mt, _fp))
                except OSError:
                    pass
            _entries.sort(reverse=True)
            for _, _fp in _entries[3:]:
                try:
                    os.unlink(_fp)
                except OSError:
                    pass
            def _bbox(valid2d):
                rows = np.any(valid2d, axis=1); cols = np.any(valid2d, axis=0)
                if not rows.any():
                    return 0, 0, 0, 0
                r0 = int(rows.argmax()); r1 = int(len(rows) - rows[::-1].argmax())
                c0 = int(cols.argmax()); c1 = int(len(cols) - cols[::-1].argmax())
                return r0, r1, c0, c1

            def _delta(a2d):
                """Delta-encode a cropped int32 index map (lossless; diffs of
                values in [-1, ~4e6] always fit int32)."""
                flat = a2d.ravel().astype(np.int64)
                return np.diff(flat, prepend=0).astype(np.int32).reshape(a2d.shape)

            _payload = {'n': np.int64(len(all_mappings))}
            for i, (my, c01, c23, uv, sw, sh) in enumerate(all_mappings):
                h, w = c01.shape
                my2 = my.reshape(h, w)
                r0, r1, c0, c1 = _bbox(my2 >= 0)
                _payload[f'ybox{i}'] = np.array([h, w, r0, c0], dtype=np.int64)
                _payload[f'my{i}'] = _delta(np.ascontiguousarray(my2[r0:r1, c0:c1]))
                _payload[f'c01_{i}'] = np.ascontiguousarray(c01[r0:r1, c0:c1])
                _payload[f'c23_{i}'] = np.ascontiguousarray(c23[r0:r1, c0:c1])
                uh, uw = uv.shape
                ur0, ur1, uc0, uc1 = _bbox(uv >= 0)
                _payload[f'uvbox{i}'] = np.array([uh, uw, ur0, uc0], dtype=np.int64)
                _payload[f'uv{i}'] = _delta(np.ascontiguousarray(uv[ur0:ur1, uc0:uc1]))
                _payload[f'sw{i}'] = np.int64(sw); _payload[f'sh{i}'] = np.int64(sh)
            _tmp_path = cache_path + f'.tmp{os.getpid()}'
            np.savez(_tmp_path, **_payload)
            _final_npz = _tmp_path + '.npz' if os.path.exists(_tmp_path + '.npz') else _tmp_path
            if _has_zstd:
                _zst_tmp = _zst_path + f'.tmp{os.getpid()}'
                try:
                    with open(_final_npz, 'rb') as _src, _zstd().open(_zst_tmp, 'wb') as _dst:
                        shutil.copyfileobj(_src, _dst)
                    os.replace(_zst_tmp, _zst_path)
                    # Prefer the zst file; remove the uncompressed duplicate to save space.
                    try:
                        os.unlink(_final_npz)
                    except OSError:
                        pass
                except Exception:
                    # If zstd compression fails for any reason, keep the uncompressed file.
                    os.replace(_final_npz, cache_path)
                    try:
                        os.unlink(_zst_tmp)
                    except OSError:
                        pass
            else:
                os.replace(_final_npz, cache_path)
        except Exception:
            pass

    return all_mappings, global_options


# ---------------------------------------------------------------------------
# Seam assignment cache
# ---------------------------------------------------------------------------
# compute_seams() is deterministic given the PTO geometry, padding, output
# dimensions and seam flags. The result is an image-index assignment array
# that can be reused across runs (image or video) when those inputs match,
# avoiding the ~100-300 ms seaming cost on every invocation.
_SEAM_CACHE_VERSION = 2

def _get_seam_cache_path(pto_file, pad, padsides, is_video_output,
                         workwidth, workheight, reverse, simple_seam, content_seam):
    """Return deterministic cache path for a seam assignment."""
    with open(pto_file, 'rb') as _f:
        _pto_bytes = _f.read()
    # Make key stable across input filenames; geometry depends on parameters only.
    _pto_norm = re.sub(rb'\s+n(?:"[^"]*"|[^\s"]+)', b'', _pto_bytes)
    _key_src = (_pto_norm +
                repr((_SEAM_CACHE_VERSION, pad, sorted(padsides), is_video_output,
                      workwidth, workheight, reverse, simple_seam, content_seam)).encode())
    _key = hashlib.sha256(_key_src).hexdigest()[:24]
    _cache_dir = os.path.join(tempfile.gettempdir(), 'stitcher_seam_cache')
    return os.path.join(_cache_dir, f'seams_{_key}.npz'), _cache_dir


def _prune_cache_dir(cache_dir, max_age_days=10, max_entries=3):
    """Prune stale cache entries and cap at the newest N files."""
    try:
        _now = time.time()
        _entries = []
        for _fn in os.listdir(cache_dir):
            _fp = os.path.join(cache_dir, _fn)
            try:
                _mt = os.path.getmtime(_fp)
                if _now - _mt > max_age_days * 86400:
                    os.unlink(_fp)
                else:
                    _entries.append((_mt, _fp))
            except OSError:
                pass
        _entries.sort(reverse=True)
        for _, _fp in _entries[max_entries:]:
            try:
                os.unlink(_fp)
            except OSError:
                pass
    except Exception:
        pass


def _prune_seam_cache(cache_dir):
    """Prune stale seam-cache entries (>10 days old) and cap at 3 newest."""
    _prune_cache_dir(cache_dir, max_age_days=10, max_entries=3)


def compute_or_load_seams(images, workwidth, workheight, pto_file, pad, padsides,
                          levels,
                          is_video_output=False,
                          reverse=False, simple_seam=False, content_seam=False,
                          verbosity=1, print_func=print):
    """Compute seam assignment and seam-mask pyramids, caching the assignment.

    The mask pyramids are fully determined by the assignment and image
    footprints, so only the assignment is stored on disk. Pyramids are rebuilt
    on load; this is faster than decompressing many float32 arrays and makes
    the cache much smaller.

    The cache is shared between image and video stitching runs as long as the
    PTO file, padding, output mode, work dimensions, levels and seam flags match.
    """
    cache_path, cache_dir = _get_seam_cache_path(
        pto_file, pad, padsides, is_video_output,
        workwidth, workheight, reverse, simple_seam, content_seam)

    try:
        if os.path.exists(cache_path):
            if verbosity >= 2:
                print_func(f"  seam cache: trying {cache_path}")
            with np.load(cache_path) as _npz:
                assignment = _npz['assignment']
                seam_present = [bool(v) for v in _npz['seam_present']]
                n = int(_npz['n'])
                levels_loaded = int(_npz['levels'])
                if (assignment.shape == (workheight, workwidth) and
                        len(seam_present) == len(images) and
                        n == len(images) and levels_loaded == levels):
                    seam_mask_cache = multiblend.build_seam_mask_cache(
                        images, assignment, workwidth, workheight, levels)
                    if verbosity >= 1:
                        print_func("  loaded seams from cache.")
                    return assignment, seam_present, seam_mask_cache
                elif verbosity >= 2:
                    print_func("  seam cache: shape/count/levels mismatch, rebuilding.")
    except Exception as _e:
        print_func(f"  seam cache load failed: {_e}", file=sys.stderr)

    assignment, seam_present = multiblend.compute_seams(
        images=images,
        workwidth=workwidth,
        workheight=workheight,
        reverse=reverse,
        simple_seam=simple_seam,
        content_seam=content_seam,
        verbosity=verbosity,
        print_func=print_func,
    )
    seam_mask_cache = multiblend.build_seam_mask_cache(
        images, assignment, workwidth, workheight, levels)

    try:
        os.makedirs(cache_dir, exist_ok=True)
        _prune_seam_cache(cache_dir)
        _tmp_path = cache_path + f'.tmp{os.getpid()}'
        _payload = {
            'assignment': np.ascontiguousarray(assignment),
            'seam_present': np.array(seam_present, dtype=np.uint8),
            'n': np.int64(len(images)),
            'levels': np.int64(levels),
        }
        np.savez_compressed(_tmp_path, **_payload)
        os.replace(_tmp_path + '.npz' if os.path.exists(_tmp_path + '.npz') else _tmp_path,
                   cache_path)
        if verbosity >= 2:
            print_func(f"  seam cache saved: {cache_path}")
    except Exception as _e:
        print_func(f"  seam cache save failed: {_e}", file=sys.stderr)

    return assignment, seam_present, seam_mask_cache


# ---------------------------------------------------------------------------
# Gap geometry cache
# ---------------------------------------------------------------------------
# The EDT index maps and feather weights used for gap filling depend only on
# the PTO geometry, padding and crop options. They can be precomputed once,
# cached on disk, and reused across runs.
_GAP_CACHE_VERSION = 4


def _get_gap_cache_path(pto_file, pad, padsides, is_video_output,
                        final_w, final_h, workwidth, workheight,
                        min_left, min_top, fisheye_mask, crop_to_content):
    """Return deterministic cache path for gap-fill geometry."""
    with open(pto_file, 'rb') as _f:
        _pto_bytes = _f.read()
    # Make key stable across input filenames; geometry depends on parameters only.
    _pto_norm = re.sub(rb'\s+n(?:"[^"]*"|[^\s"]+)', b'', _pto_bytes)
    _key_src = (_pto_norm +
                repr((_GAP_CACHE_VERSION, pad, sorted(padsides), is_video_output,
                      final_w, final_h, workwidth, workheight,
                      min_left, min_top, fisheye_mask, crop_to_content)).encode())
    _key = hashlib.sha256(_key_src).hexdigest()[:24]
    _cache_dir = os.path.join(tempfile.gettempdir(), 'stitcher_gap_cache')
    return os.path.join(_cache_dir, f'gap_{_key}.npz'), _cache_dir


def _compute_gap_geometry(gap: np.ndarray, S: int = 8, sigma_s: int = 4):
    """Precompute EDT index maps and feather weights for gap filling.

    Returns (ri, ci, blend_w, sw, sh, feather_radius). All returned arrays are
    derived from the boolean gap mask only, so they can be cached and reused.
    Index maps are stored as int16 when the canvas fits (<= 32767) to halve
    the memory footprint versus int32; feather weights are stored as uint16
    (scaled 0..65535) to halve versus float32.
    """
    H, W = gap.shape
    sw, sh = max(1, W // S), max(1, H // S)
    feather_radius = max(1, round(20 * W / 4096))

    from scipy.ndimage import distance_transform_edt as _edt

    ph = ((H + S - 1) // S) * S
    pw = ((W + S - 1) // S) * S
    gap_pad = np.zeros((ph, pw), dtype=bool)
    gap_pad[:H, :W] = gap
    gap_ds = gap_pad[::S, ::S]
    ri_ds, ci_ds = _edt(gap_ds, return_distances=False, return_indices=True)
    # int16 is sufficient for current canvas sizes (<= 8192) and saves ~50%.
    _idx_dt = np.int16 if max(H, W) <= 32767 else np.int32
    ri = np.repeat(np.repeat((ri_ds * S).astype(_idx_dt), S, axis=0), S, axis=1)[:H, :W]
    ci = np.repeat(np.repeat((ci_ds * S).astype(_idx_dt), S, axis=0), S, axis=1)[:H, :W]

    dist = _edt(~gap)
    blend_w = np.clip(dist / feather_radius, 0.0, 1.0)
    blend_w = (blend_w * 65535.0).astype(np.uint16)

    return ri, ci, blend_w, sw, sh, feather_radius


def compute_or_load_gap_geometry(gap, pto_file, pad, padsides,
                                 final_w, final_h, workwidth, workheight,
                                 min_left, min_top,
                                 fisheye_mask, crop_to_content,
                                 verbosity=1, print_func=print):
    """Compute or load cached gap-fill geometry (EDT maps + feather weights).

    The geometry depends only on the gap mask, which itself depends on the PTO
    geometry, padding and crop options. Cached files are stored under
    /tmp/stitcher_gap_cache and pruned like the seam cache.
    """
    cache_path, cache_dir = _get_gap_cache_path(
        pto_file, pad, padsides, is_video_output=False,
        final_w=final_w, final_h=final_h,
        workwidth=workwidth, workheight=workheight,
        min_left=min_left, min_top=min_top,
        fisheye_mask=fisheye_mask, crop_to_content=crop_to_content)

    H, W = gap.shape
    try:
        if os.path.exists(cache_path):
            if verbosity >= 2:
                print_func(f"  gap cache: trying {cache_path}")
            with np.load(cache_path) as _npz:
                if (int(_npz['H']) == H and int(_npz['W']) == W):
                    ri = _npz['ri']
                    ci = _npz['ci']
                    blend_w = _npz['blend_w']
                    sw = int(_npz['sw'])
                    sh = int(_npz['sh'])
                    feather_radius = int(_npz['feather_radius'])
                    if verbosity >= 1:
                        print_func("  loaded gap geometry from cache.")
                    return ri, ci, blend_w, sw, sh, feather_radius
                elif verbosity >= 2:
                    print_func("  gap cache: dimensions mismatch, rebuilding.")
    except Exception as _e:
        print_func(f"  gap cache load failed: {_e}", file=sys.stderr)

    from scipy.ndimage import distance_transform_edt as _edt, gaussian_filter
    ri, ci, blend_w, sw, sh, feather_radius = _compute_gap_geometry(gap)

    try:
        os.makedirs(cache_dir, exist_ok=True)
        _prune_cache_dir(cache_dir, max_age_days=10, max_entries=3)
        _tmp_path = cache_path + f'.tmp{os.getpid()}'
        _payload = {
            'ri': np.ascontiguousarray(ri),
            'ci': np.ascontiguousarray(ci),
            'blend_w': np.ascontiguousarray(blend_w),
            'H': np.int64(H),
            'W': np.int64(W),
            'sw': np.int64(sw),
            'sh': np.int64(sh),
            'feather_radius': np.int64(feather_radius),
        }
        np.savez_compressed(_tmp_path, **_payload)
        os.replace(_tmp_path + '.npz' if os.path.exists(_tmp_path + '.npz') else _tmp_path,
                   cache_path)
        if verbosity >= 2:
            print_func(f"  gap cache saved: {cache_path}")
    except Exception as _e:
        print_func(f"  gap cache save failed: {_e}", file=sys.stderr)

    return ri, ci, blend_w, sw, sh, feather_radius


def estimate_noise(image_plane):
    """
    Estimates the noise standard deviation of an image plane using the
    standard deviation of its Laplacian. Returns a value between 1.0 and 10.0.
    """
    try:
        from scipy.ndimage import laplace
    except ImportError:
        # Fallback to a default value if scipy is not installed
        return 4.0

    # The Laplacian filter is sensitive to high-frequency noise
    laplacian = laplace(image_plane.astype(np.float32))

    # The standard deviation of the Laplacian is a robust noise estimator
    noise_std = np.std(laplacian)

    # Clamp the value to a reasonable range to avoid extreme results
    return np.clip(noise_std, 1.0, 10.0)

def load_image_to_yuv(image_path, pad, padsides, target_w=None, target_h=None):
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

    # Resize to square-pixel display dimensions when source has non-square pixels
    # (e.g. SD 704x576 stored for 1920x1080 FOV content, display width = 1024).
    if target_w is not None and target_h is not None and (img_pil.width != target_w or img_pil.height != target_h):
        img_pil = img_pil.resize((target_w, target_h), resample_filter)

    img_ycbcr = img_pil.convert("YCbCr")
    y, u, v = img_ycbcr.split()
    
    # Use the determined resampling filter
    u_resized = u.resize((img_pil.width // 2, img_pil.height // 2), resample_filter)
    v_resized = v.resize((img_pil.width // 2, img_pil.height // 2), resample_filter)
    
    pad_t = pad if 'top' in padsides else 0
    pad_b = pad if 'bottom' in padsides else 0
    pad_l = pad if 'left' in padsides else 0
    pad_r = pad if 'right' in padsides else 0

    pad_uv_t, pad_uv_b, pad_uv_l, pad_uv_r = pad_t//2, pad_b//2, pad_l//2, pad_r//2
    pad_y_width = ((pad_t, pad_b), (pad_l, pad_r))
    pad_uv_width = ((pad_uv_t, pad_uv_b), (pad_uv_l, pad_uv_r))

    y_arr = np.array(y, np.uint8)
    padded_y = np.pad(y_arr, pad_y_width, mode='edge')
    padded_u = np.pad(np.array(u_resized, np.uint8), pad_uv_width, mode='edge')
    padded_v = np.pad(np.array(v_resized, np.uint8), pad_uv_width, mode='edge')

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
    (input_path, dw, dh, mapping, pad, padsides, devignette, fisheye_mask), out_buffers = args
    reproj_y, reproj_u, reproj_v, reproj_weights_y = out_buffers

    map_y_idx, c01, c23, map_uv_idx, sw_pto, sh_pto = mapping
    py, pu, pv, sw_orig, sh_orig = load_image_to_yuv(
        input_path, pad, padsides, target_w=sw_pto, target_h=sh_pto)

    if devignette is not None:
        gain = _build_vignette_gain(sw_orig, sh_orig, devignette)
        # Apply to original region only (not padding)
        pad_t_ = pad if 'top' in padsides else 0
        pad_l_ = pad if 'left' in padsides else 0
        _apply_vignette_y(py[pad_t_:pad_t_ + sh_orig, pad_l_:pad_l_ + sw_orig], gain)

    pad_t = pad if 'top' in padsides else 0
    pad_b = pad if 'bottom' in padsides else 0
    pad_l = pad if 'left' in padsides else 0
    pad_r = pad if 'right' in padsides else 0

    x1, y1, x2, y2 = _TIMESTAMP_BOX_HD if sh_orig >= 900 else _TIMESTAMP_BOX_SD
    # Expand by margin to cover bilinear interpolation bleed during reprojection
    _ts_m = 3
    ex1 = max(0, x1 - _ts_m)
    ey1 = max(0, y1 - _ts_m)
    ex2 = min(sw_orig, x2 + _ts_m)
    ey2 = min(sh_orig, y2 + _ts_m)
    # No pixel erasure — rely on zeroed blend weights to exclude timestamp region

    sw_padded, sh_padded = sw_orig + pad_l + pad_r, sh_orig + pad_t + pad_b

    # Weight map for the non-padded region only — computed on original size,
    # then embedded in a zero-padded canvas so padded pixels have zero weight
    # and never win seams, but their pixel data still fills the FOV extension.
    inner_weights = create_blend_weight_map(sw_orig, sh_orig)
    blend_weights_y = np.zeros((sh_padded, sw_padded), dtype=np.float32)
    blend_weights_y[pad_t:pad_t + sh_orig, pad_l:pad_l + sw_orig] = inner_weights
    blend_weights_y[ey1 + pad_t:ey2 + pad_t, ex1 + pad_l:ex2 + pad_l] = 0

    # Initialize output buffers
    reproj_y.fill(0)
    reproj_u.fill(128)
    reproj_v.fill(128)
    reproj_weights_y.fill(0)

    # Reproject into shared buffers (kernels restricted to the valid bbox)
    yr0, yr1, yc0, yc1, ur0, ur1, uc0, uc1 = _mapping_bboxes(mapping, dh, dw)
    reproject_y(py.ravel(), dw, dh, py.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_y.ravel(), fisheye_mask[0].ravel() if fisheye_mask is not None else None, None, yr0, yr1, yc0, yc1)
    reproject_uv(pu.ravel(), pv.ravel(), dw, dh, map_uv_idx.ravel(), reproj_u.ravel(), reproj_v.ravel(), fisheye_mask[1].ravel() if fisheye_mask is not None else None, None, ur0, ur1, uc0, uc1)
    reproject_float(blend_weights_y.ravel(), dw, dh, blend_weights_y.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_weights_y.ravel(), fisheye_mask[0].ravel() if fisheye_mask is not None else None, None, yr0, yr1, yc0, yc1)


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
    reproject_y(p_y, dw, dh, sw_src, map_y_idx, c01, c23, out_y, fisheye_mask=None, crop_h=None)
    reproject_uv(p_uv, p_uv, dw, dh, map_uv_idx, out_u, out_v, fisheye_mask=None, crop_h=None)
    reproject_float(p_float, dw, dh, sw_src, map_y_idx, c01, c23, out_float, fisheye_mask=None, crop_h=None)
    _yuv420_to_rgb_kernel(p_y, p_uv, p_uv,
        np.zeros(dw*dh,dtype=np.uint8), np.zeros(dw*dh,dtype=np.uint8), np.zeros(dw*dh,dtype=np.uint8),
        dh, dw, dw//2)
    _rgb_to_yuv420_kernel(p_y, p_y, p_y,
        np.zeros(dw*dh,dtype=np.uint8), np.zeros((dh//2)*(dw//2),dtype=np.uint8), np.zeros((dh//2)*(dw//2),dtype=np.uint8),
        dh, dw)

    _print("Pre-compilation complete.")

def reproject_images(pto_file, input_files, output_file, pad, num_cores, padsides, enhance, force_video_dims: bool = False, fisheye_mask: bool = False, crop_to_content: bool = True, saturation: float = 1.0, devignette=None, input_datetime: str = None):
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

        map_y_idx, c01, c23, map_uv_idx, sw_pto, sh_pto = mapping
        py, pu, pv, sw_orig, sh_orig = load_image_to_yuv(input_path, pad, padsides, target_w=sw_pto, target_h=sh_pto)

        if devignette is not None:
            gain = _build_vignette_gain(sw_orig, sh_orig, devignette)
            pad_t_ = pad if 'top' in padsides else 0
            pad_l_ = pad if 'left' in padsides else 0
            _apply_vignette_y(py[pad_t_:pad_t_ + sh_orig, pad_l_:pad_l_ + sw_orig], gain)

        y_final = np.zeros((dh, dw), dtype=np.uint8)
        u_final = np.full((dh // 2, dw // 2), 128, dtype=np.uint8)
        v_final = np.full((dh // 2, dw // 2), 128, dtype=np.uint8)

        # Precompute fisheye mask for single-image path
        if fisheye_mask:
            cx, cy = dw // 2, dh // 2
            r = min(cx, cy)
            ys, xs = np.ogrid[:dh, :dw]
            outside_y = (xs - cx) ** 2 + (ys - cy) ** 2 > r * r
            h_uv, w_uv = dh // 2, dw // 2
            cx_uv, cy_uv = w_uv // 2, h_uv // 2
            r_uv = min(cx_uv, cy_uv)
            ys_uv, xs_uv = np.ogrid[:h_uv, :w_uv]
            outside_uv = (xs_uv - cx_uv) ** 2 + (ys_uv - cy_uv) ** 2 > r_uv * r_uv
        else:
            outside_y = outside_uv = None

        yr0, yr1, yc0, yc1, ur0, ur1, uc0, uc1 = _mapping_bboxes(mapping, dh, dw)
        reproject_y(py.ravel(), dw, dh, py.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), y_final.ravel(), outside_y.ravel() if outside_y is not None else None, None, yr0, yr1, yc0, yc1)
        reproject_uv(pu.ravel(), pv.ravel(), dw, dh, map_uv_idx.ravel(), u_final.ravel(), v_final.ravel(), outside_uv.ravel() if outside_uv is not None else None, None, ur0, ur1, uc0, uc1)

        if enhance:
            _print("Applying enhancement filter...")
            seed_y = int.from_bytes(os.urandom(4), 'little')
            y_final = _enhance_filter()(y_final, t=12, log2sizex=5, log2sizey=5, dither=6, seed=seed_y)
            u_final = _enhance_filter()(u_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
            v_final = _enhance_filter()(v_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)

        if input_datetime is not None:
            _ts = datetime.datetime.strptime(input_datetime, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
            _draw_timestamp_yuv(y_final, u_final, v_final, _ts)

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
            _print(f"  Non-square pixels: '{os.path.basename(input_path)}' "
                   f"is {actual_w}x{actual_h}, will be resampled to {expected_w}x{expected_h}")

    # Eagerly compile Numba functions before entering the thread pool
    _precompile_numba_functions()

    # Precompute fisheye mask for multi-image path
    if fisheye_mask:
        cx, cy = final_w // 2, final_h // 2
        r = min(cx, cy)
        ys, xs = np.ogrid[:final_h, :final_w]
        outside_y = (xs - cx) ** 2 + (ys - cy) ** 2 > r * r
        h_uv, w_uv = final_h // 2, final_w // 2
        cx_uv, cy_uv = w_uv // 2, h_uv // 2
        r_uv = min(cx_uv, cy_uv)
        ys_uv, xs_uv = np.ogrid[:h_uv, :w_uv]
        outside_uv = (xs_uv - cx_uv) ** 2 + (ys_uv - cy_uv) ** 2 > r_uv * r_uv
    else:
        outside_y = outside_uv = None

    _print("Reprojecting source images...")
    _t_phase = time.perf_counter()

    # Shared single-camera canvas buffers — reused for each camera in turn.
    # Peak memory = 1× canvas (Y+U/2+V/2 uint8 + weight float32) instead of N×.
    _buf_y  = np.empty((final_h, final_w), dtype=np.uint8)
    _buf_u  = np.empty((final_h // 2, final_w // 2), dtype=np.uint8)
    _buf_v  = np.empty((final_h // 2, final_w // 2), dtype=np.uint8)
    _buf_w  = np.empty((final_h, final_w), dtype=np.float32)

    from scipy.ndimage import distance_transform_edt as _edt, binary_erosion as _binary_erosion

    def _build_image_info(y_snap, u_snap, v_snap, w_snap, cam_idx):
        """Build ImageInfo from snapshotted per-camera arrays (runs in a thread,
        overlapping with the next camera's Numba reprojection)."""
        mask = w_snap > 1e-9
        mask = _binary_erosion(mask, iterations=2)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any():
            return None
        r0 = int(np.argmax(rows))
        r1 = int(len(rows) - 1 - np.argmax(rows[::-1])) + 1
        c0 = int(np.argmax(cols))
        c1 = int(len(cols) - 1 - np.argmax(cols[::-1])) + 1

        # Crop YUV to bbox first, then convert only the small region to RGB —
        # avoids allocating three full-canvas uint8 arrays before discarding them.
        u2 = u_snap[r0//2:(r1+1)//2, c0//2:(c1+1)//2]
        v2 = v_snap[r0//2:(r1+1)//2, c0//2:(c1+1)//2]
        r_crop, g_crop, b_crop = yuv_to_rgb(y_snap[r0:r1, c0:c1], u2, v2)
        del u2, v2
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
                ((input_files[i], final_w, final_h, mapping, pad, padsides, devignette, (outside_y, outside_uv) if outside_y is not None else None),
                 (_buf_y, _buf_u, _buf_v, _buf_w))
            )

            # Snapshot the shared buffers — these copies are consumed by the
            # background thread while the next reprojection runs on the originals.
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
                _build_image_info, y_snap, u_snap, v_snap, w_snap, i)
            pending_idx = i

        if pending_future is not None:
            result = pending_future.result()
            if result is not None:
                images[pending_idx] = result

    images = [img for img in images if img is not None]

    del _buf_y, _buf_u, _buf_v, _buf_w
    _print(f"Reprojection complete. ({time.perf_counter() - _t_phase:.2f}s)")

    _print("Blending with multiblend (graph-cut seams + exposure correction)...")
    _t_phase = time.perf_counter()

    min_left, min_top, workwidth, workheight = multiblend.tighten(images)
    levels = multiblend.compute_levels(images, workwidth, workheight, False, 1_000_000, 0)
    _print(f"  {workwidth}x{workheight}, {levels} levels (tightened from {final_w}x{final_h})")

    assignment, _, seam_mask_cache = compute_or_load_seams(
        images=images,
        workwidth=workwidth,
        workheight=workheight,
        pto_file=pto_file,
        pad=pad,
        padsides=padsides,
        levels=levels,
        is_video_output=force_video_dims,
        simple_seam=False,
        content_seam=False,
        verbosity=0 if _quiet else 1,
        print_func=_print,
    )

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
        seam_mask_cache=seam_mask_cache,
    )
    _print(f"  blend done ({time.perf_counter() - _t_phase:.2f}s)")
    _t_phase = time.perf_counter()

    # Compute coverage in the tightened workspace (image xpos/ypos are relative
    # to the tightened bbox after tighten() shifted them).
    covered_tight = multiblend._coverage_mask(images, workwidth, workheight)
    del images, assignment

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
    if not fisheye_mask and crop_to_content:
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

    # For fisheye output the corners outside the inscribed circle are masked
    # to black after fill. We can skip filling them, but the global Gaussian
    # smooth needs valid content in a ring around the circle, so we keep a
    # margin equal to ~3 sigma of the full-resolution blur.
    S_geo = 8
    sigma_s = 4
    blur_margin = max(1, round(3 * sigma_s * S_geo))
    if fisheye_mask:
        H_gap, W_gap = gap.shape
        cx, cy = W_gap // 2, H_gap // 2
        r = min(cx, cy)
        ys, xs = np.ogrid[:H_gap, :W_gap]
        inside_fill_circle = (xs - cx) ** 2 + (ys - cy) ** 2 <= (r + blur_margin) ** 2
        gap_fill = gap & inside_fill_circle
        del inside_fill_circle
    else:
        gap_fill = gap

    n_gap = int(gap_fill.sum())

    if n_gap > 0:
        _print(f"  gap pixels: {n_gap} (filling)")
        _t_geo = time.perf_counter()
        # Gap geometry (EDT index maps + feather weights) depends only on the
        # boolean gap mask and is cached across runs. Use the full gap mask so
        # feather weights inside the circle remain accurate.
        ri, ci, blend_w, sw, sh, feather_radius = compute_or_load_gap_geometry(
            gap, pto_file, pad, padsides,
            final_w, final_h, workwidth, workheight,
            min_left, min_top, fisheye_mask, crop_to_content,
            verbosity=0 if _quiet else 1, print_func=_print)
        H, W = gap.shape
        _print(f"  gap geometry ready ({time.perf_counter() - _t_geo:.2f}s)")
        _t_gap = time.perf_counter()

        # Convert index maps to int32 once; int16 is fine for storage but
        # NumPy advanced indexing is faster with int32 coordinates.
        ri_gap = ri[gap_fill].astype(np.int32, copy=False)
        ci_gap = ci[gap_fill].astype(np.int32, copy=False)
        del ri, ci

        # Process each channel independently: nearest-pixel fill + Gaussian
        # smooth at downscale + feathered blend.
        from scipy.ndimage import gaussian_filter
        out_channels = []
        for ch in rgb_channels:
            ch_f = ch.astype(np.float32)
            # EDT fill in-place: gap pixels get nearest valid pixel colour.
            # No separate ch_fill copy needed — gap and non-gap pixels are disjoint.
            ch_f[gap_fill] = ch_f[ri_gap, ci_gap]
            # Gaussian smooth at downscale.
            src_u8 = ch_f.clip(0, 255).astype(np.uint8)
            if _cv2() is not None:
                small = _cv2().resize(src_u8, (sw, sh), interpolation=_cv2().INTER_AREA).astype(np.float32)
            else:
                from PIL import Image as _PIL2
                small = np.array(_PIL2.fromarray(src_u8).resize((sw, sh), _PIL2.BOX)).astype(np.float32)
            blurred = gaussian_filter(small, sigma=sigma_s);  del small
            blurred_u8 = blurred.clip(0, 255).astype(np.uint8)
            if _cv2() is not None:
                full = _cv2().resize(blurred_u8, (W, H), interpolation=_cv2().INTER_LINEAR).astype(np.float32)
            else:
                from PIL import Image as _PIL2
                full = np.array(_PIL2.fromarray(blurred_u8).resize((W, H), _PIL2.BILINEAR)).astype(np.float32)
            del blurred, blurred_u8
            # Feathered blend. blend_w is uint16 (0..65535) to save memory;
            # scale to [0,1] inline without a full float32 copy.
            result = ch_f - full
            result *= blend_w
            result /= 65535.0
            result += full
            del ch_f, full
            np.clip(result, 0, 255, out=result)
            out_channels.append(result.astype(np.uint8));  del result

        del blend_w
        rgb_channels = out_channels
        _print(f"  gap fill done ({time.perf_counter() - _t_gap:.2f}s)")

    del gap, gap_fill

    _t_phase = time.perf_counter()
    y_final, u_final, v_final = rgb_to_yuv(rgb_channels)

    if abs(saturation - 1.0) > 0.001:
        u_final = np.clip((u_final.astype(np.float32) - 128.0) * saturation + 128.0, 0, 255).astype(np.uint8)
        v_final = np.clip((v_final.astype(np.float32) - 128.0) * saturation + 128.0, 0, 255).astype(np.uint8)

    if enhance:
        _print("Applying enhancement filter...")
        seed_y = int.from_bytes(os.urandom(4), 'little')
        y_final = _enhance_filter()(y_final, t=8, log2sizex=5, log2sizey=5, dither=6, seed=seed_y)
        u_final = _enhance_filter()(u_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
        v_final = _enhance_filter()(v_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)

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

    if input_datetime is not None:
        _ts = datetime.datetime.strptime(input_datetime, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
        _draw_timestamp_yuv(y_final, u_final, v_final, _ts)

    _print(f"YUV conversion done ({time.perf_counter() - _t_phase:.2f}s)")

    _print("Saving final image...")
    _t_phase = time.perf_counter()
    save_image_yuv420(y_final, u_final, v_final, output_file)
    _print(f"✅ Success! Panoramic image saved to {output_file} ({time.perf_counter() - _t_phase:.2f}s)")

def _mapping_bboxes(mapping, dh, dw):
    """Bounding boxes of valid pixels in a projection mapping.
    Returns (yr0, yr1, yc0, yc1, ur0, ur1, uc0, uc1); UV coords are on the
    half-resolution grid. Lets the reproject kernels skip dead canvas regions."""
    def _bb(valid):
        rows = np.any(valid, axis=1); cols = np.any(valid, axis=0)
        if not rows.any():
            return 0, 0, 0, 0
        return (int(rows.argmax()), int(len(rows) - rows[::-1].argmax()),
                int(cols.argmax()), int(len(cols) - cols[::-1].argmax()))
    yb = _bb(mapping[0].reshape(dh, dw) >= 0)
    ub = _bb(mapping[3] >= 0)
    return yb + ub


def worker_for_video_frame(args):
    """Worker function for video frames, writing to pre-allocated buffers."""
    (idx, frame, mapping, dw, dh, blend_weights_y_src, blend_weights_uv_src, pad, padsides, devignette_gain, fisheye_mask, crop_h, map_bbox), out_buffers = args
    reproj_y, reproj_u, reproj_v, reproj_weights_y, reproj_weights_uv = out_buffers

    if frame is None: return None
    if frame.format.name != "yuv420p": frame = frame.reformat(format="yuv420p")

    sw_orig, sh_orig = frame.width, frame.height
    
    # Handle Y-plane stride (copy to make writable)
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

    if devignette_gain is not None:
        _apply_vignette_y(py_src_orig, devignette_gain)

    # No pixel erasure for timestamp — rely on zeroed blend weights to exclude
    # the timestamp region. This avoids black-pixel bleed from bilinear interpolation.

    pad_t = pad if 'top' in padsides else 0
    pad_b = pad if 'bottom' in padsides else 0
    pad_l = pad if 'left' in padsides else 0
    pad_r = pad if 'right' in padsides else 0

    if pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0:
        noise_level = estimate_noise(py_src_orig) / 2.0
        pad_uv_t, pad_uv_b, pad_uv_l, pad_uv_r = pad_t//2, pad_b//2, pad_l//2, pad_r//2
        pad_y_width = ((pad_t, pad_b), (pad_l, pad_r))
        pad_uv_width = ((pad_uv_t, pad_uv_b), (pad_uv_l, pad_uv_r))

        py_src_all = np.pad(py_src_orig, pad_y_width, mode='edge')
        pu_src_all = np.pad(pu_src_orig, pad_uv_width, mode='edge')
        pv_src_all = np.pad(pv_src_orig, pad_uv_width, mode='edge')

        blur_size = 96
        py_src_all = _blur_padded_area_numba(py_src_all.astype(np.float32), pad_t, pad_b, pad_l, pad_r, blur_size, noise_level)

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
    if reproj_weights_y is not None:
        reproj_weights_y.fill(0)
        reproj_weights_uv.fill(0)

    yr0, yr1, yc0, yc1, ur0, ur1, uc0, uc1 = map_bbox if map_bbox is not None else (0, -1, 0, -1, 0, -1, 0, -1)
    reproject_y(py_src.ravel(), dw, dh, padded_sw_y, map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_y.ravel(), fisheye_mask[0].ravel() if fisheye_mask is not None else None, crop_h, yr0, yr1, yc0, yc1)
    reproject_uv(pu_src.ravel(), pv_src.ravel(), dw, dh, map_uv_idx.ravel(), reproj_u.ravel(), reproj_v.ravel(), fisheye_mask[1].ravel() if fisheye_mask is not None else None, crop_h, ur0, ur1, uc0, uc1)
    if reproj_weights_y is not None:
        reproject_float(blend_weights_y_src.ravel(), dw, dh, blend_weights_y_src.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_weights_y.ravel(), fisheye_mask[0].ravel() if fisheye_mask is not None else None, crop_h, yr0, yr1, yc0, yc1)
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
        with _av().open(video_file) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base

            # Check for a non-zero container start time (Unix timestamp in seconds).
            container_start = container.start_time
            start_time_sec = container_start / _av().time_base if container_start is not None else 0.0

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

    except (_av().Error, IndexError) as e:
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


# ---------------------------------------------------------------------------
# Timelapse helpers
# ---------------------------------------------------------------------------

def _parse_timelapse_datetime(s):
    """Parse a timelapse datetime string, returning a timezone-aware UTC datetime."""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return dt.replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.datetime.fromisoformat(s.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    except ValueError:
        pass
    raise ValueError(f"Could not parse timelapse datetime: '{s}'")


def _parse_timelapse_duration(s):
    """Parse a duration string like '6 hours 3 minutes 10 seconds' into seconds."""
    if not s:
        return None
    s = s.strip().lower()
    total_seconds = 0
    patterns = [
        (r'(\d+)\s*days?', 86400),
        (r'(\d+)\s*hours?', 3600),
        (r'(\d+)\s*minutes?', 60),
        (r'(\d+)\s*seconds?', 1),
    ]
    found = False
    for pattern, multiplier in patterns:
        for match in re.finditer(pattern, s):
            total_seconds += int(match.group(1)) * multiplier
            found = True
    if not found:
        raise ValueError(f"Could not parse timelapse duration: '{s}'")
    return total_seconds


def _discover_timelapse_files(base_pattern, start_time, end_time, quality, station=None):
    """Discover video files for each camera in the timelapse range.

    Returns a list of lists: camera_files[cam_idx] = sorted list of
    (file_path, file_start_time, file_end_time).
    """
    quality = str(quality).lower()
    filename = "full_*.mp4" if quality == "hd" else "mini_*.mp4"

    if station:
        # Single SSH session: list specific hour directories in the time range.
        hour_patterns = []
        t = start_time.replace(minute=0, second=0, microsecond=0)
        while t <= end_time:
            ymd = t.strftime('%Y%m%d')
            hh = t.strftime('%H')
            hour_patterns.append(f"{base_pattern}/{ymd}/{hh}/{filename}")
            t += datetime.timedelta(hours=1)
        # Use compgen -G so the patterns are expanded on the remote host.
        script = 'while IFS= read -r pat; do compgen -G "$pat" 2>/dev/null || true; done'
        pattern_input = '\n'.join(hour_patterns) + '\n'
        result = subprocess.run(
            ['ssh', '-o', 'BatchMode=yes', station, 'bash', '-c', shlex.quote(script)],
            input=pattern_input, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise IOError(f"Failed to discover timelapse files on {station}: {result.stderr.strip()}")

        camera_files_by_cam = {}
        for line in result.stdout.splitlines():
            file_path = line.strip()
            if not file_path:
                continue
            cam_match = re.search(r'cam(\d+)', file_path)
            if not cam_match:
                continue
            cam_num = int(cam_match.group(1))
            parts = file_path.split('/')
            yyyymmdd = hh = mm = None
            for p in parts:
                if yyyymmdd is None and re.match(r'^\d{8}$', p):
                    yyyymmdd = p
                elif yyyymmdd is not None and hh is None and re.match(r'^\d{2}$', p):
                    hh = p
            fname = os.path.basename(file_path)
            m = re.match(r'(?:mini|full)_(\d{2})\.mp4$', fname)
            if m:
                mm = m.group(1)
            if not (yyyymmdd and hh and mm):
                continue
            try:
                file_start = datetime.datetime.strptime(f"{yyyymmdd}{hh}{mm}", "%Y%m%d%H%M").replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                continue
            file_end = file_start + datetime.timedelta(minutes=1)
            if file_end <= start_time or file_start >= end_time:
                continue
            camera_files_by_cam.setdefault(cam_num, []).append((file_path, file_start, file_end))

        if not camera_files_by_cam:
            raise ValueError(f"No remote video files found for pattern: {base_pattern}")
        sorted_camera_files = []
        for cam_num in sorted(camera_files_by_cam.keys()):
            files = camera_files_by_cam[cam_num]
            files.sort(key=lambda x: x[1])
            sorted_camera_files.append(files)
        return sorted_camera_files

    camera_dirs = sorted(glob.glob(base_pattern))
    if not camera_dirs:
        raise ValueError(f"No camera directories found for pattern: {base_pattern}")

    camera_dirs = [d for d in camera_dirs if re.search(r'cam\d+$', os.path.basename(d))]
    if not camera_dirs:
        raise ValueError(f"No camN directories found for pattern: {base_pattern}")

    camera_files = []
    for cam_dir in camera_dirs:
        files = []
        for file_path in glob.glob(os.path.join(cam_dir, "*", "*", filename)):
            parts = file_path.split(os.sep)
            yyyymmdd = hh = mm = None
            for p in parts:
                if yyyymmdd is None and re.match(r'^\d{8}$', p):
                    yyyymmdd = p
                elif yyyymmdd is not None and hh is None and re.match(r'^\d{2}$', p):
                    hh = p
            fname = os.path.basename(file_path)
            m = re.match(r'(?:mini|full)_(\d{2})\.mp4$', fname)
            if m:
                mm = m.group(1)

            if not (yyyymmdd and hh and mm):
                continue
            try:
                file_start = datetime.datetime.strptime(f"{yyyymmdd}{hh}{mm}", "%Y%m%d%H%M").replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                continue
            file_end = file_start + datetime.timedelta(minutes=1)

            if file_end <= start_time or file_start >= end_time:
                continue

            files.append((file_path, file_start, file_end))

        files.sort(key=lambda x: x[1])
        camera_files.append(files)

    return camera_files


def _build_timelapse_timeline(files, model=None):
    """Build a timeline of frame timestamps for a single camera.

    Returns a tuple (ts_arr, file_arr, frame_arr) of numpy arrays sorted by
    timestamp. Using numpy keeps memory compact (~24 bytes/frame vs ~150 bytes
    for Python tuples) and lets the OS reclaim memory immediately after deletion.
    """
    per_file_ts = []
    per_file_file = []
    per_file_frame = []

    for file_idx, (file_path, _, _) in enumerate(files):
        try:
            with _av().open(file_path) as container:
                stream = container.streams.video[0]
                time_base = stream.time_base

                container_start = container.start_time
                start_time_sec = container_start / _av().time_base if container_start is not None else 0.0

                if start_time_sec > 0:
                    # Fast path: derive exact frame timestamps from packet PTS.
                    _print(f"  Using container metadata for {file_path}          ", end='\r', flush=True)
                    raw_pts = []
                    for packet in container.demux(stream):
                        if packet.pts is None:
                            continue
                        raw_pts.append(packet.pts)
                    if not raw_pts:
                        continue
                    pts_arr = np.array(raw_pts, dtype=np.int64); del raw_pts
                    order = np.argsort(pts_arr, kind='stable')
                    n = len(pts_arr)
                    per_file_ts.append(pts_arr[order] * float(time_base))
                    per_file_file.append(np.full(n, file_idx, dtype=np.int32))
                    per_file_frame.append(np.arange(n, dtype=np.int32))
                    del pts_arr, order
                else:
                    # Fallback: OCR the first frame and assume constant frame rate.
                    _print(f"  No container metadata for {file_path}, falling back to OCR on first frame.          ", end='\r', flush=True)
                    frame_rate = stream.average_rate
                    if frame_rate is None or float(frame_rate) == 0:
                        frame_rate = 25.0
                    else:
                        frame_rate = float(frame_rate)
                    try:
                        from timestamp import get_timestamp
                        frame = next(container.decode(stream))
                        ts = get_timestamp(frame.to_image(), robust=False, model=model)
                        if ts is None:
                            ts = get_timestamp(frame.to_image(), robust=True, model=model)
                        if ts is None:
                            _print(f"  Warning: Could not extract timestamp from {file_path}. Skipping file.", file=sys.stderr)
                            continue
                        start_ts = ts.astimezone(datetime.timezone.utc).timestamp()
                        frame_count = stream.frames if stream.frames > 0 else int(frame_rate * 60)
                        per_file_ts.append(start_ts + np.arange(frame_count, dtype=np.float64) / frame_rate)
                        per_file_file.append(np.full(frame_count, file_idx, dtype=np.int32))
                        per_file_frame.append(np.arange(frame_count, dtype=np.int32))
                    except Exception as e:
                        _print(f"  Warning: OCR failed for {file_path}: {e}. Skipping file.", file=sys.stderr)
                        continue
        except Exception as e:
            _print(f"Warning: Could not process {file_path}: {e}", file=sys.stderr)

    if not per_file_ts:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    ts_arr = np.concatenate(per_file_ts)
    file_arr = np.concatenate(per_file_file)
    frame_arr = np.concatenate(per_file_frame)

    order = np.argsort(ts_arr, kind='stable')
    return ts_arr[order], file_arr[order], frame_arr[order]


def _collect_remote_lens_pto_paths(paths):
    """Collect lens.pto paths from camera directories in the given remote paths."""
    lens_paths = set()
    for path in paths:
        parts = path.split('/')
        for i, p in enumerate(parts):
            if re.match(r'cam\d+$', p):
                lens_path = '/'.join(parts[:i+1]) + '/lens.pto'
                lens_paths.add(lens_path)
                break
    return sorted(lens_paths)


def _fetch_remote_files_over_ssh(station, remote_paths, local_dir, progress_prefix="Fetching"):
    """Fetch remote files via a single tar-over-ssh session.

    remote_paths must be absolute paths on the remote host. The remote
    directory structure is preserved under local_dir. A progress bar
    showing the number of transferred files is printed.
    """
    if not remote_paths:
        return []
    os.makedirs(local_dir, exist_ok=True)
    total_files = len(remote_paths)
    file_list = '\n'.join(remote_paths) + '\n'
    ssh_cmd = ['ssh', '-o', 'BatchMode=yes', station, 'tar', '-cvhf', '-', '-T', '/dev/stdin']
    tar_cmd = ['tar', '-xf', '-', '-C', local_dir]
    ssh_proc = subprocess.Popen(ssh_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ssh_proc.stdin.write(file_list.encode())
    ssh_proc.stdin.close()
    tar_proc = subprocess.Popen(tar_cmd, stdin=ssh_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ssh_proc.stdout.close()

    transferred = [0]
    lock = threading.Lock()
    stderr_lines = []
    expected_basenames = set(os.path.basename(p) for p in remote_paths)
    def _read_stderr():
        try:
            for raw_line in ssh_proc.stderr:
                try:
                    line = raw_line.decode().strip()
                except UnicodeDecodeError:
                    continue
                stderr_lines.append(line)
                if line and not line.startswith('tar:') and os.path.basename(line) in expected_basenames:
                    with lock:
                        transferred[0] += 1
                        count = transferred[0]
                    pct = min(count / total_files, 1.0)
                    bar_len = 40
                    filled = min(int(bar_len * pct), bar_len)
                    bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
                    _print(f"\r{progress_prefix}: {bar} {count}/{total_files}", end='', flush=True)
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_read_stderr)
    stderr_thread.start()
    tar_err = tar_proc.communicate()[1]
    ssh_proc.wait()
    stderr_thread.join()

    if ssh_proc.returncode != 0:
        error_tail = ' | '.join(stderr_lines[-10:]) if stderr_lines else '(no stderr)'
        raise IOError(f"SSH fetch failed from {station}: {error_tail}")
    if tar_proc.returncode != 0:
        raise IOError(f"Tar extraction failed: {tar_err.decode().strip()}")
    # Ensure the final bar shows at least the total, but never exceeds the capped bar.
    bar_len = 40
    with lock:
        final_count = min(transferred[0], total_files)
    filled = bar_len
    bar = '[' + '#' * filled + '-' * (bar_len - filled) + ']'
    _print(f"\r{progress_prefix}: {bar} {final_count}/{total_files}", end='', flush=True)
    _print()
    return [os.path.join(local_dir, p.lstrip('/')) for p in remote_paths]


def _expand_remote_input_patterns(station, patterns):
    """Expand shell glob patterns on a remote host via a single ssh session.

    Uses bash compgen -G so wildcards are expanded on the remote host, not locally.
    """
    if not patterns:
        return []
    # compgen -G expands a glob pattern safely and returns one match per line.
    script = 'while IFS= read -r pat; do compgen -G "$pat" 2>/dev/null || true; done'
    pattern_input = '\n'.join(patterns) + '\n'
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', station, 'bash', '-c', shlex.quote(script)],
        input=pattern_input, capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise IOError(f"Failed to expand remote patterns on {station}: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _find_best_timelapse_frame(timeline, target_ts, tolerance=1.0):
    """Find the frame in timeline closest to target_ts within tolerance seconds.

    timeline is a tuple (ts_arr, file_arr, frame_arr) of sorted numpy arrays.
    target_ts is a Unix float timestamp.
    Returns (file_index, frame_index) or None.
    """
    ts_arr, file_arr, frame_arr = timeline
    if len(ts_arr) == 0:
        return None

    lo = int(np.searchsorted(ts_arr, target_ts))

    best = None
    best_diff = float('inf')
    for idx in (lo - 1, lo, lo + 1):
        if 0 <= idx < len(ts_arr):
            diff = abs(ts_arr[idx] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best = (int(file_arr[idx]), int(frame_arr[idx]))

    if best is not None and best_diff <= tolerance:
        return best
    return None


def _draw_timestamp_yuv(y_plane, u_plane, v_plane, unix_ts):
    """Overlay an ISO-format timestamp in the lower-left corner of a YUV420 frame.

    Renders white text on a dark background. Modifies the planes in-place.
    Format: YYYY-MM-DD hh:mm:ss.ff  (UTC, ff = fractional seconds, 2 digits)
    """
    from PIL import Image as _PilImg, ImageDraw as _PilDraw, ImageFont as _PilFont

    H, W = y_plane.shape
    dt = datetime.datetime.fromtimestamp(unix_ts, tz=datetime.timezone.utc)
    ff = int((unix_ts % 1) * 100)
    text = dt.strftime(f"%Y-%m-%d %H:%M:%S.{ff:02d} UTC")

    font_size = max(14, W // 128)
    font = None
    for _path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ):
        try:
            font = _PilFont.truetype(_path, font_size)
            break
        except Exception:
            pass
    if font is None:
        try:
            font = _PilFont.load_default(size=font_size)
        except TypeError:
            font = _PilFont.load_default()

    tmp = _PilImg.new('L', (W, H), 0)
    draw = _PilDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    text_bottom = bbox[3]

    margin = max(8, H // 108)
    pad_box = max(4, font_size // 8)
    pad_bottom = pad_box + max(2, font_size // 6)
    tx, ty = margin, H - text_bottom - margin - pad_bottom
    bx1, by1 = tx - pad_box + bbox[0], ty - pad_box + bbox[1]
    bx2, by2 = tx + tw + pad_box, ty + text_bottom + pad_bottom

    ry1, ry2 = max(0, by1), min(H, by2)
    rx1, rx2 = max(0, bx1), min(W, bx2)
    if ry1 >= ry2 or rx1 >= rx2:
        return

    # Semitransparent dark background (blend existing luma toward black)
    bg_alpha = 0.55
    y_plane[ry1:ry2, rx1:rx2] = (y_plane[ry1:ry2, rx1:rx2].astype(np.float32) * (1.0 - bg_alpha)).astype(np.uint8)

    draw.text((tx, ty), text, fill=235, font=font)
    tmp_region = np.array(tmp)[ry1:ry2, rx1:rx2]
    drawn = tmp_region > 0
    y_plane[ry1:ry2, rx1:rx2][drawn] = tmp_region[drawn]

    uy1, ux1 = ry1 // 2, rx1 // 2
    uy2, ux2 = (ry2 + 1) // 2, (rx2 + 1) // 2
    u_bg = u_plane[uy1:uy2, ux1:ux2].astype(np.float32)
    v_bg = v_plane[uy1:uy2, ux1:ux2].astype(np.float32)
    u_plane[uy1:uy2, ux1:ux2] = (u_bg * (1.0 - bg_alpha) + 128.0 * bg_alpha).astype(np.uint8)
    v_plane[uy1:uy2, ux1:ux2] = (v_bg * (1.0 - bg_alpha) + 128.0 * bg_alpha).astype(np.uint8)


def reproject_timelapse(pto_file, camera_files, output_file, start_time, end_time, speed_factor, output_fps, pad, num_cores, padsides, model=None, enhance=False, fisheye_mask=False, max_frames=0, level_subsample=1, crf="28", preset="ultrafast", timestamp=False, saturation=1.0, devignette=None):
    if not _av(): raise ImportError("PyAV is not installed, but video processing was requested.")

    num_images = len(camera_files)
    if num_images == 0:
        raise ValueError("No camera files provided for timelapse.")

    _print("Building timelapse frame timelines...")
    camera_timelines = []
    for cam_idx, files in enumerate(camera_files):
        timeline = _build_timelapse_timeline(files, model=model)
        _print(f"  Camera {cam_idx + 1}: {len(timeline[0])} frames from {len(files)} files.")
        camera_timelines.append(timeline)

    if speed_factor is None or speed_factor <= 0:
        raise ValueError("Timelapse speed factor must be positive.")
    if output_fps is None or output_fps <= 0:
        raise ValueError("Timelapse output framerate must be positive.")

    step_seconds = speed_factor / output_fps
    target_timestamps = []
    t = start_time
    while t <= end_time:
        target_timestamps.append(t)
        t += datetime.timedelta(seconds=step_seconds)

    _print("Selecting synchronized frames for timelapse...")
    selected_groups = []
    selected_timestamps = []
    skipped = 0
    tolerance = max(speed_factor / 25.0, 1.0)
    for target in target_timestamps:
        target_ts = target.timestamp()
        group = []
        missing = False
        for timeline in camera_timelines:
            best = _find_best_timelapse_frame(timeline, target_ts, tolerance=tolerance)
            if best is None:
                missing = True
                break
            group.append(best)
        if missing:
            skipped += 1
            continue
        selected_groups.append(group)
        selected_timestamps.append(target_ts)

    if skipped > 0:
        _print(f"Skipped {skipped} output frames due to missing synchronized frames (within {tolerance:.1f}s tolerance).")

    if not selected_groups:
        raise ValueError("No synchronized frames found in timelapse range.")

    if max_frames > 0 and len(selected_groups) > max_frames:
        selected_groups = selected_groups[:max_frames]
        selected_timestamps = selected_timestamps[:max_frames]

    _print(f"Selected {len(selected_groups)} output frames for timelapse.")

    # Free timeline data to save memory - no longer needed after frame selection
    del camera_timelines
    gc.collect()

    mappings, global_options = build_mappings(pto_file, pad, num_cores, padsides, is_video_output=True)
    final_w, final_h = global_options['final_w'], global_options['final_h']
    if final_w > 16384:
        raise ValueError(f"Output width {final_w} exceeds codec limits for H.264/libx264. PTO='{pto_file}'")
    if len(mappings) != num_images:
        raise ValueError(f"Number of cameras ({num_images}) does not match PTO ({len(mappings)}).")

    _precompile_numba_functions()

    # Build per-camera vignette gain LUTs (once — same resolution for all frames)
    _vignette_gains = [None] * num_images
    if devignette is not None:
        for i in range(num_images):
            sw_map, sh_map = mappings[i][4], mappings[i][5]
            _vignette_gains[i] = _build_vignette_gain(sw_map, sh_map, devignette)
        _print(f"  built {num_images} vignette gain LUTs")

    # Initialize fisheye mask geometry (needed for geometry precomputation)
    if fisheye_mask:
        _fy, _fx = final_h, final_w
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

    # -----------------------------------------------------------------------
    # Geometry precomputation (same as reproject_videos multi-video path)
    # -----------------------------------------------------------------------
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
        ts_x1, ts_y1, ts_x2, ts_y2 = _TIMESTAMP_BOX_HD if sh_map >= 900 else _TIMESTAMP_BOX_SD
        _ts_m = 3  # margin matching worker_for_video_frame expansion
        bw[max(0, ts_y1 - _ts_m) + _tmp_pad_t:min(sh_map, ts_y2 + _ts_m) + _tmp_pad_t,
           max(0, ts_x1 - _ts_m) + _tmp_pad_l:min(sw_map, ts_x2 + _ts_m) + _tmp_pad_l] = 0
        _tmp_blend_weights_y.append(bw)
        map_y_idx, c01, c23 = mappings[i][0], mappings[i][1], mappings[i][2]
        _tmp_reproj = np.zeros((final_h, final_w), dtype=np.float32)
        reproject_float(bw.ravel(), final_w, final_h, bw.shape[1],
                        map_y_idx.ravel(), c01.ravel(), c23.ravel(), _tmp_reproj.ravel(),
                        geo_outside_y.ravel() if geo_outside_y is not None else None)
        _tmp_weights += _tmp_reproj
    geo_gap = _tmp_weights < 1e-9
    del _tmp_weights, _tmp_reproj

    from scipy.ndimage import gaussian_filter, distance_transform_edt as _geo_edt
    H_geo, W_geo = geo_gap.shape
    S_geo = 8
    feather_radius = max(1, round(20 * W_geo / 4096))

    from scipy.ndimage import binary_erosion as _binary_erosion_cam
    cam_bboxes = []
    cam_masks = []
    cam_inpaint = []
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
        eroded_full_i = _binary_erosion_cam(mask_i, iterations=2)
        rows_e = np.any(eroded_full_i, axis=1); cols_e = np.any(eroded_full_i, axis=0)
        if not rows_e.any():
            eroded_i = np.zeros((r1_i - r0_i, c1_i - c0_i), dtype=bool)
            cam_masks.append(eroded_i)
            cam_inpaint.append(None)
            continue
        r0_i = int(np.argmax(rows_e))
        r1_i = int(len(rows_e) - np.argmax(rows_e[::-1]))
        c0_i = int(np.argmax(cols_e))
        c1_i = int(len(cols_e) - np.argmax(cols_e[::-1]))
        cam_bboxes[-1] = (r0_i, r1_i, c0_i, c1_i)
        mask_crop_i = eroded_full_i[r0_i:r1_i, c0_i:c1_i]
        cam_masks.append(mask_crop_i)
        if not mask_crop_i.all():
            H_i, W_i = mask_crop_i.shape; ds_i = 8
            ph_i = ((H_i + ds_i - 1) // ds_i) * ds_i; pw_i = ((W_i + ds_i - 1) // ds_i) * ds_i
            sp = np.zeros((ph_i, pw_i), dtype=bool)
            sp[:H_i, :W_i] = mask_crop_i
            sd = sp[::ds_i, ::ds_i]
            ri_ds_i, ci_ds_i = _geo_edt(~sd, return_distances=False, return_indices=True)
            ri_i = np.repeat(np.repeat(ri_ds_i * ds_i, ds_i, axis=0), ds_i, axis=1)[:H_i, :W_i]
            ci_i = np.repeat(np.repeat(ci_ds_i * ds_i, ds_i, axis=0), ds_i, axis=1)[:H_i, :W_i]
            # Precompute flat source/destination indices for fast per-frame inpaint
            inv_i = ~mask_crop_i
            dst_flat_i = np.flatnonzero(inv_i)
            src_flat_i = ri_i[inv_i].astype(np.intp) * W_i + ci_i[inv_i].astype(np.intp)
            cam_inpaint.append((src_flat_i, dst_flat_i))
        else:
            cam_inpaint.append(None)
    del _tmp_reproj_i

    valid_bboxes = [b for b in cam_bboxes if b is not None]
    geo_min_top = min(b[0] for b in valid_bboxes)
    geo_min_left = min(b[2] for b in valid_bboxes)
    geo_workwidth = max(b[3] - b[2] + b[2] - geo_min_left for b in valid_bboxes)
    geo_workheight = max(b[1] - b[0] + b[0] - geo_min_top for b in valid_bboxes)
    cam_tight_pos = []
    for bbox in cam_bboxes:
        if bbox is None:
            cam_tight_pos.append(None)
        else:
            r0_i, r1_i, c0_i, c1_i = bbox
            cam_tight_pos.append((r0_i - geo_min_top, c0_i - geo_min_left))

    # Mark pixels outside the multiblend work rectangle as gaps — the eroded
    # bounding boxes may exclude edge rows/columns that still have non-zero
    # weight (from uneroded maps), leaving them unfilled (dark) otherwise.
    _work_b = geo_min_top + geo_workheight
    _work_r = geo_min_left + geo_workwidth
    if geo_min_top > 0:
        geo_gap[:geo_min_top, :] = True
    if _work_b < final_h:
        geo_gap[_work_b:, :] = True
    if geo_min_left > 0:
        geo_gap[:, :geo_min_left] = True
    if _work_r < final_w:
        geo_gap[:, _work_r:] = True

    geo_crop_h = final_h
    row_has_content = np.any(~geo_gap, axis=1)
    if row_has_content.any():
        geo_crop_h = int(len(row_has_content) - np.argmax(row_has_content[::-1]))
        if geo_crop_h < final_h:
            _print(f"  will crop canvas: {final_h} -> {geo_crop_h} rows")

    out_h = _round_up_16(geo_crop_h)
    out_w = _round_up_16(final_w)
    if out_h != geo_crop_h:
        _print(f"  will round height: {geo_crop_h} -> {out_h} rows")
    if out_w != final_w:
        _print(f"  will round width: {final_w} -> {out_w} columns")

    if out_h > final_h or out_w > final_w:
        _gap_w = np.zeros((out_h, out_w), dtype=bool)
        _gap_w[:final_h, :final_w] = geo_gap
        _gap_w[final_h:, :] = True
        _gap_w[:, final_w:] = True
        geo_gap = _gap_w
    else:
        geo_gap = geo_gap[:out_h, :out_w]
    H_geo = out_h
    W_geo = out_w
    geo_sw = max(1, W_geo // S_geo)
    geo_sh = max(1, H_geo // S_geo)
    geo_sigma_s = 4
    ph_g = ((H_geo + S_geo - 1) // S_geo) * S_geo
    pw_g = ((W_geo + S_geo - 1) // S_geo) * S_geo
    gap_pad_g = np.zeros((ph_g, pw_g), dtype=bool)
    gap_pad_g[:H_geo, :W_geo] = geo_gap
    gap_ds_g = gap_pad_g[::S_geo, ::S_geo]; del gap_pad_g
    ri_ds_g, ci_ds_g = _geo_edt(gap_ds_g, return_distances=False, return_indices=True); del gap_ds_g
    geo_ri = np.repeat(np.repeat(ri_ds_g * S_geo, S_geo, axis=0), S_geo, axis=1)[:H_geo, :W_geo]; del ri_ds_g
    geo_ci = np.repeat(np.repeat(ci_ds_g * S_geo, S_geo, axis=0), S_geo, axis=1)[:H_geo, :W_geo]; del ci_ds_g
    _print(f"  computing full-res EDT on {W_geo}x{H_geo} canvas...")
    geo_dist = _geo_edt(~geo_gap)
    geo_blend_w = np.clip(geo_dist / feather_radius, 0.0, 1.0).astype(np.float32); del geo_dist
    geo_n_gap = int(geo_gap.sum())
    geo_gap_idx = np.where(geo_gap)
    geo_ri_gap = geo_ri[geo_gap_idx]
    geo_ci_gap = geo_ci[geo_gap_idx]
    _print(f"  gap pixels: {geo_n_gap}, feather: {feather_radius}px")
    _print("  geometry precomputation complete.")

    _geo_fill_u8 = [np.empty((H_geo, W_geo), dtype=np.uint8) for _ in range(3)] if geo_n_gap > 0 else None

    # -----------------------------------------------------------------------
    # Stitching pass with per-camera multi-file decoding
    # -----------------------------------------------------------------------
    _print("\nStarting timelapse stitching process...")

    out_container = _av().open(output_file, mode='w')
    out_stream = out_container.add_stream("libx264", rate=output_fps)
    out_stream.width, out_stream.height, out_stream.pix_fmt = out_w, out_h, 'yuv420p'
    out_stream.options = {"preset": preset, "crf": str(crf)}

    total_frames = len(selected_groups)

    frame_y_planes = np.empty((num_images, final_h, final_w), dtype=np.uint8)
    frame_u_planes = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.uint8)
    frame_v_planes = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.uint8)
    _canvas_r = np.empty((out_h, out_w), dtype=np.float32)
    _canvas_g = np.empty((out_h, out_w), dtype=np.float32)
    _canvas_b = np.empty((out_h, out_w), dtype=np.float32)

    blend_weights_y = _tmp_blend_weights_y
    pad_t = _tmp_pad_t; pad_b = _tmp_pad_b; pad_l = _tmp_pad_l; pad_r = _tmp_pad_r

    def _yuv_crop_inpaint(i):
        bbox = cam_bboxes[i]
        if bbox is None:
            return i, None
        r0, r1, c0, c1 = bbox
        ur0, ur1 = r0 // 2, (r1 + 1) // 2
        uc0, uc1 = c0 // 2, (c1 + 1) // 2
        y_crop_src = frame_y_planes[i][r0:r1, c0:c1]
        u_crop_src = frame_u_planes[i][ur0:ur1, uc0:uc1]
        v_crop_src = frame_v_planes[i][ur0:ur1, uc0:uc1]
        r, g, b = yuv_to_rgb(y_crop_src, u_crop_src, v_crop_src)
        h_bb, w_bb = r1 - r0, c1 - c0
        r_crop = r[:h_bb, :w_bb].copy()
        g_crop = g[:h_bb, :w_bb].copy()
        b_crop = b[:h_bb, :w_bb].copy()
        inpaint = cam_inpaint[i]
        if inpaint is not None:
            src_flat, dst_flat = inpaint
            r_crop.ravel()[dst_flat] = r_crop.ravel()[src_flat]
            g_crop.ravel()[dst_flat] = g_crop.ravel()[src_flat]
            b_crop.ravel()[dst_flat] = b_crop.ravel()[src_flat]
        return i, (r_crop, g_crop, b_crop)

    # Per-camera state for sequential file decoding
    class _CameraDecoder:
        __slots__ = ('files', 'container', 'stream', 'frame_iter', 'current_file_idx', 'current_frame_idx', 'last_frame', 'closed')
        def __init__(self, files):
            self.files = files
            self.container = None
            self.stream = None
            self.frame_iter = None
            self.current_file_idx = -1
            self.current_frame_idx = -1
            self.last_frame = None
            self.closed = False

        def _open_file(self, file_idx):
            if self.container is not None:
                try:
                    self.container.close()
                except Exception:
                    pass
            self.container = None
            self.stream = None
            self.frame_iter = None
            self.last_frame = None
            self.current_file_idx = -1
            self.current_frame_idx = -1
            if file_idx < 0 or file_idx >= len(self.files):
                return False
            file_path = self.files[file_idx][0]
            try:
                self.container = _av().open(file_path)
                self.stream = self.container.streams.video[0]
                self.stream.thread_type = 'AUTO'
                self.frame_iter = self.container.decode(self.stream)
                self.current_file_idx = file_idx
                self.current_frame_idx = -1
                return True
            except Exception as e:
                _print(f"Warning: Could not open {file_path}: {e}", file=sys.stderr)
                return False

        def get_frame(self, file_idx, frame_idx):
            if self.closed:
                return None
            if file_idx != self.current_file_idx:
                if not self._open_file(file_idx):
                    return None
            if self.current_frame_idx == frame_idx and self.last_frame is not None:
                return self.last_frame
            if self.current_frame_idx > frame_idx:
                # Cannot go backwards; should not happen with sorted targets
                return None
            try:
                while self.current_frame_idx < frame_idx:
                    self.last_frame = next(self.frame_iter)
                    self.current_frame_idx += 1
                # Now current_frame_idx == frame_idx, decode the next frame
                self.last_frame = next(self.frame_iter)
                self.current_frame_idx += 1
                return self.last_frame
            except StopIteration:
                return None

        def close(self):
            self.closed = True
            self.last_frame = None
            if self.container is not None:
                try:
                    self.container.close()
                except Exception:
                    pass

    camera_decoders = [_CameraDecoder(files) for files in camera_files]

    try:
        out_frame = _av().VideoFrame(width=out_w, height=out_h, format='yuv420p')
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
            if _cv2() is not None:
                small = _cv2().resize(buf_u8, (geo_sw, geo_sh), interpolation=_cv2().INTER_AREA).astype(np.float32)
            else:
                from PIL import Image as _PIL2
                small = np.array(_PIL2.fromarray(buf_u8).resize((geo_sw, geo_sh), _PIL2.BOX)).astype(np.float32)
            blurred_u8 = gaussian_filter(small, sigma=geo_sigma_s).clip(0, 255).astype(np.uint8)
            if _cv2() is not None:
                full = _cv2().resize(blurred_u8, (W_geo, H_geo), interpolation=_cv2().INTER_LINEAR).astype(np.float32)
            else:
                from PIL import Image as _PIL2
                full = np.array(_PIL2.fromarray(blurred_u8).resize((W_geo, H_geo), _PIL2.BILINEAR)).astype(np.float32)
            result = ch_f * geo_blend_w + full * (1.0 - geo_blend_w)
            np.clip(result, 0, 255, out=result)
            return result.astype(np.uint8)

    cached_exp_info = None
    # Valid-pixel bounding boxes per camera — reproject kernels skip dead regions.
    _map_bboxes_cams = [_mapping_bboxes(mappings[i], final_h, final_w) for i in range(num_images)]
    frame_count = 0

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        for loop_idx, (group, group_ts) in enumerate(zip(selected_groups, selected_timestamps), 1):
            final_group_frames = [None] * num_images
            for i, (file_idx, frame_idx) in enumerate(group):
                frame = camera_decoders[i].get_frame(file_idx, frame_idx)
                final_group_frames[i] = frame

            if any(f is None for f in final_group_frames):
                del final_group_frames
                gc.collect()
                continue

            frame_count += 1
            if max_frames > 0 and frame_count > max_frames:
                break

            if not _quiet and total_frames > 0 and (loop_idx % 5 == 0 or loop_idx == total_frames):
                percent_done = (loop_idx / total_frames) * 100
                if sys.stderr.isatty():
                    bar_length = 40; filled_len = int(round(bar_length * loop_idx / float(total_frames)))
                    bar = '█' * filled_len + '-' * (bar_length - filled_len)
                    sys.stderr.write(f'Stitching: [{bar}] {percent_done:.1f}% \r'); sys.stderr.flush()
                else:
                    _print(f"PROGRESS:{percent_done:.1f}", file=sys.stderr, flush=True)

            worker_args = [
                ((i, final_group_frames[i], mappings[i], final_w, final_h, blend_weights_y[i], None, pad, padsides, _vignette_gains[i], (geo_outside_y, geo_outside_uv) if geo_outside_y is not None else None, out_h, _map_bboxes_cams[i]),
                 (frame_y_planes[i], frame_u_planes[i], frame_v_planes[i], None, None))
                for i in range(num_images) if final_group_frames[i] is not None
            ]
            list(executor.map(worker_for_video_frame, worker_args))

            crop_results = dict(executor.map(_yuv_crop_inpaint, range(num_images)))
            images = []
            for i in range(num_images):
                rgb_crops = crop_results[i]
                if rgb_crops is None:
                    continue
                r0, r1, c0, c1 = cam_bboxes[i]
                tight_ypos, tight_xpos = cam_tight_pos[i]
                images.append(multiblend.ImageInfo(
                    filename="", bpp=8, width=c1 - c0, height=r1 - r0,
                    xpos=tight_xpos, ypos=tight_ypos,
                    channels=list(rgb_crops),
                    mask=cam_masks[i],
                ))

            workwidth, workheight = geo_workwidth, geo_workheight
            if frame_count == 1:
                _print("Computing seams with multiblend (first frame)...")
                levels = multiblend.compute_levels(images, workwidth, workheight, False, 1_000_000, 0)
                assignment, _, seam_mask_cache = compute_or_load_seams(
                    images=images,
                    workwidth=workwidth,
                    workheight=workheight,
                    pto_file=pto_file,
                    pad=pad,
                    padsides=padsides,
                    levels=levels,
                    is_video_output=True,
                    simple_seam=False,
                    content_seam=False,
                    verbosity=0,
                    print_func=_print,
                )

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
                seam_mask_cache=seam_mask_cache,
            )
            if recompute_exposure and 'exposure' in blend_out_info:
                cached_exp_info = blend_out_info['exposure']

            _canvas_r.fill(0); _canvas_g.fill(0); _canvas_b.fill(0)
            t, l = geo_min_top, geo_min_left
            _canvas_r[t:t + workheight, l:l + workwidth] = rgb_blended[0]
            _canvas_g[t:t + workheight, l:l + workwidth] = rgb_blended[1]
            _canvas_b[t:t + workheight, l:l + workwidth] = rgb_blended[2]
            canvas_r, canvas_g, canvas_b = _canvas_r, _canvas_g, _canvas_b

            if geo_n_gap > 0:
                futs = [executor.submit(_gap_fill_channel, ch, buf)
                        for ch, buf in zip((canvas_r, canvas_g, canvas_b), _geo_fill_u8)]
                canvas_rgb = [f.result() for f in futs]
            else:
                canvas_rgb = [np.clip(c, 0, 255).astype(np.uint8) for c in (canvas_r, canvas_g, canvas_b)]

            y_final, u_final, v_final = rgb_to_yuv(canvas_rgb)

            if abs(saturation - 1.0) > 0.001:
                u_final = np.clip((u_final.astype(np.float32) - 128.0) * saturation + 128.0, 0, 255).astype(np.uint8)
                v_final = np.clip((v_final.astype(np.float32) - 128.0) * saturation + 128.0, 0, 255).astype(np.uint8)

            if fisheye_mask:
                # Resize mask to output dimensions (may be rounded up from final_h/final_w)
                if out_h != final_h or out_w != final_w:
                    from scipy.ndimage import zoom
                    zoom_y = out_h / final_h
                    zoom_w = out_w / final_w
                    outside_y_resized = zoom(geo_outside_y, (zoom_y, zoom_w), order=0, mode='nearest').astype(bool)
                    # UV is already half resolution, so use same zoom factors to get to out_h/2, out_w/2
                    outside_uv_resized = zoom(geo_outside_uv, (zoom_y, zoom_w), order=0, mode='nearest').astype(bool)
                else:
                    outside_y_resized = geo_outside_y
                    outside_uv_resized = geo_outside_uv
                y_final[outside_y_resized] = 0
                u_final[outside_uv_resized] = 128
                v_final[outside_uv_resized] = 128

            if enhance:
                seed_y = int.from_bytes(os.urandom(4), 'little')
                y_final = _enhance_filter()(y_final, t=8, log2sizex=5, log2sizey=5, dither=6, seed=seed_y)
                u_final = _enhance_filter()(u_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
                v_final = _enhance_filter()(v_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)

            # Apply fisheye circular mask
            if fisheye_mask:
                # Resize mask to output dimensions (may be rounded up from final_h/final_w)
                if out_h != final_h or out_w != final_w:
                    from scipy.ndimage import zoom
                    zoom_y = out_h / final_h
                    zoom_w = out_w / final_w
                    outside_y_resized = zoom(geo_outside_y, (zoom_y, zoom_w), order=0, mode='nearest').astype(bool)
                    # UV is already half resolution, so use same zoom factors to get to out_h/2, out_w/2
                    outside_uv_resized = zoom(geo_outside_uv, (zoom_y, zoom_w), order=0, mode='nearest').astype(bool)
                else:
                    outside_y_resized = geo_outside_y
                    outside_uv_resized = geo_outside_uv
                y_final[outside_y_resized] = 0
                u_final[outside_uv_resized] = 128
                v_final[outside_uv_resized] = 128

            if timestamp:
                _draw_timestamp_yuv(y_final, u_final, v_final, group_ts)

            out_frame.planes[0].update(y_final); out_frame.planes[1].update(u_final); out_frame.planes[2].update(v_final)
            out_frame.pts = frame_count - 1
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)

            # Explicitly release per-frame objects to prevent memory growth
            del final_group_frames, worker_args, images, rgb_blended, canvas_rgb
            del y_final, u_final, v_final
            # Clear decoder's last_frame to avoid holding large frame buffers
            for decoder in camera_decoders:
                decoder.last_frame = None
            if frame_count % 10 == 0:
                gc.collect()

    if not _quiet and total_frames > 0 and sys.stderr.isatty(): sys.stderr.write("\n"); sys.stderr.flush()

    for packet in out_stream.encode(): out_container.mux(packet)
    out_container.close()
    for decoder in camera_decoders:
        decoder.close()
    _print(f"\n✅ Success! Timelapse video saved to {output_file}")


def reproject_videos(pto_file, input_files, output_file, pad, num_cores, padsides, use_sync=False, model=None, save_sync_file=None, load_sync_file=None, enhance=False, fisheye_mask=False, max_frames=0, level_subsample=1, crf="28", preset="ultrafast", timestamp=False, saturation=1.0, devignette=None):
    if not _av(): raise ImportError("PyAV is not installed, but video processing was requested.")

    mappings, global_options = build_mappings(pto_file, pad, num_cores, padsides, is_video_output=True)
    final_w, final_h = global_options['final_w'], global_options['final_h']
    if final_w > 16384:
        raise ValueError(f"Output width {final_w} exceeds codec limits for H.264/libx264. PTO='{pto_file}'")
    num_images = len(mappings)
    if len(input_files) != num_images: raise ValueError("Number of videos does not match PTO.")
    
    _precompile_numba_functions()

    # Build per-camera vignette gain LUTs (once — same resolution for all frames)
    _vignette_gains = [None] * num_images
    if devignette is not None:
        for i in range(num_images):
            sw_map, sh_map = mappings[i][4], mappings[i][5]
            _vignette_gains[i] = _build_vignette_gain(sw_map, sh_map, devignette)
        _print(f"  built {num_images} vignette gain LUTs")

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
            in_container = _av().open(input_path)
            in_stream = in_container.streams.video[0]
            in_stream.thread_type = 'AUTO'
            total_frames = in_stream.frames if in_stream.frames > 0 else 0
            if max_frames > 0 and (total_frames == 0 or total_frames > max_frames):
                total_frames = max_frames

            out_container = _av().open(output_file, mode='w')
            out_stream = out_container.add_stream("libx264", rate=in_stream.average_rate)
            out_stream.width, out_stream.height, out_stream.pix_fmt = dw, dh, 'yuv420p'
            out_stream.options = {"preset": preset, "crf": str(crf)}
        except _av().AVError as e:
            raise IOError(f"PyAV Error: Could not open video files for processing. Check paths and file integrity.\nDetails: {e}")

        # Precompute fisheye mask for single-video path
        if fisheye_mask:
            cx, cy = dw // 2, dh // 2
            r = min(cx, cy)
            ys, xs = np.ogrid[:dh, :dw]
            outside_y = (xs - cx) ** 2 + (ys - cy) ** 2 > r * r
            h_uv, w_uv = dh // 2, dw // 2
            cx_uv, cy_uv = w_uv // 2, h_uv // 2
            r_uv = min(cx_uv, cy_uv)
            ys_uv, xs_uv = np.ogrid[:h_uv, :w_uv]
            outside_uv = (xs_uv - cx_uv) ** 2 + (ys_uv - cy_uv) ** 2 > r_uv * r_uv
        else:
            outside_y = outside_uv = None

        map_y_idx, c01, c23, map_uv_idx, _, _ = mapping
        _yr0, _yr1, _yc0, _yc1, _ur0, _ur1, _uc0, _uc1 = _mapping_bboxes(mapping, dh, dw)
        frame_count = 0

        # Pre-compute pad dimensions once — they are constant across all frames.
        pad_t = pad if 'top' in padsides else 0; pad_b = pad if 'bottom' in padsides else 0
        pad_l = pad if 'left' in padsides else 0; pad_r = pad if 'right' in padsides else 0
        _needs_pad = pad_t > 0 or pad_b > 0 or pad_l > 0 or pad_r > 0
        pad_y_width = ((pad_t, pad_b), (pad_l, pad_r))
        pad_uv_width = ((pad_t // 2, pad_b // 2), (pad_l // 2, pad_r // 2))

        # --- Create output frame once to improve performance and stability ---
        try:
            out_frame = _av().VideoFrame(width=dw, height=dh, format='yuv420p')
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

        _sv_vgain = _vignette_gains[0]  # single-video: camera 0

        def _reproject_frame(frame, y_out, u_out, v_out):
            if frame.format.name != "yuv420p":
                frame = frame.reformat(format="yuv420p")
            sw_f, sh_f = frame.width, frame.height
            py_buf = np.asarray(frame.planes[0])
            py_s = py_buf.reshape(sh_f, py_buf.size // sh_f)[:, :sw_f]
            if _sv_vgain is not None:
                py_s = py_s.copy()
                _apply_vignette_y(py_s, _sv_vgain)
            pu_buf = np.asarray(frame.planes[1])
            pu_s = pu_buf.reshape(sh_f // 2, pu_buf.size // (sh_f // 2))[:, :sw_f // 2]
            pv_buf = np.asarray(frame.planes[2])
            pv_s = pv_buf.reshape(sh_f // 2, pv_buf.size // (sh_f // 2))[:, :sw_f // 2]
            if _needs_pad:
                py_s = np.pad(py_s, pad_y_width, mode='edge')
                pu_s = np.pad(pu_s, pad_uv_width, mode='edge')
                pv_s = np.pad(pv_s, pad_uv_width, mode='edge')
            reproject_y(py_s.ravel(), dw, dh, py_s.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), y_out.ravel(), outside_y.ravel() if outside_y is not None else None, dh, _yr0, _yr1, _yc0, _yc1)
            reproject_uv(pu_s.ravel(), pv_s.ravel(), dw, dh, map_uv_idx.ravel(), u_out.ravel(), v_out.ravel(), outside_uv.ravel() if outside_uv is not None else None, dh, _ur0, _ur1, _uc0, _uc1)
            return y_out, u_out, v_out

        def _encode_yuv(y, u, v, pts):
            if enhance:
                seed_y = int.from_bytes(os.urandom(4), 'little')
                y = _enhance_filter()(y, t=8, log2sizex=5, log2sizey=5, dither=6, seed=seed_y)
                u = _enhance_filter()(u, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
                v = _enhance_filter()(v, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
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
        _ts_m = 3  # margin matching worker_for_video_frame expansion
        bw[max(0, ts_y1 - _ts_m) + _tmp_pad_t:min(sh_map, ts_y2 + _ts_m) + _tmp_pad_t,
           max(0, ts_x1 - _ts_m) + _tmp_pad_l:min(sw_map, ts_x2 + _ts_m) + _tmp_pad_l] = 0
        _tmp_blend_weights_y.append(bw)
        # Reproject blend-weight map to canvas to accumulate coverage
        map_y_idx, c01, c23 = mappings[i][0], mappings[i][1], mappings[i][2]
        _tmp_reproj = np.zeros((final_h, final_w), dtype=np.float32)
        reproject_float(bw.ravel(), final_w, final_h, bw.shape[1],
                        map_y_idx.ravel(), c01.ravel(), c23.ravel(), _tmp_reproj.ravel())
        _tmp_weights += _tmp_reproj
    geo_gap = _tmp_weights < 1e-9
    del _tmp_weights, _tmp_reproj

    # Precompute fisheye circular mask (constant geometry)
    if fisheye_mask:
        _fy, _fx = final_h, final_w
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
            # Precompute flat source/destination indices for fast per-frame inpaint
            inv_i = ~mask_crop_i
            dst_flat_i = np.flatnonzero(inv_i)
            src_flat_i = ri_i[inv_i].astype(np.intp) * W_i + ci_i[inv_i].astype(np.intp)
            cam_inpaint.append((src_flat_i, dst_flat_i))
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

    # Mark pixels outside the multiblend work rectangle as gaps — the eroded
    # bounding boxes may exclude edge rows/columns that still have non-zero
    # weight (from uneroded maps), leaving them unfilled (dark) otherwise.
    _work_b = geo_min_top + geo_workheight
    _work_r = geo_min_left + geo_workwidth
    if geo_min_top > 0:
        geo_gap[:geo_min_top, :] = True
    if _work_b < final_h:
        geo_gap[_work_b:, :] = True
    if geo_min_left > 0:
        geo_gap[:, :geo_min_left] = True
    if _work_r < final_w:
        geo_gap[:, _work_r:] = True

    # Bottom-crop row (equirect only — fisheye video not supported)
    geo_crop_h = final_h
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
    _print(f"  computing full-res EDT on {W_geo}x{H_geo} canvas...")
    geo_dist = _geo_edt(~geo_gap)
    geo_blend_w = np.clip(geo_dist / feather_radius, 0.0, 1.0).astype(np.float32); del geo_dist
    geo_n_gap = int(geo_gap.sum())
    geo_gap_idx = np.where(geo_gap)  # precomputed tuple for fast per-frame fill
    # Precompute the EDT source row/col at gap pixels (1D arrays) — used for fast fill
    geo_ri_gap = geo_ri[geo_gap_idx]  # shape (geo_n_gap,)
    geo_ci_gap = geo_ci[geo_gap_idx]  # shape (geo_n_gap,)
    _print(f"  gap pixels: {geo_n_gap}, feather: {feather_radius}px")
    _print("  geometry precomputation complete.")

    # Precompute fisheye circular mask (constant geometry)
    if fisheye_mask:
        _fy, _fx = final_h, final_w
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
        in_containers = [_av().open(f) for f in input_files]
        in_streams = [c.streams.video[0] for c in in_containers]
        for s in in_streams: s.thread_type = 'AUTO'

        out_container = _av().open(output_file, mode='w')
        out_stream = out_container.add_stream("libx264", rate=in_streams[0].average_rate)
        out_stream.width, out_stream.height, out_stream.pix_fmt = out_w, out_h, 'yuv420p'
        out_stream.options = {"preset": preset, "crf": str(crf)}
    except _av().AVError as e:
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

    frame_y_planes = np.empty((num_images, final_h, final_w), dtype=np.uint8)
    frame_u_planes = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.uint8)
    frame_v_planes = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.uint8)
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
            src_flat, dst_flat = inpaint
            r_crop.ravel()[dst_flat] = r_crop.ravel()[src_flat]
            g_crop.ravel()[dst_flat] = g_crop.ravel()[src_flat]
            b_crop.ravel()[dst_flat] = b_crop.ravel()[src_flat]
        return i, (r_crop, g_crop, b_crop)

    frame_iters = [c.decode(s) for c, s in zip(in_containers, in_streams)]
    frame_count = 0
    loop_iterator = synchronized_frame_groups if use_sync else zip(*frame_iters)
    
    # --- Create output frame once to improve performance and stability ---
    try:
        out_frame = _av().VideoFrame(width=out_w, height=out_h, format='yuv420p')
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
            if _cv2() is not None:
                small = _cv2().resize(buf_u8, (geo_sw, geo_sh), interpolation=_cv2().INTER_AREA).astype(np.float32)
            else:
                from PIL import Image as _PIL2
                small = np.array(_PIL2.fromarray(buf_u8).resize((geo_sw, geo_sh), _PIL2.BOX)).astype(np.float32)
            blurred_u8 = gaussian_filter(small, sigma=geo_sigma_s).clip(0, 255).astype(np.uint8)
            if _cv2() is not None:
                full = _cv2().resize(blurred_u8, (W_geo, H_geo), interpolation=_cv2().INTER_LINEAR).astype(np.float32)
            else:
                from PIL import Image as _PIL2
                full = np.array(_PIL2.fromarray(blurred_u8).resize((W_geo, H_geo), _PIL2.BILINEAR)).astype(np.float32)
            result = ch_f * geo_blend_w + full * (1.0 - geo_blend_w)
            np.clip(result, 0, 255, out=result)
            return result.astype(np.uint8)

    # Cache exposure correction info to support --level-subsample.
    cached_exp_info = None
    # Valid-pixel bounding boxes per camera — reproject kernels skip dead regions.
    _map_bboxes_cams = [_mapping_bboxes(mappings[i], final_h, final_w) for i in range(num_images)]

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        current_frame_indices = [-1] * num_images
        loop_iter = iter(enumerate(loop_iterator, 1))

        def _produce_one():
            """Decode, reproject and crop the next valid frame group.
            Runs on a dedicated producer thread so it overlaps with the
            blend/encode of the previous frame. Returns None at end of input."""
            while True:
                try:
                    loop_idx, group = next(loop_iter)
                except StopIteration:
                    return None
                final_group_frames = [None] * num_images
                if use_sync:
                    target_indices = group
                    ended = False
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
                            ended = True; break
                    if ended:
                        return None
                else:
                    final_group_frames = list(group)

                if any(f is None for f in final_group_frames): continue

                worker_args = [
                    ((i, final_group_frames[i], mappings[i], final_w, final_h, blend_weights_y[i], None, pad, padsides, _vignette_gains[i], (geo_outside_y, geo_outside_uv) if geo_outside_y is not None else None, out_h, _map_bboxes_cams[i]),
                     (frame_y_planes[i], frame_u_planes[i], frame_v_planes[i], None, None))
                    for i in range(num_images) if final_group_frames[i] is not None
                ]
                list(executor.map(worker_for_video_frame, worker_args))

                # Build ImageInfo list: yuv_to_rgb + crop + inpaint per camera (parallel)
                crop_results = dict(executor.map(_yuv_crop_inpaint, range(num_images)))
                images = []
                for i in range(num_images):
                    rgb_crops = crop_results[i]
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

                # Capture the timestamp value now — the frame objects are
                # released once the next group is decoded.
                ts_val = None
                if timestamp:
                    f0 = final_group_frames[0]
                    ct = in_containers[0].start_time
                    if f0 is not None and f0.pts is not None and ct is not None and ct > 0:
                        ts_val = ct / 1_000_000 + float(f0.pts) * float(in_streams[0].time_base)
                return loop_idx, images, ts_val

        producer = ThreadPoolExecutor(max_workers=1)
        pending = producer.submit(_produce_one)
        while True:
            item = pending.result()
            if item is None: break
            frame_count += 1
            if max_frames > 0 and frame_count > max_frames: break
            # Prefetch the next frame group while this one is blended/encoded.
            pending = producer.submit(_produce_one)
            loop_idx, images, ts_val = item

            if not _quiet and total_frames > 0 and (loop_idx % 5 == 0 or loop_idx == total_frames):
                percent_done = (loop_idx / total_frames) * 100
                if sys.stderr.isatty():
                    bar_length = 40; filled_len = int(round(bar_length*loop_idx/float(total_frames)))
                    bar = '█'*filled_len + '-'*(bar_length - filled_len)
                    sys.stderr.write(f'Stitching: [{bar}] {percent_done:.1f}% \r'); sys.stderr.flush()
                else: _print(f"PROGRESS:{percent_done:.1f}", file=sys.stderr, flush=True)

            workwidth, workheight = geo_workwidth, geo_workheight
            # Seam computation — only on first frame, then reuse
            if frame_count == 1:
                _print("Computing seams with multiblend (first frame)...")
                levels = multiblend.compute_levels(images, workwidth, workheight, False, 1_000_000, 0)
                assignment, _, seam_mask_cache = compute_or_load_seams(
                    images=images,
                    workwidth=workwidth,
                    workheight=workheight,
                    pto_file=pto_file,
                    pad=pad,
                    padsides=padsides,
                    levels=levels,
                    is_video_output=True,
                    simple_seam=False,
                    content_seam=False,
                    verbosity=0,
                    print_func=_print,
                )
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
                seam_mask_cache=seam_mask_cache,
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

            if abs(saturation - 1.0) > 0.001:
                u_final = np.clip((u_final.astype(np.float32) - 128.0) * saturation + 128.0, 0, 255).astype(np.uint8)
                v_final = np.clip((v_final.astype(np.float32) - 128.0) * saturation + 128.0, 0, 255).astype(np.uint8)

            # Apply fisheye circular mask
            if fisheye_mask:
                # Resize mask to output dimensions (may be rounded up from final_h/final_w)
                if out_h != final_h or out_w != final_w:
                    from scipy.ndimage import zoom
                    zoom_y = out_h / final_h
                    zoom_w = out_w / final_w
                    outside_y_resized = zoom(geo_outside_y, (zoom_y, zoom_w), order=0, mode='nearest').astype(bool)
                    # UV is already half resolution, so use same zoom factors to get to out_h/2, out_w/2
                    outside_uv_resized = zoom(geo_outside_uv, (zoom_y, zoom_w), order=0, mode='nearest').astype(bool)
                else:
                    outside_y_resized = geo_outside_y
                    outside_uv_resized = geo_outside_uv
                y_final[outside_y_resized] = 0
                u_final[outside_uv_resized] = 128
                v_final[outside_uv_resized] = 128

            if enhance:
                seed_y = int.from_bytes(os.urandom(4), 'little')
                y_final = _enhance_filter()(y_final, t=8, log2sizex=5, log2sizey=5, dither=6, seed=seed_y)
                u_final = _enhance_filter()(u_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)
                v_final = _enhance_filter()(v_final, t=16, log2sizex=4, log2sizey=4, dither=0, seed=0)

            if timestamp and ts_val is not None:
                _draw_timestamp_yuv(y_final, u_final, v_final, ts_val)

            # --- Update and encode the single, reused output frame ---
            out_frame.planes[0].update(y_final); out_frame.planes[1].update(u_final); out_frame.planes[2].update(v_final)
            # Set the Presentation Time Stamp (PTS)
            out_frame.pts = frame_count - 1
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)

        producer.shutdown(wait=True)

    if not _quiet and total_frames > 0 and sys.stderr.isatty(): sys.stderr.write("\n"); sys.stderr.flush()

    for packet in out_stream.encode(): out_container.mux(packet)
    out_container.close()
    for c in in_containers: c.close()
    _print(f"\n✅ Success! Panoramic video saved to {output_file}")


def stitch(input_files, output_file, *, pto_file=None, projection='equirect',
           pad=0, padsides=None, enhance=False, fisheye_mask=False,
           force_video_dims=False, max_frames=0, level_subsample=1,
           sync=False, model=None, save_sync=None, load_sync=None,
           quiet=False, num_cores=None, lens_files=None,
           output_width=None, output_height=None, crop_to_content: bool = True,
           timestamp: bool = False, devignette=-0.20):
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
    lens_files : dict or None, optional
        Mapping of camera number to lens.pto path. When provided, these
        calibrations are used instead of discovering lens.pto files from the
        input paths. Useful when the inputs are local copies that no longer
        live under /meteor/cam*/.
    output_width : int or None, optional
        Force the generated PTO output canvas width. Defaults to the standard
        size for the chosen projection.
    output_height : int or None, optional
        Force the generated PTO output canvas height.
    crop_to_content : bool, optional
        For images, crop the output canvas to the last row that has any image
        content. When False, the full PTO canvas (including empty/gap rows) is
        kept and gap-filled. Default is True.
    timestamp : bool, optional
        Overlay a UTC timestamp (YYYY-MM-DD hh:mm:ss.ff) in the lower-left
        corner of each video frame. Has no effect for still-image output.
    devignette : float or None, optional
        Radial vignetting correction coefficient k1 for the model
        brightness(r) = 1 + k1*r² where r is normalised to the corner.
        Default is -0.20. Set to 0.0 or None to disable.

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
        pto_file = generate_pto_from_lens_files(
            input_files, projection,
            lens_files=lens_files,
            w=output_width, h=output_height
        )
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
            enhance, force_video_dims=force_video_dims, fisheye_mask=fisheye_mask,
            crop_to_content=crop_to_content, devignette=devignette
        )
    else:
        if len(input_files) < 2 and sync:
            sync = False
        reproject_videos(
            pto_file, input_files, output_file,
            pad, num_cores, padsides_set, sync, model,
            save_sync_file=save_sync, load_sync_file=load_sync, enhance=enhance,
            fisheye_mask=fisheye_mask, max_frames=max_frames,
            level_subsample=level_subsample, timestamp=timestamp,
            devignette=devignette
        )

    if auto_generated_pto and os.path.exists(auto_generated_pto):
        try:
            os.unlink(auto_generated_pto)
        except Exception:
            pass


def extract_camera_number_from_path(path: str) -> int:
    """Extract camera number from a path like /meteor/cam1/20260621/07/full_00.jpg -> 1"""
    # Look for 'cam' followed by a number in the path. Accept both the original
    # directory layout (/cam1/) and downloaded filenames (cam1.jpg, _cam1.jpg).
    # Use negative lookbehind/lookahead to ensure 'cam' is a standalone token.
    match = re.search(r'(?<![A-Za-z0-9])cam(\d+)(?![A-Za-z0-9])', path)
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

    w and h are rounded up to the nearest multiple of 16 so the rendered
    canvas is already the correct output size, avoiding zero-pad expansion
    that would create black edge bars and gap pixels near canvas boundaries.
    """
    w = (w + 15) & ~15
    h = (h + 15) & ~15
    f = 3 if projection == 'fisheye' else 2
    v = 190 if projection == 'fisheye' else 360
    return (f'p f{f} w{w} h{h} v{v} E0 R0 n"TIFF_m c:LZW"\n'
            f'm g1 i0 m2 p0.00784314\n')


def generate_pto_from_lens_files(input_files: list, projection: str,
                                 lens_files: dict = None,
                                 w: int = None, h: int = None) -> str:
    """Generate a PTO file from lens.pto files found relative to input files.
    
    Args:
        input_files: List of input image file paths
        projection: 'fisheye' or 'equirect'
        lens_files: Optional dict mapping camera number to lens.pto path. If
            provided, these are used instead of searching relative to the inputs.
        w: Optional output canvas width. Defaults to the standard size for the
            projection.
        h: Optional output canvas height.
    
    Returns:
        Path to the generated PTO file, or None on failure
    """
    # Define output dimensions based on projection
    explicit_size = w is not None and h is not None
    if w is None or h is None:
        if projection == 'fisheye':
            w, h = 4096, 4096
        else:  # equirect
            w, h = 4096, 2160
    
    # Find lens.pto files for each input if not supplied explicitly
    missing_lens = []
    if lens_files is None:
        lens_files = {}
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
    else:
        # Verify the explicitly supplied lens files exist.
        missing = [cam for cam, path in lens_files.items() if not os.path.exists(path)]
        if missing:
            _print(f"Error: lens.pto files not found for cameras {missing}. Cannot proceed without all calibration files.", file=sys.stderr)
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
            _av_mod_local = _av()
            if _av_mod_local:
                with _av_mod_local.open(img_path) as _vc:
                    vs = _vc.streams.video[0]
                    actual_w, actual_h = vs.width, vs.height
                break
        except Exception:
            continue

    # Track the square-pixel display width (equals actual_w for square-pixel inputs,
    # and equals cal_w * (actual_h/cal_h) for non-square SD inputs like 704x576).
    display_w = actual_w

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
                                    # Non-square pixels: SD content stored with a different aspect
                                    # ratio than the calibration (e.g. 704x576 for 1920x1080 FOV).
                                    # Use y-scale as reference and derive a square-pixel display
                                    # width so reprojection geometry is correct. The pipeline
                                    # resizes stored images to (display_w x actual_h) before reprojecting.
                                    if abs(sx - sy) / max(sx, sy, 1e-9) > 0.005:
                                        _dw = max(2, int(round(cal_w * sy)) & ~1)
                                        _print(f"  Non-square pixels: stored {actual_w}x{actual_h}, "
                                               f"display {_dw}x{actual_h} (PAR {sx/sy:.4f})")
                                        sx = _dw / cal_w  # now equals sy: uniform scale
                                        display_w = _dw
                                    else:
                                        display_w = actual_w
                                    stripped = re.sub(r'\bw\d+', f'w{display_w}', stripped)
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

    # Scale output canvas proportionally if input is not the calibration size and
    # the caller did not request a fixed output size.
    if not explicit_size and actual_w is not None:
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
        if cal_w_ref and cal_w_ref != display_w:
            scale = display_w / cal_w_ref
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
    fd_closed = False
    try:
        with os.fdopen(pto_fd, 'w') as f:
            f.write(pto_content)
        fd_closed = True  # os.fdopen closes the descriptor on exit
        _print(f"Generated PTO file with {len(lines)} lines for {len(lens_files)} cameras")

        if projection == 'fisheye':
            pto_data = pto_mapper.parse_pto_file(pto_path)
            pto_mapper.rotate_panorama(pto_data, yaw_deg=0, pitch_deg=-90, roll_deg=0)
            pto_mapper.write_pto_file(pto_data, pto_path)
            _print("Applied fisheye rotation (0,-90,0) via pto_mapper")

        return pto_path
    except Exception as e:
        _print(f"Error: Failed to write PTO file: {e}", file=sys.stderr)
        if not fd_closed:
            try:
                os.close(pto_fd)
            except OSError:
                pass
        return None


def launch_gui():
    """Launch the stitcher GUI. Called when no CLI arguments are provided."""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox, scrolledtext
    except ImportError:
        print("Error: tkinter is not available. Install python3-tk to use the GUI.", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    root.title("Stitcher")
    root.resizable(True, True)
    root.minsize(820, 940)

    def _lock_geometry():
        """Cycle all tabs so widgets get mapped, then pin the window size."""
        for i in range(nb.index("end")):
            nb.select(i)
            root.update_idletasks()
        nb.select(0)
        root.update_idletasks()
        w = max(root.winfo_width(),  820)
        h = max(root.winfo_height(), 940)
        root.geometry(f"{w}x{h}")

    root.after(100, _lock_geometry)

    # ── Shared state ─────────────────────────────────────────────────────────
    running      = threading.Event()
    cancel_proc  = [None]
    _run_start   = [None]   # wall-clock start for ETA

    # ── Helpers ───────────────────────────────────────────────────────────────
    def lbl(parent, text, col, row, **kw):
        ttk.Label(parent, text=text).grid(column=col, row=row, sticky="w", padx=5, pady=3, **kw)

    def ent(parent, var, col, row, width=28, **kw):
        e = ttk.Entry(parent, textvariable=var, width=width)
        e.grid(column=col, row=row, sticky="ew", padx=5, pady=3, **kw)
        return e

    def browseopen(var, multiple=False, filetypes=None):
        ft = filetypes or [("All files", "*.*")]
        if multiple:
            paths = filedialog.askopenfilenames(filetypes=ft)
            if paths:
                var.set(" ".join(f'"{p}"' if " " in p else p for p in paths))
        else:
            p = filedialog.askopenfilename(filetypes=ft)
            if p: var.set(p)

    def browsesave(var, filetypes=None):
        ft = filetypes or [("All files", "*.*")]
        p = filedialog.asksaveasfilename(filetypes=ft, defaultextension=ft[0][1].lstrip("*"))
        if p: var.set(p)


    # ── Load cameras.json ─────────────────────────────────────────────────────
    _cameras_json = os.path.join(os.path.dirname(__file__), '..', 'server', 'data', 'cameras.json')
    _station_map = {}   # display label → {"id": amsXXX, "ssh": amsXXX}
    try:
        import json as _json
        with open(_cameras_json) as _f:
            _cam_data = _json.load(_f)
        for _ams_id, _cfg in sorted(_cam_data.items()):
            _st = _cfg.get("station", {})
            _name = (_st.get("display_name") or _st.get("name") or _ams_id).capitalize()
            _code = _st.get("code", "")
            _label = f"{_name}  [{_code}]  ({_ams_id})" if _code else f"{_name}  ({_ams_id})"
            _station_map[_label] = _ams_id
    except Exception:
        pass   # cameras.json missing or malformed — fall back to free-text entry

    # ── Notebook ──────────────────────────────────────────────────────────────
    nb_frame = tk.Frame(root)
    nb_frame.pack(fill="both", expand=True, padx=8, pady=(8,4))
    nb_frame.pack_propagate(False)  # prevent content from resizing this frame/window
    nb = ttk.Notebook(nb_frame)
    nb.pack(fill="both", expand=True)

    # ═══════════════════════════════════════════════════════════════
    # TAB 1 – Timelapse  (primary tab, shown first)
    # ═══════════════════════════════════════════════════════════════
    tab_tl = ttk.Frame(nb)
    nb.add(tab_tl, text="  Timelapse  ")
    tab_tl.columnconfigure(1, weight=1)
    r = 0

    # ── State vars ──
    tl_station_var  = tk.StringVar()   # holds the SSH host actually passed to --station
    tl_station_label_var = tk.StringVar()  # holds the dropdown display label
    tl_proj_var     = tk.StringVar(value="fisheye")
    tl_quality_var  = tk.StringVar(value="sd")
    # Start datetime components — default to last midnight UTC
    _now = datetime.datetime.now(datetime.timezone.utc)
    _midnight = _now.replace(hour=0, minute=0, second=0, microsecond=0)
    tl_sy = tk.IntVar(value=_midnight.year)
    tl_sm = tk.IntVar(value=_midnight.month)
    tl_sd = tk.IntVar(value=_midnight.day)
    tl_sh = tk.IntVar(value=0)
    tl_smin = tk.IntVar(value=0)
    # End datetime components
    tl_ey = tk.IntVar(value=_midnight.year)
    tl_em = tk.IntVar(value=_midnight.month)
    tl_ed = tk.IntVar(value=_midnight.day)
    tl_eh = tk.IntVar(value=1)
    tl_emin = tk.IntVar(value=0)
    # Duration
    tl_dur_h_var    = tk.IntVar(value=1)
    tl_dur_m_var    = tk.IntVar(value=0)
    tl_use_end_var  = tk.BooleanVar(value=False)   # True=use end datetime, False=use duration
    tl_speed_var    = tk.StringVar(value="60")
    tl_fps_var      = tk.IntVar(value=30)
    tl_pattern_var  = tk.StringVar(value="/meteor/cam?")
    tl_output_var   = tk.StringVar()
    tl_enhance_var  = tk.BooleanVar()
    tl_timestamp_var= tk.BooleanVar()
    tl_crf_var      = tk.StringVar(value="28")
    tl_preset_var   = tk.StringVar(value="ultrafast")
    tl_saturation_var = tk.DoubleVar(value=1.0)
    tl_pto_var      = tk.StringVar()

    # Helper: read the 6 spinboxes as "YYYY-MM-DD HH:MM:00"
    def _tl_start_str():
        return f"{tl_sy.get():04d}-{tl_sm.get():02d}-{tl_sd.get():02d} {tl_sh.get():02d}:{tl_smin.get():02d}:00"
    def _tl_end_str():
        return f"{tl_ey.get():04d}-{tl_em.get():02d}-{tl_ed.get():02d} {tl_eh.get():02d}:{tl_emin.get():02d}:00"

    # Helper: build a compact datetime spinbox row (year/month/day  hour:min)
    def _dt_spinboxes(parent, y_v, mo_v, d_v, h_v, min_v):
        """Pack date+time spinboxes into parent frame.
        Increment/decrement propagate carry/borrow across the full datetime.
        Returns the frame."""
        f = ttk.Frame(parent)

        def _read_dt():
            try:
                return datetime.datetime(y_v.get(), mo_v.get(), d_v.get(),
                                         h_v.get(), min_v.get(), 0,
                                         tzinfo=datetime.timezone.utc)
            except ValueError:
                return None

        def _write_dt(dt):
            y_v.set(dt.year); mo_v.set(dt.month); d_v.set(dt.day)
            h_v.set(dt.hour); min_v.set(dt.minute)

        def _dt_spinbox(field, delta, fmt, width):
            """field: 'year'|'month'|'day'|'hour'|'minute'
               delta: timedelta (or relativedelta for month/year) magnitude."""
            str_var = tk.StringVar()
            _busy = [False]

            def _refresh(*_):
                if _busy[0]: return
                _busy[0] = True
                try:
                    val = {'year': y_v, 'month': mo_v, 'day': d_v,
                           'hour': h_v, 'minute': min_v}[field].get()
                    str_var.set(fmt.format(val))
                finally:
                    _busy[0] = False

            # Keep display in sync when any IntVar changes
            for iv in (y_v, mo_v, d_v, h_v, min_v):
                iv.trace_add("write", _refresh)
            _refresh()

            def _apply_manual(*_):
                if _busy[0]: return
                _busy[0] = True
                try:
                    raw = str_var.get().strip().lstrip("0") or "0"
                    new_val = int(raw)
                    dt = _read_dt()
                    if dt is None:
                        return
                    try:
                        new_dt = dt.replace(**{field: new_val})
                        _write_dt(new_dt)
                    except (ValueError, OverflowError):
                        pass  # invalid manual entry — leave as-is
                    finally:
                        val = {'year': y_v, 'month': mo_v, 'day': d_v,
                               'hour': h_v, 'minute': min_v}[field].get()
                        str_var.set(fmt.format(val))
                except (ValueError, tk.TclError):
                    pass
                finally:
                    _busy[0] = False

            sb = ttk.Spinbox(f, textvariable=str_var, width=width,
                             from_=0, to=9999, wrap=True)

            # Select-all on focus so user can just type to overwrite
            def _on_focus_in(event, _sb=sb):
                _sb.selection_range(0, "end")
                _sb.icursor("end")
            sb.bind("<FocusIn>", _on_focus_in)

            # Apply manual edits only on FocusOut or Return (not every keystroke)
            sb.bind("<FocusOut>", lambda e: _apply_manual())
            sb.bind("<Return>", lambda e: _apply_manual())

            def _increment(event):
                dt = _read_dt()
                if dt is None: return
                try:
                    if field == 'minute':
                        _write_dt(dt + datetime.timedelta(minutes=1))
                    elif field == 'hour':
                        _write_dt(dt + datetime.timedelta(hours=1))
                    elif field == 'day':
                        _write_dt(dt + datetime.timedelta(days=1))
                    elif field == 'month':
                        # roll month forward, carry into year
                        y, m = dt.year, dt.month + 1
                        if m > 12: y, m = y + 1, 1
                        import calendar as _cal
                        d = min(dt.day, _cal.monthrange(y, m)[1])
                        _write_dt(dt.replace(year=y, month=m, day=d))
                    elif field == 'year':
                        _write_dt(dt.replace(year=dt.year + 1))
                except (ValueError, OverflowError):
                    pass
                return "break"

            def _decrement(event):
                dt = _read_dt()
                if dt is None: return
                try:
                    if field == 'minute':
                        _write_dt(dt - datetime.timedelta(minutes=1))
                    elif field == 'hour':
                        _write_dt(dt - datetime.timedelta(hours=1))
                    elif field == 'day':
                        _write_dt(dt - datetime.timedelta(days=1))
                    elif field == 'month':
                        y, m = dt.year, dt.month - 1
                        if m < 1: y, m = y - 1, 12
                        import calendar as _cal
                        d = min(dt.day, _cal.monthrange(y, m)[1])
                        _write_dt(dt.replace(year=y, month=m, day=d))
                    elif field == 'year':
                        _write_dt(dt.replace(year=dt.year - 1))
                except (ValueError, OverflowError):
                    pass
                return "break"

            sb.bind("<<Increment>>", _increment)
            sb.bind("<<Decrement>>", _decrement)
            return sb

        _dt_spinbox('year',   None, "{:04d}", 5).pack(side="left")
        ttk.Label(f, text="-").pack(side="left")
        _dt_spinbox('month',  None, "{:02d}", 3).pack(side="left")
        ttk.Label(f, text="-").pack(side="left")
        _dt_spinbox('day',    None, "{:02d}", 3).pack(side="left")
        ttk.Label(f, text="  ").pack(side="left")
        _dt_spinbox('hour',   None, "{:02d}", 3).pack(side="left")
        ttk.Label(f, text=":").pack(side="left")
        _dt_spinbox('minute', None, "{:02d}", 3).pack(side="left")
        ttk.Label(f, text=" UTC", foreground="#888", font=("", 8)).pack(side="left", padx=2)
        return f

    # ── Station ──
    ttk.Label(tab_tl, text="Station", font=("", 10, "bold")).grid(
        column=0, row=r, columnspan=4, sticky="w", padx=5, pady=(10,2)); r+=1

    lbl(tab_tl, "Station:", 0, r)
    _station_labels = list(_station_map.keys())
    if _station_labels:
        # Dropdown from cameras.json
        station_combo = ttk.Combobox(tab_tl, textvariable=tl_station_label_var,
                                     values=_station_labels, state="readonly", width=42)
        station_combo.grid(column=1, row=r, columnspan=2, sticky="w", padx=5, pady=3)
        if _station_labels:
            station_combo.current(0)
        def _on_station_select(*_):
            lbl_sel = tl_station_label_var.get()
            tl_station_var.set(_station_map.get(lbl_sel, ""))
        tl_station_label_var.trace_add("write", _on_station_select)
        _on_station_select()
    else:
        # Fallback: plain text entry
        ent(tab_tl, tl_station_var, 1, r, width=30)
        ttk.Label(tab_tl, text="SSH hostname", foreground="#888", font=("",8)).grid(
            column=2, row=r, sticky="w", padx=4)
    r+=1

    lbl(tab_tl, "Camera pattern:", 0, r)
    ent(tab_tl, tl_pattern_var, 1, r, width=22)
    ttk.Label(tab_tl, text="glob, e.g. /meteor/cam?", foreground="#888", font=("",8)).grid(
        column=2, row=r, sticky="w", padx=4); r+=1

    ttk.Separator(tab_tl, orient="horizontal").grid(
        column=0, row=r, columnspan=4, sticky="ew", pady=6); r+=1

    # ── Time range ──
    ttk.Label(tab_tl, text="Time range (UTC)", font=("", 10, "bold")).grid(
        column=0, row=r, columnspan=4, sticky="w", padx=5, pady=(0,2)); r+=1

    lbl(tab_tl, "Start:", 0, r)
    _dt_spinboxes(tab_tl, tl_sy, tl_sm, tl_sd, tl_sh, tl_smin).grid(
        column=1, row=r, columnspan=3, sticky="w", padx=5, pady=3); r+=1

    # End / Duration toggle
    end_toggle_frame = ttk.Frame(tab_tl)
    end_toggle_frame.grid(column=0, row=r, columnspan=4, sticky="w", padx=5, pady=3); r+=1
    ttk.Radiobutton(end_toggle_frame, text="Duration:", variable=tl_use_end_var, value=False).pack(side="left")
    dur_sb_frame = ttk.Frame(end_toggle_frame)
    dur_sb_frame.pack(side="left", padx=4)
    ttk.Spinbox(dur_sb_frame, textvariable=tl_dur_h_var, from_=0, to=999, width=5).pack(side="left")
    ttk.Label(dur_sb_frame, text=" h ").pack(side="left")
    ttk.Spinbox(dur_sb_frame, textvariable=tl_dur_m_var, from_=0, to=59, width=4).pack(side="left")
    ttk.Label(dur_sb_frame, text=" min").pack(side="left")
    ttk.Label(end_toggle_frame, text="     ").pack(side="left")
    ttk.Radiobutton(end_toggle_frame, text="End time:", variable=tl_use_end_var, value=True).pack(side="left")
    end_dt_frame = _dt_spinboxes(end_toggle_frame, tl_ey, tl_em, tl_ed, tl_eh, tl_emin)
    end_dt_frame.pack(side="left", padx=4)

    def _update_end_state(*_):
        use = tl_use_end_var.get()
        for w in end_dt_frame.winfo_children():
            try: w.config(state="normal" if use else "disabled")
            except Exception: pass
        for w in dur_sb_frame.winfo_children():
            try: w.config(state="disabled" if use else "normal")
            except Exception: pass
    tl_use_end_var.trace_add("write", _update_end_state)
    _update_end_state()

    ttk.Separator(tab_tl, orient="horizontal").grid(
        column=0, row=r, columnspan=4, sticky="ew", pady=6); r+=1

    # ── Output settings ──
    ttk.Label(tab_tl, text="Output", font=("", 10, "bold")).grid(
        column=0, row=r, columnspan=4, sticky="w", padx=5, pady=(0,2)); r+=1

    lbl(tab_tl, "Projection:", 0, r)
    pf = ttk.Frame(tab_tl)
    pf.grid(column=1, row=r, columnspan=3, sticky="w", padx=5, pady=3)
    ttk.Radiobutton(pf, text="Fisheye", variable=tl_proj_var, value="fisheye").pack(side="left", padx=6)
    ttk.Radiobutton(pf, text="Equirectangular", variable=tl_proj_var, value="equirect").pack(side="left", padx=6)
    r+=1

    tl_pto_lbl = lbl(tab_tl, "Custom PTO:", 0, r)
    tl_pto_ent = ent(tab_tl, tl_pto_var, 1, r, width=30)
    tl_pto_btn = ttk.Button(tab_tl, text="Browse…",
        command=lambda: browseopen(tl_pto_var, filetypes=[("PTO","*.pto"),("All","*.*")]))
    tl_pto_btn.grid(column=2, row=r, padx=5, pady=3)
    ttk.Label(tab_tl, text="override auto-generated PTO", foreground="#888", font=("",8)).grid(
        column=3, row=r, sticky="w", padx=4)
    r+=1

    lbl(tab_tl, "Source quality:", 0, r)
    qf = ttk.Frame(tab_tl)
    qf.grid(column=1, row=r, columnspan=3, sticky="w", padx=5, pady=3)
    ttk.Radiobutton(qf, text="SD  (mini_mm.mp4)", variable=tl_quality_var, value="sd").pack(side="left", padx=6)
    ttk.Radiobutton(qf, text="HD  (full_mm.mp4)",  variable=tl_quality_var, value="hd").pack(side="left", padx=6)
    r+=1

    lbl(tab_tl, "Speed-up factor:", 0, r)
    sf = ttk.Frame(tab_tl)
    sf.grid(column=1, row=r, sticky="w", padx=5, pady=3)
    ttk.Spinbox(sf, textvariable=tl_speed_var, values=[10,30,60,120,300,600,3600], width=7).pack(side="left")
    ttk.Label(sf, text="× realtime").pack(side="left", padx=4)
    r+=1

    lbl(tab_tl, "Frame rate:", 0, r)
    ff = ttk.Frame(tab_tl)
    ff.grid(column=1, row=r, sticky="w", padx=5, pady=3)
    ttk.Spinbox(ff, textvariable=tl_fps_var, from_=1, to=120, width=5).pack(side="left")
    ttk.Label(ff, text=" fps").pack(side="left", padx=4)
    r+=1

    lbl(tab_tl, "Output file (.mp4):", 0, r)
    ent(tab_tl, tl_output_var, 1, r, width=30)
    ttk.Button(tab_tl, text="Browse…",
        command=lambda: browsesave(tl_output_var, [("MP4 video","*.mp4"),("All","*.*")])).grid(
        column=2, row=r, padx=5, pady=3); r+=1

    ttk.Checkbutton(tab_tl, text="Enhance (noise reduction)", variable=tl_enhance_var).grid(
        column=1, row=r, sticky="w", padx=5); r+=1
    ttk.Checkbutton(tab_tl, text="Overlay UTC timestamp on output", variable=tl_timestamp_var).grid(
        column=1, row=r, sticky="w", padx=5); r+=1

    lbl(tab_tl, "CRF (quality):", 0, r)
    _tl_crf_frame = ttk.Frame(tab_tl)
    _tl_crf_frame.grid(column=1, row=r, sticky="w", padx=5, pady=3)
    ttk.Spinbox(_tl_crf_frame, textvariable=tl_crf_var, from_=0, to=51, width=5).pack(side="left")
    ttk.Label(_tl_crf_frame, text=" (0=lossless, 28=default, 51=worst)", foreground="#888", font=("",8)).pack(side="left")
    r+=1

    lbl(tab_tl, "Preset:", 0, r)
    ttk.Combobox(tab_tl, textvariable=tl_preset_var, width=14,
        values=["ultrafast","superfast","veryfast","faster","fast","medium","slow","veryslow"]).grid(
        column=1, row=r, sticky="w", padx=5, pady=3); r+=1

    lbl(tab_tl, "Saturation:", 0, r)
    _tl_sat_frame = ttk.Frame(tab_tl)
    _tl_sat_frame.grid(column=1, row=r, columnspan=2, sticky="ew", padx=5, pady=3)
    _tl_sat_lbl = ttk.Label(_tl_sat_frame, text="1.0", width=4)
    _tl_sat_lbl.pack(side="right")
    def _tl_sat_update(val):
        _tl_sat_lbl.config(text=f"{float(val):.1f}")
    tk.Scale(_tl_sat_frame, variable=tl_saturation_var, from_=0.0, to=3.0, resolution=0.1,
             orient="horizontal", showvalue=False, command=_tl_sat_update).pack(side="left", fill="x", expand=True)
    r+=1

    tl_devignette_var = tk.DoubleVar(value=-0.20)
    lbl(tab_tl, "Devignette:", 0, r)
    _tl_dv_frame = ttk.Frame(tab_tl)
    _tl_dv_frame.grid(column=1, row=r, columnspan=2, sticky="ew", padx=5, pady=3)
    _tl_dv_lbl = ttk.Label(_tl_dv_frame, text="-0.20", width=5)
    _tl_dv_lbl.pack(side="right")
    def _tl_dv_update(val):
        _tl_dv_lbl.config(text=f"{float(val):.2f}")
    tk.Scale(_tl_dv_frame, variable=tl_devignette_var, from_=0.0, to=-0.5, resolution=0.01,
             orient="horizontal", showvalue=False, command=_tl_dv_update).pack(side="left", fill="x", expand=True)
    ttk.Label(tab_tl, text="0=off, negative=correct falloff", foreground="#888", font=("",8)).grid(
        column=3, row=r, sticky="w", padx=4)
    r+=1

    # ── Auto-update timelapse output filename when projection changes ──
    _TL_DEFAULTS = {"fisheye": "fisheye.mp4", "equirect": "equirect.mp4"}
    def _tl_proj_changed(*_):
        cur = tl_output_var.get()
        if cur in _TL_DEFAULTS.values() or cur == "":
            tl_output_var.set(_TL_DEFAULTS.get(tl_proj_var.get(), "fisheye.mp4"))
    tl_proj_var.trace_add("write", _tl_proj_changed)
    _tl_proj_changed()   # set initial default

    # ── Preview pane ─────────────────────────────────────────────────────────
    # Separator + label row
    ttk.Separator(tab_tl, orient="horizontal").grid(
        column=0, row=r, columnspan=4, sticky="ew", pady=(8, 2)); r += 1
    preview_hdr_var = tk.StringVar(value="Preview  (auto-updates)")
    ttk.Label(tab_tl, textvariable=preview_hdr_var, font=("", 9, "bold"),
              foreground="#555").grid(column=0, row=r, columnspan=4, sticky="w", padx=5); r += 1

    # Canvas that holds the two stitched thumbnails side by side
    tab_tl.rowconfigure(r, weight=1)
    preview_canvas = tk.Canvas(tab_tl, bg="#111", height=220, cursor="watch")
    preview_canvas.grid(column=0, row=r, columnspan=4, sticky="nsew", padx=5, pady=(2, 5))
    preview_canvas.config(cursor="")

    # Internal state
    _prev_img_refs  = [None, None]   # keep PhotoImage refs alive
    _prev_job       = [None]         # after() debounce handle
    _prev_thread    = [None]         # running background thread
    _prev_tmpdir    = [None]         # current temp dir (cleaned on next fetch)
    _prev_cancel    = [threading.Event()]   # set to abort in-flight fetch
    # Per-frame cache: (station, pattern, proj, dt) -> PIL image
    _prev_cache     = {}   # key: ('start'|'end', station, pattern, proj, dt_iso) -> PIL image

    def _preview_draw(pil_start, pil_end, label_start, label_end):
        """Draw two thumbnails side-by-side on the canvas. Called from main thread."""
        cw = preview_canvas.winfo_width()  or 700
        ch = preview_canvas.winfo_height() or 220
        preview_canvas.delete("all")
        gap = 6
        tw = (cw - gap * 3) // 2
        th = ch - 20   # leave room for label

        def _fit(img, w, h):
            iw, ih = img.size
            scale = min(w / iw, h / ih)
            return img.resize((int(iw * scale), int(ih * scale)),
                               resample=getattr(__import__("PIL.Image", fromlist=["Image"]).Image,
                                               "LANCZOS", 1))

        from PIL import Image as _PILImage, ImageTk as _PILImageTk, ImageEnhance as _PILEnhance

        _sat = tl_saturation_var.get()

        imgs = []
        for i, (pil, lbl_txt, x0) in enumerate([
                (pil_start, label_start, gap),
                (pil_end,   label_end,   gap * 2 + tw)]):
            thumb = _fit(pil, tw, th)
            if abs(_sat - 1.0) > 0.01:
                thumb = _PILEnhance.Color(thumb).enhance(_sat)
            photo = _PILImageTk.PhotoImage(thumb)
            _prev_img_refs[i] = photo
            y0 = (ch - 20 - thumb.height) // 2
            preview_canvas.create_image(x0, y0, anchor="nw", image=photo)
            preview_canvas.create_text(x0 + tw // 2, ch - 10, text=lbl_txt,
                                        fill="#aaa", font=("", 8))

    def _preview_status(msg):
        root.after(0, preview_hdr_var.set, f"Preview  —  {msg}")

    def _fetch_and_stitch(start_dt, end_dt, station, pattern, proj, cancel_ev):
        """Background thread: fetch images and stitch previews."""
        import tempfile as _tmp
        tmp = _tmp.mkdtemp(prefix="stitcher_prev_")

        # Clean up previous temp dir
        old = _prev_tmpdir[0]
        _prev_tmpdir[0] = tmp
        if old and os.path.isdir(old):
            try: shutil.rmtree(old)
            except Exception: pass

        try:
            base = pattern.rstrip("/")

            def _img_glob_for_dt(dt):
                """Return glob for mini_MM.jpg (SD) — preview always uses SD regardless of quality setting."""
                return f"{base}/{dt.strftime('%Y%m%d')}/{dt.strftime('%H')}/mini_{dt.strftime('%M')}.jpg"

            def _pto_glob():
                """Glob for lens.pto inside each camN directory."""
                return f"{base}/lens.pto"   # e.g. /meteor/cam?/lens.pto  (? already in base pattern)

            start_glob = _img_glob_for_dt(start_dt)
            end_glob   = _img_glob_for_dt(end_dt)
            pto_glob   = _pto_glob()

            if cancel_ev.is_set(): return

            _preview_status("Fetching images…")

            def _local(rp):
                return os.path.join(tmp, rp.lstrip("/"))

            def _dedup_by_cam(paths):
                """One image per camera dir (camN), keeping the first match."""
                seen = {}
                for p in sorted(paths):
                    m = re.search(r'cam(\d+)', p)
                    key = m.group(1) if m else p
                    if key not in seen:
                        seen[key] = p
                return list(seen.values())

            if station:
                # Expand globs remotely (images + lens.pto files)
                script = (
                    f'compgen -G {shlex.quote(start_glob)} 2>/dev/null || true\n'
                    f'compgen -G {shlex.quote(end_glob)}   2>/dev/null || true\n'
                    f'compgen -G {shlex.quote(pto_glob)}   2>/dev/null || true\n'
                )
                r = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", station, "bash", "-c", shlex.quote(script)],
                    capture_output=True, text=True, timeout=30
                )
                remote_files = [l.strip() for l in r.stdout.splitlines() if l.strip()]
                if not any(f.endswith(".jpg") for f in remote_files):
                    _preview_status("No images found for this time range.")
                    return
                if cancel_ev.is_set(): return
                # Fetch via tar-over-ssh
                file_list = "\n".join(remote_files) + "\n"
                ssh_proc = subprocess.Popen(
                    ["ssh", "-o", "BatchMode=yes", station, "tar", "-chf", "-", "-T", "/dev/stdin"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                tar_proc = subprocess.Popen(
                    ["tar", "-xf", "-", "-C", tmp],
                    stdin=ssh_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                ssh_proc.stdout.close()
                ssh_proc.stdin.write(file_list.encode()); ssh_proc.stdin.close()
                tar_proc.communicate(timeout=60)
                ssh_proc.wait(timeout=10)

                start_bn = f"mini_{start_dt.strftime('%M')}.jpg"
                end_bn   = f"mini_{end_dt.strftime('%M')}.jpg"
                all_fetched = []
                for dirpath, _, fnames in os.walk(tmp):
                    for fn in fnames:
                        if fn.endswith(".jpg"):
                            all_fetched.append(os.path.join(dirpath, fn))
                start_imgs = _dedup_by_cam(
                    f for f in all_fetched
                    if os.path.basename(f) == start_bn
                    and f"/{start_dt.strftime('%Y%m%d')}/{start_dt.strftime('%H')}/" in f)
                end_imgs = _dedup_by_cam(
                    f for f in all_fetched
                    if os.path.basename(f) == end_bn
                    and f"/{end_dt.strftime('%Y%m%d')}/{end_dt.strftime('%H')}/" in f)
            else:
                # Local: glob images and also find lens.pto files
                start_imgs = _dedup_by_cam(glob.glob(start_glob))
                end_imgs   = _dedup_by_cam(glob.glob(end_glob))
                # lens.pto files are already on disk — no need to copy them

            if cancel_ev.is_set(): return

            # Canvas thumbnail size: half the panel width per image
            _cw = max(preview_canvas.winfo_width() or 700, 400)
            _thumb_w = (_cw - 18) // 2   # ~half panel, matches _preview_draw gap logic
            _thumb_h = int(_thumb_w * 9 / 16)  # 16:9 aspect for preview

            # No downscaling — SD mini images are already small (~640px)
            # and downscaling breaks timestamp erasure box coordinates

            _stitch_errors = []
            def _stitch_to_jpg(imgs, out_name, cache_key):
                # Return cached PIL image if inputs haven't changed
                if cache_key in _prev_cache:
                    return _prev_cache[cache_key]
                if not imgs:
                    _stitch_errors.append(f"{out_name}: no images found")
                    return None
                scaled_imgs = list(imgs)
                out = os.path.join(tmp, out_name)
                flag = "--fisheye" if proj == "fisheye" else "--equirect"
                cmd = [sys.executable, __file__, flag]
                _dv = tl_devignette_var.get()
                if abs(_dv) > 0.001:
                    cmd += ["--devignette", f"{_dv:.2f}"]
                cmd += scaled_imgs + [out]
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    if not os.path.isfile(out):
                        err = (res.stderr or res.stdout or "").strip()
                        snippet = " | ".join(err.splitlines()[-3:]) if err else "(no output)"
                        msg = f"[preview] {out_name}: {snippet}"
                        _stitch_errors.append(msg)
                        print(f"\n--- preview stitch failed: {out_name} ---", file=sys.stderr)
                        print(f"cmd: {' '.join(cmd)}", file=sys.stderr)
                        print(err or "(no output)", file=sys.stderr)
                        return None
                    return out  # caller loads from disk and caches
                except subprocess.TimeoutExpired:
                    msg = f"[preview] {out_name}: timed out"
                    _stitch_errors.append(msg)
                    print(msg, file=sys.stderr)
                    return None
                except Exception as e:
                    msg = f"[preview] {out_name}: {e}"
                    _stitch_errors.append(msg)
                    print(msg, file=sys.stderr)
                    return None

            from PIL import Image as _PILImage
            placeholder_w, placeholder_h = 400, 225
            _ctx = (station, pattern, proj, round(tl_devignette_var.get(), 2))
            _key_s = ('start', *_ctx, start_dt.isoformat())
            _key_e = ('end',   *_ctx, end_dt.isoformat())

            def _load_or_placeholder(path):
                if path and os.path.isfile(path):
                    try:
                        return _PILImage.open(path).copy()
                    except Exception:
                        pass
                return _PILImage.new("RGB", (placeholder_w, placeholder_h), (40, 40, 40))

            need_start = _key_s not in _prev_cache
            need_end   = _key_e not in _prev_cache

            if need_start or need_end:
                parts = []
                if need_start: parts.append("start")
                if need_end:   parts.append("end")
                _preview_status(f"Stitching {' & '.join(parts)} frame…")
                if cancel_ev.is_set(): return

            def _do_start():
                if not need_start:
                    return _prev_cache[_key_s]
                out = _stitch_to_jpg(start_imgs, "preview_start.jpg", _key_s)
                pil = _load_or_placeholder(out)
                if out and os.path.isfile(out):
                    _prev_cache[_key_s] = pil
                return pil

            def _do_end():
                if not need_end:
                    return _prev_cache[_key_e]
                out = _stitch_to_jpg(end_imgs, "preview_end.jpg", _key_e)
                pil = _load_or_placeholder(out)
                if out and os.path.isfile(out):
                    _prev_cache[_key_e] = pil
                return pil

            with ThreadPoolExecutor(max_workers=2) as _pool:
                _fut_s = _pool.submit(_do_start)
                _fut_e = _pool.submit(_do_end)
                pil_s = _fut_s.result()
                pil_e = _fut_e.result()

            # Evict cache beyond 10 entries
            while len(_prev_cache) > 10:
                _prev_cache.pop(next(iter(_prev_cache)))

            if cancel_ev.is_set(): return

            lbl_s = f"Start  {start_dt.strftime('%Y-%m-%d %H:%M')} UTC"
            lbl_e = f"End    {end_dt.strftime('%Y-%m-%d %H:%M')} UTC"

            root.after(0, _preview_draw, pil_s, pil_e, lbl_s, lbl_e)
            if _stitch_errors:
                first = _stitch_errors[0][:80]
                more = f"  (+{len(_stitch_errors)-1} more)" if len(_stitch_errors) > 1 else ""
                _preview_status(f"Stitch error: {first}{more}")
            else:
                _preview_status(f"Preview updated  ({start_dt.strftime('%H:%M')} – {end_dt.strftime('%H:%M')} UTC)")

        except Exception as exc:
            if not cancel_ev.is_set():
                import traceback as _tb
                _preview_status(f"Preview error: {exc}")
                print(f"\n[preview] unhandled error: {exc}", file=sys.stderr)
                _tb.print_exc(file=sys.stderr)

    def _schedule_preview(*_):
        """Debounce: cancel any pending fetch and schedule a new one in 700ms."""
        if _prev_job[0]:
            root.after_cancel(_prev_job[0])
        _prev_cancel[0].set()                    # abort in-flight thread
        _prev_cancel[0] = threading.Event()      # fresh cancel token

        def _launch():
            _prev_job[0] = None
            # Compute start/end datetimes
            try:
                start_dt = datetime.datetime(
                    tl_sy.get(), tl_sm.get(), tl_sd.get(),
                    tl_sh.get(), tl_smin.get(), 0,
                    tzinfo=datetime.timezone.utc)
                if tl_use_end_var.get():
                    end_dt = datetime.datetime(
                        tl_ey.get(), tl_em.get(), tl_ed.get(),
                        tl_eh.get(), tl_emin.get(), 0,
                        tzinfo=datetime.timezone.utc)
                else:
                    h, m = tl_dur_h_var.get(), tl_dur_m_var.get()
                    end_dt = start_dt + datetime.timedelta(hours=h, minutes=m)
            except (ValueError, tk.TclError):
                return

            station = tl_station_var.get().strip()
            pattern = tl_pattern_var.get().strip()
            proj    = tl_proj_var.get()
            cancel_ev = _prev_cancel[0]

            _preview_status("Fetching…")
            t = threading.Thread(
                target=_fetch_and_stitch,
                args=(start_dt, end_dt, station, pattern, proj, cancel_ev),
                daemon=True
            )
            _prev_thread[0] = t
            t.start()

        _prev_job[0] = root.after(700, _launch)

    # Wire preview refresh to all relevant vars
    for _v in (tl_sy, tl_sm, tl_sd, tl_sh, tl_smin,
               tl_ey, tl_em, tl_ed, tl_eh, tl_emin,
               tl_use_end_var, tl_dur_h_var, tl_dur_m_var,
               tl_station_var, tl_pattern_var, tl_proj_var,
               tl_saturation_var, tl_devignette_var):
        _v.trace_add("write", _schedule_preview)
    # Also redraw when canvas is resized
    preview_canvas.bind("<Configure>", _schedule_preview)

    # ═══════════════════════════════════════════════════════════════
    # TAB 2 – Stitch single image / video
    # ═══════════════════════════════════════════════════════════════
    tab_stitch = ttk.Frame(nb)
    nb.add(tab_stitch, text="  Stitch Image / Video  ")
    tab_stitch.columnconfigure(1, weight=1)
    r = 0

    st_mode_var    = tk.StringVar(value="video")   # image | video
    st_proj_var    = tk.StringVar(value="fisheye")  # fisheye | equirect | custom
    st_pto_var     = tk.StringVar()
    st_inputs_var  = tk.StringVar()
    st_output_var  = tk.StringVar()
    st_station_var = tk.StringVar()        # SSH hostname passed to --station
    st_station_label_var = tk.StringVar()  # dropdown display label
    st_cam_pattern_var = tk.StringVar(value=tl_pattern_var.get())  # e.g. /meteor/cam?
    st_file_type_var = tk.StringVar(value="full")  # full | mini | image
    st_enhance_var = tk.BooleanVar()
    st_ts_var      = tk.BooleanVar()
    st_sync_var    = tk.BooleanVar()
    st_model_var   = tk.StringVar()
    st_crf_var     = tk.StringVar(value="28")
    st_preset_var  = tk.StringVar(value="ultrafast")
    st_maxfr_var   = tk.IntVar(value=0)
    st_saturation_var = tk.DoubleVar(value=1.0)
    # Single input timestamp — default to last midnight UTC
    st_dy = tk.IntVar(value=_midnight.year)
    st_dm = tk.IntVar(value=_midnight.month)
    st_dd = tk.IntVar(value=_midnight.day)
    st_dh = tk.IntVar(value=0)
    st_dmin = tk.IntVar(value=0)

    ttk.Label(tab_stitch, text="Mode", font=("",10,"bold")).grid(
        column=0, row=r, columnspan=3, sticky="w", padx=5, pady=(10,2)); r+=1
    mf2 = ttk.Frame(tab_stitch)
    mf2.grid(column=0, row=r, columnspan=3, sticky="w", padx=5, pady=3)
    ttk.Radiobutton(mf2, text="Stitch Images", variable=st_mode_var, value="image").pack(side="left", padx=6)
    ttk.Radiobutton(mf2, text="Stitch Video",  variable=st_mode_var, value="video").pack(side="left", padx=6)
    r+=1

    ttk.Separator(tab_stitch, orient="horizontal").grid(
        column=0, row=r, columnspan=3, sticky="ew", pady=5); r+=1

    ttk.Label(tab_stitch, text="Projection", font=("",10,"bold")).grid(
        column=0, row=r, columnspan=3, sticky="w", padx=5, pady=(0,2)); r+=1
    pf2 = ttk.Frame(tab_stitch)
    pf2.grid(column=0, row=r, columnspan=3, sticky="w", padx=5, pady=3)
    ttk.Radiobutton(pf2, text="Fisheye", variable=st_proj_var, value="fisheye").pack(side="left", padx=6)
    ttk.Radiobutton(pf2, text="Equirectangular", variable=st_proj_var, value="equirect").pack(side="left", padx=6)
    ttk.Radiobutton(pf2, text="Custom PTO", variable=st_proj_var, value="custom").pack(side="left", padx=6)
    r+=1

    pto_lbl2 = ttk.Label(tab_stitch, text="PTO file:")
    pto_lbl2.grid(column=0, row=r, sticky="w", padx=5, pady=3)
    pto_ent2 = ent(tab_stitch, st_pto_var, 1, r)
    pto_btn2 = ttk.Button(tab_stitch, text="Browse…",
        command=lambda: browseopen(st_pto_var, filetypes=[("PTO","*.pto"),("All","*.*")]))
    pto_btn2.grid(column=2, row=r, padx=5, pady=3); r+=1

    _ST_DEFAULTS = {
        ("image", "fisheye"):  "fisheye.jpg",
        ("image", "equirect"): "equirect.jpg",
        ("image", "custom"):   "custom.jpg",
        ("video", "fisheye"):  "fisheye.mp4",
        ("video", "equirect"): "equirect.mp4",
        ("video", "custom"):   "custom.mp4",
    }
    def _st_default_output(*_):
        cur = st_output_var.get()
        if cur in _ST_DEFAULTS.values() or cur == "":
            st_output_var.set(_ST_DEFAULTS.get((st_mode_var.get(), st_proj_var.get()), "fisheye.jpg"))
    def _on_proj2(*_):
        s = "normal" if st_proj_var.get() == "custom" else "disabled"
        pto_ent2.config(state=s); pto_btn2.config(state=s)
        _st_default_output()
    st_proj_var.trace_add("write", _on_proj2); _on_proj2()
    st_mode_var.trace_add("write", _st_default_output)
    _st_default_output()   # set initial default

    ttk.Separator(tab_stitch, orient="horizontal").grid(
        column=0, row=r, columnspan=3, sticky="ew", pady=5); r+=1

    ttk.Label(tab_stitch, text="Source", font=("",10,"bold")).grid(
        column=0, row=r, columnspan=3, sticky="w", padx=5, pady=(0,2)); r+=1

    lbl(tab_stitch, "Station:", 0, r)
    _st_station_labels = list(_station_map.keys())
    if _st_station_labels:
        st_station_combo = ttk.Combobox(tab_stitch, textvariable=st_station_label_var,
                                        values=_st_station_labels, state="readonly", width=42)
        st_station_combo.grid(column=1, row=r, columnspan=2, sticky="w", padx=5, pady=3)
        st_station_combo.current(0)
        def _on_st_station_select(*_):
            st_station_var.set(_station_map.get(st_station_label_var.get(), ""))
        st_station_label_var.trace_add("write", _on_st_station_select)
        _on_st_station_select()
    else:
        ent(tab_stitch, st_station_var, 1, r, width=30)
        ttk.Label(tab_stitch, text="SSH hostname", foreground="#888", font=("",8)).grid(
            column=2, row=r, sticky="w", padx=4)
    r+=1

    lbl(tab_stitch, "Camera pattern:", 0, r)
    ent(tab_stitch, st_cam_pattern_var, 1, r, width=22)
    ttk.Label(tab_stitch, text="glob, e.g. /meteor/cam?", foreground="#888", font=("",8)).grid(
        column=2, row=r, sticky="w", padx=4); r+=1

    lbl(tab_stitch, "File type:", 0, r)
    ttk.Combobox(tab_stitch, textvariable=st_file_type_var, width=10,
        values=["full", "mini", "image"], state="readonly").grid(
        column=1, row=r, sticky="w", padx=5, pady=3)
    ttk.Label(tab_stitch, text="full=full_mm.mp4, mini=mini_mm.mp4, image=*.jpg", foreground="#888", font=("",8)).grid(
        column=2, row=r, sticky="w", padx=4); r+=1

    # Local files row — shown only when no station is selected
    _st_local_lbl_widget = ttk.Label(tab_stitch, text="Local input files:")
    _st_local_lbl_widget.grid(column=0, row=r, sticky="w", padx=5, pady=3)
    _st_local_ent = ent(tab_stitch, st_inputs_var, 1, r)
    _st_local_btn = ttk.Button(tab_stitch, text="Browse…",
        command=lambda: browseopen(st_inputs_var, multiple=True,
            filetypes=[("Images/Videos","*.jpg *.jpeg *.png *.mp4 *.mov *.avi *.mkv"),("All","*.*")]))
    _st_local_btn.grid(column=2, row=r, padx=5, pady=3); r+=1

    def _refresh_st_source_rows(*_):
        has_station = bool(st_station_var.get().strip())
        for w in (_st_local_lbl_widget, _st_local_ent, _st_local_btn):
            w.grid_remove() if has_station else w.grid()

    st_station_var.trace_add("write", _refresh_st_source_rows)
    _refresh_st_source_rows()

    lbl(tab_stitch, "Output file:", 0, r)
    ent(tab_stitch, st_output_var, 1, r)
    ttk.Button(tab_stitch, text="Browse…",
        command=lambda: browsesave(st_output_var,
            [("JPEG","*.jpg"),("MP4","*.mp4"),("All","*.*")])).grid(
        column=2, row=r, padx=5, pady=3); r+=1

    ttk.Separator(tab_stitch, orient="horizontal").grid(
        column=0, row=r, columnspan=3, sticky="ew", pady=5); r+=1

    ttk.Checkbutton(tab_stitch, text="Enhance (noise reduction)", variable=st_enhance_var).grid(
        column=1, row=r, sticky="w", padx=5); r+=1
    ttk.Checkbutton(tab_stitch, text="Overlay UTC timestamp", variable=st_ts_var).grid(
        column=1, row=r, sticky="w", padx=5); r+=1

    # Video-only options (hidden for image mode)
    _st_sync_chk = ttk.Checkbutton(tab_stitch, text="Synchronize video streams (--sync)", variable=st_sync_var)
    _st_sync_chk.grid(column=1, row=r, sticky="w", padx=5); r+=1

    # Load model names from timestamp.py if available
    _ts_model_path = os.path.join(os.path.dirname(__file__), 'timestamp.py')
    _sync_models = []
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("_timestamp_mod", _ts_model_path)
        _ts_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_ts_mod)
        _sync_models = list(_ts_mod.FONT_DATABASE.keys())
    except Exception:
        _sync_models = ["IP8172", "IP9171", "IP8151", "IP816A",
                        "IMX291SD", "IMX291HD", "IMX307SD",
                        "IMX307HD_24x36", "IMX307HD_16x24"]
    _sync_model_values = ["(auto-detect)"] + _sync_models

    _sync_lbl_w = ttk.Label(tab_stitch, text="Sync model:")
    _sync_lbl_w.grid(column=0, row=r, sticky="w", padx=5, pady=3)
    _sync_combo = ttk.Combobox(tab_stitch, textvariable=st_model_var,
                               values=_sync_model_values, state="readonly", width=22)
    _sync_combo.grid(column=1, row=r, sticky="w", padx=5, pady=3)
    _sync_combo.current(0)
    def _on_sync_model(*_):
        v = st_model_var.get()
        if v == "(auto-detect)":
            st_model_var.set("")
    st_model_var.trace_add("write", _on_sync_model)
    r += 1

    _st_crf_lbl = ttk.Label(tab_stitch, text="CRF (quality):")
    _st_crf_lbl.grid(column=0, row=r, sticky="w", padx=5, pady=3)
    _st_crf_frame = ttk.Frame(tab_stitch)
    _st_crf_frame.grid(column=1, row=r, sticky="w", padx=5, pady=3)
    ttk.Spinbox(_st_crf_frame, textvariable=st_crf_var, from_=0, to=51, width=5).pack(side="left")
    ttk.Label(_st_crf_frame, text=" (0=lossless, 28=default, 51=worst)", foreground="#888", font=("",8)).pack(side="left")
    r+=1

    _st_preset_lbl = ttk.Label(tab_stitch, text="Preset:")
    _st_preset_lbl.grid(column=0, row=r, sticky="w", padx=5, pady=3)
    _st_preset_combo = ttk.Combobox(tab_stitch, textvariable=st_preset_var, width=14,
        values=["ultrafast","superfast","veryfast","faster","fast","medium","slow","veryslow"])
    _st_preset_combo.grid(column=1, row=r, sticky="w", padx=5, pady=3); r+=1

    _st_maxfr_lbl = ttk.Label(tab_stitch, text="Max frames (0=all):")
    _st_maxfr_lbl.grid(column=0, row=r, sticky="w", padx=5, pady=3)
    _st_maxfr_sb = ttk.Spinbox(tab_stitch, textvariable=st_maxfr_var, from_=0, to=999999, width=8)
    _st_maxfr_sb.grid(column=1, row=r, sticky="w", padx=5, pady=3); r+=1

    lbl(tab_stitch, "Saturation:", 0, r)
    _st_sat_frame = ttk.Frame(tab_stitch)
    _st_sat_frame.grid(column=1, row=r, columnspan=2, sticky="ew", padx=5, pady=3)
    _st_sat_lbl = ttk.Label(_st_sat_frame, text="1.0", width=4)
    _st_sat_lbl.pack(side="right")
    def _st_sat_update(val):
        _st_sat_lbl.config(text=f"{float(val):.1f}")
    tk.Scale(_st_sat_frame, variable=st_saturation_var, from_=0.0, to=3.0, resolution=0.1,
             orient="horizontal", showvalue=False, command=_st_sat_update).pack(side="left", fill="x", expand=True)
    r+=1

    st_devignette_var = tk.DoubleVar(value=-0.20)
    lbl(tab_stitch, "Devignette:", 0, r)
    _st_dv_frame = ttk.Frame(tab_stitch)
    _st_dv_frame.grid(column=1, row=r, columnspan=2, sticky="ew", padx=5, pady=3)
    _st_dv_lbl = ttk.Label(_st_dv_frame, text="-0.20", width=5)
    _st_dv_lbl.pack(side="right")
    def _st_dv_update(val):
        _st_dv_lbl.config(text=f"{float(val):.2f}")
    tk.Scale(_st_dv_frame, variable=st_devignette_var, from_=0.0, to=-0.5, resolution=0.01,
             orient="horizontal", showvalue=False, command=_st_dv_update).pack(side="left", fill="x", expand=True)
    r+=1

    lbl(tab_stitch, "Timestamp (UTC):", 0, r)
    _dt_spinboxes(tab_stitch, st_dy, st_dm, st_dd, st_dh, st_dmin).grid(
        column=1, row=r, columnspan=2, sticky="w", padx=5, pady=3); r+=1

    # ── Video-only visibility control ──
    _st_video_widgets = [_st_sync_chk, _sync_lbl_w, _sync_combo,
                         _st_crf_lbl, _st_crf_frame,
                         _st_preset_lbl, _st_preset_combo,
                         _st_maxfr_lbl, _st_maxfr_sb]

    def _st_update_video_opts(*_):
        is_video = st_mode_var.get() == "video"
        st = "!disabled" if is_video else "disabled"
        for w in _st_video_widgets:
            try:
                w.state([st])
            except AttributeError:
                w.configure(state="normal" if is_video else "disabled")
        # Sync model: only enabled when video AND sync is checked
        sync_st = "!disabled" if (is_video and st_sync_var.get()) else "disabled"
        try:
            _sync_lbl_w.state([sync_st]); _sync_combo.state([sync_st])
        except AttributeError:
            s = "normal" if (is_video and st_sync_var.get()) else "disabled"
            _sync_lbl_w.configure(state=s); _sync_combo.configure(state=s)

    st_mode_var.trace_add("write", _st_update_video_opts)
    st_sync_var.trace_add("write", _st_update_video_opts)
    _st_update_video_opts()

    # ── Stitch Preview ──────────────────────────────────────────────────────
    ttk.Separator(tab_stitch, orient="horizontal").grid(
        column=0, row=r, columnspan=3, sticky="ew", pady=(8, 2)); r += 1
    st_preview_hdr_var = tk.StringVar(value="Preview  (auto-updates)")
    ttk.Label(tab_stitch, textvariable=st_preview_hdr_var, font=("", 9, "bold"),
              foreground="#555").grid(column=0, row=r, columnspan=3, sticky="w", padx=5); r += 1

    tab_stitch.rowconfigure(r, weight=1)
    st_preview_canvas = tk.Canvas(tab_stitch, bg="#111", height=220)
    st_preview_canvas.grid(column=0, row=r, columnspan=3, sticky="nsew", padx=5, pady=(2, 5))

    _st_prev_img_ref = [None]
    _st_prev_job     = [None]
    _st_prev_thread  = [None]
    _st_prev_tmpdir  = [None]
    _st_prev_cancel  = [threading.Event()]
    _st_prev_cache   = {}

    def _st_preview_draw(pil_img, label_txt):
        cw = st_preview_canvas.winfo_width()
        if cw < 10:  # tab not yet displayed
            cw = max(root.winfo_width() - 30, 400)
        ch = st_preview_canvas.winfo_height()
        if ch < 10:
            ch = 220
        st_preview_canvas.delete("all")
        from PIL import Image as _PILImage, ImageTk as _PILImageTk, ImageEnhance as _PILEnhance
        iw, ih = pil_img.size
        if iw < 1 or ih < 1:
            return
        tw, th = max(cw - 12, 1), max(ch - 20, 1)
        scale = min(tw / iw, th / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        thumb = pil_img.resize((nw, nh), resample=getattr(_PILImage, "LANCZOS", 1))
        _sat = st_saturation_var.get()
        if abs(_sat - 1.0) > 0.01:
            thumb = _PILEnhance.Color(thumb).enhance(_sat)
        photo = _PILImageTk.PhotoImage(thumb)
        _st_prev_img_ref[0] = photo
        x0 = (cw - thumb.width) // 2
        y0 = (ch - 20 - thumb.height) // 2
        st_preview_canvas.create_image(x0, y0, anchor="nw", image=photo)
        st_preview_canvas.create_text(cw // 2, ch - 10, text=label_txt,
                                       fill="#aaa", font=("", 8))

    def _st_preview_status(msg):
        root.after(0, st_preview_hdr_var.set, f"Preview  —  {msg}")

    def _st_fetch_and_stitch(dt, station, pattern, proj, cancel_ev):
        import tempfile as _tmp
        tmp = _tmp.mkdtemp(prefix="stitcher_stprev_")
        old = _st_prev_tmpdir[0]
        _st_prev_tmpdir[0] = tmp
        if old and os.path.isdir(old):
            try: shutil.rmtree(old)
            except Exception: pass

        try:
            base = pattern.rstrip("/")
            img_glob = f"{base}/{dt.strftime('%Y%m%d')}/{dt.strftime('%H')}/mini_{dt.strftime('%M')}.jpg"
            pto_glob = f"{base}/lens.pto"

            if cancel_ev.is_set(): return
            _st_preview_status("Fetching images…")

            def _dedup_by_cam(paths):
                seen = {}
                for p in sorted(paths):
                    m = re.search(r'cam(\d+)', p)
                    key = m.group(1) if m else p
                    if key not in seen:
                        seen[key] = p
                return list(seen.values())

            if station:
                script = (
                    f'compgen -G {shlex.quote(img_glob)} 2>/dev/null || true\n'
                    f'compgen -G {shlex.quote(pto_glob)} 2>/dev/null || true\n'
                )
                r_result = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", station, "bash", "-c", shlex.quote(script)],
                    capture_output=True, text=True, timeout=30
                )
                remote_files = [l.strip() for l in r_result.stdout.splitlines() if l.strip()]
                if not any(f.endswith(".jpg") for f in remote_files):
                    _st_preview_status("No images found for this timestamp.")
                    return
                if cancel_ev.is_set(): return
                file_list = "\n".join(remote_files) + "\n"
                ssh_proc = subprocess.Popen(
                    ["ssh", "-o", "BatchMode=yes", station, "tar", "-chf", "-", "-T", "/dev/stdin"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                tar_proc = subprocess.Popen(
                    ["tar", "-xf", "-", "-C", tmp],
                    stdin=ssh_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                ssh_proc.stdout.close()
                ssh_proc.stdin.write(file_list.encode()); ssh_proc.stdin.close()
                tar_proc.communicate(timeout=60)
                ssh_proc.wait(timeout=10)

                bn = f"mini_{dt.strftime('%M')}.jpg"
                all_fetched = []
                for dirpath, _, fnames in os.walk(tmp):
                    for fn in fnames:
                        if fn.endswith(".jpg"):
                            all_fetched.append(os.path.join(dirpath, fn))
                imgs = _dedup_by_cam(
                    f for f in all_fetched
                    if os.path.basename(f) == bn
                    and f"/{dt.strftime('%Y%m%d')}/{dt.strftime('%H')}/" in f)
            else:
                imgs = _dedup_by_cam(glob.glob(img_glob))

            if cancel_ev.is_set(): return
            if not imgs:
                _st_preview_status("No images found for this timestamp.")
                return

            _st_preview_status("Stitching preview…")

            # No downscaling — SD mini images are already small (~640px)
            # and downscaling breaks timestamp erasure box coordinates
            scaled_imgs = list(imgs)

            if cancel_ev.is_set(): return

            out = os.path.join(tmp, "preview_stitch.jpg")
            flag = "--fisheye" if proj == "fisheye" else "--equirect"
            cmd = [sys.executable, __file__, flag]
            _dv = st_devignette_var.get()
            if abs(_dv) > 0.001:
                cmd += ["--devignette", f"{_dv:.2f}"]
            cmd += scaled_imgs + [out]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                _st_preview_status("Preview stitch timed out.")
                return

            if not os.path.isfile(out):
                err = (res.stderr or res.stdout or "").strip()
                snippet = " | ".join(err.splitlines()[-2:]) if err else "(no output)"
                _st_preview_status(f"Stitch error: {snippet[:80]}")
                return

            from PIL import Image as _I
            pil = _I.open(out).copy()
            lbl_txt = f"{dt.strftime('%Y-%m-%d %H:%M')} UTC  —  {proj}"
            root.after(0, _st_preview_draw, pil, lbl_txt)
            _st_preview_status(f"Preview updated  ({dt.strftime('%Y-%m-%d %H:%M')} UTC)")

        except Exception as exc:
            if not cancel_ev.is_set():
                _st_preview_status(f"Preview error: {exc}")
                import traceback as _tb
                _tb.print_exc(file=sys.stderr)

    def _st_schedule_preview(*_):
        if _st_prev_job[0]:
            root.after_cancel(_st_prev_job[0])
        _st_prev_cancel[0].set()
        _st_prev_cancel[0] = threading.Event()

        def _launch():
            _st_prev_job[0] = None
            try:
                dt = datetime.datetime(
                    st_dy.get(), st_dm.get(), st_dd.get(),
                    st_dh.get(), st_dmin.get(), 0,
                    tzinfo=datetime.timezone.utc)
            except (ValueError, tk.TclError):
                return
            station = st_station_var.get().strip()
            pattern = st_cam_pattern_var.get().strip()
            proj    = st_proj_var.get()
            if proj == "custom":
                proj = "fisheye"  # preview always uses fisheye or equirect
            cancel_ev = _st_prev_cancel[0]
            _st_preview_status("Fetching…")
            t = threading.Thread(
                target=_st_fetch_and_stitch,
                args=(dt, station, pattern, proj, cancel_ev),
                daemon=True
            )
            _st_prev_thread[0] = t
            t.start()

        _st_prev_job[0] = root.after(700, _launch)

    for _v in (st_dy, st_dm, st_dd, st_dh, st_dmin,
               st_station_var, st_cam_pattern_var, st_proj_var,
               st_saturation_var, st_devignette_var):
        _v.trace_add("write", _st_schedule_preview)
    # Trigger initial preview with default station
    _st_schedule_preview()

    # ═══════════════════════════════════════════════════════════════
    # TAB 3 – Log
    # ═══════════════════════════════════════════════════════════════
    tab_log = ttk.Frame(nb)
    nb.add(tab_log, text="  Log  ")
    tab_log.rowconfigure(0, weight=1)
    tab_log.columnconfigure(0, weight=1)
    log_text = scrolledtext.ScrolledText(tab_log, state="disabled", wrap="word",
                                         font=("Courier", 9), bg="#1e1e1e", fg="#d4d4d4",
                                         insertbackground="white")
    log_text.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    _last_was_cr = [False]

    def log(msg):
        log_text.config(state="normal")
        if msg.startswith("\r"):
            if _last_was_cr[0]:
                # Replace the previous progress line in-place
                log_text.delete("end-2l", "end-1c")
            log_text.insert("end", msg.lstrip("\r") + "\n")
            _last_was_cr[0] = True
        else:
            log_text.insert("end", msg + "\n")
            _last_was_cr[0] = False
        log_text.see("end")
        log_text.config(state="disabled")

    # ── Progress bar + status bar ─────────────────────────────────────────────
    progress_frame = ttk.Frame(root)
    progress_frame.pack(fill="x", padx=8, pady=(0,2))
    progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate", maximum=100)
    progress_bar.pack(fill="x", side="top", pady=(0,2))
    status_var = tk.StringVar(value="Ready")
    ttk.Label(progress_frame, textvariable=status_var, anchor="w", font=("",9)).pack(fill="x", side="top")

    def set_progress(pct, msg=""):
        progress_bar["value"] = pct
        if msg:
            status_var.set(msg)

    # ── Bottom button bar ─────────────────────────────────────────────────────
    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", padx=8, pady=(2, 8))

    run_btn = [None]
    open_btn = [None]
    _last_output = [None]   # path to last successfully produced output file

    def _open_output():
        path = _last_output[0]
        if path and os.path.isfile(path):
            try:
                subprocess.Popen(["xdg-open", path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                log(f"Could not open file: {e}")

    def _build_argv_timelapse():
        argv = []
        proj = tl_proj_var.get()
        argv.append("--fisheye" if proj == "fisheye" else "--equirect")
        argv.append("--timelapse")
        argv += ["--timelapse-start", _tl_start_str()]
        if tl_use_end_var.get():
            argv += ["--timelapse-end", _tl_end_str()]
        else:
            h, m = tl_dur_h_var.get(), tl_dur_m_var.get()
            if h == 0 and m == 0:
                raise ValueError("Specify a non-zero duration or switch to End time.")
            dur = []
            if h: dur.append(f"{h} hours")
            if m: dur.append(f"{m} minutes")
            argv += ["--timelapse-duration", " ".join(dur)]
        if tl_speed_var.get().strip() != "60":
            argv += ["--timelapse-speed", tl_speed_var.get().strip()]
        if tl_fps_var.get() != 30:
            argv += ["--timelapse-framerate", str(tl_fps_var.get())]
        if tl_quality_var.get() != "sd":
            argv += ["--timelapse-quality", tl_quality_var.get()]
        if tl_pattern_var.get().strip() and tl_pattern_var.get().strip() != "/meteor/cam?":
            argv += ["--timelapse-pattern", tl_pattern_var.get().strip()]
        if tl_station_var.get().strip():
            argv += ["--station", tl_station_var.get().strip()]
        if tl_pto_var.get().strip():
            argv += ["--pto", tl_pto_var.get().strip()]
        if tl_enhance_var.get():   argv.append("--enhance")
        if tl_timestamp_var.get(): argv.append("--timestamp")
        if tl_crf_var.get().strip() != "28":
            argv += ["--crf", tl_crf_var.get().strip()]
        if tl_preset_var.get().strip() != "ultrafast":
            argv += ["--preset", tl_preset_var.get().strip()]
        sat = tl_saturation_var.get()
        if abs(sat - 1.0) > 0.01:
            argv += ["--saturation", f"{sat:.2f}"]
        dv = tl_devignette_var.get()
        if abs(dv) > 0.001:
            argv += ["--devignette", f"{dv:.2f}"]
        out = tl_output_var.get().strip()
        if not out:
            raise ValueError("Output file is required.")
        argv.append(out)
        return argv

    def _build_argv_stitch():
        argv = []
        proj = st_proj_var.get()
        if proj == "fisheye":   argv.append("--fisheye")
        elif proj == "equirect": argv.append("--equirect")
        if st_enhance_var.get():  argv.append("--enhance")
        if st_ts_var.get():       argv.append("--timestamp")
        if st_sync_var.get():
            argv.append("--sync")
            if st_model_var.get().strip():
                argv += ["--model", st_model_var.get().strip()]
        if st_maxfr_var.get() > 0:
            argv += ["-n", str(st_maxfr_var.get())]
        if st_crf_var.get().strip() != "28":
            argv += ["--crf", st_crf_var.get().strip()]
        if st_preset_var.get().strip() != "ultrafast":
            argv += ["--preset", st_preset_var.get().strip()]
        sat = st_saturation_var.get()
        if abs(sat - 1.0) > 0.01:
            argv += ["--saturation", f"{sat:.2f}"]
        dv = st_devignette_var.get()
        if abs(dv) > 0.001:
            argv += ["--devignette", f"{dv:.2f}"]
        station = st_station_var.get().strip()
        if station:
            argv += ["--station", station]
            cam_pat = st_cam_pattern_var.get().strip()
            file_type = st_file_type_var.get().strip()
            if not cam_pat: raise ValueError("Camera pattern is required.")
            if not file_type: raise ValueError("File type is required.")
            # Build datetime-structured path: /meteor/cam?/YYYYMMDD/HH/full_MM.mp4
            y, m, d, h, min_ = st_dy.get(), st_dm.get(), st_dd.get(), st_dh.get(), st_dmin.get()
            date_dir = f"{y:04d}{m:02d}{d:02d}"
            hour_dir = f"{h:02d}"
            if file_type == "full":
                file_pat = f"full_{min_:02d}.mp4"
            elif file_type == "mini":
                file_pat = f"mini_{min_:02d}.mp4"
            else:  # image
                file_pat = "*.jpg"
            argv.append(f"{cam_pat}/{date_dir}/{hour_dir}/{file_pat}")
        else:
            argv += ["--input-datetime",
                     f"{st_dy.get():04d}-{st_dm.get():02d}-{st_dd.get():02d} "
                     f"{st_dh.get():02d}:{st_dmin.get():02d}:00"]
            raw = st_inputs_var.get().strip()
            if not raw: raise ValueError("Input files are required.")
            argv += shlex.split(raw)
        if proj == "custom":
            pto = st_pto_var.get().strip()
            if not pto: raise ValueError("PTO file is required for custom projection.")
            argv.append(pto)
        out = st_output_var.get().strip()
        if not out: raise ValueError("Output file is required.")
        argv.append(out)
        return argv

    def _run(argv):
        """Spawn stitcher subprocess and stream output. Runs in a daemon thread."""
        nb.select(tab_log)
        log("─" * 60)
        log("$ " + " ".join([os.path.basename(sys.executable), __file__] + argv))
        log("─" * 60)
        running.set()
        run_btn[0].config(state="disabled", text="⏳  Running…")
        root.after(0, set_progress, 0, "Starting…")
        _run_start[0] = __import__("time").time()

        def worker():
            cmd = [sys.executable, "-u", __file__] + argv
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    # Binary mode so \r is never eaten by Python's text decoder
                )
                cancel_proc[0] = proc

                def drain(stream, is_err):
                    buf = b""
                    while True:
                        chunk = stream.read(256)
                        if not chunk:
                            break
                        buf += chunk
                        # Split on \n and \r, preserving which delimiter was used
                        while True:
                            ni = buf.find(b'\n')
                            ri = buf.find(b'\r')
                            if ni == -1 and ri == -1:
                                break
                            if ri != -1 and (ni == -1 or ri < ni):
                                # \r-terminated: in-place overwrite signal
                                raw = buf[:ri]
                                buf = buf[ri+1:]
                                line = '\r' + raw.decode('utf-8', errors='replace').strip()
                            else:
                                raw = buf[:ni]
                                buf = buf[ni+1:]
                                line = raw.decode('utf-8', errors='replace').rstrip('\r')
                            if not line.strip('\r'):
                                continue
                            if is_err and line.lstrip('\r').startswith("PROGRESS:"):
                                try:
                                    pct = float(line.lstrip('\r').split(":", 1)[1])
                                    elapsed = __import__("time").time() - _run_start[0]
                                    if pct > 0:
                                        eta_s = elapsed / (pct / 100) - elapsed
                                        eta_str = f"ETA {int(eta_s//60)}m {int(eta_s%60):02d}s"
                                    else:
                                        eta_str = "ETA …"
                                    msg = f"Stitching… {pct:.1f}%  |  {eta_str}"
                                    root.after(0, set_progress, pct, msg)
                                except ValueError:
                                    pass
                            else:
                                root.after(0, log, line)

                t_out = threading.Thread(target=drain, args=(proc.stdout, False), daemon=True)
                t_err = threading.Thread(target=drain, args=(proc.stderr, True),  daemon=True)
                t_out.start(); t_err.start()
                t_out.join(); t_err.join()
                proc.wait()
                if proc.returncode == 0:
                    root.after(0, set_progress, 100, "Done.")
                    root.after(0, log, "\n✅  Done.")
                    if _last_output[0] and os.path.isfile(_last_output[0]):
                        root.after(0, lambda: open_btn[0].config(state="normal"))
                else:
                    root.after(0, set_progress, 0, f"Failed (exit {proc.returncode}).")
                    root.after(0, log, f"\n❌  Exited with code {proc.returncode}.")
            except Exception as e:
                root.after(0, log, f"❌  Failed to start: {e}")
                root.after(0, set_progress, 0, "Error.")
            finally:
                cancel_proc[0] = None
                running.clear()
                root.after(0, lambda: run_btn[0].config(state="normal", text="▶  Run"))

        threading.Thread(target=worker, daemon=True).start()

    def run_current():
        if running.is_set():
            return
        open_btn[0].config(state="disabled")
        tab_idx = nb.index(nb.select())
        try:
            if tab_idx == 0:    # Timelapse tab
                argv = _build_argv_timelapse()
                _last_output[0] = os.path.abspath(tl_output_var.get().strip())
            elif tab_idx == 1:  # Stitch tab
                argv = _build_argv_stitch()
                _last_output[0] = os.path.abspath(st_output_var.get().strip())
            else:
                messagebox.showinfo("Info", "Switch to the Timelapse or Stitch tab first.")
                return
        except ValueError as e:
            messagebox.showerror("Missing input", str(e))
            return
        _run(argv)

    def cancel_current():
        proc = cancel_proc[0]
        if proc:
            import signal as _signal
            try:
                proc.send_signal(_signal.SIGINT)   # raises KeyboardInterrupt in subprocess → triggers finally cleanup
            except OSError:
                proc.terminate()
            log("⚠  Cancelling — waiting for cleanup…")
            set_progress(0, "Cancelling…")
            def _wait_then_kill():
                try:
                    proc.wait(timeout=15)
                except Exception:
                    proc.kill()
                root.after(0, log, "⚠  Cancelled.")
                root.after(0, set_progress, 0, "Cancelled.")
            threading.Thread(target=_wait_then_kill, daemon=True).start()

    def quit_app():
        if running.is_set():
            if not messagebox.askyesno("Quit", "A job is still running.\nCancel it and quit?",
                                       icon="warning", default="no"):
                return
            cancel_current()
        else:
            if not messagebox.askyesno("Quit", "Quit Stitcher?", default="yes"):
                return
        root.destroy()

    run_btn[0] = ttk.Button(btn_frame, text="▶  Run", command=run_current)
    run_btn[0].pack(side="left", padx=4)
    ttk.Button(btn_frame, text="✕  Cancel", command=cancel_current).pack(side="left", padx=4)
    open_btn[0] = ttk.Button(btn_frame, text="📂  Open", command=_open_output, state="disabled")
    open_btn[0].pack(side="left", padx=4)
    ttk.Button(btn_frame, text="Quit", command=quit_app).pack(side="right", padx=4)

    root.protocol("WM_DELETE_WINDOW", quit_app)

    # Live command preview — full-width frame below buttons
    cmd_frame = ttk.Frame(root)
    cmd_frame.pack(fill="x", padx=8, pady=(0, 4))
    cmd_preview_var = tk.StringVar(value="")
    ttk.Label(cmd_frame, textvariable=cmd_preview_var, foreground="#000",
              font=("Courier", 8), wraplength=780, justify="left", anchor="w").pack(
        side="left", fill="x", expand=True)

    def _copy_cmd():
        txt = cmd_preview_var.get()
        if txt:
            root.clipboard_clear()
            root.clipboard_append(txt)
            # Also set PRIMARY selection for middle-click paste on X11
            try:
                root.selection_clear()
                root.selection_handle(lambda offset, length: txt[int(offset):int(offset)+int(length)])
                root.selection_own()
            except Exception:
                pass
    ttk.Button(cmd_frame, text="Copy", width=5, command=_copy_cmd).pack(side="right", padx=(4, 0))

    def _update_preview(*_):
        try:
            idx = nb.index(nb.select())
        except Exception:
            return
        if idx in (0, 1):
            if not cmd_frame.winfo_ismapped():
                cmd_frame.pack(fill="x", padx=8, pady=(0, 4))
            try:
                argv = _build_argv_timelapse() if idx == 0 else _build_argv_stitch()
                cmd_preview_var.set(__file__ + " " + " ".join(
                    f'"{a}"' if " " in a else a for a in argv))
            except Exception:
                cmd_preview_var.set("")
        else:
            cmd_frame.pack_forget()
            cmd_preview_var.set("")

    for v in (tl_station_var, tl_station_label_var, tl_proj_var, tl_quality_var,
              tl_sy, tl_sm, tl_sd, tl_sh, tl_smin,
              tl_ey, tl_em, tl_ed, tl_eh, tl_emin,
              tl_use_end_var, tl_dur_h_var, tl_dur_m_var,
              tl_speed_var, tl_fps_var, tl_pattern_var,
              tl_output_var, tl_enhance_var, tl_timestamp_var,
              tl_crf_var, tl_preset_var, tl_saturation_var, tl_pto_var,
              st_mode_var, st_proj_var, st_pto_var, st_inputs_var, st_output_var,
              st_station_var, st_enhance_var, st_ts_var, st_sync_var, st_model_var,
              st_crf_var, st_preset_var, st_maxfr_var,
              st_saturation_var, st_dy, st_dm, st_dd, st_dh, st_dmin,
              st_station_label_var, st_cam_pattern_var, st_file_type_var):
        v.trace_add("write", _update_preview)
    nb.bind("<<NotebookTabChanged>>", _update_preview)

    root.mainloop()


def main():
    try:
        num_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cores = os.cpu_count() or 1
    numba.set_num_threads(num_cores)

    # Launch GUI when called with no arguments or with --gui
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == '--gui'):
        launch_gui()
        return

    parser = argparse.ArgumentParser(
        description="Reproject and stitch images or videos into a panorama based on a Hugin .pto file.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:

  Single image – explicit PTO file:
    stitcher.py project.pto /meteor/cam?/20260101/00/full_00.jpg out.jpg

  Single image – auto-generate PTO from lens.pto files (equirectangular):
    stitcher.py --equirect /meteor/cam?/20260101/00/full_00.jpg out.jpg

  Single image – fisheye output with vignetting correction:
    stitcher.py --fisheye --devignette=-0.5 /meteor/cam?/20260101/00/full_00.jpg out.jpg

  Single image – scale output canvas and enhance:
    stitcher.py --equirect --output-width=1920 --enhance /meteor/cam?/20260101/00/full_00.jpg out.jpg

  Single video – stitch multiple cameras into one panoramic video:
    stitcher.py project.pto /meteor/cam?/20260101/00/full_00.mp4 out.mp4

  Single video – with timestamp overlay, better quality, stop after 300 frames:
    stitcher.py project.pto /meteor/cam?/20260101/00/full_00.mp4 out.mp4 \
      --timestamp --crf=23 --preset=medium -n 300

  Single video – synchronize streams from embedded timestamps:
    stitcher.py project.pto /meteor/cam?/20260101/00/full_00.mp4 out.mp4 --sync --save-sync=sync.json

  Single video – reuse a saved sync map:
    stitcher.py project.pto /meteor/cam?/20260101/00/full_00.mp4 out.mp4 --sync --load-sync=sync.json

  Single video – fetch inputs from a remote station over SSH:
    stitcher.py project.pto /meteor/cam?/20260101/00/full_00.mp4 out.mp4 --station=ams000

  Timelapse – one night (start/end), 60x speed, equirectangular:
    stitcher.py out.mp4 --equirect --timelapse \\
      --timelapse-start="2026-06-22 21:00:00" --timelapse-end="2026-06-23 04:00:00" \\
      --timelapse-speed=60

  Timelapse – fixed duration from start, HD quality, 25 fps:
    stitcher.py out.mp4 --equirect --timelapse \\
      --timelapse-start="2026-06-22 22:00:00" --timelapse-duration="6 hours" \\
      --timelapse-quality=hd --timelapse-framerate=25 --timelapse-speed=120

  Timelapse – custom PTO and camera pattern:
    stitcher.py out.mp4 --timelapse --pto=custom.pto \\
      --timelapse-pattern="/meteor/cam?" \\
      --timelapse-start="2026-06-22 22:00:00" --timelapse-duration="4 hours" \\
      --timelapse-speed=60
"""
    )
    parser.add_argument("pto_file", nargs='?', help="Path to the Hugin PTO project file. Required unless --fisheye or --equirect is specified.")
    parser.add_argument("input_files", nargs='*', help="One or more input image or video files (must all be same type). Required unless --timelapse is used.")
    parser.add_argument("output_file", help="Path for the output panoramic image or video.")
    parser.add_argument("--fisheye", action='store_true', help="Generate fisheye panorama (8192x8192). Automatically creates PTO from lens.pto files found two directories up from input files.")
    parser.add_argument("--equirect", action='store_true', help="Generate equirectangular panorama (3380x2240). Automatically creates PTO from lens.pto files found two directories up from input files.")
    parser.add_argument("--enhance", action='store_true', help="Apply an adaptive enhancement filter to reduce noise and artifacts.")
    parser.add_argument("--timestamp", action='store_true', help="Overlay a UTC timestamp (YYYY-MM-DD hh:mm:ss.ff) in the lower-left corner of each video frame.")
    parser.add_argument("--force-video-dims", action='store_true', help="Force codec-safe output dimensions (video rules) even when input files are images.")
    parser.add_argument("--quiet", action='store_true', help="Suppress all text output.")
    parser.add_argument("--pad", type=int, default=0, help="Pixels to pad source images before reprojection (extends edges with blurred content).")
    parser.add_argument("--padsides", type=str, default="", help="Comma-separated sides to pad: top,bottom,left,right (default: all sides if --pad > 0).")
    
    parser.add_argument("-n", "--max-frames", type=int, default=0, metavar="N",
        help="Stop after encoding N frames (video only). Useful for quick tests.")
    parser.add_argument("--level-subsample", type=int, default=1, metavar="N",
        help="Recompute exposure correction only every N frames in video mode (default: 1). Higher values are faster.")
    parser.add_argument("--output-width", type=int, default=None, metavar="W",
        help="Force the output canvas width (pixels). Scales the PTO canvas proportionally.")
    parser.add_argument("--output-height", type=int, default=None, metavar="H",
        help="Force the output canvas height (pixels). Scales the PTO canvas proportionally.")
    parser.add_argument("--crf", type=str, default="28", metavar="CRF",
        help="libx264 CRF value for video output (default: 28). Lower = better quality.")
    parser.add_argument("--preset", type=str, default="ultrafast", metavar="PRESET",
        help="libx264 preset for video output (default: ultrafast).")
    parser.add_argument("--saturation", type=float, default=1.0, metavar="S",
        help="Chroma saturation multiplier applied to the output (default: 1.0 = unchanged, >1 = more vivid).")
    parser.add_argument("--devignette", type=float, default=-0.20, metavar="K1",
        help="Radial vignetting correction coefficient k1 for the model "
             "brightness(r) = 1 + k1*r² where r is normalised to the corner. "
             "Default: -0.20. Set to 0 to disable. "
             "Typical: --devignette=-0.5 (brightens corners ~33%%).")
    parser.add_argument("--station", type=str, default=None, metavar="HOST",
        help="Fetch input files from a remote host via SSH. The paths are interpreted on the remote host; the output is written locally.")

    sync_group = parser.add_argument_group('Video Synchronization Options')
    sync_group.add_argument("--sync", action='store_true', help="Synchronize video streams by their embedded timestamps before stitching.")
    sync_group.add_argument("--model", type=str, default=None, help="Specify the model for timestamp extraction.")
    sync_group.add_argument("--save-sync", type=str, default=None, help="Save the synchronization map to a JSON file (requires --sync).")
    sync_group.add_argument("--load-sync", type=str, default=None, help="Load a pre-computed synchronization map from a JSON file (requires --sync).")

    timelapse_group = parser.add_argument_group('Timelapse Options')
    timelapse_group.add_argument("--timelapse", action='store_true', help="Create a timelapse video instead of stitching a single input.")
    timelapse_group.add_argument("--timelapse-start", type=str, default=None, help="Timelapse start time (e.g., '2026-06-22 00:00:00').")
    timelapse_group.add_argument("--timelapse-end", type=str, default=None, help="Timelapse end time (e.g., '2026-06-22 06:30:10').")
    timelapse_group.add_argument("--timelapse-duration", type=str, default=None, help="Timelapse duration as a human-readable string (e.g., '6 hours 3 minutes 10 seconds').")
    timelapse_group.add_argument("--timelapse-speed", type=float, default=None, help="Timelapse speed-up factor (e.g., 60 for 60x).")
    timelapse_group.add_argument("--timelapse-framerate", type=int, default=30, help="Output timelapse frame rate (default: 30).")
    timelapse_group.add_argument("--timelapse-quality", type=str, default='sd', help="Source quality: 'sd'/'SD' for mini_mm.mp4 or 'hd'/'HD' for full_mm.mp4 (default: sd).")
    timelapse_group.add_argument("--timelapse-pattern", type=str, default='/meteor/cam?', help="Glob pattern used to find camera directories (default: /meteor/cam?).")
    timelapse_group.add_argument("--pto", type=str, default=None, metavar="FILE",
        help="Override the auto-generated PTO file for timelapse (use a custom lens calibration).")

    parser.add_argument("--input-datetime", type=str, default=None, metavar="DT",
        help="Hint: UTC datetime of the input frames (YYYY-MM-DD HH:MM:SS). Informational only.")

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
    if args.level_subsample < 1:
        _print("Error: --level-subsample must be a positive integer.", file=sys.stderr); sys.exit(1)

    # Validate --fisheye and --equirect options
    if args.fisheye and args.equirect:
        _print("Error: --fisheye and --equirect are mutually exclusive.", file=sys.stderr); sys.exit(1)

    # Parse --devignette coefficient
    if abs(args.devignette) > 1e-9:
        devignette = args.devignette
        _print(f"INFO: Devignetting enabled — k1={devignette}")
    else:
        devignette = None
        _print("INFO: Devignetting disabled (coefficient is zero)")

    remote_temp_dir = None
    _output_file_written = [False]   # set True only on successful completion

    # --- Timelapse mode ---
    if args.timelapse:
        if not args.timelapse_start:
            _print("Error: --timelapse requires --timelapse-start.", file=sys.stderr); sys.exit(1)
        if args.timelapse_speed is None or args.timelapse_speed <= 0:
            _print("Error: --timelapse requires a positive --timelapse-speed.", file=sys.stderr); sys.exit(1)
        if args.timelapse_end and args.timelapse_duration:
            _print("Error: --timelapse-end and --timelapse-duration cannot be used together.", file=sys.stderr); sys.exit(1)
        if not args.timelapse_end and not args.timelapse_duration:
            _print("Error: --timelapse requires either --timelapse-end or --timelapse-duration.", file=sys.stderr); sys.exit(1)
        quality = args.timelapse_quality.lower()
        if quality not in ('sd', 'hd'):
            _print("Error: --timelapse-quality must be 'sd' or 'hd'.", file=sys.stderr); sys.exit(1)
        if not (args.fisheye or args.equirect):
            _print("Error: --timelapse requires either --fisheye or --equirect to choose the projection.", file=sys.stderr); sys.exit(1)

        try:
            start_time = _parse_timelapse_datetime(args.timelapse_start)
        except ValueError as e:
            _print(f"Error: {e}", file=sys.stderr); sys.exit(1)

        if args.timelapse_end:
            try:
                end_time = _parse_timelapse_datetime(args.timelapse_end)
            except ValueError as e:
                _print(f"Error: {e}", file=sys.stderr); sys.exit(1)
        else:
            try:
                duration_seconds = _parse_timelapse_duration(args.timelapse_duration)
            except ValueError as e:
                _print(f"Error: {e}", file=sys.stderr); sys.exit(1)
            end_time = start_time + datetime.timedelta(seconds=duration_seconds)

        if end_time <= start_time:
            _print("Error: Timelapse end time must be after start time.", file=sys.stderr); sys.exit(1)

        _print(f"Discovering timelapse files for {start_time} -> {end_time} ({quality.upper()} quality)...")
        camera_files = _discover_timelapse_files(args.timelapse_pattern, start_time, end_time, quality, station=args.station)
        if not camera_files or not any(camera_files):
            _print("Error: No video files found for timelapse range/pattern.", file=sys.stderr); sys.exit(1)
        for i, files in enumerate(camera_files, start=1):
            if not files:
                _print(f"Error: Camera {i} has no video files in the timelapse range.", file=sys.stderr); sys.exit(1)

        if args.station:
            remote_temp_dir = tempfile.mkdtemp(prefix='stitcher_remote_')
            _print(f"Fetching timelapse files from {args.station}...")
            all_remote_files = [f for files in camera_files for f, _, _ in files]
            lens_paths = _collect_remote_lens_pto_paths(all_remote_files)
            _print(f"Discovered {len(all_remote_files)} remote timelapse files and {len(lens_paths)} lens.pto files")
            _fetch_remote_files_over_ssh(args.station, all_remote_files + lens_paths, remote_temp_dir, progress_prefix="Fetching timelapse files")
            camera_files = [
                [(os.path.join(remote_temp_dir, f.lstrip('/')), start, end) for f, start, end in files]
                for files in camera_files
            ]

        # Generate PTO from the first file of each camera.
        projection = 'fisheye' if args.fisheye else 'equirect'
        representative_files = [files[0][0] for files in camera_files]
        if args.pto:
            auto_generated_pto = args.pto
            _print(f"Using custom PTO file: {auto_generated_pto}")
        else:
            auto_generated_pto = generate_pto_from_lens_files(representative_files, projection)
            if auto_generated_pto is None:
                _print("Error: Failed to generate PTO file from lens.pto files.", file=sys.stderr); sys.exit(1)
            _print(f"Generated PTO file: {auto_generated_pto}")

        try:
            padsides = set(s.strip() for s in args.padsides.split(',') if s.strip()) if args.padsides else ({'top','bottom','left','right'} if args.pad > 0 else set())
            reproject_timelapse(
                auto_generated_pto, camera_files, args.output_file,
                start_time, end_time, args.timelapse_speed, args.timelapse_framerate,
                args.pad, num_cores, padsides, model=args.model,
                enhance=args.enhance, fisheye_mask=args.fisheye, max_frames=args.max_frames,
                level_subsample=args.level_subsample, crf=args.crf, preset=args.preset,
                timestamp=args.timestamp, saturation=args.saturation,
                devignette=devignette
            )
            _output_file_written[0] = True
        except (ValueError, FileNotFoundError, ImportError, IOError, RuntimeError, KeyboardInterrupt) as e:
            _print(f"\n❌ An error occurred during processing:\n{e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            _print(f"\n❌ An unexpected critical error occurred:\n{e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            if not _output_file_written[0] and args.output_file and os.path.exists(args.output_file):
                try:
                    os.unlink(args.output_file)
                    _print(f"Cleaned up partial output file: {args.output_file}", file=sys.stderr)
                except Exception:
                    pass
            if auto_generated_pto and os.path.exists(auto_generated_pto):
                try:
                    os.unlink(auto_generated_pto)
                    _print(f"Cleaned up temporary PTO file: {auto_generated_pto}")
                except Exception as e:
                    _print(f"Warning: Failed to clean up temporary PTO file: {e}", file=sys.stderr)
            if remote_temp_dir and os.path.exists(remote_temp_dir):
                try:
                    shutil.rmtree(remote_temp_dir)
                    _print(f"Cleaned up remote temp directory: {remote_temp_dir}")
                except Exception as e:
                    _print(f"Warning: Failed to clean up remote temp directory: {e}", file=sys.stderr)
        return

    # --- Non-timelapse validation ---
    if len(args.input_files) < 2:
        if args.sync: _print("INFO: --sync option ignored for a single input file."); args.sync = False
    if args.station:
        if not args.input_files:
            _print("Error: --station requires input file patterns to be specified.", file=sys.stderr); sys.exit(1)
        remote_temp_dir = tempfile.mkdtemp(prefix='stitcher_remote_')
        _print(f"Fetching input files from {args.station}...")
        remote_input_files = _expand_remote_input_patterns(args.station, args.input_files)
        if not remote_input_files:
            _print("Error: No remote files matched the input patterns.", file=sys.stderr); sys.exit(1)
        lens_paths = _collect_remote_lens_pto_paths(remote_input_files)
        _print(f"Discovered {len(remote_input_files)} remote input files and {len(lens_paths)} lens.pto files")
        # Sort remote input files by camera number to match PTO file order
        try:
            remote_input_files = sorted(remote_input_files, key=lambda f: extract_camera_number_from_path(f))
        except ValueError:
            pass
        all_remote_files = remote_input_files + lens_paths
        local_paths = _fetch_remote_files_over_ssh(args.station, all_remote_files, remote_temp_dir, progress_prefix="Fetching input files")
        args.input_files = local_paths[:len(remote_input_files)]
    if not args.input_files:
        _print("Error: No input files specified.", file=sys.stderr); sys.exit(1)
    if not args.pto_file and not (args.fisheye or args.equirect):
        _print("Error: Either pto_file or --fisheye/--equirect must be specified.", file=sys.stderr); sys.exit(1)

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
        # Sort input files by camera number to match PTO file order
        try:
            args.input_files = sorted(args.input_files, key=lambda f: extract_camera_number_from_path(f))
        except ValueError:
            # If any file doesn't have a camera number, keep original order
            pass
        _print(f"Expanded input files to {len(args.input_files)} files")
        
        pto_file = generate_pto_from_lens_files(args.input_files, projection,
            w=args.output_width, h=args.output_height)
        if pto_file is None:
            _print("Error: Failed to generate PTO file from lens.pto files.", file=sys.stderr); sys.exit(1)
        args.pto_file = pto_file
        auto_generated_pto = pto_file
        _print(f"Generated PTO file: {pto_file}")

    for f in [args.pto_file] + args.input_files:
        if not os.path.exists(f):
            _print(f"Error: Input file not found: {f}", file=sys.stderr); sys.exit(1)

    is_image_output = args.output_file.lower().endswith(('.jpg', '.jpeg', '.png'))
    is_video_output = args.output_file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))

    # If output is image but inputs are videos, extract first frame from each video
    _temp_frame_files = []
    if is_image_output and any(f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')) for f in args.input_files):
        if not _av():
            _print("Error: PyAV is required to extract frames from videos for image output.", file=sys.stderr); sys.exit(1)
        _print("Extracting first frame from video inputs...")
        for vid_path in args.input_files:
            if vid_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                import tempfile as _tf
                temp_frame = _tf.NamedTemporaryFile(suffix='.jpg', delete=False)
                temp_frame.close()
                _temp_frame_files.append(temp_frame.name)
                try:
                    container = _av().open(vid_path)
                    frame = container.decode(video=0).__next__()
                    if frame.format.name != "rgb24":
                        frame = frame.reformat(format="rgb24")
                    frame.to_image().save(temp_frame.name, quality=95)
                    container.close()
                except Exception as e:
                    _print(f"Error extracting frame from {vid_path}: {e}", file=sys.stderr); sys.exit(1)
        # Replace video paths with extracted frame paths
        args.input_files = [_temp_frame_files[i] if args.input_files[i].lower().endswith(('.mp4', '.mov', '.avi', '.mkv')) else args.input_files[i] for i in range(len(args.input_files))]

    # --- Main Execution with Global Error Handling ---
    try:
        padsides = set(s.strip() for s in args.padsides.split(',') if s.strip()) if args.padsides else ({'top','bottom','left','right'} if args.pad > 0 else set())
        if is_image_output:
            reproject_images(args.pto_file, args.input_files, args.output_file, args.pad, num_cores, padsides, args.enhance, force_video_dims=args.force_video_dims, fisheye_mask=args.fisheye, saturation=args.saturation, devignette=devignette, input_datetime=args.input_datetime)
        elif is_video_output:
            reproject_videos(
                args.pto_file, args.input_files, args.output_file,
                args.pad, num_cores, padsides, args.sync, args.model,
                save_sync_file=args.save_sync, load_sync_file=args.load_sync, enhance=args.enhance,
                fisheye_mask=args.fisheye, max_frames=args.max_frames,
                level_subsample=args.level_subsample, crf=args.crf, preset=args.preset,
                timestamp=args.timestamp, saturation=args.saturation,
                devignette=devignette
            )
        else:
            _print("Error: Output file must have a supported extension (.jpg/.jpeg/.png for images, .mp4/.mov/.avi/.mkv for videos).", file=sys.stderr)
            sys.exit(1)
        _output_file_written[0] = True
    except (ValueError, FileNotFoundError, ImportError, IOError, RuntimeError, KeyboardInterrupt) as e:
        _print(f"\n❌ An error occurred during processing:\n{e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        _print(f"\n❌ An unexpected critical error occurred:\n{e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if not _output_file_written[0] and args.output_file and os.path.exists(args.output_file):
            try:
                os.unlink(args.output_file)
                _print(f"Cleaned up partial output file: {args.output_file}", file=sys.stderr)
            except Exception:
                pass
        # Clean up auto-generated PTO file
        if auto_generated_pto and os.path.exists(auto_generated_pto):
            try:
                os.unlink(auto_generated_pto)
                _print(f"Cleaned up temporary PTO file: {auto_generated_pto}")
            except Exception as e:
                _print(f"Warning: Failed to clean up temporary PTO file: {e}", file=sys.stderr)
        # Clean up temporary extracted frame files
        for tf in _temp_frame_files:
            if os.path.exists(tf):
                try:
                    os.unlink(tf)
                except Exception:
                    pass
        if remote_temp_dir and os.path.exists(remote_temp_dir):
            try:
                shutil.rmtree(remote_temp_dir)
                _print(f"Cleaned up remote temp directory: {remote_temp_dir}")
            except Exception as e:
                _print(f"Warning: Failed to clean up remote temp directory: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
