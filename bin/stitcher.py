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
import pytz


try:
    import pto_mapper
except ImportError:
    print("Error: The 'pto_mapper.py' module was not found.", file=sys.stderr)
    print("Please ensure pto_mapper.py is in the same directory as this script.", file=sys.stderr)
    sys.exit(1)

# Import get_timestamp directly from the timestamp.py file
try:
    from timestamp import get_timestamp
except ImportError:
    print("Error: The 'timestamp.py' module was not found.", file=sys.stderr)
    print("Please ensure timestamp.py is in the same directory as this script.", file=sys.stderr)
    sys.exit(1)


try:
    import av
except ImportError:
    print("Warning: 'PyAV' library not found. Video processing functionality will be unavailable.")
    av = None

try:
    import scipy.ndimage as ndimage
except ImportError:
    print("Warning: 'scipy' not found. Seam finding and leveling will be unavailable. Run 'pip install scipy'")
    ndimage = None

try:
    import pandas as pd
except ImportError:
    print("Warning: 'pandas' not found. Seam leveling performance will be degraded. Run 'pip install pandas'")
    pd = None


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
            weights[y, x] = min(dist_x, dist_y) / norm
    np.clip(weights, 0.0, 1.0, out=weights)
    return weights

@numba.njit(parallel=True, fastmath=True, cache=True)
def create_corner_penalty_map(width, height):
    penalty = np.ones((height, width), dtype=np.float32)
    corner_radius = min(width, height) * 0.33
    if corner_radius < 1.0:
        return penalty

    corners_x = np.array([0, width - 1, 0, width - 1])
    corners_y = np.array([0, 0, height - 1, height - 1])

    for y in prange(height):
        for x in range(width):
            dx = x - corners_x
            dy = y - corners_y
            dist_sq = dx**2 + dy**2
            
            min_dist = math.sqrt(np.min(dist_sq))

            if min_dist < corner_radius:
                ratio = min_dist / corner_radius
                penalty_factor = ratio * ratio
                penalty[y, x] = min(1.0, penalty_factor)
                
    return penalty

@numba.njit(parallel=True, fastmath=True)
def _apply_gain_offset_numba(plane_stack, gain_arr, offset_arr):
    num_images, h, w = plane_stack.shape
    for i in prange(num_images):
        gain = gain_arr[i]
        offset = offset_arr[i]
        for r in range(h):
            for c in range(w):
                val = plane_stack[i, r, c]
                if val > 0:
                    new_val = val * gain + offset
                    if new_val > 255: new_val = 255
                    elif new_val < 0: new_val = 0
                    plane_stack[i, r, c] = new_val

@numba.njit(parallel=True, fastmath=True)
def _apply_offset_numba(plane_stack, offset_arr):
    num_images, h, w = plane_stack.shape
    for i in prange(num_images):
        offset = offset_arr[i]
        for r in range(h):
            for c in range(w):
                val = plane_stack[i, r, c]
                if val > 0:
                    new_val = val + offset
                    if new_val > 255: new_val = 255
                    elif new_val < 0: new_val = 0
                    plane_stack[i, r, c] = new_val

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

@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def reproject_uv_float(p_float_uv_src, dw, dh, map_uv_idx, out_float_uv):
    half_w, half_h = dw // 2, dh // 2
    for y_uv in prange(half_h):
        base_uv = y_uv * half_w
        for x_uv in range(half_w):
            coffset = map_uv_idx[base_uv + x_uv]
            if coffset >= 0:
                out_float_uv[base_uv + x_uv] = p_float_uv_src[coffset]

@numba.njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def compute_map_and_weights(coords_y, sw, sh, pad):
    dh, dw, _ = coords_y.shape
    map_y_idx, c01, c23 = np.full(dh*dw,-1,dtype=np.int32), np.zeros(dh*dw,dtype=np.uint16), np.zeros(dh*dw,dtype=np.uint16)
    effective_sw, effective_sh = sw + 2 * pad, sh + pad
    for idx in prange(dh * dw):
        y_dest, x_dest = idx // dw, idx % dw
        dx_orig, dy_orig = coords_y[y_dest, x_dest, 0], coords_y[y_dest, x_dest, 1]
        if dx_orig <= -99999.0: continue
        dx, dy = dx_orig + pad, dy_orig + pad
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
def compute_uv_map(coords_uv, sw_uv, sh_uv, pad_uv):
    h_uv, w_uv, _ = coords_uv.shape
    map_uv_idx = np.full(h_uv * w_uv, -1, dtype=np.int32)
    effective_sw_uv, effective_sh_uv = sw_uv + 2 * pad_uv, sh_uv + pad_uv
    for idx in prange(h_uv * w_uv):
        y_uv, x_uv = idx // w_uv, idx % w_uv
        sx_uv_orig, sy_uv_orig = coords_uv[y_uv, x_uv, 0], coords_uv[y_uv, x_uv, 1]
        if sx_uv_orig <= -99999.0: continue
        sx_uv, sy_uv = sx_uv_orig + pad_uv, sy_uv_orig + pad_uv
        if 0 <= sx_uv < effective_sw_uv and 0 <= sy_uv < effective_sh_uv:
            map_uv_idx[idx] = int(sy_uv) * effective_sw_uv + int(sx_uv)
    return map_uv_idx.reshape(h_uv, w_uv)

def _map_one_image(args):
    # Temporarily set Numba to use 1 thread to avoid contention with the parent ThreadPoolExecutor.
    # This prevents oversubscription of threads and improves performance for this task-parallel workload.
    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        
        img, pad, final_w, final_h, orig_w, orig_h, crop_offset_x, crop_offset_y, pano_proj_f, pano_hfov = args
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

        pto_mapper.calculate_source_coords(coords_y,final_w,final_h,orig_w,orig_h,crop_offset_x,crop_offset_y,pano_proj_f,pano_hfov,sw,sh,R_pr_inv,y,src_focal,src_norm_radius,a,b,c,cx,cy,src_proj_f)

        map_y_idx, c01, c23, coords_uv = *compute_map_and_weights(coords_y,sw,sh,pad), coords_y[::2,::2]/2.
        map_uv_idx = compute_uv_map(coords_uv, sw//2, sh//2, pad//2)
        
        return (map_y_idx, c01, c23, map_uv_idx, sw, sh)
    finally:
        numba.set_num_threads(original_threads)


def build_mappings(pto_file, pad, num_workers):
    global_options, images = pto_mapper.parse_pto_file(pto_file)

    orig_w, orig_h = global_options.get('w'), global_options.get('h')
    if orig_w is None or orig_h is None: raise ValueError("PTO 'p' line must contain width 'w' and height 'h'.")
    pano_proj_f = int(global_options.get('f', 2))
    pano_hfov = global_options.get('v')
    if pano_hfov is None: raise ValueError("PTO 'p' line must have HFOV 'v' for projection calculations.")

    crop_coords = global_options.get('S')
    if crop_coords:
        left, top, right, bottom = crop_coords
        left, top, right, bottom = left & ~1, top & ~1, right & ~1, bottom & ~1
        final_w, final_h, crop_offset_x, crop_offset_y = right-left, bottom-top, left, top
    else: final_w, final_h, crop_offset_x, crop_offset_y = orig_w, orig_h, 0, 0
    global_options['final_w'], global_options['final_h'] = final_w, final_h

    task_args = [(img, pad, final_w, final_h, orig_w, orig_h, crop_offset_x, crop_offset_y, pano_proj_f, pano_hfov) for img in images]

    print("Building projection maps...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        all_mappings = list(executor.map(_map_one_image, task_args))

    return all_mappings, global_options

def load_image_to_yuv(image_path, pad):
    # Add a compatibility check for different Pillow versions
    try:
        resample_filter = Image.Resampling.BICUBIC
    except AttributeError:
        # Fallback for older versions
        resample_filter = Image.BICUBIC

    img_pil = Image.open(image_path).convert("RGB")
    img_ycbcr = img_pil.convert("YCbCr")
    y, u, v = img_ycbcr.split()
    # Use the determined resampling filter
    u_resized = u.resize((img_pil.width // 2, img_pil.height // 2), resample_filter)
    v_resized = v.resize((img_pil.width // 2, img_pil.height // 2), resample_filter)
    padded_y = np.pad(np.array(y, np.uint8), pad, mode='edge')
    padded_u = np.pad(np.array(u_resized, np.uint8), pad // 2, mode='edge')
    padded_v = np.pad(np.array(v_resized, np.uint8), pad // 2, mode='edge')
    target_h_y = img_pil.height + pad
    target_h_uv = img_pil.height // 2 + pad // 2
    return (padded_y[:target_h_y, :], padded_u[:target_h_uv, :], padded_v[:target_h_uv, :], img_pil.width, img_pil.height)

def save_image_yuv420(y_plane, u_plane, v_plane, output_path):
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

def process_and_reproject_image(args):
    """Worker function to reproject a single image, writing to pre-allocated buffers."""
    (input_path, dw, dh, mapping, pad, use_seam), out_buffers = args
    reproj_y, reproj_u, reproj_v, reproj_weights_y, reproj_weights_uv, reproj_penalty_y = out_buffers

    py, pu, pv, sw_orig, sh_orig = load_image_to_yuv(input_path, pad)
    
    sw_padded, sh_padded = sw_orig + 2 * pad, sh_orig + pad
    
    blend_weights_y = create_blend_weight_map(sw_padded, sh_padded)
    blend_weights_uv = np.ascontiguousarray(blend_weights_y[::2,::2])

    map_y_idx, c01, c23, map_uv_idx, _, _ = mapping
    
    # Initialize output buffers
    reproj_y.fill(0)
    reproj_u.fill(128)
    reproj_v.fill(128)
    reproj_weights_y.fill(0)
    reproj_weights_uv.fill(0)
    if use_seam:
        reproj_penalty_y.fill(0)

    # Reproject into shared buffers
    reproject_y(py.ravel(), dw, dh, py.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_y.ravel())
    reproject_uv(pu.ravel(), pv.ravel(), dw, dh, map_uv_idx.ravel(), reproj_u.ravel(), reproj_v.ravel())
    reproject_float(blend_weights_y.ravel(), dw, dh, blend_weights_y.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_weights_y.ravel())
    reproject_uv_float(blend_weights_uv.ravel(), dw, dh, map_uv_idx.ravel(), reproj_weights_uv.ravel())

    if use_seam:
        corner_penalty_src = create_corner_penalty_map(sw_padded, sh_padded)
        reproject_float(corner_penalty_src.ravel(), dw, dh, corner_penalty_src.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_penalty_y.ravel())

@numba.njit
def _get_all_seam_diffs(y_indices, y_stack, u_stack, v_stack, subsample):
    h, w = y_indices.shape
    diff_data = numba.typed.List()

    for r in range(0, h - 1, subsample):
        for c in range(0, w - 1, subsample):
            p1 = y_indices[r, c]
            if p1 == -1:
                continue

            # Check the immediate horizontal neighbor
            p2_h = y_indices[r, c + 1]
            if p2_h != -1 and p1 != p2_h:
                uv_r, uv_c = r // 2, c // 2
                p2_uv_c = (c + 1) // 2
                y_diff = y_stack[p1, r, c] - y_stack[p2_h, r, c + 1]
                u_diff = u_stack[p1, uv_r, uv_c] - u_stack[p2_h, uv_r, p2_uv_c]
                v_diff = v_stack[p1, uv_r, uv_c] - v_stack[p2_h, uv_r, p2_uv_c]
                diff_data.append((np.float32(p1), np.float32(p2_h), y_diff, u_diff, v_diff))

            # Check the immediate vertical neighbor
            p2_v = y_indices[r + 1, c]
            if p2_v != -1 and p1 != p2_v:
                uv_r, uv_c = r // 2, c // 2
                p2_uv_r = (r + 1) // 2
                y_diff = y_stack[p1, r, c] - y_stack[p2_v, r + 1, c]
                u_diff = u_stack[p1, uv_r, uv_c] - u_stack[p2_v, p2_uv_r, uv_c]
                v_diff = v_stack[p1, uv_r, uv_c] - v_stack[p2_v, p2_uv_r, uv_c]
                diff_data.append((np.float32(p1), np.float32(p2_v), y_diff, u_diff, v_diff))

    final_array = np.empty((len(diff_data), 5), dtype=np.float32)
    for i, item in enumerate(diff_data):
        final_array[i, 0] = item[0]
        final_array[i, 1] = item[1]
        final_array[i, 2] = item[2]
        final_array[i, 3] = item[3]
        final_array[i, 4] = item[4]
            
    return final_array

def solve_corrections(y_indices, all_y, all_u, all_v, num_images, subsample):
    if pd is None:
        raise ImportError("The 'pandas' library is required for the efficient seam leveling implementation.")

    y_stack = all_y.astype(np.float32)
    u_stack = all_u.astype(np.float32)
    v_stack = all_v.astype(np.float32)

    raw_diffs_array = _get_all_seam_diffs(y_indices, y_stack, u_stack, v_stack, subsample)
    if raw_diffs_array.shape[0] == 0:
        return np.ones(num_images), np.zeros(num_images), np.zeros(num_images), np.zeros(num_images)

    df = pd.DataFrame(raw_diffs_array, columns=['p1', 'p2', 'y_d', 'u_d', 'v_d'])
    swap_mask = df['p1'] > df['p2']
    p1_swapped = df.loc[swap_mask, 'p1'].copy()
    df.loc[swap_mask, 'p1'] = df.loc[swap_mask, 'p2']
    df.loc[swap_mask, 'p2'] = p1_swapped
    df.loc[swap_mask, ['y_d', 'u_d', 'v_d']] *= -1
    
    median_diffs_df = df.groupby(['p1', 'p2']).median()

    diffs = {}
    graph = {i: [] for i in range(num_images)}
    for (p1_raw, p2_raw), row in median_diffs_df.iterrows():
        p1, p2 = int(p1_raw), int(p2_raw)
        diffs[(p1, p2)] = (row['y_d'], row['u_d'], row['v_d'])
        if p1 >= 0 and p2 >= 0:
            if p2 not in graph[p1]: graph[p1].append(p2)
            if p1 not in graph[p2]: graph[p2].append(p1)

    y_offset, u_offset, v_offset = [np.zeros(num_images, dtype=np.float32) for _ in range(3)]
    for _ in range(100):
        next_y, next_u, next_v = [np.zeros_like(c) for c in (y_offset, u_offset, v_offset)]
        for i in range(num_images):
            if not graph.get(i): continue
            y_t, u_t, v_t = [], [], []
            for j in graph[i]:
                d = diffs.get((i,j)) or tuple(-x for x in diffs.get((j,i), (0,0,0)))
                y_t.append(y_offset[j] - d[0]); u_t.append(u_offset[j] - d[1]); v_t.append(v_offset[j] - d[2])
            if not y_t: continue
            next_y[i], next_u[i], next_v[i] = np.median(y_t), np.median(u_t), np.median(v_t)
        y_offset, u_offset, v_offset = next_y, next_u, next_v
        y_offset -= np.mean(y_offset); u_offset -= np.mean(u_offset); v_offset -= np.mean(v_offset)

    y_offset_corrected = y_stack + y_offset[:, np.newaxis, np.newaxis]

    gain_diffs = {}
    epsilon = 1e-6
    for i in range(num_images):
        for j in graph.get(i, []):
            if i >= j: continue
            seam_mask = np.zeros_like(y_indices, dtype=bool)
            seam_mask[:,:-1] |= ((y_indices[:,:-1] == i) & (y_indices[:,1:] == j)) | ((y_indices[:,:-1] == j) & (y_indices[:,1:] == i))
            seam_mask[:-1,:] |= ((y_indices[:-1,:] == i) & (y_indices[1:,:] == j)) | ((y_indices[:-1,:] == j) & (y_indices[1:,:] == i))
            if not np.any(seam_mask): continue
            
            y_i = y_offset_corrected[i][seam_mask]
            y_j = y_offset_corrected[j][seam_mask]
            if y_i.size == 0: continue
            
            log_diff = np.median(np.log(np.maximum(y_i, epsilon)) - np.log(np.maximum(y_j, epsilon)))
            gain_diffs[(i,j)] = log_diff
            
    y_log_gain = np.zeros(num_images, dtype=np.float32)
    for _ in range(100):
        next_log_gain = np.zeros_like(y_log_gain)
        for i in range(num_images):
            if not graph.get(i): continue
            targets = []
            for j in graph[i]:
                d = gain_diffs.get((i,j), -gain_diffs.get((j,i), 0))
                targets.append(y_log_gain[j] - d)
            if not targets: continue
            next_log_gain[i] = np.median(targets)
        y_log_gain = next_log_gain
        y_log_gain -= np.mean(y_log_gain)

    y_gain = np.exp(y_log_gain)
    return y_gain, y_offset, u_offset, v_offset

def _calculate_seam_blend_weights(y_weight_stack, penalty_maps_stack, seam_subsample):
    """Finds optimal seams and calculates final blending weights."""
    if ndimage is None:
        raise ImportError("The '--seamless' option requires 'scipy'. Please run 'pip install scipy'.")
    
    s = seam_subsample
    h, w = y_weight_stack.shape[1], y_weight_stack.shape[2]
    
    if s > 1:
        sub_weights = y_weight_stack[:, ::s, ::s]
        sub_dist_maps = np.array([ndimage.distance_transform_edt(w) for w in sub_weights])
        zoom_factors = (1, s, s)
        dist_maps = ndimage.zoom(sub_dist_maps, zoom_factors, order=1, prefilter=False)[:,:h,:w]
    else:
        dist_maps = np.array([ndimage.distance_transform_edt(w) for w in y_weight_stack])

    penalized_scores = dist_maps * penalty_maps_stack
    score_sum = np.sum(penalized_scores, axis=0)
    score_sum[score_sum < 1e-9] = 1.0
    final_weights_y_stack = penalized_scores / score_sum[np.newaxis, :, :]

    final_weights_uv_stack = 0.25 * (final_weights_y_stack[:, 0:h:2, 0:w:2] + final_weights_y_stack[:, 1:h:2, 0:w:2] +
                                     final_weights_y_stack[:, 0:h:2, 1:w:2] + final_weights_y_stack[:, 1:h:2, 1:w:2])

    return final_weights_y_stack, final_weights_uv_stack, penalized_scores

def _calculate_feather_blend_weights(y_weight_stack, uv_weight_stack):
    """Calculates standard feathering blending weights."""
    y_weight_sum = np.sum(y_weight_stack, axis=0)
    y_weight_sum[y_weight_sum < 1e-9] = 1.0
    final_weights_y_stack = y_weight_stack / y_weight_sum[np.newaxis, :, :]

    uv_weight_sum = np.sum(uv_weight_stack, axis=0)
    uv_weight_sum[uv_weight_sum < 1e-9] = 1.0
    final_weights_uv_stack = uv_weight_stack / uv_weight_sum[np.newaxis, :, :]
    
    return final_weights_y_stack, final_weights_uv_stack

def _blend_final_image(y_planes, u_planes, v_planes, final_weights_y, final_weights_uv, masking_stack):
    """Blends the final image using the calculated weights and corrects the background color."""
    final_h, final_w = y_planes.shape[1], y_planes.shape[2]
    y_acc = np.zeros((final_h, final_w), np.float32)
    u_acc = np.zeros((final_h//2, final_w//2), np.float32)
    v_acc = np.zeros((final_h//2, final_w//2), np.float32)
    
    for i in range(len(y_planes)):
        y_acc += y_planes[i] * final_weights_y[i]
        u_acc += u_planes[i] * final_weights_uv[i]
        v_acc += v_planes[i] * final_weights_uv[i]
        
    y_final = np.clip(y_acc, 0, 255).astype(np.uint8)
    u_final = np.clip(u_acc, 0, 255).astype(np.uint8)
    v_final = np.clip(v_acc, 0, 255).astype(np.uint8)

    # Correct background color to be neutral black (Y=0, U=128, V=128)
    unmapped_mask = np.sum(masking_stack, axis=0) < 1e-9
    unmapped_mask_uv = unmapped_mask[::2, ::2]
    u_final[unmapped_mask_uv] = 128
    v_final[unmapped_mask_uv] = 128

    return y_final, u_final, v_final

def reproject_images(pto_file, input_files, output_file, pad, use_seam, level_subsample, seam_subsample, num_cores):
    mappings, global_options = build_mappings(pto_file, pad, num_cores)
    final_w, final_h = global_options['final_w'], global_options['final_h']
    num_images = len(mappings)
    if len(input_files) != num_images:
        raise ValueError(f"Input files ({len(input_files)}) != PTO images ({num_images}).")

    all_reproj_y = np.empty((num_images, final_h, final_w), dtype=np.uint8)
    all_reproj_u = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.uint8)
    all_reproj_v = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.uint8)
    all_weights_y = np.empty((num_images, final_h, final_w), dtype=np.float32)
    all_weights_uv = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.float32)
    all_penalty_y = np.empty((num_images, final_h, final_w), dtype=np.float32) if use_seam else None

    process_args = [
        (
            (input_files[i], final_w, final_h, mappings[i], pad, use_seam),
            (all_reproj_y[i], all_reproj_u[i], all_reproj_v[i], all_weights_y[i], all_weights_uv[i], all_penalty_y[i] if use_seam else None)
        )
        for i in range(num_images)
    ]

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        list(executor.map(process_and_reproject_image, process_args))

    y_planes = all_reproj_y
    u_planes = all_reproj_u
    v_planes = all_reproj_v
    
    if use_seam:
        y_weight_stack = all_weights_y
        penalty_stack = all_penalty_y
        final_weights_y, final_weights_uv, penalized_scores = _calculate_seam_blend_weights(y_weight_stack, penalty_stack, seam_subsample)
        
        if num_images > 1:
            print("Leveling seams...")
            background_mask = np.max(y_weight_stack, axis=0) < 1e-9
            y_indices = np.argmax(penalized_scores, axis=0)
            y_indices[background_mask] = -1
            
            leveling_params = solve_corrections(y_indices, y_planes, u_planes, v_planes, num_images, level_subsample)
            y_gain, y_offset, u_offset, v_offset = leveling_params
            
            y_planes_leveled = y_planes.astype(np.float32)
            u_planes_leveled = u_planes.astype(np.float32)
            v_planes_leveled = v_planes.astype(np.float32)

            _apply_gain_offset_numba(y_planes_leveled, y_gain, y_offset)
            _apply_offset_numba(u_planes_leveled, u_offset)
            _apply_offset_numba(v_planes_leveled, v_offset)
            
            y_planes = y_planes_leveled
            u_planes = u_planes_leveled
            v_planes = v_planes_leveled
            
        masking_stack = penalized_scores
    else: # Feathering
        y_weight_stack = all_weights_y
        uv_weight_stack = all_weights_uv
        final_weights_y, final_weights_uv = _calculate_feather_blend_weights(y_weight_stack, uv_weight_stack)
        masking_stack = y_weight_stack

    y_final, u_final, v_final = _blend_final_image(y_planes, u_planes, v_planes, final_weights_y, final_weights_uv, masking_stack)

    save_image_yuv420(y_final, u_final, v_final, output_file)
    print(f"Panoramic image saved to {output_file}")

def worker_for_video_frame(args):
    """Worker function for video frames, writing to pre-allocated buffers."""
    (idx, frame, mapping, dw, dh, blend_weights_y_src, blend_weights_uv_src, pad, use_seam, corner_penalty_src), out_buffers = args
    reproj_y, reproj_u, reproj_v, reproj_weights_y, reproj_weights_uv, reproj_penalty_y = out_buffers

    if frame is None: return None
    if frame.format.name not in ("yuv420p", "yuvj420p"): frame = frame.reformat(format="yuv420p")

    sw_orig, sh_orig = frame.width, frame.height
    py_src_orig = np.asarray(frame.planes[0]).reshape(sh_orig, sw_orig)
    pu_src_orig = np.asarray(frame.planes[1]).reshape(sh_orig // 2, sw_orig // 2)
    pv_src_orig = np.asarray(frame.planes[2]).reshape(sh_orig // 2, sw_orig // 2)

    py_src_all = np.pad(py_src_orig, pad, mode='edge')
    pu_src_all = np.pad(pu_src_orig, pad // 2, mode='edge')
    pv_src_all = np.pad(pv_src_orig, pad // 2, mode='edge')
    target_h_y, target_h_uv = sh_orig + pad, sh_orig // 2 + pad // 2
    py_src, pu_src, pv_src = py_src_all[:target_h_y, :], pu_src_all[:target_h_uv, :], pv_src_all[:target_h_uv, :]

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
    reproject_uv_float(blend_weights_uv_src.ravel(), dw, dh, map_uv_idx.ravel(), reproj_weights_uv.ravel())

    # Add a check to ensure corner_penalty_src is valid before processing
    if use_seam and corner_penalty_src is not None:
        if reproj_penalty_y is not None:
            reproj_penalty_y.fill(0)
            reproject_float(corner_penalty_src.ravel(), dw, dh, corner_penalty_src.shape[1], map_y_idx.ravel(), c01.ravel(), c23.ravel(), reproj_penalty_y.ravel())
    
    return idx

def reproject_videos(pto_file, input_files, output_file, pad, use_seam, level_subsample, seam_subsample, num_cores, level_freq, use_sync=False, model=None):
    if av is None: raise ImportError("PyAV is not installed.")

    mappings, global_options = build_mappings(pto_file, pad, num_cores)
    final_w, final_h = global_options['final_w'], global_options['final_h']
    num_images = len(mappings)
    if len(input_files) != num_images: raise ValueError("Number of videos does not match PTO.")
    in_containers = [av.open(f) for f in input_files]
    in_streams = [c.streams.video[0] for c in in_containers]
    for s in in_streams: s.thread_type = 'AUTO'

    out_container = av.open(output_file, mode='w')
    out_stream = out_container.add_stream("libx264", rate=in_streams[0].average_rate)
    out_stream.width, out_stream.height, out_stream.pix_fmt = final_w, final_h, 'yuv420p'
    out_stream.options = {"preset": "fast", "crf": "28"}

    frame_iters = [c.decode(s) for c, s in zip(in_containers, in_streams)]
    frame_count = 0
    
    cached_seam_weights, target_leveling_params, previous_leveling_params = None, None, None
    recalc_frame_number = 1
    
    frame_y_planes = np.empty((num_images, final_h, final_w), dtype=np.uint8)
    frame_u_planes = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.uint8)
    frame_v_planes = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.uint8)
    frame_weights_y = np.empty((num_images, final_h, final_w), dtype=np.float32)
    frame_weights_uv = np.empty((num_images, final_h // 2, final_w // 2), dtype=np.float32)
    frame_penalty_y = np.empty((num_images, final_h, final_w), dtype=np.float32) if use_seam else None
    blend_weights_y, blend_weights_uv, corner_penalties_y = [], [], []
    for stream in in_streams:
        sw_padded, sh_padded = stream.width + 2 * pad, stream.height + pad
        weights_y = create_blend_weight_map(sw_padded, sh_padded)
        blend_weights_y.append(weights_y)
        blend_weights_uv.append(np.ascontiguousarray(weights_y[::2, ::2]))
        if use_seam:
            corner_penalties_y.append(create_corner_penalty_map(sw_padded, sh_padded))

    if use_sync:
        print("Timestamp synchronization enabled.")
        current_frames = [None] * num_images
        current_timestamps = [None] * num_images
        stream_intervals = [datetime.timedelta(seconds=4)] * num_images
        stream_ended = [False] * num_images

        def advance_stream(i, prev_ts):
            try:
                frame = next(frame_iters[i])
                
                read_ts = None
                try:
                    # Wrap timestamp reading in a try/except block ---
                    read_ts = get_timestamp(frame.to_image(), robust=False, model=model)
                    if read_ts is None:
                        read_ts = get_timestamp(frame.to_image(), robust=True, model=model)
                except ValueError as e:
                    read_ts = None # Ensure it is None if parsing fails

                expected_ts = prev_ts + stream_intervals[i]
                
                is_plausible = False
                if read_ts:
                    if abs((read_ts - expected_ts).total_seconds()) < 300: # 5-minute plausibility window
                        is_plausible = True

                if is_plausible:
                    new_ts = read_ts
                    time_diff = new_ts - prev_ts
                    if 0.1 < time_diff.total_seconds() < 300:
                        stream_intervals[i] = datetime.timedelta(seconds=(0.8 * stream_intervals[i].total_seconds() + 0.2 * time_diff.total_seconds()))
                else:
                    new_ts = expected_ts
                
                return frame, new_ts, False
            except StopIteration:
                return None, None, True

        print("Initializing frame synchronization...")
        for i in range(num_images):
            try:
                ts1, f1 = None, None
                for _ in range(200):
                    f1 = next(frame_iters[i])
                    ts1 = get_timestamp(f1.to_image(), robust=True, model=model)
                    if ts1: break
                if not ts1: raise StopIteration("Could not find a valid initial timestamp.")
                
                ts2, f2 = None, None
                for _ in range(200):
                    f2 = next(frame_iters[i])
                    ts2 = get_timestamp(f2.to_image(), robust=True, model=model)
                    if ts2 and ts2 > ts1: break
                if not ts2: raise StopIteration("Could not find a subsequent valid timestamp.")
                
                stream_intervals[i] = ts2 - ts1
                current_frames[i] = f2
                current_timestamps[i] = ts2

            except StopIteration as e:
                stream_ended[i] = True

        if any(stream_ended):
            print("\nError: One or more video streams are empty or failed to initialize. Terminating.", file=sys.stderr)
            for c in in_containers: c.close()
            out_container.close()
            return
            
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        while True:
            if use_sync:
                if any(stream_ended):
                    print("\nA video stream has ended. Terminating process.", file=sys.stderr)
                    break
                
                avg_interval = datetime.timedelta(seconds=np.mean([iv.total_seconds() for iv in stream_intervals]))
                tolerance = datetime.timedelta(seconds=avg_interval.total_seconds() * 1.5)

                while (max(current_timestamps) - min(current_timestamps)) > tolerance:
                    min_ts = min(current_timestamps)
                    min_idx = current_timestamps.index(min_ts)
                    
                    frame, new_ts, ended = advance_stream(min_idx, current_timestamps[min_idx])
                    if ended:
                        stream_ended[min_idx] = True
                        break
                    current_frames[min_idx] = frame
                    current_timestamps[min_idx] = new_ts
                
                if any(stream_ended): continue

                final_group_frames = current_frames
                frame_count += 1
                dt_object = datetime.datetime.fromtimestamp(np.mean([ts.timestamp() for ts in current_timestamps]), tz=pytz.utc)
                status_message = f"Processing synced frame: {frame_count} @ {dt_object.strftime('%Y-%m-%d %H:%M:%S')}"
                sys.stdout.write('\r' + status_message.ljust(os.get_terminal_size().columns - 1))
                sys.stdout.flush()

            else: 
                try:
                    final_group_frames = [next(it) for it in frame_iters]
                    frame_count += 1
                except StopIteration:
                    break

            is_leveling_frame = (use_seam and num_images > 1 and (frame_count == 1 or (frame_count - 1) % level_freq == 0))
            
            worker_args = [
                (
                    (i, final_group_frames[i], mappings[i], final_w, final_h, blend_weights_y[i], blend_weights_uv[i], pad, use_seam, corner_penalties_y[i] if use_seam else None),
                    (frame_y_planes[i], frame_u_planes[i], frame_v_planes[i], frame_weights_y[i], frame_weights_uv[i], frame_penalty_y[i] if use_seam else None)
                ) for i in range(num_images)
            ]
            
            list(executor.map(worker_for_video_frame, worker_args))
            
            y_planes, u_planes, v_planes = frame_y_planes, frame_u_planes, frame_v_planes
            if use_seam:
                y_weight_stack, penalty_stack = frame_weights_y, frame_penalty_y
                if cached_seam_weights is None:
                    cached_seam_weights = _calculate_seam_blend_weights(y_weight_stack, penalty_stack, seam_subsample)
                final_weights_y, final_weights_uv, penalized_scores = cached_seam_weights
                masking_stack = penalized_scores
                if is_leveling_frame:
                    y_indices = np.argmax(penalized_scores, axis=0)
                    y_indices[np.max(frame_weights_y, axis=0) < 1e-9] = -1
                    new_params = solve_corrections(y_indices, y_planes, u_planes, v_planes, num_images, level_subsample)
                    previous_leveling_params = target_leveling_params if target_leveling_params is not None else new_params
                    target_leveling_params = new_params
                    recalc_frame_number = frame_count
                if target_leveling_params:
                    alpha = min((frame_count - recalc_frame_number) / level_freq, 1.0)
                    p_gain, p_y_off, p_u_off, p_v_off = previous_leveling_params
                    t_gain, t_y_off, t_u_off, t_v_off = target_leveling_params
                    interp_gain, interp_y_off, interp_u_off, interp_v_off = (1 - alpha) * p_gain + alpha * t_gain, (1 - alpha) * p_y_off + alpha * t_y_off, (1 - alpha) * p_u_off + alpha * t_u_off, (1 - alpha) * p_v_off + alpha * t_v_off
                    y_planes_leveled, u_planes_leveled, v_planes_leveled = y_planes.astype(np.float32), u_planes.astype(np.float32), v_planes.astype(np.float32)
                    _apply_gain_offset_numba(y_planes_leveled, interp_gain, interp_y_off)
                    _apply_offset_numba(u_planes_leveled, interp_u_off)
                    _apply_offset_numba(v_planes_leveled, interp_v_off)
                    y_planes, u_planes, v_planes = y_planes_leveled, u_planes_leveled, v_planes_leveled
            else: # Feathering
                final_weights_y, final_weights_uv = _calculate_feather_blend_weights(frame_weights_y, frame_weights_uv)
                masking_stack = frame_weights_y

            y_final, u_final, v_final = _blend_final_image(y_planes, u_planes, v_planes, final_weights_y, final_weights_uv, masking_stack)
            out_frame = av.VideoFrame(width=final_w, height=final_h, format='yuv420p')
            out_frame.planes[0].update(y_final); out_frame.planes[1].update(u_final); out_frame.planes[2].update(v_final)
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)

            if use_sync:
                for i in range(num_images):
                    if stream_ended[i]: continue
                    frame, new_ts, ended = advance_stream(i, current_timestamps[i])
                    if ended:
                        stream_ended[i] = True
                        continue
                    current_frames[i] = frame
                    current_timestamps[i] = new_ts

    for packet in out_stream.encode(): out_container.mux(packet)
    out_container.close()
    for c in in_containers: c.close()
    print(f"\nPanoramic video saved to {output_file}")

def main():
    try:
        num_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cores = os.cpu_count() or 1
    print(f"INFO: Detected {num_cores} available CPU cores.")
    numba.set_num_threads(num_cores)
    
    parser = argparse.ArgumentParser(description="Reproject and stitch images or videos into a panorama.")
    parser.add_argument("pto_file", help="Path to the PTO project file.")
    parser.add_argument("input_files", nargs='+', help="One or more input image or video files.")
    parser.add_argument("output_file", help="Path for the output panoramic image or video.")
    parser.add_argument("--pad", type=int, default=32, help="Padding (pixels) for seamless stitching. Defaults to 32.")
    parser.add_argument("--seamless", action='store_true', help="Enable optimal seam finding with feathering and leveling.")
    parser.add_argument("--level-subsample", type=int, default=16, help="Subsampling factor for seam leveling to improve performance. Higher is faster. Defaults to 16.")
    parser.add_argument("--seam-subsample", type=int, default=16, help="Subsampling factor for the seam finder to improve performance. Higher is faster. Defaults to 16.")
    parser.add_argument("--level-freq", type=int, default=8, help="Frequency of seam leveling recalculation for videos. Defaults to 8 frames.")
    parser.add_argument("--sync", action='store_true', help="For videos, synchronize streams by their embedded timestamps before stitching.")
    parser.add_argument("--model", type=str, default=None, help="Specify the model for timestamp extraction.")
    args = parser.parse_args()
    
    if len(args.input_files) < 2:
        args.seamless = False
        args.sync = False
    if not args.input_files:
        print("Error: No input files specified.", file=sys.stderr)
        sys.exit(1)
        
    is_image_input = all(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in args.input_files)
    is_video_input = all(f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')) for f in args.input_files)
    
    try:
        if is_image_input:
            reproject_images(args.pto_file, args.input_files, args.output_file, args.pad, args.seamless, args.level_subsample, args.seam_subsample, num_cores)
        elif is_video_input:
            reproject_videos(args.pto_file, args.input_files, args.output_file, args.pad, args.seamless, args.level_subsample, args.seam_subsample, num_cores, args.level_freq, args.sync, args.model)
        else:
            print("Error: Input files must all be either images or videos.", file=sys.stderr)
            sys.exit(1)
    except (ValueError, FileNotFoundError, ImportError, KeyboardInterrupt) as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
