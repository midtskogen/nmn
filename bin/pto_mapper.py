#!/usr/bin/env python3

import numpy as np
import numba
from numba import prange
import math
import re
import unittest
import os
import sys
import random

# --- PTO Parsing and Core Mapping Logic ---
def parse_pto_file(pto_file):
    """
    Parses a Hugin PTO project file to extract panorama and image parameters.

    Args:
        pto_file (str): Path to the .pto file.

    Returns:
        tuple: A tuple containing (global_options, images), where
               global_options is a dict of panorama settings (p-line) and
               images is a list of dicts for each image's settings (i-lines).
    """
    global_options, images = {}, []
    with open(pto_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            line_type = line[0]
            if line_type not in ('p', 'i', 'm'): continue # Also parse 'm' lines for i-line modifiers
            
            params, content_parts = {}, line.split(' ', 1)
            if len(content_parts) < 2: continue
            content = content_parts[1]
            
            # Regex to handle parameters, including quoted values for filenames (n)
            pattern = r'([a-zA-Z]+)(?:"([^"]*)"|(\S+))'
            
            for key, val_quoted, val_unquoted in re.findall(pattern, content):
                value_str = val_quoted if val_quoted else val_unquoted
                if key == 'S':
                    try:
                        coords = [int(c) for c in value_str.split(',')]
                        if len(coords) == 4: params[key] = (coords[0], coords[1], coords[2], coords[3])
                    except (ValueError, IndexError): print(f"Warning: Malformed crop 'S' parameter: {value_str}")
                    continue
                try:
                    value = float(value_str)
                    if value == int(value): value = int(value)
                    params[key] = value
                except (ValueError, TypeError): params[key] = value_str
            
            if line_type == 'p':
                global_options.update(params)
            elif line_type == 'i':
                images.append(params)
            elif line_type == 'm':
                 # Apply modifier to the last image line
                 if images:
                     images[-1].update(params)

    if not global_options or not images: raise ValueError("Could not parse panorama or image lines from PTO file.")
    return global_options, images

def write_pto_file(pto_data, pto_filename, optimize_vars=None):
    """
    Writes panorama and image parameters to a Hugin PTO project file.

    Args:
        pto_data (tuple): A tuple containing (global_options, images).
        pto_filename (str): Path to the output .pto file.
        optimize_vars (list, optional): A list of variable names to mark for
                                        optimization (e.g., ['p0', 'y0']).
    """
    global_options, images = pto_data

    with open(pto_filename, 'w') as f:
        f.write("# Hugin project file created by pto_mapper\n")
        # Write panorama line (p)
        p_line = ['p']
        p_keys = ['f', 'w', 'h', 'v', 'E', 'R', 'S', 'n', 't', 'r'] # Added 'r' to known keys
        
        # Write known keys in order
        for key in p_keys:
            if key in global_options:
                val = global_options[key]
                if isinstance(val, str) and (' ' in val or not val):
                    p_line.append(f'{key}"{val}"')
                else:
                    p_line.append(f"{key}{val}")
        
        # Write any other keys
        for key, value in global_options.items():
            if key not in p_keys:
                if isinstance(value, str) and (' ' in value or not value):
                    p_line.append(f'{key}"{value}"')
                else:
                    p_line.append(f"{key}{value}")
        f.write(" ".join(p_line) + '\n\n')

        # Write image lines (i)
        i_keys = ['w', 'h', 'f', 'v', 'y', 'p', 'r', 'a', 'b', 'c', 'd', 'e', 'g', 't', 'S', 'n']
        for img in images:
            i_line = ['i']
            for key in i_keys:
                if key in img:
                    val = img[key]
                    if isinstance(val, str) and (' ' in val or not val):
                        i_line.append(f'{key}"{val}"')
                    else:
                        i_line.append(f"{key}{val}")
            
            for key, value in img.items():
                if key not in i_keys:
                    if isinstance(value, str) and (' ' in value or not value):
                         i_line.append(f'{key}"{value}"')
                    else:
                        i_line.append(f"{key}{value}")

            f.write(" ".join(i_line) + '\n')
            
        # Add optimization variables if specified
        if optimize_vars:
            f.write('\n# specify variables that should be optimized\n')
            for var in optimize_vars:
                f.write(f"v {var}\n")
            f.write("v\n") # Add the terminating 'v' line


@numba.njit(fastmath=True, cache=True)
def create_pr_rotation_matrix(p, r):
    """
    Creates a combined pitch and roll rotation matrix. Used internally.
    """
    p_rad, r_rad = math.radians(p), math.radians(r)
    cos_p, sin_p = math.cos(p_rad), math.sin(p_rad)
    cos_r, sin_r = math.cos(r_rad), math.sin(r_rad)
    R_pitch = np.array([[1.,0,0],[0,cos_p,-sin_p],[0,sin_p,cos_p]],dtype=np.float32)
    R_roll = np.array([[cos_r,-sin_r,0],[sin_r,cos_r,0],[0,0,1.]],dtype=np.float32)
    return np.dot(R_pitch, R_roll)

def map_pano_to_image(pto_data, pano_x, pano_y, restrict_to_bounds=False):
    """
    Maps a coordinate from the final panorama to a source image.

    It iterates through all source images and returns the corresponding
    coordinate for the first image that contains the point.

    Args:
        pto_data (tuple): The (global_options, images) tuple from parse_pto_file().
        pano_x (float): The x-coordinate in the panorama.
        pano_y (float): The y-coordinate in the panorama.
        restrict_to_bounds (bool): If True, returns None for coordinates that
            fall outside the source image's dimensions. Defaults to False.

    Returns:
        tuple: A tuple (image_number, x, y) if a mapping is found.
        None: If the panorama coordinate does not map to any source image.
    """
    global_options, images = pto_data
    orig_w, orig_h = global_options.get('w'), global_options.get('h')
    pano_proj_f = int(global_options.get('f', 2))
    pano_hfov = global_options.get('v')
    pano_r = global_options.get('r', 0.0)
    if orig_w is None or orig_h is None or pano_hfov is None:
        raise ValueError("PTO 'p' line must contain 'w', 'h', and 'v' parameters.")

    if pano_r != 0.0:
        pano_r_rad = math.radians(pano_r)
        cos_r, sin_r = math.cos(pano_r_rad), math.sin(pano_r_rad)
        center_x, center_y = orig_w / 2.0, orig_h / 2.0
        translated_x, translated_y = pano_x - center_x, pano_y - center_y
        rotated_x = translated_x * cos_r - translated_y * sin_r
        rotated_y = translated_x * sin_r + translated_y * cos_r
        pano_x, pano_y = rotated_x + center_x, rotated_y + center_y

    pano_hfov_rad = math.radians(pano_hfov)

    vec_3d = np.empty(3, dtype=np.float32)
    if pano_proj_f == 2:  # Equirectangular
        pano_yaw = (pano_x / orig_w - 0.5) * 2.0 * math.pi
        pitch = -(pano_y / orig_h - 0.5) * math.pi
        cos_pitch = math.cos(pitch)
        vec_3d[:] = cos_pitch * math.sin(pano_yaw), math.sin(pitch), -cos_pitch * math.cos(pano_yaw)
    elif pano_proj_f == 0:  # Rectilinear
        pano_focal = (orig_w / 2.0) / math.tan(pano_hfov_rad / 2.0) if pano_hfov_rad > 0 else 1.0
        x_norm, y_norm = pano_x - orig_w / 2.0, -(pano_y - orig_h / 2.0)
        norm = math.sqrt(x_norm**2 + y_norm**2 + pano_focal**2)
        if norm == 0: norm = 1.0
        vec_3d[:] = x_norm / norm, y_norm / norm, -pano_focal / norm
    elif pano_proj_f == 3: # Fisheye
        x_norm = pano_x - orig_w / 2.0
        y_norm = -(pano_y - orig_h / 2.0)
        r = math.sqrt(x_norm**2 + y_norm**2)
        fisheye_focal = (orig_w / 2.0) / (pano_hfov_rad / 2.0) if pano_hfov_rad > 0 else 1.0
        theta = r / fisheye_focal
        if theta > math.pi: return None
        phi = math.atan2(y_norm, x_norm)
        sin_theta, cos_theta = math.sin(theta), math.cos(theta)
        vec_3d[:] = sin_theta * math.cos(phi), sin_theta * math.sin(phi), -cos_theta
    else:
        return None

    for i, img in enumerate(images):
        sw, sh, fov, src_proj_f = img.get('w'), img.get('h'), img.get('v'), int(img.get('f', 0))
        fov_rad = math.radians(fov)
        
        src_focal = 0
        if src_proj_f == 0: src_focal = sw / (2*math.tan(fov_rad/2)) if fov_rad > 0 else 0
        elif src_proj_f == 3: src_focal = sw / fov_rad if fov_rad > 0 else 0
        else: continue

        y, p, r = img.get('y', 0), img.get('p', 0), -img.get('r', 0)
        a, b, c = img.get('a', 0), img.get('b', 0), img.get('c', 0)
        cx, cy = -img.get('d', 0), img.get('e', 0)
        
        R_pr_inv = create_pr_rotation_matrix(p, r).T
        camera_yaw_rad = math.radians(y)
        
        world_pitch = math.asin(np.clip(vec_3d[1], -1.0, 1.0))
        world_yaw = math.atan2(vec_3d[0], -vec_3d[2])
        adjusted_yaw = world_yaw - camera_yaw_rad
        cos_pitch = math.cos(world_pitch)
        vec_3d_adjusted = np.array([cos_pitch*math.sin(adjusted_yaw), vec_3d[1], -cos_pitch*math.cos(adjusted_yaw)], dtype=np.float32)
        vec_rot = np.dot(R_pr_inv, vec_3d_adjusted)

        x_rot, y_rot, z_rot = vec_rot[0], vec_rot[1], vec_rot[2]
        is_valid_src = True
        if src_proj_f == 0:
            if z_rot >= -1e-6: is_valid_src = False
            else: x_ideal, y_ideal = src_focal * x_rot / (-z_rot), src_focal * y_rot / (-z_rot)
        elif src_proj_f == 3:
            theta = math.atan2(math.sqrt(x_rot**2+y_rot**2), -z_rot)
            phi = math.atan2(y_rot,x_rot)
            r_ideal = src_focal*theta
            x_ideal, y_ideal = r_ideal * math.cos(phi), r_ideal * math.sin(phi)
        else: is_valid_src = False
        
        if not is_valid_src: continue

        src_norm_radius = min(sw,sh)/2.
        r_ideal_val = math.sqrt(x_ideal**2 + y_ideal**2)
        mag = 1.0
        
        if src_norm_radius > 1e-6:
            r_norm = r_ideal_val / src_norm_radius
            d_coeff = 1.0 - (a + b + c)
            derivative = d_coeff + r_norm * (2.0*c + r_norm * (3.0*b + r_norm * 4.0*a))
            if derivative < 0.0: continue
            mag = d_coeff + r_norm * (c + r_norm * (b + r_norm * a))
        
        x_dist, y_dist = x_ideal * mag, y_ideal * mag
        sx, sy = (x_dist - cx) + sw / 2.0, -(y_dist - cy) + sh / 2.0

        if not restrict_to_bounds or (0 <= sx < sw and 0 <= sy < sh):
            return i, sx, sy

    return None

def map_image_to_pano(pto_data, image_index, x, y):
    """
    Maps a coordinate from a source image to the final panorama.
    """
    global_options, images = pto_data
    if image_index >= len(images):
        raise IndexError("Image index is out of bounds.")

    img = images[image_index]
    pano_proj_f = int(global_options.get('f', 2))
    pano_hfov = global_options.get('v')
    orig_w, orig_h = global_options.get('w'), global_options.get('h')
    pano_r = global_options.get('r', 0.0)
    if orig_w is None or orig_h is None or pano_hfov is None:
        raise ValueError("PTO 'p' line must contain 'w', 'h', and 'v' parameters.")

    sw, sh = img.get('w'), img.get('h')
    cx, cy = -img.get('d', 0), img.get('e', 0)
    
    x_dist_centered = (x - sw / 2.0) + cx
    y_dist_centered = -(y - sh / 2.0) + cy
    r_dist = math.sqrt(x_dist_centered**2 + y_dist_centered**2)
    
    a, b, c = img.get('a', 0), img.get('b', 0), img.get('c', 0)
    r_ideal = r_dist
    
    if abs(a) > 1e-9 or abs(b) > 1e-9 or abs(c) > 1e-9:
        src_norm_radius = min(sw, sh) / 2.0
        if src_norm_radius > 1e-6:
            a_ = a / (src_norm_radius**3); b_ = b / (src_norm_radius**2)
            c_ = c / src_norm_radius; d_ = 1.0 - (a + b + c)
            for _ in range(10):
                r_ideal_2 = r_ideal**2; r_ideal_3 = r_ideal_2 * r_ideal
                f_r = (a_ * r_ideal_3 * r_ideal) + (b_ * r_ideal_3) + (c_ * r_ideal_2) + (d_ * r_ideal) - r_dist
                f_prime_r = (4.0 * a_ * r_ideal_3) + (3.0 * b_ * r_ideal_2) + (2.0 * c_ * r_ideal) + d_
                if abs(f_prime_r) < 1e-9: break
                r_ideal -= f_r / f_prime_r

    if r_dist > 1e-9:
        ratio = r_ideal / r_dist
        x_ideal, y_ideal = x_dist_centered * ratio, y_dist_centered * ratio
    else:
        x_ideal, y_ideal = 0.0, 0.0

    fov, src_proj_f = img.get('v'), int(img.get('f', 0))
    fov_rad = math.radians(fov)
    vec_cam = np.empty(3, dtype=np.float32)

    if src_proj_f == 0:
        src_focal = sw / (2 * math.tan(fov_rad / 2)) if fov_rad > 0 else 0
        vec_cam[:] = x_ideal, y_ideal, -src_focal
    elif src_proj_f == 3:
        src_focal = sw / fov_rad if fov_rad > 0 else 0
        r_theta = math.sqrt(x_ideal**2 + y_ideal**2)
        theta = r_theta / src_focal if src_focal > 1e-6 else 0
        phi = math.atan2(y_ideal, x_ideal)
        vec_cam[:] = math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), -math.cos(theta)
    else:
        return None

    norm = np.linalg.norm(vec_cam)
    if norm > 1e-6: vec_cam /= norm

    p, r, yaw = img.get('p', 0), -img.get('r', 0), img.get('y', 0)
    R_pr = create_pr_rotation_matrix(p, r)
    vec_3d_adjusted = np.dot(R_pr, vec_cam)

    pitch_adjusted = math.asin(np.clip(vec_3d_adjusted[1], -1.0, 1.0))
    yaw_adjusted = math.atan2(vec_3d_adjusted[0], -vec_3d_adjusted[2])

    world_yaw = yaw_adjusted + math.radians(yaw)
    cos_world_pitch = math.cos(pitch_adjusted)
    world_x = cos_world_pitch * math.sin(world_yaw)
    world_y = math.sin(pitch_adjusted)
    world_z = -cos_world_pitch * math.cos(world_yaw)
    
    pano_hfov_rad = math.radians(pano_hfov)
    unrotated_pano_x, unrotated_pano_y = None, None
    if pano_proj_f == 2:
        pano_yaw = math.atan2(world_x, -world_z)
        pano_pitch = math.asin(world_y)
        unrotated_pano_x = (pano_yaw / (2 * math.pi) + 0.5) * orig_w
        unrotated_pano_y = (-pano_pitch / math.pi + 0.5) * orig_h
    elif pano_proj_f == 0:
        pano_focal = (orig_w / 2.0) / math.tan(pano_hfov_rad / 2.0) if pano_hfov_rad > 0 else 1.0
        if world_z >= 0: return None
        unrotated_pano_x = world_x * (-pano_focal / world_z) + orig_w / 2.0
        unrotated_pano_y = -(world_y * (-pano_focal / world_z)) + orig_h / 2.0
    elif pano_proj_f == 3:
        fisheye_focal = (orig_w / 2.0) / (pano_hfov_rad / 2.0) if pano_hfov_rad > 0 else 1.0
        theta = math.atan2(math.sqrt(world_x**2 + world_y**2), -world_z)
        phi = math.atan2(world_y, world_x)
        r_pano = theta * fisheye_focal
        unrotated_pano_x = r_pano * math.cos(phi) + orig_w / 2.0
        unrotated_pano_y = -(r_pano * math.sin(phi)) + orig_h / 2.0
    else:
        return None

    if unrotated_pano_x is None:
        return None

    if pano_r != 0.0:
        pano_r_rad = math.radians(-pano_r) # Use negative angle for inverse rotation
        cos_r, sin_r = math.cos(pano_r_rad), math.sin(pano_r_rad)
        center_x, center_y = orig_w / 2.0, orig_h / 2.0
        translated_x, translated_y = unrotated_pano_x - center_x, unrotated_pano_y - center_y
        rotated_x = translated_x * cos_r - translated_y * sin_r
        rotated_y = translated_x * sin_r + translated_y * cos_r
        return rotated_x + center_x, rotated_y + center_y
    else:
        return unrotated_pano_x, unrotated_pano_y

@numba.njit(parallel=True, fastmath=True, cache=True)
def calculate_source_coords(coords_y, final_w, final_h, orig_w, orig_h, crop_offset_x, crop_offset_y, pano_proj_f, pano_hfov, sw, sh, R_pr_inv, camera_yaw, src_focal, src_norm_radius, a, b, c, cx, cy, src_proj_f, pano_r=0.0):
    """
    This is the core JIT-compiled reprojection function.
    It is kept here to be used by the stitching process. It maps a grid of
    panorama coordinates to a single source image's coordinate system.
    """
    pano_hfov_rad = math.radians(pano_hfov)
    pano_focal = (orig_w / 2.0) / math.tan(pano_hfov_rad / 2.0) if pano_hfov_rad > 0 else 1.
    camera_yaw_rad = math.radians(camera_yaw)
    INVALID_COORD = -99999.0

    has_rotation = pano_r != 0.0
    if has_rotation:
        pano_r_rad = math.radians(pano_r)
        cos_r, sin_r = math.cos(pano_r_rad), math.sin(pano_r_rad)
        center_x, center_y = orig_w / 2.0, orig_h / 2.0

    for y_dest in prange(final_h):
        for x_dest in range(final_w):
            pano_x, pano_y = x_dest + crop_offset_x, y_dest + crop_offset_y
            
            if has_rotation:
                translated_x, translated_y = pano_x - center_x, pano_y - center_y
                rotated_x = translated_x * cos_r - translated_y * sin_r
                rotated_y = translated_x * sin_r + translated_y * cos_r
                pano_x, pano_y = rotated_x + center_x, rotated_y + center_y

            vec_3d = np.empty(3, dtype=np.float32)
            is_valid_pano_pixel = True
            
            if pano_proj_f == 2: # Equirectangular
                pano_yaw = (pano_x / orig_w - 0.5) * 2.0 * math.pi
                pitch = -(pano_y / orig_h - 0.5) * math.pi
                cos_pitch = math.cos(pitch)
                vec_3d[0], vec_3d[1], vec_3d[2] = cos_pitch * math.sin(pano_yaw), math.sin(pitch), -cos_pitch * math.cos(pano_yaw)
            elif pano_proj_f == 0: # Rectilinear
                x_norm, y_norm = pano_x - orig_w / 2.0, -(pano_y - orig_h / 2.0)
                norm = math.sqrt(x_norm**2 + y_norm**2 + pano_focal**2)
                if norm == 0: norm = 1.
                vec_3d[0], vec_3d[1], vec_3d[2] = x_norm / norm, y_norm / norm, -pano_focal / norm
            elif pano_proj_f == 3: # Fisheye
                x_norm = pano_x - orig_w / 2.0
                y_norm = -(pano_y - orig_h / 2.0)
                r = math.sqrt(x_norm**2 + y_norm**2)
                fisheye_f = (orig_w / 2.0) / (pano_hfov_rad / 2.0) if pano_hfov_rad > 0 else 1.
                if r > fisheye_f * math.pi: is_valid_pano_pixel = False
                else:
                    theta, phi, sin_theta = r / fisheye_f, math.atan2(y_norm, x_norm), math.sin(r/fisheye_f)
                    vec_3d[0], vec_3d[1], vec_3d[2] = sin_theta*math.cos(phi), sin_theta*math.sin(phi), -math.cos(theta)
            else:
                is_valid_pano_pixel = False

            if not is_valid_pano_pixel:
                coords_y[y_dest,x_dest,0], coords_y[y_dest,x_dest,1] = INVALID_COORD, INVALID_COORD
                continue
            
            world_pitch, world_yaw = math.asin(vec_3d[1]), math.atan2(vec_3d[0], -vec_3d[2])
            adjusted_yaw, cos_pitch = world_yaw - camera_yaw_rad, math.cos(world_pitch)
            vec_3d_adjusted = np.array([cos_pitch*math.sin(adjusted_yaw), vec_3d[1], -cos_pitch*math.cos(adjusted_yaw)], dtype=np.float32)
            vec_rot = np.dot(R_pr_inv, vec_3d_adjusted)
            
            x_rot,y_rot,z_rot,x_ideal,y_ideal,is_valid_src = vec_rot[0],vec_rot[1],vec_rot[2], 0., 0., True
            if src_proj_f == 0:
                if z_rot >= -1e-6: is_valid_src = False
                else: x_ideal, y_ideal = src_focal * x_rot / (-z_rot), src_focal * y_rot / (-z_rot)
            elif src_proj_f == 3:
                theta = math.atan2(math.sqrt(x_rot**2+y_rot**2),-z_rot)
                phi, r_ideal = math.atan2(y_rot,x_rot), src_focal*theta
                x_ideal, y_ideal = r_ideal * math.cos(phi), r_ideal * math.sin(phi)
            else: is_valid_src = False
            
            if not is_valid_src:
                coords_y[y_dest,x_dest,0], coords_y[y_dest,x_dest,1] = INVALID_COORD, INVALID_COORD
                continue

            r_ideal_val = math.sqrt(x_ideal**2 + y_ideal**2)
            mag = 1.0
            is_valid_dist = True

            if src_norm_radius > 1e-6:
                r_norm = r_ideal_val / src_norm_radius
                d_coeff = 1.0 - (a + b + c)
                derivative = d_coeff + r_norm * (2.0*c + r_norm * (3.0*b + r_norm * 4.0*a))
                if derivative < 0.0:
                    is_valid_dist = False
                else:
                    mag = d_coeff + r_norm * (c + r_norm * (b + r_norm * a))

            if not is_valid_dist:
                coords_y[y_dest, x_dest, 0] = INVALID_COORD
                coords_y[y_dest, x_dest, 1] = INVALID_COORD
            else:
                x_dist, y_dist = x_ideal * mag, y_ideal * mag
                x_shifted, y_shifted = x_dist - cx, y_dist - cy
                coords_y[y_dest, x_dest, 0] = x_shifted + sw / 2.0
                coords_y[y_dest, x_dest, 1] = -y_shifted + sh / 2.0

class TestPtoMapping(unittest.TestCase):
    # Class variable to hold the command-line override file
    pto_file_override = None
    
    def setUp(self):
        """
        Create a dummy PTO file for testing, or use the one
        provided via command line.
        """
        self.created_dummy_file = False
        if TestPtoMapping.pto_file_override:
            self.pto_filename = TestPtoMapping.pto_file_override
            print(f"\n--- Using provided test file: {self.pto_filename} ---")
        else:
            self.pto_filename = "test_project.pto"
            self.created_dummy_file = True
            pto_content = ""
            with open(self.pto_filename, "w") as f:
                f.write(pto_content)

    def tearDown(self):
        """Remove the dummy PTO file only if it was created by setUp."""
        if self.created_dummy_file:
            os.remove(self.pto_filename)

    def test_roundtrip_mapping_randomized(self):
        """
        Tests mapping from panorama to image and back for 100000 random
        points. Points that don't map to an image are ignored.
        Failures are collected and reported at the end.
        """
        pto_data = parse_pto_file(self.pto_filename)
        
        global_options = pto_data[0]
        pano_w = global_options['w']
        pano_h = global_options['h']
        pano_proj_f = int(global_options.get('f', 2))
        
        num_to_generate = 100000
        mapped_points_tested = 0
        successful_roundtrips = 0
        failures = []
        
        print(f"\n--- Testing {num_to_generate} random coordinates (w={pano_w}, h={pano_h}) ---")
        
        for i in range(num_to_generate):
            orig_pano_x = random.uniform(0, pano_w)
            orig_pano_y = random.uniform(0, pano_h)
            
            # For equirectangular, exclude the poles to avoid instability
            if pano_proj_f == 2:
                # pitch_deg = (0.5 - y/h) * 180
                pitch_deg = (0.5 - (orig_pano_y / pano_h)) * 180.0
                if abs(pitch_deg) > 89.0:
                    continue

            # For testing, we must restrict to bounds to have a valid round-trip
            pano_to_img_result = map_pano_to_image(pto_data, orig_pano_x, orig_pano_y, restrict_to_bounds=True)
            
            if pano_to_img_result is None:
                continue
            
            mapped_points_tested += 1
            img_idx, src_x, src_y = pano_to_img_result
            
            img_to_pano_result = map_image_to_pano(pto_data, img_idx, src_x, src_y)
            if img_to_pano_result is None:
                failures.append(f"Point {i+1} failed: Reverse mapping returned None for img {img_idx} at ({src_x:.2f}, {src_y:.2f})")
                continue

            final_pano_x, final_pano_y = img_to_pano_result
            
            # Check if the coordinates match within a 1.0 pixel tolerance
            x_ok = abs(orig_pano_x - final_pano_x) <= 1.0
            y_ok = abs(orig_pano_y - final_pano_y) <= 1.0

            if x_ok and y_ok:
                successful_roundtrips += 1
            else:
                msg = f"Point {i+1} failed:"
                if not x_ok:
                    msg += f" X mismatch (Orig: {orig_pano_x:.2f}, Final: {final_pano_x:.2f})"
                if not y_ok:
                    msg += f" Y mismatch (Orig: {orig_pano_y:.2f}, Final: {final_pano_y:.2f})"
                failures.append(msg)
        
        # --- Final Report ---
        print(f"--- Test Summary ---")
        print(f"Total points generated: {num_to_generate}")
        print(f"Points that mapped to an image: {mapped_points_tested}")
        print(f"Successful round-trips: {successful_roundtrips}")
        print(f"Failed round-trips: {len(failures)}")

        # Don't fail if no points landed in the project, as some PTOs are sparse
        if mapped_points_tested == 0:
            print("Warning: No random points mapped to any source image. Test could not be completed.")
            return

        if failures:
            # Report the first 10 failures for brevity
            report = "\n".join(failures[:10])
            self.fail(f"{len(failures)} round-trip mapping failures occurred:\n{report}")


if __name__ == '__main__':
    num_args = len(sys.argv)

    # Dispatch based on number of command-line arguments
    if num_args not in [2, 4, 5]:
        # Print usage information if the number of arguments is incorrect
        print("Usage:")
        print(f"  Run unit test: python {sys.argv[0]} <pto_file>")
        print(f"  Pano -> Image: python {sys.argv[0]} <pto_file> <pano_x> <pano_y> [--restrict]")
        print(f"  Image -> Pano: python {sys.argv[0]} <pto_file> <src_x> <src_y> <image_index>")
        sys.exit(1)

    pto_path = sys.argv[1]
    if not os.path.exists(pto_path):
        print(f"Error: PTO file not found at '{pto_path}'")
        sys.exit(1)

    # Mode 1: Run the unit test with the given pto file
    if num_args == 2:
        TestPtoMapping.pto_file_override = pto_path
        # Remove the argument from sys.argv for unittest framework
        sys.argv.pop(1)
        unittest.main()
        sys.exit(0)

    # For direct mapping, parse the PTO file first
    try:
        pto_data = parse_pto_file(pto_path)
    except Exception as e:
        print(f"Error parsing PTO file: {e}")
        sys.exit(1)

    # Determine which mapping mode is being invoked
    is_pano_to_image = (num_args == 4) or (num_args == 5 and sys.argv[4] == '--restrict')

    if is_pano_to_image:
        # Mode 2: Forward mapping (Panorama -> Image)
        try:
            pano_x = float(sys.argv[2])
            pano_y = float(sys.argv[3])
        except ValueError:
            print("Error: Panorama coordinates <pano_x> and <pano_y> must be numbers.")
            sys.exit(1)
        
        restrict_bounds = (num_args == 5 and sys.argv[4] == '--restrict')
        result = map_pano_to_image(pto_data, pano_x, pano_y, restrict_to_bounds=restrict_bounds)
        
        if result:
            print(f"{result[1]} {result[2]} {result[0]}")
        else:
            print("None")

    elif num_args == 5:
        # Mode 3: Reverse mapping (Image -> Panorama)
        try:
            src_x = float(sys.argv[2])
            src_y = float(sys.argv[3])
            img_idx = int(sys.argv[4])
        except ValueError:
            print("Error: Source coordinates <src_x>, <src_y> must be numbers and <image_index> must be an integer.")
            sys.exit(1)
            
        try:
            result = map_image_to_pano(pto_data, img_idx, src_x, src_y)
            if result:
                print(f"{result[0]} {result[1]}")
            else:
                print("None")
        except IndexError:
            print(f"Error: Image index {img_idx} is out of bounds.")
            sys.exit(1)
