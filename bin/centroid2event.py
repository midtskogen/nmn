#!/usr/bin/python3
"""
Generates an event file from a Hugin .pto project file and a centroid data file.
This script maps celestial coordinates (azimuth, altitude) from the centroid file
to pixel coordinates within a specified source image of the panorama.
"""

import argparse
import math
from datetime import datetime
import numpy as np

# Assuming pto_mapper.py is in the same directory or accessible in the Python path
import pto_mapper

def az_alt_to_pano_coord(az, alt, p_line):
    """
    Converts celestial coordinates (az, alt) to panorama pixel coordinates (pano_x, pano_y)
    using the linear scaling defined by the panorama's dimensions and FOV.
    """
    pano_w = p_line.get('w')
    pano_h = p_line.get('h')
    # For an f2 projection (equirectangular), HFOV is 360 and VFOV is 180.
    pano_hfov = p_line.get('v', 360.0)
    pano_vfov = 180.0

    # Calculate pixels per degree. e.g., 36000px / 360deg = 100 px/deg
    pixels_per_deg_x = pano_w / pano_hfov
    pixels_per_deg_y = pano_h / pano_vfov

    # Calculate panorama coordinates based on this linear scaling.
    pano_x = az * pixels_per_deg_x
    # Y=0 is Alt=90. Convert altitude to zenith distance (angle from the top pole).
    zenith_angle = 90.0 - alt
    pano_y = zenith_angle * pixels_per_deg_y

    return pano_x, pano_y

def map_pano_to_image_coord(pto_data, pano_x, pano_y, image_index):
    """
    Maps a panorama coordinate (pano_x, pano_y) to a source image coordinate (sx, sy)
    using the full geometric model from the PTO file.
    """
    global_options, images = pto_data
    if image_index >= len(images):
        raise IndexError(f"Image index {image_index} is out of bounds.")

    # Step 1: Convert panorama coordinate to a 3D vector.
    # This logic is taken from pto_mapper.map_pano_to_image
    orig_w, orig_h = global_options.get('w'), global_options.get('h')
    pano_proj_f = int(global_options.get('f', 2))
    pano_hfov = global_options.get('v')
    pano_hfov_rad = math.radians(pano_hfov)

    vec_3d = np.empty(3, dtype=np.float32)
    if pano_proj_f == 2:  # Equirectangular
        pano_yaw = (pano_x / orig_w - 0.5) * 2.0 * math.pi
        pitch = -(pano_y / orig_h - 0.5) * math.pi
        cos_pitch = math.cos(pitch)
        vec_3d[:] = cos_pitch * math.sin(pano_yaw), math.sin(pitch), -cos_pitch * math.cos(pano_yaw)
    else:
        # This implementation currently only supports equirectangular panoramas
        return None

    # The rest of this function faithfully reproduces the logic for a single image
    # from the original pto_mapper.map_pano_to_image function.

    img = images[image_index]
    sw, sh = img.get('w'), img.get('h')
    fov, src_proj_f = img.get('v'), int(img.get('f', 0))
    fov_rad = math.radians(fov)
    
    src_focal = 0
    if src_proj_f == 0: src_focal = sw / (2 * math.tan(fov_rad / 2)) if fov_rad > 0 else 0
    elif src_proj_f == 3: src_focal = sw / fov_rad if fov_rad > 0 else 0
    else: return None

    # Step 2: Rotate the 3D vector using the original pto_mapper logic.
    y, p, r = img.get('y', 0), img.get('p', 0), -img.get('r', 0)
    a, b, c = img.get('a', 0), img.get('b', 0), img.get('c', 0)
    cx, cy = -img.get('d', 0), img.get('e', 0)
    
    R_pr_inv = pto_mapper.create_pr_rotation_matrix(p, r).T
    camera_yaw_rad = math.radians(y)
    
    world_pitch = math.asin(np.clip(vec_3d[1], -1.0, 1.0))
    world_yaw = math.atan2(vec_3d[0], -vec_3d[2])
    adjusted_yaw = world_yaw - camera_yaw_rad
    cos_pitch_adj = math.cos(world_pitch)
    
    vec_3d_adjusted = np.array([
        cos_pitch_adj * math.sin(adjusted_yaw),
        vec_3d[1],
        -cos_pitch_adj * math.cos(adjusted_yaw)
    ], dtype=np.float32)
    
    vec_rot = np.dot(R_pr_inv, vec_3d_adjusted)

    # Step 3: Project the rotated vector into the source image's 2D plane.
    x_rot, y_rot, z_rot = vec_rot[0], vec_rot[1], vec_rot[2]
    
    x_ideal, y_ideal = 0., 0.
    is_valid_src = True
    if src_proj_f == 0:
        if z_rot >= -1e-6: is_valid_src = False
        else: x_ideal, y_ideal = src_focal * x_rot / (-z_rot), src_focal * y_rot / (-z_rot)
    elif src_proj_f == 3:
        theta = math.atan2(math.sqrt(x_rot**2 + y_rot**2), -z_rot)
        phi = math.atan2(y_rot, x_rot)
        r_ideal = src_focal * theta
        x_ideal, y_ideal = r_ideal * math.cos(phi), r_ideal * math.sin(phi)
    else: is_valid_src = False
    if not is_valid_src: return None

    # Step 4: Apply lens distortion to find the final pixel coordinate.
    src_norm_radius = min(sw, sh) / 2.0
    r_ideal_val = math.sqrt(x_ideal**2 + y_ideal**2)
    mag = 1.0
    if src_norm_radius > 1e-6:
        r_norm = r_ideal_val / src_norm_radius
        d_coeff = 1.0 - (a + b + c)
        derivative = d_coeff + r_norm * (2.0*c + r_norm * (3.0*b + r_norm * 4.0*a))
        if derivative < 0.0: return None
        mag = d_coeff + r_norm * (c + r_norm * (b + r_norm * a))

    x_dist, y_dist = x_ideal * mag, y_ideal * mag
    sx, sy = (x_dist - cx) + sw / 2.0, -(y_dist - cy) + sh / 2.0
    
    return sx, sy

def midpoint(az1, alt1, az2, alt2):
    """Calculates the midpoint and angular distance of a great-circle arc."""
    lon1, lat1 = math.radians(az1), math.radians(alt1)
    lon2, lat2 = math.radians(az2), math.radians(alt2)

    Bx = math.cos(lat2) * math.cos(lon2 - lon1)
    By = math.cos(lat2) * math.sin(lon2 - lon1)

    lat3 = math.atan2(math.sin(lat1) + math.sin(lat2),
                      math.sqrt((math.cos(lat1) + Bx)**2 + By**2))
    lon3 = lon1 + math.atan2(By, math.cos(lat1) + Bx)

    a = math.sin((lat2 - lat1) / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return math.degrees(lon3), math.degrees(lat3), math.degrees(c)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Make an event file from a lens file and centroid file.')
    parser.add_argument('-i', '--image', dest='image', help='which image in the .pto file to use (default: 0)', default=0, type=int)
    parser.add_argument(action='store', dest='ptofile', help='input .pto file')
    parser.add_argument(action='store', dest='centroid', help='input centroid file')
    args = parser.parse_args()

    try:
        pto_data = pto_mapper.parse_pto_file(args.ptofile)
        global_options = pto_data[0]
    except Exception as e:
        print(f"Error: Could not parse PTO file '{args.ptofile}'.\n{e}")
        return

    try:
        img = pto_data[1][args.image]
        width = img.get('w')
        height = img.get('h')
    except IndexError:
        print(f"Error: Image index {args.image} is out of range for the provided PTO file.")
        return

    timestamps, timestamps_full, coordinates, positions = [], [], [], []

    try:
        with open(args.centroid, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) < 8: continue

                az, alt = float(parts[3]), float(parts[2])

                # STEP 1: Convert (az, alt) to linear panorama coords
                pano_x, pano_y = az_alt_to_pano_coord(az, alt, global_options)
                
                # STEP 2: Map panorama coords to the source image
                mapped_pos = map_pano_to_image_coord(pto_data, pano_x, pano_y, args.image)
                
                if mapped_pos:
                    positions.append(f"{mapped_pos[0]:.2f},{mapped_pos[1]:.2f}")
                    coordinates.append(f"{az},{alt}")
                    
                    ts_str = f"{parts[6]} {parts[7]}"
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                    epoch_time = (ts - datetime(1970, 1, 1)).total_seconds()
                    
                    timestamps.append(str(epoch_time))
                    timestamps_full.append(f"{ts_str} {parts[8]}")

    except FileNotFoundError:
        print(f"Error: Centroid file not found at '{args.centroid}'")
        return
    except (ValueError, IndexError) as e:
        print(f"Error processing centroid file. Please check format. Details: {e}")
        return

    if not positions:
        print("Warning: No valid positions could be mapped. Check your input files and image index.")
        return

    # --- Print Output ---
    print("[trail]")
    print(f"frames = {len(positions)}")
    if len(timestamps) > 1:
        duration = float(timestamps[-1]) - float(timestamps[0])
        print(f"duration = {duration:.2f}")
    else:
        print("duration = 0.00")
    print(f"positions = {' '.join(positions)}")
    print(f"coordinates = {' '.join(coordinates)}")
    print(f"timestamps = {' '.join(timestamps)}")

    first_coord = [float(c) for c in coordinates[0].split(',')]
    last_coord = [float(c) for c in coordinates[-1].split(',')]
    mid_az, mid_alt, arc = midpoint(first_coord[0], first_coord[1], last_coord[0], last_coord[1])
    
    print(f"midpoint = {mid_az:.2f},{mid_alt:.2f}")
    print(f"arc = {arc:.2f}")

    zeros = " ".join(["0"] * len(coordinates))
    print(f"brightness = {zeros}")
    print(f"frame_brightness = {zeros}")
    print(f"size = {zeros}")
    
    print("\n[video]")
    print(f"start = {timestamps_full[0]} ({timestamps[0]})")
    print(f"end = {timestamps_full[-1]} ({timestamps[-1]})")
    print(f"width = {width}")
    print(f"height = {height}")

if __name__ == '__main__':
    main()
