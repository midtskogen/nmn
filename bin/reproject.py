#!/usr/bin/env python3

import math
import argparse
import numpy as np
from collections import OrderedDict
import pto_mapper

def euler_to_matrix(yaw_deg, pitch_deg, roll_deg):
    """Converts Euler angles (degrees) to a 3x3 rotation matrix (Y-X-Z order)."""
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz

def matrix_to_euler(matrix):
    """Extracts Euler angles (degrees) from a 3x3 rotation matrix."""
    sp = -matrix[1, 2]
    sp = np.clip(sp, -1.0, 1.0)
    pitch_deg = np.rad2deg(np.arcsin(sp))
    if np.isclose(np.abs(sp), 1.0):
        roll_deg, yaw_deg = 0.0, np.rad2deg(np.arctan2(-matrix[2, 0], matrix[0, 0]))
    else:
        cp = np.cos(np.deg2rad(pitch_deg))
        yaw_deg = np.rad2deg(np.arctan2(matrix[0, 2] / cp, matrix[2, 2] / cp))
        roll_deg = np.rad2deg(np.arctan2(matrix[1, 0] / cp, matrix[1, 1] / cp))
    return yaw_deg, pitch_deg, roll_deg

# --- PTO File Writing ---

def _build_line(params):
    """Constructs a .pto line from a dictionary of parameters."""
    line_type = params.get('_type', 'i')
    parts = [line_type]
    for key, value in params.items():
        if key == '_type': continue
        if isinstance(value, str):
            parts.append(f'{key}"{value}"')
        elif isinstance(value, float):
            parts.append(f'{key}{value:.8f}')
        else:
            parts.append(f'{key}{value}')
    return ' '.join(parts)

def write_pto(filename, global_options, images):
    """Writes project data to a .pto file."""
    with open(filename, 'w', encoding='utf-8') as f:
        if global_options:
            f.write(_build_line(global_options) + '\n')
        for img in images:
            f.write(_build_line(img) + '\n')

# --- Main Script Logic ---

def pos(s):
    """Custom argparse type for 'az,alt' coordinates."""
    try:
        az, alt = map(float, s.split(','))
        return az, alt
    except:
        raise argparse.ArgumentTypeError("Position must be in 'az,alt' format")

def midpoint(az1, alt1, az2, alt2):
    """Calculates the midpoint and arc distance on a sphere."""
    x1, y1 = math.radians(az1), math.radians(alt1)
    x2, y2 = math.radians(az2), math.radians(alt2)
    Bx, By = math.cos(y2) * math.cos(x2 - x1), math.cos(y2) * math.sin(x2 - x1)
    y3_den_sq = (math.cos(y1) + Bx)**2 + By**2
    y3 = math.atan2(math.sin(y1) + math.sin(y2), math.sqrt(y3_den_sq))
    x3 = x1 + math.atan2(By, math.cos(y1) + Bx)
    a = math.sin((y2 - y1) / 2)**2 + math.cos(y1) * math.cos(y2) * math.sin((x2 - x1) / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return math.degrees(x3), math.degrees(y3), math.degrees(c)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Change projection and centre of a .pto file.')
    parser.add_argument('infile', help="Source .pto file")
    parser.add_argument('pos', type=pos, help="Center position: az,alt")
    parser.add_argument('-e', '--endpos', type=pos, help="End position (for FOV calculation)")
    parser.add_argument('-o', '--output', help='Destination .pto file')
    parser.add_argument('-g', '--grid', dest='gridfile', help='Generate a grid .pto file')
    parser.add_argument('-f', '--fov', type=float, default=0.0, help='Horizontal field of view (default: auto)')
    parser.add_argument('--width', type=int, default=2560, help='Destination canvas width')
    parser.add_argument('--height', type=int, default=2560, help='Destination canvas height')
    parser.add_argument('-s', '--size', type=int, help='Set destination canvas width and height')
    
    args = parser.parse_args()

    # --- Set default arguments ---
    if args.endpos is None: args.endpos = args.pos
    if args.output is None:
        base, ext = args.infile.rsplit('.', 1)
        args.output = f"{base}_reproj.{ext}"
    if args.size is not None: args.width = args.height = args.size

    # --- Calculations ---
    midaz, midalt, arc = midpoint(args.pos[0], args.pos[1], args.endpos[0], args.endpos[1])
    hfov = max(args.fov, arc * 1.4 * max(1, float(args.width) / args.height))

    # --- Grid File Generation (Optional) ---
    if args.gridfile:
        # Define the full set of parameters in the desired order to match the original output
        grid_img = OrderedDict([
            ('_type', 'i'), ('w', args.width), ('h', args.height), ('f', 0), ('v', hfov),
            ('Ra', 0), ('Rb', 0), ('Rc', 0), ('Rd', 0), ('Re', 0),
            ('Eev', 0), ('Er', 1), ('Eb', 1),
            ('r', 0), ('p', midalt), ('y', midaz - 180),
            ('TrX', 0), ('TrY', 0), ('TrZ', 0), ('Tpy', 0), ('Tpp', 0), ('j', 0),
            ('a', 0), ('b', 0), ('c', 0), ('d', 0), ('e', 0), ('g', 0), ('t', 0),
            ('Va', 1), ('Vb', 0), ('Vc', 0), ('Vd', 0), ('Vx', 0), ('Vy', 0), ('Vm', 5),
            ('n', "dummy.jpg")
        ])
        # A grid file typically doesn't need a 'p' line, but you could add one if needed
        write_pto(args.gridfile, global_options=None, images=[grid_img])
        print(f"✅ Grid file written to {args.gridfile}")

    # --- Main Reprojection ---
    try:
        # Use pto_mapper to parse the file
        global_options, images = pto_mapper.parse_pto_file(args.infile)
    except (ValueError, FileNotFoundError) as e:
        print(f"❌ Error: Could not read or parse PTO file. {e}")
        return

    # Create the combined rotation matrix
    T_yaw = euler_to_matrix(yaw_deg=180 - midaz, pitch_deg=0, roll_deg=0)
    T_pitch = euler_to_matrix(yaw_deg=0, pitch_deg=-midalt, roll_deg=0)
    T_combined = T_pitch @ T_yaw
    
    # Process only the first image
    img_to_process = images[0]
    
    # Apply the transformation
    y, p, r = img_to_process.get('y', 0.0), img_to_process.get('p', 0.0), img_to_process.get('r', 0.0)
    M_new = T_combined @ euler_to_matrix(y, p, r)
    y_new, p_new, r_new = matrix_to_euler(M_new)
    
    # Update image and panorama parameters
    img_to_process.update({'y': y_new, 'p': p_new, 'r': r_new})
    global_options.update({'f': 0, 'v': hfov, 'w': args.width, 'h': args.height, 'y': 0, 'p': 0, 'r': 0})
    
    write_pto(args.output, global_options, images=[img_to_process])
    print(f"✅ Reprojected PTO file written to {args.output}")

if __name__ == '__main__':
    main()
