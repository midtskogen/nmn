#!/usr/bin/python3

# Improve calibration of a pto file using stars in the image.
# Usage: recalibrate.py <Unix timestamp> <input pto file> <image file> <output pto file>
import sys
import ephem
from datetime import datetime, UTC
import math
import pto_mapper
import argparse
from wand.image import Image
from wand.drawing import Drawing
from wand.color import Color
import os
import tempfile
import subprocess
from astropy.io import fits
import configparser
from stars import cat

parser = argparse.ArgumentParser(description='Improve calibration of a pto file using stars in the image. This tool works in these steps: 1) Use the timestamp, input .pto file and image to look for stars nearby their expected positions. 2) Get the coordinates of the actual positions of the stars. 3) Reoptimise the pto file and output it into a new file.')

parser.add_argument('-n', '--number', dest='objects', help='maximum number of objects (default: 500)', default=500, type=int)
parser.add_argument('-f', '--faintest', dest='faintest', help='faintest objects to include (default: 3)', default=3, type=float)
parser.add_argument('-a', '--altitude', dest='sunalt', help="don't recalibrate if the sun is this many degrees below the horizon or higher (default: -5)", default=-5, type=float)
parser.add_argument('-b', '--brightest', dest='brightest', help='brightest objects to include (default: -5)', default=-5, type=float)
parser.add_argument('-x', '--longitude', dest='longitude', help='observer longitude', type=float)
parser.add_argument('-y', '--latitude', dest='latitude', help='observer latitude', type=float)
parser.add_argument('-e', '--elevation', dest='elevation', help='observer elevation (m)', type=float)
parser.add_argument('-t', '--temperature', dest='temperature', help='observer temperature (C)', type=float)
parser.add_argument('-p', '--pressure', dest='pressure', help='observer air pressure (hPa)', type=float)
parser.add_argument('-i', '--image', dest='image', help='which image in the .pto file to use (default: 0)', default=0, type=int)
parser.add_argument('-r', '--radius', dest='radius', help='search radius around stars (degrees, default: 0.8)', default=0.8, type=float)
parser.add_argument('-g', '--gaussian', dest='blur', help='amount of blurring (percentage of radius default: 50)', default=50, type=float)
parser.add_argument('-q', '--sigma', dest='sigma', help='noise level (default: 20)', default=20.0, type=float)
parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='be more verbose')
parser.add_argument('-l', '--lens_optimise', action='store_true', dest='lensopt', help='optimise lens, not only orientation')

parser.add_argument(action='store', dest='timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00UTC)')
parser.add_argument(action='store', dest='infile', help='Hugin .pto file (input)')
parser.add_argument(action='store', dest='picture', help='Image file')
parser.add_argument(action='store', dest='outfile', help='Hugin .pto file (output)')


def format_pto_line(line_type, params):
    """Helper function to format a line for the PTO file."""
    parts = [line_type]
    for key, val in params.items():
        if isinstance(val, str) and ' ' in val:
            parts.append(f'{key}"{val}"')
        else:
            parts.append(f'{key}{val}')
    return ' '.join(parts)

def write_pto_for_optimisation(outfile, pto_data, image_index, control_points, lens_opt):
    """
    Writes a new PTO file configured for optimizing a single image against itself.
    """
    global_options, images = pto_data
    img_to_optimise = images[image_index].copy()

    with open(outfile, 'w') as f:
        f.write(format_pto_line('p', global_options) + '\n')

        # Write two 'i' lines for the same image, which is how autooptimiser
        # works when optimizing one image against a set of control points.
        img_line_0 = img_to_optimise.copy()
        img_line_0['n'] = 0 # First image reference
        f.write(format_pto_line('i', img_line_0) + '\n')

        img_line_1 = img_to_optimise.copy()
        img_line_1['n'] = 1 # Second image reference
        f.write(format_pto_line('i', img_line_1) + '\n')

        # Write control points linking the two image references
        for cp in control_points:
            x_expected, y_expected, x_found, y_found = cp
            f.write(f'c n0 N1 x{x_found:.4f} y{y_found:.4f} X{x_expected:.4f} Y{y_expected:.4f}\n')

        # Define the optimization variables in the correct multi-line 'v' format
        f.write('\n# specify variables that should be optimized\n')
        
        # Base variables for orientation and FOV
        base_vars = ['v', 'r', 'p', 'y']
        for var in base_vars:
            f.write(f"v {var}{image_index}\n")

        if lens_opt:
            # Add lens distortion parameters if requested
            lens_vars = ['a', 'b', 'c', 'd', 'e']
            for var in lens_vars:
                 f.write(f"v {var}{image_index}\n")
        
        # Final terminating 'v' line required by autooptimiser
        f.write('v\n')

        # Final line required by autooptimiser
        f.write('*\n')


def recalibrate(timestamp, infile, picture, outfile, pos, image, radius = 1.0, lensopt = False, faintest = 3, brightest = -5, objects = 500, blur = 50, verbose = False, sigma = 20):
    out_stream = sys.stdout if verbose else subprocess.DEVNULL
    
    if verbose:
        print(f"Parsing PTO file: {infile}")
    try:
        pto_data = pto_mapper.parse_pto_file(infile)
        global_options, images = pto_data
    except (ValueError, FileNotFoundError) as e:
        print(f"Error parsing PTO file '{infile}': {e}", file=sys.stderr)
        exit(1)

    img_params = images[image]
    width, height = img_params['w'], img_params['h']
    pano_width, pano_height = global_options['w'], global_options['h']
    
    if global_options.get('f') != 2:
        print("Error: The panorama projection (p-line 'f' parameter) must be 2 (Equirectangular).", file=sys.stderr)
        exit(1)

    img_hfov = img_params.get('v', 180)
    pixel_radius = radius * width / img_hfov
    blur_radius = pixel_radius * blur / 100
    
    if verbose:
        print(f"Using image {image}: width={width}, height={height}, HFOV={img_hfov:.2f} deg")
        print(f"Search radius is {radius:.2f} deg, which corresponds to {pixel_radius:.2f} pixels.")

    starlist = []

    def test_body(body, name, faintest, brightest):
        """
        Computes celestial object position, maps it to an image coordinate,
        and adds it to the list of expected star locations.
        """
        body.compute(pos)
        az_rad, alt_rad = float(repr(body.az)), float(repr(body.alt))
        alt_deg = math.degrees(alt_rad)
        if alt_deg < -0.5:
            return 0
        
        alt_corrected_deg = alt_deg + 0.01666 / math.tan(math.radians(alt_deg + (7.31/(alt_deg + 4.4))))
        alt_corrected_rad = math.radians(alt_corrected_deg)

        if body.mag > faintest or body.mag < brightest:
            return 0

        pano_x = (az_rad / (2 * math.pi)) * pano_width
        pano_y = ((-alt_corrected_rad / math.pi) + 0.5) * pano_height

        result = pto_mapper.map_pano_to_image(pto_data, pano_x, pano_y, restrict_to_bounds=True)

        if result:
            mapped_image_index, x, y = result
            if mapped_image_index == image:
                if verbose:
                    print(f"  Found expected position for {name} (mag: {body.mag:.2f}) at ({x:.2f}, {y:.2f})")
                starlist.append((x, y, 0, 0))
                return 1
        return 0

    if verbose:
        print(f"Calculating expected positions for up to {objects} celestial objects (mag {brightest} to {faintest})...")
    
    count = 0
    for (body, name) in [ (ephem.Sun(), "Sol"), (ephem.Moon(), "Moon") , (ephem.Mercury(), "Mercury"), (ephem.Venus(), "Venus"), (ephem.Mars(), "Mars"), (ephem.Jupiter(), "Jupiter"), (ephem.Saturn(), "Saturn") ]:
        count += test_body(body, name, faintest, brightest)
        if count >= objects:
            break

    if count < objects:
        for (ra, pmra, dec, pmdec, mag, name) in cat:
            if (mag <= faintest and mag >= brightest):
                body = ephem.FixedBody()
                body._ra, body._pmra, body._dec, body._pmdec, body._epoch = str(ra), pmra, str(dec), pmdec, ephem.J2000
                count += test_body(body, name, faintest, brightest)
                if count >= objects:
                    break
    
    if not starlist:
        print("No expected stars could be mapped to the input image. Check PTO parameters and timestamp.", file=sys.stderr)
        exit(1)

    if verbose:
        print("Generating star mask for feature detection...")

    stars = Image(width=width, height=height, background=Color('black'))
    pic = Image(filename=picture)

    with Drawing() as draw:
        draw.fill_color = Color('white')
        draw.stroke_color = Color('white')
        for (x, y, _, _) in starlist:
            draw.circle((x, y), (x + pixel_radius, y))
        draw(stars)
        stars.gaussian_blur(blur_radius, blur_radius)

    pic.gaussian_blur(blur_radius/32, blur_radius/32)
    
    masked = stars.clone()
    with Drawing() as draw:
        draw.composite(operator='bumpmap', left=0, top=0,
                       width=pic.width, height=pic.height, image=pic)
        draw(masked)

    with tempfile.NamedTemporaryFile(prefix='recalibrate_', suffix='.png', dir='/tmp', delete=False) as temp:
        temp_name = temp.name
        masked.format = 'png'
        masked.save(file=temp)

    if verbose:
        print(f"Running solve-field to find actual star positions (sigma={sigma})...")
    try:
        subprocess.run(['solve-field', '--sigma', str(sigma), '--just-augment', temp_name], check=True, stdout=out_stream, stderr=out_stream)
        axyfile = os.path.splitext(temp_name)[0] + '.axy'
        with fits.open(axyfile) as hdul:
            axy_data = hdul[1].data
        os.unlink(axyfile)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running solve-field: {e}", file=sys.stderr)
        exit(1)
    finally:
        os.unlink(temp_name)

    def dist(x1, y1, x2, y2):
        return math.sqrt((x2-x1)**2+(y2-y1)**2)

    if verbose:
        print("Matching expected stars to found stars...")
    remap = []
    found_stars = [(row[0]-1, row[1]-1) for row in axy_data]

    for (x_expected, y_expected, _, _) in starlist:
        min_dist = float('inf')
        best_match = None
        for (x_found, y_found) in found_stars:
            d = dist(x_expected, y_expected, x_found, y_found)
            if d < min_dist:
                min_dist = d
                best_match = (x_found, y_found)
        
        if min_dist <= pixel_radius:
            remap.append((x_expected, y_expected, best_match[0], best_match[1]))

    if verbose:
        print(f'Stars expected: {len(starlist)}')
        print(f'Stars found: {len(found_stars)}')
        print(f'Stars remapped: {len(remap)}')
        for (x1, y1, x2, y2) in remap:
            print(f'  {x1:8.3f},{y1:8.3f} -> {x2:8.3f},{y2:8.3f}')
            
    if not remap:
        print("Could not remap any stars. No control points generated. Check search radius or image quality.", file=sys.stderr)
        exit(1)

    if verbose:
        print(f"Writing new PTO file for optimisation to {outfile}...")
    write_pto_for_optimisation(outfile, pto_data, image, remap, lensopt)
    
    # Optional: uncomment to see the PTO file before autooptimiser runs
    # with open(outfile, 'r') as f:
    #     print(f.read())

    try:
        if verbose: print(f"Running cpclean on {outfile}...")
        subprocess.run(['cpclean', '-n', '1', '-o', outfile, outfile], check=True, stdout=out_stream, stderr=out_stream)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('Failed to run cpclean. Please ensure Hugin tools are in your PATH.', file=sys.stderr)

    try:
        if verbose: print(f"Running autooptimiser on {outfile}...")
        subprocess.run(['autooptimiser', '-n', '-o', outfile, outfile], check=True, stdout=out_stream, stderr=out_stream)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print('Failed to run autooptimiser. Please ensure Hugin tools are in your PATH.', file=sys.stderr)
    
    if verbose:
        print("Recalibration complete.")


if __name__ == '__main__':
    args = parser.parse_args()
    pos = ephem.Observer()
    config = configparser.ConfigParser()
    
    config.read(['/etc/meteor.cfg', os.path.expanduser('~/meteor.cfg')])
    try:
        pos.lat = config.get('astronomy', 'latitude')
        pos.lon = config.get('astronomy', 'longitude')
        pos.elevation = float(config.get('astronomy', 'elevation'))
        pos.temp = float(config.get('astronomy', 'temperature'))
        pos.pressure = float(config.get('astronomy', 'pressure'))
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Configuration error: {e}. Please ensure meteor.cfg is set up or provide location via command line.", file=sys.stderr)
        if not (args.latitude and args.longitude):
            exit(1)

    if args.longitude is not None:
        pos.lon = str(args.longitude)
    if args.latitude is not None:
        pos.lat = str(args.latitude)
    if args.elevation is not None:
        pos.elevation = args.elevation
    if args.temperature is not None:
        pos.temp = args.temperature
    if args.pressure is not None:
        pos.pressure = args.pressure

    pos.date = datetime.fromtimestamp(float(args.timestamp), UTC)

    sun = ephem.Sun(pos)
    sun_alt_deg = math.degrees(sun.alt)
    if sun_alt_deg > args.sunalt:
        print(f"Sun is too high (altitude: {sun_alt_deg:.2f} deg, limit: {args.sunalt:.2f} deg). Skipping recalibration.", file=sys.stderr)
        exit(0)

    recalibrate(
        args.timestamp, args.infile, args.picture, args.outfile, pos, 
        args.image, args.radius, args.lensopt, args.faintest, 
        args.brightest, args.objects, args.blur, args.verbose, args.sigma
    )
