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

def setup_arg_parser():
    """Sets up and returns the argument parser."""
    parser = argparse.ArgumentParser(
        description='Improve calibration of a pto file using stars in the image. This tool works in these steps: '
                    '1) Use the timestamp, input .pto file and image to look for stars nearby their expected positions. '
                    '2) Get the coordinates of the actual positions of the stars. '
                    '3) Reoptimise the pto file and output it into a new file.'
    )

    # Positional Arguments
    parser.add_argument('timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00UTC)')
    parser.add_argument('infile', help='Hugin .pto file (input)')
    parser.add_argument('picture', help='Image file')
    parser.add_argument('outfile', help='Hugin .pto file (output)')

    # General Options
    parser.add_argument('-c', '--config', dest='configfile', help='filename for a config file')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='be more verbose')
    parser.add_argument('-i', '--image', dest='image', help='which image in the .pto file to use (default: 0)', default=0, type=int)

    # Observer Location and Environment (can be overridden by config)
    parser.add_argument('-x', '--longitude', dest='longitude', help='observer longitude', type=float)
    parser.add_argument('-y', '--latitude', dest='latitude', help='observer latitude', type=float)
    parser.add_argument('-e', '--elevation', dest='elevation', help='observer elevation (m)', type=float)
    parser.add_argument('-t', '--temperature', dest='temperature', help='observer temperature (C)', type=float)
    parser.add_argument('-p', '--pressure', dest='pressure', help='observer air pressure (hPa)', type=float)

    # Star Selection & Sun Altitude
    parser.add_argument('-n', '--number', dest='objects', help='maximum number of objects (default: 500)', default=500, type=int)
    parser.add_argument('-f', '--faintest', dest='faintest', help='faintest objects to include (default: 3)', default=3, type=float)
    parser.add_argument('-b', '--brightest', dest='brightest', help='brightest objects to include (default: -5)', default=-5, type=float)
    parser.add_argument('-a', '--altitude', dest='sunalt', help="don't recalibrate if the sun is this many degrees below the horizon or higher (default: -5)", default=-5, type=float)
    
    # Image Processing and Optimisation
    parser.add_argument('-r', '--radius', dest='radius', help='search radius around stars (degrees, default: 0.8)', default=0.8, type=float)
    parser.add_argument('-g', '--gaussian', dest='blur', help='amount of blurring (percentage of radius default: 50)', default=50, type=float)
    parser.add_argument('-q', '--sigma', dest='sigma', help='noise level for solve-field (default: 20)', default=20.0, type=float)
    parser.add_argument('-l', '--lens-optimise', action='store_true', dest='lensopt', help='optimise lens, not only orientation')

    return parser

def format_pto_line(line_type, params):
    """Helper function to format a line for the PTO file."""
    parts = [line_type]
    for key, val in params.items():
        # Hugin format requires quotes around values with spaces
        if isinstance(val, str) and ' ' in val:
            parts.append(f'{key}"{val}"')
        else:
            parts.append(f'{key}{val}')
    return ' '.join(parts)

def write_pto_for_optimisation(outfile, pto_data, image_index, control_points, lens_opt):
    """
    Writes a new PTO file configured for optimizing a single image against itself.
    This setup is required by Hugin's autooptimiser tool.
    """
    global_options, images = pto_data
    img_to_optimise = images[image_index]

    with open(outfile, 'w') as f:
        # Write panorama line
        f.write(format_pto_line('p', global_options) + '\n\n')

        # Write two 'i' lines for the same image. This is a trick to make
        # autooptimiser work with control points on a single image.
        img_line_params = img_to_optimise.copy()
        img_line_params['n'] = 0
        f.write(format_pto_line('i', img_line_params) + '\n')
        img_line_params['n'] = 1
        f.write(format_pto_line('i', img_line_params) + '\n\n')

        # Write control points linking the two image "references"
        # n0 refers to the first 'i' line, N1 to the second.
        # The 'found' coordinates are on image 0, 'expected' on image 1.
        for cp in control_points:
            x_expected, y_expected, x_found, y_found = cp
            f.write(f'c n0 N1 x{x_found:.4f} y{y_found:.4f} X{x_expected:.4f} Y{y_expected:.4f}\n')

        # Define the optimization variables in the multi-line 'v' format
        f.write('\n# Variables to be optimized\n')
        base_vars = ['v', 'r', 'p', 'y'] # HFOV, roll, pitch, yaw
        
        # We optimize variables for the first image 'i n0'
        for var in base_vars:
            f.write(f"v {var}0\n")

        if lens_opt:
            lens_vars = ['a', 'b', 'c', 'd', 'e'] # Lens distortion parameters
            for var in lens_vars:
                 f.write(f"v {var}0\n")
        
        # Final terminating 'v' line and '*' are required by autooptimiser
        f.write('v\n')
        f.write('*\n')

def recalibrate(timestamp, infile, picture, outfile, pos, **kwargs):
    """Main recalibration logic."""
    verbose = kwargs.get('verbose', False)
    image_idx = kwargs.get('image', 0)
    radius = kwargs.get('radius', 1.0)
    lensopt = kwargs.get('lensopt', False)
    faintest = kwargs.get('faintest', 3)
    brightest = kwargs.get('brightest', -5)
    objects = kwargs.get('objects', 500)
    blur = kwargs.get('blur', 50)
    sigma = kwargs.get('sigma', 20)
    
    out_stream = sys.stdout if verbose else subprocess.DEVNULL
    
    if verbose: print(f"Parsing PTO file: {infile}")
    try:
        pto_data = pto_mapper.parse_pto_file(infile)
        global_options, images = pto_data
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: Could not parse PTO file '{infile}': {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure we are using an equirectangular panorama, as calculations depend on it
    if global_options.get('f') != 2:
        print("Error: The panorama projection (p-line 'f' parameter) must be 2 (Equirectangular).", file=sys.stderr)
        sys.exit(1)

    img_params = images[image_idx]
    width, height = img_params['w'], img_params['h']
    pano_width, pano_height = global_options['w'], global_options['h']
    img_hfov = img_params.get('v', 180) # Default to 180 HFOV if not specified
    
    pixel_radius = radius * width / img_hfov
    blur_radius = pixel_radius * blur / 100
    
    if verbose:
        print(f"Using image {image_idx}: width={width}, height={height}, HFOV={img_hfov:.2f} deg")
        print(f"Search radius is {radius:.2f} deg, which corresponds to {pixel_radius:.2f} pixels.")

    starlist = []

    def test_body(body, name, faintest, brightest):
        """
        Computes celestial object position, maps it to an image coordinate,
        and adds it to the list of expected star locations if visible.
        Returns 1 if the star is added, 0 otherwise.
        """
        body.compute(pos)
        az_rad, alt_rad = float(repr(body.az)), float(repr(body.alt))
        alt_deg = math.degrees(alt_rad)
        
        # Ignore objects below the horizon (with a small margin)
        if alt_deg < -0.5:
            return 0
        
        # Apply a simple atmospheric refraction correction
        alt_corrected_deg = alt_deg + 0.01666 / math.tan(math.radians(alt_deg + (7.31/(alt_deg + 4.4))))
        alt_corrected_rad = math.radians(alt_corrected_deg)

        if not (brightest <= body.mag <= faintest):
            return 0

        # Convert celestial coordinates (az, alt) to panoramic coordinates (x, y)
        pano_x = (az_rad / (2 * math.pi)) * pano_width
        pano_y = ((-alt_corrected_rad / math.pi) + 0.5) * pano_height

        # Map panoramic coordinates to a specific image's pixel coordinates
        result = pto_mapper.map_pano_to_image(pto_data, pano_x, pano_y, restrict_to_bounds=True)

        if result:
            mapped_image_index, x, y = result
            if mapped_image_index == image_idx:
                if verbose:
                    print(f"  Found expected position for {name} (mag: {body.mag:.2f}) at ({x:.2f}, {y:.2f})")
                starlist.append({'x_exp': x, 'y_exp': y})
                return 1
        return 0

    if verbose:
        print(f"Calculating expected positions for up to {objects} celestial objects (mag {brightest} to {faintest})...")
    
    count = 0
    # Check for planets, Sun, and Moon first
    solar_system_bodies = [
        (ephem.Sun(), "Sol"), (ephem.Moon(), "Moon"), (ephem.Mercury(), "Mercury"), 
        (ephem.Venus(), "Venus"), (ephem.Mars(), "Mars"), (ephem.Jupiter(), "Jupiter"), 
        (ephem.Saturn(), "Saturn")
    ]
    for body, name in solar_system_bodies:
        if count >= objects: break
        count += test_body(body, name, faintest, brightest)

    # Then fill with stars from the catalog
    if count < objects:
        for ra, pmra, dec, pmdec, mag, name in cat:
            if brightest <= mag <= faintest:
                body = ephem.FixedBody()
                body._ra, body._pmra, body._dec, body._pmdec, body._epoch = str(ra), pmra, str(dec), pmdec, ephem.J2000
                count += test_body(body, name, faintest, brightest)
                if count >= objects:
                    break
    
    if not starlist:
        print("Error: No expected stars could be mapped to the input image. Check PTO parameters, timestamp, and location.", file=sys.stderr)
        sys.exit(1)

    if verbose: print("Generating star mask for feature detection...")
    
    # Create a mask image with white circles at expected star locations
    with Image(width=width, height=height, background=Color('black')) as stars_mask:
        with Drawing() as draw:
            draw.fill_color = Color('white')
            for star in starlist:
                draw.circle((star['x_exp'], star['y_exp']), (star['x_exp'] + pixel_radius, star['y_exp']))
            draw(stars_mask)
        stars_mask.gaussian_blur(blur_radius, blur_radius)

        # Use the mask to highlight potential stars in the actual picture
        with Image(filename=picture) as pic:
            pic.gaussian_blur(radius=blur_radius/32, sigma=blur_radius/32) # Slight blur to reduce noise
            with stars_mask.clone() as masked:
                masked.composite(pic, operator='bumpmap', left=0, top=0)
                
                # Use a temporary file for solve-field
                with tempfile.NamedTemporaryFile(prefix='recalibrate_', suffix='.png', delete=False) as temp:
                    temp_name = temp.name
                    # Explicitly set the format before saving to the file handle
                    masked.format = 'png'
                    masked.save(file=temp)

    axyfile = os.path.splitext(temp_name)[0] + '.axy'
    if verbose: print(f"Running solve-field to find actual star positions (sigma={sigma})...")
    
    try:
        # solve-field finds star-like features and writes their coordinates to an .axy file
        subprocess.run(
            ['solve-field', '--sigma', str(sigma), '--just-augment', temp_name], 
            check=True, stdout=out_stream, stderr=out_stream
        )
        with fits.open(axyfile) as hdul:
            axy_data = hdul[1].data
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: Failed to run solve-field. Is astrometry.net installed and in your PATH? Details: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up temporary files
        if os.path.exists(axyfile): os.unlink(axyfile)
        if os.path.exists(temp_name): os.unlink(temp_name)

    # Match found stars to expected stars
    if verbose: print("Matching expected stars to found stars...")
    remap = []
    # FITS files are 1-indexed, so convert to 0-indexed pixel coordinates
    found_stars = [(row[0]-1, row[1]-1) for row in axy_data]

    for star in starlist:
        x_expected, y_expected = star['x_exp'], star['y_exp']
        min_dist = float('inf')
        best_match = None
        for (x_found, y_found) in found_stars:
            d = math.hypot(x_found - x_expected, y_found - y_expected)
            if d < min_dist:
                min_dist = d
                best_match = (x_found, y_found)
        
        # Only consider a match if it's within the search radius
        if min_dist <= pixel_radius:
            remap.append((x_expected, y_expected, best_match[0], best_match[1]))

    if verbose:
        print(f'Stars expected: {len(starlist)}')
        print(f'Stars found by solve-field: {len(found_stars)}')
        print(f'Stars successfully remapped: {len(remap)}')
        for (x1, y1, x2, y2) in remap:
            print(f'  {x1:8.3f},{y1:8.3f} -> {x2:8.3f},{y2:8.3f}')
            
    if not remap:
        print("Error: Could not remap any stars. No control points generated. Try increasing the search radius or checking image quality.", file=sys.stderr)
        sys.exit(1)

    # Write a temporary PTO file for Hugin tools
    with tempfile.NamedTemporaryFile(prefix='opt_', suffix='.pto', delete=False, mode='w') as temp_pto:
        temp_pto_name = temp_pto.name
    
    if verbose: print(f"Writing new PTO file for optimisation to {temp_pto_name}...")
    write_pto_for_optimisation(temp_pto_name, pto_data, image_idx, remap, lensopt)

    # Run Hugin's cpclean and autooptimiser to refine the PTO parameters
    try:
        if verbose: print(f"Running cpclean on {temp_pto_name}...")
        # cpclean removes outlier control points
        subprocess.run(['cpclean', '-n', '1', '-o', temp_pto_name, temp_pto_name], check=True, stdout=out_stream, stderr=out_stream)
        
        if verbose: print(f"Running autooptimiser on {temp_pto_name}...")
        # autooptimiser refines the image position and lens parameters
        subprocess.run(['autooptimiser', '-n', '-o', outfile, temp_pto_name], check=True, stdout=out_stream, stderr=out_stream)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: Failed to run Hugin tools (cpclean/autooptimiser). Are they in your PATH? Details: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(temp_pto_name): os.unlink(temp_pto_name)

    if verbose: print(f"Recalibration complete. Optimised PTO file saved to {outfile}.")

def main():
    """Main execution function."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    
    # Determine which config file to use
    if args.configfile:
        config_files = [args.configfile]
    else:
        # Default search order, with corrected path for user's config
        config_files = [os.path.expanduser('~/meteor.cfg'), '/etc/meteor.cfg']

    read_files = config.read(config_files)
    if args.verbose and read_files:
        print(f"Read configuration from: {', '.join(read_files)}")
    
    # A helper to load config values or use command-line arguments as a fallback
    def get_config_or_arg(section, option, arg_val, is_float=False):
        if arg_val is not None:
            return arg_val
        try:
            val = config.get(section, option)
            return float(val) if is_float else val
        except (configparser.NoSectionError, configparser.NoOptionError):
            return None

    # Step 1: Get location values and store them in temporary variables
    lat = get_config_or_arg('astronomy', 'latitude', args.latitude)
    lon = get_config_or_arg('astronomy', 'longitude', args.longitude)
    elev = get_config_or_arg('astronomy', 'elevation', args.elevation, is_float=True)
    temp = get_config_or_arg('astronomy', 'temperature', args.temperature, is_float=True)
    pressure = get_config_or_arg('astronomy', 'pressure', args.pressure, is_float=True)

    # Step 2: Validate that the essential location data exists
    if lat is None or lon is None or elev is None:
        print("Error: Observer location (latitude, longitude, elevation) must be provided.", file=sys.stderr)
        print("Please provide them via command-line arguments (-y, -x, -e) or a config file.", file=sys.stderr)
        sys.exit(1)

    # Step 3: Now that validation is complete, create and configure the observer
    pos = ephem.Observer()
    pos.lat = str(lat)
    pos.lon = str(lon)
    pos.elevation = elev
    if temp is not None:
        pos.temp = temp
    if pressure is not None:
        pos.pressure = pressure

    # Set the time of observation
    try:
        pos.date = datetime.fromtimestamp(float(args.timestamp), UTC)
    except ValueError:
        print(f"Error: Invalid timestamp '{args.timestamp}'", file=sys.stderr)
        sys.exit(1)

    # Check sun's altitude to ensure it's dark enough for stars to be visible
    sun = ephem.Sun()
    sun.compute(pos)
    sun_alt_deg = math.degrees(sun.alt)
    if sun_alt_deg > args.sunalt:
        print(f"Sun is too high (altitude: {sun_alt_deg:.2f} deg, limit: {args.sunalt:.2f} deg). Skipping recalibration.")
        sys.exit(0) # Exit successfully as this is a conditional skip, not an error.

    # Pass all keyword arguments to the main function
    recalibrate(
        args.timestamp, args.infile, args.picture, args.outfile, pos, 
        image=args.image, radius=args.radius, lensopt=args.lensopt, 
        faintest=args.faintest, brightest=args.brightest, objects=args.objects, 
        blur=args.blur, verbose=args.verbose, sigma=args.sigma
    )

if __name__ == '__main__':
    main()
