#!/usr/bin/env python3

import os
import json
import math
import logging

# --- Try to import pto_mapper.py ---
# This attempts to import the necessary functions for camera calibration transformations.
# The PTO_MAPPER_AVAILABLE flag is used by other modules to conditionally enable features
# that rely on this functionality.
try:
    from pto_mapper import map_pano_to_image
    PTO_MAPPER_AVAILABLE = True
except ImportError as e:
    PTO_MAPPER_AVAILABLE = False
    # The calling scripts will log a warning if the import fails.
    pass

# --- Configuration Constants ---
# Establishes base paths for all necessary directories and configuration files.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOCK_DIR = os.path.join(BASE_DIR, 'locks')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
STATIONS_FILE = os.path.join(BASE_DIR, 'stations.json')
CAMERAS_FILE = os.path.join(BASE_DIR, 'cameras.json')

# --- Physical Constants ---
EARTH_RADIUS_KM = 6371.0


def update_status(status_file, status, data={}):
    """
    Writes a status update to a JSON file for a given task.
    This allows the frontend to poll for the progress of a long-running background process.
    """
    if status_file:
        try:
            with open(status_file, 'w') as f:
                json.dump({"status": status, **data}, f)
        except IOError as e:
            logging.error(f"Could not write to status file {status_file}: {e}")


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two latitude/longitude points in kilometers
    using the Haversine formula.
    """
    R = EARTH_RADIUS_KM
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    a = math.sin(dLat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def cross_track_distance(p_lat, p_lon, p1_lat, p1_lon, p2_lat, p2_lon):
    """
    Calculates the shortest distance from a point (p) to a great-circle path
    defined by two other points (p1 and p2). This is used to determine if a
    flight path passes close to a ground station.
    Returns the distance in kilometers.
    """
    # Distance from the start of the path (p1) to the query point (p).
    d_13 = haversine_distance(p1_lat, p1_lon, p_lat, p_lon)
    
    # Convert all coordinates to radians for trigonometric calculations.
    lat1_rad, lon1_rad = math.radians(p1_lat), math.radians(p1_lon)
    lat2_rad, lon2_rad = math.radians(p2_lat), math.radians(p2_lon)
    lat3_rad, lon3_rad = math.radians(p_lat), math.radians(p_lon)
    
    # Bearing from path start (p1) to the query point (p3).
    y13 = math.sin(lon3_rad - lon1_rad) * math.cos(lat3_rad)
    x13 = math.cos(lat1_rad) * math.sin(lat3_rad) - math.sin(lat1_rad) * math.cos(lat3_rad) * math.cos(lon3_rad - lon1_rad)
    brg_13 = math.atan2(y13, x13)

    # Bearing of the great-circle path itself (from p1 to p2).
    y12 = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
    x12 = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
    brg_12 = math.atan2(y12, x12)
    
    # If the path has zero length, the distance is simply the distance to the point.
    d_12 = haversine_distance(p1_lat, p1_lon, p2_lat, p2_lon)
    if d_12 == 0:
        return d_13

    # Check if the closest point on the great-circle is beyond the segment endpoints.
    # If so, the shortest distance is to one of the endpoints.
    d_23 = haversine_distance(p2_lat, p2_lon, p_lat, p_lon)
    if d_23**2 > d_13**2 + d_12**2 or d_13**2 > d_23**2 + d_12**2:
        return min(d_13, d_23)

    # Calculate the cross-track distance using the standard formula.
    return abs(math.asin(math.sin(d_13 / EARTH_RADIUS_KM) * math.sin(brg_13 - brg_12)) * EARTH_RADIUS_KM)


def is_sky_coord_in_view(pto_data, az_deg, alt_deg):
    """
    Checks if a given azimuth and altitude coordinate is visible within a camera's
    calibrated field of view, using the provided PTO (Hugin project) data.
    
    Returns:
        tuple: A tuple (bool, tuple or None).
               The boolean is True if the coordinate is in the camera's view.
               The tuple contains the (x, y) pixel coordinates if found, otherwise None.
    """
    if not PTO_MAPPER_AVAILABLE or pto_data is None:
        # If the pto_mapper library isn't available or no calibration data is provided,
        # we cannot determine the field of view. As a fallback, we assume the coordinate
        # is visible to avoid incorrectly discarding valid passes.
        return True, None

    try:
        # Pano-tools use a specific coordinate system where the horizontal axis is azimuth
        # and the vertical axis is related to altitude, with Y increasing downwards from the horizon.
        orig_pano_x = az_deg * 100
        orig_pano_y = (90 - alt_deg) * 100
        
        # Use the pto_mapper function to transform the panoramic coordinate to a pixel coordinate.
        # `restrict_to_bounds=True` ensures it returns None if the point is outside the image frame.
        pixel_coords = map_pano_to_image(pto_data, orig_pano_x, orig_pano_y, restrict_to_bounds=True)
        
        return (True, pixel_coords) if pixel_coords else (False, None)
    except Exception as e:
        # Log any errors from the mapping function for debugging but don't crash the prediction.
        # An error in mapping is treated as the point not being in view.
        logging.debug(f"pto_mapper.map_pano_to_image raised an exception: {e}")
        return False, None
