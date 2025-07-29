#!/usr/bin/python3
"""
Associates observed meteors with known meteor showers.

This script takes the observational data of a meteor (radiant position, speed, 
and date) and compares it against a catalog of known meteor showers to find 
the best match.
"""

import showerdata
from datetime import datetime
import numpy as np
from spiceypy import radrec, rpd, vsep, dpr
from typing import List, Tuple, Optional

def _get_day_of_year(date_str: str, date_format: str) -> Optional[int]:
    """
    Safely parse a date string and return the day of the year.
    Returns None if the date format is invalid.
    """
    try:
        return datetime.strptime(date_str, date_format).timetuple().tm_yday
    except ValueError:
        return None

def _interpol_radiant(rad_dates: List[str], coords: List[float], obs_yday: int) -> Optional[float]:
    """
    Interpolates radiant coordinates (RA or Dec) to the observation date.

    Args:
        rad_dates: A list of radiant dates in 'MM-DD' format.
        coords: A list of corresponding radiant coordinates (RA or Dec).
        obs_yday: The day of the year of the observation.

    Returns:
        The interpolated coordinate value, or None if interpolation fails.
    """
    date_yday = [_get_day_of_year(d, '%m-%d') for d in rad_dates]
    
    # Filter out any dates that failed to parse
    valid_data = [(day, coord) for day, coord in zip(date_yday, coords) if day is not None]
    if not valid_data:
        return None

    date_yday, coords = zip(*valid_data)
    date_yday = list(date_yday)

    # Handle showers that cross the year-end boundary (e.g., Quadrantids)
    if max(date_yday) - min(date_yday) > 180:
        for i, day in enumerate(date_yday):
            if day < 180:  # Heuristic: dates in the first half of the year belong to the next year
                date_yday[i] += 365

    # Fallback to mean if quadratic fit isn't possible
    if len(date_yday) < 3:
        return np.mean(coords) if coords else None

    # Fit a 2nd-degree polynomial to the data
    try:
        poly_fit = np.polyfit(date_yday, coords, 2)
        return np.polyval(poly_fit, obs_yday)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback if fitting fails
        return np.mean(coords)


def _radiant_separation(ra_obs: float, decl_obs: float, shower: 'showerdata.ShowerInfo', obs_yday: int) -> float:
    """
    Calculates the angular separation between an observed radiant and a shower's radiant.

    Args:
        ra_obs: Observed Right Ascension in degrees.
        decl_obs: Observed Declination in degrees.
        shower: A shower data object.
        obs_yday: The day of the year of the observation.

    Returns:
        The angular separation in degrees. Returns a large value (999.0) on failure.
    """
    ra_shower = _interpol_radiant(shower.rad_date, shower.ra, obs_yday)
    dec_shower = _interpol_radiant(shower.rad_date, shower.dec, obs_yday)

    if ra_shower is None or dec_shower is None:
        return 999.0

    # Convert RA/Dec to Cartesian vectors and find the separation
    shower_vec = radrec(1, ra_shower * rpd(), dec_shower * rpd())
    obs_vec = radrec(1, ra_obs * rpd(), decl_obs * rpd())
    
    return vsep(obs_vec, shower_vec) * dpr()

def showerassoc(ra_obs: float, decl_obs: float, speed_obs: float, obs_date_str: str) -> Tuple[str, str]:
    """
    Associates an observed meteor with a known meteor shower. This is the main
    function intended for external calls.

    Args:
        ra_obs: Observed Right Ascension in degrees.
        decl_obs: Observed Declination in degrees.
        speed_obs: Observed atmospheric speed in km/s.
        obs_date_str: The observation date in 'YYYY-MM-DD' format.

    Returns:
        A tuple containing the full name and the 3-letter code of the best-matched shower.
        Returns ('', '') if no suitable match is found.
    """
    obs_date_obj = datetime.strptime(obs_date_str, '%Y-%m-%d')
    obs_yday = obs_date_obj.timetuple().tm_yday

    best_score = 0.0
    best_match_name = ''
    best_match_code = ''

    for shower in showerdata.showerlist:
        start_yday = _get_day_of_year(shower.beg_date, '%m-%d')
        end_yday = _get_day_of_year(shower.end_date, '%m-%d')

        if start_yday is None or end_yday is None:
            print(f"Warning: Invalid date format in data for shower {shower.name}. Skipping.")
            continue
        
        current_obs_yday = obs_yday
        date_match = False
        
        # Handle showers active across the new year
        if start_yday > end_yday:
            is_in_activity_period = (obs_yday >= start_yday or obs_yday <= end_yday)
            if is_in_activity_period:
                date_match = True
                if obs_yday <= end_yday: # Observation is in the next year (e.g., Jan for Quadrantids)
                    current_obs_yday += 365
        else:
            if start_yday <= obs_yday <= end_yday:
                date_match = True

        if not date_match:
            continue

        # Calculate separation and velocity difference
        separation = _radiant_separation(ra_obs, decl_obs, shower, current_obs_yday)
        if separation > 10.0:  # Angular separation threshold
            continue

        velocity_error = abs(shower.v_inf - speed_obs) / shower.v_inf
        if velocity_error >= 0.2:  # Velocity difference threshold
            continue
            
        # Calculate a score for the match. Lower separation and velocity error result in a higher score.
        score = (1.0 / (velocity_error + 0.1)) + (1.0 / ((separation + 1.0) / 55.0))
        
        if score > best_score:
            best_score = score
            best_match_name = shower.name
            best_match_code = shower.name_sg

    return best_match_name, best_match_code

def check_shower_data_integrity():
    """
    Checks the integrity of the shower data lists and plots the radiant positions.
    """
    import matplotlib.pyplot as plt

    for shower in showerdata.showerlist:
        if not (len(shower.ra) == len(shower.dec) == len(shower.rad_date)):
            print(f"List length mismatch in shower: {shower.name}")
        for date_str in shower.rad_date:
            if _get_day_of_year(date_str, '%m-%d') is None:
                print(f"Bad date format '{date_str}' in shower: {shower.name}")
        
        plt.plot(shower.ra, shower.dec, 'o-', label=shower.name)

    plt.xlabel("R.A. (deg)")
    plt.ylabel("Dec. (deg)")
    plt.title("Shower Radiant Positions")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # To run a check on the shower data, uncomment the following line:
    # check_shower_data_integrity()

    test_cases = [
        {'name': 'Lyrid', 'ra': 271.5, 'dec': 33.5, 'spd': 49.0, 'date': '2024-04-22'},
        {'name': 'Quadrantid (start of year)', 'ra': 230.0, 'dec': 49.0, 'spd': 41.0, 'date': '2024-01-03'},
        {'name': 'Quadrantid (end of year)', 'ra': 230.0, 'dec': 49.0, 'spd': 41.0, 'date': '2024-12-29'},
        {'name': 'No match (sporadic)', 'ra': 230.0, 'dec': 50.0, 'spd': 40.0, 'date': '2024-06-01'},
        {'name': 'Leonid', 'ra': 152.0, 'dec': 22.0, 'spd': 71.0, 'date': '2024-11-17'},
        {'name': 'May Camelopardalid', 'ra': 120., 'dec': 81., 'spd': 18., 'date': '2024-05-16'}
    ]

    print("Running meteor shower association tests...")
    for test in test_cases:
        print(f"\n--- Testing for: {test['name']} on {test['date']} ---")
        match_name, match_code = showerassoc(test['ra'], test['dec'], test['spd'], test['date'])
        if match_name:
            print(f"  -> Match found: {match_name} ({match_code})")
        else:
            print("  -> No match found.")
