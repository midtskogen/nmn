#!/usr/bin/env python3
"""
Calculates and plots the orbit of a meteor based on observational data.
"""

# 1. Standard Library Imports
import os
import argparse

# 2. Third-Party Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 3. SPICE and Local Application Imports
from spiceypy import (
    bodvrd, spkezr, str2et, georec, unorm, vsub,
    vadd, vscl, pxform, mxv, recrad, radrec, oscelt, convrt,
    et2utc, conics, ktotal, kdata, furnsh, SpiceyError
)
from fbspd_merge import readres
import showerassoc

# --- Module-Level Constants ---
REF_FRAME_ECLIPTIC = "ECLIPJ2000"
REF_FRAME_J2000 = "J2000"
REF_FRAME_IAU_EARTH = "IAU_EARTH"
ABERRATION_CORRECTION = "NONE"
SOLAR_SYSTEM_BARYCENTER = "Sun"
SIDEREAL_DAY_SECONDS = 23.9344696 * 3600.0

def _load_spice_kernels():
    """
    Finds and loads the required SPICE kernels.
    """
    possible_paths = [
        './data/',
        '/var/www/html/bin/data',
        os.path.expanduser('~/spice/data/')
    ]
    kernel_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if not kernel_path:
        print("Error: SPICE kernel directory not found.")
        return False

    kernels_to_load = [
        "lsk/naif0012.tls",
        "spk/planets/de440.bsp",
        "pck/pck00010.tpc",
        "pck/gm_de440.tpc"
    ]

    print(f"--- Using SPICE kernel base path: {kernel_path} ---")
    try:
        for kernel in kernels_to_load:
            full_path = os.path.join(kernel_path, kernel)
            if not os.path.exists(full_path):
                print(f"Error: Required SPICE kernel not found: {full_path}")
                return False
            furnsh(full_path)

        if ktotal('ALL') == 0:
            print("CRITICAL ERROR: No kernels were loaded by SPICE.")
            return False

    except SpiceyError as e:
        print(f"An error occurred during SPICE kernel loading: {e}")
        return False

    return True


def _plot_orbit(et, meteor_elements, doplot):
    """
    Generates and displays or saves a 3D plot of the meteoroid's orbit.
    """
    if doplot not in ['show', 'save', 'both']:
        return

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    fig.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, color='lightgray', linestyle=':', linewidth=0.5)

    _, gm_sun = bodvrd("SUN", "GM", 1)
    gm_sun = gm_sun[0]
    
    km_per_au = convrt(1.0, "AU", "KM")

    planet_params = {
        "Venus": {"name": "Venus", "color": "#C77C00"},
        "Earth": {"name": "Jorda", "color": "#0077BE"},
        "MARS BARYCENTER": {"name": "Mars", "color": "#D73A00"},
        "Jupiter barycenter": {"name": "Jupiter", "color": "#A0522D"}
    }

    q_km, e = meteor_elements[0], meteor_elements[1]
    if e < 1:
        aphelion_au = (q_km / km_per_au) / (1 - e) * (1 + e)
        if aphelion_au > 6.0:
            planet_params["Saturn barycenter"] = {"name": "Saturn", "color": "#C19A6B"}

    legend_handles = []
    for planet_id, params in planet_params.items():
        state, _ = spkezr(planet_id, et, REF_FRAME_ECLIPTIC, ABERRATION_CORRECTION, SOLAR_SYSTEM_BARYCENTER)
        
        pos_au = state[:3] / km_per_au
        
        p_elts = oscelt(state, et, gm_sun)
        a_p = p_elts[0] / (1 - p_elts[1]) if p_elts[1] < 1 else float('inf')
        t_p = 2 * np.pi * np.sqrt(a_p**3 / gm_sun) if a_p > 0 and np.isfinite(a_p) else float('inf')
        
        if np.isfinite(t_p):
            orbit_path_km = np.array([conics(p_elts, et + i * (t_p / 500))[:3] for i in range(501)])
            orbit_path_au = orbit_path_km / km_per_au
            ax.plot(orbit_path_au[:, 0], orbit_path_au[:, 1], orbit_path_au[:, 2], '-', color=params["color"], linewidth=1.5, alpha=0.7)
        
        h, = ax.plot([pos_au[0]], [pos_au[1]], [pos_au[2]], 'o', color=params["color"], markersize=5, markeredgecolor='k', mew=0.5, label=params["name"])
        ax.text(pos_au[0] * 1.05, pos_au[1] * 1.05, pos_au[2], params["name"], color='k', fontsize=9)
        legend_handles.append(h)
    
    h, = ax.plot([0], [0], [0], 'o', color='yellow', markersize=8, markeredgecolor='k', mew=0.5, label='Sola')
    legend_handles.append(h)

    if e < 1:
        a = q_km / (1.0 - e)
        period = 2 * np.pi * np.sqrt(a**3 / gm_sun)
        et_start, et_end = et, et + period
    else:
        time_span = 5e8
        et_start, et_end = et - time_span, et + time_span

    times = np.linspace(et_start, et_end, 2001)
    orbit_km = np.array([conics(meteor_elements, t)[:3] for t in times])
    orbit_au = orbit_km / km_per_au

    x, y, z = orbit_au[:, 0], orbit_au[:, 1], orbit_au[:, 2]

    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    colors = np.where(z[:-1] >= 0, 'green', 'red')
    
    lc = Line3DCollection(segments, colors=colors, linewidths=2)
    ax.add_collection(lc)

    # --- RESTORED: Plot droplines for the meteoroid ---
    step = max(1, len(x) // 50)
    # Plot "shadow" dots on the ecliptic plane
    ax.plot(x[::step], y[::step], 0, '.', color='gray', markersize=1, alpha=0.5)
    # Plot the vertical droplines
    for i in range(0, len(x), step):
        ax.plot([x[i], x[i]], [y[i], y[i]], [z[i], 0], '-', color='gray', linewidth=0.5, alpha=0.4)

    legend_handles.extend([
        Line2D([0], [0], color='green', lw=1.5, label='Meteoroidbane (over ekliptikken)'),
        Line2D([0], [0], color='red', lw=1.5, label='Meteoroidbane (under ekliptikken)')
    ])

    max_range = np.max(np.ptp(orbit_au, axis=0)) * 1.1
    mid_points = np.mean(orbit_au, axis=0)
    ax.set_xlim(mid_points[0] - max_range / 2, mid_points[0] + max_range / 2)
    ax.set_ylim(mid_points[1] - max_range / 2, mid_points[1] + max_range / 2)
    ax.set_zlim(mid_points[2] - max_range / 2, mid_points[2] + max_range / 2)
    
    ax.set_title('Meteoroidens heliosentriske bane', fontsize=16, y=1, fontstyle="oblique")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_label_text('')

    ax.view_init(elev=30., azim=30)
    ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 0.92))

    if doplot == 'save' or doplot == 'both':
        plt.savefig('orbit.svg', bbox_inches='tight', pad_inches=0.2)
        print("Plot saved to orbit.svg")
    if doplot == 'show' or doplot == 'both':
        plt.show()

def calc_azalt(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    Calculate initial bearing and incidence angle using NumPy for robustness.
    """
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = np.radians([lat1, lon1, lat2, lon2])
    
    dLon = lon2_rad - lon1_rad
    dLat = lat2_rad - lat1_rad
    a = np.sin(dLat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dLon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist_km = 6371 * c

    x = np.sin(dLon) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dLon)
    bearing = np.degrees(np.arctan2(x, y))

    incidence = np.degrees(np.arctan2(alt1 - alt2, dist_km)) if dist_km > 0 else (90.0 if alt1 > alt2 else -90.0)
    
    return bearing, incidence

def orbit(spd_success, v_obs, avg_speed, resname, datestr, timestr, doplot=''):
    """
    Calculates radiant and orbital elements for a meteor observation.
    """
    if not _load_spice_kernels():
        return 0, 0, (0,) * 7, '', '', False

    resdat = readres(resname)
    lon = float(resdat.long1[0])
    lat = float(resdat.lat1[0])
    height = float(resdat.height[0])
    
    azimuth, altitude = calc_azalt(
        lat, lon, height,
        float(resdat.lat1[1]), float(resdat.long1[1]), float(resdat.height[1])
    )
    
    velocity = v_obs if spd_success else avg_speed
    azimuth -= 180.0

    utc_time_str = f"{datestr.replace('-', ' ')} {timestr} UTC"
    et = str2et(utc_time_str)
    
    _, radii_e = bodvrd("EARTH", "RADII", 3)
    
    local_rotation_spd = np.cos(np.radians(lat)) * 2 * np.pi * (radii_e[0] + height) / SIDEREAL_DAY_SECONDS
    
    spd_south = np.cos(np.radians(azimuth)) * np.cos(np.radians(altitude)) * velocity
    spd_west = np.sin(np.radians(azimuth)) * np.cos(np.radians(altitude)) * velocity
    spd_down = np.sin(np.radians(altitude)) * velocity
    
    new_spd_west = spd_west - local_rotation_spd
    v_rotcorr = np.sqrt(spd_south**2 + new_spd_west**2 + spd_down**2)
    
    v_horizontal = np.sqrt(spd_south**2 + new_spd_west**2)
    alt_rc = np.degrees(np.arctan2(spd_down, v_horizontal)) if v_horizontal > 0 else 90.0
    az_rc = np.degrees(np.arctan2(new_spd_west, spd_south))
    
    _, gm_e = bodvrd("EARTH", "GM", 1)
    gm_e = gm_e[0]
    v_esc = np.sqrt(2 * gm_e / (height + radii_e[0]))
    
    if v_rotcorr < v_esc:
        print("Warning: Velocity is below Earth escape velocity. No orbit calculated.")
        return 0, 0, (0,) * 7, '', '', False
        
    v_orbit = np.sqrt(v_rotcorr**2 - v_esc**2)
    zd = 90 - alt_rc
    r = (v_rotcorr - v_orbit) / (v_rotcorr + v_orbit) if (v_rotcorr + v_orbit) > 0 else 0
    dzd = 2 * np.arctan(r * np.tan(np.radians(zd / 2)))
    czd = np.degrees(dzd) + zd
    calt = 90 - czd

    lon_rad, lat_rad = np.radians([lon, lat])
    flatcoeff = (radii_e[0] - radii_e[2]) / radii_e[0]
    pos_vec = georec(lon_rad, lat_rad, height, radii_e[0], flatcoeff)
    
    zenith = unorm(pos_vec)[0]
    east = unorm(vsub(georec(lon_rad + np.radians(1./60), lat_rad, height, radii_e[0], flatcoeff), pos_vec))[0]
    north = unorm(vsub(georec(lon_rad, lat_rad + np.radians(1./60), height, radii_e[0], flatcoeff), pos_vec))[0]
    
    alt_rad, az_rad = np.radians([calt, az_rc])
    objvec = vadd(
        vscl(np.sin(alt_rad), zenith),
        vscl(np.cos(alt_rad), vadd(vscl(np.cos(az_rad), north), vscl(np.sin(az_rad), east)))
    )
    
    j2000_vec = mxv(pxform(REF_FRAME_IAU_EARTH, REF_FRAME_J2000, et), objvec)
    _, ra_rad, dec_rad = recrad(j2000_vec)
    
    eclipj2000_vec = mxv(pxform(REF_FRAME_IAU_EARTH, REF_FRAME_ECLIPTIC, et), objvec)
    _, era_rad, edec_rad = recrad(eclipj2000_vec)
    
    meteor_vel_vec = radrec(-v_orbit, era_rad, edec_rad)
    earth_state, _ = spkezr('EARTH', et, REF_FRAME_ECLIPTIC, ABERRATION_CORRECTION, SOLAR_SYSTEM_BARYCENTER)
    
    meteoroid_state = np.concatenate((earth_state[:3], earth_state[3:] + meteor_vel_vec))
    
    _, gm_sun = bodvrd("SUN", "GM", 1)
    elements = oscelt(meteoroid_state, et, gm_sun[0])

    orbelts = (
        convrt(elements[0], "KM", "AU"),
        elements[1],
        np.degrees(elements[2]),
        np.degrees(elements[3]),
        np.degrees(elements[4]),
        np.degrees(elements[5]),
        et2utc(elements[6], "C", 0)
    )
    
    showername, showername_sg = showerassoc.showerassoc(np.degrees(ra_rad), np.degrees(dec_rad), v_rotcorr, datestr)
    
    if doplot:
        _plot_orbit(et, elements, doplot)
        
    return np.degrees(ra_rad), np.degrees(dec_rad), orbelts, showername, showername_sg, True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot meteor orbit.")
    parser.add_argument('res_file', help="Result file from metrack")
    parser.add_argument('date', help="Date string YYYY-MM-DD")
    parser.add_argument('time', help="Time string HH:MM:SS")
    parser.add_argument('--speed', type=float, default=20.0, help="Observed speed in km/s")
    parser.add_argument('--plot', choices=['show', 'save', 'both'], default='', help="Plotting option")
    
    args = parser.parse_args()

    ra, dec, orbelts, shower, _, valid = orbit(
        spd_success=True, 
        v_obs=args.speed, 
        avg_speed=0, 
        resname=args.res_file, 
        datestr=args.date, 
        timestr=args.time, 
        doplot=args.plot
    )
    
    if valid:
        print("\n--- Calculation Successful ---")
        print(f"Radiant (RA, Dec): {ra:.3f}°, {dec:.3f}°")
        print(f"Associated Shower: {shower if shower else 'Sporadic'}")
        print("\nOrbital Elements (ECLIPJ2000):")
        print(f"  Perifocal Distance (q): {orbelts[0]:.3f} AU")
        print(f"  Eccentricity (e):       {orbelts[1]:.3f}")
        print(f"  Inclination (i):        {orbelts[2]:.3f}°")
        print(f"  Long. Asc. Node (Ω):    {orbelts[3]:.3f}°")
        print(f"  Arg. of Periapse (ω):   {orbelts[4]:.3f}°")
        print("------------------------------")
    else:
        print("\n--- Calculation Failed ---")
