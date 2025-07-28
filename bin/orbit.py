#!/usr/bin/env python3
# Converted to Python 3

# Changelist
# 2015-08-14 Added clf()

from fbspd_merge import ResData, readres
from spiceypy import *
import matplotlib.pyplot as plt
import math
import showerassoc
import numpy as np


def show_planets(et_start, plane='XY'):
    '''
    Plots the orbits and positions of planets for a given time.
    The reference plane is the ecliptic.
    Three projections may be selected: 'XY', 'XZ' or 'YZ'
    '''
    ref    = "ECLIPJ2000"
    corr   = "NONE"
    obs    = "Sun"
    targ   = "Earth"
    planetlist=["Venus", "Earth", "Mars", "Jupiter barycenter"]
    planetname=["Venus", "Earth", "Mars", "Jupiter" ]
    planetcol =["y",  "b",  "m",  "c"]
    planetyr  =[225/365., 1., 687/365., 4333/365.]

    et_end = et_start + 365. * 24 * 3600

    steps = 1000
    dt = (et_end-et_start)/steps

    for i in range(len(planetlist)):
        x=[]
        y=[]
        for t in range(steps+1):
            tm = et_start +  t*dt*planetyr[i]
            state, lt = spkezr(  planetlist[i], tm, ref, corr, obs )
            if plane == 'XY':
                x.append(state[0])
                y.append(state[1])
            elif plane == 'XZ':
                x.append(state[0])
                y.append(state[2])
            elif plane == 'YZ':
                x.append(state[1])
                y.append(state[2])
            else:
                print("Illegal plane specification")
                exit()

        plt.plot(x,y, color=planetcol[i], linestyle='-')
        plt.plot(x[0],y[0], 'o', color=planetcol[i])
        plt.text(x[0],y[0],planetname[i])

    state, lt = spkezr(  "Sun", et_start, ref, corr, obs )
    plt.plot(state[0], state[1], 'yo')
    plt.text(state[0], state[1], "Sun")
    plt.gca().set_aspect('equal', adjustable='box')


# Calculate initial bearing and incidence.
def calc_azalt(lat1, lon1, alt1, lat2, lon2, alt2):
    lat1, lon1, lat2, lon2 = math.radians(lat1), math.radians(lon1), math.radians(lat2), math.radians(lon2)
    dist = math.acos(math.sin(lat1)*math.sin(lat2) + math.cos(lat1)*math.cos(lat2)*math.cos(lon2-lon1)) * 6371
    bearing = math.degrees(math.atan2(math.sin(lon2-lon1)*math.cos(lat2),
                                      math.cos(lat1)*math.sin(lat2) -
                                      math.sin(lat1)*math.cos(lat2)*math.cos(lon2-lon1)))
    if dist > 0:
        incidence = math.degrees(math.atan2(alt1-alt2, dist))
    else:
        incidence = 90.0 if alt1 > alt2 else -90.0
    
    return bearing, incidence
                                   
def orbit(spd_success, v_obs, avg_speed, resname, datestr, timestr, doplot=''):
    '''
    NAME:
        orbit
    PURPOSE:
        Calculates radiant and orbital elements for a meteor observation.
    CALLING SEQUENCE:
        ra, dec, orbelts, showername, showername_sg, valid = orbit(<spd_success>, <v_obs>, <avg_speed>, <resname>, <datestr>, <timestr> [, doplot='<option>'])
    INPUTS:
        <spd_success> Boolean, true if a atmospheric entry speed has been
        calculated.
        <v_obs> The fitted atmospheric entry speed.
        <avg_speed> Observed average speed, only used if the entry speed
        is not available.
        <resname> Name of the result file created by metrack.py 
        <datestr> UTC date, format YYYY-MM-DD
        <timestr> UTC time, format HH:MM:SS
    OPTIONAL INPUTS
        doplot='<option>'
          '' : Do not output any graphics          
          'show' : Show graphics on screen
          'save' : Save graphics to file 'orbit.svg'
          'both' : Show and save
    OUTPUTS:
        The R.A. and declination of the radiant, corrected for Earth's
        rotation speed and Zenith attraction.
        Tuple of orbital elements in the ECLIPJ2000 system:    
         Perifocal distance [AU]
         Eccentricity       
         Inclination        [deg]
         Long. Asc. Node    [deg]
         Arg. of Periapse   [deg]
         Mean Anom @ Epoch  [deg]
         Epoch              [UTC]
         G*M_Sun
        Shower name (if any)
        Shower name singular (if any)
        Validity flag (boolean)
    '''

    import time
    import os

    resdat = readres(resname)
    lon = resdat.long1[0]
    lat = resdat.lat1[0]
    height = resdat.height[0]
    az, alt = calc_azalt(resdat.lat1[0], resdat.long1[0], resdat.height[0], resdat.lat1[1], resdat.long1[1], resdat.height[1])

    if not spd_success:
        v_obs = avg_speed

    # Define kernel path. Check for a few common locations.
    kernelpath = None
    possible_paths = [
        './data/',
        '/var/www/html/bin/data',
        os.path.expanduser('~/spice/data/')
    ]
    for p in possible_paths:
        if os.path.exists(p):
            kernelpath = p
            break
    
    if not kernelpath:
        print("SPICE kernel directory not found. Please check paths in orbit.py")
        return 0, 0, (0, 0, 0, 0, 0, 0, ''), '', '', False

    #furnsh( os.path.join(kernelpath, "lsk/naif0012.tls") )
    furnsh( os.path.join(kernelpath, "lsk/naif0010.tls") )
    furnsh( os.path.join(kernelpath, "spk/planets/de421.bsp") )
    #furnsh( os.path.join(kernelpath, "pck/gm_de431.tpc") )
    furnsh( os.path.join(kernelpath, "pck/de-403-masses.tpc") )
    furnsh( os.path.join(kernelpath, "pck/pck00010.tpc") )

    lon_rad = lon * rpd()
    lat_rad = lat * rpd()
    tm = datestr.replace('-',' ') + ' ' + timestr + ' UTC'

    # Input az is course, i.e. moving direction
    az = az - 180. # Working az is direction coming from

    et = str2et( tm )

    # Read Earth radius (3 axes)
    n, radii_e = bodvrd( "EARTH", "RADII", 3 )

    # Earth rotation speed 
    sidday = 23.9344696 * 3600 # sec
    locspd = math.cos(lat_rad)*math.pi*2*(radii_e[0]+height) / sidday

    spd_south = math.cos(az * rpd()) * math.cos(alt * rpd()) * v_obs
    spd_west = math.sin(az * rpd()) * math.cos(alt * rpd()) * v_obs
    spd_down = math.sin(alt * rpd()) * v_obs
    new_spd_west = spd_west - locspd

    v_rotcorr = math.sqrt(spd_south**2 + new_spd_west**2 + spd_down**2)
    v_hor = math.sqrt(spd_south**2 + new_spd_west**2)

    alt_rc = 90 - math.atan2(v_hor, spd_down) * dpr() if spd_down > 0 else 0
    az_rc = math.atan2(new_spd_west, spd_south) * dpr()

    # Read gravitational constant * Mass_Earth, unit: Kilometers
    try:
        n, GM_e = bodvrd( "EARTH", "GM", 1 )
        GM_e = GM_e[0]
    except SpiceyError:
        # Fallback if GM is not in the kernel under that name
        n, GM_e = bodvrd( "EARTH BARYCENTER", "GM", 1 )
        GM_e = GM_e[0]


    # Find Earth escape velocity in km/sec
    v_esc = math.sqrt( 2 * GM_e / ( height + radii_e[0] ) )

    if v_rotcorr < v_esc:
        # Sub-orbital velocity!
        print("Warning: Velocity is below Earth escape velocity. No orbit calculated.")
        return 0, 0, (0, 0, 0, 0, 0, 0, ''), '', '', False
 
    v_orbit = math.sqrt(v_rotcorr**2 - v_esc**2)

    # Correct for Zenith attraction
    zd = 90 - alt_rc
    if (v_rotcorr + v_orbit) > 0:
        r = ( v_rotcorr - v_orbit ) / ( v_rotcorr + v_orbit )
        dzd = 2 * math.atan( r * math.tan(math.radians(zd/2)) )
        czd = math.degrees(dzd) + zd
    else:
        czd = zd
    calt = 90 - czd

    flatcoeff = (radii_e[0]-radii_e[2])/radii_e[0];
    vec = georec( lon_rad, lat_rad, height, radii_e[0], flatcoeff )
    
    zenith = unorm( vec )[0]

    dlon = lon_rad + (1./60)*rpd()
    dvec = georec( dlon, lat_rad, height, radii_e[0], flatcoeff )
    east = unorm(vsub(dvec,vec))[0]

    dlat = lat_rad + (1./60)*rpd()
    dvec = georec( lon_rad, dlat, height, radii_e[0], flatcoeff )
    north = unorm(vsub(dvec,vec))[0]

    alt_rad = calt * rpd()
    az_rad = az_rc * rpd()

    objvec =  vadd( vscl(math.sin(alt_rad), zenith) ,  vscl(math.cos(alt_rad) ,  vadd( vscl(math.cos(az_rad), north) , vscl(math.sin(az_rad), east) ) ) )

    pxmat = pxform("IAU_EARTH", "J2000", et)
    j2objvec = mxv(pxmat, objvec)
    ra_rad , dec_rad = recrad(j2objvec)[1:]

    showername, showername_sg = showerassoc.showerassoc(ra_rad*dpr(), dec_rad*dpr(), v_rotcorr, datestr)

    epxmat = pxform("IAU_EARTH", "ECLIPJ2000", et)
    ej2objvec = mxv(epxmat, objvec)
    era_rad, edec_rad = recrad(ej2objvec)[1:]

    # Calculate the meteoroid's velocity vector relative to the Earth
    espdvec = radrec( -v_orbit, era_rad, edec_rad )

    # Get Earth's state vector (position and velocity) relative to the Sun
    state, lt = spkezr('EARTH', et, "ECLIPJ2000", "NONE", "Sun")

    # Combine Earth's state with the meteoroid's relative velocity
    # to get the meteoroid's final state vector relative to the Sun.
    modstate = np.array([
        state[0],
        state[1],
        state[2],
        state[3] + espdvec[0],
        state[4] + espdvec[1],
        state[5] + espdvec[2]
    ])

    n, gm = bodvrd("SUN", "GM", 1)
    gm = gm[0]
    elts = oscelt( modstate, et, gm)

    orbelts = [ convrt(elts[0], "KM", "AU"), \
                elts[1], \
                elts[2] * dpr(), \
                elts[3] * dpr(), \
                elts[4] * dpr(), \
                elts[5] * dpr(), \
                et2utc( elts[6], "C", 0) ]

    if doplot != '':
        plt.figure(1, figsize=(10,10))
        plt.clf()
        ax = plt.subplot(111)
        ax.set_facecolor('k')
        plt.axis('off')
        show_planets(et, plane='XY')

        if elts[1] < 1:
            a = elts[0] / (1. - elts[1])
            t = 2 * math.pi * math.sqrt(a**3 / elts[7])
            et_end = et + t
            et_start = et
        else:
            et_end   = et + 1e8
            et_start = et - 1e8

        steps = 1000
        dt = (et_end-et_start)/steps

        xh=[]
        yh=[]
        zh=[]
        xl=[]
        yl=[]
        zl=[]

        for t_step in range(steps+1):
            tm = et_start +  t_step*dt
            state = conics( elts, tm )
            if state[2] > 0:
                xh.append(state[0])
                yh.append(state[1])
                zh.append(state[2])
            else:
                xl.append(state[0])
                yl.append(state[1])
                zl.append(state[2])


        # Plot the meteoroid's orbit
        plt.plot(xh,yh, 'g-')
        plt.plot(xl,yl, 'r--')

        # --- ADD THIS NEW BLOCK TO AUTO-TRIM THE PLOT ---
        all_x_coords = []
        all_y_coords = []

        # Get coordinates from all plotted lines (planets and meteoroid)
        for line in ax.get_lines():
            all_x_coords.extend(line.get_xdata())
            all_y_coords.extend(line.get_ydata())

        # Auto-scale plot to fit all data with some padding
        if all_x_coords and all_y_coords:
            min_x, max_x = min(all_x_coords), max(all_x_coords)
            min_y, max_y = min(all_y_coords), max(all_y_coords)

            # Determine the center and the maximum range needed
            range_x = max_x - min_x
            range_y = max_y - min_y
            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2
            max_range = max(range_x, range_y)

            # Set new limits to create a square box around the data with 2% padding
            padding = max_range * 0.02
            ax.set_xlim(center_x - max_range / 2 - padding, center_x + max_range / 2 + padding)
            ax.set_ylim(center_y - max_range / 2 - padding, center_y + max_range / 2 + padding)
        # --- END OF NEW BLOCK ---

        if doplot == 'save' or doplot == 'both':
            plt.savefig('orbit.svg')
        if doplot == 'show' or doplot == 'both':
            plt.show()

    return ra_rad*dpr(), dec_rad*dpr(), orbelts, showername, showername_sg, True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate and plot meteor orbit.")
    parser.add_argument('res_file', help="Result file from metrack")
    parser.add_argument('date', help="Date string YYYY-MM-DD")
    parser.add_argument('time', help="Time string HH:MM:SS")
    parser.add_argument('--speed', type=float, default=20.0, help="Observed speed in km/s")
    parser.add_argument('--plot', choices=['show', 'save', 'both'], default='', help="Plotting option")
    
    args = parser.parse_args()

    orbit(True, args.speed, 0, args.res_file, args.date, args.time, doplot=args.plot)
