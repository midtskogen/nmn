#!/usr/bin/env python3
# Converted to Python 3

import showerdata
import time
from spiceypy import radrec, rpd, vsep, dpr
import numpy as np



def interpol(coordlist, datelist, date):
    '''
    Interpolate ephemerisis table data to date of observation
    '''
    ndat = len(coordlist)
    datenum=[]
    for d in datelist:
        try:
            t = time.strptime(d, '%m-%d')
            yday = t.tm_yday
            datenum.append(yday)
        except ValueError:
            print(f"Warning: Could not parse date '{d}' in showerdata. Skipping.")
            return None


    # Handle swarms occurring over year changes
    if datenum:
        minpos = datenum.index(min(datenum))
        if minpos > 0:
            # Check if the range wraps around the new year
            if max(datenum) - min(datenum) > 180: # Heuristic for year wrap
                for i in range(minpos):
                    datenum[i] = datenum[i] + 365
    
    if len(datenum) < 3: # Cannot do a quadratic fit with less than 3 points
        if len(datenum) > 0:
             return np.mean(coordlist) # Fallback to mean
        else:
             return None

    p = np.polyfit(datenum, coordlist, 2)
    
    fitval = np.polyval(p, date)

    return fitval



def radsep(ra_obs, decl_obs, shower, yday):
    '''
    Calculate angular separation between observed radiant and
    ephmerisis radiant.
    Input: radiant ra, decl in degrees
    Output: Separation in degrees
    '''
    ra_shower = interpol(shower.ra,  shower.rad_date, yday)
    dec_shower= interpol(shower.dec, shower.rad_date, yday)

    if ra_shower is None or dec_shower is None:
        return 999 # Return a large separation if interpolation fails

    showervec = radrec(1, ra_shower*rpd(), dec_shower*rpd())
    obsvec = radrec(1, ra_obs*rpd(), decl_obs*rpd())

    sep = vsep(obsvec, showervec)

    return sep*dpr()



def check_showerdata():
    '''
    Check if lengths of lists are identical and format of date-string
    '''

    import matplotlib.pyplot as plt

    for i in showerdata.showerlist:
        if not ( len(i.ra) == len(i.dec) == len(i.rad_date)):
            print("List length mismatch")
            print(i.name)
        for j in i.rad_date:
            try:
                tmp = time.strptime(j, '%m-%d')
            except ValueError:
                 print(f"Bad date format '{j}' in shower '{i.name}'")
        plt.plot(i.ra, i.dec, 'o-', label=i.name)
    plt.xlabel("R.A. (deg)")
    plt.ylabel("Dec. (deg)")
    plt.title("Shower Radiant Positions")
    plt.grid(True)
    plt.show()



def showerassoc(ra_obs, decl_obs, spd_obs, indate):
    '''
    Try to associate observed meteor with known shower.
    Input: radiant ra, decl in degrees
    Input: speed in km/s
    '''

    try:
        t = time.strptime(indate, '%Y-%m-%d')
    except ValueError:
        print(f"Error: Invalid date format for observation: {indate}")
        return '', ''


    bestscore = 0.
    matchname = ''
    matchname_sg = ''

    nshowers = len(showerdata.showerlist)
    for i in range(nshowers):
        yday_obs = t.tm_yday
        try:
            tb = time.strptime(showerdata.showerlist[i].beg_date,'%m-%d')
            yday_beg = tb.tm_yday
            te = time.strptime(showerdata.showerlist[i].end_date,'%m-%d')
            yday_end = te.tm_yday
        except ValueError:
            print(f"Warning: Invalid date format in shower data for {showerdata.showerlist[i].name}. Skipping.")
            continue

        datematch = False
        
        # Handle showers that cross the new year
        if yday_beg > yday_end:
            if yday_obs >= yday_beg or yday_obs <= yday_end:
                datematch = True
                if yday_obs <= yday_end: # Adjust observation day for interpolation
                    yday_obs += 365
        else:
            if yday_beg <= yday_obs <= yday_end:
                datematch = True

        if datematch:
            #print("Datematch for", showerdata.showerlist[i].name)

            sep = radsep(ra_obs, decl_obs, showerdata.showerlist[i], yday_obs)
            #print("Separation:", sep)
            if sep < 10: 
                radmatch = True
            else:
                radmatch = False

            spderr = abs(showerdata.showerlist[i].v_inf - spd_obs) / showerdata.showerlist[i].v_inf
            #print("Spderr:", spderr)
            if spderr < 0.2: 
                speedmatch = True
            else:
                speedmatch = False

            if radmatch and speedmatch:
                score = (1./(spderr+0.1)) + 1./((sep+1)/55)
                if score > bestscore:
                    bestscore = score
                    matchname = showerdata.showerlist[i].name
                    matchname_sg = showerdata.showerlist[i].name_sg

    return matchname, matchname_sg



if __name__ == "__main__":

    # check_showerdata()

    print(showerassoc( 265., 35., 45., '2014-04-19')) # Lyride
    print("----------------------")
    print(showerassoc( 230., 50., 40., '2014-01-01')) # Quadrantide
    print("----------------------")
    print(showerassoc( 230., 50., 40., '2014-12-31')) # Quadrantide
    print("----------------------")
    print(showerassoc( 230., 50., 40., '2014-06-01')) # No match
    print("----------------------")
    print(showerassoc( 151., 22., 65., '2014-11-06')) # Leonide, date overlapping other swarms
    print("----------------------")
    print(showerassoc( 120., 81., 35., '2014-05-16')) # May Camelopardalide
