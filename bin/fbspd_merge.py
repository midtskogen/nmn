#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted to Python 3

# 2040413: If a fit to merged data fails, result is replaced with best single station fit, if possible.
# 20140418 Added site code in title string
# 20150906 Added abs() to fitting function to force meaningful solution


import numpy as np

earth_radius = 6371.0 # km


def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def get_sitecoord_fromdat(inname):

    import os

    path, longname = os.path.split(inname)
    (name, ext) = os.path.splitext(longname)

    # Pre-declare data lists
    longarr = []
    latarr =  []
    az1arr =  []
    az2arr =  []
    alt1arr=  []
    alt2arr = []
    weights = []
    duration =[]
    lenarr =  []
    colarr =  []
    namarr =  []
    secarr =  []
    obsheightarr = []

    # Read in data
    with open(os.path.join(path, name+ext), 'r') as f:
        for line in f:
            words=line.split()
            if not words:
                continue
            if words[0] == 'borders':
                borders = [float(words[1]), float(words[2]), float(words[3]), float(words[4]) ]
                autoborders = False
            elif line[0] != '#':
                longarr.append(float(words[0]))
                latarr.append(float(words[1]))
                az1arr.append(float(words[2]))
                az2arr.append(float(words[3]))
                alt1arr.append(float(words[4]))
                alt2arr.append(float(words[5]))
                weights.append(float(words[6]))
                duration.append(float(words[7]))
                lenarr.append(float(words[8]))
                colarr.append( (int(words[9]), int(words[10]), int(words[11])) )
                nam=''
                nam_nsegs=0
                for n in words[12:]: 
                    if not is_number(n):
                        nam=nam+' '+n
                        nam_nsegs += 1
                nam=nam[1:]
                namarr.append(nam)
                try:
                    secarr.append(float(words[12+nam_nsegs]))
                except (IndexError, ValueError):
                    # Will this have any use in the future?
                    secarr.append(None)
                try:
                    obsheightarr.append(float(words[13+nam_nsegs])/1000.)
                except (IndexError, ValueError):
                    obsheightarr.append(0.)
                
    sitecoords = []
    for i in range(len(longarr)):
      sitecoords.append([namarr[i], longarr[i], latarr[i], obsheightarr[i]])
    
    return sitecoords


class ResData:
    '''
    Container data type for .res file
    '''
    def __init__(self):
        self.ndata  = 0.
        self.long1  = np.array([])
        self.lat1   = np.array([])
        self.long2  = np.array([])
        self.lat2   = np.array([])
        self.height = np.array([])
        self.desc   = np.array([])



def readres(inname):
    '''
    Read in data from .res file and return them in data type
    '''

    import os

    path, longname = os.path.split(inname)
    (name, ext) = os.path.splitext(longname)

    # Pre-declare data lists
    long1  = []
    lat1   = []
    long2  = []
    lat2   = []
    height = []
    desc   = []

    # Read in data
    with open(os.path.join(path, name+ext), 'r') as f:
        for line in f:
            if line and line[0] != '#':
                words=line.split()
                long1.append(float(words[0]))
                lat1.append(float(words[1]))
                long2.append(float(words[2]))
                lat2.append(float(words[3]))
                height.append(float(words[4]))
                desc.append(words[5])

    data = ResData()

    data.ndata = len(long1)
    data.long1 =  np.array(long1)
    data.lat1   = np.array(lat1)
    data.long2  = np.array(long2)
    data.lat2   = np.array(lat2)
    data.height = np.array(height)
    data.desc   = np.array(desc)

    return data



class CenData:
    '''
    Container data type for centroid file
    '''
    def __init__(self):
        self.ndata   = 0.
        self.seqid   = np.array([])
        self.reltime = np.array([])
        self.cenalt  = np.array([])
        self.cenaz   = np.array([])
        self.censig  = np.array([])
        self.sitestr = np.array([])
        self.datestr = np.array([])
        self.timestr = np.array([])



def readcen(inname):
    '''
    Read in data from centroid file and return them in data type
    '''

    import os

    path, longname = os.path.split(inname)
    (name, ext) = os.path.splitext(longname)

    # Pre-declare data lists
    seqid   = []
    reltime = []
    cenalt  = []
    cenaz   = []
    censig  = []
    sitestr = []
    datestr = []
    timestr = []

    # Read in data
    with open(os.path.join(path, name+ext), 'r') as f:
        for line in f:
            if line and line[0] != '#':
                words=line.split()
                seqid.append(int(words[0]))
                reltime.append(float(words[1]))
                cenalt.append(float(words[2]))
                cenaz.append(float(words[3]))
                censig.append(float(words[4]))
                sitestr.append(words[5])
                datestr.append(words[6])
                timestr.append(words[7])

    data = CenData()

    data.ndata   = len(seqid)
    data.seqid   = np.array(seqid)
    data.reltime = np.array(reltime)
    data.cenalt  = np.array(cenalt)
    data.cenaz   = np.array(cenaz)
    data.censig  = np.array(censig)
    data.sitestr = np.array(sitestr)
    data.datestr = np.array(datestr)
    data.timestr = np.array(timestr)

    return data


def lonlat2xyz(lon, lat, height = 0):
    """
    Convert geographic lon/lat/height on reference ellipse to cartesian coordiantes
    """
    
    # http://www.navipedia.net/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
    #
    f = 1.0/298.257223563 # Earth flattening factor WGS-84
    e = np.sqrt(2*f - f*f)
    N = earth_radius / np.sqrt(1 - e*e * np.sin(np.radians(lat))*np.sin(np.radians(lat)))

    v = np.zeros(3)
    v[0] = (N + height) * np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    v[1] = (N + height) * np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    v[2] = (N*(1 - e*e) + height)  * np.sin(np.radians(lat))

    return v


def lonlat2xyz_globe(lon, lat, height = 0):
    """
    Convert geographic lon/lat/height to cartesian coordiantes
    """
    
    v = np.zeros(3)
    v[0] = np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    v[1] = np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    v[2] = np.sin(np.radians(lat))
    v = v * (earth_radius + height)

    return v


def xyz2lonlat(v):
    """
    Convert cartesian coordinates to geographical lon/lat/height on reference ellipse
    """

    # http://www.navipedia.net/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
    #
    f = 1.0/298.257223563 # Earth flattening factor WGS-84
    e = np.sqrt(2*f - f*f)

    lon    = np.degrees(np.arctan2(v[1], v[0]))
    
    p = np.sqrt(v[0]*v[0] + v[1]*v[1])
    lat_old = np.degrees(np.arctan2(v[2], (1.0 - e*e)*p))

    diff = 1
    nsteps = 0

    while (diff > 1e-10):
      nsteps = nsteps + 1
      N = earth_radius / np.sqrt(1.0 - e*e*np.sin(np.radians(lat_old))*np.sin(np.radians(lat_old)))
      height = p / np.cos(np.radians(lat_old)) - N
      lat    = np.degrees(np.arctan2(v[2], p * (1.0 - e*e * N / (N + height))))
      
      diff = abs((lat - lat_old))
      #print(diff, N, lon, lat, height)
      
      lat_old      = lat
      
      if (nsteps > 1000):
        print(f"Warning - did not reach full convergence in xyz2lonlat: current diff is {diff}\n")
        break

    return lon, lat, height


def xyz2lonlat_globe(v):
    """
    Convert cartesian coordinates to geographical lon/lat/height
    """

    lon    = np.degrees(np.arctan2(v[1], v[0]))
    lat    = np.degrees(np.arctan2(v[2], np.sqrt(v[0]**2 + v[1]**2)))
    height = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2) - earth_radius
    
    return lon, lat, height


def closest_point(p1, p2, u1, u2, return_points=False):
    """
    Find the point that is closest to two lines, where the lines are
    defined by positions p1 and p2, with directional vectors u1 and u2
    """

    p21 = p2 - p1
    m   = np.cross(u2, u1)
    m2  = float(np.dot(m, m))

    if m2 < 1e-9: # Lines are parallel
        if return_points:
            return p1, p1 + np.dot(p21, u1) * u1
        else:
            return p1 + np.dot(p21, u1) * u1 / 2.0


    R = np.cross(p21, m/m2)
    t1 = np.dot(R, u2)
    t2 = np.dot(R, u1)

    cross_1 = p1 + t1*u1
    cross_2 = p2 + t2*u2

    if return_points: return cross_1, cross_2

    return (cross_1 + cross_2)/2.0


def dist_line_line(p1, u1, p2, u2):
    """
    Calculate the distance between two vectors. These are defined by
    positions p1 and p2, with directional vectors u1 and u2
    """

    # Setup an array of two equations based on the fact that
    # the shortest line connecting both vectors must be perpendicular
    # to both.

    a = np.array( [[np.dot(u1, u1), -np.dot(u1, u2)],
                   [np.dot(u2, u1), -np.dot(u2, u2)]] )
    b = np.array(  [np.dot(u1, (p2 - p1)),
                    np.dot(u2, (p2 - p1))] )

    #print(p1, u1)
    #print(p2, u2)
    #print(a, b)

    try:
        # Solve the set of two equations
        [s, t] = np.linalg.solve(a, b)
    except np.linalg.LinAlgError: # Singular matrix, lines are parallel
        return np.sqrt(np.dot(p2-p1, p2-p1))


    # Then, the vector connecting both lines is    
    pq = (p1 + s*u1) - (p2 + t*u2)

    # The length of this vector is
    dist = np.sqrt( np.dot(pq, pq) )
    
    return dist


def intersec_line_plane(line_ref, line_vec, plane_ref, plane_vec):
    '''
    Determine the point where a line intersects a plane.
    '''

    # Line is defined by reference point and directional vector
    # Plane is defined by reference point and normal vector
    
    dot_prod = np.dot(plane_vec, line_vec)
    if abs(dot_prod) < 1e-9: # Line is parallel to the plane
        return line_ref # No unique intersection

    factor = -np.dot(plane_vec, line_ref - plane_ref) / dot_prod
    vec_to_intersection = line_vec * factor
    intersec_point = line_ref + vec_to_intersection
    #print(intersec_vec)
    #print("The following must be zero: %5.3f" % np.dot(intersec_point - plane_ref, plane_norm))
    
    return intersec_point


def rotation_matrix(axis,theta):
    # from http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def altaz2xyz(alt, az, lon, lat):
    """
    Convert alt/az to vector in cartesian space
    """

    v = np.array([-1,0,0]) # alt=0, az=0
    # rotate alt
    axis = np.array([0,1,0])
    theta = -np.radians(alt) # alt
    v = np.dot(rotation_matrix(axis,theta),v)

    # rotate az
    axis = np.array([0,0,1])
    theta = np.radians(az) # az 
    v = np.dot(rotation_matrix(axis,theta),v)

    # rotate geographic latitude
    axis = np.array([0,1,0])
    theta = np.radians(lat-90) # latitude
    v = np.dot(rotation_matrix(axis,theta),v)

    # rotate geographic longitude
    axis = np.array([0,0,1])
    theta = -np.radians(lon) # longitude
    v = np.dot(rotation_matrix(axis,theta),v)

    return v



def linfunc(x, a, b):
    '''
    Template for weighted fit using scipy.optimize.curve_fit
    '''
    return a*x + b



def expfunc(x, a, b, c, d):
    '''
    Template for weighted fit using scipy.optimize.curve_fit
    '''
    # Use abs() to force solution to physically meningful regime
    return -abs(a)*np.exp(abs(b)*x) + abs(a)*abs(b)*x + abs(a) + c*x + d


def expfunc_1stder(x, a, b, c, d):
    '''
    First derivative of expfunc()
    '''
    # Use abs() to force solution to physically meningful regime
    return -abs(a)*abs(b)*np.exp(abs(b)*x) + abs(a)*abs(b) + c


def expfunc_2ndder(x, a, b, c, d):
    '''
    Second derivative of expfunc()
    '''
    # Use abs() to force solution to physically meningful regime
    return -abs(a)*abs(b)*abs(b)*np.exp(abs(b)*x)



def guess_expfit(x, y, weights):
    '''
    Try fitting with different initial guesses on parameters
    Return on first success or when all fails
    '''
    guesslist = [ [0.1,  1.0, 20.0, 1.0],
                  [0.1,  1.0, 20.0, 0.0],
                  [0.1,  1.0, 20.0, -1.0],
                  [1.0,  1.0, 25.0, 0.0],
                  [1e-3, 1.0, 25.0, 0.0],
                  [1e-6, 1.0, 25.0, 0.0],
                  [0.1,  1.0, 10.0, 0.0],
                  [0.1,  1.0, 20.0, 0.0],
                  [0.1,  1.0, 30.0, 0.0],
                  [0.1,  1.0, 40.0, 0.0],
                  [0.1,  1.0, 50.0, 0.0],
                  [0.1,  1.0, 60.0, 0.0],
                ]

    success = False
    par = np.array([0, 0, 0, 0])
    
    current_x, current_y, current_weights = np.copy(x), np.copy(y), np.copy(weights)

    while not success and len(current_x) > 4:
        for g in guesslist:
            par, success = try_expfit(current_x, current_y, current_weights, g)
            if success:
                break
        if not success:
            # Try to eliminate strongly decelerated/afterglow part of track
            current_x, current_y, current_weights = current_x[:-1], current_y[:-1], current_weights[:-1]

    return par, success, len(current_x)
        


def try_expfit(x, y, weights, guess=None):
    '''
    Tries to fit a lin-exp function to data.
    Returns parameters and True if the fit was possible and the
    result plausible.
    '''
    from scipy.optimize import curve_fit

    success = True
    try:
        exp_params, pcov = curve_fit(expfunc, x, y,  guess, weights)
    except Exception:
        success = False
        #print("Failed to fit data")
        exp_params=np.array([0,0,0,0])
    if success:
        exp_params[0] = -abs(exp_params[0])
        exp_params[1] =  abs(exp_params[1])
        # Allow slightly incorrect values:
        if (exp_params[2] <0) or (exp_params[2] >300):

            success = False
            #print("Fit was not reasonable")
            #print(exp_params)
            exp_params=np.array([0,0,0,0])

    if success:
        exp_fitval = expfunc(x, exp_params[0], exp_params[1], exp_params[2], exp_params[3])
        exp_speed  = expfunc_1stder(x, exp_params[0], exp_params[1], exp_params[2], exp_params[3])
        #print("Min speed", min(exp_speed))
        exp_accel  = expfunc_2ndder(x, exp_params[0], exp_params[1], exp_params[2], exp_params[3])
        #print("Max accel", max(exp_accel))
        if min(exp_speed) < -1 or max(exp_accel) > 0:
            success = False
            #print("Fit yielded unreasonable speed/accel")
            #print(exp_params)
            exp_params=np.array([0,0,0,0])

    #if success:
    #    print("try_expfit Success!")
    return exp_params, success


def chainlength(alltimearr, allposarr, offsets):

    timearr = []
    posarr = []

    for j in range(len(alltimearr)):
      for k in range(len(alltimearr[j])):
         timearr.append(alltimearr[j][k] + offsets[j])
         posarr.append(allposarr[j][k])
    
    timearr = np.array(timearr)
    posarr = np.array(posarr)

    sortidx = np.argsort(posarr)
    sorted_timearr = timearr[sortidx]
    sorted_posarr = posarr[sortidx]
    if len(sorted_timearr) < 2:
        return 1e10
    seglengths = (sorted_timearr[1:] - sorted_timearr[0:-1])**2 + (sorted_posarr[1:] - sorted_posarr[0:-1])**2
    return seglengths.sum()


def minimize_chainlength(alltimearr, allposarr):

    # Use 1-d 'grid-search' to find minmimize chainlength in time domain

    n_series = len(alltimearr)
    offsets = np.zeros(n_series)
    best_offsets = np.zeros(n_series)

    if (n_series < 2): return best_offsets
    
    for i in range(1, n_series):
      bestcl = 1e10
      for try_offset in np.arange(-10, 10, 0.01):
        thiscl = chainlength([alltimearr[0], alltimearr[i]], [allposarr[0], allposarr[i]], [0, try_offset])
        if (thiscl < bestcl):
          best_offsets[i] = try_offset
          bestcl = thiscl

    return best_offsets


def fbspd(resname, cennames, datname, doplot='', posdata=False, debug=False):
    '''
    Determines speed and acceleration profiles for meteors from
    trajectory parameters and centroid files.
    A list of centroid files can be provided - a profile is fitted to each and
    the centroid data are merged from all plausible fits and a final profile
    fitted.
    A boolean describing the success of creating a final fit is returned, also
    the fitted initial speed is returned.
    Speed/accel graphs can be optionally saved/displayed.
    Created july 2013
    Revised july 23, 2013
    '''

    import matplotlib
    if doplot == 'save':
      if (matplotlib.get_backend() != 'agg'): matplotlib.use('agg')
    import matplotlib.pyplot as plt

    import numpy.linalg
    from scipy.optimize import curve_fit

    resdat = readres(resname)

    pathpos1 = lonlat2xyz(resdat.long1[0], resdat.lat1[0], resdat.height[0])
    pathpos2 = lonlat2xyz(resdat.long1[1], resdat.lat1[1], resdat.height[1])
    p1p2 = pathpos2 - pathpos1
    p1p2norm = p1p2 / np.sqrt(np.dot(p1p2, p1p2))

    sitedata = []
    if (datname):
       sitedata = get_sitecoord_fromdat(datname)
    else:
       # If no datfile given, do the best we can (no altitude info available though)
       print("Warning: not .dat file given, so no height data available for sites. Assuming zero heights.")
       for i in range(2, len(resdat.long1), 1):
         sitedata.append([resdat.desc[i], resdat.long1[i], resdat.lat1[i], 0])

    allcendat=[]
    for cenname in cennames:
        try:
            tmp = readcen(cenname)
            allcendat.append(tmp)
        except FileNotFoundError:
            print(f"Warning: Centroid file not found: {cenname}")


    allposarr = []
    alltimearr = []
    allheightarr = []
    allsigarr = []
    alldistarr = []
    mergedposarr=[]
    mergedheightarr=[]
    mergedreltime=[]
    mergedsigarr=[]
    mergeddistarr=[]
    mergednames = ""

    reffit = None

    if doplot != '':
       plt.close('all')

    if (debug and (doplot == 'show' or doplot == 'both')):
       # For checking the timing offset
       plt.figure(4)
       plt.subplot(211)
       plt.title('Individual tracks')
       plt.ylabel('Position [km]')
       syms=['g.','r.','b.','y.']

    cendat_lenghts = []
    for n in range(len(allcendat)): cendat_lenghts.append(allcendat[n].ndata)
    sorted_cendat_index = sorted(range(len(cendat_lenghts)), key=lambda k: cendat_lenghts[k], reverse=True)

    for nn in range(len(allcendat)):
        allcenidx = sorted_cendat_index[nn]
        cendat = allcendat[allcenidx]

        site_lon, site_lat, site_height = (None, None, None)
        for site in sitedata:
          if (cendat.sitestr[0] == site[0]):
            site_lon, site_lat, site_height = site[1], site[2], site[3]
            break
        
        if site_lon is None:
            print(f"Warning: Could not find site coordinates for {cendat.sitestr[0]}. Skipping this centroid file.")
            continue


        sitepos = lonlat2xyz(site_lon, site_lat, site_height)

        # Alt : 0 @ horizon, 90 @ zenith
        # Az : 0 @ north, 90 @ east

        # Long: 0 @ Meridian, positive east
        # Lat: 0 @ Equator, positive north

        posarr = []
        timearr = []
        distarr = []
        heightarr = []
        sigarr = []

        timearr = allcendat[allcenidx].reltime
        
        for i in range(cendat.ndata):

            v = altaz2xyz(cendat.cenalt[i], cendat.cenaz[i], site_lon, site_lat)

            coords = []
            # Determine the point on the track that is closest to the observed line of sight
            (intersec_on_los, intersec_on_track) = closest_point(sitepos, pathpos1, v, p1p2norm, return_points=True)

            # Where is this point located along the track?
            posarr.append(np.sqrt(np.sum(np.dot(intersec_on_track - pathpos1, intersec_on_track - pathpos1) / np.dot(p1p2norm, p1p2norm))) * np.sign(np.dot(intersec_on_track - pathpos1, p1p2norm)))
            # And determine the height of this particular point
            poslon, poslat, posheight = xyz2lonlat(intersec_on_track)
            heightarr.append(posheight)

            # Find smallest distance between path line and line of sight (for debugging purposes only)
            pq = dist_line_line(sitepos, v, pathpos1, p1p2norm)
            distarr.append(abs(pq))

            # Rough estimate of measurement uncertainty
            sitedist = np.sqrt(np.dot(intersec_on_los - sitepos, intersec_on_los - sitepos))
            # Using pixel scale of 0.25 deg/pix:
            # Not valid for all cameras!
            possig = sitedist * np.tan(np.radians(0.25*cendat.censig[i]))
            if (cendat.censig[i] == 0.00): possig = 1e-30
            sigarr.append(possig)

        measerr = np.array(sigarr)

        # Now try fitting both with exponential and linear functions:
        exp_params, success, n_ok = guess_expfit(cendat.reltime, np.array(posarr), measerr)
        lin_params, pcov = curve_fit(linfunc, cendat.reltime, posarr, None, measerr)

        allposarr.append(posarr)
        alltimearr.append(timearr)
        allheightarr.append(heightarr)
        allsigarr.append(sigarr)
        alldistarr.append(distarr)

        if success:
            chi2_exp = np.sum(((measerr*(np.array(posarr) - expfunc(cendat.reltime, exp_params[0], exp_params[1], exp_params[2], exp_params[3])))**2)[0:n_ok-1])
            chi2_lin = np.sum(((measerr*(np.array(posarr) - linfunc(cendat.reltime, lin_params[0], lin_params[1])))**2)[0:n_ok-1])

            if chi2_exp < chi2_lin:
                linfit = [exp_params[2], exp_params[3]]
                if (debug and (doplot == 'show' or doplot == 'both')):
                  plt.plot(cendat.reltime, posarr, syms[nn % 4])
                  plt.plot(cendat.reltime, linfunc(cendat.reltime, exp_params[2], exp_params[3]), 'y-')
                  plt.plot(cendat.reltime, expfunc(cendat.reltime, exp_params[0], exp_params[1], exp_params[2], exp_params[3]), 'b-')
            else:
                linfit = [lin_params[0], lin_params[1]]
                if debug: print("Fallback to linear fit!")
                if (debug and (doplot == 'show' or doplot == 'both')):
                  plt.plot(cendat.reltime, posarr, syms[nn % 4])
                  plt.plot(cendat.reltime, linfunc(cendat.reltime, lin_params[0], lin_params[1]), 'b-')

        else:
            if debug: print("Fallback to linear fit!")
            linfit = [lin_params[0], lin_params[1]]

        if not reffit: reffit = linfit

        mergednames = mergednames + " " + cendat.sitestr[0]

    adjtimearr = minimize_chainlength(alltimearr, allposarr)
    if debug: print(adjtimearr)
    
    for i in range(len(allposarr)):
        for j in range(len(allposarr[i])):
            mergedposarr.append(allposarr[i][j])
            mergedheightarr.append(allheightarr[i][j])
            mergedreltime.append(alltimearr[i][j] + adjtimearr[i]) # With adjustment!
            mergedsigarr.append(allsigarr[i][j])
            mergeddistarr.append(alldistarr[i][j])

    if not mergedreltime:
        print("No valid centroid data to process.")
        return False, 0

    if (debug and (doplot == 'show' or doplot == 'both')):
       plt.figure(4)
       plt.subplot(211)
       plt.title('Individual tracks')
       plt.ylabel('Position [km]')
       syms=['g.','r.','b.','y.']
       for k in range(len(allcendat)):
          plt.plot(alltimearr[sorted_cendat_index[k]], allposarr[sorted_cendat_index[k]], syms[k % 4])
    
    if (debug and (doplot == 'show' or doplot == 'both')):
       plt.subplot(212)
       plt.title('Merged tracks')
       plt.ylabel('Position [km]')
       plt.xlabel('Time [s]')
       plt.plot(mergedreltime, mergedposarr, 'g.')

    mergedreltime = np.array(mergedreltime)
    mergedposarr=np.array(mergedposarr)
    mergedheightarr=np.array(mergedheightarr)
    mergedsigarr=np.array(mergedsigarr)
    mergeddistarr=np.array(mergeddistarr)

    sortidx = np.argsort(mergedreltime)
    mergedreltime = mergedreltime[sortidx]
    mergedposarr = mergedposarr[sortidx]
    mergedheightarr = mergedheightarr[sortidx]
    mergedsigarr = mergedsigarr[sortidx]
    mergeddistarr = mergeddistarr[sortidx]

    measerr = mergedsigarr

    exp_params, success, n_ok = guess_expfit(mergedreltime, mergedposarr, measerr)
    lin_params, pcov = curve_fit(linfunc, mergedreltime, mergedposarr, None, measerr)

    if success:
        chi2_exp = np.sum(((measerr*(mergedposarr - expfunc(mergedreltime, exp_params[0], exp_params[1], exp_params[2], exp_params[3])))**2)[0:n_ok-1])
        chi2_lin = np.sum(((measerr*(mergedposarr - linfunc(mergedreltime, lin_params[0], lin_params[1])))**2)[0:n_ok-1])

        if chi2_exp < chi2_lin:
            linfit = [exp_params[2], exp_params[3]]
        else:
            linfit = [lin_params[0], lin_params[1]]
            if debug: print("Fallback to linear fit!")
    else:
        if debug: print("Fallback to linear fit!")
        linfit = [lin_params[0], lin_params[1]]

    adjtime = linfit[1]/linfit[0] if linfit[0] != 0 else 0
    mergedreltime = mergedreltime + adjtime
    if debug: print(linfit, adjtime)
    
    exp_params, success, n_ok = guess_expfit(mergedreltime, mergedposarr, measerr)
    lin_params, pcov = curve_fit(linfunc, mergedreltime, mergedposarr, None, measerr)

    if success:
        chi2_exp = np.sum(((measerr*(mergedposarr - expfunc(mergedreltime, exp_params[0], exp_params[1], exp_params[2], exp_params[3])))**2)[0:n_ok-1])
        chi2_lin = np.sum(((measerr*(mergedposarr - linfunc(mergedreltime, lin_params[0], lin_params[1])))**2)[0:n_ok-1])

        if chi2_exp > chi2_lin:
            exp_params = [0, 0, lin_params[0], lin_params[1]]
            n_ok = len(mergedreltime)
            if debug: print("Fallback to linear fit!")
    else:
        if debug: print("Fallback to linear fit!")
        n_ok = len(mergedreltime)
        exp_params = [0, 0, lin_params[0], lin_params[1]]

    exp_fitval = expfunc(mergedreltime, exp_params[0], exp_params[1], exp_params[2], exp_params[3])
    exp_fitdev = mergedposarr - exp_fitval
    exp_speed  = expfunc_1stder(mergedreltime, exp_params[0], exp_params[1], exp_params[2], exp_params[3])
    exp_accel  = expfunc_2ndder(mergedreltime, exp_params[0], exp_params[1], exp_params[2], exp_params[3])

    if doplot != '':
        plt.figure(1, figsize=(10,7))
        plt.subplot(211)
        plt.title('Tilbakelagt strekning')
        plt.errorbar(mergedreltime, mergedposarr, yerr=mergedsigarr, fmt='r.')
        plt.plot(mergedreltime[n_ok:], mergedposarr[n_ok:], 'y.')
        plt.ylabel('Posisjon [km]')
        plt.xlabel('Tid [s]')
        if debug: plt.plot(mergedreltime, linfunc(mergedreltime, exp_params[2], exp_params[3]), 'y-')
        plt.plot(mergedreltime, exp_fitval, 'b-')

        plt.subplot(212)
        plt.plot(mergedreltime, exp_fitdev, 'b-')
        plt.errorbar(mergedreltime, exp_fitdev, yerr=mergedsigarr, fmt='b.')
        plt.plot(mergedreltime[n_ok:], exp_fitdev[n_ok:], 'y.')
        plt.ylabel('Avvik [km]')
        plt.xlabel('Tid [s]')
        plt.plot(mergedreltime, np.zeros(len(mergedreltime)), 'r--')
        plt.tight_layout()

        if doplot == 'save' or doplot == 'both':
            plt.savefig('posvstime.svg')

        if debug: 
            plt.figure(3)
            plt.errorbar(mergeddistarr, mergedposarr, xerr=mergedsigarr, fmt='b.')
            plt.plot(np.zeros(len(mergedposarr)), mergedposarr, 'r--')
            plt.ylabel('Position [km]')
            plt.xlabel('Path deviation [km]')

        plt.figure(2, figsize=(10,7))
        plt.subplot(211)
        plt.ylim(max([0., min(exp_speed)*0.95]), max(exp_speed)*1.05)
        plt.plot(mergedreltime, exp_speed, 'b-')
        plt.ylabel('Hastighet [km/s]')
        plt.xlabel('Tid [s]')

        plt.subplot(212)
        plt.ylim(min([-0.1, min(exp_accel)*0.95]), max([0.1, max(exp_accel)*1.05]))
        plt.plot(mergedreltime, exp_accel, 'b-')
        plt.ylabel(u'Aksellerasjon [km/s²]')
        plt.xlabel('Tid [s]')
        plt.tight_layout()

        if doplot == 'save' or doplot == 'both':
            print(f"Entry speed {exp_params[2]:.2f} km/s")
            plt.savefig('spd_acc.svg')

        if doplot == 'show' or doplot == 'both':
            plt.show()
            print(f"Entry speed {exp_params[2]:.2f} km/s")
            if (posdata):
              npoints = len(mergedreltime)
              for i in range(npoints):
                print(f"{mergedreltime[i]:f3} {mergedheightarr[i]:f3} {mergedposarr[i]:f3} {exp_speed[i]:f3}")

    return success, lin_params[0]



if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Fit speed and acceleration profiles to meteor centroid data.',
        epilog="Example: fbspd_merge.py -r meteor.res -d meteor.dat -c cen1.txt,cen2.txt -o show")
    parser.add_argument('-r', '--res', dest='resname', default='',
        help='Name of input file with .res extension.')
    parser.add_argument('-d', '--dat', dest='datname', default='',
        help='Name of input file with .dat extension.')
    parser.add_argument('-c', '--cen', dest='cennames', default=[], type=lambda s: s.split(','),
        help='Comma-separated list of input files with centroid data.')
    parser.add_argument('-p', '--posdata', dest='posdata', default=False, action='store_true',
        help='Provide additional positional data.')
    parser.add_argument('-o', '--output', dest='output', default='', choices=['','show','save','both'],
        help='show: Display graphics, save: save graphics to SVG, both: Display and save. Default is no output.')
    parser.add_argument('-v', '--verbose', dest='debug', default=False, action="store_true",
        help='Provide a little more output and plots.')


    args = parser.parse_args()

    if not args.resname or not args.cennames:
        parser.print_help()
        sys.exit(1)

    fbspd(args.resname, args.cennames, args.datname, doplot=args.output, posdata=args.posdata, debug=args.debug)
