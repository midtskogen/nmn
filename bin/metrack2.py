#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageOps
import cairosvg
from scour import scour
try:
    from PIL import Image, ImageOps
    import cairosvg
    from scour import scour
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

earth_radius = 6371.0  # km


class MetrackInfo:
    '''
    Structure for information on a track fit
    '''

    def __init__(self):
        self.date = ''
        self.error = 1e9
        self.fit_quality = 0.
        self.start_height = 0.
        self.end_height = 0.
        self.ground_track = 0.
        self.course = 0.
        self.incidence = 0.
        self.speed = 0.
        self.radiant_ra = 0.
        self.radiant_dec = 0.
        self.shower = ''


def clean_svg(svg_path):
    """
    Optimizes an SVG file in place using the Scour library.
    """
    if not LIBS_AVAILABLE:
        print("Warning: Scour library not installed. Cannot clean SVG.")
        print("Please run: pip install scour")
        return

    try:
        # Set options for Scour (these are common defaults)
        options = scour.parse_args([])
        options.remove_metadata = True
        options.strip_comments = True
        options.enable_viewboxing = True
        options.indent_type = 'none'  # To make the file compact

        # Read the original SVG file
        with open(svg_path, 'r', encoding='utf-8') as f:
            in_svg = f.read()
        
        # Clean the SVG data in memory
        cleaned_svg = scour.scourString(in_svg, options)
        
        # Overwrite the original file with the cleaned version
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_svg)
            
    except Exception as e:
        print(f"An error occurred while cleaning SVG {svg_path}: {e}")

def altaz_to_radec(longitude, latitude, alt, az, timestamp):
    import ephem
    import datetime
    from datetime import timezone

    obs = ephem.Observer()
    obs.long = str(longitude)
    obs.lat = str(latitude)
    # Disregard from atmospheric refration. Meteor track is calculated
    # before entry into atmosphere
    obs.pressure = 0

    obs.epoch = ephem.J2000
    # Updated to use the recommended timezone-aware method
    obs.date = ephem.Date(datetime.datetime.fromtimestamp(timestamp, timezone.utc))

    return obs.radec_of(str(az), str(alt))


def radec_eqlonlat(ra, dec):
    import ephem

    # Epoch J2000 implicitly assumed
    equ_coordinates = ephem.Equatorial(ra, dec)
    ecl_coordinates = ephem.Ecliptic(equ_coordinates)

    return ecl_coordinates.lon, ecl_coordinates.lat


def lonlat2xyz(lon, lat, height=0):
    """
    Convert geographic lon/lat/height on reference ellipse to cartesian coordiantes
    """

    # http://www.navipedia.net/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion
    #
    f = 1.0 / 298.257223563  # Earth flattening factor WGS-84
    e = np.sqrt(2 * f - f * f)
    N = earth_radius / np.sqrt(1 - e * e * np.sin(np.radians(lat)) * np.sin(np.radians(lat)))

    v = np.zeros(3)
    v[0] = (N + height) * np.cos(np.radians(lon)) * np.cos(np.radians(lat))
    v[1] = (N + height) * np.sin(np.radians(lon)) * np.cos(np.radians(lat))
    v[2] = (N * (1 - e * e) + height) * np.sin(np.radians(lat))

    return v


def lonlat2xyz_globe(lon, lat, height=0):
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
    f = 1.0 / 298.257223563  # Earth flattening factor WGS-84
    e = np.sqrt(2 * f - f * f)

    lon = np.degrees(np.arctan2(v[1], v[0]))

    p = np.sqrt(v[0] * v[0] + v[1] * v[1])
    lat_old = np.degrees(np.arctan2(v[2], (1.0 - e * e) * p))

    diff = 1
    nsteps = 0

    while (diff > 1e-10):
        nsteps = nsteps + 1
        N = earth_radius / np.sqrt(1.0 - e * e * np.sin(np.radians(lat_old)) * np.sin(np.radians(lat_old)))
        height = p / np.cos(np.radians(lat_old)) - N
        lat = np.degrees(np.arctan2(v[2], p * (1.0 - e * e * N / (N + height))))

        diff = abs((lat - lat_old))
        # print(diff, N, lon, lat, height)

        lat_old = lat

        if (nsteps > 1000):
            print("Warning - did not reach full convergence in xyz2lonlat: current diff is %f\n" % diff)
            break

    return lon, lat, height


def xyz2lonlat_globe(v):
    """
    Convert cartesian coordinates to geographical lon/lat/height
    """

    lon = np.degrees(np.arctan2(v[1], v[0]))
    lat = np.degrees(np.arctan2(v[2], np.sqrt(v[0] ** 2 + v[1] ** 2)))
    height = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) - earth_radius

    return lon, lat, height


def azdist2lonlat(lon1, lat1, azimuth, distance):
    """
    Calculate the longitude and latitude of a point that is at a
    given a great circle distance and azimuth from position (lon1, lat1)
    """

    distance = distance / earth_radius

    lat2 = 90 - np.degrees(np.arccos(np.cos(distance) * np.cos(np.radians(90 - lat1))
                                      + np.sin(distance) * np.sin(np.radians(90 - lat1)) * np.cos(np.radians(azimuth))
                                      ))
    lon2 = lon1 + np.degrees(np.arcsin(np.sin(distance) * np.sin(np.radians(azimuth))
                                      / np.sin(np.radians(90 - lat2))
                                      ))
    return lon2, lat2


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    km = earth_radius * c
    return km


def rotation_matrix(axis, theta):
    # from http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])


def altaz2xyz(alt, az, lon, lat):
    """
    Convert alt/az to vector in cartesian space
    """

    v = np.array([-1, 0, 0])  # alt=0, az=0
    # rotate alt
    axis = np.array([0, 1, 0])
    theta = -np.radians(alt)  # alt
    v = np.dot(rotation_matrix(axis, theta), v)

    # rotate az
    axis = np.array([0, 0, 1])
    theta = np.radians(az)  # az
    v = np.dot(rotation_matrix(axis, theta), v)

    # rotate geographic latitude
    axis = np.array([0, 1, 0])
    theta = np.radians(lat - 90)  # latitude
    v = np.dot(rotation_matrix(axis, theta), v)

    # rotate geographic longitude
    axis = np.array([0, 0, 1])
    theta = -np.radians(lon)  # longitude
    v = np.dot(rotation_matrix(axis, theta), v)

    return v


def closest_point(p1, p2, u1, u2, return_points=False):
    """
    Find the point that is closest to two lines, where the lines are
    defined by positions p1 and p2, with directional vectors u1 and u2
    """

    p21 = p2 - p1
    m = np.cross(u2, u1)
    m2 = float(np.dot(m, m))

    R = np.cross(p21, m / m2)
    t1 = np.dot(R, u2)
    t2 = np.dot(R, u1)

    cross_1 = p1 + t1 * u1
    cross_2 = p2 + t2 * u2

    if return_points: return cross_1, cross_2

    return (cross_1 + cross_2) / 2.0


def dist_line_line(p1, u1, p2, u2):
    """
    Calculate the distance between two vectors. These are defined by
    positions p1 and p2, with directional vectors u1 and u2
    """

    # Setup an array of two equations based on the fact that
    # the shortest line connecting both vectors must be perpendicular
    # to both.

    a = np.array([[np.dot(u1, u1), -np.dot(u1, u2)],
                   [np.dot(u2, u1), -np.dot(u2, u2)]])
    b = np.array([np.dot(u1, (p2 - p1)),
                    np.dot(u2, (p2 - p1))])

    # print(p1, u1)
    # print(p2, u2)
    # print(a, b)

    # Solve the set of two equations
    [s, t] = np.linalg.solve(a, b)

    # print(s, t)

    # Then, the vector connecting both lines is
    pq = (p1 + s * u1) - (p2 + t * u2)

    # The length of this vector is
    dist = np.sqrt(np.dot(pq, pq))

    return dist


def intersec_line_plane(line_ref, line_vec, plane_ref, plane_vec):
    """
    Determine the point where a line intersects a plane.
    Line is defined by reference point 'line_ref' and directional vector 'line_vec'
    Plane is defined by reference point 'plane_ref' and normal vector 'plane_vec'
    """

    factor = -np.dot(plane_vec, line_ref - plane_ref) / np.dot(plane_vec, line_vec)
    vec_to_intersection = line_vec * factor
    intersec_point = line_ref + vec_to_intersection
    # print(intersec_vec)
    # print("The following must be zero: %5.3f" % np.dot(intersec_point - plane_ref, plane_norm))

    return intersec_point


def chisq_of_fit(track_longvec, los_refs, los_vecs, weights):
    """
    Return the chi-square value of a given track and observed lines of sight.
    The track is defined as a single vector with six elements. The first three
    are the reference point, and the last three the directional vector.
    """

    chisq = 0

    n_los = len(los_refs)
    n_obs = len(weights)

    track_ref = track_longvec[:3]
    track_vec = track_longvec[3:]

    for i in range(n_los):
        dist = dist_line_line(track_ref, track_vec, los_refs[i], los_vecs[i])
        chisq = chisq + (weights[i % n_obs] * dist) ** 2

    return chisq


def plot_height(track_start, track_end, cross_pos, indata, lenarr, colarr, borders=None, doplot=None,
                autoborders=True, azonly=False, mapres='i'):
    """
    Show the vertical path of a track and the corresponding observations
    """

    import os
    import pylab
    from PIL import Image

    n_los = len(cross_pos)
    n_obs = n_los // 2
    n_steps = 100

    site_names = indata['namarr']
    plot_title = indata['name']

    start_lon, start_lat, start_height = xyz2lonlat(track_start)
    end_lon, end_lat, end_height = xyz2lonlat(track_end)

    los_lons = []
    los_lats = []
    los_heights = []
    los_dists = []
    for i in range(n_los):
        los_lon, los_lat, los_height = xyz2lonlat(cross_pos[i])
        los_lons.append(los_lon)
        los_lats.append(los_lat)
        los_heights.append(los_height)
        dist_along_ground = haversine(start_lon, start_lat, los_lon, los_lat)
        los_dists.append(dist_along_ground)

    xstart = 0
    xend = haversine(start_lon, start_lat, end_lon, end_lat)

    x_track = np.zeros(n_steps)
    y_track = np.zeros(n_steps)
    for i in np.arange(n_steps):
        track_pos = track_start + float(i) / (n_steps - 1) * (track_end - track_start)
        lon, lat, height = xyz2lonlat(track_pos)
        x_track[i] = haversine(start_lon, start_lat, lon, lat)
        y_track[i] = height

    xmin = min(xstart, min(los_dists))
    xmax = max(xend, max(los_dists))
    xrng = xmax - xmin
    xmin = xmin - 0.05 * xrng
    xmax = xmax + 0.05 * xrng

    ymin = min(min(y_track), min(los_heights))
    ymax = max(max(y_track), max(los_heights))
    yrng = ymax - ymin
    ymin = ymin - 0.05 * yrng
    ymax = ymax + 0.05 * yrng

    pylab.close()
    pylab.figure(1)
    pylab.figure(figsize=(10, 8))
    pylab.plot(los_dists, los_heights, 'ro')
    pylab.plot(x_track, y_track, 'g-')
    for i in range(n_los):
        pylab.text(los_dists[i], los_heights[i], site_names[i])
    pylab.xlim(min(xmin, xmax), max(xmin, xmax))
    pylab.ylim(min(ymin, ymax), max(ymin, ymax))
    # pylab.title(plot_title)
    pylab.xlabel('Tilbakelagt strekning over bakken [km]')
    pylab.ylabel('Høgde [km]')
    if doplot == 'save' or doplot == 'both':
        pylab.savefig('height.svg')
    # svgim = Image.open('height.svg')
    # jpgim = Image.new("RGB", svgim.size, (255, 255, 255))
    # jpgim.paste(svgim, (0, 0), svgim)
    # jpgim.save('height.jpg', format='JPEG')
    # os.remove('height.svg')
    if doplot == 'show' or doplot == 'both':
        pylab.ioff()
        pylab.draw()


def plot_map(track_start, track_end, cross_pos, indata, lenarr, colarr, borders=None, doplot=None,
             autoborders=True, azonly=False, mapres='i'):
    """
    Show a map of the track and the corresponding lines of sight using Cartopy
    """

    import os
    import pylab
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 1000000000

    # Import Cartopy modules
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io.img_tiles import OSM
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    try:
        from cartopy.io import shapereader
        has_shapereader = True
    except ImportError:
        has_shapereader = False

    n_obs = len(indata['azarr']) // 2

    pylab.figure(figsize=(8, 8))
    site_lon = indata['longarr']
    site_lat = indata['latarr']
    site_names = indata['namarr']
    plot_title = indata['name']

    if autoborders:
        if azonly:
            lon_left = min(site_lon) - 1.
            lon_right = max(site_lon) + 1.
            lat_bot = min(site_lat) - 0.5
            lat_top = max(site_lat) + 0.5
        else:
            n_los = len(cross_pos)
            los_lons = []
            los_lats = []
            for i in range(n_los):
                los_lon, los_lat, los_height = xyz2lonlat(cross_pos[i])
                los_lons.append(los_lon)
                los_lats.append(los_lat)
            lon_left = min([min(los_lons), min(site_lon)]) - 1.
            lon_right = max([max(los_lons), max(site_lon)]) + 1.
            lat_bot = min([min(los_lats), min(site_lat)]) - 0.5
            lat_top = max([max(los_lats), max(site_lat)]) + 0.5
    else:
        lon_left = borders[0]
        lon_right = borders[1]
        lat_bot = borders[2]
        lat_top = borders[3]

    # Define color scheme
    bordercol = '#000000'
    watercol = '#0060B0'
    gridcol = '#808080'
    landcol = '#009000'

    sublen = 0.9
    colgrad = 0.7

    # Set up the projection - use Gnomonic projection similar to original
    central_lon = np.mean([lon_left, lon_right])
    central_lat = np.mean([lat_bot, lat_top])

    proj = ccrs.Gnomonic(central_longitude=central_lon, central_latitude=central_lat)
    ax = pylab.axes(projection=proj)

    # Set map extent
    ax.set_extent([lon_left, lon_right, lat_bot, lat_top], crs=ccrs.PlateCarree())

    # Add map features
    # Add high resolution coastlines and borders
    resolution_map = {'c': '110m', 'l': '50m', 'i': '10m', 'h': '10m', 'f': '10m'}
    resolution = resolution_map.get(mapres, '10m')

    # Add topography/bathymetry - equivalent to basemap's etopo()
    try:
        # Try to add shaded relief or topography
        lat_span = abs(lat_top - lat_bot)
        zoom_level = int(np.log2(360 / (lat_span + 1)))
        zoom_level = max(6, min(zoom_level, 9))
        ax.add_image(OSM(), 6)
    except:
        try:
            # Alternative: add land/ocean with topographic shading
            ax.add_feature(cfeature.LAND, color=landcol, alpha=0.8)
            ax.add_feature(cfeature.OCEAN, color=watercol, alpha=0.8)
            # Add topographic shading
            ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', resolution,
                                                       edgecolor='face',
                                                       facecolor=cfeature.COLORS['land']))
        except:
            # Final fallback
            ax.add_feature(cfeature.LAND, color=landcol, alpha=0.8)
            ax.add_feature(cfeature.OCEAN, color=watercol, alpha=0.8)

    # Add coastlines, borders, and rivers
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution), color=bordercol, linewidth=0.4)
    ax.add_feature(cfeature.BORDERS.with_scale(resolution), color=bordercol, linewidth=0.4)
    ax.add_feature(cfeature.RIVERS.with_scale(resolution), color=watercol, linewidth=0.3)

    # Add lakes
    ax.add_feature(cfeature.LAKES.with_scale(resolution), color=watercol, alpha=0.8)

    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color=gridcol, alpha=0.5, linestyle='-')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Add administrative boundaries if shapefiles are available
    start_lon, start_lat, start_height = xyz2lonlat(
        track_start) if track_start is not None else (None, None, None)
    end_lon, end_lat, end_height = xyz2lonlat(track_end) if track_end is not None else (None, None, None)

    # Try to add administrative boundaries (simplified version without specific shapefiles)
    if has_shapereader:
        try:
            # Add country boundaries with higher detail
            countries = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_0_countries',
                scale=resolution,
                facecolor='none'
            )
            ax.add_feature(countries, edgecolor='black', linewidth=0.7)

            # Add state/province boundaries
            states = cfeature.NaturalEarthFeature(
                category='cultural',
                name='admin_1_states_provinces_lines',
                scale=resolution,
                facecolor='none'
            )
            ax.add_feature(states, edgecolor='grey', linewidth=0.3)
        except:
            pass  # Fallback if detailed features are not available

    # Color scheme for plotting
    mycolors = True  # Assume we have background colors
    if mycolors:
        startcol = '#5499c7'
        endcol = '#1a5276'
        arrowcol = '#b03a2e'
        arrowcol = 'blue'
        textcol = 'black'
    else:
        startcol = '#59b2b2'
        endcol = '#80ffff'
        arrowcol = 'white'
        textcol = 'yellow'

    # Plot observation sites and lines of sight
    for i in range(n_obs):

        if not mycolors:
            endcol = '#' + hex(int(colarr[i % n_obs][0]))[2:] \
                     + hex(int(colarr[i % n_obs][1]))[2:] \
                     + hex(int(colarr[i % n_obs][2]))[2:]
            startcol = '#' + hex(int(colgrad * colarr[i % n_obs][0]))[2:] \
                       + hex(int(colgrad * colarr[i % n_obs][1]))[2:] \
                       + hex(int(colgrad * colarr[i % n_obs][2]))[2:]
            arrowcol = 'white'

        # Plot observation site
        ax.plot(site_lon[i], site_lat[i], 'r*', markersize=8, transform=ccrs.PlateCarree())
        ax.text(site_lon[i], site_lat[i], '  ' + site_names[i], color=textcol, size=10,
                transform=ccrs.PlateCarree(), horizontalalignment='left')

        if azonly:
            azpos = azdist2lonlat(site_lon[i], site_lat[i], indata['azarr'][i], lenarr[i])
            azpos2 = azdist2lonlat(site_lon[i + n_obs], site_lat[i + n_obs], indata['azarr'][i + n_obs], lenarr[i])

            # Plot azimuth lines
            ax.plot([site_lon[i], azpos[0]], [site_lat[i], azpos[1]],
                    color=startcol, transform=ccrs.PlateCarree())
            ax.plot([site_lon[i], azpos2[0]], [site_lat[i], azpos2[1]],
                    color=endcol, transform=ccrs.PlateCarree())
        else:
            if cross_pos is not None:
                los_lon, los_lat, los_height = xyz2lonlat(cross_pos[i])
                los_lon2, los_lat2, los_height2 = xyz2lonlat(cross_pos[i + n_obs])

                # Plot lines of sight
                ax.plot([site_lon[i], los_lon], [site_lat[i], los_lat],
                        color=startcol, transform=ccrs.PlateCarree())
                ax.plot([site_lon[i], los_lon2], [site_lat[i], los_lat2],
                        color=endcol, transform=ccrs.PlateCarree())

    if not azonly and track_start is not None and track_end is not None:
        # Plot arrow to indicate the track
        ax.annotate('', xy=(start_lon, start_lat), xytext=(end_lon, end_lat),
                    arrowprops=dict(arrowstyle='<-', linewidth=1.5, color=arrowcol),
                    transform=ccrs.PlateCarree())
    elif azonly:
        # Plot azimuth uncertainty arcs
        for i in range(n_obs):
            az1 = indata['azarr'][i]
            az2 = indata['azarr'][i + n_obs]
            if (az2 > az1 + 180):
                az2 = az2 - 360
            if (az2 < az1 - 180):
                az2 = az2 + 360
            if az2 < az1: az1, az2 = az2, az1

            # Create arc points
            angles = np.arange(int(az1), int(az2), 2)
            if len(angles) > 1:
                arc_lons = []
                arc_lats = []
                for angle in angles:
                    azpos = azdist2lonlat(site_lon[i], site_lat[i], angle, lenarr[i] * sublen)
                    arc_lons.append(azpos[0])
                    arc_lats.append(azpos[1])

                ax.plot(arc_lons, arc_lats, color='red', linewidth=2, transform=ccrs.PlateCarree())

    # ax.set_title(plot_title)

    if doplot == 'save' or doplot == 'both':
        pylab.savefig('map.svg', bbox_inches='tight', dpi=150)
        clean_svg('map.svg')
    if doplot == 'show' or doplot == 'both':
        pylab.show()

    pylab.close()


def fit_track(data, optimize=True):
    """
    Perform the actual fitting of a meteor track to the observations given in 'data'.
    """

    # First create vectors of each line of sight
    nobs = len(data['longarr'])
    posvecarr = []
    losvecarr = []
    normvec = []
    weights = []

    # Read in the data
    for i in range(nobs):
        lon = data['longarr'][i]
        lat = data['latarr'][i]
        height = data['obsheightarr'][i]
        alt = data['altarr'][i]
        az = data['azarr'][i]
        posvecarr.append(lonlat2xyz(lon, lat, height))
        losvecarr.append(altaz2xyz(alt, az, lon, lat))
        weights.append(data['weights'][i])

    # Find normal to each plane suspended by pairs of vectors
    for i in range(nobs // 2):
        normvec.append(np.cross(losvecarr[i], losvecarr[i + nobs // 2]))
        normvec[i] = normvec[i] / np.sqrt(np.sum(normvec[i] * normvec[i]))

    # Find direction of intersection between pairs of planes
    # Summing the results has the nice property that the weights are factored in automatically.
    # The weights calculated below include two components, one based on the weights given in the
    # list of observations (.dat file), and one derived from the vector product between the sightlines,
    # which automatically assignes higher weight to pairs of lines that are perpendicular,
    # and lower weight to lines that are almost parallel.
    dirvecs = []
    vecweights = []
    fit_quality = 0
    for i in range(nobs // 2):
        for j in range(i + 1, nobs // 2):
            thisvec = np.cross(normvec[j], normvec[i])
            dirvecs.append(thisvec / np.sqrt(np.sum(thisvec * thisvec)))
            fit_quality = fit_quality + np.sqrt(np.sum(thisvec * thisvec))
            vecweights.append(np.sqrt(np.sum(thisvec * thisvec) * data['weights'][i] * data['weights'][j]))
            # vecweights.append(data['weights'][i] * data['weights'][j]) # np.sqrt(np.sum(thisvec*thisvec)))
            # print(i, j, data['namarr'][i], data['namarr'][j], thisvec, vecweights[-1])

    # Find intersection of each station's first line of sight with planes suspended by other stations
    # (corresponding to start of path for each pair of stations)
    start_coords = []
    end_coords = []

    if (fit_quality < 0.02):
        print("Warning: very small angle between observed directions")
        print("Trying my best by turning off optimization")
        optimize = False

    if optimize:

        for i in range(nobs // 2):
            for j in range(i + 1, nobs // 2):
                plane_ref = posvecarr[j]
                plane_norm = normvec[j]
                los_ref = posvecarr[i]
                los_vec = losvecarr[i]

                start_coords.append(intersec_line_plane(los_ref, los_vec, plane_ref, plane_norm))

                los_ref = posvecarr[i + nobs // 2]
                los_vec = losvecarr[i + nobs // 2]

                end_coords.append(intersec_line_plane(los_ref, los_vec, plane_ref, plane_norm))

                plane_ref = posvecarr[i]
                plane_norm = normvec[i]
                los_ref = posvecarr[j]
                los_vec = losvecarr[j]

                start_coords.append(intersec_line_plane(los_ref, los_vec, plane_ref, plane_norm))

                los_ref = posvecarr[j + nobs // 2]
                los_vec = losvecarr[j + nobs // 2]

                end_coords.append(intersec_line_plane(los_ref, los_vec, plane_ref, plane_norm))

    else:

        for i in range(nobs // 2):
            for j in range(i + 1, nobs // 2):
                los_ref1 = posvecarr[i]
                los_vec1 = losvecarr[i]
                los_ref2 = posvecarr[j]
                los_vec2 = losvecarr[j]

                # Append twice to make result compatible with common-plane method
                start_coords.append(closest_point(los_ref1, los_ref2, los_vec1, los_vec2))
                start_coords.append(closest_point(los_ref1, los_ref2, los_vec1, los_vec2))

                los_ref1 = posvecarr[i + nobs // 2]
                los_vec1 = losvecarr[i + nobs // 2]
                los_ref2 = posvecarr[j + nobs // 2]
                los_vec2 = losvecarr[j + nobs // 2]

                # Append twice to make result compatible with common-plane method
                end_coords.append(closest_point(los_ref1, los_ref2, los_vec1, los_vec2))
                end_coords.append(closest_point(los_ref1, los_ref2, los_vec1, los_vec2))

    # Renormalize weights (not really necessary)
    weights = np.array(weights)
    weights = 2 * weights / np.sum(weights)
    vecweights = np.array(vecweights)
    if (np.sum(vecweights) == 0): raise ValueError("ERROR - cannot solve with only one observation")
    vecweights = vecweights / np.sum(vecweights)

    # for c in start_coords: print(xyz2lonlat(c))
    # for c in end_coords: print(xyz2lonlat(c))

    npairs = nobs // 2 * (nobs // 2 - 1) // 2

    # Now combine the results
    start_coord = np.zeros(3)
    end_coord = np.zeros(3)
    best_fit = np.zeros(3)
    # The number of pairs is the length of vecweights
    num_pairs_actual = len(vecweights)

    for i in range(num_pairs_actual):
        # Note the definition of the weights in vecweights, see above
        start_coord = start_coord + (start_coords[i * 2] * vecweights[i] + start_coords[i * 2 + 1] * vecweights[i])
        end_coord = end_coord + (end_coords[i * 2] * vecweights[i] + end_coords[i * 2 + 1] * vecweights[i])
    # Correct for the fact that the sum above goes twice over each pair
    start_coord = start_coord / 2
    end_coord = end_coord / 2

    # Make a first guess at the fit based on start- and end-coordinates.
    guess_fit = end_coord - start_coord

    if optimize:
        # Make sure the dirvecs point in the right direction (start -> end)
        # and combine the dirvecs to get common-plane solution
        for i in range(num_pairs_actual):
            if (np.dot(guess_fit, dirvecs[i]) < 0): dirvecs[i] = -dirvecs[i]
            best_fit = best_fit + vecweights[i] * dirvecs[i]
    else:
        # Do not use common-plane solution when explicitly not optimizing
        best_fit = guess_fit

    # Renormalize, otherwise projection on this vector will not work (well?)
    best_fit = best_fit / np.sqrt(np.sum(best_fit * best_fit))

    # Make sure that track vector points from start to end
    if (np.dot((end_coord - start_coord), best_fit) < 0):
        print("Reversing track to match start to end")
        best_fit = -best_fit

    start_lon, start_lat, start_height = xyz2lonlat(start_coord)
    end_lon, end_lat, end_height = xyz2lonlat(end_coord)

    # print(start_coord)
    # print(end_coord)
    # print((end_coord - start_coord) / np.sqrt(np.sum((end_coord - start_coord)*(end_coord - start_coord))))
    # print(best_fit)
    # print("Start position corresponds to... long %7.4f / lat %7.4f / height %5.2f"% (start_lon, start_lat, start_height))
    # print("End position corresponds to... long %7.4f / lat %7.4f / height %5.2f"% (end_lon, end_lat, end_height))

    testpar = np.zeros(6)
    testpar[:3] = start_coord.copy()
    testpar[3:] = best_fit.copy()

    track_chi2 = chisq_of_fit(testpar, posvecarr, losvecarr, weights)

    cross_pos = []

    if optimize:

        #########################
        # Optimize the solution #
        #########################

        # print("Old Chi2 of fit is %5.2f" % chisq_of_fit(testpar, posvecarr, losvecarr, weights))

        import scipy
        from scipy.optimize import fmin_powell

        # print(testpar)
        newpar, fopt, direc, niter, nfuncalls, warnflag = fmin_powell(chisq_of_fit, testpar,
                                                                     args=(posvecarr, losvecarr, weights),
                                                                     disp=False, full_output=True)
        if (warnflag != 0):
            print("WARNING: Chi2 minimization failed - reverting to old solution")
            newpar = testpar
        # print(newpar)

        track_chi2 = chisq_of_fit(newpar, posvecarr, losvecarr, weights)
        # print("New Chi2 of fit is %5.2f" % track_chi2)
        start_coord = newpar[:3]
        best_fit = newpar[3:]

        #########################

        # Make sure that track vector was not inadvertently inverted during optimization of the solution
        if (np.dot(testpar[3:], best_fit) < 0):
            print("Oops - track got inverted by optimization... fixing...")
            best_fit = -best_fit

            # Construct the vertical plane of flight (old - no longer needed)
    # track_plane = np.cross(best_fit, start_coord)

    # Find where the lines are closest to the recovered solution
    dist_along_track = []
    closest_on_los = []
    closest_on_track = []
    for i in range(nobs):
        closest_on_los, closest_on_track = closest_point(posvecarr[i], start_coord, losvecarr[i], best_fit,
                                                         return_points=True)
        cross_pos.append(closest_on_los)
        dist_along_track.append(np.sqrt(np.dot(closest_on_track - start_coord, closest_on_track - start_coord)
                                        / np.dot(best_fit, best_fit))
                                * np.sign(np.dot(closest_on_track - start_coord, best_fit)))

    adjusted_start = start_coord + min(dist_along_track) * best_fit
    adjusted_end = start_coord + max(dist_along_track) * best_fit

    start_lon, start_lat, start_height = xyz2lonlat(adjusted_start)
    end_lon, end_lat, end_height = xyz2lonlat(adjusted_end)

    # Check that start/end heights are consistent - swap if necessary? (forcing start always higher than end)
    # (This sometimes happens when the track is highly inclined, and the chi2 minimization above inverted the track)
    if end_height > start_height:
        print("WARNING: start height is lower than end height")
    # adjusted_start, adjusted_end = adjusted_end, adjusted_start
    # start_lon, start_lat, start_height = xyz2lonlat(adjusted_start)
    # end_lon, end_lat, end_height = xyz2lonlat(adjusted_end)

    # print("Start position corresponds to... long %7.4f / lat %7.4f / height %5.2f"% (start_lon, start_lat, start_height))
    # print("End position corresponds to... long %7.4f / lat %7.4f / height %5.2f"% (end_lon, end_lat, end_height))

    return adjusted_start, adjusted_end, cross_pos, track_chi2, fit_quality


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def metrack(inname, doplot='', accept_err=0, mapres='i', azonly=False, autoborders=False, timestamp=None,
            optimize=True, writestat=False):
    '''
    For plotting meteor observations.
    Calculates a  trajectory throught the atmosphere and plots a map from a
    list of observers.
    Optionally plots to a SVG figure.

    Input file format:

    Optional border indication: Line starting with "borders" followed
    by four floating-point dreree numbers:
    borders longitude_left longitude_right latitude_bottom latitude_top

    Observation line: A list of the following data:
    longitude latitude azimuth_start azimuth_end altitude_start altitude_end weight duration distance red green blue name time height
    Longitude, latitude and altitude are floating point degrees
    Weight: Fitting weight (unused here)
    Duration: floating point seconds (unused here)
    Length: Graphics pointer length in kilometers
    Red, green, blue: integers 0-255
    Name: All following non-digit strings are interpreted as name.
        May contain spaces.
    Time: Unix time stamp (seconds since Jan 01 1970. (UTC))
        Not used in this program.
    Height: Observer height over sea level in metres.
        Assumes zero height if missing.

    Example:
    # Comment line
    borders 8.0 14.0 53.5 56.5
    10.1575 57.2089 182.0 352.0 69.1 23.8 1.  5.7  100 128 255 255 A Sorensen
    12.563  55.698  316.6 327.7 20.3  8.8 1.  3.84 300 128 255 255 KBH
    12.563  55.698  316.6 327.7 20.3  8.8 1.  3.84 300 128 255 255 KBH 1429086497.1 50.0
    '''

    import os
    import time
    try:
        import showerassoc
    except ImportError:
        print("Warning: 'showerassoc' module not found. Shower association will be skipped.")
        showerassoc = None
    import configparser

    import matplotlib
    if doplot == 'save':
        matplotlib.use('Agg')
    from matplotlib import pylab
    pylab.get_backend()

    path, longname = os.path.split(inname)

    if path != '':
        path = path + '/'
    elem = os.path.splitext(longname)
    name = elem[0]
    if len(elem) < 2:
        ext = '.dat'
    else:
        ext = elem[1]

    # Default borders
    borders = [7.0, 16.0, 54.0, 58.0]

    # Pre-declare data lists
    longarr = []
    latarr = []
    az1arr = []
    az2arr = []
    alt1arr = []
    alt2arr = []
    weights = []
    duration = []
    lenarr = []
    colarr = []
    namarr = []
    secarr = []
    obsheightarr = []

    # Read in data
    f = open(path + name + ext, 'r')
    for line in f:
        words = line.split()
        if not words:
            continue
        if words[0] == 'borders':
            borders = [float(words[1]), float(words[2]), float(words[3]), float(words[4])]
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
            colarr.append((int(words[9]), int(words[10]), int(words[11])))
            nam = ''
            nam_nsegs = 0
            for n in words[12:]:
                if not is_number(n):
                    nam = nam + ' ' + n
                    nam_nsegs += 1
                else:
                    break
            nam = nam[1:]
            namarr.append(nam)
            try:
                secarr.append(float(words[12 + nam_nsegs]))
            except:
                # Will this have any use in the future?
                secarr.append(None)
            try:
                obsheightarr.append(float(words[13 + nam_nsegs]) / 1000.)
            except:
                obsheightarr.append(0.)

    f.close()
    ndata = len(longarr)

    longarr.extend(longarr)
    latarr.extend(latarr)
    azarr = az1arr + az2arr
    altarr = alt1arr + alt2arr
    weights.extend(weights)
    duration.extend(duration)
    namarr.extend(namarr)
    obsheightarr.extend(obsheightarr)

    # Calculate an average timestamp for this event
    if timestamp is None:
        valid_secarr = [s for s in secarr if s is not None and s != 0]
        if len(valid_secarr) > 0:
            timestamp = np.mean(valid_secarr)
        else:
            timestamp = time.time()
            print("WARNING: Reverting to current time as timestamp for this event -> RA and Dec may be wrong!")

    longarr = np.array(longarr)
    latarr = np.array(latarr)
    az1arr = np.array(az1arr)
    az2arr = np.array(az2arr)
    azarr = np.array(azarr)
    alt1arr = np.array(alt1arr)
    alt2arr = np.array(alt2arr)
    altarr = np.array(altarr)
    weights = np.array(weights)
    duration = np.array(duration)
    lenarr = np.array(lenarr)
    colarr = np.array(colarr)
    obsheightarr = np.array(obsheightarr)

    indata = {'longarr': longarr, \
              'latarr': latarr, \
              'obsheightarr': obsheightarr, \
              'azarr': azarr, \
              'altarr': altarr, \
              'weights': weights, \
              'duration': duration, \
              'namarr': namarr, \
              'name': name}

    ndata2 = ndata * 2

    if azonly:
        plot_map(None, None, None, indata, lenarr, colarr, borders, doplot='show', autoborders=autoborders,
                 azonly=azonly, mapres=mapres)
        return

    # Determine best fit to these observations using common plane solution as first guess
    track_start, track_end, cross_pos, track_err, fit_quality = fit_track(indata, optimize=optimize)

    start_lon, start_lat, start_height = xyz2lonlat(track_start)
    end_lon, end_lat, end_height = xyz2lonlat(track_end)

    # Create and derotate dirvec to starting point (first longitude, then latitude!)
    dirvec = track_end - track_start

    # rotate geographic longitude
    axis = np.array([0, 0, 1])
    theta = -np.radians(-start_lon)  # longitude
    dirvec = np.dot(rotation_matrix(axis, theta), dirvec)

    # rotate geographic latitude
    axis = np.array([0, 1, 0])
    theta = np.radians(90 - start_lat)  # latitude
    dirvec = np.dot(rotation_matrix(axis, theta), dirvec)

    # Now extract the local direction and incidence angle
    best_fit_az = np.degrees(np.arctan2(dirvec[1], -dirvec[0])) % 360
    best_fit_alpha = np.degrees(np.arctan2(dirvec[2], np.sqrt(dirvec[0] ** 2 + dirvec[1] ** 2)))

    # print("This direction corresponds to... Az = %5.2f / Alpha = %5.3f" % (best_fit_az, best_fit_alpha))

    # Prepare and fill return variable

    info = MetrackInfo()
    info.error = track_err
    info.fit_quality = fit_quality
    info.start_height = start_height
    info.end_height = end_height
    info.ground_track = haversine(start_lon, start_lat, end_lon, end_lat)
    fall = start_height - end_height
    info.incidence = -best_fit_alpha
    # Line below was valid in old version. Now incidence is already determined at start of track
    # info.incidence   += info.ground_track / (2 * 1.852 * 60) # Adjust incidence to start of track, not middle of track
    info.course = best_fit_az

    radiant_ra, radiant_dec = altaz_to_radec(start_lon, start_lat, info.incidence, (info.course + 180) % 360,
                                             timestamp)
    radiant_ecllong, radiant_ecllat = radec_eqlonlat(radiant_ra, radiant_dec)
    info.radiant_ra = np.degrees(float(radiant_ra))
    info.radiant_dec = np.degrees(float(radiant_dec))
    info.radiant_ecllong = np.degrees(float(radiant_ecllong))
    info.radiant_ecllat = np.degrees(float(radiant_ecllat))

    # print(time.ctime(timestamp), cross_long[endid], cross_lat[endid], info.incidence, (info.course + 180) % 360, radiant_ra, radiant_dec)

    all_airspeed = []
    for i in range(ndata):
        airtrack = np.sqrt(np.dot(cross_pos[i + ndata] - cross_pos[i], cross_pos[i + ndata] - cross_pos[i]))
        if duration[i] > 0:
            airspeed = airtrack / duration[i]
            if doplot == 'show':
                print('Speed from %s : %6.1f km/s' % (namarr[i], airspeed))
            all_airspeed.append(airspeed)
    if all_airspeed:
        info.speed = sum(all_airspeed) / len(all_airspeed)

    info.date = time.asctime(time.gmtime(timestamp))
    info.timestamp = timestamp
    if showerassoc:
        info.shower, _ = showerassoc.showerassoc(info.radiant_ra, info.radiant_dec, info.speed,
                                                 time.strftime("%Y-%m-%d", time.localtime(timestamp)))
    else:
        info.shower = 'N/A'

    if doplot == 'show' or doplot == 'both':
        print('[track]')
        print('startheight = %7.2f km' % info.start_height)
        print('endheight   = %7.2f km' % info.end_height)
        print('groundtrack = %7.2f km' % info.ground_track)
        print('course      = %7.2f deg' % info.course)
        print('incidence   = %7.2f deg' % info.incidence)
        print('[fit]')
        print('error       = %7.2f' % info.error)
        print('quality     = %7.2f' % info.fit_quality)
        print('[radiant]')
        print('ra         = %s' % radiant_ra)
        print('dec         = %s' % radiant_dec)
        print('ecl_long    = %s deg' % info.radiant_ecllong)
        print('ecl_lat     = %s deg' % info.radiant_ecllat)
        print('shower      = %s' % info.shower)

    # Write results to file
    with open(name + '.res', 'w') as f:
        f.write('%8.4f %10.6f %8.4f %8.4f %6.1f %s\n' % (start_lon, start_lat, start_lon, start_lat, start_height, 'Start'))
        f.write('%8.4f %10.6f %8.4f %8.4f %6.1f %s\n' % (end_lon, end_lat, end_lon, end_lat, end_height, 'End'))
        for i in range(ndata2):
            end_lon, end_lat, end_height = xyz2lonlat(cross_pos[i])
            f.write('%6.2f %6.2f %8.4f %8.4f %6.1f %s\n' % (
            indata['longarr'][i], indata['latarr'][i], end_lon, end_lat, end_height, indata['namarr'][i]))

    if writestat:
        with open(name + '.stat', 'w') as f:
            infofile = configparser.RawConfigParser()

            infofile.add_section('track')
            infofile.set('track', 'startheight', '%.1f km' % info.start_height)
            infofile.set('track', 'endheight', '%.1f km' % info.end_height)
            infofile.set('track', 'groundtrack', '%.1f km' % info.ground_track)
            infofile.set('track', 'course', '%.1f deg' % info.course)
            infofile.set('track', 'incidence', '%.1f deg' % info.incidence)
            infofile.set('track', 'speed', '%.1f km/s' % info.speed)
            infofile.set('track', 'speed_source', 'average')

            infofile.add_section('fit')
            infofile.set('fit', 'error', '%.1f' % info.error)
            infofile.set('fit', 'quality', '%.2f' % info.fit_quality)

            infofile.add_section('radiant')
            infofile.set('radiant', 'ra', '%.2f deg' % info.radiant_ra)
            infofile.set('radiant', 'dec', '%.2f deg' % info.radiant_dec)
            infofile.set('radiant', 'ecl_long', '%.2f deg' % info.radiant_ecllong)
            infofile.set('radiant', 'ecl_lat', '%.2f deg' % info.radiant_ecllat)
            infofile.set('radiant', 'shower', '%s' % info.shower)
            infofile.set('radiant', 'zenith_attractor', 'uncorrected')

            infofile.add_section('date')
            infofile.set('date', 'timestamp', '%f' % info.timestamp)
            infofile.set('date', 'date', '%s' % info.date)

            infofile.write(f)

    if doplot != '':
        plot_height(track_start, track_end, cross_pos, indata, lenarr, colarr, borders, doplot=doplot,
                    autoborders=autoborders, azonly=azonly, mapres=mapres)
        plot_map(track_start, track_end, cross_pos, indata, lenarr, colarr, borders, doplot=doplot,
                 autoborders=autoborders, azonly=azonly, mapres=mapres)

    return info


if __name__ == "__main__":
    import sys
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Atmospheric meteor trajectory fitting. This program reads a data file describing a set of meteor observations. A trajectory through the atmosphere is calculated and the result is printed to screen and saved to a .res text file. Optionally, maps of the height and ground path can be displayed and/or saved.',
        epilog='Example: ./metrack.py obs_20120403.dat -m i -e 10 -o both')
    parser.add_argument('--version', action='version', version='%(prog)s 2024-07-26 (converted from 2015-11-24)')
    parser.add_argument('input_file', nargs='?', default=None,
                        help='Name of input file with .dat extension.')
    parser.add_argument('-i', '--input', default=None,
                        help='Name of input file with .dat extension (overrides positional argument).')
    parser.add_argument('-o', '--output', default='', choices=['', 'show', 'save', 'both'],
                        help='show: Display graphics, save: save graphics to SVG, both: Display and save. Default is no output.')
    parser.add_argument('-m', '--map', dest='mapres', default='i', choices=['c', 'l', 'i', 'h', 'f'],
                        help='Map detail level. c : crude, l: low, i: intermediate (default), f: full')
    parser.add_argument('-e', '--err', dest='errlim', default=1.0, type=float,
                        help='Deprecated option - kept for compatibility')
    parser.add_argument('-a', '--azonly', action='store_true', default=False,
                        help='If set, an azimuth-map is plotted and no fitting is performed. Default=off')
    parser.add_argument('-b', '--borders-auto', dest='autoborders', action='store_true', default=False,
                        help='If set, map boundaries will be determined automatically. Default=off')
    parser.add_argument('-t', '--timestamp', dest='timestamp', default=None, type=float,
                        help="Force numerical timestamp of observation (seconds since Epoch)")
    parser.add_argument('--noopt', action='store_false', dest='optimize',
                        help="Turn off optimization and common-plane solution - use closest line-of-sight crossings instead")

    options = parser.parse_args()

    name = options.input if options.input is not None else options.input_file

    if name is None:
        print("No input file given!")
        sys.exit()

    if options.azonly and options.output == '':
        options.output = 'show'

    retval = metrack(name, options.output, options.errlim, options.mapres, options.azonly, options.autoborders,
                     timestamp=options.timestamp, optimize=options.optimize, writestat=True)
