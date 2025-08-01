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

# Import Plotly, with a friendly error message if it's not installed.
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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
    Generates and displays or saves a 3D plot of the meteoroid's orbit using Matplotlib.
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

    all_planets = {
        "MERCURY": {"name": "Merkur", "color": "#B7A9A3"},
        "Venus": {"name": "Venus", "color": "#C77C00"},
        "Earth": {"name": "Jorda", "color": "#0077BE"},
        "MARS BARYCENTER": {"name": "Mars", "color": "#D73A00"},
        "Jupiter barycenter": {"name": "Jupiter", "color": "#A0522D"},
        "Saturn barycenter": {"name": "Saturn", "color": "#C19A6B"},
        "Uranus barycenter": {"name": "Uranus", "color": "#94D2E2"},
        "Neptune barycenter": {"name": "Neptune", "color": "#3F54B5"}
    }

    planets_to_plot = {
        "Venus": all_planets["Venus"],
        "Earth": all_planets["Earth"],
        "MARS BARYCENTER": all_planets["MARS BARYCENTER"],
        "Jupiter barycenter": all_planets["Jupiter barycenter"]
    }

    q_km, e = meteor_elements[0], meteor_elements[1]
    if e < 1:
        aphelion_au = (q_km / km_per_au) * (1 + e) / (1 - e)
        if aphelion_au < 1.67:
            planets_to_plot["MERCURY"] = all_planets["MERCURY"]
        if aphelion_au > 6.0:
            planets_to_plot["Saturn barycenter"] = all_planets["Saturn barycenter"]
        if aphelion_au > 10.12:
            planets_to_plot["Uranus barycenter"] = all_planets["Uranus barycenter"]
            if "Venus" in planets_to_plot:
                del planets_to_plot["Venus"]
        if aphelion_au > 20.1:
            planets_to_plot["Neptune barycenter"] = all_planets["Neptune barycenter"]
    elif e > 1:
        planets_to_plot["Saturn barycenter"] = all_planets["Saturn barycenter"]
        planets_to_plot["Uranus barycenter"] = all_planets["Uranus barycenter"]
            
    legend_handles = []
    for planet_id, params in planets_to_plot.items():
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

    if e < 1:
        aphelion_au = (q_km / km_per_au) * (1 + e) / (1 - e)
        max_radius = int(np.ceil(aphelion_au))
        if max_radius > 0:
            circle_interval = max(1, int(np.ceil(max_radius / 10.0)))
            theta_circle = np.linspace(0, 2 * np.pi, 200)
            for r in range(circle_interval, max_radius + 1, circle_interval):
                x_circle = r * np.cos(theta_circle)
                y_circle = r * np.sin(theta_circle)
                ax.plot(x_circle, y_circle, 0, color='blue', linestyle='--', linewidth=0.75, alpha=0.6)
            for i in range(12):
                angle = i * (2 * np.pi / 12)
                x_line = [0, max_radius * np.cos(angle)]
                y_line = [0, max_radius * np.sin(angle)]
                ax.plot(x_line, y_line, [0, 0], color='blue', linestyle='--', linewidth=0.75, alpha=0.6)
        tp_et = meteor_elements[6]
        a_km = q_km / (1.0 - e)
        period_sec = 2 * np.pi * np.sqrt(a_km**3 / gm_sun)
        et_aphelion = tp_et + period_sec / 2.0
        aphelion_state_km = conics(meteor_elements, et_aphelion)
        aphelion_pos_au = aphelion_state_km[:3] / km_per_au
        ax.plot([aphelion_pos_au[0]], [aphelion_pos_au[1]], [0], 'x', color='darkviolet', markersize=7, zorder=10)
        ax.plot([aphelion_pos_au[0], aphelion_pos_au[0]], [aphelion_pos_au[1], aphelion_pos_au[1]], [aphelion_pos_au[2], 0], color='darkviolet', linestyle='--', linewidth=1)
        ax.plot([aphelion_pos_au[0]], [aphelion_pos_au[1]], [aphelion_pos_au[2]], 'D', color='darkviolet', markersize=6, label='Aphelium', zorder=10)
        legend_handles.append(Line2D([0], [0], marker='D', color='w', label='Aphelium', markerfacecolor='darkviolet', markersize=7))

    step = max(1, len(x) // 50)
    ax.plot(x[::step], y[::step], 0, '.', color='gray', markersize=1, alpha=0.5)
    for i in range(0, len(x), step):
        ax.plot([x[i], x[i]], [y[i], y[i]], [z[i], 0], '-', color='gray', linewidth=0.5, alpha=0.4)

    legend_handles.extend([
        Line2D([0], [0], color='green', lw=1.5, label='Meteoroidebane (over ekliptikken)'),
        Line2D([0], [0], color='red', lw=1.5, label='Meteoroidebane (under ekliptikken)')
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
    ax.legend(handles=legend_handles, loc='lower left')

    if doplot == 'save' or doplot == 'both':
        plt.savefig('orbit.svg', bbox_inches='tight', pad_inches=0.2)
        print("Plot saved to orbit.svg")
    if doplot == 'show' or doplot == 'both':
        plt.show()


def _create_ecliptic_grid_traces(max_radius):
    """Generates traces for the ecliptic grid lines."""
    traces = []
    theta = np.linspace(0, 2 * np.pi, 200)
    circle_interval = max(1, int(np.ceil(max_radius / 10.0)))
    
    # Concentric circles for AU distances
    for r in range(circle_interval, max_radius + 1, circle_interval):
        traces.append(go.Scatter3d(
            x=r * np.cos(theta), y=r * np.sin(theta), z=np.zeros_like(theta),
            mode='lines', line=dict(color='blue', width=0.5, dash='dash'),
            hoverinfo='none', showlegend=False
        ))
        
    # Radial lines
    for i in range(12):
        angle = i * (2 * np.pi / 12)
        traces.append(go.Scatter3d(
            x=[0, max_radius * np.cos(angle)], y=[0, max_radius * np.sin(angle)], z=[0, 0],
            mode='lines', line=dict(color='blue', width=1, dash='dash'),
            hoverinfo='none', showlegend=False
        ))
    return traces


def _plot_orbit_interactive(et, meteor_elements):
    """
    Generates and saves an interactive 3D plot of the meteoroid's orbit
    with an autorotation feature.
    """
    if not PLOTLY_AVAILABLE:
        print("\nError: Plotly is not installed. Please run: pip install plotly")
        return

    # --- Initial Setup ---
    gm_sun = bodvrd("SUN", "GM", 1)[1][0]
    km_per_au = convrt(1.0, "AU", "KM")
    q_km, e = meteor_elements[0], meteor_elements[1]
    is_elliptic = e < 1.0

    traces = []

    # --- 1. Meteor Orbit Calculation ---
    if is_elliptic:
        a = q_km / (1.0 - e)
        period = 2 * np.pi * np.sqrt(a**3 / gm_sun)
        et_start, et_end = et, et + period
    else: # Hyperbolic or Parabolic
        time_span = 5e8
        et_start, et_end = et - time_span, et + time_span

    times = np.linspace(et_start, et_end, 2001)
    orbit_km = np.array([conics(meteor_elements, t)[:3] for t in times])
    orbit_au = orbit_km / km_per_au
    x, y, z = orbit_au.T

    # --- 2. Meteor Trace Generation ---
    common_hover_label = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.3)", font=dict(size=10, color="black"))
    common_hover_template = 'X: %{x:.2f} AU<br>Y: %{y:.2f} AU<br>Z: %{z:.2f} AU<extra></extra>'

    for color, condition in [('green', z >= 0), ('red', z < 0)]:
        indices = np.where(condition)[0]
        if not len(indices):
            continue
            
        x_segment, y_segment, z_segment = [], [], []
        for i in range(len(indices) - 1):
            current_idx, next_idx = indices[i], indices[i+1]
            x_segment.append(x[current_idx]); y_segment.append(y[current_idx]); z_segment.append(z[current_idx])
            if next_idx - current_idx > 1:
                x_segment.append(None); y_segment.append(None); z_segment.append(None)
        
        last_idx = indices[-1]
        x_segment.append(x[last_idx]); y_segment.append(y[last_idx]); z_segment.append(z[last_idx])

        traces.append(go.Scatter3d(
            x=x_segment, y=y_segment, z=z_segment, mode='lines',
            name='Meteoroide', line=dict(color=color, width=4),
            hoverlabel=common_hover_label, hovertemplate=common_hover_template
        ))

    # --- 3. Planetary Orbits and Positions ---
    all_planets = {
        "MERCURY": {"name": "Merkur", "color": "#B7A9A3"}, "VENUS": {"name": "Venus", "color": "#C77C00"},
        "EARTH": {"name": "Jorda", "color": "#0077BE"}, "MARS BARYCENTER": {"name": "Mars", "color": "#D73A00"},
        "JUPITER BARYCENTER": {"name": "Jupiter", "color": "#A0522D"}, "SATURN BARYCENTER": {"name": "Saturn", "color": "#C19A6B"},
        "URANUS BARYCENTER": {"name": "Uranus", "color": "#94D2E2"}, "NEPTUNE BARYCENTER": {"name": "Neptune", "color": "#3F54B5"}
    }
    
    planets_to_plot = {"VENUS", "EARTH", "MARS BARYCENTER", "JUPITER BARYCENTER"}
    if is_elliptic:
        aphelion_au = (q_km / km_per_au) * (1 + e) / (1 - e)
        if aphelion_au < 1.67: planets_to_plot.add("MERCURY")
        if aphelion_au > 6.0:  planets_to_plot.add("SATURN BARYCENTER")
        if aphelion_au > 10.12:
            planets_to_plot.add("URANUS BARYCENTER")
            planets_to_plot.discard("VENUS")
        if aphelion_au > 20.1: planets_to_plot.add("NEPTUNE BARYCENTER")
    else:  # For hyperbolic/parabolic orbits
        planets_to_plot.add("SATURN BARYCENTER")
        planets_to_plot.add("URANUS BARYCENTER")

    for planet_id in planets_to_plot:
        params = all_planets[planet_id]
        state, _ = spkezr(planet_id, et, REF_FRAME_ECLIPTIC, ABERRATION_CORRECTION, SOLAR_SYSTEM_BARYCENTER)
        pos_au = state[:3] / km_per_au
        p_elts = oscelt(state, et, gm_sun)
        
        traces.append(go.Scatter3d(
            x=[pos_au[0]], y=[pos_au[1]], z=[pos_au[2]], mode='markers+text',
            marker=dict(color=params["color"], size=5, line=dict(color='black', width=1)),
            text=params["name"], textposition="top center", name=params["name"],
            hoverinfo='text', textfont=dict(size=7), showlegend=False
        ))
        
        a_p = p_elts[0] / (1 - p_elts[1]) if p_elts[1] < 1 else float('inf')
        if a_p > 0 and np.isfinite(a_p):
            t_p = 2 * np.pi * np.sqrt(a_p**3 / gm_sun)
            orbit_path_km = np.array([conics(p_elts, et + i * (t_p / 500))[:3] for i in range(501)])
            traces.append(go.Scatter3d(
                x=orbit_path_km[:, 0] / km_per_au, y=orbit_path_km[:, 1] / km_per_au, z=orbit_path_km[:, 2] / km_per_au,
                mode='lines', line=dict(color=params["color"], width=3), name=params["name"], hoverinfo='none'
            ))

    # --- 4. Sun, Ecliptic Grid, and Aphelion Marker ---
    traces.append(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', name='Sola',
                               marker=dict(color='yellow', size=8, line=dict(color='black', width=1)),
                               hoverinfo='name', showlegend=False))
    
    if is_elliptic:
        max_radius = int(np.ceil(aphelion_au))
        if max_radius > 0:
            traces.extend(_create_ecliptic_grid_traces(max_radius))
        
        et_aphelion = meteor_elements[6] + period / 2.0
        aphelion_pos_au = conics(meteor_elements, et_aphelion)[:3] / km_per_au
        
        traces.append(go.Scatter3d(
            x=[aphelion_pos_au[0]], y=[aphelion_pos_au[1]], z=[aphelion_pos_au[2]], mode='markers', name='Aphelium', 
            marker=dict(symbol='diamond', color='darkviolet', size=2), hoverlabel=common_hover_label, hovertemplate=common_hover_template
        ))
        traces.append(go.Scatter3d(
            x=[aphelion_pos_au[0]], y=[aphelion_pos_au[1]], z=[0], mode='markers', name='Aphelium (projisert)', 
            marker=dict(symbol='cross', color='darkviolet', size=2), showlegend=False
        ))
        traces.append(go.Scatter3d(
            x=[aphelion_pos_au[0], aphelion_pos_au[0]], y=[aphelion_pos_au[1], aphelion_pos_au[1]], z=[aphelion_pos_au[2], 0], 
            mode='lines', name='Aphelium droplinje', line=dict(color='darkviolet', width=2, dash='dash'), hoverinfo='none', showlegend=False
        ))

    # --- 5. Orbit Projection and Droplines on Ecliptic ---
    step = max(1, len(x) // 50)
    traces.append(go.Scatter3d(x=x[::step], y=y[::step], z=np.zeros_like(x[::step]), mode='markers',
                               marker=dict(color='dimgray', size=1.5), name='Baneprojeksjon', showlegend=False, hoverinfo='none'))
    
    for i in range(0, len(x), step):
        traces.append(go.Scatter3d(x=[x[i], x[i]], y=[y[i], y[i]], z=[z[i], 0], mode='lines',
                                   line=dict(color='dimgray', width=1.5, dash='dot'), showlegend=False, hoverinfo='none'))
                                   
    # --- 6. Animation Frame Generation ---
    frames = []
    num_frames = 360
    initial_eye = dict(x=2.5, y=0.01, z=1.2)
    radius = np.sqrt(initial_eye['x']**2 + initial_eye['y']**2)
    
    for k in range(num_frames):
        theta = (k / num_frames) * 2 * np.pi
        eye_x = radius * np.cos(theta)
        eye_y = radius * np.sin(theta)
        frames.append(go.Frame(
            layout=dict(scene=dict(camera=dict(eye=dict(x=eye_x, y=eye_y, z=initial_eye['z']))))
        ))

    # --- 7. Figure Layout and Generation ---
    layout = go.Layout(
        title_text='Meteoroidens heliosentriske bane', title_x=0.5, title_y=1,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            x=1, y=1, xanchor='right', yanchor='top', font=dict(size=8),
            bgcolor='rgba(224,222,224,0.5)', bordercolor='gray', borderwidth=1,
            itemsizing='constant'
        ),
        scene=dict(
            xaxis_title='X (AU)', yaxis_title='Y (AU)', zaxis_title='Z (AU)',
            aspectmode='data', dragmode='turntable',
            camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=initial_eye),
            xaxis=dict(showspikes=True, spikecolor="rgba(128,128,128,0.9)", spikethickness=1),
            yaxis=dict(showspikes=True, spikecolor="rgba(128,128,128,0.9)", spikethickness=1),
            zaxis=dict(showspikes=True, spikecolor="rgba(128,128,128,0.9)", spikethickness=1)
        ),
        updatemenus=[dict(
            type='buttons',
            active=0,
            showactive=True,
            y=0.95, x=0.05,
            xanchor='left', yanchor='top',
            pad=dict(t=0, r=10),
            font=dict(size=10), 
            bgcolor='rgba(255, 255, 255, 0.5)',
            buttons=[
                dict(label='▶ Play',
                     method='animate',
                     args=[None, dict(frame=dict(duration=20, redraw=True), transition=dict(duration=0), fromcurrent=True)]),
                dict(label='⏸ Pause',
                     method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate',
                                        transition=dict(duration=0))])
            ]
        )]
    )
    
    fig = go.Figure(data=traces, layout=layout, frames=frames)
    fig.write_html("orbit.html", include_plotlyjs='cdn')
    print("\nInteractive plot with autorotation saved to orbit.html")


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

def orbit(spd_success, v_obs, avg_speed, resname, datestr, timestr, doplot='', interactive=False):
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
    
    if interactive:
        _plot_orbit_interactive(et, elements)
    if doplot:
        _plot_orbit(et, elements, doplot)
        
    return np.degrees(ra_rad), np.degrees(dec_rad), orbelts, showername, showername_sg, True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and plot meteor orbit. Can generate a static plot and/or an interactive HTML file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('res_file', help="Result file from metrack")
    parser.add_argument('date', help="Date string YYYY-MM-DD")
    parser.add_argument('time', help="Time string HH:MM:SS")
    parser.add_argument('--speed', type=float, default=20.0, help="Observed speed in km/s")
    
    parser.add_argument('--plot', choices=['show', 'save', 'both'], default='', 
                            help="Generate static plot using Matplotlib.\n"
                                 " 'show': Display the plot.\n"
                                 " 'save': Save plot as orbit.svg.\n"
                                 " 'both': Show and save the plot.")
    parser.add_argument('--interactive', action='store_true',
                            help="Generate an interactive HTML file named orbit.html using Plotly.\n"
                                 "This option requires the 'plotly' library to be installed.")
    
    args = parser.parse_args()

    ra, dec, orbelts, shower, _, valid = orbit(
        spd_success=True, 
        v_obs=args.speed, 
        avg_speed=0, 
        resname=args.res_file, 
        datestr=args.date, 
        timestr=args.time, 
        doplot=args.plot,
        interactive=args.interactive
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
