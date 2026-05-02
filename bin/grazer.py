#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grazer: Earth Grazer Meteor Trajectory Solver (Robust Version)

Usage:
    python3 grazer.py <event_directory>
"""

import argparse
import glob
import math
import os
import sys
import datetime
import numpy as np
import scipy.optimize
from pathlib import Path

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Constants
R_EARTH = 6371.0  # km
GM = 3.986004418e5  # Earth gravitational parameter km^3/s^2
OMEGA_EARTH = 7.2921159e-5  # Earth rotation rate rad/s

# --- Coordinate Transformations ---

def geodetic_to_ecef(lat_deg, lon_deg, alt_km):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    a = 6378.137
    f = 1 / 298.257223563
    e2 = 2*f - f*f
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    x = (N + alt_km) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt_km) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e2) + alt_km) * np.sin(lat_rad)
    return np.array([x, y, z])

def ecef_to_geodetic(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    # Simple spherical approximation for logging/plotting
    lat = np.degrees(np.arcsin(z / r))
    lon = np.degrees(np.arctan2(y, x))
    alt = r - R_EARTH
    return lat, lon, alt

def az_alt_to_unit_vec_ecef(lat_deg, lon_deg, az_deg, alt_deg):
    az_rad = np.radians(az_deg)
    alt_rad = np.radians(alt_deg)
    u_east = np.sin(az_rad) * np.cos(alt_rad)
    u_north = np.cos(az_rad) * np.cos(alt_rad)
    u_up = np.sin(alt_rad)
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    x = -sin_lon * u_east - sin_lat * cos_lon * u_north + cos_lat * cos_lon * u_up
    y = cos_lon * u_east - sin_lat * sin_lon * u_north + cos_lat * sin_lon * u_up
    z = cos_lat * u_north + sin_lat * u_up
    return np.array([x, y, z])

def ecef_to_eci(pos_ecef, t_offset_seconds):
    """Rotate ECEF to ECI (forward in time)"""
    theta = OMEGA_EARTH * t_offset_seconds
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = pos_ecef
    x_new = x * c - y * s
    y_new = x * s + y * c
    return np.array([x_new, y_new, z])

# --- Logic ---

def intersect_ray_sphere(ray_origin, ray_dir, radius):
    """Find intersection of a ray P + t*D with a sphere of given radius."""
    a = 1.0
    b = 2 * np.dot(ray_origin, ray_dir)
    c = np.dot(ray_origin, ray_origin) - radius**2
    delta = b*b - 4*a*c
    if delta < 0: return None
    t1 = (-b - np.sqrt(delta)) / (2*a)
    t2 = (-b + np.sqrt(delta)) / (2*a)
    if t1 > 0: return ray_origin + t1 * ray_dir
    if t2 > 0: return ray_origin + t2 * ray_dir
    return None

class GrazerSolver:
    def __init__(self, observations):
        # Sort by time
        self.obs = sorted(observations, key=lambda x: x['t'])
        self.ref_time = self.obs[0]['t']
        for o in self.obs:
            o['t'] -= self.ref_time
        self.t_max_raw = self.obs[-1]['t']

    def spatial_loss(self, params):
        state0 = params
        pos0 = state0[:3]
        vel0 = state0[3:]
        
        # --- 1. Physical Constraints (Penalties) ---
        penalty = 0.0
        
        # Speed constraint (10 km/s to 74 km/s)
        speed = np.linalg.norm(vel0)
        if speed > 74.0: penalty += (speed - 74.0)**2 * 10000
        elif speed < 10.0: penalty += (10.0 - speed)**2 * 10000
            
        # Altitude constraint (Start must be < 200km and > 50km)
        r_mag = np.linalg.norm(pos0)
        alt = r_mag - R_EARTH
        if alt > 200.0: penalty += (alt - 200.0)**2 * 10000
        elif alt < 0.0: penalty += (0.0 - alt)**2 * 10000

        if penalty > 1e6: return penalty

        # --- 2. Integrate Orbit ---
        from scipy.integrate import odeint
        # Integrate over full raw duration to ensure we cover all points for the check
        t_span = np.linspace(0, self.t_max_raw, 30)
        
        def derivs(s, t):
            r = s[:3]; v = s[3:]
            r_mag = np.linalg.norm(r)
            a = -GM * r / (r_mag**3)
            return np.concatenate((v, a))
            
        path_states = odeint(derivs, state0, t_span)
        path_points = path_states[:, :3]
        
        # --- 3. Geometric Error ---
        total_dist_sq = 0.0
        step = max(1, len(self.obs) // 100)
        
        for i in range(0, len(self.obs), step):
            o = self.obs[i]
            ray_origin = o['pos_eci']
            ray_dir = o['dir_eci']
            
            # Vectorized distance to line segment check
            w = path_points - ray_origin
            proj = np.sum(w * ray_dir, axis=1)
            perp_dist_sq = np.sum((w - np.outer(proj, ray_dir))**2, axis=1)
            
            min_dist_sq = np.min(perp_dist_sq)
            total_dist_sq += min_dist_sq
            
        return total_dist_sq + penalty

    def fit_spatial(self):
        print("Solving for spatial trajectory (Gravity-curved orbit)...")
        
        # --- Robust Initialization ---
        # 1. Estimate Start Position (intersect first ray with 100km shell)
        start_ray = self.obs[0]
        p_start_guess = intersect_ray_sphere(start_ray['pos_eci'], start_ray['dir_eci'], R_EARTH + 100.0)
        if p_start_guess is None:
            p_start_guess = start_ray['pos_eci'] + start_ray['dir_eci'] * 100.0
            
        # 2. Estimate End Position (intersect last ray with 80km shell)
        end_ray = self.obs[-1]
        p_end_guess = intersect_ray_sphere(end_ray['pos_eci'], end_ray['dir_eci'], R_EARTH + 80.0)
        if p_end_guess is None:
             p_end_guess = end_ray['pos_eci'] + end_ray['dir_eci'] * 100.0

        # 3. Estimate Velocity
        path_vec = p_end_guess - p_start_guess
        dist = np.linalg.norm(path_vec)
        if dist < 1.0: path_vec = np.array([1.0, 0.0, 0.0])
            
        dir_guess = path_vec / np.linalg.norm(path_vec)
        speed_guess = 30.0 
        vel_guess = dir_guess * speed_guess
        
        initial_state = np.concatenate((p_start_guess, vel_guess))
        print(f"  Initial Guess Altitude: {np.linalg.norm(p_start_guess) - R_EARTH:.2f} km")
        print(f"  Initial Guess Velocity: {vel_guess} ({speed_guess} km/s)")

        # --- Optimization ---
        res = scipy.optimize.minimize(
            self.spatial_loss, 
            initial_state, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'xatol': 1e-3, 'disp': True}
        )
        return res.x

    def temporal_fit(self, state_vector):
        print("Solving for temporal profile (Deceleration)...")
        t_max = self.t_max_raw
        t_eval = np.linspace(0, t_max, 200)
        
        from scipy.integrate import odeint
        def derivs(s, t):
            r = s[:3]; v = s[3:]; r_mag = np.linalg.norm(r)
            a = -GM * r / (r_mag**3)
            return np.concatenate((v, a))
        
        path_states = odeint(derivs, state_vector, t_eval)
        path_pos = path_states[:, :3]
        
        # Arc Lengths
        dists = np.sqrt(np.sum(np.diff(path_pos, axis=0)**2, axis=1))
        arc_lengths = np.concatenate(([0], np.cumsum(dists)))
        
        # Project observations
        obs_s = []
        obs_t = []
        for o in self.obs:
            ray_origin = o['pos_eci']
            ray_dir = o['dir_eci']
            w = path_pos - ray_origin
            proj = np.sum(w * ray_dir, axis=1)
            dists_sq = np.sum((w - np.outer(proj, ray_dir))**2, axis=1)
            idx = np.argmin(dists_sq)
            obs_s.append(arc_lengths[idx])
            obs_t.append(o['t'])
            
        # Linear/Quadratic Fit
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        X = np.array(obs_t).reshape(-1, 1)
        y = np.array(obs_s)
        
        model_lin = RANSACRegressor(LinearRegression(), min_samples=0.5, residual_threshold=5.0)
        model_lin.fit(X, y)
        v_avg = model_lin.estimator_.coef_[0]
        print(f"  Average Speed (Linear Fit): {v_avg:.2f} km/s")

        X_poly = np.hstack((X, X**2))
        model_quad = RANSACRegressor(LinearRegression(), min_samples=0.5, residual_threshold=5.0)
        try:
            model_quad.fit(X_poly, y)
            est = model_quad.estimator_
            s0 = est.intercept_
            v0 = est.coef_[0]
            accel = 2 * est.coef_[1]
        except:
            print("  Quadratic fit failed, falling back to linear.")
            s0 = model_lin.estimator_.intercept_
            v0 = v_avg
            accel = 0.0

        # DETERMINE VALID END TIME (Based on Inliers)
        # We only want to integrate the trajectory as far as the *valid* observations go.
        try:
            inlier_mask = model_quad.inlier_mask_
        except:
            inlier_mask = model_lin.inlier_mask_
            
        valid_t = np.array(obs_t)[inlier_mask]
        t_max_inlier = np.max(valid_t)
        
        print(f"  Valid Trajectory Duration: {t_max_inlier:.2f}s (Raw: {self.t_max_raw:.2f}s)")

        return s0, v0, accel, obs_t, obs_s, model_quad, t_max_inlier

# --- Data Helpers ---

def load_station_map(event_dir):
    obs_files = glob.glob(str(Path(event_dir) / "obs_*.txt"))
    if not obs_files: return {}
    obs_file = sorted(obs_files)[-1]
    station_map = {}
    with open(obs_file, 'r') as f:
        for line in f:
            parts = line.split()
            try:
                name_idx = -1
                for i in range(10, len(parts)):
                    try: float(parts[i])
                    except ValueError: 
                        name_idx = i
                        break
                if name_idx != -1:
                    code = parts[name_idx]
                    lon = float(parts[0])
                    lat = float(parts[1])
                    station_map[code] = {'lat': lat, 'lon': lon, 'alt': 0.1} 
            except Exception: pass
    return station_map

def collect_centroids(event_dir, station_map):
    files = glob.glob(str(Path(event_dir) / "*" / "*" / "centroid.txt"))
    if not files: files = glob.glob(str(Path(event_dir) / "**" / "centroid.txt"), recursive=True)
    observations = []
    
    for cfile in files:
        path_parts = Path(cfile).parts
        station_name_dir = path_parts[-3] 
        with open(cfile, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 7: continue
                try:
                    alt = float(parts[2])
                    az = float(parts[3])
                    code = parts[5]
                    ts_str = " ".join(parts[6:]).replace(" UTC", "")
                    try: dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
                    except ValueError: dt = datetime.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                    t_unix = dt.replace(tzinfo=datetime.timezone.utc).timestamp()
                    
                    st = None
                    if code in station_map: st = station_map[code]
                    else:
                        for k, v in station_map.items():
                            if k.lower() in station_name_dir.lower():
                                st = v; break
                    if not st: continue
                    
                    pos_ecef = geodetic_to_ecef(st['lat'], st['lon'], st['alt'])
                    dir_ecef = az_alt_to_unit_vec_ecef(st['lat'], st['lon'], az, alt)
                    
                    observations.append({
                        'pos_ecef': pos_ecef, 'dir_ecef': dir_ecef, 't': t_unix, 'code': code
                    })
                except Exception: pass
                
    if not observations: return []
    t0 = min(o['t'] for o in observations)
    for o in observations:
        dt = o['t'] - t0
        o['pos_eci'] = ecef_to_eci(o['pos_ecef'], dt)
        o['dir_eci'] = ecef_to_eci(o['dir_ecef'], dt)
    print(f"Loaded {len(observations)} valid data points.")
    return observations

def main():
    parser = argparse.ArgumentParser(description="Earth Grazer Trajectory Solver")
    parser.add_argument("event_dir", help="Path to the event directory")
    args = parser.parse_args()
    
    print(f"--- Analyzing Event in {args.event_dir} ---")
    station_map = load_station_map(args.event_dir)
    if not station_map: sys.exit("Error: Could not load station coordinates.")
    obs = collect_centroids(args.event_dir, station_map)
    if not obs: sys.exit("Error: No centroid data found.")
        
    solver = GrazerSolver(obs)
    final_state = solver.fit_spatial()
    s0, v0, accel, obs_t, obs_s, time_model, t_end_inlier = solver.temporal_fit(final_state)
    
    print("\n--- Final Results ---")
    pos_final = final_state[:3]
    vel_final = final_state[3:]
    
    # Start Point
    lat, lon, alt = ecef_to_geodetic(pos_final[0], pos_final[1], pos_final[2])
    speed = np.linalg.norm(vel_final)
    print(f"Start Point (Lat/Lon/Alt): {lat:.4f}, {lon:.4f}, {alt:.2f} km")

    # End Point (Robust)
    # Integrate ONLY to t_end_inlier
    from scipy.integrate import odeint
    def derivs(s, t):
        r = s[:3]; v = s[3:]; r_mag = np.linalg.norm(r)
        a = -GM * r / (r_mag**3)
        return np.concatenate((v, a))
    
    state_end = odeint(derivs, final_state, [0, t_end_inlier])[-1]
    pos_end_eci = state_end[:3]
    
    # Rotate End ECI back to ECEF
    theta = OMEGA_EARTH * t_end_inlier
    c, s = np.cos(theta), np.sin(theta)
    x_i, y_i, z_i = pos_end_eci
    x_ecef = x_i * c + y_i * s
    y_ecef = -x_i * s + y_i * c
    z_ecef = z_i
    
    lat_e, lon_e, alt_e = ecef_to_geodetic(x_ecef, y_ecef, z_ecef)
    print(f"End Point (Lat/Lon/Alt):   {lat_e:.4f}, {lon_e:.4f}, {alt_e:.2f} km")
    
    # Perigee Calculation
    r_vec = pos_final
    v_vec = vel_final
    h_vec = np.cross(r_vec, v_vec)
    mu = GM
    e_vec = (np.cross(v_vec, h_vec) / mu) - (r_vec / np.linalg.norm(r_vec))
    e = np.linalg.norm(e_vec)
    h_mag = np.linalg.norm(h_vec)
    rp = (h_mag**2) / (mu * (1 + e))
    perigee_alt = rp - 6371.0
    print(f"Calculated Perigee Alt:    {perigee_alt:.2f} km")
    
    print(f"Initial Velocity Vector:   {vel_final}")
    print(f"Estimated Entry Speed:     {speed:.2f} km/s")
    print(f"Deceleration:              {accel:.4f} km/s^2")
    
    # Plotting
    if MATPLOTLIB_AVAILABLE:
        fig = plt.figure(figsize=(12, 6))
        
        # 3D Plot
        ax = fig.add_subplot(121, projection='3d')
        # Only plot the valid duration
        t_span = np.linspace(0, t_end_inlier, 100)
        path = odeint(derivs, final_state, t_span)
        ax.plot(path[:,0], path[:,1], path[:,2], 'b-', label='Fit Orbit', linewidth=2)
        
        # Plot rays
        for i, o in enumerate(obs):
            if i % 20 == 0: 
                p = o['pos_eci']; d = o['dir_eci']
                line = np.array([p, p + d*200])
                ax.plot(line[:,0], line[:,1], line[:,2], 'r-', alpha=0.3)
        ax.set_title('3D Trajectory')
        
        # Time Plot
        ax2 = fig.add_subplot(122)
        ax2.scatter(obs_t, obs_s, c='k', s=2, label='Obs')
        
        # Visualize the valid range
        ax2.axvline(x=t_end_inlier, color='g', linestyle='--', label='End Valid')
        
        X_plot = np.linspace(min(obs_t), max(obs_t), 100).reshape(-1, 1)
        X_poly = np.hstack((X_plot, X_plot**2))
        y_plot = time_model.predict(X_poly)
        ax2.plot(X_plot, y_plot, 'r-', label='Fit')
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Dist (km)')
        ax2.set_title(f'Decel: {accel:.3f} km/s2')
        plt.tight_layout()
        
        # Save Plot
        out_file = os.path.join(args.event_dir, "grazer_plot.png")
        plt.savefig(out_file)
        print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    main()
