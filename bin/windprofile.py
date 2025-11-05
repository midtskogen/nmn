#!/usr/bin/env python3

# This can be used in two ways:
#
# As a script:
# $ windprofile.py -d $(date +%s -u -d "yesterday 10:00") 59.809 5.227
#
# This will fetch a wind profile from the Open-Meteo API for the given
# timestamp and location.
# Default output is "wind_profile.csv" (can be changed with the -o option).
#
# As a module:
# import windprofile
# success, matched_time = windprofile.get_wind_profile(lat, lon, timestamp, 'output.csv')
# if success:
#   windprofile.generate_wind_plot('output.csv', translations_dict, matched_time, 'nb_')

import csv
import requests
import argparse
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
# Use timezone.utc for Python < 3.11 compatibility
from datetime import datetime, timezone, timedelta

def get_wind_profile(latitude, longitude, timestamp, csv_output_file, dataset_file=None):
    """
    Fetches an upper-air wind profile from the Open-Meteo API for a specific
    lat, lon, and time, and saves it to a CSV file.
    
    NOTE: The Open-Meteo forecast API only provides pressure-level (high-altitude)
    data for the last ~14 days. This function will gracefully fail for
    older events.

    Args:
        latitude (float): Latitude.
        longitude (float): Longitude.
        timestamp (int or float): Unix timestamp (UTC).
        csv_output_file (str): Path to save the output CSV.
        dataset_file (str, optional): This argument is ignored, but kept
                                     for compatibility with fetch.py.

    Returns:
        tuple: (bool: True on success, False on failure,
                str: The matched data timestamp string, or None on failure)
    """
    try:
        # Convert timestamp to datetime object and required date string
        # Use timezone.utc for compatibility with Python < 3.11
        utc = datetime.fromtimestamp(int(timestamp), timezone.utc)
        date_str = utc.strftime('%Y-%m-%d')
        now = datetime.now(timezone.utc)

        # --- Open-Meteo API setup ---
        
        # Check if the requested date is older than ~14 days.
        # The 'forecast' API only holds ~14-15 days of historical data.
        # The 'archive' API does not provide this pressure-level data.
        cutoff_days = 14
        if (now - utc).days >= cutoff_days:
            logging.warning(f"Event date {date_str} is >= {cutoff_days} days old. High-altitude wind profile data is not available via this API.")
            return False, None

        logging.info(f"Date {date_str} is < {cutoff_days} days old, using forecast API.")
        api_url = "https://api.open-meteo.com/v1/forecast"

        # We only need data up to 30km (approx 10 hPa)
        # Use a more detailed list of pressure levels for higher resolution.
        pressure_levels_hpa = [
            1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 
            550, 500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100,
            70, 50, 30, 20, 10
        ]
        
        # Base variables (relativehumidity removed as requested)
        variables_forecast = ["geopotential_height", "windspeed", "winddirection", "temperature"]
        
        # Forecast API needs variables and levels combined
        hourly_params = []
        for var in variables_forecast:
            for level in pressure_levels_hpa:
                level_str = str(level).replace('.', '_')
                hourly_params.append(f"{var}_{level_str}hPa")
        
        hourly_string = ",".join(hourly_params)
        
        params = {
            "latitude": round(latitude, 2),
            "longitude": round(longitude, 2),
            "start_date": date_str,
            "end_date": date_str,
            "hourly": hourly_string,
            "timezone": "UTC",
            "wind_speed_unit": "ms"
        }

        logging.info(f"Fetching wind profile from Open-Meteo for {date_str}...")
        
        # --- Make API Request ---
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()  # Raise an exception for 4xx/5xx status
        
        response_json = response.json()
        
        hourly_data = response_json.get("hourly", {})

        if "time" not in hourly_data:
            logging.error(f"Open-Meteo API response is missing 'time' data in 'hourly' block: {response_json}")
            return False, None

        # --- Process API Response ---
        
        # Find the hour in the response that is closest to our event timestamp
        hourly_times = hourly_data["time"]
        # Convert API times (string) to timestamps for comparison
        api_timestamps = [int(datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp()) for t in hourly_times]
        
        hour_index = np.argmin(np.abs(np.array(api_timestamps) - timestamp))
        matched_time_str = hourly_times[hour_index]
        
        logging.info(f"Event time {utc} matched to API data for {matched_time_str}")

        data = []
        
        # Process RECENT (Forecast) data
        # Data is in keyed arrays, e.g., "windspeed_1000hPa"
        for level in pressure_levels_hpa:
            try:
                level_str = str(level).replace('.', '_')
                
                # Use 'geopotential_height' (Forecast API name), which is already in meters.
                alt_m = hourly_data[f"geopotential_height_{level_str}hPa"][hour_index]
                temp_c = hourly_data[f"temperature_{level_str}hPa"][hour_index]
                # Use the correct variable names (no underscores)
                speed_ms = hourly_data[f"windspeed_{level_str}hPa"][hour_index]
                dir_deg = hourly_data[f"winddirection_{level_str}hPa"][hour_index]
                pres_pa = level * 100

                # Check for None/null values
                if any(v is None for v in [alt_m, temp_c, speed_ms, dir_deg]):
                    logging.warning(f"Skipping pressure level {level}hPa due to null data.")
                    continue

                temp_k = round(temp_c + 273.15, 2)
                # The data is now guaranteed to be in m/s
                data.append([alt_m, temp_k, pres_pa, speed_ms, dir_deg])

            except KeyError:
                logging.debug(f"No data for pressure level {level}hPa.")
                continue
            except (TypeError, ValueError) as e:
                    logging.warning(f"Error processing data for level {level}hPa: {e}. Skipping.")
                    continue

        if not data:
            logging.error("No valid data points extracted from Open-Meteo response.")
            return False, None

        logging.info(f"Finished processing Open-Meteo data. Extracted {len(data)} profile points.")

        # Write data to CSV, sorted by altitude ascending
        with open(csv_output_file, "w", newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Update header to remove RHum_pct
            writer.writerow(["# Height_m", "Temp_K", "Pressure_Pa", "WindSpeed_ms", "WindDir_deg"])
            writer.writerows(sorted(data, key=lambda x: x[0]))
        
        return True, matched_time_str

    except requests.exceptions.RequestException as e:
        # This will catch 4xx errors for data that is too old
        logging.error(f"Failed to fetch data from Open-Meteo (data might be too old): {e}")
        return False, None
    except Exception as e:
        logging.error(f"An error occurred in get_wind_profile: {e}", exc_info=True)
        return False, None


def generate_wind_plot(csv_path, translations: dict, matched_data_time: str, output_prefix: str = ''):
    """
    Generates a wind profile plot from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file generated by get_wind_profile.
        translations (dict): Dictionary for localization of plot labels.
        matched_data_time (str): The ISO 8601 timestamp string of the data, e.g. "2025-10-22T10:00".
        output_prefix (str, optional): Language prefix (e.g., 'nb_') for the output file.
    """
    try:
        altitudes_km_raw = []
        wind_speeds_raw = []
        wind_dirs_raw = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                try:
                    alt_m = float(row[0])
                    # Only plot up to 30 km as requested
                    if 0 <= alt_m <= 30000:
                        altitudes_km_raw.append(alt_m / 1000.0)
                        # Read from correct column (index 3)
                        wind_speeds_raw.append(float(row[3]))
                        # Read from correct column (index 4)
                        wind_dirs_raw.append(float(row[4]))
                except (ValueError, IndexError):
                    continue
        
        if not altitudes_km_raw:
            logging.warning("No data found to plot in wind profile CSV.")
            return

        # --- Process Wind Direction for wrapping ---
        # We need to find jumps > 180 degrees (e.g., 350 -> 10) and insert NaN
        # to break the line plot.
        
        alt_plot = []
        dir_plot = []
        
        if len(wind_dirs_raw) > 0:
            # Add the first point
            alt_plot.append(altitudes_km_raw[0])
            dir_plot.append(wind_dirs_raw[0])
            
            # Iterate from the second point
            for i in range(1, len(wind_dirs_raw)):
                # Get the last valid (non-NaN) plotted value
                j = -1
                while j >= -len(dir_plot) and np.isnan(dir_plot[j]):
                    j -= 1
                prev_dir = dir_plot[j]

                curr_dir = wind_dirs_raw[i]

                # This is the wrap-around check
                if abs(curr_dir - prev_dir) > 180:
                    # Insert a NaN point to break the line
                    alt_plot.append(np.nan)
                    dir_plot.append(np.nan)
                
                # Add the current point
                alt_plot.append(altitudes_km_raw[i])
                dir_plot.append(curr_dir)
        
        # --- End Wind Direction processing ---

        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7), sharey=True)
        
        # --- Updated Title ---
        title = translations.get("wind_profile", "Wind Profile")
        if matched_data_time:
            # Format time from '2025-10-22T10:00' to '2025-10-22 10:00 UTC'
            try:
                dt_obj = datetime.fromisoformat(matched_data_time.replace('Z', '+00:00'))
                formatted_time = dt_obj.strftime('%Y-%m-%d %H:%M UTC')
                title = f"{title}\n({formatted_time})"
            except ValueError:
                title = f"{title}\n({matched_data_time})" # fallback
        fig.suptitle(title, fontsize=16)
        # --- End Updated Title ---

        # Plot 1: Wind Speed
        ax1.plot(wind_speeds_raw, altitudes_km_raw, 'b-') # Use original data
        ax1.set_xlabel(translations.get("wind_speed_ms", "Wind Speed (m/s)"), fontsize=12)
        ax1.set_ylabel(translations.get("altitude_km", "Altitude (km)"), fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_ylim(0, 30) # Lock Y-axis from 0 to 30 km

        # Plot 2: Wind Direction
        ax2.plot(dir_plot, alt_plot, 'r-') # Use processed data with NaNs
        ax2.set_xlabel(translations.get("wind_direction_deg", "Wind Direction (°)"), fontsize=12)
        ax2.set_xticks([0, 90, 180, 270, 360])
        ax2.set_xlim(0, 360)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_filename = f"{output_prefix}wind_profile.svg"
        plt.savefig(output_filename)
        plt.close(fig)
        logging.info(f"Wind profile plot saved to {output_filename}")

    except FileNotFoundError:
        logging.error(f"Could not find wind profile CSV at {csv_path} to generate plot.")
    except Exception as e:
        logging.error(f"Failed to generate wind plot: {e}", exc_info=True)


if __name__ == "__main__":
    # This part runs only if the script is executed directly
    parser = argparse.ArgumentParser(description='Extract wind profile from Open-Meteo API.')
    parser.add_argument(action='store', dest='latitude', help='Latitude (decimal degrees)')
    parser.add_argument(action='store', dest='longitude', help='Longitude (decimal degrees)')
    parser.add_argument('-d', '--date', dest='timestamp', help='Unix timestamp (UTC). If not given, will use current time.')
    parser.add_argument('-o', '--output', dest='csv', help='CSV output file', default='wind_profile.csv')
    
    args = parser.parse_args()

    # Configure basic logging for script use
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    if args.timestamp:
        timestamp = int(args.timestamp)
    else:
        logging.info("No timestamp provided, using current time.")
        # Use timezone.utc for compatibility with Python < 3.11
        timestamp = int(datetime.now(timezone.utc).timestamp())

    success, matched_time = get_wind_profile(
        # Convert command-line args to float, as they are read as strings
        latitude=float(args.latitude),
        longitude=float(args.longitude),
        timestamp=timestamp,
        csv_output_file=args.csv
    )

    if success:
        print(f"Successfully created wind profile at {args.csv}")
        # Generate a sample plot (using default English translations)
        try:
            import matplotlib.pyplot as plt
            print("Generating sample plot wind_profile.svg...")
            sample_translations = {
                "wind_profile": "Wind Profile",
                "wind_speed_ms": "Wind Speed (m/s)",
                "altitude_km": "Altitude (km)",
                "wind_direction_deg": "Wind Direction (°)"
            }
            generate_wind_plot(args.csv, sample_translations, matched_time)
        except ImportError:
            print("Install matplotlib to generate a sample plot.")
    else:
        print("Failed to create wind profile.")
