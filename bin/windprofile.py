#!/usr/bin/env python3

# This can be used in two ways:
#
# $ windprofile.py -d $(date +%s -u -d "yesterday 10:00") 59.809 5.227

# This will download the global GFS model (13 GB!) for the Unix timestamp
# specified by the -d option and create a wind profile for 59.809N 5.227E,
# default output is "wind_profile.csv" (can be changed with the -o
# option).

# $ windprofile.py 59.809 5.227 gfs.t18z.atmanl.nc

# As above but this will assume that the GFS model has already been
# downloaded as gfs.t18z.atmanl.nc.

# Note that the data is only available for a limited time on the server
# (about 10 days, it seems).

import csv
import requests
import argparse
import xarray as xr
import numpy as np
import metpy.calc as met
import os
from metpy.units import units
from datetime import datetime, timezone, UTC
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extract wind profile.')
parser.add_argument(action='store', dest='latitude', help='latitude')
parser.add_argument(action='store', dest='longitude', help='longitude')
parser.add_argument(action='store', nargs='?', dest='dataset', help='input GFS dataset file (automatically download if not specified)')
parser.add_argument('-d', '--date', dest='timestamp', help='date')
parser.add_argument('-o', '--output', dest='csv', help='csv output file', default='wind_profile.csv')

args = parser.parse_args()

if args.timestamp:
    # Convert from Unix timestamp
    utc = datetime.fromtimestamp(int(args.timestamp), UTC).replace(tzinfo=timezone.utc)
    year = utc.year
    month = str(utc.month).zfill(2)
    day = str(utc.day).zfill(2)
    hour = str(int(((utc.hour + 3) % 24) / 6) * 6).zfill(2)

    if args.dataset:
        print(f"Timestamp specified, ignoring the \"{args.dataset}\" argument")

    url=f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{year}{month}{day}/{hour}/atmos/gfs.t{hour}z.atmanl.nc"
    dest=f"gfs_{year}{month}{day}{hour}.nc"
    if os.path.exists(dest):
        print(f"{dest} seems to have been downloaded already, not downloading again")
    else:
        print(f"Downloading {url} as {dest}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(dest, 'wb') as file, tqdm(desc=dest, total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

    args.dataset = dest

# Open the dataset
ds = xr.open_dataset(args.dataset)

# Get the dimension values
grid_xt_size = ds.sizes['grid_xt']
grid_yt_size = ds.sizes['grid_yt']
pfull_size = ds.sizes['pfull']

# Find the grid indices
lat_idx = np.clip(round(grid_yt_size * (90 - float(args.latitude)) / 180), 0, grid_yt_size - 1)
lon_idx = np.clip(round(grid_xt_size * (float(args.longitude)) / 360), 0, grid_xt_size - 1)

data = []
altitude = 0
for p in range(pfull_size-1, -1, -1):
    pres = round(float(ds['pfull'].isel(pfull=p).values) * 100, 1)
    ugrd = ds['ugrd'].isel(time=0, pfull=p, grid_yt=lat_idx, grid_xt=lon_idx).values
    vgrd = ds['vgrd'].isel(time=0, pfull=p, grid_yt=lat_idx, grid_xt=lon_idx).values
    spfh = ds['spfh'].isel(time=0, pfull=p, grid_yt=lat_idx, grid_xt=lon_idx).values
    temp = round(float(ds['tmp'].isel(time=0, pfull=p, grid_yt=lat_idx, grid_xt=lon_idx).values), 1)
    rh = round((met.relative_humidity_from_specific_humidity(pres * units.hPa, temp * units.degK, spfh).magnitude*100), 1)

    wind_speed = round(float(met.wind_speed(units.Quantity(ugrd, 'm/s'), units.Quantity(vgrd, 'm/s')).magnitude), 1)
    wind_dir = round(float(met.wind_direction(units.Quantity(ugrd, 'm/s'), units.Quantity(vgrd, 'm/s')).magnitude), 1)
    altitude = altitude - ds['delz'].isel(time=0, pfull=p, grid_yt=lat_idx, grid_xt=lon_idx).values
    alt = round(altitude, 1)
    print("Working: {:2.1%}".format(alt / 30000), end='\r')
    data.append([alt, temp, pres, rh, wind_speed, wind_dir])
    if altitude > 30000:
        break
print("               ", end='\r')

with open(args.csv, "w") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["# Height", "TempK", "Press", "RHum", "Wind", "WDir"])
    writer.writerows(data)
