#!/usr/bin/env python3
"""
Creates a .kml file for Google Earth, displaying a meteor trajectory and
observers' lines of sight. Reads a .res file created by metrack.py.
"""

import sys
import argparse
from pathlib import Path
from collections import namedtuple
import simplekml

# --- Constants for KML Styling ---
HEIGHT_MULTIPLIER = 1000  # Convert sight point height from km to meters for KML
KML_POINT_ICON = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
TRAJECTORY_COLOR = simplekml.Color.yellow
TRAJECTORY_WALL_COLOR = '7700ff00'  # Transparent green
SIGHT_LINE_COLOR = simplekml.Color.red

# --- Data Structures for Parsed Data ---
TrajectoryPoint = namedtuple('TrajectoryPoint', ['lon', 'lat', 'height', 'desc'])
# Added 'elevation' to the Observer tuple to store the observer's ground height
Observer = namedtuple('Observer', ['lon', 'lat', 'sight_lon', 'sight_lat', 'sight_height', 'name', 'elevation'])


def parse_res_file(filepath: Path) -> tuple[list[TrajectoryPoint], list[Observer]]:
    """
    Parses a .res file, separating trajectory points from observer data.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Error: Input file not found at {filepath}")

    with filepath.open('r') as f:
        lines = [line.split() for line in f if line.strip() and not line.startswith('#')]

    if len(lines) < 2:
        raise ValueError("Error: Not enough data. A .res file must contain at least two lines for the trajectory.")

    traj_lines = lines[:2]
    trajectory_points = [
        TrajectoryPoint(lon=float(ln[0]), lat=float(ln[1]), height=float(ln[4]), desc=ln[5])
        for ln in traj_lines
    ]

    observer_lines = lines[2:]
    observers = []
    for ln in observer_lines:
        observers.append(
            Observer(
                lon=float(ln[0]), lat=float(ln[1]),
                sight_lon=float(ln[2]), sight_lat=float(ln[3]),
                sight_height=float(ln[4]), name=ln[5],
                elevation=float(ln[6]) if len(ln) >= 7 else 0.0
            )
        )
    return trajectory_points, observers


def create_kml(doc_name: str, trajectory: list[TrajectoryPoint], observers: list[Observer]) -> simplekml.Kml:
    """
    Creates a KML object with the meteor trajectory and observer data.
    """
    kml = simplekml.Kml(name=doc_name)
    start_point, end_point = trajectory

    traj_line = kml.newlinestring(
        name="Trajectory",
        description="Meteor track extended to ground",
        extrude=1,
        altitudemode=simplekml.AltitudeMode.absolute,
        coords=[
            (start_point.lon, start_point.lat, start_point.height * HEIGHT_MULTIPLIER),
            (end_point.lon, end_point.lat, end_point.height * HEIGHT_MULTIPLIER)
        ]
    )
    traj_line.style.linestyle.color = TRAJECTORY_COLOR
    traj_line.style.linestyle.width = 3
    traj_line.style.polystyle.color = TRAJECTORY_WALL_COLOR

    style = simplekml.Style()
    style.iconstyle.icon.href = KML_POINT_ICON
    style.iconstyle.scale = 0.5
    style.labelstyle.scale = 0.8

    for point in trajectory:
        pnt = kml.newpoint(name=point.desc, coords=[(point.lon, point.lat)])
        pnt.style = style

    for obs in observers:
        pnt = kml.newpoint(name=obs.name, coords=[(obs.lon, obs.lat)])
        pnt.style = style

        los_line = kml.newlinestring(
            name=f'Line of Sight: {obs.name}',
            altitudemode=simplekml.AltitudeMode.absolute,
            coords=[
                (obs.lon, obs.lat, obs.elevation * HEIGHT_MULTIPLIER),
                (obs.sight_lon, obs.sight_lat, obs.sight_height * HEIGHT_MULTIPLIER)
            ]
        )

        los_line.style.linestyle.color = SIGHT_LINE_COLOR
        los_line.style.linestyle.width = 1

    return kml


def fb2kml(inname: str = ''):
    """
    Creates a .kml file from a .res file.

    Args:
        inname: The path to the input .res file.
    """
    if not inname:
        print("Error: No input file name provided.", file=sys.stderr)
        return

    input_path = Path(inname)
    output_path = input_path.with_suffix('.kml')

    try:
        trajectory, observers = parse_res_file(input_path)
        kml_doc = create_kml(input_path.stem, trajectory, observers)
        kml_doc.save(output_path)
        print(f"✅ KML file successfully created: {output_path}")

    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Creates a .kml file from a .res file, showing a meteor trajectory and observer lines of sight.',
        epilog='Example: ./fb2kml.py obs_20110424.res'
    )
    parser.add_argument('input_file', help='The path to the input .res file.')
    args = parser.parse_args()
    
    fb2kml(args.input_file)
