#!/usr/bin/env python3
# Converted to Python 3

def fb2kml(inname = ''):
    """
    Creates a .kml file for Google Earth, displaying a meteor trajectory 
    and observers' lines of sight. Reads in a .res file created by metrack.py.
    """

    import os
    import numpy
    import simplekml
    import argparse

    path, longname = os.path.split(inname)

    if path != '':
        path=path+('/')
    elem = longname.split('.')
    name = elem[0]
    if len(elem) < 2:
        ext='res'
    else:
        ext=elem[1]

    # Pre-declare data lists
    long1  = []
    lat1   = []
    long2  = []
    lat2   = []
    height = []
    desc   = []

    # Read in data
    try:
        with open(os.path.join(path, name + '.' + ext), 'r') as f:
            for line in f:
                if line and line[0] != '#':
                    words=line.split()
                    long1.append(float(words[0]))
                    lat1.append(float(words[1]))
                    long2.append(float(words[2]))
                    lat2.append(float(words[3]))
                    height.append(float(words[4]))
                    desc.append(words[5])
    except FileNotFoundError:
        print(f"Error: Input file not found at {os.path.join(path, name + '.' + ext)}")
        return

    ndata = len(long1)
    if ndata < 2:
        print("Error: Not enough data in the file to generate a KML.")
        return

    long1  = numpy.array(long1)
    lat1    = numpy.array(lat1)
    long2  = numpy.array(long2)
    lat2   = numpy.array(lat2)
    height = numpy.array(height)
    desc   = numpy.array(desc)

    kml = simplekml.Kml()
    kml.document.name = name

    # Create points for observers and start/end points
    # The original range was complex, simplifying it. It iterates through all observers + start/end points.
    for i in range(ndata):
        # Only create points for observers and start/end, not the line-of-sight intersections
        if i < 2 or (i >= 2 and desc[i] not in ['Start', 'End']):
             pnt = kml.newpoint(name=desc[i], coords=[(long1[i],lat1[i])])
             pnt.style.labelstyle.scale = 1.5
             pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
             pnt.style.iconstyle.scale = 0.5


    # Create lines of sight for observers
    for i in range(2, ndata):
        lin = kml.newlinestring(altitudemode=simplekml.AltitudeMode.absolute,
              name = 'Line of sight: ' + desc[i],
              description="Line of sight",
              coords=[(long1[i], lat1[i], 0),
              (long2[i], lat2[i], 1000*height[i])])
        lin.style.linestyle.color = simplekml.Color.red
        lin.style.linestyle.width = 1

    # Create trajectory line
    lin = kml.newlinestring(name="Trajectory", 
        description="Meteor track extended to ground",
        extrude=1,
        altitudemode=simplekml.AltitudeMode.absolute,
        coords=[(long1[0],lat1[0],1000*height[0]),
                (long1[1],lat1[1],1000*height[1])])
    lin.style.linestyle.color='ff00ffff' # yellow
    lin.style.polystyle.color='7700ff00' # transparent green
    lin.style.polystyle.outline=1
    lin.style.linestyle.width = 2

    kml.save(os.path.join(path, name + '.kml'))
    print(f"KML file saved to {os.path.join(path, name + '.kml')}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Creates a .kml file for Google Earth, displaying a meteor trajectory and observers line of sight. Reads in a .res file created by metrack.py',
        epilog='Example: ./fb2kml.py obs_20110424.res'
    )
    parser.add_argument('input_file', help='The input .res file to process.')
    
    args = parser.parse_args()

    if not args.input_file:
        print("No input file given!")
        sys.exit(1)

    fb2kml(args.input_file)
