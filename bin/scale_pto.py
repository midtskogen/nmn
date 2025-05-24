#!/usr/bin/python

gpl = r"""
    scale_pto - recreate a panorama with differently sized
                versions of the images it contains

    Copyright (C) 2010  Kay F. Jahnke

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import sys
import numbers
import argparse
import parse_pto
import random
import os

# scale single int or float values

def simple_scale ( feature , factor ) :

    if type ( feature ) == int :
        return int ( feature * factor + 0.5 ) # rounds
    elif type ( feature ) == float :
        return feature * factor

    raise TypeError ( "simple_scale: cannot scale type %s" , type ( feature ) )

# scale with two different factors

def xy_scale ( feature , xfactor , yfactor ) :

    if isinstance ( feature , numbers.Number ) :
        # something simple, pass to simple_scale with xfactor
        return simple_scale ( feature , xfactor )

    # next bit works for all assignable sequences if they contain numbers

    for nr in range ( len ( feature ) ) :
        if nr % 2 : # odd, use yfactor
            feature [ nr ] = simple_scale ( feature [ nr ] , yfactor )
        else :
            feature [ nr ] = simple_scale ( feature [ nr ] , xfactor )
    return feature

# we need this for croppings, which we parse to 4-tuples

def rect_scale ( rect , xfactor , yfactor ) :

    # rectangles are lrtb in hugin

    return ( simple_scale ( rect[0] , xfactor ) ,
             simple_scale ( rect[1] , xfactor ) ,
             simple_scale ( rect[2] , yfactor ) ,
             simple_scale ( rect[3] , yfactor ) )

# main processing routine

def scale_panorama ( scan , xfactor , yfactor , first ) :
    
    # scale all the i lines

    try :

        for line in scan.i :

            # modify the width and height
            line.w.value = simple_scale ( line.w.value , xfactor )
            line.h.value = simple_scale ( line.h.value , yfactor )

            # modify d and e, which are in pixels, if they aren't back refs
            if line.d.type != 'b' :
                line.d.value = simple_scale ( line.d.value , xfactor )
            if line.e.type != 'b' :
                line.e.value = simple_scale ( line.e.value , yfactor )

            # modify crop
            if hasattr ( line , 'S' ) :
                line.S.value = rect_scale ( line.S.value , xfactor , yfactor )
 
            if hasattr ( line , 'C' ) :
                line.S.value = rect_scale ( line.C.value , xfactor , yfactor )
            if first:
                break
                
    except AttributeError :
         pass

    # scale control points

    try :

        for line in scan.c :

            line.x.value = simple_scale ( line.x.value , xfactor )
            line.y.value = simple_scale ( line.y.value , yfactor )
            if not first:
                line.X.value = simple_scale ( line.X.value , xfactor )
                line.Y.value = simple_scale ( line.Y.value , yfactor )

    except AttributeError:
        pass


    # scale masks

    try :
        
        for line in scan.k :

            # this is a bit more involved...
            # and note that we read and write out floats, even though
            # hugin usually writes integers. Seems hugin doesn't mind.
            
            k_values = [ float ( z ) for z in line.p.value.split() ]

            xy_scale ( k_values , xfactor , yfactor )

            result = '%f' % k_values[0]
            for z in k_values[1:] :
                result += ' %f' % z

            line.p.value = result
            
    except AttributeError:
        pass

# in case the user has provided a set of replacement images

def replace_images ( scan , images ) :

    if len ( scan.i ) != len ( images ) :
        raise IndexError ( "wrong number of replacement images" )

    for pair in zip ( scan.i , images ) :
        pair[0].n.value = pair[1]

def main() :
    
    # we create an argument parser
    
    parser = argparse.ArgumentParser (
        formatter_class=argparse.RawDescriptionHelpFormatter ,
        description = gpl + '''
    This script is used to scale pto files. The scaling is
    passed as an arbitrary float factor, which functions as
    a multiplicator for all values which are coordinates.
    Optionally a second factor can be passed to scale differently
    in the horizontal and vertical - this should be rarely needed.
    The script scales:
    - image size
    - control points
    - d and e
    - croppings
    - masks
    If you pass a set of images after the other parameters, the script
    will fill these into the pto to use instead of the ones present.
    You have to pass the correct number of images or none, in which
    case the image names stay the same.
    The filled-in images aren't checked to have the specified sizes.
    known issues: scaling of integral values may introduce imprecisions.
    they are scaled as floats and then rounded and converted to int.
    ''' )
    
    parser.add_argument('-x', '--x-factor',
                        metavar='<factor>',
                        required=True,
                        type=float,
                        help='horizontal scaling factor')

    parser.add_argument('-y', '--y-factor',
                        metavar='<factor>',
                        required=False,
                        default=0.0,
                        type=float,
                        help='vertical scaling factor, default == x')

    parser.add_argument('-f', '--first',
                        required=False,
                        default=False,
                        action="store_true",
                        help='scale first image only')

    parser.add_argument('-p', '--pto',
                        metavar='<pto file>',
                        required=True,
                        type=argparse.FileType('r'),
                        help='pto file to be processed')

    parser.add_argument('-o', '--output',
                        metavar='<output file>',
                        type=argparse.FileType('w'),
                        required=True,
                        help='output file')

    parser.add_argument('images',
                        metavar='<image>',
                        type=str,
                        nargs='*',
                        help='images to use instead of those in pto')

    if len ( sys.argv ) < 4 :
        parser.print_help()
        return

    args = parser.parse_args( sys.argv[1:] )
    if args.y_factor == 0.0 :
        args.y_factor = args.x_factor

    scan = parse_pto.pto_scan ( args.pto )
    scan.make_member_access()

    scale_panorama ( scan , args.x_factor , args.y_factor , args.first )

    if len ( args.images ) :
        replace_images ( scan , args.images )

    scan.pto ( args.output )


# are we main? if so, do the main thing...

if __name__ == "__main__":
    main()
