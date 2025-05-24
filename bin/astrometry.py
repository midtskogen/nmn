#!/usr/bin/env python3

# Improve calibration of a pto file using stars in the image.
# Usage: astrometry.py <Unix timestamp> <image file> <output pto file>

# echo convert -pointsize 12 -strokewidth 2 -draw \"fill none stroke white $crosses\" $text $(~/bin/brightstar.py -f 3.5 --latitude 54.09625 --longitude 11.9208 $(date +%s -u -d "2016-03-02 21:34:45 UTC") y.pto 2> /dev/null | sed 's/[(),]//g;s/'\''//g' | awk '{x=$1; y=$2; az=$3; alt=$4; $1=$2=$3=$4=""; sub(/ */, ""); printf("-stroke white -fill none -draw \"circle %f,%f %f,%f\" -stroke none -fill white -annotate +%f+%f \"%s [%.2f %.2f]\"\n", x, y, x+7, y, x+11, y-4, $0, az, alt)}') x.jpg labels.png | bash; eog labels.png

from __future__ import print_function
import sys
import ephem
import glob
from datetime import datetime, UTC
import math
import argparse
import os
import tempfile
import subprocess
import astropy.io.fits as pyfits
import configparser
from PIL import Image

parser = argparse.ArgumentParser(description='Use astrometry.net to analyse an image and produce a Hugin .pto file.')

parser.add_argument('-x', '--longitude', dest='longitude', help='observer longitude', type=float)
parser.add_argument('-y', '--latitude', dest='latitude', help='observer latitude', type=float)
parser.add_argument('-e', '--elevation', dest='elevation', help='observer elevation (m)', type=float)
parser.add_argument('-t', '--temperature', dest='temperature', help='observer temperature (C)', type=float)
parser.add_argument('-p', '--pressure', dest='pressure', help='observer air pressure (hPa)', type=float)
parser.add_argument('-q', '--sigma', dest='sigma', help='noise level (default: 20)', default=20.0, type=float)
parser.add_argument('-P', '--projection', dest='projection', help='source image projection (default: 0 = gnomonic)', default=0, type=int)
parser.add_argument('--odds-to-solve', dest='odds', help='odds ratio at which to consider a field solved (default: 1e9)', default=1e9, type=float)
parser.add_argument('-c', '--code-tolerance', dest='tolerance', help='matching distance for quads (default: 0.01)', default=0.01, type=float)
parser.add_argument('-L', '--scale-low', dest='low', help='lower bound of image scale estimate (default: 10)', default=10, type=float)
parser.add_argument('-H', '--scale-high', dest='high', help='upper bound of image scale estimate (default: 180)', default=180, type=float)
parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='be more verbose')

parser.add_argument(action='store', dest='timestamp', help='Unix timestamp (seconds since 1970-01-01 00:00:00UTC)')
parser.add_argument(action='store', dest='picture', help='Image file')
parser.add_argument(action='store', dest='outfile', help='Hugin .pto file (output)')
args = parser.parse_args()


pos = ephem.Observer()

config = configparser.ConfigParser()
config.read(['/etc/meteor.cfg', os.path.expanduser('~/meteor.cfg')])

pos.lat = config.get('astronomy', 'latitude')
pos.lon = config.get('astronomy', 'longitude')
pos.elevation = float(config.get('astronomy', 'elevation'))
pos.temp = float(config.get('astronomy', 'temperature'))
pos.pressure = float(config.get('astronomy', 'pressure'))

if args.longitude:
    pos.lon = str(args.longitude)
if args.latitude:
    pos.lat = str(args.latitude)
if args.elevation:
    pos.elevation = args.elevation
if args.temperature:
    pos.temp = args.temperature
if args.pressure:
    pos.pressure = args.pressure

img = Image.open(args.picture)
width, height = img.size

pos.date = datetime.fromtimestamp(float(args.timestamp), UTC).strftime('%Y-%m-%d %H:%M:%S')
temp = tempfile.NamedTemporaryFile(prefix='astrometry_', dir='.', delete=True)
out = sys.stdout if args.verbose else open(os.devnull, 'wb')
subprocess.Popen(['solve-field', '-L', str(args.low), '-H', str(args.high), '--odds-to-solve', str(args.odds), '-c', str(args.tolerance), '--sigma', str(args.sigma), '-o', os.path.basename(temp.name), args.picture], stdout=out, stderr=out).wait()
corrfile = os.path.splitext(os.path.basename(temp.name))[0] + '.corr'
axyfile = os.path.splitext(os.path.basename(temp.name))[0] + '.axy'
rdlsfile = os.path.splitext(os.path.basename(temp.name))[0] + '.rdls'
xylsfile = os.path.splitext(os.path.basename(temp.name))[0] + '-indx.xyls'

corr = pyfits.open(corrfile)

for f in glob.glob(os.path.basename(temp.name) + '?*'):
    os.unlink(f)

cat = [
    (  6.752478, -16.716117, -1.43, 'Sirius'),
    (  6.399197, -52.695661, -0.46, 'Canopus'),
    ( 18.615650,  38.783692,  0.03, 'Vega'),
    (  5.278156,  45.997992,  0.08, 'Capella'),
    ( 14.261019,  19.182414,  0.16, 'Arcturus'),
    (  5.242297,  -8.201642,  0.28, 'Rigel'),
    ( 14.660136, -60.833975,  0.30, 'Rigel Kentaurus'),
    (  7.655033,   5.224994,  0.40, 'Procyon'),
    (  1.628569, -57.236758,  0.54, 'Achernar'),
    (  5.919531,   7.407064,  0.57, 'Betelgeuse'),
    ( 14.063722, -60.373039,  0.64, 'Agena'),
    ( 19.846389,   8.868322,  0.93, 'Altair'),
    (  4.598678,  16.509300,  0.99, 'Aldebaran'),
    ( 13.419883, -11.161322,  1.06, 'Spica'),
    ( 16.490128, -26.432003,  1.07, 'Antares'),
    (  7.755264,  28.026197,  1.22, 'Pollux'),
    ( 22.960847, -29.622236,  1.23, 'Fomalhaut'),
    ( 12.443306, -63.099092,  1.28, 'Acrux'),
    ( 12.795350, -59.688764,  1.31, 'Mimosa'),
    ( 20.690531,  45.280339,  1.33, 'Deneb'),
    ( 14.659744, -60.837156,  1.35, 'Cen Alpha2'),
    ( 10.139531,  11.967208,  1.41, 'Regulus'),
    (  6.977097, -28.972083,  1.53, 'Adhara'),
    (  7.576628,  31.888278,  1.58, 'Castor'),
    ( 12.443456, -63.099522,  1.58, 'Cru Alpha2'),
    ( 17.560144, -37.103822,  1.63, 'Shaula'),
    ( 12.519433, -57.113211,  1.65, 'Gacrux'),
    (  5.418850,   6.349703,  1.66, 'Bellatrix'),
    (  9.219994, -69.717208,  1.67, 'Miaplacidus'),
    (  5.438197,  28.607450,  1.68, 'Alnath'),
    (  5.603558,  -1.201919,  1.72, 'Alnilam'),
    (  5.679314,  -1.942572,  1.74, 'Alnitak'),
    ( 12.900486,  55.959822,  1.76, 'Alioth'),
    ( 22.137219, -46.960975,  1.77, 'Alnair'),
    (  8.158875, -47.336589,  1.79, 'Suhail al Muhlif'),
    ( 18.402867, -34.384617,  1.81, 'Kaus Australis'),
    (  3.405381,  49.861181,  1.81, 'Mirfak'),
    ( 11.062131,  61.751033,  1.82, 'Dubhe'),
    (  7.139856, -26.393200,  1.84, 'Wesen'),
    ( 17.621981, -42.997825,  1.86, 'Sargas'),
    ( 13.792344,  49.313264,  1.86, 'Alkaid'),
    (  5.992144,  44.947433,  1.90, 'Menkalinan'),
    ( 16.811081, -69.027714,  1.91, 'TrA Alpha'),
    ( 20.427461, -56.735089,  1.92, 'Peacock'),
    (  8.745064, -54.708822,  1.94, 'Vel Delta'),
    (  8.375233, -59.509483,  1.95, 'Avior'),
    (  6.378331, -17.955917,  1.96, 'Mirzam'),
    (  9.459789,  -8.658603,  1.99, 'Alphard'),
    (  2.530300,  89.264108,  2.00, 'Polaris'),
    (  6.628528,  16.399253,  2.02, 'Alhena'),
    (  2.119558,  23.462422,  2.02, 'Hamal'),
    (  0.726492, -17.986606,  2.05, 'Diphda'),
    (  0.139794,  29.090431,  2.06, 'Alpheratz'),
    ( 14.845092,  74.155506,  2.06, 'Kochab'),
    (  5.795942,  -9.669606,  2.06, 'Saiph'),
    ( 18.921092, -26.296722,  2.07, 'Nunki'),
    (  1.162200,  35.620558,  2.08, 'Mirak'),
    ( 14.111375, -36.369956,  2.08, 'Menkent'),
    ( 17.582242,  12.560033,  2.09, 'Rasalhague'),
    (  3.136147,  40.955647,  2.11, 'Algol'),
    ( 22.711125, -46.884578,  2.12, 'Gru Beta'),
    ( 11.817661,  14.572061,  2.13, 'Denebola'),
    (  2.064986,  42.329725,  2.17, 'Almach'),
    ( 12.691956, -48.959889,  2.18, 'Cen Gamma'),
    (  0.945147,  60.716742,  2.18, 'Cas Gamma-27A'),
    (  9.133267, -43.432589,  2.21, 'Alsuhail'),
    ( 15.578131,  26.714694,  2.22, 'Alphecca'),
    (  8.059736, -40.003147,  2.22, 'Naos'),
    ( 13.398761,  54.925361,  2.22, 'Mizar'),
    (  5.533444,  -0.299081,  2.23, 'Mintaka'),
    ( 10.332875,  19.841489,  2.23, 'Algieba'),
    ( 20.370472,  40.256681,  2.23, 'Sadr'),
    ( 17.943436,  51.488894,  2.23, 'Etamin'),
    (  0.675122,  56.537331,  2.25, 'Schedar'),
    (  9.284836, -59.275228,  2.25, 'Aspidiske'),
    ( 13.664794, -53.466394,  2.28, 'Cen Epsilon'),
    (  0.152969,  59.149781,  2.28, 'Caph'),
    ( 16.836058, -34.293231,  2.29, 'Sco Epsilon-26'),
    ( 14.698822, -47.388200,  2.29, 'Lup Alpha'),
    ( 16.005556, -22.621731,  2.30, 'Dschubba'),
    ( 14.591783, -42.157825,  2.34, 'Cen Eta'),
    ( 11.030689,  56.382428,  2.35, 'Merak'),
    ( 17.708131, -39.029983,  2.39, 'Sco Kappa'),
    ( 21.736433,   9.875011,  2.39, 'Enif'),
    (  0.438069, -42.305981,  2.40, 'Ankaa'),
    ( 17.172969, -15.724911,  2.43, 'Sabik'),
    ( 11.897181,  53.694761,  2.43, 'Phecda'),
    (  7.401583, -29.303103,  2.46, 'Aludra'),
    ( 23.062906,  28.082789,  2.47, 'Scheat'),
    ( 21.309658,  62.585572,  2.47, 'Alderamin'),
    (  9.368561, -55.010669,  2.48, 'Vel Kappa'),
    ( 23.079347,  15.205264,  2.49, 'Markab'),
    ( 20.770189,  33.970256,  2.49, 'Gienah Cygni'),
    ( 14.749783,  27.074222,  2.50, 'Boo Epsilon-36A'),
    ( 13.925664, -47.288375,  2.53, 'Cen Zeta'),
    (  3.037992,   4.089733,  2.55, 'Menkar'),
    ( 11.235139,  20.523717,  2.56, 'Zosma'),
    ( 16.619317, -10.567089,  2.58, 'Oph Zeta-13'),
    ( 12.139306, -50.722425,  2.58, 'Cen Delta'),
    ( 12.263436, -17.541931,  2.59, 'Gienah'),
    (  5.545506, -17.822289,  2.59, 'Arneb'),
    ( 19.043531, -29.880106,  2.61, 'Ascella'),
    ( 15.283447,  -9.382917,  2.61, 'Lib Beta-27'),
    ( 16.090619, -19.805453,  2.62, 'Graffias'),
    ( 15.737797,   6.425628,  2.63, 'Cor Serpentis'),
    (  5.995353,  37.212586,  2.65, 'Aur Theta-37A'),
    (  1.910669,  20.808036,  2.66, 'Sheratan'),
    ( 12.573119, -23.396758,  2.66, 'Crv Beta-9'),
    (  5.660817, -34.074108,  2.66, 'Col Alpha'),
    ( 13.911411,  18.397717,  2.68, 'Muphrid'),
    (  4.949894,  33.166089,  2.68, 'Aur Iota-3'),
    ( 17.512733, -37.295811,  2.68, 'Lesath'),
    ( 14.975536, -43.133961,  2.68, 'Lup Beta'),
    (  1.430264,  60.235283,  2.68, 'Ruchbah'),
    ( 12.619728, -69.135564,  2.69, 'Mus Alpha'),
    ( 18.349900, -29.828103,  2.70, 'Kaus Media'),
    ( 19.770994,  10.613261,  2.71, 'Tarazed'),
    (  7.285711, -37.097469,  2.71, 'Pup Pi'),
    ( 10.779494, -49.420256,  2.72, 'Vel Mu'),
    ( 16.239094,  -3.694322,  2.73, 'Yed Prior'),
    ( 16.399858,  61.514214,  2.73, 'Dra Eta-14A'),
    ( 10.715944, -64.394450,  2.74, 'Car Theta'),
    ( 14.847975, -16.041778,  2.75, 'Lib Alpha-9A'),
    ( 13.343283, -36.712294,  2.76, 'Cen Iota'),
    ( 17.724542,   4.567303,  2.77, 'Cebalrai'),
    ( 16.503667,  21.489614,  2.78, 'Kornephoros'),
    ( 15.585681, -41.166758,  2.78, 'Lup Gamma'),
    (  5.590550,  -5.909903,  2.78, 'Nair'),
    (  5.130831,  -5.086447,  2.79, 'Cursa'),
    ( 12.252422, -58.748928,  2.79, 'Cru Delta'),
    ( 17.507211,  52.301386,  2.80, 'Rastaban'),
    ( 17.421664, -55.529883,  2.82, 'Ara Beta'),
    (  0.429186, -77.254244,  2.82, 'Hyi Beta'),
    (  0.220597,  15.183594,  2.83, 'Algenib'),
    (  8.125736, -24.304325,  2.83, 'Pup Rho-15'),
    ( 18.466178, -25.421700,  2.83, 'Kaus Borealis'),
    ( 16.598042, -28.216017,  2.83, 'Al Niyat'),
    ( 13.036278,  10.959150,  2.84, 'Vindemiatrix'),
    (  5.470756, -20.759442,  2.84, 'Nihal'),
    ( 15.919044, -63.430728,  2.84, 'TrA Beta'),
    ( 21.784014, -16.127286,  2.85, 'Deneb Algedi'),
    ( 16.688100,  31.602725,  2.85, 'Her Zeta-40'),
    ( 17.530692, -49.876144,  2.85, 'Ara Alpha'),
    ( 22.308358, -60.259589,  2.86, 'Tuc Alpha'),
    (  1.979497, -61.569858,  2.86, 'Hyi Alpha'),
    (  3.791411,  24.105136,  2.87, 'Alcyone'),
    (  3.902200,  31.883636,  2.88, 'Per Zeta-44A'),
    (  7.576694,  31.888492,  2.88, 'Gem Alpha-66B'),
    ( 15.315161, -68.679544,  2.88, 'TrA Gamma'),
    ( 15.980864, -26.114106,  2.89, 'Sco Pi-6'),
    ( 12.933797,  38.318381,  2.89, 'Cor Carioli'),
    ( 21.525981,  -5.571172,  2.89, 'Sadalsuud'),
    (  7.452511,   8.289317,  2.89, 'Gomeisa'),
    ( 19.162731, -21.023614,  2.90, 'Albaldah'),
    (  6.382675,  22.513586,  2.90, 'Calx'),
    ( 16.353144, -25.592797,  2.91, 'Alniyat'),
    (  3.964231,  40.010214,  2.91, 'Per Epsilon-45A'),
    ( 19.749578,  45.130808,  2.91, 'Cyg Delta-18A'),
    ( 22.096400,  -0.319850,  2.94, 'Sadalmelik'),
    ( 22.716706,  30.221244,  2.94, 'Matar'),
    (  6.832269, -50.614558,  2.94, 'Pup Tau'),
    (  3.079942,  53.506439,  2.94, 'Per Gamma-23A'),
    (  3.967158, -13.508517,  2.96, 'Zaurak'),
    ( 12.497739, -16.515433,  2.97, 'Algorab'),
    (  9.764186,  23.774256,  2.97, 'Algenubi'),
    ( 19.090169,  13.863478,  2.99, 'Al Okab Australis'),
    (  5.627414,  21.142550,  3.00, 'Tau Zeta-123'),
    ( 13.315361, -23.171511,  3.00, 'Hya Gamma-46'),
    ( 16.864508, -38.047381,  3.00, 'Sco Mu1'),
    ( 11.161058,  44.498486,  3.00, 'UMa Psi-52'),
    ( 12.168744, -22.619767,  3.01, 'Minkar'),
    (  6.732203,  25.131125,  3.01, 'Mebsuta'),
    ( 21.898811, -37.364853,  3.01, 'Gru Gamma'),
    ( 17.793078, -40.126997,  3.01, 'Sco Iota1'),
    (  9.785033, -65.072006,  3.01, 'Car Upsilon'),
    (  6.338553, -30.063367,  3.02, 'Furud'),
    (  2.159064,  34.987297,  3.02, 'Tri Beta-4'),
    (  3.715417,  47.787553,  3.02, 'Per Delta-39'),
    (  5.032814,  43.823308,  3.03, 'Almaaz'),
    ( 15.345478,  71.834017,  3.03, 'Pherkad'),
    (  7.050408, -23.833292,  3.04, 'CMa Omicron-24'),
    ( 14.534631,  38.308253,  3.04, 'Seginus'),
    ( 10.372150,  41.499517,  3.05, 'Tania Australis'),
    ( 19.512022,  27.959681,  3.08, 'Albireo'),
    ( 19.209250,  67.661542,  3.08, 'Altais'),
    ( 12.771336, -68.108119,  3.08, 'Mus Beta'),
    ( 20.350189, -14.781367,  3.09, 'Dabih'),
    ( 10.827081, -16.193647,  3.11, 'Hya Nu'),
    (  5.849331, -35.768308,  3.11, 'Wazn'),
    ( 20.626119, -47.291503,  3.11, 'Ind Alpha'),
    ( 16.977003, -55.990142,  3.11, 'Ara Zeta'),
    (  8.923231,   5.945564,  3.12, 'Hya Zeta-16'),
    ( 11.596356, -63.019842,  3.12, 'Cen Lambda'),
    ( 17.250531,  24.839206,  3.13, 'Her Delta-65A'),
    (  9.350917,  34.392561,  3.13, 'Lyn Alpha-40'),
    ( 18.293789, -36.761686,  3.13, 'Sgr Eta'),
    ( 14.986022, -42.104186,  3.13, 'Cen Kappa'),
    ( 17.250786,  36.809161,  3.14, 'Her Pi-67'),
    (  8.986792,  48.041825,  3.14, 'Talitha'),
    (  9.520367, -57.034378,  3.16, 'N VEL'),
    ( 18.760942, -26.990778,  3.17, 'Sgr Phi-27'),
    (  5.108581,  41.234475,  3.17, 'Aur Eta-10'),
    (  5.091017, -22.371033,  3.18, 'Lep Epsilon-2'),
    (  6.629353, -43.195933,  3.18, 'Pup Nu'),
    (  9.547619,  51.677300,  3.18, 'UMa Theta-25'),
    ( 14.708450, -64.975139,  3.18, 'Cir Alpha'),
    ( 17.146444,  65.714683,  3.18, 'Aldhibah'),
    ( 17.830967, -37.043303,  3.19, 'Basanismus'),
    (  4.830669,   6.961275,  3.19, 'Tabit'),
    ( 16.961139,   9.375033,  3.19, 'Oph Kappa-27'),
    ( 21.215608,  30.226917,  3.20, 'Cyg Zeta-64A'),
    (  2.971019, -40.304672,  3.22, 'Acamar'),
    ( 15.356200, -40.647517,  3.22, 'Lup Delta'),
    ( 23.655792,  77.632275,  3.22, 'Cep Gamma-35'),
    ( 21.477667,  70.560717,  3.23, 'Alfirk'),
    ( 16.305358,  -4.692511,  3.24, 'Yed Posterior'),
    ( 20.188414,  -0.821461,  3.25, 'Aql Theta-65'),
    ( 18.355167,  -2.898825,  3.25, 'Ser Eta-58'),
    ( 18.982394,  32.689558,  3.25, 'Sulafat'),
    (  6.803181, -61.941392,  3.25, 'Pic Alpha'),
    ( 17.366828, -24.999544,  3.26, 'Oph Theta-42'),
    ( 14.106194, -26.682361,  3.26, 'Hya Pi-49'),
    (  7.487175, -43.301433,  3.26, 'Pup Sigma'),
    (  4.566606, -55.044975,  3.26, 'Dor Alpha'),
    (  3.787317, -74.238961,  3.26, 'Hyi Gamma'),
    ( 22.910836, -15.820819,  3.27, 'Skat'),
    (  0.655467,  30.861025,  3.27, 'And Delta-31A'),
    ( 15.067839, -25.281964,  3.28, 'Brachium'),
    (  5.215528, -16.205469,  3.29, 'Lep Mu-5'),
    (  6.247961,  22.506800,  3.30, 'Propus'),
    ( 12.257100,  57.032617,  3.30, 'Megraz'),
    ( 15.415492,  58.966067,  3.30, 'Edasich'),
    ( 10.228950, -70.037903,  3.30, 'Car Omega'),
    ( 19.115669, -27.670422,  3.32, 'Sgr Tau-40'),
    ( 17.202553, -43.239189,  3.32, 'Sco Eta'),
    (  1.101403, -46.718414,  3.32, 'Phe Beta'),
    ( 17.423239, -56.377728,  3.32, 'Ara Gamma'),
    ( 17.983775,  -9.773631,  3.32, 'Oph Nu-64'),
    ( 11.237336,  15.429569,  3.33, 'Chertan'),
    (  7.821572, -24.859786,  3.33, 'Asmidiske'),
    (  6.754822,  12.895592,  3.34, 'Alzirr'),
    ( 22.180911,  58.201261,  3.34, 'Cep Zeta-21'),
    (  4.240411, -62.473858,  3.34, 'Ret Alpha'),
    (  1.906592,  63.670103,  3.35, 'Segin'),
    (  8.504408,  60.718169,  3.36, 'Muscida'),
    ( 17.244128,  14.390333,  3.37, 'Rasalgethi'),
    ( 19.424972,   3.114775,  3.37, 'Aql Delta-30A'),
    ( 13.578219,  -0.595819,  3.38, 'Heze'),
    ( 15.378019, -44.689608,  3.38, 'Lup Epsilon'),
    (  5.407950,  -2.397147,  3.39, 'Ori Eta-28A'),
    (  5.585633,   9.934158,  3.39, 'Meissa'),
    ( 13.825078, -41.687708,  3.40, 'Cen Nu'),
    (  8.779586,   6.418808,  3.40, 'Hya Epsilon-11A'),
    (  4.477706,  15.870883,  3.41, 'Tau Theta-78A'),
    (  3.086275,  38.840275,  3.41, 'Per Rho-25'),
    ( 15.204747, -52.099247,  3.41, 'Lup Zeta'),
    ( 22.691033,  10.831364,  3.42, 'Homam'),
    (  4.011339,  12.490347,  3.42, 'Tau Lambda-35'),
    ( 17.774314,  27.720675,  3.42, 'Her Mu-86A'),
    (  1.884697,  29.578828,  3.42, 'Atria'),
    ( 12.926725,   3.397469,  3.42, 'Auva'),
    ( 20.754828,  61.838781,  3.42, 'Cep Eta-3'),
    ( 20.749303, -66.203211,  3.42, 'Pav Beta'),
    ( 16.002036, -38.396706,  3.43, 'Lup Eta'),
    (  9.182803, -58.966894,  3.43, 'a Car'),
    ( 10.278172,  23.417311,  3.44, 'Adhafera'),
    ( 10.284942,  42.914364,  3.44, 'Tania Borealis'),
    (  1.472758, -43.318233,  3.44, 'Phe Gamma'),
    ( 19.104150,  -4.882556,  3.44, 'Aql Lambda-16'),
    (  0.818414,  57.815186,  3.45, 'Cas Eta-24A'),
    (  1.143164, -10.182264,  3.46, 'Deneb Algenubi'),
    (  7.946308, -52.982361,  3.46, 'Car Chi'),
    ( 10.332950,  19.840750,  3.47, 'Leo Gamma-41B'),
    (  7.028653, -27.934831,  3.47, 'CMa Sigma-22A'),
    (  2.721678,   3.235819,  3.47, 'Al Kaff'),
    ( 15.258378,  33.314833,  3.47, 'Boo Delta-49A'),
    ( 13.826942, -42.473731,  3.47, 'Cen Mu'),
    ( 12.694344,  -1.449375,  3.48, 'Porrima'),
    ( 16.714936,  38.922256,  3.48, 'Her Eta-44A'),
    ( 15.032433,  40.390567,  3.48, 'Nekkar'),
    (  1.734467, -15.937481,  3.49, 'Cet Tau-52'),
    ( 11.307983,  33.094306,  3.49, 'Alula Borealis'),
    ( 18.449561, -45.968458,  3.49, 'Tel Alpha'),
    ( 22.809250, -51.316864,  3.49, 'Gru Epsilon'),
    ( 12.694311,  -1.449431,  3.50, 'Vir Gamma-29B'),
    ( 22.828006,  66.200408,  3.50, 'Cep Iota-32'),
    ( 19.979286,  19.492147,  3.51, 'Sge Gamma-12'),
    ( 22.833386,  24.601578,  3.51, 'Peg Mu-48'),
    ( 10.122208,  16.762664,  3.52, 'Leo Eta-30'),
    ( 18.834664,  33.362667,  3.52, 'Sheliak'),
    (  8.275256,   9.185544,  3.52, 'Altarf'),
    ( 18.962167, -21.106653,  3.53, 'Sgr Xi2-37'),
    (  6.830683, -32.508478,  3.53, 'CMa Kappa-13'),
    ( 22.169997,   6.197864,  3.53, 'Biham'),
    (  3.720806,  -9.763394,  3.53, 'Eri Delta-23'),
    (  9.685842,   9.892308,  3.53, 'Subra'),
    ( 17.626444, -15.398558,  3.54, 'Ser Xi-55'),
    (  4.476944,  19.180431,  3.54, 'Ain'),
    (  7.335383,  21.982319,  3.54, 'Wasat'),
    ( 11.550033, -31.857625,  3.54, 'Hya Xi'),
    (  9.947706, -54.567792,  3.54, 'Vel Phi'),
    (  5.782594, -14.821950,  3.55, 'Lep Zeta-14'),
    ( 15.827003,  -3.430208,  3.55, 'Ser Mu-32'),
    ( 14.323394, -46.058094,  3.55, 'Lup Iota'),
    (  2.275164, -51.512164,  3.55, 'Eri Phi'),
    ( 20.145447, -66.182069,  3.55, 'Pav Delta'),
    (  0.323797,  -8.823922,  3.55, 'Shemali'),
    (  4.298239, -33.798347,  3.56, 'Eri Upsilon-41A'),
    ( 15.363436, -36.261375,  3.56, 'Lup Phi1'),
    ( 16.872261, -38.017536,  3.56, 'Sco Mu2'),
    ( 11.322347, -14.778542,  3.57, 'Crt Delta-12'),
    (  7.740792,  24.397992,  3.57, 'Gem Kappa-77'),
    ( 14.530497,  30.371436,  3.57, 'Boo Rho-25'),
    ( 18.350939,  72.732842,  3.57, 'Dra Chi-44A'),
    ( 20.300906, -12.544853,  3.58, 'Cap Alpha-6A'),
    (  7.301550,  16.540383,  3.58, 'Gem Lambda-54'),
    (  9.060425,  47.156525,  3.58, 'UMa Kappa-12'),
    (  9.511667, -40.466769,  3.59, 'Vel Psi'),
    (  1.633211,  48.628214,  3.59, '51 And'),
    ( 12.356003, -60.401147,  3.59, 'Cru Epsilon'),
    ( 17.762219, -64.723872,  3.59, 'Pav Eta'),
    ( 11.844922,   1.764717,  3.60, 'Zavijava'),
    (  5.741053, -22.448383,  3.60, 'Lep Gamma-13A'),
    ( 15.617069, -28.135081,  3.60, 'Lib Upsilon-39'),
    (  8.671553, -52.921886,  3.60, 'Omicron Velorum'),
    ( 17.518308, -60.683847,  3.60, 'Ara Delta'),
    (  5.293442,  -6.844408,  3.60, 'Ori Tau-20A'),
    ( 10.176467, -12.354083,  3.61, 'Hya Lambda-41A'),
    (  6.879817,  33.961253,  3.61, 'Gem Theta-34A'),
    ( 16.909725, -42.361314,  3.61, 'Sco Zeta2'),
    ( 13.037850, -71.548856,  3.61, 'Mus Delta'),
    (  1.400392,  -8.183256,  3.61, 'Cet Theta-45A'),
    (  3.413553,   9.028869,  3.61, 'Tau Omicron-1'),
    (  2.833064,  27.260508,  3.62, 'Ari 41A'),
    (  7.754250, -37.968583,  3.62, 'c Pup'),
    ( 20.625817,  14.595092,  3.63, 'Rotanev'),
    (  1.524725,  15.345822,  3.63, 'Psc Eta-99'),
    (  3.819372,  24.053417,  3.63, 'Atlas'),
    ( 18.096803, -30.424092,  3.63, 'Nash'),
    ( 11.760117, -66.728764,  3.63, 'Mus Lambda'),
    ( 23.032017,  42.325975,  3.64, 'And Omicron-1A'),
    (  4.329889,  15.627642,  3.65, 'Hyadum I'),
    ( 20.913500, -58.454156,  3.65, 'Ind Beta'),
    (  9.525475,  63.061861,  3.65, 'UMa 23A'),
    ( 14.073153,  64.375850,  3.65, 'Thuban'),
    ( 22.480525,  -0.020522,  3.66, 'Aqr Zeta-55B'),
    ( 15.769794,  15.421825,  3.66, 'Ser Beta-28A'),
    ( 15.644269, -29.777753,  3.66, 'Lib Tau-40'),
    ( 15.463814,  29.105703,  3.66, 'Nusakan'),
    ( 18.110519, -50.091478,  3.67, 'Ara Theta'),
    ( 21.668183, -16.662308,  3.68, 'Nashira'),
    ( 23.157444, -21.172411,  3.68, 'Aqr 88'),
    (  0.616189,  53.896908,  3.68, 'Cas Zeta-17'),
    (  4.853433,   5.605103,  3.68, 'Ori Pi4-3'),
    (  8.726539, -33.186386,  3.69, 'Pyx Alpha'),
    ( 17.962747,  29.247881,  3.70, 'Her Xi-92'),
    ( 23.286094,   3.282289,  3.70, 'Psc Gamma-6'),
    ( 11.767503,  47.779406,  3.70, 'UMa Chi-63'),
    (  4.904194,   2.440672,  3.71, 'Ori Pi5-8'),
    (  3.747928,  24.113339,  3.71, 'Electra'),
    ( 15.846936,   4.477731,  3.71, 'Ser Epsilon-37'),
    ( 21.082183,  43.927853,  3.71, 'Cyg Xi-62'),
    (  5.940081, -14.167700,  3.72, 'Lep Eta-16'),
    (  1.932631, -51.608894,  3.72, 'Eri Chi'),
    ( 19.921886,   6.406764,  3.72, 'Alshain'),
    ( 18.122494,   9.563847,  3.72, 'Oph 72'),
    (  1.857675, -10.335039,  3.73, 'Baten Kaitos'),
    ( 14.770811,   1.892886,  3.73, 'Vir 109'),
    (  5.992122,  54.284656,  3.73, 'Aur Delta-33A'),
    (  3.548844,  -9.458261,  3.73, 'Eri Epsilon-18'),
    (  3.452819,   9.732681,  3.73, 'Tau Xi-2'),
    ( 16.365336,  19.153131,  3.74, 'Her Gamma-20A'),
    (  3.325278, -21.757864,  3.74, 'Eri Tau-16'),
    ( 21.246525,  38.045317,  3.74, 'Cyg Tau-65A'),
    ( 17.892147,  56.872642,  3.74, 'Grumium'),
    ( 21.691292, -77.390047,  3.74, 'Oct Nu'),
    ( 21.444453, -22.411333,  3.75, 'Cap Zeta-34'),
    ( 17.798211,   2.707275,  3.75, 'Oph Gamma-62'),
    (  9.069244, -47.097736,  3.75, 'CD-46 4883'),
    ( 22.876911,  -7.579600,  3.75, 'Aqr Lambda-73'),
    (  4.382247,  17.542514,  3.76, 'Hyadum II'),
    (  5.041303,  41.075839,  3.76, 'Sadatoni'),
    (  2.844947,  55.895497,  3.76, 'Per Eta-15A'),
    ( 16.829767, -59.041378,  3.76, 'Ara Eta'),
    (  5.560422, -62.489825,  3.76, 'Dor Beta'),
    (  7.145797, -70.498931,  3.76, 'Vol Gamma2'),
    ( 19.078050, -21.741494,  3.77, 'Sgr Omicron-39'),
    ( 22.116853,  25.345111,  3.77, 'Peg Iota-24'),
    ( 19.495100,  51.729778,  3.77, 'Cyg Iota-10'),
    (  8.428942, -66.136889,  3.77, 'Vol Beta'),
    ( 20.794597,  -9.495775,  3.77, 'Albali'),
    ( 14.685819,  13.728300,  3.78, 'Boo Zeta-30A'),
    ( 14.685806,  13.728408,  3.78, 'Boo Zeta-30B'),
    ( 20.660636,  15.912081,  3.78, 'Sualocin'),
    (  5.855361, -20.879089,  3.78, 'Lep Delta-15'),
    (  3.753231,  42.578547,  3.78, 'Per Nu-41A'),
    ( 22.521528,  50.282492,  3.78, 'Lac Alpha-7'),
    (  9.849822,  59.038736,  3.78, 'UMa Upsilon-29'),
    (  7.428778,  27.798081,  3.79, 'Gem Iota-60'),
    ( 11.303039,  31.529236,  3.79, 'Alula Australis'),
    ( 10.888531,  34.214872,  3.79, 'Praecipua'),
    (  3.158269,  44.857544,  3.79, 'Per Kappa-27'),
    (  5.645769,  -2.600069,  3.80, 'Ori Sigma-48A'),
    (  3.201258, -28.987619,  3.80, 'Fornacis'),
    ( 19.285047,  53.368458,  3.80, 'Cyg Kappa-1'),
    (  4.592511, -30.562342,  3.81, 'Theemim'),
    ( 17.657747,  46.006333,  3.81, 'Her Iota-85A'),
    ( 20.227197,  46.741328,  3.81, 'Cyg 31A'),
    ( 11.523394,  69.331075,  3.81, 'Giausar'),
    ( 14.797697, -79.044750,  3.81, 'Aps Alpha'),
    ( 10.434842, -16.836292,  3.82, 'Hya Mu-42'),
    (  2.034117,   2.763758,  3.82, 'Al Rischa'),
    (  2.034083,   2.763792,  3.82, 'Psc Alpha-113B'),
    ( 15.712378,  26.295639,  3.82, 'CrB Gamma-8'),
    (  9.314067,  36.802597,  3.82, 'Lyn 38A'),
    ( 13.971186, -42.100753,  3.82, 'Cen Phi'),
    ( 19.789794,  18.534281,  3.83, 'Sge Delta-7A'),
    (  4.476250,  15.962181,  3.84, 'Tau Theta-77B'),
    ( 18.229392, -21.058833,  3.84, 'Sgr Mu-13A'),
    (  6.902208, -24.184211,  3.84, 'CMa Omicron-16'),
    ( 18.125708,  28.762489,  3.84, 'Her Omicron-103'),
    ( 17.937550,  37.250539,  3.84, 'Her Theta-91'),
    (  3.736661, -64.806903,  3.84, 'Ret Beta'),
    ( 19.802875,  70.267931,  3.84, 'Dra Epsilon-63'),
    ( 12.541114, -72.132989,  3.84, 'Mus Gamma'),
    ( 22.360939,  -1.387331,  3.85, 'Sadachbia'),
    ( 15.940883,  15.661617,  3.85, 'Ser Gamma-41A'),
    ( 18.394969,  21.769753,  3.85, 'Her 109A'),
    (  6.368564, -33.436400,  3.85, 'Col Delta'),
    ( 12.628378, -48.541303,  3.85, 'Cen Tau'),
    ( 16.257297, -63.685681,  3.85, 'TrA Delta'),
    ( 18.586786,  -8.244072,  3.85, 'Sct Alpha'),
    ( 10.546853,   9.306586,  3.85, 'Leo Rho-47'),
    (  3.738647,  32.288247,  3.86, 'Atik'),
    (  4.233364, -42.294369,  3.86, 'Hor Alpha'),
    ( 13.977986, -44.803583,  3.86, 'Cen Upsilon1'),
    ( 15.198911, -48.737819,  3.86, 'Lup Kappa1'),
    (  5.788081, -51.066514,  3.86, 'Pic Beta'),
    ( 13.398981,  54.921822,  3.86, 'UMa Zeta-79B'),
    ( 19.874547,   1.005661,  3.87, 'Aql Eta-55'),
    (  4.636339, -14.304019,  3.87, 'Eri 53'),
    (  5.520211, -35.470519,  3.87, 'Col Epsilon'),
    (  0.945892,  38.499344,  3.87, 'And Mu-37A'),
    ( 23.626067,  46.458153,  3.87, 'And Lambda-16A'),
    ( 14.717672,  -5.658206,  3.87, 'Vir Mu-107'),
    ( 16.557522, -78.897147,  3.87, 'Aps Gamma'),
    (  1.892169,  19.293853,  3.88, 'Mesarthim'),
    (  3.763781,  24.367750,  3.88, 'Maia'),
    (  9.879394,  26.006950,  3.88, 'Rasalas'),
    ( 15.948078, -29.214072,  3.88, 'Sco Rho-5'),
    ( 23.172650, -45.246711,  3.88, 'Gru Iota'),
    (  0.156844, -45.747425,  3.88, 'Phe Epsilon'),
    ( 12.331767,  -0.666803,  3.89, 'Zaniah'),
    (  9.239406,   2.314281,  3.89, 'Hya Theta-22A'),
    ( 11.350114, -54.491019,  3.89, 'Cen Pi'),
    ( 12.558039,  69.788239,  3.89, 'Dra Kappa-5'),
    ( 16.515228,   1.983922,  3.90, 'Marfik'),
    (  9.664267,  -1.142811,  3.90, 'Hya Iota-35'),
    ( 19.938436,  35.083425,  3.90, 'Cyg Eta-21A'),
    ( 16.329011,  46.313367,  3.90, 'Her Tau-22'),
    (  4.052606,   5.989306,  3.90, 'Tau Nu-38'),
    (  2.940458,  -8.898144,  3.90, 'Azha'),
    ( 17.004825,  30.926406,  3.91, 'Her Epsilon-58'),
    ( 15.085303, -47.051244,  3.91, 'Lup Pi'),
    ( 12.467328, -50.230636,  3.91, 'Cen Sigma'),
    ( 15.592106, -14.789536,  3.92, 'Lib Gamma-38A'),
    ( 19.361211, -17.847197,  3.93, 'Sgr Rho-44'),
    (  8.744750,  18.154308,  3.93, 'Cnc Delta-47'),
    (  4.605317,  -3.352458,  3.93, 'Eri Nu-48'),
    ( 20.952894,  41.167142,  3.94, 'Cyg Nu-58'),
    (  5.985781, -42.815136,  3.94, 'Col Eta'),
    (  1.520861, -49.072703,  3.94, 'Phe Delta'),
    ( 21.263731,   5.247844,  3.94, 'Equ Alpha-8A'),
    (  2.904294,  52.762478,  3.94, 'Per Tau-18A'),
    (  7.687453,  -9.551131,  3.94, 'Mon Alpha-26'),
    (  6.611400, -19.255878,  3.95, 'CMa Nu2-7'),
    ( 16.113453, -20.669192,  3.95, 'Sco Omega-9'),
    (  0.436722, -43.679831,  3.95, 'Phe Kappa'),
    (  2.057253,  72.421294,  3.95, 'Cas 50'),
    (  7.697017, -72.606097,  3.95, 'Vol Zeta'),
    ( 20.009875, -72.910503,  3.95, 'Pav Epsilon'),
    ( 11.398736,  10.529508,  3.96, 'Leo Iota-78'),
    ( 23.382842, -20.100581,  3.96, 'Aqr 98'),
    ( 22.775522,  23.565653,  3.96, 'Peg Lambda-47'),
    ( 15.849317, -33.627167,  3.96, 'Lup Chi-5'),
    (  4.400617, -34.016847,  3.96, 'Eri 43'),
    ( 19.398106, -40.615939,  3.96, 'Ruchbat'),
    ( 22.487825, -43.495564,  3.96, 'Gru Delta1'),
    ( 12.194197, -52.368461,  3.96, 'Cen Rho'),
    (  8.668372, -35.308353,  3.97, 'Pyx Beta'),
    (  5.858167,  39.148481,  3.97, 'Aur Nu-32'),
    (  9.010658,  41.782911,  3.97, 'UMa 10'),
    (  7.280506, -67.957153,  3.97, 'Vol Delta'),
    ( 18.010756,   2.931567,  3.98, 'Oph 67A'),
    (  7.730131, -28.954825,  3.98, 'Pup 3'),
    (  1.139744, -55.245761,  3.98, 'Phe Zeta'),
    (  6.247592,  -6.274775,  3.98, 'Mon Gamma-5'),
    ( 21.566347,  45.591836,  3.99, 'Cyg Rho-73'),
    (  5.588138,  -5.391110,  4.00, 'M42')
]
coords = []

def dist(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

for (x, y, _, _, _, _, ra, dec, _, _, w, _, _) in corr[1].data:
    if w > 0.9:
        mindist = 99999
        name = ''
        mag = 99
        for (r, d, m, n) in cat:
            d = dist(ra*24/360, dec, r, d)
            if d < mindist:
                mindist = d
                name = n
                mag = m

        if mindist < 0.001:
            body = ephem.FixedBody()
            body._ra, body._dec, body._epoch = str(ra * 24 / 360), str(dec), ephem.J2000
            body.compute(pos)
            az = math.degrees(float(repr(body.az)))
            alt = math.degrees(float(repr(body.alt)))
            alt2 = alt + 0.01666 / math.tan(math.pi/180*(alt + (7.31/(alt + 4.4))))
            coords.append((x, y, az, alt2, ra, dec, name, mag))
            #print(x, y, az, alt2, body.ra, body.dec, w, name, mag)
            print('%4.0f %4.0f: %s' % (x, y, name))

with open(args.outfile, 'w') as f:
    print('p f2 w36000 h18000 v360  E0 R0 n"TIFF_m c:LZW"', file=f)
    print('m g1 i0 f0 m2 p0.00784314', file=f)
    print('i w%d h%d f%d v25 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p0 y0 TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5 n"%s"' % (width, height, args.projection, args.picture), file=f)
    print('i w36000 h18000 f4 v360 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p0 y0 TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a0 b0 c0 d0 e0 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5 n"dummy.jpg"', file=f)
    print('', file=f)
    print('v v0', file=f)
    print('v r0', file=f)
    print('v p0', file=f)
    print('v y0', file=f)
    print('', file=f)
    for (x, y, az, alt, ra, dec, name, mag) in coords:
        print('c n0 N1 x%f y%f X%f Y%f t0 # %s (%s)' % (x-1, y-1, az*100, (90-alt)*100, name, mag), file=f)

subprocess.Popen(['autooptimiser', '-n', args.outfile, '-o', args.outfile], stdout=out, stderr=out).wait()

with open(args.outfile, 'a') as f:
    print('v a0', file=f)
    print('v b0', file=f)
    print('v c0', file=f)
    print('v d0', file=f)
    print('v e0', file=f)

subprocess.Popen(['autooptimiser', '-n', args.outfile, '-o', args.outfile], stdout=out, stderr=out).wait()

#subprocess.Popen(['bin/drawgrid.py', args.outfile], stdout=out, stderr=out).wait()
