#!/bin/bash

fov=115
line=0
rm -f stars.tmp.id calibration.log

file=$1
ts=$2
lat=$3
long=$4

# Use astrometry.net to map x,y to ra,dec
function solve {
    count=$((
	flock 9 || exit 1
	let count=$(cat counter.txt)+1
	echo $count > counter.txt
	echo $count
    ) 9> counter.lock)
    pto_var --set p0=$3 --set y0=$2 --opt r0,p0,y0,a0,b0,c0,d0,e0,v0 -o lens2_$count.pto lens.pto 2>> calibration.log > /dev/null
    sed 's/^p.*/p f3 w800 h800 v40 E0 R0 n"TIFF_m c:LZW"/' lens2_$count.pto > lens_$count.pto
    rm -f lens2_$count.pto

    nona -z DEFLATE -o out_$count- -m TIFF_m lens_$count.pto $1 dummy.jpg
    
    # Try to solve it
    mkdir -p tmp_$count
    rm -f out_$count.corr solutions_$count.txt
    touch solutions_$count.txt

    solve-field -p --scale-low 15 --odds-to-solve 10000000 -c 0.05 -d 20 -l 600 -m tmp_$count --sigma 1 -o out_$count --overwrite out_$count-0000.tif 2>> calibration.log > /dev/null
    #solve-field -p --scale-low 20 -l 30 -m tmp_$count -b $c --sigma 0 -o out_$count --overwrite out_$count-0000.tif 2>> calibration.log > /dev/null
    
    # If solution, find ra/dec's and corresponding x,y in original picture
    if [ -e out_$count.corr ]; then
	# Get x y ra dec from .corr file
	(echo 'import pyfits';echo 'for x in map (lambda x:(x[0], x[1], x[6], x[7]), pyfits.open("'out_$count.corr'")[1].data): print " ".join(map(str, x))') | python > solutions_$count.txt
	echo $2 >> yaws.txt
	echo $3 >> pitches.txt
    fi
    rm -f tmp_$count/*
}

export -f solve

# Create dummy frame where x=az y=90-alt
convert -size 360x180 xc:darkblue dummy.jpg

# Create initial Hugin project file
pto_gen -f $fov $file -p 3 -o /dev/stdout 2>> calibration.log | sed 's/ r:CROP//' > lens.pto
pto_gen -f 360 $file -p 4 dummy.jpg -o /dev/stdout 2>> calibration.log | sed 's/ r:CROP//;s/w360 h180/w36000 h18000/' > dummy.pto
pto_merge lens.pto dummy.pto -o lens.pto 2>> calibration.log > /dev/null
pto_var --opt r0,p0,y0,a0,b0,c0,d0,e0,v0 -o lens.pto lens.pto 2>> calibration.log > /dev/null

# Thoby fisheye
#sed 's/ f3 / f20 /' lens.pto > lens3.pto; mv lens3.pto lens.pto

convert -blur 0x32 $file blur.png
convert $file blur.png -fx "u-v" -brightness-contrast 45x60 clean.jpg
convert -blur 0x32 clean.jpg blur.png
convert clean.jpg blur.png -fx "u-2*min(v+0.25,1)" -brightness-contrast 45x65 clean2.jpg
mv clean2.jpg clean.jpg

step=10
width=$(identify -format %w $1)
height=$(identify -format %h $1)
let minyaw=-$fov/2
let maxyaw=$fov
let minpitch=-$fov*$height/$width/2
let maxpitch=$fov*$height/$width/2
 
clean=clean.jpg

rm -f solutions.txt out_.* out_-*
rm -f yaws.txt pitches.txt
pitches=""
for pitch in $(seq $minpitch $step $maxpitch); do
    pitches="$pitches $pitch"
done
yaws=""
for yaw in $(seq $minyaw $step $maxyaw); do
    yaws="$yaws $yaw"
done
echo 0 > counter.txt
parallel -u -j $(echo $(parallel --number-of-cores) | bc) solve $clean {1} {2} $astrometry ::: $yaws ::: $pitches

if [ -e yaws.txt ]; then
    let minyaw=$(sort -n yaws.txt | head -n1)-5
    let maxyaw=$(sort -nr yaws.txt | head -n1)+5
fi
if [ -e patches.txt ]; then
    let minpitch=$(sort -n pitches.txt | head -n1)-5
    let maxpitch=$(sort -nr pitches.txt | head -n1)+5
fi

# Remap x,y back to original frame
for i in $(seq 1 $(cat counter.txt)); do
    awk '{print $1" "$2}' solutions_$i.txt | pano_trafo -r lens_$i.pto 0 2>> calibration.log | paste - solutions_$i.txt | awk '{print $1" "$2" "$5*24/360" "$6}' >> solutions.txt
    done

# Wrap up the solutions for this frame
touch solutions.txt
awk '{printf("%.1f %.1f %.3f %.2f\n", $1, $2, $3, $4)}' solutions.txt | sort -u -k3,4 | sort -u -k1,2 > solutions_tmp.txt
awk '{system("~/bin/findstar.py "$3" "$4)}' solutions_tmp.txt | paste solutions_tmp.txt - > $clean.solutions
echo -n "Stars identified: "
wc -l < $clean.solutions

echo $ts" "$(awk '{print $1","$2}' < clean.jpg.solutions) > stars.txt
awk '{i=i+1;print i" "$3","$4" "$5" "$6" "$7" "$8}' < clean.jpg.solutions | sed 's/,/ /' > stars.id

if [ ! -s stars.id ]; then
    echo No solution.
    exit 1
fi

cp lens.pto lens.pto.bak
cp stars.id stars.id.bak


pano_modify --canvas=36000x18000 --fov=360x180 --projection=2 lens.pto -o lens.pto > /dev/null 2>> calibration.log

~/bin/map.py --latitude $lat --longitude $long stars.txt stars.id | awk '{printf("c n0 N1 x%s y%s X%s Y%s t0\n", $1, $2, $3*100, (90-$4)*100)}' > stars.lens
cat stars.lens >> lens.pto

cat stars.id
cpclean -n 1 lens.pto -o lens.pto 2>&1 | grep Removed
error=$(autooptimiser -n lens.pto -o lens2.pto 2>&1 | grep units | tail -n1)

echo $error | awk '{printf("Average error: %.3f arc minutes\n", $4*60/100)}'

poor=$(echo $error | awk '{ print($4*60/100 > 5) }')

if [ "$poor" -eq 1 ]; then
    echo Poor solution.  Improve by eliminating stars.

    num=$(wc -l < stars.lens)
    len=$(wc -l < lens2.pto)
    for i in $(seq 1 10 $num); do
	cpclean -n 1 lens2.pto -o lens2.pto 2> /dev/null >> calibration.log
	error=$(autooptimiser -n lens2.pto -o lens2.pto 2> /dev/null | grep units | tail -n1)
	dist=$(grep f3 lens2.pto | sed 's/.* a//;s/ d.*//;s/[bc]//g'|awk 'function abs(x){return ((x < 0.0) ? -x : x)} {print abs($1)+abs($2)+abs($3)}')
	lastlen=$len
	len=$(wc -l < lens2.pto)
	echo "Lens distortion: "$dist
	if [ $(echo $dist | awk '{ print($1 < 0.1)}') -eq 1 ] || [ $len -eq $lastlen ] ; then
	    break
	fi
    done

    echo $error | awk '{printf("Average error: %.3f arc minutes\n", $4*60/100)}'
fi

~/bin/drawgrid.py lens2.pto
composite -blend 40 $1 grid.png grid.jpg
echo convert -pointsize 12 $(~/bin/brightstar.py --latitude $lat --longitude $long $ts lens2.pto 2> /dev/null | sed 's/[(),]//g;s/'\''//g' | awk '{x=$1; y=$2; az=$3; alt=$4; $1=$2=$3=$4=""; sub(/ */, ""); printf("-stroke white -fill none -draw \"circle %f,%f %f,%f\" -stroke none -fill white -annotate +%f+%f \"%s [%.2f %.2f]\"\n", x, y, x+7, y, x+11, y-4, $0, az, alt)}') grid.png grid-labels.png | bash
composite -blend 40 $1 grid-labels.png grid-labels.jpg
